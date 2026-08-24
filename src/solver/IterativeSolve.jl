# IterativeSolve.jl — GMRES iterative solver via Krylov.jl
#
# Provides iterative solve alternatives to the dense direct factorization
# in Solve.jl, with support for near-field sparse preconditioning.

export solve_gmres, solve_gmres_adjoint

const _DEFAULT_MAX_GMRES_WORKSPACE_BYTES = 512 * 1024 * 1024

function _gmres_workspace_bytes(problem_size::Int, memory::Int)
    effective_memory = min(problem_size, memory)
    # Krylov.GmresSolver stores x, w, `effective_memory` Arnoldi vectors,
    # three length-memory scalar arrays, and packed upper-triangular R.
    # Δx, p, and q are zero-length for the unpreconditioned constructor and
    # are allocated lazily when the corresponding option requires them, so
    # reserve three additional problem-size vectors conservatively.
    total = BigInt(sizeof(ComplexF64)) * (
        BigInt(effective_memory + 5) * problem_size +
        2BigInt(effective_memory) +
        (BigInt(effective_memory) * (effective_memory + 1)) ÷ 2
    )
    total += BigInt(sizeof(Float64)) * effective_memory
    total <= typemax(Int) ||
        throw(ArgumentError("GMRES workspace estimate overflows Int"))
    return Int(total)
end

function _preflight_gmres_workspace(
        problem_size::Int, memory::Int, max_workspace_bytes::Integer)
    return _enforce_payload_limit(
        _gmres_workspace_bytes(problem_size, memory),
        max_workspace_bytes,
        "GMRES Krylov workspace", "max_workspace_bytes")
end

@inline function _as_complex_rhs(rhs::AbstractVector{<:Number})
    if rhs isa Vector{ComplexF64}
        return rhs
    end
    return Vector{ComplexF64}(rhs)
end

@inline function _gmres_final_residual(stats)
    return hasproperty(stats, :residuals) && !isempty(stats.residuals) ?
           stats.residuals[end] : NaN
end

function _assert_gmres_converged(stats, label::AbstractString; tol::Float64, maxiter::Int)
    solved = hasproperty(stats, :solved) ? Bool(stats.solved) : false
    inconsistent = hasproperty(stats, :inconsistent) ?
                   Bool(stats.inconsistent) : false
    solved && !inconsistent && return stats
    niter = hasproperty(stats, :niter) ? stats.niter : missing
    status = hasproperty(stats, :status) ? stats.status : "unknown"
    resid = _gmres_final_residual(stats)
    error("$label GMRES did not converge consistently: niter=$niter, " *
          "status=$status, solved=$solved, inconsistent=$inconsistent, " *
          "final_residual=$resid, tol=$tol, maxiter=$maxiter")
end

function _assert_gmres_result(x::AbstractVector, stats, label::AbstractString;
                              tol::Float64, maxiter::Int)
    _assert_gmres_converged(stats, label; tol=tol, maxiter=maxiter)
    @inbounds for i in eachindex(x)
        isfinite(x[i]) ||
            error("$label GMRES returned a non-finite solution component at index $i: $(x[i])")
    end
    return stats
end

@inline function _validate_gmres_options(tol::Float64, maxiter::Int,
                                         memory::Int, precond_side::Symbol)
    (isfinite(tol) && tol > 0.0) ||
        throw(ArgumentError("GMRES tol must be finite and positive, got $tol"))
    maxiter >= 1 ||
        throw(ArgumentError("GMRES maxiter must be at least 1, got $maxiter"))
    memory >= 1 ||
        throw(ArgumentError("GMRES memory must be at least 1, got $memory"))
    precond_side in (:left, :right) ||
        throw(ArgumentError("Invalid precond_side: $precond_side (expected :left or :right)"))
    return nothing
end

function _gmres_max_component_exponent(values)
    maximum_exponent = typemin(Int)
    @inbounds for value in values
        real_value = abs(Float64(real(value)))
        imag_value = abs(Float64(imag(value)))
        !iszero(real_value) &&
            (maximum_exponent = max(
                maximum_exponent, exponent(real_value)))
        !iszero(imag_value) &&
            (maximum_exponent = max(
                maximum_exponent, exponent(imag_value)))
    end
    return maximum_exponent
end

@inline function _gmres_known_operator_values(A::AbstractMatrix)
    if A isa StridedMatrix
        return A
    elseif A isa SparseMatrixCSC
        return nonzeros(A)
    end
    return nothing
end

@inline function _gmres_scale_component(
        component::Float64,
        shift::Int,
        label::AbstractString)
    scaled = ldexp(component, shift)
    isfinite(scaled) ||
        throw(OverflowError(
            "$label scaling produced a non-finite component."))
    !iszero(component) && iszero(scaled) &&
        throw(ArgumentError(
            "$label spans too much exponent range for lossless " *
            "power-of-two GMRES scaling."))
    return scaled
end

@inline function _gmres_scale_value(
        value::Number,
        shift::Int,
        label::AbstractString)
    converted = ComplexF64(value)
    return ComplexF64(
        _gmres_scale_component(real(converted), shift, label),
        _gmres_scale_component(imag(converted), shift, label),
    )
end

function _gmres_scaled_operator(
        A::StridedMatrix,
        shift::Int,
        label::AbstractString)
    scaled = Matrix{ComplexF64}(undef, size(A))
    @inbounds for index in eachindex(A)
        scaled[index] = _gmres_scale_value(A[index], shift, label)
    end
    return scaled
end

function _gmres_scaled_operator(
        A::SparseMatrixCSC,
        shift::Int,
        label::AbstractString)
    scaled = SparseMatrixCSC{ComplexF64,Int}(A)
    @inbounds for index in eachindex(nonzeros(scaled))
        nonzeros(scaled)[index] =
            _gmres_scale_value(nonzeros(scaled)[index], shift, label)
    end
    return scaled
end

function _gmres_scaled_problem(
        A::AbstractMatrix,
        rhs::Vector{ComplexF64},
        label::AbstractString,
        max_workspace_bytes::Int,
        krylov_workspace_bytes::Int)
    operator_values = _gmres_known_operator_values(A)
    operator_exponent = operator_values === nothing ?
                        typemin(Int) :
                        _gmres_max_component_exponent(operator_values)
    rhs_exponent = _gmres_max_component_exponent(rhs)
    operator_extreme = operator_exponent != typemin(Int) &&
                       (operator_exponent < -128 ||
                        operator_exponent > 128)
    rhs_extreme = any(
        value -> _ieee_dense_extreme_factor(value, Float64), rhs)
    (operator_extreme || rhs_extreme) || return A, rhs, 0

    operator_shift = operator_exponent == typemin(Int) ?
                     0 : -operator_exponent
    solution_shift = rhs_exponent == typemin(Int) ?
                     0 : operator_shift + rhs_exponent
    rhs_shift = rhs_exponent == typemin(Int) ?
                operator_shift : -rhs_exponent

    extra_bytes = BigInt(0)
    if !iszero(operator_shift)
        if A isa StridedMatrix
            extra_bytes += BigInt(sizeof(ComplexF64)) * length(A)
        elseif A isa SparseMatrixCSC
            # Stored nonzeros are copied into a ComplexF64 sparse matrix; CSC
            # column pointers and row indices are retained alongside them.
            extra_bytes += BigInt(sizeof(ComplexF64) + sizeof(Int)) * nnz(A)
            extra_bytes += BigInt(sizeof(Int)) * (size(A, 2) + 1)
        end
    end
    !iszero(rhs_shift) &&
        (extra_bytes += BigInt(sizeof(ComplexF64)) * length(rhs))
    total_bytes = BigInt(krylov_workspace_bytes) + extra_bytes
    total_bytes <= max_workspace_bytes ||
        throw(ArgumentError(
            "$label Krylov and scaling workspaces require $total_bytes raw " *
            "bytes, exceeding max_workspace_bytes=$max_workspace_bytes"))

    scaled_A = if iszero(operator_shift)
        A
    elseif A isa StridedMatrix || A isa SparseMatrixCSC
        _gmres_scaled_operator(A, operator_shift, "$label operator")
    else
        # An unknown matrix-free operator cannot be safely rescaled without
        # knowing whether its own mul! already loses exponent range. Variable
        # scaling still normalizes a tiny or huge RHS.
        solution_shift = rhs_exponent == typemin(Int) ? 0 : rhs_exponent
        rhs_shift = rhs_exponent == typemin(Int) ? 0 : -rhs_exponent
        A
    end

    scaled_rhs = if iszero(rhs_shift)
        rhs
    else
        result = similar(rhs)
        @inbounds for index in eachindex(rhs)
            result[index] = _gmres_scale_value(
                rhs[index], rhs_shift, "$label RHS")
        end
        result
    end
    return scaled_A, scaled_rhs, solution_shift
end

function _gmres_recover_solution!(
        solution::Vector{ComplexF64},
        shift::Int,
        label::AbstractString)
    iszero(shift) && return solution
    @inbounds for index in eachindex(solution)
        value = solution[index]
        real_value = ldexp(real(value), shift)
        imag_value = ldexp(imag(value), shift)
        converted = ComplexF64(real_value, imag_value)
        isfinite(converted) ||
            throw(OverflowError(
                "$label solution is outside the representable " *
                "ComplexF64 range at index $index."))
        solution[index] = converted
    end
    return solution
end

@noinline function _dense_true_residual_ratio_bigfloat(
        A::AbstractMatrix,
        x::AbstractVector,
        rhs::AbstractVector)
    return setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        residual_squared = zero(BigFloat)
        rhs_squared = zero(BigFloat)
        @inbounds for row in axes(A, 1)
            total = zero(Complex{BigFloat})
            for column in axes(A, 2)
                total += Complex{BigFloat}(A[row, column]) *
                         Complex{BigFloat}(x[column])
            end
            rhs_value = Complex{BigFloat}(rhs[row])
            residual_squared += abs2(total - rhs_value)
            rhs_squared += abs2(rhs_value)
        end
        if iszero(rhs_squared)
            return iszero(residual_squared) ? 0.0 : Inf
        end
        return Float64(sqrt(residual_squared / rhs_squared))
    end
end

@inline function _gmres_is_stored_dense(A::AbstractMatrix)
    return A isa StridedMatrix ||
           ((A isa Adjoint || A isa Transpose) &&
            parent(A) isa StridedMatrix)
end

function _true_residual_ratio(
        A::AbstractMatrix,
        x::AbstractVector,
        rhs::Vector{ComplexF64},
        label::AbstractString)
    if _gmres_is_stored_dense(A) &&
       (_ieee_dense_product_requires_fallback(A, x, Float64) ||
        any(value -> _ieee_dense_extreme_factor(value, Float64), rhs))
        return _dense_true_residual_ratio_bigfloat(A, x, rhs)
    end

    residual = Vector{ComplexF64}(undef, length(rhs))
    mul!(residual, A, x)
    @inbounds for index in eachindex(residual, rhs)
        residual[index] -= rhs[index]
        isfinite(residual[index]) ||
            error("$label true residual is non-finite at index $index.")
    end
    rhs_norm = norm(rhs)
    residual_norm = norm(residual)
    if iszero(rhs_norm)
        return iszero(residual_norm) ? 0.0 : Inf
    end
    return residual_norm / rhs_norm
end

function _assert_true_residual(A::AbstractMatrix, x::AbstractVector, rhs::AbstractVector,
                               label::AbstractString;
                               tol::Float64,
                               factor::Float64=100.0)
    (isfinite(factor) && factor > 0.0) ||
        throw(ArgumentError(
            "true residual factor must be finite and positive, got $factor"))
    rhs_c = _as_complex_rhs(rhs)
    relres = _true_residual_ratio(A, x, rhs_c, label)
    limit = factor * tol
    isfinite(limit) ||
        throw(ArgumentError(
            "true residual limit must be finite, got factor*tol=$limit"))
    isfinite(relres) && relres <= limit && return relres
    error("$label GMRES true residual too large: relative_residual=$relres, " *
          "limit=$limit, tol=$tol, factor=$factor")
end

"""
    solve_gmres(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8,
                maxiter=200, memory=20, max_workspace_bytes=536_870_912,
                verbose=false,
                check_gmres_convergence=true)

Solve Z x = rhs using GMRES from Krylov.jl.

If `preconditioner` is an `AbstractPreconditionerData`, it is applied via:
- `precond_side=:left` (default): left preconditioner M in Krylov.gmres
- `precond_side=:right`: right preconditioner N in Krylov.gmres

Returns `(x, stats)` where `stats` is the Krylov.jl convergence info.
By default, an unconverged or non-finite result throws. Set
`check_gmres_convergence=false` only when intentionally inspecting a partial
iterate and its `stats`.
`max_workspace_bytes` bounds the raw payload of the Krylov basis, solver
vectors, and Hessenberg/rotation storage before Krylov allocates them.
"""
function solve_gmres(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                     preconditioner::Union{Nothing, AbstractPreconditionerData}=nothing,
                     precond_side::Symbol=:left,
                     tol::Float64=1e-8,
                     maxiter::Int=200,
                     memory::Int=20,
                     max_workspace_bytes::Integer=
                         _DEFAULT_MAX_GMRES_WORKSPACE_BYTES,
                     verbose::Bool=false,
                     check_gmres_convergence::Bool=true)
    _validate_gmres_options(tol, maxiter, memory, precond_side)
    _validate_linear_system_inputs(Z, rhs, "forward GMRES")
    workspace_limit = _validated_resource_limit(
        "max_workspace_bytes", max_workspace_bytes)
    krylov_workspace_bytes = _preflight_gmres_workspace(
        size(Z, 1), memory, workspace_limit)
    rhs_c = _as_complex_rhs(rhs)
    scaled_Z, scaled_rhs, solution_shift =
        _gmres_scaled_problem(
            Z, rhs_c, "forward GMRES",
            workspace_limit, krylov_workspace_bytes)

    if preconditioner === nothing
        x, stats = Krylov.gmres(scaled_Z, scaled_rhs;
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :right
        N_op = NearFieldOperator(preconditioner)
        x, stats = Krylov.gmres(scaled_Z, scaled_rhs;
                                 N=N_op,
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :left
        M = NearFieldOperator(preconditioner)
        x, stats = Krylov.gmres(scaled_Z, scaled_rhs;
                                 M=M,
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    _gmres_recover_solution!(x, solution_shift, "forward GMRES")
    _assert_finite_linear_vector(x, "forward GMRES solution")
    check_gmres_convergence &&
        _assert_gmres_converged(
            stats, "forward"; tol=tol, maxiter=maxiter)
    return x, stats
end

"""
    solve_gmres_adjoint(Z, rhs; preconditioner=nothing, precond_side=:left,
                        tol=1e-8, maxiter=200, memory=20,
                        max_workspace_bytes=536_870_912, verbose=false,
                        check_gmres_convergence=true)

Solve Z† x = rhs using GMRES, with the adjoint preconditioner Z_nf⁻ᴴ.

This is used for the adjoint linear system in sensitivity analysis:
  Z†(θ) λ = ∂Φ/∂I*

Returns `(x, stats)`. By default, an unconverged or non-finite result throws.
Set `check_gmres_convergence=false` only when intentionally inspecting a
partial iterate and its `stats`.
"""
function solve_gmres_adjoint(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                              preconditioner::Union{Nothing, AbstractPreconditionerData}=nothing,
                              precond_side::Symbol=:left,
                              tol::Float64=1e-8,
                              maxiter::Int=200,
                              memory::Int=20,
                              max_workspace_bytes::Integer=
                                  _DEFAULT_MAX_GMRES_WORKSPACE_BYTES,
                              verbose::Bool=false,
                              check_gmres_convergence::Bool=true)
    _validate_gmres_options(tol, maxiter, memory, precond_side)
    _validate_linear_system_inputs(Z, rhs, "adjoint GMRES")
    workspace_limit = _validated_resource_limit(
        "max_workspace_bytes", max_workspace_bytes)
    krylov_workspace_bytes = _preflight_gmres_workspace(
        size(Z, 1), memory, workspace_limit)
    rhs_c = _as_complex_rhs(rhs)
    scaled_Z, scaled_rhs, solution_shift =
        _gmres_scaled_problem(
            Z, rhs_c, "adjoint GMRES",
            workspace_limit, krylov_workspace_bytes)

    if preconditioner === nothing
        x, stats = Krylov.gmres(adjoint(scaled_Z), scaled_rhs;
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :right
        N_adj = NearFieldAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(adjoint(scaled_Z), scaled_rhs;
                                 N=N_adj,
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :left
        M_adj = NearFieldAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(adjoint(scaled_Z), scaled_rhs;
                                 M=M_adj,
                                 memory=memory,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    _gmres_recover_solution!(x, solution_shift, "adjoint GMRES")
    _assert_finite_linear_vector(x, "adjoint GMRES solution")
    check_gmres_convergence &&
        _assert_gmres_converged(
            stats, "adjoint"; tol=tol, maxiter=maxiter)
    return x, stats
end
