# Solve.jl — Forward and adjoint linear system solves

export solve_forward, solve_system, assemble_full_Z, assemble_full_Z!,
       make_mass_regularizer, make_left_preconditioner,
       select_preconditioner, transform_patch_matrices, prepare_conditioned_system

@inline function _validate_known_matrix_entries(
    matrix::AbstractMatrix,
    label::AbstractString,
)
    values = if matrix isa StridedMatrix
        matrix
    elseif matrix isa SparseMatrixCSC
        nonzeros(matrix)
    elseif matrix isa LocalMassMatrix
        matrix.vals
    else
        return nothing
    end
    all(isfinite, values) ||
        throw(ArgumentError("$label must contain only finite values"))
    return nothing
end

function _validate_linear_system_inputs(
    matrix::AbstractMatrix,
    rhs::AbstractVector,
    label::AbstractString,
)
    N = size(matrix, 1)
    N >= 1 ||
        throw(ArgumentError("$label matrix must be nonempty"))
    size(matrix, 2) == N ||
        throw(DimensionMismatch(
            "$label matrix must be square, got size $(size(matrix))"))
    length(rhs) == N ||
        throw(DimensionMismatch(
            "$label RHS length $(length(rhs)) must equal matrix size $N"))
    _validate_known_matrix_entries(matrix, label)
    all(isfinite, rhs) ||
        throw(ArgumentError("$label RHS must contain only finite values"))
    return N
end

@inline function _assert_finite_linear_vector(
    vector::AbstractVector,
    label::AbstractString,
)
    @inbounds for i in eachindex(vector)
        isfinite(vector[i]) ||
            error("$label contains a non-finite value at index $i: $(vector[i])")
    end
    return vector
end

"""
    solve_forward(Z, v; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, verbose_gmres=false, check_gmres_convergence=true, check_true_residual=false, true_residual_factor=100.0)

Solve Z I = v. Uses direct factorization by default, or GMRES when `solver=:gmres`.

# Arguments
- `solver`: `:direct` for LU factorization, `:gmres` for preconditioned GMRES
- `preconditioner`: a preconditioner object (e.g., `AbstractPreconditionerData`), or `nothing`
- `gmres_precond_side`: `:left` or `:right` preconditioner application side
- `gmres_tol`: relative tolerance for GMRES convergence
- `gmres_maxiter`: maximum GMRES iterations
- `gmres_memory`: Krylov restart/memory parameter
- `check_gmres_convergence`: throw an error if GMRES returns an unconverged solve
- `check_true_residual`: additionally verify `norm(Z*x-v)/norm(v)`
- `true_residual_factor`: allowed true-residual multiple of `gmres_tol`
"""
function solve_forward(Z::AbstractMatrix{<:Number}, v::AbstractVector{<:Number};
                       solver::Symbol=:direct,
                       preconditioner=nothing,
                       gmres_precond_side::Symbol=:left,
                       gmres_tol::Float64=1e-8,
                       gmres_maxiter::Int=200,
                       gmres_memory::Int=20,
                       verbose_gmres::Bool=false,
                       check_gmres_convergence::Bool=true,
                       check_true_residual::Bool=false,
                       true_residual_factor::Float64=100.0)
    solver in (:direct, :gmres) ||
        throw(ArgumentError(
            "Unknown solver: $solver (expected :direct or :gmres)"))
    if solver == :direct
        Z isa Matrix ||
            throw(ArgumentError(
                "Direct solver requires a dense Matrix; use solver=:gmres for operator-based systems."))
        _validate_linear_system_inputs(Z, v, "forward solve")
        return _assert_finite_linear_vector(
            Z \ v, "direct forward solution")
    else
        x, stats = solve_gmres(Z, v;
                                preconditioner=preconditioner,
                                precond_side=gmres_precond_side,
                                tol=gmres_tol, maxiter=gmres_maxiter,
                                memory=gmres_memory,
                                verbose=verbose_gmres,
                                check_gmres_convergence=check_gmres_convergence)
        check_true_residual &&
            _assert_true_residual(Z, x, v, "forward";
                                  tol=gmres_tol,
                                  factor=true_residual_factor)
        return _assert_finite_linear_vector(x, "GMRES forward solution")
    end
end

"""
    solve_system(Z, rhs; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, check_gmres_convergence=true, check_true_residual=false, true_residual_factor=100.0)

General linear solve Z x = rhs with solver dispatch.
"""
function solve_system(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                      solver::Symbol=:direct,
                      preconditioner=nothing,
                      gmres_precond_side::Symbol=:left,
                      gmres_tol::Float64=1e-8,
                      gmres_maxiter::Int=200,
                      gmres_memory::Int=20,
                      check_gmres_convergence::Bool=true,
                      check_true_residual::Bool=false,
                      true_residual_factor::Float64=100.0)
    return solve_forward(Z, rhs; solver=solver, preconditioner=preconditioner,
                          gmres_tol=gmres_tol, gmres_maxiter=gmres_maxiter,
                          gmres_precond_side=gmres_precond_side,
                          gmres_memory=gmres_memory,
                          check_gmres_convergence=check_gmres_convergence,
                          check_true_residual=check_true_residual,
                          true_residual_factor=true_residual_factor)
end

"""
    assemble_full_Z(Z_efie, Mp, theta; reactive=false)

Assemble the full MoM matrix: Z(θ) = Z_efie + Z_imp(θ)

For resistive impedance (default):  Z_imp = -Σ_p θ_p M_p
For reactive impedance:             Z_imp = -Σ_p (iθ_p) M_p
"""
function assemble_full_Z(Z_efie::Matrix{<:Number},
                         Mp::Vector{<:AbstractMatrix},
                         theta::AbstractVector;
                         reactive::Bool=false)
    Z = copy(Z_efie)
    assemble_full_Z!(Z, Z_efie, Mp, theta; reactive=reactive)
    return Z
end

"""
    assemble_full_Z!(Z, Z_efie, Mp, theta; reactive=false)

In-place variant: writes Z(θ) = Z_efie + Z_imp(θ) into pre-allocated `Z`.
"""
function assemble_full_Z!(Z::Matrix{<:Number},
                          Z_efie::Matrix{<:Number},
                          Mp::Vector{<:AbstractMatrix},
                          theta::AbstractVector;
                          reactive::Bool=false)
    size(Z) == size(Z_efie) ||
        throw(DimensionMismatch(
            "output Z has size $(size(Z)), expected $(size(Z_efie))"))
    _validate_impedance_inputs(Mp, theta, size(Z_efie))
    copyto!(Z, Z_efie)
    for p in eachindex(theta)
        coeff = reactive ? (1im * theta[p]) : theta[p]
        _add_scaled_matrix!(Z, -coeff, Mp[p])
    end
    all(isfinite, Z) ||
        error("full impedance-loaded system contains non-finite entries")
    return Z
end

function _validated_mass_matrix_size(Mp::Vector{<:AbstractMatrix})
    N = first(_validate_mass_matrix_sizes(Mp))
    @inbounds for p in eachindex(Mp)
        _validate_known_matrix_entries(Mp[p], "patch mass matrix")
    end
    return N
end

function _validated_conditioning_matrix(
    matrix,
    N::Int,
    label::AbstractString,
)
    converted = try
        Matrix{ComplexF64}(matrix)
    catch err
        throw(ArgumentError(
            "$label must be convertible to a dense ComplexF64 matrix: $(sprint(showerror, err))"))
    end
    size(converted) == (N, N) ||
        throw(DimensionMismatch(
            "$label has size $(size(converted)), expected ($N, $N)"))
    all(isfinite, converted) ||
        throw(ArgumentError("$label must contain only finite values"))
    return converted
end

function _validate_conditioning_factor(
    factor,
    N::Int,
    label::AbstractString,
)
    size(factor) == (N, N) ||
        throw(DimensionMismatch(
            "$label has size $(size(factor)), expected ($N, $N)"))
    if applicable(issuccess, factor) && !issuccess(factor)
        throw(ArgumentError("$label is not a successful factorization"))
    end
    if hasproperty(factor, :factors)
        factors = getproperty(factor, :factors)
        factors isa AbstractMatrix &&
            _validate_known_matrix_entries(factors, label)
    end
    return factor
end

@inline function _ldiv_reusing_input(factor, rhs)
    if applicable(ldiv!, factor, rhs)
        ldiv!(factor, rhs)
        return rhs
    end
    return factor \ rhs
end

"""
    make_mass_regularizer(Mp)

Build a Hermitian positive-semidefinite mass-based regularizer from patch
mass matrices:
  R = Σ_p M_p

Returns a dense `ComplexF64` matrix so it can be used directly in
regularized solves.
"""
function make_mass_regularizer(Mp::Vector{<:AbstractMatrix})
    N = _validated_mass_matrix_size(Mp)
    R = zeros(ComplexF64, N, N)
    for p in eachindex(Mp)
        _add_scaled_matrix!(R, one(ComplexF64), Mp[p])
    end
    all(isfinite, R) ||
        error("mass regularizer accumulation produced non-finite values")

    # Enforce Hermitian symmetry in place, avoiding two extra N×N matrices.
    @inbounds for j in 1:N
        R[j, j] = complex(real(R[j, j]), 0.0)
        for i in 1:(j - 1)
            value = 0.5 * (R[i, j] + conj(R[j, i]))
            R[i, j] = value
            R[j, i] = conj(value)
        end
    end
    all(isfinite, R) ||
        error("mass regularizer symmetrization produced non-finite values")
    return R
end

"""
    make_left_preconditioner(Mp; eps_rel=1e-8)

Build a simple mass-based left preconditioner matrix:
  M = R + ϵ I,  R = Σ_p M_p

`eps_rel` scales the diagonal shift as
  ϵ = eps_rel * max(tr(R)/N, 1).
"""
function make_left_preconditioner(Mp::Vector{<:AbstractMatrix};
                                  eps_rel::Float64=1e-8)
    (isfinite(eps_rel) && eps_rel > 0.0) ||
        throw(ArgumentError(
            "eps_rel must be finite and positive, got $eps_rel"))
    R = make_mass_regularizer(Mp)
    N = size(R, 1)
    scale = max(real(tr(R)) / N, 1.0)
    isfinite(scale) ||
        error("mass preconditioner scale is non-finite")
    ϵ = eps_rel * scale
    isfinite(ϵ) && ϵ > 0.0 ||
        error("mass preconditioner diagonal shift is non-finite or nonpositive")
    @inbounds for i in 1:N
        R[i, i] += ϵ
    end
    all(isfinite, R) ||
        error("mass preconditioner contains non-finite values")
    return R
end

"""
    select_preconditioner(Mp;
                          mode=:off,
                          preconditioner_M=nothing,
                          n_threshold=256,
                          iterative_solver=false,
                          eps_rel=1e-6)

Select the effective left preconditioner matrix used by the solver.

Modes:
- `:off`: disable preconditioning (unless `preconditioner_M` is provided).
- `:on`: always build/use a mass-based preconditioner.
- `:auto`: enable only when `iterative_solver=true` or `N >= n_threshold`.

If `preconditioner_M` is provided, it takes precedence over `mode`.

Returns `(M_eff, enabled, reason)` where:
- `M_eff` is either a dense `ComplexF64` matrix or `nothing`,
- `enabled` indicates whether preconditioning is active,
- `reason` is a short status string for logging/debugging.
"""
function select_preconditioner(Mp::Vector{<:AbstractMatrix};
                               mode::Symbol=:off,
                               preconditioner_M=nothing,
                               n_threshold::Int=256,
                               iterative_solver::Bool=false,
                               eps_rel::Float64=1e-6)
    mode ∈ (:off, :on, :auto) ||
        throw(ArgumentError(
            "Invalid preconditioner mode: $mode (expected :off, :on, or :auto)"))
    n_threshold >= 0 ||
        throw(ArgumentError(
            "n_threshold must be nonnegative, got $n_threshold"))
    (isfinite(eps_rel) && eps_rel > 0.0) ||
        throw(ArgumentError(
            "eps_rel must be finite and positive, got $eps_rel"))
    N = _validated_mass_matrix_size(Mp)

    if preconditioner_M !== nothing
        matrix = _validated_conditioning_matrix(
            preconditioner_M, N, "preconditioner_M")
        return matrix, true, "user-provided preconditioner"
    end

    if mode == :off
        return nothing, false, "mode=:off"
    elseif mode == :on
        return make_left_preconditioner(Mp; eps_rel=eps_rel), true, "mode=:on"
    else
        if iterative_solver
            return make_left_preconditioner(Mp; eps_rel=eps_rel), true, "mode=:auto (iterative_solver=true)"
        elseif N >= n_threshold
            return make_left_preconditioner(Mp; eps_rel=eps_rel), true, "mode=:auto (N=$N >= $n_threshold)"
        else
            return nothing, false, "mode=:auto (N=$N < $n_threshold)"
        end
    end
end

"""
    transform_patch_matrices(Mp; preconditioner_M=nothing, preconditioner_factor=nothing)

Transform derivative blocks under left preconditioning:
  M_p_tilde = M^{-1} M_p

When `preconditioner_M === nothing`, returns `Mp` unchanged.
If `preconditioner_factor` is provided, it is reused instead of factorizing
`preconditioner_M`.

Returns `(Mp_tilde, factor)` where `factor` is `nothing` for the unpreconditioned
case.
"""
function transform_patch_matrices(Mp::Vector{<:AbstractMatrix};
                                  preconditioner_M=nothing,
                                  preconditioner_factor=nothing)
    N = _validated_mass_matrix_size(Mp)
    if preconditioner_M === nothing && preconditioner_factor === nothing
        return Mp, nothing
    end

    preconditioner_matrix =
        preconditioner_M === nothing ? nothing :
        _validated_conditioning_matrix(
            preconditioner_M, N, "preconditioner_M")
    fac = if preconditioner_factor === nothing
        lu(preconditioner_matrix)
    else
        _validate_conditioning_factor(
            preconditioner_factor, N, "preconditioner_factor")
    end

    Mp_tilde = Vector{Matrix{ComplexF64}}(undef, length(Mp))
    @inbounds for p in eachindex(Mp)
        transformed = Matrix{ComplexF64}(Mp[p])
        Mp_tilde[p] = _ldiv_reusing_input(fac, transformed)
        all(isfinite, Mp_tilde[p]) ||
            error("transformed patch matrix $p contains non-finite values")
    end
    return Mp_tilde, fac
end

"""
    prepare_conditioned_system(Z_raw, rhs;
                               regularization_alpha=0.0,
                               regularization_R=nothing,
                               preconditioner_M=nothing,
                               preconditioner_factor=nothing)

Build the linear system used by forward/adjoint solves:
  Z_reg = Z_raw + αR
  Z_eff = M^{-1} Z_reg
  rhs_eff = M^{-1} rhs

If no regularization/preconditioning is requested, returns `(Z_raw, rhs, nothing)`.
Returns `(Z_eff, rhs_eff, factor)` where `factor` is the LU factorization used
for preconditioning (or `nothing`).
"""
function prepare_conditioned_system(Z_raw::Matrix{<:Number},
                                    rhs::Vector{<:Number};
                                    regularization_alpha::Float64=0.0,
                                    regularization_R=nothing,
                                    preconditioner_M=nothing,
                                    preconditioner_factor=nothing)
    N = _validate_linear_system_inputs(
        Z_raw, rhs, "conditioned system")
    (isfinite(regularization_alpha) && regularization_alpha >= 0.0) ||
        throw(ArgumentError(
            "regularization_alpha must be finite and nonnegative, got $regularization_alpha"))
    Z_eff = Matrix{ComplexF64}(Z_raw)
    rhs_eff = Vector{ComplexF64}(rhs)

    if regularization_alpha > 0.0
        regularization_R === nothing &&
            throw(ArgumentError(
                "regularization_alpha is positive but regularization_R is nothing"))
        R = _validated_conditioning_matrix(
            regularization_R, N, "regularization_R")
        @inbounds @simd for i in eachindex(Z_eff, R)
            Z_eff[i] += regularization_alpha * R[i]
        end
        all(isfinite, Z_eff) ||
            error("regularized system matrix contains non-finite values")
    end

    if preconditioner_M === nothing && preconditioner_factor === nothing
        return Z_eff, rhs_eff, nothing
    end

    preconditioner_matrix =
        preconditioner_M === nothing ? nothing :
        _validated_conditioning_matrix(
            preconditioner_M, N, "preconditioner_M")
    fac = if preconditioner_factor === nothing
        preconditioner_matrix === nothing &&
            throw(ArgumentError(
                "preconditioner_M is required when preconditioner_factor is nothing"))
        lu(preconditioner_matrix)
    else
        _validate_conditioning_factor(
            preconditioner_factor, N, "preconditioner_factor")
    end
    Z_eff = _ldiv_reusing_input(fac, Z_eff)
    rhs_eff = _ldiv_reusing_input(fac, rhs_eff)
    all(isfinite, Z_eff) ||
        error("conditioned system matrix contains non-finite values")
    all(isfinite, rhs_eff) ||
        error("conditioned RHS contains non-finite values")
    return Z_eff, rhs_eff, fac
end
