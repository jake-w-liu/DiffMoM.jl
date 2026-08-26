# IterativeSolve.jl — GMRES iterative solver via Krylov.jl
#
# Provides iterative solve alternatives to the dense direct factorization
# in Solve.jl, with support for near-field sparse preconditioning.

export solve_gmres, solve_gmres_adjoint

const _DEFAULT_MAX_GMRES_WORKSPACE_BYTES = 512 * 1024 * 1024
# A flagged complex matrix-vector term performs several 4352-bit operations.
# Bound that exceptional work independently of the ordinary Krylov workspace.
const _DEFAULT_MAX_TRUE_RESIDUAL_EXACT_TERMS = 2_000_000

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

@inline function _gmres_stored_leaf(A::AbstractMatrix)
    current = A
    while current isa Adjoint || current isa Transpose ||
          current isa Hermitian || current isa LinearAlgebra.Symmetric
        current = parent(current)
    end
    return current
end

@inline function _gmres_is_stored_dense(A::AbstractMatrix)
    return _gmres_stored_leaf(A) isa StridedMatrix
end

@inline function _gmres_is_stored_sparse(A::AbstractMatrix)
    return _gmres_stored_leaf(A) isa
           SparseArrays.AbstractSparseMatrixCSC
end

@inline function _gmres_is_stored_local_mass(A::AbstractMatrix)
    return A isa LocalMassMatrix ||
           ((A isa Adjoint || A isa Transpose) &&
            parent(A) isa LocalMassMatrix)
end

@inline function _true_residual_sparse_storage_bytes(A::AbstractMatrix)
    sparse_matrix = _gmres_stored_leaf(A)
    structured_wrapper = false
    current = A
    while current !== sparse_matrix
        structured_wrapper |= current isa Hermitian ||
                              current isa LinearAlgebra.Symmetric
        current = parent(current)
    end
    stored_entries = if structured_wrapper
        2 * BigInt(nnz(sparse_matrix))
    else
        BigInt(nnz(sparse_matrix))
    end
    return BigInt(sizeof(eltype(sparse_matrix)) + sizeof(Int)) *
           stored_entries +
           BigInt(sizeof(Int)) * (size(A, 2) + 1)
end

@inline function _true_residual_sparse_materialization_copies(
        A::AbstractMatrix)
    A isa SparseArrays.AbstractSparseMatrixCSC && return 0
    depth = 0
    current = A
    while current isa Adjoint || current isa Transpose ||
          current isa Hermitian || current isa LinearAlgebra.Symmetric
        depth += 1
        current = parent(current)
    end
    current isa SparseArrays.AbstractSparseMatrixCSC || return 0
    # Recursive sparse materialization can retain one completed inner CSC while
    # allocating the outer CSC. Deeper intermediates do not add another live
    # layer, so two full logical copies are the conservative nested peak.
    return depth == 1 ? 1 : 2
end

function _true_residual_workspace_bytes(
        A::AbstractMatrix,
    output_length::Int;
    include_sparse_transpose::Bool=false)
    length_big = BigInt(output_length)
    array_bytes = BigInt(sizeof(ComplexF64)) * length_big
    array_bytes += 2 * cld(length_big, 64) * sizeof(UInt64)
    if _gmres_is_stored_dense(A)
        array_bytes += 2 * BigInt(sizeof(Float64)) * length_big
    elseif _gmres_is_stored_sparse(A) || _gmres_is_stored_local_mass(A)
        array_bytes += 4 * BigInt(sizeof(Float64)) * length_big
    end
    materialization_peak = BigInt(0)
    retained_sparse_output = BigInt(0)
    if _gmres_is_stored_sparse(A) &&
       !(A isa SparseArrays.AbstractSparseMatrixCSC)
        sparse_storage = _true_residual_sparse_storage_bytes(A)
        materialization_peak =
            _true_residual_sparse_materialization_copies(A) * sparse_storage
        retained_sparse_output = sparse_storage
    end
    transpose_bytes = BigInt(0)
    if include_sparse_transpose
        _gmres_is_stored_sparse(A) ||
            error("internal sparse true-residual workspace request used a non-sparse matrix")
        transpose_bytes = _true_residual_sparse_storage_bytes(A)
    end
    retained_phase = array_bytes + retained_sparse_output + transpose_bytes
    total = max(materialization_peak, retained_phase)
    total <= typemax(Int) ||
        throw(ArgumentError(
            "true-residual workspace estimate overflows Int"))
    return Int(total)
end

function _true_residual_dense_fallback_rows(
        A::AbstractMatrix,
        x::AbstractVector,
        product::Vector{ComplexF64})
    # Certify each stored row independently so one ill-conditioned reduction
    # does not move the full residual calculation to BigFloat.
    fallback_rows = falses(length(product))
    exact_rows = falses(length(product))
    real_error_bounds = zeros(Float64, length(product))
    imag_error_bounds = zeros(Float64, length(product))
    error_factor = _ieee_product_error_factor(
        Float64, 2, size(A, 2))
    @inbounds for row in axes(A, 1)
        real_magnitude = 0.0
        imag_magnitude = 0.0
        has_extreme_factor = false
        for column in axes(A, 2)
            matrix_value = A[row, column]
            vector_value = x[column]
            if !iszero(matrix_value) && !iszero(vector_value)
                has_extreme_factor |=
                    _ieee_dense_extreme_factor(matrix_value, Float64) ||
                    _ieee_dense_extreme_factor(vector_value, Float64)
            end
            matrix_real = Float64(real(matrix_value))
            matrix_imag = Float64(imag(matrix_value))
            vector_real = Float64(real(vector_value))
            vector_imag = Float64(imag(vector_value))
            real_magnitude += abs(matrix_real * vector_real) +
                              abs(matrix_imag * vector_imag)
            imag_magnitude += abs(matrix_real * vector_imag) +
                              abs(matrix_imag * vector_real)
        end
        result_value = product[row]
        real_error_bounds[row] = error_factor * real_magnitude
        imag_error_bounds[row] = error_factor * imag_magnitude
        sequential_real, sequential_imag,
        real_operations_were_exact, imag_operations_were_exact =
            _ieee_dense_row_exact_reduction(
                A, x, row, Float64, true)
        exact_rows[row] = !has_extreme_factor &&
                          real_operations_were_exact &&
                          imag_operations_were_exact &&
                          real(result_value) == sequential_real &&
                          imag(result_value) == sequential_imag
        real_is_suspicious = _ieee_product_component_is_suspicious(
            real(result_value), real_magnitude, error_factor)
        imag_is_suspicious = _ieee_product_component_is_suspicious(
            imag(result_value), imag_magnitude, error_factor)
        fallback_rows[row] = has_extreme_factor ||
                             ((real_is_suspicious || imag_is_suspicious) &&
                              !exact_rows[row])
    end
    return (
        fallback=fallback_rows,
        exact=exact_rows,
        real_error=real_error_bounds,
        imag_error=imag_error_bounds,
    )
end

function _true_residual_sparse_fallback_rows(
        A::SparseArrays.AbstractSparseMatrixCSC,
        x::AbstractVector,
        product::Vector{ComplexF64})
    fallback_rows = falses(length(product))
    exact_rows = trues(length(product))
    real_magnitudes = zeros(Float64, length(product))
    imag_magnitudes = zeros(Float64, length(product))
    sequential_real = zeros(Float64, length(product))
    sequential_imag = zeros(Float64, length(product))
    rows = rowvals(A)
    values = nonzeros(A)
    @inbounds for column in axes(A, 2)
        vector_value = x[column]
        vector_real = Float64(real(vector_value))
        vector_imag = Float64(imag(vector_value))
        for position in nzrange(A, column)
            row = rows[position]
            matrix_value = values[position]
            if !iszero(matrix_value) && !iszero(vector_value)
                fallback_rows[row] |=
                    _ieee_dense_extreme_factor(matrix_value, Float64) ||
                    _ieee_dense_extreme_factor(vector_value, Float64)
            end
            matrix_real = Float64(real(matrix_value))
            matrix_imag = Float64(imag(matrix_value))
            real_magnitudes[row] += abs(matrix_real * vector_real) +
                                    abs(matrix_imag * vector_imag)
            imag_magnitudes[row] += abs(matrix_real * vector_imag) +
                                    abs(matrix_imag * vector_real)
            sequential_real[row], first_real_exact =
                _ieee_add_product_with_exactness(
                    sequential_real[row], matrix_real, vector_real, 1.0)
            sequential_real[row], second_real_exact =
                _ieee_add_product_with_exactness(
                    sequential_real[row], matrix_imag, vector_imag, -1.0)
            sequential_imag[row], first_imag_exact =
                _ieee_add_product_with_exactness(
                    sequential_imag[row], matrix_real, vector_imag, 1.0)
            sequential_imag[row], second_imag_exact =
                _ieee_add_product_with_exactness(
                    sequential_imag[row], matrix_imag, vector_real, 1.0)
            exact_rows[row] &= first_real_exact && second_real_exact &&
                               first_imag_exact && second_imag_exact
        end
    end
    error_factor = _ieee_product_error_factor(
        Float64, 2, size(A, 2))
    real_error_bounds = real_magnitudes
    imag_error_bounds = imag_magnitudes
    @inbounds for row in eachindex(product)
        exact_rows[row] &= !fallback_rows[row] &&
                           real(product[row]) == sequential_real[row] &&
                           imag(product[row]) == sequential_imag[row]
        product_is_suspicious =
            _ieee_product_component_is_suspicious(
                real(product[row]), real_error_bounds[row], error_factor) ||
            _ieee_product_component_is_suspicious(
                imag(product[row]), imag_error_bounds[row], error_factor)
        real_error_bounds[row] *= error_factor
        imag_error_bounds[row] *= error_factor
        fallback_rows[row] |= product_is_suspicious && !exact_rows[row]
    end
    return (
        fallback=fallback_rows,
        exact=exact_rows,
        real_error=real_error_bounds,
        imag_error=imag_error_bounds,
    )
end

function _true_residual_local_mass_fallback_rows(
        A::AbstractMatrix,
        x::AbstractVector,
        product::Vector{ComplexF64})
    matrix = (A isa Adjoint || A isa Transpose) ? parent(A) : A
    matrix isa LocalMassMatrix ||
        error("internal LocalMassMatrix residual analysis received $(typeof(A))")
    wrapped = A isa Adjoint || A isa Transpose
    conjugate_values = A isa Adjoint
    fallback_rows = falses(length(product))
    exact_rows = trues(length(product))
    A isa Transpose && fill!(exact_rows, false)
    real_magnitudes = zeros(Float64, length(product))
    imag_magnitudes = zeros(Float64, length(product))
    sequential_real = zeros(Float64, length(product))
    sequential_imag = zeros(Float64, length(product))
    @inbounds for position in eachindex(matrix.vals)
        row = wrapped ? matrix.cols[position] : matrix.rows[position]
        column = wrapped ? matrix.rows[position] : matrix.cols[position]
        stored_value = matrix.vals[position]
        matrix_value = conjugate_values ? conj(stored_value) : stored_value
        vector_value = x[column]
        if !iszero(matrix_value) && !iszero(vector_value)
            fallback_rows[row] |=
                _ieee_dense_extreme_factor(matrix_value, Float64) ||
                _ieee_dense_extreme_factor(vector_value, Float64)
        end
        matrix_real = Float64(real(matrix_value))
        matrix_imag = Float64(imag(matrix_value))
        vector_real = Float64(real(vector_value))
        vector_imag = Float64(imag(vector_value))
        real_magnitudes[row] += abs(matrix_real * vector_real) +
                                abs(matrix_imag * vector_imag)
        imag_magnitudes[row] += abs(matrix_real * vector_imag) +
                                abs(matrix_imag * vector_real)
        sequential_real[row], first_real_exact =
            _ieee_add_product_with_exactness(
                sequential_real[row], matrix_real, vector_real, 1.0)
        sequential_real[row], second_real_exact =
            _ieee_add_product_with_exactness(
                sequential_real[row], matrix_imag, vector_imag, -1.0)
        sequential_imag[row], first_imag_exact =
            _ieee_add_product_with_exactness(
                sequential_imag[row], matrix_real, vector_imag, 1.0)
        sequential_imag[row], second_imag_exact =
            _ieee_add_product_with_exactness(
                sequential_imag[row], matrix_imag, vector_real, 1.0)
        exact_rows[row] &= first_real_exact && second_real_exact &&
                           first_imag_exact && second_imag_exact
    end
    error_factor = _ieee_product_error_factor(
        Float64, 2, size(matrix, 2))
    real_error_bounds = real_magnitudes
    imag_error_bounds = imag_magnitudes
    @inbounds for row in eachindex(product)
        exact_rows[row] &= !fallback_rows[row] &&
                           real(product[row]) == sequential_real[row] &&
                           imag(product[row]) == sequential_imag[row]
        product_is_suspicious =
            _ieee_product_component_is_suspicious(
                real(product[row]), real_error_bounds[row], error_factor) ||
            _ieee_product_component_is_suspicious(
                imag(product[row]), imag_error_bounds[row], error_factor)
        real_error_bounds[row] *= error_factor
        imag_error_bounds[row] *= error_factor
        fallback_rows[row] |= product_is_suspicious && !exact_rows[row]
    end
    return (
        fallback=fallback_rows,
        exact=exact_rows,
        real_error=real_error_bounds,
        imag_error=imag_error_bounds,
    )
end

function _true_residual_fallback_rows(
        A::AbstractMatrix,
        x::AbstractVector,
        product::Vector{ComplexF64})
    if _gmres_is_stored_dense(A)
        return _true_residual_dense_fallback_rows(A, x, product)
    elseif A isa SparseArrays.AbstractSparseMatrixCSC
        return _true_residual_sparse_fallback_rows(A, x, product)
    elseif _gmres_is_stored_local_mass(A)
        return _true_residual_local_mass_fallback_rows(A, x, product)
    end
    return nothing
end

@inline function _true_residual_component_addition_is_exact(
        first::Float64,
        second::Float64,
        combined::Float64)
    (isfinite(first) && isfinite(second) && isfinite(combined)) || return false
    virtual_second = combined - first
    virtual_first = combined - virtual_second
    error = (first - virtual_first) + (second - virtual_second)
    return iszero(error)
end

@inline function _true_residual_difference_is_exact(
        product::ComplexF64,
        rhs::ComplexF64,
        difference::ComplexF64)
    return _true_residual_component_addition_is_exact(
               real(product), -real(rhs), real(difference)) &&
           _true_residual_component_addition_is_exact(
               imag(product), -imag(rhs), imag(difference))
end

@inline function _true_residual_difference_requires_exact(
        product::ComplexF64,
        rhs::ComplexF64,
        difference::ComplexF64,
        real_product_error::Float64,
        imag_product_error::Float64)
    isfinite(difference) || return true
    real_scale = abs(real(product)) + abs(real(rhs))
    imag_scale = abs(imag(product)) + abs(imag(rhs))
    real_bound = real_product_error +
                 _SCALED_SUM_CANCELLATION_FACTOR * real_scale
    imag_bound = imag_product_error +
                 _SCALED_SUM_CANCELLATION_FACTOR * imag_scale
    return !isfinite(real_bound) || !isfinite(imag_bound) ||
           (!iszero(real_bound) &&
            abs(real(difference)) <= real_bound) ||
           (!iszero(imag_bound) &&
            abs(imag(difference)) <= imag_bound)
end

@inline function _true_residual_lassq_component(
        scale::Float64,
        sumsq::Float64,
        component::Float64)
    magnitude = abs(component)
    iszero(magnitude) && return scale, sumsq
    if scale < magnitude
        ratio = scale / magnitude
        return magnitude, 1.0 + sumsq * ratio^2
    end
    ratio = magnitude / scale
    return scale, sumsq + ratio^2
end

function _true_residual_scaled_sumsq(
        values::AbstractVector,
        selected_rows::Union{Nothing,BitVector}=nothing;
        selected::Bool=true)
    scale = 0.0
    sumsq = 1.0
    @inbounds for index in eachindex(values)
        if selected_rows === nothing || selected_rows[index] == selected
            value = ComplexF64(values[index])
            scale, sumsq = _true_residual_lassq_component(
                scale, sumsq, real(value))
            scale, sumsq = _true_residual_lassq_component(
                scale, sumsq, imag(value))
        end
    end
    return scale, iszero(scale) ? 0.0 : sumsq
end

@inline function _true_residual_bigfloat_sumsq(
        scale::Float64,
        sumsq::Float64)
    iszero(scale) && return zero(BigFloat)
    return BigFloat(scale)^2 * BigFloat(sumsq)
end

function _true_residual_exact_term_count(
        A::AbstractMatrix,
        fallback_rows::BitVector)
    if _gmres_is_stored_dense(A)
        return BigInt(count(fallback_rows)) * size(A, 2)
    elseif A isa SparseArrays.AbstractSparseMatrixCSC
        count_terms = BigInt(0)
        rows = rowvals(A)
        @inbounds for position in eachindex(nonzeros(A))
            fallback_rows[rows[position]] && (count_terms += 1)
        end
        return count_terms
    elseif _gmres_is_stored_local_mass(A)
        matrix = (A isa Adjoint || A isa Transpose) ? parent(A) : A
        wrapped = A isa Adjoint || A isa Transpose
        count_terms = BigInt(0)
        @inbounds for position in eachindex(matrix.vals)
            row = wrapped ? matrix.cols[position] : matrix.rows[position]
            fallback_rows[row] && (count_terms += 1)
        end
        return count_terms
    end
    error("internal exact true-residual count received $(typeof(A))")
end

function _enforce_true_residual_exact_work(
        exact_terms::Integer,
        exact_rows::Integer,
        label::AbstractString)
    exact_terms >= 0 && exact_rows >= 0 ||
        error("internal exact true-residual work counts must be nonnegative")
    exact_work = BigInt(exact_terms) + exact_rows
    exact_work <= _DEFAULT_MAX_TRUE_RESIDUAL_EXACT_TERMS ||
        throw(ArgumentError(
            "$label exact true-residual work requires $exact_work terms, " *
            "exceeding the limit of " *
            "$_DEFAULT_MAX_TRUE_RESIDUAL_EXACT_TERMS"))
    return Int(exact_work)
end

function _preflight_true_residual_exact_work(
        A::AbstractMatrix,
        fallback_rows::BitVector,
        label::AbstractString)
    exact_terms = _true_residual_exact_term_count(A, fallback_rows)
    return _enforce_true_residual_exact_work(
        exact_terms, count(fallback_rows), label)
end

@noinline function _mixed_local_mass_true_residual_ratio_bigfloat(
        A::AbstractMatrix,
        x::AbstractVector,
        rhs::AbstractVector,
        residual::Vector{ComplexF64},
        fallback_rows::BitVector)
    matrix = (A isa Adjoint || A isa Transpose) ? parent(A) : A
    matrix isa LocalMassMatrix ||
        error("internal LocalMassMatrix exact residual received $(typeof(A))")
    wrapped = A isa Adjoint || A isa Transpose
    conjugate_values = A isa Adjoint
    order = wrapped ? matrix.col_order : eachindex(matrix.vals)
    residual_scale, residual_sumsq = _true_residual_scaled_sumsq(
        residual, fallback_rows; selected=false)
    rhs_scale, rhs_sumsq = _true_residual_scaled_sumsq(rhs)
    return setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        residual_squared = _true_residual_bigfloat_sumsq(
            residual_scale, residual_sumsq)
        rhs_squared = _true_residual_bigfloat_sumsq(
            rhs_scale, rhs_sumsq)
        position = firstindex(order)
        @inbounds for row in axes(A, 1)
            if fallback_rows[row]
                total = zero(Complex{BigFloat})
                while position <= lastindex(order)
                    entry = order[position]
                    target_row = wrapped ?
                                 matrix.cols[entry] : matrix.rows[entry]
                    target_row == row || break
                    source_column = wrapped ?
                                    matrix.rows[entry] : matrix.cols[entry]
                    stored_value = matrix.vals[entry]
                    matrix_value = conjugate_values ?
                                   conj(stored_value) : stored_value
                    total += Complex{BigFloat}(matrix_value) *
                             Complex{BigFloat}(x[source_column])
                    position += 1
                end
                residual_squared += abs2(
                    total - Complex{BigFloat}(rhs[row]))
            else
                while position <= lastindex(order)
                    entry = order[position]
                    target_row = wrapped ?
                                 matrix.cols[entry] : matrix.rows[entry]
                    target_row == row || break
                    position += 1
                end
            end
        end
        if iszero(rhs_squared)
            return iszero(residual_squared) ? 0.0 : Inf
        end
        return Float64(sqrt(residual_squared / rhs_squared))
    end
end

@noinline function _mixed_true_residual_ratio_bigfloat(
        A::AbstractMatrix,
        x::AbstractVector,
        rhs::AbstractVector,
        residual::Vector{ComplexF64},
        fallback_rows::BitVector)
    _gmres_is_stored_local_mass(A) &&
        return _mixed_local_mass_true_residual_ratio_bigfloat(
            A, x, rhs, residual, fallback_rows)
    residual_scale, residual_sumsq = _true_residual_scaled_sumsq(
        residual, fallback_rows; selected=false)
    rhs_scale, rhs_sumsq = _true_residual_scaled_sumsq(rhs)
    return setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        residual_squared = _true_residual_bigfloat_sumsq(
            residual_scale, residual_sumsq)
        rhs_squared = _true_residual_bigfloat_sumsq(
            rhs_scale, rhs_sumsq)
        @inbounds for row in axes(A, 1)
            if fallback_rows[row]
                total = zero(Complex{BigFloat})
                for column in axes(A, 2)
                    total += Complex{BigFloat}(A[row, column]) *
                             Complex{BigFloat}(x[column])
                end
                residual_squared += abs2(
                    total - Complex{BigFloat}(rhs[row]))
            end
        end
        if iszero(rhs_squared)
            return iszero(residual_squared) ? 0.0 : Inf
        end
        return Float64(sqrt(residual_squared / rhs_squared))
    end
end

@noinline function _mixed_true_residual_ratio_bigfloat(
        A::SparseArrays.AbstractSparseMatrixCSC,
        x::AbstractVector,
        rhs::AbstractVector,
        residual::Vector{ComplexF64},
        fallback_rows::BitVector)
    # CSC is column-oriented. A bounded transpose lets the exceptional path
    # revisit only flagged rows without retaining one BigFloat sum per row.
    row_matrix = copy(transpose(A))
    columns = rowvals(row_matrix)
    values = nonzeros(row_matrix)
    residual_scale, residual_sumsq = _true_residual_scaled_sumsq(
        residual, fallback_rows; selected=false)
    rhs_scale, rhs_sumsq = _true_residual_scaled_sumsq(rhs)
    return setprecision(BigFloat, _IEEE_DENSE_PRODUCT_FALLBACK_PRECISION) do
        residual_squared = _true_residual_bigfloat_sumsq(
            residual_scale, residual_sumsq)
        rhs_squared = _true_residual_bigfloat_sumsq(
            rhs_scale, rhs_sumsq)
        @inbounds for row in axes(A, 1)
            if fallback_rows[row]
                total = zero(Complex{BigFloat})
                for position in nzrange(row_matrix, row)
                    total += Complex{BigFloat}(values[position]) *
                             Complex{BigFloat}(x[columns[position]])
                end
                residual_squared += abs2(
                    total - Complex{BigFloat}(rhs[row]))
            end
        end
        if iszero(rhs_squared)
            return iszero(residual_squared) ? 0.0 : Inf
        end
        return Float64(sqrt(residual_squared / rhs_squared))
    end
end

function _true_residual_ratio(
        A::AbstractMatrix,
        x::AbstractVector,
        rhs::Vector{ComplexF64},
        label::AbstractString)
    base_workspace_bytes = _true_residual_workspace_bytes(A, length(rhs))
    _enforce_payload_limit(
        base_workspace_bytes,
        _DEFAULT_MAX_GMRES_WORKSPACE_BYTES,
        "$label true-residual workspace",
        "true-residual workspace limit",
    )
    residual_operator = if _gmres_is_stored_sparse(A) &&
                           !(A isa SparseArrays.AbstractSparseMatrixCSC)
        sparse(A)
    else
        A
    end
    residual = Vector{ComplexF64}(undef, length(rhs))
    mul!(residual, residual_operator, x)
    analysis = _true_residual_fallback_rows(
        residual_operator, x, residual)
    @inbounds for index in eachindex(residual, rhs)
        product = residual[index]
        difference = product - rhs[index]
        if analysis === nothing
            isfinite(difference) ||
                error("$label true residual is non-finite at index $index.")
        else
            fallback_rows = analysis.fallback
            if !fallback_rows[index] &&
               _true_residual_difference_requires_exact(
                   product,
                   rhs[index],
                   difference,
                   analysis.real_error[index],
                   analysis.imag_error[index],
               )
                fallback_rows[index] =
                    !(analysis.exact[index] &&
                      _true_residual_difference_is_exact(
                          product, rhs[index], difference))
            end
        end
        residual[index] = difference
    end
    if analysis !== nothing && any(analysis.fallback)
        _preflight_true_residual_exact_work(
            residual_operator, analysis.fallback, label)
        if residual_operator isa SparseArrays.AbstractSparseMatrixCSC
            exact_workspace_bytes = _true_residual_workspace_bytes(
                A, length(rhs); include_sparse_transpose=true)
            _enforce_payload_limit(
                exact_workspace_bytes,
                _DEFAULT_MAX_GMRES_WORKSPACE_BYTES,
                "$label sparse true-residual workspace",
                "true-residual workspace limit",
            )
        end
        return _mixed_true_residual_ratio_bigfloat(
            residual_operator, x, rhs, residual, analysis.fallback)
    end
    rhs_norm = norm(rhs)
    residual_norm = norm(residual)
    if iszero(rhs_norm)
        return iszero(residual_norm) ? 0.0 : Inf
    end
    return residual_norm / rhs_norm
end

function _validated_true_residual_limit(
        tol::Float64,
        factor::Float64)
    (isfinite(factor) && factor > 0.0) ||
        throw(ArgumentError(
            "true residual factor must be finite and positive, got $factor"))
    limit = factor * tol
    isfinite(limit) ||
        throw(ArgumentError(
            "true residual limit must be finite, got factor*tol=$limit"))
    return limit
end

function _assert_true_residual(A::AbstractMatrix, x::AbstractVector, rhs::AbstractVector,
                               label::AbstractString;
                               tol::Float64,
                               factor::Float64=100.0)
    limit = _validated_true_residual_limit(tol, factor)
    rhs_c = _as_complex_rhs(rhs)
    relres = _true_residual_ratio(A, x, rhs_c, label)
    isfinite(relres) && relres <= limit && return relres
    error("$label GMRES true residual too large: relative_residual=$relres, " *
          "limit=$limit, tol=$tol, factor=$factor")
end

"""
    solve_gmres(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8,
                maxiter=200, memory=20, max_workspace_bytes=536_870_912,
                verbose=false,
                check_gmres_convergence=true, check_true_residual=true,
                true_residual_factor=100.0)

Solve Z x = rhs using GMRES from Krylov.jl.

If `preconditioner` is an `AbstractPreconditionerData`, it is applied via:
- `precond_side=:left` (default): left preconditioner M in Krylov.gmres
- `precond_side=:right`: right preconditioner N in Krylov.gmres

GMRES is restarted after `memory` inner iterations, so the preflighted Krylov
basis size remains an actual workspace bound rather than an allocation hint.

Returns `(x, stats)` where `stats` is the Krylov.jl convergence info.
By default, an unconverged, non-finite, or excessive true-residual result throws.
Set both checks to `false` only when intentionally inspecting a partial iterate
and its `stats`.
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
                     check_gmres_convergence::Bool=true,
                     check_true_residual::Bool=true,
                     true_residual_factor::Float64=100.0)
    _validate_gmres_options(tol, maxiter, memory, precond_side)
    check_true_residual &&
        _validated_true_residual_limit(tol, true_residual_factor)
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
                                 restart=true,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :right
        N_op = NearFieldOperator(preconditioner)
        x, stats = Krylov.gmres(scaled_Z, scaled_rhs;
                                 N=N_op,
                                 memory=memory,
                                 restart=true,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :left
        M = NearFieldOperator(preconditioner)
        x, stats = Krylov.gmres(scaled_Z, scaled_rhs;
                                 M=M,
                                 memory=memory,
                                 restart=true,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    _gmres_recover_solution!(x, solution_shift, "forward GMRES")
    _assert_finite_linear_vector(x, "forward GMRES solution")
    check_gmres_convergence &&
        _assert_gmres_converged(
            stats, "forward"; tol=tol, maxiter=maxiter)
    check_true_residual &&
        _assert_true_residual(
            Z, x, rhs_c, "forward";
            tol=tol, factor=true_residual_factor)
    return x, stats
end

"""
    solve_gmres_adjoint(Z, rhs; preconditioner=nothing, precond_side=:left,
                        tol=1e-8, maxiter=200, memory=20,
                        max_workspace_bytes=536_870_912, verbose=false,
                        check_gmres_convergence=true, check_true_residual=true,
                        true_residual_factor=100.0)

Solve Z† x = rhs using GMRES, with the adjoint preconditioner Z_nf⁻ᴴ.

The adjoint solve uses the same restarted `memory` workspace contract as the
forward solve.

This is used for the adjoint linear system in sensitivity analysis:
  Z†(θ) λ = ∂Φ/∂I*

Returns `(x, stats)`. By default, an unconverged, non-finite, or excessive
true-residual result throws. Set both checks to `false` only when intentionally
inspecting a partial iterate and its `stats`.
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
                              check_gmres_convergence::Bool=true,
                              check_true_residual::Bool=true,
                              true_residual_factor::Float64=100.0)
    _validate_gmres_options(tol, maxiter, memory, precond_side)
    check_true_residual &&
        _validated_true_residual_limit(tol, true_residual_factor)
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
                                 restart=true,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :right
        N_adj = NearFieldAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(adjoint(scaled_Z), scaled_rhs;
                                 N=N_adj,
                                 memory=memory,
                                 restart=true,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    elseif precond_side == :left
        M_adj = NearFieldAdjointOperator(preconditioner)
        x, stats = Krylov.gmres(adjoint(scaled_Z), scaled_rhs;
                                 M=M_adj,
                                 memory=memory,
                                 restart=true,
                                 rtol=tol, atol=0.0,
                                 itmax=maxiter,
                                 verbose=(verbose ? 1 : 0))
    end
    _gmres_recover_solution!(x, solution_shift, "adjoint GMRES")
    _assert_finite_linear_vector(x, "adjoint GMRES solution")
    check_gmres_convergence &&
        _assert_gmres_converged(
            stats, "adjoint"; tol=tol, maxiter=maxiter)
    check_true_residual &&
        _assert_true_residual(
            adjoint(Z), x, rhs_c, "adjoint";
            tol=tol, factor=true_residual_factor)
    return x, stats
end
