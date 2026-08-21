# QMatrix.jl — Build the Q matrix for the quadratic far-field objective
#
# Q_mn = Σ_q w_q (p†·g_m)* (p†·g_n)
# J(θ) = I† Q I  (radiated power in selected direction/polarization)

export FarFieldQMatrix, SumQMatrix, build_Q, build_Q_operator, apply_Q, pol_linear_x, pol_linear_y, cap_mask, direction_mask

# A primitive G'WGx term contains at most six input factors. Restricting each
# to ±128 binary exponents keeps the product, complex-component additions,
# and any addressable reduction count inside the normal Float64 exponent range.
const _FARFIELD_Q_SAFE_FACTOR_EXPONENT = 128

@inline function _farfield_q_extreme_factor(value::Number)
    scale = max(abs(real(value)), abs(imag(value)))
    isfinite(scale) || return true
    iszero(scale) && return false
    value_exponent = exponent(scale)
    return value_exponent < -_FARFIELD_Q_SAFE_FACTOR_EXPONENT ||
           value_exponent > _FARFIELD_Q_SAFE_FACTOR_EXPONENT
end

function _farfield_q_has_extreme_operator_factor(
        G_mat::Matrix{ComplexF64},
        weights::Vector{Float64},
        pol::Matrix{ComplexF64})
    return any(_farfield_q_extreme_factor, G_mat) ||
           any(_farfield_q_extreme_factor, weights) ||
           any(_farfield_q_extreme_factor, pol)
end

"""
    FarFieldQMatrix

Matrix-free representation of the Hermitian far-field objective matrix
`Q = G' W G`. It stores the radiation-vector matrix and applies `Q*x`
without forming the dense `N x N` matrix.
"""
struct FarFieldQMatrix <: AbstractMatrix{ComplexF64}
    G_mat::Matrix{ComplexF64}
    weights::Vector{Float64}
    pol::Matrix{ComplexF64}
    mask::Union{Nothing,BitVector}
    N::Int
    work::Vector{ComplexF64}
    work_lock::ReentrantLock
    extreme_operator_factor::Bool
end

FarFieldQMatrix(G_mat::Matrix{ComplexF64},
                weights::Vector{Float64},
                pol::Matrix{ComplexF64},
                mask::Union{Nothing,BitVector},
                N::Int) =
    FarFieldQMatrix(
        G_mat, weights, pol, mask, N,
        zeros(ComplexF64, N), ReentrantLock(),
        _farfield_q_has_extreme_operator_factor(
            G_mat, weights, pol))

FarFieldQMatrix(G_mat::Matrix{ComplexF64},
                weights::Vector{Float64},
                pol::Matrix{ComplexF64},
                mask::Union{Nothing,BitVector},
                N::Int,
                work::Vector{ComplexF64},
                work_lock::ReentrantLock) =
    FarFieldQMatrix(
        G_mat, weights, pol, mask, N, work, work_lock,
        _farfield_q_has_extreme_operator_factor(
            G_mat, weights, pol))

# A primitive real term in G'WGx contains up to six finite Float64 factors.
# Their exact product needs at most 12588 bits; the number of addressable
# quadrature/input terms adds fewer than 70 bits. Keep the remainder as guard
# precision in the exceptional path.
const _FARFIELD_Q_FALLBACK_PRECISION = 12800
const _FARFIELD_Q_SCALED_OUTPUT_FALLBACK_PRECISION = 4352

@inline function _farfield_q_has_subnormal_component(value::ComplexF64)
    scale = max(abs(real(value)), abs(imag(value)))
    return !iszero(scale) && scale < floatmin(Float64)
end

@inline function _farfield_q_product_underflowed(
        left::Number,
        right::Number,
        product::ComplexF64)
    product_scale = max(abs(real(product)), abs(imag(product)))
    product_scale >= floatmin(Float64) && return false
    return !iszero(product_scale) ||
           (!iszero(left) && !iszero(right))
end

@inline function _farfield_q_projection_underflowed(
        polarization::SVector{3,ComplexF64},
        radiation::SVector{3,ComplexF64},
        projection::ComplexF64)
    _farfield_q_has_subnormal_component(projection) && return true
    iszero(projection) || return false
    @inbounds for component in 1:3
        left = conj(polarization[component])
        right = radiation[component]
        term = left * right
        _farfield_q_product_underflowed(left, right, term) && return true
    end
    return false
end

@inline function _farfield_q_projection_bigfloat(
        Q::FarFieldQMatrix,
        q::Int,
        column::Int)
    offset = 3 * (q - 1)
    total = zero(Complex{BigFloat})
    @inbounds for component in 1:3
        total += conj(Complex{BigFloat}(Q.pol[component, q])) *
                 Complex{BigFloat}(Q.G_mat[offset + component, column])
    end
    return total
end

@noinline function _farfield_q_product_bigfloat!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64})
    return setprecision(BigFloat, _FARFIELD_Q_FALLBACK_PRECISION) do
        quadrature_count = length(Q.weights)
        if Q.N <= quadrature_count
            totals = fill(zero(Complex{BigFloat}), Q.N)
            @inbounds for q in 1:quadrature_count
                Q.mask !== nothing && !Q.mask[q] && continue
                projected_input = zero(Complex{BigFloat})
                for n in 1:Q.N
                    projected_input +=
                        _farfield_q_projection_bigfloat(Q, q, n) *
                        Complex{BigFloat}(input[n])
                end
                weight = BigFloat(Q.weights[q])
                for m in 1:Q.N
                    totals[m] += weight *
                        conj(_farfield_q_projection_bigfloat(Q, q, m)) *
                        projected_input
                end
            end
            @inbounds for m in 1:Q.N
                converted = ComplexF64(totals[m])
                isfinite(converted) ||
                    throw(OverflowError(
                        "FarFieldQMatrix product is outside the " *
                        "representable ComplexF64 range at row $m."))
                result[m] = converted
            end
        else
            projected_inputs = fill(
                zero(Complex{BigFloat}), quadrature_count)
            @inbounds for q in 1:quadrature_count
                Q.mask !== nothing && !Q.mask[q] && continue
                for n in 1:Q.N
                    projected_inputs[q] +=
                        _farfield_q_projection_bigfloat(Q, q, n) *
                        Complex{BigFloat}(input[n])
                end
            end
            @inbounds for m in 1:Q.N
                total = zero(Complex{BigFloat})
                for q in 1:quadrature_count
                    Q.mask !== nothing && !Q.mask[q] && continue
                    total += BigFloat(Q.weights[q]) *
                        conj(_farfield_q_projection_bigfloat(Q, q, m)) *
                        projected_inputs[q]
                end
                converted = ComplexF64(total)
                isfinite(converted) ||
                    throw(OverflowError(
                        "FarFieldQMatrix product is outside the " *
                        "representable ComplexF64 range at row $m."))
                result[m] = converted
            end
        end
        return result
    end
end

@inline function _farfield_q_requires_checked_products(
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64})
    # The public matrix stores mutable arrays.  Recompute the classification so
    # caller mutation cannot invalidate the construction-time cache and route
    # an extreme product through the unchecked kernel.
    (Q.extreme_operator_factor ||
     _farfield_q_has_extreme_operator_factor(
         Q.G_mat, Q.weights, Q.pol)) && return true
    return any(_farfield_q_extreme_factor, input)
end

function _farfield_q_product_unchecked!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64})
    fill!(result, zero(ComplexF64))
    quadrature_count = length(Q.weights)
    @inbounds for q in 1:quadrature_count
        Q.mask !== nothing && !Q.mask[q] && continue
        offset = 3 * (q - 1)
        polarization = SVector{3,ComplexF64}(
            Q.pol[1, q], Q.pol[2, q], Q.pol[3, q])
        projected_input = zero(ComplexF64)
        for n in 1:Q.N
            radiation = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, n],
                Q.G_mat[offset + 2, n],
                Q.G_mat[offset + 3, n])
            projected_input += dot(polarization, radiation) * input[n]
        end
        isfinite(projected_input) ||
            return _farfield_q_product_bigfloat!(result, Q, input)

        weight = Q.weights[q]
        for m in 1:Q.N
            radiation = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, m],
                Q.G_mat[offset + 2, m],
                Q.G_mat[offset + 3, m])
            result[m] +=
                weight * conj(dot(polarization, radiation)) *
                projected_input
        end
    end
    all(isfinite, result) ||
        return _farfield_q_product_bigfloat!(result, Q, input)
    return result
end

function _farfield_q_product_checked!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64})
    fill!(result, zero(ComplexF64))
    quadrature_count = length(Q.weights)
    @inbounds for q in 1:quadrature_count
        Q.mask !== nothing && !Q.mask[q] && continue
        offset = 3 * (q - 1)
        polarization = SVector{3,ComplexF64}(
            Q.pol[1, q], Q.pol[2, q], Q.pol[3, q])
        projected_input = zero(ComplexF64)
        for n in 1:Q.N
            radiation = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, n],
                Q.G_mat[offset + 2, n],
                Q.G_mat[offset + 3, n])
            projection = dot(polarization, radiation)
            _farfield_q_projection_underflowed(
                polarization, radiation, projection) &&
                return _farfield_q_product_bigfloat!(result, Q, input)
            term = projection * input[n]
            _farfield_q_product_underflowed(
                projection, input[n], term) &&
                return _farfield_q_product_bigfloat!(result, Q, input)
            projected_input += term
            _farfield_q_has_subnormal_component(projected_input) &&
                return _farfield_q_product_bigfloat!(result, Q, input)
        end
        isfinite(projected_input) ||
            return _farfield_q_product_bigfloat!(result, Q, input)

        weight = Q.weights[q]
        for m in 1:Q.N
            radiation = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, m],
                Q.G_mat[offset + 2, m],
                Q.G_mat[offset + 3, m])
            projection = dot(polarization, radiation)
            _farfield_q_projection_underflowed(
                polarization, radiation, projection) &&
                return _farfield_q_product_bigfloat!(result, Q, input)
            weighted_projection = weight * conj(projection)
            _farfield_q_product_underflowed(
                weight, conj(projection), weighted_projection) &&
                return _farfield_q_product_bigfloat!(result, Q, input)
            contribution = weighted_projection * projected_input
            _farfield_q_product_underflowed(
                weighted_projection, projected_input, contribution) &&
                return _farfield_q_product_bigfloat!(result, Q, input)
            result[m] += contribution
            _farfield_q_has_subnormal_component(result[m]) &&
                return _farfield_q_product_bigfloat!(result, Q, input)
        end
    end
    all(isfinite, result) ||
        return _farfield_q_product_bigfloat!(result, Q, input)
    return result
end

function _farfield_q_product!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64})
    if _farfield_q_requires_checked_products(Q, input)
        return _farfield_q_product_checked!(result, Q, input)
    end
    return _farfield_q_product_unchecked!(result, Q, input)
end

@noinline function _farfield_q_scaled_output_bigfloat(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    return setprecision(
            BigFloat, _FARFIELD_Q_SCALED_OUTPUT_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(alpha_scale) *
                Complex{BigFloat}(value)
        if !overwrite
            total += Complex{BigFloat}(beta_scale) *
                     Complex{BigFloat}(previous)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "FarFieldQMatrix scaled output is outside the " *
                "representable ComplexF64 range at row $row."))
        return converted
    end
end

@inline function _farfield_q_scaled_output(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    needs_fallback =
        _ieee_dense_extreme_factor(value, Float64) ||
        _ieee_dense_extreme_factor(alpha_scale, Float64) ||
        (!overwrite &&
         (_ieee_dense_extreme_factor(previous, Float64) ||
          _ieee_dense_extreme_factor(beta_scale, Float64)))
    needs_fallback && return _farfield_q_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, overwrite, row)
    alpha_term = alpha_scale * value
    if overwrite
        converted = ComplexF64(alpha_term)
        return isfinite(converted) ? converted :
               _farfield_q_scaled_output_bigfloat(
                   value, previous, alpha_scale, beta_scale, true, row)
    end
    beta_term = beta_scale * previous
    combined = alpha_term + beta_term
    magnitude_sum =
        max(abs(real(alpha_term)), abs(imag(alpha_term))) +
        max(abs(real(beta_term)), abs(imag(beta_term)))
    converted = ComplexF64(combined)
    if isfinite(converted) && isfinite(magnitude_sum) &&
       !_scaled_sum_requires_exact(alpha_term, beta_term, combined)
        return converted
    end
    return _farfield_q_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, false, row)
end

const _SUM_Q_PRODUCT_FALLBACK_PRECISION = 4608
const _SUM_Q_SCALED_OUTPUT_FALLBACK_PRECISION = 4352

struct SumQMatrix{A<:AbstractMatrix{ComplexF64},B<:AbstractMatrix{ComplexF64}} <: AbstractMatrix{ComplexF64}
    A::A
    B::B
    work_a::Vector{ComplexF64}
    work_b::Vector{ComplexF64}
    work_lock::ReentrantLock
end

function SumQMatrix{A,B}(matrix_a::A, matrix_b::B) where {
        A<:AbstractMatrix{ComplexF64},B<:AbstractMatrix{ComplexF64}}
    size(matrix_a) == size(matrix_b) ||
        throw(DimensionMismatch(
            "summed Q matrices must have the same size"))
    row_count = size(matrix_a, 1)
    return SumQMatrix{A,B}(
        matrix_a, matrix_b,
        zeros(ComplexF64, row_count),
        zeros(ComplexF64, row_count),
        ReentrantLock())
end

SumQMatrix(matrix_a::A, matrix_b::B) where {
        A<:AbstractMatrix{ComplexF64},B<:AbstractMatrix{ComplexF64}} =
    SumQMatrix{A,B}(matrix_a, matrix_b)

function sum_q_matrix(A::AbstractMatrix{ComplexF64}, B::AbstractMatrix{ComplexF64})
    return SumQMatrix(A, B)
end

Base.size(Q::SumQMatrix) = size(Q.A)
Base.eltype(::SumQMatrix) = ComplexF64

@noinline function _sum_q_entry_bigfloat(
        value_a::ComplexF64,
        value_b::ComplexF64,
        row::Int,
        column::Int)
    return setprecision(BigFloat, _SUM_Q_PRODUCT_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(value_a) + Complex{BigFloat}(value_b)
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "SumQMatrix entry is outside the representable " *
                "ComplexF64 range at index ($row, $column)."))
        return converted
    end
end

function Base.getindex(Q::SumQMatrix, row::Int, column::Int)
    value_a = Q.A[row, column]
    value_b = Q.B[row, column]
    combined = value_a + value_b
    real_magnitude = abs(real(value_a)) + abs(real(value_b))
    imag_magnitude = abs(imag(value_a)) + abs(imag(value_b))
    if isfinite(combined) &&
       isfinite(real_magnitude) &&
       isfinite(imag_magnitude) &&
       !_scaled_sum_requires_exact(value_a, value_b, combined)
        return combined
    end
    return _sum_q_entry_bigfloat(
        value_a, value_b, row, column)
end

@noinline function _sum_q_scaled_product_bigfloat!(
        output::Vector{ComplexF64},
        Q::SumQMatrix,
        input::AbstractVector{ComplexF64},
        previous::AbstractVector{ComplexF64},
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool)
    return setprecision(BigFloat, _SUM_Q_PRODUCT_FALLBACK_PRECISION) do
        alpha_big = Complex{BigFloat}(alpha_scale)
        beta_big = overwrite ? zero(Complex{BigFloat}) :
                   Complex{BigFloat}(beta_scale)
        @inbounds for row in eachindex(output)
            product = zero(Complex{BigFloat})
            for column in axes(Q, 2)
                input_value = Complex{BigFloat}(input[column])
                product +=
                    Complex{BigFloat}(Q.A[row, column]) * input_value
                product +=
                    Complex{BigFloat}(Q.B[row, column]) * input_value
            end
            total = alpha_big * product
            if !overwrite
                total += beta_big * Complex{BigFloat}(previous[row])
            end
            converted = ComplexF64(total)
            isfinite(converted) ||
                throw(OverflowError(
                    "SumQMatrix scaled product is outside the " *
                    "representable ComplexF64 range at row $row."))
            output[row] = converted
        end
        return output
    end
end

@noinline function _sum_q_scaled_output_bigfloat(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    return setprecision(
            BigFloat, _SUM_Q_SCALED_OUTPUT_FALLBACK_PRECISION) do
        total = Complex{BigFloat}(alpha_scale) *
                Complex{BigFloat}(value)
        if !overwrite
            total += Complex{BigFloat}(beta_scale) *
                     Complex{BigFloat}(previous)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "SumQMatrix scaled output is outside the " *
                "representable ComplexF64 range at row $row."))
        return converted
    end
end

@inline function _sum_q_scaled_output(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    needs_fallback =
        _ieee_dense_extreme_factor(value, Float64) ||
        _ieee_dense_extreme_factor(alpha_scale, Float64) ||
        (!overwrite &&
         (_ieee_dense_extreme_factor(previous, Float64) ||
          _ieee_dense_extreme_factor(beta_scale, Float64)))
    needs_fallback && return _sum_q_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, overwrite, row)
    alpha_term = alpha_scale * value
    if overwrite
        converted = ComplexF64(alpha_term)
        return isfinite(converted) ? converted :
               _sum_q_scaled_output_bigfloat(
                   value, previous, alpha_scale, beta_scale, true, row)
    end

    beta_term = beta_scale * previous
    combined = alpha_term + beta_term
    real_magnitude = abs(real(alpha_term)) + abs(real(beta_term))
    imag_magnitude = abs(imag(alpha_term)) + abs(imag(beta_term))
    converted = ComplexF64(combined)
    if isfinite(converted) &&
       isfinite(real_magnitude) &&
       isfinite(imag_magnitude) &&
       !_scaled_sum_requires_exact(alpha_term, beta_term, combined)
        return converted
    end
    return _sum_q_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, false, row)
end

function _sum_q_child_products!(
        Q::SumQMatrix,
        input::AbstractVector{ComplexF64})
    try
        mul!(Q.work_a, Q.A, input)
        mul!(Q.work_b, Q.B, input)
    catch error
        error isa OverflowError || rethrow()
        return false
    end
    return all(isfinite, Q.work_a) && all(isfinite, Q.work_b)
end

@inline function _sum_q_matrix_has_extreme_factor(matrix::Matrix{ComplexF64})
    return any(value -> _ieee_dense_extreme_factor(value, Float64), matrix)
end

@inline function _sum_q_matrix_has_extreme_factor(
        matrix::SparseArrays.AbstractSparseMatrixCSC{ComplexF64})
    return any(
        value -> _ieee_dense_extreme_factor(value, Float64),
        nonzeros(matrix))
end

@inline _sum_q_matrix_has_extreme_factor(matrix::FarFieldQMatrix) =
    matrix.extreme_operator_factor ||
    _farfield_q_has_extreme_operator_factor(
        matrix.G_mat, matrix.weights, matrix.pol)

@inline function _sum_q_matrix_has_extreme_factor(matrix::SumQMatrix)
    return _sum_q_matrix_has_extreme_factor(matrix.A) ||
           _sum_q_matrix_has_extreme_factor(matrix.B)
end

# Generic matrices may be matrix-free: enumerating their entries here changes
# an O(N) `mul!` into O(N^2) work (and an entry can itself be expensive).
# Storage-backed matrix types and DiffMoM's structured types have specialized
# methods above.  For an unknown matrix, inspect the computed child product and
# fail over only if that product exposes an exceptional result.
@inline _sum_q_matrix_has_extreme_factor(::AbstractMatrix) = false

@inline _sum_q_exact_entries_supported(::Matrix{ComplexF64}) = true
@inline _sum_q_exact_entries_supported(
    ::SparseArrays.AbstractSparseMatrixCSC{ComplexF64}) = true
@inline _sum_q_exact_entries_supported(::FarFieldQMatrix) = true
@inline _sum_q_exact_entries_supported(Q::SumQMatrix) =
    _sum_q_exact_entries_supported(Q.A) &&
    _sum_q_exact_entries_supported(Q.B)
@inline _sum_q_exact_entries_supported(::AbstractMatrix) = false

@inline function _sum_q_combination_requires_fallback(
        first::ComplexF64, second::ComplexF64, combined::ComplexF64)
    _scaled_sum_requires_exact(first, second, combined) && return true
    @inbounds for component in 1:2
        first_part = component == 1 ? real(first) : imag(first)
        second_part = component == 1 ? real(second) : imag(second)
        combined_part = component == 1 ? real(combined) : imag(combined)
        magnitude = abs(first_part) + abs(second_part)
        (!isfinite(magnitude) ||
         (!iszero(magnitude) && abs(combined_part) < floatmin(Float64))) &&
            return true
    end
    return false
end

@inline function _sum_q_requires_exact_product(
        Q::SumQMatrix, input::AbstractVector{ComplexF64})
    any(value -> _ieee_dense_extreme_factor(value, Float64), input) &&
        return true
    return _sum_q_matrix_has_extreme_factor(Q.A) ||
           _sum_q_matrix_has_extreme_factor(Q.B)
end

function LinearAlgebra.mul!(result::AbstractVector{ComplexF64},
                            Q::SumQMatrix,
                            x::AbstractVector{ComplexF64})
    return LinearAlgebra.mul!(
        result, Q, x, one(ComplexF64), zero(ComplexF64))
end

function LinearAlgebra.mul!(result::AbstractVector{ComplexF64},
                            Q::SumQMatrix,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    length(result) == size(Q, 1) || throw(DimensionMismatch("result length $(length(result)) != $(size(Q, 1))"))
    length(x) == size(Q, 2) || throw(DimensionMismatch("input length $(length(x)) != $(size(Q, 2))"))
    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(result, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            lock(Q.work_lock)
            try
                @inbounds for row in eachindex(result)
                    Q.work_b[row] = _sum_q_scaled_output(
                        zero(ComplexF64), result[row], alpha_scale,
                        beta_scale, false, row)
                end
                copyto!(result, Q.work_b)
            finally
                unlock(Q.work_lock)
            end
        end
        return result
    end
    xread = Base.mightalias(result, x) ? copy(x) : x
    overwrite = iszero(beta_scale)
    lock(Q.work_lock)
    try
        fallback_required = _sum_q_requires_exact_product(Q, xread) ||
                            !_sum_q_child_products!(Q, xread)
        if !fallback_required
            @inbounds for row in eachindex(Q.work_a)
                value_a = Q.work_a[row]
                value_b = Q.work_b[row]
                combined = value_a + value_b
                real_magnitude = abs(real(value_a)) + abs(real(value_b))
                imag_magnitude = abs(imag(value_a)) + abs(imag(value_b))
                if !isfinite(combined) ||
                   !isfinite(real_magnitude) ||
                   !isfinite(imag_magnitude) ||
                   _sum_q_combination_requires_fallback(
                       value_a, value_b, combined)
                    fallback_required = true
                    break
                end
                Q.work_a[row] = combined
            end
        end

        if fallback_required
            (_sum_q_exact_entries_supported(Q.A) &&
             _sum_q_exact_entries_supported(Q.B)) ||
                throw(OverflowError(
                    "SumQMatrix cannot certify an exceptional product for " *
                    "a generic matrix-free child without exact-entry support"))
            _sum_q_scaled_product_bigfloat!(
                Q.work_b, Q, xread, result,
                alpha_scale, beta_scale, overwrite)
        else
            @inbounds for row in eachindex(Q.work_a)
                Q.work_b[row] = _sum_q_scaled_output(
                    Q.work_a[row], result[row],
                    alpha_scale, beta_scale, overwrite, row)
            end
        end
        copyto!(result, Q.work_b)
    finally
        unlock(Q.work_lock)
    end
    return result
end

Base.size(Q::FarFieldQMatrix) = (Q.N, Q.N)
Base.eltype(::FarFieldQMatrix) = ComplexF64

@noinline function _farfield_q_entry_bigfloat(
        Q::FarFieldQMatrix,
        m::Int,
        n::Int)
    return setprecision(BigFloat, _FARFIELD_Q_FALLBACK_PRECISION) do
        total = zero(Complex{BigFloat})
        @inbounds for q in eachindex(Q.weights)
            Q.mask !== nothing && !Q.mask[q] && continue
            total += BigFloat(Q.weights[q]) *
                conj(_farfield_q_projection_bigfloat(Q, q, m)) *
                _farfield_q_projection_bigfloat(Q, q, n)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "FarFieldQMatrix entry is outside the representable " *
                "ComplexF64 range at index ($m, $n)."))
        return converted
    end
end

function _validate_q_inputs(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                            pol::Matrix{ComplexF64}, mask)
    NΩ = _validate_sph_grid(grid)
    size(G_mat, 1) == 3 * NΩ ||
        throw(DimensionMismatch(
            "G_mat has $(size(G_mat, 1)) rows, expected $(3 * NΩ)"))
    size(pol) == (3, NΩ) ||
        throw(DimensionMismatch(
            "pol has size $(size(pol)), expected (3, $NΩ)"))
    all(isfinite, G_mat) ||
        throw(ArgumentError("G_mat must contain only finite values"))
    all(isfinite, pol) ||
        throw(ArgumentError("pol must contain only finite values"))
    if mask !== nothing
        mask isa AbstractVector{Bool} ||
            throw(ArgumentError("mask must be a boolean vector"))
        length(mask) == NΩ ||
            throw(DimensionMismatch(
                "mask length $(length(mask)) != $NΩ"))
    end
    return NΩ, size(G_mat, 2)
end

function Base.getindex(Q::FarFieldQMatrix, m::Int, n::Int)
    1 <= m <= Q.N || throw(BoundsError(Q, (m, n)))
    1 <= n <= Q.N || throw(BoundsError(Q, (m, n)))
    NΩ = length(Q.weights)
    if !(Q.extreme_operator_factor ||
         _farfield_q_has_extreme_operator_factor(
             Q.G_mat, Q.weights, Q.pol))
        val = zero(ComplexF64)
        @inbounds for q in 1:NΩ
            Q.mask !== nothing && !Q.mask[q] && continue
            offset = 3 * (q - 1)
            polarization = SVector{3,ComplexF64}(
                Q.pol[1, q], Q.pol[2, q], Q.pol[3, q])
            radiation_m = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, m],
                Q.G_mat[offset + 2, m],
                Q.G_mat[offset + 3, m])
            radiation_n = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, n],
                Q.G_mat[offset + 2, n],
                Q.G_mat[offset + 3, n])
            val += Q.weights[q] *
                   conj(dot(polarization, radiation_m)) *
                   dot(polarization, radiation_n)
        end
        return isfinite(val) ? val :
               _farfield_q_entry_bigfloat(Q, m, n)
    end

    val = zero(ComplexF64)
    @inbounds for q in 1:NΩ
        if Q.mask !== nothing && !Q.mask[q]
            continue
        end
        offset = 3 * (q - 1)
        polarization = SVector{3,ComplexF64}(
            Q.pol[1, q], Q.pol[2, q], Q.pol[3, q])
        radiation_m = SVector{3,ComplexF64}(
            Q.G_mat[offset + 1, m],
            Q.G_mat[offset + 2, m],
            Q.G_mat[offset + 3, m])
        radiation_n = SVector{3,ComplexF64}(
            Q.G_mat[offset + 1, n],
            Q.G_mat[offset + 2, n],
            Q.G_mat[offset + 3, n])
        projection_m = dot(polarization, radiation_m)
        projection_n = dot(polarization, radiation_n)
        (_farfield_q_projection_underflowed(
             polarization, radiation_m, projection_m) ||
         _farfield_q_projection_underflowed(
             polarization, radiation_n, projection_n)) &&
            return _farfield_q_entry_bigfloat(Q, m, n)
        weighted_projection = Q.weights[q] * conj(projection_m)
        _farfield_q_product_underflowed(
            Q.weights[q], conj(projection_m), weighted_projection) &&
            return _farfield_q_entry_bigfloat(Q, m, n)
        contribution = weighted_projection * projection_n
        _farfield_q_product_underflowed(
            weighted_projection, projection_n, contribution) &&
            return _farfield_q_entry_bigfloat(Q, m, n)
        val += contribution
        _farfield_q_has_subnormal_component(val) &&
            return _farfield_q_entry_bigfloat(Q, m, n)
    end
    return isfinite(val) ? val : _farfield_q_entry_bigfloat(Q, m, n)
end

function LinearAlgebra.mul!(result::AbstractVector{ComplexF64},
                            Q::FarFieldQMatrix,
                            I_coeffs::AbstractVector{ComplexF64})
    return LinearAlgebra.mul!(
        result, Q, I_coeffs, one(ComplexF64), zero(ComplexF64))
end

function LinearAlgebra.mul!(result::AbstractVector{ComplexF64},
                            Q::FarFieldQMatrix,
                            I_coeffs::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    length(result) == Q.N || throw(DimensionMismatch("result length $(length(result)) != $(Q.N)"))
    length(I_coeffs) == Q.N || throw(DimensionMismatch("input length $(length(I_coeffs)) != $(Q.N)"))
    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(result, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            @inbounds for m in eachindex(result)
                result[m] = _farfield_q_scaled_output(
                    zero(ComplexF64), result[m], alpha_scale,
                    beta_scale, false, m)
            end
        end
        return result
    end
    input_read = Base.mightalias(result, I_coeffs) ? copy(I_coeffs) : I_coeffs
    overwrite = iszero(beta_scale)
    lock(Q.work_lock)
    try
        _farfield_q_product!(Q.work, Q, input_read)
        @inbounds for m in eachindex(result)
            result[m] = _farfield_q_scaled_output(
                Q.work[m], result[m], alpha_scale, beta_scale,
                overwrite, m)
        end
    finally
        unlock(Q.work_lock)
    end
    return result
end

@noinline function _build_q_checked(
        G_mat::Matrix{ComplexF64},
        weights::Vector{Float64},
        pol::Matrix{ComplexF64},
        mask,
        N::Int)
    mask_copy = mask === nothing ? nothing : BitVector(mask)
    operator = FarFieldQMatrix(
        G_mat, weights, pol, mask_copy, N,
        ComplexF64[], ReentrantLock(), true)
    result = zeros(ComplexF64, N, N)
    @inbounds for column in 1:N
        for row in 1:column
            value = operator[row, column]
            result[row, column] = value
            result[column, row] = conj(value)
        end
    end
    return result
end

"""
    build_Q(G_mat, grid, pol;
            mask=nothing,
            max_work_bytes=2_000_000_000)

Build the Hermitian PSD matrix Q from radiation vectors and polarization.

  G_mat: (3*NΩ, N) radiation vector matrix
  grid:  SphGrid with quadrature weights
  pol:   (3, NΩ) complex polarization vectors (unit, transverse to r̂)
  mask:  optional BitVector of length NΩ selecting target directions

Returns Q ∈ C^{N×N}, Hermitian positive semidefinite.
"""
function build_Q(
        G_mat::Matrix{ComplexF64}, grid::SphGrid,
        pol::Matrix{ComplexF64};
        mask=nothing,
        max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    NΩ, N = _validate_q_inputs(G_mat, grid, pol, mask)
    projection_bytes = _checked_array_payload_bytes(
        ComplexF64, NΩ, N; label="build_Q projection workspace")
    matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, N, N; label="build_Q output matrix")
    checked_products = _farfield_q_has_extreme_operator_factor(
        G_mat, grid.w, pol)
    # The checked path snapshots an optional mask in a BitVector-backed
    # FarFieldQMatrix while it fills the dense result.  Charge the chunk array
    # as simultaneous workspace; otherwise a limit equal to the output alone
    # silently permits extra operation-owned storage.
    mask_copy_bytes = checked_products && mask !== nothing ?
        _checked_array_payload_bytes(
            UInt64, cld(NΩ, 8sizeof(UInt64));
            label="build_Q checked mask workspace") : 0
    work_bytes = try
        checked_products ?
            Base.Checked.checked_add(matrix_bytes, mask_copy_bytes) :
            Base.Checked.checked_add(projection_bytes, matrix_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("build_Q raw-work estimate overflows Int"))
    end
    _enforce_payload_limit(
        work_bytes, max_work_bytes,
        "build_Q dense output and workspace", "max_work_bytes")
    if checked_products
        return _build_q_checked(
            G_mat, grid.w, pol, mask, N)
    end

    # Compute scalar projections: y_q_n = p†(r̂_q) · g_n(r̂_q)
    # y is (NΩ, N)
    y = zeros(ComplexF64, NΩ, N)
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        p = SVector{3,ComplexF64}(pol[1, q], pol[2, q], pol[3, q])  # avoid slice alloc
        idx = 3 * (q - 1)
        for n in 1:N
            gn = SVector{3,ComplexF64}(G_mat[idx+1, n], G_mat[idx+2, n], G_mat[idx+3, n])
            y[q, n] = dot(p, gn)
        end
    end

    # Q_mn = Σ_q w_q conj(y_qm) y_qn
    Q = zeros(ComplexF64, N, N)
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        wq = grid.w[q]
        for m in 1:N
            ym = conj(y[q, m])
            for n in 1:N
                Q[m, n] += wq * ym * y[q, n]
            end
        end
    end

    return Q
end

"""
    build_Q_operator(G_mat, grid, pol; mask=nothing)

Build a matrix-free far-field objective operator with the same action as
`build_Q(G_mat, grid, pol; mask)`, but without forming the dense matrix.
"""
function build_Q_operator(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                          pol::Matrix{ComplexF64}; mask=nothing)
    _, N = _validate_q_inputs(G_mat, grid, pol, mask)
    mask_copy = mask === nothing ? nothing : BitVector(mask)
    return FarFieldQMatrix(G_mat, copy(grid.w), pol, mask_copy, N)
end

"""
    apply_Q(G_mat, grid, pol, I_coeffs; mask=nothing)

Apply Q*I without forming Q explicitly.
Returns Q*I ∈ C^N.
"""
function apply_Q(G_mat::Matrix{ComplexF64}, grid::SphGrid,
                 pol::Matrix{ComplexF64}, I_coeffs::Vector{ComplexF64};
                 mask=nothing)
    NΩ, N = _validate_q_inputs(G_mat, grid, pol, mask)
    length(I_coeffs) == N ||
        throw(DimensionMismatch(
            "I_coeffs length $(length(I_coeffs)) != $N"))
    all(isfinite, I_coeffs) ||
        throw(ArgumentError(
            "I_coeffs must contain only finite values"))

    result = zeros(ComplexF64, N)
    extreme_operator_factor =
        _farfield_q_has_extreme_operator_factor(
            G_mat, grid.w, pol)
    if extreme_operator_factor ||
       any(_farfield_q_extreme_factor, I_coeffs)
        mask_copy = mask === nothing ? nothing : BitVector(mask)
        operator = FarFieldQMatrix(
            G_mat, grid.w, pol, mask_copy, N,
            result, ReentrantLock(), extreme_operator_factor)
        return _farfield_q_product!(
            result, operator, I_coeffs)
    end
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        p = SVector{3,ComplexF64}(
            pol[1, q], pol[2, q], pol[3, q])
        wq = grid.w[q]

        # Compute y_q = p† · E∞(r̂_q) = Σ_n I_n (p† · g_n)
        yq = zero(ComplexF64)
        for n in 1:N
            idx = 3 * (q - 1)
            gn = SVector{3,ComplexF64}(G_mat[idx+1, n], G_mat[idx+2, n], G_mat[idx+3, n])
            yq += dot(p, gn) * I_coeffs[n]
        end

        # Accumulate: (Q*I)_m += w_q conj(p†·g_m) y_q
        for m in 1:N
            idx = 3 * (q - 1)
            gm = SVector{3,ComplexF64}(G_mat[idx+1, m], G_mat[idx+2, m], G_mat[idx+3, m])
            result[m] += wq * conj(dot(p, gm)) * yq
        end
    end

    return result
end

"""
    pol_linear_x(grid)

Generate x-polarized far-field polarization vectors (θ̂ component for
broadside radiation along z).
Returns (3, NΩ) complex matrix.
"""
function pol_linear_x(grid::SphGrid)
    NΩ = _validate_sph_grid(grid)
    pol = zeros(ComplexF64, 3, NΩ)
    for q in 1:NΩ
        θ = grid.theta[q]
        φ = grid.phi[q]
        # θ̂ unit vector
        theta_hat = Vec3(cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ))
        pol[:, q] = theta_hat
    end
    return pol
end

"""
    pol_linear_y(grid)

Generate the orthogonal far-field polarization vectors (`φ̂` component), which
correspond to y-polarized broadside radiation and the TE/s-polarized basis for
the common `φ = 0` incidence plane in periodic workflows.
Returns `(3, NΩ)` complex matrix.
"""
function pol_linear_y(grid::SphGrid)
    NΩ = _validate_sph_grid(grid)
    pol = zeros(ComplexF64, 3, NΩ)
    for q in 1:NΩ
        φ = grid.phi[q]
        phi_hat = Vec3(-sin(φ), cos(φ), 0.0)
        pol[:, q] = phi_hat
    end
    return pol
end

"""
    cap_mask(grid; theta_max=π/18)

Create a mask selecting directions within a cone of half-angle θ_max
around the z-axis (broadside).
"""
function cap_mask(grid::SphGrid; theta_max=π/18)
    NΩ = _validate_sph_grid(grid)
    isfinite(theta_max) && 0 <= theta_max <= π ||
        throw(ArgumentError(
            "theta_max must be finite and lie in [0, π], got $theta_max"))
    mask = BitVector(undef, NΩ)
    @inbounds for q in 1:NΩ
        mask[q] = grid.theta[q] <= theta_max
    end
    return mask
end

"""
    direction_mask(grid, direction; half_angle=π/18)

Create a mask selecting directions within a cone of `half_angle` (radians)
around an arbitrary `direction` vector. Generalizes `cap_mask` to any direction.

# Example: backscatter mask for incidence from +z
```julia
mask = direction_mask(grid, Vec3(0,0,-1); half_angle=10*π/180)
```
"""
function direction_mask(grid::SphGrid, direction::Vec3; half_angle::Real=π/18)
    NΩ = _validate_sph_grid(grid)
    isfinite(half_angle) && 0 <= half_angle <= π ||
        throw(ArgumentError(
            "half_angle must be finite and lie in [0, π], got $half_angle"))
    all(isfinite, direction) ||
        throw(ArgumentError("direction components must be finite"))
    direction_scale = maximum(abs, direction)
    direction_scale > 0 ||
        throw(ArgumentError("direction must have a nonzero norm"))
    scaled_direction = direction / direction_scale
    scaled_norm = norm(scaled_direction)
    isfinite(scaled_norm) && scaled_norm > 0 ||
        throw(ArgumentError("direction must have a finite, nonzero norm"))
    d = scaled_direction / scaled_norm
    threshold = cos(half_angle)
    mask = BitVector(undef, NΩ)
    @inbounds for q in 1:NΩ
        mask[q] = grid.rhat[1, q] * d[1] +
                  grid.rhat[2, q] * d[2] +
                  grid.rhat[3, q] * d[3] >= threshold
    end
    return mask
end
