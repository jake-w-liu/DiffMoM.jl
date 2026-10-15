# QMatrix.jl — Build the Q matrix for the quadratic far-field objective
#
# Q_mn = Σ_q w_q (p†·g_m)* (p†·g_n)
# J(θ) = I† Q I  (radiated power in selected direction/polarization)

export FarFieldQMatrix, SumQMatrix, build_Q, build_Q_operator, apply_Q, pol_linear_x, pol_linear_y, cap_mask, direction_mask

function _validate_hermitian_objective_matrix(
        Q::AbstractMatrix,
        label::AbstractString)
    LinearAlgebra.ishermitian(Q) ||
        throw(ArgumentError(
            "$label must be Hermitian because adjoint gradients use Q*I " *
            "as the derivative of Re(I' * Q * I). Symmetrize the matrix " *
            "or rebuild it with build_Q."))
    return nothing
end

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

The reusable workspaces belong to the operator and must not overlap each
other or the input/output vectors. External input and output may alias.
"""
struct FarFieldQMatrix <: AbstractMatrix{ComplexF64}
    G_mat::Matrix{ComplexF64}
    weights::Vector{Float64}
    pol::Matrix{ComplexF64}
    mask::Union{Nothing,BitVector}
    N::Int
    work::Vector{ComplexF64}
    error_bounds::Vector{ComplexF64}
    work_lock::ReentrantLock
    extreme_operator_factor::Bool
    function FarFieldQMatrix(G_mat,weights,pol,mask,N,work,error_bounds,
            work_lock,extreme_operator_factor)
        # Preserve the default constructor's field conversions before checking
        # ownership: conversion may create disjoint copies of borrowed inputs.
        operator = new(G_mat,weights,pol,mask,N,work,error_bounds,
                       work_lock,extreme_operator_factor)
        _q_validate_owned_graph(operator)
        return operator
    end
end

FarFieldQMatrix(G_mat::Matrix{ComplexF64},
                weights::Vector{Float64},
                pol::Matrix{ComplexF64},
                mask::Union{Nothing,BitVector},
                N::Int) =
    FarFieldQMatrix(
        G_mat, weights, pol, mask, N,
        zeros(ComplexF64, N), zeros(ComplexF64, N), ReentrantLock(),
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
        G_mat, weights, pol, mask, N, work,
        zeros(ComplexF64, N), work_lock,
        _farfield_q_has_extreme_operator_factor(
            G_mat, weights, pol))

# A primitive real term in G'WGx contains up to six finite Float64 factors.
# Their exact product needs at most 12588 bits; the number of addressable
# quadrature/input terms adds fewer than 70 bits. Keep the remainder as guard
# precision in the exceptional path.
const _FARFIELD_Q_FALLBACK_PRECISION = 12800
const _FARFIELD_Q_SCALED_OUTPUT_FALLBACK_PRECISION = 4352

@inline function _farfield_q_product_underflowed(
        left::Number,
        right::Number,
        product::ComplexF64)
    product_scale = max(abs(real(product)), abs(imag(product)))
    product_scale >= floatmin(Float64) && return false
    return !iszero(product_scale) ||
           (!iszero(left) && !iszero(right))
end

@inline function _farfield_q_product_requires_exact(
        left::Number,
        right::Number,
        product::ComplexF64)
    _farfield_q_product_underflowed(left, right, product) && return true
    left_value = ComplexF64(left)
    right_value = ComplexF64(right)
    real_magnitude =
        abs(real(left_value) * real(right_value)) +
        abs(imag(left_value) * imag(right_value))
    imag_magnitude =
        abs(real(left_value) * imag(right_value)) +
        abs(imag(left_value) * real(right_value))
    return _matrixfree_complex_reduction_requires_exact(
        product, real_magnitude, imag_magnitude, 2)
end

@inline function _farfield_q_projection_checked(
        polarization::SVector{3,ComplexF64},
        radiation::SVector{3,ComplexF64})
    total = zero(ComplexF64)
    @inbounds for component in 1:3
        left = conj(polarization[component])
        right = radiation[component]
        term = left * right
        _farfield_q_product_requires_exact(left, right, term) &&
            return total, true
        next_total = total + term
        (!isfinite(next_total) ||
         _scaled_sum_requires_exact(total, term, next_total)) &&
            return next_total, true
        total = next_total
    end
    return total, false
end

@inline function _farfield_q_projection_with_bounds(
        polarization::SVector{3,ComplexF64},
        radiation::SVector{3,ComplexF64})
    projection = zero(ComplexF64)
    real_bound = 0.0
    imag_bound = 0.0
    @inbounds for component in 1:3
        polarization_value = polarization[component]
        radiation_value = radiation[component]
        polarization_real = real(polarization_value)
        polarization_imag = imag(polarization_value)
        radiation_real = real(radiation_value)
        radiation_imag = imag(radiation_value)
        projection += conj(polarization_value) * radiation_value
        real_bound +=
            abs(polarization_real * radiation_real) +
            abs(polarization_imag * radiation_imag)
        imag_bound +=
            abs(polarization_real * radiation_imag) +
            abs(polarization_imag * radiation_real)
    end
    return projection, real_bound, imag_bound
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
        input::AbstractVector{ComplexF64},
        alpha_scale::Number=one(ComplexF64),
        beta_scale::Number=zero(ComplexF64),
        previous::Union{Nothing,AbstractVector{ComplexF64}}=nothing)
    bits = previous === nothing ? _FARFIELD_Q_FALLBACK_PRECISION :
           _farfield_q_scaled_product_precision(
               Q, input, alpha_scale, beta_scale, previous)
    return setprecision(BigFloat, bits) do
        alpha_big = Complex{BigFloat}(alpha_scale)
        beta_big = Complex{BigFloat}(beta_scale)
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
                total = totals[m]
                if previous !== nothing
                    total = alpha_big * total
                    iszero(beta_scale) ||
                        (total += beta_big * Complex{BigFloat}(previous[m]))
                end
                converted = ComplexF64(total)
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
                if previous !== nothing
                    total = alpha_big * total
                    iszero(beta_scale) ||
                        (total += beta_big * Complex{BigFloat}(previous[m]))
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

function _farfield_q_product_checked!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64},
        defer_exact::Val{DEFER}=Val(false)) where {DEFER}
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
            projection, projection_requires_exact =
                _farfield_q_projection_checked(
                    polarization, radiation)
            projection_requires_exact &&
                return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)
            term = projection * input[n]
            _farfield_q_product_requires_exact(
                projection, input[n], term) &&
                return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)
            next_projected_input = projected_input + term
            (!isfinite(next_projected_input) ||
             _scaled_sum_requires_exact(
                 projected_input, term, next_projected_input)) &&
                return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)
            projected_input = next_projected_input
        end
        isfinite(projected_input) ||
            return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)

        weight = Q.weights[q]
        for m in 1:Q.N
            radiation = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, m],
                Q.G_mat[offset + 2, m],
                Q.G_mat[offset + 3, m])
            projection, projection_requires_exact =
                _farfield_q_projection_checked(
                    polarization, radiation)
            projection_requires_exact &&
                return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)
            weighted_projection = weight * conj(projection)
            _farfield_q_product_requires_exact(
                weight, conj(projection), weighted_projection) &&
                return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)
            contribution = weighted_projection * projected_input
            _farfield_q_product_requires_exact(
                weighted_projection, projected_input, contribution) &&
                return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)
            next_result = result[m] + contribution
            (!isfinite(next_result) ||
             _scaled_sum_requires_exact(
                 result[m], contribution, next_result)) &&
                return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)
            result[m] = next_result
        end
    end
    all(isfinite, result) ||
        return DEFER ? false : _farfield_q_product_bigfloat!(result, Q, input)
    return result
end

@inline function _farfield_q_vector3_has_extreme_factor(
        values::SVector{3,ComplexF64})
    return _farfield_q_extreme_factor(values[1]) ||
           _farfield_q_extreme_factor(values[2]) ||
           _farfield_q_extreme_factor(values[3])
end

@inline function _farfield_q_certificate_term_count(
        column_count::Int,
        quadrature_count::Int)
    # This deliberately overestimates the longest rounding chain through the
    # two projections and the input/quadrature reductions.  The downstream
    # helper contributes a further 16eps factor.
    limit = (typemax(Int) - 256) ÷ 16
    if column_count > limit ||
       quadrature_count > limit - column_count
        return typemax(Int)
    end
    return 16 * (column_count + quadrature_count) + 256
end

function _farfield_q_product_certified!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64},
        defer_exact::Val{DEFER}=Val(false)) where {DEFER}
    # Accumulate componentwise absolute bounds alongside the ordinary
    # factorized product.  If the rounded output is not safely separated from
    # its forward-error envelope, recompute through the exact BigFloat-aware
    # kernel.  This keeps the common path allocation-free without paying the
    # per-operation cancellation checks at every inner-loop instruction.
    if Q.extreme_operator_factor ||
       any(_farfield_q_extreme_factor, input)
        return _farfield_q_product_checked!(result, Q, input, defer_exact)
    end

    fill!(result, zero(ComplexF64))
    fill!(Q.error_bounds, zero(ComplexF64))
    quadrature_count = length(Q.weights)
    @inbounds for q in 1:quadrature_count
        Q.mask !== nothing && !Q.mask[q] && continue
        weight = Q.weights[q]
        _farfield_q_extreme_factor(weight) &&
            return _farfield_q_product_checked!(result, Q, input, defer_exact)
        offset = 3 * (q - 1)
        polarization = SVector{3,ComplexF64}(
            Q.pol[1, q], Q.pol[2, q], Q.pol[3, q])
        _farfield_q_vector3_has_extreme_factor(polarization) &&
            return _farfield_q_product_checked!(result, Q, input, defer_exact)

        projected_input = zero(ComplexF64)
        projected_real_bound = 0.0
        projected_imag_bound = 0.0
        for n in 1:Q.N
            radiation = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, n],
                Q.G_mat[offset + 2, n],
                Q.G_mat[offset + 3, n])
            _farfield_q_vector3_has_extreme_factor(radiation) &&
                return _farfield_q_product_checked!(result, Q, input, defer_exact)
            projection, projection_real_bound, projection_imag_bound =
                _farfield_q_projection_with_bounds(
                    polarization, radiation)
            input_value = input[n]
            projected_input += projection * input_value
            input_real = abs(real(input_value))
            input_imag = abs(imag(input_value))
            projected_real_bound +=
                projection_real_bound * input_real +
                projection_imag_bound * input_imag
            projected_imag_bound +=
                projection_real_bound * input_imag +
                projection_imag_bound * input_real
        end

        absolute_weight = abs(weight)
        for m in 1:Q.N
            radiation = SVector{3,ComplexF64}(
                Q.G_mat[offset + 1, m],
                Q.G_mat[offset + 2, m],
                Q.G_mat[offset + 3, m])
            projection, projection_real_bound, projection_imag_bound =
                _farfield_q_projection_with_bounds(
                    polarization, radiation)
            result[m] +=
                weight * conj(projection) * projected_input

            previous_bound = Q.error_bounds[m]
            real_bound = real(previous_bound) + absolute_weight *
                (projection_real_bound * projected_real_bound +
                 projection_imag_bound * projected_imag_bound)
            imag_bound = imag(previous_bound) + absolute_weight *
                (projection_real_bound * projected_imag_bound +
                 projection_imag_bound * projected_real_bound)
            Q.error_bounds[m] = ComplexF64(real_bound, imag_bound)
        end
    end

    term_count = _farfield_q_certificate_term_count(
        Q.N, quadrature_count)
    @inbounds for row in eachindex(result)
        value = result[row]
        bounds = Q.error_bounds[row]
        if !isfinite(value) ||
           !isfinite(bounds) ||
           _matrixfree_complex_reduction_requires_exact(
               value, real(bounds), imag(bounds), term_count)
            return _farfield_q_product_checked!(result, Q, input, defer_exact)
        end
    end
    return result
end

# Float64/ComplexF64 scales are the ordinary BLAS path. Wider significands
# must keep their owned bits even when the caller's BigFloat precision is low.
function _farfield_q_scale_requires_exact(scale::Number)
    for component in (real(scale), imag(scale))
        component isa Base.IEEEFloat && continue
        if component isa Integer
            abs(widen(component)) <= 9007199254740992 && continue
        end
        # An owned wide value can still be exactly an ordinary IEEE scale.
        # Equality compares the stored value; it does not round it under the
        # caller's current BigFloat precision.
        component isa Union{BigFloat,Integer} && isfinite(component) &&
            Float64(component) == component && continue
        return true
    end
    return false
end

function _farfield_q_scaled_product_precision(Q, input, alpha, beta, previous)
    ranges = (_local_mass_dyadic_span((alpha,)),
              _local_mass_dyadic_span(Q.weights),
              _local_mass_dyadic_span(Q.pol),
              _local_mass_dyadic_span(Q.G_mat),
              _local_mass_dyadic_span(input),
              _local_mass_dyadic_span((beta,)),
              _local_mass_dyadic_span(iszero(beta) ? () : previous))
    bits = if any(isnothing, ranges)
        # Non-dyadic Number inputs retain a finite approximation budget.
        15360
    else
        # alpha*w*conj(pol*G)*(pol*G)*x has seven primitive factors.
        lower = min(BigInt(ranges[1][1]) + ranges[2][1] + 2ranges[3][1] +
                    2ranges[4][1] + ranges[5][1],
                    BigInt(ranges[6][1]) + ranges[7][1])
        upper = max(BigInt(ranges[1][2]) + ranges[2][2] + 2ranges[3][2] +
                    2ranges[4][2] + ranges[5][2] + 7,
                    BigInt(ranges[6][2]) + ranges[7][2] + 2)
        # Nine projection-component pairs, at most64 real terms per complex
        # product, for every quadrature/input pair; beta*y adds two terms.
        terms = 576BigInt(Q.N) * length(Q.weights) + 2
        _local_mass_dyadic_precision(upper, lower, terms)
    end
    # The factorized kernel retains min(N,NΩ) complex wide values. Include
    # their MPFR significands and a bounded allowance for live scalar terms.
    scalar_payload = BigInt(sizeof(BigFloat)) + 8cld(BigInt(bits), 64)
    payload = _checked_payload_sum(
        "FarFieldQMatrix exact scaled product workspace",
        (32 + 2BigInt(min(Q.N, length(Q.weights)))) * scalar_payload)
    _enforce_payload_limit(payload, _default_max_dense_payload_bytes(),
        "FarFieldQMatrix exact scaled product workspace", "available_memory_bytes")
    return bits
end

function _q_validate_external_workspace(result, input, first, second, check_input)
    (length(first) == length(result) && length(second) == length(result)) ||
        throw(DimensionMismatch("Q product workspace lengths must match the output"))
    (Base.mightalias(first, second) ||
     Base.mightalias(result, first) || Base.mightalias(result, second) ||
     (check_input &&
      (Base.mightalias(input, first) || Base.mightalias(input, second)))) &&
        throw(ArgumentError(
            "Q product input/output must not overlap the operator's workspaces"))
    return nothing
end
_q_validate_workspace_tree(::AbstractMatrix, result, input, check_input) = nothing
_q_validate_workspace_tree(Q::FarFieldQMatrix, result, input, check_input) =
    _q_validate_external_workspace(result,input,Q.work,Q.error_bounds,check_input)

function _farfield_q_product!(
        result::Vector{ComplexF64},
        Q::FarFieldQMatrix,
        input::AbstractVector{ComplexF64})
    return _farfield_q_product_certified!(result, Q, input)
end

@noinline function _farfield_q_scaled_output_bigfloat(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    bits = max(_FARFIELD_Q_SCALED_OUTPUT_FALLBACK_PRECISION,
        _local_mass_dyadic_scaled_sum_precision(
            (alpha_scale, beta_scale), (value, previous), 2))
    return setprecision(BigFloat, bits) do
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

@inline function _farfield_q_try_scaled_output(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool)
    needs_fallback =
        _farfield_q_scale_requires_exact(alpha_scale) ||
        _ieee_dense_extreme_factor(value, Float64) ||
        _ieee_dense_extreme_factor(alpha_scale, Float64) ||
        (!overwrite &&
         (_farfield_q_scale_requires_exact(beta_scale) ||
          _ieee_dense_extreme_factor(previous, Float64) ||
          _ieee_dense_extreme_factor(beta_scale, Float64)))
    needs_fallback && return zero(ComplexF64), true
    alpha_term = alpha_scale * value
    if overwrite
        converted = ComplexF64(alpha_term)
        return converted, !isfinite(converted)
    end
    beta_term = beta_scale * previous
    combined = alpha_term + beta_term
    magnitude_sum =
        max(abs(real(alpha_term)), abs(imag(alpha_term))) +
        max(abs(real(beta_term)), abs(imag(beta_term)))
    converted = ComplexF64(combined)
    if isfinite(converted) && isfinite(magnitude_sum) &&
       !_scaled_sum_requires_exact(alpha_term, beta_term, combined)
        return converted, false
    end
    return converted, true
end

@inline function _farfield_q_scaled_output(
        value::ComplexF64, previous::ComplexF64,
        alpha_scale::Number, beta_scale::Number,
        overwrite::Bool, row::Int)
    converted, requires_exact = _farfield_q_try_scaled_output(
        value, previous, alpha_scale, beta_scale, overwrite)
    return requires_exact ? _farfield_q_scaled_output_bigfloat(
        value, previous, alpha_scale, beta_scale, overwrite, row) : converted
end

const _SUM_Q_PRODUCT_FALLBACK_PRECISION = 4608
const _SUM_Q_SCALED_OUTPUT_FALLBACK_PRECISION = 4352

struct SumQMatrix{A<:AbstractMatrix{ComplexF64},B<:AbstractMatrix{ComplexF64}} <: AbstractMatrix{ComplexF64}
    A::A
    B::B
    work_a::Vector{ComplexF64}
    work_b::Vector{ComplexF64}
    work_lock::ReentrantLock
    function SumQMatrix{A,B}(first,second,work_a,work_b,work_lock) where {
            A<:AbstractMatrix{ComplexF64},B<:AbstractMatrix{ComplexF64}}
        operator = new{A,B}(first,second,work_a,work_b,work_lock)
        _q_validate_owned_graph(operator)
        return operator
    end
end
SumQMatrix(first::A,second::B,work_a::Vector{ComplexF64},
        work_b::Vector{ComplexF64},work_lock::ReentrantLock) where {
        A<:AbstractMatrix{ComplexF64},B<:AbstractMatrix{ComplexF64}} =
    SumQMatrix{A,B}(first,second,work_a,work_b,work_lock)

# Every owned write buffer must be disjoint from all known read storage and
# every other owned write buffer in the graph. Check this once: these types'
# array references are immutable, while ordinary coefficient values may mutate.
# Deduplicate shared Q nodes so SumQMatrix(child,child) remains valid.
function _q_collect_owned_storage!(reads,writes,seen,Q::FarFieldQMatrix)
    haskey(seen,Q) && return
    seen[Q] = nothing
    append!(reads,(Q.G_mat,Q.weights,Q.pol))
    Q.mask === nothing || push!(reads,Q.mask)
    append!(writes,(Q.work,Q.error_bounds))
end
function _q_collect_owned_storage!(reads,writes,seen,Q::SumQMatrix)
    haskey(seen,Q) && return
    seen[Q] = nothing
    append!(writes,(Q.work_a,Q.work_b))
    _q_collect_owned_storage!(reads,writes,seen,Q.A)
    _q_collect_owned_storage!(reads,writes,seen,Q.B)
end
_q_collect_owned_storage!(reads,writes,seen,Q::StridedMatrix) = push!(reads,Q)
function _q_collect_owned_storage!(reads,writes,seen,Q::SparseArrays.AbstractSparseMatrixCSC)
    append!(reads,(Q.colptr,Q.rowval,nonzeros(Q)))
end
_q_collect_owned_storage!(reads,writes,seen,Q::Diagonal) = push!(reads,Q.diag)
_q_collect_owned_storage!(reads,writes,seen,Q::Union{LinearAlgebra.Adjoint,
        LinearAlgebra.Transpose,LinearAlgebra.Symmetric,LinearAlgebra.Hermitian}) =
    _q_collect_owned_storage!(reads,writes,seen,parent(Q))
# Base.mightalias exposes storage for other array views/wrappers without
# enumerating entries. Unknown matrix-free children own hidden-cache contracts.
_q_collect_owned_storage!(reads,writes,seen,Q::AbstractMatrix) = push!(reads,Q)
function _q_validate_owned_graph(Q)
    reads,writes = Any[],Any[]
    _q_collect_owned_storage!(reads,writes,IdDict{Any,Nothing}(),Q)
    for (index,work) in enumerate(writes)
        for coefficients in reads
            Base.mightalias(work,coefficients) && throw(ArgumentError(
                "Q operator workspace must not overlap retained coefficient storage"))
        end
        for previous in 1:index-1
            Base.mightalias(work,writes[previous]) && throw(ArgumentError(
                "Q operator workspaces must be mutually disjoint"))
        end
    end
    return nothing
end
function _q_validate_workspace_tree(Q::SumQMatrix, result, input, check_input)
    _q_validate_external_workspace(result,input,Q.work_a,Q.work_b,check_input)
    _q_validate_workspace_tree(Q.A,result,input,check_input)
    _q_validate_workspace_tree(Q.B,result,input,check_input)
    return nothing
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
LinearAlgebra.ishermitian(Q::SumQMatrix) =
    LinearAlgebra.ishermitian(Q.A) && LinearAlgebra.ishermitian(Q.B)

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
    bits = _sum_q_scaled_product_precision(
        Q, input, alpha_scale, beta_scale, previous)
    return setprecision(BigFloat, bits) do
        alpha_big = Complex{BigFloat}(alpha_scale)
        beta_big = overwrite ? zero(Complex{BigFloat}) :
                   Complex{BigFloat}(beta_scale)
        @inbounds for row in eachindex(output)
            product = _q_wide_product_row(Q, row, input)
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
    bits = max(_SUM_Q_SCALED_OUTPUT_FALLBACK_PRECISION,
        _local_mass_dyadic_scaled_sum_precision(
            (alpha_scale, beta_scale), (value, previous), 2))
    return setprecision(BigFloat, bits) do
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
        _farfield_q_scale_requires_exact(alpha_scale) ||
        _ieee_dense_extreme_factor(value, Float64) ||
        _ieee_dense_extreme_factor(alpha_scale, Float64) ||
        (!overwrite &&
         (_farfield_q_scale_requires_exact(beta_scale) ||
          _ieee_dense_extreme_factor(previous, Float64) ||
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

# Bounds for the original primitive terms in a row product, before child
# operators or caller scales can narrow them. The final field counts live
# recursive levels for the scalar-workspace preflight.
function _q_product_dyadic_bounds(Q::Union{Matrix{ComplexF64},
        SparseArrays.AbstractSparseMatrixCSC{ComplexF64}}, input)
    values = Q isa SparseArrays.AbstractSparseMatrixCSC ? nonzeros(Q) : Q
    matrix_range = _local_mass_dyadic_span(values)
    input_range = _local_mass_dyadic_span(input)
    (matrix_range === nothing || input_range === nothing) && return nothing
    return (BigInt(matrix_range[1])+input_range[1],
            BigInt(matrix_range[2])+input_range[2]+2,
            2BigInt(length(input)), 1)
end
function _q_product_dyadic_bounds(Q::FarFieldQMatrix, input)
    ranges = (_local_mass_dyadic_span(Q.weights),
              _local_mass_dyadic_span(Q.pol),
              _local_mass_dyadic_span(Q.G_mat),
              _local_mass_dyadic_span(input))
    any(isnothing, ranges) && return nothing
    return (BigInt(ranges[1][1])+2ranges[2][1]+2ranges[3][1]+ranges[4][1],
            BigInt(ranges[1][2])+2ranges[2][2]+2ranges[3][2]+ranges[4][2]+6,
            288BigInt(Q.N)*length(Q.weights), 1)
end
function _q_product_dyadic_bounds(Q::SumQMatrix, input)
    first = _q_product_dyadic_bounds(Q.A, input)
    second = _q_product_dyadic_bounds(Q.B, input)
    (first === nothing || second === nothing) && return nothing
    return (min(first[1],second[1]), max(first[2],second[2]),
            first[3]+second[3], max(first[4],second[4])+1)
end
_q_product_depth(::AbstractMatrix) = 1
_q_product_depth(Q::SumQMatrix) = max(_q_product_depth(Q.A),_q_product_depth(Q.B))+1

function _sum_q_scaled_product_precision(Q, input, alpha, beta, previous)
    product = _q_product_dyadic_bounds(Q,input)
    scale = _local_mass_dyadic_span((alpha,))
    prior = _local_mass_dyadic_span(iszero(beta) ? () : previous)
    prior_scale = _local_mass_dyadic_span((beta,))
    bits = if any(isnothing,(product,scale,prior,prior_scale))
        15360
    else
        lower = min(product[1]+scale[1], BigInt(prior[1])+prior_scale[1])
        upper = max(product[2]+scale[2]+1, BigInt(prior[2])+prior_scale[2]+2)
        _local_mass_dyadic_precision(upper,lower,2product[3]+2)
    end
    depth = product === nothing ? _q_product_depth(Q) : product[4]
    payload = _checked_payload_sum("SumQMatrix exact scaled product workspace",
        32BigInt(depth+1)*(sizeof(BigFloat)+8cld(BigInt(bits),64)))
    _enforce_payload_limit(payload,_default_max_dense_payload_bytes(),
        "SumQMatrix exact scaled product workspace","available_memory_bytes")
    return bits
end

function _q_wide_product_row(Q::Union{Matrix{ComplexF64},
        SparseArrays.AbstractSparseMatrixCSC{ComplexF64}}, row, input)
    total = zero(Complex{BigFloat})
    @inbounds for column in axes(Q,2)
        total += Complex{BigFloat}(Q[row,column])*Complex{BigFloat}(input[column])
    end
    return total
end
function _q_wide_product_row(Q::FarFieldQMatrix, row, input)
    total = zero(Complex{BigFloat})
    @inbounds for q in eachindex(Q.weights)
        Q.mask !== nothing && !Q.mask[q] && continue
        projected = zero(Complex{BigFloat})
        for column in axes(Q,2)
            projected += _farfield_q_projection_bigfloat(Q,q,column)*
                         Complex{BigFloat}(input[column])
        end
        total += BigFloat(Q.weights[q])*
                 conj(_farfield_q_projection_bigfloat(Q,q,row))*projected
    end
    return total
end
_q_wide_product_row(Q::SumQMatrix, row, input) =
    _q_wide_product_row(Q.A,row,input)+_q_wide_product_row(Q.B,row,input)

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
    _q_validate_workspace_tree(Q,result,x,!iszero(alpha_scale))
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
        exact_entries_supported = _sum_q_exact_entries_supported(Q.A) &&
                                  _sum_q_exact_entries_supported(Q.B)
        wide_scales = _farfield_q_scale_requires_exact(alpha_scale) ||
                      (!overwrite && _farfield_q_scale_requires_exact(beta_scale))
        fallback_required = (exact_entries_supported && wide_scales) ||
                            _sum_q_requires_exact_product(Q, xread) ||
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

        if !fallback_required
            @inbounds for row in eachindex(Q.work_a)
                converted, needs_exact = _farfield_q_try_scaled_output(
                    Q.work_a[row],result[row],alpha_scale,beta_scale,overwrite)
                if needs_exact
                    if exact_entries_supported
                        fallback_required = true
                        break
                    end
                    # Opaque children expose their completed finite products.
                    # Preserve that existing contract for wider Number scales
                    # without requiring an unavailable child entry expansion.
                    converted = _sum_q_scaled_output(
                        Q.work_a[row],result[row],alpha_scale,beta_scale,overwrite,row)
                end
                Q.work_b[row] = converted
            end
        end
        if fallback_required
            exact_entries_supported ||
                throw(OverflowError(
                    "SumQMatrix cannot certify an exceptional product for " *
                    "a generic matrix-free child without exact-entry support"))
            _sum_q_scaled_product_bigfloat!(
                Q.work_b, Q, xread, result,
                alpha_scale, beta_scale, overwrite)
        end
        copyto!(result, Q.work_b)
    finally
        unlock(Q.work_lock)
    end
    return result
end

Base.size(Q::FarFieldQMatrix) = (Q.N, Q.N)
Base.eltype(::FarFieldQMatrix) = ComplexF64
LinearAlgebra.ishermitian(::FarFieldQMatrix) = true

@noinline function _farfield_q_entry_bigfloat_arrays(
        G_mat::Matrix{ComplexF64},
        weights::Vector{Float64},
        pol::Matrix{ComplexF64},
        mask,
        m::Int,
        n::Int)
    return setprecision(BigFloat, _FARFIELD_Q_FALLBACK_PRECISION) do
        total = zero(Complex{BigFloat})
        @inbounds for q in eachindex(weights)
            mask !== nothing && !mask[q] && continue
            offset = 3 * (q - 1)
            projection_m = zero(Complex{BigFloat})
            projection_n = zero(Complex{BigFloat})
            for component in 1:3
                polarization = conj(Complex{BigFloat}(pol[component, q]))
                projection_m += polarization *
                    Complex{BigFloat}(G_mat[offset + component, m])
                projection_n += polarization *
                    Complex{BigFloat}(G_mat[offset + component, n])
            end
            total += BigFloat(weights[q]) *
                conj(projection_m) * projection_n
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "FarFieldQMatrix entry is outside the representable " *
                "ComplexF64 range at index ($m, $n)."))
        return converted
    end
end

@noinline function _farfield_q_entry_bigfloat(
        Q::FarFieldQMatrix,
        m::Int,
        n::Int)
    return _farfield_q_entry_bigfloat_arrays(
        Q.G_mat, Q.weights, Q.pol, Q.mask, m, n)
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
        projection_m, projection_m_requires_exact =
            _farfield_q_projection_checked(
                polarization, radiation_m)
        projection_m_requires_exact &&
            return _farfield_q_entry_bigfloat(Q, m, n)
        contribution = if m == n
            projection_squared = abs2(projection_m)
            ((!iszero(projection_m) && iszero(projection_squared)) ||
             (!iszero(projection_squared) &&
              projection_squared < floatmin(Float64))) &&
                return _farfield_q_entry_bigfloat(Q, m, n)
            weighted_squared = Q.weights[q] * projection_squared
            (!isfinite(weighted_squared) ||
             (!iszero(Q.weights[q]) && !iszero(projection_squared) &&
              iszero(weighted_squared))) &&
                return _farfield_q_entry_bigfloat(Q, m, n)
            ComplexF64(weighted_squared)
        else
            projection_n, projection_n_requires_exact =
                _farfield_q_projection_checked(
                    polarization, radiation_n)
            projection_n_requires_exact &&
                return _farfield_q_entry_bigfloat(Q, m, n)
            weighted_projection = Q.weights[q] * conj(projection_m)
            _farfield_q_product_requires_exact(
                Q.weights[q], conj(projection_m), weighted_projection) &&
                return _farfield_q_entry_bigfloat(Q, m, n)
            product = weighted_projection * projection_n
            _farfield_q_product_requires_exact(
                weighted_projection, projection_n, product) &&
                return _farfield_q_entry_bigfloat(Q, m, n)
            product
        end
        next_val = val + contribution
        (!isfinite(next_val) ||
         _scaled_sum_requires_exact(val, contribution, next_val)) &&
            return _farfield_q_entry_bigfloat(Q, m, n)
        val = next_val
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
    _q_validate_external_workspace(
        result,I_coeffs,Q.work,Q.error_bounds,!iszero(alpha_scale))
    if iszero(alpha_scale)
        if iszero(beta_scale)
            fill!(result, zero(ComplexF64))
        elseif beta_scale != one(beta_scale)
            lock(Q.work_lock)
            try
                @inbounds for m in eachindex(result)
                    Q.work[m] = _farfield_q_scaled_output(
                        zero(ComplexF64), result[m], alpha_scale,
                        beta_scale, false, m)
                end
                copyto!(result, Q.work)
            finally
                unlock(Q.work_lock)
            end
        end
        return result
    end
    input_read = Base.mightalias(result, I_coeffs) ? copy(I_coeffs) : I_coeffs
    overwrite = iszero(beta_scale)
    lock(Q.work_lock)
    try
        requires_exact = _farfield_q_scale_requires_exact(alpha_scale) ||
            (!overwrite && _farfield_q_scale_requires_exact(beta_scale))
        # Defer exceptional products until alpha and beta can join the same
        # wide reduction; an unscaled Q*x may overflow or round to zero.
        requires_exact || (requires_exact =
            _farfield_q_product_certified!(Q.work, Q, input_read, Val(true)) === false)
        if !requires_exact
            @inbounds for m in eachindex(result)
                converted, needs_exact = _farfield_q_try_scaled_output(
                    Q.work[m], result[m], alpha_scale, beta_scale, overwrite)
                if needs_exact
                    requires_exact = true
                    break
                end
                Q.error_bounds[m] = converted
            end
        end
        if requires_exact
            _farfield_q_product_bigfloat!(
                Q.work, Q, input_read, alpha_scale, beta_scale, result)
            copyto!(result, Q.work)
        else
            # Reuse the no-longer-needed certificate buffer so failure never
            # leaves a partially scaled user output, including aliased input.
            copyto!(result, Q.error_bounds)
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
        ComplexF64[], ComplexF64[], ReentrantLock(), true)
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

@inline function _q_neumaier_add(
        total::Float64, correction::Float64, value::Float64)
    combined = total + value
    if abs(total) >= abs(value)
        correction += (total - combined) + value
    else
        correction += (value - combined) + total
    end
    return combined, correction
end

function _farfield_q_entry_compensated(
        projections::Matrix{ComplexF64},
        weights::Vector{Float64},
        mask,
        row::Int,
        column::Int)
    real_total = 0.0
    real_correction = 0.0
    imag_total = 0.0
    imag_correction = 0.0
    @inbounds for q in eachindex(weights)
        mask !== nothing && !mask[q] && continue
        row_projection = conj(projections[q, row])
        weighted_projection = weights[q] * row_projection
        _farfield_q_product_requires_exact(
            weights[q], row_projection, weighted_projection) &&
            return zero(ComplexF64), true
        contribution = weighted_projection * projections[q, column]
        _farfield_q_product_requires_exact(
            weighted_projection, projections[q, column], contribution) &&
            return zero(ComplexF64), true
        real_total, real_correction = _q_neumaier_add(
            real_total, real_correction, real(contribution))
        imag_total, imag_correction = _q_neumaier_add(
            imag_total, imag_correction, imag(contribution))
        (isfinite(real_total) && isfinite(real_correction) &&
         isfinite(imag_total) && isfinite(imag_correction)) ||
            return zero(ComplexF64), true
    end
    value = ComplexF64(
        real_total + real_correction,
        imag_total + imag_correction,
    )
    return value, !isfinite(value)
end

function _build_q_from_projections!(
        result::Matrix{ComplexF64},
        projections::Matrix{ComplexF64},
        weighted_projections::Matrix{ComplexF64},
        G_mat::Matrix{ComplexF64},
        weights::Vector{Float64},
        pol::Matrix{ComplexF64},
        mask,
        N::Int)
    active_terms = 0
    @inbounds for q in eachindex(weights)
        active = mask === nothing || mask[q]
        active_terms += active && !iszero(weights[q])
        weight = active ? weights[q] : 0.0
        for column in 1:N
            weighted_projections[q, column] =
                weight * projections[q, column]
        end
    end
    mul!(result, adjoint(projections), weighted_projections)

    # BLAS handles the ordinary O(N²NΩ) product. Recompute an entry when one
    # component is small enough to have lost a cancellation-sensitive product.
    # The diagonal supplies a Cauchy-Schwarz bound without another NΩ×N
    # workspace or a second matrix product.
    column_has_real = falses(N)
    column_has_imag = falses(N)
    @inbounds for column in 1:N, q in eachindex(weights)
        mask !== nothing && !mask[q] && continue
        projection = projections[q, column]
        column_has_real[column] |= !iszero(real(projection))
        column_has_imag[column] |= !iszero(imag(projection))
    end
    error_factor = min(1.0, 32eps(Float64) * max(active_terms, 1))
    @inbounds for column in 1:N
        column_norm = sqrt(max(real(result[column, column]), 0.0))
        for row in 1:column
            value = result[row, column]
            row_norm = sqrt(max(real(result[row, row]), 0.0))
            magnitude_bound = row_norm * column_norm
            real_component_possible =
                (column_has_real[row] && column_has_real[column]) ||
                (column_has_imag[row] && column_has_imag[column])
            imag_component_possible =
                (column_has_real[row] && column_has_imag[column]) ||
                (column_has_imag[row] && column_has_real[column])
            component_requires_check = row != column &&
                !iszero(magnitude_bound) &&
                ((real_component_possible &&
                  abs(real(value)) <= error_factor * magnitude_bound) ||
                 (imag_component_possible &&
                  abs(imag(value)) <= error_factor * magnitude_bound))
            if !isfinite(value) ||
               component_requires_check
                value, requires_exact = _farfield_q_entry_compensated(
                    projections, weights, mask, row, column)
                requires_exact &&
                    (value = _farfield_q_entry_bigfloat_arrays(
                        G_mat, weights, pol, mask, row, column))
            end
            row == column && (value = ComplexF64(real(value), 0.0))
            result[row, column] = value
            result[column, row] = conj(value)
        end
    end
    return result
end

"""
    build_Q(G_mat, grid, pol;
            mask=nothing,
            max_work_bytes=Sys.free_memory())

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
        max_work_bytes::Integer=_default_max_dense_payload_bytes())
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
    weighted_projection_bytes = checked_products ? 0 : projection_bytes
    component_flag_bytes = checked_products ? 0 :
        _checked_array_payload_bytes(
            UInt64, 2, cld(N, 8sizeof(UInt64));
            label="build_Q component-flag workspace")
    work_bytes = try
        checked_products ?
            Base.Checked.checked_add(matrix_bytes, mask_copy_bytes) :
            Base.Checked.checked_add(
                Base.Checked.checked_add(
                    Base.Checked.checked_add(
                        projection_bytes, weighted_projection_bytes),
                    component_flag_bytes),
                matrix_bytes)
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
    projection_requires_exact = false
    for q in 1:NΩ
        if mask !== nothing && !mask[q]
            continue
        end
        p = SVector{3,ComplexF64}(pol[1, q], pol[2, q], pol[3, q])  # avoid slice alloc
        idx = 3 * (q - 1)
        for n in 1:N
            gn = SVector{3,ComplexF64}(G_mat[idx+1, n], G_mat[idx+2, n], G_mat[idx+3, n])
            projection, requires_exact =
                _farfield_q_projection_checked(p, gn)
            if requires_exact
                projection_requires_exact = true
                break
            end
            y[q, n] = projection
        end
        projection_requires_exact && break
    end

    # Q_mn = Σ_q w_q conj(y_qm) y_qn.
    Q = zeros(ComplexF64, N, N)
    projection_requires_exact &&
        return _build_q_checked(G_mat, grid.w, pol, mask, N)
    weighted_y = similar(y)
    return _build_q_from_projections!(
        Q, y, weighted_y, G_mat, grid.w, pol, mask, N)
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

@noinline function _farfield_q_apply_bigfloat!(
        result::Vector{ComplexF64},
        G_mat::Matrix{ComplexF64},
        weights::Vector{Float64},
        pol::Matrix{ComplexF64},
        mask,
        N::Int,
        input::Vector{ComplexF64})
    mask_copy = mask === nothing ? nothing : BitVector(mask)
    operator = FarFieldQMatrix(
        G_mat, weights, pol, mask_copy, N,
        result, ComplexF64[], ReentrantLock(), true)
    return _farfield_q_product_bigfloat!(result, operator, input)
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
            result, ComplexF64[], ReentrantLock(),
            extreme_operator_factor)
        return _farfield_q_product!(
            result, operator, I_coeffs)
    end
    fallback_required = false
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
            projection, projection_requires_exact =
                _farfield_q_projection_checked(p, gn)
            if projection_requires_exact
                fallback_required = true
                break
            end
            term = projection * I_coeffs[n]
            if _farfield_q_product_requires_exact(
                    projection, I_coeffs[n], term)
                fallback_required = true
                break
            end
            next_yq = yq + term
            if !isfinite(next_yq) ||
               _scaled_sum_requires_exact(yq, term, next_yq)
                fallback_required = true
                break
            end
            yq = next_yq
        end
        fallback_required && break

        # Accumulate: (Q*I)_m += w_q conj(p†·g_m) y_q
        for m in 1:N
            idx = 3 * (q - 1)
            gm = SVector{3,ComplexF64}(G_mat[idx+1, m], G_mat[idx+2, m], G_mat[idx+3, m])
            projection, projection_requires_exact =
                _farfield_q_projection_checked(p, gm)
            if projection_requires_exact
                fallback_required = true
                break
            end
            weighted_projection = wq * conj(projection)
            if _farfield_q_product_requires_exact(
                    wq, conj(projection), weighted_projection)
                fallback_required = true
                break
            end
            contribution = weighted_projection * yq
            if _farfield_q_product_requires_exact(
                    weighted_projection, yq, contribution)
                fallback_required = true
                break
            end
            next_result = result[m] + contribution
            if !isfinite(next_result) ||
               _scaled_sum_requires_exact(
                   result[m], contribution, next_result)
                fallback_required = true
                break
            end
            result[m] = next_result
        end
        fallback_required && break
    end

    fallback_required && return _farfield_q_apply_bigfloat!(
        result, G_mat, grid.w, pol, mask, N, I_coeffs)

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
around an arbitrary `direction` vector. Both the requested direction and each
stored grid direction are normalized before the angular comparison. Generalizes
`cap_mask` to any direction.

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
        x = grid.rhat[1, q]
        y = grid.rhat[2, q]
        z = grid.rhat[3, q]
        inverse_norm = inv(hypot(x, y, z))
        cosine = clamp(
            (x * d[1] + y * d[2] + z * d[3]) * inverse_norm,
            -1.0,
            1.0,
        )
        mask[q] = cosine >= threshold
    end
    return mask
end
