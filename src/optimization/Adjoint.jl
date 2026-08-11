# Adjoint.jl — Adjoint solve and gradient evaluation
#
# Adjoint eq:  Z† λ = ∂Φ/∂I* = Q I
# Gradient:    ∂J/∂θ_p = -2 Re{ λ† (∂Z/∂θ_p) I }
#            = -2 Re{ λ† (-M_p) I }
#            = +2 Re{ λ† M_p I }

export solve_adjoint, solve_adjoint_rhs, gradient_impedance, compute_objective

# Expanding one component of left' * A * right produces at most four real
# triple products per stored matrix entry. Three finite Float64 coefficients
# need at most 6295 bits, and an addressable matrix has at most typemax(Int)
# entries. The extra 65 bits cover the full sum, with the remaining precision
# as guard margin.
const _IEEE_BILINEAR_FALLBACK_PRECISION = 6656

@inline function _bilinear_component_bigfloat(
    left_real::BigFloat,
    left_imag::BigFloat,
    matrix_real::BigFloat,
    matrix_imag::BigFloat,
    right_real::BigFloat,
    right_imag::BigFloat,
    ::Val{:real},
)
    product_real = matrix_real * right_real - matrix_imag * right_imag
    product_imag = matrix_real * right_imag + matrix_imag * right_real
    return left_real * product_real + left_imag * product_imag
end

@inline function _bilinear_component_bigfloat(
    left_real::BigFloat,
    left_imag::BigFloat,
    matrix_real::BigFloat,
    matrix_imag::BigFloat,
    right_real::BigFloat,
    right_imag::BigFloat,
    ::Val{:imag},
)
    product_real = matrix_real * right_real - matrix_imag * right_imag
    product_imag = matrix_real * right_imag + matrix_imag * right_real
    return left_real * product_imag - left_imag * product_real
end

@noinline function _quadratic_objective_bigfloat(
    I::Vector{<:Number},
    Q::Matrix{<:Number},
    ::Type{T},
) where {T<:AbstractFloat}
    if _ieee_bilinear_superaccumulator_supported(I, Q, I, T)
        return _bilinear_component_ieee_exact(
            I, Q, I, Val(:real), T, "quadratic objective")::T
    end
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        total = zero(BigFloat)
        @inbounds for row in axes(Q, 1)
            left_value = I[row]
            iszero(left_value) && continue
            left_real = BigFloat(real(left_value))
            left_imag = BigFloat(imag(left_value))
            for column in axes(Q, 2)
                matrix_value = Q[row, column]
                right_value = I[column]
                (iszero(matrix_value) || iszero(right_value)) && continue
                matrix_real = BigFloat(real(matrix_value))
                matrix_imag = BigFloat(imag(matrix_value))
                right_real = BigFloat(real(right_value))
                right_imag = BigFloat(imag(right_value))
                total += _bilinear_component_bigfloat(
                    left_real, left_imag,
                    matrix_real, matrix_imag,
                    right_real, right_imag,
                    Val(:real),
                )
            end
        end
        converted = convert(T, total)
        isfinite(converted) ||
            throw(OverflowError(
                "quadratic objective is outside the representable $T range"))
        return converted
    end
end

@inline _bilinear_component(value::Number, ::Val{:real}) = real(value)
@inline _bilinear_component(value::Number, ::Val{:imag}) = imag(value)

@inline function _ieee_bilinear_values_require_fallback(
    values,
    ::Type{R},
) where {R<:Union{Float32,Float64}}
    @inbounds for value in values
        _ieee_dense_extreme_factor(value, R) && return true
    end
    return false
end

@inline function _ieee_bilinear_matrix_requires_fallback(
    matrix::AbstractMatrix,
    ::Type{R},
) where {R<:Union{Float32,Float64}}
    return _ieee_bilinear_values_require_fallback(matrix, R)
end

@inline function _ieee_bilinear_matrix_requires_fallback(
    matrix::SparseArrays.AbstractSparseMatrixCSC,
    ::Type{R},
) where {R<:Union{Float32,Float64}}
    return _ieee_bilinear_values_require_fallback(nonzeros(matrix), R)
end

@inline function _ieee_bilinear_matrix_requires_fallback(
    matrix::LocalMassMatrix,
    ::Type{R},
) where {R<:Union{Float32,Float64}}
    return _ieee_bilinear_values_require_fallback(matrix.vals, R)
end

# The ±128 Float64 component band used by dense products gives exact
# three-factor quanta of at least 2^-540 and an addressable reduction below
# 2^451. The Float32 ±16 band gives quanta of at least 2^-120 and reductions
# below 2^115. Within those bands, staged products cannot lose a representable
# term solely to range; exceptional operands use the exact accumulator.
function _ieee_bilinear_requires_fallback(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    ::Type{R},
) where {R<:Union{Float32,Float64}}
    _ieee_bilinear_values_require_fallback(left, R) && return true
    _ieee_bilinear_values_require_fallback(right, R) && return true
    return _ieee_bilinear_matrix_requires_fallback(matrix, R)
end

@inline _ieee_bilinear_accumulator_precision(::Type{Float64}) =
    _IEEE_BILINEAR_FALLBACK_PRECISION
@inline _ieee_bilinear_accumulator_precision(::Type{Float32}) = 960
@inline _ieee_bilinear_base_exponent(::Type{Float64}) = -3222
@inline _ieee_bilinear_base_exponent(::Type{Float32}) = -447

mutable struct _IEEEBilinearAccumulator{R<:Union{Float32,Float64}}
    words::Vector{Int128}
    pending_terms::Int
end

function _IEEEBilinearAccumulator(::Type{R}) where {
        R<:Union{Float32,Float64}}
    precision_bits = _ieee_bilinear_accumulator_precision(R)
    # One extra word receives the final signed carry.
    words = zeros(Int128, cld(precision_bits, 64) + 1)
    return _IEEEBilinearAccumulator{R}(words, 0)
end

@inline function _ieee_signed_mantissa_exponent(value::Float64)
    bits = reinterpret(UInt64, value)
    exponent_bits = (bits >> 52) & UInt64(0x7ff)
    fraction = bits & ((UInt64(1) << 52) - UInt64(1))
    exponent_bits == UInt64(0x7ff) &&
        throw(ArgumentError("non-finite Float64 in exact bilinear accumulator"))
    if iszero(exponent_bits)
        iszero(fraction) && return Int64(0), 0
        mantissa = Int64(fraction)
        (bits >> 63) != 0 && (mantissa = -mantissa)
        return mantissa, -1074
    end
    mantissa = Int64(fraction | (UInt64(1) << 52))
    (bits >> 63) != 0 && (mantissa = -mantissa)
    return mantissa, Int(exponent_bits) - 1023 - 52
end

@inline function _ieee_signed_mantissa_exponent(value::Float32)
    bits = reinterpret(UInt32, value)
    exponent_bits = (bits >> 23) & UInt32(0xff)
    fraction = bits & ((UInt32(1) << 23) - UInt32(1))
    exponent_bits == UInt32(0xff) &&
        throw(ArgumentError("non-finite Float32 in exact bilinear accumulator"))
    if iszero(exponent_bits)
        iszero(fraction) && return Int64(0), 0
        mantissa = Int64(fraction)
        (bits >> 31) != 0 && (mantissa = -mantissa)
        return mantissa, -149
    end
    mantissa = Int64(fraction | (UInt32(1) << 23))
    (bits >> 31) != 0 && (mantissa = -mantissa)
    return mantissa, Int(exponent_bits) - 127 - 23
end

@inline function _ieee_bilinear_add_limb!(
    words::Vector{Int128},
    limb::UInt64,
    bit_position::Int,
    negative::Bool,
)
    iszero(limb) && return nothing
    word_index = (bit_position >>> 6) + 1
    bit_shift = bit_position & 63
    word_index <= length(words) ||
        error("exact bilinear accumulator capacity invariant violated")
    low = limb << bit_shift
    low_delta = Int128(low)
    words[word_index] += negative ? -low_delta : low_delta
    if !iszero(bit_shift)
        high = limb >> (64 - bit_shift)
        if !iszero(high)
            word_index += 1
            word_index <= length(words) ||
                error("exact bilinear accumulator carry invariant violated")
            high_delta = Int128(high)
            words[word_index] += negative ? -high_delta : high_delta
        end
    end
    return nothing
end

function _ieee_bilinear_normalize!(
    accumulator::_IEEEBilinearAccumulator,
)
    words = accumulator.words
    word_base = Int128(1) << 64
    @inbounds for index in firstindex(words):(lastindex(words) - 1)
        carry = fld(words[index], word_base)
        words[index] -= carry * word_base
        words[index + 1] += carry
    end
    accumulator.pending_terms = 0
    return accumulator
end

@inline function _ieee_bilinear_add_triple!(
    accumulator::_IEEEBilinearAccumulator{R},
    first::R,
    second::R,
    third::R,
    coefficient_sign::Int,
) where {R<:Union{Float32,Float64}}
    (iszero(first) || iszero(second) || iszero(third)) && return nothing
    first_mantissa, first_exponent =
        _ieee_signed_mantissa_exponent(first)
    second_mantissa, second_exponent =
        _ieee_signed_mantissa_exponent(second)
    third_mantissa, third_exponent =
        _ieee_signed_mantissa_exponent(third)

    negative = (first_mantissa < 0) ⊻ (second_mantissa < 0) ⊻
               (third_mantissa < 0) ⊻ (coefficient_sign < 0)
    first_unsigned = UInt64(abs(first_mantissa))
    second_unsigned = UInt64(abs(second_mantissa))
    third_unsigned = UInt64(abs(third_mantissa))

    first_product = UInt128(first_unsigned) * UInt128(second_unsigned)
    first_low = UInt64(first_product & UInt128(typemax(UInt64)))
    first_high = UInt64(first_product >> 64)
    low_product = UInt128(first_low) * UInt128(third_unsigned)
    high_product = UInt128(first_high) * UInt128(third_unsigned) +
                   (low_product >> 64)
    limb_0 = UInt64(low_product & UInt128(typemax(UInt64)))
    limb_1 = UInt64(high_product & UInt128(typemax(UInt64)))
    limb_2 = UInt64(high_product >> 64)

    bit_position = first_exponent + second_exponent + third_exponent -
                   _ieee_bilinear_base_exponent(R)
    bit_position >= 0 ||
        error("exact bilinear accumulator exponent invariant violated")
    _ieee_bilinear_add_limb!(
        accumulator.words, limb_0, bit_position, negative)
    _ieee_bilinear_add_limb!(
        accumulator.words, limb_1, bit_position + 64, negative)
    _ieee_bilinear_add_limb!(
        accumulator.words, limb_2, bit_position + 128, negative)

    accumulator.pending_terms += 1
    # Normalizing periodically gives a formal Int128 headroom bound even for
    # reductions approaching the maximum addressable matrix length.
    accumulator.pending_terms == (1 << 30) &&
        _ieee_bilinear_normalize!(accumulator)
    return nothing
end

@inline function _ieee_bilinear_accumulate_entry!(
    accumulator::_IEEEBilinearAccumulator{R},
    left_value::Number,
    matrix_value::Number,
    right_value::Number,
    ::Val{:real},
) where {R<:Union{Float32,Float64}}
    left_real = R(real(left_value))
    left_imag = R(imag(left_value))
    matrix_real = R(real(matrix_value))
    matrix_imag = R(imag(matrix_value))
    right_real = R(real(right_value))
    right_imag = R(imag(right_value))
    _ieee_bilinear_add_triple!(
        accumulator, left_real, matrix_real, right_real, 1)
    _ieee_bilinear_add_triple!(
        accumulator, left_real, matrix_imag, right_imag, -1)
    _ieee_bilinear_add_triple!(
        accumulator, left_imag, matrix_real, right_imag, 1)
    _ieee_bilinear_add_triple!(
        accumulator, left_imag, matrix_imag, right_real, 1)
    return nothing
end

@inline function _ieee_bilinear_accumulate_entry!(
    accumulator::_IEEEBilinearAccumulator{R},
    left_value::Number,
    matrix_value::Number,
    right_value::Number,
    ::Val{:imag},
) where {R<:Union{Float32,Float64}}
    left_real = R(real(left_value))
    left_imag = R(imag(left_value))
    matrix_real = R(real(matrix_value))
    matrix_imag = R(imag(matrix_value))
    right_real = R(real(right_value))
    right_imag = R(imag(right_value))
    _ieee_bilinear_add_triple!(
        accumulator, left_real, matrix_real, right_imag, 1)
    _ieee_bilinear_add_triple!(
        accumulator, left_real, matrix_imag, right_real, 1)
    _ieee_bilinear_add_triple!(
        accumulator, left_imag, matrix_real, right_real, -1)
    _ieee_bilinear_add_triple!(
        accumulator, left_imag, matrix_imag, right_imag, 1)
    return nothing
end

function _accumulate_bilinear_component_ieee!(
    accumulator::_IEEEBilinearAccumulator,
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    component,
)
    @inbounds for row in axes(matrix, 1)
        left_value = left[row]
        iszero(left_value) && continue
        for column in axes(matrix, 2)
            matrix_value = matrix[row, column]
            right_value = right[column]
            (iszero(matrix_value) || iszero(right_value)) && continue
            _ieee_bilinear_accumulate_entry!(
                accumulator, left_value, matrix_value,
                right_value, component)
        end
    end
    return accumulator
end

function _accumulate_bilinear_component_ieee!(
    accumulator::_IEEEBilinearAccumulator,
    left::AbstractVector,
    matrix::SparseArrays.AbstractSparseMatrixCSC,
    right::AbstractVector,
    component,
)
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    @inbounds for column in axes(matrix, 2)
        right_value = right[column]
        iszero(right_value) && continue
        for position in nzrange(matrix, column)
            left_value = left[rows[position]]
            matrix_value = values[position]
            (iszero(left_value) || iszero(matrix_value)) && continue
            _ieee_bilinear_accumulate_entry!(
                accumulator, left_value, matrix_value,
                right_value, component)
        end
    end
    return accumulator
end

function _accumulate_bilinear_component_ieee!(
    accumulator::_IEEEBilinearAccumulator,
    left::AbstractVector,
    matrix::LocalMassMatrix,
    right::AbstractVector,
    component,
)
    @inbounds for position in eachindex(matrix.vals)
        left_value = left[matrix.rows[position]]
        matrix_value = matrix.vals[position]
        right_value = right[matrix.cols[position]]
        (iszero(left_value) || iszero(matrix_value) ||
         iszero(right_value)) && continue
        _ieee_bilinear_accumulate_entry!(
            accumulator, left_value, matrix_value,
            right_value, component)
    end
    return accumulator
end

function _ieee_bilinear_finish(
    accumulator::_IEEEBilinearAccumulator{R},
    label::AbstractString,
) where {R<:Union{Float32,Float64}}
    _ieee_bilinear_normalize!(accumulator)
    words = accumulator.words
    integer_total = BigInt(words[end])
    @inbounds for index in (lastindex(words) - 1):-1:firstindex(words)
        integer_total = (integer_total << 64) + UInt64(words[index])
    end
    precision_bits = _ieee_bilinear_accumulator_precision(R)
    return setprecision(BigFloat, precision_bits) do
        exact_value = ldexp(
            BigFloat(integer_total), _ieee_bilinear_base_exponent(R))
        converted = R(exact_value)
        isfinite(converted) ||
            throw(OverflowError(
                "$label is outside the representable $R range"))
        return converted
    end
end

@inline _ieee_bilinear_supported_eltype(
    ::Type{T}, ::Type{Float64}) where {T} =
    T <: Union{Float32,Float64,ComplexF32,ComplexF64}
@inline _ieee_bilinear_supported_eltype(
    ::Type{T}, ::Type{Float32}) where {T} =
    T <: Union{Float32,ComplexF32}

@inline function _ieee_bilinear_values_are_finite(values)
    @inbounds for value in values
        isfinite(value) || return false
    end
    return true
end

@inline _ieee_bilinear_matrix_is_finite(matrix::AbstractMatrix) =
    _ieee_bilinear_values_are_finite(matrix)
@inline _ieee_bilinear_matrix_is_finite(
    matrix::SparseArrays.AbstractSparseMatrixCSC) =
    _ieee_bilinear_values_are_finite(nonzeros(matrix))
@inline _ieee_bilinear_matrix_is_finite(matrix::LocalMassMatrix) =
    _ieee_bilinear_values_are_finite(matrix.vals)

function _ieee_bilinear_superaccumulator_supported(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    ::Type{R},
) where {R<:Union{Float32,Float64}}
    _ieee_bilinear_supported_eltype(eltype(left), R) || return false
    _ieee_bilinear_supported_eltype(eltype(matrix), R) || return false
    _ieee_bilinear_supported_eltype(eltype(right), R) || return false
    _ieee_bilinear_values_are_finite(left) || return false
    _ieee_bilinear_matrix_is_finite(matrix) || return false
    return _ieee_bilinear_values_are_finite(right)
end

@noinline function _bilinear_component_ieee_exact(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    component,
    ::Type{R},
    label::AbstractString,
) where {R<:Union{Float32,Float64}}
    accumulator = _IEEEBilinearAccumulator(R)
    _accumulate_bilinear_component_ieee!(
        accumulator, left, matrix, right, component)
    return _ieee_bilinear_finish(accumulator, label)
end

function _accumulate_bilinear_component_bigfloat(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    component,
)
    total = zero(BigFloat)
    @inbounds for row in axes(matrix, 1)
        left_value = left[row]
        iszero(left_value) && continue
        left_real = BigFloat(real(left_value))
        left_imag = BigFloat(imag(left_value))
        for column in axes(matrix, 2)
            matrix_value = matrix[row, column]
            right_value = right[column]
            (iszero(matrix_value) || iszero(right_value)) && continue
            total += _bilinear_component_bigfloat(
                left_real, left_imag,
                BigFloat(real(matrix_value)),
                BigFloat(imag(matrix_value)),
                BigFloat(real(right_value)),
                BigFloat(imag(right_value)),
                component,
            )
        end
    end
    return total
end

function _accumulate_bilinear_component_bigfloat(
    left::AbstractVector,
    matrix::SparseArrays.AbstractSparseMatrixCSC,
    right::AbstractVector,
    component,
)
    total = zero(BigFloat)
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    @inbounds for column in axes(matrix, 2)
        right_value = right[column]
        iszero(right_value) && continue
        right_real = BigFloat(real(right_value))
        right_imag = BigFloat(imag(right_value))
        for position in nzrange(matrix, column)
            left_value = left[rows[position]]
            matrix_value = values[position]
            (iszero(left_value) || iszero(matrix_value)) && continue
            total += _bilinear_component_bigfloat(
                BigFloat(real(left_value)),
                BigFloat(imag(left_value)),
                BigFloat(real(matrix_value)),
                BigFloat(imag(matrix_value)),
                right_real, right_imag,
                component,
            )
        end
    end
    return total
end


function _accumulate_bilinear_component_bigfloat(
    left::AbstractVector,
    matrix::LocalMassMatrix,
    right::AbstractVector,
    component,
)
    total = zero(BigFloat)
    @inbounds for position in eachindex(matrix.vals)
        left_value = left[matrix.rows[position]]
        matrix_value = matrix.vals[position]
        right_value = right[matrix.cols[position]]
        (iszero(left_value) || iszero(matrix_value) ||
         iszero(right_value)) && continue
        total += _bilinear_component_bigfloat(
            BigFloat(real(left_value)),
            BigFloat(imag(left_value)),
            BigFloat(real(matrix_value)),
            BigFloat(imag(matrix_value)),
            BigFloat(real(right_value)),
            BigFloat(imag(right_value)),
            component,
        )
    end
    return total
end

@noinline function _bilinear_component_bigfloat(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    component,
    ::Type{T},
    label::AbstractString,
) where {T<:AbstractFloat}
    if _ieee_bilinear_superaccumulator_supported(
            left, matrix, right, T)
        return _bilinear_component_ieee_exact(
            left, matrix, right, component, T, label)::T
    end
    return setprecision(BigFloat, _IEEE_BILINEAR_FALLBACK_PRECISION) do
        total = _accumulate_bilinear_component_bigfloat(
            left, matrix, right, component)
        converted = convert(T, total)
        isfinite(converted) ||
            throw(OverflowError(
                "$label is outside the representable $T range"))
        return converted
    end
end

function _finite_bilinear_component(
    left::AbstractVector,
    matrix::AbstractMatrix,
    right::AbstractVector,
    component,
    label::AbstractString,
)
    value = _bilinear_component(
        _dot_left_matrix_right(left, matrix, right), component)
    value_type = typeof(value)
    if value_type <: Union{Float32,Float64}
        if !isfinite(value) || _ieee_bilinear_requires_fallback(
                left, matrix, right, value_type)
            return _bilinear_component_bigfloat(
                left, matrix, right, component, value_type, label)
        end
        return value
    end
    isfinite(value) || error("$label produced a non-finite value")
    return value
end

"""
    compute_objective(I, Q)

Compute the quadratic objective J = Re(I† Q I).
"""
function compute_objective(I::Vector{<:Number}, Q::Matrix{<:Number})
    _validate_linear_system_inputs(Q, I, "quadratic objective")
    value = real(_dot_left_matrix_right(I, Q, I))
    value_type = typeof(value)
    if value_type <: Union{Float32,Float64}
        if !isfinite(value) || _ieee_bilinear_requires_fallback(
                I, Q, I, value_type)
            return _quadratic_objective_bigfloat(I, Q, value_type)
        end
        return value
    end
    isfinite(value) ||
        error("quadratic objective produced a non-finite value")
    return value
end

@inline function _quadratic_objective_from_product(
    I::Vector{<:Number},
    Q::Matrix{<:Number},
    QI::AbstractVector{<:Number},
)
    value = real(dot(I, QI))
    value_type = typeof(value)
    if isfinite(value) &&
       (!(value_type <: Union{Float32,Float64}) ||
        !_ieee_bilinear_requires_fallback(I, Q, I, value_type))
        return value
    end
    return compute_objective(I, Q)
end

@inline function _quadratic_objective_from_product(
    I::Vector{<:Number},
    Q::Matrix{<:Number},
    QI::AbstractVector{<:Number},
    product_used_fallback::Bool,
)
    value = real(dot(I, QI))
    !product_used_fallback && isfinite(value) && return value
    return compute_objective(I, Q)
end

"""
    solve_adjoint(Z, Q, I; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, check_gmres_convergence=true, check_true_residual=false, true_residual_factor=100.0)

Solve the adjoint system: Z† λ = Q I
Returns λ ∈ C^N.

When `solver=:gmres`, uses GMRES with the adjoint preconditioner P⁻ᴴ.
By default, an unconverged GMRES solve throws instead of returning an
unverified adjoint vector.
"""
function solve_adjoint(Z::AbstractMatrix{<:Number}, Q::Matrix{<:Number},
                       I::AbstractVector{<:Number};
                       solver::Symbol=:direct,
                       preconditioner=nothing,
                       gmres_precond_side::Symbol=:left,
                       gmres_tol::Float64=1e-8,
                       gmres_maxiter::Int=200,
                       gmres_memory::Int=20,
                       check_gmres_convergence::Bool=true,
                       check_true_residual::Bool=false,
                       true_residual_factor::Float64=100.0)
    solver in (:direct, :gmres) ||
        throw(ArgumentError(
            "Unknown solver: $solver (expected :direct or :gmres)"))
    N = _validate_linear_system_inputs(Z, I, "adjoint solve")
    size(Q) == (N, N) ||
        throw(DimensionMismatch(
            "adjoint objective matrix has size $(size(Q)), expected ($N, $N)"))
    _validate_known_matrix_entries(Q, "adjoint objective matrix")
    rhs = _finite_matrix_vector_product(Q, I, "adjoint RHS")
    if solver == :direct
        Z isa Matrix ||
            throw(ArgumentError(
                "Direct adjoint solver requires a dense Matrix; use solver=:gmres for operator-based systems."))
        adjoint_Z = adjoint(Z)
        return _solve_factored_linear_system(
            lu(adjoint_Z), adjoint_Z, rhs, "direct adjoint solution")
    else
        x, stats = solve_gmres_adjoint(Z, rhs;
                                        preconditioner=preconditioner,
                                        precond_side=gmres_precond_side,
                                        tol=gmres_tol, maxiter=gmres_maxiter,
                                        memory=gmres_memory,
                                        check_gmres_convergence=check_gmres_convergence)
        check_true_residual &&
            _assert_true_residual(adjoint(Z), x, rhs, "adjoint";
                                  tol=gmres_tol,
                                  factor=true_residual_factor)
        return _assert_finite_linear_vector(x, "GMRES adjoint solution")
    end
end

"""
    solve_adjoint_rhs(Z, rhs; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, check_gmres_convergence=true, check_true_residual=false, true_residual_factor=100.0)

Solve the adjoint system Z† λ = rhs where rhs is pre-computed.
Unlike `solve_adjoint(Z, Q, I)` which internally computes rhs = Q*I,
this accepts the RHS directly — useful when using `apply_Q` for matrix-free
Q application or for multi-angle objectives.
"""
function solve_adjoint_rhs(Z::AbstractMatrix{<:Number}, rhs::AbstractVector{<:Number};
                           solver::Symbol=:direct,
                           preconditioner=nothing,
                           gmres_precond_side::Symbol=:left,
                           gmres_tol::Float64=1e-8,
                           gmres_maxiter::Int=200,
                           gmres_memory::Int=20,
                           check_gmres_convergence::Bool=true,
                           check_true_residual::Bool=false,
                           true_residual_factor::Float64=100.0)
    solver in (:direct, :gmres) ||
        throw(ArgumentError(
            "Unknown solver: $solver (expected :direct or :gmres)"))
    if solver == :direct
        Z isa Matrix ||
            throw(ArgumentError(
                "Direct adjoint solver requires a dense Matrix; use solver=:gmres for operator-based systems."))
        _validate_linear_system_inputs(Z, rhs, "adjoint solve")
        complex_rhs = _as_complex_rhs(rhs)
        adjoint_Z = adjoint(Z)
        return _solve_factored_linear_system(
            lu(adjoint_Z), adjoint_Z, complex_rhs,
            "direct adjoint solution")
    else
        x, stats = solve_gmres_adjoint(Z, _as_complex_rhs(rhs);
                                        preconditioner=preconditioner,
                                        precond_side=gmres_precond_side,
                                        tol=gmres_tol, maxiter=gmres_maxiter,
                                        memory=gmres_memory,
                                        check_gmres_convergence=check_gmres_convergence)
        check_true_residual &&
            _assert_true_residual(adjoint(Z), x, rhs, "adjoint";
                                  tol=gmres_tol,
                                  factor=true_residual_factor)
        return _assert_finite_linear_vector(x, "GMRES adjoint solution")
    end
end

"""
    gradient_impedance(Mp, I, lambda; reactive=false)

Compute the adjoint gradient for impedance parameters:
  g[p] = -2 Re{ λ† (∂Z/∂θ_p) I }

For resistive impedance (Z_s = θ_p, default):
  ∂Z/∂θ_p = -M_p  →  g[p] = +2 Re{ λ† M_p I }

For reactive impedance (Z_s = iθ_p, reactive=true):
  ∂Z/∂θ_p = -iM_p  →  g[p] = +2 Re{ i λ† M_p I } = -2 Im{ λ† M_p I }

Returns g ∈ R^P.
"""
function gradient_impedance(Mp::Vector{<:AbstractMatrix},
                            I::Vector{<:Number},
                            lambda::Vector{<:Number};
                            reactive::Bool=false)
    matrix_size = _validate_mass_matrix_sizes(Mp)
    length(I) == matrix_size[2] ||
        throw(DimensionMismatch(
            "I length $(length(I)) != $(matrix_size[2])"))
    length(lambda) == matrix_size[1] ||
        throw(DimensionMismatch(
            "lambda length $(length(lambda)) != $(matrix_size[1])"))
    all(isfinite, I) ||
        throw(ArgumentError("I must contain only finite values"))
    all(isfinite, lambda) ||
        throw(ArgumentError("lambda must contain only finite values"))
    @inbounds for p in eachindex(Mp)
        _validate_known_matrix_entries(Mp[p], "patch mass matrix")
    end
    P = length(Mp)
    g = zeros(Float64, P)
    component = reactive ? Val(:imag) : Val(:real)
    for p in 1:P
        lMI_component = _finite_bilinear_component(
            lambda, Mp[p], I, component,
            "gradient_impedance bilinear product")
        if reactive
            # ∂Z/∂θ_p = -iM_p
            # g[p] = -2 Re{ λ† (-iM_p) I } = 2 Re{ i λ† M_p I } = -2 Im{ λ† M_p I }
            g[p] = -2 * lMI_component
        else
            # ∂Z/∂θ_p = -M_p
            # g[p] = -2 Re{ λ† (-M_p) I } = 2 Re{ λ† M_p I }
            g[p] = 2 * lMI_component
        end
    end
    all(isfinite, g) ||
        error("gradient_impedance produced non-finite values")
    return g
end
