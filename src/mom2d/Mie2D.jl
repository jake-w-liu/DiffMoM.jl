# Mie2D.jl — 2D Mie series for circular cylinder (TM polarization)
#
# Provides exact scattered field for validation of the 2D VIE solver.
# Convention: exp(+iωt), H₀⁽²⁾ for outgoing waves.

export mie_coefficients_2d, mie_scattered_field_2d, mie_total_field_2d

const _MIE2D_FALLBACK_PRECISION = 256
const _MIE2D_SMALL_ARGUMENT_CUTOFF = 1e-25
const _MIE2D_INTERNAL_SERIES_LIMIT = 0.5
const _MAX_MIE2D_ORDER = 100_000
const _MAX_MIE2D_FALLBACK_PRECISION = 16_384
const _MAX_MIE2D_EXACT_WORK = 2_000_000
const _MIE2D_AMOS_INTERNAL_ARGUMENT_LIMIT = 1.0e8
const _MAX_MIE2D_INTERNAL_PAIR_ORDER = 256
const _MAX_MIE2D_SMALL_SERIES_ORDER = 256
const _MIE2D_SEQUENCE_ORDER_THRESHOLD = 64
const _DEFAULT_MAX_MIE2D_FIELD_TERMS = 50_000_000

@inline function _mie2d_high_order_precision(x::Float64, N::Int)
    requested = 512 + 8N + 4max(0, -exponent(x))
    requested <= _MAX_MIE2D_FALLBACK_PRECISION ||
        throw(ArgumentError(
            "2D Mie order $N requires $requested exact bits, exceeding " *
            "the supported limit $_MAX_MIE2D_FALLBACK_PRECISION"))
    N + 1 <= _MAX_MIE2D_EXACT_WORK ÷ requested ||
        throw(ArgumentError(
            "2D Mie order $N exceeds the high-order exact-work limit " *
            "$_MAX_MIE2D_EXACT_WORK"))
    return requested
end

@inline function _validate_mie2d_exact_pair_work(N::Int, precision::Int)
    # Each order evaluates J_{n-1}, J_n, and J_{n+1}; forming their leading
    # powers performs exactly 1 + 3N(N+1)/2 arbitrary-precision updates.
    # N is validated at 100,000, so this exact count fits Int64.
    leading_updates = 1 + 3 * N * (N + 1) ÷ 2
    leading_updates <= _MAX_MIE2D_EXACT_WORK ÷ precision ||
        throw(ArgumentError(
            "internal 2D Mie order $N exceeds the exact-work limit " *
            "$_MAX_MIE2D_EXACT_WORK"))
    return nothing
end


@inline function _validate_mie2d_internal_pair_order(
        N::Int, magnitude::Float64)
    if magnitude <= _MIE2D_INTERNAL_SERIES_LIMIT &&
       N > _MAX_MIE2D_INTERNAL_PAIR_ORDER
        throw(ArgumentError(
            "internal 2D Mie order $N exceeds the supported small-internal " *
            "pair limit $_MAX_MIE2D_INTERNAL_PAIR_ORDER"))
    end
    return nothing
end

@inline function _mie2d_small_argument_precision(
        k0::Float64, a::Float64, order::Int)
    # The dielectric numerator can cancel through several powers of k0*a.
    # Float64 operands have at most 53 significant bits, so an exponent-based
    # guard supplies enough bits to settle every representable coefficient,
    # while retaining the fixed low-allocation precision at ordinary scale.
    product_exponent = exponent(k0) + exponent(a)
    requested = 256 + 4order + 2max(0, -product_exponent)
    return min(_MAX_MIE2D_FALLBACK_PRECISION, requested)
end

@inline function _mie2d_normalized_pair(
        value::ComplexF64, derivative::ComplexF64)
    scale = max(
        abs(real(value)), abs(imag(value)),
        abs(real(derivative)), abs(imag(derivative)))
    (isfinite(scale) && scale > 0.0) ||
        error("internal 2D Mie function pair cannot be normalized")
    return value / scale, derivative / scale
end

function _mie2d_internal_log_derivative(
        order::Int, z::ComplexF64)
    order >= 1 ||
        throw(ArgumentError(
            "cylindrical logarithmic derivative requires positive order"))
    start = Base.checked_add(
        order, max(32, ceil(Int, sqrt(80.0 * order))))
    following_ratio = zero(ComplexF64)
    ratio = zero(ComplexF64)
    inverse_z = inv(z)
    @inbounds for current in start:-1:order
        ratio = inv((2current) * inverse_z - following_ratio)
        isfinite(ratio) ||
            error("internal 2D Mie logarithmic derivative is non-finite")
        following_ratio = ratio
    end
    derivative = inv(ratio) - order * inverse_z
    isfinite(derivative) ||
        error("internal 2D Mie logarithmic derivative is non-finite")
    return derivative
end

function _mie2d_internal_pair_at_order(
        material_root::ComplexF64, x::Float64, order::Int)
    z = material_root * x
    if order >= 1 && order > abs(z) + 32
        derivative = _mie2d_internal_log_derivative(order, z)
        scale = max(1.0, abs(real(derivative)), abs(imag(derivative)))
        return ComplexF64(inv(scale), 0.0), derivative / scale
    end
    scaled = abs(imag(z)) > 500.0
    value = scaled ? besseljx(order, z) : besselj(order, z)
    next_value = scaled ? besseljx(order + 1, z) : besselj(order + 1, z)
    previous_value = order == 0 ? -next_value :
        (scaled ? besseljx(order - 1, z) : besselj(order - 1, z))
    return _mie2d_normalized_pair(
        value, (previous_value - next_value) / 2)
end

function _mie2d_internal_function_pairs(
        material_root::ComplexF64, x::Float64, N::Int)
    z = material_root * x
    (isfinite(z) && !iszero(z)) ||
        throw(ArgumentError(
            "internal 2D Mie size parameter must be finite and nonzero, got $z"))

    _validate_mie2d_internal_pair_order(N, abs(z))

    if abs(z) <= _MIE2D_INTERNAL_SERIES_LIMIT &&
       (iszero(z) || abs(z) < 1e-25)
        precision = _mie2d_small_argument_precision(x, 1.0, N)
        _validate_mie2d_exact_pair_work(N, precision)
        return setprecision(BigFloat, precision) do
            z_big = Complex{BigFloat}(z)
            result = Vector{Tuple{ComplexF64,ComplexF64}}(undef, N + 1)
            @inbounds for order in 0:N
                value, derivative =
                    _besselj_and_derivative_big_2d(order, z_big)
                scale = max(
                    abs(real(value)), abs(imag(value)),
                    abs(real(derivative)), abs(imag(derivative)))
                scale > 0 ||
                    error("internal 2D Mie exact function pair is zero")
                result[order + 1] = (
                    ComplexF64(value / scale),
                    ComplexF64(derivative / scale),
                )
            end
            result
        end
    end

    # Raw Amos values eventually underflow for the minimal high-order
    # solution even though the normalized value/derivative pair is finite.
    # Build each order independently so the downward logarithmic-derivative
    # path can take over beyond the turning region. This also covers the
    # exponentially scaled imaginary-argument branch without retaining an
    # additional O(N) value table.
    pairs = Vector{Tuple{ComplexF64,ComplexF64}}(undef, N + 1)
    @inbounds for order in 0:N
        pairs[order + 1] = _mie2d_internal_pair_at_order(
            material_root, x, order)
    end
    return pairs
end

function _besselj_integer_series_big_2d(
    order::Int,
    z::Complex{BigFloat},
)
    order >= 0 || throw(ArgumentError("Bessel order must be nonnegative."))
    half_z = z / 2
    term = one(z)
    @inbounds for factor in 1:order
        term *= half_z / factor
    end
    value = term
    recurrence_factor = -(half_z * half_z)
    for series_order in 1:128
        term *= recurrence_factor /
                (BigFloat(series_order) *
                 (BigFloat(order) + BigFloat(series_order)))
        updated = value + term
        updated == value && return updated
        value = updated
    end
    error("complex Bessel-J power series did not converge.")
end

@inline function _besselj_and_derivative_big_2d(
    order::Int,
    z::Complex{BigFloat},
)
    value = _besselj_integer_series_big_2d(order, z)
    next_value = _besselj_integer_series_big_2d(order + 1, z)
    previous_value = order == 0 ? -next_value :
                     _besselj_integer_series_big_2d(order - 1, z)
    return value, (previous_value - next_value) / 2
end

@inline function _exterior_bessel_values_big_2d(order::Int, x::BigFloat)
    Jn = besselj(order, x)
    Yn = bessely(order, x)
    Jnext = besselj(order + 1, x)
    Ynext = bessely(order + 1, x)
    Jprevious = order == 0 ? -Jnext : besselj(order - 1, x)
    Yprevious = order == 0 ? -Ynext : bessely(order - 1, x)
    dJn = (Jprevious - Jnext) / 2
    dYn = (Yprevious - Ynext) / 2
    return Jn, Complex{BigFloat}(Jn, -Yn), dJn,
           Complex{BigFloat}(dJn, -dYn)
end

function _small_mie2d_fallback_supported(
    k0::Float64,
    a::Float64,
    eps_r::Float64,
    pec::Bool,
)
    (pec || eps_r == 0.0) && return true
    return setprecision(BigFloat, 128) do
        internal_size = BigFloat(k0) * BigFloat(a) * sqrt(abs(BigFloat(eps_r)))
        internal_size <= BigFloat(_MIE2D_INTERNAL_SERIES_LIMIT)
    end
end

@inline function _mie2d_large_imaginary_log_derivative(
        order::Int, magnitude::BigFloat)
    # J_n(i y) = i^n I_n(y).  Differentiate the exponentially scaled
    # large-y expansion of I_n so exp(y) is never materialized.
    mu = BigFloat(4) * BigFloat(order)^2
    inverse = inv(magnitude)
    term = one(BigFloat)
    series = term
    derivative_series = zero(BigFloat)
    for index in 1:64
        odd = BigFloat(2index - 1)
        term *= -(mu - odd^2) * inverse / (BigFloat(8index))
        updated = series + term
        derivative_series -= BigFloat(index) * term * inverse
        if updated == series
            series = updated
            break
        end
        series = updated
    end
    ratio = one(BigFloat) - inverse / 2 + derivative_series / series
    return Complex{BigFloat}(zero(BigFloat), -ratio)
end

function _mie2d_besselj_values_miller_big(
        argument::BigFloat, maximum_order::Int)
    start = Base.checked_add(
        maximum_order,
        max(32, ceil(Int, sqrt(80.0 * max(maximum_order, 1)))))
    values = Vector{BigFloat}(undef, Base.checked_add(start, 2))
    values[start + 2] = zero(BigFloat)
    values[start + 1] = one(BigFloat)
    inverse = inv(argument)
    @inbounds for order in (start - 1):-1:0
        values[order + 1] = BigFloat(2order + 2) * inverse *
                            values[order + 2] - values[order + 3]
    end
    reference_zero = besselj(0, argument)
    reference_one = besselj(1, argument)
    reference_order = abs(reference_zero) >= abs(reference_one) ? 0 : 1
    reference = reference_order == 0 ? reference_zero : reference_one
    scale = reference / values[reference_order + 1]
    result = Vector{BigFloat}(undef, maximum_order + 1)
    @inbounds for order in 0:maximum_order
        result[order + 1] = values[order + 1] * scale
    end
    return result
end

function _mie2d_besselj_values_big(
        argument::BigFloat, maximum_order::Int)
    BigFloat(maximum_order) > argument &&
        return _mie2d_besselj_values_miller_big(
            argument, maximum_order)
    result = Vector{BigFloat}(undef, maximum_order + 1)
    result[1] = besselj(0, argument)
    maximum_order == 0 && return result
    result[2] = besselj(1, argument)
    inverse = inv(argument)
    @inbounds for order in 1:(maximum_order - 1)
        result[order + 2] = BigFloat(2order) * inverse *
                            result[order + 1] - result[order]
    end
    return result
end

function _mie2d_besselj_values_miller_float!(
        result::Vector{Float64},
        ratios::Vector{Float64},
        argument::Float64,
        maximum_order::Int)
    length(result) >= maximum_order + 1 ||
        throw(DimensionMismatch("Mie2D J workspace is too small"))
    length(ratios) >= maximum_order ||
        throw(DimensionMismatch("Mie2D ratio workspace is too small"))
    start = Base.checked_add(
        maximum_order,
        max(32, ceil(Int, sqrt(80.0 * max(maximum_order, 1)))))
    following_ratio = 0.0
    inverse = inv(argument)
    @inbounds for order in start:-1:1
        ratio = inv((2.0 * order) * inverse - following_ratio)
        isnan(ratio) &&
            error("exterior 2D Mie Bessel ratio is indeterminate")
        order <= maximum_order && (ratios[order] = ratio)
        following_ratio = ratio
    end

    reference_zero = besselj(0, argument)
    reference_one = besselj(1, argument)
    result[1] = reference_zero
    maximum_order == 0 && return result
    result[2] = reference_one
    @inbounds for order in 2:maximum_order
        ratio = ratios[order]
        result[order + 1] = if isfinite(ratio)
            result[order] * ratio
        else
            # A cylindrical-Bessel zero makes J_n/J_{n-1} genuinely
            # infinite.  Evaluate that isolated numerator directly, then
            # continue with the stable downward ratios at later orders.
            besselj(order, argument)
        end
    end
    return result
end

function _mie2d_besselj_values_miller_float(
        argument::Float64, maximum_order::Int)
    result = Vector{Float64}(undef, maximum_order + 1)
    ratios = Vector{Float64}(undef, maximum_order)
    return _mie2d_besselj_values_miller_float!(
        result, ratios, argument, maximum_order)
end

function _mie2d_exterior_sequences_float!(
        bessel_j::Vector{Float64},
        bessel_y::Vector{Float64},
        ratios::Vector{Float64},
        argument::Float64,
        maximum_order::Int)
    requested_order = Base.checked_add(maximum_order, 1)
    required_length = Base.checked_add(requested_order, 1)
    length(bessel_j) >= required_length ||
        throw(DimensionMismatch("Mie2D J workspace is too small"))
    length(bessel_y) >= required_length ||
        throw(DimensionMismatch("Mie2D Y workspace is too small"))
    length(ratios) >= requested_order ||
        throw(DimensionMismatch("Mie2D ratio workspace is too small"))
    if Float64(requested_order) > argument
        _mie2d_besselj_values_miller_float!(
            bessel_j, ratios, argument, requested_order)
    else
        bessel_j[1] = besselj(0, argument)
        bessel_j[2] = besselj(1, argument)
        inverse = inv(argument)
        @inbounds for order in 1:(requested_order - 1)
            bessel_j[order + 2] = (2.0 * order) * inverse *
                                          bessel_j[order + 1] -
                                          bessel_j[order]
        end
    end

    bessel_y[1] = bessely(0, argument)
    bessel_y[2] = bessely(1, argument)
    inverse = inv(argument)
    @inbounds for order in 1:(requested_order - 1)
        bessel_y[order + 2] = (2.0 * order) * inverse *
                                      bessel_y[order + 1] -
                                      bessel_y[order]
    end
    return bessel_j, bessel_y
end

function _mie2d_exterior_sequences_float(
        argument::Float64, maximum_order::Int)
    requested_order = Base.checked_add(maximum_order, 1)
    bessel_j = Vector{Float64}(undef, requested_order + 1)
    bessel_y = Vector{Float64}(undef, requested_order + 1)
    ratios = Vector{Float64}(undef, requested_order)
    return _mie2d_exterior_sequences_float!(
        bessel_j, bessel_y, ratios, argument, maximum_order)
end

@inline function _mie2d_exterior_pair_float(
        bessel_j::Vector{Float64},
        bessel_y::Vector{Float64},
        order::Int,
        inverse_argument::Float64)
    Jn = bessel_j[order + 1]
    Yn = bessel_y[order + 1]
    dJn = order == 0 ? -bessel_j[2] :
          bessel_j[order] - order * inverse_argument * Jn
    dYn = order == 0 ? -bessel_y[2] :
          bessel_y[order] - order * inverse_argument * Yn
    return Jn, ComplexF64(Jn, -Yn), dJn, ComplexF64(dJn, -dYn)
end

@noinline function _mie2d_fill_high_order_tail_big!(
        coefficients::Vector{ComplexF64}, center::Int,
        first_order::Int, maximum_order::Int,
        x::Float64, eps_r::Float64; pec::Bool=false)
    precision = _mie2d_high_order_precision(x, maximum_order)
    return setprecision(BigFloat, precision) do
        xb = BigFloat(x)
        inverse = inv(xb)
        bessel_j = _mie2d_besselj_values_big(xb, max(maximum_order, 1))
        y_previous = bessely(0, xb)
        y_current = bessely(1, xb)
        material_root = eps_r == 0.0 ? zero(Complex{BigFloat}) :
            sqrt(Complex{BigFloat}(BigFloat(eps_r), zero(BigFloat)))
        material_root_float = eps_r == 0.0 ? zero(ComplexF64) :
            sqrt(complex(eps_r))
        internal = material_root * xb

        # Once the internal order is beyond its turning region, compute every
        # needed J_n'/J_n ratio in one high-precision downward recurrence.
        # Promoting a Float64 continued-fraction result here loses roughly one
        # ulp per recurrence step and can corrupt a still-representable tiny
        # coefficient after a long explicit tail.
        first_log_order = if eps_r == 0.0
            maximum_order + 1
        else
            max(first_order, 1, floor(Int, abs(internal) + 32) + 1)
        end
        internal_ratios = Vector{Complex{BigFloat}}()
        if first_log_order <= maximum_order
            resize!(internal_ratios, maximum_order - first_log_order + 1)
            recurrence_start = Base.checked_add(
                maximum_order,
                max(32, ceil(Int, sqrt(80.0 * maximum_order))))
            following_ratio = zero(Complex{BigFloat})
            inverse_internal = inv(internal)
            @inbounds for current in recurrence_start:-1:first_log_order
                ratio = inv(BigFloat(2current) * inverse_internal -
                            following_ratio)
                isfinite(ratio) ||
                    error("high-order internal 2D Mie ratio is non-finite")
                current <= maximum_order &&
                    (internal_ratios[current - first_log_order + 1] = ratio)
                following_ratio = ratio
            end
        end

        @inbounds for order in 0:maximum_order
            Jn = bessel_j[order + 1]
            dJn = order == 0 ? -bessel_j[2] :
                bessel_j[order] - BigFloat(order) * inverse * Jn
            Yn = order == 0 ? y_previous : y_current
            dYn = order == 0 ? -y_current :
                y_previous - BigFloat(order) * inverse * y_current

            if order >= first_order
                Hn = Complex{BigFloat}(Jn, -Yn)
                dHn = Complex{BigFloat}(dJn, -dYn)
                coefficient = if pec
                    -Jn / Hn
                elseif eps_r == 0.0
                    order_big = BigFloat(order)
                    -(order_big * Jn - xb * dJn) /
                     (order_big * Hn - xb * dHn)
                else
                    first, second = if order >= first_log_order
                        ratio = internal_ratios[
                            order - first_log_order + 1]
                        derivative = inv(ratio) -
                            BigFloat(order) / internal
                        material_root * derivative,
                        one(Complex{BigFloat})
                    else
                        internal_value, internal_derivative =
                            _mie2d_internal_pair_at_order(
                                material_root_float, x, order)
                        material_root *
                            Complex{BigFloat}(internal_derivative),
                        Complex{BigFloat}(internal_value)
                    end
                    pair_scale = max(abs(first), abs(second))
                    pair_scale > 0 ||
                        error("high-order internal 2D Mie pair is zero")
                    first /= pair_scale
                    second /= pair_scale
                    -(first * Jn - second * dJn) /
                     (first * Hn - second * dHn)
                end
                converted = ComplexF64(coefficient)
                isfinite(converted) ||
                    error("2D Mie coefficient at order $order is non-finite")
                coefficients[center + order] = converted
                coefficients[center - order] = converted
            end

            if order >= 1 && order < maximum_order
                y_next = BigFloat(2order) * inverse * y_current - y_previous
                y_previous, y_current = y_current, y_next
            end
        end
        return coefficients
    end
end

@noinline function _mie_coefficients_large_internal_2d(
        k0a::Float64, eps_r::Float64, N::Int, coefficient_count::Int)
    eps_r != 0.0 ||
        throw(ArgumentError("large-internal fallback requires nonzero eps_r"))
    precision = min(
        _MAX_MIE2D_FALLBACK_PRECISION,
        max(512, 256 + max(0, exponent(k0a)) +
                 max(0, cld(exponent(eps_r), 2))))
    N + 1 <= _MAX_MIE2D_EXACT_WORK ÷ precision ||
        throw(ArgumentError(
            "large-internal 2D Mie order $N exceeds the precision-weighted " *
            "exact-work limit $_MAX_MIE2D_EXACT_WORK"))
    return setprecision(BigFloat, precision) do
        xb = BigFloat(k0a)
        root = sqrt(Complex{BigFloat}(BigFloat(eps_r), zero(BigFloat)))
        internal = root * xb

        exterior_miller = BigFloat(N) > xb ?
            _mie2d_besselj_values_miller_big(xb, N) : nothing
        exterior_previous = besselj(0, xb)
        exterior_current = besselj(1, xb)
        exterior_y_previous = bessely(0, xb)
        exterior_y_current = bessely(1, xb)
        positive_material = eps_r > 0.0
        internal_real = positive_material ? real(internal) : zero(BigFloat)
        internal_previous = positive_material ?
            besselj(0, internal_real) : zero(BigFloat)
        internal_current = positive_material ?
            besselj(1, internal_real) : zero(BigFloat)
        inverse_exterior = inv(xb)
        inverse_internal = inv(internal)

        coefficients = Vector{ComplexF64}(undef, coefficient_count)
        center = N + 1
        @inbounds for order in 0:N
            exterior_value = exterior_miller === nothing ?
                (order == 0 ? exterior_previous : exterior_current) :
                exterior_miller[order + 1]
            exterior_y = order == 0 ?
                exterior_y_previous : exterior_y_current
            exterior_derivative = if order == 0
                -(exterior_miller === nothing ?
                  exterior_current : exterior_miller[2])
            elseif exterior_miller !== nothing
                exterior_miller[order] - BigFloat(order) *
                    inverse_exterior * exterior_miller[order + 1]
            else
                exterior_previous - BigFloat(order) * inverse_exterior *
                                    exterior_current
            end
            exterior_y_derivative = if order == 0
                -exterior_y_current
            else
                exterior_y_previous - BigFloat(order) * inverse_exterior *
                                      exterior_y_current
            end
            internal_value, internal_derivative = if positive_material
                value = order == 0 ? internal_previous : internal_current
                derivative = if order == 0
                    -internal_current
                else
                    internal_previous - BigFloat(order) * inverse_internal *
                                        internal_current
                end
                Complex{BigFloat}(value, 0),
                Complex{BigFloat}(derivative, 0)
            else
                Complex{BigFloat}(one(BigFloat), 0),
                _mie2d_large_imaginary_log_derivative(
                    order, abs(internal))
            end
            first = root * internal_derivative
            second = internal_value
            scale = max(abs(first), abs(second))
            scale > 0 ||
                error("large-internal 2D Mie boundary pair is zero")
            first /= scale
            second /= scale
            hankel = Complex{BigFloat}(exterior_value, -exterior_y)
            hankel_derivative = Complex{BigFloat}(
                exterior_derivative, -exterior_y_derivative)
            coefficient =
                -(first * exterior_value - second * exterior_derivative) /
                 (first * hankel - second * hankel_derivative)
            converted = ComplexF64(coefficient)
            isfinite(converted) ||
                error("large-internal 2D Mie coefficient at order $order " *
                      "is non-finite")
            coefficients[center + order] = converted
            coefficients[center - order] = converted

            if order >= 1 && order < N
                exterior_next = exterior_miller === nothing ?
                    BigFloat(2order) * inverse_exterior *
                    exterior_current - exterior_previous : zero(BigFloat)
                exterior_y_next = BigFloat(2order) * inverse_exterior *
                                  exterior_y_current - exterior_y_previous
                internal_next = positive_material ?
                    BigFloat(2order) * real(inverse_internal) *
                    internal_current - internal_previous : zero(BigFloat)
                if exterior_miller === nothing
                    exterior_previous, exterior_current =
                        exterior_current, exterior_next
                end
                exterior_y_previous, exterior_y_current =
                    exterior_y_current, exterior_y_next
                internal_previous, internal_current =
                    internal_current, internal_next
            end
        end
        return coefficients, N
    end
end

function _mie_coefficients_small_bigfloat_2d(
    k0::Float64,
    a::Float64,
    eps_r::Float64,
    N::Int,
    coefficient_count::Int,
    pec::Bool,
)
    N <= _MAX_MIE2D_SMALL_SERIES_ORDER ||
        throw(ArgumentError(
            "small-argument 2D Mie order $N exceeds the supported " *
            "series limit $_MAX_MIE2D_SMALL_SERIES_ORDER"))
    precision = _mie2d_small_argument_precision(k0, a, N)
    # A dielectric order evaluates the internal J series at n-1, n, and n+1.
    # Count their leading-factor recurrences instead of pretending this
    # quadratic kernel is one unit per order. PEC/ENZ do not evaluate the
    # internal series and keep the linear precision-weighted model.
    work_units = if pec || eps_r == 0.0
        N + 1
    else
        1 + 3 * N * (N + 1) ÷ 2
    end
    work_units <= _MAX_MIE2D_EXACT_WORK ÷ precision ||
        throw(ArgumentError(
            "small-argument 2D Mie order $N exceeds the " *
            "precision-weighted exact-work limit " *
            "$_MAX_MIE2D_EXACT_WORK"))
    coefficients = Vector{ComplexF64}(undef, coefficient_count)
    center = N + 1
    setprecision(BigFloat, precision) do
        x = BigFloat(k0) * BigFloat(a)
        material_root = pec || eps_r == 0.0 ?
                        zero(Complex{BigFloat}) :
                        sqrt(Complex{BigFloat}(BigFloat(eps_r), zero(BigFloat)))
        internal_argument = material_root * x

        @inbounds for order in 0:N
            Jn, Hn, dJn, dHn = _exterior_bessel_values_big_2d(order, x)
            coefficient = if pec
                -Jn / Hn
            elseif eps_r == 0.0
                order_big = BigFloat(order)
                -(order_big * Jn - x * dJn) /
                 (order_big * Hn - x * dHn)
            else
                Jinternal, dJinternal =
                    _besselj_and_derivative_big_2d(order, internal_argument)
                -(material_root * dJinternal * Jn - Jinternal * dJn) /
                 (material_root * dJinternal * Hn - Jinternal * dHn)
            end
            converted = ComplexF64(coefficient)
            isfinite(converted) ||
                error("2D Mie coefficient at order $order is non-finite.")
            coefficients[center + order] = converted
            coefficients[center - order] = converted
        end
    end
    return coefficients, N
end

@inline function _validated_mie2d_positive(value::Float64,
                                           label::AbstractString)
    (isfinite(value) && value > 0.0) ||
        throw(ArgumentError("$label must be finite and positive, got $value"))
    return value
end

function _validated_mie2d_order(k0a::Float64,
                                nmax::Union{Nothing,Int})
    (isfinite(k0a) && k0a >= 0.0) ||
        throw(ArgumentError(
            "2D Mie size parameter k0*a must be finite and nonnegative, got $k0a"))
    N = if nmax === nothing
        estimate = k0a + 4 * cbrt(k0a) + 2
        (isfinite(estimate) && estimate <= typemax(Int)) ||
            throw(ArgumentError(
                "automatic 2D Mie truncation order is not representable for size parameter $k0a"))
        order = try
            ceil(Int, estimate)
        catch err
            err isa Union{InexactError,OverflowError} || rethrow()
            throw(ArgumentError(
                "automatic 2D Mie truncation order is not representable for size parameter $k0a"))
        end
        max(10, order)
    else
        nmax >= 0 ||
            throw(ArgumentError("nmax must be nonnegative, got $nmax"))
        nmax
    end
    coefficient_count = try
        Base.checked_add(Base.checked_mul(2, N), 1)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("nmax=$N is too large to index a coefficient vector"))
    end
    N <= _MAX_MIE2D_ORDER ||
        throw(ArgumentError(
            "2D Mie order $N exceeds the supported limit " *
            "$_MAX_MIE2D_ORDER"))
    return N, coefficient_count
end

"""
    mie_coefficients_2d(k0, a, eps_r; nmax=nothing, pec=false)

Compute 2D Mie scattering coefficients cₙ for a circular cylinder.

Arguments:
- `k0`: free-space wavenumber
- `a`: cylinder radius
- `eps_r`: relative permittivity (ignored if `pec=true`)
- `nmax`: maximum order (auto-determined if `nothing`)
- `pec`: if true, compute PEC cylinder coefficients

Returns `(c, N)` where `c` is the coefficient vector indexed from -N:N
(stored as c[n + N + 1]) and `N` is the (auto-selected) truncation order.
"""
function mie_coefficients_2d(k0::Float64, a::Float64, eps_r::Float64;
                              nmax::Union{Nothing,Int}=nothing, pec::Bool=false)
    k0 = _validated_mie2d_positive(k0, "k0")
    a = _validated_mie2d_positive(a, "a")
    k0a = k0 * a
    isfinite(k0a) ||
        throw(ArgumentError(
            "2D Mie size parameter k0*a must be representable as Float64; " *
            "got k0=$k0 and a=$a"))
    N, coefficient_count = _validated_mie2d_order(k0a, nmax)
    if !pec
        isfinite(eps_r) ||
            throw(ArgumentError("eps_r must be finite, got $eps_r"))
    end

    # A cylinder identical to the exterior medium has exactly zero contrast.
    # Returning the analytic result also avoids overflow in high-order Hankel
    # intermediates for electrically tiny matched cylinders.
    if !pec && eps_r == 1.0
        return zeros(ComplexF64, coefficient_count), N
    end

    if k0a <= _MIE2D_SMALL_ARGUMENT_CUTOFF &&
       _small_mie2d_fallback_supported(k0, a, eps_r, pec)
        N <= _MAX_MIE2D_SMALL_SERIES_ORDER ||
            throw(ArgumentError(
                "small-argument 2D Mie order $N exceeds the supported " *
                "series limit $_MAX_MIE2D_SMALL_SERIES_ORDER"))
        return _mie_coefficients_small_bigfloat_2d(
            k0, a, eps_r, N, coefficient_count, pec)
    end

    if !pec && eps_r != 0.0
        root_magnitude = sqrt(abs(eps_r))
        large_internal = root_magnitude >
                         _MIE2D_AMOS_INTERNAL_ARGUMENT_LIMIT / k0a
        if large_internal
            return _mie_coefficients_large_internal_2d(
                k0a, eps_r, N, coefficient_count)
        end
        internal_magnitude = root_magnitude * k0a
        _validate_mie2d_internal_pair_order(N, internal_magnitude)
    end

    stable_limit = k0a + 4cbrt(k0a) + 2
    explicit_high_order_tail =
        nmax !== nothing && N > 10 && N > k0a &&
        Float64(N - 1) >= stable_limit
    if explicit_high_order_tail
        _mie2d_high_order_precision(k0a, N)
    end

    c = Vector{ComplexF64}(undef, coefficient_count)
    center = N + 1
    # Small ordinary requests retain the allocation-free scalar SpecialFunctions
    # path.  Beyond this bounded threshold, compute each exterior sequence once
    # so explicit industrial orders do O(N), rather than O(N^2), recurrence
    # work in OpenLibm's individual integer-order calls.
    use_exterior_sequences = N > _MIE2D_SEQUENCE_ORDER_THRESHOLD
    exterior_j, exterior_y = use_exterior_sequences ?
        _mie2d_exterior_sequences_float(k0a, N) : (nothing, nothing)
    inverse_k0a = inv(k0a)
    exact_tail_first_order = explicit_high_order_tail ?
        max(11, ceil(Int, stable_limit) + 1) : N + 1
    exact_tail_filled = false

    if pec
        for n in 0:N
            Jn, Hn = if use_exterior_sequences
                sequence_j, sequence_h, _, _ = _mie2d_exterior_pair_float(
                    exterior_j, exterior_y, n, inverse_k0a)
                sequence_j, sequence_h
            else
                scalar_j = besselj(n, k0a)
                scalar_j, besselh(n, 2, k0a)
            end
            coefficient = -Jn / Hn
            if !isfinite(coefficient)
                _mie2d_fill_high_order_tail_big!(
                    c, center, n, N, k0a, eps_r; pec=true)
                exact_tail_filled = true
                break
            end
            c[center + n] = coefficient
            c[center - n] = coefficient
        end
    elseif eps_r == 0.0
        # Analytic k1→0 limit. For order q=|n|,
        # k1*Jq'(k1*a)/Jq(k1*a) → q/a.
        for n in 0:N
            Jn_k0a, Hn_k0a, dJn_k0a, dHn_k0a = if use_exterior_sequences
                _mie2d_exterior_pair_float(
                    exterior_j, exterior_y, n, inverse_k0a)
            else
                scalar_j = besselj(n, k0a)
                scalar_j_previous = besselj(n - 1, k0a)
                scalar_h = besselh(n, 2, k0a)
                scalar_h_previous = besselh(n - 1, 2, k0a)
                scalar_j, scalar_h,
                scalar_j_previous - n * inverse_k0a * scalar_j,
                scalar_h_previous - n * inverse_k0a * scalar_h
            end
            coefficient =
                -(n * Jn_k0a - k0a * dJn_k0a) /
                 (n * Hn_k0a - k0a * dHn_k0a)
            if !isfinite(coefficient)
                _mie2d_fill_high_order_tail_big!(
                    c, center, n, N, k0a, eps_r)
                exact_tail_filled = true
                break
            end
            c[center + n] = coefficient
            c[center - n] = coefficient
        end
    else
        material_root = sqrt(complex(eps_r))
        k1a = material_root * k0a
        isfinite(k1a) ||
            throw(ArgumentError(
                "internal 2D Mie size parameter must be finite, got $k1a"))

        use_internal_pairs =
            abs(k1a) <= _MIE2D_INTERNAL_SERIES_LIMIT ||
            abs(imag(k1a)) > 500.0
        internal_pairs = use_internal_pairs ?
            _mie2d_internal_function_pairs(material_root, k0a, N) : nothing
        for n in 0:N
            # Bessel function derivatives using recurrence:
            # f'_n(x) = f_{n-1}(x) - (n/x) f_n(x)
            Jn_k0a, Hn_k0a, dJn_k0a, dHn_k0a = if use_exterior_sequences
                _mie2d_exterior_pair_float(
                    exterior_j, exterior_y, n, inverse_k0a)
            else
                scalar_j = besselj(n, k0a)
                scalar_j_previous = besselj(n - 1, k0a)
                scalar_h = besselh(n, 2, k0a)
                scalar_h_previous = besselh(n - 1, 2, k0a)
                scalar_j, scalar_h,
                scalar_j_previous - n * inverse_k0a * scalar_j,
                scalar_h_previous - n * inverse_k0a * scalar_h
            end

            # Beyond the internal turning region raw Amos values for the
            # minimal J solution underflow. Switch per order to a normalized
            # logarithmic-derivative pair even when lower orders were ordinary.
            Jn_k1a, dJn_k1a = if use_internal_pairs
                @inbounds internal_pairs[n + 1]
            elseif n >= 1 && n > abs(k1a) + 32
                _mie2d_internal_pair_at_order(material_root, k0a, n)
            else
                value = besselj(n, k1a)
                previous = besselj(n - 1, k1a)
                value, previous - (n / k1a) * value
            end

            # Divide the boundary-condition numerator and denominator by k0.
            # The dimensionless material root and size parameters stay finite
            # when k0 and a are individually at opposite Float64 extremes.
            num = -(material_root * dJn_k1a * Jn_k0a -
                    Jn_k1a * dJn_k0a)
            den =  (material_root * dJn_k1a * Hn_k0a -
                    Jn_k1a * dHn_k0a)
            coefficient = num / den
            if !isfinite(coefficient)
                _mie2d_fill_high_order_tail_big!(
                    c, center, n, N, k0a, eps_r)
                exact_tail_filled = true
                break
            end
            c[center + n] = coefficient
            c[center - n] = coefficient
        end
    end

    if explicit_high_order_tail && !exact_tail_filled
        _mie2d_fill_high_order_tail_big!(
            c, center, exact_tail_first_order, N, k0a, eps_r; pec=pec)
    end

    return c, N
end

function _validate_mie2d_observations(k0::Float64, a::Float64,
                                      r_obs::AbstractVector{Vec2},
                                      phi_inc::Float64)
    isfinite(phi_inc) ||
        throw(ArgumentError("phi_inc must be finite, got $phi_inc"))
    @inbounds for m in eachindex(r_obs)
        point = r_obs[m]
        all(isfinite, point) ||
            throw(ArgumentError(
                "r_obs[$m] components must be finite, got $point"))
        rho = hypot(point[1], point[2])
        tolerance = 16 * eps(max(rho, a))
        near_surface = rho > 0.0 &&
                       rho + tolerance >= a &&
                       rho >= a * (1.0 - 64eps(Float64))
        rho >= a || near_surface ||
            throw(DomainError(
                rho,
                "r_obs[$m] lies inside the cylinder: radius $rho < a=$a"))
    end
    return nothing
end

"""
    mie_scattered_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0, nmax=nothing,
                           pec=false, max_field_terms=50_000_000,
                           max_output_bytes=2_000_000_000)

Compute exact scattered field at observation points for a circular cylinder.
Observation points must lie on or outside the cylinder (`ρ ≥ a`); the
exterior-series surface limit is supported.
`max_output_bytes` caps the raw payload of the returned vector before field
work is performed.

E_z^scat(ρ,φ) = E₀ Σ_n (-i)ⁿ cₙ Hₙ⁽²⁾(k₀ρ) eⁱⁿᶠ

Arguments:
- `r_obs`: vector of observation positions (Vec2)
- `phi_inc`: incident angle (0 = +x direction)
"""
function mie_scattered_field_2d(k0::Float64, a::Float64, eps_r::Float64,
                                 r_obs::AbstractVector{Vec2};
                                 phi_inc::Float64=0.0, nmax=nothing,
                                 pec::Bool=false,
                                 max_field_terms::Int=
                                     _DEFAULT_MAX_MIE2D_FIELD_TERMS,
                                 max_output_bytes::Integer=
                                     _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    k0 = _validated_mie2d_positive(k0, "k0")
    a = _validated_mie2d_positive(a, "a")
    if !pec
        isfinite(eps_r) ||
            throw(ArgumentError("eps_r must be finite, got $eps_r"))
    end
    matched_medium = !pec && eps_r == 1.0
    max_field_terms >= 1 ||
        throw(ArgumentError(
            "max_field_terms must be positive, got $max_field_terms"))
    M = length(r_obs)
    output_bytes = _checked_array_payload_bytes(
        ComplexF64, M; label="2D Mie scattered-field output")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "2D Mie scattered-field output", "max_output_bytes")
    _validate_mie2d_observations(k0, a, r_obs, phi_inc)
    requested_order, _ = _validated_mie2d_order(k0 * a, nmax)
    matched_medium && return zeros(ComplexF64, length(r_obs))
    small_coefficient_product = k0 * a <= _MIE2D_SMALL_ARGUMENT_CUTOFF &&
        _small_mie2d_fallback_supported(k0, a, eps_r, pec)
    exact_observation_count = 0
    ordinary_observation_count = 0
    maximum_exact_precision = 0
    @inbounds for point in r_obs
        radial_argument = k0 * hypot(point[1], point[2])
        uncertain = small_coefficient_product ||
                    !isfinite(radial_argument) ||
                    iszero(radial_argument) ||
                    eps(radial_argument) >= 0.125 ||
                    exponent(radial_argument) > 40
        if uncertain
            exact_observation_count += 1
            requested_precision = max(
                2304,
                256 + 4requested_order + 2max(0, -exponent(k0)) +
                2max(0, -exponent(a)),
            )
            requested_precision <= _MAX_MIE2D_FALLBACK_PRECISION ||
                throw(ArgumentError(
                    "2D Mie exceptional observation requires " *
                    "$requested_precision exact bits, exceeding the limit " *
                    "$_MAX_MIE2D_FALLBACK_PRECISION"))
            maximum_exact_precision = max(
                maximum_exact_precision, requested_precision)
        else
            ordinary_observation_count += 1
        end
    end
    terms_per_ordinary_observation = try
        requested_order <= _MIE2D_SEQUENCE_ORDER_THRESHOLD ?
            Base.Checked.checked_add(
                Base.Checked.checked_mul(2, requested_order), 1) :
            Base.Checked.checked_add(requested_order, 1)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("2D Mie ordinary field work overflows Int"))
    end
    ordinary_terms = try
        Base.Checked.checked_mul(
            ordinary_observation_count, terms_per_ordinary_observation)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("2D Mie ordinary field work overflows Int"))
    end
    ordinary_terms <= max_field_terms ||
        throw(ArgumentError(
            "2D Mie field evaluation requires $ordinary_terms ordinary " *
            "series terms, exceeding max_field_terms=$max_field_terms"))
    if exact_observation_count > 0 || small_coefficient_product
        linear_units = requested_order + 1
        small_dielectric = small_coefficient_product && !pec && eps_r != 0.0
        coefficient_units = small_dielectric ?
            1 + 3 * requested_order * (requested_order + 1) ÷ 2 :
            linear_units
        observation_units = small_dielectric ? coefficient_units : linear_units
        coefficient_precision = small_coefficient_product ?
            _mie2d_small_argument_precision(k0, a, requested_order) : 0
        coefficient_work = coefficient_units * coefficient_precision
        available_observation_work =
            _MAX_MIE2D_EXACT_WORK - coefficient_work
        per_observation_work = observation_units * maximum_exact_precision
        available_observation_work >= 0 &&
            (iszero(per_observation_work) ||
             exact_observation_count <=
                available_observation_work ÷ per_observation_work) ||
            throw(ArgumentError(
                "2D Mie exceptional observations exceed the exact-work " *
                "limit " *
                "$_MAX_MIE2D_EXACT_WORK"))
    end
    c, N = mie_coefficients_2d(k0, a, eps_r; nmax=nmax, pec=pec)

    E_scat = zeros(ComplexF64, M)
    sequence_workspace = N > _MIE2D_SEQUENCE_ORDER_THRESHOLD ? (
        Vector{Float64}(undef, N + 2),
        Vector{Float64}(undef, N + 2),
        Vector{Float64}(undef, N + 1),
    ) : nothing

    for m in 1:M
        rho = hypot(r_obs[m][1], r_obs[m][2])
        phi = atan(r_obs[m][2], r_obs[m][1])
        phase_delta = rem2pi(
            rem2pi(phi, RoundNearest) -
            rem2pi(phi_inc, RoundNearest), RoundNearest)
        radial_argument = k0 * rho
        radial_phase_uncertain = if isfinite(radial_argument) &&
                                    radial_argument > 0.0
            argument_ulp = eps(radial_argument)
            argument_ulp >= 0.125 ||
                exponent(radial_argument) > 40
        else
            true
        end
        if small_coefficient_product ||
           radial_phase_uncertain
            E_scat[m] = _mie_scattered_point_big_2d(
                k0, a, eps_r, r_obs[m], phi_inc, N, pec, m;
                coefficients=!small_coefficient_product ? c : nothing)
            continue
        end

        if N <= _MIE2D_SEQUENCE_ORDER_THRESHOLD
            @inbounds for n in -N:N
                coefficient = c[n + N + 1]
                iszero(coefficient) && continue
                E_scat[m] += (-im + 0.0)^n * coefficient *
                             besselh(n, 2, radial_argument) *
                             cis(rem2pi(n * phase_delta, RoundNearest))
            end
        else
            radial_j, radial_y, radial_ratios = sequence_workspace
            _mie2d_exterior_sequences_float!(
                radial_j, radial_y, radial_ratios, radial_argument, N)
            @inbounds for n in 0:N
                coefficient = c[N + 1 + n]
                iszero(coefficient) && continue
                radial_hankel = ComplexF64(
                    radial_j[n + 1], -radial_y[n + 1])
                angular = n == 0 ? 1.0 :
                          2cos(rem2pi(n * phase_delta, RoundNearest))
                E_scat[m] += (-im + 0.0)^n * coefficient *
                             radial_hankel * angular
            end
        end
        isfinite(E_scat[m]) ||
            error("2D Mie scattered field at observation $m is non-finite")
    end

    return E_scat
end

@inline function _mie2d_incident_phase_requires_exact(
        k0::Float64, incident_angle::Float64,
        direction::Vec2, observation::Vec2)
    first = direction[1] * observation[1]
    second = direction[2] * observation[2]
    @inbounds for index in 1:2
        primitive_direction = direction[index]
        primitive_coordinate = observation[index]
        product = primitive_direction * primitive_coordinate
        if !iszero(primitive_direction) && !iszero(primitive_coordinate) &&
           (!isfinite(product) || abs(product) < floatmin(Float64))
            return true
        end
    end
    total = first + second
    phase = k0 * total
    if !iszero(k0) && !iszero(total) &&
       (!isfinite(phase) || abs(phase) < floatmin(Float64))
        return true
    end
    isfinite(first) && isfinite(second) && isfinite(phase) || return true
    magnitude = abs(first) + abs(second)
    if signbit(first) != signbit(second) &&
       abs(total) <= 64eps(Float64) * magnitude
        return true
    end
    # `direction` contains rounded Float64 trigonometric values.  Bound the
    # phase error caused by those rounded values before multiplying by a large
    # observation coordinate; promoting them after the fact cannot recover
    # the API angle.  The exact branch evaluates sin/cos at the stored angle.
    direction_error_bound = abs(k0) * (
        abs(observation[1]) * eps(abs(direction[1])) +
        abs(observation[2]) * eps(abs(direction[2])))
    (!isfinite(direction_error_bound) ||
     direction_error_bound > 64eps(Float64)) && return true
    phase_ulp = eps(abs(phase))
    return phase_ulp >= 0.125 ||
           exponent(max(abs(first), abs(second), abs(k0))) > 40
end

@inline function _mie2d_incident_phase_precision(
        k0::Float64, incident_angle::Float64, observation::Vec2)
    coordinate_exponent = 0
    @inbounds for component in observation
        iszero(component) && continue
        coordinate_exponent = max(
            coordinate_exponent, exponent(abs(component)))
    end
    phase_exponent = max(
        0, exponent(abs(k0)) + coordinate_exponent)
    angle_exponent = iszero(incident_angle) ? 0 :
        max(0, exponent(abs(incident_angle)))
    requested = 256 + phase_exponent + angle_exponent
    requested <= _MAX_MIE2D_FALLBACK_PRECISION ||
        throw(ArgumentError(
            "2D Mie incident phase requires $requested exact bits, " *
            "exceeding the limit $_MAX_MIE2D_FALLBACK_PRECISION"))
    return max(512, requested)
end

@noinline function _mie_scattered_point_big_2d(
        k0::Float64, a::Float64, eps_r::Float64, point::Vec2,
        phi_inc::Float64, N::Int, pec::Bool, index;
        coefficients::Union{Nothing,Vector{ComplexF64}}=nothing)
    requested_precision = max(
        2304,
        256 + 4N + 2max(0, -exponent(k0)) +
        2max(0, -exponent(a)))
    requested_precision <= _MAX_MIE2D_FALLBACK_PRECISION ||
        throw(ArgumentError(
            "2D Mie exceptional observation requires $requested_precision " *
            "exact bits, exceeding the limit " *
            "$_MAX_MIE2D_FALLBACK_PRECISION"))
    recomputes_dielectric_coefficients =
        coefficients === nothing && !pec && eps_r != 0.0
    work_units = recomputes_dielectric_coefficients ?
        1 + 3 * N * (N + 1) ÷ 2 : N + 1
    work_units <= _MAX_MIE2D_EXACT_WORK ÷ requested_precision ||
        throw(ArgumentError(
            "2D Mie exceptional observation exceeds the exact-work " *
            "limit $_MAX_MIE2D_EXACT_WORK"))
    precision = requested_precision
    return setprecision(BigFloat, precision) do
        xb = BigFloat(k0) * BigFloat(a)
        rho = hypot(BigFloat(point[1]), BigFloat(point[2]))
        radial = BigFloat(k0) * rho
        phi = atan(BigFloat(point[2]), BigFloat(point[1]))
        incident_angle = BigFloat(phi_inc)
        material_root = pec || eps_r == 0.0 ? zero(Complex{BigFloat}) :
                        sqrt(Complex{BigFloat}(BigFloat(eps_r), zero(BigFloat)))
        internal = material_root * xb
        total = zero(Complex{BigFloat})
        @inbounds for order in 0:N
            coefficient = if coefficients !== nothing
                Complex{BigFloat}(coefficients[N + 1 + order])
            else
                Jn, Hn, dJn, dHn =
                    _exterior_bessel_values_big_2d(order, xb)
                if pec
                    -Jn / Hn
                elseif eps_r == 0.0
                    order_big = BigFloat(order)
                    -(order_big * Jn - xb * dJn) /
                     (order_big * Hn - xb * dHn)
                elseif abs(internal) <= _MIE2D_INTERNAL_SERIES_LIMIT
                    Jinternal, dJinternal =
                        _besselj_and_derivative_big_2d(order, internal)
                    -(material_root * dJinternal * Jn - Jinternal * dJn) /
                     (material_root * dJinternal * Hn - Jinternal * dHn)
                else
                    throw(ArgumentError(
                        "2D Mie exceptional observation requires an " *
                        "internal argument within the supported series range"))
                end
            end
            radial_hankel = Complex{BigFloat}(
                besselj(order, radial), -bessely(order, radial))
            angular = order == 0 ? one(BigFloat) :
                      2cos(BigFloat(order) * (phi - incident_angle))
            term = Complex{BigFloat}(zero(BigFloat), -one(BigFloat))^order *
                   coefficient * radial_hankel * angular
            total += term
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            error("2D Mie scattered field at observation $index is non-finite")
        return converted
    end
end

@noinline function _mie_incident_phase_big_2d(
        k0::Float64,
        incident_angle::Float64,
        observation::Vec2,
        index)
    precision = _mie2d_incident_phase_precision(
        k0, incident_angle, observation)
    return setprecision(BigFloat, precision) do
        angle = BigFloat(incident_angle)
        direction_x = cos(angle)
        direction_y = sin(angle)
        phase = BigFloat(k0) *
                (direction_x * BigFloat(observation[1]) +
                 direction_y * BigFloat(observation[2]))
        value = ComplexF64(
            exp(Complex{BigFloat}(zero(BigFloat), -phase)))
        isfinite(value) ||
            error(
                "2D Mie incident field at observation $index is non-finite")
        return value
    end
end

"""
    mie_total_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0, nmax=nothing,
                       pec=false, max_field_terms=50_000_000,
                       max_output_bytes=2_000_000_000)

Compute exact total field (incident + scattered) at observation points on or
outside the cylinder (ρ ≥ a). `max_output_bytes` caps the raw payload of
the returned vector before incident- or scattered-field work is performed.
"""
function mie_total_field_2d(k0::Float64, a::Float64, eps_r::Float64,
                             r_obs::AbstractVector{Vec2};
                             phi_inc::Float64=0.0, nmax=nothing,
                             pec::Bool=false,
                             max_field_terms::Int=
                                 _DEFAULT_MAX_MIE2D_FIELD_TERMS,
                             max_output_bytes::Integer=
                                 _DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    k0 = _validated_mie2d_positive(k0, "k0")
    a = _validated_mie2d_positive(a, "a")
    if !pec
        isfinite(eps_r) ||
            throw(ArgumentError("eps_r must be finite, got $eps_r"))
    end
    output_bytes = _checked_array_payload_bytes(
        ComplexF64, length(r_obs); label="2D Mie total-field output")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "2D Mie total-field output", "max_output_bytes")
    _validate_mie2d_observations(k0, a, r_obs, phi_inc)
    khat = Vec2(cos(phi_inc), sin(phi_inc))
    exact_incident_work = 0
    for point in r_obs
        if _mie2d_incident_phase_requires_exact(
                k0, phi_inc, khat, point)
            point_work = 3 * _mie2d_incident_phase_precision(
                k0, phi_inc, point)
            point_work <= _MAX_MIE2D_EXACT_WORK - exact_incident_work ||
                throw(ArgumentError(
                    "2D Mie incident phases require more than the " *
                    "$_MAX_MIE2D_EXACT_WORK exact-work-unit limit"))
            exact_incident_work += point_work
        end
    end

    E_total = mie_scattered_field_2d(k0, a, eps_r, r_obs;
                                     phi_inc=phi_inc, nmax=nmax, pec=pec,
                                     max_field_terms=max_field_terms,
                                     max_output_bytes=max_output_bytes)
    for m in eachindex(r_obs)
        incident = if _mie2d_incident_phase_requires_exact(
                k0, phi_inc, khat, r_obs[m])
            _mie_incident_phase_big_2d(k0, phi_inc, r_obs[m], m)
        else
            exp(-im * (k0 * dot(khat, r_obs[m])))
        end
        E_total[m] += incident
        isfinite(E_total[m]) ||
            error("2D Mie total field at observation $m is non-finite")
    end
    return E_total
end
