# Mie.jl — sphere Mie-theory reference utilities

export mie_s1s2_pec, mie_bistatic_rcs_pec
export mie_s1s2_dielectric, mie_bistatic_rcs_dielectric

const _MAX_MIE_ORDER = 100_000
const _MIE_STACKLESS_LOG_DERIVATIVE_ORDER = 64
const _MAX_MIE_CONTINUED_FRACTION_ITERATIONS = 100_000
const _MIE_CONTINUED_FRACTION_TOLERANCE = 8 * eps(Float64)
const _MIE_CONTINUED_FRACTION_TINY = 1.0e-300
const _MIE_FALLBACK_PRECISION = 512
const _MIE_EXTERIOR_SERIES_THRESHOLD = 1.0
const _MIE_EXACT_SERIES_PRECISION = 2304
const _MAX_MIE_EXACT_PRECISION = 16_384
const _MAX_MIE_EXACT_WORK = 2_000_000
# Retain at least half of the Float64 significand on the ordinary path.
# Colder, more strongly cancelled cases use the bounded BigFloat kernel.
const _MIE_CANCELLATION_THRESHOLD = sqrt(eps(Float64))
# Covers two complex triple products, their subtraction, and coefficient
# division when propagating an absolute forward-error bound.
const _MIE_COEFFICIENT_ERROR_FACTOR = 64eps(Float64)

@inline function _validate_mie_exact_work(
    nstop::Int,
    precision::Int=_MIE_EXACT_SERIES_PRECISION,
)
    nstop >= 1 || throw(ArgumentError("exact Mie order must be positive"))
    precision >= 2 ||
        throw(ArgumentError("exact Mie precision must be at least 2 bits"))
    precision <= _MAX_MIE_EXACT_PRECISION ||
        throw(ArgumentError(
            "exact Mie precision $precision exceeds the supported limit " *
            "$_MAX_MIE_EXACT_PRECISION bits"))
    # The cold kernels store and advance O(nstop) precision-bit values. Charge
    # both axes so a high explicit order cannot combine with range-recovery
    # precision to create unbounded work.
    work = BigInt(max(nstop, 1)) * BigInt(precision)
    work <= BigInt(_MAX_MIE_EXACT_WORK) ||
        throw(ArgumentError(
            "exact Mie order $nstop at $precision-bit precision exceeds " *
            "the exact-work limit $_MAX_MIE_EXACT_WORK"))
    return nothing
end

@inline function _mie_exact_series_precision(x::Real, nstop::Int)
    x > 0 || throw(ArgumentError("exact Mie size parameter must be positive"))
    _validate_mie_exact_work(nstop, _MIE_EXACT_SERIES_PRECISION)
    return _MIE_EXACT_SERIES_PRECISION
end

@inline function _mie_dielectric_exact_series_precision(
    x::Real,
    epsc::ComplexF64,
    muc::ComplexF64,
    nstop::Int,
)
    x > 0 || throw(ArgumentError("exact Mie size parameter must be positive"))
    size_exponent = exponent(x)
    # Magnetodielectric parameters provide two independent cancellation
    # degrees of freedom.  For n=1, the first surviving term can be O(x^3);
    # for n>=2 the first two even denominator terms can vanish but the x^4
    # term cannot.  Retain that worst exponent span plus a fixed guard for
    # Float64 material-data precision and the final rounded reduction.
    cancellation_order = nstop == 1 ? 3 : 4
    cancellation_bits = Base.checked_mul(
        cancellation_order, max(0, -size_exponent))
    required = Base.checked_add(256, cancellation_bits)
    precision = max(_MIE_EXACT_SERIES_PRECISION, required)
    # Large oscillatory internal arguments need enough bits for range
    # reduction of sin/cos/Bessel seeds.  Form the exponent in BigFloat so the
    # material product and m*x cannot overflow before the bound is known.
    internal_exponent = setprecision(BigFloat, _MIE_FALLBACK_PRECISION) do
        refractive_index = sqrt(
            Complex{BigFloat}(epsc) * Complex{BigFloat}(muc))
        internal_argument = refractive_index * BigFloat(x)
        iszero(internal_argument) ? typemin(Int) :
            exponent(abs(internal_argument))
    end
    internal_exponent > 0 &&
        (precision = max(
            precision, Base.checked_add(256, internal_exponent)))
    _validate_mie_exact_work(nstop, precision)
    return precision
end

@inline function _mie_exterior_initial_pair(x::Float64)
    if x > _MIE_EXTERIOR_SERIES_THRESHOLD
        sin_x = sin(x)
        cos_x = cos(x)
        return sin_x / x,
               -cos_x / x,
               sin_x / x^2 - cos_x / x,
               -cos_x / x^2 - sin_x / x
    end

    x2 = x * x
    j0 = 1.0
    j1 = x / 3
    term0 = 1.0
    term1 = j1
    for k in 1:12
        term0 *= -x2 / ((2k) * (2k + 1))
        j0 += term0
        term1 *= -x2 / ((2k) * (2k + 3))
        j1 += term1
        abs(term0) + abs(term1) <= eps(Float64) *
                                   (abs(j0) + abs(j1)) && break
    end
    inverse_x = inv(x)
    y0 = -cos(x) * inverse_x
    y1 = -cos(x) * inverse_x^2 - sin(x) * inverse_x
    return j0, y0, j1, y1
end

@inline function _mie_requires_exact_exterior(
    x::Float64,
    nstop::Int,
)
    _, exponent = frexp(x)
    inverse_exponent = max(0, -exponent)
    Base.checked_mul(nstop + 1, inverse_exponent) >= 1000 && return true
    nstop <= x && return false

    # Propagate a cancellation-free log upper bound for the Neumann/Riccati
    # recurrence. Triangle inequality makes this conservative even near the
    # turning region, where the double-factorial asymptotic is not a bound.
    chi_previous = -cos(x)
    chi_current = -cos(x) / x - sin(x)
    turning_order = min(nstop - 1, max(1, floor(Int, x)))
    @inbounds for order in 1:(turning_order - 1)
        chi_following = ((2order + 1) / x) * chi_current - chi_previous
        isfinite(chi_following) || return true
        chi_previous, chi_current = chi_current, chi_following
    end

    log_previous = log(max(abs(chi_previous), floatmin(Float64)))
    log_current = log(max(abs(chi_current), floatmin(Float64)))
    limit = log(floatmax(Float64)) - 8.0
    @inbounds for order in turning_order:(nstop - 1)
        log_factor = log(Float64(2order + 1)) - log(x)
        first = log_factor + log_current
        maximum_log = max(first, log_previous)
        log_following = maximum_log +
                        log(exp(first - maximum_log) +
                            exp(log_previous - maximum_log))
        log_following >= limit && return true
        log_previous, log_current = log_current, log_following
    end
    return false
end

@inline function _mie_exterior_is_overtruncated(x::Float64, nstop::Int)
    nstop > 3 || return false
    stable_limit = x + 4cbrt(x) + 2
    # The automatic order is ceil(stable_limit), so it remains on the normal
    # recurrence path.  Comparing nstop - 1 avoids converting an enormous
    # finite size parameter to Int for explicit, bounded orders.
    return Float64(nstop - 1) >= stable_limit
end

@inline function _mie_spherical_bessel_j_series(order::Int, x::Float64)
    order >= 0 || throw(ArgumentError("spherical-Bessel order must be nonnegative"))
    leading = one(Float64)
    @inbounds for index in 1:order
        leading *= x / Float64(2index + 1)
    end
    term = leading
    value = term
    x_squared = x * x
    @inbounds for series_order in 1:128
        term *= -x_squared /
                (Float64(2series_order) *
                 Float64(2order + 2series_order + 1))
        updated = value + term
        updated == value && return value
        value = updated
    end
    return value
end

@inline function _mie_stable_spherical_bessel_j(order::Int, x::Float64)
    # SpecialFunctions intentionally returns zero for positive spherical
    # orders at |x| <= sqrt(eps).  The convergent series retains representable
    # small values and is also stable throughout the exterior series range.
    return x <= _MIE_EXTERIOR_SERIES_THRESHOLD ?
        _mie_spherical_bessel_j_series(order, x) :
        sphericalbesselj(order, x)
end

@inline function _mie_riccati_psi_series(
    order::Int,
    argument::Complex{BigFloat},
    leading::Complex{BigFloat},
)
    term = one(argument)
    value_series = term
    derivative_series = BigFloat(order + 1) * term
    argument_squared = argument * argument
    for series_order in 1:128
        term *= -argument_squared /
                (BigFloat(2series_order) *
                 BigFloat(2order + 2series_order + 1))
        updated = value_series + term
        derivative_series +=
            BigFloat(order + 1 + 2series_order) * term
        if updated == value_series
            value_series = updated
            break
        end
        value_series = updated
    end
    return leading * value_series,
           (leading / argument) * derivative_series
end

@inline function _mie_riccati_pair_big(
        order::Int, argument::Complex{BigFloat},
        leading::Complex{BigFloat})
    return _mie_riccati_psi_series(order, argument, leading)
end

function _mie_riccati_psi_pairs_miller_big(
        argument::BigFloat, nstop::Int)
    start = Base.checked_add(
        nstop, max(32, ceil(Int, sqrt(80.0 * nstop))))
    values = Vector{BigFloat}(undef, Base.checked_add(start, 2))
    values[start + 2] = zero(BigFloat)
    values[start + 1] = one(BigFloat)
    @inbounds for order in (start - 1):-1:0
        values[order + 1] =
            (BigFloat(2order + 3) / argument) * values[order + 2] -
            values[order + 3]
    end
    psi_zero = sin(argument)
    psi_one = psi_zero / argument - cos(argument)
    reference_order = abs(psi_zero) >= abs(psi_one) ? 0 : 1
    reference = reference_order == 0 ? psi_zero : psi_one
    scale = reference / values[reference_order + 1]
    result = Vector{Tuple{BigFloat,BigFloat}}(undef, nstop)
    previous = psi_zero
    @inbounds for order in 1:nstop
        current = values[order + 1] * scale
        derivative = previous - (BigFloat(order) / argument) * current
        result[order] = (current, derivative)
        previous = current
    end
    return result
end

function _mie_riccati_psi_pairs_forward_big(
        argument::BigFloat, nstop::Int)
    result = Vector{Tuple{BigFloat,BigFloat}}(undef, nstop)
    previous = sin(argument)
    current = previous / argument - cos(argument)
    inverse = inv(argument)
    @inbounds for order in 1:nstop
        derivative = previous - BigFloat(order) * inverse * current
        result[order] = (current, derivative)
        if order < nstop
            following = BigFloat(2order + 1) * inverse * current - previous
            previous, current = current, following
        end
    end
    return result
end

@inline function _mie_riccati_psi_pairs_big(
        argument::BigFloat, nstop::Int)
    # Forward recurrence is stable on the oscillatory side of the turning
    # point. Miller recurrence is used only once the requested order extends
    # beyond the exterior size parameter.
    return BigFloat(nstop) <= argument ?
        _mie_riccati_psi_pairs_forward_big(argument, nstop) :
        _mie_riccati_psi_pairs_miller_big(argument, nstop)
end

function _mie_internal_function_pairs_big(
        nstop::Int, argument::Complex{BigFloat})
    pairs = Vector{Tuple{Complex{BigFloat},Complex{BigFloat}}}(
        undef, nstop)
    if abs(imag(argument)) > 10_000
        limiting_derivative = Complex{BigFloat}(
            zero(BigFloat), -sign(imag(argument)))
        scale = max(one(BigFloat), abs(limiting_derivative))
        @inbounds for order in 1:nstop
            pairs[order] = (
                Complex{BigFloat}(inv(scale), 0),
                limiting_derivative / scale,
            )
        end
        return pairs
    end
    previous = sin(argument)
    current = previous / argument - cos(argument)
    inverse = inv(argument)
    @inbounds for order in 1:nstop
        derivative = previous - BigFloat(order) * inverse * current
        scale = max(abs(current), abs(derivative))
        (isfinite(scale) && scale > 0) ||
            error("internal exact Mie function pair cannot be normalized")
        pairs[order] = (current / scale, derivative / scale)
        if order < nstop
            following = BigFloat(2order + 1) * inverse * current - previous
            pair_scale = max(abs(current), abs(following))
            previous, current = current / pair_scale, following / pair_scale
        end
    end
    return pairs
end

@noinline function _mie_s1s2_pec_exact_exterior(
    x::Float64,
    cosine::Float64,
    nstop::Int,
)
    precision = _mie_exact_series_precision(x, nstop)
    return setprecision(BigFloat, precision) do
        xb = BigFloat(x)
        argument = Complex{BigFloat}(xb, zero(BigFloat))
        leading = argument
        use_exterior_series = xb <= 1
        exterior_pairs = use_exterior_series ? nothing :
            _mie_riccati_psi_pairs_big(xb, nstop)
        chi_previous = -cos(xb)
        chi_current = -cos(xb) / xb - sin(xb)
        cosine_big = BigFloat(cosine)
        pi_previous2 = zero(BigFloat)
        pi_previous1 = one(BigFloat)
        first_sum = zero(Complex{BigFloat})
        second_sum = zero(Complex{BigFloat})

        for order in 1:nstop
            use_exterior_series &&
                (leading *= argument / BigFloat(2order + 1))
            psi, psi_derivative = if use_exterior_series
                _mie_riccati_pair_big(order, argument, leading)
            else
                pair = @inbounds exterior_pairs[order]
                Complex{BigFloat}(pair[1], zero(BigFloat)),
                Complex{BigFloat}(pair[2], zero(BigFloat))
            end
            chi_derivative = chi_previous -
                             (BigFloat(order) / xb) * chi_current
            xi = psi - Complex{BigFloat}(zero(BigFloat), chi_current)
            xi_derivative = psi_derivative -
                            Complex{BigFloat}(zero(BigFloat), chi_derivative)
            coefficient_a = -psi_derivative / xi_derivative
            coefficient_b = -psi / xi

            pi_order, pi_previous = if order == 1
                (one(BigFloat), zero(BigFloat))
            else
                ((BigFloat(2order - 1) / BigFloat(order - 1)) *
                     cosine_big * pi_previous1 -
                 (BigFloat(order) / BigFloat(order - 1)) * pi_previous2,
                 pi_previous1)
            end
            tau_order = BigFloat(order) * cosine_big * pi_order -
                        BigFloat(order + 1) * pi_previous
            angular_scale = BigFloat(2order + 1) /
                            BigFloat(order * (order + 1))
            first_sum += angular_scale *
                         (coefficient_a * pi_order +
                          coefficient_b * tau_order)
            second_sum += angular_scale *
                          (coefficient_a * tau_order +
                           coefficient_b * pi_order)

            if order >= 2
                pi_previous2, pi_previous1 = pi_previous1, pi_order
            end
            if order < nstop
                chi_next = (BigFloat(2order + 1) / xb) * chi_current -
                           chi_previous
                chi_previous, chi_current = chi_current, chi_next
            end
        end
        converted = ComplexF64(first_sum), ComplexF64(second_sum)
        return _assert_finite_mie_amplitudes(
            converted[1], converted[2], "PEC Mie series")
    end
end

@noinline function _mie_s1s2_dielectric_exact_exterior(
    x::Float64,
    cosine::Float64,
    epsc::ComplexF64,
    muc::ComplexF64,
    nstop::Int,
)
    precision = _mie_dielectric_exact_series_precision(
        x, epsc, muc, nstop)
    return setprecision(BigFloat, precision) do
        xb = BigFloat(x)
        exterior_argument = Complex{BigFloat}(xb, zero(BigFloat))
        epsilon = Complex{BigFloat}(epsc)
        permeability = Complex{BigFloat}(muc)
        refractive_index = sqrt(epsilon * permeability)
        internal_argument = refractive_index * xb
        use_internal_series = abs(internal_argument) <= 1
        # This path is also the correctness retry for cancelled constitutive
        # contrasts, so do not seed it with already rounded internal pairs.
        internal_function_pairs = use_internal_series ? nothing :
            _mie_internal_function_pairs_big(nstop, internal_argument)
        exterior_leading = exterior_argument
        use_exterior_series = xb <= 1
        exterior_pairs = use_exterior_series ? nothing :
            _mie_riccati_psi_pairs_big(xb, nstop)
        internal_leading = internal_argument
        chi_previous = -cos(xb)
        chi_current = -cos(xb) / xb - sin(xb)
        cosine_big = BigFloat(cosine)
        pi_previous2 = zero(BigFloat)
        pi_previous1 = one(BigFloat)
        first_sum = zero(Complex{BigFloat})
        second_sum = zero(Complex{BigFloat})

        for order in 1:nstop
            denominator = BigFloat(2order + 1)
            use_exterior_series &&
                (exterior_leading *= exterior_argument / denominator)
            use_internal_series &&
                (internal_leading *= internal_argument / denominator)
            psi, psi_derivative = if use_exterior_series
                _mie_riccati_pair_big(
                    order, exterior_argument, exterior_leading)
            else
                pair = @inbounds exterior_pairs[order]
                Complex{BigFloat}(pair[1], zero(BigFloat)),
                Complex{BigFloat}(pair[2], zero(BigFloat))
            end
            interior_function, interior_derivative = if use_internal_series
                _mie_riccati_psi_series(
                    order, internal_argument, internal_leading)
            else
                function_pair, derivative_pair =
                    @inbounds internal_function_pairs[order]
                Complex{BigFloat}(function_pair),
                Complex{BigFloat}(derivative_pair)
            end
            chi_derivative = chi_previous -
                             (BigFloat(order) / xb) * chi_current
            xi = psi - Complex{BigFloat}(zero(BigFloat), chi_current)
            xi_derivative = psi_derivative -
                            Complex{BigFloat}(zero(BigFloat), chi_derivative)

            numerator_a = refractive_index * interior_function * psi_derivative -
                          permeability * interior_derivative * psi
            denominator_a = refractive_index * interior_function * xi_derivative -
                            permeability * interior_derivative * xi
            numerator_b = permeability * interior_function * psi_derivative -
                          refractive_index * interior_derivative * psi
            denominator_b = permeability * interior_function * xi_derivative -
                            refractive_index * interior_derivative * xi
            coefficient_a = -numerator_a / denominator_a
            coefficient_b = -numerator_b / denominator_b

            pi_order, pi_previous = if order == 1
                (one(BigFloat), zero(BigFloat))
            else
                ((BigFloat(2order - 1) / BigFloat(order - 1)) *
                     cosine_big * pi_previous1 -
                 (BigFloat(order) / BigFloat(order - 1)) * pi_previous2,
                 pi_previous1)
            end
            tau_order = BigFloat(order) * cosine_big * pi_order -
                        BigFloat(order + 1) * pi_previous
            angular_scale = BigFloat(2order + 1) /
                            BigFloat(order * (order + 1))
            first_sum += angular_scale *
                         (coefficient_a * pi_order +
                          coefficient_b * tau_order)
            second_sum += angular_scale *
                          (coefficient_a * tau_order +
                           coefficient_b * pi_order)

            if order >= 2
                pi_previous2, pi_previous1 = pi_previous1, pi_order
            end
            if order < nstop
                chi_next = (BigFloat(2order + 1) / xb) * chi_current -
                           chi_previous
                chi_previous, chi_current = chi_current, chi_next
            end
        end
        converted = ComplexF64(first_sum), ComplexF64(second_sum)
        return _assert_finite_mie_amplitudes(
            converted[1], converted[2], "dielectric Mie series")
    end
end

function _mie_internal_function_pairs(
        nstop::Int, z::ComplexF64)
    pairs = Vector{Tuple{ComplexF64,ComplexF64}}(undef, nstop)
    if _mie_use_scaled_forward_recurrence(z, nstop)
        previous, current = _mie_scaled_riccati_initial_pair(z)
        inverse_z = inv(z)
        @inbounds for order in 1:nstop
            derivative = previous - (order * inverse_z) * current
            pairs[order] = _mie_normalized_pair(
                current, derivative,
                "internal Mie scaled forward recurrence")
            if order < nstop
                following = ((2order + 1) * inverse_z) * current - previous
                previous, current = _mie_normalized_pair(
                    current, following,
                    "internal Mie scaled forward recurrence")
            end
        end
        return pairs
    end

    derivatives = nstop > _MIE_STACKLESS_LOG_DERIVATIVE_ORDER ?
        _mie_riccati_log_derivatives(nstop, z) : nothing
    @inbounds for order in 1:nstop
        derivative = derivatives === nothing ?
            _mie_riccati_log_derivative(order, z) : derivatives[order]
        pairs[order] = _mie_log_derivative_pair(derivative)
    end
    return pairs
end

@inline function _validated_mie_positive(value::Float64,
                                         label::AbstractString)
    (isfinite(value) && value > 0.0) ||
        throw(ArgumentError("$label must be finite and positive, got $value"))
    return value
end

@inline function _validated_mie_cosine(value::Float64,
                                       label::AbstractString)
    (isfinite(value) && abs(value) <= 1.0 + 1e-12) ||
        throw(ArgumentError("$label must be finite and satisfy |$label| <= 1, got $value"))
    return clamp(value, -1.0, 1.0)
end

@inline function _validated_mie_material(value, label::AbstractString)
    converted = try
        ComplexF64(value)
    catch err
        err isa Union{MethodError,InexactError,TypeError,ArgumentError,OverflowError} ||
            rethrow()
        throw(ArgumentError("$label must be convertible to ComplexF64, got $value"))
    end
    (isfinite(real(converted)) && isfinite(imag(converted))) ||
        throw(ArgumentError("$label must be finite, got $value"))
    abs(converted) > 0.0 ||
        throw(ArgumentError("$label must be nonzero, got $value"))
    return converted
end

function _validated_mie_order(nmax, size_parameter::Float64)
    _validated_mie_positive(size_parameter, "Mie size parameter")
    if nmax === nothing
        estimate = size_parameter + 4 * cbrt(size_parameter) + 2
        (isfinite(estimate) && estimate <= typemax(Int)) ||
            throw(ArgumentError(
                "automatic Mie truncation order is not representable for size parameter $size_parameter"))
        order = try
            ceil(Int, estimate)
        catch err
            err isa Union{InexactError,OverflowError} || rethrow()
            throw(ArgumentError(
                "automatic Mie truncation order is not representable for size parameter $size_parameter"))
        end
        order = max(3, order)
        order <= _MAX_MIE_ORDER ||
            throw(ArgumentError(
                "automatic Mie truncation order $order exceeds the supported " *
                "limit $_MAX_MIE_ORDER"))
        return order
    end
    order = try
        Int(nmax)
    catch err
        err isa Union{MethodError,InexactError,TypeError,ArgumentError,OverflowError} ||
            rethrow()
        throw(ArgumentError("nmax must be an integer-valued number, got $nmax"))
    end
    order >= 1 ||
        throw(ArgumentError("nmax must be at least 1, got $order"))
    order <= _MAX_MIE_ORDER ||
        throw(ArgumentError(
            "nmax=$order exceeds the supported Mie order limit " *
            "$_MAX_MIE_ORDER"))
    return order
end

function _validated_mie_exceptional_product_order(nmax, x::BigFloat)
    x > 0 || throw(ArgumentError("Mie size parameter must be positive"))
    if nmax === nothing
        x < BigFloat(floatmin(Float64)) && return 3
        throw(ArgumentError(
            "automatic Mie truncation is unavailable when k*a exceeds " *
            "the Float64 range; provide an explicit bounded nmax"))
    end
    return _validated_mie_order(nmax, 1.0)
end

@noinline function _mie_exact_size_parameter(k::Float64, a::Float64)
    return setprecision(BigFloat, _MIE_EXACT_SERIES_PRECISION) do
        BigFloat(k) * BigFloat(a)
    end
end

@noinline function _mie_material_refractive_index_exact(
    epsc::ComplexF64,
    muc::ComplexF64,
)
    return setprecision(BigFloat, _MIE_FALLBACK_PRECISION) do
        product = Complex{BigFloat}(epsc) * Complex{BigFloat}(muc)
        converted = ComplexF64(sqrt(product))
        (isfinite(converted) && !iszero(converted)) ||
            throw(ArgumentError(
                "eps_r * mu_r has no representable nonzero square root"))
        return converted
    end
end

@inline function _mie_material_refractive_index(
    epsc::ComplexF64,
    muc::ComplexF64,
)
    product = epsc * muc
    if isfinite(product) && !iszero(product)
        refractive_index = sqrt(product)
        isfinite(refractive_index) && !iszero(refractive_index) &&
            return refractive_index
    end
    return _mie_material_refractive_index_exact(epsc, muc)
end

@noinline function _mie_internal_size_parameter_exact(
    refractive_index::ComplexF64,
    x::Float64,
)
    return setprecision(BigFloat, _MIE_FALLBACK_PRECISION) do
        exact = Complex{BigFloat}(refractive_index) * BigFloat(x)
        converted = ComplexF64(exact)
        (isfinite(converted) && !iszero(converted)) ||
            throw(ArgumentError(
                "internal Mie size parameter is outside the representable " *
                "nonzero ComplexF64 range"))
        return converted
    end
end

@inline function _mie_internal_size_parameter(
    refractive_index::ComplexF64,
    x::Float64,
)
    z = refractive_index * x
    isfinite(z) && !iszero(z) && return z
    return _mie_internal_size_parameter_exact(refractive_index, x)
end

@inline function _mie_lentz_nonzero(value::ComplexF64)
    magnitude = abs(value)
    magnitude >= _MIE_CONTINUED_FRACTION_TINY && return value
    iszero(magnitude) &&
        return ComplexF64(_MIE_CONTINUED_FRACTION_TINY, 0.0)
    return value * (_MIE_CONTINUED_FRACTION_TINY / magnitude)
end

@inline function _mie_riccati_log_derivative(
    n::Int,
    z::ComplexF64,
)
    inverse_z = inv(z)
    isfinite(inverse_z) ||
        throw(ArgumentError(
            "internal Mie size parameter reciprocal is not representable"))

    # D_n(z) = psi'_n(z)/psi_n(z) = J_(n-1/2)(z)/J_(n+1/2)(z) - n/z.
    # Evaluate the Bessel ratio as a continued fraction with the modified
    # Lentz algorithm. This remains bounded when sin(z) and cos(z) themselves
    # overflow and avoids the unstable forward recurrence for n >> abs(z).
    b0 = (2n + 1) * inverse_z
    fraction = _mie_lentz_nonzero(b0)
    numerator_factor = fraction
    denominator_factor = 0.0 + 0.0im

    for iteration in 1:_MAX_MIE_CONTINUED_FRACTION_ITERATIONS
        b = (2n + 1 + 2iteration) * inverse_z
        denominator_factor = _mie_lentz_nonzero(b - denominator_factor)
        denominator_factor = inv(denominator_factor)
        numerator_factor = _mie_lentz_nonzero(
            b - inv(numerator_factor))
        update = numerator_factor * denominator_factor
        fraction *= update

        (isfinite(update) && isfinite(fraction)) ||
            error(
                "internal Mie logarithmic-derivative continued fraction " *
                "produced a non-finite value at order $n")
        if abs(update - 1) <= _MIE_CONTINUED_FRACTION_TOLERANCE
            result = fraction - n * inverse_z
            isfinite(result) ||
                error(
                    "internal Mie logarithmic derivative is non-finite at " *
                    "order $n")
            return result
        end
    end

    throw(ArgumentError(
        "internal Mie logarithmic derivative at order $n requires more than " *
        "$_MAX_MIE_CONTINUED_FRACTION_ITERATIONS continued-fraction iterations"))
end

function _mie_riccati_log_derivatives(
    nstop::Int,
    z::ComplexF64,
)
    values = Vector{ComplexF64}(undef, nstop)
    values[nstop] = _mie_riccati_log_derivative(nstop, z)
    inverse_z = inv(z)
    @inbounds for n in (nstop - 1):-1:1
        order_over_z = (n + 1) * inverse_z
        values[n] = order_over_z - inv(values[n + 1] + order_over_z)
        isfinite(values[n]) ||
            error(
                "internal Mie logarithmic derivative is non-finite at order $n")
    end
    return values
end

@inline function _mie_pair_component_scale(
    first::ComplexF64,
    second::ComplexF64,
)
    return max(
        abs(real(first)), abs(imag(first)),
        abs(real(second)), abs(imag(second)),
    )
end

@inline function _mie_normalized_pair(
    first::ComplexF64,
    second::ComplexF64,
    label::AbstractString,
)
    scale = _mie_pair_component_scale(first, second)
    (isfinite(scale) && scale > 0.0) ||
        error("$label cannot be normalized from $first and $second")
    return first / scale, second / scale
end

@inline function _mie_log_derivative_pair(
    derivative::ComplexF64,
)
    scale = max(
        1.0, abs(real(derivative)), abs(imag(derivative)))
    return ComplexF64(inv(scale), 0.0), derivative / scale
end

@inline function _mie_use_scaled_forward_recurrence(
    z::ComplexF64,
    nstop::Int,
)
    # The normalized forward transfer is a small perturbation of a unitary
    # swap when |z| dominates the accumulated recurrence coefficients. At this
    # conservative boundary, sum((2n+1)/|z|, n=0:N) is below about 0.26.
    threshold = 4.0 * nstop * nstop + 32.0
    return abs(z) >= threshold
end

@inline function _mie_scaled_riccati_initial_pair(z::ComplexF64)
    real_z = real(z)
    imaginary_z = imag(z)
    sine_real, cosine_real = sincos(real_z)
    decay_argument = -2.0 * abs(imaginary_z)
    decayed = exp(decay_argument)
    one_minus_decayed = -expm1(decay_argument)
    even_scale = 0.5 * (1.0 + decayed)
    odd_scale = 0.5 * sign(imaginary_z) * one_minus_decayed

    scaled_sine = ComplexF64(
        sine_real * even_scale,
        cosine_real * odd_scale,
    )
    scaled_cosine = ComplexF64(
        cosine_real * even_scale,
        -sine_real * odd_scale,
    )
    inverse_z = inv(z)
    psi_zero = scaled_sine
    psi_one = scaled_sine * inverse_z - scaled_cosine
    return _mie_normalized_pair(
        psi_zero, psi_one, "internal Mie scaled forward recurrence")
end

@inline function _mie_product_requires_exact(
    first::ComplexF64,
    second::ComplexF64,
    third::ComplexF64,
    intermediate::ComplexF64,
    product::ComplexF64,
)
    isfinite(intermediate) && isfinite(product) || return true
    (iszero(first) || iszero(second) || iszero(third)) && return false
    intermediate_scale = abs(intermediate)
    product_scale = abs(product)
    return !isfinite(intermediate_scale) || !isfinite(product_scale) ||
           iszero(intermediate_scale) || iszero(product_scale)
end

@inline function _mie_coefficient_error_bound(
    numerator::ComplexF64,
    denominator::ComplexF64,
    numerator_magnitude::Float64,
    denominator_magnitude::Float64,
    coefficient::ComplexF64,
)
    isfinite(numerator) && isfinite(denominator) &&
        isfinite(numerator_magnitude) &&
        isfinite(denominator_magnitude) &&
        isfinite(coefficient) || return Inf
    numerator_error =
        _MIE_COEFFICIENT_ERROR_FACTOR * numerator_magnitude
    denominator_error =
        _MIE_COEFFICIENT_ERROR_FACTOR * denominator_magnitude
    denominator_margin = abs(denominator) - denominator_error
    denominator_margin > 0.0 || return Inf
    bound = (numerator_error + abs(coefficient) * denominator_error) /
            denominator_margin + 8eps(Float64) * abs(coefficient)
    return isfinite(bound) ? bound : Inf
end

@inline function _mie_series_requires_exact(
    value::ComplexF64,
    magnitude::Float64,
)
    isfinite(value) && isfinite(magnitude) || return true
    iszero(magnitude) && return false
    return abs(value) <= _MIE_CANCELLATION_THRESHOLD * magnitude
end

@inline function _mie_coefficient_error_requires_exact(
    value::ComplexF64,
    error_bound::Float64,
)
    isfinite(value) && isfinite(error_bound) || return true
    iszero(error_bound) && return false
    return iszero(value) ||
           error_bound >= _MIE_CANCELLATION_THRESHOLD * abs(value)
end

@inline function _mie_dielectric_coefficients(
    scaled_m::ComplexF64,
    scaled_mu::ComplexF64,
    interior_function::ComplexF64,
    interior_derivative::ComplexF64,
    psi_n::Float64,
    psi_p_n::Float64,
    xi_n::ComplexF64,
    xi_p_n::ComplexF64,
)
    psi = ComplexF64(psi_n)
    psi_derivative = ComplexF64(psi_p_n)

    m_function = scaled_m * interior_function
    mu_function = scaled_mu * interior_function
    m_derivative = scaled_m * interior_derivative
    mu_derivative = scaled_mu * interior_derivative

    num_a_first = m_function * psi_derivative
    num_a_second = mu_derivative * psi
    den_a_first = m_function * xi_p_n
    den_a_second = mu_derivative * xi_n
    num_b_first = mu_function * psi_derivative
    num_b_second = m_derivative * psi
    den_b_first = mu_function * xi_p_n
    den_b_second = m_derivative * xi_n

    products_require_exact =
        _mie_product_requires_exact(
            scaled_m, interior_function, psi_derivative,
            m_function, num_a_first) ||
        _mie_product_requires_exact(
            scaled_mu, interior_derivative, psi,
            mu_derivative, num_a_second) ||
        _mie_product_requires_exact(
            scaled_m, interior_function, xi_p_n,
            m_function, den_a_first) ||
        _mie_product_requires_exact(
            scaled_mu, interior_derivative, xi_n,
            mu_derivative, den_a_second) ||
        _mie_product_requires_exact(
            scaled_mu, interior_function, psi_derivative,
            mu_function, num_b_first) ||
        _mie_product_requires_exact(
            scaled_m, interior_derivative, psi,
            m_derivative, num_b_second) ||
        _mie_product_requires_exact(
            scaled_mu, interior_function, xi_p_n,
            mu_function, den_b_first) ||
        _mie_product_requires_exact(
            scaled_m, interior_derivative, xi_n,
            m_derivative, den_b_second)
    products_require_exact &&
        return 0.0 + 0.0im, 0.0 + 0.0im, Inf, Inf, true

    num_a = num_a_first - num_a_second
    den_a = den_a_first - den_a_second
    num_b = num_b_first - num_b_second
    den_b = den_b_first - den_b_second
    all(isfinite, (num_a, den_a, num_b, den_b)) ||
        return 0.0 + 0.0im, 0.0 + 0.0im, Inf, Inf, true

    coefficient_a = -num_a / den_a
    coefficient_b = -num_b / den_b
    isfinite(coefficient_a) && isfinite(coefficient_b) ||
        return 0.0 + 0.0im, 0.0 + 0.0im, Inf, Inf, true

    num_a_magnitude = abs(num_a_first) + abs(num_a_second)
    den_a_magnitude = abs(den_a_first) + abs(den_a_second)
    num_b_magnitude = abs(num_b_first) + abs(num_b_second)
    den_b_magnitude = abs(den_b_first) + abs(den_b_second)
    coefficient_a_error = _mie_coefficient_error_bound(
        num_a, den_a, num_a_magnitude, den_a_magnitude, coefficient_a)
    coefficient_b_error = _mie_coefficient_error_bound(
        num_b, den_b, num_b_magnitude, den_b_magnitude, coefficient_b)
    coefficients_require_exact =
        !isfinite(coefficient_a_error) || !isfinite(coefficient_b_error)
    return coefficient_a, coefficient_b,
           coefficient_a_error, coefficient_b_error,
           coefficients_require_exact
end

@inline function _assert_finite_mie_amplitudes(S1::ComplexF64,
                                               S2::ComplexF64,
                                               label::AbstractString)
    (isfinite(S1) && isfinite(S2)) ||
        error("$label produced non-finite scattering amplitudes; check the size parameter, material parameters, and truncation order")
    return S1, S2
end

@inline function _validated_mie_direction(value::Vec3,
                                          label::AbstractString)
    all(isfinite, value) ||
        throw(ArgumentError("$label components must be finite, got $value"))
    scale = max(abs(value[1]), abs(value[2]), abs(value[3]))
    scale > 0.0 ||
        throw(ArgumentError("$label must be nonzero"))
    scaled = value / scale
    scaled_norm = norm(scaled)
    (isfinite(scaled_norm) && scaled_norm > 0.0) ||
        throw(ArgumentError("$label must have a finite, nonzero norm"))
    return scaled / scaled_norm
end

@inline function _mie_rcs_from_amplitude(fvec::CVec3,
                                         label::AbstractString)
    amplitude_norm = norm(fvec)
    isfinite(amplitude_norm) ||
        error("$label produced a non-finite scattering amplitude")
    sigma = 4π * amplitude_norm * amplitude_norm
    isfinite(sigma) ||
        throw(OverflowError("$label is outside the representable Float64 range"))
    return sigma
end

@noinline function _mie_rcs_from_scaled_components(
        first::ComplexF64, first_weight::Float64,
        second::ComplexF64, second_weight::Float64,
        k::Float64, label::AbstractString)
    return setprecision(BigFloat, _MIE_FALLBACK_PRECISION) do
        first_big = Complex{BigFloat}(first) * BigFloat(first_weight)
        second_big = Complex{BigFloat}(second) * BigFloat(second_weight)
        norm_squared = abs2(first_big) + abs2(second_big)
        sigma = Float64(4BigFloat(π) * norm_squared / BigFloat(k)^2)
        isfinite(sigma) ||
            throw(OverflowError(
                "$label is outside the representable Float64 range"))
        return sigma
    end
end

@noinline function _mie_bistatic_rcs_pec_big(
        x, cosine::Float64, nstop::Int,
        first_weight::Float64, second_weight::Float64,
        k::Float64)
    precision = _mie_exact_series_precision(x, nstop)
    return setprecision(BigFloat, precision) do
        xb = BigFloat(x)
        argument = Complex{BigFloat}(xb, zero(BigFloat))
        leading = argument
        use_exterior_series = xb <= 1
        exterior_pairs = use_exterior_series ? nothing :
            _mie_riccati_psi_pairs_big(xb, nstop)
        chi_previous = -cos(xb)
        chi_current = -cos(xb) / xb - sin(xb)
        pi_previous2 = BigFloat(0)
        pi_previous1 = BigFloat(1)
        first_sum = zero(Complex{BigFloat})
        second_sum = zero(Complex{BigFloat})
        for order in 1:nstop
            use_exterior_series &&
                (leading *= argument / BigFloat(2order + 1))
            psi, psi_derivative = if use_exterior_series
                _mie_riccati_pair_big(order, argument, leading)
            else
                pair = @inbounds exterior_pairs[order]
                Complex{BigFloat}(pair[1], zero(BigFloat)),
                Complex{BigFloat}(pair[2], zero(BigFloat))
            end
            chi_derivative = chi_previous -
                             (BigFloat(order) / xb) * chi_current
            xi = psi - Complex{BigFloat}(0, chi_current)
            xi_derivative = psi_derivative -
                            Complex{BigFloat}(0, chi_derivative)
            coefficient_a = -psi_derivative / xi_derivative
            coefficient_b = -psi / xi
            pi_order, pi_previous = if order == 1
                (BigFloat(1), BigFloat(0))
            else
                (((BigFloat(2order - 1) / BigFloat(order - 1)) *
                  BigFloat(cosine) * pi_previous1 -
                  (BigFloat(order) / BigFloat(order - 1)) * pi_previous2),
                 pi_previous1)
            end
            tau_order = BigFloat(order) * BigFloat(cosine) * pi_order -
                        BigFloat(order + 1) * pi_previous
            angular_scale = BigFloat(2order + 1) /
                            BigFloat(order * (order + 1))
            first_sum += angular_scale *
                         (coefficient_a * pi_order + coefficient_b * tau_order)
            second_sum += angular_scale *
                          (coefficient_a * tau_order + coefficient_b * pi_order)
            if order >= 2
                pi_previous2, pi_previous1 = pi_previous1, pi_order
            end
            if order < nstop
                chi_next = (BigFloat(2order + 1) / xb) * chi_current -
                           chi_previous
                chi_previous, chi_current = chi_current, chi_next
            end
        end
        amplitude_squared = abs2(first_sum * BigFloat(first_weight)) +
                            abs2(second_sum * BigFloat(second_weight))
        sigma = Float64(
            4BigFloat(π) * amplitude_squared / BigFloat(k)^2)
        isfinite(sigma) ||
            throw(OverflowError(
                "PEC bistatic RCS is outside the representable Float64 range"))
        return sigma
    end
end

@noinline function _mie_bistatic_rcs_dielectric_big(
        x, cosine::Float64,
        epsc::ComplexF64, muc::ComplexF64, nstop::Int,
        first_weight::Float64, second_weight::Float64,
        k::Float64)
    precision = _mie_dielectric_exact_series_precision(
        x, epsc, muc, nstop)
    return setprecision(BigFloat, precision) do
        xb = BigFloat(x)
        exterior_argument = Complex{BigFloat}(xb, 0)
        epsilon = Complex{BigFloat}(epsc)
        permeability = Complex{BigFloat}(muc)
        refractive_index = sqrt(epsilon * permeability)
        internal_argument = refractive_index * xb
        use_internal_series = abs(internal_argument) <= 1
        internal_function_pairs = if use_internal_series
            nothing
        else
            converted_internal = ComplexF64(internal_argument)
            if isfinite(converted_internal) && !iszero(converted_internal)
                _mie_internal_function_pairs(nstop, converted_internal)
            else
                _mie_internal_function_pairs_big(
                    nstop, internal_argument)
            end
        end
        exterior_leading = exterior_argument
        use_exterior_series = xb <= 1
        exterior_pairs = use_exterior_series ? nothing :
            _mie_riccati_psi_pairs_big(xb, nstop)
        internal_leading = internal_argument
        chi_previous = -cos(xb)
        chi_current = -cos(xb) / xb - sin(xb)
        pi_previous2 = BigFloat(0)
        pi_previous1 = BigFloat(1)
        first_sum = zero(Complex{BigFloat})
        second_sum = zero(Complex{BigFloat})
        for order in 1:nstop
            denominator = BigFloat(2order + 1)
            use_exterior_series &&
                (exterior_leading *= exterior_argument / denominator)
            use_internal_series &&
                (internal_leading *= internal_argument / denominator)
            psi, psi_derivative = if use_exterior_series
                _mie_riccati_pair_big(
                    order, exterior_argument, exterior_leading)
            else
                pair = @inbounds exterior_pairs[order]
                Complex{BigFloat}(pair[1], zero(BigFloat)),
                Complex{BigFloat}(pair[2], zero(BigFloat))
            end
            interior_function, interior_derivative = if use_internal_series
                _mie_riccati_psi_series(
                    order, internal_argument, internal_leading)
            else
                function_pair, derivative_pair =
                    @inbounds internal_function_pairs[order]
                Complex{BigFloat}(function_pair),
                Complex{BigFloat}(derivative_pair)
            end
            chi_derivative = chi_previous -
                             (BigFloat(order) / xb) * chi_current
            xi = psi - Complex{BigFloat}(0, chi_current)
            xi_derivative = psi_derivative -
                            Complex{BigFloat}(0, chi_derivative)
            numerator_a = refractive_index * interior_function * psi_derivative -
                          permeability * interior_derivative * psi
            denominator_a = refractive_index * interior_function * xi_derivative -
                            permeability * interior_derivative * xi
            numerator_b = permeability * interior_function * psi_derivative -
                          refractive_index * interior_derivative * psi
            denominator_b = permeability * interior_function * xi_derivative -
                            refractive_index * interior_derivative * xi
            coefficient_a = -numerator_a / denominator_a
            coefficient_b = -numerator_b / denominator_b
            pi_order, pi_previous = if order == 1
                (BigFloat(1), BigFloat(0))
            else
                (((BigFloat(2order - 1) / BigFloat(order - 1)) *
                  BigFloat(cosine) * pi_previous1 -
                  (BigFloat(order) / BigFloat(order - 1)) * pi_previous2),
                 pi_previous1)
            end
            tau_order = BigFloat(order) * BigFloat(cosine) * pi_order -
                        BigFloat(order + 1) * pi_previous
            angular_scale = BigFloat(2order + 1) /
                            BigFloat(order * (order + 1))
            first_sum += angular_scale *
                         (coefficient_a * pi_order + coefficient_b * tau_order)
            second_sum += angular_scale *
                          (coefficient_a * tau_order + coefficient_b * pi_order)
            if order >= 2
                pi_previous2, pi_previous1 = pi_previous1, pi_order
            end
            if order < nstop
                chi_next = (BigFloat(2order + 1) / xb) * chi_current -
                           chi_previous
                chi_previous, chi_current = chi_current, chi_next
            end
        end
        amplitude_squared = abs2(first_sum * BigFloat(first_weight)) +
                            abs2(second_sum * BigFloat(second_weight))
        sigma = Float64(
            4BigFloat(π) * amplitude_squared / BigFloat(k)^2)
        isfinite(sigma) ||
            throw(OverflowError(
                "dielectric bistatic RCS is outside the representable " *
                "Float64 range"))
        return sigma
    end
end

@inline function _mie_rcs_from_components(
        first::ComplexF64, first_weight::Float64,
        second::ComplexF64, second_weight::Float64,
        k::Float64, label::AbstractString)
    first_component = first * first_weight
    second_component = second * second_weight
    weighted_component_is_unresolved(amplitude, weight) = begin
        iszero(weight) && return false
        @inbounds for primitive in (real(amplitude), imag(amplitude))
            iszero(primitive) && continue
            product = primitive * weight
            (!isfinite(product) || abs(product) < floatmin(Float64)) &&
                return true
        end
        return false
    end
    if weighted_component_is_unresolved(first, first_weight) ||
       weighted_component_is_unresolved(second, second_weight)
        return _mie_rcs_from_scaled_components(
            first, first_weight, second, second_weight, k, label)
    end
    first_scaled = first_component / k
    second_scaled = second_component / k
    if isfinite(first_scaled) && isfinite(second_scaled) &&
       !(iszero(first_scaled) && !iszero(first_component)) &&
       !(iszero(second_scaled) && !iszero(second_component))
        scale = max(abs(first_scaled), abs(second_scaled))
        if isfinite(scale) && scale > 0.0
            normalized = abs2(first_scaled / scale) +
                         abs2(second_scaled / scale)
            sigma = (4π * normalized) * scale * scale
            isfinite(sigma) && !(
                iszero(sigma) && normalized > 0.0 && scale > 0.0) &&
                return sigma
        elseif iszero(scale)
            return 0.0
        end
    end
    return _mie_rcs_from_scaled_components(
        first, first_weight, second, second_weight, k, label)
end

"""
    mie_s1s2_pec(x, μ; nmax=nothing)

Compute Mie scattering amplitudes `(S1, S2)` for a PEC sphere at size
parameter `x = k a`, evaluated at `μ = cos(γ)` where `γ` is the scattering
angle (angle between incident propagation direction and observation direction).
"""
function mie_s1s2_pec(x::Float64, μ::Float64; nmax=nothing)
    x = _validated_mie_positive(x, "x")
    μ = _validated_mie_cosine(μ, "μ")
    nstop = _validated_mie_order(nmax, x)
    over_truncated = _mie_exterior_is_overtruncated(x, nstop)
    if _mie_requires_exact_exterior(x, nstop)
        _validate_mie_exact_work(nstop)
        return _mie_s1s2_pec_exact_exterior(x, μ, nstop)
    end

    j_nm1, y_nm1, j_n, y_n = _mie_exterior_initial_pair(x)
    if over_truncated
        j_nm1 = _mie_stable_spherical_bessel_j(0, x)
        j_n = _mie_stable_spherical_bessel_j(1, x)
    end

    # Angular functions:
    #   π_n = P_n^1(μ)/sin(θ),  τ_n = dP_n^1(μ)/dθ
    # with π_0 = 0, π_1 = 1 and
    #   π_n = ((2n-1)/(n-1)) μ π_{n-1} - (n/(n-1)) π_{n-2},  n≥2
    #   τ_n = n μ π_n - (n+1) π_{n-1}
    π_prev2 = 0.0   # π_0
    π_prev1 = 1.0   # π_1

    S1 = 0.0 + 0.0im
    S2 = 0.0 + 0.0im
    S1_magnitude = 0.0
    S2_magnitude = 0.0

    for n in 1:nstop
        psi_nm1 = x * j_nm1
        xi_nm1 = x * (j_nm1 - 1im * y_nm1)
        psi_n = x * j_n
        xi_n = x * (j_n - 1im * y_n)
        psi_p_n = psi_nm1 - (n / x) * psi_n
        xi_p_n = xi_nm1 - (n / x) * xi_n
        a_n = -psi_p_n / xi_p_n
        b_n = -psi_n / xi_n

        if n == 1
            π_n = 1.0
            π_nm1 = 0.0   # π_0
        else
            π_n = ((2n - 1) / (n - 1)) * μ * π_prev1 - (n / (n - 1)) * π_prev2
            π_nm1 = π_prev1
        end

        τ_n = n * μ * π_n - (n + 1) * π_nm1

        c = (2n + 1) / (n * (n + 1))
        S1_term = c * (a_n * π_n + b_n * τ_n)
        S2_term = c * (a_n * τ_n + b_n * π_n)
        S1 += S1_term
        S2 += S2_term
        S1_magnitude += abs(S1_term)
        S2_magnitude += abs(S2_term)

        if n >= 2
            π_prev2, π_prev1 = π_prev1, π_n
        end
        if n < nstop
            recurrence = (2n + 1) / x
            j_nm1, j_n = j_n, over_truncated ?
                _mie_stable_spherical_bessel_j(n + 1, x) :
                recurrence * j_n - j_nm1
            y_nm1, y_n = y_n, recurrence * y_n - y_nm1
        end
    end

    if _mie_series_requires_exact(S1, S1_magnitude) ||
       _mie_series_requires_exact(S2, S2_magnitude)
        _validate_mie_exact_work(nstop)
        return _mie_s1s2_pec_exact_exterior(x, μ, nstop)
    end

    return _assert_finite_mie_amplitudes(S1, S2, "PEC Mie series")
end

"""
    mie_s1s2_dielectric(x, cosγ, eps_r; mu_r=1, nmax=nothing)

Compute Mie scattering amplitudes `(S1, S2)` for a homogeneous isotropic
dielectric/magnetodielectric sphere in vacuum. `x = k0*a` is the exterior size
parameter and `cosγ` is the cosine of the scattering angle.

The coefficients use outgoing spherical Hankel functions of the second kind,
matching the package-wide `exp(+iωt)` convention.
"""
function mie_s1s2_dielectric(x::Float64, cosγ::Float64, eps_r;
                             mu_r=1.0 + 0im, nmax=nothing)
    x = _validated_mie_positive(x, "x")
    cosγ = _validated_mie_cosine(cosγ, "cosγ")
    epsc = _validated_mie_material(eps_r, "eps_r")
    muc = _validated_mie_material(mu_r, "mu_r")
    nstop = _validated_mie_order(nmax, x)
    epsc == 1.0 + 0.0im && muc == 1.0 + 0.0im &&
        return 0.0 + 0.0im, 0.0 + 0.0im
    over_truncated = _mie_exterior_is_overtruncated(x, nstop)
    if _mie_requires_exact_exterior(x, nstop)
        _validate_mie_exact_work(nstop)
        return _mie_s1s2_dielectric_exact_exterior(
            x, cosγ, epsc, muc, nstop)
    end
    m = _mie_material_refractive_index(epsc, muc)
    z_product = m * x
    if !isfinite(z_product) || iszero(z_product)
        _validate_mie_exact_work(nstop)
        return _mie_s1s2_dielectric_exact_exterior(
            x, cosγ, epsc, muc, nstop)
    end
    z = _mie_internal_size_parameter(m, x)
    inverse_z = inv(z)
    maximum_seed = Float64(2nstop + 3) * inverse_z
    if !isfinite(inverse_z) || !isfinite(maximum_seed)
        _validate_mie_exact_work(nstop)
        return _mie_s1s2_dielectric_exact_exterior(
            x, cosγ, epsc, muc, nstop)
    end

    j_nm1, y_nm1, j_n, y_n = _mie_exterior_initial_pair(x)
    if over_truncated
        j_nm1 = _mie_stable_spherical_bessel_j(0, x)
        j_n = _mie_stable_spherical_bessel_j(1, x)
    end

    use_scaled_forward = _mie_use_scaled_forward_recurrence(z, nstop)
    log_derivatives = !use_scaled_forward &&
                      nstop > _MIE_STACKLESS_LOG_DERIVATIVE_ORDER ?
        _mie_riccati_log_derivatives(nstop, z) : nothing
    internal_previous, internal_current = use_scaled_forward ?
        _mie_scaled_riccati_initial_pair(z) :
        (0.0 + 0.0im, 0.0 + 0.0im)
    inverse_internal_z = inv(z)
    scaled_m, scaled_mu = _mie_normalized_pair(
        m, muc, "dielectric Mie material factors")

    π_prev2 = 0.0
    π_prev1 = 1.0
    S1 = 0.0 + 0.0im
    S2 = 0.0 + 0.0im
    S1_magnitude = 0.0
    S2_magnitude = 0.0
    S1_coefficient_error = 0.0
    S2_coefficient_error = 0.0

    for n in 1:nstop
        psi_nm1 = x * j_nm1
        xi_nm1 = x * (j_nm1 - 1im * y_nm1)
        psi_n = x * j_n
        xi_n = x * (j_n - 1im * y_n)
        psi_p_n = psi_nm1 - (n / x) * psi_n
        xi_p_n = xi_nm1 - (n / x) * xi_n

        interior_function, interior_derivative = if use_scaled_forward
            derivative = internal_previous -
                         (n * inverse_internal_z) * internal_current
            _mie_normalized_pair(
                internal_current,
                derivative,
                "internal Mie scaled forward recurrence",
            )
        else
            log_derivative = log_derivatives === nothing ?
                _mie_riccati_log_derivative(n, z) :
                @inbounds(log_derivatives[n])
            _mie_log_derivative_pair(log_derivative)
        end

        # The leading minus keeps the phase convention aligned with the PEC
        # h^(2) implementation above. RCS is invariant to the resulting global
        # scattered-field phase, but amplitude users expect one convention.
        a_n, b_n, a_n_error, b_n_error, coefficients_require_exact =
            _mie_dielectric_coefficients(
                scaled_m,
                scaled_mu,
                interior_function,
                interior_derivative,
                psi_n,
                psi_p_n,
                xi_n,
                xi_p_n,
            )
        if coefficients_require_exact
            _validate_mie_exact_work(nstop)
            return _mie_s1s2_dielectric_exact_exterior(
                x, cosγ, epsc, muc, nstop)
        end

        if n == 1
            π_n = 1.0
            π_nm1 = 0.0
        else
            π_n = ((2n - 1) / (n - 1)) * cosγ * π_prev1 -
                  (n / (n - 1)) * π_prev2
            π_nm1 = π_prev1
        end

        τ_n = n * cosγ * π_n - (n + 1) * π_nm1
        c = (2n + 1) / (n * (n + 1))
        S1_term = c * (a_n * π_n + b_n * τ_n)
        S2_term = c * (a_n * τ_n + b_n * π_n)
        S1 += S1_term
        S2 += S2_term
        S1_magnitude += abs(S1_term)
        S2_magnitude += abs(S2_term)
        S1_coefficient_error += abs(c) *
            (abs(π_n) * a_n_error + abs(τ_n) * b_n_error)
        S2_coefficient_error += abs(c) *
            (abs(τ_n) * a_n_error + abs(π_n) * b_n_error)

        if n >= 2
            π_prev2, π_prev1 = π_prev1, π_n
        end
        if n < nstop
            exterior_recurrence = (2n + 1) / x
            j_nm1, j_n = j_n, over_truncated ?
                _mie_stable_spherical_bessel_j(n + 1, x) :
                exterior_recurrence * j_n - j_nm1
            y_nm1, y_n = y_n, exterior_recurrence * y_n - y_nm1
            if use_scaled_forward
                internal_next =
                    ((2n + 1) * inverse_internal_z) * internal_current -
                    internal_previous
                internal_previous, internal_current = _mie_normalized_pair(
                    internal_current,
                    internal_next,
                    "internal Mie scaled forward recurrence",
                )
            end
        end
    end

    if _mie_series_requires_exact(S1, S1_magnitude) ||
       _mie_series_requires_exact(S2, S2_magnitude) ||
       _mie_coefficient_error_requires_exact(
           S1, S1_coefficient_error) ||
       _mie_coefficient_error_requires_exact(
           S2, S2_coefficient_error)
        _validate_mie_exact_work(nstop)
        return _mie_s1s2_dielectric_exact_exterior(
            x, cosγ, epsc, muc, nstop)
    end

    return _assert_finite_mie_amplitudes(
        S1, S2, "dielectric Mie series")
end

function _orthonormal_to(v::Vec3)
    tmp = abs(v[1]) < 0.9 ? Vec3(1.0, 0.0, 0.0) : Vec3(0.0, 1.0, 0.0)
    u = cross(v, tmp)
    return u / norm(u)
end

"""
    mie_bistatic_rcs_pec(k, a, k_inc_hat, pol_inc, rhat; nmax=nothing)

Compute PEC-sphere bistatic RCS (linear units, m²) for fixed incident
propagation direction `k_inc_hat` (unit vector), incident polarization
`pol_inc` (unit vector orthogonal to `k_inc_hat`), and observation direction
`rhat` (unit vector).
"""
function mie_bistatic_rcs_pec(k::Float64, a::Float64,
                              k_inc_hat::Vec3, pol_inc::Vec3, rhat::Vec3;
                              nmax=nothing)
    k = _validated_mie_positive(k, "k")
    a = _validated_mie_positive(a, "a")

    khat = _validated_mie_direction(k_inc_hat, "k_inc_hat")
    phat = _validated_mie_direction(pol_inc, "pol_inc")
    rhat_u = _validated_mie_direction(rhat, "rhat")

    abs(dot(khat, phat)) < 1e-10 ||
        throw(ArgumentError("pol_inc must be orthogonal to k_inc_hat"))

    μ = clamp(dot(khat, rhat_u), -1.0, 1.0)

    e_perp = cross(khat, rhat_u)
    if norm(e_perp) < 1e-12
        e_perp = _orthonormal_to(khat)
    else
        e_perp /= norm(e_perp)
    end
    e_par_i = cross(e_perp, khat)
    e_par_i /= norm(e_par_i)
    e_par_s = cross(e_perp, rhat_u)
    e_par_s /= norm(e_par_s)

    coeff_perp = dot(phat, e_perp)
    coeff_para = dot(phat, e_par_i)

    x = k * a
    if !isfinite(x) || iszero(x)
        x_exact = _mie_exact_size_parameter(k, a)
        nstop = _validated_mie_exceptional_product_order(nmax, x_exact)
        _validate_mie_exact_work(nstop)
        return _mie_bistatic_rcs_pec_big(
            x_exact, μ, nstop, coeff_perp, coeff_para, k)
    end
    S1, S2 = mie_s1s2_pec(x, μ; nmax=nmax)

    weighted_first = S1 * coeff_perp
    weighted_second = S2 * coeff_para
    ordinary_sigma = _mie_rcs_from_components(
        S1, coeff_perp, S2, coeff_para, k, "PEC bistatic RCS")
    if x <= 1.0 &&
       (iszero(ordinary_sigma) ||
        (iszero(weighted_first) && !iszero(coeff_perp)) ||
        (iszero(weighted_second) && !iszero(coeff_para)))
        nstop = _validated_mie_order(nmax, x)
        _validate_mie_exact_work(nstop)
        return _mie_bistatic_rcs_pec_big(
            x, μ, nstop, coeff_perp, coeff_para, k)
    end

    return ordinary_sigma
end

"""
    mie_bistatic_rcs_dielectric(k, a, k_inc_hat, pol_inc, rhat, eps_r; mu_r=1, nmax=nothing)

Compute exact homogeneous-sphere bistatic RCS (m²) from dielectric Mie theory.
The exterior medium is vacuum and `pol_inc` must be transverse to
`k_inc_hat`.
"""
function mie_bistatic_rcs_dielectric(k::Float64, a::Float64,
                                     k_inc_hat::Vec3, pol_inc::Vec3,
                                     rhat::Vec3, eps_r;
                                     mu_r=1.0 + 0im, nmax=nothing)
    k = _validated_mie_positive(k, "k")
    a = _validated_mie_positive(a, "a")

    khat = _validated_mie_direction(k_inc_hat, "k_inc_hat")
    phat = _validated_mie_direction(pol_inc, "pol_inc")
    rhat_u = _validated_mie_direction(rhat, "rhat")

    abs(dot(khat, phat)) < 1e-10 ||
        throw(ArgumentError("pol_inc must be orthogonal to k_inc_hat"))

    cosγ = clamp(dot(khat, rhat_u), -1.0, 1.0)
    epsc = _validated_mie_material(eps_r, "eps_r")
    muc = _validated_mie_material(mu_r, "mu_r")
    # Keep the analytic matched-medium shortcut consistent with the ordinary
    # amplitude path: an explicitly supplied truncation order is still part of
    # the public input contract and must be validated even though no series is
    # needed for the zero-contrast result.
    nmax === nothing || _validated_mie_order(nmax, 1.0)
    epsc == 1.0 + 0.0im && muc == 1.0 + 0.0im && return 0.0
    e_perp = cross(khat, rhat_u)
    if norm(e_perp) < 1e-12
        e_perp = _orthonormal_to(khat)
    else
        e_perp /= norm(e_perp)
    end
    e_par_i = cross(e_perp, khat)
    e_par_i /= norm(e_par_i)
    e_par_s = cross(e_perp, rhat_u)
    e_par_s /= norm(e_par_s)

    coeff_perp = dot(phat, e_perp)
    coeff_para = dot(phat, e_par_i)

    x = k * a
    if !isfinite(x) || iszero(x)
        x_exact = _mie_exact_size_parameter(k, a)
        nstop = _validated_mie_exceptional_product_order(nmax, x_exact)
        _validate_mie_exact_work(nstop)
        return _mie_bistatic_rcs_dielectric_big(
            x_exact, cosγ, epsc, muc, nstop,
            coeff_perp, coeff_para, k)
    end
    S1, S2 = mie_s1s2_dielectric(x, cosγ, epsc;
                                 mu_r=muc, nmax=nmax)

    weighted_first = S1 * coeff_perp
    weighted_second = S2 * coeff_para
    ordinary_sigma = _mie_rcs_from_components(
        S1, coeff_perp, S2, coeff_para, k,
        "dielectric bistatic RCS")
    if x <= 1.0 &&
       (iszero(ordinary_sigma) ||
        (iszero(weighted_first) && !iszero(coeff_perp)) ||
        (iszero(weighted_second) && !iszero(coeff_para)))
        nstop = _validated_mie_order(nmax, x)
        _validate_mie_exact_work(nstop)
        return _mie_bistatic_rcs_dielectric_big(
            x, cosγ, epsc, muc, nstop,
            coeff_perp, coeff_para, k)
    end

    return ordinary_sigma
end
