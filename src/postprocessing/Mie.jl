# Mie.jl — sphere Mie-theory reference utilities

export mie_s1s2_pec, mie_bistatic_rcs_pec
export mie_s1s2_dielectric, mie_bistatic_rcs_dielectric

const _MAX_MIE_ORDER = 100_000
const _MIE_STACKLESS_LOG_DERIVATIVE_ORDER = 64
const _MAX_MIE_CONTINUED_FRACTION_ITERATIONS = 100_000
const _MIE_CONTINUED_FRACTION_TOLERANCE = 8 * eps(Float64)
const _MIE_CONTINUED_FRACTION_TINY = 1.0e-300
const _MIE_FALLBACK_PRECISION = 512

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
    num_a = scaled_m * interior_function * psi_p_n -
            scaled_mu * interior_derivative * psi_n
    den_a = scaled_m * interior_function * xi_p_n -
            scaled_mu * interior_derivative * xi_n
    num_b = scaled_mu * interior_function * psi_p_n -
            scaled_m * interior_derivative * psi_n
    den_b = scaled_mu * interior_function * xi_p_n -
            scaled_m * interior_derivative * xi_n
    return -num_a / den_a, -num_b / den_b
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

    sin_x = sin(x)
    cos_x = cos(x)
    j_nm1 = sin_x / x
    y_nm1 = -cos_x / x
    j_n = sin_x / x^2 - cos_x / x
    y_n = -cos_x / x^2 - sin_x / x

    # Angular functions:
    #   π_n = P_n^1(μ)/sin(θ),  τ_n = dP_n^1(μ)/dθ
    # with π_0 = 0, π_1 = 1 and
    #   π_n = ((2n-1)/(n-1)) μ π_{n-1} - (n/(n-1)) π_{n-2},  n≥2
    #   τ_n = n μ π_n - (n+1) π_{n-1}
    π_prev2 = 0.0   # π_0
    π_prev1 = 1.0   # π_1

    S1 = 0.0 + 0.0im
    S2 = 0.0 + 0.0im

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
        S1 += c * (a_n * π_n + b_n * τ_n)
        S2 += c * (a_n * τ_n + b_n * π_n)

        if n >= 2
            π_prev2, π_prev1 = π_prev1, π_n
        end
        if n < nstop
            recurrence = (2n + 1) / x
            j_nm1, j_n = j_n, recurrence * j_n - j_nm1
            y_nm1, y_n = y_n, recurrence * y_n - y_nm1
        end
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
    m = _mie_material_refractive_index(epsc, muc)
    z = _mie_internal_size_parameter(m, x)

    sin_x = sin(x)
    cos_x = cos(x)
    j_nm1 = sin_x / x
    y_nm1 = -cos_x / x
    j_n = sin_x / x^2 - cos_x / x
    y_n = -cos_x / x^2 - sin_x / x

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
        a_n, b_n = _mie_dielectric_coefficients(
            scaled_m,
            scaled_mu,
            interior_function,
            interior_derivative,
            psi_n,
            psi_p_n,
            xi_n,
            xi_p_n,
        )

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
        S1 += c * (a_n * π_n + b_n * τ_n)
        S2 += c * (a_n * τ_n + b_n * π_n)

        if n >= 2
            π_prev2, π_prev1 = π_prev1, π_n
        end
        if n < nstop
            exterior_recurrence = (2n + 1) / x
            j_nm1, j_n = j_n, exterior_recurrence * j_n - j_nm1
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
    S1, S2 = mie_s1s2_pec(k * a, μ; nmax=nmax)

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

    fvec = ((S1 * coeff_perp) .* e_perp .+ (S2 * coeff_para) .* e_par_s) / (1im * k)

    return _mie_rcs_from_amplitude(fvec, "PEC bistatic RCS")
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
    S1, S2 = mie_s1s2_dielectric(k * a, cosγ, eps_r;
                                 mu_r=mu_r, nmax=nmax)

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

    fvec = ((S1 * coeff_perp) .* e_perp .+ (S2 * coeff_para) .* e_par_s) / (1im * k)

    return _mie_rcs_from_amplitude(fvec, "dielectric bistatic RCS")
end
