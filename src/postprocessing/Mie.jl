# Mie.jl — sphere Mie-theory reference utilities

export mie_s1s2_pec, mie_bistatic_rcs_pec
export mie_s1s2_dielectric, mie_bistatic_rcs_dielectric

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
        return max(3, order)
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
    return order
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
    material_product = epsc * muc
    (isfinite(material_product) && abs(material_product) > 0.0) ||
        throw(ArgumentError(
            "eps_r * mu_r must be finite and nonzero, got $material_product"))
    m = sqrt(material_product)
    z = m * x
    (isfinite(z) && abs(z) > 0.0) ||
        throw(ArgumentError(
            "internal Mie size parameter must be finite and nonzero, got $z"))
    nstop = _validated_mie_order(nmax, abs(z))

    sin_x = sin(x)
    cos_x = cos(x)
    j_nm1 = sin_x / x
    y_nm1 = -cos_x / x
    j_n = sin_x / x^2 - cos_x / x
    y_n = -cos_x / x^2 - sin_x / x

    sin_z = sin(z)
    cos_z = cos(z)
    j_m_nm1 = sin_z / z
    j_m_n = sin_z / z^2 - cos_z / z

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

        psi_m_nm1 = z * j_m_nm1
        psi_m_n = z * j_m_n
        psi_m_p_n = psi_m_nm1 - (n / z) * psi_m_n

        num_a = m * psi_m_n * psi_p_n - muc * psi_n * psi_m_p_n
        den_a = m * psi_m_n * xi_p_n - muc * xi_n * psi_m_p_n
        num_b = muc * psi_m_n * psi_p_n - m * psi_n * psi_m_p_n
        den_b = muc * psi_m_n * xi_p_n - m * xi_n * psi_m_p_n

        # The leading minus keeps the phase convention aligned with the PEC
        # h^(2) implementation above. RCS is invariant to the resulting global
        # scattered-field phase, but amplitude users expect one convention.
        a_n = -num_a / den_a
        b_n = -num_b / den_b

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
            interior_recurrence = (2n + 1) / z
            j_m_nm1, j_m_n =
                j_m_n, interior_recurrence * j_m_n - j_m_nm1
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
