# Mie.jl — sphere Mie-theory reference utilities

export mie_s1s2_pec, mie_bistatic_rcs_pec
export mie_s1s2_dielectric, mie_bistatic_rcs_dielectric

function _sph_bessel_jy_arrays(x::Float64, nmax::Int)
    x > 0 || error("x must be positive")
    nmax >= 1 || error("nmax must be >= 1")

    j = zeros(Float64, nmax + 1)  # j[n+1] = j_n
    y = zeros(Float64, nmax + 1)  # y[n+1] = y_n

    j[1] = sin(x) / x
    y[1] = -cos(x) / x
    j[2] = sin(x) / x^2 - cos(x) / x
    y[2] = -cos(x) / x^2 - sin(x) / x

    for n in 1:(nmax - 1)
        j[n + 2] = ((2n + 1) / x) * j[n + 1] - j[n]
        y[n + 2] = ((2n + 1) / x) * y[n + 1] - y[n]
    end

    return j, y
end

function _sph_bessel_j_arrays(z::ComplexF64, nmax::Int)
    abs(z) > 0 || error("z must be nonzero")
    nmax >= 1 || error("nmax must be >= 1")

    j = zeros(ComplexF64, nmax + 1)  # j[n+1] = j_n
    j[1] = sin(z) / z
    j[2] = sin(z) / z^2 - cos(z) / z

    for n in 1:(nmax - 1)
        j[n + 2] = ((2n + 1) / z) * j[n + 1] - j[n]
    end

    return j
end

function _riccati_psi_arrays(z::ComplexF64, nmax::Int)
    j = _sph_bessel_j_arrays(z, nmax)
    psi = zeros(ComplexF64, nmax + 1)
    psi_p = zeros(ComplexF64, nmax + 1)
    for n in 0:nmax
        psi[n + 1] = z * j[n + 1]
    end
    for n in 1:nmax
        psi_p[n + 1] = psi[n] - (n / z) * psi[n + 1]
    end
    return psi, psi_p
end

function _riccati_exterior_arrays(x::Float64, nmax::Int)
    j, y = _sph_bessel_jy_arrays(x, nmax)
    psi = zeros(ComplexF64, nmax + 1)
    xi = zeros(ComplexF64, nmax + 1)
    psi_p = zeros(ComplexF64, nmax + 1)
    xi_p = zeros(ComplexF64, nmax + 1)
    for n in 0:nmax
        psi[n + 1] = x * j[n + 1]
        xi[n + 1] = x * (j[n + 1] - 1im * y[n + 1])  # hankel-2
    end
    for n in 1:nmax
        psi_p[n + 1] = psi[n] - (n / x) * psi[n + 1]
        xi_p[n + 1] = xi[n] - (n / x) * xi[n + 1]
    end
    return psi, xi, psi_p, xi_p
end

function _mie_nmax(x::Float64)
    return max(3, ceil(Int, x + 4 * x^(1 / 3) + 2))
end

"""
    mie_s1s2_pec(x, μ; nmax=nothing)

Compute Mie scattering amplitudes `(S1, S2)` for a PEC sphere at size
parameter `x = k a`, evaluated at `μ = cos(γ)` where `γ` is the scattering
angle (angle between incident propagation direction and observation direction).
"""
function mie_s1s2_pec(x::Float64, μ::Float64; nmax=nothing)
    abs(μ) <= 1.0 + 1e-12 || error("μ must satisfy |μ|<=1")
    nstop = nmax === nothing ? _mie_nmax(x) : Int(nmax)
    nstop >= 1 || error("nmax must be >= 1")

    j, y = _sph_bessel_jy_arrays(x, nstop)

    psi = zeros(Float64, nstop + 1)          # ψ_n
    xi = zeros(ComplexF64, nstop + 1)        # ξ_n
    for n in 0:nstop
        psi[n + 1] = x * j[n + 1]
        xi[n + 1] = x * (j[n + 1] - 1im * y[n + 1])   # hankel-2
    end

    psi_p = zeros(Float64, nstop + 1)        # ψ'_n
    xi_p = zeros(ComplexF64, nstop + 1)      # ξ'_n
    for n in 1:nstop
        psi_p[n + 1] = psi[n] - (n / x) * psi[n + 1]
        xi_p[n + 1] = xi[n] - (n / x) * xi[n + 1]
    end

    a = zeros(ComplexF64, nstop)
    b = zeros(ComplexF64, nstop)
    for n in 1:nstop
        a[n] = -psi_p[n + 1] / xi_p[n + 1]
        b[n] = -psi[n + 1] / xi[n + 1]
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

    for n in 1:nstop
        if n == 1
            π_n = 1.0
            π_nm1 = 0.0   # π_0
        else
            π_n = ((2n - 1) / (n - 1)) * μ * π_prev1 - (n / (n - 1)) * π_prev2
            π_nm1 = π_prev1
        end

        τ_n = n * μ * π_n - (n + 1) * π_nm1

        c = (2n + 1) / (n * (n + 1))
        S1 += c * (a[n] * π_n + b[n] * τ_n)
        S2 += c * (a[n] * τ_n + b[n] * π_n)

        if n >= 2
            π_prev2, π_prev1 = π_prev1, π_n
        end
    end

    return S1, S2
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
    x > 0 || error("x must be positive")
    abs(cosγ) <= 1.0 + 1e-12 || error("cosγ must satisfy |cosγ|<=1")
    epsc = ComplexF64(eps_r)
    muc = ComplexF64(mu_r)
    abs(epsc) > 0 || error("eps_r must be nonzero")
    abs(muc) > 0 || error("mu_r must be nonzero")

    nstop = nmax === nothing ? _mie_nmax(abs(sqrt(epsc * muc)) * x) : Int(nmax)
    nstop >= 1 || error("nmax must be >= 1")

    m = sqrt(epsc * muc)
    psi, xi, psi_p, xi_p = _riccati_exterior_arrays(x, nstop)
    psi_m, psi_m_p = _riccati_psi_arrays(m * x, nstop)

    a = zeros(ComplexF64, nstop)
    b = zeros(ComplexF64, nstop)
    for n in 1:nstop
        num_a = m * psi_m[n + 1] * psi_p[n + 1] -
                muc * psi[n + 1] * psi_m_p[n + 1]
        den_a = m * psi_m[n + 1] * xi_p[n + 1] -
                muc * xi[n + 1] * psi_m_p[n + 1]
        num_b = muc * psi_m[n + 1] * psi_p[n + 1] -
                m * psi[n + 1] * psi_m_p[n + 1]
        den_b = muc * psi_m[n + 1] * xi_p[n + 1] -
                m * xi[n + 1] * psi_m_p[n + 1]

        # The leading minus keeps the phase convention aligned with the PEC
        # h^(2) implementation above. RCS is invariant to the resulting global
        # scattered-field phase, but amplitude users expect one convention.
        a[n] = -num_a / den_a
        b[n] = -num_b / den_b
    end

    π_prev2 = 0.0
    π_prev1 = 1.0
    S1 = 0.0 + 0.0im
    S2 = 0.0 + 0.0im

    for n in 1:nstop
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
        S1 += c * (a[n] * π_n + b[n] * τ_n)
        S2 += c * (a[n] * τ_n + b[n] * π_n)

        if n >= 2
            π_prev2, π_prev1 = π_prev1, π_n
        end
    end

    return S1, S2
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
    k > 0 || error("k must be positive")
    a > 0 || error("a must be positive")

    khat = k_inc_hat / norm(k_inc_hat)
    phat = pol_inc / norm(pol_inc)
    rhat_u = rhat / norm(rhat)

    abs(dot(khat, phat)) < 1e-10 || error("pol_inc must be orthogonal to k_inc_hat")

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

    return 4π * real(dot(fvec, fvec))
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
    k > 0 || error("k must be positive")
    a > 0 || error("a must be positive")

    khat = k_inc_hat / norm(k_inc_hat)
    phat = pol_inc / norm(pol_inc)
    rhat_u = rhat / norm(rhat)

    abs(dot(khat, phat)) < 1e-10 || error("pol_inc must be orthogonal to k_inc_hat")

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

    return 4π * real(dot(fvec, fvec))
end
