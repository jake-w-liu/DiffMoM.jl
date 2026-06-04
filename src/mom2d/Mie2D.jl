# Mie2D.jl — 2D Mie series for circular cylinder (TM polarization)
#
# Provides exact scattered field for validation of the 2D VIE solver.
# Convention: exp(+iωt), H₀⁽²⁾ for outgoing waves.

export mie_coefficients_2d, mie_scattered_field_2d, mie_total_field_2d

"""
    mie_coefficients_2d(k0, a, eps_r; nmax=nothing, pec=false)

Compute 2D Mie scattering coefficients cₙ for a circular cylinder.

Arguments:
- `k0`: free-space wavenumber
- `a`: cylinder radius
- `eps_r`: relative permittivity (ignored if `pec=true`)
- `nmax`: maximum order (auto-determined if `nothing`)
- `pec`: if true, compute PEC cylinder coefficients

Returns vector `c` indexed from -nmax:nmax (stored as c[n + nmax + 1]).
"""
function mie_coefficients_2d(k0::Float64, a::Float64, eps_r::Float64;
                              nmax::Union{Nothing,Int}=nothing, pec::Bool=false)
    @assert k0 > 0 "Wavenumber must be positive"
    @assert a > 0 "Radius must be positive"

    k0a = k0 * a
    N = nmax === nothing ? max(10, ceil(Int, k0a + 4 * k0a^(1/3) + 2)) : nmax

    c = Vector{ComplexF64}(undef, 2N + 1)  # c[n + N + 1] for n = -N:N

    if pec
        for n in -N:N
            Jn = besselj(n, k0a)
            Hn = besselh(n, 2, k0a)
            c[n + N + 1] = -Jn / Hn
        end
    else
        k1 = k0 * sqrt(complex(eps_r))
        k1a = k1 * a

        for n in -N:N
            # Bessel function derivatives using recurrence:
            # f'_n(x) = f_{n-1}(x) - (n/x) f_n(x)
            Jn_k0a = besselj(n, k0a)
            Jnm1_k0a = besselj(n - 1, k0a)
            dJn_k0a = Jnm1_k0a - (n / k0a) * Jn_k0a

            Hn_k0a = besselh(n, 2, k0a)
            Hnm1_k0a = besselh(n - 1, 2, k0a)
            dHn_k0a = Hnm1_k0a - (n / k0a) * Hn_k0a

            Jn_k1a = besselj(n, k1a)
            Jnm1_k1a = besselj(n - 1, k1a)
            dJn_k1a = Jnm1_k1a - (n / k1a) * Jn_k1a

            num = -(k1 * dJn_k1a * Jn_k0a - k0 * Jn_k1a * dJn_k0a)
            den =  (k1 * dJn_k1a * Hn_k0a - k0 * Jn_k1a * dHn_k0a)
            c[n + N + 1] = num / den
        end
    end

    return c, N
end

"""
    mie_scattered_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0, nmax=nothing, pec=false)

Compute exact scattered field at observation points for a circular cylinder.

E_z^scat(ρ,φ) = E₀ Σ_n (-i)ⁿ cₙ Hₙ⁽²⁾(k₀ρ) eⁱⁿᶠ

Arguments:
- `r_obs`: vector of observation positions (Vec2)
- `phi_inc`: incident angle (0 = +x direction)
"""
function mie_scattered_field_2d(k0::Float64, a::Float64, eps_r::Float64,
                                 r_obs::AbstractVector{Vec2};
                                 phi_inc::Float64=0.0, nmax=nothing, pec::Bool=false)
    c, N = mie_coefficients_2d(k0, a, eps_r; nmax=nmax, pec=pec)

    M = length(r_obs)
    E_scat = zeros(ComplexF64, M)

    for m in 1:M
        rho = sqrt(dot(r_obs[m], r_obs[m]))
        phi = atan(r_obs[m][2], r_obs[m][1])

        for n in -N:N
            E_scat[m] += (-im + 0.0)^n * c[n + N + 1] * besselh(n, 2, k0 * rho) *
                         exp(im * n * (phi - phi_inc))
        end
    end

    return E_scat
end

"""
    mie_total_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0, nmax=nothing, pec=false)

Compute exact total field (incident + scattered) at observation points outside
the cylinder (ρ > a).
"""
function mie_total_field_2d(k0::Float64, a::Float64, eps_r::Float64,
                             r_obs::AbstractVector{Vec2};
                             phi_inc::Float64=0.0, nmax=nothing, pec::Bool=false)
    E_scat = mie_scattered_field_2d(k0, a, eps_r, r_obs;
                                      phi_inc=phi_inc, nmax=nmax, pec=pec)

    khat = Vec2(cos(phi_inc), sin(phi_inc))
    E_total = Vector{ComplexF64}(undef, length(r_obs))
    for m in eachindex(r_obs)
        E_total[m] = exp(-im * k0 * dot(khat, r_obs[m])) + E_scat[m]
    end
    return E_total
end
