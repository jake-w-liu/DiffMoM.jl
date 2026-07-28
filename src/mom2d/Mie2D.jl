# Mie2D.jl — 2D Mie series for circular cylinder (TM polarization)
#
# Provides exact scattered field for validation of the 2D VIE solver.
# Convention: exp(+iωt), H₀⁽²⁾ for outgoing waves.

export mie_coefficients_2d, mie_scattered_field_2d, mie_total_field_2d

@inline function _validated_mie2d_positive(value::Float64,
                                           label::AbstractString)
    (isfinite(value) && value > 0.0) ||
        throw(ArgumentError("$label must be finite and positive, got $value"))
    return value
end

function _validated_mie2d_order(k0a::Float64,
                                nmax::Union{Nothing,Int})
    (isfinite(k0a) && k0a > 0.0) ||
        throw(ArgumentError(
            "2D Mie size parameter k0*a must be finite and positive, got $k0a"))
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
    N, coefficient_count = _validated_mie2d_order(k0a, nmax)
    if !pec
        isfinite(eps_r) ||
            throw(ArgumentError("eps_r must be finite, got $eps_r"))
    end

    c = Vector{ComplexF64}(undef, coefficient_count)
    center = N + 1

    if pec
        for n in 0:N
            coefficient = -besselj(n, k0a) / besselh(n, 2, k0a)
            isfinite(coefficient) ||
                error("PEC 2D Mie coefficient at order $n is non-finite")
            c[center + n] = coefficient
            c[center - n] = coefficient
        end
    elseif eps_r == 0.0
        # Analytic k1→0 limit. For order q=|n|,
        # k1*Jq'(k1*a)/Jq(k1*a) → q/a.
        for n in 0:N
            Jn_k0a = besselj(n, k0a)
            Jnm1_k0a = besselj(n - 1, k0a)
            dJn_k0a = Jnm1_k0a - (n / k0a) * Jn_k0a
            Hn_k0a = besselh(n, 2, k0a)
            Hnm1_k0a = besselh(n - 1, 2, k0a)
            dHn_k0a = Hnm1_k0a - (n / k0a) * Hn_k0a
            radial_ratio = n / a
            coefficient =
                -(radial_ratio * Jn_k0a - k0 * dJn_k0a) /
                 (radial_ratio * Hn_k0a - k0 * dHn_k0a)
            isfinite(coefficient) ||
                error("ENZ 2D Mie coefficient at order $n is non-finite")
            c[center + n] = coefficient
            c[center - n] = coefficient
        end
    else
        k1 = k0 * sqrt(complex(eps_r))
        k1a = k1 * a
        isfinite(k1a) ||
            throw(ArgumentError(
                "internal 2D Mie size parameter must be finite, got $k1a"))

        for n in 0:N
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
            coefficient = num / den
            isfinite(coefficient) ||
                error("dielectric 2D Mie coefficient at order $n is non-finite")
            c[center + n] = coefficient
            c[center - n] = coefficient
        end
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
        rho + tolerance >= a ||
            throw(DomainError(
                rho,
                "r_obs[$m] lies inside the cylinder: radius $rho < a=$a"))
        argument = k0 * rho
        (isfinite(argument) && argument > 0.0) ||
            throw(ArgumentError(
                "k0*norm(r_obs[$m]) must be finite and positive, got $argument"))
    end
    return nothing
end

"""
    mie_scattered_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0, nmax=nothing, pec=false)

Compute exact scattered field at observation points for a circular cylinder.
Observation points must lie on or outside the cylinder (`ρ ≥ a`); the
exterior-series surface limit is supported.

E_z^scat(ρ,φ) = E₀ Σ_n (-i)ⁿ cₙ Hₙ⁽²⁾(k₀ρ) eⁱⁿᶠ

Arguments:
- `r_obs`: vector of observation positions (Vec2)
- `phi_inc`: incident angle (0 = +x direction)
"""
function mie_scattered_field_2d(k0::Float64, a::Float64, eps_r::Float64,
                                 r_obs::AbstractVector{Vec2};
                                 phi_inc::Float64=0.0, nmax=nothing, pec::Bool=false)
    k0 = _validated_mie2d_positive(k0, "k0")
    a = _validated_mie2d_positive(a, "a")
    _validate_mie2d_observations(k0, a, r_obs, phi_inc)
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
        isfinite(E_scat[m]) ||
            error("2D Mie scattered field at observation $m is non-finite")
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
    E_total = mie_scattered_field_2d(k0, a, eps_r, r_obs;
                                     phi_inc=phi_inc, nmax=nmax, pec=pec)

    khat = Vec2(cos(phi_inc), sin(phi_inc))
    for m in eachindex(r_obs)
        E_total[m] += exp(-im * k0 * dot(khat, r_obs[m]))
        isfinite(E_total[m]) ||
            error("2D Mie total field at observation $m is non-finite")
    end
    return E_total
end
