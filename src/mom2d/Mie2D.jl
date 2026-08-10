# Mie2D.jl — 2D Mie series for circular cylinder (TM polarization)
#
# Provides exact scattered field for validation of the 2D VIE solver.
# Convention: exp(+iωt), H₀⁽²⁾ for outgoing waves.

export mie_coefficients_2d, mie_scattered_field_2d, mie_total_field_2d

const _MIE2D_FALLBACK_PRECISION = 256
const _MIE2D_SMALL_ARGUMENT_CUTOFF = 1e-25
const _MIE2D_INTERNAL_SERIES_LIMIT = 0.5

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

function _mie_coefficients_small_bigfloat_2d(
    k0::Float64,
    a::Float64,
    eps_r::Float64,
    N::Int,
    coefficient_count::Int,
    pec::Bool,
)
    coefficients = Vector{ComplexF64}(undef, coefficient_count)
    center = N + 1
    setprecision(BigFloat, _MIE2D_FALLBACK_PRECISION) do
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

    # A cylinder identical to the exterior medium has exactly zero contrast.
    # Returning the analytic result also avoids overflow in high-order Hankel
    # intermediates for electrically tiny matched cylinders.
    if !pec && eps_r == 1.0
        return zeros(ComplexF64, coefficient_count), N
    end

    if k0a <= _MIE2D_SMALL_ARGUMENT_CUTOFF &&
       _small_mie2d_fallback_supported(k0, a, eps_r, pec)
        return _mie_coefficients_small_bigfloat_2d(
            k0, a, eps_r, N, coefficient_count, pec)
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
                                      phi_inc::Float64;
                                      require_representable_phase::Bool=true)
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
        if require_representable_phase
            argument = k0 * rho
            (isfinite(argument) && argument > 0.0) ||
                throw(ArgumentError(
                    "k0*norm(r_obs[$m]) must be finite and positive, got $argument"))
        end
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
    if !pec
        isfinite(eps_r) ||
            throw(ArgumentError("eps_r must be finite, got $eps_r"))
    end
    matched_medium = !pec && eps_r == 1.0
    _validate_mie2d_observations(
        k0, a, r_obs, phi_inc;
        require_representable_phase=!matched_medium,
    )
    matched_medium && return zeros(ComplexF64, length(r_obs))
    c, N = mie_coefficients_2d(k0, a, eps_r; nmax=nmax, pec=pec)

    M = length(r_obs)
    E_scat = zeros(ComplexF64, M)

    for m in 1:M
        rho = hypot(r_obs[m][1], r_obs[m][2])
        phi = atan(r_obs[m][2], r_obs[m][1])

        for n in -N:N
            coefficient = c[n + N + 1]
            iszero(coefficient) && continue
            E_scat[m] += (-im + 0.0)^n * coefficient * besselh(n, 2, k0 * rho) *
                         exp(im * n * (phi - phi_inc))
        end
        isfinite(E_scat[m]) ||
            error("2D Mie scattered field at observation $m is non-finite")
    end

    return E_scat
end

@noinline function _mie_incident_phase_big_2d(
        k0::Float64,
        direction::Vec2,
        observation::Vec2,
        index)
    return setprecision(BigFloat, _MIE2D_FALLBACK_PRECISION) do
        phase = BigFloat(k0) *
                (BigFloat(direction[1]) * BigFloat(observation[1]) +
                 BigFloat(direction[2]) * BigFloat(observation[2]))
        value = ComplexF64(
            exp(Complex{BigFloat}(zero(BigFloat), -phase)))
        isfinite(value) ||
            error(
                "2D Mie incident field at observation $index is non-finite")
        return value
    end
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
        phase = k0 * dot(khat, r_obs[m])
        incident = isfinite(phase) ?
                   exp(-im * phase) :
                   _mie_incident_phase_big_2d(
                       k0, khat, r_obs[m], m)
        E_total[m] += incident
        isfinite(E_total[m]) ||
            error("2D Mie total field at observation $m is non-finite")
    end
    return E_total
end
