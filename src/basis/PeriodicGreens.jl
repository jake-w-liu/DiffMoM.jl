# PeriodicGreens.jl — 2D-periodic Green's function via Helmholtz-Ewald summation
#
# For a 2D lattice with vectors a1 = (dx, 0, 0), a2 = (0, dy, 0),
# the quasi-periodic Green's function is:
#
#   G_per(r, r') = Σ_{m,n} G_0(r, r' + R_mn) × exp(-i k_∥ · R_mn)
#
# where k_∥ = (kx, ky) is the Bloch wave vector and G_0 = exp(-ikR)/(4πR).
#
# Implementation: Helmholtz-Ewald splitting (Capolino/Wilton/Johnson, IEEE TAP 2005).
#   G_per = S_spatial + S_spectral
#
# Both sums converge exponentially with splitting parameter E.
# The periodic correction ΔG = G_per - G_0 is decomposed as:
#   ΔG = self_correction + spatial_images + spectral_sum
#
# Numerical stability: E is set to max(sqrt(π/A), k/(2√α_max)) where α_max = 2.
# This keeps k²/(4E²) ≤ 2, avoiding catastrophic cancellation between spatial
# and spectral sums (which individually grow as exp(k²/(4E²))).
# For large periods, N_spectral is automatically enlarged to include all
# propagating Floquet modes plus evanescent convergence margin.
#
# Convention: exp(+iωt), G_0 = exp(-ikR)/(4πR)
#
# NOTE:
# The topology workflows use coplanar source and observation meshes, but the
# Ewald kernel itself accepts nonzero vertical separation Δz. The grounded-EFIE
# image block relies on this path to couple the real sheet to its PEC image.

export greens_periodic_correction, PeriodicLattice

# Maximum allowed exponent k²/(4E²). Both the spatial self-correction and
# the spectral sum grow as exp(α), and must cancel to give O(1) result.
# With α = 2: exp(2) ≈ 7.4, losing < 1 digit. Safe for any period.
const _EWALD_MAX_EXP_ARG = 2.0
# Every truncation is traversed as a two-dimensional `(2N+1)^2` lattice.
# Bound a single traversal to one million terms: this keeps explicit Floquet
# storage below about 72 MB and prevents an accepted input from committing the
# process to a practically unbounded loop. The derived order is 499, yielding
# 998,001 terms; order 500 would exceed the budget.
const _MAX_PERIODIC_TERM_COUNT = 1_000_000
const _MAX_PERIODIC_TRUNCATION =
    (isqrt(_MAX_PERIODIC_TERM_COUNT) - 1) ÷ 2

@inline function _periodic_term_count(order::Int)
    side = Base.checked_add(Base.checked_mul(2, order), 1)
    return Base.checked_mul(side, side)
end

@inline function _finite_periodic_parameter(name::AbstractString, value::Real)
    value_f = Float64(value)
    isfinite(value_f) ||
        throw(ArgumentError("$name must be finite, got $value"))
    return value_f
end

@inline function _positive_periodic_parameter(name::AbstractString, value::Real)
    value_f = _finite_periodic_parameter(name, value)
    value_f > 0.0 ||
        throw(ArgumentError("$name must be positive, got $value"))
    return value_f
end

@inline function _periodic_truncation_order(name::AbstractString, value::Integer)
    value >= 0 ||
        throw(ArgumentError("$name must be nonnegative, got $value"))
    value <= _MAX_PERIODIC_TRUNCATION ||
        throw(ArgumentError(
            "$name=$value is too large; maximum supported value is " *
            "$_MAX_PERIODIC_TRUNCATION " *
            "($_MAX_PERIODIC_TERM_COUNT lattice terms)"
        ))
    order = Int(value)
    _periodic_term_count(order) <= _MAX_PERIODIC_TERM_COUNT ||
        error("periodic truncation resource-bound invariant violated")
    return order
end

function _auto_periodic_truncation(name::AbstractString, value::Float64)
    (isfinite(value) && value >= 0.0 && value <= _MAX_PERIODIC_TRUNCATION) ||
        throw(ArgumentError(
            "lattice parameters require unsupported $name=$value; " *
            "reduce the period or wavenumber"
        ))
    return _periodic_truncation_order(name, ceil(Int, value))
end

"""
    PeriodicLattice

2D lattice parameters for Ewald-accelerated periodic Green's function.
"""
struct PeriodicLattice
    dx::Float64                  # period in x
    dy::Float64                  # period in y
    kx_bloch::Float64            # Bloch phase shift x (from incident angle)
    ky_bloch::Float64            # Bloch phase shift y (from incident angle)
    k::Float64                   # free-space wavenumber
    E::Float64                   # Ewald splitting parameter
    N_spatial::Int               # truncation order for spatial sum
    N_spectral::Int              # truncation order for spectral sum

    function PeriodicLattice(dx::Float64, dy::Float64,
                             kx_bloch::Float64, ky_bloch::Float64,
                             k::Float64, E::Float64,
                             N_spatial::Int, N_spectral::Int)
        _positive_periodic_parameter("dx", dx)
        _positive_periodic_parameter("dy", dy)
        _finite_periodic_parameter("kx_bloch", kx_bloch)
        _finite_periodic_parameter("ky_bloch", ky_bloch)
        _positive_periodic_parameter("k", k)
        _positive_periodic_parameter("E", E)
        _periodic_truncation_order("N_spatial", N_spatial)
        _periodic_truncation_order("N_spectral", N_spectral)
        return new(dx, dy, kx_bloch, ky_bloch, k, E, N_spatial, N_spectral)
    end
end

function PeriodicLattice(dx::Real, dy::Real,
                         kx_bloch::Real, ky_bloch::Real,
                         k::Real, E::Real,
                         N_spatial::Integer, N_spectral::Integer)
    return PeriodicLattice(
        Float64(dx), Float64(dy), Float64(kx_bloch), Float64(ky_bloch),
        Float64(k), Float64(E),
        _periodic_truncation_order("N_spatial", N_spatial),
        _periodic_truncation_order("N_spectral", N_spectral),
    )
end

@inline function _validated_lattice_wavenumber(k::Real,
                                               lattice::PeriodicLattice)
    kw = _positive_periodic_parameter("k", k)
    kw == lattice.k && return kw
    isapprox(kw, lattice.k; rtol=1e-12, atol=0.0) ||
        throw(ArgumentError(
            "k must match lattice.k: got k=$kw and lattice.k=$(lattice.k). " *
            "Construct a new PeriodicLattice for this wavenumber."
        ))
    return lattice.k
end

"""
    PeriodicLattice(dx, dy, theta_inc, phi_inc, k; N_spatial=4, N_spectral=4)

Construct a PeriodicLattice with Ewald splitting from physical parameters.

- `theta_inc`, `phi_inc`: incident angles (radians)
- `k`: free-space wavenumber
- `N_spatial`, `N_spectral`: truncation orders (minimum values;
   automatically increased for large periods to maintain convergence)

Periods and `k` must be finite and positive, angles must be finite, and
truncation orders must be between zero and `_MAX_PERIODIC_TRUNCATION` (499),
inclusive. Invalid inputs throw `ArgumentError`.

For periods d >> λ, the splitting parameter E is increased above sqrt(π/A)
to maintain numerical stability, and N_spectral is enlarged to include all
propagating Floquet modes.
"""
function PeriodicLattice(dx::Real, dy::Real, theta_inc::Real, phi_inc::Real, k::Real;
                         N_spatial::Int=4, N_spectral::Int=4)
    dxf = _positive_periodic_parameter("dx", dx)
    dyf = _positive_periodic_parameter("dy", dy)
    theta_f = _finite_periodic_parameter("theta_inc", theta_inc)
    phi_f = _finite_periodic_parameter("phi_inc", phi_inc)
    kf = _positive_periodic_parameter("k", k)
    Ns = _periodic_truncation_order("N_spatial", N_spatial)
    Nf_input = _periodic_truncation_order("N_spectral", N_spectral)

    kx = kf * sin(theta_f) * cos(phi_f)
    ky = kf * sin(theta_f) * sin(phi_f)

    # Optimal splitting parameter (balances spatial/spectral work for d ~ λ)
    E_opt = sqrt(π) / sqrt(dxf) / sqrt(dyf)
    (isfinite(E_opt) && E_opt > 0.0) ||
        throw(ArgumentError(
            "dx=$dxf and dy=$dyf are outside the supported Float64 range " *
            "for Ewald splitting"
        ))

    # Minimum E to keep k²/(4E²) ≤ MAX_EXP_ARG (numerical stability)
    # Both spatial self-correction and spectral sum grow as exp(k²/(4E²));
    # their cancellation loses log10(exp(α)) digits of precision.
    E_min = kf / (2 * sqrt(_EWALD_MAX_EXP_ARG))

    E = max(E_opt, E_min)

    # Auto-compute N_spectral: must include ALL propagating Floquet modes
    # (which have erfc values of order exp(kz²/(4E²))) plus evanescent margin.
    # Propagating modes: |p| ≤ k*dx/(2π), |q| ≤ k*dy/(2π)
    # Evanescent convergence: need |kz|/(2E) > M where erfc(M) < eps.
    #   M ≈ 5 gives erfc(5) ≈ 1.5e-12.
    #   |kz| > 2E*M → κ_t > sqrt(k² + 4E²M²)
    #   N_spectral > d * sqrt(k² + 4E²M²) / (2π)
    M_erfc = 5.0  # erfc(5) ≈ 1.5e-12
    spectral_radius = hypot(kf, (2 * M_erfc) * E)
    Nf_x = _auto_periodic_truncation(
        "N_spectral", (dxf / (2π)) * spectral_radius
    )
    Nf_y = _auto_periodic_truncation(
        "N_spectral", (dyf / (2π)) * spectral_radius
    )
    Nf = max(Nf_input, Nf_x, Nf_y)

    return PeriodicLattice(dxf, dyf, kx, ky, kf, E, Ns, Nf)
end

# ─────────────────────────────────────────────────────────────────
# Ewald spatial kernel
# ─────────────────────────────────────────────────────────────────

"""
    _ewald_spatial_kernel(R, k, E)

Ewald-damped spatial kernel for the Helmholtz Green's function:

  K_sp(R) = Re[exp(-ikR) erfc(ER - ik/(2E))] / (4πR)

This is real-valued and decays as exp(-E²R²) for large R.
"""
function _ewald_spatial_kernel(R::Float64, k::Float64, E::Float64)
    spatial_argument = E * R
    phase_ratio = (k / E) / 2
    if isinf(spatial_argument) && isfinite(phase_ratio)
        return 0.0
    end
    (isfinite(spatial_argument) && isfinite(phase_ratio)) ||
        throw(OverflowError(
            "periodic Green spatial kernel arguments are outside the " *
            "supported Float64 range"))

    # With s=ER and q=k/(2E), the oscillatory factor cancels exactly:
    #   exp(-ikR) erfc(s-iq) = exp(q²-s²) erfcx(s-iq).
    # This form also lets the final division by R rescue an erfc value that
    # would underflow if it were rounded before the complete kernel is formed.
    scaled_erfc_real = real(erfcx(spatial_argument - im * phase_ratio))
    isfinite(scaled_erfc_real) ||
        throw(OverflowError(
            "periodic Green spatial kernel is outside the representable " *
            "Float64 range"))
    iszero(scaled_erfc_real) && return 0.0
    exponent = phase_ratio * phase_ratio -
               spatial_argument * spatial_argument
    if exponent == -Inf
        return 0.0
    end
    isfinite(exponent) ||
        throw(OverflowError(
            "periodic Green spatial kernel is outside the representable " *
            "Float64 range"))

    numerator = exp(exponent) * scaled_erfc_real
    result = (numerator / R) / (4π)
    if isfinite(result) && abs(result) >= floatmin(Float64)
        return result
    end
    log_magnitude = exponent + log(abs(scaled_erfc_real)) -
                    log(R) - log(4π)
    result = copysign(exp(log_magnitude), scaled_erfc_real)
    isfinite(result) ||
        throw(OverflowError(
            "periodic Green spatial kernel is outside the representable " *
            "Float64 range"))
    return result
end

# ─────────────────────────────────────────────────────────────────
# Self-correction: K_sp(R) - G_0(R) for the (0,0) lattice site
# ─────────────────────────────────────────────────────────────────

const _EWALD_SELF_FALLBACK_PRECISION = 256
const _EWALD_SELF_SERIES_TERMS = 1024

function _ewald_self_dimensionless_real_big(q::BigFloat)
    q_squared = q * q
    tolerance_scale = 16eps(BigFloat)
    dawson = q
    term = q
    if abs(q) <= 8
        for order in 1:_EWALD_SELF_SERIES_TERMS
            term *= (-2q_squared) / (2order + 1)
            dawson_next = dawson + term
            dawson = dawson_next
            if abs(term) <= tolerance_scale * max(abs(dawson), one(BigFloat))
                return (4 / sqrt(big(π))) * (2q * dawson - 1)
            end
        end
        throw(ErrorException(
            "periodic Green self-correction Dawson series did not converge"))
    end

    # For large q, evaluate 2q*Dawson(q)-1 directly from its asymptotic
    # series. This avoids subtracting two nearly equal Float64 values.
    inverse_two_q_squared = inv(2q_squared)
    term = inverse_two_q_squared
    series_sum = term
    previous_magnitude = abs(term)
    for order in 2:_EWALD_SELF_SERIES_TERMS
        next_term = term * (2order - 1) * inverse_two_q_squared
        next_magnitude = abs(next_term)
        if next_magnitude >= previous_magnitude
            return (4 / sqrt(big(π))) * series_sum
        end
        series_sum += next_term
        if next_magnitude <= tolerance_scale * abs(series_sum)
            return (4 / sqrt(big(π))) * series_sum
        end
        term = next_term
        previous_magnitude = next_magnitude
    end
    throw(ErrorException(
        "periodic Green self-correction asymptotic series did not converge"))
end

@noinline function _ewald_self_correction_exact_scale(
        k::Float64,
        E::Float64)
    return setprecision(BigFloat, _EWALD_SELF_FALLBACK_PRECISION) do
        k_big = BigFloat(k)
        E_big = BigFloat(E)
        ratio_big = k_big / E_big
        half_ratio_big = ratio_big / 2
        dimensionless_real =
            _ewald_self_dimensionless_real_big(half_ratio_big)
        real_big = E_big * exp(half_ratio_big * half_ratio_big) *
                   dimensionless_real / (8big(π))
        imag_big = k_big / (4big(π))
        result = ComplexF64(Float64(real_big), Float64(imag_big))
        isfinite(result) ||
            throw(OverflowError(
                "periodic Green self correction is outside the " *
                "representable ComplexF64 range"))
        return result
    end
end

"""
    _ewald_self_correction(R, k, E)

Self-correction: K_sp(R) - G_0(R) for the (m=0, n=0) Ewald term.

This is the difference between the Ewald spatial kernel and the
free-space Green's function at the same point. It is smooth
everywhere, with an analytical limit at R → 0 via L'Hôpital.
"""
function _ewald_self_correction(R::Float64, k::Float64, E::Float64)
    k_radius = k * R
    E_radius = E * R
    near_origin = iszero(R) ||
                  (isfinite(k_radius) && isfinite(E_radius) &&
                   max(abs(k_radius), abs(E_radius)) <= 1.0e-6)
    if near_origin
        # R → 0 limit (L'Hôpital on the 0/0 form):
        #   C_self = [2ik erfc(ik/(2E)) - (4E/√π) exp(k²/(4E²))] / (8π)
        #
        # Numerically stable form using erfcx to avoid overflow:
        #   erfc(z) = exp(-z²) erfcx(z), with z = ik/(2E), z² = -k²/(4E²)
        #   C_self = exp(k²/(4E²)) [2ik erfcx(ik/(2E)) - 4E/√π] / (8π)
        if iszero(k)
            return ComplexF64(-(E / (2π)) / √π, 0.0)
        end
        ratio = k / E
        half_ratio = ratio / 2
        exp_arg = half_ratio * half_ratio
        z0 = im * half_ratio
        # Factor k before forming the bracket and divide by 8π first. This
        # avoids k², E², 2k, and 4E intermediate overflow while preserving
        # the analytical R→0 limit.
        inverse_ratio = E / k
        scale = k / (8π)
        if isfinite(ratio) && abs(ratio) >= floatmin(Float64) &&
           isfinite(inverse_ratio) &&
           abs(inverse_ratio) >= floatmin(Float64) &&
           isfinite(scale) && abs(scale) >= floatmin(Float64) &&
           isfinite(exp_arg)
            dimensionless = 2im * erfcx(z0) -
                            (4 * inverse_ratio) / √π
            result = scale * exp(exp_arg) * dimensionless
            if isfinite(result) &&
               (iszero(real(result)) ||
                abs(real(result)) >= floatmin(Float64)) &&
               (iszero(imag(result)) ||
                abs(imag(result)) >= floatmin(Float64))
                return result
            end
        end
        return _ewald_self_correction_exact_scale(k, E)
    end

    # For R > 0: compute K_sp(R) - G_0(R) directly
    K_sp = _ewald_spatial_kernel(R, k, E)
    G_0 = (_periodic_rwg_bloch_phase(k, R) / R) / (4π)
    return K_sp - G_0
end

# ─────────────────────────────────────────────────────────────────
# Spectral sum utilities
# ─────────────────────────────────────────────────────────────────

const _PERIODIC_LONGITUDINAL_FALLBACK_PRECISION = 256
const _PERIODIC_AREA_FALLBACK_PRECISION = 256
const _PERIODIC_PHASE_FALLBACK_PRECISION = 2304

@noinline function _periodic_transverse_phase_exact(
        kx::Float64,
        ky::Float64,
        x::Float64,
        y::Float64)
    return setprecision(BigFloat, _PERIODIC_PHASE_FALLBACK_PRECISION) do
        argument = BigFloat(kx) * BigFloat(x) +
                   BigFloat(ky) * BigFloat(y)
        ComplexF64(exp(Complex{BigFloat}(0, -argument)))
    end
end

@inline function _periodic_phase_term_requires_exact(
        first::Float64,
        second::Float64,
        product::Float64)
    (iszero(first) || iszero(second)) && return false
    return !isfinite(product) || iszero(product) ||
           abs(product) < floatmin(Float64)
end

@inline function _periodic_transverse_phase(
        kx::Float64,
        ky::Float64,
        x::Float64,
        y::Float64)
    x_product = kx * x
    y_product = ky * y
    if _periodic_phase_term_requires_exact(kx, x, x_product) ||
       _periodic_phase_term_requires_exact(ky, y, y_product)
        return _periodic_transverse_phase_exact(kx, ky, x, y)
    end

    x_error = fma(kx, x, -x_product)
    y_error = fma(ky, y, -y_product)
    if !(isfinite(x_error) && isfinite(y_error))
        return _periodic_transverse_phase_exact(kx, ky, x, y)
    end
    reduced_x = rem2pi(
        rem2pi(x_product, RoundNearest) +
        rem2pi(x_error, RoundNearest),
        RoundNearest,
    )
    reduced_y = rem2pi(
        rem2pi(y_product, RoundNearest) +
        rem2pi(y_error, RoundNearest),
        RoundNearest,
    )
    reduced = rem2pi(reduced_x + reduced_y, RoundNearest)
    phase = cis(-reduced)
    return isfinite(real(phase)) && isfinite(imag(phase)) ?
           ComplexF64(phase) :
           _periodic_transverse_phase_exact(kx, ky, x, y)
end

@inline function _periodic_spectral_vertical_kernel(
        kz::ComplexF64,
        zk::ComplexF64,
        E::Float64,
        separation::Float64)
    if iszero(imag(kz))
        magnitude = real(kz)
        negative_phase = _periodic_rwg_bloch_phase(magnitude, separation)
        positive_phase = conj(negative_phase)
        scaled_separation = E * separation
        if isinf(scaled_separation)
            selected = scaled_separation > 0 ? negative_phase : positive_phase
            return selected / (2im * kz)
        end
        isfinite(scaled_separation) ||
            throw(OverflowError(
                "periodic Green vertical separation is outside the supported range"))
        return (
            negative_phase * erfc(zk - scaled_separation) +
            positive_phase * erfc(zk + scaled_separation)
        ) / (4im * kz)
    end

    # For kz=-iγ, evaluating exp(+γ|z|)*erfc(q+E|z|)
    # separately creates Inf*0.  Combine that term with erfcx instead:
    # exp(+2qs)erfc(q+s) = exp(-q²-s²)erfcx(q+s).
    gamma = -imag(kz)
    gamma > 0 ||
        throw(ArgumentError(
            "periodic Green evanescent longitudinal wavenumber has the wrong branch"))
    q = real(zk)
    abs_separation = abs(separation)
    s = E * abs_separation
    decay = gamma * abs_separation
    first_term = exp(-decay) * erfc(q - s)
    second_exponent = -(q * q) - s * s
    second_term = exp(second_exponent) * erfcx(q + s)
    value = ((first_term + second_term) / gamma) / 4
    isfinite(value) ||
        throw(OverflowError(
            "periodic Green evanescent spectral kernel is outside the representable range"))
    return ComplexF64(value)
end

@noinline function _periodic_scale_by_cell_area_exact(
        value::ComplexF64,
        dx::Float64,
        dy::Float64)
    return setprecision(BigFloat, _PERIODIC_AREA_FALLBACK_PRECISION) do
        scaled_big = Complex{BigFloat}(value) /
                     (BigFloat(dx) * BigFloat(dy))
        scaled = ComplexF64(scaled_big)
        isfinite(scaled) ||
            throw(OverflowError(
                "periodic Green spectral contribution is outside the " *
                "representable ComplexF64 range"))
        return scaled
    end
end

@inline function _periodic_scale_by_cell_area_component(
        value::Float64,
        denominator_mantissa::Float64,
        denominator_exponent::Int)
    iszero(value) && return value, true
    scaled = if denominator_exponent >= 0
        ldexp(value, -denominator_exponent) / denominator_mantissa
    else
        ldexp(value / denominator_mantissa, -denominator_exponent)
    end
    certified = isfinite(scaled) &&
                !iszero(scaled) &&
                abs(scaled) >= floatmin(Float64)
    return scaled, certified
end

@inline function _periodic_scale_by_cell_area_wide(
        value::ComplexF64,
        dx::Float64,
        dy::Float64)
    dx_mantissa, dx_exponent = frexp(dx)
    dy_mantissa, dy_exponent = frexp(dy)
    denominator_mantissa = dx_mantissa * dy_mantissa
    denominator_exponent = dx_exponent + dy_exponent
    if denominator_mantissa < 0.5
        denominator_mantissa *= 2
        denominator_exponent -= 1
    end
    real_scaled, real_certified =
        _periodic_scale_by_cell_area_component(
            real(value), denominator_mantissa, denominator_exponent)
    imag_scaled, imag_certified =
        _periodic_scale_by_cell_area_component(
            imag(value), denominator_mantissa, denominator_exponent)
    if (iszero(real(value)) || real_certified) &&
       (iszero(imag(value)) || imag_certified)
        return ComplexF64(real_scaled, imag_scaled)
    end
    return _periodic_scale_by_cell_area_exact(value, dx, dy)
end

@inline function _periodic_scale_by_cell_area(
        value::ComplexF64,
        dx::Float64,
        dy::Float64)
    iszero(value) && return value
    area = dx * dy
    if isfinite(area) && area >= floatmin(Float64)
        scaled = value / area
        if isfinite(scaled) &&
           !((iszero(real(scaled)) && !iszero(real(value))) ||
             (iszero(imag(scaled)) && !iszero(imag(value))))
            return scaled
        end
    end
    return _periodic_scale_by_cell_area_wide(value, dx, dy)
end

@noinline function _periodic_longitudinal_magnitude_exact(
    k::Float64,
    kappa_x::Float64,
    kappa_y::Float64,
    label::AbstractString,
)
    return setprecision(
            BigFloat, _PERIODIC_LONGITUDINAL_FALLBACK_PRECISION) do
        k_big = BigFloat(k)
        kappa_x_big = BigFloat(kappa_x)
        kappa_y_big = BigFloat(kappa_y)
        radicand = k_big * k_big -
                   kappa_x_big * kappa_x_big -
                   kappa_y_big * kappa_y_big
        propagating = radicand > 0
        magnitude_big = sqrt(abs(radicand))
        magnitude = Float64(magnitude_big)
        isfinite(magnitude) ||
            throw(OverflowError("$label is outside the Float64 range"))
        if !iszero(magnitude_big) && iszero(magnitude)
            throw(ArgumentError("$label is below the Float64 range"))
        end
        return magnitude, propagating
    end
end

@inline function _periodic_longitudinal_magnitude(
    k::Float64,
    kappa_x::Float64,
    kappa_y::Float64,
    label::AbstractString,
)
    isfinite(k) && k >= 0.0 ||
        throw(ArgumentError("k must be finite and nonnegative, got $k"))
    all(isfinite, (kappa_x, kappa_y)) ||
        throw(ArgumentError(
            "transverse wavevector must be finite, got " *
            "($kappa_x, $kappa_y)"))

    scale = max(k, abs(kappa_x), abs(kappa_y))
    iszero(scale) && return 0.0, false
    k_scaled = k / scale
    kappa_x_scaled = kappa_x / scale
    kappa_y_scaled = kappa_y / scale
    k_squared = k_scaled * k_scaled
    transverse_squared =
        kappa_x_scaled * kappa_x_scaled +
        kappa_y_scaled * kappa_y_scaled
    radicand_scaled = k_squared - transverse_squared
    uncertainty = 64eps(Float64) * (k_squared + transverse_squared)
    if abs(radicand_scaled) <= uncertainty
        return _periodic_longitudinal_magnitude_exact(
            k, kappa_x, kappa_y, label)
    end

    magnitude = scale * sqrt(abs(radicand_scaled))
    if !isfinite(magnitude) || iszero(magnitude)
        return _periodic_longitudinal_magnitude_exact(
            k, kappa_x, kappa_y, label)
    end
    return magnitude, radicand_scaled > 0.0
end

"""
    _spectral_kz(k, kappa_x, kappa_y)

Compute kz = sqrt(k² - κ²) with branch cut ensuring Im(kz) ≤ 0
(outgoing/decaying convention for exp(+iωt)).
"""
function _spectral_kz(k::Float64, kappa_x::Float64, kappa_y::Float64)
    magnitude, propagating = _periodic_longitudinal_magnitude(
        k, kappa_x, kappa_y, "spectral longitudinal wavevector")
    return propagating ?
           ComplexF64(magnitude, 0.0) :
           ComplexF64(0.0, -magnitude)
end

# ─────────────────────────────────────────────────────────────────
# Main function: periodic correction via Ewald
# ─────────────────────────────────────────────────────────────────

"""
    greens_periodic_correction(r, rp, k, lattice)

Periodic correction to the free-space Green's function via Ewald summation:

    ΔG(r, r') = G_per(r, r') - G_0(r, r')

Decomposed into three exponentially convergent sums:
1. **Self-correction**: K_sp(R) - G_0(R) at the (0,0) lattice site
2. **Spatial images**: Σ_{(m,n)≠(0,0)} phase_mn × K_sp(R_mn)
3. **Spectral sum**: Floquet mode expansion with erfc damping

Numerically stable for any period via E-clamping (see `PeriodicLattice`).
The call-site `k` must match `lattice.k`, because the Bloch phase, Ewald split,
and spectral truncation stored in the lattice depend on that wavenumber.
"""
function greens_periodic_correction(r::SVector{3,<:Real},
                                    rp::SVector{3,<:Real}, k::Real,
                                    lattice::PeriodicLattice)
    kw = _validated_lattice_wavenumber(k, lattice)
    dx = lattice.dx
    dy = lattice.dy
    kx = lattice.kx_bloch
    ky = lattice.ky_bloch
    E  = lattice.E
    Ns = lattice.N_spatial
    Nf = lattice.N_spectral

    CT = ComplexF64
    val = zero(CT)

    # Observation-source displacement
    rx = _finite_periodic_parameter("r[1]", r[1])
    ry = _finite_periodic_parameter("r[2]", r[2])
    rz = _finite_periodic_parameter("r[3]", r[3])
    rpx = _finite_periodic_parameter("rp[1]", rp[1])
    rpy = _finite_periodic_parameter("rp[2]", rp[2])
    rpz = _finite_periodic_parameter("rp[3]", rp[3])
    drho_x = rx - rpx
    drho_y = ry - rpy
    drho_z = rz - rpz
    (isfinite(drho_x) && isfinite(drho_y) && isfinite(drho_z)) ||
        throw(ArgumentError(
            "r-rp is outside the supported Float64 coordinate range"
        ))

    # Vertical separation drho_z = z - z' is supported: the spatial sum and
    # self-correction already use the full 3D distance, and the spectral sum below
    # carries the Δz-dependent Ewald kernel. drho_z = 0 recovers the coplanar form.

    # ── 1. Self-correction: (m=0, n=0) term ──
    R_self = hypot(hypot(drho_x, drho_y), drho_z)
    val += _ewald_self_correction(R_self, kw, E)

    # ── 2. Spatial images: (m,n) ≠ (0,0) with Ewald damping ──
    @inbounds for m in -Ns:Ns
        for n in -Ns:Ns
            (m == 0 && n == 0) && continue

            # Image displacement
            sx = m * dx
            sy = n * dy
            R_mn = hypot(
                hypot(drho_x - sx, drho_y - sy), drho_z)

            # Bloch phase: exp(-i k_∥ · R_mn)
            phase = _periodic_transverse_phase(kx, ky, sx, sy)

            # Ewald-damped spatial kernel (real-valued)
            K_sp = _ewald_spatial_kernel(R_mn, kw, E)

            val += phase * K_sp
        end
    end

    # ── 3. Spectral sum: Floquet modes with erfc damping ──
    @inbounds for p in -Nf:Nf
        for q in -Nf:Nf
            # Floquet wave vector (transverse)
            kappa_x = kx + 2π * p / dx
            kappa_y = ky + 2π * q / dy

            # z-component with proper branch cut
            kz = _spectral_kz(kw, kappa_x, kappa_y)

            # Skip Wood anomaly (kz ≈ 0). Relative threshold needed because
            # off-axis modes at the light cone (e.g. p²+q² = (d/λ)² via
            # Pythagorean triples) can have |kz| ≈ 3e-6 from floating-point
            # error in κ² - k², far above an absolute 1e-12 threshold.
            # Nearest non-Wood mode has |kz| ≥ k/d_λ, giving >1e4 safety margin.
            abs(kz) < 1e-6 * kw && continue

            # Phase from observation-source offset
            phase_spec = _periodic_transverse_phase(
                kappa_x, kappa_y, drho_x, drho_y)

            # Ewald-damped spectral kernel with vertical separation Δz = drho_z.
            # Reduces to erfc(ikz/(2E))/(2ikz) at Δz = 0 and to the physical
            # plane-wave factor e^{-ikz|Δz|}/(2ikz) as E → ∞ (no Ewald damping):
            #   spec(Δz) = [ e^{-ikz Δz} erfc(ikz/2E - E Δz)
            #              + e^{+ikz Δz} erfc(ikz/2E + E Δz) ] / (4 i kz)
            zk = im * kz / (2E)
            spec_val = _periodic_spectral_vertical_kernel(
                kz, zk, E, drho_z)

            val += _periodic_scale_by_cell_area(
                phase_spec * spec_val, dx, dy)
        end
    end

    isfinite(val) ||
        throw(OverflowError(
            "periodic Green correction is outside the representable " *
            "ComplexF64 range"))
    return val
end
