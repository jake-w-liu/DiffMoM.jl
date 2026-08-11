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
    z = E * R - im * k / (2E)
    return real(exp(-im * k * R) * erfc(z)) / (4π * R)
end

# ─────────────────────────────────────────────────────────────────
# Self-correction: K_sp(R) - G_0(R) for the (0,0) lattice site
# ─────────────────────────────────────────────────────────────────

"""
    _ewald_self_correction(R, k, E)

Self-correction: K_sp(R) - G_0(R) for the (m=0, n=0) Ewald term.

This is the difference between the Ewald spatial kernel and the
free-space Green's function at the same point. It is smooth
everywhere, with an analytical limit at R → 0 via L'Hôpital.
"""
function _ewald_self_correction(R::Float64, k::Float64, E::Float64)
    if R < 1e-14
        # R → 0 limit (L'Hôpital on the 0/0 form):
        #   C_self = [2ik erfc(ik/(2E)) - (4E/√π) exp(k²/(4E²))] / (8π)
        #
        # Numerically stable form using erfcx to avoid overflow:
        #   erfc(z) = exp(-z²) erfcx(z), with z = ik/(2E), z² = -k²/(4E²)
        #   C_self = exp(k²/(4E²)) [2ik erfcx(ik/(2E)) - 4E/√π] / (8π)
        exp_arg = k^2 / (4E^2)
        z0 = im * k / (2E)
        bracket = 2im * k * erfcx(z0) - 4E / √π
        return exp(exp_arg) * bracket / (8π)
    end

    # For R > 0: compute K_sp(R) - G_0(R) directly
    K_sp = _ewald_spatial_kernel(R, k, E)
    G_0 = exp(-im * k * R) / (4π * R)
    return K_sp - G_0
end

# ─────────────────────────────────────────────────────────────────
# Spectral sum utilities
# ─────────────────────────────────────────────────────────────────

const _PERIODIC_LONGITUDINAL_FALLBACK_PRECISION = 256

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
    A  = dx * dy  # unit cell area

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
    R_self = sqrt(drho_x^2 + drho_y^2 + drho_z^2)
    val += _ewald_self_correction(R_self, kw, E)

    # ── 2. Spatial images: (m,n) ≠ (0,0) with Ewald damping ──
    @inbounds for m in -Ns:Ns
        for n in -Ns:Ns
            (m == 0 && n == 0) && continue

            # Image displacement
            sx = m * dx
            sy = n * dy
            R_mn = sqrt((drho_x - sx)^2 + (drho_y - sy)^2 + drho_z^2)

            # Bloch phase: exp(-i k_∥ · R_mn)
            phase = exp(-im * (kx * sx + ky * sy))

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
            phase_spec = exp(-im * (kappa_x * drho_x + kappa_y * drho_y))

            # Ewald-damped spectral kernel with vertical separation Δz = drho_z.
            # Reduces to erfc(ikz/(2E))/(2ikz) at Δz = 0 and to the physical
            # plane-wave factor e^{-ikz|Δz|}/(2ikz) as E → ∞ (no Ewald damping):
            #   spec(Δz) = [ e^{-ikz Δz} erfc(ikz/2E - E Δz)
            #              + e^{+ikz Δz} erfc(ikz/2E + E Δz) ] / (4 i kz)
            zk = im * kz / (2E)
            Edz = E * drho_z
            spec_val = (exp(-im * kz * drho_z) * erfc(zk - Edz) +
                        exp( im * kz * drho_z) * erfc(zk + Edz)) / (4im * kz)

            val += phase_spec * spec_val / A
        end
    end

    return val
end
