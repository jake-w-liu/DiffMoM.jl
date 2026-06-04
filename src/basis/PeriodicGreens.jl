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
# The current Ewald correction path is implemented for coplanar periodic surfaces
# (z = const over the unit cell), which is the use case in this repository's
# periodic topology workflows. Non-coplanar source/observation offsets are
# rejected at runtime to avoid silently returning non-physical values.

export greens_periodic_correction, PeriodicLattice

# Maximum allowed exponent k²/(4E²). Both the spatial self-correction and
# the spectral sum grow as exp(α), and must cancel to give O(1) result.
# With α = 2: exp(2) ≈ 7.4, losing < 1 digit. Safe for any period.
const _EWALD_MAX_EXP_ARG = 2.0

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
end

"""
    PeriodicLattice(dx, dy, theta_inc, phi_inc, k; N_spatial=4, N_spectral=4)

Construct a PeriodicLattice with Ewald splitting from physical parameters.

- `theta_inc`, `phi_inc`: incident angles (radians)
- `k`: free-space wavenumber
- `N_spatial`, `N_spectral`: truncation orders (minimum values;
   automatically increased for large periods to maintain convergence)

For periods d >> λ, the splitting parameter E is increased above sqrt(π/A)
to maintain numerical stability, and N_spectral is enlarged to include all
propagating Floquet modes.
"""
function PeriodicLattice(dx::Real, dy::Real, theta_inc::Real, phi_inc::Real, k::Real;
                         N_spatial::Int=4, N_spectral::Int=4)
    kx = k * sin(theta_inc) * cos(phi_inc)
    ky = k * sin(theta_inc) * sin(phi_inc)
    kf = Float64(k)

    # Optimal splitting parameter (balances spatial/spectral work for d ~ λ)
    E_opt = sqrt(π / (dx * dy))

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
    Nf = N_spectral
    M_erfc = 5.0  # erfc(5) ≈ 1.5e-12
    Nf_x = ceil(Int, dx * sqrt(kf^2 + 4 * E^2 * M_erfc^2) / (2π))
    Nf_y = ceil(Int, dy * sqrt(kf^2 + 4 * E^2 * M_erfc^2) / (2π))
    Nf = max(N_spectral, Nf_x, Nf_y)

    return PeriodicLattice(dx, dy, kx, ky, kf, E, N_spatial, Nf)
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

"""
    _spectral_kz(k, kappa_x, kappa_y)

Compute kz = sqrt(k² - κ²) with branch cut ensuring Im(kz) ≤ 0
(outgoing/decaying convention for exp(+iωt)).
"""
function _spectral_kz(k::Float64, kappa_x::Float64, kappa_y::Float64)
    kt_sq = kappa_x^2 + kappa_y^2
    kz = sqrt(complex(k^2 - kt_sq))
    # Enforce Im(kz) ≤ 0 for outgoing wave convention
    if imag(kz) > 0
        kz = -kz
    end
    return kz
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
"""
function greens_periodic_correction(r::SVector{3}, rp::SVector{3}, k,
                                    lattice::PeriodicLattice)
    dx = lattice.dx
    dy = lattice.dy
    kx = lattice.kx_bloch
    ky = lattice.ky_bloch
    kw = Float64(k)
    E  = lattice.E
    Ns = lattice.N_spatial
    Nf = lattice.N_spectral
    A  = dx * dy  # unit cell area

    CT = ComplexF64
    val = zero(CT)

    # Observation-source displacement
    drho_x = r[1] - rp[1]
    drho_y = r[2] - rp[2]
    drho_z = r[3] - rp[3]

    if abs(drho_z) > 1e-12
        throw(ArgumentError(
            "greens_periodic_correction currently supports coplanar points only " *
            "(|z-z'| <= 1e-12). Got |z-z'|=$(abs(drho_z))."
        ))
    end

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

            # Ewald-damped spectral kernel: erfc(ikz/(2E)) / (2ikz)
            spec_val = erfc(im * kz / (2E)) / (2im * kz)

            val += phase_spec * spec_val / A
        end
    end

    return val
end
