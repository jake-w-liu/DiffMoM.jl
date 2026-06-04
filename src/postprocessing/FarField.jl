# FarField.jl — Far-field radiation pattern computation
#
# E∞(r̂) = (ik η₀)/(4π) r̂ × [r̂ × ∫ J(r') exp(ik r̂·r') dS']
#        = Σ_n I_n g_n(r̂)

export make_sph_grid, radiation_vectors, compute_farfield, incident_farfield

"""
    make_sph_grid(Ntheta, Nphi)

Create a spherical grid using a uniform midpoint rule in θ and φ, with
quadrature weights w = sin(θ) dθ dφ.
Returns a SphGrid.
"""
function make_sph_grid(Ntheta::Int, Nphi::Int)
    # Simple uniform grid (sufficient for moderate resolution)
    dtheta = π / Ntheta
    dphi   = 2π / Nphi

    NΩ = Ntheta * Nphi
    rhat  = zeros(3, NΩ)
    theta = zeros(NΩ)
    phi   = zeros(NΩ)
    w     = zeros(NΩ)

    idx = 0
    for it in 1:Ntheta
        θ = (it - 0.5) * dtheta
        for ip in 1:Nphi
            φ = (ip - 0.5) * dphi
            idx += 1
            theta[idx] = θ
            phi[idx]   = φ
            rhat[1, idx] = sin(θ) * cos(φ)
            rhat[2, idx] = sin(θ) * sin(φ)
            rhat[3, idx] = cos(θ)
            w[idx] = sin(θ) * dtheta * dphi
        end
    end

    return SphGrid(rhat, theta, phi, w)
end

"""
    radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=376.730313668)

Compute the per-basis radiation vectors g_n(r̂_q) for all basis functions
and all grid directions.

Returns G_mat of size (3*NΩ, N) such that
  G_mat[(3*(q-1)+1):(3*q), n] = g_n(r̂_q) ∈ C³
"""
function radiation_vectors(mesh::TriMesh, rwg::RWGData, grid::SphGrid, k;
                           quad_order::Int=3, eta0=376.730313668)
    N = rwg.nedges
    NΩ = length(grid.w)
    Nt = ntriangles(mesh)
    prefactor = 1im * k * eta0 / (4π)

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # Precompute quad points and areas per triangle
    quad_pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    @inbounds for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
    end

    # Precompute rhat Vec3
    rhat_vec = Vector{Vec3}(undef, NΩ)
    @inbounds for q in 1:NΩ
        rhat_vec[q] = Vec3(grid.rhat[1, q], grid.rhat[2, q], grid.rhat[3, q])
    end

    # Precompute phase exp(ik * rhat · rp) per (grid direction, triangle, quad point).
    # This only depends on (rhat, rp), not on the basis function, so it is shared
    # across all RWG functions touching the same triangle.
    # Layout: phase_cache[q_dir, q_surf, t]
    phase_cache = Array{ComplexF64, 3}(undef, NΩ, Nq, Nt)
    Threads.@threads for t in 1:Nt
        @inbounds for q_surf in 1:Nq
            rp = quad_pts[t][q_surf]
            for q_dir in 1:NΩ
                phase_cache[q_dir, q_surf, t] = exp(1im * k * dot(rhat_vec[q_dir], rp))
            end
        end
    end

    # Assemble G_mat with parallelization over basis functions
    G_mat = zeros(ComplexF64, 3 * NΩ, N)

    Threads.@threads for n in 1:N
        @inbounds for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]

            for q_surf in 1:Nq
                rp = pts[q_surf]
                fn = eval_rwg(rwg, n, rp, t)
                wt = wq[q_surf] * 2 * A

                for q_dir in 1:NΩ
                    rh = rhat_vec[q_dir]
                    phase = phase_cache[q_dir, q_surf, t]

                    contrib = fn * (wt * phase)
                    rh_cross_N_cross = rh * dot(rh, contrib) - contrib

                    idx = 3 * (q_dir - 1)
                    G_mat[idx+1, n] += prefactor * rh_cross_N_cross[1]
                    G_mat[idx+2, n] += prefactor * rh_cross_N_cross[2]
                    G_mat[idx+3, n] += prefactor * rh_cross_N_cross[3]
                end
            end
        end
    end

    return G_mat
end

"""
    compute_farfield(G_mat, I_coeffs, NΩ)

Compute the far-field E∞(r̂_q) = Σ_n I_n g_n(r̂_q) for all grid points.
Returns a (3, NΩ) complex matrix.
"""
function compute_farfield(G_mat::Matrix{ComplexF64}, I_coeffs::Vector{ComplexF64}, NΩ::Int)
    # G_mat is (3*NΩ, N), I_coeffs is (N,)
    E_flat = G_mat * I_coeffs   # (3*NΩ,)
    E = reshape(E_flat, 3, NΩ)
    return E
end

# ══════════════════════════════════════════════════════════════════════
# Incident-field far-field amplitudes
# ══════════════════════════════════════════════════════════════════════
#
# `incident_farfield(excitation, r_hat, k)` returns the asymptotic
# amplitude `E_inc^∞(r̂)` such that `E_inc(r·r̂) ~ E_inc^∞(r̂)·e^{-ikr}/r`
# as `r → ∞`. Summed with `compute_farfield` (scattered contribution
# from solved currents) this gives the **total** far-field pattern of a
# finite scatterer illuminated by a point-like source — the true
# `r → ∞` limit, not a workaround evaluated at large finite `r`.
#
# `PlaneWave` has no 1/r decay in the incident field and is excluded
# (its "far-field" contribution is a delta function in the incident
# direction — not a radiation pattern).

"""
    incident_farfield(excitation, r_hat, k) -> CVec3

Asymptotic amplitude of the incident electric field radiated by
`excitation` in direction `r̂`, such that
`E_inc(r·r̂) ~ incident_farfield(...)·exp(-ikr)/r` as `r → ∞`.
"""
function incident_farfield(excitation::AbstractExcitation, ::Vec3, ::Real)
    error("incident_farfield is not defined for excitation type $(typeof(excitation)).")
end

function incident_farfield(mono::MonopoleExcitation, r_hat::Vec3, k::Real)
    # Closed-form far-field `E_θ^∞(r̂) = iηk sinθ/(4π) · θ̂ · J`
    # with J the wire-current phase integral, computed by Simpson over
    # the appropriate axial range for each source model:
    #   include_image=true  : dipole-plus-image on z' ∈ [-h, +h]
    #   include_image=false : physical half-wire on z' ∈ [ 0, +h]
    # Returns zero below the ground plane when image theory is in
    # effect (cosθ < 0), and at the axial null (sinθ = 0).
    η0 = 376.730313668
    rh = Vec3(r_hat) / max(norm(Vec3(r_hat)), 1e-30)
    ax = mono.axis
    cosθ = clamp(dot(rh, ax), -1.0, 1.0)

    mono.include_image && cosθ < 0 &&
        return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)

    sinθ = sqrt(max(0.0, 1.0 - cosθ^2))
    sinθ > 1e-12 || return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    θ_hat = (cosθ * rh - Vec3(ax)) / sinθ

    I_0 = -1im * 2π * mono.amplitude / η0
    h = mono.height
    λ = 2π / k
    z_lo = mono.include_image ? -h : 0.0
    span = h - z_lo
    N = max(64, 2 * Int(ceil(50.0 * span / λ)))
    iseven(N) || (N += 1)
    dz = span / N

    integ = 0.0 + 0im
    @inbounds for i in 0:N
        z = z_lo + i * dz
        I_z = I_0 * sin(k * (h - abs(z)))  # abs(z) reduces to z for z ≥ 0
        w = (i == 0 || i == N) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        integ += w * I_z * exp(1im * k * z * cosθ)
    end
    integ *= dz / 3.0

    Eθ_far = 1im * η0 * k * sinθ / (4π) * integ
    phase = exp(1im * k * dot(rh, Vec3(mono.position)))
    return CVec3(Eθ_far * θ_hat) * phase
end

function incident_farfield(dipole::DipoleExcitation, r_hat::Vec3, k::Real)
    # Cross-checked against `dipole_incident_field` in this file (keep the
    # 1/R term, multiply by R·exp(+ikR)):
    #   Electric: E∞(r̂) = +k²/(4πε₀) · (r̂×p)×r̂ = +k²/(4πε₀) · perp(p)
    #   Magnetic: E∞(r̂) = +iη₀·k²/(4π) · (m × r̂)
    rh = Vec3(r_hat) / max(norm(Vec3(r_hat)), 1e-30)
    ϵ0 = 8.854187817e-12; μ0 = 4π * 1e-7
    η0 = sqrt(μ0 / ϵ0)
    if dipole.type == :electric
        perp = dipole.moment - rh * dot(rh, dipole.moment)
        E_far = (k^2 / (4π * ϵ0)) * perp
    elseif dipole.type == :magnetic
        E_far = 1im * (η0 * k^2 / (4π)) * cross(dipole.moment, rh)
    else
        error("Dipole type must be :electric or :magnetic, got $(dipole.type).")
    end
    phase = exp(1im * k * dot(rh, dipole.position))
    return CVec3(E_far) * phase
end

function incident_farfield(loop::LoopExcitation, r_hat::Vec3, k::Real)
    # A small circular current loop with current I and radius a radiates
    # in the far field as an equivalent magnetic dipole with moment
    # m = I·π·a²·n̂ placed at the loop centre.
    m = loop.current * π * loop.radius^2 * loop.normal
    dip = DipoleExcitation(loop.center, m, loop.normal, :magnetic, loop.frequency)
    return incident_farfield(dip, r_hat, k)
end

function incident_farfield(pat::PatternFeedExcitation, r_hat::Vec3, k::Real)
    # `PatternFeedExcitation` is already a tabulated far-field:
    # E_inc(r·r̂) ~ (Fθ(θ,ϕ)·θ̂ + Fϕ(θ,ϕ)·ϕ̂) · exp(-ikR)/R
    # (with R = |r − phase_center|). At the far field, R ≈ r − r̂·pc,
    # so r·E·exp(+ikr) = E_far(r̂) · exp(+ik·r̂·phase_center).
    pat.frequency > 0 || error("Pattern feed frequency must be positive.")
    rh = Vec3(r_hat) / max(norm(Vec3(r_hat)), 1e-30)

    θ = acos(clamp(rh[3], -1.0, 1.0))
    ϕ = atan(rh[2], rh[1])
    if ϕ < 0
        ϕ += 2π
    end
    _, eθ, eϕ = _spherical_basis(θ, ϕ)

    Fθ = _pattern_interp(pat, pat.Ftheta, θ, ϕ)
    Fϕ = _pattern_interp(pat, pat.Fphi, θ, ϕ)
    if pat.convention == :exp_minus_iwt
        Fθ = conj(Fθ)
        Fϕ = conj(Fϕ)
    end

    E_far = Fθ * eθ + Fϕ * eϕ
    phase = exp(1im * k * dot(rh, pat.phase_center))
    return CVec3(E_far) * phase
end

function incident_farfield(multi::MultiExcitation, r_hat::Vec3, k::Real)
    length(multi.excitations) == length(multi.weights) ||
        error("MultiExcitation has mismatched excitation/weight lengths.")
    E = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    for (exc, w) in zip(multi.excitations, multi.weights)
        E = E + w * incident_farfield(exc, r_hat, k)
    end
    return E
end
