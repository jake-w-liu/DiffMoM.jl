# PeriodicMetrics.jl — Scattering metrics for periodic metasurfaces
#
# Computes reflection coefficients, Floquet mode efficiencies,
# and periodic RCS for metasurface unit cells.
#
# For a metasurface illuminated by a plane wave at angle (θ_inc, φ_inc),
# the reflected field is decomposed into Floquet harmonics (m,n) with
# propagation directions determined by grating equations:
#   kx_mn = kx_inc + 2πm/dx
#   ky_mn = ky_inc + 2πn/dy
#   kz_mn = sqrt(k² - kx_mn² - ky_mn²)  (propagating if real)
#
# Reflection coefficient normalization (Fix 1.3):
#   R_mn = -(η₀ k)/(2 κz_mn E₀) × (ê_pol · J̃_mn)
#   where J̃_mn = (1/A) ∫_cell J(r') exp(i κ_t·r') dS'
#   For PEC at normal incidence: R₀₀ = -1 (verified).

export floquet_modes, reflection_coefficients, reflection_coefficient_vectors
export reflected_power_fractions, transmission_coefficients, specular_rcs_objective
export power_balance
export FloquetMode

function _specular_objective_polarization(grid::SphGrid, polarization)
    if polarization isa Symbol
        if polarization in (:x, :theta, :tm)
            return pol_linear_x(grid)
        elseif polarization in (:y, :phi, :te)
            return pol_linear_y(grid)
        else
            throw(ArgumentError(
                "Unsupported polarization=$polarization " *
                "(expected :x/:theta/:tm, :y/:phi/:te, or a (3, NΩ) polarization matrix)."
            ))
        end
    elseif polarization isa AbstractMatrix
        NΩ = length(grid.w)
        size(polarization) == (3, NΩ) ||
            throw(DimensionMismatch(
                "Custom polarization matrix must have size (3, $NΩ); got $(size(polarization))."
            ))
        return ComplexF64.(Matrix(polarization))
    else
        throw(ArgumentError(
            "Unsupported polarization input of type $(typeof(polarization)). " *
            "Pass a supported Symbol or a (3, NΩ) polarization matrix."
        ))
    end
end

function _assert_coplanar_periodic_metrics_mesh(mesh::TriMesh; atol::Float64=1e-12)
    zvals = @view mesh.xyz[3, :]
    zmin = minimum(zvals)
    zmax = maximum(zvals)
    if abs(zmax - zmin) > atol
        throw(ArgumentError(
            "reflection_coefficients currently supports coplanar unit-cell meshes only " *
            "(max z spread <= $(atol)). Got z spread=$(abs(zmax - zmin))."
        ))
    end
end

"""
    FloquetMode

A single Floquet diffraction order (m, n).
"""
struct FloquetMode
    m::Int                      # x-order
    n::Int                      # y-order
    kx::Float64                 # kx of this mode
    ky::Float64                 # ky of this mode
    kz::ComplexF64              # kz (real = propagating, imaginary = evanescent)
    propagating::Bool           # true if kz is real (mode carries power)
    theta_r::Float64            # reflection angle theta (NaN if evanescent)
    phi_r::Float64              # reflection angle phi (NaN if evanescent)
end

function _mode_transverse_projection(pol::SVector{3,<:Real}, mode::FloquetMode, k::Real)
    khat = SVector(mode.kx / k, mode.ky / k, real(mode.kz) / k)
    pol_real = SVector(Float64(pol[1]), Float64(pol[2]), Float64(pol[3]))
    pol_mode_raw = pol_real - dot(pol_real, khat) * khat
    pol_mode_norm = norm(pol_mode_raw)
    return pol_mode_norm < 1e-12 ? nothing : pol_mode_raw / pol_mode_norm
end

function _floquet_current_fourier_coefficients(mesh::TriMesh, rwg::RWGData,
                                               I_coeffs::Vector{<:Number},
                                               k::Real, lattice::PeriodicLattice;
                                               quad_order::Int=3,
                                               N_orders::Int=3)
    _assert_coplanar_periodic_metrics_mesh(mesh)
    _assert_boundary_touching_periodic_mesh_requires_bloch(mesh, lattice, rwg)

    modes = floquet_modes(k, lattice; N_orders=N_orders)
    A_cell = lattice.dx * lattice.dy

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    Nt = ntriangles(mesh)
    N = rwg.nedges

    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas = [triangle_area(mesh, t) for t in 1:Nt]

    tri_to_basis = [Int[] for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n)
        push!(tri_to_basis[rwg.tminus[n]], n)
    end

    zero_vec = SVector{3,ComplexF64}(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    J_tildes = fill(zero_vec, length(modes))

    for (mi, mode) in enumerate(modes)
        if !mode.propagating
            continue
        end

        integral = zero_vec
        for t in 1:Nt
            At = areas[t]
            for q in 1:Nq
                rq = quad_pts[t][q]

                J_rq = zero_vec
                for n_idx in tri_to_basis[t]
                    fn = eval_rwg(rwg, n_idx, rq, t)
                    J_rq += I_coeffs[n_idx] * fn
                end

                phase = exp(im * (mode.kx * rq[1] + mode.ky * rq[2]))
                integral += J_rq * phase * wq[q] * (2 * At)
            end
        end

        J_tildes[mi] = integral / A_cell
    end

    return modes, J_tildes
end

"""
    floquet_modes(k, lattice; N_orders=3)

Enumerate all Floquet modes (m, n) for the given lattice and classify
them as propagating or evanescent.

Returns a vector of FloquetMode structs.
"""
function floquet_modes(k::Real, lattice::PeriodicLattice; N_orders::Int=3)
    modes = FloquetMode[]

    for m in -N_orders:N_orders
        for n in -N_orders:N_orders
            kx_mn = lattice.kx_bloch + 2π * m / lattice.dx
            ky_mn = lattice.ky_bloch + 2π * n / lattice.dy
            kt2 = kx_mn^2 + ky_mn^2

            kz2 = k^2 - kt2
            if kz2 > 0
                kz = sqrt(kz2)
                theta_r = acos(clamp(real(kz) / k, -1.0, 1.0))
                phi_r = atan(ky_mn, kx_mn)
                push!(modes, FloquetMode(m, n, kx_mn, ky_mn, kz, true, theta_r, phi_r))
            else
                kz = im * sqrt(-kz2)
                push!(modes, FloquetMode(m, n, kx_mn, ky_mn, kz, false, NaN, NaN))
            end
        end
    end

    return modes
end

"""
    reflection_coefficients(mesh, rwg, I_coeffs, k, lattice; kwargs...)

Compute properly normalized complex reflection coefficients for each
propagating Floquet mode.

The reflection coefficient for mode (m,n) is:
  R_mn = -(η₀ k)/(2 κz_mn E₀) × (ê_pol · J̃_mn)
where J̃_mn = (1/A) ∫_cell J(r') exp(i κ_t·r') dS' is the current
Fourier coefficient, ê_pol is the incident polarization, and E₀ is
the incident field amplitude.

Sanity check: for a PEC plate at normal incidence, R₀₀ = -1.

Returns (modes, R_coeffs) where R_coeffs[i] is the complex reflection
coefficient for modes[i].
"""
function reflection_coefficients(mesh::TriMesh, rwg::RWGData,
                                 I_coeffs::Vector{<:Number},
                                 k::Real, lattice::PeriodicLattice;
                                 quad_order::Int=3, N_orders::Int=3,
                                 E0::Float64=1.0,
                                 pol::SVector{3,Float64}=SVector(1.0, 0.0, 0.0),
                                 eta0::Float64=376.730313668)
    modes, J_tildes = _floquet_current_fourier_coefficients(
        mesh, rwg, I_coeffs, k, lattice; quad_order=quad_order, N_orders=N_orders
    )

    R_coeffs = zeros(ComplexF64, length(modes))

    for (mi, mode) in enumerate(modes)
        if !mode.propagating
            continue
        end

        # Use a mode-transverse co-polar vector obtained by projecting the
        # incident polarization onto the mode's transverse plane.
        #
        # This avoids overestimating mode amplitudes when the global incident
        # polarization has a component parallel to the reflected mode direction.
        pol_mode = _mode_transverse_projection(pol, mode, k)
        if isnothing(pol_mode)
            continue
        end

        # Co-polar reflection coefficient:
        #   R_mn = -(η₀ k)/(2 κz_mn E₀) × (ê_mode · J̃_mn)
        # where ê_mode is transverse to this mode's propagation direction.
        kz_mn = real(mode.kz)
        R_coeffs[mi] = -(eta0 * k) / (2 * kz_mn * E0) * dot(pol_mode, J_tildes[mi])
    end

    return modes, R_coeffs
end

"""
    reflection_coefficient_vectors(mesh, rwg, I_coeffs, k, lattice; kwargs...)

Compute the full mode-transverse reflected electric-field amplitude vector for
each propagating Floquet order. Unlike `reflection_coefficients`, which reports
one scalar co-polar projection per order, this vector form retains both
orthogonal transverse polarizations and is therefore the correct quantity for
total reflected-power budgets.

Returns `(modes, R_vecs)`, where `R_vecs[i]` is a three-component complex vector
normalized by the incident field amplitude.
"""
function reflection_coefficient_vectors(mesh::TriMesh, rwg::RWGData,
                                        I_coeffs::Vector{<:Number},
                                        k::Real, lattice::PeriodicLattice;
                                        quad_order::Int=3, N_orders::Int=3,
                                        E0::Float64=1.0,
                                        eta0::Float64=376.730313668)
    modes, J_tildes = _floquet_current_fourier_coefficients(
        mesh, rwg, I_coeffs, k, lattice; quad_order=quad_order, N_orders=N_orders
    )

    zero_vec = SVector{3,ComplexF64}(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    R_vecs = fill(zero_vec, length(modes))

    for (mi, mode) in enumerate(modes)
        if !mode.propagating
            continue
        end

        kz_mn = real(mode.kz)
        khat = SVector(mode.kx / k, mode.ky / k, kz_mn / k)
        J_transverse = J_tildes[mi] - khat * dot(khat, J_tildes[mi])
        R_vecs[mi] = -(eta0 * k) / (2 * kz_mn * E0) * J_transverse
    end

    return modes, R_vecs
end

"""
    reflected_power_fractions(modes, R_vecs, k)

Return the reflected power fraction carried by each Floquet order from full
vector reflection amplitudes. The total reflected fraction is `sum(p)`.
"""
function reflected_power_fractions(modes::Vector{FloquetMode},
                                   R_vecs::Vector{SVector{3,ComplexF64}},
                                   k::Real)
    length(modes) == length(R_vecs) ||
        throw(DimensionMismatch("modes length ($(length(modes))) != R_vecs length ($(length(R_vecs)))"))

    p = zeros(Float64, length(modes))
    for (i, mode) in enumerate(modes)
        if mode.propagating
            p[i] = real(dot(R_vecs[i], R_vecs[i])) * real(mode.kz) / k
        end
    end
    return p
end

"""
    transmission_coefficients(modes, R_coeffs; incident_order=(0, 0))

Compute transmitted Floquet amplitudes from reflection amplitudes for a free-standing
electric-current sheet model in identical media above and below the sheet.

Exact thin-sheet relation (no branch selection):
- An infinitesimal electric surface current radiates a field whose tangential component is
  equal on both sides of the sheet, so the forward-scattered amplitude equals the
  backward-scattered (reflection) amplitude. Field continuity of the total tangential field
  then gives, for the incident order `(m,n) = incident_order`,
      T₀₀ = 1 + R₀₀
  (incident wave plus forward-scattered = 1 + R). This recovers the physical limits
  exactly: a PEC sheet `R = -1` ⇒ `T = 0`; a transparent sheet `R = 0` ⇒ `T = 1`.
  For a lossless (reactive) sheet it satisfies `|R₀₀|² + |T₀₀|² = 1` identically, so the
  power budget closes without any heuristic branch choice.
- Non-incident propagating orders are purely scattered; by the same forward/backward
  symmetry their transmitted amplitude equals the reflected amplitude (`T_mn = R_mn`).
"""
function transmission_coefficients(modes::Vector{FloquetMode},
                                   R_coeffs::Vector{ComplexF64};
                                   incident_order::Tuple{Int,Int}=(0, 0))
    length(modes) == length(R_coeffs) ||
        throw(DimensionMismatch("modes length ($(length(modes))) != R_coeffs length ($(length(R_coeffs)))"))

    T_coeffs = similar(R_coeffs)
    m_inc, n_inc = incident_order
    for (i, mode) in enumerate(modes)
        if mode.m == m_inc && mode.n == n_inc
            # Exact incident-order transmission for a free-standing current sheet.
            T_coeffs[i] = 1 + R_coeffs[i]
        else
            # Non-incident orders: forward amplitude equals the reflected amplitude.
            T_coeffs[i] = R_coeffs[i]
        end
    end
    return T_coeffs
end

"""
    power_balance(I_coeffs, Z_pen, A_cell, k, modes, R_coeffs;
                  eta0=376.730313668, E0=1.0,
                  transmission=:none, T_coeffs=nothing, incident_order=(0, 0))

Compute the power balance for a periodic metasurface unit cell.

Returns a NamedTuple:
`(P_inc, P_refl, P_abs, P_trans, P_resid, refl_frac, abs_frac, trans_frac, resid_frac)`

- P_inc:   incident power through the unit cell = |E₀|² A / (2η₀)
- P_refl:  reflected power = Σ_mn |R_mn|² (kz_mn/k) P_inc  [propagating modes]
- P_abs:   absorbed by SIMP penalty impedance = ½ Re(I† Z_pen I)
- P_trans: transmitted power (depends on `transmission` mode)
  - `:none`: 0 (legacy behavior)
  - `:closure`: `max(P_inc - P_refl - P_abs, 0)` (conservation-based estimate)
  - `:floquet`: Σ_mn |T_mn|² (kz_mn/k) P_inc using `T_coeffs` or inferred from `R`
- P_resid: residual power = P_inc - P_refl - P_abs - P_trans
- refl_frac: P_refl / P_inc
- abs_frac:  P_abs / P_inc
- trans_frac: P_trans / P_inc
- resid_frac: P_resid / P_inc
"""
function power_balance(I_coeffs::Vector{<:Number},
                       Z_pen::AbstractMatrix,
                       A_cell::Real,
                       k::Real,
                       modes::Vector{FloquetMode},
                       R_coeffs::Vector{ComplexF64};
                       eta0::Float64=376.730313668,
                       E0::Float64=1.0,
                       transmission::Symbol=:none,
                       T_coeffs::Union{Nothing,Vector{ComplexF64}}=nothing,
                       incident_order::Tuple{Int,Int}=(0, 0))
    # Per-mode z-directed Poynting flux ∝ |coef|²·real(kz)/k. The incident power
    # crossing the cell is the z-flux of the incident order, which carries
    # cos(θ_inc) = real(kz_inc)/k. Normalizing the fractions by this (rather than
    # by the unprojected |E0|²A/2η) is required for energy conservation at oblique
    # incidence; at normal incidence kz_inc = k and nothing changes.
    base = abs(E0)^2 * A_cell / (2 * eta0)
    inc_idx = findfirst(mode -> mode.m == incident_order[1] && mode.n == incident_order[2], modes)
    cos_inc = inc_idx === nothing ? 1.0 : real(modes[inc_idx].kz) / k
    P_inc = base * cos_inc

    # Reflected power from Floquet modes (z-directed flux)
    P_refl = 0.0
    for (i, mode) in enumerate(modes)
        if mode.propagating
            P_refl += abs2(R_coeffs[i]) * real(mode.kz) / k
        end
    end
    P_refl *= base

    # Power absorbed by SIMP penalty impedance
    P_abs = 0.5 * real(dot(I_coeffs, Z_pen * I_coeffs))

    # Transmitted power
    P_trans = 0.0
    if transmission == :none
        P_trans = 0.0
    elseif transmission == :closure
        # Conservation-based transmission estimate from unaccounted non-absorbed power.
        P_trans = clamp(P_inc - P_refl - P_abs, 0.0, P_inc)
    elseif transmission == :floquet
        Tc = isnothing(T_coeffs) ?
            transmission_coefficients(modes, R_coeffs; incident_order=incident_order) :
            T_coeffs
        length(Tc) == length(modes) ||
            throw(DimensionMismatch("T_coeffs length ($(length(Tc))) != modes length ($(length(modes)))"))

        for (i, mode) in enumerate(modes)
            if mode.propagating
                P_trans += abs2(Tc[i]) * real(mode.kz) / k
            end
        end
        P_trans *= base
    else
        error("Unknown transmission mode: $transmission (expected :none, :closure, or :floquet)")
    end

    refl_frac = P_refl / P_inc
    abs_frac = P_abs / P_inc
    trans_frac = P_trans / P_inc
    P_resid = P_inc - P_refl - P_abs - P_trans
    resid_frac = P_resid / P_inc

    return (P_inc=P_inc, P_refl=P_refl, P_abs=P_abs, P_trans=P_trans, P_resid=P_resid,
            refl_frac=refl_frac, abs_frac=abs_frac, trans_frac=trans_frac, resid_frac=resid_frac)
end

"""
    specular_rcs_objective(mesh, rwg, grid, k, lattice;
                           quad_order=3, half_angle=π/18, polarization=:x)

Build a Q matrix targeting the specular reflection direction.

For normal incidence: specular = broadside (θ=0).
For oblique incidence: specular direction from Snell's law.

`half_angle` sets the cone half-angle (radians) around the specular direction.
`polarization` selects the projection polarization. Supported symbols:

- `:x`, `:theta`, `:tm` → `pol_linear_x(grid)` (`θ̂` basis)
- `:y`, `:phi`, `:te` → `pol_linear_y(grid)` (`φ̂` basis)

You may also pass a custom polarization matrix of size `(3, NΩ)`.

Returns Q ∈ C^{N×N} such that J = Re(I† Q I) measures cone-integrated
specular scattered power in the chosen polarization.
"""
function specular_rcs_objective(mesh::TriMesh, rwg::RWGData,
                                grid::SphGrid, k::Real,
                                lattice::PeriodicLattice;
                                quad_order::Int=3,
                                half_angle::Float64=π/18,
                                polarization=:x)
    # Specular (0,0) reflected order keeps the incident transverse wavevector and
    # flips only kz, so r̂ = (kx_bloch, ky_bloch, +kz_inc)/k: θ_r = θ_inc and
    # φ_r = φ_inc = atan(ky_bloch, kx_bloch) (no +π — that would point at the
    # mirror azimuth and miss the specular lobe at oblique incidence).
    theta_spec = asin(clamp(sqrt(lattice.kx_bloch^2 + lattice.ky_bloch^2) / k, 0.0, 1.0))
    phi_spec = atan(lattice.ky_bloch, lattice.kx_bloch)

    # Build direction mask for specular cone
    spec_dir = Vec3(sin(theta_spec) * cos(phi_spec),
                    sin(theta_spec) * sin(phi_spec),
                    cos(theta_spec))
    mask = direction_mask(grid, spec_dir; half_angle=half_angle)

    # Build radiation vectors and Q matrix
    G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=quad_order)
    pol = _specular_objective_polarization(grid, polarization)
    Q = build_Q(G_mat, grid, pol; mask=mask)

    return Q
end
