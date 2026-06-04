# PhysicalOptics.jl — Physical Optics (PO) high-frequency RCS solver
#
# Computes scattered far-field and bistatic RCS using the PO approximation:
#   Illuminated face: J_s = 2(n̂ × H_inc)
#   Shadow face:      J_s = 0
#
# Works directly on triangle meshes — no RWG basis functions needed.
# Uses analytical phase integration (exact for linear phase over triangles),
# matching the POFacets 4.5 algorithm (Jenn, NPS).

export POResult, solve_po

"""
    POResult

Result from the PO solver containing far-field, surface currents,
illumination mask, and problem metadata.
"""
struct POResult
    E_ff::Matrix{ComplexF64}     # (3, NΩ) scattered far-field
    J_s::Vector{CVec3}           # (Nt,) PO surface current per triangle centroid
    illuminated::BitVector       # (Nt,) which triangles are illuminated
    grid::SphGrid
    freq_hz::Float64
    k::Float64
end

# ─── Analytical phase integral helpers (POFacets G.m / fact.m) ───

"""
Recursive helper G(n, w) for Taylor-series phase integral (POFacets G.m).
"""
function _po_G(n::Int, w)
    jw = 1im * w
    g = (exp(jw) - 1) / jw
    for m in 1:n
        g_prev = g
        g = (exp(jw) - m * g_prev) / jw
    end
    return g
end

"""
    _phase_integral_analytical(k, delta_k, v1, v2, v3, Area; Lt=1e-5, Nt=5)

Compute ∫_triangle exp(jk δk·r') dS' analytically using the POFacets formula.

Uses vertex-based decomposition with Taylor-series special cases for
small phase differences (avoids division by zero when δk is nearly
perpendicular to an edge).
"""
function _phase_integral_analytical(k::Float64, delta_k::Vec3,
                                    p1::Vec3, p2::Vec3, p3::Vec3,
                                    Area::Float64;
                                    Lt::Float64=1e-5, Nt::Int=5)
    # Phase at vertices relative to v3
    Dp = k * dot(Vec3(p1 - p3), delta_k)
    Dq = k * dot(Vec3(p2 - p3), delta_k)
    Do = k * dot(p3, delta_k)

    DD = Dq - Dp
    expDo = exp(1im * Do)
    expDp = exp(1im * Dp)
    expDq = exp(1im * Dq)

    Ic::ComplexF64 = zero(ComplexF64)

    if abs(Dp) < Lt && abs(Dq) >= Lt
        # Special case 1: Dp small, Dq not
        sic = zero(ComplexF64)
        for n in 0:Nt
            sic += (1im * Dp)^n / factorial(n) *
                   (-1.0 / (n + 1) + expDq * _po_G(n, ComplexF64(-Dq)))
        end
        Ic = sic * 2 * Area * expDo / (1im * Dq)
    elseif abs(Dp) < Lt && abs(Dq) < Lt
        # Special case 2: both small
        sic = zero(ComplexF64)
        for n in 0:Nt
            for nn in 0:Nt
                sic += (1im * Dp)^n * (1im * Dq)^nn / factorial(nn + n + 2)
            end
        end
        Ic = sic * 2 * Area * expDo
    elseif abs(Dp) >= Lt && abs(Dq) < Lt
        # Special case 3: Dq small, Dp not
        sic = zero(ComplexF64)
        for n in 0:Nt
            sic += (1im * Dq)^n / factorial(n) *
                   _po_G(n + 1, ComplexF64(-Dp)) / (n + 1)
        end
        Ic = sic * 2 * Area * expDo * expDp
    elseif abs(Dp) >= Lt && abs(Dq) >= Lt && abs(DD) < Lt
        # Special case 4: DD small
        sic = zero(ComplexF64)
        for n in 0:Nt
            sic += (1im * DD)^n / factorial(n) *
                   (-_po_G(n, ComplexF64(Dq)) + expDq / (n + 1))
        end
        Ic = sic * 2 * Area * expDo / (1im * Dq)
    else
        # General case: all phase differences large enough
        Ic = 2 * Area * expDo *
             (expDp / (Dp * DD) - expDq / (Dq * DD) - 1.0 / (Dp * Dq))
    end

    return Ic
end

"""
    solve_po(mesh, freq_hz, excitation; grid, c0=299792458.0, eta0=376.730313668)

Compute the Physical Optics scattered far-field for a PEC body.

# Arguments
- `mesh::TriMesh`: triangle mesh of the scatterer
- `freq_hz`: frequency in Hz
- `excitation::PlaneWaveExcitation`: incident plane wave
- `grid::SphGrid`: spherical observation grid (default: 36×72)
- `c0, eta0`: physical constants

# Returns
`POResult` with far-field `E_ff`, surface currents `J_s`, illumination mask, etc.

# Physics
For a plane wave E_inc = E₀ pol exp(-jk·r), the PO surface current on
illuminated faces is J_s = 2(n̂ × H_inc), where H_inc = (k̂ × E_inc)/η₀.
The scattered far-field is computed using the analytical phase integral
over each triangle (exact for the linear phase exp(jk δk·r')).
"""
function solve_po(mesh::TriMesh, freq_hz::Real, excitation::PlaneWaveExcitation;
                  grid::SphGrid=make_sph_grid(36, 72),
                  c0::Float64=299792458.0,
                  eta0::Float64=376.730313668)
    Nt = ntriangles(mesh)
    NΩ = length(grid.w)

    k = 2π * freq_hz / c0
    k_vec = excitation.k_vec
    E0 = excitation.E0
    pol = excitation.pol

    # Incident direction unit vector (propagation direction)
    k_hat = Vec3(k_vec / norm(k_vec))

    # H_inc polarization direction: (k̂ × pol)
    h_pol = cross(k_hat, Vec3(pol))

    # ─── Phase 1: Determine illumination and PO surface currents ───
    illuminated = falses(Nt)
    J_s = Vector{CVec3}(undef, Nt)
    V_t = Vector{Vec3}(undef, Nt)

    # Precompute per-triangle vertices and areas
    tri_v1 = Vector{Vec3}(undef, Nt)
    tri_v2 = Vector{Vec3}(undef, Nt)
    tri_v3 = Vector{Vec3}(undef, Nt)
    tri_area = Vector{Float64}(undef, Nt)

    for t in 1:Nt
        v1 = _mesh_vertex(mesh, mesh.tri[1, t])
        v2 = _mesh_vertex(mesh, mesh.tri[2, t])
        v3 = _mesh_vertex(mesh, mesh.tri[3, t])
        tri_v1[t] = v1
        tri_v2[t] = v2
        tri_v3[t] = v3
        tri_area[t] = 0.5 * norm(cross(v2 - v1, v3 - v1))

        n_hat = triangle_normal(mesh, t)

        # Illumination: k̂ · n̂ ≤ 0 means wave propagates against normal
        if dot(k_hat, n_hat) <= 0.0
            illuminated[t] = true
            V_t[t] = cross(n_hat, h_pol)  # n̂ × (k̂ × pol)

            rc = triangle_center(mesh, t)
            phase = exp(-1im * dot(k_vec, rc))
            J_s[t] = CVec3(complex.(2.0 * E0 / eta0 * V_t[t] * phase))
        else
            illuminated[t] = false
            V_t[t] = Vec3(0.0, 0.0, 0.0)
            J_s[t] = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
        end
    end

    # ─── Phase 2: Far-field integration ───
    # E_scat(r̂) = (-jk E₀ / 2π) Σ_t [r̂ × (r̂ × V_t)] × I_t
    # where I_t = ∫_t exp(jk(r̂ - k̂)·r') dS'  (analytical phase integral)

    prefactor = -1im * k * E0 / (2π)

    # Precompute rhat Vec3 (avoids column-slice allocation in hot loop)
    rhat_vec = Vector{Vec3}(undef, NΩ)
    @inbounds for q in 1:NΩ
        rhat_vec[q] = Vec3(grid.rhat[1, q], grid.rhat[2, q], grid.rhat[3, q])
    end

    E_ff = zeros(ComplexF64, 3, NΩ)

    for q in 1:NΩ
        r_hat = rhat_vec[q]
        # Phase: exp(jk(r̂ - k̂_prop)·r')
        delta_k = r_hat - k_hat

        E_q = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)

        for t in 1:Nt
            !illuminated[t] && continue

            # Analytical phase integral over triangle
            I_t = _phase_integral_analytical(k, delta_k,
                      tri_v1[t], tri_v2[t], tri_v3[t], tri_area[t])

            # r̂ × (r̂ × V_t) = (r̂·V_t)r̂ - V_t
            Vt = V_t[t]
            proj = r_hat * dot(r_hat, Vt) - Vt

            E_q += CVec3(complex.(prefactor * proj * I_t))
        end

        E_ff[1, q] = E_q[1]
        E_ff[2, q] = E_q[2]
        E_ff[3, q] = E_q[3]
    end

    return POResult(E_ff, J_s, illuminated, grid, Float64(freq_hz), k)
end
