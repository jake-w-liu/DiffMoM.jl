# PTD.jl — Physical Theory of Diffraction (PO + Ufimtsev fringe correction)
#
# Adds edge-diffraction corrections to the PO solution. The fringe
# The fringe = exact_edge - PO_edge uses cot(ψ)-csc(ψ) = -tan(ψ/2),
# which is finite everywhere (no shadow/reflection boundary singularities).
#
# Convention: exp(+jωt) time dependence throughout.
#
# Reference: Ufimtsev, "Fundamentals of the Physical Theory of Diffraction"

export DiffractionEdge, extract_diffraction_edges, PTDResult, solve_ptd

# ═══════════════════════════════════════════════════════════════════
# DiffractionEdge struct
# ═══════════════════════════════════════════════════════════════════

"""
    DiffractionEdge

A diffraction edge extracted from a triangle mesh, storing the local wedge
geometry needed for PTD computations.

For interior edges, `face_o` and `face_n` are the two adjacent faces and
`alpha` is the exterior wedge angle. For boundary edges, `face_n == 0` and
`alpha == 2π` (half-plane).
"""
struct DiffractionEdge
    v1::Int                    # vertex index 1
    v2::Int                    # vertex index 2
    p1::Vec3                   # vertex 1 position
    p2::Vec3                   # vertex 2 position
    center::Vec3               # edge midpoint
    tangent::Vec3              # unit tangent (p2-p1)/|p2-p1|
    length::Float64            # edge length
    face_o::Int                # "outer" face index (lower face index)
    face_n::Int                # "inner" face index (0 for boundary)
    normal_o::Vec3             # unit normal of face_o
    normal_n::Vec3             # unit normal of face_n (zero for boundary)
    alpha::Float64             # exterior wedge angle (radians), in (0, 2π]
    uo::Vec3                   # outward unit vector in o-face plane, perp to tangent
end

# ═══════════════════════════════════════════════════════════════════
# PTDResult struct
# ═══════════════════════════════════════════════════════════════════

"""
    PTDResult

Result from the PTD solver containing combined PO+PTD far-field,
individual components for diagnostics, and diffraction edge data.
"""
struct PTDResult
    E_ff::Matrix{ComplexF64}       # (3, NΩ) combined PO+PTD far-field
    E_ff_po::Matrix{ComplexF64}    # (3, NΩ) PO-only far-field
    E_ff_ptd::Matrix{ComplexF64}   # (3, NΩ) PTD edge correction only
    J_s::Vector{CVec3}             # PO surface currents
    illuminated::BitVector         # PO illumination mask
    edges::Vector{DiffractionEdge} # diffraction edges found
    grid::SphGrid
    freq_hz::Float64
    k::Float64
end

# ═══════════════════════════════════════════════════════════════════
# Edge extraction (unchanged — working correctly)
# ═══════════════════════════════════════════════════════════════════

@inline function _safe_uo(t::Vec3, n::Vec3)
    u = cross(t, n)
    nu = norm(u)
    if nu <= 1e-12
        ref = abs(t[1]) < 0.9 ? Vec3(1.0, 0.0, 0.0) : Vec3(0.0, 1.0, 0.0)
        u = cross(t, ref)
        nu = norm(u)
        nu <= 1e-12 && return Vec3(0.0, 0.0, 1.0)
    end
    return u / nu
end

"""
    extract_diffraction_edges(mesh; min_dihedral_deg=5.0, include_boundary=true)

Extract diffraction-feature edges from a triangle mesh.
Interior edges with dihedral angle above `min_dihedral_deg` are kept.
Boundary edges (single adjacent face) are treated as half-planes (α = 2π).
"""
function extract_diffraction_edges(mesh::TriMesh;
                                    min_dihedral_deg::Float64=5.0,
                                    include_boundary::Bool=true)
    min_dihedral = deg2rad(min_dihedral_deg)
    Nt = ntriangles(mesh)

    edgemap = Dict{Tuple{Int,Int}, Vector{NTuple{3,Int}}}()
    for t in 1:Nt
        i1, i2, i3 = mesh.tri[1, t], mesh.tri[2, t], mesh.tri[3, t]
        for (a, b) in ((i1, i2), (i2, i3), (i3, i1))
            key = a < b ? (a, b) : (b, a)
            push!(get!(edgemap, key, NTuple{3,Int}[]), (t, a, b))
        end
    end

    out = DiffractionEdge[]
    for recs in values(edgemap)
        if length(recs) == 1
            include_boundary || continue
            rec = recs[1]
            fo = rec[1]; va, vb = rec[2], rec[3]
            p1 = _mesh_vertex(mesh, va); p2 = _mesh_vertex(mesh, vb)
            e = p2 - p1; le = norm(e)
            le > 1e-14 || continue
            t = e / le; c = (p1 + p2) / 2.0
            no = triangle_normal(mesh, fo)
            nn = Vec3(0.0, 0.0, 0.0)
            uo = _safe_uo(t, no)
            push!(out, DiffractionEdge(va, vb, p1, p2, c, t, le,
                                        fo, 0, no, nn, 2π, uo))
            continue
        end
        length(recs) == 2 || continue
        rec1, rec2 = recs[1], recs[2]
        rec_o, rec_n = rec1[1] <= rec2[1] ? (rec1, rec2) : (rec2, rec1)
        fo, fn = rec_o[1], rec_n[1]
        va, vb = rec_o[2], rec_o[3]
        p1 = _mesh_vertex(mesh, va); p2 = _mesh_vertex(mesh, vb)
        e = p2 - p1; le = norm(e)
        le > 1e-14 || continue
        t = e / le; c = (p1 + p2) / 2.0
        no = triangle_normal(mesh, fo); nn = triangle_normal(mesh, fn)
        y = dot(t, cross(no, nn))
        x = clamp(dot(no, nn), -1.0, 1.0)
        δ = atan(y, x)
        abs(δ) >= min_dihedral || continue
        α = π + δ
        α <= 0.0 && (α += 2π); α > 2π && (α -= 2π)
        (α > 1e-10 && α <= 2π) || continue
        uo = _safe_uo(t, no)
        un = _safe_uo(t, nn)
        αchk = mod(atan(dot(t, cross(uo, un)), dot(uo, un)), 2π)
        if abs(αchk - α) > 1e-6
            uo = -uo
        end
        push!(out, DiffractionEdge(va, vb, p1, p2, c, t, le,
                                    fo, fn, no, nn, α, uo))
    end
    return out
end

# ═══════════════════════════════════════════════════════════════════
# PTD physics: azimuth, fringe coefficients, polarization basis
# ═══════════════════════════════════════════════════════════════════

"""
    _ptd_edge_azimuth(dir, edge) -> Union{Float64, Nothing}

Compute the azimuthal angle of direction `dir` in the edge-fixed coordinate
system. The angle is measured from the o-face normal `n̂_o` (φ=0), with
`t_o = ê × n̂_o` at φ=π/2. Returns `nothing` if `dir` is nearly parallel
to the edge tangent.
"""
@inline function _ptd_edge_azimuth(dir::Vec3, edge::DiffractionEdge)
    # Project direction perpendicular to edge tangent
    d_perp = dir - dot(dir, edge.tangent) * edge.tangent
    nd = norm(d_perp)
    nd > 1e-12 || return nothing
    d_hat = d_perp / nd
    # Measure angle: x-axis = t_o = ê × n̂_o (= uo), y-axis = n̂_o
    # This matches the edge-fixed frame where φ=0 is along the outward
    # tangential direction in the o-face plane.
    t_o = edge.uo
    return mod(atan(dot(d_hat, edge.normal_o), dot(d_hat, t_o)) + 2π, 2π)
end

"""
    _stable_XminusTan(u, n) -> Float64

Compute X - (1/2)tan(u) stably using a common-denominator formula.

Both X = (sin(π/n)/n) / [cos(π/n) - cos(2u/n)] and (1/2)tan(u) diverge
at the shadow boundary (where cos(π/n) = cos(2u/n)), but their difference
is finite. The common-denominator form avoids catastrophic cancellation.
"""
@inline function _stable_XminusTan(u::Float64, n::Float64; cap::Float64=10.0)
    cos_u = cos(u)
    sin_u = sin(u)
    sin_pi_n = sin(π / n)
    cos_pi_n = cos(π / n)
    D = cos_pi_n - cos(2u / n)

    denom = 2n * cos_u * D
    if abs(denom) < 1e-12
        # L'Hôpital limit: at cos(u) = 0 AND D = 0 → limit is 0
        # at cos(u) = 0 but D ≠ 0 → 1/(2cos(u)) diverges → clamp
        return abs(cos_u) < 1e-6 && abs(D) < 1e-6 ? 0.0 :
               clamp(-(1.0 + sin_u) / max(abs(2cos_u), 1e-10) * sign(cos_u), -cap, cap)
    end

    num = 2cos_u * sin_pi_n - n * sin_u * D
    return clamp(num / denom, -cap, cap)
end

"""
    _stable_YplusTan(v, n, sign_Y, sign_tan) -> Float64

Compute sign_Y * Y + sign_tan * (1/2)tan(v) stably.

Y = (sin(π/n)/n) / [cos(π/n) - cos(2v/n)]. Near the reflection boundary
where both Y and tan diverge, the combined expression stays finite.
"""
@inline function _stable_YplusTan(v::Float64, n::Float64, sign_Y::Int, sign_tan::Int;
                                    cap::Float64=10.0)
    cos_v = cos(v)
    sin_v = sin(v)
    sin_pi_n = sin(π / n)
    cos_pi_n = cos(π / n)
    D = cos_pi_n - cos(2v / n)

    denom = 2n * cos_v * D
    if abs(denom) < 1e-12
        return abs(cos_v) < 1e-6 && abs(D) < 1e-6 ? 0.0 :
               clamp(sign_tan * sin_v / max(abs(2cos_v), 1e-10) * sign(cos_v), -cap, cap)
    end

    num = sign_Y * 2cos_v * sin_pi_n + sign_tan * n * sin_v * D
    return clamp(num / denom, -cap, cap)
end

"""
    _ptd_fringe_fg(n, delta_s, delta_i, gamma) -> (f, g)

Compute the real-valued PTD fringe coefficients f and g for a PEC wedge,
following Sáez de Adana et al., eqs. 4.131-4.136.

Uses the bottom-side illuminated formula (eq 4.133-4.134) which matches
the azimuth convention where δⁱ is computed from k̂.

Uses numerically stable combined computation of (X - tan) and (Y + tan)
to avoid catastrophic cancellation at shadow/reflection boundaries.
"""
function _ptd_fringe_fg(n::Float64, delta_s::Float64, delta_i::Float64,
                         gamma::Float64)
    u = 0.5 * (delta_s - delta_i)
    v = 0.5 * (delta_s + delta_i)
    # A = X - (1/2)tan(u)  [stable, bounded at shadow boundary]
    A = _stable_XminusTan(u, n)

    # Bottom-side formula (eq 4.133-4.134):
    # f = X - Y - 1/2 tan(u) - 1/2 tan(γ-v)
    # g = X + Y - 1/2 tan(u) + 1/2 tan(γ-v)
    #
    # For n=2: tan(2π-v) = -tan(v), so -1/2 tan(γ-v) = +1/2 tan(v).
    # Thus: -Y - 1/2 tan(γ-v) = -Y + 1/2 tan(v) → use _stable_YplusTan(v, n, -1, +1)
    # This is CRITICAL: Y and tan(γ-v) BOTH diverge at the reflection boundary
    # (v = γ/2 for n=2), and their combination must be computed stably.
    B_bot = _stable_YplusTan(v, n, -1, +1)   # -Y + 1/2 tan(v) = -Y - 1/2 tan(γ-v) for n=2
    C_bot = _stable_YplusTan(v, n, +1, -1)   # +Y - 1/2 tan(v) = +Y + 1/2 tan(γ-v) for n=2

    f = A + B_bot
    g = A + C_bot

    return (f, g)
end

"""
    _ptd_beta_phi_basis(ray_hat, tangent) -> Union{Tuple{Vec3,Vec3}, Nothing}

Compute edge-fixed polarization basis vectors (β̂, φ̂) for a ray direction,
following Balanis Eq. 3.63-3.66.

Returns `(β̂, φ̂)` or `nothing` if the ray is nearly parallel to the edge.
"""
@inline function _ptd_beta_phi_basis(ray_hat::Vec3, tangent::Vec3)
    v = -cross(tangent, ray_hat)
    nv = norm(v)
    nv > 1e-12 || return nothing
    φ̂ = v / nv
    β̂v = cross(φ̂, ray_hat)
    nβ = norm(β̂v)
    nβ > 1e-12 || return nothing
    return (β̂v / nβ, φ̂)
end

# ═══════════════════════════════════════════════════════════════════
# Main PTD solver
# ═══════════════════════════════════════════════════════════════════

"""
    solve_ptd(mesh, freq_hz, excitation; grid, c0, eta0, min_dihedral_deg, include_boundary)

Compute the PO+PTD scattered far-field for a PEC body.

Calls `solve_po` for the PO contribution, then adds PTD fringe corrections
from diffraction edges using the Sáez de Adana et al. formulation
(eqs 4.131-4.146): fringe = exact_edge - PO_edge.

# Returns
`PTDResult` with combined PO+PTD far-field, individual components, and edge data.
"""
function solve_ptd(mesh::TriMesh, freq_hz::Real, excitation::PlaneWaveExcitation;
                   grid::SphGrid=make_sph_grid(36, 72),
                   c0::Float64=299792458.0,
                   eta0::Float64=376.730313668,
                   min_dihedral_deg::Float64=5.0,
                   include_boundary::Bool=true)
    # ── Phase 1: PO solution ──
    po = solve_po(mesh, freq_hz, excitation; grid=grid, c0=c0, eta0=eta0)

    # ── Phase 2: Extract diffraction edges ──
    edges = extract_diffraction_edges(mesh;
                min_dihedral_deg=min_dihedral_deg,
                include_boundary=include_boundary)

    k = po.k
    NΩ = length(grid.w)

    k_vec = excitation.k_vec
    k_hat = Vec3(k_vec / norm(k_vec))   # propagation direction
    E0    = excitation.E0
    pol   = Vec3(excitation.pol)

    # ── Phase 3: PTD fringe far-field (eqs 4.145-4.146) ──
    #
    # E_θ = -(1/(2π)) × [-t̂·Ēⁱ/sin²β × f - t̂·(k̂ᵢ×Ēⁱ)/sin²β × g]
    #        × L × sinc_phase
    # E_φ = -(1/(2π)) × [-t̂·Ēⁱ/sin²β × f + t̂·(k̂ᵢ×Ēⁱ)/sin²β × g]
    #        × L × sinc_phase
    #
    # where f, g are real-valued PTD fringe coefficients (eqs 4.131-4.136)
    # and θ̂, φ̂ are the far-field polarization basis vectors.

    E_ff_ptd = zeros(ComplexF64, 3, NΩ)

    prefactor = -1.0 / (2π)

    for q in 1:NΩ
        r_hat = Vec3(grid.rhat[:, q])

        Ex = complex(0.0)
        Ey = complex(0.0)
        Ez = complex(0.0)

        for edge in edges
            ê = edge.tangent
            L = edge.length
            Q₀ = edge.center
            γ = edge.alpha   # exterior wedge angle

            # ── Cone angle (sin²β) ──
            sin_beta = norm(cross(ê, k_hat))
            sin_beta_s = norm(cross(ê, r_hat))
            sin_beta > 1e-4 || continue
            sin_beta_s > 1e-4 || continue
            sin2_beta = sin_beta^2

            # ── Edge azimuth angles ──
            delta_i = _ptd_edge_azimuth(k_hat, edge)
            delta_s = _ptd_edge_azimuth(r_hat, edge)
            isnothing(delta_i) && continue
            isnothing(delta_s) && continue

            # ── PTD fringe coefficients (real-valued, eq 4.137-4.138) ──
            n = γ / π
            f_ptd, g_ptd = _ptd_fringe_fg(n, delta_s, delta_i, γ)

            # ── Incident field at edge midpoint ──
            E_inc_Q0 = pol * E0 * exp(-1im * k * dot(k_hat, Q₀))

            # Field projections onto edge tangent (eqs 4.145-4.146):
            # t̂·Ēⁱ = electric field component along edge
            # t̂·(k̂ᵢ×Ēⁱ) = (k̂×E) component along edge ∝ H tangential
            tE = dot(ê, E_inc_Q0)                              # t̂·Ēⁱ
            tH = dot(ê, cross(k_hat, E_inc_Q0))                # t̂·(k̂ᵢ×Ēⁱ)

            # ── Scattered field ──
            # Edge-fixed basis: ê_⊥ (soft/electric) and r̂×ê (hard/magnetic)
            ê_perp = ê - dot(ê, r_hat) * r_hat
            r_cross_e = cross(r_hat, ê)

            E_soft = prefactor / sin2_beta * (-f_ptd * tE) * ê_perp
            E_hard = prefactor / sin2_beta * (-g_ptd * tH) * r_cross_e

            # ── Edge line integral (sinc × phase) ──
            delta_k = r_hat - k_hat
            q_edge = dot(delta_k, ê)
            kqL_half = k * q_edge * L / 2.0
            sinc_val = abs(kqL_half) > 1e-15 ? sin(kqL_half) / kqL_half : 1.0
            phase = exp(1im * k * dot(delta_k, Q₀))

            line_integral = L * sinc_val * phase

            # ── Assemble scattered vector ──
            E_vec = (E_soft + E_hard) * line_integral

            Ex += E_vec[1]
            Ey += E_vec[2]
            Ez += E_vec[3]
        end

        E_ff_ptd[1, q] = Ex
        E_ff_ptd[2, q] = Ey
        E_ff_ptd[3, q] = Ez
    end

    # ── Phase 4: Combine PO + PTD ──
    E_ff_combined = po.E_ff + E_ff_ptd

    return PTDResult(E_ff_combined, po.E_ff, E_ff_ptd,
                     po.J_s, po.illuminated, edges,
                     grid, Float64(freq_hz), k)
end
