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
    face_o::Int                # canonical first wedge face
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

const _PTD_EXACT_PRECISION = 8704
const _MAX_PTD_EXACT_DIRECTION_VALUES = 12_288
const _DEFAULT_MAX_PTD_WORK_BYTES = 512 * 1024 * 1024

function _ptd_additional_work_bytes(edge_count::Int, direction_count::Int)
    total = BigInt(sizeof(DiffractionEdge)) * edge_count
    # PTD-only and combined far fields plus cached observation directions.
    total += BigInt(6 * sizeof(ComplexF64) + sizeof(Vec3)) * direction_count
    total <= typemax(Int) ||
        throw(ArgumentError("PTD raw-workspace estimate overflows Int"))
    return Int(total)
end

function _preflight_ptd_work(
        Nt::Int,
        edge_count::Int,
        direction_count::Int,
        max_work_bytes::Integer)
    limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    required = try
        Base.Checked.checked_add(
            _po_work_bytes(Nt, direction_count),
            _ptd_additional_work_bytes(edge_count, direction_count))
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("PO+PTD raw-workspace estimate overflows Int"))
    end
    required <= limit ||
        throw(ArgumentError(
            "PO+PTD output and workspace requires $required raw bytes, " *
            "exceeding max_work_bytes=$limit"))
    return required, limit
end

@inline function _ptd_register_exact_direction!(
        exact_totals::Dict{Int,Vector{Complex{BigFloat}}},
        direction::Int,
        current_count::Int)
    haskey(exact_totals, direction) && return current_count
    next_count = Base.checked_add(current_count, 1)
    next_count <= _MAX_PTD_EXACT_DIRECTION_VALUES ÷ 3 ||
        throw(ArgumentError(
            "solve_ptd exact-direction accumulation exceeds the " *
            "$_MAX_PTD_EXACT_DIRECTION_VALUES-value resource limit"))
    return next_count
end

@noinline function _ptd_incident_components_exact(
    amplitude::Float64,
    polarization::Vec3,
    incident_direction::Vec3,
    tangent::Vec3,
)
    return setprecision(BigFloat, _PTD_EXACT_PRECISION) do
        incident = ntuple(component ->
            BigFloat(amplitude) * BigFloat(polarization[component]), 3)
        tangent_electric = sum(
            BigFloat(tangent[component]) * incident[component]
            for component in 1:3)
        magnetic = (
            BigFloat(incident_direction[2]) * incident[3] -
                BigFloat(incident_direction[3]) * incident[2],
            BigFloat(incident_direction[3]) * incident[1] -
                BigFloat(incident_direction[1]) * incident[3],
            BigFloat(incident_direction[1]) * incident[2] -
                BigFloat(incident_direction[2]) * incident[1],
        )
        tangent_magnetic = sum(
            BigFloat(tangent[component]) * magnetic[component]
            for component in 1:3)
        return tangent_electric, tangent_magnetic
    end
end

@noinline function _ptd_accumulate_direction_exact!(
    exact_totals::Dict{Int,Vector{Complex{BigFloat}}},
    field::Matrix{ComplexF64},
    direction::Int,
    prefactor::Float64,
    sin2_beta::Float64,
    f_ptd::Float64,
    g_ptd::Float64,
    tE::Number,
    tH::Number,
    beta_hat::Vec3,
    phi_hat::Vec3,
    length::Float64,
    sinc_value::Float64,
    phase::ComplexF64,
)
    return setprecision(BigFloat, _PTD_EXACT_PRECISION) do
        totals = get!(exact_totals, direction) do
            Complex{BigFloat}[
                Complex{BigFloat}(field[component, direction])
                for component in 1:3]
        end
        common = BigFloat(prefactor) * BigFloat(length) *
                 BigFloat(sinc_value) * Complex{BigFloat}(phase) /
                 BigFloat(sin2_beta)
        @inbounds for component in 1:3
            contribution = common * (
                -BigFloat(f_ptd) * Complex{BigFloat}(tE) *
                    BigFloat(beta_hat[component]) -
                BigFloat(g_ptd) * Complex{BigFloat}(tH) *
                    BigFloat(phi_hat[component]))
            totals[component] += contribution
        end
    end
    return nothing
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
@inline function _validated_min_dihedral(min_dihedral_deg::Float64)
    (isfinite(min_dihedral_deg) &&
     0.0 <= min_dihedral_deg <= 180.0) ||
        throw(ArgumentError(
            "min_dihedral_deg must be finite and in [0, 180], got $min_dihedral_deg"))
    return deg2rad(min_dihedral_deg)
end

function _extract_diffraction_edges_validated(
    mesh::TriMesh,
    min_dihedral::Float64,
    include_boundary::Bool,
)
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
            isfinite(le) && le > 0.0 ||
                error("boundary edge ($va, $vb) has invalid length $le")
            t = e / le; c = _safe_edge_midpoint(p1, p2)
            no = triangle_normal(mesh, fo)
            nn = Vec3(0.0, 0.0, 0.0)
            uo = _safe_uo(t, no)
            push!(out, DiffractionEdge(va, vb, p1, p2, c, t, le,
                                        fo, 0, no, nn, 2π, uo))
            continue
        end
        length(recs) == 2 || continue
        rec1, rec2 = recs[1], recs[2]
        # Each record retains the edge direction induced by its consistently
        # oriented face.  Swapping the two faces also reverses that direction,
        # so dot(t, cross(no, nn)) and the exterior wedge angle are invariant.
        # Select by the face's mesh-independent canonical vertex key: unlike
        # sorting normals or coordinates independently, this keeps the whole
        # oriented face record together under rigid rotations and face-column
        # permutations.
        face1 = sort((mesh.tri[1, rec1[1]],
                      mesh.tri[2, rec1[1]],
                      mesh.tri[3, rec1[1]]))
        face2 = sort((mesh.tri[1, rec2[1]],
                      mesh.tri[2, rec2[1]],
                      mesh.tri[3, rec2[1]]))
        key1 = (face1[1], face1[2], face1[3])
        key2 = (face2[1], face2[2], face2[3])
        rec_o, rec_n = isless(key2, key1) ? (rec2, rec1) : (rec1, rec2)
        no = triangle_normal(mesh, rec_o[1])
        nn = triangle_normal(mesh, rec_n[1])
        fo, fn = rec_o[1], rec_n[1]
        va, vb = rec_o[2], rec_o[3]
        p1 = _mesh_vertex(mesh, va); p2 = _mesh_vertex(mesh, vb)
        e = p2 - p1; le = norm(e)
        isfinite(le) && le > 0.0 ||
            error("interior edge ($va, $vb) has invalid length $le")
        t = e / le; c = _safe_edge_midpoint(p1, p2)
        y = dot(t, cross(no, nn))
        x = clamp(dot(no, nn), -1.0, 1.0)
        δ = atan(y, x)
        abs(δ) > min_dihedral || continue

        # With outward normals and the edge direction inherited from face_o,
        # δ is the signed material fold angle. PTD uses the complementary
        # free-space exterior angle α = nπ.
        α = π - δ
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

function extract_diffraction_edges(mesh::TriMesh;
                                    min_dihedral_deg::Float64=5.0,
                                    include_boundary::Bool=true)
    min_dihedral = _validated_min_dihedral(min_dihedral_deg)
    assert_mesh_quality(
        mesh; allow_boundary=true, require_closed=false)
    return _extract_diffraction_edges_validated(
        mesh, min_dihedral, include_boundary)
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
    if abs(denom) < sqrt(eps(Float64))
        return setprecision(BigFloat, 512) do
            ub = BigFloat(u)
            nb = BigFloat(n)
            value = (sin(BigFloat(π) / nb) / nb) /
                    (cos(BigFloat(π) / nb) - cos(2ub / nb)) -
                    tan(ub) / 2
            clamp(Float64(value), -cap, cap)
        end
    end

    num = 2cos_u * sin_pi_n - n * sin_u * D
    return clamp(num / denom, -cap, cap)
end

"""
    _stable_YplusTanG(v, n, γ, sign_Y, sign_tan) -> Float64

Compute  sign_Y · Y + sign_tan · (1/2)·tan(γ − v)  stably for a general
wedge of exterior angle γ = n·π.

`Y = (sin(π/n)/n) / [cos(π/n) − cos(2v/n)]`. Both `Y` and `tan(γ−v)` are
evaluated through a single common-denominator expression

    S = [ sign_Y·a·cos(γ−v) + sign_tan·(1/2)·sin(γ−v)·D ] / [ D·cos(γ−v) ],

with `a = sin(π/n)/n` and `D = cos(π/n) − cos(2v/n)`. This is the exact
value of `sign_Y·Y + sign_tan·(1/2)tan(γ−v)` (verified algebraically) and
avoids catastrophic cancellation when the two large terms partially cancel
away from a pole.

For the half-plane case `n = 2` (γ = 2π) one has `tan(2π−v) = −tan(v)`, so
this reduces *exactly* to the legacy `sign_Y·Y − sign_tan·(1/2)tan(v)`
expression (and to its fallback) — i.e. the previous code is recovered with
the tan sign flipped, which is why the caller flips `sign_tan` below.

At a true reflection boundary `Y` and `tan(γ−v)` diverge together; the GTD
fringe coefficient genuinely blows up there (ray theory limitation). One-sided
values are clamped to `±cap`. Exactly on the boundary there is no unique
one-sided value, so `_ptd_fringe_fg` uses the symmetric principal-value
convention and omits that edge correction.
"""
@inline function _stable_YplusTanG(v::Float64, n::Float64, γ::Float64,
                                     sign_Y::Int, sign_tan::Int; cap::Float64=10.0)
    a = sin(π / n) / n                 # = sin(π/n)/n
    D = cos(π / n) - cos(2v / n)       # Y = a/D, diverges at the shadow boundary
    γv = γ - v
    cos_γv = cos(γv)
    sin_γv = sin(γv)

    denom = D * cos_γv
    if abs(denom) < sqrt(eps(Float64))
        return setprecision(BigFloat, 512) do
            vb = BigFloat(v)
            nb = BigFloat(n)
            gammab = BigFloat(γ)
            y_value = (sin(BigFloat(π) / nb) / nb) /
                      (cos(BigFloat(π) / nb) - cos(2vb / nb))
            value = BigFloat(sign_Y) * y_value +
                    BigFloat(sign_tan) * tan(gammab - vb) / 2
            clamp(Float64(value), -cap, cap)
        end
    end

    num = sign_Y * a * cos_γv + sign_tan * 0.5 * sin_γv * D
    return clamp(num / denom, -cap, cap)
end

"""
    _ptd_fringe_fg(n, delta_s, delta_i, gamma) -> (f, g)

Compute the real-valued PTD fringe coefficients f and g for a PEC wedge,
following Sáez de Adana et al., eqs. 4.131-4.136.

Uses the bottom-side illuminated formula (eq 4.133-4.134) which matches
the azimuth convention where δⁱ is computed from k̂.

Uses numerically stable combined computation of (X − tan) and (Y ± tan)
to avoid catastrophic cancellation at shadow/reflection boundaries. The
`(1/2)tan(γ−v)` reflection-boundary term is kept exact for *any* wedge
angle γ = n·π (not only the half-plane n = 2), via `_stable_YplusTanG`.
"""
function _ptd_fringe_fg(n::Float64, delta_s::Float64, delta_i::Float64,
                         gamma::Float64)
    u = 0.5 * (delta_s - delta_i)
    v = 0.5 * (delta_s + delta_i)
    # A = X - (1/2)tan(u)  [stable, bounded at shadow boundary]
    A = _stable_XminusTan(u, n; cap=Inf)

    # Bottom-side formula (eq 4.133-4.134):
    # f = X - Y - 1/2 tan(u) - 1/2 tan(γ-v)
    # g = X + Y - 1/2 tan(u) + 1/2 tan(γ-v)
    #
    # The (1/2)tan(γ-v) term is computed exactly through _stable_YplusTanG,
    # which pairs it with ∓Y in a common-denominator form. Y and tan(γ-v)
    # both diverge at the reflection boundary and must be combined stably.
    # For n=2, γ=2π and tan(2π-v)=-tan(v), so this reduces exactly to the
    # previous half-plane expression (verified to ≤1e-10).
    B_bot = _stable_YplusTanG(v, n, gamma, -1, -1; cap=Inf)
    C_bot = _stable_YplusTanG(v, n, gamma, +1, +1; cap=Inf)

    f = clamp(A + B_bot, -10.0, 10.0)
    g = clamp(A + C_bot, -10.0, 10.0)
    if !isfinite(f) || !isfinite(g)
        # At the exact GTD ray boundary, the two one-sided limits diverge with
        # opposite signs.  A coordinate-independent symmetric convention is
        # to omit the singular edge term; neighboring rays retain their
        # correctly signed, capped values.
        return 0.0, 0.0
    end

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
    solve_ptd(mesh, freq_hz, excitation; grid, c0, eta0,
              min_dihedral_deg, include_boundary,
              max_work_bytes=536_870_912)

Compute the PO+PTD scattered far-field for a PEC body.

Calls `solve_po` for the PO contribution, then adds PTD fringe corrections
from diffraction edges using the Sáez de Adana et al. formulation
(eqs 4.131-4.146): fringe = exact_edge - PO_edge.
`max_work_bytes` bounds the combined raw payload of the PO and PTD outputs and
construction workspaces before either solver allocates its field arrays.

# Returns
`PTDResult` with combined PO+PTD far-field, individual components, and edge data.
"""
function solve_ptd(mesh::TriMesh, freq_hz::Real, excitation::PlaneWaveExcitation;
                   grid::SphGrid=make_sph_grid(36, 72),
                   c0::Float64=299792458.0,
                   eta0::Float64=376.730313668,
                   min_dihedral_deg::Float64=5.0,
                   include_boundary::Bool=true,
                   max_work_bytes::Integer=_DEFAULT_MAX_PTD_WORK_BYTES)
    min_dihedral = _validated_min_dihedral(min_dihedral_deg)

    # Validate and reject unsupported interior wedges before the potentially
    # expensive PO integration.  `solve_po` repeats these inexpensive guards
    # for its own public contract once the geometry is known to be supported.
    _, _, _, validated_k_hat =
        _validate_po_inputs(grid, freq_hz, excitation, c0, eta0)
    assert_mesh_quality(mesh; allow_boundary=true, require_closed=false)
    edges = _extract_diffraction_edges_validated(
        mesh, min_dihedral, include_boundary)

    # The implemented Sáez de Adana coefficient branch is the validated
    # half-plane/boundary formula. Applying its bottom-side convention to an
    # interior wedge requires a physical illuminated-side classification;
    # choosing a side by face order, vertex labels, coordinates, or normals
    # makes the field depend on mesh representation. Fail closed until both
    # interior side branches are implemented and benchmarked.
    any(edge -> edge.face_n != 0, edges) &&
        throw(ArgumentError(
            "solve_ptd currently supports boundary half-plane edges only; " *
            "interior-wedge PTD requires an invariant illuminated-side formulation"))

    _, work_limit = _preflight_ptd_work(
        ntriangles(mesh), length(edges), length(grid.w), max_work_bytes)

    # ── Phase 1: PO solution ──
    po = solve_po(
        mesh, freq_hz, excitation;
        grid=grid, c0=c0, eta0=eta0,
        max_work_bytes=work_limit)

    k = po.k
    NΩ = length(grid.w)

    k_vec = excitation.k_vec
    k_hat = validated_k_hat
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
    exact_direction_totals = Dict{Int,Vector{Complex{BigFloat}}}()
    exact_direction_count = 0

    prefactor = -1.0 / (2π)

    # Precompute direction Vec3s once (avoids a column-slice allocation per direction).
    rhat_vec = Vector{Vec3}(undef, NΩ)
    @inbounds for q in 1:NΩ
        rhat_vec[q] = Vec3(grid.rhat[1, q], grid.rhat[2, q], grid.rhat[3, q])
    end

    # Edge-dependent incident quantities are invariant across observation
    # directions. Keeping the edge loop outermost avoids recomputing them NΩ
    # times without allocating an O(number-of-edges) workspace.
    for edge in edges
        ê = edge.tangent
        L = edge.length
        Q₀ = edge.center
        γ = edge.alpha   # exterior wedge angle

        # ── Incident cone angle and azimuth ──
        sin_beta = norm(cross(ê, k_hat))
        sin_beta > 1e-4 || continue
        sin2_beta = sin_beta^2
        delta_i = _ptd_edge_azimuth(k_hat, edge)
        isnothing(delta_i) && continue

        # The edge-line phase below already contains the incident propagation
        # term.  Keep these coefficients as the unphased incident amplitudes
        # so translating the edge does not apply that phase twice.
        incident_requires_exact =
            _ieee_dense_extreme_factor(E0, Float64)
        @inbounds for component in 1:3
            incident_requires_exact |=
                _ieee_dense_extreme_factor(pol[component], Float64) ||
                _ieee_dense_extreme_factor(k_hat[component], Float64) ||
                _ieee_dense_extreme_factor(ê[component], Float64)
        end
        tE, tH = if incident_requires_exact
            _ptd_incident_components_exact(
                E0, pol, k_hat, ê)
        else
            incident_amplitude = pol * E0
            all(isfinite, incident_amplitude) ||
                throw(OverflowError(
                    "PTD incident edge field overflowed"))
            dot(ê, incident_amplitude),
            dot(ê, cross(k_hat, incident_amplitude))
        end
        incident_center_phase = _source_directional_phase(
            k, k_vec, k_hat, Q₀, -1.0, "PTD incident edge phase")

        for q in 1:NΩ
            supplied_r_hat = rhat_vec[q]
            r_hat = _validated_farfield_direction(supplied_r_hat)

            # ── Scattered cone angle and azimuth ──
            sin_beta_s = norm(cross(ê, r_hat))
            sin_beta_s > 1e-4 || continue
            delta_s = _ptd_edge_azimuth(r_hat, edge)
            isnothing(delta_s) && continue

            # ── PTD fringe coefficients (real-valued, eq 4.137-4.138) ──
            n = γ / π
            f_ptd, g_ptd = _ptd_fringe_fg(n, delta_s, delta_i, γ)

            # ── Scattered field ──
            basis = _ptd_beta_phi_basis(r_hat, ê)
            isnothing(basis) && continue
            beta_hat, phi_hat = basis

            # ── Edge line integral (sinc × phase) ──
            delta_k = r_hat - k_hat
            q_edge = dot(delta_k, ê)
            kqL_half = (k * q_edge) * (L / 2.0)
            sinc_val = if isfinite(kqL_half)
                abs(kqL_half) > 1e-15 ?
                    sin(kqL_half) / kqL_half : 1.0
            else
                setprecision(BigFloat, 4352) do
                    argument = BigFloat(k) * BigFloat(q_edge) *
                               BigFloat(L) / 2
                    converted = Float64(sin(argument) / argument)
                    isfinite(converted) ||
                        throw(OverflowError(
                            "PTD edge sinc factor is outside the Float64 range"))
                    converted
                end
            end
            outgoing_center_phase = _source_directional_phase(
                k, supplied_r_hat, r_hat, Q₀, 1.0,
                "PTD scattered edge phase")
            phase = outgoing_center_phase * incident_center_phase

            # Treat the complete edge contribution as one product.  Forming
            # the field coefficient before multiplying by a tiny edge length
            # can overflow even when the physical contribution is finite.
            needs_exact = incident_requires_exact ||
                          haskey(exact_direction_totals, q)
            @inbounds for factor in (
                prefactor, sin2_beta, f_ptd, g_ptd,
                real(tE), imag(tE), real(tH), imag(tH),
                L, sinc_val, real(phase), imag(phase),
            )
                needs_exact |= _ieee_dense_extreme_factor(factor, Float64)
            end
            if needs_exact
                exact_direction_count = _ptd_register_exact_direction!(
                    exact_direction_totals, q, exact_direction_count)
                _ptd_accumulate_direction_exact!(
                    exact_direction_totals, E_ff_ptd, q,
                    prefactor, sin2_beta,
                    f_ptd, g_ptd, tE, tH, beta_hat, phi_hat,
                    L, sinc_val, phase)
                continue
            end

            common = (prefactor * L * sinc_val * phase) / sin2_beta
            E_vec = common *
                (-f_ptd * tE * beta_hat - g_ptd * tH * phi_hat)
            if !all(isfinite, E_vec)
                exact_direction_count = _ptd_register_exact_direction!(
                    exact_direction_totals, q, exact_direction_count)
                _ptd_accumulate_direction_exact!(
                    exact_direction_totals, E_ff_ptd, q,
                    prefactor, sin2_beta,
                    f_ptd, g_ptd, tE, tH, beta_hat, phi_hat,
                    L, sinc_val, phase)
                continue
            end

            updated = CVec3(
                E_ff_ptd[1, q] + E_vec[1],
                E_ff_ptd[2, q] + E_vec[2],
                E_ff_ptd[3, q] + E_vec[3],
            )
            if !all(isfinite, updated)
                exact_direction_count = _ptd_register_exact_direction!(
                    exact_direction_totals, q, exact_direction_count)
                _ptd_accumulate_direction_exact!(
                    exact_direction_totals, E_ff_ptd, q,
                    prefactor, sin2_beta,
                    f_ptd, g_ptd, tE, tH, beta_hat, phi_hat,
                    L, sinc_val, phase)
                continue
            end
            E_ff_ptd[1, q] = updated[1]
            E_ff_ptd[2, q] = updated[2]
            E_ff_ptd[3, q] = updated[3]
        end
    end
    for (direction, totals) in exact_direction_totals
        @inbounds for component in 1:3
            converted = ComplexF64(totals[component])
            isfinite(converted) ||
                throw(OverflowError(
                    "PTD far field is outside the ComplexF64 range at " *
                    "direction $direction"))
            E_ff_ptd[component, direction] = converted
        end
    end
    all(isfinite, E_ff_ptd) ||
        throw(OverflowError("PTD far field contains non-finite values"))

    # ── Phase 4: Combine PO + PTD ──
    E_ff_combined = po.E_ff + E_ff_ptd
    all(isfinite, E_ff_combined) ||
        throw(OverflowError("combined PO+PTD far field is non-finite"))

    return PTDResult(E_ff_combined, po.E_ff, E_ff_ptd,
                     po.J_s, po.illuminated, edges,
                     grid, po.freq_hz, k)
end
