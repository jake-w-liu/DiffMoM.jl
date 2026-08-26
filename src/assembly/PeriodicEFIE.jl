# PeriodicEFIE.jl — Periodic EFIE assembly via image correction
#
# Assembles Z_per = Z_free + Z_correction, where:
#   Z_free      = standard free-space EFIE (existing code, handles singularity)
#   Z_correction = contribution from periodic images (m,n) ≠ (0,0)
#
# The correction uses greens_periodic_correction() which is smooth (no 1/R
# singularity), so standard product quadrature suffices for all entries.

export assemble_Z_efie_periodic

const _DEFAULT_MAX_PERIODIC_EFIE_CACHE_BYTES = 2_000_000_000
const _DEFAULT_MAX_PERIODIC_GREEN_TERMS = 500_000_000

function _enforce_periodic_efie_cache_limit(
        estimated_bytes::Int,
        max_cache_bytes::Integer,
        label::AbstractString="periodic EFIE auxiliary storage")
    limit = _validated_resource_limit("max_cache_bytes", max_cache_bytes)
    estimated_bytes <= limit ||
        throw(ArgumentError(
            "$label requires at most $estimated_bytes estimated bytes, " *
            "exceeding max_cache_bytes=$limit"))
    return estimated_bytes
end

struct _PeriodicTriangleIncidence
    offsets::Vector{Int}
    entries::Vector{NTuple{2,Int}}
    max_incident::Int
    active_triangles::Int
end

struct _PeriodicSpatialEwaldTerm
    m::Int
    n::Int
    sx::Float64
    sy::Float64
    phase::ComplexF64
end

struct _PeriodicSpectralEwaldTerm
    kappa_x::Float64
    kappa_y::Float64
    kz::ComplexF64
    zk::ComplexF64
end

struct _PeriodicEFIEGeometryCache{TD,TV}
    wq::Vector{Float64}
    Nq::Int
    quad_pts::Vector{Vector{Vec3}}
    areas::Vector{Float64}
    div_vals::Matrix{TD}
    rwg_vals::Vector{NTuple{2,Vector{TV}}}
    incidence::_PeriodicTriangleIncidence
    spatial_terms::Vector{_PeriodicSpatialEwaldTerm}
    spectral_terms::Vector{_PeriodicSpectralEwaldTerm}
    ntasks::Int
end

function _periodic_efie_cache_bytes(
        N::Int,
        Nt::Int,
        Nq::Int,
        ntasks::Int,
        max_incident::Int,
        ::Type{Tcoef},
        ::Type{TVec};
        point_cache_count::Int=1,
        spatial_term_count::Int=0,
        spectral_term_count::Int=0) where {Tcoef,TVec}
    all(>=(0), (N, Nt, Nq, ntasks, max_incident)) ||
        throw(ArgumentError(
            "periodic EFIE cache dimensions must be nonnegative"))
    point_cache_count >= 1 ||
        throw(ArgumentError("point_cache_count must be positive"))
    spatial_term_count >= 0 ||
        throw(ArgumentError("spatial term count must be nonnegative"))
    spectral_term_count >= 0 ||
        throw(ArgumentError("spectral term count must be nonnegative"))

    point_payload = BigInt(point_cache_count) * Nt * (
        sizeof(Vector{Vec3}) + _EFIE_EMPTY_VECTOR_BYTES +
        BigInt(Nq) * sizeof(Vec3))
    basis_payload = BigInt(N) * (
        2sizeof(Tcoef) +
        sizeof(NTuple{2,Vector{TVec}}) +
        2 * _EFIE_EMPTY_VECTOR_BYTES + 2BigInt(Nq) * sizeof(TVec))
    incidence_payload =
        BigInt(Nt) * sizeof(Int) +
        (BigInt(Nt) + 1) * sizeof(Int) +
        2BigInt(N) * sizeof(NTuple{2,Int})
    quadrature_payload =
        2 * _EFIE_EMPTY_VECTOR_BYTES +
        BigInt(Nq) * (sizeof(SVector{2,Float64}) + sizeof(Float64))
    task_payload = BigInt(ntasks) * (
        BigInt(Nq) * Nq * sizeof(ComplexF64) +
        BigInt(max_incident) * N * sizeof(ComplexF64))
    lock_payload = BigInt(N) * sizeof(Threads.SpinLock)
    ewald_payload =
        2 * _EFIE_EMPTY_VECTOR_BYTES +
        BigInt(spatial_term_count) * sizeof(_PeriodicSpatialEwaldTerm) +
        BigInt(spectral_term_count) * sizeof(_PeriodicSpectralEwaldTerm)
    raw_bytes = point_payload + BigInt(Nt) * sizeof(Float64) +
                basis_payload + incidence_payload + quadrature_payload +
                task_payload + lock_payload + ewald_payload
    estimated = cld(5 * raw_bytes, 4)
    estimated <= typemax(Int) ||
        throw(ArgumentError(
            "periodic EFIE auxiliary-storage estimate overflows Int"))
    return Int(estimated)
end

function _build_periodic_ewald_terms(
        lattice::PeriodicLattice,
        active_triangles::Int)
    if iszero(active_triangles)
        return _PeriodicSpatialEwaldTerm[], _PeriodicSpectralEwaldTerm[]
    end

    Ns = lattice.N_spatial
    Nf = lattice.N_spectral
    spatial_count = _periodic_term_count(Ns) - 1
    spectral_capacity = _periodic_term_count(Nf)
    spatial_terms = Vector{_PeriodicSpatialEwaldTerm}(undef, spatial_count)
    spectral_terms = Vector{_PeriodicSpectralEwaldTerm}(undef, spectral_capacity)

    spatial_idx = 1
    @inbounds for m in -Ns:Ns, n in -Ns:Ns
        iszero(m) && iszero(n) && continue
        sx = m * lattice.dx
        sy = n * lattice.dy
        phase = _periodic_spatial_image_phase(
            lattice, m, n, sx, sy)
        spatial_terms[spatial_idx] =
            _PeriodicSpatialEwaldTerm(m, n, sx, sy, phase)
        spatial_idx += 1
    end
    resize!(spatial_terms, spatial_idx - 1)

    spectral_idx = 1
    @inbounds for p in -Nf:Nf, q in -Nf:Nf
        kappa_x = lattice.kx_bloch + 2π * p / lattice.dx
        kappa_y = lattice.ky_bloch + 2π * q / lattice.dy
        kz = _spectral_kz(lattice.k, kappa_x, kappa_y)
        _periodic_is_wood_anomaly(kz, lattice.k) && continue
        zk = im * kz / (2lattice.E)
        spectral_terms[spectral_idx] =
            _PeriodicSpectralEwaldTerm(
                kappa_x, kappa_y, kz, zk)
        spectral_idx += 1
    end
    resize!(spectral_terms, spectral_idx - 1)
    return spatial_terms, spectral_terms
end

function _greens_periodic_correction_cached(
        r::SVector{3,<:Real},
        rp::SVector{3,<:Real},
        k::Float64,
        lattice::PeriodicLattice,
        spatial_terms::Vector{_PeriodicSpatialEwaldTerm},
        spectral_terms::Vector{_PeriodicSpectralEwaldTerm})
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
            "r-rp is outside the supported Float64 coordinate range"))

    R_self = hypot(hypot(drho_x, drho_y), drho_z)
    value = ComplexF64(_ewald_self_correction(R_self, k, lattice.E))
    @inbounds for term in spatial_terms
        spatial_value = _periodic_spatial_image_kernel(
            drho_x, drho_y, drho_z,
            term.m, term.n, lattice, term.sx, term.sy)
        iszero(spatial_value) && continue
        value += term.phase * spatial_value
    end

    @inbounds for term in spectral_terms
        phase = _periodic_transverse_phase(
            term.kappa_x, term.kappa_y, drho_x, drho_y)
        spectral_value = _periodic_spectral_vertical_kernel(
            term.kz, term.zk, lattice.E, drho_z)
        value += _periodic_scale_by_cell_area(
            phase * spectral_value, lattice.dx, lattice.dy)
    end
    isfinite(value) ||
        throw(OverflowError(
            "periodic Green correction is outside the representable " *
            "ComplexF64 range"))
    return value
end

function _periodic_efie_green_terms(
        active_triangles::Int,
        quadrature_count::Int,
        lattice::PeriodicLattice,
        symmetric::Bool;
        image_block::Bool=false)
    active_triangles >= 0 ||
        throw(ArgumentError("active triangle count must be nonnegative"))
    quadrature_count >= 0 ||
        throw(ArgumentError("quadrature count must be nonnegative"))
    triangle_pairs = symmetric ?
        BigInt(active_triangles) * (active_triangles + 1) ÷ 2 :
        BigInt(active_triangles)^2
    terms_per_green = BigInt(_periodic_term_count(lattice.N_spatial)) +
                      _periodic_term_count(lattice.N_spectral) +
                      (image_block ? 1 : 0)
    terms = triangle_pairs * quadrature_count^2 * terms_per_green
    terms <= typemax(Int) ||
        throw(ArgumentError(
            "periodic EFIE Green-series work estimate overflows Int"))
    return Int(terms)
end

function _build_periodic_triangle_incidence(rwg::RWGData, Nt::Int)
    N = rwg.nedges
    N >= 0 || throw(ArgumentError("RWG edge count must be nonnegative"))
    length(rwg.tplus) >= N && length(rwg.tminus) >= N ||
        throw(DimensionMismatch(
            "RWG triangle-support arrays must contain nedges=$N entries"))

    counts = zeros(Int, Nt)
    @inbounds for n in 1:N
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        1 <= tp <= Nt ||
            throw(ArgumentError(
                "RWG $n has out-of-range plus-triangle index $tp"))
        1 <= tm <= Nt ||
            throw(ArgumentError(
                "RWG $n has out-of-range minus-triangle index $tm"))
        counts[tp] = Base.checked_add(counts[tp], 1)
        counts[tm] = Base.checked_add(counts[tm], 1)
    end

    max_incident = maximum(counts; init=0)
    active_triangles = count(!iszero, counts)
    offsets = Vector{Int}(undef, Nt + 1)
    offsets[1] = 1
    @inbounds for t in 1:Nt
        offsets[t + 1] = Base.checked_add(offsets[t], counts[t])
        counts[t] = offsets[t]
    end
    entries = Vector{NTuple{2,Int}}(undef, offsets[end] - 1)
    @inbounds for n in 1:N
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        entries[counts[tp]] = (n, 1)
        counts[tp] += 1
        entries[counts[tm]] = (n, 2)
        counts[tm] += 1
    end
    return _PeriodicTriangleIncidence(
        offsets, entries, max_incident, active_triangles)
end

@inline function _periodic_incidence_range(
        incidence::_PeriodicTriangleIncidence, triangle::Int)
    return incidence.offsets[triangle]:(incidence.offsets[triangle + 1] - 1)
end

function _build_periodic_efie_geometry_cache(
        mesh::TriMesh,
        rwg::RWGData,
        lattice::PeriodicLattice;
        quad_order::Int,
        point_cache_count::Int,
        image_block::Bool,
        max_cache_bytes::Integer,
        max_green_terms::Integer)
    cache_limit = _validated_resource_limit(
        "max_cache_bytes", max_cache_bytes)
    term_limit = _validated_resource_limit(
        "max_green_terms", max_green_terms)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    TVec = SVector{3,Tcoef}
    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    ntasks = max(1, min(Threads.nthreads(), Nt))

    # Reject from scalar dimensions before the first mesh-sized allocation.
    minimum_bytes = _periodic_efie_cache_bytes(
        N, Nt, Nq, ntasks, 0, Tcoef, TVec;
        point_cache_count=point_cache_count)
    _enforce_periodic_efie_cache_limit(minimum_bytes, cache_limit)

    incidence = _build_periodic_triangle_incidence(rwg, Nt)
    spatial_term_count = iszero(incidence.active_triangles) ?
        0 : _periodic_term_count(lattice.N_spatial) - 1
    spectral_term_count = iszero(incidence.active_triangles) ?
        0 : _periodic_term_count(lattice.N_spectral)
    cache_bytes = _periodic_efie_cache_bytes(
        N, Nt, Nq, ntasks, incidence.max_incident, Tcoef, TVec;
        point_cache_count=point_cache_count,
        spatial_term_count=spatial_term_count,
        spectral_term_count=spectral_term_count)
    _enforce_periodic_efie_cache_limit(cache_bytes, cache_limit)
    symmetric = _periodic_correction_is_symmetric(rwg, lattice)
    green_terms = _periodic_efie_green_terms(
        incidence.active_triangles, Nq, lattice, symmetric;
        image_block=image_block)
    green_terms <= term_limit ||
        throw(ArgumentError(
            "periodic EFIE assembly requires $green_terms Green-series " *
            "terms, exceeding max_green_terms=$term_limit"))

    spatial_terms, spectral_terms = _build_periodic_ewald_terms(
        lattice, incidence.active_triangles)

    quad_pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    @inbounds for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
    end

    div_vals = zeros(Tcoef, 2, N)
    rwg_vals = Vector{NTuple{2,Vector{TVec}}}(undef, N)
    @inbounds for n in 1:N
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        div_vals[1, n] = div_rwg(rwg, n, tp)
        div_vals[2, n] = div_rwg(rwg, n, tm)
        vals_p = [eval_rwg(rwg, n, quad_pts[tp][q], tp) for q in 1:Nq]
        vals_m = [eval_rwg(rwg, n, quad_pts[tm][q], tm) for q in 1:Nq]
        rwg_vals[n] = (vals_p, vals_m)
    end

    return _PeriodicEFIEGeometryCache(
        wq, Nq, quad_pts, areas, div_vals, rwg_vals,
        incidence, spatial_terms, spectral_terms, ntasks),
        cache_bytes, green_terms
end

function _assert_coplanar_periodic_mesh(mesh::TriMesh; atol::Float64=1e-12)
    isfinite(atol) && atol >= 0.0 ||
        throw(ArgumentError(
            "PeriodicEFIE coplanarity tolerance must be finite and " *
            "nonnegative, got $atol"))
    zvals = @view mesh.xyz[3, :]
    zmin = minimum(zvals)
    zmax = maximum(zvals)
    spread = abs(zmax - zmin)
    if !_periodic_coordinate_within(zmax, zmin, atol)
        throw(ArgumentError(
            "PeriodicEFIE requires a coplanar unit-cell mesh with z spread " *
            "<= $atol; got z spread=$spread. Project the mesh vertices " *
            "onto one z plane before assembly."
        ))
    end
end

function _mesh_has_unitcell_boundary_edges(mesh::TriMesh, lattice::PeriodicLattice;
                                           atol_abs::Float64=1e-12,
                                           atol_rel::Float64=1e-9,
                                           max_cache_bytes::Integer=
                                               _DEFAULT_MAX_PERIODIC_EFIE_CACHE_BYTES)
    isfinite(atol_abs) && atol_abs >= 0.0 ||
        throw(ArgumentError(
            "atol_abs must be finite and nonnegative, got $atol_abs"))
    isfinite(atol_rel) && 0.0 <= atol_rel < 0.5 ||
        throw(ArgumentError(
            "atol_rel must be finite and in [0, 0.5), got $atol_rel"))
    absolute_tolerance = min(lattice.dx, lattice.dy) > 2atol_abs ?
        atol_abs : 0.0
    tol_x = max(absolute_tolerance, atol_rel * lattice.dx)
    tol_y = max(absolute_tolerance, atol_rel * lattice.dy)
    xmin = -0.5 * lattice.dx
    xmax =  0.5 * lattice.dx
    ymin = -0.5 * lattice.dy
    ymax =  0.5 * lattice.dy

    Nt = ntriangles(mesh)
    edge_record_count = Base.checked_mul(3, Nt)
    raw_bytes = BigInt(edge_record_count) * sizeof(NTuple{2,Int})
    estimated_bytes = cld(5 * raw_bytes, 4)
    estimated_bytes <= typemax(Int) ||
        throw(ArgumentError(
            "periodic boundary-edge workspace estimate overflows Int"))
    _enforce_periodic_efie_cache_limit(
        Int(estimated_bytes), max_cache_bytes,
        "periodic boundary-edge workspace")
    edge_records = Vector{NTuple{2,Int}}(undef, edge_record_count)
    record_idx = 1
    for t in 1:Nt
        for le in 1:3
            v1 = mesh.tri[le, t]
            v2 = mesh.tri[mod1(le + 1, 3), t]
            key = v1 < v2 ? (v1, v2) : (v2, v1)
            edge_records[record_idx] = key
            record_idx += 1
        end
    end
    sort!(edge_records)

    first_record = 1
    while first_record <= edge_record_count
        next_edge = first_record + 1
        @inbounds while next_edge <= edge_record_count &&
                        edge_records[next_edge] == edge_records[first_record]
            next_edge += 1
        end
        if next_edge != first_record + 1
            first_record = next_edge
            continue
        end

        va, vb = edge_records[first_record]
        xa = mesh.xyz[1, va]; xb = mesh.xyz[1, vb]
        ya = mesh.xyz[2, va]; yb = mesh.xyz[2, vb]

        on_xmin =
            _periodic_boundary_coordinate_within(xa, xmin, tol_x) &&
            _periodic_boundary_coordinate_within(xb, xmin, tol_x)
        on_xmax =
            _periodic_boundary_coordinate_within(xa, xmax, tol_x) &&
            _periodic_boundary_coordinate_within(xb, xmax, tol_x)
        on_ymin =
            _periodic_boundary_coordinate_within(ya, ymin, tol_y) &&
            _periodic_boundary_coordinate_within(yb, ymin, tol_y)
        on_ymax =
            _periodic_boundary_coordinate_within(ya, ymax, tol_y) &&
            _periodic_boundary_coordinate_within(yb, ymax, tol_y)

        if on_xmin || on_xmax || on_ymin || on_ymax
            return true
        end
        first_record = next_edge
    end

    return false
end

function _assert_boundary_touching_periodic_mesh_requires_bloch(mesh::TriMesh,
                                                                lattice::PeriodicLattice,
                                                                rwg::Union{Nothing,RWGData}=nothing;
                                                                max_cache_bytes::Integer=
                                                                    _DEFAULT_MAX_PERIODIC_EFIE_CACHE_BYTES)
    isnothing(rwg) && return
    rwg.has_periodic_bloch && return
    _mesh_has_unitcell_boundary_edges(
        mesh, lattice; max_cache_bytes=max_cache_bytes) || return
    throw(ArgumentError(
        "Mesh has conductor boundary edges on the unit-cell boundary, but RWG basis " *
        "does not carry Bloch-periodic boundary pairing. Build RWG with " *
        "`build_rwg_periodic(mesh, lattice; ...)` for boundary-touching periodic cells."
    ))
end

"""
    assemble_Z_efie_periodic(mesh, rwg, k, lattice;
                             quad_order=3, eta0=376.730313668,
                             max_work_bytes=2_000_000_000,
                             max_cache_bytes=2_000_000_000,
                             max_adjacency_pairs=20_000_000,
                             max_green_terms=500_000_000)

Assemble the dense periodic EFIE matrix `Z_per ∈ C^{N×N}` for a unit cell
with 2D periodicity defined by `lattice::PeriodicLattice`.

Strategy:
  Z_per = Z_free + Z_correction

- Z_free: standard free-space EFIE (with singularity extraction for self-cells)
- Z_correction: image sum using ΔG = G_per - G_0 (smooth, no singularity)

Both use the mixed-potential form:
  Z_mn = -iωμ₀ [ ∫∫ f_m·f_n G dS dS' - (1/k²) ∫∫ (∇·f_m)(∇'·f_n) G dS dS' ]

`max_work_bytes` bounds the three dense matrices. `max_cache_bytes` bounds
each sequential auxiliary cache and the boundary-edge validation workspace,
while `max_adjacency_pairs` bounds free-space triangle adjacency records.
`max_green_terms` bounds the complete Ewald-series work of the correction.
Lattice-dependent Ewald terms are computed once and shared read-only by all
triangle-pair tasks.
"""
function assemble_Z_efie_periodic(mesh::TriMesh, rwg::RWGData, k,
                                  lattice::PeriodicLattice;
                                  quad_order::Int=3,
                                  eta0::Float64=376.730313668,
                                  max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                  max_cache_bytes::Integer=
                                      _DEFAULT_MAX_PERIODIC_EFIE_CACHE_BYTES,
                                  max_adjacency_pairs::Integer=
                                      _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                                  max_green_terms::Integer=
                                      _DEFAULT_MAX_PERIODIC_GREEN_TERMS)
    _validate_mesh_rwg_pair(mesh, rwg)
    work_bytes = _checked_array_payload_bytes(
        ComplexF64, 3, rwg.nedges, rwg.nedges;
        label="periodic EFIE dense work matrices")
    work_limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    _enforce_payload_limit(
        work_bytes, work_limit,
        "periodic EFIE dense work matrices", "max_work_bytes")
    cache_limit = _validated_resource_limit(
        "max_cache_bytes", max_cache_bytes)
    adjacency_limit = _validated_nonnegative_resource_limit(
        "max_adjacency_pairs", max_adjacency_pairs)
    green_limit = _validated_resource_limit(
        "max_green_terms", max_green_terms)
    tri_quad_rule(quad_order)

    kw = _validated_lattice_wavenumber(k, lattice)
    _validated_efie_prefactors(kw, eta0)
    _assert_coplanar_periodic_mesh(mesh)
    _assert_boundary_touching_periodic_mesh_requires_bloch(
        mesh, lattice, rwg; max_cache_bytes=cache_limit)

    # Step 1: Free-space EFIE (handles self-cell singularity)
    Z_free = assemble_Z_efie(mesh, rwg, kw;
                             quad_order=quad_order, eta0=eta0,
                             mesh_precheck=false,
                             max_output_bytes=work_limit,
                             max_cache_bytes=cache_limit,
                             max_adjacency_pairs=adjacency_limit)

    # Step 2: Periodic image correction (smooth, no singularity)
    Z_corr = _assemble_periodic_correction(mesh, rwg, kw, lattice;
                                            quad_order=quad_order, eta0=eta0,
                                            max_cache_bytes=cache_limit,
                                            max_green_terms=green_limit)

    Z_periodic = Z_free + Z_corr
    all(isfinite, Z_periodic) ||
        throw(OverflowError(
            "periodic EFIE matrix contains entries outside the " *
            "representable ComplexF64 range"))
    return Z_periodic
end

"""
Return `true` when the assembled periodic correction `Z_corr` is provably
symmetric (`Z_corr[m,n] == Z_corr[n,m]`), so the assembly may compute only the
upper triangle and mirror it (matching `assemble_Z_efie`).

Symmetry requires BOTH:

1. Zero Bloch phase (`kx_bloch == ky_bloch == 0`): the quasi-periodic correction
   is then reciprocal, `ΔG(a,b) == ΔG(b,a)`, so the kernel is symmetric.
2. Real RWG coefficients: with real `f_m`/`∇·f_m` the entry kernel
   `dot(f_m,f_n) - conj(∇·f_m)(∇·f_n)/k²` is symmetric under `m↔n`.

Both are needed independently: a Bloch-paired RWG carrying complex coefficients
combined with a zero-phase lattice (a mismatched build) yields a reciprocal `ΔG`
but a *non*-symmetric `Z_corr`. Checking the coefficients directly guards that
case rather than trusting `kx_bloch/ky_bloch` alone. Boundary-paired RWGs built
at normal incidence have phase `exp(0)=1`, so their coefficients are real and the
fast path still applies.
"""
function _periodic_correction_is_symmetric(rwg::RWGData, lattice::PeriodicLattice)
    (iszero(lattice.kx_bloch) && iszero(lattice.ky_bloch)) || return false
    all(isreal, rwg.coeff_plus) || return false
    all(isreal, rwg.coeff_minus) || return false
    return true
end

# Merge the rows accumulated for one source triangle into the global matrix.
# A task holds each row lock only once per source triangle, rather than once per
# scalar contribution. This bounds synchronization overhead while keeping the
# per-task scratch proportional to the number of basis functions incident on one
# triangle.
function _merge_periodic_triangle_rows!(
        Z, row_locks, row_buffer,
        incidence::_PeriodicTriangleIncidence,
        incident_range::UnitRange{Int})
    N = size(Z, 2)
    @inbounds for (slot, incident_idx) in enumerate(incident_range)
        m_idx, _ = incidence.entries[incident_idx]
        row_lock = row_locks[m_idx]
        lock(row_lock)
        try
            for n_idx in 1:N
                Z[m_idx, n_idx] += row_buffer[slot, n_idx]
            end
        finally
            unlock(row_lock)
        end
    end
    return nothing
end

# The symmetric triangle-pair sweep stores half of every same-triangle block
# and one orientation of every distinct-triangle block. Reciprocity then makes
# P + transpose(P) the complete matrix. This in-place form avoids allocating a
# transposed copy.
function _complete_periodic_triangle_symmetry!(Z)
    N = size(Z, 1)
    @inbounds for m_idx in 1:N
        Z[m_idx, m_idx] += Z[m_idx, m_idx]
        for n_idx in (m_idx + 1):N
            z = Z[m_idx, n_idx] + Z[n_idx, m_idx]
            Z[m_idx, n_idx] = z
            Z[n_idx, m_idx] = z
        end
    end
    return Z
end

"""
Assemble the periodic correction matrix using ΔG = G_per - G_0.
Since ΔG is smooth everywhere, standard product quadrature is used for all entries.

Memory: each source/observation triangle pair `(ts, tn)` is streamed through a
single reused `Nq×Nq` ΔG block (`O(Nq²)`, independent of `Nt`) and scattered into
one global `N×N` result through per-task row buffers. The scratch is
`O(nthreads·(Nq² + dmax·N))`, where `dmax` is the maximum number of RWGs incident
on one triangle, rather than `O(nthreads·N²)`. Each ΔG triangle-pair block is still
evaluated exactly once (Ewald sums are unchanged).

The auxiliary-cache estimate includes quadrature geometry, compact
triangle-to-RWG incidence, cached spatial/spectral Ewald terms, row locks, and
all task slabs/row buffers. It excludes the dense result, which the public
entry point charges to `max_work_bytes`.

Symmetry: when `_periodic_correction_is_symmetric` holds, only target triangles
`tn ≥ ts` are evaluated, halving the Ewald work. The missing triangle orientation
is restored after the parallel sweep from reciprocity. Otherwise the full `tn`
sweep is used.
"""
function _assemble_periodic_correction(mesh::TriMesh, rwg::RWGData, k,
                                       lattice::PeriodicLattice;
                                       quad_order::Int=3,
                                       eta0::Float64=376.730313668,
                                       max_cache_bytes::Integer=
                                           _DEFAULT_MAX_PERIODIC_EFIE_CACHE_BYTES,
                                       max_green_terms::Integer=
                                           _DEFAULT_MAX_PERIODIC_GREEN_TERMS)
    kw = _validated_lattice_wavenumber(k, lattice)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    _, inv_k2, omega_mu0 = _validated_efie_prefactors(kw, eta0)
    cache, _, _ = _build_periodic_efie_geometry_cache(
        mesh, rwg, lattice;
        quad_order=quad_order,
        point_cache_count=1,
        image_block=false,
        max_cache_bytes=max_cache_bytes,
        max_green_terms=max_green_terms)
    wq = cache.wq
    Nq = cache.Nq
    quad_pts = cache.quad_pts
    areas = cache.areas
    div_vals = cache.div_vals
    rwg_vals = cache.rwg_vals
    incidence = cache.incidence

    CT = ComplexF64
    symmetric = _periodic_correction_is_symmetric(rwg, lattice)

    # Partition source triangles across tasks. Each task owns only a small
    # `max_incident×N` row buffer and one quadrature slab; completed rows are
    # merged under per-output-row locks. `@spawn` tasks may migrate between
    # threads, so scratch is bound to chunk index `c`, not `threadid()`.
    # Interleaved chunks balance the triangular symmetric sweep.
    ntasks = cache.ntasks
    max_incident = incidence.max_incident
    Z_corr = zeros(CT, N, N)
    row_locks = [Threads.SpinLock() for _ in 1:N]

    @sync for c in 1:ntasks
        Threads.@spawn begin
            slab = Matrix{CT}(undef, Nq, Nq)
            row_buffer = zeros(CT, max_incident, N)
            for ts in c:ntasks:Nt
                incident_s = _periodic_incidence_range(incidence, ts)
                isempty(incident_s) && continue
                fill!(row_buffer, zero(CT))
                Am = areas[ts]

                tn_start = symmetric ? ts : 1
                @inbounds for tn in tn_start:Nt
                    incident_t = _periodic_incidence_range(incidence, tn)
                    isempty(incident_t) && continue
                    An = areas[tn]
                    for qn in 1:Nq, qm in 1:Nq
                        slab[qm, qn] =
                            _greens_periodic_correction_cached(
                                quad_pts[ts][qm], quad_pts[tn][qn], kw, lattice,
                                cache.spatial_terms, cache.spectral_terms)
                    end

                    wAA = (2 * Am) * (2 * An)
                    for (source_slot, source_idx) in enumerate(incident_s)
                        m_idx, itm = incidence.entries[source_idx]
                        dvm = div_vals[itm, m_idx]
                        fm_vals = itm == 1 ? rwg_vals[m_idx][1] : rwg_vals[m_idx][2]
                        conj_dvm_ik2 = conj(dvm) * inv_k2

                        for target_idx in incident_t
                            n_idx, itn = incidence.entries[target_idx]
                            dvn = div_vals[itn, n_idx]
                            fn_vals = itn == 1 ? rwg_vals[n_idx][1] : rwg_vals[n_idx][2]
                            dvmn_inv_k2 = conj_dvm_ik2 * dvn

                            val = zero(CT)
                            for qm in 1:Nq
                                fm = fm_vals[qm]
                                wqm = wq[qm]
                                for qn in 1:Nq
                                    fn = fn_vals[qn]
                                    dG = slab[qm, qn]
                                    vec_part = dot(fm, fn) * dG
                                    scl_part = dvmn_inv_k2 * dG
                                    val += (vec_part - scl_part) * (wqm * wq[qn])
                                end
                            end
                            val *= wAA

                            # Same-triangle blocks occur in both P and transpose(P)
                            # during symmetry completion, so store half here.
                            if symmetric && tn == ts
                                val *= 0.5
                            end
                            row_buffer[source_slot, n_idx] += val
                        end
                    end
                end
                _merge_periodic_triangle_rows!(
                    Z_corr, row_locks, row_buffer, incidence, incident_s)
            end
        end
    end

    symmetric && _complete_periodic_triangle_symmetry!(Z_corr)
    Z_corr .*= -1im * omega_mu0

    all(isfinite, Z_corr) ||
        throw(OverflowError(
            "periodic EFIE correction contains entries outside the " *
            "representable ComplexF64 range"))

    return Z_corr
end
