# SurfaceIE3D.jl -- Dense dielectric surface integral equations
#
# This file implements closed-surface PMCHWT and Muller block systems using RWG
# electric and magnetic equivalent surface currents. The electric-current EFIE
# blocks reuse the package's singularity-treated EFIE assembly. The magnetic
# field K operator is assembled by principal-value product quadrature, with
# higher-order quadrature on near-singular panel pairs (those sharing at least
# one vertex: self, edge-adjacent, and vertex-touching).

export DielectricMedium3D, DielectricSIEResult3D
export dielectric_medium_3d, assemble_magnetic_field_operator_3d
export MatrixFreeMagneticFieldOperator3D, MatrixFreeDielectricSIE3D
export matrixfree_magnetic_field_operator_3d, matrixfree_dielectric_sie_operator_3d
export assemble_dielectric_sie_rhs_3d, assemble_dielectric_sie_3d
export assemble_pmchwt_3d, assemble_muller_3d, solve_dielectric_sie_3d

struct DielectricMedium3D
    eps_r::ComplexF64
    mu_r::ComplexF64
    k::ComplexF64
    eta::ComplexF64
end

struct DielectricSIEResult3D{TA<:AbstractMatrix{ComplexF64},TLU,TStats}
    J::Vector{ComplexF64}
    M::Vector{ComplexF64}
    A::TA
    rhs::Vector{ComplexF64}
    A_LU::TLU
    solver::Symbol
    stats::TStats
    formulation::Symbol
    exterior::DielectricMedium3D
    interior::DielectricMedium3D
end

struct MatrixFreeMagneticFieldOperator3D{
        TRWG<:RWGData} <: AbstractMatrix{ComplexF64}
    mesh::TriMesh
    rwg::TRWG
    k::ComplexF64
    wq::Vector{Float64}
    pts::Vector{Vector{Vec3}}
    areas::Vector{Float64}
    wq_hi::Vector{Float64}
    pts_hi::Vector{Vector{Vec3}}
    areas_hi::Vector{Float64}
    near_pairs::TriangleAdjacency
end

# Matches the exact ComplexF64 row-reduction bound used by the matrix-free
# EFIE primitive. The ordinary path remains allocation-free; only an exponent-
# range overflow or an overflowing sum of term magnitudes enters this path.
const _MFIE_MATVEC_FALLBACK_PRECISION_3D = 4352
const _DEFAULT_MAX_SURFACE_CACHE_BYTES_3D = 2_000_000_000
const _DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D = 20_000_000
const _SURFACE_EMPTY_VECTOR_BYTES_3D = Base.summarysize(Vec3[])

mutable struct MatrixFreeDielectricSIE3D{
        TZe<:MatrixFreeEFIEOperator,
        TZh<:MatrixFreeEFIEOperator,
        TK<:MatrixFreeMagneticFieldOperator3D,
        TG<:AbstractMatrix{ComplexF64}} <: AbstractMatrix{ComplexF64}
    formulation::Symbol
    exterior::DielectricMedium3D
    interior::DielectricMedium3D
    Ze_ext::TZe
    Ze_int::TZe
    Zh_ext::TZh
    Zh_int::TZh
    K_ext::TK
    K_int::TK
    c_ze_ext::ComplexF64
    c_ze_int::ComplexF64
    c_zh_ext::ComplexF64
    c_zh_int::ComplexF64
    # Second-kind identity (nhat x Gram) residue omitted by the PV K operator.
    # c_g_e/c_g_h are the off-diagonal Gram coefficients for the E-/H-rows.
    # For PMCHWT they are zero (equal row weights); the matvec then skips Gram.
    Gram::TG
    c_g_e::ComplexF64
    c_g_h::ComplexF64
    work_J::Vector{ComplexF64}
    work_M::Vector{ComplexF64}
    tmp1::Vector{ComplexF64}
    tmp2::Vector{ComplexF64}
    tmp3::Vector{ComplexF64}
    tmp4::Vector{ComplexF64}
    tmp5::Vector{ComplexF64}
    work_lock::ReentrantLock
end

function MatrixFreeDielectricSIE3D(
        formulation::Symbol,
        exterior::DielectricMedium3D,
        interior::DielectricMedium3D,
        Ze_ext::TZe,
        Ze_int::TZe,
        Zh_ext::TZh,
        Zh_int::TZh,
        K_ext::TK,
        K_int::TK,
        c_ze_ext::ComplexF64,
        c_ze_int::ComplexF64,
        c_zh_ext::ComplexF64,
        c_zh_int::ComplexF64,
        Gram::TG,
        c_g_e::ComplexF64,
        c_g_h::ComplexF64,
        work_J::Vector{ComplexF64},
        work_M::Vector{ComplexF64},
        tmp1::Vector{ComplexF64},
        tmp2::Vector{ComplexF64},
        tmp3::Vector{ComplexF64},
        tmp4::Vector{ComplexF64},
        tmp5::Vector{ComplexF64},
    ) where {
        TZe<:MatrixFreeEFIEOperator,
        TZh<:MatrixFreeEFIEOperator,
        TK<:MatrixFreeMagneticFieldOperator3D,
        TG<:AbstractMatrix{ComplexF64},
    }
    return MatrixFreeDielectricSIE3D{TZe,TZh,TK,TG}(
        formulation, exterior, interior,
        Ze_ext, Ze_int, Zh_ext, Zh_int, K_ext, K_int,
        c_ze_ext, c_ze_int, c_zh_ext, c_zh_int,
        Gram, c_g_e, c_g_h,
        work_J, work_M, tmp1, tmp2, tmp3, tmp4, tmp5,
        ReentrantLock(),
    )
end

function _finite_complex_surface_3d(x, label::AbstractString)
    z = ComplexF64(x)
    isfinite(real(z)) && isfinite(imag(z)) ||
        error("$label must be finite, got $z.")
    return z
end

function _validated_surface_wavenumber_3d(k)
    k isa Number ||
        throw(ArgumentError(
            "magnetic-field wavenumber must be numeric, got $(typeof(k))"))
    kc = try
        ComplexF64(k)
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError(
            "magnetic-field wavenumber is outside the ComplexF64 range"))
    end
    isfinite(real(kc)) && isfinite(imag(kc)) ||
        throw(ArgumentError(
            "magnetic-field wavenumber must be finite, got $kc"))
    return kc
end

const _SURFACE_MEDIUM_FALLBACK_PRECISION = 256

function _dielectric_medium_bigfloat_3d(k0::Float64, eta0::Float64,
                                         eps_r::ComplexF64,
                                         mu_r::ComplexF64)
    return setprecision(BigFloat, _SURFACE_MEDIUM_FALLBACK_PRECISION) do
        eps_b = Complex{BigFloat}(eps_r)
        mu_b = Complex{BigFloat}(mu_r)
        k_medium = ComplexF64(BigFloat(k0) * sqrt(eps_b * mu_b))
        eta_medium = ComplexF64(BigFloat(eta0) * sqrt(mu_b / eps_b))
        return k_medium, eta_medium
    end
end

function dielectric_medium_3d(k0::Real, eps_r=1.0 + 0im, mu_r=1.0 + 0im;
                              eta0::Real=_ETA0_DDA)
    k0f = Float64(k0)
    isfinite(k0f) && k0f > 0 ||
        error("k0 must be finite and positive.")
    eta0f = Float64(eta0)
    isfinite(eta0f) && eta0f > 0 ||
        error("eta0 must be finite and positive.")
    epsc = _finite_complex_surface_3d(eps_r, "eps_r")
    muc = _finite_complex_surface_3d(mu_r, "mu_r")
    abs(epsc) > 0 || error("eps_r must be nonzero.")
    abs(muc) > 0 || error("mu_r must be nonzero.")
    k_medium = k0f * sqrt(epsc * muc)
    eta_medium = eta0f * sqrt(muc / epsc)
    if !(isfinite(k_medium) && !iszero(k_medium) &&
         isfinite(eta_medium) && !iszero(eta_medium))
        k_medium, eta_medium = _dielectric_medium_bigfloat_3d(
            k0f, eta0f, epsc, muc)
    end
    isfinite(k_medium) && !iszero(k_medium) ||
        error("derived medium wavenumber must be finite and nonzero; got $k_medium.")
    isfinite(eta_medium) && !iszero(eta_medium) ||
        error("derived medium wave impedance must be finite and nonzero; got $eta_medium.")
    return DielectricMedium3D(epsc, muc, k_medium, eta_medium)
end

function _assert_closed_surface_sie_3d(mesh::TriMesh, rwg::RWGData;
                                       mesh_precheck::Bool=true,
                                       area_tol_rel::Float64=1e-12)
    rwg.has_periodic_bloch &&
        error("Dielectric 3D SIE requires non-periodic closed-surface RWG basis functions.")
    mesh === rwg.mesh || error("RWG data must be built from the same mesh object.")
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=false,
            require_closed=true,
            area_tol_rel=area_tol_rel,
        )
    end
    return nothing
end

# Triangle pairs that are near-singular for the magnetic-field K kernel, i.e.
# pairs sharing at least one vertex (this subsumes edge-sharing pairs, which
# share two vertices). These pairs use the higher-order quadrature rule. The
# 1/R singularity of grad(G) makes both edge-touching AND vertex-touching pairs
# under-integrated by the coarse far rule; promoting vertex-touching pairs to
# the high-order rule measurably reduces the K-operator quadrature error toward
# a fine-quadrature reference (verified ~0.99%->0.27% on an icosphere).
function _surface_cache_work_bytes_3d(
        fixed_payload_bytes::Int,
        triangle_count::Int,
        pair_records::Int)
    fixed_payload_bytes >= 0 ||
        throw(ArgumentError("fixed surface-cache payload must be nonnegative"))
    triangle_count >= 0 ||
        throw(ArgumentError("triangle count must be nonnegative"))
    pair_records >= 0 ||
        throw(ArgumentError("near-pair record count must be nonnegative"))

    vertex_record_bytes = BigInt(3) * triangle_count *
                          sizeof(NTuple{2,Int})
    pair_bytes = BigInt(pair_records) * sizeof(NTuple{2,Int})
    degree_bytes = BigInt(triangle_count) * sizeof(Int)
    offset_bytes = (BigInt(triangle_count) + 1) * sizeof(Int)
    neighbor_bytes = BigInt(2) * pair_records * sizeof(Int)
    adjacency_peak = vertex_record_bytes + pair_bytes + degree_bytes +
                     offset_bytes + neighbor_bytes
    retained_peak = BigInt(fixed_payload_bytes) + offset_bytes +
                    neighbor_bytes
    estimated = cld(5 * max(adjacency_peak, retained_peak), 4)
    estimated <= typemax(Int) ||
        throw(ArgumentError("surface-cache storage estimate overflows Int"))
    return Int(estimated)
end

function _surface_cache_retained_bytes_3d(
        fixed_payload_bytes::Int,
        triangle_count::Int,
        near_pairs::Int)
    fixed_payload_bytes >= 0 ||
        throw(ArgumentError("fixed surface-cache payload must be nonnegative"))
    triangle_count >= 0 ||
        throw(ArgumentError("triangle count must be nonnegative"))
    near_pairs >= 0 ||
        throw(ArgumentError("near-pair count must be nonnegative"))
    raw_bytes = BigInt(fixed_payload_bytes) +
                (BigInt(triangle_count) + 1) * sizeof(Int) +
                BigInt(2) * near_pairs * sizeof(Int)
    estimated = cld(5 * raw_bytes, 4)
    estimated <= typemax(Int) ||
        throw(ArgumentError(
            "retained magnetic-field cache estimate overflows Int"))
    return Int(estimated)
end

function _triangle_near_pairs_3d(
        mesh::TriMesh;
        fixed_payload_bytes::Int=0,
        max_cache_bytes::Integer=_DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
        max_near_pairs::Integer=_DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    Nt = ntriangles(mesh)
    cache_limit = _validated_resource_limit(
        "max_cache_bytes", max_cache_bytes)
    pair_limit = try
        Int(max_near_pairs)
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError("max_near_pairs is outside the Int range"))
    end
    pair_limit >= 0 ||
        throw(ArgumentError(
            "max_near_pairs must be nonnegative, got $max_near_pairs"))
    minimum_bytes = _surface_cache_work_bytes_3d(
        fixed_payload_bytes, Nt, 0)
    minimum_bytes <= cache_limit ||
        throw(ArgumentError(
            "magnetic-field cache requires at least $minimum_bytes " *
            "estimated bytes, exceeding max_cache_bytes=$cache_limit"))

    nrecords = Base.checked_mul(3, Nt)
    vertex_records = Vector{NTuple{2,Int}}(undef, nrecords)
    record_idx = 1
    for t in 1:Nt
        for iv in 1:3
            vertex_records[record_idx] = (mesh.tri[iv, t], t)
            record_idx += 1
        end
    end
    sort!(vertex_records)

    # Reserve the exact number of candidate pairs before cross-vertex
    # deduplication. Edge-sharing triangles appear once for each shared vertex.
    pair_capacity = 0
    first_record = 1
    while first_record <= nrecords
        next_vertex = first_record + 1
        @inbounds while next_vertex <= nrecords &&
                        vertex_records[next_vertex][1] == vertex_records[first_record][1]
            next_vertex += 1
        end
        degree = next_vertex - first_record
        first_factor, second_factor = iseven(degree) ?
            (degree ÷ 2, degree - 1) :
            (degree, (degree - 1) ÷ 2)
        remaining_pairs = pair_limit - pair_capacity
        if !iszero(first_factor) &&
           second_factor > remaining_pairs ÷ first_factor
            throw(ArgumentError(
                "triangle near-pair construction exceeds " *
                "max_near_pairs=$pair_limit"))
        end
        pair_capacity += first_factor * second_factor
        first_record = next_vertex
    end
    work_bytes = _surface_cache_work_bytes_3d(
        fixed_payload_bytes, Nt, pair_capacity)
    work_bytes <= cache_limit ||
        throw(ArgumentError(
            "magnetic-field cache requires at most $work_bytes estimated " *
            "bytes, exceeding max_cache_bytes=$cache_limit"))

    pairs = NTuple{2,Int}[]
    sizehint!(pairs, pair_capacity)
    first_record = 1
    while first_record <= nrecords
        next_vertex = first_record + 1
        @inbounds while next_vertex <= nrecords &&
                        vertex_records[next_vertex][1] == vertex_records[first_record][1]
            next_vertex += 1
        end
        @inbounds for i in first_record:(next_vertex - 1)
            for j in (i + 1):(next_vertex - 1)
                t1 = vertex_records[i][2]
                t2 = vertex_records[j][2]
                t1 == t2 && continue
                push!(pairs, t1 < t2 ? (t1, t2) : (t2, t1))
            end
        end
        first_record = next_vertex
    end

    return _triangle_adjacency_from_pairs!(pairs, Nt)
end

function _surface_quad_cache_3d(mesh::TriMesh, xi, wq::Vector{Float64})
    Nt = ntriangles(mesh)
    pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    for t in 1:Nt
        pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
    end
    return wq, pts, areas
end

function _surface_cache_fixed_payload_bytes_3d(
        triangle_count::Int,
        quadrature_count::Int,
        singular_quadrature_count::Int)
    triangle_payload = BigInt(triangle_count) * (
        2 * sizeof(Vector{Vec3}) + 2 * _SURFACE_EMPTY_VECTOR_BYTES_3D +
        (quadrature_count + singular_quadrature_count) * sizeof(Vec3) +
        2 * sizeof(Float64))
    weight_payload = BigInt(
        quadrature_count + singular_quadrature_count) * sizeof(Float64)
    total = triangle_payload + weight_payload
    total <= typemax(Int) ||
        throw(ArgumentError(
            "magnetic-field cache fixed-payload estimate overflows Int"))
    return Int(total)
end

function _build_surface_operator_cache_3d(
        mesh::TriMesh,
        quad_order::Int,
        singular_quad_order::Int;
        max_cache_bytes::Integer=_DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
        max_near_pairs::Integer=_DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    xi, wq = tri_quad_rule(quad_order)
    xi_hi, wq_hi = tri_quad_rule(singular_quad_order)
    fixed_payload_bytes = _surface_cache_fixed_payload_bytes_3d(
        ntriangles(mesh), length(wq), length(wq_hi))
    near_pairs = _triangle_near_pairs_3d(
        mesh;
        fixed_payload_bytes=fixed_payload_bytes,
        max_cache_bytes=max_cache_bytes,
        max_near_pairs=max_near_pairs)
    _, pts, areas = _surface_quad_cache_3d(mesh, xi, wq)
    _, pts_hi, areas_hi = _surface_quad_cache_3d(mesh, xi_hi, wq_hi)
    return wq, pts, areas, wq_hi, pts_hi, areas_hi, near_pairs
end

"""
    _ncross_gram_3d(mesh, rwg; quad_order=3)

Assemble the second-kind identity overlap matrix
`G[m,n] = <f_m, nhat x f_n>` integrated over the surface. Because two RWG
functions overlap only on triangles they both support, the integral is taken
over the common triangle(s) using the outward triangle normal `nhat`. This is
the principal-value residue ("nhat x Gram") term that the magnetic-field K
operator omits; in the Muller system it appears off-diagonal with coefficient
`(c_ext - c_int)*(1/2)`. (It cancels identically in PMCHWT, where the two
off-diagonal coefficients are equal, so PMCHWT is unaffected.)
"""
function _ncross_gram_3d(mesh::TriMesh, rwg::RWGData;
                         quad_order::Int=3,
                         max_storage_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    xi, wq = tri_quad_rule(quad_order)
    Nt = ntriangles(mesh)
    N = rwg.nedges
    pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    normals = Vector{Vec3}(undef, Nt)
    for t in 1:Nt
        pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
        normals[t] = triangle_normal(mesh, t)
    end
    tri_to_basis = [Int[] for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n)
        push!(tri_to_basis[rwg.tminus[n]], n)
    end

    nentries = 0
    try
        for basis_on_t in tri_to_basis
            nentries = Base.Checked.checked_add(
                nentries,
                Base.Checked.checked_mul(length(basis_on_t), length(basis_on_t)))
        end
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("nhat x Gram triplet count overflows Int"))
    end
    index_bytes = _checked_array_payload_bytes(
        Int, 2, nentries; label="nhat x Gram row/column triplets")
    value_bytes = _checked_array_payload_bytes(
        ComplexF64, nentries; label="nhat x Gram value triplets")
    triplet_bytes = try
        Base.Checked.checked_add(index_bytes, value_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("nhat x Gram triplet payload estimate overflows Int"))
    end
    _enforce_payload_limit(
        triplet_bytes, max_storage_bytes,
        "nhat x Gram triplets", "max_storage_bytes")

    rows = Vector{Int}(undef, nentries)
    cols = Vector{Int}(undef, nentries)
    vals = Vector{ComplexF64}(undef, nentries)
    entry = 1
    @inbounds for t in 1:Nt
        A = areas[t]
        nh = normals[t]
        ptst = pts[t]
        basis_on_t = tri_to_basis[t]
        for m in basis_on_t, n in basis_on_t
            acc = 0.0 + 0.0im
            if m != n
                twice_area = 2 * A
                if isfinite(twice_area)
                    for q in eachindex(wq)
                        r = ptst[q]
                        fm = eval_rwg(rwg, m, r, t)
                        fn = eval_rwg(rwg, n, r, t)
                        acc += wq[q] * dot(fm, cross(nh, fn)) * twice_area
                    end
                else
                    for q in eachindex(wq)
                        r = ptst[q]
                        fm = eval_rwg(rwg, m, r, t)
                        fn = eval_rwg(rwg, n, r, t)
                        acc += wq[q] * dot(fm, cross(nh, fn))
                    end
                    acc = _local_surface_mass_scale(acc, A, t, m, n)
                end
            end
            rows[entry] = m
            cols[entry] = n
            vals[entry] = acc
            entry += 1
        end
    end
    entry == nentries + 1 || error("internal nhat x Gram triplet count mismatch")
    return LocalMassMatrix(N, rows, cols, vals)
end

@inline function _mfie_triangle_pair_entry_3d(rwg::RWGData, m::Int, n::Int,
                                             tm::Int, tn::Int,
                                             k, wq, pts, areas)
    val = 0.0 + 0.0im
    Am = areas[tm]
    An = areas[tn]
    @inbounds for qm in eachindex(wq)
        rm = pts[tm][qm]
        fm = eval_rwg(rwg, m, rm, tm)
        for qn in eachindex(wq)
            rn = pts[tn][qn]
            fn = eval_rwg(rwg, n, rn, tn)
            kernel = cross(_grad_greens_unchecked(rm, rn, k), fn)
            val += wq[qm] * wq[qn] * dot(fm, kernel) * (2 * Am) * (2 * An)
        end
    end
    return val
end

@inline function _finalize_mfie_entry_3d(value, m::Int, n::Int)
    entry = ComplexF64(value)
    isfinite(entry) ||
        throw(OverflowError(
            "magnetic-field operator entry ($m, $n) is outside the " *
            "representable ComplexF64 range"))
    return entry
end

"""
    assemble_magnetic_field_operator_3d(mesh, rwg, k; quad_order=3,
                                        max_output_bytes=2_000_000_000,
                                        max_cache_bytes=2_000_000_000,
                                        max_near_pairs=20_000_000)

Assemble the dense magnetic-field principal-value operator
`K[m,n] = <f_m, PV ∫ grad(G) x f_n dS'>`. This is the off-diagonal surface
current coupling used in PMCHWT/Muller systems.
"""
function assemble_magnetic_field_operator_3d(mesh::TriMesh, rwg::RWGData, k;
                                             quad_order::Int=3,
                                             singular_quad_order::Int=7,
                                             mesh_precheck::Bool=true,
                                             area_tol_rel::Float64=1e-12,
                                             max_output_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                             max_cache_bytes::Integer=
                                                 _DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
                                             max_near_pairs::Integer=
                                                 _DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    kc = _validated_surface_wavenumber_3d(k)
    output_bytes = _checked_array_payload_bytes(
        ComplexF64, rwg.nedges, rwg.nedges;
        label="magnetic-field operator matrix")
    _enforce_payload_limit(
        output_bytes, max_output_bytes,
        "magnetic-field operator matrix", "max_output_bytes")
    _assert_closed_surface_sie_3d(mesh, rwg;
                                  mesh_precheck=mesh_precheck,
                                  area_tol_rel=area_tol_rel)
    N = rwg.nedges
    wq, pts, areas, wq_hi, pts_hi, areas_hi, near_pairs =
        _build_surface_operator_cache_3d(
            mesh, quad_order, singular_quad_order;
            max_cache_bytes=max_cache_bytes,
            max_near_pairs=max_near_pairs)
    K = zeros(ComplexF64, N, N)

    Threads.@threads for m in 1:N
        @inbounds for n in 1:N
            acc = 0.0 + 0.0im
            for tm in (rwg.tplus[m], rwg.tminus[m])
                for tn in (rwg.tplus[n], rwg.tminus[n])
                    if tm == tn || near_pairs[tm, tn]
                        acc += _mfie_triangle_pair_entry_3d(
                            rwg, m, n, tm, tn, kc, wq_hi, pts_hi, areas_hi,
                        )
                    else
                        acc += _mfie_triangle_pair_entry_3d(
                            rwg, m, n, tm, tn, kc, wq, pts, areas,
                        )
                    end
                end
            end
            K[m, n] = _finalize_mfie_entry_3d(acc, m, n)
        end
    end
    return K
end

function matrixfree_magnetic_field_operator_3d(mesh::TriMesh, rwg::RWGData, k;
                                               quad_order::Int=3,
                                               singular_quad_order::Int=7,
                                               mesh_precheck::Bool=true,
                                               area_tol_rel::Float64=1e-12,
                                               max_cache_bytes::Integer=
                                                   _DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
                                               max_near_pairs::Integer=
                                                   _DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    kc = _validated_surface_wavenumber_3d(k)
    _assert_closed_surface_sie_3d(mesh, rwg;
                                  mesh_precheck=mesh_precheck,
                                  area_tol_rel=area_tol_rel)
    wq, pts, areas, wq_hi, pts_hi, areas_hi, near_pairs =
        _build_surface_operator_cache_3d(
            mesh, quad_order, singular_quad_order;
            max_cache_bytes=max_cache_bytes,
            max_near_pairs=max_near_pairs)
    return MatrixFreeMagneticFieldOperator3D(mesh, rwg, kc,
                                             wq, pts, areas,
                                             wq_hi, pts_hi, areas_hi,
                                             near_pairs)
end

function _matrixfree_magnetic_field_operator_with_wavenumber_3d(
        base::MatrixFreeMagneticFieldOperator3D,
        k)
    kc = _validated_surface_wavenumber_3d(k)
    return MatrixFreeMagneticFieldOperator3D(
        base.mesh,
        base.rwg,
        kc,
        base.wq,
        base.pts,
        base.areas,
        base.wq_hi,
        base.pts_hi,
        base.areas_hi,
        base.near_pairs,
    )
end

function _matrixfree_efie_operator_with_prefactors_3d(
        base::MatrixFreeEFIEOperator,
        k,
        eta0)
    cache = _efie_cache_with_prefactors(base.cache, k, eta0)
    return MatrixFreeEFIEOperator{ComplexF64,typeof(cache)}(cache)
end

function _estimated_surface_payload_bytes_3d(
        payload_bytes::Int,
        label::AbstractString)
    payload_bytes >= 0 ||
        throw(ArgumentError("$label payload must be nonnegative"))
    estimated = cld(5 * BigInt(payload_bytes), 4)
    estimated <= typemax(Int) ||
        throw(ArgumentError("$label payload estimate overflows Int"))
    return Int(estimated)
end

Base.size(A::MatrixFreeMagneticFieldOperator3D) = (A.rwg.nedges, A.rwg.nedges)
Base.eltype(::Type{<:MatrixFreeMagneticFieldOperator3D}) = ComplexF64
Base.eltype(::MatrixFreeMagneticFieldOperator3D) = ComplexF64

@inline function _mfie_entry_3d(A::MatrixFreeMagneticFieldOperator3D, m::Int, n::Int)
    1 <= m <= A.rwg.nedges || throw(BoundsError(A, (m, n)))
    1 <= n <= A.rwg.nedges || throw(BoundsError(A, (m, n)))
    acc = 0.0 + 0.0im
    for tm in (A.rwg.tplus[m], A.rwg.tminus[m])
        for tn in (A.rwg.tplus[n], A.rwg.tminus[n])
            if tm == tn || A.near_pairs[tm, tn]
                acc += _mfie_triangle_pair_entry_3d(
                    A.rwg, m, n, tm, tn, A.k, A.wq_hi, A.pts_hi, A.areas_hi,
                )
            else
                acc += _mfie_triangle_pair_entry_3d(
                    A.rwg, m, n, tm, tn, A.k, A.wq, A.pts, A.areas,
                )
            end
        end
    end
    return _finalize_mfie_entry_3d(acc, m, n)
end

Base.getindex(A::MatrixFreeMagneticFieldOperator3D, i::Int, j::Int) =
    _mfie_entry_3d(A, i, j)

@noinline function _matrixfree_mfie_row_bigfloat_3d(
        A::MatrixFreeMagneticFieldOperator3D,
        x::AbstractVector,
        row::Int)
    return setprecision(BigFloat, _MFIE_MATVEC_FALLBACK_PRECISION_3D) do
        total = zero(Complex{BigFloat})
        @inbounds for column in axes(A, 2)
            total += Complex{BigFloat}(_mfie_entry_3d(A, row, column)) *
                     Complex{BigFloat}(x[column])
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "matrix-free magnetic-field operator output is outside " *
                "the representable ComplexF64 range at row $row."))
        return converted
    end
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::MatrixFreeMagneticFieldOperator3D,
                            x::AbstractVector)
    N = size(A, 1)
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))
    xread = Base.mightalias(y, x) ? copy(x) : x
    @inbounds for m in 1:N
        acc = 0.0 + 0.0im
        magnitude_sum = 0.0
        needs_fallback = false
        try
            for n in 1:N
                term = _mfie_entry_3d(A, m, n) * xread[n]
                next_acc = acc + term
                magnitude_sum += max(abs(real(term)), abs(imag(term)))
                if !isfinite(next_acc) || !isfinite(magnitude_sum)
                    needs_fallback = true
                    break
                end
                acc = next_acc
            end
        catch err
            err isa OverflowError || rethrow()
            needs_fallback = true
        end
        y[m] = needs_fallback ?
               _matrixfree_mfie_row_bigfloat_3d(A, xread, m) : acc
    end
    return y
end

function Base.:*(A::MatrixFreeMagneticFieldOperator3D, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, x)
    return y
end

function _assemble_plane_wave_h_rhs_3d(mesh::TriMesh, rwg::RWGData,
                                       pw::PlaneWaveExcitation, eta::Complex;
                                       quad_order::Int=3)
    plane_wave_geometry = _validate_plane_wave_excitation_geometry(pw)
    isfinite(real(eta)) && isfinite(imag(eta)) && !iszero(eta) ||
        throw(ArgumentError(
            "exterior wave impedance must be finite and nonzero, got $eta."))
    N = rwg.nedges
    wq, quad_pts, areas = _excitation_quadrature_cache(mesh, quad_order)
    khat = plane_wave_geometry.direction
    rhs = zeros(ComplexF64, N)
    for n in 1:N
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = areas[t]
            pts = quad_pts[t]
            for q in eachindex(wq)
                rq = pts[q]
                fn = eval_rwg(rwg, n, rq, t)
                Einc = _plane_wave_field_unchecked(
                    rq, pw.k_vec, pw.E0, pw.pol)
                Hinc = cross(khat, Einc) / eta
                rhs[n] += _excitation_surface_term(fn, Hinc, A, wq[q])
            end
        end
    end
    all(isfinite, rhs) ||
        throw(OverflowError(
            "magnetic plane-wave RHS contains entries outside the " *
            "representable ComplexF64 range"))
    return rhs
end

@inline function _validated_surface_sie_formulation_3d(
        formulation::Symbol)
    formulation in (:pmchwt, :muller) ||
        throw(ArgumentError(
            "formulation must be :pmchwt or :muller, got $formulation."))
    return formulation
end

"""
    assemble_dielectric_sie_rhs_3d(mesh, rwg, excitation, exterior; quad_order=3,
                                   formulation=:pmchwt, interior=nothing)

Assemble the PMCHWT/Muller right-hand side `[v_E; v_H]` for a plane-wave
incident field in the exterior medium.

The incident field lives only in the exterior region, so the tested forcing is
`[v_E; v_H]` for PMCHWT. For Muller the exterior equations are scaled by the
exterior row weights, so the RHS becomes `[c_ze_ext*v_E; c_zh_ext*v_H]`
consistently with the weighted block matrix. Pass the `interior` medium to
enable the Muller scaling.
"""
function assemble_dielectric_sie_rhs_3d(mesh::TriMesh, rwg::RWGData,
                                        excitation::PlaneWaveExcitation,
                                        exterior::DielectricMedium3D;
                                        quad_order::Int=3,
                                        formulation::Symbol=:pmchwt,
                                        interior::Union{Nothing,DielectricMedium3D}=nothing)
    _validated_surface_sie_formulation_3d(formulation)
    formulation == :muller && interior === nothing &&
        throw(ArgumentError(
            "Muller RHS scaling requires the interior medium."))
    isfinite(real(exterior.k)) && isfinite(imag(exterior.k)) ||
        throw(ArgumentError(
            "exterior wavenumber must be finite, got $(exterior.k)."))
    isfinite(real(exterior.eta)) && isfinite(imag(exterior.eta)) &&
        !iszero(exterior.eta) ||
        throw(ArgumentError(
            "exterior wave impedance must be finite and nonzero, got $(exterior.eta)."))
    exterior_k_imag_tol = max(1e-10 * max(abs(real(exterior.k)), 1.0), 1e-12)
    abs(imag(exterior.k)) <= exterior_k_imag_tol ||
        throw(ArgumentError(
            "PlaneWaveExcitation uses a real k_vec and requires a real exterior wavenumber; got $(exterior.k)."))
    _validate_plane_wave_wavenumber(
        excitation, abs(real(exterior.k)), "dielectric SIE")

    vE = assemble_excitation(mesh, rwg, excitation; quad_order=quad_order)
    vH = _assemble_plane_wave_h_rhs_3d(mesh, rwg, excitation, exterior.eta;
                                       quad_order=quad_order)
    rhs = if formulation == :muller
        c_ze_ext, _, c_zh_ext, _ =
            _surface_sie_coefficients_3d(:muller, exterior, interior)
        vcat(c_ze_ext .* vE, c_zh_ext .* vH)
    else
        vcat(vE, vH)
    end
    all(isfinite, rhs) ||
        throw(OverflowError(
            "dielectric SIE RHS contains entries outside the " *
            "representable ComplexF64 range"))
    return rhs
end

function _validated_surface_sie_dense_work_3d(
        N::Int,
        matrix_count::Int,
        max_work_bytes::Integer,
        label::AbstractString)
    work_bytes = _checked_array_payload_bytes(
        ComplexF64, matrix_count, N, N;
        label="$label work matrices")
    work_limit = _validated_resource_limit(
        "max_work_bytes", max_work_bytes)
    _enforce_payload_limit(
        work_bytes, work_limit,
        "$label work matrices", "max_work_bytes")
    return work_limit
end

function _surface_sie_blocks_3d(mesh::TriMesh, rwg::RWGData, k0::Real,
                                epsr_in, mur_in, epsr_ext, mur_ext;
                                formulation::Symbol,
                                quad_order::Int=3,
                                singular_quad_order::Int=7,
                                eta0::Real=_ETA0_DDA,
                                mesh_precheck::Bool=true,
                                area_tol_rel::Float64=1e-12,
                                max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                max_cache_bytes::Integer=
                                    _DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
                                max_adjacency_pairs::Integer=
                                    _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                                max_near_pairs::Integer=
                                    _DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    _validated_surface_sie_formulation_3d(formulation)
    work_limit = _validated_surface_sie_dense_work_3d(
        rwg.nedges, 10, max_work_bytes,
        "dense dielectric SIE assembly")
    _assert_closed_surface_sie_3d(mesh, rwg;
                                  mesh_precheck=mesh_precheck,
                                  area_tol_rel=area_tol_rel)

    exterior = dielectric_medium_3d(k0, epsr_ext, mur_ext; eta0=eta0)
    interior = dielectric_medium_3d(k0, epsr_in, mur_in; eta0=eta0)

    Ze_ext = assemble_Z_efie(mesh, rwg, exterior.k;
                             quad_order=quad_order,
                             eta0=exterior.eta,
                             mesh_precheck=false,
                             max_output_bytes=work_limit,
                             max_cache_bytes=max_cache_bytes,
                             max_adjacency_pairs=max_adjacency_pairs)
    Ze_int = assemble_Z_efie(mesh, rwg, interior.k;
                             quad_order=quad_order,
                             eta0=interior.eta,
                             mesh_precheck=false,
                             max_output_bytes=work_limit,
                             max_cache_bytes=max_cache_bytes,
                             max_adjacency_pairs=max_adjacency_pairs)
    Zh_ext = assemble_Z_efie(mesh, rwg, exterior.k;
                             quad_order=quad_order,
                             eta0=1 / exterior.eta,
                             mesh_precheck=false,
                             max_output_bytes=work_limit,
                             max_cache_bytes=max_cache_bytes,
                             max_adjacency_pairs=max_adjacency_pairs)
    Zh_int = assemble_Z_efie(mesh, rwg, interior.k;
                             quad_order=quad_order,
                             eta0=1 / interior.eta,
                             mesh_precheck=false,
                             max_output_bytes=work_limit,
                             max_cache_bytes=max_cache_bytes,
                             max_adjacency_pairs=max_adjacency_pairs)
    K_ext = assemble_magnetic_field_operator_3d(
        mesh, rwg, exterior.k;
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        mesh_precheck=false,
        max_output_bytes=work_limit,
        max_cache_bytes=max_cache_bytes,
        max_near_pairs=max_near_pairs,
    )
    K_int = assemble_magnetic_field_operator_3d(
        mesh, rwg, interior.k;
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        mesh_precheck=false,
        max_output_bytes=work_limit,
        max_cache_bytes=max_cache_bytes,
        max_near_pairs=max_near_pairs,
    )

    # Row weights (PMCHWT => all 1; Muller => mu/eps weights). The SAME weights
    # multiply the diagonal T blocks and the off-diagonal K blocks of each region.
    c_ze_ext, c_ze_int, c_zh_ext, c_zh_int =
        _surface_sie_coefficients_3d(formulation, exterior, interior)

    # Fill the returned block matrix in place. Together with the six regional
    # operators above this keeps the dense resident payload to ten N-by-N
    # matrices; scalar combinations do not create additional dense temporaries.
    N = rwg.nedges
    A = Matrix{ComplexF64}(undef, 2N, 2N)
    A11 = @view A[1:N, 1:N]
    A12 = @view A[1:N, (N + 1):(2N)]
    A21 = @view A[(N + 1):(2N), 1:N]
    A22 = @view A[(N + 1):(2N), (N + 1):(2N)]
    A11 .= c_ze_ext .* Ze_ext .+ c_ze_int .* Ze_int
    A22 .= c_zh_ext .* Zh_ext .+ c_zh_int .* Zh_int
    A12 .= .-(c_ze_ext .* K_ext .+ c_ze_int .* K_int)
    A21 .= c_zh_ext .* K_ext .+ c_zh_int .* K_int

    # The second-kind identity is local. Keep its O(Nt) triplets compact and
    # accumulate them directly into the two off-diagonal dense blocks.
    if (c_ze_ext != c_ze_int) || (c_zh_ext != c_zh_int)
        Gram = _ncross_gram_3d(
            mesh, rwg;
            quad_order=quad_order,
            max_storage_bytes=work_limit)
        _add_scaled_matrix!(
            A12, -(c_ze_ext - c_ze_int) * 0.5, Gram)
        _add_scaled_matrix!(
            A21, (c_zh_ext - c_zh_int) * 0.5, Gram)
    end
    all(isfinite, A) ||
        throw(OverflowError(
            "dielectric SIE matrix contains entries outside the " *
            "representable ComplexF64 range"))
    return A, exterior, interior
end

@inline function _surface_sie_pair_weights_3d(exterior::ComplexF64,
                                              interior::ComplexF64,
                                              singular_message::String)
    scale = max(abs(real(exterior)), abs(imag(exterior)),
                abs(real(interior)), abs(imag(interior)))
    scale > 0 || error(singular_message)
    exterior_scaled = exterior / scale
    interior_scaled = interior / scale
    sum_scaled = exterior_scaled + interior_scaled
    iszero(sum_scaled) && error(singular_message)
    interior_weight = interior_scaled / sum_scaled
    exterior_weight = exterior_scaled / sum_scaled
    isfinite(interior_weight) && isfinite(exterior_weight) ||
        error("Muller formulation produced non-finite material weights.")
    return interior_weight, exterior_weight
end

function _surface_sie_coefficients_3d(formulation::Symbol,
                                      exterior::DielectricMedium3D,
                                      interior::DielectricMedium3D)
    if formulation == :pmchwt
        return (1.0 + 0im, 1.0 + 0im, 1.0 + 0im, 1.0 + 0im)
    elseif formulation == :muller
        c_mu_int, c_mu_ext = _surface_sie_pair_weights_3d(
            exterior.mu_r, interior.mu_r,
            "Muller formulation is singular for mu_ext + mu_in = 0.")
        c_eps_int, c_eps_ext = _surface_sie_pair_weights_3d(
            exterior.eps_r, interior.eps_r,
            "Muller formulation is singular for eps_ext + eps_in = 0.")
        return c_mu_int, c_mu_ext, c_eps_int, c_eps_ext
    else
        _validated_surface_sie_formulation_3d(formulation)
        error("unreachable dielectric SIE formulation branch")
    end
end

function matrixfree_dielectric_sie_operator_3d(mesh::TriMesh, rwg::RWGData,
                                               k0::Real, epsr_in=1.0 + 0im;
                                               mur_in=1.0 + 0im,
                                               epsr_ext=1.0 + 0im,
                                               mur_ext=1.0 + 0im,
                                               formulation::Symbol=:pmchwt,
                                               quad_order::Int=3,
                                               singular_quad_order::Int=7,
                                               eta0::Real=_ETA0_DDA,
                                               mesh_precheck::Bool=true,
                                               area_tol_rel::Float64=1e-12,
                                               max_gram_storage_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                               max_cache_bytes::Integer=
                                                   _DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
                                               max_adjacency_pairs::Integer=
                                                   _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                                               max_near_pairs::Integer=
                                                   _DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    _validated_surface_sie_formulation_3d(formulation)
    _assert_closed_surface_sie_3d(mesh, rwg;
                                  mesh_precheck=mesh_precheck,
                                  area_tol_rel=area_tol_rel)
    exterior = dielectric_medium_3d(k0, epsr_ext, mur_ext; eta0=eta0)
    interior = dielectric_medium_3d(k0, epsr_in, mur_in; eta0=eta0)
    c_ze_ext, c_ze_int, c_zh_ext, c_zh_int =
        _surface_sie_coefficients_3d(formulation, exterior, interior)

    N = rwg.nedges
    cache_limit = _validated_resource_limit(
        "max_cache_bytes", max_cache_bytes)
    workspace_payload = _checked_array_payload_bytes(
        ComplexF64, 7, N;
        label="matrix-free dielectric SIE work vectors")
    workspace_bytes = _estimated_surface_payload_bytes_3d(
        workspace_payload, "matrix-free dielectric SIE work vectors")
    workspace_bytes < cache_limit ||
        throw(ArgumentError(
            "matrix-free dielectric SIE work vectors require an estimated " *
            "$workspace_bytes bytes, exceeding max_cache_bytes=$cache_limit"))

    # All four EFIE blocks have identical geometric quadrature/RWG data. Build
    # that data once, then replace only the validated medium prefactors.
    Ze_ext = matrixfree_efie_operator(mesh, rwg, exterior.k;
                                      quad_order=quad_order,
                                      eta0=exterior.eta,
                                      mesh_precheck=false,
                                      max_cache_bytes=
                                          cache_limit - workspace_bytes,
                                      max_adjacency_pairs=max_adjacency_pairs)
    Ze_int = _matrixfree_efie_operator_with_prefactors_3d(
        Ze_ext, interior.k, interior.eta)
    Zh_ext = _matrixfree_efie_operator_with_prefactors_3d(
        Ze_ext, exterior.k, 1 / exterior.eta)
    Zh_int = _matrixfree_efie_operator_with_prefactors_3d(
        Ze_ext, interior.k, 1 / interior.eta)

    efie_cache = Ze_ext.cache
    efie_vector_type = eltype(fieldtype(eltype(efie_cache.rwg_vals), 1))
    efie_fixed_payload = _efie_cache_fixed_payload_bytes(
        N,
        ntriangles(mesh),
        efie_cache.Nq,
        length(efie_cache.wq_hi),
        eltype(efie_cache.div_vals),
        efie_vector_type)
    efie_cache_bytes = _efie_cache_retained_bytes(
        efie_fixed_payload,
        ntriangles(mesh),
        _adjacent_pair_count(efie_cache.adjacent))
    reserved_bytes = try
        Base.Checked.checked_add(workspace_bytes, efie_cache_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "matrix-free dielectric SIE cache estimate overflows Int"))
    end
    reserved_bytes < cache_limit ||
        throw(ArgumentError(
            "matrix-free dielectric SIE EFIE cache and work vectors require " *
            "an estimated $reserved_bytes bytes, exceeding " *
            "max_cache_bytes=$cache_limit"))

    # The two K blocks likewise differ only in wavenumber. Charge the retained
    # EFIE cache and reusable work vectors before building their shared geometry.
    K_ext = matrixfree_magnetic_field_operator_3d(
        mesh, rwg, exterior.k;
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        mesh_precheck=false,
        max_cache_bytes=cache_limit - reserved_bytes,
        max_near_pairs=max_near_pairs,
    )
    magnetic_fixed_payload = _surface_cache_fixed_payload_bytes_3d(
        ntriangles(mesh), length(K_ext.wq), length(K_ext.wq_hi))
    magnetic_cache_bytes = _surface_cache_retained_bytes_3d(
        magnetic_fixed_payload,
        ntriangles(mesh),
        _adjacent_pair_count(K_ext.near_pairs))
    total_cache_bytes = try
        Base.Checked.checked_add(reserved_bytes, magnetic_cache_bytes)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError(
            "matrix-free dielectric SIE cache estimate overflows Int"))
    end
    total_cache_bytes <= cache_limit ||
        throw(ArgumentError(
            "matrix-free dielectric SIE caches and work vectors require an " *
            "estimated $total_cache_bytes bytes, exceeding " *
            "max_cache_bytes=$cache_limit"))
    K_int = _matrixfree_magnetic_field_operator_with_wavenumber_3d(
        K_ext, interior.k)

    # Off-diagonal second-kind identity (nhat x Gram) residue coefficients.
    # E-row K block enters with sign -, H-row with sign +.
    c_g_e = -(c_ze_ext - c_ze_int) * 0.5
    c_g_h = (c_zh_ext - c_zh_int) * 0.5
    Gram = (c_g_e != 0 || c_g_h != 0) ?
        _ncross_gram_3d(
            mesh, rwg;
            quad_order=quad_order,
            max_storage_bytes=max_gram_storage_bytes) :
        zeros(ComplexF64, 0, 0)
    return MatrixFreeDielectricSIE3D(
        formulation, exterior, interior,
        Ze_ext, Ze_int, Zh_ext, Zh_int, K_ext, K_int,
        ComplexF64(c_ze_ext), ComplexF64(c_ze_int),
        ComplexF64(c_zh_ext), ComplexF64(c_zh_int),
        Gram, ComplexF64(c_g_e), ComplexF64(c_g_h),
        zeros(ComplexF64, N), zeros(ComplexF64, N),
        zeros(ComplexF64, N), zeros(ComplexF64, N),
        zeros(ComplexF64, N), zeros(ComplexF64, N),
        zeros(ComplexF64, N),
    )
end

Base.size(A::MatrixFreeDielectricSIE3D) = (2 * A.Ze_ext.cache.rwg.nedges,
                                           2 * A.Ze_ext.cache.rwg.nedges)
Base.eltype(::Type{<:MatrixFreeDielectricSIE3D}) = ComplexF64
Base.eltype(::MatrixFreeDielectricSIE3D) = ComplexF64

function Base.getindex(A::MatrixFreeDielectricSIE3D, row::Int, col::Int)
    N = A.Ze_ext.cache.rwg.nedges
    1 <= row <= 2N || throw(BoundsError(A, (row, col)))
    1 <= col <= 2N || throw(BoundsError(A, (row, col)))
    if row <= N && col <= N
        return A.c_ze_ext * A.Ze_ext[row, col] + A.c_ze_int * A.Ze_int[row, col]
    elseif row <= N
        # E-row off-diagonal: -(c_ze_ext K_ext + c_ze_int K_int) + c_g_e * Gram
        c = col - N
        val = -(A.c_ze_ext * A.K_ext[row, c] + A.c_ze_int * A.K_int[row, c])
        if A.c_g_e != 0
            val += A.c_g_e * A.Gram[row, c]
        end
        return val
    elseif col <= N
        # H-row off-diagonal: (c_zh_ext K_ext + c_zh_int K_int) + c_g_h * Gram
        r = row - N
        val = A.c_zh_ext * A.K_ext[r, col] + A.c_zh_int * A.K_int[r, col]
        if A.c_g_h != 0
            val += A.c_g_h * A.Gram[r, col]
        end
        return val
    else
        r = row - N
        c = col - N
        return A.c_zh_ext * A.Zh_ext[r, c] + A.c_zh_int * A.Zh_int[r, c]
    end
end

@inline function _copy_block_inputs_3d!(J::Vector{ComplexF64}, M::Vector{ComplexF64},
                                       x::AbstractVector)
    N = length(J)
    @inbounds for j in 1:N
        J[j] = ComplexF64(x[j])
        M[j] = ComplexF64(x[N + j])
    end
    return nothing
end

@noinline function _surface_sie_block_sum_bigfloat_3d(
        A::MatrixFreeDielectricSIE3D,
        row::Int,
        electric_row::Bool)
    return setprecision(BigFloat, _MFIE_MATVEC_FALLBACK_PRECISION_3D) do
        total = if electric_row
            Complex{BigFloat}(A.c_ze_ext) *
                Complex{BigFloat}(A.tmp1[row]) +
            Complex{BigFloat}(A.c_ze_int) *
                Complex{BigFloat}(A.tmp2[row]) -
            Complex{BigFloat}(A.c_ze_ext) *
                Complex{BigFloat}(A.tmp3[row]) -
            Complex{BigFloat}(A.c_ze_int) *
                Complex{BigFloat}(A.tmp4[row]) +
            Complex{BigFloat}(A.c_g_e) *
                Complex{BigFloat}(A.tmp5[row])
        else
            Complex{BigFloat}(A.c_zh_ext) *
                Complex{BigFloat}(A.tmp1[row]) +
            Complex{BigFloat}(A.c_zh_int) *
                Complex{BigFloat}(A.tmp2[row]) +
            Complex{BigFloat}(A.c_zh_ext) *
                Complex{BigFloat}(A.tmp3[row]) +
            Complex{BigFloat}(A.c_zh_int) *
                Complex{BigFloat}(A.tmp4[row]) +
            Complex{BigFloat}(A.c_g_h) *
                Complex{BigFloat}(A.tmp5[row])
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "matrix-free dielectric SIE block output is outside the " *
                "representable ComplexF64 range at row " *
                "$(electric_row ? row : length(A.tmp1) + row)."))
        return converted
    end
end

@inline function _surface_sie_block_sum_3d(
        A::MatrixFreeDielectricSIE3D,
        row::Int,
        electric_row::Bool)
    terms = if electric_row
        (
            A.c_ze_ext * A.tmp1[row],
            A.c_ze_int * A.tmp2[row],
            -A.c_ze_ext * A.tmp3[row],
            -A.c_ze_int * A.tmp4[row],
            A.c_g_e * A.tmp5[row],
        )
    else
        (
            A.c_zh_ext * A.tmp1[row],
            A.c_zh_int * A.tmp2[row],
            A.c_zh_ext * A.tmp3[row],
            A.c_zh_int * A.tmp4[row],
            A.c_g_h * A.tmp5[row],
        )
    end

    value = zero(ComplexF64)
    magnitude_sum = 0.0
    @inbounds for term in terms
        next_value = value + term
        magnitude_sum += max(abs(real(term)), abs(imag(term)))
        if !isfinite(next_value) || !isfinite(magnitude_sum)
            return _surface_sie_block_sum_bigfloat_3d(
                A, row, electric_row)
        end
        value = next_value
    end
    return value
end

@noinline function _surface_sie_scaled_output_bigfloat_3d(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    return setprecision(BigFloat, _MFIE_MATVEC_FALLBACK_PRECISION_3D) do
        total = Complex{BigFloat}(alpha_scale) *
                Complex{BigFloat}(value)
        if !overwrite
            total += Complex{BigFloat}(beta_scale) *
                     Complex{BigFloat}(previous)
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "matrix-free dielectric SIE scaled output is outside the " *
                "representable ComplexF64 range at row $row."))
        return converted
    end
end

@inline function _surface_sie_scaled_output_3d(
        value::ComplexF64,
        previous::ComplexF64,
        alpha_scale::Number,
        beta_scale::Number,
        overwrite::Bool,
        row::Int)
    _dda_scaled_output_requires_exact_3d(
        value, previous, alpha_scale, beta_scale, overwrite) &&
        return _surface_sie_scaled_output_bigfloat_3d(
            value, previous, alpha_scale, beta_scale, overwrite, row)

    alpha_term = alpha_scale * value
    if overwrite
        converted = ComplexF64(alpha_term)
        return isfinite(converted) ? converted :
               _surface_sie_scaled_output_bigfloat_3d(
                   value, previous, alpha_scale, beta_scale, true, row)
    end

    beta_term = beta_scale * previous
    combined = alpha_term + beta_term
    magnitude_sum =
        max(abs(real(alpha_term)), abs(imag(alpha_term))) +
        max(abs(real(beta_term)), abs(imag(beta_term)))
    converted = ComplexF64(combined)
    if isfinite(converted) && isfinite(magnitude_sum) &&
       !_scaled_sum_requires_exact(alpha_term, beta_term, combined)
        return converted
    end
    return _surface_sie_scaled_output_bigfloat_3d(
        value, previous, alpha_scale, beta_scale, false, row)
end

@inline function _surface_sie_scale_output_only_3d!(
        output::AbstractVector{ComplexF64}, beta_scale::Number)
    if iszero(beta_scale)
        fill!(output, zero(ComplexF64))
    elseif beta_scale != one(beta_scale)
        @inbounds for row in eachindex(output)
            output[row] = _surface_sie_scaled_output_3d(
                zero(ComplexF64), output[row],
                zero(ComplexF64), beta_scale, false, row)
        end
    end
    return output
end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            A::MatrixFreeDielectricSIE3D,
                            x::AbstractVector{ComplexF64},
                            alpha_scale::Number,
                            beta_scale::Number)
    N2 = size(A, 1)
    N = div(N2, 2)
    length(x) == N2 || throw(DimensionMismatch("x length must be $N2."))
    length(y) == N2 || throw(DimensionMismatch("y length must be $N2."))

    if iszero(alpha_scale)
        return _surface_sie_scale_output_only_3d!(y, beta_scale)
    end

    overwrite = iszero(beta_scale)
    lock(A.work_lock)
    try
        _copy_block_inputs_3d!(A.work_J, A.work_M, x)

        # E-row: c_ze_ext Ze_ext J + c_ze_int Ze_int J
        #        - (c_ze_ext K_ext + c_ze_int K_int) M + c_g_e Gram M
        mul!(A.tmp1, A.Ze_ext, A.work_J)
        mul!(A.tmp2, A.Ze_int, A.work_J)
        mul!(A.tmp3, A.K_ext, A.work_M)
        mul!(A.tmp4, A.K_int, A.work_M)
        if A.c_g_e != 0
            mul!(A.tmp5, A.Gram, A.work_M)
        end
        @inbounds for j in 1:N
            v = _surface_sie_block_sum_3d(A, j, true)
            y[j] = _surface_sie_scaled_output_3d(
                v, y[j], alpha_scale, beta_scale, overwrite, j)
        end

        # H-row: (c_zh_ext K_ext + c_zh_int K_int) J + c_g_h Gram J
        #        + c_zh_ext Zh_ext M + c_zh_int Zh_int M
        mul!(A.tmp1, A.K_ext, A.work_J)
        mul!(A.tmp2, A.K_int, A.work_J)
        mul!(A.tmp3, A.Zh_ext, A.work_M)
        mul!(A.tmp4, A.Zh_int, A.work_M)
        if A.c_g_h != 0
            mul!(A.tmp5, A.Gram, A.work_J)
        end
        @inbounds for j in 1:N
            v = _surface_sie_block_sum_3d(A, j, false)
            idx = N + j
            y[idx] = _surface_sie_scaled_output_3d(
                v, y[idx], alpha_scale, beta_scale, overwrite, idx)
        end
    finally
        unlock(A.work_lock)
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                   A::MatrixFreeDielectricSIE3D,
                   x::AbstractVector{ComplexF64}) =
    LinearAlgebra.mul!(y, A, x, one(ComplexF64), zero(ComplexF64))

function Base.:*(A::MatrixFreeDielectricSIE3D, x::AbstractVector)
    y = zeros(ComplexF64, size(A, 1))
    mul!(y, A, _complex_vector_input(x))
    return y
end

"""
    assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in;
                               formulation=:pmchwt,
                               max_work_bytes=2_000_000_000, ...)

Assemble a dense closed-surface dielectric SIE matrix for isotropic homogeneous
interior/exterior media. Unknowns are stacked RWG coefficients `[J; M]`.
"""
function assemble_dielectric_sie_3d(mesh::TriMesh, rwg::RWGData, k0::Real,
                                    epsr_in=1.0 + 0im;
                                    mur_in=1.0 + 0im,
                                    epsr_ext=1.0 + 0im,
                                    mur_ext=1.0 + 0im,
                                    formulation::Symbol=:pmchwt,
                                    quad_order::Int=3,
                                    singular_quad_order::Int=7,
                                    eta0::Real=_ETA0_DDA,
                                    mesh_precheck::Bool=true,
                                    area_tol_rel::Float64=1e-12,
                                    max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                    max_cache_bytes::Integer=
                                        _DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
                                    max_adjacency_pairs::Integer=
                                        _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                                    max_near_pairs::Integer=
                                        _DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    A, _, _ = _surface_sie_blocks_3d(
        mesh, rwg, k0, epsr_in, mur_in, epsr_ext, mur_ext;
        formulation=formulation,
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        eta0=eta0,
        mesh_precheck=mesh_precheck,
        area_tol_rel=area_tol_rel,
        max_work_bytes=max_work_bytes,
        max_cache_bytes=max_cache_bytes,
        max_adjacency_pairs=max_adjacency_pairs,
        max_near_pairs=max_near_pairs,
    )
    return A
end

assemble_pmchwt_3d(mesh::TriMesh, rwg::RWGData, k0::Real, epsr_in=1.0 + 0im; kwargs...) =
    assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in; formulation=:pmchwt, kwargs...)

# Second-kind Müller system: μ/ε-weighted diagonal T AND off-diagonal K blocks,
# weighted RHS, plus the (n̂× Gram) identity residue the principal-value K operator
# omits. Verified to match PMCHWT currents to <1% on a sphere (tightening under
# refinement), so it is a validated alternative to assemble_pmchwt_3d.
assemble_muller_3d(mesh::TriMesh, rwg::RWGData, k0::Real, epsr_in=1.0 + 0im; kwargs...) =
    assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in; formulation=:muller, kwargs...)

function _split_surface_currents_3d(x::AbstractVector{ComplexF64}, N::Int)
    length(x) == 2N || error("surface-current vector length ($(length(x))) must be 2N ($(2N)).")
    return x[1:N], x[(N + 1):(2N)]
end

"""
    solve_dielectric_sie_3d(mesh, rwg, k0, epsr_in, rhs; formulation=:pmchwt, solver=:direct, ...)

Solve a dense PMCHWT/Muller dielectric SIE system. `rhs` may be either a
length-`2N` vector or a `PlaneWaveExcitation`. Set `solver=:gmres` to use the
matrix-free operator instead of forming the dense matrix.
"""
function solve_dielectric_sie_3d(mesh::TriMesh, rwg::RWGData, k0::Real,
                                 epsr_in, rhs::AbstractVector{<:Number};
                                 mur_in=1.0 + 0im,
                                 epsr_ext=1.0 + 0im,
                                 mur_ext=1.0 + 0im,
                                 formulation::Symbol=:pmchwt,
                                 solver::Symbol=:direct,
                                 quad_order::Int=3,
                                 singular_quad_order::Int=7,
                                 eta0::Real=_ETA0_DDA,
                                 mesh_precheck::Bool=true,
                                 area_tol_rel::Float64=1e-12,
                                 tol::Float64=1e-8,
                                 maxiter::Int=200,
                                 memory::Int=20,
                                 verbose::Bool=false,
                                 check_gmres_convergence::Bool=true,
                                 max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                 max_gram_storage_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                 max_cache_bytes::Integer=
                                     _DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
                                 max_adjacency_pairs::Integer=
                                     _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                                 max_near_pairs::Integer=
                                     _DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    solver in (:direct, :gmres) ||
        error("Unsupported dielectric SIE solver: $solver (expected :direct or :gmres).")
    _validated_surface_sie_formulation_3d(formulation)
    if solver == :direct
        _validated_surface_sie_dense_work_3d(
            rwg.nedges, 10, max_work_bytes,
            "direct dielectric SIE solve")
    end
    rhsv = _copy_finite_complex_vector_3d(
        rhs, 2 * rwg.nedges, "rhs")
    if solver == :direct
        A, exterior, interior = _surface_sie_blocks_3d(
            mesh, rwg, k0, epsr_in, mur_in, epsr_ext, mur_ext;
            formulation=formulation,
            quad_order=quad_order,
            singular_quad_order=singular_quad_order,
            eta0=eta0,
            mesh_precheck=mesh_precheck,
            area_tol_rel=area_tol_rel,
            max_work_bytes=max_work_bytes,
            max_cache_bytes=max_cache_bytes,
            max_adjacency_pairs=max_adjacency_pairs,
            max_near_pairs=max_near_pairs,
        )
        fac = _factor_dense_linear_system(
            A, ComplexF64, "direct dielectric SIE factorization")
        x = _solve_factored_linear_system(
            fac, A, rhsv, "direct dielectric SIE solution")
        J, M = _split_surface_currents_3d(x, rwg.nedges)
        return DielectricSIEResult3D(copy(J), copy(M), A, rhsv, fac,
                                     :direct, nothing,
                                     formulation, exterior, interior)
    elseif solver == :gmres
        A = matrixfree_dielectric_sie_operator_3d(
            mesh, rwg, k0, epsr_in;
            mur_in=mur_in,
            epsr_ext=epsr_ext,
            mur_ext=mur_ext,
            formulation=formulation,
            quad_order=quad_order,
            singular_quad_order=singular_quad_order,
            eta0=eta0,
            mesh_precheck=mesh_precheck,
            area_tol_rel=area_tol_rel,
            max_gram_storage_bytes=max_gram_storage_bytes,
            max_cache_bytes=max_cache_bytes,
            max_adjacency_pairs=max_adjacency_pairs,
            max_near_pairs=max_near_pairs,
        )
        x, stats = solve_gmres(
            A, rhsv;
            memory=memory,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
            check_gmres_convergence=check_gmres_convergence,
        )
        J, M = _split_surface_currents_3d(x, rwg.nedges)
        return DielectricSIEResult3D(copy(J), copy(M), A, rhsv, nothing,
                                     :gmres, stats,
                                     formulation, A.exterior, A.interior)
    end
end

function solve_dielectric_sie_3d(mesh::TriMesh, rwg::RWGData, k0::Real,
                                 epsr_in, excitation::PlaneWaveExcitation;
                                 mur_in=1.0 + 0im,
                                 epsr_ext=1.0 + 0im,
                                 mur_ext=1.0 + 0im,
                                 formulation::Symbol=:pmchwt,
                                 solver::Symbol=:direct,
                                 quad_order::Int=3,
                                 singular_quad_order::Int=7,
                                 eta0::Real=_ETA0_DDA,
                                 mesh_precheck::Bool=true,
                                 area_tol_rel::Float64=1e-12,
                                 tol::Float64=1e-8,
                                 maxiter::Int=200,
                                 memory::Int=20,
                                 verbose::Bool=false,
                                 check_gmres_convergence::Bool=true,
                                 max_work_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                 max_gram_storage_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                                 max_cache_bytes::Integer=
                                     _DEFAULT_MAX_SURFACE_CACHE_BYTES_3D,
                                 max_adjacency_pairs::Integer=
                                     _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS,
                                 max_near_pairs::Integer=
                                     _DEFAULT_MAX_SURFACE_NEAR_PAIRS_3D)
    solver in (:direct, :gmres) ||
        error("Unsupported dielectric SIE solver: $solver (expected :direct or :gmres).")
    _validated_surface_sie_formulation_3d(formulation)
    if solver == :direct
        _validated_surface_sie_dense_work_3d(
            rwg.nedges, 10, max_work_bytes,
            "direct dielectric SIE solve")
    end
    exterior = dielectric_medium_3d(k0, epsr_ext, mur_ext; eta0=eta0)
    interior = dielectric_medium_3d(k0, epsr_in, mur_in; eta0=eta0)
    rhs = assemble_dielectric_sie_rhs_3d(
        mesh, rwg, excitation, exterior;
        quad_order=quad_order,
        formulation=formulation,
        interior=interior,
    )
    return solve_dielectric_sie_3d(
        mesh, rwg, k0, epsr_in, rhs;
        mur_in=mur_in,
        epsr_ext=epsr_ext,
        mur_ext=mur_ext,
        formulation=formulation,
        solver=solver,
        quad_order=quad_order,
        singular_quad_order=singular_quad_order,
        eta0=eta0,
        mesh_precheck=mesh_precheck,
        area_tol_rel=area_tol_rel,
        tol=tol,
        maxiter=maxiter,
        memory=memory,
        verbose=verbose,
        check_gmres_convergence=check_gmres_convergence,
        max_work_bytes=max_work_bytes,
        max_gram_storage_bytes=max_gram_storage_bytes,
        max_cache_bytes=max_cache_bytes,
        max_adjacency_pairs=max_adjacency_pairs,
        max_near_pairs=max_near_pairs,
    )
end
