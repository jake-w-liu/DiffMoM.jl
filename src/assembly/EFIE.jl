# EFIE.jl — Dense and matrix-free EFIE operators
#
# Z_mn = <f_m, T[f_n]>  (EFIE operator only, without impedance sheet)
#
# Uses mixed-potential form:
#   Z_mn = -iωμ₀ [ ∫∫ f_m(r)·f_n(r') G(r,r') dS dS'
#                   - (1/k²) ∫∫ (∇·f_m)(∇'·f_n) G(r,r') dS dS' ]

export assemble_Z_efie
export MatrixFreeEFIEOperator, MatrixFreeEFIEAdjointOperator
export matrixfree_efie_operator, matrixfree_efie_adjoint_operator
export efie_entry

const _DEFAULT_MAX_EFIE_CACHE_BYTES = 2_000_000_000
const _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS = 20_000_000
const _EFIE_EMPTY_VECTOR_BYTES = Base.summarysize(Vec3[])

"""
Compact row storage for a symmetric adjacency relation between mesh triangles.

Neighbors of triangle `t` occupy
`neighbors[offsets[t]:(offsets[t + 1] - 1)]`. Storage is `O(Nt + Nadj)`
instead of the `O(Nt^2)` bits required by a `BitMatrix`. The EFIE builder
stores edge neighbors; the dielectric-SIE builder stores vertex neighbors.
"""
struct TriangleAdjacency <: AbstractMatrix{Bool}
    offsets::Vector{Int}
    neighbors::Vector{Int}
end

Base.size(adjacency::TriangleAdjacency) = begin
    Nt = length(adjacency.offsets) - 1
    (Nt, Nt)
end
Base.IndexStyle(::Type{TriangleAdjacency}) = IndexCartesian()

@inline function _has_triangle_neighbor(adjacency::TriangleAdjacency,
                                        t1::Int, t2::Int)
    @inbounds for idx in adjacency.offsets[t1]:(adjacency.offsets[t1 + 1] - 1)
        adjacency.neighbors[idx] == t2 && return true
    end
    return false
end

@inline function Base.getindex(adjacency::TriangleAdjacency, t1::Int, t2::Int)
    @boundscheck checkbounds(adjacency, t1, t2)
    return t1 != t2 && _has_triangle_neighbor(adjacency, t1, t2)
end

function _triangle_adjacency_from_pairs!(pairs::Vector{NTuple{2,Int}}, Nt::Int)
    sort!(pairs)
    if !isempty(pairs)
        write_idx = 1
        @inbounds for read_idx in 2:length(pairs)
            if pairs[read_idx] != pairs[write_idx]
                write_idx += 1
                pairs[write_idx] = pairs[read_idx]
            end
        end
        resize!(pairs, write_idx)
    end

    degree = zeros(Int, Nt)
    @inbounds for (t1, t2) in pairs
        degree[t1] += 1
        degree[t2] += 1
    end

    offsets = Vector{Int}(undef, Nt + 1)
    offsets[1] = 1
    @inbounds for t in 1:Nt
        offsets[t + 1] = offsets[t] + degree[t]
        degree[t] = offsets[t]  # reuse as the fill cursor
    end

    neighbors = Vector{Int}(undef, offsets[end] - 1)
    @inbounds for (t1, t2) in pairs
        neighbors[degree[t1]] = t2
        degree[t1] += 1
        neighbors[degree[t2]] = t1
        degree[t2] += 1
    end

    return TriangleAdjacency(offsets, neighbors)
end

"""
Build a deduplicated triangle edge-adjacency table in compact row storage.

The temporary edge records and triangle pairs are sorted in place. This avoids
one heap-allocated vector per mesh edge and leaves only the two compact vectors
resident in `EFIEApplyCache`.
"""
function _efie_cache_work_bytes(
        fixed_payload_bytes::Int,
        triangle_count::Int,
        pair_records::Int)
    fixed_payload_bytes >= 0 ||
        throw(ArgumentError("fixed EFIE cache payload must be nonnegative"))
    triangle_count >= 0 ||
        throw(ArgumentError("triangle count must be nonnegative"))
    pair_records >= 0 ||
        throw(ArgumentError("adjacency-pair count must be nonnegative"))

    edge_record_bytes = BigInt(3) * triangle_count *
                        sizeof(NTuple{3,Int})
    pair_bytes = BigInt(pair_records) * sizeof(NTuple{2,Int})
    degree_bytes = BigInt(triangle_count) * sizeof(Int)
    offset_bytes = (BigInt(triangle_count) + 1) * sizeof(Int)
    neighbor_bytes = BigInt(2) * pair_records * sizeof(Int)
    adjacency_peak = edge_record_bytes + pair_bytes + degree_bytes +
                     offset_bytes + neighbor_bytes
    retained_peak = BigInt(fixed_payload_bytes) + offset_bytes +
                    neighbor_bytes
    estimated = cld(5 * max(adjacency_peak, retained_peak), 4)
    estimated <= typemax(Int) ||
        throw(ArgumentError("EFIE cache storage estimate overflows Int"))
    return Int(estimated)
end

function _efie_cache_retained_bytes(
        fixed_payload_bytes::Int,
        triangle_count::Int,
        adjacent_pairs::Int)
    fixed_payload_bytes >= 0 ||
        throw(ArgumentError("fixed EFIE cache payload must be nonnegative"))
    triangle_count >= 0 ||
        throw(ArgumentError("triangle count must be nonnegative"))
    adjacent_pairs >= 0 ||
        throw(ArgumentError("adjacent-pair count must be nonnegative"))
    raw_bytes = BigInt(fixed_payload_bytes) +
                (BigInt(triangle_count) + 1) * sizeof(Int) +
                BigInt(2) * adjacent_pairs * sizeof(Int)
    estimated = cld(5 * raw_bytes, 4)
    estimated <= typemax(Int) ||
        throw(ArgumentError("retained EFIE cache estimate overflows Int"))
    return Int(estimated)
end

function _build_triangle_adjacency(
        mesh::TriMesh;
        fixed_payload_bytes::Int=0,
        max_cache_bytes::Integer=_DEFAULT_MAX_EFIE_CACHE_BYTES,
        max_adjacency_pairs::Integer=_DEFAULT_MAX_EFIE_ADJACENCY_PAIRS)
    Nt = ntriangles(mesh)
    cache_limit = _validated_resource_limit(
        "max_cache_bytes", max_cache_bytes)
    pair_limit = _validated_nonnegative_resource_limit(
        "max_adjacency_pairs", max_adjacency_pairs)
    minimum_bytes = _efie_cache_work_bytes(
        fixed_payload_bytes, Nt, 0)
    minimum_bytes <= cache_limit ||
        throw(ArgumentError(
            "EFIE cache requires at least $minimum_bytes estimated bytes, " *
            "exceeding max_cache_bytes=$cache_limit"))

    nrecords = Base.checked_mul(3, Nt)
    edge_records = Vector{NTuple{3,Int}}(undef, nrecords)

    record_idx = 1
    @inbounds for t in 1:Nt
        v1 = mesh.tri[1, t]
        v2 = mesh.tri[2, t]
        v3 = mesh.tri[3, t]
        for (va, vb) in ((v1, v2), (v2, v3), (v3, v1))
            edge_records[record_idx] =
                va < vb ? (va, vb, t) : (vb, va, t)
            record_idx += 1
        end
    end
    sort!(edge_records)

    pair_records = 0
    first_record = 1
    while first_record <= nrecords
        next_edge = first_record + 1
        @inbounds while next_edge <= nrecords &&
                        edge_records[next_edge][1] == edge_records[first_record][1] &&
                        edge_records[next_edge][2] == edge_records[first_record][2]
            next_edge += 1
        end

        group_count = next_edge - first_record
        first_factor, second_factor = iseven(group_count) ?
            (group_count ÷ 2, group_count - 1) :
            (group_count, (group_count - 1) ÷ 2)
        remaining_pairs = pair_limit - pair_records
        if !iszero(first_factor) &&
           second_factor > remaining_pairs ÷ first_factor
            throw(ArgumentError(
                "triangle adjacency exceeds " *
                "max_adjacency_pairs=$pair_limit"))
        end
        pair_records += first_factor * second_factor
        first_record = next_edge
    end
    work_bytes = _efie_cache_work_bytes(
        fixed_payload_bytes, Nt, pair_records)
    work_bytes <= cache_limit ||
        throw(ArgumentError(
            "EFIE cache requires at most $work_bytes estimated bytes, " *
            "exceeding max_cache_bytes=$cache_limit"))

    pairs = NTuple{2,Int}[]
    sizehint!(pairs, pair_records)
    first_record = 1
    while first_record <= nrecords
        next_edge = first_record + 1
        @inbounds while next_edge <= nrecords &&
                        edge_records[next_edge][1] == edge_records[first_record][1] &&
                        edge_records[next_edge][2] == edge_records[first_record][2]
            next_edge += 1
        end

        # All triangles in this group share the same mesh edge. Generating every
        # pair preserves the previous behavior for non-manifold edges as well.
        @inbounds for i in first_record:(next_edge - 1)
            for j in (i + 1):(next_edge - 1)
                t1 = edge_records[i][3]
                t2 = edge_records[j][3]
                t1 == t2 && continue
                push!(pairs, t1 < t2 ? (t1, t2) : (t2, t1))
            end
        end
        first_record = next_edge
    end

    # A malformed/duplicate face can make a triangle pair share more than one
    # recorded edge. The common converter sorts and compacts those duplicates.
    return _triangle_adjacency_from_pairs!(pairs, Nt)
end

function _efie_cache_fixed_payload_bytes(
        N::Int,
        Nt::Int,
        Nq::Int,
        Nq_hi::Int,
        ::Type{Tcoef},
        ::Type{TVec}) where {Tcoef,TVec}
    triangle_payload = BigInt(Nt) * (
        sizeof(Vector{Vec3}) + _EFIE_EMPTY_VECTOR_BYTES + Nq * sizeof(Vec3) +
        sizeof(Float64) +
        sizeof(Vector{Vec3}) + _EFIE_EMPTY_VECTOR_BYTES +
        Nq_hi * sizeof(Vec3))
    basis_payload = BigInt(N) * (
        2 * sizeof(Int) + 2 * sizeof(Tcoef) +
        sizeof(NTuple{2,Vector{TVec}}) +
        2 * _EFIE_EMPTY_VECTOR_BYTES + 2 * Nq * sizeof(TVec) +
        sizeof(NTuple{2,Vector{TVec}}) +
        2 * _EFIE_EMPTY_VECTOR_BYTES + 2 * Nq_hi * sizeof(TVec))
    quadrature_payload = BigInt(Nq + Nq_hi) * sizeof(Float64)
    total = triangle_payload + basis_payload + quadrature_payload
    total <= typemax(Int) ||
        throw(ArgumentError("EFIE cache fixed-payload estimate overflows Int"))
    return Int(total)
end

@inline _adjacent_pair_count(adjacency::TriangleAdjacency) =
    length(adjacency.neighbors) ÷ 2

struct EFIEApplyCache{TK, Tω, TD, TV}
    mesh::TriMesh
    rwg::RWGData
    k::TK
    inv_k2::TK                           # precomputed 1/k²
    omega_mu0::Tω
    wq::Vector{Float64}
    Nq::Int
    quad_pts::Vector{Vector{Vec3}}
    areas::Vector{Float64}
    tri_ids::Matrix{Int}                 # (2, N)
    div_vals::Matrix{TD}                 # (2, N)
    rwg_vals::Vector{NTuple{2,Vector{TV}}}
    # Adjacent-cell near-singular integration data
    adjacent::TriangleAdjacency          # compact edge-adjacency rows
    wq_hi::Vector{Float64}              # high-order quadrature weights
    quad_pts_hi::Vector{Vector{Vec3}}    # high-order quadrature points per triangle
    rwg_vals_hi::Vector{NTuple{2,Vector{TV}}}  # RWG values at high-order quad pts
end

# Two finite ComplexF64 factors expand to four exact Float64 products. Their
# component sums need fewer than 4196 bits across the full IEEE exponent range;
# the remainder is a guard margin for conversion back to the original type.
const _EFIE_PRODUCT_FALLBACK_PRECISION = 4352

@noinline function _efie_product_bigfloat(
        first::Number, second::Number, ::Type{T}) where {T<:Number}
    return setprecision(BigFloat, _EFIE_PRODUCT_FALLBACK_PRECISION) do
        if T <: Real
            return convert(T, BigFloat(first) * BigFloat(second))
        end
        return convert(
            T, Complex{BigFloat}(first) * Complex{BigFloat}(second))
    end
end

@inline function _validated_efie_prefactors(k, eta0)
    k isa Number ||
        throw(ArgumentError(
            "EFIE wavenumber must be numeric, got $(typeof(k))"))
    isfinite(k) && !iszero(k) ||
        throw(ArgumentError(
            "EFIE wavenumber must be finite and nonzero, got $k"))
    eta0 isa Number ||
        throw(ArgumentError(
            "EFIE eta0 must be numeric, finite, and nonzero, got $eta0"))
    isfinite(eta0) && !iszero(eta0) ||
        throw(ArgumentError(
            "EFIE eta0 must be numeric, finite, and nonzero, got $eta0"))

    kw = float(k)
    eta = float(eta0)
    k2 = kw * kw
    isfinite(k2) && !iszero(k2) ||
        throw(OverflowError(
            "EFIE wavenumber squared is not representable for k=$k"))
    inv_k2 = inv(k2)
    isfinite(inv_k2) && !iszero(inv_k2) ||
        throw(OverflowError(
            "EFIE inverse wavenumber squared is not representable for k=$k"))
    omega_mu0 = kw * eta
    if !isfinite(omega_mu0) || iszero(omega_mu0)
        omega_mu0 = _efie_product_bigfloat(
            kw, eta, typeof(omega_mu0))
    end
    isfinite(omega_mu0) && !iszero(omega_mu0) ||
        throw(OverflowError(
            "EFIE k*eta0 prefactor is not representable for k=$k and eta0=$eta0"))
    return kw, inv_k2, omega_mu0
end

function _build_efie_cache(
        mesh::TriMesh,
        rwg::RWGData,
        k;
        quad_order::Int=3,
        eta0=376.730313668,
        max_cache_bytes::Integer=_DEFAULT_MAX_EFIE_CACHE_BYTES,
        max_adjacency_pairs::Integer=_DEFAULT_MAX_EFIE_ADJACENCY_PAIRS)
    _validate_mesh_rwg_pair(mesh, rwg)
    kw, inv_k2, omega_mu0 = _validated_efie_prefactors(k, eta0)
    N = rwg.nedges
    Nt = ntriangles(mesh)
    Tcoef = promote_type(eltype(rwg.coeff_plus), eltype(rwg.coeff_minus))
    TVec = SVector{3,Tcoef}

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)
    xi_hi, wq_hi = tri_quad_rule(7)
    Nq_hi = length(wq_hi)
    fixed_payload_bytes = _efie_cache_fixed_payload_bytes(
        N, Nt, Nq, Nq_hi, Tcoef, TVec)
    adjacent = _build_triangle_adjacency(
        mesh;
        fixed_payload_bytes=fixed_payload_bytes,
        max_cache_bytes=max_cache_bytes,
        max_adjacency_pairs=max_adjacency_pairs)

    # Precompute quadrature points and areas for all triangles
    quad_pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
    end

    # High-order quadrature for inner integration of self/adjacent cells
    quad_pts_hi = Vector{Vector{Vec3}}(undef, Nt)
    for t in 1:Nt
        quad_pts_hi[t] = tri_quad_points(mesh, t, xi_hi)
    end

    tri_ids = zeros(Int, 2, N)
    div_vals = zeros(Tcoef, 2, N)
    rwg_vals = Vector{NTuple{2,Vector{TVec}}}(undef, N)
    rwg_vals_hi = Vector{NTuple{2,Vector{TVec}}}(undef, N)

    for n in 1:N
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        tri_ids[1, n] = tp
        tri_ids[2, n] = tm
        div_vals[1, n] = div_rwg(rwg, n, tp)
        div_vals[2, n] = div_rwg(rwg, n, tm)

        vals_p = [eval_rwg(rwg, n, quad_pts[tp][q], tp) for q in 1:Nq]
        vals_m = [eval_rwg(rwg, n, quad_pts[tm][q], tm) for q in 1:Nq]
        rwg_vals[n] = (vals_p, vals_m)

        vals_p_hi = [eval_rwg(rwg, n, quad_pts_hi[tp][q], tp) for q in 1:Nq_hi]
        vals_m_hi = [eval_rwg(rwg, n, quad_pts_hi[tm][q], tm) for q in 1:Nq_hi]
        rwg_vals_hi[n] = (vals_p_hi, vals_m_hi)
    end

    return EFIEApplyCache(mesh, rwg, kw, inv_k2, omega_mu0, wq, Nq, quad_pts, areas,
                          tri_ids, div_vals, rwg_vals,
                          adjacent, wq_hi, quad_pts_hi, rwg_vals_hi)
end

function _efie_cache_with_prefactors(
        cache::EFIEApplyCache,
        k,
        eta0)
    kw, inv_k2, omega_mu0 = _validated_efie_prefactors(k, eta0)
    return EFIEApplyCache(
        cache.mesh,
        cache.rwg,
        kw,
        inv_k2,
        omega_mu0,
        cache.wq,
        cache.Nq,
        cache.quad_pts,
        cache.areas,
        cache.tri_ids,
        cache.div_vals,
        cache.rwg_vals,
        cache.adjacent,
        cache.wq_hi,
        cache.quad_pts_hi,
        cache.rwg_vals_hi,
    )
end

@inline function _is_adjacent(cache::EFIEApplyCache, t1::Int, t2::Int)
    return _has_triangle_neighbor(cache.adjacent, t1, t2)
end

@inline function _finalize_efie_entry(
        cache::EFIEApplyCache, value, m::Int, n::Int)
    entry = -1im * cache.omega_mu0 * value
    isfinite(entry) ||
        throw(OverflowError(
            "EFIE entry ($m, $n) is outside the representable " *
            "ComplexF64 range."))
    return entry
end

@inline function _efie_entry(cache::EFIEApplyCache, m::Int, n::Int)
    # For non-Bloch RWG, normalize to canonical order (m ≤ n) so that
    # _efie_entry(c, i, j) == _efie_entry(c, j, i) bitwise.
    # This keeps dense assembly (symmetry exploit) and on-the-fly getindex consistent.
    if !cache.rwg.has_periodic_bloch && m > n
        return _efie_entry(cache, n, m)
    end
    CT = ComplexF64
    val = zero(CT)
    inv_k2 = cache.inv_k2

    @inbounds for itm in 1:2
        tm = cache.tri_ids[itm, m]
        Am = cache.areas[tm]
        dvm = cache.div_vals[itm, m]
        fm_vals = _rwg_vals(cache, m, itm)
        fm_vals_hi = _rwg_vals_hi(cache, m, itm)

        for itn in 1:2
            tn = cache.tri_ids[itn, n]
            An = cache.areas[tn]
            dvn = cache.div_vals[itn, n]
            fn_vals = _rwg_vals(cache, n, itn)
            fn_vals_hi = _rwg_vals_hi(cache, n, itn)

            if tm == tn
                # Self-cell singularity extraction (high-order quadrature)
                val += _self_cell_cached(
                    cache.mesh, tm,
                    fm_vals_hi,
                    fn_vals_hi,
                    dvm, dvn,
                    Am, cache.k, inv_k2,
                    cache.wq_hi,
                    cache.quad_pts_hi[tm],
                )
            elseif _is_adjacent(cache, tm, tn)
                # Adjacent-cell near-singular integration
                val += _adjacent_cell_cached(
                    cache.mesh, tn,
                    cache.quad_pts[tm], cache.quad_pts[tn],
                    fm_vals, fn_vals,
                    fm_vals_hi, fn_vals_hi,
                    dvm, dvn,
                    Am, An,
                    cache.wq, cache.k, inv_k2,
                    cache.wq_hi, cache.quad_pts_hi[tm], cache.quad_pts_hi[tn],
                )
            else
                # Non-adjacent: standard product quadrature
                dvmn_inv_k2 = conj(dvm) * dvn * inv_k2
                for qm in 1:cache.Nq
                    rm = cache.quad_pts[tm][qm]
                    fm = fm_vals[qm]

                    for qn in 1:cache.Nq
                        rn = cache.quad_pts[tn][qn]
                        fn = fn_vals[qn]

                        G = _greens_unchecked(rm, rn, cache.k)

                        vec_part = dot(fm, fn) * G
                        scl_part = dvmn_inv_k2 * G
                        weight = cache.wq[qm] * cache.wq[qn] * (2 * Am) * (2 * An)

                        val += (vec_part - scl_part) * weight
                    end
                end
            end
        end
    end

    return _finalize_efie_entry(cache, val, m, n)
end

@inline function _rwg_vals(cache::EFIEApplyCache, n::Int, it::Int)
    if it == 1
        return cache.rwg_vals[n][1]
    end
    return cache.rwg_vals[n][2]
end

@inline function _rwg_vals_hi(cache::EFIEApplyCache, n::Int, it::Int)
    if it == 1
        return cache.rwg_vals_hi[n][1]
    end
    return cache.rwg_vals_hi[n][2]
end

# ── Internal cached versions of self/adjacent cell (no eval_rwg calls) ──

"""
Self-cell contribution using precomputed high-order RWG values.
Avoids calling eval_rwg in the hot inner loops.
"""
@inline function _self_cell_cached(
    mesh::TriMesh, tm::Int,
    fm_hi::Vector{<:SVector{3,<:Number}},
    fn_hi::Vector{<:SVector{3,<:Number}},
    div_m::Number, div_n::Number,
    Am::Float64, k, inv_k2,
    wq_hi, quad_pts_tm_hi::Vector{Vec3})

    Nq_hi = length(wq_hi)
    CT = complex(typeof(real(k)))

    V1 = _mesh_vertex(mesh, mesh.tri[1, tm])
    V2 = _mesh_vertex(mesh, mesh.tri[2, tm])
    V3 = _mesh_vertex(mesh, mesh.tri[3, tm])

    inv4pi = 1.0 / (4π)
    dvmn_inv_k2 = conj(div_m) * div_n * inv_k2

    # ── Smooth part: high-order product quadrature with G_smooth ──
    val_smooth = zero(CT)
    @inbounds for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = fm_hi[qm]
        for qn in 1:Nq_hi
            rn = quad_pts_tm_hi[qn]
            fn = fn_hi[qn]

            Gs = _greens_smooth_unchecked(rm, rn, k)
            vec_part = dot(fm, fn) * Gs
            scl_part = dvmn_inv_k2 * Gs
            weight = wq_hi[qm] * wq_hi[qn] * (2 * Am) * (2 * Am)
            val_smooth += (vec_part - scl_part) * weight
        end
    end

    # ── Singular part: semi-analytical with high-order outer quadrature ──
    val_singular = zero(CT)
    @inbounds for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = fm_hi[qm]

        S = _analytical_integral_1overR_unchecked(rm, V1, V2, V3)
        inner_scalar = inv4pi * S

        # Scalar potential singular part
        scl_sing = dvmn_inv_k2 * inner_scalar

        # Vector potential singular part: leading term
        fn_at_rm = fn_hi[qm]
        vec_lead = dot(fm, fn_at_rm) * inner_scalar

        # Vector potential singular part: remainder (bounded integrand)
        vec_rem = zero(CT)
        for qn in 1:Nq_hi
            rn = quad_pts_tm_hi[qn]
            fn = fn_hi[qn]

            R_vec = rm - rn
            R = hypot(hypot(R_vec[1], R_vec[2]), R_vec[3])
            if iszero(R)
                continue
            end

            delta_fn = fn - fn_at_rm
            vec_rem += dot(fm, delta_fn) * (inv4pi / R) * wq_hi[qn] * (2 * Am)
        end

        outer_weight = wq_hi[qm] * (2 * Am)
        val_singular += ((vec_lead + vec_rem) - scl_sing) * outer_weight
    end

    return val_smooth + val_singular
end

"""
Adjacent-cell contribution using precomputed high-order RWG values.
"""
@inline function _adjacent_cell_cached(
    mesh::TriMesh, tn::Int,
    quad_pts_tm::Vector{Vec3}, quad_pts_tn::Vector{Vec3},
    fm_vals::Vector{<:SVector{3,<:Number}},
    fn_vals::Vector{<:SVector{3,<:Number}},
    fm_hi::Vector{<:SVector{3,<:Number}},
    fn_hi::Vector{<:SVector{3,<:Number}},
    div_m::Number, div_n::Number,
    Am::Float64, An::Float64,
    wq, k, inv_k2,
    wq_hi, quad_pts_tm_hi::Vector{Vec3}, quad_pts_tn_hi::Vector{Vec3})

    Nq = length(wq)
    Nq_hi = length(wq_hi)
    CT = complex(typeof(real(k)))

    V1n = _mesh_vertex(mesh, mesh.tri[1, tn])
    V2n = _mesh_vertex(mesh, mesh.tri[2, tn])
    V3n = _mesh_vertex(mesh, mesh.tri[3, tn])

    inv4pi = 1.0 / (4π)
    dvmn_inv_k2 = conj(div_m) * div_n * inv_k2

    # ── Smooth part: standard product quadrature with G_smooth ──
    val_smooth = zero(CT)
    @inbounds for qm in 1:Nq
        rm = quad_pts_tm[qm]
        fm = fm_vals[qm]
        for qn in 1:Nq
            rn = quad_pts_tn[qn]
            fn = fn_vals[qn]

            Gs = _greens_smooth_unchecked(rm, rn, k)
            vec_part = dot(fm, fn) * Gs
            scl_part = dvmn_inv_k2 * Gs
            weight = wq[qm] * wq[qn] * (2 * Am) * (2 * An)
            val_smooth += (vec_part - scl_part) * weight
        end
    end

    # ── Singular 1/(4πR) part: semi-analytical ──
    val_singular = zero(CT)
    @inbounds for qm in 1:Nq_hi
        rm = quad_pts_tm_hi[qm]
        fm = fm_hi[qm]

        # Analytical inner integral: S = ∫_{T_n} 1/|rm - r'| dS'
        S = _analytical_integral_1overR_unchecked(rm, V1n, V2n, V3n)
        inner_scalar = inv4pi * S

        # Scalar potential singular part
        scl_sing = dvmn_inv_k2 * inner_scalar

        # Vector potential singular part: ∫_{T_n} f_n(r') / (4πR) dS'
        vec_sing = zero(CT)
        for qn in 1:Nq_hi
            rn = quad_pts_tn_hi[qn]
            fn_qn = fn_hi[qn]

            R_vec = rm - rn
            R = hypot(hypot(R_vec[1], R_vec[2]), R_vec[3])
            if iszero(R)
                continue
            end

            vec_sing += dot(fm, fn_qn) * (inv4pi / R) * wq_hi[qn] * (2 * An)
        end

        outer_weight = wq_hi[qm] * (2 * Am)
        val_singular += (vec_sing - scl_sing) * outer_weight
    end

    return val_smooth + val_singular
end

"""
    assemble_Z_efie(mesh, rwg, k;
                    quad_order=3,
                    mesh_precheck=true,
                    allow_boundary=true,
                    require_closed=false,
                    max_output_bytes=2_000_000_000,
                    max_cache_bytes=2_000_000_000,
                    max_adjacency_pairs=20_000_000)

Assemble the dense EFIE matrix `Z_efie ∈ C^{N×N}`.
`k` is the wavenumber (can be complex for complex-step).
"""
function assemble_Z_efie(mesh::TriMesh, rwg::RWGData, k;
                         quad_order::Int=3, eta0=376.730313668,
                         mesh_precheck::Bool=true,
                         allow_boundary::Bool=true,
                         require_closed::Bool=false,
                         area_tol_rel::Float64=1e-12,
                         max_output_bytes::Integer=
                             _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
                         max_cache_bytes::Integer=
                             _DEFAULT_MAX_EFIE_CACHE_BYTES,
                         max_adjacency_pairs::Integer=
                             _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS)
    _validate_mesh_rwg_pair(mesh, rwg)
    N = rwg.nedges
    matrix_bytes = _checked_array_payload_bytes(
        ComplexF64, N, N; label="dense EFIE matrix")
    _enforce_payload_limit(
        matrix_bytes, max_output_bytes,
        "dense EFIE matrix", "max_output_bytes")
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    cache = _build_efie_cache(
        mesh, rwg, k;
        quad_order=quad_order,
        eta0=eta0,
        max_cache_bytes=max_cache_bytes,
        max_adjacency_pairs=max_adjacency_pairs)
    CT = ComplexF64
    Z = zeros(CT, N, N)

    # Exploit Z symmetry for non-Bloch RWG (real coefficients → Z_mn = Z_nm)
    if !rwg.has_periodic_bloch
        Threads.@threads for m in 1:N
            @inbounds for n in m:N
                z = _efie_entry(cache, m, n)
                Z[m, n] = z
                Z[n, m] = z
            end
        end
    else
        Threads.@threads for m in 1:N
            @inbounds for n in 1:N
                Z[m, n] = _efie_entry(cache, m, n)
            end
        end
    end

    return Z
end

"""
Matrix-free EFIE operator `A` such that `A * x` computes the EFIE matvec
without allocating a dense `N×N` matrix.
"""
struct MatrixFreeEFIEOperator{T, TC<:EFIEApplyCache} <: AbstractMatrix{T}
    cache::TC
end

"""
Adjoint matrix-free EFIE operator `Aᴴ` used by GMRES adjoint solves.
"""
struct MatrixFreeEFIEAdjointOperator{T, TO<:MatrixFreeEFIEOperator{T}} <: AbstractMatrix{T}
    op::TO
end

# A finite Float64 is an integer multiple of 2^-1074 with an integer
# coefficient of at most 2098 bits. A complex product therefore needs at most
# 4196 bits, and summing both real products across any Int-indexable row adds
# fewer than 64 bits. This precision makes the exceptional reduction exact for
# ComplexF64 inputs while leaving the ordinary zero-allocation path unchanged.
const _EFIE_MATVEC_FALLBACK_PRECISION = 4352

# With every real component exponent in [-128, 128], one ComplexF64 product
# and any Int-indexable row reduction remain inside the normal Float64 range.
# More extreme factors need the exact accumulator even when the rounded term
# itself is finite: individually lost products can later cancel to a
# representable row result.
@inline function _matrixfree_extreme_component(value::Real)
    magnitude = abs(Float64(value))
    isfinite(magnitude) || return true
    iszero(magnitude) && return false
    value_exponent = exponent(magnitude)
    return value_exponent < -128 || value_exponent > 128
end

@inline function _matrixfree_extreme_factor(value::Number)
    return _matrixfree_extreme_component(real(value)) ||
           _matrixfree_extreme_component(imag(value))
end

@inline function _matrixfree_complex_reduction_requires_exact(
        value::Number,
        real_magnitude::Float64,
        imag_magnitude::Float64,
        term_count::Int)
    error_factor = min(
        1.0,
        16eps(Float64) * Float64(max(term_count, 1)),
    )
    return (!iszero(real_magnitude) &&
            abs(Float64(real(value))) <= error_factor * real_magnitude) ||
           (!iszero(imag_magnitude) &&
            abs(Float64(imag(value))) <= error_factor * imag_magnitude)
end

Base.size(A::MatrixFreeEFIEOperator) = (A.cache.rwg.nedges, A.cache.rwg.nedges)
Base.eltype(::MatrixFreeEFIEOperator{T}) where {T} = T

Base.size(A::MatrixFreeEFIEAdjointOperator) = size(A.op)
Base.eltype(::MatrixFreeEFIEAdjointOperator{T}) where {T} = T

function efie_entry(A::MatrixFreeEFIEOperator, m::Int, n::Int)
    return _efie_entry(A.cache, m, n)
end

function Base.getindex(A::MatrixFreeEFIEOperator, i::Int, j::Int)
    return efie_entry(A, i, j)
end

function Base.getindex(A::MatrixFreeEFIEAdjointOperator, i::Int, j::Int)
    return conj(efie_entry(A.op, j, i))
end

@noinline function _matrixfree_efie_row_bigfloat(
        A::MatrixFreeEFIEOperator{ComplexF64},
        x::AbstractVector,
        row::Int)
    return setprecision(BigFloat, _EFIE_MATVEC_FALLBACK_PRECISION) do
        total = zero(Complex{BigFloat})
        @inbounds for column in axes(A, 2)
            total += Complex{BigFloat}(efie_entry(A, row, column)) *
                     Complex{BigFloat}(x[column])
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "matrix-free EFIE output is outside the representable " *
                "ComplexF64 range at row $row."))
        return converted
    end
end

@noinline function _matrixfree_efie_adjoint_row_bigfloat(
        A::MatrixFreeEFIEAdjointOperator{ComplexF64},
        x::AbstractVector,
        row::Int)
    return setprecision(BigFloat, _EFIE_MATVEC_FALLBACK_PRECISION) do
        total = zero(Complex{BigFloat})
        @inbounds for column in axes(A, 2)
            total += conj(Complex{BigFloat}(
                         efie_entry(A.op, column, row))) *
                     Complex{BigFloat}(x[column])
        end
        converted = ComplexF64(total)
        isfinite(converted) ||
            throw(OverflowError(
                "adjoint matrix-free EFIE output is outside the " *
                "representable ComplexF64 range at row $row."))
        return converted
    end
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::MatrixFreeEFIEOperator{T}, x::AbstractVector) where {T<:Number}
    N = size(A, 1)
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    xread = Base.mightalias(y, x) ? copy(x) : x
    @inbounds for m in 1:N
        acc = zero(T)
        real_magnitude = 0.0
        imag_magnitude = 0.0
        needs_fallback = false
        try
            for n in 1:N
                entry = efie_entry(A, m, n)
                input = xread[n]
                if !iszero(entry) && !iszero(input) &&
                   (_matrixfree_extreme_factor(entry) ||
                    _matrixfree_extreme_factor(input))
                    needs_fallback = true
                    break
                end
                term = entry * input
                next_acc = acc + term
                real_magnitude += abs(Float64(real(term)))
                imag_magnitude += abs(Float64(imag(term)))
                if !isfinite(next_acc) || !isfinite(real_magnitude) ||
                   !isfinite(imag_magnitude)
                    needs_fallback = true
                    break
                end
                acc = next_acc
            end
        catch err
            err isa OverflowError || rethrow()
            needs_fallback = true
        end
        needs_fallback |= _matrixfree_complex_reduction_requires_exact(
            acc, real_magnitude, imag_magnitude, N)
        y[m] = needs_fallback ?
               _matrixfree_efie_row_bigfloat(A, xread, m) : acc
    end
    return y
end

function Base.:*(A::MatrixFreeEFIEOperator{T}, x::AbstractVector) where {T<:Number}
    y = zeros(T, size(A, 1))
    mul!(y, A, x)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::MatrixFreeEFIEAdjointOperator{T}, x::AbstractVector) where {T<:Number}
    N = size(A, 1)
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    xread = Base.mightalias(y, x) ? copy(x) : x
    @inbounds for n in 1:N
        acc = zero(T)
        real_magnitude = 0.0
        imag_magnitude = 0.0
        needs_fallback = false
        try
            for m in 1:N
                entry = conj(efie_entry(A.op, m, n))
                input = xread[m]
                if !iszero(entry) && !iszero(input) &&
                   (_matrixfree_extreme_factor(entry) ||
                    _matrixfree_extreme_factor(input))
                    needs_fallback = true
                    break
                end
                term = entry * input
                next_acc = acc + term
                real_magnitude += abs(Float64(real(term)))
                imag_magnitude += abs(Float64(imag(term)))
                if !isfinite(next_acc) || !isfinite(real_magnitude) ||
                   !isfinite(imag_magnitude)
                    needs_fallback = true
                    break
                end
                acc = next_acc
            end
        catch err
            err isa OverflowError || rethrow()
            needs_fallback = true
        end
        needs_fallback |= _matrixfree_complex_reduction_requires_exact(
            acc, real_magnitude, imag_magnitude, N)
        y[n] = needs_fallback ?
               _matrixfree_efie_adjoint_row_bigfloat(A, xread, n) : acc
    end
    return y
end

function Base.:*(A::MatrixFreeEFIEAdjointOperator{T}, x::AbstractVector) where {T<:Number}
    y = zeros(T, size(A, 1))
    mul!(y, A, x)
    return y
end

LinearAlgebra.adjoint(A::MatrixFreeEFIEOperator{T}) where {T<:Number} =
    MatrixFreeEFIEAdjointOperator{T,typeof(A)}(A)

"""
    matrixfree_efie_operator(mesh, rwg, k; kwargs...)

Create a matrix-free EFIE operator for GMRES matvecs without dense `Z` allocation.
"""
function matrixfree_efie_operator(mesh::TriMesh, rwg::RWGData, k;
                                  quad_order::Int=3, eta0=376.730313668,
                                  mesh_precheck::Bool=true,
                                  allow_boundary::Bool=true,
                                  require_closed::Bool=false,
                                  area_tol_rel::Float64=1e-12,
                                  max_cache_bytes::Integer=
                                      _DEFAULT_MAX_EFIE_CACHE_BYTES,
                                  max_adjacency_pairs::Integer=
                                      _DEFAULT_MAX_EFIE_ADJACENCY_PAIRS)
    _validate_mesh_rwg_pair(mesh, rwg)
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end
    cache = _build_efie_cache(
        mesh, rwg, k;
        quad_order=quad_order,
        eta0=eta0,
        max_cache_bytes=max_cache_bytes,
        max_adjacency_pairs=max_adjacency_pairs)
    return MatrixFreeEFIEOperator{ComplexF64,typeof(cache)}(cache)
end

"""
    matrixfree_efie_adjoint_operator(A)

Return matrix-free adjoint operator `Aᴴ` for Krylov adjoint solves.
"""
matrixfree_efie_adjoint_operator(A::MatrixFreeEFIEOperator) = adjoint(A)
