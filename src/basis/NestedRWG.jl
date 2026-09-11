export NestedRWGPair, build_nested_rwg_pair

"""
    NestedRWGPair

One uniform midpoint refinement on a fixed faceted surface. `P` injects coarse
RWG coefficients into the fine basis; `Q` selects native fine coordinates
that complete `P` to a nonsingular basis. `parent_triangles` maps each fine
triangle to its coarse parent. No vertex is projected onto another surface.

`pivot_ratio` is the smallest/largest absolute diagonal entry of the
rank-revealing QR factor, not a condition-number estimate. The complement is
a coordinate completion, not an orthogonal complement in field energy.
"""
struct NestedRWGPair
    coarse_mesh::TriMesh
    fine_mesh::TriMesh
    coarse_rwg::RWGData{Float64}
    fine_rwg::RWGData{Float64}
    P::SparseMatrixCSC{Float64,Int}
    Q::SparseMatrixCSC{Float64,Int}
    parent_triangles::Vector{Int}
    pivot_rows::Vector{Int}
    pivot_ratio::Float64
    work_bytes_upper_bound::Int
    signature::UInt
end

function _nested_pair_signature(pair::NestedRWGPair)
    return _content_fingerprint(
        (pair.coarse_mesh, pair.fine_mesh, pair.coarse_rwg, pair.fine_rwg,
         pair.P, pair.Q, pair.parent_triangles, pair.pivot_rows,
         pair.pivot_ratio, pair.work_bytes_upper_bound), UInt(0))
end

function _validate_nested_pair(pair::NestedRWGPair)
    pair.signature == _nested_pair_signature(pair) ||
        throw(ArgumentError("nested RWG geometry or maps changed; rebuild the pair"))
    return nothing
end

function _nested_rwg_work_bytes(mesh::TriMesh)
    nv = BigInt(nvertices(mesh))
    nt = BigInt(ntriangles(mesh))
    # A manifold triangular mesh has at most 3Nt/2 interior edges. Midpoint
    # subdivision gives 4Nt triangles and at most Nv+3Nt vertices. Reserve
    # dense worst-case sparse-QR fill and its factor/work arrays before setup.
    nc = 3nt ÷ 2
    nf = 6nt
    qr_bytes = 8sizeof(Float64) * (nc * nf + nc^2)
    # Connectivity, edge maps, both RWG records, sparse triplet builders,
    # parent/pivot vectors and sparse construction transients. This includes
    # a conservative per-edge dictionary/object allowance.
    geometry_bytes = 1024 * (nv + 3nt + nf) + 128 * (4nt + nf)
    total = qr_bytes + geometry_bytes
    total <= typemax(Int) ||
        throw(ArgumentError("nested RWG workspace estimate overflows Int"))
    return Int(total)
end

function _midpoint_rwg_injection(
        coarse::RWGData{Float64}, fine::RWGData{Float64}, parents::Vector{Int})
    support = [Int[] for _ in 1:ntriangles(coarse.mesh)]
    for edge in 1:coarse.nedges
        push!(support[coarse.tplus[edge]], edge)
        push!(support[coarse.tminus[edge]], edge)
    end
    rows = Int[]
    columns = Int[]
    coefficients = Float64[]
    sizehint!(rows, 3fine.nedges)
    sizehint!(columns, 3fine.nedges)
    sizehint!(coefficients, 3fine.nedges)
    for edge in 1:fine.nedges
        triangle = fine.tplus[edge]
        parent = parents[triangle]
        first = _mesh_vertex(fine.mesh, fine.evert[1, edge])
        second = _mesh_vertex(fine.mesh, fine.evert[2, edge])
        midpoint = _safe_edge_midpoint(first, second)
        tangent = (second - first) / fine.len[edge]
        toward_edge = midpoint - _mesh_vertex(fine.mesh, fine.vplus_opp[edge])
        conormal = toward_edge - dot(toward_edge, tangent) * tangent
        conormal_length = norm(conormal)
        isfinite(conormal_length) && conormal_length > 0 ||
            throw(ArgumentError("fine RWG edge $edge has an unresolved conormal"))
        conormal /= conormal_length
        trace = dot(eval_rwg(fine, edge, midpoint, triangle), conormal)
        isfinite(trace) && trace > 0 ||
            throw(ArgumentError("fine RWG edge $edge has an invalid normal trace"))
        for coarse_edge in support[parent]
            coefficient =
                dot(eval_rwg(coarse, coarse_edge, midpoint, parent), conormal) / trace
            isfinite(coefficient) ||
                throw(ArgumentError("nested RWG injection contains a non-finite coefficient"))
            if !iszero(coefficient)
                push!(rows, edge)
                push!(columns, coarse_edge)
                push!(coefficients, coefficient)
            end
        end
    end
    return sparse(rows, columns, coefficients, fine.nedges, coarse.nedges)
end

"""
    build_nested_rwg_pair(coarse_mesh; max_work_bytes=536_870_912,
                          rank_rtol=sqrt(eps(Float64)))

Subdivide every coarse triangle into four coplanar triangles using the
canonical midpoint-refinement routine. Construct the fine RWG injection from
oriented normal traces, and use sparse rank-revealing QR to select a native
coordinate complement. Reject invalid meshes, unresolved coordinates, deficient
rank, and a workspace budget below the conservative preallocation bound.

This supports standard nonperiodic RWG spaces on fixed faceted surfaces with
or without boundary. It does not project a faceted sphere onto a smooth sphere
or establish that independently assembled coarse and fine operators agree.
Use `P' * Z_f * P` and `P' * b_f` for exact finite Galerkin restriction.
"""
function build_nested_rwg_pair(coarse_mesh::TriMesh;
        max_work_bytes::Integer=512 * 1024 * 1024,
        rank_rtol::Real=sqrt(eps(Float64)))
    tolerance = Float64(rank_rtol)
    isfinite(tolerance) && 0 < tolerance < 1 ||
        throw(ArgumentError("rank_rtol must be finite and strictly between zero and one"))
    limit = _validated_resource_limit("max_work_bytes", max_work_bytes)
    work_bytes = _nested_rwg_work_bytes(coarse_mesh)
    _enforce_payload_limit(work_bytes, limit, "nested RWG workspace", "max_work_bytes")
    coarse = build_rwg(coarse_mesh)
    coarse.nedges > 0 ||
        throw(ArgumentError("coarse mesh must contain at least one RWG unknown"))
    refinement = _midpoint_refine_once(coarse_mesh, limit)
    refinement.mesh === nothing &&
        throw(ArgumentError("nested refinement stopped: $(refinement.stop_reason)"))
    fine_mesh = refinement.mesh
    fine = build_rwg(fine_mesh)
    # _midpoint_refine_once emits four consecutive children for each parent.
    parents = [cld(triangle, 4) for triangle in 1:ntriangles(fine_mesh)]
    injection = _midpoint_rwg_injection(coarse, fine, parents)
    factor = LinearAlgebra.qr(sparse(transpose(injection)); tol=tolerance)
    diagonal = abs.(LinearAlgebra.diag(factor.R))
    length(diagonal) == coarse.nedges && !isempty(diagonal) ||
        throw(ArgumentError("nested RWG injection is rank deficient"))
    largest = maximum(diagonal)
    ratio = iszero(largest) ? 0.0 : minimum(diagonal) / largest
    isfinite(ratio) && ratio > tolerance ||
        throw(ArgumentError("nested RWG injection fails the rank threshold"))
    pivots = Vector{Int}(factor.pcol[1:coarse.nedges])
    remaining = trues(fine.nedges)
    remaining[pivots] .= false
    selected = findall(remaining)
    complement = sparse(
        selected, collect(1:length(selected)), ones(length(selected)),
        fine.nedges, length(selected))
    payload = (
        coarse_mesh, fine_mesh, coarse, fine, injection, complement,
        parents, pivots, ratio, work_bytes)
    return NestedRWGPair(payload..., _content_fingerprint(payload, UInt(0)))
end
