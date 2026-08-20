# ClusterTree.jl — Binary space-partitioning tree for H-matrix blocking
#
# Bisects RWG basis function centers along the longest bounding-box dimension.
# Used by ACA.jl to determine admissible (low-rank) vs inadmissible (dense) blocks.

export ClusterNode, ClusterTree
export build_cluster_tree, cluster_diameter, cluster_distance, is_admissible
export is_leaf, leaf_nodes

const _DEFAULT_MAX_CLUSTER_TREE_NODES = 30_000_000
const _DEFAULT_MAX_CLUSTER_TREE_STORAGE_BYTES = 2_000_000_000

function _cluster_tree_resource_bounds(N::Int, leaf_size::Int)
    target_leaves = cld(N, leaf_size)
    leaf_count = 1
    while leaf_count < target_leaves
        leaf_count = try
            Base.Checked.checked_mul(leaf_count, 2)
        catch err
            err isa OverflowError || rethrow()
            throw(ArgumentError("cluster-tree leaf-count estimate overflows Int"))
        end
    end
    node_count = try
        Base.Checked.checked_sub(
            Base.Checked.checked_mul(2, leaf_count), 1)
    catch err
        err isa OverflowError || rethrow()
        throw(ArgumentError("cluster-tree node-count estimate overflows Int"))
    end
    storage = BigInt(node_count) * sizeof(ClusterNode) +
              BigInt(3) * N * sizeof(Int)
    storage <= typemax(Int) ||
        throw(ArgumentError("cluster-tree storage estimate overflows Int"))
    return node_count, Int(storage)
end

"""
    ClusterNode

One node in the binary cluster tree.
- `indices`: range into the permutation array (contiguous)
- `bbox_min`, `bbox_max`: axis-aligned bounding box
- `left`, `right`: child indices (0 = this node is a leaf)
- `level`: depth in the tree (root = 0)
"""
struct ClusterNode
    indices::UnitRange{Int}
    bbox_min::Vec3
    bbox_max::Vec3
    left::Int
    right::Int
    level::Int
end

"""
    ClusterTree

Binary cluster tree with flat node storage.
- `nodes`: vector of `ClusterNode`, root is `nodes[1]`
- `perm`: tree-order → original index mapping
- `iperm`: original index → tree-order mapping
- `leaf_size`: maximum cluster size at leaves
"""
struct ClusterTree
    nodes::Vector{ClusterNode}
    perm::Vector{Int}
    iperm::Vector{Int}
    leaf_size::Int
end

"""
    build_cluster_tree(centers;
                       leaf_size=64,
                       max_nodes=30_000_000,
                       max_storage_bytes=2_000_000_000)

Build a binary cluster tree by recursive bisection along the longest
bounding-box axis. `centers` is a `Vector{Vec3}` of point locations
(typically from `rwg_centers`).
"""
function build_cluster_tree(
        centers::Vector{Vec3};
        leaf_size::Int=64,
        max_nodes::Integer=_DEFAULT_MAX_CLUSTER_TREE_NODES,
        max_storage_bytes::Integer=_DEFAULT_MAX_CLUSTER_TREE_STORAGE_BYTES)
    N = length(centers)
    N > 0 ||
        throw(ArgumentError("build_cluster_tree: centers must not be empty"))
    leaf_size >= 1 ||
        throw(ArgumentError(
            "build_cluster_tree: leaf_size must be at least 1, got $leaf_size"))
    node_limit = _validated_resource_limit("max_nodes", max_nodes)
    node_bound, storage_bound = _cluster_tree_resource_bounds(N, leaf_size)
    node_bound <= node_limit ||
        throw(ArgumentError(
            "cluster tree requires at most $node_bound nodes, exceeding " *
            "max_nodes=$node_limit"))
    _enforce_payload_limit(
        storage_bound, max_storage_bytes,
        "cluster-tree arrays", "max_storage_bytes")
    @inbounds for i in eachindex(centers)
        all(isfinite, centers[i]) ||
            throw(ArgumentError(
                "build_cluster_tree: center $i must be finite, got $(centers[i])"))
    end

    perm = collect(1:N)
    nodes = ClusterNode[]
    sizehint!(nodes, node_bound)

    function _build!(lo::Int, hi::Int, level::Int)
        # Compute bounding box
        bmin = Vec3(Inf, Inf, Inf)
        bmax = Vec3(-Inf, -Inf, -Inf)
        for k in lo:hi
            c = centers[perm[k]]
            bmin = Vec3(min(bmin[1], c[1]), min(bmin[2], c[2]), min(bmin[3], c[3]))
            bmax = Vec3(max(bmax[1], c[1]), max(bmax[2], c[2]), max(bmax[3], c[3]))
        end

        span = bmax - bmin
        all(isfinite, span) ||
            throw(ArgumentError(
                "build_cluster_tree: cluster coordinate extent is too large"))

        count = hi - lo + 1
        if count <= leaf_size
            push!(nodes, ClusterNode(lo:hi, bmin, bmax, 0, 0, level))
            return length(nodes)
        end

        # Split along longest axis
        axis = 1
        if span[2] > span[axis]
            axis = 2
        end
        if span[3] > span[axis]
            axis = 3
        end

        # Median split: sort perm[lo:hi] by coordinate along axis
        sort!(@view(perm[lo:hi]); by=k -> centers[k][axis])
        mid = (lo + hi) >> 1  # bisect

        # Reserve slot for this node
        push!(nodes, ClusterNode(lo:hi, bmin, bmax, 0, 0, level))
        my_idx = length(nodes)

        left_idx = _build!(lo, mid, level + 1)
        right_idx = _build!(mid + 1, hi, level + 1)

        # Patch children
        nodes[my_idx] = ClusterNode(lo:hi, bmin, bmax, left_idx, right_idx, level)
        return my_idx
    end

    _build!(1, N, 0)

    # Build inverse permutation
    iperm = Vector{Int}(undef, N)
    for k in 1:N
        iperm[perm[k]] = k
    end

    return ClusterTree(nodes, perm, iperm, leaf_size)
end

"""
    cluster_diameter(tree, node_idx)

Maximum dimension of the bounding box of cluster `node_idx`.
"""
function cluster_diameter(tree::ClusterTree, node_idx::Int)
    node = tree.nodes[node_idx]
    span = node.bbox_max - node.bbox_min
    all(isfinite, span) && all(x -> x >= 0.0, span) ||
        throw(ArgumentError(
            "cluster $node_idx has invalid bounding-box extents"))
    return max(span[1], span[2], span[3])
end

"""
    cluster_distance(tree, i, j)

Minimum distance between the bounding boxes of clusters `i` and `j`.
Returns 0 if the boxes overlap.
"""
function cluster_distance(tree::ClusterTree, i::Int, j::Int)
    ni = tree.nodes[i]
    nj = tree.nodes[j]
    gap1 = max(ni.bbox_min[1] - nj.bbox_max[1],
               nj.bbox_min[1] - ni.bbox_max[1], 0.0)
    gap2 = max(ni.bbox_min[2] - nj.bbox_max[2],
               nj.bbox_min[2] - ni.bbox_max[2], 0.0)
    gap3 = max(ni.bbox_min[3] - nj.bbox_max[3],
               nj.bbox_min[3] - ni.bbox_max[3], 0.0)
    (isfinite(gap1) && isfinite(gap2) && isfinite(gap3)) ||
        throw(OverflowError(
            "cluster bounding-box separation is non-finite"))
    distance = norm(Vec3(gap1, gap2, gap3))
    isfinite(distance) ||
        throw(OverflowError("cluster distance overflowed"))
    return distance
end

@noinline function _cluster_admissibility_exact(
        tree::ClusterTree, i::Int, j::Int, eta::Float64)
    ni = tree.nodes[i]
    nj = tree.nodes[j]
    zero_exact = Rational{BigInt}(0)

    gaps = ntuple(component -> max(
        Rational{BigInt}(ni.bbox_min[component]) -
            Rational{BigInt}(nj.bbox_max[component]),
        Rational{BigInt}(nj.bbox_min[component]) -
            Rational{BigInt}(ni.bbox_max[component]),
        zero_exact,
    ), 3)
    spans_i = ntuple(component ->
        Rational{BigInt}(ni.bbox_max[component]) -
        Rational{BigInt}(ni.bbox_min[component]), 3)
    spans_j = ntuple(component ->
        Rational{BigInt}(nj.bbox_max[component]) -
        Rational{BigInt}(nj.bbox_min[component]), 3)
    diameter = min(max(spans_i...), max(spans_j...))
    eta_exact = Rational{BigInt}(eta)
    squared_distance =
        gaps[1] * gaps[1] + gaps[2] * gaps[2] + gaps[3] * gaps[3]
    return diameter * diameter <=
           eta_exact * eta_exact * squared_distance
end

"""
    is_admissible(tree, i, j; eta=1.5)

Test the standard H-matrix admissibility condition:
  min(diam(i), diam(j)) <= eta * dist(i, j)

Returns `true` if the block (i, j) can be approximated as low-rank.
"""
function is_admissible(tree::ClusterTree, i::Int, j::Int; eta::Float64=1.5)
    (isfinite(eta) && eta > 0.0) ||
        throw(ArgumentError(
            "eta must be finite and positive, got $eta"))
    d = cluster_distance(tree, i, j)
    d <= 0.0 && return false  # overlapping or touching
    diam_min = min(cluster_diameter(tree, i), cluster_diameter(tree, j))
    scaled_distance = eta * d
    isinf(scaled_distance) && return true
    iszero(scaled_distance) && return iszero(diam_min)

    # Bounding-box subtraction, the Euclidean norm, and eta scaling each
    # round independently. Only the narrow comparison boundary needs an exact
    # rational squared-distance decision; ordinary block traversal remains on
    # the allocation-free Float64 path.
    comparison_scale = max(diam_min, scaled_distance)
    uncertainty = 64 * eps(comparison_scale)
    separation = abs(diam_min - scaled_distance)
    isfinite(uncertainty) && separation > uncertainty &&
        return diam_min <= scaled_distance
    return _cluster_admissibility_exact(tree, i, j, eta)
end

"""
    is_leaf(tree, node_idx)

Return `true` if the node is a leaf (no children).
"""
is_leaf(tree::ClusterTree, node_idx::Int) = tree.nodes[node_idx].left == 0

"""
    leaf_nodes(tree)

Return indices of all leaf nodes in the tree.
"""
function leaf_nodes(tree::ClusterTree)
    leaves = Int[]
    for i in eachindex(tree.nodes)
        if is_leaf(tree, i)
            push!(leaves, i)
        end
    end
    return leaves
end
