# ClusterTree.jl — Binary space-partitioning tree for H-matrix blocking
#
# Bisects RWG basis function centers along the longest bounding-box dimension.
# Used by ACA.jl to determine admissible (low-rank) vs inadmissible (dense) blocks.

export ClusterNode, ClusterTree
export build_cluster_tree, cluster_diameter, cluster_distance, is_admissible
export is_leaf, leaf_nodes

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
    build_cluster_tree(centers; leaf_size=64)

Build a binary cluster tree by recursive bisection along the longest
bounding-box axis. `centers` is a `Vector{Vec3}` of point locations
(typically from `rwg_centers`).
"""
function build_cluster_tree(centers::Vector{Vec3}; leaf_size::Int=64)
    N = length(centers)
    N > 0 || error("build_cluster_tree: empty centers")
    leaf_size >= 1 || error("build_cluster_tree: leaf_size must be >= 1")

    perm = collect(1:N)
    nodes = ClusterNode[]

    function _build!(lo::Int, hi::Int, level::Int)
        # Compute bounding box
        bmin = Vec3(Inf, Inf, Inf)
        bmax = Vec3(-Inf, -Inf, -Inf)
        for k in lo:hi
            c = centers[perm[k]]
            bmin = Vec3(min(bmin[1], c[1]), min(bmin[2], c[2]), min(bmin[3], c[3]))
            bmax = Vec3(max(bmax[1], c[1]), max(bmax[2], c[2]), max(bmax[3], c[3]))
        end

        count = hi - lo + 1
        if count <= leaf_size
            push!(nodes, ClusterNode(lo:hi, bmin, bmax, 0, 0, level))
            return length(nodes)
        end

        # Split along longest axis
        span = bmax - bmin
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
    d2 = 0.0
    for ax in 1:3
        gap = max(ni.bbox_min[ax] - nj.bbox_max[ax],
                  nj.bbox_min[ax] - ni.bbox_max[ax],
                  0.0)
        d2 += gap * gap
    end
    return sqrt(d2)
end

"""
    is_admissible(tree, i, j; eta=1.5)

Test the standard H-matrix admissibility condition:
  min(diam(i), diam(j)) <= eta * dist(i, j)

Returns `true` if the block (i, j) can be approximated as low-rank.
"""
function is_admissible(tree::ClusterTree, i::Int, j::Int; eta::Float64=1.5)
    d = cluster_distance(tree, i, j)
    d <= 0.0 && return false  # overlapping or touching
    diam_min = min(cluster_diameter(tree, i), cluster_diameter(tree, j))
    return diam_min <= eta * d
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
