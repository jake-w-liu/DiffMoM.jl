# Octree.jl — Spatial octree for MLFMA
#
# Builds an octree over RWG basis function centers for use by the MLFMA
# operator. Each non-empty leaf box stores a contiguous range of BFs
# (via permutation). Neighbor and interaction lists are precomputed.

export Octree, OctreeBox, OctreeLevel, build_octree

struct OctreeBox
    id::Int                         # 1-based index within level
    ijk::NTuple{3,Int}              # integer grid position (0-based)
    center::Vec3                    # physical center of box
    edge_length::Float64
    bf_range::UnitRange{Int}        # range into perm[] (empty 1:0 if no BFs at this level)
    children::Vector{Int}           # child box IDs at next finer level
    parent::Int                     # parent box ID at next coarser level (0 for root)
    neighbors::Vector{Int}          # same-level neighbor box IDs (including self)
    interaction_list::Vector{Int}   # same-level far-interaction box IDs
end

struct OctreeLevel
    id::Int                         # 1 = root (coarsest), nLevels = leaf (finest)
    edge_length::Float64
    boxes::Vector{OctreeBox}
    ijk_map::Dict{NTuple{3,Int}, Int}  # (i,j,k) → box index
end

struct Octree
    nLevels::Int
    levels::Vector{OctreeLevel}     # levels[1] = root, levels[nLevels] = leaf
    perm::Vector{Int}               # tree-order → original BF index
    iperm::Vector{Int}              # original → tree-order BF index
    N::Int                          # total BFs
    origin::Vec3                    # lower corner of bounding cube
    root_edge::Float64              # edge length of root box
end

"""
    build_octree(centers, k; leaf_lambda=0.25)

Build an octree over RWG basis function centers for MLFMA.

`centers` is a `Vector{Vec3}` of BF center positions.
`k` is the wavenumber. Leaf boxes have edge length `leaf_lambda * λ`.

Returns an `Octree` with BFs permuted for spatial locality.
"""
function build_octree(centers::Vector{Vec3}, k::Float64; leaf_lambda::Float64=0.25)
    N = length(centers)
    N > 0 || error("build_octree: empty centers")
    λ = 2π / k
    leaf_edge = leaf_lambda * λ

    # 1. Bounding cube
    cmin = Vec3(Inf, Inf, Inf)
    cmax = Vec3(-Inf, -Inf, -Inf)
    for c in centers
        cmin = Vec3(min(cmin[1], c[1]), min(cmin[2], c[2]), min(cmin[3], c[3]))
        cmax = Vec3(max(cmax[1], c[1]), max(cmax[2], c[2]), max(cmax[3], c[3]))
    end
    span = cmax - cmin
    max_span = max(span[1], span[2], span[3], leaf_edge)
    # Pad by 1% to avoid edge cases
    pad = 0.01 * max_span
    origin = cmin - Vec3(pad, pad, pad)
    cube_edge = max_span + 2 * pad

    # 2. Number of levels
    nLevels = max(2, ceil(Int, log2(cube_edge / leaf_edge)) + 1)
    # Adjust leaf_edge so root box = leaf_edge * 2^(nLevels-1)
    root_edge = leaf_edge * (1 << (nLevels - 1))

    # Re-center origin so the root box is centered on the geometry
    geo_center = (cmin + cmax) / 2
    origin = geo_center - Vec3(root_edge / 2, root_edge / 2, root_edge / 2)

    # 3. Assign each BF to a leaf box
    leaf_ijk = Vector{NTuple{3,Int}}(undef, N)
    for n in 1:N
        rel = centers[n] - origin
        i = clamp(floor(Int, rel[1] / leaf_edge), 0, (1 << (nLevels - 1)) - 1)
        j = clamp(floor(Int, rel[2] / leaf_edge), 0, (1 << (nLevels - 1)) - 1)
        kk = clamp(floor(Int, rel[3] / leaf_edge), 0, (1 << (nLevels - 1)) - 1)
        leaf_ijk[n] = (i, j, kk)
    end

    # 4. Sort BFs by leaf box (lexicographic on ijk) for spatial locality
    perm = sortperm(leaf_ijk)
    iperm = Vector{Int}(undef, N)
    for n in 1:N
        iperm[perm[n]] = n
    end

    # 5. Build leaf level
    sorted_ijk = leaf_ijk[perm]
    leaf_boxes = OctreeBox[]
    leaf_ijk_map = Dict{NTuple{3,Int}, Int}()

    n_start = 1
    while n_start <= N
        cur_ijk = sorted_ijk[n_start]
        n_end = n_start
        while n_end < N && sorted_ijk[n_end + 1] == cur_ijk
            n_end += 1
        end
        box_id = length(leaf_boxes) + 1
        box_center = origin + Vec3(
            (cur_ijk[1] + 0.5) * leaf_edge,
            (cur_ijk[2] + 0.5) * leaf_edge,
            (cur_ijk[3] + 0.5) * leaf_edge,
        )
        push!(leaf_boxes, OctreeBox(
            box_id, cur_ijk, box_center, leaf_edge,
            n_start:n_end,  # bf_range
            Int[],          # children (leaf has none)
            0,              # parent (set later)
            Int[],          # neighbors (set later)
            Int[],          # interaction_list (set later)
        ))
        leaf_ijk_map[cur_ijk] = box_id
        n_start = n_end + 1
    end

    leaf_level = OctreeLevel(nLevels, leaf_edge, leaf_boxes, leaf_ijk_map)

    # 6. Build coarser levels bottom-up
    levels = Vector{OctreeLevel}(undef, nLevels)
    levels[nLevels] = leaf_level

    for l in (nLevels - 1):-1:1
        child_level = levels[l + 1]
        edge_l = leaf_edge * (1 << (nLevels - l))
        parent_boxes = OctreeBox[]
        parent_ijk_map = Dict{NTuple{3,Int}, Int}()

        for cbox in child_level.boxes
            p_ijk = (cbox.ijk[1] >> 1, cbox.ijk[2] >> 1, cbox.ijk[3] >> 1)
            if !haskey(parent_ijk_map, p_ijk)
                pid = length(parent_boxes) + 1
                parent_ijk_map[p_ijk] = pid
                p_center = origin + Vec3(
                    (p_ijk[1] + 0.5) * edge_l,
                    (p_ijk[2] + 0.5) * edge_l,
                    (p_ijk[3] + 0.5) * edge_l,
                )
                push!(parent_boxes, OctreeBox(
                    pid, p_ijk, p_center, edge_l,
                    1:0,        # bf_range (non-leaf)
                    Int[],      # children (populated below)
                    0,          # parent (set in next iteration)
                    Int[],      # neighbors
                    Int[],      # interaction_list
                ))
            end
        end

        # Link children → parent
        for (ci, cbox) in enumerate(child_level.boxes)
            p_ijk = (cbox.ijk[1] >> 1, cbox.ijk[2] >> 1, cbox.ijk[3] >> 1)
            pid = parent_ijk_map[p_ijk]
            push!(parent_boxes[pid].children, ci)
            # Update child's parent
            child_level.boxes[ci] = OctreeBox(
                cbox.id, cbox.ijk, cbox.center, cbox.edge_length,
                cbox.bf_range, cbox.children, pid,
                cbox.neighbors, cbox.interaction_list,
            )
        end

        levels[l] = OctreeLevel(l, edge_l, parent_boxes, parent_ijk_map)
    end

    # 7. Compute neighbors and interaction lists at each level
    for l in 1:nLevels
        _compute_neighbors!(levels[l])
    end
    for l in 2:nLevels
        _compute_interaction_list!(levels, l)
    end

    return Octree(nLevels, levels, perm, iperm, N, origin, root_edge)
end

"""
Compute neighbor lists for all boxes at a level.
Neighbors = boxes within ±1 in each ijk dimension (including self).
"""
function _compute_neighbors!(level::OctreeLevel)
    for (bi, box) in enumerate(level.boxes)
        nbrs = Int[]
        for di in -1:1, dj in -1:1, dk in -1:1
            nbr_ijk = (box.ijk[1] + di, box.ijk[2] + dj, box.ijk[3] + dk)
            nbr_id = get(level.ijk_map, nbr_ijk, 0)
            if nbr_id > 0
                push!(nbrs, nbr_id)
            end
        end
        level.boxes[bi] = OctreeBox(
            box.id, box.ijk, box.center, box.edge_length,
            box.bf_range, box.children, box.parent,
            nbrs, box.interaction_list,
        )
    end
end

"""
Compute interaction lists at level l (requires level l-1 neighbors to be set).

Interaction list of box c at level l = children of parent's neighbors at level l,
minus c's own neighbors at level l.
"""
function _compute_interaction_list!(levels::Vector{OctreeLevel}, l::Int)
    child_level = levels[l]
    parent_level = levels[l - 1]

    for (bi, box) in enumerate(child_level.boxes)
        pid = box.parent
        pid > 0 || continue
        parent_box = parent_level.boxes[pid]

        # Collect all children of parent's neighbors
        candidates = Set{Int}()
        for pnbr_id in parent_box.neighbors
            pnbr = parent_level.boxes[pnbr_id]
            for child_id in pnbr.children
                push!(candidates, child_id)
            end
        end

        # Remove own neighbors (including self)
        nbr_set = Set(box.neighbors)
        ilist = sort!(collect(setdiff(candidates, nbr_set)))

        child_level.boxes[bi] = OctreeBox(
            box.id, box.ijk, box.center, box.edge_length,
            box.bf_range, box.children, box.parent,
            box.neighbors, ilist,
        )
    end
end
