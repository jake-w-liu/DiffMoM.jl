# API: Octree Spatial Decomposition

## Purpose

The octree module provides the spatial data structure used by MLFMA to partition the scatterer geometry into a hierarchy of boxes. The octree determines which BF pairs interact via near-field (direct computation) versus far-field (multipole translation), and provides the BF reordering for spatial locality.

---

## Types

### `OctreeBox`

A single box at one level of the octree.

```julia
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
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `Int` | 1-based box index within its level. |
| `ijk` | `NTuple{3,Int}` | Integer grid coordinates (0-based) at this level. |
| `center` | `Vec3` | Physical center position of the box (meters). |
| `edge_length` | `Float64` | Box edge length in meters. |
| `bf_range` | `UnitRange{Int}` | Range into `octree.perm[]` for BFs in this box. Empty (`1:0`) for non-leaf boxes. |
| `children` | `Vector{Int}` | Child box IDs at the next finer level. Empty for leaf boxes. |
| `parent` | `Int` | Parent box ID at the next coarser level. `0` for root-level boxes. |
| `neighbors` | `Vector{Int}` | Same-level boxes within +/-1 in each grid dimension (including self). Near-field interactions. |
| `interaction_list` | `Vector{Int}` | Same-level boxes that are children of parent's neighbors but NOT self's neighbors. Far-field (multipole) interactions. |

---

### `OctreeLevel`

One level of the octree hierarchy.

```julia
struct OctreeLevel
    id::Int                         # 1 = root (coarsest), nLevels = leaf (finest)
    edge_length::Float64
    boxes::Vector{OctreeBox}
    ijk_map::Dict{NTuple{3,Int}, Int}
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `Int` | Level index: `1` = root (coarsest), `nLevels` = leaf (finest). |
| `edge_length` | `Float64` | Box edge length at this level (meters). |
| `boxes` | `Vector{OctreeBox}` | All non-empty boxes at this level. |
| `ijk_map` | `Dict{NTuple{3,Int}, Int}` | Maps grid coordinates `(i,j,k)` to box index within `boxes`. |

---

### `Octree`

Complete octree data structure for MLFMA.

```julia
struct Octree
    nLevels::Int
    levels::Vector{OctreeLevel}     # levels[1] = root, levels[nLevels] = leaf
    perm::Vector{Int}               # tree-order → original BF index
    iperm::Vector{Int}              # original → tree-order BF index
    N::Int                          # total BFs
    origin::Vec3                    # lower corner of bounding cube
    root_edge::Float64              # edge length of root box
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `nLevels` | `Int` | Total number of octree levels (root = 1, leaf = nLevels). |
| `levels` | `Vector{OctreeLevel}` | Hierarchy from coarsest (`levels[1]`) to finest (`levels[nLevels]`). |
| `perm` | `Vector{Int}` | Permutation from tree ordering to original BF index: `perm[tree_idx] = orig_idx`. |
| `iperm` | `Vector{Int}` | Inverse permutation: `iperm[orig_idx] = tree_idx`. |
| `N` | `Int` | Total number of basis functions. |
| `origin` | `Vec3` | Lower corner of the bounding cube (meters). |
| `root_edge` | `Float64` | Edge length of the root-level bounding cube (meters). |

---

## Functions

### `build_octree(centers, k; leaf_lambda=0.25)`

Build an octree over RWG basis function centers for MLFMA.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `centers` | `Vector{Vec3}` | -- | BF center positions (from `rwg_centers`). |
| `k` | `Float64` | -- | Wavenumber (rad/m). |
| `leaf_lambda` | `Float64` | `0.25` | Leaf box edge length in wavelengths. |

**Returns:** `Octree` with BFs permuted for spatial locality.

**Algorithm:**

1. Compute a bounding cube enclosing all BF centers (with 1% padding)
2. Determine the number of levels: `nLevels = ceil(log2(cube_edge / leaf_edge)) + 1`
3. Assign each BF to a leaf box based on its grid coordinates
4. Sort BFs by leaf box (lexicographic on `(i,j,k)`) for spatial locality
5. Build coarser levels bottom-up by grouping child boxes `(i>>1, j>>1, k>>1)`
6. Compute neighbor lists (boxes within +/-1 in each dimension)
7. Compute interaction lists (children of parent's neighbors, minus own neighbors)

**Example:**

```julia
centers = rwg_centers(mesh, rwg)
k = 2pi * freq / c0
octree = build_octree(centers, k; leaf_lambda=1.0)
println("Levels: ", octree.nLevels)
println("Leaf boxes: ", length(octree.levels[end].boxes))
```

---

## Octree Decomposition

The octree partitions the BF-pair interaction space into two disjoint sets:

- **Near-field pairs:** BFs in neighboring leaf boxes (distance <= 1 box apart). These are computed directly using the full EFIE kernel and stored in a sparse matrix `Z_near`.
- **Far-field pairs:** BFs in non-neighboring boxes at any level. These are computed via the multipole translation algorithm (aggregation -> translation -> disaggregation).

The decomposition is verified to be complete: every BF pair `(m, n)` is accounted for exactly once, either in the near-field or via the interaction lists across all levels.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/fast/Octree.jl` | `Octree`, `OctreeBox`, `OctreeLevel`, `build_octree` |
| `src/fast/MLFMA.jl` | Consumer of octree data structures. See [mlfma.md](mlfma.md). |
