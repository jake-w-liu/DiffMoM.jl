# API: Spatial Patch Assignment

## Purpose

Reference for automatic spatial partitioning of mesh triangles into impedance design patches. These utilities replace manual `PatchPartition` construction with spatial-aware assignment strategies that are essential for practical optimization of complex geometries (e.g., "coat the nose cone", "add absorber to leading edges").

All functions return a `PatchPartition` compatible with `precompute_patch_mass` and the full optimization pipeline.

---

## Grid-Based Partitioning

### `assign_patches_grid(mesh; nx=4, ny=4, nz=1)`

Partition mesh triangles by dividing the bounding box into an `nx * ny * nz` grid. Each occupied cell becomes a patch. Empty cells are skipped and patch IDs are renumbered consecutively.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `nx` | `Int` | `4` | Number of grid divisions along x. |
| `ny` | `Int` | `4` | Number of grid divisions along y. |
| `nz` | `Int` | `1` | Number of grid divisions along z. Set to 1 for planar or nearly-planar geometries. |

**Returns:** `PatchPartition` with `P <= nx * ny * nz` patches (empty cells skipped).

**Algorithm:**
1. Compute triangle centroids via `triangle_center(mesh, t)`.
2. Compute the bounding box of all centroids (with small epsilon padding).
3. Map each centroid to a grid cell `(ix, iy, iz)` via floor division.
4. Convert to a linear index: `id = ix + iy*nx + iz*nx*ny + 1`.
5. Renumber to consecutive patch IDs (skip empty cells).

**Example:**

```julia
mesh = make_rect_plate(1.0, 1.0, 20, 20)

# 4x4 grid → up to 16 patches
partition = assign_patches_grid(mesh; nx=4, ny=4, nz=1)
println("Patches: ", partition.P)   # 16 for a flat plate

# Finer spatial resolution
partition_fine = assign_patches_grid(mesh; nx=8, ny=8, nz=1)
println("Patches: ", partition_fine.P)   # 64
```

**Choosing grid resolution:**

| Geometry | Recommended `(nx, ny, nz)` | Typical P |
|----------|----------------------------|-----------|
| Flat plate | `(4, 4, 1)` to `(8, 8, 1)` | 16--64 |
| Cylinder | `(4, 4, 4)` | 16--64 |
| Aircraft-like | `(6, 4, 2)` | 24--48 |
| Sphere | `(4, 4, 4)` | 20--64 (many cells empty) |

---

## Region-Based Partitioning

### `assign_patches_by_region(mesh, regions)`

Assign triangles to patches based on spatial predicate functions. Each element of `regions` is a function `f(centroid::Vec3) -> Bool`. Triangle `t` is assigned to the first region whose predicate returns `true`. Unmatched triangles are collected into an extra "background" patch (the last patch).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh` | `TriMesh` | Triangle mesh. |
| `regions` | `Vector{<:Function}` | Predicate functions, tested in order. First match wins. |

**Returns:** `PatchPartition` with `P = length(regions) + 1` patches (including background).

**Example:**

```julia
# Split mesh into "nose" (x > 0.5) and "tail" (x <= 0.5)
regions = [
    region_halfspace(axis=:x, threshold=0.5, above=true),   # patch 1: nose
    region_halfspace(axis=:x, threshold=0.5, above=false),   # patch 2: tail
]
partition = assign_patches_by_region(mesh, regions)
# patch 3 = background (if any triangles match neither, which can't happen here)
```

**Controlling PEC regions:** To keep some patches as PEC (uncoated), set their box constraints to zero in the optimizer:

```julia
# Only optimize patches 1 and 2; patch 3 (background) stays PEC
lb = [0.0, 0.0, 0.0]
ub = [500.0, 500.0, 0.0]   # ub=lb=0 for PEC patches
```

---

## Region Predicate Constructors

### `region_halfspace(; axis, threshold, above=true)`

Create a predicate selecting triangles whose centroid satisfies `centroid[axis] >= threshold` (if `above=true`) or `centroid[axis] < threshold` (if `above=false`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `axis` | `Symbol` | -- | `:x`, `:y`, or `:z`. |
| `threshold` | `Float64` | -- | Coordinate threshold (meters). |
| `above` | `Bool` | `true` | `true` = select triangles at or above threshold. |

**Returns:** `Function` (predicate `Vec3 -> Bool`).

**Example:**

```julia
# Select upper half of geometry
upper = region_halfspace(axis=:z, threshold=0.0, above=true)

# Select left side
left = region_halfspace(axis=:y, threshold=0.0, above=false)
```

---

### `region_sphere(; center, radius)`

Create a predicate selecting triangles whose centroid is within `radius` of `center`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `center` | `Vec3` | Center of the selection sphere (meters). |
| `radius` | `Float64` | Selection radius (meters). |

**Returns:** `Function` (predicate `Vec3 -> Bool`).

**Example:**

```julia
# Select triangles near the tip of a nose cone
nose_tip = region_sphere(center=Vec3(0.5, 0.0, 0.0), radius=0.1)
```

---

### `region_box(; lo, hi)`

Create a predicate selecting triangles whose centroid is inside the axis-aligned box `[lo, hi]`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `lo` | `Vec3` | Lower corner of the box (meters). |
| `hi` | `Vec3` | Upper corner of the box (meters). |

**Returns:** `Function` (predicate `Vec3 -> Bool`).

**Example:**

```julia
# Select a rectangular region on a plate
wing_section = region_box(lo=Vec3(-0.2, -0.5, -0.01),
                           hi=Vec3(0.2, 0.5, 0.01))
```

---

### Combining Predicates

Predicates are standard Julia functions and can be composed with boolean logic:

```julia
# Select triangles that are both in the upper half AND within a sphere
upper = region_halfspace(axis=:z, threshold=0.0, above=true)
near_center = region_sphere(center=Vec3(0,0,0), radius=0.3)

# Combine: Julia closure
upper_and_near = c -> upper(c) && near_center(c)

regions = [upper_and_near]
partition = assign_patches_by_region(mesh, regions)
```

---

## K-Means Partitioning

### `assign_patches_uniform(mesh; n_patches)`

Partition all triangles into `n_patches` spatial groups using k-means clustering of triangle centroids. This produces spatially compact patches of roughly equal size, regardless of geometry.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh` | `TriMesh` | Triangle mesh. |
| `n_patches` | `Int` | Desired number of patches (must be >= 1 and <= `ntriangles(mesh)`). |

**Returns:** `PatchPartition` with `P <= n_patches` patches (may be fewer if clusters merge).

**Algorithm:**
1. Compute triangle centroids.
2. Initialize k-means cluster centers via random permutation (seed=42 for determinism).
3. Run Lloyd's algorithm (max 100 iterations) assigning each triangle to its nearest center.
4. Renumber to consecutive IDs if any clusters are empty.

**Example:**

```julia
mesh = make_rect_plate(1.0, 1.0, 20, 20)

# 10 spatially-uniform patches
partition = assign_patches_uniform(mesh; n_patches=10)
println("Patches: ", partition.P)   # 10
```

**When to use k-means vs grid:**

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| `assign_patches_grid` | Axis-aligned, predictable layout, fast | Poor for curved or irregular shapes |
| `assign_patches_uniform` | Adapts to geometry shape, equal-size patches | Slightly slower (iterative), non-deterministic without fixed seed |
| `assign_patches_by_region` | Maximum user control, semantically meaningful | Requires manual predicate construction |

---

## Integration with Optimization Pipeline

All patch assignment functions return `PatchPartition`, which feeds directly into the standard optimization pipeline:

```julia
# 1. Assign patches (any method)
partition = assign_patches_grid(mesh; nx=4, ny=4)

# 2. Precompute patch mass matrices
Mp = precompute_patch_mass(mesh, rwg, partition)

# 3. Set up optimization
theta0 = zeros(partition.P)
lb = zeros(partition.P)          # passive resistive sheets
ub = fill(500.0, partition.P)

# 4. Run optimizer (dense or MLFMA-based)
theta_opt, trace = optimize_lbfgs(Z_efie, Mp, v, Q, theta0;
    lb=lb, ub=ub, maxiter=100)

# Or with MLFMA:
theta_opt, trace = optimize_multiangle_rcs(mlfma_op, Mp, configs, theta0;
    lb=lb, ub=ub, preconditioner=P_nf)
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/assembly/SpatialPatches.jl` | `assign_patches_grid`, `assign_patches_by_region`, `assign_patches_uniform`, `region_halfspace`, `region_sphere`, `region_box` |
| `src/Types.jl` | `PatchPartition` type definition |
| `src/assembly/Impedance.jl` | `precompute_patch_mass` (consumes `PatchPartition`) |

---

## Exercises

- **Basic:** Create a 4x4 grid partition on a rectangular plate mesh and verify that every triangle has a valid patch ID in `1:P`.
- **Practical:** Use `region_halfspace` to split a mesh into "top" and "bottom" halves. Precompute mass matrices and verify that `sum(Mp[1])` and `sum(Mp[2])` are roughly proportional to the area of each half.
- **Challenge:** Compare `assign_patches_grid(mesh; nx=4, ny=4)` against `assign_patches_uniform(mesh; n_patches=16)` on a sphere mesh. Visualize the resulting patch assignments and discuss which produces more spatially uniform patches on curved surfaces.
