# API: Mesh Utilities

## Purpose

Reference for mesh generation, OBJ file I/O, quality diagnostics, repair, and coarsening. These functions handle the first stage of any MoM simulation: creating or importing a triangle mesh and ensuring it is suitable for RWG basis construction and EFIE assembly.

A mesh that fails quality checks will cause errors or silently wrong results in downstream stages. Always run quality checks on imported meshes before proceeding.

---

## Creation and IO

### `make_rect_plate(Lx, Ly, Nx, Ny)`

Creates a triangulated rectangular plate in the xy-plane, centered at the origin. This is the most common mesh for development and testing.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Lx` | `Real` | Plate length in x-direction (meters). For a half-wavelength plate at 1 GHz: `Lx = 0.15`. |
| `Ly` | `Real` | Plate length in y-direction (meters). |
| `Nx` | `Int` | Number of cells (subdivisions) along x. Must be >= 1. More cells = finer mesh = more RWG basis functions. |
| `Ny` | `Int` | Number of cells along y. Must be >= 1. |

**Returns:** `TriMesh` with `(Nx+1)*(Ny+1)` vertices and `2*Nx*Ny` triangles (each rectangular cell is split into 2 triangles).

**Choosing Nx, Ny:** A rule of thumb for MoM is ~10 edges per wavelength. For a square plate of side L at wavelength lambda:
- `Nx = Ny = round(Int, 10 * L / lambda)` gives approximately lambda/10 edge length.
- Doubling `Nx, Ny` roughly quadruples the RWG count (and 16x the matrix fill time, since assembly is O(N^2)).

---

### `make_rect_plate_graded(Lx, Ly, Nx, Ny; grading_factor=3.0)`

Creates a triangulated rectangular plate in the xy-plane with graded mesh density near the edges. Same topology as `make_rect_plate` but vertex positions are redistributed using a tanh grading function that clusters elements near the plate boundaries.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Lx` | `Real` | -- | Plate length in x-direction (meters). |
| `Ly` | `Real` | -- | Plate length in y-direction (meters). |
| `Nx` | `Int` | -- | Number of cells along x. |
| `Ny` | `Int` | -- | Number of cells along y. |
| `grading_factor` | `Real` | `3.0` | Controls edge clustering strength. Must be positive. |

**Returns:** `TriMesh` with `(Nx+1)*(Ny+1)` vertices and `2*Nx*Ny` triangles (same count as `make_rect_plate`).

**Grading factor guide:**

| `grading_factor` | Edge-to-center density ratio | Use case |
|-------------------|------------------------------|----------|
| `1.0` | ~1.5:1 (nearly uniform) | Mild refinement near edges |
| `3.0` (default) | ~5:1 | Standard edge clustering |
| `5.0` | ~10:1 | Strong edge clustering |

Values above 6 may create highly skewed center elements.

**Example:**
```julia
mesh = make_rect_plate_graded(0.1, 0.1, 12, 12; grading_factor=3.0)
rwg = build_rwg(mesh)
```

---

### `make_parabolic_reflector(D, f, Nr, Nphi; center=Vec3(0,0,0))`

Creates an open parabolic reflector mesh aligned with +z, with surface equation:

```math
z = \frac{x^2 + y^2}{4f}, \qquad x^2 + y^2 \le (D/2)^2.
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `D` | `Real` | Aperture diameter (meters). |
| `f` | `Real` | Focal length (meters). The focal point is at `(0, 0, f)` relative to the apex. Common f/D ratios: 0.3--0.5. |
| `Nr` | `Int` | Number of radial rings (>= 2). More rings = finer radial resolution. |
| `Nphi` | `Int` | Number of azimuthal samples per ring (>= 3). Typically 20--40 for smooth curvature. |
| `center` | `Vec3` | Reflector apex location, default `Vec3(0,0,0)`. |

**Returns:** `TriMesh` with `1 + Nr*Nphi` vertices and `Nphi + 2*(Nr-1)*Nphi` triangles.

**Note:** This is an open surface (has boundary edges). Use `allow_boundary=true` in mesh checks and `build_rwg`.

**Example:**
```julia
mesh = make_parabolic_reflector(0.30, 0.105, 8, 28)  # D=30cm, f=10.5cm
report = assert_mesh_quality(mesh; allow_boundary=true, require_closed=false)
println((nvertices(mesh), ntriangles(mesh)))
```

---

### `read_obj_mesh(path)`

Reads a triangle mesh from a Wavefront OBJ file. OBJ is a widely supported text format exported by most CAD tools and mesh generators.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `AbstractString` | Path to the OBJ file. |

**Returns:** `TriMesh`

**Supported OBJ records:**
- `v x y z` -- vertex coordinates
- `f i j k ...` -- polygon faces (polygons with more than 3 vertices are fan-triangulated automatically)

Texture and normal indices (`f v/t/n`) are parsed but ignored. Both positive and negative OBJ vertex indices are supported.

**Typical workflow for imported meshes:**
```julia
mesh_raw = read_obj_mesh("input.obj")
rep = repair_mesh_for_simulation(mesh_raw)   # clean up
mesh = rep.mesh                              # use repaired mesh
```

---

### `write_obj_mesh(path, mesh; header="Exported by DiffMoM")`

Writes a `TriMesh` to a Wavefront OBJ file. Useful for exporting meshes for visualization in external tools (MeshLab, Blender, etc.) or for archiving simulation meshes.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `AbstractString` | Output file path. |
| `mesh` | `TriMesh` | Triangle mesh to write. |
| `header` | `AbstractString` | Comment line written at the top of the file. |

**Returns:** `path` (the output file path as a string).

---

## Geometry Helpers

These functions compute basic geometric properties of individual triangles. They are used internally by assembly routines and are also useful for custom analysis.

### `triangle_area(mesh, t)`

Compute the area of triangle `t` using the cross-product formula: `A = ||(v2-v1) x (v3-v1)|| / 2`.

**Parameters:** `mesh::TriMesh`, `t::Int` (1-based triangle index).

**Returns:** `Float64` area in m^2.

---

### `triangle_center(mesh, t)`

Compute the centroid of triangle `t`: the average of its three vertex positions.

**Parameters:** `mesh::TriMesh`, `t::Int` (1-based triangle index).

**Returns:** `Vec3` centroid coordinates in meters.

---

### `triangle_normal(mesh, t)`

Compute the outward unit normal of triangle `t`, defined by the right-hand rule applied to the vertex ordering. The normal direction depends on the winding order of the triangle's vertices.

**Parameters:** `mesh::TriMesh`, `t::Int` (1-based triangle index).

**Returns:** `Vec3` unit normal vector.

---

### `mesh_unique_edges(mesh)`

Return all unique undirected edges of the mesh as vertex-index pairs `(i, j)` with `i < j`. This includes both interior edges (shared by two triangles) and boundary edges (belonging to one triangle).

**Parameters:** `mesh::TriMesh`

**Returns:** `Vector{Tuple{Int,Int}}`.

---

### `mesh_wireframe_segments(mesh)`

Build line-segment arrays for lightweight 3D wireframe visualization. Each edge contributes `(p1, p2, NaN)` to coordinate vectors, which is the format expected by `Plots.path3d`.

**Parameters:** `mesh::TriMesh`

**Returns:** named tuple `(x, y, z, n_edges)`.

---

## Quality Checks

Mesh quality checks catch problems that would cause assembly failures or silently incorrect results. Run these on every imported mesh before building RWG basis functions.

**What the checks detect:**

| Problem | Consequence if undetected |
|---------|--------------------------|
| Invalid triangles (repeated vertex indices) | Assembly crash or NaN in matrix |
| Degenerate triangles (zero or near-zero area) | Singular matrix entries, division by zero |
| Non-manifold edges (shared by 3+ triangles) | RWG basis construction failure |
| Orientation conflicts (inconsistent winding) | Wrong sign in matrix entries, incorrect physics |
| Boundary edges (edge with only 1 triangle) | Not an error for open surfaces; error for closed bodies |

### `mesh_quality_report(mesh; area_tol_rel=1e-12, check_orientation=true)`

Compute comprehensive mesh-quality diagnostics.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh to check. |
| `area_tol_rel` | `Float64` | `1e-12` | Relative tolerance for degenerate triangle detection. A triangle is "degenerate" if its area is less than `area_tol_rel * bbox_diagonal^2`. The default is extremely conservative; increase to `1e-8` if your mesh has intentionally tiny triangles. |
| `check_orientation` | `Bool` | `true` | Whether to check winding consistency across shared edges. Disable only if you know the mesh has intentional orientation discontinuities (rare). |

**Returns:** Named tuple with fields:
- `n_vertices`, `n_triangles`, `n_edges_total`, `n_interior_edges`, `n_boundary_edges`, `n_nonmanifold_edges`, `n_orientation_conflicts`
- `n_invalid_triangles`, `n_degenerate_triangles`
- `invalid_triangles`, `degenerate_triangles` (index vectors)
- `area_tol_abs` (the absolute tolerance used, derived from `area_tol_rel`)

---

### `mesh_quality_ok(report; allow_boundary=true, require_closed=false)`

Return `true` if a mesh-quality report passes all hard checks.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `report` | NamedTuple | -- | Output from `mesh_quality_report`. |
| `allow_boundary` | `Bool` | `true` | Allow boundary edges. Set to `true` for open surfaces (plates, reflectors). Set to `false` for closed surfaces where every edge must be interior. |
| `require_closed` | `Bool` | `false` | Require that the surface is closed (zero boundary edges). Use for enclosed PEC bodies (spheres, aircraft, etc.) where boundary edges indicate a mesh defect. |

**Returns:** `Bool`

**Checks performed:**
- No invalid triangles (repeated vertex indices)
- No degenerate triangles (near-zero area)
- No non-manifold edges (3+ incident triangles)
- No orientation conflicts (inconsistent winding across shared edges)
- Boundary edges: allowed unless `allow_boundary=false` or `require_closed=true`

---

### `assert_mesh_quality(mesh; allow_boundary=true, require_closed=false, area_tol_rel=1e-12)`

Run mesh-quality checks and **throw a detailed error** if the mesh is unsuitable for simulation. This is the recommended one-call check: it runs `mesh_quality_report` internally and provides a human-readable error message listing all problems found.

**Parameters:** Same as `mesh_quality_report` plus the `allow_boundary`/`require_closed` flags from `mesh_quality_ok`.

**Returns:** The computed quality report on success (for inspection if desired).

---

## Repair and Coarsening

These functions fix problematic meshes (typically from CAD exports) and optionally reduce mesh density to control problem size.

### `repair_mesh_for_simulation(mesh; ...)`

Repair a triangle mesh so it passes solver prechecks. This is the primary repair entry point.

**Full signature:**
```julia
repair_mesh_for_simulation(mesh;
    allow_boundary=true, require_closed=false, area_tol_rel=1e-12,
    drop_invalid=true, drop_degenerate=true, fix_orientation=true,
    strict_nonmanifold=true, auto_drop_nonmanifold=true)
```

**Pipeline (executed in order):**

1. **Drop invalid triangles** (`drop_invalid=true`): Remove triangles with repeated vertex indices.
2. **Drop degenerate triangles** (`drop_degenerate=true`): Remove triangles with near-zero area (controlled by `area_tol_rel`).
3. **Drop non-manifold triangles** (`auto_drop_nonmanifold=true`): Iteratively remove triangles that create non-manifold edges (edges shared by 3+ triangles). This is common in messy CAD exports.
4. **Fix orientation** (`fix_orientation=true`): Walk the mesh and flip triangles to achieve consistent winding across all interior edges.
5. **Final check**: Verify the repaired mesh passes `mesh_quality_ok`.

**Key parameters:**

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `auto_drop_nonmanifold` | `true` | Set to `false` for strict fail-fast validation (error instead of auto-fix). |
| `strict_nonmanifold` | `true` | Set to `false` only if you want to tolerate non-manifold edges (not recommended). |
| `allow_boundary` | `true` | Set to `false` for closed surfaces. |
| `require_closed` | `false` | Set to `true` for enclosed bodies (sphere, aircraft hull). |

**Returns:** Named tuple containing:
- `mesh::TriMesh`: The repaired mesh.
- `before`, `cleaned`, `after`: Quality reports at each stage.
- `removed_invalid`, `removed_degenerate`: Indices of removed triangles.
- `removed_nonmanifold::Int`: Count of triangles dropped during non-manifold cleanup.
- `flipped_triangles`: Indices of triangles whose orientation was flipped.
- `area_tol_abs`: Absolute area tolerance used.

---

### `repair_obj_mesh(input_path, output_path; kwargs...)`

Convenience wrapper: read an OBJ, repair it, and write the repaired mesh to a new OBJ file.

**Parameters:**
- `input_path::AbstractString`: Path to input OBJ file.
- `output_path::AbstractString`: Path to output OBJ file.
- `kwargs...`: Forwarded to `repair_mesh_for_simulation`.

**Returns:** Same metadata as `repair_mesh_for_simulation`, plus the `output_path`.

---

### `coarsen_mesh_to_target_rwg(mesh, target_rwg; ...)`

Auto-coarsen a mesh by voxel clustering to approach a target RWG basis count. This is useful when an imported mesh is too fine (too many unknowns) for the available memory or computation time.

**Full signature:**
```julia
coarsen_mesh_to_target_rwg(mesh, target_rwg;
    max_iters=10, allow_boundary=true, require_closed=false,
    area_tol_rel=1e-12, strict_nonmanifold=true)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Input mesh (typically already repaired). |
| `target_rwg` | `Int` | -- | Target number of RWG basis functions. The actual count will be close but may not match exactly, since coarsening is discrete. |
| `max_iters` | `Int` | `10` | Maximum binary-search iterations for finding the right voxel size. |

**Returns:** Named tuple `(mesh, rwg_count, target_rwg, best_gap, iterations)`.

**Tip:** Use `estimate_dense_matrix_gib(target_rwg)` to check memory requirements before choosing a target.

---

### `cluster_mesh_vertices(mesh, h)`

Low-level voxel clustering: partition space into cubic cells of side `h`, replace all vertices in each cell with their centroid, and remap triangles. Degenerate and duplicate triangles created by remapping are removed.

**Parameters:**
- `mesh::TriMesh`: Input mesh.
- `h::Float64`: Voxel cell size in meters (> 0). Smaller `h` = less coarsening.

**Returns:** `TriMesh` (coarsened mesh).

---

### `drop_nonmanifold_triangles(mesh; max_passes=8)`

Iteratively remove triangles attached to non-manifold edges (edges with more than two incident triangles). Each pass may expose new non-manifold edges, so multiple passes are needed.

**Parameters:**
- `mesh::TriMesh`: Input mesh.
- `max_passes::Int=8`: Maximum iteration count. Usually converges in 2--3 passes.

**Returns:** `TriMesh` with only manifold and boundary edges.

---

### `estimate_dense_matrix_gib(N)`

Estimate memory (GiB) for a dense `ComplexF64` N x N matrix (16 bytes per entry). Use this before assembly to verify the problem fits in RAM.

**Parameters:** `N::Integer` -- matrix dimension (= number of RWG basis functions).

**Returns:** `Float64` estimated GiB.

**Reference values:**

| N | Memory |
|---|--------|
| 500 | 0.004 GiB |
| 2000 | 0.06 GiB |
| 5000 | 0.37 GiB |
| 10000 | 1.5 GiB |
| 20000 | 6.0 GiB |

---

## Resolution Diagnostics

These functions assess whether a mesh is adequately resolved for MoM simulation at a given frequency. The core criterion is that the maximum edge length should be no larger than `lambda / points_per_wavelength` (typically lambda/10). Under-resolved meshes produce inaccurate MoM solutions without any obvious error message -- always check resolution before running a simulation.

### `mesh_resolution_report(mesh, freq_hz; points_per_wavelength=10.0, c0=299792458.0)`

Compute electrical mesh-resolution diagnostics for MoM at the specified frequency.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh to check. |
| `freq_hz` | `Real` | -- | Simulation frequency in Hz. |
| `points_per_wavelength` | `Real` | `10.0` | Target resolution: `target_max_edge = lambda / points_per_wavelength`. The standard MoM rule of thumb is 10 edges per wavelength. Use 15--20 for high-accuracy studies. |
| `c0` | `Real` | `299792458.0` | Speed of light (m/s). |

**Returns:** Named tuple with fields:

| Field | Type | Description |
|-------|------|-------------|
| `freq_hz` | `Float64` | Frequency used. |
| `wavelength_m` | `Float64` | Wavelength `lambda = c0 / freq_hz`. |
| `points_per_wavelength` | `Float64` | Resolution criterion used. |
| `target_max_edge_m` | `Float64` | Target maximum edge length `lambda / ppw`. |
| `n_vertices`, `n_triangles`, `n_edges` | `Int` | Mesh size counts. |
| `edge_min_m`, `edge_median_m`, `edge_mean_m`, `edge_max_m` | `Float64` | Edge length statistics. |
| `edge_p95_m` | `Float64` | 95th-percentile edge length. |
| `edge_median_over_lambda`, `edge_p95_over_lambda`, `edge_max_over_lambda` | `Float64` | Edge lengths normalized by wavelength. |
| `meets_target` | `Bool` | `true` if `edge_max_m <= target_max_edge_m`. |

**Example:**

```julia
freq = 1e9
report = mesh_resolution_report(mesh, freq)
println("λ = ", report.wavelength_m, " m")
println("Max edge: ", report.edge_max_m, " m (", report.edge_max_over_lambda, " λ)")
println("Meets λ/10 target: ", report.meets_target)
```

---

### `mesh_resolution_ok(report; criterion=:max)`

Evaluate a `mesh_resolution_report` against a selected criterion.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `report` | NamedTuple | -- | Output from `mesh_resolution_report`. |
| `criterion` | `Symbol` | `:max` | Which edge statistic to check: `:max` (strictest -- all edges must pass), `:p95` (allows 5% outliers), `:median` (most lenient). |

**Returns:** `Bool` -- `true` if the selected edge statistic is at most `target_max_edge_m`.

**Interpretation:**

| Criterion | Passes when | Use case |
|-----------|-------------|----------|
| `:max` | Every edge meets the target | Default, recommended for standard simulations |
| `:p95` | 95% of edges meet the target | Meshes with a few long edges at boundaries or corners |
| `:median` | The median edge is acceptable | Quick feasibility check |

---

## Mesh Refinement

These functions increase mesh density via uniform midpoint subdivision. Each refinement pass splits every triangle into 4 by inserting midpoint vertices on each edge, halving all edge lengths. Use these when an imported mesh is too coarse for the simulation frequency.

### `refine_mesh_to_target_edge(mesh, target_max_edge_m; max_iters=8, max_triangles=2_000_000)`

Uniformly refine a triangle mesh via midpoint subdivision until `edge_max <= target_max_edge_m` or limits are reached.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Input mesh. |
| `target_max_edge_m` | `Real` | -- | Target maximum edge length (meters). |
| `max_iters` | `Int` | `8` | Maximum refinement passes. Each pass multiplies the triangle count by 4. |
| `max_triangles` | `Int` | `2_000_000` | Safety limit on total triangles. Refinement stops before exceeding this. |

**Returns:** Named tuple:

| Field | Type | Description |
|-------|------|-------------|
| `mesh` | `TriMesh` | Refined mesh. |
| `iterations` | `Int` | Number of refinement passes performed. |
| `converged` | `Bool` | `true` if `edge_max <= target_max_edge_m`. |
| `target_max_edge_m` | `Float64` | The target used. |
| `edge_max_before_m`, `edge_max_after_m` | `Float64` | Max edge length before and after refinement. |
| `triangles_before`, `triangles_after` | `Int` | Triangle count before and after. |
| `history_edge_max_m`, `history_triangles` | `Vector` | Per-iteration history for diagnostics. |

**Scaling:** Each pass multiplies triangles by 4 and halves the maximum edge length. After `k` passes: `Nt_new = 4^k * Nt_original`, `h_max_new ~ h_max / 2^k`.

---

### `refine_mesh_for_mom(mesh, freq_hz; points_per_wavelength=10.0, max_iters=8, max_triangles=2_000_000, c0=299792458.0)`

Refine a mesh to satisfy a frequency-based MoM edge-length target: `target_max_edge = lambda / points_per_wavelength`. This is a convenience wrapper that computes the target from the frequency and calls `refine_mesh_to_target_edge`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Input mesh. |
| `freq_hz` | `Real` | -- | Simulation frequency (Hz). |
| `points_per_wavelength` | `Real` | `10.0` | Resolution target. |
| `max_iters` | `Int` | `8` | Maximum refinement passes. |
| `max_triangles` | `Int` | `2_000_000` | Triangle count limit. |
| `c0` | `Real` | `299792458.0` | Speed of light (m/s). |

**Returns:** Same as `refine_mesh_to_target_edge`, plus `report_before` and `report_after` (resolution reports at the given frequency before and after refinement).

**Example:**

```julia
mesh_coarse = read_obj_mesh("antenna.obj")
result = refine_mesh_for_mom(mesh_coarse, 3e9; points_per_wavelength=10)
println("Before: ", result.report_before.edge_max_over_lambda, " λ max edge")
println("After:  ", result.report_after.edge_max_over_lambda, " λ max edge")
println("Triangles: ", result.triangles_before, " → ", result.triangles_after)
mesh_fine = result.mesh
```

---

## Typical Safe Pipeline

A robust workflow for imported OBJ meshes:

```julia
# 1. Import
mesh_raw = read_obj_mesh("input.obj")

# 2. Repair (fixes degenerate/non-manifold/orientation issues)
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, strict_nonmanifold=true)
mesh_ok = rep.mesh
println("Removed $(rep.removed_nonmanifold) non-manifold triangles")
println("Flipped $(length(rep.flipped_triangles)) triangle orientations")

# 3. Coarsen to target size (optional, if mesh is too fine)
target_N = 400
mem_gib = estimate_dense_matrix_gib(target_N)
println("Target N=$target_N requires ~$(round(mem_gib, digits=3)) GiB")
coarse = coarsen_mesh_to_target_rwg(mesh_ok, target_N)
mesh_sim = coarse.mesh
println("Achieved N=$(coarse.rwg_count) (target was $target_N)")
```

---

## Multi-Format Mesh I/O

These functions (in `src/geometry/MeshIO.jl`) extend mesh I/O beyond OBJ to support STL, Gmsh MSH, and CAD conversion.

### `read_stl_mesh(path; merge_tol=0.0)`

Read a triangle mesh from an STL file. Both binary and ASCII STL are auto-detected.

STL stores three vertices per facet with no shared-vertex topology, so duplicate vertices must be merged. With the default `merge_tol=0.0`, vertices are merged when their coordinates are bitwise identical after Float32→Float64 conversion. Set `merge_tol` to a small positive value if your exporter introduces floating-point noise.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `AbstractString` | -- | Path to the STL file. |
| `merge_tol` | `Float64` | `0.0` | Vertex merge tolerance. `0.0` = exact merge (bitwise). Positive values use grid-based quantization. |

**Returns:** `TriMesh`

**Example:**
```julia
mesh = read_stl_mesh("model.stl")
report = assert_mesh_quality(mesh; allow_boundary=true)
```

---

### `write_stl_mesh(path, mesh; header="...", ascii=false)`

Write a `TriMesh` to an STL file. Default is binary (compact, fast). Set `ascii=true` for human-readable ASCII STL.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `AbstractString` | -- | Output file path. |
| `mesh` | `TriMesh` | -- | Triangle mesh to write. |
| `header` | `AbstractString` | `"Exported by DiffMoM"` | Header string (80 chars max for binary). |
| `ascii` | `Bool` | `false` | Write ASCII STL instead of binary. |

**Returns:** `path`

**Note:** Binary STL uses Float32 for coordinates, so vertex positions lose precision beyond ~7 significant digits. For archiving with full Float64 precision, use OBJ.

---

### `read_msh_mesh(path)`

Read a triangle surface mesh from a Gmsh MSH file (v2 or v4 ASCII).

Only 3-node triangle elements (Gmsh element type 2) are extracted. Lines, quads, tetrahedra, and other element types are silently ignored. Node IDs are remapped to 1-based contiguous indices.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `AbstractString` | Path to the MSH file. |

**Returns:** `TriMesh`

**Typical workflow (STEP → MSH → TriMesh):**
```bash
gmsh -2 model.step -o model.msh -clmax 0.01
```
```julia
mesh = read_msh_mesh("model.msh")
```

---

### `read_mesh(path)`

Unified mesh reader that dispatches by file extension:

| Extension | Reader |
|-----------|--------|
| `.obj` | `read_obj_mesh` |
| `.stl` | `read_stl_mesh` |
| `.msh` | `read_msh_mesh` |

Throws an error on unsupported extensions.

**Example:**
```julia
mesh = read_mesh("input.stl")     # auto-detects STL
mesh = read_mesh("model.msh")     # auto-detects Gmsh MSH
```

---

### `write_mesh(path, mesh; kwargs...)`

Unified mesh writer that dispatches by file extension:

| Extension | Writer |
|-----------|--------|
| `.obj` | `write_obj_mesh` |
| `.stl` | `write_stl_mesh` |

Keyword arguments are forwarded to the underlying writer.

---

### `convert_cad_to_mesh(cad_path, output_path; mesh_size=0.0, gmsh_exe="gmsh")`

Convert a CAD file (STEP, IGES, BREP) to a triangle surface mesh by calling the Gmsh CLI. Gmsh must be installed and on PATH.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cad_path` | `AbstractString` | -- | Input CAD file (.step, .stp, .iges, .igs, .brep). |
| `output_path` | `AbstractString` | -- | Output mesh file (.msh, .stl, or .obj). |
| `mesh_size` | `Float64` | `0.0` | Maximum element size (`-clmax`). `0.0` = Gmsh default. |
| `gmsh_exe` | `AbstractString` | `"gmsh"` | Path to the Gmsh executable. |

**Returns:** `TriMesh` (the imported mesh).

**Example:**
```julia
mesh = convert_cad_to_mesh("antenna.step", "antenna.msh"; mesh_size=0.005)
rep = repair_mesh_for_simulation(mesh)
```

**Install Gmsh:** Download from [gmsh.info](https://gmsh.info). On macOS: `brew install gmsh`. On Ubuntu: `apt install gmsh`.

---

## Notes

- Use `require_closed=true` for closed PEC bodies (spheres, aircraft) where boundary edges indicate a mesh defect.
- Keep `allow_boundary=true` for open surfaces (plates, reflectors, strips).
- The repair pipeline is idempotent: running it twice produces the same result.
- STL binary uses Float32 for vertex coordinates; use OBJ for full Float64 precision.
- For STEP/IGES import, install Gmsh and use `convert_cad_to_mesh` or manually run `gmsh -2 model.step -o model.msh`.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/geometry/Mesh.jl` | Mesh creation, geometry, quality, repair, coarsening, refinement |
| `src/geometry/MeshIO.jl` | Multi-format I/O: STL, Gmsh MSH, unified dispatcher, CAD conversion |
| `examples/06_aircraft_rcs.jl` | End-to-end OBJ import -> repair -> coarsen -> RCS workflow |
| `examples/12_plate_rcs_stl_roundtrip.jl` | Mesh import/export and visualization checks |
| `examples/convert_aircraft_mat_to_obj.py` | MAT-to-OBJ conversion helper (SciPy) |

---

## Exercises

- **Basic:** Run `mesh_quality_report` on a mesh before and after `repair_mesh_for_simulation`. Compare the report fields.
- **Practical:** Import an OBJ mesh, coarsen it to several target RWG counts (200, 400, 800), and plot `estimate_dense_matrix_gib` vs actual RWG count.
- **Challenge:** Write a script that sweeps voxel cell size `h` in `cluster_mesh_vertices` and plots the resulting RWG count vs `h`. At what `h` does the mesh become too coarse (quality checks fail)?
- **Format I/O:** Export a mesh to both OBJ and STL, read each back, and compare vertex coordinates. Verify that STL introduces Float32 rounding while OBJ preserves full precision.
