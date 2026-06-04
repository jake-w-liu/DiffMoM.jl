# Chapter 1: Mesh Pipeline

## Purpose

Establish a robust geometry preprocessing workflow that transforms raw CAD meshes into simulation-ready triangulated surfaces suitable for RWG-based MoM. The mesh pipeline ensures topological correctness, manages computational complexity through coarsening, and provides quality guarantees required for stable EFIE solutions. This chapter covers mesh representation, quality diagnostics, repair algorithms, coarsening strategies, and integration with the RWG basis function generation.

---

## Learning Goals

After this chapter, you should be able to:

1. Understand the `TriMesh` data structure and its role in the MoM pipeline.
2. Diagnose mesh quality issues using comprehensive validation reports.
3. Apply repair algorithms to fix common CAD mesh problems.
4. Implement coarsening strategies to control problem size while preserving geometric fidelity.
5. Estimate memory requirements for dense MoM matrices.
6. Design reproducible mesh preprocessing workflows for complex platforms.
7. Troubleshoot mesh-related solver failures and performance bottlenecks.

---

## 1) Mathematical and Computational Foundations

### 1.1 Surface Representation for MoM

The Method of Moments with RWG basis functions requires a **piecewise-planar triangulated surface** $\Gamma$ represented as a union of flat triangles:

```math
\Gamma = \bigcup_{n=1}^{N_t} T_n
```

where each triangle $T_n$ is defined by three vertices $\mathbf{v}_{n1}, \mathbf{v}_{n2}, \mathbf{v}_{n3} \in \mathbb{R}^3$. The RWG basis functions are defined on **pairs of triangles sharing an edge**, imposing specific topological requirements:

- **Manifold edges**: Each interior edge must be shared by exactly two triangles
- **Consistent orientation**: Triangle normals must follow a consistent ordering (typically counterclockwise when viewed from outside)
- **Non-degenerate geometry**: Triangle areas must be non-zero, edges non-zero length

### 1.2 Topological Requirements for RWG Bases

The RWG basis function $\mathbf{f}_m(\mathbf{r})$ is defined on a pair of triangles $T_m^+$ and $T_m^-$ sharing edge $m$. This requires:

1. **Edge-based connectivity**: Each interior edge maps to exactly two triangles
2. **Orientation consistency**: The edge direction must be opposite in the two triangles for proper current continuity
3. **Surface closure**: For closed surfaces, every edge must be interior (shared by two triangles); for open surfaces, boundary edges are allowed but require special treatment

### 1.3 Computational Complexity Considerations

The dense MoM matrix has size $N \times N$ where $N$ is the number of RWG basis functions (approximately equal to the number of interior edges). Memory usage grows as $O(N^2)$:

```math
\text{Memory (GB)} \approx \frac{16 \times N^2}{1024^3} \quad \text{(complex double precision)}
```

For $N = 10,000$, matrix storage requires ~1.5 GB; for $N = 50,000$, ~37 GB. This motivates **mesh coarsening** to control problem size while preserving essential geometric features.

---

## 2) Implementation in `DiffMoM.jl`

### 2.1 Core Data Structure: `TriMesh`

The `TriMesh` type stores surface geometry in minimal format:

```julia
struct TriMesh
    xyz::Matrix{Float64}  # (3, Nv) vertex coordinates
    tri::Matrix{Int}      # (3, Nt) triangle vertex indices (1-based)
end
```

Key properties:
- **Vertex coordinates**: Stored as columns for memory efficiency
- **Triangle indexing**: 1-based indices into vertex columns
- **Implicit topology**: Edge connectivity derived from triangle adjacency

Helper functions:
- `nvertices(mesh)`: Returns $N_v$ (number of vertices)
- `ntriangles(mesh)`: Returns $N_t$ (number of triangles)
- `triangle_area(mesh, t)`: Area of triangle $t$
- `triangle_center(mesh, t)`: Centroid of triangle $t$
- `triangle_normal(mesh, t)`: Unit normal vector (right-hand rule from vertex order)

### 2.2 Mesh Generation and Import

#### Analytical Plate Generation
For benchmark problems, generate rectangular plates:

```julia
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
```

Creates a plate in the $xy$-plane centered at origin with $(N_x+1)\times(N_y+1)$ vertices and $2N_xN_y$ triangles. The triangulation follows a regular grid with alternating diagonal orientation.

#### OBJ File Import
For complex CAD geometry:

```julia
mesh = read_obj_mesh("aircraft.obj")
```

The importer handles:
- **Polygon faces**: Automatically triangulates $n$-gons via fan triangulation
- **Vertex normals/textures**: Ignored (only geometry retained)
- **Relative/absolute paths**: Standard OBJ syntax
- **Multiple objects**: Combined into single mesh

### 2.3 Quality Diagnostics System

The `mesh_quality_report` function performs comprehensive validation:

```julia
report = mesh_quality_report(mesh; 
    area_tol_rel=1e-12, 
    check_orientation=true)
```

#### Validation Categories:

1. **Invalid triangles**: Index out of bounds or repeated vertices
2. **Degenerate triangles**: Area below tolerance $A_{\min} = \epsilon_{\text{rel}} \times L_{\text{bbox}}^2$
3. **Boundary edges**: Edges with only one incident triangle (acceptable for open surfaces)
4. **Non-manifold edges**: Edges with ≥3 incident triangles (topologically invalid)
5. **Orientation conflicts**: Interior edges where adjacent triangles disagree on edge direction

#### Interpretation and Acceptance:

```julia
ok = mesh_quality_ok(report;
    allow_boundary=true,
    require_closed=false)
```

Typical acceptance criteria for RWG generation:
- **Closed surfaces**: `require_closed=true`
- **Open surfaces**: `allow_boundary=true`

### 2.4 Mesh Repair Pipeline

For CAD meshes with topological defects:

```julia
rep = repair_mesh_for_simulation(mesh;
    allow_boundary=true,
    strict_nonmanifold=true,
    fix_orientation=true,
    area_tol_rel=1e-12)
```

#### Repair Algorithms:

1. **Degenerate triangle removal**: Triangles with area below threshold
2. **Duplicate face elimination**: Identical triangles (within tolerance)
3. **Orientation fixing**: Flood-fill algorithm to establish consistent normals
4. **Non-manifold resolution**: 
   - If `strict_nonmanifold=true`: Remove triangles causing non-manifold edges
   - If `strict_nonmanifold=false`: Allow non-manifold edges (not recommended for RWG)

#### OBJ-Level Wrapper:

```julia
repair_obj_mesh("input.obj", "output.obj")
```

Produces a cleaned OBJ file with metadata about removed elements.

### 2.5 Coarsening for Computational Feasibility

To control problem size while preserving shape:

```julia
coarse = coarsen_mesh_to_target_rwg(mesh, target_rwg;
    max_iters=10)
```

#### Coarsening Strategy:

1. **Vertex clustering**: Group vertices within distance $\delta = \alpha \times L_{\text{bbox}}$
2. **Edge collapse**: Remove short edges while maintaining manifold property
3. **Quality preservation**: Reject operations that create degenerate triangles
4. **Iterative refinement**: Repeat until target RWG count reached or convergence

#### RWG Count Estimation:

The number of RWG basis functions $N_{\text{RWG}}$ is approximately:

```math
N_{\text{RWG}} \approx \frac{3}{2} N_t - N_v + \chi
```

where $\chi$ is the Euler characteristic ($\chi = 2$ for closed sphere-like surfaces). The coarsening algorithm uses this approximation to track progress toward the target.

### 2.6 Memory Estimation

Before committing to solve, estimate memory requirements:

```julia
gib = estimate_dense_matrix_gib(N)
println("Estimated matrix memory: $gib GB")
```

Implementation:
- Complex double precision: 16 bytes per entry
- Dense storage: $16 \times N^2$ bytes
- Conversion to GiB: divide by $1024^3$

#### Practical Guidelines:
- **Desktop (< 32 GB RAM)**: $N \lesssim 40,000$
- **Workstation (128 GB RAM)**: $N \lesssim 80,000$  
- **Cluster/HPC**: $N \gtrsim 100,000$ with out-of-core or iterative solvers

---

## 3) Practical Workflow Examples

### 3.1 Complete CAD-to-Simulation Pipeline

```julia
using DiffMoM

# 1. Import CAD mesh
mesh_raw = read_obj_mesh("aircraft.obj")
println("Raw mesh: $(nvertices(mesh_raw)) vertices, $(ntriangles(mesh_raw)) triangles")

# 2. Quality assessment
report_raw = mesh_quality_report(mesh_raw)
println("Non-manifold edges: $(report_raw.n_nonmanifold_edges)")
println("Orientation conflicts: $(report_raw.n_orientation_conflicts)")

# 3. Repair topological defects
rep = repair_mesh_for_simulation(mesh_raw;
    allow_boundary=true,
    strict_nonmanifold=true,
    fix_orientation=true)
mesh_repaired = rep.mesh
println("Repaired: removed $(length(rep.removed_invalid) + length(rep.removed_degenerate)) triangles")

# 4. Coarsen to target complexity
target_rwg = 5000
coarse = coarsen_mesh_to_target_rwg(mesh_repaired, target_rwg)
mesh_final = coarse.mesh
println("Coarsened: $(ntriangles(mesh_final)) triangles → ~$(coarse.rwg_count) RWG")

# 5. Final validation
assert_mesh_quality(mesh_final; 
    allow_boundary=true, 
    require_closed=false)

# 6. RWG generation
rwg = build_rwg(mesh_final; 
    precheck=true, 
    allow_boundary=true, 
    require_closed=false)
println("Actual RWG count: $(rwg.nedges)")

# 7. Memory estimation
gib = estimate_dense_matrix_gib(rwg.nedges)
println("Matrix memory estimate: $(round(gib, digits=2)) GB")
```

### 3.2 Batch Processing for Multiple Geometries

```julia
function process_geometry(input_path, output_path, target_rwg)
    # Load and repair
    mesh = read_obj_mesh(input_path)
    rep = repair_mesh_for_simulation(mesh)
    
    # Coarsen if needed
    rwg_est = build_rwg(rep.mesh; precheck=true, allow_boundary=true, require_closed=false).nedges
    if rwg_est > target_rwg
        coarse = coarsen_mesh_to_target_rwg(rep.mesh, target_rwg)
        mesh = coarse.mesh
    end
    
    # Save processed mesh
    write_obj_mesh(output_path, mesh)
    
    # Generate quality report
    report = mesh_quality_report(mesh)
    return (mesh=mesh, report=report)
end
```

### 3.3 Quality Inspection Workflow

```julia
using DiffMoM

mesh = read_obj_mesh("platform.obj")
report = mesh_quality_report(mesh)
println(report)

# Optional visual check
plot_mesh_wireframe(mesh; title="Raw mesh")

# Repair and re-check
rep = repair_mesh_for_simulation(mesh)
println(rep.after)
plot_mesh_wireframe(rep.mesh; title="Repaired mesh")
```

---

## 4) Troubleshooting Common Issues

### 4.1 Diagnostic Decision Tree

1. **`build_rwg` fails with "non-manifold edge"**
   - Run `mesh_quality_report` to identify problematic edges
   - Use `repair_mesh_for_simulation` with `strict_nonmanifold=true`
   - Manually inspect edge connectivity with `mesh_unique_edges`

2. **Excessive memory usage**
   - Estimate RWG count before solve: `build_rwg(mesh; precheck=true, allow_boundary=true, require_closed=false).nedges`
   - Coarsen mesh: `coarsen_mesh_to_target_rwg`
   - Consider iterative solver or domain decomposition

3. **Poor solution accuracy after coarsening**
   - Compare key geometric features before/after coarsening
   - Increase `target_rwg` (retain more unknowns)
   - Use multi-resolution approach: solve coarse, refine regionally

4. **OBJ import fails or produces degenerate geometry**
   - Check OBJ format compliance (triangles vs. polygons)
   - Scale geometry: CAD often uses mm, MoM expects meters
   - Use external repair tools (MeshLab, Blender) for severe defects

### 4.2 Performance Optimization

- **Batch processing**: Repair/coarsen multiple meshes offline
- **Parallel coarsening**: Independent regions can be processed concurrently
- **Incremental refinement**: Start coarse, refine based on solution sensitivity
- **Caching**: Save processed meshes to avoid recomputation

### 4.3 Validation Metrics

Establish quality metrics for processed meshes:
- **Aspect ratio**: Triangle quality $\text{area} / \text{perimeter}^2$
- **Edge length uniformity**: Coefficient of variation
- **Feature preservation**: Hausdorff distance from original
- **RWG suitability**: Percentage of edges with exactly two incident triangles

---

## 5) Advanced Topics

### 5.1 Adaptive Mesh Refinement

Combine coarsening with refinement based on solution features:

```julia
# Initial coarse solve
coarse = coarsen_mesh_to_target_rwg(mesh, 2000)
mesh_coarse = coarse.mesh
rwg_coarse = build_rwg(mesh_coarse)
Z = assemble_Z_efie(mesh_coarse, rwg_coarse, k; quad_order=3, eta0=η0)
v = assemble_v_plane_wave(mesh_coarse, rwg_coarse, k_vec, E0, pol_inc; quad_order=3)
I_coarse = solve_forward(Z, v)

# Region-selective refinement is not implemented in the current package.
# Typical practice: repeat solve with higher target_rwg and compare observables.
```

### 5.2 Multi-Resolution Workflows

1. **Coarse exploration**: Rapid parameter sweeps on coarse mesh
2. **Medium fidelity**: Design optimization with balanced accuracy/speed
3. **Fine validation**: Final verification on detailed mesh

### 5.3 Integration with External Tools

- **MeshLab**: Advanced repair and processing scripts
- **Blender Python API**: Automated geometry processing
- **CAD kernels**: Direct import of STEP/IGES files (future extension)

---

## 6) Code Mapping

### 6.1 Primary Implementation Files

- **Mesh data structures and algorithms**: `src/geometry/Mesh.jl`
  - `make_rect_plate`, `read_obj_mesh`, `write_obj_mesh`
  - `mesh_quality_report`, `mesh_quality_ok`, `assert_mesh_quality`
  - `repair_mesh_for_simulation`, `repair_obj_mesh`
  - `coarsen_mesh_to_target_rwg`, `estimate_dense_matrix_gib`

- **RWG basis generation**: `src/basis/RWG.jl`
  - `build_rwg`
  - Pre-check integration with mesh quality system

- **Visualization utilities**: `src/postprocessing/Visualization.jl`
  - Mesh inspection and quality visualization

### 6.2 Example Scripts

- **Mesh repair + coarsening demonstration**: `examples/06_aircraft_rcs.jl`
- **STL roundtrip and mesh handling**: `examples/12_plate_rcs_stl_roundtrip.jl`
- **Sphere benchmark mesh flow**: `examples/04_pec_sphere_mie.jl`

### 6.3 Supporting Functions

- **Geometry utilities**: `src/geometry/Mesh.jl` (`triangle_area`, `triangle_center`, `triangle_normal`)
- **Performance estimation**: `src/geometry/Mesh.jl` (`estimate_dense_matrix_gib`)

---

## 7) Exercises

### 7.1 Basic Level

1. **Mesh quality assessment**:
   - Load an OBJ file from the `data/` directory
   - Generate a comprehensive quality report
   - Identify and categorize all topological issues
   - Document the repair steps needed

2. **Coarsening experiment**:
   - Take a rectangular plate with $N_x = N_y = 20$
   - Coarsen to target RWG counts of 1000, 2000, 3000
   - Plot triangle count vs. RWG count and memory estimate
   - Visually compare geometry preservation

### 7.2 Intermediate Level

3. **Repair algorithm analysis**:
   - Intentionally corrupt a mesh (duplicate faces, flipped normals)
   - Apply repair pipeline with different parameter settings
   - Quantify repair effectiveness (triangles removed, orientation fixed)
   - Propose improvements to the repair algorithm

4. **Memory scalability study**:
   - Generate plate meshes with $N_x = N_y = 5, 10, 20, 40$
   - Estimate RWG count and memory requirements
   - Compare estimates with actual `build_rwg` results
   - Derive empirical scaling laws

### 7.3 Advanced Level

5. **Adaptive coarsening design**:
   - Implement curvature-based coarsening that preserves high-curvature regions
   - Compare with uniform coarsening on a complex geometry
   - Evaluate impact on RCS accuracy for different coarsening strategies

6. **Pipeline automation**:
   - Create a batch processing system for multiple CAD files
   - Generate comprehensive reports (quality metrics, repair statistics)
   - Implement automatic parameter tuning based on mesh characteristics

---

## 8) Chapter Checklist

Before proceeding to forward simulation, ensure you can:

- [ ] Import OBJ files and generate analytical meshes
- [ ] Interpret mesh quality reports and identify critical issues
- [ ] Apply repair algorithms to fix common topological defects
- [ ] Coarsen meshes to meet computational constraints
- [ ] Estimate memory requirements for MoM matrices
- [ ] Validate mesh suitability for RWG basis generation
- [ ] Design reproducible mesh processing workflows

---

## 9) Further Reading

- **Computational geometry**: O'Rourke, *Computational Geometry in C* (1998)
- **Mesh generation**: Shewchuk, *Delaunay Refinement Algorithms* (2002)
- **RWG basis requirements**: Rao et al., *Electromagnetic Scattering by Surfaces of Arbitrary Shape* (1982)
- **Mesh repair algorithms**: Attene et al., *Mesh Repair* (2013)
- **Multi-resolution modeling**: Katz & Tal, *Mesh Simplification* (2003)
- **CAD interoperability**: Pratt & Anderson, *A Shape Representation for CAD/CAM* (2005)
