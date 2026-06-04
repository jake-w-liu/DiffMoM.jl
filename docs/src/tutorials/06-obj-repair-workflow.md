# Tutorial: OBJ Repair Workflow

## Purpose

CAD‑exported meshes often contain defects that violate the topological assumptions of RWG basis functions. This tutorial teaches a defensive mesh‑preparation pipeline that diagnoses and repairs common OBJ flaws, ensuring robust MoM simulation.

You will learn to:

- **Identify mesh defects** (non‑manifold edges, degenerate triangles, orientation conflicts) using `mesh_quality_report`.
- **Apply automated repair** with `repair_obj_mesh` or `repair_mesh_for_simulation`, understanding the trade‑offs of each flag.
- **Visualize repaired and coarsened meshes** side‑by‑side to verify geometric fidelity.
- **Enforce a quality checklist** with `assert_mesh_quality` before simulation.
- **Decide when to coarsen** based on dense‑matrix memory estimates.

**Why repair matters:** A single non‑manifold edge can cause RWG construction to fail; degenerate triangles lead to singular integrals; inconsistent normals break the EFIE sign conventions. Automated repair transforms a “dirty” CAD export into a simulation‑ready mesh in seconds.

---

## Learning Goals

After this tutorial, you should be able to:

1. **Diagnose mesh quality issues** using `mesh_quality_report` and interpret its output.
2. **Repair OBJ files** via the demo script (`examples/06_aircraft_rcs.jl`) and programmatic API (`repair_mesh_for_simulation`).
3. **Choose appropriate repair flags** for open surfaces, closed scatterers, and partially defective meshes.
4. **Estimate memory footprint** with `estimate_dense_matrix_gib` and decide whether coarsening is needed.
5. **Generate wireframe previews** with `save_mesh_preview` to visually compare original, repaired, and coarsened versions.
6. **Enforce pre‑simulation quality gates** using `assert_mesh_quality`.

---

## Common Mesh Defects and Their Impact

| Defect | Description | MoM Impact |
|--------|-------------|------------|
| **Non‑manifold edge** | Edge shared by >2 triangles (common in CAD “seams”) | RWG construction fails; basis functions undefined |
| **Degenerate triangle** | Zero‑area triangle (collapsed vertices) | Singular mass matrix; integration weights NaN |
| **Invalid triangle** | Vertex index out of bounds (corrupt OBJ) | Array bounds error during assembly |
| **Orientation conflict** | Adjacent triangles have opposing normals (inside‑out flip) | EFIE sign errors; scattered field phase reversed |
| **Boundary edge** | Edge belonging to only one triangle (open surface) | OK for open surfaces; must set `allow_boundary=true` |
| **Non‑watertight mesh** | Holes or missing triangles (unintended gaps) | May still simulate but physics unrealistic |

The repair pipeline detects each defect and applies a configurable action (remove, flip, or tolerate).

---

## Step‑by‑Step Workflow

### 1) Diagnose Mesh Quality

Before repairing, inspect the raw OBJ file:

```julia
using DiffMoM

mesh = read_obj_mesh("Airplane.obj")
report = mesh_quality_report(mesh)
```

Output includes counts of each defect type and a pass/fail verdict. For example:

```
Mesh quality report:
  vertices: 12345
  triangles: 24684
  boundary edges: 124 (open surface)
  non‑manifold edges: 12 **FAIL**
  degenerate triangles: 0
  invalid triangles: 0
  orientation conflicts: 48 **FAIL**
  watertight: false (boundary edges present)
  genus (if closed): N/A
Verdict: FAIL – needs repair
```

**Interpretation:** This mesh has non‑manifold edges (must fix) and orientation conflicts (should fix). Boundary edges are acceptable for an airplane (open surface).

### 2) Repair via Command‑Line Script

The simplest way to exercise the repair workflow is the bundled demo:

```bash
julia --project=. examples/06_aircraft_rcs.jl
```

Inside the script, repair is performed with conservative defaults:

```julia
result = repair_mesh_for_simulation(mesh;
    allow_boundary=true,
    auto_drop_nonmanifold=true,
)
mesh_repaired = result.mesh
```

For direct programmatic use, prefer `repair_mesh_for_simulation` (next section).
When you run the demo script, it prints a repair summary:

```
Repairing mesh...
  Removed invalid: ...
  Removed degenerate: ...
  Removed non-manifold: ...
  Flipped: ...
  Repaired mesh: ... vertices, ... triangles
```

Non‑manifold edges are removed (along with adjacent triangles), orientation conflicts are fixed by flipping normals, and boundary edges are preserved.

### 3) Programmatic Repair with Fine‑Grained Control

For integration into custom pipelines, use `repair_mesh_for_simulation`:

```julia
result = repair_mesh_for_simulation(
    mesh;
    allow_boundary=true,      # permit open surfaces
    require_closed=false,     # do not demand watertightness
    drop_invalid=true,        # discard triangles with out‑of‑bounds indices
    drop_degenerate=true,     # discard zero‑area triangles
    fix_orientation=true,     # flip normals to achieve consistent outward orientation
    strict_nonmanifold=true,  # remove triangles attached to non‑manifold edges
)

mesh_repaired = result.mesh
println("Flipped $(length(result.flipped_triangles)) triangles")
println("Removed $(length(result.removed_degenerate)) degenerate triangles")
println("Removed $(length(result.removed_invalid)) invalid triangles")
```

The function returns a named tuple containing before/after statistics and lists of modified triangles.

### 4) Estimate Memory Footprint

After repair, build RWG basis and estimate dense‑matrix memory:

```julia
rwg = build_rwg(mesh_repaired; precheck=true, allow_boundary=true)
gib = estimate_dense_matrix_gib(rwg.nedges)
println("RWG unknowns: $(rwg.nedges)")
println("Dense matrix memory: $gib GiB")
```

If `gib` exceeds available RAM (e.g., > 32 GiB on a typical workstation), coarsening is necessary.

### 5) Coarsen to Target RWG Count

Coarsening reduces triangle count while preserving sharp features:

```julia
target_rwg = 500
coarse_result = coarsen_mesh_to_target_rwg(
    mesh_repaired,
    target_rwg;
    max_iters=10,
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
    strict_nonmanifold=true,
)

mesh_coarse = coarse_result.mesh
rwg_coarse = build_rwg(mesh_coarse; precheck=true, allow_boundary=true)
println("Coarsened unknowns: $(rwg_coarse.nedges)")
println("Coarsening iterations: $(coarse_result.iterations)")
println("Target gap: $(coarse_result.best_gap)")  # difference from target
```

**Key parameters:**

- `area_tol_rel`: relative area change allowed per edge‑collapse (smaller preserves volume).
- `strict_nonmanifold`: prevent creation of new non‑manifold edges during collapse.

### 6) Visualize Repaired vs Coarsened Mesh

Generate a side‑by‑side wireframe preview programmatically:

```julia
seg_rep = mesh_wireframe_segments(mesh_repaired)
seg_coa = mesh_wireframe_segments(mesh_coarse)

preview = save_mesh_preview(
    mesh_repaired,
    mesh_coarse,
    "mesh_preview";
    title_a = "Repaired mesh\nV=$(nvertices(mesh_repaired)), T=$(ntriangles(mesh_repaired))",
    title_b = "Coarsened simulation mesh\nV=$(nvertices(mesh_coarse)), T=$(ntriangles(mesh_coarse))",
    color_a = :steelblue,
    color_b = :darkorange,
    camera = (30, 30),
    size = (1200, 520),
)
```

The output includes PNG and PDF files with matched axis limits for easy comparison.

### 7) Enforce Quality Checklist

Before simulation, assert that the mesh passes all required checks:

```julia
assert_mesh_quality(mesh_coarse;
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
)
```

If any condition fails, `assert_mesh_quality` throws an informative error. This gate ensures that later stages (RWG build, EFIE assembly) will not encounter unexpected mesh defects.

---

## Repair Flag Decision Guide

| Scenario | `allow_boundary` | `require_closed` | `strict_nonmanifold` | `fix_orientation` |
|----------|------------------|------------------|-----------------------|-------------------|
| **Closed PEC scatterer** (sphere, cube) | `false` | `true` | `true` | `true` |
| **Open surface** (plate, antenna) | `true` | `false` | `true` | `true` |
| **Complex CAD with seams** (airplane, vehicle) | `true` | `false` | `false` | `true` |
| **Quick visualization only** | `true` | `false` | `false` | `false` |

**When to relax `strict_nonmanifold`:** Some CAD exports contain intentional non‑manifold edges (e.g., where wings meet fuselage). Setting `strict_nonmanifold=false` keeps those edges but may cause RWG construction to fail. In that case, consider manually cleaning the mesh in a CAD tool before import.

**When to relax `fix_orientation`:** If the mesh already has consistent outward normals (e.g., from a carefully prepared OBJ), you can skip orientation fixing to avoid unnecessary flips.

---

## Troubleshooting

### Repair fails: “Mesh still has non‑manifold edges after repair”

**Cause:** The mesh contains complex non‑manifold junctions that cannot be resolved automatically (e.g., three surfaces meeting along a line).

**Solutions:**

- Set `strict_nonmanifold=false` and accept that RWG construction may fail.
- Manually edit the mesh in Blender/MeshLab to separate touching surfaces.
- Use `drop_nonmanifold_triangles(mesh)` to aggressively remove problematic regions.

### Orientation flipping creates inward normals

**Cause:** The repair algorithm uses a heuristic (majority‑vote) to decide “outward” direction; if the mesh is highly irregular, it may choose wrong.

**Solutions:**

- Inspect the mesh visually with `plot_mesh_wireframe` and check normals.
- Manually flip normals in external software and re‑export.
- Disable `fix_orientation` and rely on external correction.

### Coarsening creates degenerate triangles

**Cause:** Edge‑collapse can produce zero‑area sliver triangles when vertices are nearly coincident.

**Solutions:**

- Reduce `area_tol_rel` (e.g., to `1e‑4`) to limit collapse severity.
- Post‑coarsening, run `repair_mesh_for_simulation` with `drop_degenerate=true`.
- Increase `target_rwg` to allow less aggressive coarsening.

### Memory estimate unrealistic for given RWG count

`estimate_dense_matrix_gib` assumes a dense complex double‑precision matrix (16 bytes per entry). If you plan to use a sparse solver or single precision, the actual memory will be lower.

**Adjustment:** Multiply the estimate by your storage factor (e.g., 0.5 for single precision, 0.1 for sparse‑matrix format).

### Visualization backend fails to initialize

**Cause:** `Plots.jl` backend (typically GR) is not available or cannot initialize in the current environment.

**Solutions:**

- Ensure `Plots.jl` and its backend dependencies are installed (`Pkg.instantiate()` in project root).
- On headless servers, prefer non-interactive runs and write files via `save_mesh_preview(...)`.
- If plotting is unavailable, skip preview generation and inspect OBJ files directly in MeshLab/Blender.

---

## Code Mapping

| Task | Function | Source File | Key Lines |
|------|----------|-------------|-----------|
| **OBJ reading** | `read_obj_mesh` | `src/geometry/Mesh.jl` | 100–120 |
| **Quality report** | `mesh_quality_report` | `src/geometry/Mesh.jl` | 250–280 |
| **Mesh repair** | `repair_mesh_for_simulation` | `src/geometry/Mesh.jl` | 300–350 |
| **OBJ repair script** | `repair_obj_mesh` | `src/geometry/Mesh.jl` | 360–400 |
| **Coarsening** | `coarsen_mesh_to_target_rwg` | `src/geometry/Mesh.jl` | 400–450 |
| **Memory estimate** | `estimate_dense_matrix_gib` | `src/geometry/Mesh.jl` | 200–220 |
| **Wireframe segments** | `mesh_wireframe_segments` | `src/geometry/Mesh.jl` | 500–520 |
| **Mesh preview** | `save_mesh_preview` | `src/postprocessing/Visualization.jl` | 150–200 |
| **Quality assertion** | `assert_mesh_quality` | `src/geometry/Mesh.jl` | 280–300 |

**Scripts:**

- `examples/06_aircraft_rcs.jl` – end-to-end mesh repair/coarsen/solve demo.
- Use `save_mesh_preview` from `src/postprocessing/Visualization.jl` for side-by-side preview generation.

---

## Exercises

### Basic (45 minutes)

1. **Diagnose a defective mesh**: Download an OBJ from an online repository (e.g., Thingiverse). Run `mesh_quality_report` and list all defects.
2. **Repair with default flags**: Use `repair_mesh_for_simulation` to produce a repaired mesh. Verify that the output passes `assert_mesh_quality`.
3. **Estimate memory**: Build RWG for the repaired mesh and compute dense‑matrix memory. Would it fit on your machine?

### Practical (90 minutes)

1. **Compare repair flags**: Repair the same mesh with `strict_nonmanifold=true` vs `false`. How many triangles are removed in each case? Visualize the differences.
2. **Coarsening trade‑off**: Starting from a repaired mesh, coarsen to target RWG counts of 200, 500, and 1000. Plot number of triangles vs coarsening iteration count. Does the algorithm converge faster for larger targets?
3. **Orientation consistency**: Create a simple cube OBJ with deliberately flipped normals on two adjacent faces. Repair with `fix_orientation=true`. Verify that all normals now point outward.

### Advanced (2 hours)

1. **Custom repair pipeline**: Write a function that repairs a mesh, then coarsens until memory estimate ≤ 2 GiB, then runs `assert_mesh_quality`. Apply it to three different CAD models.
2. **Non‑manifold preservation**: For a mesh with intentional non‑manifold edges (e.g., a wing‑fuselage junction), modify the repair algorithm to keep those edges while still making the mesh RWG‑compatible. Hint: duplicate vertices along the junction.
3. **Volume preservation metric**: Implement a metric that compares the volume enclosed by the original mesh vs the coarsened mesh (for closed surfaces). Plot volume error vs coarsening ratio.

---

## Tutorial Checklist

Before feeding any imported mesh into MoM simulation, ensure you have:

- [ ] **Run `mesh_quality_report`** and understood the defect types present.
- [ ] **Repaired the mesh** with appropriate flags for your physics (open/closed, tolerate non‑manifold).
- [ ] **Estimated memory footprint** and coarsened if necessary.
- [ ] **Generated a visual preview** to confirm geometric fidelity after repair/coarsening.
- [ ] **Asserted quality** with `assert_mesh_quality` and received a “PASS” verdict.
- [ ] **Saved the simulation‑ready OBJ** for reproducibility.

---

## Further Reading

- **Paper Appendix A** – Mesh preparation pipeline and defect classification.
- **`src/geometry/Mesh.jl`** – Full implementation of repair, coarsening, and quality checks.
- **`src/postprocessing/Visualization.jl`** – Wireframe plotting and mesh preview utilities.
- **Tutorial 5: Airplane RCS** – Application of the repair workflow to a realistic scattering problem.
- **Advanced Workflows, Chapter 1: Complex OBJ Platforms** – Strategies for handling pathological CAD exports.
