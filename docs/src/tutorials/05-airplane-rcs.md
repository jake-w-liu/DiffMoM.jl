# Tutorial: Airplane RCS

## Purpose

Real‑world electromagnetic simulations involve complex CAD geometries that require robust preprocessing before MoM analysis. This tutorial demonstrates a complete workflow for scattering analysis of an airplane model:

- **Import an OBJ mesh** (typical CAD output) and scale it to physical dimensions.
- **Repair mesh defects** (non‑manifold edges, degenerate triangles, flipped normals) that would break RWG basis functions.
- **Coarsen the mesh** to stay within dense‑matrix memory limits while preserving geometric features.
- **Solve PEC scattering** for a given incidence direction and compute bistatic RCS cuts.
- **Visualize results** with heuristic plots that highlight monostatic and bistatic signatures.

**Key insight:** The bottleneck for large platforms is not the solver time but the $O(N^2)$ memory of the dense MoM matrix. Strategic coarsening allows you to explore scattering trends with manageable unknown counts (e.g., $N \lesssim 500$), providing qualitative insight before investing in fast‑method accelerators.

---

## Learning Goals

After this tutorial, you should be able to:

1. **Prepare imported meshes** using `read_obj_mesh`, `repair_mesh_for_simulation`, and `coarsen_mesh_to_target_rwg`.
2. **Estimate memory requirements** with `estimate_dense_matrix_gib(N)` and choose a feasible target RWG count.
3. **Run the automated airplane RCS pipeline** and adapt frequency/coarsening settings by editing script parameters.
4. **Interpret reported RCS diagnostics** (bistatic statistics and monostatic backscatter).
5. **Trade off accuracy vs speed** by adjusting the target RWG count and understanding coarsening artifacts.
6. **Generate mesh preview images** in a custom post-processing step when needed.

---

## Workflow Overview

The airplane RCS demo follows a defensive pipeline designed to handle real‑world CAD imperfections:

```
OBJ import → repair defects → optional coarsening → EFIE assembly →
forward solve → far‑field → bistatic + monostatic RCS diagnostics
```

Each step is logged for auditability.

---

## Step‑by‑Step Walkthrough

### 1) Obtain an Airplane OBJ File

The bundled script loads `examples/demo_aircraft.obj` by default. For a quick
run, no extra file setup is needed.
If you want to use your own OBJ, copy the script and replace `obj_path`
with your mesh path.

### 2) Run the Pipeline with Default Parameters

From the project root:

```bash
julia --project=. examples/06_aircraft_rcs.jl
```

This example script currently takes no CLI arguments. It runs a fixed demo
workflow (repair + optional coarsening + solve + RCS diagnostics) and prints
results to the console.

### 3) Understand the Console Output

The script prints a log like:

```
============================================================
Example 06: Aircraft RCS Pipeline
============================================================

Loading: .../examples/demo_aircraft.obj
  Raw mesh: ... vertices, ... triangles

Repairing mesh...
  Removed invalid: ...
  Removed degenerate: ...
  Removed non-manifold: ...

── Solving PEC scattering ──
  Unknowns = ...
  Assembly: ... s
  Solve: ... s, residual = ...

── Bistatic RCS statistics ──
  Min: ... dBsm
  Max: ... dBsm
  Backscatter (nose-on): ... dBsm
  P_rad/P_in = ...
```

**Key metrics:**

- **Unknown count**: Reported after RWG build; controls dense-memory feasibility.
- **Relative residual**: Should be < 1e‑10 for a well‑conditioned solve.
- **Monostatic RCS**: Reported in both m² and dBsm (dB relative to 1 m²).

### 4) Inspect Repaired/Coarsened Geometry

Use `mesh_repaired` and `mesh_coarse` in a custom script if you want to export
OBJ files or save side-by-side preview figures with `save_mesh_preview`.

### 5) Post-Process RCS

The demo computes `σ` and `σ_dB` in memory. To create publication plots or CSV
exports, add a short post-processing block that writes these arrays.

---

## Detailed Explanation of Each Step

### Mesh Import and Scaling

```julia
mesh_in = read_obj_mesh(input_path)
mesh_scaled = TriMesh(mesh_in.xyz .* scale_to_m, copy(mesh_in.tri))
```

CAD models often use arbitrary units (mm, inches, dimensionless). The `scale_to_m` parameter converts to meters, which are required for physical wavenumber $k = 2\pi/\lambda$.

### Mesh Repair

`repair_mesh_for_simulation` performs a series of sanity fixes:

| Operation | Purpose |
|-----------|---------|
| `drop_degenerate=true` | Remove triangles with zero area |
| `fix_orientation=true` | Ensure consistent outward‑pointing normals |
| `strict_nonmanifold=true` | Remove non‑manifold edges (where >2 triangles meet) |
| `allow_boundary=true` | Permit open surfaces (e.g., aircraft without interior) |
| `require_closed=false` | Do not insist on watertight mesh |

The function returns a named tuple with before/after quality reports and
repair counters (flipped triangles, removed invalid/degenerate/non-manifold
triangles, and final mesh quality status).

### Coarsening to Target RWG Count

If the repaired mesh yields more RWG functions than `target_rwg`, the script calls `coarsen_mesh_to_target_rwg`. This iterative algorithm:

1. Computes a curvature‑weighted edge‑collapse priority.
2. Collapses the lowest‑priority edge that does not violate quality thresholds.
3. Rebuilds RWG basis and checks count.
4. Repeats until `RWG ≤ target_rwg` or max iterations reached.

Coarsening preserves sharp features (wing edges, tail fins) while reducing triangle density on flat regions. The `area_tol_rel` parameter prevents collapse that would change local area by more than a fraction.

**Trade‑off:** Aggressive coarsening reduces accuracy but keeps memory feasible. A good rule of thumb is to target $N$ such that `estimate_dense_matrix_gib(N)` is ≤ 0.5× available RAM.

### EFIE Assembly and Solve

Standard MoM workflow with default quadrature order 3. The forward solve uses direct factorization (LU) because $N$ is small after coarsening.

### Far‑Field and RCS Computation

A spherical grid with 121 θ points (1.5° resolution) and 36 φ points (10° resolution) is used for performance. The bistatic RCS is computed for all directions; a φ ≈ 0° cut is extracted for plotting. Monostatic RCS is sampled at the backscatter direction (θ = 180°, φ = 0°).

### Optional Exports

The current script reports results to stdout. If you need persistent artifacts
(CSV/OBJ/preview images), add explicit write steps with:
`write_obj_mesh`, `DelimitedFiles.writedlm` (or CSV.jl), and
`save_mesh_preview`.

---

## Interpretation Guidelines

### Coarsening Artifacts

| Artifact | Effect on RCS | Mitigation |
|----------|---------------|------------|
| **Smoothed curvatures** (e.g., rounded fuselage) | Shifts specular lobe angles, reduces creeping‑wave contributions | Increase `target_rwg`; use curvature‑weighted coarsening (already default) |
| **Lost small features** (antenna, wingtips) | Eliminates high‑frequency scattering | Preserve feature edges via manual mesh preprocessing |
| **Increased triangle aspect ratio** | May degrade EFIE accuracy, raise condition number | Set stricter `area_tol_rel` (e.g., 1e‑4) |

### RCS Pattern Features

- **Specular lobe**: Strong peak near θ = 0° (forward scattering) and θ = 180° (backscatter). Width inversely proportional to electrical size.
- **Nulls**: Directions where scattered field cancels; sensitive to geometry details.
- **Sidelobes**: Secondary maxima caused by diffraction at edges (wing, tail).

With coarse meshes, nulls may fill in and sidelobes may be suppressed. The overall pattern shape should still resemble the expected scattering physics.

### Monostatic RCS Validity

The reported monostatic value is a **single sample** on the angular grid. The `angular_error_deg` column gives the angular distance to the exact backscatter direction; if > grid resolution, consider interpolating between adjacent grid points.

---

## Troubleshooting

### Error: “Mesh has non‑manifold edges after repair”

**Cause:** CAD export contains topological errors that automatic repair cannot fix.

**Solutions:**

- Manually clean mesh in Blender/MeshLab before import.
- Set `strict_nonmanifold=false` (may produce unphysical currents).
- Use `drop_nonmanifold_triangles(mesh)` to remove offending triangles.

### Error: Coarsening fails to reach target RWG within max iterations

**Cause:** Mesh is already near minimal triangle count for its genus.

**Solutions:**

- Increase `target_rwg` to a more realistic value.
- Reduce `area_tol_rel` to allow more aggressive collapses.
- Manually simplify mesh in external tool (e.g., MeshLab’s quadratic edge collapse).

### Warning: “Mesh preview generation failed” (custom scripts)

**Cause:** Plotly export backend not available.

**Solutions:**

- Ignore if preview is optional.
- Install `PlotlySupply` and `PlotlyKaleido`.
- Generate only numerical diagnostics when running headless.

### Poor Residual (> 1e‑10)

**Cause:** Ill‑conditioned $Z$ matrix due to coarsening‑induced poorly‑shaped triangles.

**Solutions:**

- Enable preconditioning: modify script to call `make_left_preconditioner`.
- Increase `target_rwg` to improve mesh quality.

### Monostatic RCS varies wildly with small incidence‑angle changes

**Cause:** Coarse angular grid misses sharp backscatter peaks.

**Solutions:**

- Use finer spherical grid (`make_sph_grid(361, 72)`).
- Interpolate RCS near backscatter direction using neighboring samples.

---

## Code Mapping

| Task | Function | Source File | Key Lines |
|------|----------|-------------|-----------|
| **OBJ import** | `read_obj_mesh` | `src/geometry/Mesh.jl` | 100–120 |
| **Mesh repair** | `repair_mesh_for_simulation` | `src/geometry/Mesh.jl` | 300–350 |
| **Coarsening** | `coarsen_mesh_to_target_rwg` | `src/geometry/Mesh.jl` | 400–450 |
| **RWG building** | `build_rwg` | `src/basis/RWG.jl` | 50–80 |
| **Memory estimate** | `estimate_dense_matrix_gib` | `src/geometry/Mesh.jl` | 200–220 |
| **Mesh preview** | `save_mesh_preview` | `src/postprocessing/Visualization.jl` | 150–200 |
| **Complete pipeline (repair + solve + diagnostics)** | `06_aircraft_rcs.jl` | `examples/` | full script |
| **Plotting helpers** | `save_mesh_preview`, `plot_mesh_wireframe` | `src/postprocessing/Visualization.jl` | use in custom scripts |

---

## Exercises

### Basic (60 minutes)

1. **Run the pipeline** with a simple sphere OBJ (generate with `write_obj_mesh`). Compare monostatic RCS with the Mie solution (Tutorial 4). How does coarsening affect accuracy?
2. **Vary target RWG** (200, 400, 600) and plot monostatic RCS vs unknown count. Is there convergence?
3. **Inspect coarsening artifacts**: Export `mesh` after coarsening with `write_obj_mesh`, load it in a mesh viewer, and identify regions where triangle density is disproportionately reduced.

### Practical (90 minutes)

1. **Import a real CAD model** (e.g., vehicle, drone). Adjust `scale_to_m` to achieve plausible physical size (e.g., 5 m wingspan). Run the pipeline and discuss plausible vs unphysical RCS features.
2. **Implement adaptive coarsening**: Modify the script to coarsen until memory estimate ≤ 2 GiB (instead of fixed RWG target). Use `estimate_dense_matrix_gib` in the loop.
3. **Add incidence‑angle sweep**: Modify `examples/06_aircraft_rcs.jl` to compute monostatic RCS for 5 incidence directions (θ = 0°, 45°, 90°, 135°, 180°). Plot RCS vs incidence angle.

### Advanced (2 hours)

1. **Compare with fast‑method prototype**: Replace the dense direct solve with an iterative solver (GMRES) and left preconditioning. Measure time/memory vs dense solve for N = 500, 1000, 2000.
2. **Quantify coarsening error**: For a sphere, compute MAE(dB) between RCS of original mesh and coarsened mesh (same frequency). Plot error vs coarsening ratio (triangles‑after / triangles‑before).
3. **Extend to impedance boundary condition**: Modify the pipeline to simulate a partially‑absorbing airplane (e.g., carbon‑composite skin) by adding resistive impedance sheets. Use `assemble_full_Z` with `reactive=false` and small θ values (~100 Ω). Compare RCS patterns with PEC case.

---

## Tutorial Checklist

Before applying the pipeline to your own CAD models, ensure you can:

- [ ] **Run the demo** with the provided sphere OBJ and obtain plausible RCS outputs.
- [ ] **Interpret memory estimates** and adjust target RWG accordingly.
- [ ] **Identify mesh defects** in the repair log and know how to fix them externally.
- [ ] **Explain trade‑offs** of coarsening: which scattering features are preserved, which are lost.
- [ ] **Generate and interpret** the heuristic RCS plots, noting the effect of angular sampling.

---

## Further Reading

- **Paper Section 6.2** – Complex‑platform workflow and coarsening strategy.
- **`src/geometry/Mesh.jl`** – Implementation of mesh repair and coarsening algorithms.
- **`src/postprocessing/Visualization.jl`** – Mesh preview and wireframe plotting utilities.
- **Tutorial 6: OBJ Repair Workflow** – Deep dive into mesh repair and quality metrics.
- **Advanced Workflows, Chapter 2: Large‑Problem Strategy** – Scaling beyond dense‑matrix limits.
