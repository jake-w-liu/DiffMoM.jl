# Visualization

## Purpose

This chapter treats visualization as part of the *numerical verification loop*:
before assembly/solve, you confirm geometry integrity, orientation plausibility,
and scale realism from package-native plotting utilities.

---

## Learning Goals

After this chapter, you should be able to:

1. Use package plotting tools to validate repair/coarsening results quickly.
2. Understand why shared axis limits and equal aspect ratio are required for
   physically meaningful visual comparisons.
3. Export reproducible preview artifacts for reports and experiment logs.

---

## 1) Importance of Visualization in MoM Pipeline

Many forward-solve failures are geometry failures in disguise:

- unintended topology changes after repair,
- over-aggressive coarsening,
- unit-scale mistakes (mm treated as m, etc.),
- disconnected surface fragments.

A mesh preview often reveals these issues faster than matrix diagnostics.

---

## 2) Implementation in `DiffMoM.jl`

### 2.1 Core Functions

The visualization module (`src/postprocessing/Visualization.jl`) provides three main user‑facing functions:

- **`plot_mesh_wireframe(mesh; kwargs...)`** – 3D wireframe plot of a single mesh with consistent axis scaling and physical units (meters).
- **`plot_mesh_comparison(mesh_a, mesh_b; kwargs...)`** – Side‑by‑side wireframe plots with **shared axis limits** and **equal aspect ratio** for reliable visual comparison.
- **`save_mesh_preview(mesh_a, mesh_b, out_prefix; kwargs...)`** – Generate and save comparison plots as PNG and PDF files, returning paths for logging.

### 2.2 Underlying Algorithms

**Wireframe segment extraction (`mesh_wireframe_segments` in `src/geometry/Mesh.jl`):** For each interior edge (shared by two triangles) and each boundary edge (belonging to one triangle), the function creates a line segment between the two edge vertices. Segments are concatenated with `NaN` separators to produce a single continuous path for efficient plotting with `PlotlySupply`. The wireframe draws ALL unique edges regardless of manifold status.

**Shared axis limits (`_realistic_axis_limits`):** To prevent visual distortion when comparing two meshes, the function computes a common cubic bounding box:

1. Compute global minima and maxima across **all** input meshes for each coordinate axis.
2. Determine the maximum span $s = \max(x_{\max} - x_{\min}, y_{\max} - y_{\min}, z_{\max} - z_{\min})$.
3. Define a cube centered at the global centroid with half‑side length $s/2 \times (1 + \epsilon)$ where $\epsilon$ is a small padding fraction (`pad_frac`).
4. Set identical `xlims`, `ylims`, `zlims` for all subplots.

This guarantees that geometric differences reflect true shape changes, not arbitrary axis scaling.

**Aspect ratio enforcement:** Both subplots in `plot_mesh_comparison` use `aspectmode = "cube"` internally, ensuring that a unit length in the $x$, $y$, and $z$ directions occupies the same screen distance. Combined with shared limits, this makes visual comparison trustworthy.

### 2.3 Keyword Arguments and Customization

Each function accepts standard `PlotlySupply` keywords (e.g., `color`, `linewidth`, `camera`, `title`) plus package‑specific options:

- **`camera = (30, 30)`** – azimuth and elevation angles in degrees.
- **`pad_frac = 0.04`** – fractional padding added to the cubic bounding box.
- **`size = (1200, 520)`** – total figure size in pixels (width × height).
- **`color_a`, `color_b`** – distinct colors for the two meshes in comparison plots.

All functions return a `PlotlySupply` plot object. PNG/PDF/SVG export is handled by PlotlySupply.

### 2.4 Integration with the MoM Pipeline

Visualization is designed to be inserted at critical points in the mesh‑processing workflow:

1. **After repair** – verify topology corrections and orientation consistency.
2. **After coarsening** – assess geometric fidelity relative to the original mesh.
3. **Before RWG assembly** – confirm scale and overall shape are physically plausible.

The `save_mesh_preview` function produces reproducible artifacts for reports, experiment logs, and continuous‑integration checks.

---

## 3) Practical Workflow Examples

### 3.1 Single-Mesh Sanity View

```julia
using DiffMoM

mesh = read_obj_mesh("airplane_repaired.obj")

p = plot_mesh_wireframe(
    mesh;
    title = "Repaired mesh",
    color = :steelblue,
    camera = (30, 30),
    linewidth = 0.7,
)
display(p)
```

What to check visually:

1. gross geometry and connectivity,
2. expected length scale (meters in this package),
3. no obvious detached fragments.

---

### 3.2 Repaired vs Simulation Mesh Comparison

This is the most useful diagnostic before a large run.

```julia
mesh_rep = read_obj_mesh("airplane_repaired.obj")
mesh_sim = read_obj_mesh("airplane_coarse.obj")

p = plot_mesh_comparison(
    mesh_rep,
    mesh_sim;
    title_a = "Repaired mesh",
    title_b = "Simulation mesh",
    color_a = :steelblue,
    color_b = :darkorange,
    camera = (30, 30),
    size = (1200, 520),
    pad_frac = 0.04,
)
display(p)
```

Because limits are shared and aspect ratio is equal, you can reliably answer:

- did coarsening preserve the dominant shape?
- are appendages (wings/tail/fins) still represented?
- is the model scale still realistic?

---

### 3.3 Saving Artifacts for Reproducibility

```julia
preview = save_mesh_preview(
    mesh_rep,
    mesh_sim,
    "figs/airplane_mesh_preview";
    title_a = "Repaired mesh",
    title_b = "Simulation mesh",
    camera = (30, 30),
)

println(preview.png_path)
println(preview.pdf_path)
```

The returned paths make it easy to log generated assets in scripts/CI.

---

### 3.4 Annotated Workflow Before EFIE Assembly

```julia
using DiffMoM

# (A) Import + repair
mesh_raw = read_obj_mesh("Airplane.obj")
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true)
mesh_rep = rep.mesh

# (B) Coarsen to target unknown count
coarse = coarsen_mesh_to_target_rwg(mesh_rep, 600)
mesh_sim = coarse.mesh

# (C) Visual check (shared limits + equal aspect)
plot_mesh_comparison(mesh_rep, mesh_sim;
    title_a = "Repaired (high detail)",
    title_b = "Simulation (coarsened)",
    # internally uses aspectmode = "cube"
)

# (D) Then build RWG and solve
rwg = build_rwg(mesh_sim; precheck=true, allow_boundary=true)
```

This ordering avoids expensive solves on obviously bad meshes.

---

### 3.5 Practical Interpretation Tips

1. **“Broken-looking” coarsened mesh** can be acceptable if global scattering
   observables are your target and topology remains valid.
2. **Large geometric distortion** in critical features means coarsening is too
   aggressive for your intended observable.
3. Always pair visual checks with `mesh_quality_report` counts; the two are
   complementary.

---

## 4) Troubleshooting Common Visualization Issues

### 4.1 Empty or Incomplete Wireframe

**Symptoms:** The plot appears empty, shows only a few edges, or lacks expected detail.

**Possible causes and remedies:**

1. **Mesh scale mismatch:** If the mesh coordinates are in millimeters but the axis limits are set for meters, the geometry may be outside the viewport. Use `plot_mesh_wireframe` with default limits (automatically computed) or check `_realistic_axis_limits` output.
2. **Disconnected components:** If the mesh contains disconnected surface fragments, some may appear outside the viewport. Run `mesh_quality_report` to identify topological issues.
3. **Camera orientation:** The default camera `(30, 30)` may be looking from a direction that hides features. Adjust the `camera` keyword (azimuth, elevation) to rotate the view.

### 4.2 Misleading Visual Comparison

**Symptoms:** Two meshes appear dramatically different in size or shape even though they represent the same geometry.

**Checklist:**

- **Shared axis limits:** Verify that `plot_mesh_comparison` is used (not two separate calls to `plot_mesh_wireframe`). Independent plots autoscale axes differently.
- **Equal aspect ratio:** The implementation uses `aspectmode = "cube"` internally (the default in `plot_mesh_comparison`).
- **Padding factor:** If `pad_frac` is too large (e.g., `> 0.5`), the cubic bounding box may dwarf small geometric details. Reduce `pad_frac` to `0.02`–`0.05`.

### 4.3 Missing Plotting Backend

**Symptoms:** Julia throws an error about the plotting backend not being installed.

**Solution:** The package uses PlotlySupply for plotting and file export. Ensure PlotlySupply is installed and up to date:
```julia
import Pkg
Pkg.add("PlotlySupply")
```

### 4.4 File‑Save Failures

**Symptoms:** `save_mesh_preview` returns but no PNG/PDF files are created, or permissions errors occur.

**Diagnostics:**

- Check that the output directory exists (the function creates parent directories via `mkpath`).
- Ensure write permissions for the destination.
- PNG, PDF, and SVG export are handled by PlotlySupply. Ensure PlotlySupply is installed and working.

---

## 5) Code Mapping

- Plot implementations: `src/postprocessing/Visualization.jl`
- Wireframe segment generation: `src/geometry/Mesh.jl`
- End-to-end examples:
  - `examples/06_aircraft_rcs.jl`
  - `examples/12_plate_rcs_stl_roundtrip.jl`

---

## 6) Exercises

- Basic: render one repaired mesh and annotate `(Nv, Nt, Nedges)` in title.
- Intermediate: compare two target-RWG coarsened meshes and justify which one
  you would trust for monostatic RCS.
- Challenge: create a scripted pre-solve gate that fails if visual preview and
   quality report suggest severe geometric loss.

---

## 7) Chapter Checklist

Before relying on visualization for mesh validation, ensure you can:

- [ ] Generate a wireframe plot of a single mesh with correct scale and orientation.
- [ ] Compare two meshes side‑by‑side using `plot_mesh_comparison` with shared axis limits and equal aspect ratio.
- [ ] Save comparison plots as PNG/PDF files for documentation and reproducibility.
- [ ] Interpret visual differences in the context of mesh repair and coarsening.
- [ ] Troubleshoot common plotting issues (empty frames, misleading scaling, missing backends).
- [ ] Integrate visual checks into a scripted pre‑solve validation gate.

---

## 8) Further Reading

- **PlotlySupply:** The plotting backend used by the package for 3D visualization and file export (PNG/PDF/SVG).
- **Mesh processing for EM:** Shepard, *Mesh Generation and Quality Criteria for Computational Electromagnetics* (2002).
- **Visual debugging in scientific computing:** Johansson & Forssén, *Visualization as a Tool for Debugging Numerical Software* (2016).
- **Package examples:** `examples/06_aircraft_rcs.jl` and `examples/12_plate_rcs_stl_roundtrip.jl` demonstrate end‑to‑end workflows.
