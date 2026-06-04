# API: Visualization

## Purpose

Reference for mesh plotting helpers that produce interactive 3D wireframe views using [PlotlySupply.jl](https://github.com/plotly/PlotlySupply.jl). These are used for quick mesh inspection, comparing original vs. repaired/coarsened meshes, and generating publication-ready figures.

---

## When to Use

- **Before simulation:** Visually verify that an imported OBJ mesh looks correct (no missing faces, no gross distortion) and that coarsening preserved the geometry.
- **During debugging:** Confirm that mesh repair (vertex merging, non-manifold removal) did not damage the surface.
- **For publication:** Generate side-by-side comparison plots of different mesh densities.

---

## `plot_mesh_wireframe(mesh; kwargs...)`

Create an interactive 3D wireframe plot of a single mesh.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh to plot. |
| `color` | Symbol or String | `:steelblue` | Line color for the wireframe edges. Any CSS color name or hex string works (e.g., `:red`, `"#ff0000"`). |
| `title` | `AbstractString` | `"Mesh"` | Plot title displayed above the 3D viewport. |
| `camera` | `Tuple{Real,Real}` | `(30, 30)` | Camera view angles `(azimuth, elevation)` in degrees. `(0, 0)` looks along +x; `(0, 90)` looks straight down from +z. |
| `linewidth` | `Real` | `0.7` | Wireframe line width in pixels. Increase to 1.0--1.5 for coarse meshes; decrease to 0.3--0.5 for dense meshes. |
| `xlims`, `ylims`, `zlims` | Tuple or `nothing` | `nothing` | Axis limits `(min, max)` in meters. Default: auto-scaled from mesh extents. Set manually to align multiple plots. |
| `size` | `Tuple{Int,Int}` | `(700, 500)` | Figure size `(width, height)` in pixels. |
| `guidefontsize` | `Int` | `12` | Font size for axis labels ("x (m)", "y (m)", "z (m)"). |
| `tickfontsize` | `Int` | `10` | Font size for axis tick numbers. |
| `titlefontsize` | `Int` | `12` | Font size for the plot title. |
| `kwargs...` | -- | -- | Additional keyword arguments forwarded to `relayout!` for fine-grained Plotly layout control. |

**Returns:** PlotlySupply plot object. Display it in a Jupyter notebook or Pluto cell by returning it; save it with `savefig(p, "path.png")`.

**How it works:** Internally calls `mesh_wireframe_segments(mesh)` (from `Mesh.jl`) to extract all triangle edges as disconnected line segments, then renders them as a single Plotly `scatter3d` trace with `mode="lines"`. The axes use equal aspect ratio (`aspectmode = "cube"`) so the geometry is not distorted.

**Example:**

```julia
mesh = read_obj_mesh("data/airplane.obj")
p = plot_mesh_wireframe(mesh; title="Airplane", camera=(45, 30), linewidth=0.5)
# In a notebook, just return p to display
# To save: savefig(p, "airplane_wireframe.png"; width=700, height=500)
```

---

## `plot_mesh_comparison(mesh_a, mesh_b; kwargs...)`

Create side-by-side 3D wireframe plots with shared axis limits and equal aspect ratio. Useful for comparing a repaired mesh against the original, or a coarsened mesh against the full-resolution version.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh_a` | `TriMesh` | -- | Left-panel mesh. |
| `mesh_b` | `TriMesh` | -- | Right-panel mesh. |
| `title_a` | `AbstractString` | `"Mesh A"` | Title for the left panel. |
| `title_b` | `AbstractString` | `"Mesh B"` | Title for the right panel. |
| `color_a` | Symbol or String | `:steelblue` | Line color for mesh A. |
| `color_b` | Symbol or String | `:darkorange` | Line color for mesh B. |
| `camera` | `Tuple{Real,Real}` | `(30, 30)` | Camera angles applied to both panels (same viewpoint for fair comparison). |
| `size` | `Tuple{Int,Int}` | `(1200, 520)` | Total figure size `(width, height)` in pixels. The two panels split the width evenly. |
| `pad_frac` | `Float64` | `0.04` | Fractional padding around the combined bounding box of both meshes. `0.04` = 4% padding on each side. |
| `linewidth` | `Real` | `0.7` | Line width for both wireframes. |
| `guidefontsize` | `Int` | `10` | Axis label font size (smaller default to fit two panels). |
| `tickfontsize` | `Int` | `8` | Tick font size. |
| `titlefontsize` | `Int` | `10` | Subplot title font size. |
| `kwargs...` | -- | -- | Forwarded to `relayout!`. |

**Returns:** PlotlySupply plot object with two side-by-side 3D subplots.

**Shared axis limits:** Both panels use the same axis range computed from the combined bounding box of `mesh_a` and `mesh_b`. This ensures geometry changes (e.g., from coarsening) are visually apparent without scale distortion.

**Example:**

```julia
mesh_orig = read_obj_mesh("data/airplane.obj")
rep = repair_mesh_for_simulation(mesh_orig)
p = plot_mesh_comparison(mesh_orig, rep.mesh;
    title_a="Original ($(ntriangles(mesh_orig)) tri)",
    title_b="Repaired ($(ntriangles(rep.mesh)) tri)",
    camera=(60, 25))
```

---

## `save_mesh_preview(mesh_a, mesh_b, out_prefix; kwargs...)`

Generate a side-by-side comparison plot and save it as both PNG and PDF. This is a convenience wrapper around `plot_mesh_comparison` for batch processing.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh_a` | `TriMesh` | Left-panel mesh. |
| `mesh_b` | `TriMesh` | Right-panel mesh. |
| `out_prefix` | `AbstractString` | Output file prefix (without extension). E.g., `"figs/airplane_preview"` produces `"figs/airplane_preview.png"` and `"figs/airplane_preview.pdf"`. |
| `kwargs...` | -- | Forwarded to `plot_mesh_comparison`. |

**Returns:** Named tuple `(plot, png_path, pdf_path)`:

| Field | Type | Description |
|-------|------|-------------|
| `plot` | PlotlySupply plot | The generated plot object (can be displayed or further customized). |
| `png_path` | `String` | Full path to the saved PNG file. |
| `pdf_path` | `String` | Full path to the saved PDF file. |

**Directory creation:** Parent directories are created automatically if they do not exist (via `mkpath`).

**Example:**

```julia
mesh_rep = read_obj_mesh("data/airplane_repaired.obj")
coa = coarsen_mesh_to_target_rwg(mesh_rep, 500)
preview = save_mesh_preview(mesh_rep, coa.mesh, "figs/airplane_mesh_preview";
    title_a="Repaired", title_b="Coarsened (N~500)")
println("Saved: ", preview.png_path)
println("Saved: ", preview.pdf_path)
```

---

## Camera Angle Guide

The `camera` parameter is `(azimuth, elevation)` in degrees:

| View | Camera Setting | Description |
|------|---------------|-------------|
| Default 3D | `(30, 30)` | Standard isometric-like view. Good general-purpose default. |
| Front | `(0, 0)` | Looking along the +x axis (sees the yz-plane). |
| Top-down | `(0, 90)` | Looking straight down from +z (sees the xy-plane). |
| Side | `(90, 0)` | Looking along the +y axis (sees the xz-plane). |
| Steep angle | `(45, 60)` | Emphasizes the z-dimension; good for parabolic reflectors. |

---

## Notes

- **Backend:** Uses PlotlySupply.jl (a lightweight Plotly.js wrapper), not Plots.jl. Plots are interactive in Jupyter/Pluto — you can rotate, zoom, and pan with the mouse.
- **Performance:** Wireframe rendering handles meshes with thousands of triangles without issue. For very large meshes (>50k triangles), the browser-rendered Plotly plot may become sluggish; consider coarsening first for visualization.
- **Aspect ratio:** All plots use `aspectmode = "cube"` to ensure equal scaling on all three axes. This prevents flat structures from appearing stretched.
- **Transparent background:** The 3D scene background is transparent (`rgba(0,0,0,0)`), which works well when embedding in papers or presentations.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/postprocessing/Visualization.jl` | `plot_mesh_wireframe`, `plot_mesh_comparison`, `save_mesh_preview` |
| `src/geometry/Mesh.jl` | `mesh_wireframe_segments` (extracts edge segments from `TriMesh`) |

**Example scripts:** `examples/06_aircraft_rcs.jl`, `examples/12_plate_rcs_stl_roundtrip.jl`

---

## Exercises

- **Basic:** Plot a single mesh with two different camera settings (e.g., default `(30, 30)` and top-down `(0, 90)`) and compare the views.
- **Practical:** Use `save_mesh_preview` to compare a repaired mesh and a coarsened version. Include the triangle count in the subplot titles.
- **Challenge:** Generate previews at three coarsening levels (e.g., target N = 200, 500, 1000) and assess geometric fidelity tradeoffs visually.
