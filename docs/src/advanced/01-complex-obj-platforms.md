# Complex OBJ Platforms

## Purpose

Describe robust workflows for importing and simulating complex CAD/OBJ
platforms (e.g., aircraft-like geometry) within the package’s dense-MoM
envelope.

---

## Learning Goals

After this chapter, you should be able to:

1. Diagnose OBJ geometry issues that break simulation.
2. Repair/coarsen meshes to a stable simulation-ready state.
3. Run platform RCS heuristics with reproducible preprocessing.

---

## 1) Common OBJ Failure Modes

Complex CAD meshes exported as OBJ files often contain geometric and topological defects that prevent successful RWG discretization and EFIE assembly. The package’s mesh‑quality pipeline (`src/geometry/Mesh.jl`) detects these issues early, but understanding their nature helps you decide which repairs are essential.

### 1.1 Topological Defects

- **Non‑manifold edges:** An edge shared by three or more triangles violates the manifold requirement for RWG bases. The repair routine either splits the edge (creating duplicate vertices) or, if `strict_nonmanifold=true`, removes offending triangles.
- **Inconsistent triangle orientation:** Adjacent triangles should have consistent outward‑pointing normals (right‑hand rule from vertex order). Inconsistent orientation breaks current continuity across edges. The repair function `repair_mesh_for_simulation` flips triangles to achieve consistent orientation.
- **Disconnected fragments:** The mesh may consist of several isolated components (e.g., separate parts of an assembly). For a single scattering body, all triangles must form a connected surface. The quality report lists component counts; you may need to manually remove irrelevant fragments.

### 1.2 Geometric Defects

- **Degenerate triangles:** Triangles with zero area (collinear vertices) or near‑zero area cause singular integrals and ill‑conditioned mass matrices. The repair pipeline automatically discards them when `drop_degenerate=true`.
- **Duplicate faces:** Identical triangles (same vertex indices) waste computation and can lead to singular linear systems. The repair step removes duplicates.
- **Unrealistic scale:** CAD models often use millimeters, while the MoM formulation expects meters. Scaling errors of $10^3$ change electrical size dramatically. Always verify physical dimensions with `plot_mesh_wireframe` and compare with expected wavelength.

### 1.3 Diagnostic Tools

- **`mesh_quality_report(mesh)`** – prints counts of vertices, triangles, manifold edges, boundary edges, non‑manifold edges, degenerate triangles, and connected components.
- **`plot_mesh_wireframe`** – visual inspection of overall shape, scale, and connectivity.
- **`repair_mesh_for_simulation`** – returns a named tuple detailing flipped triangles, removed triangles, and before/after topology diagnostics.

**Reference:** Part II, Chapter 1 (Mesh Pipeline) provides a comprehensive treatment of mesh quality criteria and repair algorithms.

---

## 2) Recommended Platform Workflow

### 2.1 Step‑by‑Step Procedure

```julia
using DiffMoM

# 1. Import raw CAD mesh (units: meters)
mesh_raw = read_obj_mesh("aircraft.obj")

# 2. Repair topology and orientation
repair_result = repair_mesh_for_simulation(
    mesh_raw;
    allow_boundary=true,          # permit open surfaces (e.g., aircraft fuselage)
    require_closed=false,         # allow boundaries (not a closed body)
    drop_invalid=true,            # remove degenerate triangles
    drop_degenerate=true,
    fix_orientation=true,         # enforce consistent triangle normals
    strict_nonmanifold=true,      # abort if non‑manifold edges cannot be repaired
)
mesh_repaired = repair_result.mesh

# 3. Coarsen to target RWG count (if needed)
target_rwg = 500  # adjust based on memory constraints
coarse_result = coarsen_mesh_to_target_rwg(
    mesh_repaired, target_rwg;
    max_iters=10,
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
    strict_nonmanifold=true,
)
mesh_sim = coarse_result.mesh

# 4. Build RWG basis functions
rwg = build_rwg(mesh_sim; precheck=true, allow_boundary=true)

# 5. Forward solve and RCS extraction (as in Part II, Chapter 2)
k = 2π * freq / 299792458.0
Z = assemble_Z_efie(mesh_sim, rwg, k)
v = assemble_v_plane_wave(...)
I = solve_forward(Z, v)
grid = make_sph_grid(...)
G = radiation_vectors(...)
E_ff = compute_farfield(...)
σ = bistatic_rcs(...)
```

### 2.2 Critical Parameter Choices

- **`allow_boundary`:** Set `true` for open surfaces (aircraft, vehicles, antennas), `false` for closed PEC bodies (spheres, capsules). Boundary edges are allowed but produce half‑RWG functions.
- **`require_closed`:** Usually `false` for CAD platforms; set `true` only when the geometry must be watertight.
- **`strict_nonmanifold`:** `true` forces the repair to fail if non‑manifold edges remain; `false` attempts to fix them by splitting vertices.
- **Target RWG count:** Choose based on available memory: $N \approx 10,000$ fits in ~1.5 GB (complex double‑precision dense matrix). Use `estimate_dense_matrix_gib(N)` for a precise estimate.

### 2.3 Visualization at Each Stage

Insert visual checks after repair and coarsening:

```julia
using Plots
gr()

# Compare repaired vs. simulation mesh
plot_mesh_comparison(
    mesh_repaired, mesh_sim;
    title_a = "Repaired (detailed)",
    title_b = "Simulation (coarsened)",
    color_a = :steelblue,
    color_b = :darkorange,
    camera = (30, 30),
)
savefig("platform_mesh_comparison.png")
```

Visual confirmation prevents costly solves on geometrically corrupted meshes.

---

## 3) Reproducibility Pattern

Platform‑level simulations are computationally expensive; saving intermediate artifacts ensures that results can be audited, compared, and reproduced months later. The package provides utilities to automate this logging.

### 3.1 Artifact Checklist

For each platform run, archive the following:

| Artifact | Purpose | How to generate |
|----------|---------|-----------------|
| **Raw OBJ** | Original geometry | `read_obj_mesh` input |
| **Repaired OBJ** | Topologically correct mesh | `write_obj_mesh("platform_repaired.obj", mesh_repaired)` |
| **Coarsened OBJ** | Mesh used for simulation | `write_obj_mesh("platform_coarse.obj", mesh_sim)` |
| **Mesh preview** | Visual comparison | `save_mesh_preview(mesh_repaired, mesh_sim, "figs/platform_preview")` |
| **Quality reports** | Quantitative mesh statistics | `mesh_quality_report` output (save as text file) |
| **Metadata CSV** | Run parameters and counts | Custom table with `Nv`, `Nt`, `N_{\mathrm{RWG}}`, frequency, scale factor, repair flags |
| **RCS results** | Final observables | CSV outputs from `bistatic_rcs`, `backscatter_rcs` |

### 3.2 Automated Logging Example

```julia
function run_platform_study(obj_path, freq_ghz, target_rwg; out_dir="results")
    mkpath(out_dir)
    mkpath(joinpath(out_dir, "meshes"))
    mkpath(joinpath(out_dir, "figs"))
    mkpath(joinpath(out_dir, "reports"))

    # ... perform import, repair, coarsening, solve ...

    # Save artifacts
    write_obj_mesh(joinpath(out_dir, "meshes", "repaired.obj"), mesh_repaired)
    write_obj_mesh(joinpath(out_dir, "meshes", "coarse.obj"), mesh_sim)
    save_mesh_preview(mesh_repaired, mesh_sim,
        joinpath(out_dir, "figs", "preview"))
    
    # Write metadata
    open(joinpath(out_dir, "metadata.txt"), "w") do io
        println(io, "Platform: $obj_path")
        println(io, "Frequency: $freq_ghz GHz")
        println(io, "Vertices (repaired): $(nvertices(mesh_repaired))")
        println(io, "Triangles (repaired): $(ntriangles(mesh_repaired))")
        println(io, "RWG unknowns: $(rwg.nedges)")
    end
end
```

### 3.3 Version Control Considerations

Store raw OBJ files and the script that produced the results; treat repaired/coarsened meshes as derived data (regeneratable). Commit the script and metadata, but exclude large binary OBJ files and CSV results from version control (use `.gitignore`). Instead, archive them on a data‑storage system with persistent identifiers.

---

## 4) Performance Reality

Complex CAD platforms typically contain $10^4$–$10^6$ triangles, far beyond the dense‑MoM limit ($N \lesssim 5\times10^4$ on a workstation). Coarsening is therefore unavoidable, but it must be done **deliberately**, tracking quantitative metrics to ensure observables of interest remain stable.

### 4.1 Coarsening Trade‑Offs

- **Unknown count $N$:** Directly controls memory ($\sim 16 N^2$ bytes) and solve time ($\sim O(N^3)$ for dense factorization). Use `estimate_dense_matrix_gib(N)` to predict memory usage.
- **Geometric fidelity:** Aggressive coarsening distorts small features (antenna feeds, edges, slots) that may critically affect scattering. Visual comparison (`plot_mesh_comparison`) is qualitative; quantitative shape metrics (Hausdorff distance, volume change) are not yet implemented in the package.
- **Observable stability:** The ultimate criterion is **convergence of the target observable** (e.g., monostatic RCS at a few angles) with increasing $N$. Run a coarsening ladder (e.g., $N = 200, 400, 800, 1600$) and plot the observable vs. $N$; stop when relative change falls below your tolerance (e.g., 1 dB for RCS).

### 4.2 Practical Coarsening Strategy

1. **Start with a representative frequency** (usually the highest frequency of interest).
2. **Choose a target RWG count** that fits your memory budget, leaving room for multiple right‑hand sides and far‑field matrices.
3. **Generate a coarsening ladder** with `coarsen_mesh_to_target_rwg` using different target counts.
4. **Solve each level** and extract key observables (backscatter RCS at a few aspect angles, total scattered power, etc.).
5. **Decide on the coarsest mesh** that keeps observables within acceptable error bounds.

### 4.3 Example: Aircraft RCS Convergence

The script `examples/06_aircraft_rcs.jl` includes an optional coarsening step. Extend it to loop over target RWG counts and produce a convergence table:

```julia
targets = [200, 400, 800, 1600, 3200]
backscatter_db = Float64[]
for tg in targets
    coarse = coarsen_mesh_to_target_rwg(mesh_repaired, tg)
    mesh_tg = coarse.mesh
    # ... solve and compute monostatic RCS ...
    push!(backscatter_db, 10*log10(σ_mono))
end
```

Plot `backscatter_db` vs. `targets` to identify the “knee” where further refinement yields diminishing returns.

---

## 5) Code Mapping

### 5.1 Primary Source Files

- **`src/geometry/Mesh.jl`** – contains `read_obj_mesh`, `write_obj_mesh`, `repair_mesh_for_simulation`, `coarsen_mesh_to_target_rwg`, `mesh_quality_report`, and `mesh_wireframe_segments`. This module handles all geometry import, repair, coarsening, and quality diagnostics.
- **`src/postprocessing/Visualization.jl`** – provides `plot_mesh_wireframe`, `plot_mesh_comparison`, `save_mesh_preview` for visual validation.
- **`src/assembly/EFIE.jl`**, **`src/assembly/Excitation.jl`**, **`src/solver/Solve.jl`** – assemble and solve the forward system on the coarsened mesh.

### 5.2 Example Scripts

- **`examples/06_aircraft_rcs.jl`** – complete workflow for a complex platform: import, repair, coarsening, solve, far‑field and RCS extraction. This script is the reference implementation for this chapter.
- **Visualization helpers** – `plot_mesh_wireframe` and `plot_mesh_comparison` in `src/postprocessing/Visualization.jl`.

### 5.3 Supporting Utilities

- **`src/geometry/Mesh.jl`** – triangle operations (`triangle_area`, `triangle_center`, `triangle_normal`) and `estimate_dense_matrix_gib`.

---

## 6) Exercises

### 6.1 Basic Level

1. **OBJ import and repair:** Choose a CAD model (e.g., from [NASA 3D Resources](https://nasa3d.arc.nasa.gov/models) or your own). Load it with `read_obj_mesh`, run `repair_mesh_for_simulation`, and inspect the repair result with `mesh_quality_report`. Save the repaired mesh as an OBJ file.
2. **Visual sanity check:** Use `plot_mesh_wireframe` to view the raw and repaired meshes side‑by‑side. Verify that scale is plausible (meters) and that no obvious fragments are missing.

### 6.2 Intermediate Level

3. **Coarsening ladder:** For the repaired mesh, create three coarsened versions with target RWG counts of 300, 600, and 1200. For each level, compute the monostatic RCS at broadside incidence ($\theta=0^\circ$, $\phi=0^\circ$) and plot RCS (dBsm) vs. target count. Determine the coarsest mesh that keeps RCS within 2 dB of the finest mesh.
4. **Memory estimation:** Use `estimate_dense_matrix_gib` to predict memory usage for each coarsening level. Decide which level fits your available RAM.

### 6.3 Advanced Level

5. **Automated reproducibility script:** Write a function that ingests an OBJ path, a frequency, and a target RWG count, then runs the full pipeline (repair, coarsen, solve, RCS extraction) and saves all artifacts (OBJs, preview plots, metadata, RCS CSV) in a timestamped directory. Add command‑line argument parsing.
6. **Sensitivity to repair parameters:** Investigate how `strict_nonmanifold`, `allow_boundary`, and `require_closed` affect the repaired mesh topology and the resulting RCS. Present a decision tree for choosing these flags based on geometry characteristics.

---

## 7) Chapter Checklist

Before applying the platform workflow to a new CAD model, ensure you can:

- [ ] Diagnose common OBJ failure modes using `mesh_quality_report`.
- [ ] Repair a raw OBJ mesh with appropriate flags (`allow_boundary`, `strict_nonmanifold`, etc.).
- [ ] Coarsen a mesh to a target RWG count while monitoring geometric fidelity.
- [ ] Generate side‑by‑side visual comparisons of repaired and simulation meshes.
- [ ] Save all intermediate artifacts (OBJ, plots, metadata) for reproducibility.
- [ ] Perform a coarsening‑convergence study for your observable of interest.
- [ ] Estimate memory requirements for a given unknown count.

---

## 8) Further Reading

- **CAD for EM:** Davidson, *CAD for Microwave and Electromagnetic Systems* (2012) – discusses geometry cleanup and meshing for computational electromagnetics.
- **Mesh repair algorithms:** Attene et al., *Mesh Repair* (2013) – survey of techniques for fixing non‑manifold edges, degenerate triangles, and orientation inconsistencies.
- **Platform RCS prediction:** Knott et al., *Radar Cross Section* (1993) – classic text on scattering from complex targets.
- **Package examples:** `examples/06_aircraft_rcs.jl` – a ready‑to‑run implementation of the workflow described in this chapter.
