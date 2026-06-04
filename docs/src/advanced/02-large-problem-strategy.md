# Large-Problem Strategy

## Purpose

The current package prioritizes **reference‑correct dense EFIE solves**,
ensuring exact algebraic solutions for validation and research reproducibility.
However, dense Method of Moments (MoM) has inherent scaling limits governed by
$O(N^2)$ memory and $O(N^3)$ solve‑time complexity, where $N$ is the number of
RWG unknowns.

This chapter provides a practical strategy for “as large as possible” runs
within the current implementation, explains the available cost‑control levers,
and outlines the transition path when your target problem exceeds the dense‑algebra
envelope.

---

---

## Learning Goals

After this chapter, you should be able to:

1. Estimate memory and solve‑time cost from RWG unknown count before assembling any matrices.
2. Understand the mathematical scaling laws and their practical implications for dense MoM.
3. Apply geometry coarsening and scenario simplification to fit problems within available resources.
4. Interpret preconditioning options and know when they help (and when they don’t).
5. Build a reproducible large‑mesh workflow: repair → coarsen → solve → audit.
6. Diagnose common large‑problem failures and implement tiered validation.
7. Plan a transition to fast‑method extensions when dense limits are reached.

---

## 1) Mathematical Foundation: Dense MoM Scaling Laws

The Method of Moments (MoM) with RWG basis functions and dense matrix algebra
follows well‑known scaling laws that determine the maximum feasible problem size
on a given hardware platform.

### Memory Scaling: $O(N^2)$

The dense impedance matrix $\mathbf{Z} \in \mathbb{C}^{N \times N}$ stores
$N^2$ complex entries. With `ComplexF64` (16 bytes per entry), the raw matrix
memory is

```math
\text{Memory}_{\text{raw}} = 16 N^2 \ \text{bytes}.
```

The package provides a helper function that converts this to GiB:

```julia
estimate_dense_matrix_gib(N)   # returns GiB
```

> **Important**: Factorizations (LU, Cholesky) and temporary buffers require
> additional memory beyond the raw matrix. Treat `estimate_dense_matrix_gib(N)`
> as a **lower bound**; a realistic budget is **2–3×** this estimate.

### Time Scaling: $O(N^3)$

Direct factorization (LU) of a dense $N \times N$ matrix scales as $O(N^3)$.
Assembly of $\mathbf{Z}$ also scales as $O(N^2)$ (or worse with high‑order
quadrature). Consequently, doubling $N$ increases memory by **4×** and
factorization time by **8×**.

### Practical Limits Table

| Unknowns (N) | Raw Matrix (GiB) | Realistic Memory (GiB) | LU Flops (≈) | Typical Solve Time* |
|--------------|------------------|------------------------|--------------|---------------------|
| 500          | 0.004            | 0.01–0.02              | 0.1 Tflop    | < 1 s               |
| 1,000        | 0.016            | 0.03–0.05              | 1 Tflop      | 1–5 s               |
| 2,000        | 0.064            | 0.13–0.20              | 8 Tflop      | 10–30 s             |
| 5,000        | 0.40             | 0.8–1.2                | 125 Tflop    | 2–10 min            |
| 10,000       | 1.6              | 3–5                    | 1 Pflop      | 20–60 min           |
| 20,000       | 6.4              | 12–20                  | 8 Pflop      | hours–days          |

\* Times are indicative for a modern desktop CPU (single‑threaded LU). Parallel
BLAS can reduce factor‑solve time but not the asymptotic scaling.

### Reference to the Paper

The dense EFIE formulation follows the standard Galerkin discretization described
in the accompanying paper `bare_jrnl.tex` (Eq. 1–4). The $O(N^2)$ memory and
$O(N^3)$ solve complexity are inherent to the dense‑algebra approach; fast
methods (FMM/MLFMM) break these scalings but are outside the current release.

---

## 2) Practical Preflight Checklist and Memory Estimation

Before committing to expensive matrix assembly, perform a lightweight preflight
analysis to verify feasibility and avoid out‑of‑memory failures.

### Step‑by‑Step Preflight

```julia
using DiffMoM

# 1. Load and repair mesh (no assembly yet)
mesh = read_obj_mesh("platform.obj")
mesh_scaled = TriMesh(mesh.xyz .* 0.001, copy(mesh.tri))   # scale to meters
repair = repair_mesh_for_simulation(
    mesh_scaled;
    allow_boundary=true,
    require_closed=false,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=true,
    strict_nonmanifold=true,
)
mesh_ok = repair.mesh

# 2. Build RWG to count unknowns
rwg = build_rwg(mesh_ok; precheck=true, allow_boundary=true)
N = rwg.nedges
println("RWG unknowns: $N")

# 3. Estimate memory
mem_gib = estimate_dense_matrix_gib(N)
println("Raw matrix estimate: $mem_gib GiB")
println("Realistic budget: $(round(3 * mem_gib, digits=2)) GiB (3× factor)")

# 4. Mesh quality report
report = mesh_quality_report(mesh_ok)
println("Mesh quality summary:")
println("  boundary edges: $(report.n_boundary_edges)")
println("  non‑manifold edges: $(report.n_nonmanifold_edges)")
println("  orientation conflicts: $(report.n_orientation_conflicts)")

# 5. Decide: proceed, coarsen, or abort
if mem_gib > 4.0   # e.g., more than 4 GiB raw matrix
    println("Problem too large for direct solve; apply coarsening.")
    target_rwg = 800   # adjust based on available memory
    coarse_result = coarsen_mesh_to_target_rwg(mesh_ok, target_rwg; max_iters=10)
    mesh_ok = coarse_result.mesh
    rwg = build_rwg(mesh_ok; precheck=true, allow_boundary=true)
    N = rwg.nedges
    println("Coarsened to $N unknowns")
end
```

### Tiered Acceptance Test for Coarsened Results

When coarsening is applied, adopt a tiered validation strategy:

1. **Mesh‑quality tier**: `mesh_quality_ok` passes with your chosen tolerances.
2. **Solver‑residual tier**: Relative residual `norm(Z*I - v)/norm(v)` < 1e‑8.
3. **Internal‑consistency tier**: Far‑field power conservation holds within
   a few percent (check `energy_ratio` function in `src/postprocessing/Diagnostics.jl`).
4. **Engineering‑observable tier**: Key metrics (main lobe, monostatic RCS,
   trend vs. parameter) are stable across coarsening levels.
5. **External‑validation tier** (optional): Compare with a high‑fidelity
   reference on a small subset of scenarios.

This tiered approach makes the cost‑vs‑fidelity trade‑off explicit and
reproducible.

---

## 3) Cost‑Control Lever 1: Geometry Coarsening

For complex CAD/OBJ platforms, the most effective lever is **geometry coarsening**,
which reduces the number of RWG unknowns while preserving the overall shape.
The package provides an automated routine:

```julia
coarse_result = coarsen_mesh_to_target_rwg(mesh, target_rwg; max_iters=10, ...)
```

### How Coarsening Works

The algorithm (`src/geometry/Mesh.jl:461`) iterates:

1. **Vertex clustering**: Voxelize the bounding box with cell size $h$ and
   replace all vertices within a cell by their centroid.
2. **Non‑manifold removal**: Drop triangles that share an edge with more than
   two triangles (topologically invalid for RWG).
3. **Repair**: Fix triangle orientation, remove degenerate/invalid triangles.
4. **RWG recount**: Build RWG on the candidate mesh and compare to `target_rwg`.
5. **Adaptive $h$ adjustment**: If the count is outside ±15% of target,
   scale $h$ by $(N_{\text{current}} / N_{\text{target}})^{1/3}$ and repeat.

The loop stops when the count is within tolerance or `max_iters` is reached.

### Choosing `target_rwg`

Start from your available memory budget. For example, if you have 16 GiB RAM
and want to keep the raw matrix below 4 GiB:

```julia
max_mem_gib = 4.0
max_N = floor(Int, sqrt(max_mem_gib * 1024^3 / 16))
target_rwg = clamp(max_N, 200, 800)   # keep within reasonable range
```

**Typical guidelines:**

- **First pass**: 200–400 unknowns (fast, good for trend discovery).
- **Production run**: 500–800 unknowns (balance accuracy/time).
- **Validation run**: 1000–1500 unknowns (only if memory permits).

### Coarsening Trade‑Off Analysis

Coarsening reduces geometric detail, which affects:

- **High‑frequency features**: Small curvatures, sharp edges, and fine
  protrusions are smoothed or eliminated.
- **Electrical size**: The effective electrical size ($ka$) may shift slightly
  because the bounding‑box dimensions change.
- **Mesh quality**: The algorithm enforces manifoldness and orientation, often
  improving simulation robustness.

**Recommendation**: Always save the original, repaired, and coarsened meshes
(e.g., as `.obj` files) to make the fidelity reduction fully reproducible.

### Example: Airplane RCS with Coarsening

The script `examples/06_aircraft_rcs.jl` demonstrates the complete workflow:

```bash
julia --project=. examples/06_aircraft_rcs.jl
```

This runs the bundled aircraft demo mesh through repair, optional coarsening,
and a PEC RCS solve. It prints solve diagnostics and bistatic/monostatic
statistics to the console.

---

## 4) Cost‑Control Lever 2: Scenario Simplification

Even with a fixed mesh, total runtime scales linearly with the number of
scenarios. Common multipliers are:

- **Multiple frequencies** (broadband sweeps).
- **Multiple incidence angles** (monostatic/radar cross‑section maps).
- **Dense angular sampling** (bistatic pattern cuts).
- **Optimization iterations** (gradient‑based design loops).

### Staged Study Design

Instead of running all scenarios at high resolution, adopt a **two‑stage
screening** strategy:

```julia
# Stage 1: Coarse screening
freqs_coarse = range(1e9, 10e9, length=5)           # 5 frequencies
angles_coarse = range(0, 180, length=19)            # 19 azimuth angles
results_coarse = run_sweep(mesh, rwg, freqs_coarse, angles_coarse)

# Identify interesting regions
interesting_freqs = filter(f -> results_coarse[f].peak > threshold, freqs_coarse)
interesting_angles = filter(θ -> results_coarse[θ].null < threshold, angles_coarse)

# Stage 2: Refined study only on interesting subsets
freqs_fine = refine_range(interesting_freqs, 10)     # 10 points per region
angles_fine = refine_range(interesting_angles, 10)
results_fine = run_sweep(mesh, rwg, freqs_fine, angles_fine)
```

This reduces the total number of solves by orders of magnitude while preserving
accuracy where it matters.

### Far‑Field Sampling Strategy

The function `make_sph_grid(Nθ, Nφ)` creates a spherical grid with $N_\theta$
polar and $N_\varphi$ azimuthal samples. The total number of far‑field points is
$N_\Omega = N_\theta \times N_\varphi$. For pattern cuts, you can use a reduced
grid:

```julia
# Full sphere for 3D pattern (expensive)
grid_full = make_sph_grid(181, 361)   # 65,341 points

# Single φ‑cut for 2D pattern (cheap)
phi_fixed = 0.0
theta_range = range(0, π, length=181)
grid_cut = make_sph_grid(theta_range, [phi_fixed])   # 181 points
```

### Optimization‑Specific Simplifications

For gradient‑based optimization (`src/optimization/Optimize.jl`):

- Start with a **coarse mesh** (low $N$) for initial exploration.
- Use **coarse far‑field grid** (e.g., $N_\theta=31$, $N_\varphi=36$) for
  objective evaluation.
- Enable **preconditioning** (`:auto` with `iterative_solver=true`) to improve
  convergence of the adjoint solves.
- Limit the number of L‑BFGS iterations (`maxiter=20–50`) and check for
  stagnation.

The example `examples/05_solver_methods.jl` shows recommended settings
for large iterative runs.

---

## 5) Preconditioning: What It Does and Does Not Do

The package supports optional **mass‑based left preconditioning** with three
modes: `:off`, `:on`, `:auto`, plus the ability to supply a custom preconditioner
matrix.

### Mathematical Role of Preconditioning

The EFIE matrix $\mathbf{Z}$ can be ill‑conditioned for electrically large
structures or dense discretizations. Left preconditioning solves

```math
\mathbf{M}^{-1} \mathbf{Z} \,\mathbf{I} = \mathbf{M}^{-1} \mathbf{v},
```

where $\mathbf{M}$ is a preconditioner designed to approximate $\mathbf{Z}^{-1}$.
The package constructs $\mathbf{M}$ from the patch‑mass matrices
$\mathbf{M}_p$ (see `src/solver/Solve.jl:67`):

```math
\mathbf{M} = \sum_p \mathbf{M}_p + \epsilon \mathbf{I},
```

with a small regularization $\epsilon$ (default $10^{-8}$ relative to the
diagonal norm). This effectively normalizes the rows of $\mathbf{Z}$ and
improves the condition number.

### What Preconditioning **Does**

- **Improves numerical conditioning**: Reduces the risk of large residual errors
  from finite‑precision arithmetic.
- **Accelerates iterative solvers**: The package includes GMRES via Krylov.jl
  (`solve_gmres`), and preconditioning cuts the iteration count. A near-field
  sparse preconditioner is available via `build_nearfield_preconditioner`.
- **Stabilizes adjoint gradients**: Better‑conditioned forward solves lead to
  more accurate gradient computations in optimization.

### What Preconditioning **Does Not** Do

- **Does not reduce memory**: $\mathbf{M}$ is a dense matrix (sum of patch mass
  matrices plus regularization), and $\mathbf{Z}$ remains dense $O(N^2)$.
- **Does not change scaling**: Factorizing $\mathbf{Z}$ still costs $O(N^3)$.
- **Does not enable “large‑scale” dense solves**: The fundamental scaling wall
  remains.

### Auto‑Mode Logic

The `:auto` mode (`src/solver/Solve.jl:106`) decides whether to apply preconditioning
based on:

1. **User override**: If `preconditioner_M` is provided, use it.
2. **Explicit mode**: `:on` always builds $\mathbf{M}$; `:off` never does.
3. **Auto rules**:
   - If `iterative_solver=true` (recommended for large runs), enable.
   - If $N \ge$ `n_threshold` (default 256), enable.
   - Otherwise, disable.

The example `examples/05_solver_methods.jl` demonstrates the decision
logic and provides recommended settings for large/iterative workflows.

### Connection to the Paper

The mass‑based preconditioner corresponds to the diagonal scaling discussed in
the paper `bare_jrnl.tex` (Section 2.3) as a means to improve iterative
convergence. In the current dense‑direct context, it serves as a safeguard
against poor conditioning.

---

## 6) Implementation Boundaries and Transition to Fast Methods

### Current Envelope

The package supports both **dense direct solves** and **accelerated methods**.
The reliable workflow for large geometry is:

1. **Mesh repair** (`repair_mesh_for_simulation`).
2. **Controlled coarsening** (`coarsen_mesh_to_target_rwg`).
3. **Solve**: dense direct (`assemble_Z_efie` + `solve_forward`), dense GMRES
   (`solve_gmres` with `build_nearfield_preconditioner`), or ACA-compressed
   GMRES (`build_aca_operator`). The high-level `solve_scattering` function
   selects the method automatically based on problem size.
4. **Strong validation diagnostics** (residual, power conservation, etc.).

**Practical limits** with typical desktop hardware (32 GiB RAM, 8 cores):

- **Maximum unknowns**: ≈10,000–15,000 (raw matrix 1.6–3.6 GiB, realistic
  memory 5–11 GiB).
- **Maximum electrical size**: $ka \approx 20$–30 for closed PEC objects
  (depending on discretization density).
- **Typical production runs**: 500–2,000 unknowns, completing in seconds to
  minutes.

### When to Consider Fast‑Method Extensions

If your project requires:

- **>15,000 unknowns** (e.g., detailed vehicle/aircraft at high frequency),
- **>50 wavelengths electrical size**,
- **Many right‑hand sides** (hundreds of incidence angles),
- **Inverse design with >10,000 design variables**,

then use the package's built-in accelerated methods:

- **ACA H-matrix compression**: `build_aca_operator` constructs an ACA-compressed
  operator with $O(N \log^2 N)$ matvec cost, avoiding full dense assembly.
- **GMRES iterative solver**: `solve_gmres` (via Krylov.jl) with left or right
  preconditioning.
- **Near-field preconditioner**: `build_nearfield_preconditioner(Z, mesh, rwg, cutoff)`
  substantially reduces iteration counts and often shows weak growth with N in
  tested ranges.
- **Automatic method selection**: `solve_scattering` picks dense direct ($N \le 2000$),
  dense GMRES ($N \le 10000$), or ACA GMRES ($N > 10000$) automatically.

For problems exceeding the ACA envelope, full FMM/MLFMM methods would be the
next algorithmic step.

### Planning Your Project

| Requirement | Recommended Approach |
|-------------|----------------------|
| Validation / research reproducibility | Use dense direct solves with $N \le 5,000$. |
| Parameter sweeps on moderate platforms | Coarsen to $N \le 1,500$, stage scenarios. |
| Inverse design on small‑medium domains | Use dense + adjoint, $N \le 2,500$, preconditioning `:auto`. |
| Large platforms / high frequency | ACA GMRES via `build_aca_operator` + `solve_scattering`, or coarsen further. |

---

## 7) Complete Workflow Example

### For Complex OBJ Platforms

```bash
julia --project=. examples/06_aircraft_rcs.jl
```

This command runs the bundled demo pipeline end-to-end: load OBJ, repair,
coarsen if needed, solve, and report RCS statistics.

### For Preconditioning Tuning

```bash
julia --project=. examples/05_solver_methods.jl
```

This script demonstrates the `:auto` decision logic and shows recommended
settings for large/iterative optimization runs.

### Custom Workflow Script Template

For your own projects, adapt the following template:

```julia
using DiffMoM

function my_large_simulation(obj_path, freq_ghz, scale_to_m, target_rwg)
    # 1. Load and repair
    mesh = read_obj_mesh(obj_path)
    mesh_scaled = TriMesh(mesh.xyz .* scale_to_m, copy(mesh.tri))
    repair = repair_mesh_for_simulation(mesh_scaled; ...)
    mesh_ok = repair.mesh
    
    # 2. Coarsen if needed
    rwg = build_rwg(mesh_ok; precheck=true, allow_boundary=true)
    if rwg.nedges > target_rwg
        coarse = coarsen_mesh_to_target_rwg(mesh_ok, target_rwg; max_iters=10)
        mesh_ok = coarse.mesh
        rwg = build_rwg(mesh_ok; precheck=true, allow_boundary=true)
    end
    
    # 3. Estimate memory and warn
    mem_gib = estimate_dense_matrix_gib(rwg.nedges)
    @info "Memory estimate: $mem_gib GiB raw, $(3*mem_gib) GiB realistic"
    
    # 4. Solve
    k = 2π * freq_ghz*1e9 / 299792458.0
    Z = assemble_Z_efie(mesh_ok, rwg, k; quad_order=3)
    v = assemble_v_plane_wave(mesh_ok, rwg, ...)
    I = solve_forward(Z, v)
    
    # 5. Validate
    residual = norm(Z*I - v) / norm(v)
    @assert residual < 1e-8 "Solver residual too large: $residual"
    
    # 6. Save artifacts
    write_obj_mesh("repaired.obj", mesh_ok)
    # ... save results
end
```

---

## 8) Troubleshooting Common Large‑Problem Issues

### Memory Exhaustion During Assembly

**Symptom**: Julia crashes or throws `OutOfMemoryError` when calling
`assemble_Z_efie`.

**Diagnosis**:
- Check `rwg.nedges` and compute `estimate_dense_matrix_gib(N)`.
- If raw matrix > 25% of available RAM, factorizations will likely exceed memory.

**Remedy**:
- Coarsen mesh to lower $N$.
- Reduce quadrature order (`quad_order=2` instead of `3`).
- Close other memory‑hungry applications.

### Poor Solver Residual After Coarsening

**Symptom**: `norm(Z*I - v)/norm(v)` is large (e.g., > 1e‑6).

**Possible causes**:
1. Mesh quality degraded after coarsening (non‑manifold edges, degenerate triangles).
2. Electrical size changed significantly (coarsening altered bounding box).
3. Ill‑conditioning due to low‑frequency breakdown (if $ka \ll 1$).

**Diagnostic steps**:
```julia
report = mesh_quality_report(mesh)
println(report)
# Check boundary edges, non‑manifold edges, orientation conflicts
```

**Remedy**:
- Increase `area_tol_rel` in `coarsen_mesh_to_target_rwg` (default 1e‑12).
- Enable preconditioning (`:on` or `:auto`).
- If $ka < 0.5$, consider using a combined field integral equation (CFIE)
  (not yet implemented in this package).

### Far‑Field Pattern Artifacts

**Symptom**: Unphysical spikes, deep nulls, or asymmetry in the far‑field pattern.

**Diagnosis**:
- Usually caused by mesh defects (holes, non‑manifold edges) that create
  spurious currents.
- Could also be due to insufficient quadrature for near‑singular integrals.

**Remedy**:
- Run `repair_mesh_for_simulation` with `strict_nonmanifold=true`.
- Increase quadrature order (`quad_order=4`) in `assemble_Z_efie` and
  `radiation_vectors`.
- Verify power conservation: total radiated power vs. incident power.

### Slow Assembly Time

**Symptom**: `assemble_Z_efie` takes much longer than expected.

**Diagnosis**:
- Assembly scales as $O(N^2)$ with quadrature cost.
- High‑order quadrature (`quad_order=5`) increases constant factor.

**Remedy**:
- Use `quad_order=3` (default) unless high accuracy is required.
- For large $N$, consider pre‑computing singularity‑extraction tables
  (future optimization).

### Optimization Fails to Converge

**Symptom**: L‑BFGS iterations oscillate or stagnate.

**Diagnosis**:
- Gradient inaccuracy due to poor conditioning.
- Objective function too sensitive to small mesh changes.

**Remedy**:
- Enable preconditioning (`:auto` with `iterative_solver=true`).
- Increase `auto_precondition_eps_rel` (default 1e‑6) to 1e‑5.
- Reduce design‑variable count (coarsen parameterization).
- Use a smaller step size (`alpha0=1e‑3`).

---

## 9) Code Mapping

| Functionality | Source File | Key Functions / Lines |
|---------------|-------------|-----------------------|
| Memory estimation | `src/geometry/Mesh.jl` | `estimate_dense_matrix_gib` (line 328) |
| Mesh coarsening | `src/geometry/Mesh.jl` | `coarsen_mesh_to_target_rwg` (line 461) |
| Mesh repair | `src/geometry/Mesh.jl` | `repair_mesh_for_simulation` (line …) |
| Preconditioner construction | `src/solver/Solve.jl` | `make_left_preconditioner` (line 67), `select_preconditioner` (line 106) |
| Conditioned system solve | `src/solver/Solve.jl` | `prepare_conditioned_system` (line 179) |
| Optimization with preconditioning | `src/optimization/Optimize.jl` | `optimize_lbfgs`, `optimize_directivity` |
| Large OBJ demo | `examples/06_aircraft_rcs.jl` | Full workflow with repair, coarsen, solve, RCS |
| Auto‑preconditioning example | `examples/05_solver_methods.jl` | Decision logic and recommended settings |

---

## 10) Exercises

### Basic

1. **Memory estimation**: Pick three `target_rwg` values (200, 500, 1000) and
   compute `estimate_dense_matrix_gib(N)` for each. Convert the results to MiB
   and GiB.
2. **Coarsening effect**: Load a simple plate mesh (`make_rect_plate`), coarsen
   it with `target_rwg=50`, and compare the number of vertices, triangles, and
   RWG edges before/after.
3. **Preconditioning logic**: Run `examples/05_solver_methods.jl` and
   explain why each of the three test cases enables or disables preconditioning.

### Practical

4. **Two‑level RCS comparison**: Run `06_aircraft_rcs.jl` with `target_rwg=200`
   and `target_rwg=500`. Compare the monostatic RCS (dBsm), solver residual,
   and total runtime. Are the results qualitatively similar?
5. **Mesh‑quality audit**: Take a complex OBJ (e.g., from the `data/` directory),
   repair it with `repair_mesh_for_simulation`, and generate a full
   `mesh_quality_report`. Identify any remaining boundary edges or non‑manifold
   edges.
6. **Staged frequency sweep**: Implement a two‑stage sweep from 1 GHz to 10 GHz.
   Use 5 coarse points, identify the frequency with highest monostatic RCS,
   then run a refined sweep with 10 points around that peak.

### Advanced

7. **Memory‑budget planner**: Write a function that, given available RAM (GiB)
   and a safety factor $f$, returns the maximum $N$ that can be solved directly.
   Test it with RAM = 8 GiB, $f = 0.3$ (i.e., use at most 30% of RAM for the
   raw matrix).
8. **Coarsening‑convergence study**: For a fixed OBJ, run coarsening with
   `target_rwg = 100, 200, 400, 800`. Plot monostatic RCS vs. $N$ and determine
   the point where the result stabilizes within 1 dB.
9. **Preconditioning impact**: Modify `05_solver_methods.jl` to solve a
    larger problem ($N \approx 2000$) with and without preconditioning. Compare
    the condition number (estimate via `cond(Z)`) and the solver residual.

---

## 11) Chapter Checklist

Before moving to the next chapter, verify you can:

- [ ] Compute memory requirements for a given RWG count using `estimate_dense_matrix_gib`.
- [ ] Explain why dense MoM scales as $O(N^2)$ memory and $O(N^3)$ solve time.
- [ ] Use `coarsen_mesh_to_target_rwg` to reduce a complex OBJ mesh to a target unknown count.
- [ ] Describe the trade‑offs introduced by geometry coarsening (loss of detail, possible electrical‑size shift).
- [ ] Design a two‑stage scenario sweep to reduce computational cost.
- [ ] Explain what preconditioning does and does not do for dense solves.
- [ ] Choose appropriate `preconditioning` settings (`:off`, `:on`, `:auto`) for a given problem.
- [ ] Diagnose and remedy common large‑problem issues (memory exhaustion, poor residual, pattern artifacts).
- [ ] State the practical limits of the current dense implementation and when to consider fast‑method extensions.

---

## 12) Further Reading

- **Paper `bare_jrnl.tex`**: Section 2.3 discusses diagonal preconditioning for iterative solvers; Section 3.1 outlines the dense EFIE discretization and its scaling properties.
- **Computational Electromagnetics Texts**:
  - *Field Computation by Moment Methods* (Harrington, 1993) – classic reference on MoM.
  - *Fast and Efficient Algorithms in Computational Electromagnetics* (Chew et al., 2001) – covers fast multipole and other scaling improvements.
- **Julia Performance Tips**: [Julia Manual: Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/) – guidance on memory management and efficient array usage.
- **Mesh Processing**: *Polygon Mesh Processing* (Botsch et al., 2010) – algorithms for mesh repair, coarsening, and quality metrics.

---
