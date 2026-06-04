# Debugging Playbook

## Purpose

Method‑of‑Moments (MoM) simulations can fail for many reasons: mesh defects,
unit mismatches, ill‑conditioned matrices, far‑field projection errors, or
gradient inconsistencies. Debugging without a systematic approach wastes hours
on false leads.

This chapter provides a **fixed diagnostic order** that moves from cheap,
high‑impact checks to expensive, detailed verifications. Following this playbook
lets you isolate root causes quickly and produce reproducible bug reports.

---

## Learning Goals

After this chapter, you should be able to:

1. Map common symptoms (RWG build failure, large residual, non‑physical patterns,
   inconsistent objectives, gradient mismatch) to likely root causes.
2. Execute the seven‑step triage order, applying the cheapest diagnostic first.
3. Use built‑in functions (`mesh_quality_report`, `condition_diagnostics`,
   `energy_ratio`, etc.) to quantify problems.
4. Produce a minimal reproducibility packet that captures all necessary
   information for bug reports or external help.
5. Avoid common debugging pitfalls (skipping steps, misinterpreting unit errors,
   comparing nulls instead of beam features).

---

## 1) One‑Page Triage Order

**Never skip steps.** Each step is cheap compared to the next, and skipping
often leads to false conclusions. Apply this order every time a simulation
behaves unexpectedly.

### Step 1: Mesh Sanity

**Check**: Topology, degeneracy, orientation, non‑manifold edges.

**Tools**:
```julia
report = mesh_quality_report(mesh; area_tol_rel=1e-12, check_orientation=true)
mesh_quality_ok(report; allow_boundary=true, require_closed=false)
```

**Action**: If `mesh_quality_ok` returns `false`, repair with
`repair_mesh_for_simulation`. Save the repaired mesh for reproducibility.

### Step 2: Units Sanity

**Check**: Geometry scale vs. wavelength.

**Tools**:
```julia
λ = c0 / freq
L = _bbox_diagonal(mesh)   # or compute characteristic length
println("Electrical size L/λ = $(L/λ)")
```

**Action**: If `L/λ` is orders of magnitude off your intended regime (e.g.,
0.001 instead of 10), you likely have a unit mismatch (mm vs m, GHz vs Hz).
Fix scaling before proceeding.

### Step 3: Linear‑System Sanity

**Check**: Solver residual and matrix conditioning.

**Tools**:
```julia
r_rel = norm(Z * I - v) / max(norm(v), 1e-30)
diag = condition_diagnostics(Z)
```

**Action**: If `r_rel > 1e-8`, enable preconditioning (`:auto` or `:on`) or
regularization. If condition number > 1e12, consider coarsening the mesh or
adjusting the electrical size.

### Step 4: Far‑Field Sanity

**Check**: Transversality (`r̂·E ≈ 0`) and power conservation.

**Tools**:
```julia
transversality_error = maximum(abs.(sum(rhat .* E_ff, dims=1)))
rho = energy_ratio(I, v, E_ff, grid)
```

**Action**: Transversality error should be < 1e‑10; `rho` should be ≈1 for
lossless PEC. Failures indicate far‑field assembly or quadrature issues.

### Step 5: Objective Sanity

**Check**: Consistency between quadratic form `I†Q I` and direct angular
integration.

**Tools**:
```julia
quad_val = real(dot(I, Q * I))
direct_val = sum(grid.w[q] * abs2(dot(pol_vec[:,q], E_ff[:,q])) for q in 1:NΩ)
rel_err = abs(quad_val - direct_val) / max(abs(quad_val), 1e-30)
```

**Action**: If `rel_err > 1e-10`, debug `Q` construction (`build_Q`) and
far‑field projection.

### Step 6: Gradient Sanity

**Check**: Adjoint gradient matches finite‑difference gradient.

**Tools**:
```julia
I = solve_forward(Z, v)
lambda = solve_adjoint(Z, Q, I)
grad_adj = gradient_impedance(Mp, I, lambda; reactive=true)
grad_fd = [fd_grad(p -> objective_with_theta(p), θ0, p; h=1e-6) for p in eachindex(θ0)]
max_rel_err = maximum(abs.(grad_adj - grad_fd) ./ max.(abs.(grad_adj), 1e-12))
```

**Action**: If `max_rel_err > 1e-6`, verify that the same preconditioned
operators are used in forward and adjoint solves, and that derivative blocks
(`∂Z/∂θ`) are transformed consistently.

### Step 7: External Comparison

**Only after steps 1‑6 pass**, compare against an external reference (another
solver, measurement, analytic solution). Compare beam‑centric features (main
lobe, sidelobes) first, not null regions.

**Why this order works**: Each step depends on the correctness of the previous
ones. A unit mistake can cause terrible conditioning, which then breaks gradient
accuracy. Fixing the unit mistake solves both downstream issues.

---

## 2) Symptom → Likely Cause Map

### A) RWG Build or Assembly Fails Immediately

**Typical error messages**: “Triangle index out of bounds”, “degenerate triangle”,
“non‑manifold edge”, “orientation conflict”.

**Likely causes**:
- Invalid triangles (vertex index out of range).
- Degenerate triangles (area below `area_tol_rel`).
- Non‑manifold edges (more than two triangles sharing an edge).
- Orientation conflicts (adjacent triangles with inconsistent normal directions).

**Diagnostic steps**:

1. **Run mesh quality report**:
   ```julia
   report = mesh_quality_report(mesh; area_tol_rel=1e-12, check_orientation=true)
   println(report)
   ```

2. **Check if mesh passes simulation requirements**:
   ```julia
   ok = mesh_quality_ok(report; allow_boundary=true, require_closed=false)
   ```

3. **If `ok` is false**, repair the mesh:
   ```julia
   fixed = repair_mesh_for_simulation(
       mesh;
       allow_boundary=true,
       require_closed=false,
       drop_invalid=true,
       drop_degenerate=true,
       fix_orientation=true,
       strict_nonmanifold=true,
   )
   mesh = fixed.mesh
   ```

4. **Re‑run RWG build**:
   ```julia
   rwg = build_rwg(mesh; precheck=true, allow_boundary=true)
   ```

**Source functions**: `src/geometry/Mesh.jl` (`mesh_quality_report`, `repair_mesh_for_simulation`),
`src/basis/RWG.jl` (`build_rwg`).

---

### B) Solve Completes but Residual Is Large

**Typical symptom**: `norm(Z*I - v)/norm(v)` > 1e‑6 (or even > 1e‑2).

**Likely causes**:
- Ill‑conditioned matrix (low‑frequency breakdown, dense discretization).
- Unit mismatch causing extreme electrical size (`L/λ` ≪ 1 or ≫ 1000).
- Corrupted assembly inputs (wrong `k`, `eta0`, quadrature order).

**Diagnostic steps**:

1. **Check electrical size** (see Step 2 of triage order).
2. **Compute residual**:
   ```julia
   r_rel = norm(Z * I - v) / max(norm(v), 1e-30)
   ```
3. **Inspect condition diagnostics**:
   ```julia
   diag = condition_diagnostics(Z)
   println("Condition number estimate: $(diag.cond)")
   ```
4. **Enable preconditioning**:
   ```julia
   I_gmres = solve_forward(Z, v; solver=:gmres)
   r_rel_gmres = norm(Z * I_gmres - v) / norm(v)
   ```
   If switching to GMRES with a preconditioner reduces residual, conditioning was the issue.

5. **Verify assembly parameters**: Double‑check `k`, `eta0`, `quad_order`, and
   mesh scaling.

**Source functions**: `src/solver/Solve.jl` (`solve_forward`),
`src/postprocessing/Diagnostics.jl` (`condition_diagnostics`),
`src/assembly/EFIE.jl` (`assemble_Z_efie`).

---

### C) Far‑Field Pattern Looks Non‑Physical

**Typical symptoms**: Spikes, deep nulls, asymmetry, non‑zero radial component.

**Likely causes**:
- Far‑field vector assembly mismatch (`radiation_vectors` vs `compute_farfield`).
- Polarization projector misalignment (`pol_linear_x` vs `pol_linear_y`).
- Grid normalization error (weights `grid.w` not matching solid‑angle elements).

**Diagnostic steps**:

1. **Check transversality**:
   ```julia
   rhat = grid.rhat  # shape (3, NΩ)
   E_ff = compute_farfield(G_mat, I, NΩ)
   radial = sum(rhat .* E_ff, dims=1)
   max_radial = maximum(abs.(radial))
   ```
   Should be < 1e‑10.

2. **Check power conservation**:
   ```julia
   rho = energy_ratio(I, v, E_ff, grid)
   ```
   For lossless PEC, `rho` should be ≈1 (within 1 %).

3. **Compare with direct integration**:
   ```julia
   P_ff = sum(grid.w[q] * real(dot(E_ff[:,q], E_ff[:,q])) for q in 1:NΩ)
   P_inc = input_power(I, v)   # compute from MoM currents and excitation
   ```
4. **Inspect polarization vectors**: Ensure `pol_mat` matches intended
   polarization (e.g., `pol_linear_x` for x‑polarized far field).

**Source functions**: `src/postprocessing/FarField.jl` (`radiation_vectors`, `compute_farfield`),
`src/postprocessing/Diagnostics.jl` (`energy_ratio`).

---

### D) Objective Values Seem Inconsistent

**Typical symptom**: `I†Q I` differs from direct angular summation by more than
1e‑10 relative error.

**Likely causes**:
- `Q` construction mismatch (`build_Q` with wrong `mask` or `pol_mat`).
- Far‑field projection `G_mat` not aligned with `Q`’s internal `G`.
- Weight vector `grid.w` not correctly accounted for in `Q`.

**Diagnostic steps**:

1. **Compute both values**:
   ```julia
   quad_val = real(dot(I, Q * I))
   direct_val = sum(grid.w[q] * abs2(dot(pol_vec[:,q], E_ff[:,q])) for q in 1:NΩ)
   rel_err = abs(quad_val - direct_val) / max(abs(quad_val), 1e-30)
   ```

2. **If `rel_err` large**, verify `Q` construction:
   ```julia
   Q2 = build_Q(G_mat, grid, pol_mat; mask=mask)
   norm(Q - Q2)   # should be tiny
   ```

3. **Check mask alignment**: Ensure `mask` is a `BitVector` of length `NΩ` and
   matches the intended angular region.

4. **Check polarization matrix**: `pol_mat` should have shape `(3, NΩ)` and be
   orthogonal to `rhat`.

**Source functions**: `src/optimization/QMatrix.jl` (`build_Q`), `src/postprocessing/FarField.jl` (`pol_linear_x`, …).

---

### E) Adjoint Gradient Does Not Match Finite Difference

**Typical symptom**: Relative error between adjoint gradient and finite‑difference
gradient > 1e‑6.

**Likely causes**:
- Inconsistent preconditioning between forward and adjoint solves.
- Derivative blocks (`∂Z/∂θ`) not transformed under left preconditioning.
- Poor finite‑difference step (`h` too large or too small).
- Complex‑step artifacts (if using complex‑step differentiation with conjugation).

**Diagnostic steps**:

1. **Use central finite differences**:
   ```julia
   function fd_gradient(f, θ0; h=1e-6)
       grad = similar(θ0)
       for i in eachindex(θ0)
           θp = copy(θ0); θp[i] += h
           θm = copy(θ0); θm[i] -= h
           grad[i] = (f(θp) - f(θm)) / (2h)
       end
       return grad
   end
   ```

2. **Ensure same preconditioned operator**:
   - Forward solve uses `M⁻¹ Z` and `M⁻¹ v`.
   - Adjoint solve must use the same `M⁻¹` (via `prepare_conditioned_system`).

3. **Check derivative‑block transformation**:
   ```julia
   Mp_tilde, factor = transform_patch_matrices(Mp;
       preconditioner_M=M, preconditioner_factor=factor)
   ```
   Use `Mp_tilde` in the gradient computation.

4. **Verify gradient on a small problem** (`Nt ≤ 10`) where finite differences
   are cheap and reliable.

**Source functions**: `src/optimization/Adjoint.jl` (`solve_adjoint`, `gradient_impedance`),
`src/solver/Solve.jl` (`transform_patch_matrices`), `src/optimization/Verification.jl` (`fd_grad`).

---

## 3) Units and Scale Debugging (High‑Impact)

Unit mistakes are among the most common—and most subtle—sources of error.
A mesh designed in mm but interpreted as meters, or a frequency given in GHz
instead of Hz, can produce electrical sizes orders of magnitude off, leading to
catastrophic conditioning issues or nonsensical patterns.

### Quick Diagnostic

```julia
using DiffMoM

c0 = 299792458.0
freq = 3e9   # double‑check units: Hz, not GHz!
λ = c0 / freq

# Characteristic length: bounding-box diagonal
mins = [minimum(@view mesh.xyz[i, :]) for i in 1:3]
maxs = [maximum(@view mesh.xyz[i, :]) for i in 1:3]
L = sqrt(sum((maxs .- mins).^2))
println("Electrical size L/λ = $(L/λ)")
```

### Expected Regimes

- **Electrically small**: `L/λ < 0.1`. May cause low‑frequency breakdown
  (ill‑conditioned EFIE). Consider using a combined field integral equation
  (CFIE) if available.
- **Electrically moderate**: `0.1 ≤ L/λ ≤ 30`. Ideal for dense MoM.
- **Electrically large**: `L/λ > 30`. Dense MoM becomes expensive; consider
  coarsening or fast‑method extensions.

If `L/λ` is wildly different from your intended regime (e.g., 0.001 instead of
10), you likely have a **unit mismatch**.

### Common Unit Pitfalls

1. **CAD units**: Many CAD tools default to mm. The package expects **meters**.
   Scale the mesh: `mesh.xyz .*= 0.001`.
2. **Frequency**: Functions expect Hz, not GHz or MHz. Convert:
   `freq_Hz = freq_GHz * 1e9`.
3. **Wave number `k`**: `k = 2π / λ`. Ensure `λ` is in meters.
4. **Impedance `η₀`**: Default is 376.730313668 Ω (free space). Do not change
   unless simulating a different medium.

### Action Plan

If the electrical size is wrong:

1. **Scale the mesh** (if CAD units are mm):
   ```julia
   mesh_scaled = TriMesh(mesh.xyz .* 0.001, copy(mesh.tri))
   ```
2. **Re‑run preflight checks** (mesh quality, RWG count, memory estimate).
3. **Re‑assemble and solve** with corrected scaling.

**Do not proceed** with debugging conditioning, gradients, or pattern errors
until the electrical size is correct. Unit errors mask all downstream issues.

---

## 4) Preconditioning Debugging Rules

Left preconditioning (`M^{-1} Z I = M^{-1} v`) improves conditioning but
introduces extra complexity. When preconditioning is enabled, follow these rules
to avoid subtle errors.

### Rule 1: Same Conditioned Operator in Forward and Adjoint Solves

The forward solve uses `M^{-1} Z` and `M^{-1} v`. The adjoint solve must use the
**same** preconditioned operator, not the original `Z`. The package handles this
automatically when you pass `preconditioner_M` (or `solver=:gmres`) to
`compute_adjoint_gradient`. Verify by checking that the `factor` returned by
`prepare_conditioned_system` is reused.

### Rule 2: Transform Derivative Blocks Consistently

The derivative blocks `∂Z/∂θ` (stored in `Mp`) must be transformed under the same
preconditioner. Use `transform_patch_matrices`:

```julia
Mp_tilde, factor = transform_patch_matrices(Mp;
    preconditioner_M=M, preconditioner_factor=factor)
```

Then use `Mp_tilde` in gradient computations. Skipping this step will cause
gradient mismatch.

### Rule 3: Compare Against Unconditioned Results on a Small Benchmark

As a sanity check, run a small problem (`N ≤ 100`) with and without
preconditioning:

```julia
I1 = solve_forward(Z, v; solver=:direct)
I2 = solve_forward(Z, v; solver=:gmres)
rel_diff = norm(I1 - I2) / norm(I1)
```

The difference should be small (≲ 1e‑10). If it is large, preconditioner
construction may be flawed.

### Rule 4: Verify Solution Invariance

Left preconditioning should not change the solution (up to numerical rounding).
Solving `Z I = v` and `M^{-1} Z I = M^{-1} v` must give the same `I`. The package
ensures this by using a stable LU factorization of `M`. If you suspect
preconditioner issues, test with a diagonal `M` (e.g., `M = Diagonal(rand(N))`).

### Debugging Steps

1. **Check residual of preconditioned system**:
   ```julia
   I_pre = solve_forward(Z, v; solver=:gmres)
   r_pre = norm(Z * I_pre - v) / norm(v)
   ```
   Should be < 1e‑8.

2. **Compare gradients with finite differences** on a tiny problem, with
   preconditioning both on and off. Gradients should match within 1e‑6 relative
   error.

3. **Inspect the preconditioner matrix `M`**:
   ```julia
   M = make_left_preconditioner(Mp; eps_rel=1e-8)
   cond(M)   # should be moderate (≲ 1e6)
   ```

**Source functions**: `src/solver/Solve.jl` (`make_left_preconditioner`,
`prepare_conditioned_system`, `transform_patch_matrices`).

---

## 5) Cross‑Validation Debugging (After Internal Checks)

After all internal checks pass (mesh, units, linear system, far‑field, objective,
gradient), you may still observe discrepancies with an external reference:
another solver, measurement, or analytic solution. Before concluding the package
is wrong, systematically compare **conventions and feature alignment**.

### Step 1: Verify Convention Alignment

- **Time sign**: The package uses $\exp(-i\omega t)$ (engineering convention).
  If the reference uses $\exp(+i\omega t)$, fields will be complex conjugates.
- **Phase reference**: Check that the phase center (coordinate origin) matches.
- **Polarization definition**: Linear polarization along $x$ vs $y$, circular
  handedness.
- **Units**: Double‑check frequency, length, and impedance units.

### Step 2: Verify Geometry and Excitation Match

- **Mesh**: Are the triangles identical? Even small differences (vertex order,
  rounding) can affect results.
- **Excitation**: Plane‑wave direction, polarization vector, amplitude.
- **Frequency**: Exact same frequency (not just close).

### Step 3: Compare Beam‑Centric Features First

Do **not** start with aggregate error metrics (mean‑square difference over all
angles). Those are dominated by null regions, which are sensitive to tiny phase
shifts. Instead, compare:

- **Main‑beam direction** (peak angle).
- **Main‑beam gain** (directivity at peak).
- **First sidelobe level**.
- **Null locations** (only after main‑beam agreement).

### Step 4: Inspect Null‑Region Residuals Only After Beam Agreement

If the main beam matches but nulls differ, the discrepancy may be numerically
insignificant for your application (e.g., beam‑steering). Nulls are sensitive to
small errors in mesh, quadrature, and solver tolerance.

### Step 5: Use Normalized Error Metrics

When quantifying differences, use metrics that reflect your application:

- For beam‑steering: angular error of peak (degrees), gain difference at target (dB).
- For pattern matching: mean‑square error weighted by pattern power (not uniform).
- For RCS: dB difference at specified angles.

### Example Workflow

```julia
# Load external pattern (theta_ext, phi_ext, E_ext)
# Compute package pattern on the same angular grid
grid_ext = make_sph_grid(theta_ext, phi_ext)
E_pkg = compute_farfield(...)

# Compare main beam
idx_peak_ext = argmax(abs2.(E_ext))
idx_peak_pkg = argmax(abs2.(E_pkg))
angular_error = acos(clamp(dot(grid_ext.rhat[:,idx_peak_ext],
                               grid_ext.rhat[:,idx_peak_pkg]), -1.0, 1.0)) * 180/π

# Compare gain at target direction
gain_diff = 10*log10(abs2(E_pkg[target_idx]) / abs2(E_ext[target_idx]))
```

**Takeaway**: Internal consistency is necessary but not sufficient for external
agreement. Methodical comparison of conventions and beam features isolates true
discrepancies from irrelevant differences.

---

## 6) Minimal Reproducibility Packet (for Issues)

When you encounter a bug or unexpected behavior and need help (from the
development team, colleagues, or your future self), a **minimal reproducibility
packet** is essential. It should contain all the information needed to reproduce
the issue **exactly**, without requiring access to your local files or environment.

### Required Items

1. **Input mesh** (as an OBJ file) or a script that generates the mesh (e.g.,
   `make_rect_plate` call).
2. **Frequency, polarization, incidence definition** (code snippet).
3. **Mesh‑quality report** (output of `mesh_quality_report`).
4. **RWG unknown count `N` and memory estimate** (`estimate_dense_matrix_gib(N)`).
5. **Solver residual** (`norm(Z*I - v)/norm(v)`).
6. **Condition diagnostic** (output of `condition_diagnostics(Z)`).
7. **Energy ratio** (`energy_ratio(I, v, E_ff, grid)`).
8. **Exact commit hash** of the package (`git rev-parse HEAD`).
9. **Exact run command** (including Julia version and project activation).
10. **Generated CSV outputs** (if any) or a summary of the unexpected results.

### Optional but Helpful

- **Script that reproduces the issue** (standalone, ≤ 50 lines).
- **Plot of the unexpected pattern** (PNG or PDF).
- **Comparison with expected results** (if available).

### Example Packet

```
Issue: Far‑field pattern shows spurious spikes at θ ≈ 90°.

Reproducibility packet:

- Mesh: `bug_report_mesh.obj` (attached)
- Script:
  using DiffMoM
  mesh = read_obj_mesh("bug_report_mesh.obj")
  freq = 3e9
  k = 2π * freq / 299792458.0
  rwg = build_rwg(mesh)
  Z = assemble_Z_efie(mesh, rwg, k; quad_order=3)
  ...
- Mesh quality: boundary_edges=12, nonmanifold_edges=0, orientation_conflicts=0
- N = 1256, memory estimate = 0.03 GiB
- Residual = 2.4e-5
- Condition estimate = 1.2e8
- Energy ratio = 0.87
- Commit: a1b2c3d4e5f6
- Command: `julia --project=. bug_script.jl`
- Output CSV: `bug_output.csv`
```

### Why This Works

A complete packet lets the helper immediately run the same code, see the same
numbers, and focus on the root cause instead of asking for missing pieces. It
typically reduces debug time by an order of magnitude.

---

## 7) Recommended Commands

### Full Internal Regression

Run the entire test suite to verify the package is working correctly on your
system:

```bash
julia --project=. test/runtests.jl
```

This checks mesh utilities, EFIE assembly, forward/adjoint solves, far‑field
projection, and gradient consistency. Any failure indicates a broken installation
or environment mismatch.

### Mesh Repair Sanity Check

Run the built-in repair/coarsen workflow and inspect the reported changes:

```bash
julia --project=. examples/06_aircraft_rcs.jl
```

The script prints repair counters and the final mesh size used for solving.

### Large OBJ Smoke Run

Test the complete workflow (repair, coarsen, solve, RCS) on a complex platform:

```bash
julia --project=. examples/06_aircraft_rcs.jl
```

This command runs the bundled aircraft demo end-to-end (repair, coarsening,
solve, RCS diagnostics).

### Quick Preflight Check

For any mesh, run a lightweight preflight to estimate memory and detect obvious
issues:

```julia
using DiffMoM
mesh = read_obj_mesh("my_mesh.obj")
rwg = build_rwg(mesh; precheck=true)
println("RWG unknowns: $(rwg.nedges)")
println("Memory estimate: $(estimate_dense_matrix_gib(rwg.nedges)) GiB")
report = mesh_quality_report(mesh)
println(report)
```

### Gradient Verification

On a small problem, run the package test suite (which includes gradient gates):

```bash
julia --project=. test/runtests.jl
```

Or write a minimal verification script using `solve_adjoint`, `gradient_impedance`,
and `fd_grad` (see Exercise 8).

### Profiling and Benchmarking

To identify performance bottlenecks, run with Julia’s built‑in profiler:

```bash
julia --project=. --track-allocation=user examples/06_aircraft_rcs.jl
```

Or use `@time` inside your script to time assembly, solve, and far‑field steps.

---

## 8) Code Mapping

| Component | Source File | Key Functions |
|-----------|-------------|---------------|
| Mesh diagnostics, repair, coarsening | `src/geometry/Mesh.jl` | `mesh_quality_report`, `repair_mesh_for_simulation`, `coarsen_mesh_to_target_rwg`, `estimate_dense_matrix_gib` |
| EFIE assembly | `src/assembly/EFIE.jl`, `src/assembly/Excitation.jl` | `assemble_Z_efie`, `assemble_v_plane_wave` |
| Solving and conditioning | `src/solver/Solve.jl` | `solve_forward`, `make_left_preconditioner`, `prepare_conditioned_system` |
| Far‑field and Q matrices | `src/postprocessing/FarField.jl`, `src/optimization/QMatrix.jl` | `radiation_vectors`, `compute_farfield`, `build_Q` |
| Diagnostics | `src/postprocessing/Diagnostics.jl` | `energy_ratio`, `condition_diagnostics`, `bistatic_rcs`, `backscatter_rcs` |
| Adjoint gradient | `src/optimization/Adjoint.jl` | `solve_adjoint`, `gradient_impedance`, `compute_objective` |
| Gradient verification | `src/optimization/Verification.jl` | `fd_grad`, `complex_step_grad`, `verify_gradient` |
| End‑to‑end tests | `test/runtests.jl` | Test suite that exercises all components |

> Note: The exact location of adjoint and verification functions may vary; check
> the `src/` directory for the latest organization.

---

## 9) Exercises

### Basic

1. **Intentional mesh corruption**: Flip triangle winding on a subset of a simple
   plate mesh. Run `mesh_quality_report` and `mesh_quality_ok`. Which checks fail
   first? Does `build_rwg` succeed?
2. **Unit mismatch detection**: Scale a plate mesh by 1000 (simulating mm‑to‑m
   error). Compute `L/λ` and compare with the correct scaling. Which step in the
   triage order catches this mistake?
3. **Residual check**: Solve a small problem, then artificially perturb the
   solution vector `I`. Compute the relative residual `norm(Z*I - v)/norm(v)`.
   How large a perturbation raises the residual above 1e‑8?

### Practical

4. **Preconditioning comparison**: Solve a moderately sized problem (≈500
   unknowns) with `solver=:direct` and `solver=:gmres`. Compare residuals, solution
   vectors, and solve times. When does preconditioning help most?
5. **Far‑field transversality**: Compute the far field of a PEC plate and evaluate
   `max(abs.(rhat·E_ff))`. Is it below 1e‑10? If not, increase quadrature order
   and retry.
6. **Objective consistency**: For a given current `I`, compute `I†Q I` and the
   direct angular sum. Vary the angular grid resolution (`make_sph_grid(31,36)`
   vs `(181,361)`). Does the relative error change?
7. **Debug‑report template**: Create a Markdown template that includes all
   sections of the minimal reproducibility packet. Use it to document a simulated
   bug (e.g., the unit‑scale error from exercise 2).

### Advanced

8. **Gradient verification script**: Write a standalone script that, for a small
   plate mesh, computes the adjoint gradient of the beam‑steering objective and
   compares it with central finite differences. Report the maximum relative error.
9. **Condition‑number study**: For a fixed plate, vary the electrical size
   `L/λ` from 0.1 to 10 (by changing frequency). Compute the condition number
   estimate (`condition_diagnostics`) and plot vs. `L/λ`. Identify the regime
   where preconditioning becomes necessary.
 10. **Cross‑validation with analytic solution**: Compare the scattered field of a
     PEC sphere (Mie series) with the MoM solution at a few frequencies. Follow
     the cross‑validation steps (Section 5) and quantify the agreement.

---

## 10) Chapter Checklist

Before moving to the next chapter, verify you can:

- [ ] Recite the seven‑step triage order and explain why skipping steps leads to false conclusions.
- [ ] Map symptoms (RWG build failure, large residual, non‑physical pattern, inconsistent objective, gradient mismatch) to likely causes.
- [ ] Use `mesh_quality_report`, `condition_diagnostics`, and `energy_ratio` to quantify problems, and manually verify objective consistency by comparing `real(dot(I, Q * I))` against direct angular integration.
- [ ] Detect unit mistakes by computing `L/λ` and comparing with the intended regime.
- [ ] Apply the four preconditioning debugging rules to ensure gradient consistency.
- [ ] Follow the cross‑validation workflow when comparing with external references.
- [ ] Assemble a minimal reproducibility packet that captures all necessary information for a bug report.

---

## 11) Further Reading

- **Paper `bare_jrnl.tex`**: Section 2.3 discusses preconditioning; Section 3.2 covers far‑field projection and `Q` matrices; Appendix A may include verification benchmarks.
- **Debugging Numerical Software**: *The Science of Debugging* (Telles & Hsieh, 2001) – general principles that apply to MoM debugging.
- **Julia Debugging Tools**: `Debugger.jl`, `Infiltrator.jl`, and `JET.jl` can help track down type instabilities and logical errors.
- **MoM Validation Techniques**: *Accuracy Considerations for Integral Equation MoM* (Peterson et al., 1998) – discusses convergence tests, residual norms, and cross‑validation.

---
