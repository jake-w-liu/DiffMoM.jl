# Optimization Workflow

## Purpose

This chapter covers the inverse-design workflow from problem setup and
optimizer configuration through trace inspection and post-optimization
validation.

---

## Learning Goals

After this chapter, you should be able to:

1. Configure a reproducible optimization run with appropriate box constraints, conditioning options, and stopping criteria.
2. Interpret optimization traces (objective value $J$, gradient norm $\|\mathbf{g}\|$, iteration count) to diagnose convergence issues.
3. Apply practical safeguards: parameter initialization, bounds enforcement, regularization/preconditioning selection.
4. Validate optimization results through independent far‑field recomputation and sensitivity analysis.
5. Log the inputs, settings, and seeds needed to repeat a run.

---

## 1. Overview of the Optimization Loop

### 1.1 High‑Level Pipeline

The inverse‑design pipeline in `DiffMoM.jl` follows a standard projected L‑BFGS framework adapted to the EFIE–MoM context:

```
    Initialization
        ↓
    For k = 1, 2, … until convergence:
        ↓
    1. Assemble Z(θ⁽ᵏ⁾) = Z_EFIE + Z_imp(θ⁽ᵏ⁾)
        ↓
    2. Solve forward system Z(θ⁽ᵏ⁾) I⁽ᵏ⁾ = v
        ↓
    3. Evaluate objective J⁽ᵏ⁾ (quadratic or ratio)
        ↓
    4. Solve adjoint system(s) for λ⁽ᵏ⁾
        ↓
    5. Compute gradient g⁽ᵏ⁾ = ∂J/∂θ
        ↓
    6. Update θ via projected L‑BFGS with backtracking line search
        ↓
    7. Check stopping criteria → continue or exit
```

Each gradient evaluation requires one forward solve and one adjoint solve for a
quadratic objective, or two adjoint solves for a ratio objective. Backtracking
can add multiple forward trial solves before a step is accepted.

### 1.2 Two Optimizer Entry Points

The package provides two main optimizer functions, both built on the same L‑BFGS core:

- **`optimize_lbfgs`** – for a single quadratic objective $J = \mathbf{I}^\dagger \mathbf{Q} \mathbf{I}$.
- **`optimize_directivity`** – for a ratio objective $J = (\mathbf{I}^\dagger \mathbf{Q}_t \mathbf{I}) / (\mathbf{I}^\dagger \mathbf{Q}_{\mathrm{tot}} \mathbf{I})$.

Both support box constraints (`lb`, `ub`), reactive/resistive parameterization, and the full conditioning/preconditioning options described in Chapter 5.

---

## 2. Problem Setup and Configuration

### 2.1 Geometry and Discretization

1. **Mesh generation**: Create a surface mesh of the metasurface (e.g., rectangular plate, arbitrary shape from OBJ file). Ensure triangle quality; badly shaped elements can degrade solution accuracy.
2. **RWG basis**: Call `build_rwg(mesh)` to generate basis functions.
3. **Patch partition**: Define how the mesh is divided into design patches. A fine partition (one patch per triangle) gives maximum control but many parameters; a coarse partition reduces dimensionality at the cost of spatial resolution.
4. **Patch mass matrices**: Precompute $\mathbf{M}_p$ via `precompute_patch_mass`. This is a one‑time cost amortized over all optimization iterations.

### 2.2 Incident Field and EFIE Matrix

- **Incident field**: Assemble the right‑hand side $\mathbf{v}$ using `assemble_v_plane_wave` (or other incident‑field routines).
- **EFIE matrix**: Assemble $\mathbf{Z}_{\mathrm{EFIE}}$ via `assemble_Z_efie`. This matrix is independent of $\boldsymbol{\theta}$ and can be reused throughout the optimization.

### 2.3 Objective Matrices

- Build radiation vectors once: `G = radiation_vectors(mesh, rwg, grid, k)`.
- Use `build_Q(G, grid, pol; mask=...)` to construct:
  - $\mathbf{Q}_t$: target-region mask (e.g., cone around steering angle).
  - $\mathbf{Q}_{\mathrm{tot}}$: total-region mask (or `mask=nothing` for full sphere).
- Use `cap_mask` directly for broadside cones; for off-broadside targets,
  provide a custom Boolean mask over `grid.theta`/`grid.phi`.

### 2.4 Initial Parameter Vector

The initial guess $\boldsymbol{\theta}^{(0)}$ can significantly influence convergence:

- **Zero initialization** ($\theta_p = 0$): Starts from a PEC surface (no impedance). Often works for broadside steering but may stall for off‑broadside targets due to symmetry.
- **Phase‑ramp initialization**: For steering to angle $\theta_0$, set $\theta_p = \beta \cdot (\mathbf{r}_p \cdot \hat{\mathbf{k}}_0)$, where $\mathbf{r}_p$ is the patch centroid and $\hat{\mathbf{k}}_0$ is the desired wavevector direction. This mimics a linear phase gradient.
- **Random initialization**: Small random values can break symmetry and help escape poor local minima.
- **Previous solution**: Warm‑start from a previously optimized design for a similar target.

---

## 3. Optimizer Parameters and Their Effects

### 3.1 L‑BFGS‑Specific Parameters

| Parameter | What to inspect when tuning |
|-----------|-----------------------------|
| `maxiter` | Whether the trace reaches the iteration cap before another stopping condition. |
| `tol` | Repeatability of the final objective and observables as the absolute gradient-norm threshold changes. |
| `m_lbfgs` | Stored curvature history, runtime, and accepted-step progress. |

The line search uses backtracking Armijo and starts each search at step length
1.0. The `alpha0` keyword controls the initial inverse-Hessian scaling before
any curvature pairs have been accumulated. Read the current keyword defaults
from the `optimize_lbfgs` or `optimize_directivity` docstring.

### 3.2 Box Constraints (`lb`, `ub`)

Physical realizability often imposes bounds on impedance values:

- **Passive resistive sheets**: $\theta_p \ge 0$ (set `lb=0.0`, `ub=Inf`).
- **Reactive sheets**: $\theta_p \in [-X_{\max}, X_{\max}]$ where $X_{\max}$ is a practical reactance limit (e.g., 500 $\Omega$).
- **Active sheets**: May allow negative resistance ($\theta_p < 0$) for gain, but stability must be considered.

Box constraints are enforced via **projection**: after each L‑BFGS step, parameters are clipped to `[lb, ub]`. Pass `lb=nothing` and `ub=nothing` (the defaults) for unconstrained optimization.

### 3.3 Conditioning and Solver Options

- `solver`: `:direct` for LU factorization or `:gmres` for Krylov.jl; the choice applies to both forward and adjoint solves.
- `nf_preconditioner`: Near-field sparse preconditioner for GMRES. Build one with `build_nearfield_preconditioner(Z, mesh, rwg, cutoff)`.
- `gmres_tol`: GMRES convergence tolerance.
- `gmres_maxiter`: Maximum GMRES iterations.
- `preconditioning`: `:off`, `:on`, or `:auto` (see Chapter 5). In `:auto` mode, `select_preconditioner` uses `iterative_solver` and the configured unknown-count threshold.
- `iterative_solver`: A Boolean flag passed to `select_preconditioner` that influences auto‑preconditioning decisions. When `true`, the auto‑preconditioner logic accounts for the use of an iterative solver. Note: this does **not** switch the solver itself; use the `solver` keyword (`:direct` or `:gmres`) for that.
- `regularization_alpha`: Weight applied to the selected regularization matrix.

### 3.4 Verbosity and Tracing

- `verbose=true`: Print iteration progress (objective value, gradient norm).
- The trace is returned as the second element of `(theta_opt, trace)`.
  `optimize_directivity` records `(iter, J, gnorm)`; `optimize_lbfgs` also
  records cumulative forward- and adjoint-solve counts as `n_fwd` and `n_adj`.

---

## 4. Monitoring Convergence and Diagnosing Issues

### 4.1 What to Monitor

Every optimization trace contains these fields per iteration:

- **Iteration number** `iter`.
- **Objective value** `J`.
- **Gradient norm** `gnorm`.

Plotting `J` against iteration shows the accepted objective sequence.

### 4.2 Trace Checks

- **Objective direction**: Accepted iterates should reduce `J` for minimization
  or increase it for maximization.
- **Gradient threshold**: Compare the final `gnorm` directly with `tol`.
- **Solve work**: For `optimize_lbfgs`, differences in `n_fwd` expose the
  number of line-search trial solves between accepted iterates.

### 4.3 Common Convergence Problems and Remedies

| Symptom | Checks to run |
|---------|---------------|
| **Objective moves in the wrong direction** | Confirm `maximize`, recompute the objective from the returned parameters, and check the solve residuals. |
| **Gradient norm stops changing** | Compare the adjoint gradient with finite differences, inspect active bounds, and compare solver residuals. |
| **Many forward solves between iterates** | Inspect line-search diagnostics, scaling, active bounds, and the directional derivative. |
| **Run stops before the iteration cap with `gnorm > tol`** | Read the verbose line-search diagnostic and check whether projection prevented a distinct trial point or Armijo rejected every trial. |

### 4.4 Stopping Criteria

An optimizer run ends when any of the following occurs:

1. **Gradient norm criterion**: $\|\mathbf{g}\| \le \mathtt{tol}$.
2. **Iteration limit**: the `maxiter` loop is exhausted.
3. **Line-search failure**: no trial step satisfies the acceptance condition.

Treat the stopping path as one diagnostic. Verify the returned objective and
gradient, the underlying solve residual, and the target observable before
accepting the result.

---

## 5. Practical Run Template

### 5.1 Complete Script for Beam Steering

The following template shows the checked end-to-end workflow:

```julia
using DiffMoM
using Random

# ------------------------------
# 1. Seed for reproducibility
# ------------------------------
Random.seed!(12345)

# ------------------------------
# 2. Geometry and discretization
# ------------------------------
λ = 1.0
k = 2π / λ
mesh = make_rect_plate(4λ, 4λ, 40, 40)
rwg = build_rwg(mesh)

# ------------------------------
# 3. Patch partition (e.g., 10×10 patches)
# ------------------------------
ntri = ntriangles(mesh)
partition = PatchPartition(collect(1:ntri), ntri)  # one patch per triangle (simple runnable setup)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
P = length(Mp)

# ------------------------------
# 4. Incident field (broadside)
# ------------------------------
k_vec = Vec3(0.0, 0.0, -k)
pol_inc = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, pol_inc; quad_order=3)

# ------------------------------
# 5. EFIE matrix (parameter‑independent)
# ------------------------------
Z_efie = assemble_Z_efie(mesh, rwg, k)

# ------------------------------
# 6. Objective matrices
# ------------------------------
θ_steer = 30.0   # degrees
grid = make_sph_grid(64, 128)
G = radiation_vectors(mesh, rwg, grid, k)
pol = pol_linear_x(grid)
θ0 = deg2rad(θ_steer)
Δθ = deg2rad(10.0)
mask_target = abs.(grid.theta .- θ0) .<= Δθ
Q_target = build_Q(G, grid, pol; mask=mask_target)
Q_total = build_Q(G, grid, pol)   # full-sphere reference

# ------------------------------
# 7. Initial parameters (zero reactance → PEC reference)
# ------------------------------
theta0 = zeros(P)

# ------------------------------
# 8. Bounds (realistic reactance limits)
# ------------------------------
lb = fill(-500.0, P)
ub = fill( 500.0, P)

# ------------------------------
# 9. Run optimization
# ------------------------------
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta0;
    reactive=true,
    maxiter=100,
    tol=1e-6,
    m_lbfgs=10,
    lb=lb,
    ub=ub,
    verbose=true
)

# ------------------------------
# 10. Save results
# ------------------------------
using JLD2
@save "optimization_result.jld2" theta_opt trace mesh rwg partition
```

### 5.2 Logging for Reproducibility

Always record **all** input parameters and random seeds. A simple logging function:

```julia
function log_optimization_run(config, trace)
    open("run_$(Dates.now()).log", "w") do io
        println(io, "DiffMoM.jl optimization log")
        println(io, "=====================================")
        println(io, "Timestamp: ", now())
        println(io, "Git commit: ", read(`git rev-parse HEAD`, String))
        println(io, "\nConfiguration:")
        for (key, val) in config
            println(io, "  $key = $val")
        end
        println(io, "\nTrace summary:")
        println(io, "  Final J = ", trace[end].J)
        println(io, "  Final ‖g‖ = ", trace[end].gnorm)
        println(io, "  Iterations = ", length(trace))
    end
end
```

---

## 6. Post‑Optimization Validation

### 6.1 Independent Far‑Field Recalculation

Never trust the objective value from the optimization trace alone. Re‑assemble $\mathbf{Z}$ with `theta_opt`, solve for $\mathbf{I}$, and compute the far‑field pattern independently:

```julia
Z_opt = assemble_full_Z(Z_efie, Mp, theta_opt; reactive=true)
I_opt = solve_forward(Z_opt, v)

# Compute far-field using radiation vectors
G_mat = radiation_vectors(mesh, rwg, grid, k)
E_ff = compute_farfield(G_mat, I_opt, length(grid.w))

# Compute directivity ratio directly
f_opt = real(dot(I_opt, Q_target * I_opt))
g_opt = real(dot(I_opt, Q_total * I_opt))
J_opt = f_opt / g_opt
println("Recomputed J = ", J_opt, " (trace reported ", trace[end].J, ")")
```

Discrepancies $> 1\%$ may indicate numerical issues during optimization.

### 6.2 Perturbation Sensitivity

Test robustness by perturbing the optimized design:

1. **Frequency shift**: Evaluate $J$ at $f_0 \pm \Delta f$ (e.g., $\Delta f = 0.05 f_0$). A sharp drop indicates narrow‑band performance.
2. **Incidence angle variation**: Evaluate $J$ for slightly different incident angles. This tests beam‑steering stability.
3. **Parameter noise**: Add Gaussian noise to `theta_opt` (e.g., $\sigma = 0.01 \cdot \mathrm{range}(\theta)$) and recompute $J$. A sensitive design may degrade rapidly.

### 6.3 Visualization

- **Impedance distribution**: Plot `theta_opt` on the mesh surface using `plot_mesh_wireframe` or `plot_mesh_comparison` from `src/postprocessing/Visualization.jl`.
- **Far‑field pattern**: 2D cuts or 3D radiation plots.
- **Optimization trace**: $J$ vs. iteration, $\|\mathbf{g}\|$ vs. iteration.

### 6.4 Comparison with Baseline

Compare the optimized design against relevant baselines:

- **PEC reference** ($\boldsymbol{\theta} = \mathbf{0}$).
- **Uniform impedance** (constant $\theta$).
- **Phase‑gradient heuristic** (analytic linear ramp).

Quantify improvement in directivity, sidelobe suppression, or other metrics.

---

## 7. Advanced Workflow Considerations

### 7.1 Multi‑Start Optimization

To reduce the risk of poor local minima, run multiple optimizations from different initial guesses and select the best result:

```julia
n_restarts = 5
best_J = -Inf
best_theta = nothing
best_trace = nothing

for r in 1:n_restarts
    theta0_r = randn(P) .* 100.0   # random initial reactance
    theta_opt_r, trace_r = optimize_directivity(..., theta0_r, ...)
    J_r = trace_r[end].J
    if J_r > best_J
        best_J = J_r
        best_theta = theta_opt_r
        best_trace = trace_r
    end
end
```

### 7.2 Continuation Methods

For difficult problems (e.g., large steering angles), use a continuation strategy:

1. Optimize for a small steering angle (e.g., 10°).
2. Use the result as initial guess for a larger angle (e.g., 20°).
3. Repeat until target angle is reached.

This can avoid symmetry traps and improve convergence.

### 7.3 Constraint Handling Beyond Box Bounds

If additional constraints are needed (e.g., total power consumption, manufacturing constraints), consider:

- **Penalty method**: Add penalty terms to the objective.
- **Augmented Lagrangian**: Not supported by the package; use an external
  constrained optimizer if penalty terms are insufficient.
- **Filtering**: Post‑process `theta_opt` to enforce smoothness or discreteness.

### 7.4 Warm‑Starting Across Frequencies

When designing over a frequency band, optimize at the center frequency first, then use the solution as initial guess for neighboring frequencies. This exploits frequency smoothness.

---

## 8. Troubleshooting Checklist

Use this checklist when an optimization fails or behaves unexpectedly:

- [ ] **Mesh quality**: Are any triangles degenerate or excessively skewed?
- [ ] **Condition number**: Compute $\kappa(\mathbf{Z})$ for initial $\boldsymbol{\theta}$; if $>10^{10}$, enable regularization/preconditioning.
- [ ] **Gradient verification**: Run a finite‑difference check on the initial guess; relative error should be $<10^{-6}$.
- [ ] **Bounds feasibility**: Is the initial guess within `[lb, ub]`? Are bounds too restrictive?
- [ ] **Incident field**: Does `v` have reasonable magnitude? Is the polarization correct?
- [ ] **Objective matrices**: Do $\mathbf{Q}_t$ and $\mathbf{Q}_{\mathrm{tot}}$ have expected ranks? (Check `rank(Matrix(Q))`.)
- [ ] **Memory limits**: For large $P$, `m_lbfgs` may exceed available memory; reduce it.
- [ ] **Random seed**: For stochastic initialization, record the seed to reproduce results.
- [ ] **Verbose output**: Enable `verbose=true` to see iteration‑by‑iteration progress.

---

## 9. Code Mapping

- **`src/optimization/Optimize.jl`** – Core optimizers `optimize_lbfgs` and `optimize_directivity`.
- **`src/optimization/Adjoint.jl`** – Gradient computation (`gradient_impedance`) and adjoint solves.
- **`src/solver/Solve.jl`** – Forward solve (`solve_forward`) and conditioned system preparation.
- **`src/assembly/Impedance.jl`** – Patch mass matrices and impedance assembly.
- **`src/optimization/QMatrix.jl`** – Objective matrix construction.
- **`src/postprocessing/Visualization.jl`** – Plotting utilities (`plot_mesh_wireframe`, `plot_mesh_comparison`, `save_mesh_preview`).
- **`examples/03_beamsteering_physical_unitcell.jl`** – Complete beam‑steering example and demonstration of the full workflow.

---

## 10. Exercises

### 10.1 Conceptual Questions

1. **Convergence diagnosis**: An optimization trace shows $J$ decreasing for 50 iterations, then oscillating with no further improvement. The gradient norm stagnates around $10^{-2}$. List three possible causes and how you would test each.
2. **Boundary effects**: Why might an optimization with tight box constraints (e.g., $\theta_p \in [-100, 100]$) converge to a solution where many parameters are exactly at the bounds? Is this desirable?
3. **Reproducibility**: You run the same optimization twice with identical inputs but get slightly different final objective values ($\Delta J \approx 10^{-4}$). What could explain this? How would you ensure bit‑wise reproducibility?

### 10.2 Derivation Tasks

1. **Projected L‑BFGS**: Describe how the projection step after an L‑BFGS update affects the Hessian approximation. Why is projected L‑BFGS suitable for box constraints but not for general nonlinear constraints?
2. **Line‑search conditions**: The backtracking line search uses Armijo conditions. Write these conditions mathematically and explain their role in ensuring sufficient decrease.

### 10.3 Coding Exercises

1. **Basic optimization**: Run the template from Section 5.1 for a $2\lambda \times 2\lambda$ plate with steering angles $0^\circ$, $30^\circ$, and $60^\circ$. Compare convergence speed and final directivity.
2. **Bound sensitivity**: Repeat the same optimization with three different bound sets: $[-100, 100]$, $[-500, 500]$, and $[-1000, 1000]$. How do the bounds affect the final solution and convergence?
3. **Trace analysis**: Write a script that loads a saved trace and plots `[t.J for t in trace]` vs. iteration and `[t.gnorm for t in trace]` vs. iteration. Annotate the plots with key events (e.g., "gradient norm plateau").

### 10.4 Advanced Challenges

1. **Multi‑start wrapper**: Implement a general multi‑start wrapper function that takes an optimizer, a distribution for initial guesses, and a number of restarts, and returns the best solution. Apply it to a challenging steering problem and compare with single‑start results.
2. **Continuation method**: Implement a continuation loop that incrementally increases the steering angle, using each optimized solution as the initial guess for the next angle. Test on a range from $0^\circ$ to $60^\circ$ and plot final directivity vs. steering angle.
3. **Custom callback**: Extend the optimizer to accept a user callback function that is called each iteration and can modify parameters or stop optimization. Use the callback to implement an adaptive regularization scheme that increases `regularization_alpha` when the condition number exceeds a threshold.

---

## 11. Chapter Checklist

After studying this chapter, you should be able to:

- [ ] **Configure** a complete optimization run with appropriate geometry, patch partition, incident field, objective matrices, and initial parameters.
- [ ] **Select** optimizer parameters (`maxiter`, `tol`, `m_lbfgs`, `lb`, `ub`, `solver`, `preconditioning`) based on problem size and conditioning.
- [ ] **Monitor** convergence through iteration traces and diagnose common issues (oscillations, stagnation, small step lengths).
- [ ] **Validate** optimization results by independent far‑field recomputation, perturbation sensitivity analysis, and comparison with baselines.
- [ ] **Ensure** reproducibility by logging all inputs, random seeds, and software versions.
- [ ] **Apply** advanced techniques (multi‑start, continuation, warm‑starting) for challenging design problems.

---

## 12. Further Reading

1. **L‑BFGS and projected gradient methods**:
   - Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2nd ed.). Springer. (Chapters 4, 5, and 16 cover line‑search methods, L‑BFGS, and projected gradients.)
   - Byrd, R. H., et al. (1995). *A limited memory algorithm for bound constrained optimization*. SIAM Journal on Scientific Computing, 16(5), 1190–1208.

2. **Convergence diagnosis and debugging**:
   - Dennis, J. E., & Schnabel, R. B. (1996). *Numerical methods for unconstrained optimization and nonlinear equations*. SIAM. (Chapter 6 discusses convergence tests and failure modes.)
   - Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). *Trust‑region methods*. SIAM. (Although focused on trust‑region methods, the discussion of convergence diagnostics is widely applicable.)

3. **Reproducibility in computational science**:
   - Sandve, G. K., et al. (2013). *Ten simple rules for reproducible computational research*. PLOS Computational Biology, 9(10), e1003285.
   - Stodden, V., et al. (2016). *Enhancing reproducibility for computational methods*. Science, 354(6317), 1240–1241.

4. **Inverse design case studies in electromagnetics**:
   - Molesky, S., et al. (2018). *Inverse design in nanophotonics*. Nature Photonics, 12(11), 659–670.
   - Christiansen, R. E., & Sigmund, O. (2021). *Compact 200‑line MATLAB code for inverse design in photonics by topology optimization*. Computer Physics Communications, 269, 108117.

5. **Julia optimization libraries**:
   - `Optim.jl` – General‑purpose optimization library (includes L‑BFGS).
   - `LineSearches.jl` – Advanced line‑search algorithms.
   - `NLopt.jl` – Interface to the NLopt nonlinear‑optimization library.

---

Continue to Part VI — Validation for internal consistency checks, gradient
verification, and cross‑validation with external codes.
