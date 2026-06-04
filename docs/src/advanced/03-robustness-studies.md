# Robustness Studies

## Purpose

An optimized design at a single frequency and incidence angle can **detune**
when the scenario changes—a phenomenon known as **narrowband behavior**.
Robustness studies quantify how much performance degrades under perturbations
(frequency shift, angle shift, manufacturing tolerances, etc.) and help
distinguish between:

- **Expected physical detuning** (inherent to the design physics), and
- **Implementation errors** (solver failures, mesh defects, gradient inaccuracies).

This chapter explains the built‑in robustness‑sweep workflow, how to interpret
its metrics, and how to use the results to guide robust‑design strategies.

---

---

## Learning Goals

After this chapter, you should be able to:

1. Define robustness metrics that align with your beam‑steering objective.
2. Understand the mathematical formulation of multi‑scenario objectives.
3. Run the built‑in frequency/incidence perturbation workflow and interpret its
   output tables.
4. Diagnose whether performance degradation is expected physical detuning or a
   sign of implementation error.
5. Extend the robustness sweep to custom perturbation sets (e.g., manufacturing
   tolerances).
6. Implement a simple multi‑scenario objective for robust design.

---

## 1) Mathematical Foundation: Multi‑Scenario Objectives

### Beam‑Steering Objective with Scenario Dependence

The core beam‑steering objective (paper `bare_jrnl.tex`, Eq. 19) is a ratio of
quadratic forms:

```math
J(\theta;\xi)
= 
\frac{\mathbf I(\theta;\xi)^\dagger \mathbf Q_{\text{target}}(\xi)\,\mathbf I(\theta;\xi)}
{\mathbf I(\theta;\xi)^\dagger \mathbf Q_{\text{total}}(\xi)\,\mathbf I(\theta;\xi)},
```

where $\theta$ are the design variables (patch impedances), $\xi$ denotes the
scenario parameters (frequency $f$, incidence angle $(\theta_{\text{inc}},\phi_{\text{inc}})$,
polarization, etc.), $\mathbf I(\theta;\xi)$ is the forward solution of the EFIE
under scenario $\xi$, and $\mathbf Q_{\text{target}},\mathbf Q_{\text{total}}$
are the far‑field projection matrices defined in `src/postprocessing/FarField.jl`.

### Why Single‑Scenario Designs Are Narrowband

When optimized only at a nominal scenario $\xi_0$, the gradient
$\nabla_\theta J(\theta;\xi_0)$ drives the design to a local maximum **for that
specific scenario**. However, the mapping from impedance pattern $\theta$ to
scattered field is dispersive:

- The phase progression required for steering depends on $k = 2\pi/\lambda$.
- Reactive loading (via $\theta$) produces a frequency‑dependent phase shift.
- A fixed impedance pattern that steers at $f_0$ will generally steer to a
  different direction at $f \neq f_0$, and may lose directivity altogether.

Thus, small perturbations in $\xi$ can **shift the global maximum away from the
target direction** even if the gain **at the target angle** remains positive.
This is expected physical detuning, not a numerical error.

### Robustness as a Multi‑Scenario Optimization Problem

To achieve robustness, we can replace the single‑scenario objective with an
aggregate over a set of scenarios $\{\xi_s\}_{s=1}^S$:

```math
J_{\text{mean}}(\theta) = \frac{1}{S}\sum_{s=1}^S J(\theta;\xi_s), \qquad
J_{\text{worst}}(\theta) = \min_{s} J(\theta;\xi_s).
```

The gradient of $J_{\text{mean}}$ is simply the average of the per‑scenario
gradients, which can be computed with the same adjoint machinery described in
the paper (Eq. 25). The package currently provides evaluation tools; robust
**design** requires extending the optimization loop to handle multiple scenarios.

---

## 2) Metrics You Should Track

The robustness‑sweep script (`validation/robustness/run_robustness_sweep.jl`)
computes five key metrics for each perturbed scenario. Understanding what each
metric measures is crucial for correct interpretation.

### Objective‑Space Metrics

1. **`J_opt_pct`** – Optimized objective ratio $J(\theta_{\text{opt}};\xi)$,
   expressed as a percentage (×100). This is the value of the beam‑steering
   ratio achieved by the previously optimized impedance pattern $\theta_{\text{opt}}$
   under the perturbed scenario $\xi$.

2. **`J_pec_pct`** – PEC baseline ratio $J(\theta_{\text{PEC}};\xi)$, where
   $\theta_{\text{PEC}} = 0$ (all patches perfectly conducting). This gives the
   “no‑design” reference; any improvement over PEC is due to the impedance pattern.

### Pattern‑Space Metrics

3. **`gain_target_dB`** – Directivity gain (dB) at the **target angle**
   (e.g., 30° from broadside). Computed as:
   ```math
   G_{\text{target}} = D_{\text{opt}}(\theta_{\text{target}}) - D_{\text{PEC}}(\theta_{\text{target}}),
   ```
   where $D$ is directivity in dBi. Positive gain means the design still
   enhances radiation at the intended direction.

4. **`peak_theta_opt_deg`** – Angular location (degrees) of the **global peak**
   of the optimized pattern on the $\phi \approx 0$ cut. If this angle drifts
   away from the target, the beam has steered off‑target.

5. **`peak_opt_dBi`** – Directivity (dBi) at that global peak. Indicates how
   much peak directivity is retained.

### Two Complementary Questions

Together these metrics answer two distinct engineering questions:

- **Does the design still help at the target direction?** → Look at `gain_target_dB`.
- **Does the design still peak at the target direction?** → Look at `peak_theta_opt_deg`.

A robust design would keep both `gain_target_dB` positive and
`peak_theta_opt_deg` close to the target angle across the perturbation range.

### Internal Consistency Checks

The script also verifies internal consistency (power conservation, solver
residuals) and will warn if any scenario fails those checks—a sign of possible
implementation error rather than physical detuning.

---

## 3) Running the Robustness Sweep

### Basic Command

From the project root:

```bash
julia --project=. validation/robustness/run_robustness_sweep.jl
```

This executes the default sweep, which perturbs:

- Frequency: ±2 % around the design frequency (3 GHz).
- Incidence angle: ±2° around broadside.

### Output Files

The script writes a CSV table to `data/robustness_sweep.csv` with columns:

| Column | Description |
|--------|-------------|
| `case` | Label (e.g., `f_-2pct`, `ang_-2deg`) |
| `freq_GHz` | Frequency in GHz |
| `theta_inc_deg` | Incidence angle (degrees from broadside) |
| `J_opt_pct` | Optimized objective ratio (%) |
| `J_pec_pct` | PEC baseline ratio (%) |
| `gain_target_dB` | Directivity gain at target angle (dB) |
| `target_theta_deg` | Target angle (should be constant) |
| `peak_theta_opt_deg` | Actual peak angle of optimized pattern (deg) |
| `peak_opt_dBi` | Peak directivity of optimized pattern (dBi) |

### Customizing the Sweep

To study different perturbation ranges, modify the `cases` DataFrame inside the
script (lines 60‑64). For example, to sweep frequency from 2.8 GHz to 3.2 GHz
in 5 steps:

```julia
cases = DataFrame(
    case = ["f_$(i)" for i in 1:5],
    freq_GHz = range(2.8, 3.2, length=5),
    theta_inc_deg = zeros(5),
)
```

You can also add combined perturbations (frequency + angle) by adding rows with
both fields varied.

### Prerequisites

The script expects a previously optimized impedance pattern stored in
`data/beam_steer_impedance.csv`. This file is generated by the paper validation
driver `validation/paper/run_beam_steering_case.jl`. If missing, run:

```bash
julia --project=. validation/paper/run_beam_steering_case.jl
```

before executing the robustness sweep.

### Runtime and Memory

Each scenario requires a full forward solve (dense LU) and far‑field evaluation.
With the default plate geometry (≈1,200 unknowns), the whole sweep finishes in
seconds. For larger geometries, consider reducing the number of perturbation
points or using a coarsened mesh (see Chapter 2).

---

## 4) Interpreting a Typical Detuning Pattern

### Common Observation

For a design optimized at $f_0$ and $\theta_{\text{inc}}=0^\circ$, a small
frequency increase ($f = f_0 + \Delta f$) often produces:

- **`gain_target_dB`** remains positive (design still enhances target direction),
- **`peak_theta_opt_deg`** shifts toward broadside ($0^\circ$),
- **`J_opt_pct`** decreases (lower beam‑steering ratio).

This is **expected physical detuning**, not a solver bug.

### Physical Explanation

Beam steering with a reactive impedance surface works by imposing a phase
gradient $\nabla \phi(x,y)$ that compensates the incident phase slope. The
required gradient is

```math
\nabla \phi = k (\hat{\mathbf r}_{\text{target}} - \hat{\mathbf r}_{\text{inc}}),
```

where $k = 2\pi/\lambda$. When frequency changes, $k$ changes, but the
impedance pattern $\theta$ (and thus the phase gradient it produces) is fixed.
Hence the beam points to a different direction:

```math
\hat{\mathbf r}_{\text{actual}} \approx \hat{\mathbf r}_{\text{inc}} + \frac{1}{k}\nabla\phi.
```

If $\nabla\phi$ was designed for $k_0$, then at $k \neq k_0$ the beam steers to
$\hat{\mathbf r}_{\text{actual}} \neq \hat{\mathbf r}_{\text{target}}$.

### Incidence‑Angle Perturbations

Similarly, changing the incidence angle changes the required compensation
$\hat{\mathbf r}_{\text{target}} - \hat{\mathbf r}_{\text{inc}}$. A fixed
impedance pattern cannot adapt, so the beam moves.

### Key Takeaway

**“Still better than PEC at target”** and **“global maximum moved away from target”**
can coexist. This is a fundamental bandwidth limitation of static impedance
surfaces, not a flaw in the solver or optimization.

### Quantitative Example

From the default sweep, you might see:

| Case | `gain_target_dB` (dB) | `peak_theta_opt_deg` (deg) |
|------|-----------------------|----------------------------|
| `f_nom` | +3.2 | 30.0 |
| `f_+2pct` | +2.1 | 26.5 |
| `ang_+2deg` | +2.8 | 28.0 |

The design still improves target‑angle directivity (`gain_target_dB` > 0) but
the beam peak has drifted several degrees off‑target.

---

## 5) When to Suspect an Implementation Problem

Robustness degradation is **expected** when the physics changes. It becomes
**suspicious** only when accompanied by signs of numerical or algorithmic failure.

### Diagnostic Decision Tree

For each perturbed scenario, ask:

1. **Did the solver residual exceed tolerance?**
   - Check `norm(Z*I - v)/norm(v)` (printed to console in verbose mode).
   - Acceptable: < 1e‑8. Suspicious: > 1e‑6.

2. **Did power conservation fail?**
   - Compare total radiated power to incident power (computed internally).
   - Acceptable: within 1 %. Suspicious: > 5 % discrepancy.

3. **Is the PEC reference behaving inconsistently?**
   - `J_pec_pct` should vary smoothly with frequency/angle.
   - Sudden jumps or non‑monotonic changes may indicate mesh or quadrature issues.

4. **Are there internal‑consistency warnings?**
   - Inspect `energy_ratio(I, v, E_ff, grid)` and objective consistency
     `abs(real(dot(I, Q*I)) - projected_power(...))`.
   - Any large mismatch should be investigated.

5. **Do very small perturbations produce discontinuous jumps?**
   - Changing frequency by 0.1 % should change metrics by a small amount.
   - If results jump erratically, suspect numerical conditioning (enable
     preconditioning or increase quadrature order).

### If All Checks Are Clean

If the solver residual is small, power is conserved, and the PEC reference is
smooth, then the observed degradation is almost certainly **physical detuning**,
not an implementation error. Proceed to robust‑design strategies (Section 7).

### If Any Check Fails

First verify the mesh quality (`mesh_quality_report`) and consider increasing
the quadrature order (`quad_order=4`). If the problem persists, examine the
specific scenario’s excitation and matrix conditioning; the issue may be
low‑frequency breakdown or a nearly singular EFIE matrix.

---

## 6) Toward Robust Design (Not Just Robustness Evaluation)

The built‑in sweep **evaluates** robustness of a previously optimized design.
To **design** a robust impedance pattern from the start, you need a
multi‑scenario objective.

### Multi‑Scenario Objective Formulations

Two common aggregates are:

```math
J_{\text{mean}}(\theta) = \frac{1}{S}\sum_{s=1}^{S} J(\theta;\xi_s), \qquad
J_{\text{worst}}(\theta) = \min_{s} J(\theta;\xi_s).
```

- **Mean objective** smooths performance across scenarios; its gradient is the
  average of per‑scenario gradients (easy to compute).
- **Worst‑case objective** maximizes the minimum performance; its gradient
  requires sub‑gradient methods or smoothing approximations.

### Gradient Computation

For $J_{\text{mean}}$, the gradient with respect to $\theta$ is

```math
\nabla_\theta J_{\text{mean}} = \frac{1}{S}\sum_{s=1}^S \nabla_\theta J(\theta;\xi_s),
```

where each term $\nabla_\theta J(\theta;\xi_s)$ is computed via the **adjoint
method** described in the paper (Eq. 25). The same adjoint solve can be reused
across scenarios because the adjoint source depends only on the forward solution
and the $Q$ matrices.

### Implementation Sketch

Using existing package primitives, a robust‑design loop looks like:

```julia
function robust_gradient(theta, scenarios)
    grad_total = zero(theta)
    for ξ in scenarios
        # Forward solve
        Z = assemble_Z_efie(..., ξ)
        v = assemble_v_plane_wave(..., ξ)
        I = solve_forward(Z, v)
        # Adjoint solve + impedance gradient
        λ = solve_adjoint(Z, Q_target, I)
        grad_s = gradient_impedance(Mp, I, λ; reactive=true)
        grad_total .+= grad_s
    end
    return grad_total / length(scenarios)
end
```

Then plug `robust_gradient` into L‑BFGS (`optimize_lbfgs`) or a gradient‑based
optimizer.

### Practical Considerations

- **Computational cost**: Each scenario requires a dense LU factorization.
  Keep the scenario set small (3–5 points) or use coarsened meshes.
- **Scenario selection**: Choose perturbations that represent realistic
  uncertainties (frequency band, angular sector, manufacturing tolerances).
- **Regularization**: Adding a small penalty on $\theta$ variation can improve
  smoothness and manufacturability.

### Example: Two‑Frequency Robust Design

As a starting experiment, optimize for $f_0$ and $f_0 + \Delta f$ simultaneously
using $J_{\text{mean}}$. Compare its robustness to a single‑frequency design
using the built‑in sweep.

---

## 7) Recommended Reporting Style

In publications and engineering reports, avoid a binary “robust/not robust”
label. Instead, provide a transparent table or plot that separates **target‑angle
gain** from **peak‑angle drift**.

### Table Template

| Scenario | $f$ (GHz) | $\theta_{\text{inc}}$ (deg) | $G_{\text{target}}$ (dB) | $\theta_{\text{peak}}$ (deg) | $J_{\text{opt}}$ (%) |
|----------|-----------|-----------------------------|--------------------------|------------------------------|----------------------|
| Nominal  | 3.00      | 0.0                         | +3.2                     | 30.0                         | 85.4                 |
| $f$‑2 %  | 2.94      | 0.0                         | +2.8                     | 28.5                         | 82.1                 |
| $f$+2 %  | 3.06      | 0.0                         | +2.1                     | 26.5                         | 78.9                 |
| $\theta_{\text{inc}}$‑2° | 3.00 | –2.0                        | +2.9                     | 28.0                         | 83.2                 |
| $\theta_{\text{inc}}$+2° | 3.00 | +2.0                        | +2.7                     | 27.5                         | 81.8                 |

### Plot Recommendations

- **Subplot (a)**: `gain_target_dB` vs. frequency (or angle).
- **Subplot (b)**: `peak_theta_opt_deg` vs. frequency (or angle).
- **Subplot (c)**: Far‑field patterns at nominal and extreme scenarios.

### Interpretation Text

“The design retains positive target‑angle gain (> 2 dB) across the perturbation
range, indicating it still enhances directivity in the intended direction.
However, the beam peak drifts up to 3.5° off‑target, revealing a steering‑angle
sensitivity of approximately 1.75 °/ %. This is consistent with the theoretical
phase‑gradient detuning described in Section 4.”

### Why This Matters

Separating the two metrics lets readers judge whether the design is **useful**
(gain retained) and **accurate** (peak on target) under perturbations. It also
sets a clear baseline for comparing different robust‑design strategies.

---

## 8) Code Mapping

| Component | Source File | Key Functions / Lines |
|-----------|-------------|-----------------------|
| Robustness sweep driver | `validation/robustness/run_robustness_sweep.jl` | `main()` (line 38), `build_target_mask` (line 20), `mean_dir_at_theta` (line 31) |
| Beam‑steering objective & gradient | `src/optimization/Optimize.jl`, `src/optimization/Adjoint.jl` | `optimize_directivity`, `solve_adjoint`, `gradient_impedance` |
| EFIE assembly | `src/assembly/EFIE.jl` | `assemble_Z_efie` (line …) |
| Excitation assembly | `src/assembly/Excitation.jl` | `assemble_v_plane_wave` (line …) |
| Far‑field matrices | `src/postprocessing/FarField.jl`, `src/optimization/QMatrix.jl` | `radiation_vectors`, `build_Q` |
| Diagnostics | `src/postprocessing/Diagnostics.jl` | `energy_ratio`, `projected_power`, `condition_diagnostics` |
| Mesh & RWG utilities | `src/geometry/Mesh.jl`, `src/basis/RWG.jl` | `make_rect_plate`, `build_rwg` |

> Note: Line numbers are approximate; refer to the latest source files.

---

## 9) Exercises

### Basic

1. **Run the default sweep** and identify which metric first shows significant
   detuning. Is it `gain_target_dB` or `peak_theta_opt_deg`?
2. **Plot the results**: Load `data/robustness_sweep.csv` and create a simple
   plot of `gain_target_dB` vs. `freq_GHz`. Describe the trend.
3. **Check internal consistency**: Run the sweep with verbose output (add
   `verbose=true` to the solver calls) and verify that all solver residuals are
   below 1e‑8.

### Practical

4. **Extend the frequency sweep**: Modify `run_robustness_sweep.jl` to include
   five frequency points from 2.8 GHz to 3.2 GHz (keeping incidence angle zero).
   Compare the trends of `peak_theta_opt_deg` and `J_opt_pct`.
5. **Incidence‑angle sweep**: Create a new sweep that varies the incidence angle
   from –5° to +5° in 1° steps (frequency fixed at 3 GHz). Which perturbation
   (frequency or angle) causes larger peak drift?
6. **Mesh‑size impact**: Repeat the default sweep with a coarser plate mesh
   (e.g., `Nx=Ny=8` instead of 12). Does coarsening change the detuning pattern
   qualitatively?

### Advanced

7. **Two‑scenario robust design**: Implement a simple two‑scenario mean objective
   that optimizes for both 2.94 GHz and 3.06 GHz simultaneously. Use the same
   L‑BFGS wrapper but compute gradients as the average of the two scenario
   gradients. Compare its robustness (using the built‑in sweep) to the original
   single‑frequency design.
8. **Gradient verification**: For a small problem (Nx=Ny=4), compute the
   gradient of $J_{\text{mean}}$ via finite differences (perturb each $\theta_i$
   by 1e‑6) and compare to the adjoint‑based gradient. Report the maximum
   relative error.
9. **Manufacturing‑tolerance study**: Simulate random impedance variations by
   adding Gaussian noise (standard deviation 5 Ω) to the optimized $\theta$
   vector. Run the robustness sweep on 10 perturbed copies and compute the
   standard deviation of `gain_target_dB`. Is the design sensitive to small
    manufacturing errors?

---

## 10) Chapter Checklist

Before moving to the next chapter, verify you can:

- [ ] Explain why single‑scenario beam‑steering designs are inherently narrowband.
- [ ] List the five key metrics reported by the robustness sweep and interpret what each measures.
- [ ] Run the built‑in robustness sweep and customize the perturbation set.
- [ ] Distinguish between expected physical detuning and signs of implementation error using the diagnostic decision tree.
- [ ] Sketch a multi‑scenario objective (mean or worst‑case) and describe how its gradient would be computed.
- [ ] Present robustness results in a publication‑ready table that separates target‑angle gain from peak‑angle drift.

---

## 11) Further Reading

- **Paper `bare_jrnl.tex`**: Eq. 19 defines the beam‑steering ratio; Eq. 25 gives the adjoint gradient; Section 4 discusses bandwidth limitations of static impedance surfaces.
- **Robust Optimization in Electromagnetics**:
  - *Design of Electromagnetic Structures Using Robust Optimization* (Forrester et al., 2008) – introduces worst‑case and statistical approaches.
  - *Multi‑scenario gradient‑based topology optimization of microwave devices* (Jensen, 2014) – extends adjoint methods to multiple frequencies.
- **Julia DataFrames & CSV**: The robustness sweep uses `DataFrames.jl` and `CSV.jl` for output; see their documentation for advanced table manipulation.
- **Beam‑Steering Bandwidth Fundamentals**: *Array Antenna Handbook* (Hansen, 2009) – covers phased‑array bandwidth limitations, relevant to impedance‑surface steering.

---
