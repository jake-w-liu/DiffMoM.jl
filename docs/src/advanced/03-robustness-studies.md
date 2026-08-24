# Robustness studies

Use the robustness sweep to evaluate one saved impedance design at nearby
frequencies and incidence angles. The script measures performance, but it does
not optimize across scenarios or decide whether a change is physical.

## What the sweep evaluates

The driver is `validation/robustness/run_robustness_sweep.jl`. It fixes the
physical plate and impedance pattern, then rebuilds the excitation, EFIE matrix,
and far-field operators for each scenario.

The geometry, reference frequency, target cone, spherical grid, and scenario
table are defined together in the driver's `main` function. Keep that setup
aligned with `validation/paper/run_beam_steering_case.jl`; the saved-design
loader rejects a geometry mismatch before solving.

For each case, the script compares the saved reactive design with a PEC plate.
Its objective is the ratio

```math
J(\boldsymbol\theta;\xi)=
\frac{\mathbf I^\dagger
      \mathbf Q_{\mathrm{target}}\mathbf I}
     {\mathbf I^\dagger
      \mathbf Q_{\mathrm{total}}\mathbf I},
```

where $\xi$ identifies the frequency and incidence angle. Both Q matrices use
the same linear-x polarization projection. The numerator covers the target
cone, while the denominator covers the full grid.

## Prerequisite

Generate the saved design from the project root:

```bash
julia --project=. validation/paper/run_beam_steering_case.jl
```

That command must create `data/beam_steer_impedance.csv` with one `theta_opt`
value per triangle. The robustness driver verifies the required columns,
triangle order, finite values, row count, and triangle-center coordinates
before using the design.

## Run the sweep

```bash
julia --project=. validation/robustness/run_robustness_sweep.jl
```

The driver writes `data/robustness_sweep.csv` with these columns:

| Column | Meaning |
|:--|:--|
| `case` | Scenario label |
| `freq_GHz` | Frequency in GHz |
| `theta_inc_deg` | Incidence angle in degrees |
| `J_opt_pct` | Saved-design objective, multiplied by 100 |
| `J_pec_pct` | PEC objective, multiplied by 100 |
| `gain_target_dB` | Saved-design directivity minus PEC directivity at the sampled angle nearest $30^\circ$ |
| `target_theta_deg` | Sampled target angle used for that comparison |
| `peak_theta_opt_deg` | Peak angle of the saved design on the sampled $\phi\approx0$ cut |
| `peak_opt_dBi` | Directivity at that sampled peak |
| `residual_pec`, `residual_opt` | True relative residuals for both solves |
| `energy_ratio_pec`, `energy_ratio_opt` | Sampled $P_{\mathrm{rad}}/P_{\mathrm{in}}$ checks |

`gain_target_dB` answers whether the design improves the sampled target
direction relative to PEC. `peak_theta_opt_deg` answers where the largest
sampled value occurs on the selected cut. Report both. A positive target gain
does not imply that the peak remains on target.

## Qualification checks

The driver fails before writing its CSV when a solve misses its printed
residual or power-balance gate, a metric is non-finite, or the saved design is
outside its declared box. Those checks establish numerical consistency for the
fixed run; they do not establish mesh or angular-grid convergence.

The executable thresholds are the `ROBUSTNESS_MAX_*` constants in
`validation/robustness/run_robustness_sweep.jl`. The verification labels read
those constants, so the reported gates match the comparisons being applied.

Before assigning a physical cause to a change, record at least:

1. mesh refinement or coarsening results for the reported metrics;
2. angular-grid refinement results for the target gain and peak angle;
3. the effect of another supported triangle quadrature order; and
4. the same checks for the PEC reference.

Choose tolerances before running the qualification study. Base them on the
required accuracy and observed convergence, not on fixed values copied from
another geometry.

If a metric changes after tightening a failed check, the sweep has not isolated
physical detuning. If all declared checks pass and the change remains, the
evidence supports, but does not by itself prove, a physical interpretation.

## Change the scenarios

Edit the `cases` DataFrame in the driver. For example, this table evaluates five
frequencies from 2.8 GHz to 3.2 GHz at normal incidence:

```julia
cases = DataFrame(
    case = ["f_$(i)" for i in 1:5],
    freq_GHz = collect(range(2.8, 3.2; length=5)),
    theta_inc_deg = zeros(5),
)
```

Add rows with both fields changed to study combined perturbations. Keep the
saved impedance length and the fixed mesh partition consistent.

Each row assembles and factorizes a dense system and constructs a radiation
matrix. Measure runtime and peak memory on the target machine before increasing
the mesh, grid, or scenario count.

## Multi-scenario design is separate

The sweep reuses one saved design. It does not compute a design that is robust
across frequencies or angles.

A mean objective over scenarios would be

```math
J_{\mathrm{mean}}(\boldsymbol\theta)
=\sum_{s=1}^{S} w_s J(\boldsymbol\theta;\xi_s),
\qquad \sum_{s=1}^{S}w_s=1.
```

Its gradient is the same weighted sum of per-scenario gradients. A
frequency-dependent implementation must rebuild every frequency-dependent
operator, solve the forward and adjoint systems for each scenario, combine the
gradients with the declared weights, and verify the result by finite
differences.

`optimize_multiangle_rcs` is the package's supported multi-incidence optimizer
for its weighted RCS objective at one wavenumber. It is not a replacement for a
frequency-dependent directivity-ratio optimizer.

## Report the result

State the mesh, quadrature order, angular grid, target definition, scenario
table, solver settings, and declared qualification tolerances. For every row,
report frequency, incidence angle, target gain, peak angle, and optimized
objective. Also report the minimum target gain and maximum absolute peak-angle
error over the declared scenario set. Keep the residual and convergence
evidence with the result.

## Code map

| Task | Source |
|:--|:--|
| Sweep setup and CSV output | `validation/robustness/run_robustness_sweep.jl` |
| Directivity-ratio optimization | `src/optimization/Optimize.jl` |
| Q-matrix construction | `src/optimization/QMatrix.jl` |
| Adjoint solves and impedance gradients | `src/optimization/Adjoint.jl` |
| Multi-incidence RCS optimization | `src/optimization/MultiAngleRCS.jl` |
| EFIE and excitation assembly | `src/assembly/EFIE.jl`, `src/assembly/Excitation.jl` |
| Far-field evaluation and diagnostics | `src/postprocessing/FarField.jl`, `src/postprocessing/Diagnostics.jl` |
