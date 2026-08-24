# Internal consistency

Internal checks compare quantities produced by the same discretized problem.
They can detect inconsistent inputs or algebra, but they do not replace an
analytical or independent-solver reference.

For local scattered and total fields, also run the
[Rayleigh-sphere validation](06-near-total-field-rayleigh-sphere.md).

## True linear residual

For $ZI=v$, measure the residual against the original matrix and right-hand
side:

```julia
using LinearAlgebra

rhs_norm = norm(v)
iszero(rhs_norm) && error("relative residual is undefined for a zero RHS")
relative_residual = norm(Z * current - v) / rhs_norm
```

This checks the algebraic solve. It does not check the mesh, quadrature,
formulation, or observation operator. Set a gate from the chosen solver
tolerance and reference case. GMRES workflows should keep the package's
original-system true-residual check enabled.

For a dense matrix, record singular-value diagnostics separately:

```julia
condition = condition_diagnostics(Z)
println(condition)
```

`condition_diagnostics` returns `cond`, `sv_max`, and `sv_min`. A large or
changing condition number is evidence for further comparison, not proof of the
cause of an observable error.

## PEC power balance

`radiated_power` computes

```math
P_{\mathrm{rad}}=
\frac{1}{2\eta_0}\sum_qw_q
|\mathbf E_\infty(\hat{\mathbf r}_q)|^2,
```

and `input_power` computes

```math
P_{\mathrm{in}}=-\frac12\operatorname{Re}(I^\dagger v).
```

```julia
P_rad = radiated_power(E_ff, grid; eta0=eta0)
P_in = input_power(current, v)
ratio = energy_ratio(current, v, E_ff, grid; eta0=eta0)

println((P_rad=P_rad, P_in=P_in, ratio=ratio))
```

For a lossless PEC benchmark, refine the mesh, triangle quadrature, and
spherical grid and check whether `ratio` approaches one. For a passive
resistive impedance sheet with positive input and radiated power, absorption
can make the ratio smaller than one. Do not apply a PEC gate to that model.

`energy_ratio` throws when input power is exactly zero. A negative input power
or a non-finite ratio should be preserved and investigated instead of being
hidden by an absolute value.

## Far-field transversality

The far-field amplitude should be transverse to each stored direction. Use a
normalized check so weak samples do not dominate solely because of scale:

```julia
radial_error = maximum(
    abs(dot(@view(grid.rhat[:, q]), @view(E_ff[:, q]))) /
    max(norm(@view(E_ff[:, q])), eps(Float64))
    for q in eachindex(grid.w)
)
```

Choose the gate from a verified case and record the angular grid.

## Direct versus Q-matrix objective

`build_Q` represents the same discrete projected-field sum as
`projected_power` when both receive identical radiation vectors, grid,
polarization matrix, and mask:

```julia
mask = trues(length(grid.w))
Q = build_Q(G_mat, grid, polarization; mask=mask)

direct_value = projected_power(
    E_ff, grid, polarization; mask=mask)
quadratic_value = real(dot(current, Q * current))

scale = max(abs(direct_value), abs(quadratic_value))
relative_difference = iszero(scale) ? 0.0 :
    abs(direct_value - quadratic_value) / scale
```

This projected objective omits the $1/(2\eta_0)$ conversion used by
`radiated_power`. Compare like with like. A mismatch requires checking the
shared inputs and dimensions before changing either implementation.

## Reciprocity diagnostic

The regression suite checks transpose symmetry of a fixed PEC EFIE case:

```julia
symmetry_error = norm(Z_efie - transpose(Z_efie)) / norm(Z_efie)
```

Record the mesh, quadrature, and formulation with this value. Do not copy the
test's numerical threshold to a different formulation without a convergence
study.

## Executable convergence study

Run:

```bash
julia --project=. --startup-file=no validation/paper/run_convergence_study.jl
```

The driver writes `data/convergence_study.csv` and
`data/gradient_verification.csv` and exits nonzero when a declared gate fails.
[Gradient verification](02-gradient-verification.md) owns the gate description;
the `CONVERGENCE_MAX_*` constants in
`validation/paper/run_convergence_study.jl` own their executable values. For
another geometry, declare new gates from its required accuracy and measured
convergence.

The optional paper aggregator is:

```bash
julia --project=. --startup-file=no \
  validation/paper/generate_consistency_report.jl
```

It requires generated artifacts from several workflows. When optional inputs
are missing, the script reports that state rather than manufacturing their
metrics.

## Diagnosis order

When an internal gate fails:

1. preserve the exact command, input, and measured value;
2. check finiteness, dimensions, units, and mesh policy;
3. recompute the true residual against the original system;
4. rebuild paired quantities from the same grid, constants, and masks;
5. refine the spherical grid while holding the solved current fixed;
6. refine the mesh and reassemble; and
7. compare supported triangle quadrature orders.

Change one factor per comparison. A failed power, symmetry, or conditioning
check narrows the next experiment but does not identify a cause by itself.

## Evidence record

Keep these values with a validation result:

- mesh quality and resolution reports;
- RWG count, frequency, excitation, and physical constants;
- triangle and spherical quadrature settings;
- solver method, tolerance, iteration status, and true residual;
- `P_in`, `P_rad`, and their ratio;
- transversality error;
- direct and quadratic objective values and their shared inputs; and
- condition and reciprocity diagnostics where applicable.

## Code map

| Task | Source |
|:--|:--|
| Power, RCS, and condition diagnostics | `src/postprocessing/Diagnostics.jl` |
| Far-field construction | `src/postprocessing/FarField.jl` |
| Q matrices and projected objectives | `src/optimization/QMatrix.jl` |
| Dense and iterative solves | `src/solver/Solve.jl`, `src/solver/IterativeSolve.jl` |
| Convergence gates | `validation/paper/run_convergence_study.jl` |
| Optional aggregate report | `validation/paper/generate_consistency_report.jl` |
