# Far field, Q objectives, and RCS

This chapter maps solved RWG coefficients to far-field amplitudes, power
diagnostics, quadratic objectives, and radar cross section.

## Conventions and array shapes

The package uses the $e^{+i\omega t}$ convention. `radiation_vectors` returns
the asymptotic scattered-field amplitude $\mathbf E_\infty$ used in

```math
\mathbf E^{\mathrm{sca}}(R\hat{\mathbf r})
\sim \frac{e^{-ikR}}{R}\mathbf E_\infty(\hat{\mathbf r}).
```

For $N$ RWG unknowns and $N_\Omega$ observation directions:

- `grid.rhat` has shape `(3, N_omega)`;
- `G_mat` has shape `(3 * N_omega, N)`;
- `I` has length `N`; and
- `E_ff` has shape `(3, N_omega)`.

The three rows of `E_ff` are Cartesian components. `compute_farfield` validates
these dimensions and returns the reshaped product `G_mat * I`.

## Build a spherical midpoint grid

```julia
using DiffMoM
using LinearAlgebra

grid = make_sph_grid(36, 72)
N_omega = length(grid.w)
```

`make_sph_grid(Ntheta, Nphi)` uses midpoint samples:

```math
\theta_i=(i-\tfrac12)\frac{\pi}{N_\theta},\qquad
\phi_j=(j-\tfrac12)\frac{2\pi}{N_\phi},
```

with weights $\sin\theta_i\,\Delta\theta\,\Delta\phi$. The poles and
$\phi=0$ are not stored exactly. Use the recorded angles rather than labelling a
sample as an endpoint.

The constructor checks `max_points` and `max_raw_bytes` before allocating the
grid. Refine the grid until the required integral, peak, or cut is stable under
the study's declared tolerance.

## Compute the scattered far field

Assume `mesh`, `rwg`, `k`, and a solved `Vector{ComplexF64}` named `I` are
available:

```julia
G_mat = radiation_vectors(
    mesh,
    rwg,
    grid,
    k;
    quad_order=3,
    eta0=376.730313668,
)
E_ff = compute_farfield(G_mat, I, N_omega)
```

`radiation_vectors` checks its output, workspace, loop-count, and exceptional
exact-work limits before or during construction. Its dense output payload is

```math
16(3N_\Omega)N\ \text{bytes}
```

for `ComplexF64`. This matrix can dominate postprocessing memory even when the
forward system itself fits. Reduce or batch angular sampling only after checking
the effect on the reported observable.

The field should be transverse to each observation direction. Use a normalized
diagnostic:

```julia
radial_error = maximum(
    abs(dot(@view(grid.rhat[:, q]), @view(E_ff[:, q]))) /
    max(norm(@view(E_ff[:, q])), eps(Float64))
    for q in eachindex(grid.w)
)
```

Choose its acceptance threshold from a verified reference or convergence study.

## Power diagnostics

`radiated_power` evaluates

```math
P_{\mathrm{rad}}=
\frac{1}{2\eta_0}\sum_q w_q
\left|\mathbf E_\infty(\hat{\mathbf r}_q)\right|^2.
```

`input_power` evaluates

```math
P_{\mathrm{in}}=-\frac12\operatorname{Re}(\mathbf I^\dagger\mathbf v).
```

For a PEC reference:

```julia
P_rad = radiated_power(E_ff, grid; eta0=376.730313668)
P_in = input_power(I, v)
rho = energy_ratio(I, v, E_ff, grid; eta0=376.730313668)

println((P_rad=P_rad, P_in=P_in, ratio=rho))
```

For a lossless PEC benchmark, `rho` should converge toward one as the mesh,
triangle quadrature, and spherical grid are refined. The tolerance is
benchmark-specific. A resistive impedance model can absorb power, so do not
reuse a PEC gate without accounting for absorption.

## Polarization-projected objectives

`projected_power` and the Q-matrix path use

```math
J=\sum_{q\in\mathcal M}w_q
\left|\mathbf p_q^\dagger
\mathbf E_\infty(\hat{\mathbf r}_q)\right|^2.
```

This quantity omits the $1/(2\eta_0)$ factor used by `radiated_power`. It is a
weighted projected-field objective, not power in watts. The omission cancels in
ratios built from matching Q matrices.

The built-in polarization matrices are spherical basis projections:

- `pol_linear_x(grid)` returns $\hat{\boldsymbol\theta}$ at each sample; and
- `pol_linear_y(grid)` returns $\hat{\boldsymbol\phi}$ at each sample.

Their names describe the corresponding broadside linear polarizations. At a
general direction they should not be interpreted as fixed Cartesian x and y
vectors.

Build a target cone and compare the direct and quadratic forms:

```julia
pol = pol_linear_x(grid)
target_direction = Vec3(sind(30.0), 0.0, cosd(30.0))
mask = direction_mask(
    grid,
    target_direction;
    half_angle=deg2rad(5.0),
)

Q_target = build_Q(G_mat, grid, pol; mask=mask)
direct_value = projected_power(E_ff, grid, pol; mask=mask)
quadratic_value = real(dot(I, Q_target * I))

scale = max(abs(direct_value), abs(quadratic_value))
relative_difference = iszero(scale) ? 0.0 :
    abs(direct_value - quadratic_value) / scale
println("Q/direct relative difference = ", relative_difference)
```

`build_Q` returns a dense Hermitian positive-semidefinite `N x N` matrix and
preflights its output and workspace. Use `build_Q_operator` when the Q matrix
itself should remain matrix-free, or `apply_Q` when only one `Q * I` product is
needed:

```julia
Q_operator = build_Q_operator(G_mat, grid, pol; mask=mask)
QI = apply_Q(G_mat, grid, pol, I; mask=mask)
```

These alternatives avoid the dense Q output, but they still retain the supplied
radiation matrix and polarization data.

`cap_mask(grid; theta_max=...)` selects a cone about positive z.
`direction_mask` accepts any finite nonzero direction and normalizes it.

## Directivity-ratio objective

A common beam-steering objective is

```math
J_{\mathrm{ratio}}=
\frac{\mathbf I^\dagger Q_{\mathrm{target}}\mathbf I}
     {\mathbf I^\dagger Q_{\mathrm{total}}\mathbf I}.
```

Construct both matrices from the same `G_mat`, `grid`, and `pol`:

```julia
Q_total = build_Q(G_mat, grid, pol)
numerator = real(dot(I, Q_target * I))
denominator = real(dot(I, Q_total * I))
iszero(denominator) && error("directivity-ratio denominator is zero")
J_ratio = numerator / denominator
```

`optimize_directivity` implements this ratio and its two-adjoint quotient-rule
gradient. See [Tutorial 3](../tutorials/03-beam-steering-design.md) for the
complete workflow.

## Bistatic and backscatter RCS

For plane-wave amplitude `E0`, `bistatic_rcs` computes each sample as

```math
\sigma(\hat{\mathbf r}_q)=
\frac{4\pi\left|\mathbf E_\infty(\hat{\mathbf r}_q)\right|^2}
     {|E_0|^2}.
```

```julia
sigma = bistatic_rcs(E_ff; E0=1.0)
sigma_dbsm = 10 .* log10.(max.(sigma, 1e-30))
```

The linear result is in square metres when the field and geometry use the
documented SI convention. State the dB floor whenever plotting or comparing
`sigma_dbsm`.

`backscatter_rcs` takes the incident propagation direction and samples the grid
nearest its negative:

```julia
k_inc_hat = Vec3(0.0, 0.0, -1.0)
back = backscatter_rcs(E_ff, grid, k_inc_hat; E0=1.0)

println((
    sigma=back.sigma,
    theta=back.theta,
    phi=back.phi,
    angular_error_deg=back.angular_error_deg,
))
```

The result is a nearest-grid sample, not an interpolation. Refine the angular
grid and report `angular_error_deg` when the exact direction matters. The
nearest-direction comparison normalizes tolerated floating-point norm drift in
custom `SphGrid.rhat` columns.

For an incident wave propagating along negative z, forward scattering is toward
negative z and backscatter is toward positive z. Do not infer these directions
from the labels of a plot axis.

## Analytical comparison

The PEC sphere validator compares the complete EFIE, solve, far-field, and RCS
path with `mie_bistatic_rcs_pec`:

```bash
julia --project=. --startup-file=no validation/mie/validate_mie_rcs.jl
```

Compare the analytical and numerical values at identical stored directions,
with the same sphere radius, propagation direction, polarization, incident
amplitude, units, and dB floor. Inspect linear-scale error near nulls.

## Resource and convergence checklist

- [ ] Record `N`, `N_omega`, `quad_order`, `eta0`, and all resource-limit
  keywords.
- [ ] Account for the `G_mat` payload before constructing it.
- [ ] Refine the triangle quadrature and spherical grid independently.
- [ ] Verify normalized transversality and benchmark-specific power balance.
- [ ] Compare `projected_power` with `real(dot(I, Q * I))` from identical inputs.
- [ ] State the polarization projection, mask, RCS normalization, and dB floor.
- [ ] Report the nearest-grid angular error for backscatter samples.

## Code map

| Task | Source |
|:--|:--|
| Spherical grid, radiation matrix, and far field | `src/postprocessing/FarField.jl` |
| Power, projected field, RCS, and conditioning diagnostics | `src/postprocessing/Diagnostics.jl` |
| Dense and matrix-free Q operators, polarizations, and masks | `src/optimization/QMatrix.jl` |
| PEC and dielectric Mie references | `src/postprocessing/Mie.jl` |
