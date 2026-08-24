# Fields, Q objectives, and RCS API

This page connects solved RWG coefficients to near fields, far fields,
quadratic objectives, power diagnostics, and radar cross section. Exact
signatures and defaults are rendered from the source under
[Optimization docstrings](exported-optimization.md) and
[Postprocessing docstrings](exported-postprocessing.md).

For the complete data flow and conventions, see
[Far field, Q objectives, and RCS](../package-basics/03-farfield-q-rcs.md).

## Spherical sampling and radiation vectors

```julia
using DiffMoM

grid = make_sph_grid(36, 72)
G = radiation_vectors(
    mesh,
    rwg,
    grid,
    k;
    quad_order=3,
)
E_ff = compute_farfield(G, current, length(grid.w))
```

For $N$ RWG coefficients and $N_\Omega$ directions:

- `grid.rhat` has shape `(3, N_omega)`;
- `G` has shape `(3 * N_omega, N)`; and
- `E_ff` has shape `(3, N_omega)`.

`make_sph_grid` stores midpoint samples rather than the polar and azimuthal
endpoints. Label a cut with its stored angle. Its point-count and raw-payload
limits are checked before allocation.

`radiation_vectors` returns the asymptotic amplitude used in

```math
\mathbf E^{\mathrm{sca}}(R\hat{\mathbf r})
\sim \frac{e^{-ikR}}{R}\mathbf E_\infty(\hat{\mathbf r}).
```

The radiation matrix is dense. Its raw `ComplexF64` payload is
$16(3N_\Omega)N$ bytes, before quadrature caches and workspace. The API exposes
limits for output bytes, total work bytes, interaction terms, and exceptional
high-precision work.

## Incident-source far field

`incident_farfield(source, direction, k)` returns the asymptotic incident-field
amplitude for finite radiating sources. Supported sources are monopoles,
electric or magnetic dipoles, loops, pattern feeds, and weighted combinations
of those models.

A plane wave has no decaying $1/R$ radiation amplitude, so this function
rejects it. Ports, delta gaps, and imported local RHS models are also outside
this API.

## Near and total electric fields

`compute_nearfield` evaluates the scattered electric field from the solved RWG
coefficients. `compute_total_field` adds a pointwise incident field from the
same excitation model used to assemble the right-hand side.

```julia
points = [
    Vec3(0.0, 0.0, 0.15),
    Vec3(0.02, 0.0, 0.18),
]

E_sca = compute_nearfield(mesh, rwg, current, points, k)
E_total = compute_total_field(
    mesh,
    rwg,
    current,
    excitation,
    points,
    k,
)
```

Accepted observation inputs are one `Vec3`, a vector of `Vec3` values, or a
real matrix of shape `(3, Nobs)`. One point returns `CVec3`; multiple points
return a `(3, Nobs)` `Matrix{ComplexF64}`.

Pointwise total-field evaluation supports plane waves, dipoles, loops,
monopoles, pattern feeds, electric-field imports, and weighted combinations in
which every child supports the operation. It rejects ports, delta gaps, and
surface-current-density imports because those models define a right-hand side,
not an incident electric field at an arbitrary observation point.

On-surface evaluation is unsupported. With surface checks active, observation
points on the mesh are rejected. Nearby off-surface points use the package's
singularity-subtracted quadrature. Work-byte, interaction-count, and
exceptional-precision limits bound the direct evaluation.

## Dense and matrix-free Q objectives

For a polarization matrix `pol` and optional direction mask, the package uses

```math
Q=G^\dagger W G,
\qquad
J=I^\dagger QI,
```

where the projection onto `pol[:, q]` is included in the per-direction rows of
the product.

```julia
pol = pol_linear_x(grid)
mask = direction_mask(
    grid,
    Vec3(sind(30.0), 0.0, cosd(30.0));
    half_angle=deg2rad(5.0),
)

Q = build_Q(G, grid, pol; mask=mask)
value = real(dot(current, Q * current))

Q_operator = build_Q_operator(G, grid, pol; mask=mask)
QI = apply_Q(G, grid, pol, current; mask=mask)
```

Use:

- `build_Q` when callers need the dense Hermitian matrix;
- `build_Q_operator` when repeated products should avoid dense Q storage; and
- `apply_Q` for a single product.

All three retain the supplied radiation matrix and polarization data. The
dense builder also allocates projected and weighted-projected work matrices and
the final Q matrix; its work limit accounts for all three.

For ordinary finite inputs, dense construction uses a BLAS matrix product.
Entries whose value is small relative to a Cauchy--Schwarz norm bound are
recomputed with compensated summation. Exceptional non-finite or
range-sensitive inputs use the bounded high-precision path. The lower triangle
is copied as the exact conjugate of the upper triangle.

`FarFieldQMatrix` implements `AbstractMatrix{ComplexF64}`, indexed access,
`mul!`, and a lock-protected reusable workspace.
`SumQMatrix` represents the lazy sum of two same-size Q operators.

`pol_linear_x` and `pol_linear_y` return the spherical $\hat\theta$ and
$\hat\phi$ projections used by the package. `cap_mask` selects a cone about
positive z; `direction_mask` accepts and normalizes any finite nonzero target
direction.

## Power and conditioning diagnostics

```julia
P_rad = radiated_power(E_ff, grid)
P_in = input_power(current, rhs)
ratio = energy_ratio(current, rhs, E_ff, grid)
projected = projected_power(E_ff, grid, pol; mask=mask)
condition = condition_diagnostics(Z)
```

`radiated_power` includes the $1/(2\eta_0)$ factor. `projected_power` is the
weighted polarization projection used by Q objectives and omits that factor.
Compare it with `real(dot(current, Q * current))`, not directly with radiated
power in watts.

For a lossless PEC benchmark, power ratio should approach one under mesh,
triangle-quadrature, and spherical-grid refinement. A resistive surface can
absorb power, so it needs a model-specific gate.

## Radar cross section

```julia
sigma = bistatic_rcs(E_ff; E0=1.0)
back = backscatter_rcs(E_ff, grid, incident_direction; E0=1.0)
```

`bistatic_rcs` returns one linear RCS value per stored direction.
`backscatter_rcs` accepts the incident propagation direction and returns the
stored sample nearest its negative, including the selected angles and angular
error. Refine the grid or use a separately evaluated direction when that error
matters.

State the incident amplitude and any logarithmic floor when converting to
dBsm. Retain linear values near pattern nulls.

## Analytical references

`mie_s1s2_pec` and `mie_bistatic_rcs_pec` provide the PEC-sphere reference.
`mie_s1s2_dielectric` and `mie_bistatic_rcs_dielectric` cover homogeneous
dielectric or magnetodielectric spheres. Their direction, polarization,
material, truncation, and work-limit contracts are in the source docstrings.

Run the repository PEC benchmark with:

```bash
julia --project=. --startup-file=no validation/mie/validate_mie_rcs.jl
```

## Source map

| Area | Source |
|:--|:--|
| Spherical grids, radiation vectors, and incident far fields | `src/postprocessing/FarField.jl` |
| Scattered and total fields | `src/postprocessing/NearField.jl` |
| Power, conditioning, and RCS | `src/postprocessing/Diagnostics.jl` |
| Dense and matrix-free Q operators | `src/optimization/QMatrix.jl` |
| Mie references | `src/postprocessing/Mie.jl` |
