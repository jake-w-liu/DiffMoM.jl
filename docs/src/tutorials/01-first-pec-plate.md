# Tutorial 1: First PEC plate scattering run

This tutorial assembles and solves the electric field integral equation for a
PEC plate under plane-wave illumination. It also checks the solve residual,
power balance, and off-surface field decomposition.

## Run the packaged example

From the repository root:

```bash
julia --project=. examples/01_pec_plate_basics.jl
```

The example reports the mesh size, matrix memory estimate, solve residual,
bistatic and backscatter RCS, energy ratio, and dense-matrix condition number.
Record the values produced by your run. The documentation does not substitute a
fixed transcript for runtime output.

For a mesh sweep with executable acceptance gates, run:

```bash
julia --project=. validation/paper/run_convergence_study.jl
```

That driver writes `data/convergence_study.csv` and
`data/gradient_verification.csv`. It exits with an error if a declared energy-
or gradient-error gate fails. The `CONVERGENCE_MAX_*` constants in
`validation/paper/run_convergence_study.jl` own the executable limits.

## Build the problem

Start Julia with the project active, then load the package and `LinearAlgebra`:

```julia
using DiffMoM
using LinearAlgebra

frequency = 3.0e9
c0 = 299792458.0
wavelength = c0 / frequency
k = 2π / wavelength
eta0 = 376.730313668
```

All geometry values in this example are in metres and frequency is in hertz.
Create a $0.1$ m square plate and its RWG basis:

```julia
mesh = make_rect_plate(0.1, 0.1, 5, 5)
rwg = build_rwg(mesh)

println("vertices = ", nvertices(mesh))
println("triangles = ", ntriangles(mesh))
println("RWG unknowns = ", rwg.nedges)
println(mesh_resolution_report(mesh, frequency))
```

`make_rect_plate` places the plate in the xy plane. `build_rwg` creates one
unknown for each eligible shared edge.

## Assemble the EFIE and excitation

```julia
Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)

k_vec = Vec3(0.0, 0.0, -k)
polarization = Vec3(1.0, 0.0, 0.0)
excitation = make_plane_wave(k_vec, 1.0, polarization)
v = assemble_excitation(mesh, rwg, excitation; quad_order=3)
```

The wave propagates in the negative z direction and is polarized along x. The
polarization must be transverse to `k_vec`.

The EFIE matrix is dense. Its element payload alone is approximately
`16 * rwg.nedges^2` bytes for `ComplexF64`. Factorization and assembly
workspaces require additional memory. Check the package estimate before
increasing the mesh:

```julia
println("matrix payload estimate (GiB) = ",
        estimate_dense_matrix_gib(rwg.nedges))
```

## Solve and check the residual

```julia
I = solve_forward(Z, v; solver=:direct)
relative_residual = norm(Z * I - v) / norm(v)

isfinite(relative_residual) || error("non-finite solve residual")
println("relative residual = ", relative_residual)
```

The residual checks the algebraic solve, not the accuracy of the mesh,
quadrature, or physical model. Choose a residual tolerance for the study before
interpreting the result. If it fails, preserve the value and inspect
`condition_diagnostics(Z)` before changing the solver.

## Compute the far field and power balance

```julia
grid = make_sph_grid(18, 36)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
E_ff = compute_farfield(G, I, length(grid.w))

sigma = bistatic_rcs(E_ff; E0=1.0)
backscatter = backscatter_rcs(E_ff, grid, k_vec; E0=1.0)
ratio = energy_ratio(I, v, E_ff, grid; eta0=eta0)

all(isfinite, sigma) || error("non-finite bistatic RCS")
isfinite(backscatter.sigma) || error("non-finite backscatter RCS")
isfinite(ratio) || error("non-finite energy ratio")

println("RCS range = ", extrema(sigma))
println("backscatter RCS = ", backscatter.sigma)
println("P_rad / P_in = ", ratio)
```

For a lossless PEC case, `energy_ratio` should converge toward one as the mesh,
triangle quadrature, and spherical quadrature are refined. A small solve residual
does not guarantee that power balance has converged.

Use a controlled refinement study:

1. keep the geometry and frequency fixed;
2. refine the mesh and record the metric of interest;
3. refine the spherical grid independently;
4. compare supported triangle quadrature orders; and
5. accept a result only when it meets tolerances chosen for the application.

## Sample scattered and total fields

Observation points must remain off the surface. Use the same excitation object
that produced the right-hand side:

```julia
points = [
    Vec3(0.0, 0.0, 0.15),
    Vec3(0.02, 0.0, 0.18),
]

E_sca = compute_nearfield(
    mesh, rwg, I, points, k;
    quad_order=3,
    eta0=eta0,
)
E_total = compute_total_field(
    mesh, rwg, I, excitation, points, k;
    quad_order=3,
    eta0=eta0,
)

E_incident = hcat((
    plane_wave_field(point, k_vec, 1.0, polarization)
    for point in points
)...)
field_decomposition_error =
    norm((E_total - E_sca) - E_incident) / max(norm(E_incident), eps())
println("field decomposition error = ", field_decomposition_error)
```

`compute_nearfield` returns the scattered field. `compute_total_field` returns
the incident field plus the scattered field. On-surface evaluation is not
supported.

For an analytical local-field benchmark, see the
[near and total field Rayleigh-sphere validation](../validation/06-near-total-field-rayleigh-sphere.md)
and `examples/21_near_total_field_rayleigh_sphere.jl`.

## Diagnose a failed check

Follow this order so that one change tests one hypothesis:

1. Confirm the units and calculate the electrical size.
2. Run `mesh_quality_report(mesh)` and `mesh_resolution_report(mesh, frequency)`.
3. Record the true residual and `condition_diagnostics(Z)`.
4. Refine the spherical grid while holding the solved current fixed.
5. Refine the mesh and reassemble the system.
6. Compare supported triangle quadrature orders.

Do not assign a cause from the residual, condition number, or energy ratio alone.
The [debugging playbook](../advanced/04-debugging-playbook.md) gives executable
checks for each stage.

## Complete script

The executable version is `examples/01_pec_plate_basics.jl`. Use it as the
starting point for changes to frequency, plate size, mesh density, incidence,
or polarization. Re-run the same residual and convergence checks after each
change.

## Code map

| Task | Source |
|:--|:--|
| Plate mesh and mesh reports | `src/geometry/Mesh.jl` |
| RWG basis construction | `src/basis/RWG.jl` |
| EFIE assembly and excitation | `src/assembly/EFIE.jl`, `src/assembly/Excitation.jl` |
| Direct and GMRES solves | `src/solver/Solve.jl` |
| Near and total fields | `src/postprocessing/NearField.jl` |
| Far field | `src/postprocessing/FarField.jl` |
| Power, RCS, and conditioning diagnostics | `src/postprocessing/Diagnostics.jl` |
| Mesh and gradient sweep | `validation/paper/run_convergence_study.jl` |
