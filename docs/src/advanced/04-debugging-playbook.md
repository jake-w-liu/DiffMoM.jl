# Debugging playbook

Use this chapter to collect evidence before assigning a cause. Each check has a
specific output. A failed check narrows the next experiment, but it does not by
itself prove why the failure occurred.

## Triage order

### 1. Check the mesh

Set the boundary policy to match the physical model.

```julia
report = mesh_quality_report(
    mesh; area_tol_rel=1e-12, check_orientation=true)
ok = mesh_quality_ok(
    report; allow_boundary=true, require_closed=false)
println(report)
println("mesh accepted: $ok")
```

If `ok` is false, inspect the reported defect counts before repairing anything.
`repair_mesh_for_simulation` can remove invalid, degenerate, duplicate, and
non-manifold faces and can make neighboring face windings consistent. It cannot
decide whether a boundary or removed feature is physically intentional.

```julia
repaired = repair_mesh_for_simulation(
    mesh;
    allow_boundary=true,
    require_closed=false,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=true,
    auto_drop_nonmanifold=true,
    strict_nonmanifold=true,
)
mesh = repaired.mesh
```

### 2. Check units and electrical size

Geometry is in metres and frequency is in hertz. Compute a characteristic
length directly from the coordinates instead of calling an internal helper.

```julia
c0 = 299792458.0
lambda0 = c0 / freq_hz
mins = [minimum(@view mesh.xyz[i, :]) for i in 1:3]
maxs = [maximum(@view mesh.xyz[i, :]) for i in 1:3]
bbox_diagonal = sqrt(sum(abs2, maxs .- mins))
println("bounding-box diagonal / wavelength = $(bbox_diagonal / lambda0)")

resolution = mesh_resolution_report(mesh, freq_hz)
println(resolution)
```

Compare these values with the intended geometry and frequency. A mismatch is
evidence of a unit or input error. It is not a reason to rescale the model until
the expected physical dimensions are known.

### 3. Check the linear system

Measure the true residual against the original matrix and right-hand side.

```julia
using LinearAlgebra

vnorm = norm(v)
iszero(vnorm) && error("relative residual is undefined for a zero RHS")
relative_residual = norm(Z * I - v) / vnorm
diagnostics = condition_diagnostics(Z)
println("relative residual: $relative_residual")
println(diagnostics)
```

Choose a residual gate for the solver tolerance and reference problem. A large
condition number is a diagnostic, not proof that the matrix caused an observed
field error. On a tractable case, compare direct and GMRES solutions and keep
GMRES true-residual checking enabled.

```julia
I_direct = solve_forward(Z, v; solver=:direct)
I_gmres = solve_forward(
    Z, v;
    solver=:gmres,
    gmres_tol=1e-8,
    check_gmres_convergence=true,
    check_true_residual=true,
)
solution_scale = norm(I_direct)
solution_difference = iszero(solution_scale) ?
    norm(I_direct - I_gmres) : norm(I_direct - I_gmres) / solution_scale
```

If a preconditioner is needed, build one through a supported overload of
`build_nearfield_preconditioner` and compare setup time, fill, iteration count,
and the true residual.

### 4. Check far-field transversality and power

Normalize the radial component by the field magnitude so the result has a clear
scale.

```julia
N_omega = length(grid.w)
E_ff = compute_farfield(G_mat, I, N_omega)
radial_error = maximum(
    abs(dot(@view(grid.rhat[:, q]), @view(E_ff[:, q]))) /
    max(norm(@view(E_ff[:, q])), eps(Float64))
    for q in 1:N_omega
)
rho = energy_ratio(I, v, E_ff, grid)
println("normalized radial error: $radial_error")
println("P_rad / P_in: $rho")
```

For a lossless PEC benchmark, compare `rho` with a threshold established by a
mesh and angular-grid convergence study. For a lossy impedance model, a value
below one may represent absorbed power. Do not reuse a PEC gate for that case.

### 5. Check the quadratic objective

`build_Q` uses the same grid weights, mask, polarization, and radiation matrix
as the direct projected-field sum. Compare the two constructions with identical
inputs.

```julia
mask = trues(N_omega)
Q = build_Q(G_mat, grid, pol_mat; mask=mask)
quadratic_value = real(dot(I, Q * I))
direct_value = sum(
    grid.w[q] * abs2(dot(@view(pol_mat[:, q]), @view(E_ff[:, q])))
    for q in 1:N_omega if mask[q]
)
scale = max(abs(quadratic_value), abs(direct_value))
relative_difference = iszero(scale) ? 0.0 :
    abs(quadratic_value - direct_value) / scale
println("objective relative difference: $relative_difference")
```

A mismatch can come from any input to the two calculations. Rebuild both from
the same variables before changing package code.

### 6. Check the adjoint gradient

Use central finite differences for real objectives that contain conjugation.
Run a step-size sweep because one finite-difference step cannot separate
truncation error from rounding error.

```julia
function objective(theta)
    Z_theta = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
    I_theta = solve_forward(Z_theta, v)
    return compute_objective(I_theta, Q)
end

indices = 1:min(5, length(theta0))
for h in (1e-4, 1e-5, 1e-6, 1e-7)
    errors = map(indices) do p
        g_fd = fd_grad(objective, theta0, p; h=h)
        scale = max(abs(g_adj[p]), abs(g_fd))
        iszero(scale) ? 0.0 : abs(g_adj[p] - g_fd) / scale
    end
    println("h=$h, max symmetric relative error=$(maximum(errors))")
end
```

`complex_step_grad` is only valid when the objective is holomorphic in the
perturbed parameter and real at real input. A quadratic objective such as
`real(I' * Q * I)` contains conjugation, so use finite differences for that
function.

### 7. Compare with an independent reference

Run external or analytical comparisons after the internal inputs and identities
are understood. Record these conventions on both sides:

- The package uses the $e^{+i\omega t}$ time convention and
  $G=e^{-ikR}/(4\pi R)$.
- `k_vec` points in the propagation direction.
- Length is in metres, frequency is in hertz, and RCS is in square metres before
  conversion to dBsm.
- Spherical $\theta$ is measured from positive z and $\phi$ is the xy-plane
  azimuth.
- The meshes, polarization vectors, phase origins, material parameters, and
  observation directions must match.

Compare named observables such as peak direction, level at a specified angle,
or an RCS cut. State the interpolation and dB floor. Deep nulls need a
linear-scale comparison because a small absolute difference can produce a large
dB difference.

## Two kinds of preconditioning

The package has two related but different mechanisms.

Mass-based conditioning constructs

```math
Z_{\mathrm{eff}} = M^{-1}Z, \qquad v_{\mathrm{eff}} = M^{-1}v.
```

If this algebraic transformation is used for an impedance gradient, transform
the derivative blocks too.

```julia
M = make_left_preconditioner(Mp; eps_rel=1e-8)
Z_eff, v_eff, factor = prepare_conditioned_system(
    Z, v; preconditioner_M=M)
Mp_eff, _ = transform_patch_matrices(
    Mp; preconditioner_M=M, preconditioner_factor=factor)
```

The high-level optimizers handle this path when their `preconditioner_M` or
`preconditioning` options are used. Keep the same effective system and
derivative blocks in the forward, adjoint, and gradient calculations.

A near-field GMRES preconditioner is a Krylov solver aid. Pass it through the
`preconditioner` or `nf_preconditioner` keyword. `solve_gmres_adjoint` applies
the corresponding adjoint preconditioner. Do not manually transform `Mp` merely
because GMRES uses a near-field preconditioner.

## Symptom-driven checks

| Symptom | First evidence to collect | Next controlled comparison |
|---|---|---|
| RWG build or assembly throws | `mesh_quality_report` and the exact exception | Repair a copy, then rerun the same precheck |
| Direct or GMRES residual misses its gate | Original-system true residual and `condition_diagnostics` | Direct versus GMRES on a smaller equivalent case |
| Far field has unexpected radial content | Normalized radial error and array shapes | Rebuild `G_mat`, `grid`, and `E_ff` together |
| Power ratio misses its benchmark gate | `P_in`, `P_rad`, mesh level, and grid level | Mesh and angular-grid refinement |
| `I'QI` differs from a direct sum | Shared `G_mat`, `pol_mat`, mask, and weights | Reconstruct both values in one scope |
| Adjoint and finite difference disagree | Step sweep and forward/adjoint residuals | Small problem with conditioning off, then on |
| External solver disagrees | Convention and geometry record | Analytical case or identical-mesh comparison |

This table lists tests, not causes. Report a cause only after a controlled
comparison changes the suspected factor while holding the other inputs fixed.

## Reproducibility packet

Include enough material to rerun the failing command without local state:

1. A mesh fixture or deterministic mesh-construction code.
2. The complete excitation, frequency, material, and solver settings.
3. `mesh_quality_report`, `mesh_resolution_report`, and the RWG count.
4. The original-system true residual and condition diagnostics when a dense
   matrix is available.
5. Far-field grid dimensions, power values, and the failed observable.
6. Gradient step sizes and the compared adjoint and finite-difference values.
7. The Julia version, package commit, thread count, and exact run command.
8. The smallest generated artifact needed to demonstrate the mismatch.

Avoid placeholder output. Attach the values from the failing run and label any
external dependency that is required to reproduce it.

## Commands

Run the regression suite from the repository root:

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
```

Run the analytical PEC sphere validator:

```bash
julia --project=. --startup-file=no validation/mie/validate_mie_rcs.jl
```

Run the gradient and mesh-convergence study:

```bash
julia --project=. --startup-file=no validation/paper/run_convergence_study.jl
```

The aircraft example needs a user-supplied `examples/demo_aircraft.obj`:

```bash
julia --project=. --startup-file=no examples/06_aircraft_rcs.jl
```

For a file elsewhere, set `DMOM_AIRCRAFT_OBJ=/absolute/path/to/model.obj` on
that command.

Use Julia's `Profile` standard library or BenchmarkTools for a bounded hot-path
measurement. Record warm-up, thread count, input sizes, allocations, and the
exact command. A timing without those inputs is not a reusable baseline.

## Code map

| Area | Source | Public entry points |
|---|---|---|
| Mesh diagnostics and repair | `src/geometry/Mesh.jl` | `mesh_quality_report`, `repair_mesh_for_simulation`, `mesh_resolution_report` |
| OBJ input and output | `src/geometry/Mesh.jl` | `read_obj_mesh`, `write_obj_mesh` |
| STL, MSH, and format-dispatch I/O | `src/geometry/MeshIO.jl` | `read_stl_mesh`, `read_msh_mesh`, `read_mesh`, `write_mesh` |
| EFIE assembly and excitation | `src/assembly/EFIE.jl`, `src/assembly/Excitation.jl` | `assemble_Z_efie`, `assemble_excitation` |
| Dense and iterative solves | `src/solver/Solve.jl`, `src/solver/IterativeSolve.jl` | `solve_forward`, `solve_gmres` |
| Near-field preconditioning | `src/solver/NearFieldPreconditioner.jl` | `build_nearfield_preconditioner` |
| Far field, power, and conditioning | `src/postprocessing/FarField.jl`, `src/postprocessing/Diagnostics.jl` | `radiation_vectors`, `compute_farfield`, `energy_ratio`, `condition_diagnostics` |
| Objectives and gradients | `src/optimization/QMatrix.jl`, `src/optimization/Adjoint.jl` | `build_Q`, `solve_adjoint`, `gradient_impedance` |
| Gradient verification | `src/optimization/Verification.jl` | `fd_grad`, `complex_step_grad`, `verify_gradient` |

## Checklist

- [ ] Record the exact input, command, Julia version, commit, and thread count.
- [ ] Check mesh policy, dimensions, units, and resolution before assembly.
- [ ] Measure the true residual against the original system.
- [ ] Normalize transversality checks and use benchmark-specific power gates.
- [ ] Rebuild objective comparisons from the same grid and polarization data.
- [ ] Verify nonholomorphic objectives with a finite-difference step sweep.
- [ ] Distinguish mass conditioning from a near-field Krylov preconditioner.
- [ ] Align conventions before interpreting an external comparison.
