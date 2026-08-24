# Large-problem strategy

Choose a solver path only after measuring the mesh size, electrical
resolution, retained operator storage, preconditioner storage, Krylov
workspace, and postprocessing arrays. There is no hardware-independent maximum
problem size.

## Dense costs

For $N$ RWG unknowns, one dense `ComplexF64` matrix has raw payload

```math
16N^2\ \text{bytes}.
```

`estimate_dense_matrix_gib(N)` returns this value in GiB:

| $N$ | One matrix (GiB) |
|--:|--:|
| 500 | 0.0037 |
| 1,000 | 0.0149 |
| 2,000 | 0.0596 |
| 5,000 | 0.3725 |
| 10,000 | 1.4901 |
| 20,000 | 5.9605 |

For fixed triangle quadrature, dense EFIE assembly performs work proportional
to $N^2$. Dense LU factorization has cubic asymptotic cost. Factorization,
assembly caches, right-hand sides, and later operators require memory beyond
the one-matrix table.

The radiation matrix can be another large dense allocation. For $N_\Omega$
directions, its payload is

```math
16(3N_\Omega)N\ \text{bytes}.
```

Dense Q matrices add another $16N^2$ bytes each. Use
`build_Q_operator` or `apply_Q` when a dense Q output is unnecessary, while
remembering that those paths still retain the supplied radiation matrix.

## Preflight the exact mesh

```julia
using DiffMoM

raw = read_obj_mesh("platform.obj")
scaled = TriMesh(raw.xyz .* 1e-3, copy(raw.tri))
repair = repair_mesh_for_simulation(
    scaled;
    allow_boundary=true,
    require_closed=false,
    auto_drop_nonmanifold=true,
    strict_nonmanifold=true,
)
mesh = repair.mesh

rwg = build_rwg(
    mesh;
    precheck=true,
    allow_boundary=true,
    require_closed=false,
)
N = rwg.nedges

quality = mesh_quality_report(mesh)
resolution = mesh_resolution_report(mesh, frequency)
println((
    unknowns=N,
    one_dense_matrix_gib=estimate_dense_matrix_gib(N),
    quality=quality,
    resolution=resolution,
))
```

Set the scale factor, boundary policy, and frequency from the physical model.
The example's `1e-3` conversion is appropriate only for source coordinates in
millimetres.

Before running, account for:

1. the selected base operator and its construction workspace;
2. direct-factorization or GMRES workspace;
3. near-field or mass preconditioner storage;
4. every retained right-hand side and solution;
5. radiation vectors for the requested directions;
6. dense or matrix-free objective operators; and
7. output and validation artifacts.

Use each API's byte and work limits to fail before its main allocation. Measure
peak resident memory on a smaller representative case because raw-payload
formulas do not include allocator and library overhead.

## Solver paths

The high-level workflow accepts five method values:

| Method | Base representation | Solve |
|:--|:--|:--|
| `:dense_direct` | Dense EFIE matrix | LU |
| `:dense_gmres` | Dense EFIE matrix | GMRES |
| `:aca_gmres` | ACA block operator | GMRES |
| `:mlfma` | MLFMA operator | GMRES |
| `:auto` | Selected by RWG count | Selected with the representation |

`solve_scattering(...; method=:auto)` compares $N$ with
`dense_direct_limit`, `dense_gmres_limit`, and `mlfma_threshold` in that order.
The canonical defaults are in the `solve_scattering` docstring rendered under
[Core docstrings](../api/exported-core.md).

These count thresholds select code paths. They do not certify memory,
convergence, approximation error, or mesh resolution. Override `method` and
the thresholds only with measured evidence.

```julia
result = solve_scattering(
    mesh,
    frequency,
    excitation;
    method=:auto,
    error_on_underresolved=true,
    check_gmres_convergence=true,
    check_true_residual=true,
    gmres_tol=1e-6,
    true_residual_factor=100.0,
    verbose=true,
)
```

For iterative paths, the workflow verifies the true residual against the
selected dense, ACA, or MLFMA operator by default. Setting
`check_gmres_convergence=false` or `check_true_residual=false` is appropriate
only when deliberately inspecting a partial iterate and its diagnostics.

## Compare accelerated methods

ACA and MLFMA avoid the dense base matrix, but their resource use depends on
geometry, frequency, ranks or sampling, near-field structure, and the requested
tolerances. For a tractable reference case, record:

- construction time and peak memory;
- ACA block counts and ranks, or MLFMA level and sampling data;
- preconditioner setup, fill, and memory;
- GMRES iterations, reported residual, and true residual;
- current difference against the reference; and
- differences in the final observable.

`examples/05_solver_methods.jl` compares dense direct, dense GMRES with a
near-field preconditioner, unpreconditioned dense GMRES, and ACA GMRES on one
plate. Its settings and timings belong to that case.

## Two preconditioning mechanisms

### Near-field Krylov preconditioning

`build_nearfield_preconditioner` constructs a solver aid for dense,
matrix-free EFIE, ACA, MLFMA, geometry-based, or preassembled sparse inputs,
depending on the overload. It can use sparse LU, incomplete LU where
supported, or a diagonal factorization.

A preconditioner may reduce iteration count, but it has setup and storage cost.
Compare with and without it while holding the operator, tolerance, and true
residual gate fixed.

### Mass-based algebraic conditioning

Optimization APIs also support a dense mass-based left transformation:

```math
Z_{\mathrm{eff}}=M^{-1}Z,
\qquad
v_{\mathrm{eff}}=M^{-1}v.
```

`make_left_preconditioner` forms

```math
M=\sum_pM_p+\epsilon I.
```

This path leaves a selected dense base matrix dense. For an impedance gradient,
transform every derivative block with `transform_patch_matrices` as
$M^{-1}M_p$. Do not perform this derivative transformation merely because a
GMRES near-field preconditioner is active.

## Geometry coarsening

When the full mesh exceeds the resource envelope, one option is
`coarsen_mesh_to_target_rwg`. It voxel-clusters vertices, repairs candidates,
and returns the closest valid RWG count found within its bounded search.

```julia
coarse = coarsen_mesh_to_target_rwg(
    mesh,
    800;
    max_iters=10,
    allow_boundary=true,
    require_closed=false,
)
mesh_coarse = coarse.mesh
```

The requested count is not an accuracy condition. Coarsening can change small
features, edges, curvature, openings, and bounding dimensions. Record the
achieved count and compare the required observable over several target counts.
Save or hash the exact simulation mesh.

## Reduce scenario cost explicitly

Frequency, incidence, and optimization sweeps multiply the cost of operator
construction and solves. A staged sweep can reduce work, but coarse screening
can miss narrow features. Declare the coarse grid, the rule used to select
refinement regions, and an independently chosen refinement check.

Angular postprocessing has a separate cost. `make_sph_grid(Ntheta, Nphi)`
creates `Ntheta * Nphi` midpoint samples. For a cut, use a small azimuth count
only when its midpoint angle matches the intended quantity, and label the cut
with the stored angle rather than an unstored endpoint.

## Acceptance ladder

For each retained mesh and method, apply checks in this order:

1. mesh quality and the intended boundary policy;
2. electrical-resolution report;
3. operator construction and resource preflights;
4. convergence status and original-operator true residual;
5. far-field transversality and a case-appropriate power check;
6. approximation error against a tractable dense or analytical reference;
7. mesh, quadrature, and angular-grid convergence of the reported observable;
8. repeatability under the recorded thread and library configuration.

Passing one tier does not imply the next.

## Diagnose resource or convergence failures

| Observation | Evidence to collect | Controlled comparison |
|:--|:--|:--|
| Dense allocation is rejected | `N`, requested bytes, API limit | Smaller mesh or explicitly selected accelerated operator |
| GMRES status or true residual fails | Status, iterations, reported and true residuals | Same operator with preconditioner off and on |
| ACA result differs from dense | ACA settings, block/rank data, true residual | Tighten approximation on a tractable fixed mesh |
| MLFMA result differs from dense | Level, sampling, near-field, and solve settings | Refine MLFMA settings on the same fixed mesh |
| Coarsened observable changes | Exact meshes, achieved counts, resolution | Add an intermediate or finer mesh |
| Far-field feature moves | Stored directions and angular error | Refine only the observation grid first |

Do not report a cause until one controlled comparison changes the suspected
factor while the other inputs remain fixed.

## Code map

| Area | Source |
|:--|:--|
| High-level method selection | `src/Workflow.jl` |
| Dense and matrix-free EFIE | `src/assembly/EFIE.jl` |
| ACA | `src/fast/ACA.jl` |
| MLFMA | `src/fast/MLFMA.jl` |
| GMRES | `src/solver/IterativeSolve.jl` |
| Near-field preconditioners | `src/solver/NearFieldPreconditioner.jl` |
| Mass conditioning | `src/solver/Solve.jl` |
| Mesh coarsening and storage estimate | `src/geometry/Mesh.jl` |
| Far-field storage and computation | `src/postprocessing/FarField.jl` |
