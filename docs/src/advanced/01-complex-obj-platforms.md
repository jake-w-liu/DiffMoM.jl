# Complex OBJ platforms

Large CAD exports need two separate decisions before a solve:

1. Is the imported surface structurally acceptable for RWG discretization?
2. Does the chosen mesh and observable fit the accuracy and resource budget?

Repair answers the first question. Resolution, convergence, and measured memory
answer the second.

## Establish the geometry boundary

Load the file with explicit resource limits appropriate to the expected model:

```julia
using DiffMoM

raw = read_obj_mesh(
    "platform.obj";
    max_input_bytes=500_000_000,
    max_vertices=2_000_000,
    max_triangles=4_000_000,
    max_raw_bytes=250_000_000,
    max_line_bytes=1_048_576,
)
```

OBJ has no unit field. Measure the bounding box and apply a documented scale
factor before interpreting frequency or wavelength:

```julia
mins = vec(minimum(raw.xyz; dims=2))
maxs = vec(maximum(raw.xyz; dims=2))
println("raw extents: ", maxs - mins)

scale_to_metres = 1.0e-3
scaled = TriMesh(raw.xyz .* scale_to_metres, copy(raw.tri))
```

A plausible picture is not enough to establish the scale. Compare the measured
dimensions with the source model's declared units.

## Apply a deliberate topology policy

An open sheet and a closed vehicle shell need different gates.

For an intentionally open surface:

```julia
repair = repair_mesh_for_simulation(
    scaled;
    allow_boundary=true,
    require_closed=false,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=true,
    auto_drop_nonmanifold=true,
    strict_nonmanifold=true,
)
mesh = repair.mesh
```

For a body that must be watertight, set `allow_boundary=false` and
`require_closed=true`. The stricter call prevents a damaged shell from being
silently treated as an intentional open surface.

Review every removal count and both `repair.before` and `repair.after`.
Automatic non-manifold cleanup deletes all faces attached to an over-shared
edge. That may remove a junction that should instead be remodeled.

Connected-component reporting is outside the mesh-repair API. Compute or
inspect connectivity separately when detached CAD fragments are possible.

## Preflight the solve, not one matrix

```julia
rwg = build_rwg(mesh; precheck=true, allow_boundary=true)
println("RWG unknowns: ", rwg.nedges)
println("one dense matrix: ", estimate_dense_matrix_gib(rwg.nedges), " GiB")
```

`estimate_dense_matrix_gib` covers the raw payload of one dense
`ComplexF64` matrix. A complete dense workflow also needs factorization,
assembly workspace, right-hand sides, and postprocessing storage.

Far-field radiation vectors can dominate a workflow that otherwise fits. Their
dense output scales with both RWG count and observation count. Use the
`max_output_bytes` and `max_work_bytes` arguments on the relevant API and reduce
the angular grid only when the resulting sampling still meets the observable's
accuracy requirement.

## Choose a path from measured constraints

### Dense path

Use dense assembly only when its API preflight and measured peak memory fit the
environment. `solve_scattering` also enforces `max_dense_matrix_bytes` before a
dense matrix allocation.

### Coarsened dense path

```julia
coarse = coarsen_mesh_to_target_rwg(
    mesh,
    target_rwg;
    max_iters=10,
    allow_boundary=true,
    require_closed=false,
)
mesh_coarse = coarse.mesh
```

Treat the returned count as the fact. Voxel clustering can lose tips, seams,
slots, and narrow gaps. A coarsening ladder must compare the declared RCS or
field observable, not only triangle counts.

### ACA or MLFMA path

Keep the repaired mesh and select an accelerated operator when its measured
accuracy and memory fit better:

```julia
result = solve_scattering(
    mesh,
    freq,
    source;
    method=:auto,
    check_resolution=true,
    check_true_residual=true,
)
```

Automatic selection uses configured RWG-count thresholds. It does not prove
that the selected ACA tolerance, MLFMA settings, preconditioner, or mesh is
adequate. Record the selected method, setup memory, iteration count, convergence
status, and true residual.

## Build a convergence record

For each mesh or solver setting, retain:

| Evidence | Why it is needed |
|---|---|
| Imported and simulation geometry hashes | Identify the actual surfaces |
| Scale factor and bounding box | Establish physical dimensions |
| Repair configuration and before/after reports | Record topology changes |
| Actual triangle and RWG counts | Avoid treating targets as achieved values |
| Frequency and resolution report | Establish electrical sampling |
| Operator and solver configuration | Reproduce the discrete problem |
| Convergence status and true residual | Check the linear solve |
| Angular grid and nearest-direction error | Interpret sampled RCS |
| Peak memory and elapsed times | Support resource claims |
| Linear and dB observables | Separate null-floor effects from field changes |

Use filenames or a manifest schema chosen by the calling project. The package
does not impose an artifact layout for platform studies.

## Visual checks

```julia
preview = save_mesh_preview(
    mesh,
    mesh_coarse,
    "figures/platform_mesh";
    title_a="Repaired",
    title_b="Coarsened",
)
```

The PNG/PDF comparison can expose scale mistakes, missing components, and gross
feature loss. Keep numerical mesh reports alongside it; rendering does not test
manifold topology, orientation direction, or wavelength resolution.

## Example solve skeleton

```julia
freq = 1.0e9
k = 2pi * freq / 299792458.0
source = make_plane_wave(
    Vec3(0.0, 0.0, -k),
    1.0,
    Vec3(1.0, 0.0, 0.0),
)

result = solve_scattering(
    mesh,
    freq,
    source;
    method=:auto,
    check_resolution=true,
    check_gmres_convergence=true,
    check_true_residual=true,
)
```

For a new platform, define explicit residual, mesh, angular, power, and
observable-convergence gates in the study driver. The generic example
`examples/06_aircraft_rcs.jl` prints diagnostics but does not impose those
application-level thresholds.

## Source map

| Task | Source |
|---|---|
| OBJ import and export | `src/geometry/Mesh.jl` |
| Other mesh formats | `src/geometry/MeshIO.jl` |
| Quality, repair, and coarsening | `src/geometry/Mesh.jl` |
| High-level solver selection | `src/Workflow.jl` |
| ACA | `src/fast/ACA.jl` |
| MLFMA | `src/fast/MLFMA.jl` |
| Mesh previews | `src/postprocessing/Visualization.jl` |
| Platform demonstration | `examples/06_aircraft_rcs.jl` |
