# Tutorial: RCS from an imported platform mesh

`examples/06_aircraft_rcs.jl` demonstrates OBJ import, repair, bounded
coarsening, dense EFIE solution, and RCS diagnostics. The repository does not
include an aircraft model.

## Prepare the input

Place a model at:

```text
examples/demo_aircraft.obj
```

The script exits with status 1 if that file is absent. It has no command-line
options, so copy the script when changing its path or fixed parameters.

OBJ coordinates are interpreted as metres. If the source file uses another
unit, scale a copy of its coordinate matrix before repair:

```julia
using DiffMoM

raw = read_obj_mesh(input_path)
scale_to_metres = 1.0e-3
scaled = TriMesh(raw.xyz .* scale_to_metres, copy(raw.tri))
```

Do not infer the unit from the OBJ format; it does not encode one.

## Run the example

```bash
julia --project=. examples/06_aircraft_rcs.jl
```

The fixed workflow is:

```text
read OBJ -> repair -> optional coarsening -> resolution check ->
dense EFIE solve -> far field -> RCS and power diagnostics
```

The top-level `target_rwg` assignment and its coarsening condition in
`examples/06_aircraft_rcs.jl` own the example's resource bound. It is a runtime
bound, not an accuracy target.

The script's `freq` assignment and under-resolution branch own its frequency
policy. That branch lowers the frequency; it does not refine the mesh. Read the
effective frequency and edge-to-wavelength ratio from the run output.

## Read the reported values

The example prints:

- the raw, repaired, and final mesh counts;
- repair counters;
- maximum edge length relative to wavelength;
- EFIE assembly time;
- direct-solve time and relative residual;
- minimum, maximum, and mean sampled RCS in dBsm;
- the nearest-grid backscatter sample;
- `P_rad / P_in`; and
- `cond(Z)`.

These are measurements, not executable acceptance gates. The script does not
fail on a large residual, power mismatch, poor condition number, or unstable
RCS. Add application-specific gates before using it as a validation driver.

## Adapt the preprocessing

### Repair

The example calls:

```julia
repair = repair_mesh_for_simulation(
    scaled;
    allow_boundary=true,
    auto_drop_nonmanifold=true,
)
mesh_repaired = repair.mesh
```

The other defaults remove invalid, degenerate, and duplicate triangles, compact
unreferenced vertices, repair neighboring winding conflicts, and reject any
non-manifold edges left after cleanup.

`allow_boundary=true` permits open surfaces. That may fit a sheet or an
intentionally open platform model, but it is not a substitute for checking
whether holes are physically intended.

Inspect at least these returned fields:

```julia
println(repair.before)
println(repair.after)
println(repair.removed_invalid)
println(repair.removed_degenerate)
println(repair.removed_duplicate)
println(repair.removed_nonmanifold)
println(repair.flipped_triangles)
```

Orientation repair makes adjacent triangles consistent. It does not determine
which side of each connected component is outward.

### Coarsening

```julia
coarse = coarsen_mesh_to_target_rwg(
    mesh_repaired,
    target_rwg;
    max_iters=10,
    allow_boundary=true,
    require_closed=false,
)
mesh_sim = coarse.mesh
```

The coarsener clusters vertices into voxels, repairs each candidate, and
returns the closest valid count it found. An input no more than 15 percent over
the target is returned unchanged. A candidate within 15 percent of the target
ends the search early. Otherwise, the closest candidate after `max_iters` is
returned.

Record `coarse.rwg_count`, `coarse.best_gap`, and `coarse.iterations`; do not
assume the requested count was reached. Voxel clustering can remove small
features and alter curved surfaces.

### Memory preflight

```julia
rwg = build_rwg(mesh_sim; allow_boundary=true)
matrix_gib = estimate_dense_matrix_gib(rwg.nedges)
println("one dense ComplexF64 matrix: $matrix_gib GiB")
```

This estimate covers one matrix payload. Factorization, assembly workspace,
vectors, and radiation matrices require additional memory. Use the resource
limits on the selected APIs and measure the complete workflow before increasing
the demonstration size.

## Solve and sample RCS

The example uses a plane wave propagating along `-z` with x polarization:

```julia
k_vec = Vec3(0.0, 0.0, -k)
source = make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0))
Z = assemble_Z_efie(mesh_sim, rwg, k)
v = assemble_excitation(mesh_sim, rwg, source)
I = solve_forward(Z, v)
```

It samples `make_sph_grid(36, 72)`. Because this is a midpoint grid, exact
backscatter may lie between samples. In an adapted script, keep the angular
error returned by `backscatter_rcs`:

```julia
grid = make_sph_grid(36, 72)
G = radiation_vectors(mesh_sim, rwg, grid, k)
E_ff = compute_farfield(G, I, length(grid.w))
sigma = bistatic_rcs(E_ff; E0=1.0)
back = backscatter_rcs(E_ff, grid, k_vec; E0=1.0)

println(back.sigma)
println(back.angular_error_deg)
```

## Establish a usable mesh

Choose a bounded ladder of target counts that fits the resource budget. For
each completed level, retain:

- the actual mesh and RWG counts;
- mesh-quality and resolution reports;
- solve residual and condition diagnostics;
- the same linear-scale RCS observables;
- angular sampling and backscatter angular error; and
- runtime and peak memory.

Compare every level with the finest completed level. A stable result for one
observable does not establish convergence for the whole pattern.

## Optional artifacts

The example writes no result files. Add explicit exports when needed:

```julia
write_obj_mesh("platform_sim.obj", mesh_sim)
preview = save_mesh_preview(
    mesh_repaired,
    mesh_sim,
    "figures/platform_mesh";
    title_a="Repaired",
    title_b="Simulation mesh",
)
```

`save_mesh_preview` writes both PNG and PDF files. Keep its rendered output with
the numerical quality reports; a visual check does not replace topology or
resolution gates.

## Source map

| Task | Source |
|---|---|
| OBJ import and export | `src/geometry/Mesh.jl` |
| Repair and coarsening | `src/geometry/Mesh.jl` |
| RWG construction | `src/basis/RWG.jl` |
| EFIE assembly | `src/assembly/EFIE.jl` |
| RCS and power diagnostics | `src/postprocessing/Diagnostics.jl` |
| Mesh preview | `src/postprocessing/Visualization.jl` |
| Fixed demonstration | `examples/06_aircraft_rcs.jl` |
