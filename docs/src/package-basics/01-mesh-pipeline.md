# Mesh pipeline

Use this workflow to import or generate a triangle mesh, apply an explicit
boundary policy, repair supported defects, build RWG basis functions, and
preflight resource use.

## Mesh and RWG requirements

`TriMesh` stores vertex coordinates and triangle connectivity:

```julia
struct TriMesh
    xyz::Matrix{Float64}  # size (3, number of vertices)
    tri::Matrix{Int}      # size (3, number of triangles)
end
```

Each standard RWG basis function is associated with an edge shared by exactly
two triangles. `build_rwg` skips boundary edges and rejects invalid,
degenerate, duplicate, non-manifold, or orientation-conflicting topology when
its default precheck is enabled.

Choose the boundary policy from the physical model:

| Model | `allow_boundary` | `require_closed` |
|:--|:--:|:--:|
| Open sheet or plate | `true` | `false` |
| Closed surface | `false` | `true` |

`require_closed=true` rejects any boundary edge. It does not determine whether
the connected components or enclosed volume match the intended geometry.

## Generate or import a mesh

The package includes analytical plate, disk, and reflector constructors:

```julia
plate = make_rect_plate(0.2, 0.1, 12, 6)
disk = make_circular_plate(0.05, 8, 48)
reflector = make_parabolic_reflector(0.3, 0.12, 12, 72)
```

For OBJ input:

```julia
mesh = read_obj_mesh("geometry.obj")
```

`read_obj_mesh` accepts `v` records and faces with at least three vertices.
Polygon faces are fan-triangulated. Positive and negative OBJ vertex indices
are supported; texture and normal indices are ignored. The reader enforces
input-size, line-size, vertex-count, triangle-count, and raw-output limits.

STL and Gmsh MSH readers, plus format-dispatch helpers, are defined in
`src/geometry/MeshIO.jl`.

OBJ has no mandatory unit. Scale coordinates explicitly before topology repair
when the source file is not in metres:

```julia
scale_to_m = 1e-3
mesh_m = TriMesh(mesh.xyz .* scale_to_m, copy(mesh.tri))
```

Record the scale factor. A topology report cannot detect a unit error.

## Inspect mesh quality

```julia
report = mesh_quality_report(
    mesh_m;
    area_tol_rel=1e-12,
    check_orientation=true,
)

println(report)
accepted = mesh_quality_ok(
    report;
    allow_boundary=true,
    require_closed=false,
)
```

The returned named tuple contains:

- vertex, triangle, total-edge, interior-edge, and boundary-edge counts;
- non-finite vertex and invalid-triangle indices;
- degenerate and duplicate triangle indices;
- non-manifold-edge and orientation-conflict counts; and
- the mesh scale and absolute area tolerance used by the check.

The area threshold is
$\texttt{area_tol_rel}\,L_{\mathrm{bbox}}^2$. A report does not measure
connected components, genus, absolute outward orientation, geometric fidelity,
or electromagnetic accuracy.

To fail with a detailed message instead of receiving a Boolean:

```julia
report = assert_mesh_quality(
    mesh_m;
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
)
```

## Repair supported defects

```julia
repair = repair_mesh_for_simulation(
    mesh_m;
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=true,
    auto_drop_nonmanifold=true,
    strict_nonmanifold=true,
)
mesh_repaired = repair.mesh
```

The repair routine can:

- remove invalid and degenerate triangles;
- remove later duplicate faces regardless of winding;
- compact unreferenced vertices;
- drop triangles incident to non-manifold edges when
  `auto_drop_nonmanifold=true`; and
- make neighboring triangle windings consistent when
  `fix_orientation=true`.

`strict_nonmanifold=true` rejects the result if such edges remain after the
selected cleanup. Orientation propagation does not determine the absolute
outward direction of a component. Removing faces can also remove an intended
feature, so inspect the returned `before`, `cleaned`, and `after` reports and
the repaired geometry.

The result also records removed triangle and vertex indices, the old-to-new
vertex map, non-manifold removal count, flipped triangles, and the area
tolerance. For file-to-file repair:

```julia
repair_obj_mesh(
    "input.obj",
    "repaired.obj";
    allow_boundary=true,
    require_closed=false,
)
```

## Build RWG basis functions

```julia
rwg = build_rwg(
    mesh_repaired;
    precheck=true,
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
)

println((triangles=ntriangles(mesh_repaired), unknowns=rwg.nedges))
```

For a valid non-periodic mesh, `rwg.nedges` is the number of edges with exactly
two incident triangles. Boundary edges do not receive half-RWG functions.

## Check electrical resolution

```julia
frequency = 3.0e9
resolution = mesh_resolution_report(
    mesh_repaired,
    frequency;
    points_per_wavelength=10.0,
)

println(resolution)
@assert mesh_resolution_ok(resolution; criterion=:max)
```

The report records minimum, median, mean, 95th-percentile, and maximum unique
edge lengths, together with their wavelength ratios. The default acceptance
uses the maximum edge against `wavelength / points_per_wavelength`.

This geometric rule is a preflight, not an observable-error bound. Establish
accuracy with mesh and quadrature convergence or an analytical reference.

## Estimate dense storage

One dense `N x N` `ComplexF64` matrix has a raw payload of

```math
16N^2\ \text{bytes}.
```

```julia
matrix_gib = estimate_dense_matrix_gib(rwg.nedges)
println((unknowns=rwg.nedges, one_matrix_gib=matrix_gib))
```

This value excludes factorization, assembly caches, right-hand sides,
preconditioners, radiation matrices, Q matrices, and runtime overhead. Use the
resource limits on the selected APIs and measure peak memory for the complete
workflow.

## Coarsen toward a target count

```julia
coarse = coarsen_mesh_to_target_rwg(
    mesh_repaired,
    500;
    max_iters=10,
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
    strict_nonmanifold=true,
)

mesh_simulation = coarse.mesh
println((
    requested=coarse.target_rwg,
    achieved=coarse.rwg_count,
    gap=coarse.best_gap,
    iterations=coarse.iterations,
))
```

The routine voxel-clusters vertices, cleans and repairs each candidate, and
counts its interior edges. It returns immediately when the input is no more
than 15 percent above the target. During the search it returns on a candidate
within 15 percent, or otherwise returns the closest valid candidate found in
`max_iters`.

The target is not an exact-count promise. Voxel clustering can alter curved
surfaces, sharp edges, gaps, and small features. Save the achieved count,
inspect the returned mesh, and compare the required observable across a bounded
sequence of target counts.

## Complete reusable function

```julia
function prepare_obj(
    input_path,
    output_path;
    scale_to_m=1.0,
    target_rwg=nothing,
    allow_boundary=true,
    require_closed=false,
)
    raw = read_obj_mesh(input_path)
    scaled = TriMesh(raw.xyz .* scale_to_m, copy(raw.tri))
    repaired = repair_mesh_for_simulation(
        scaled;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
        auto_drop_nonmanifold=true,
        strict_nonmanifold=true,
    )

    prepared = repaired.mesh
    initial_rwg = build_rwg(
        prepared;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
    )

    coarsening = nothing
    if target_rwg !== nothing && initial_rwg.nedges > target_rwg
        coarsening = coarsen_mesh_to_target_rwg(
            prepared,
            target_rwg;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
        )
        prepared = coarsening.mesh
    end

    final_report = assert_mesh_quality(
        prepared;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
    )
    final_rwg = build_rwg(
        prepared;
        allow_boundary=allow_boundary,
        require_closed=require_closed,
    )
    write_obj_mesh(output_path, prepared)

    return (
        mesh=prepared,
        rwg=final_rwg,
        report=final_report,
        repair=repaired,
        coarsening=coarsening,
    )
end
```

## Visual inspection

```julia
preview = save_mesh_preview(
    mesh_repaired,
    mesh_simulation,
    "figures/mesh_comparison";
    title_a="Repaired",
    title_b="Simulation mesh",
)

println((png=preview.png_path, pdf=preview.pdf_path))
```

The preview is a geometry check, not a topology or convergence test.

## Code map

| Area | Source |
|:--|:--|
| Analytical meshes, OBJ I/O, quality, repair, coarsening, and resolution | `src/geometry/Mesh.jl` |
| STL, MSH, and dispatch I/O | `src/geometry/MeshIO.jl` |
| RWG construction | `src/basis/RWG.jl` |
| Mesh previews | `src/postprocessing/Visualization.jl` |
| Imported-platform example | `examples/06_aircraft_rcs.jl` |

## Pre-solve checklist

- [ ] Record source units and the applied scale factor.
- [ ] Apply the intended open or closed boundary policy.
- [ ] Inspect every repair removal and the resulting geometry.
- [ ] Run `assert_mesh_quality` on the exact simulation mesh.
- [ ] Record the achieved RWG count and electrical-resolution report.
- [ ] Preflight the complete solver and postprocessing memory, not one matrix
  alone.
- [ ] Establish mesh and quadrature convergence for the reported observable.
