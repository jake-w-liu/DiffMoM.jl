# Tutorial: OBJ diagnosis and repair

RWG construction assumes valid triangle connectivity, manifold interior edges,
and consistent neighboring winding. Use the mesh-quality and repair APIs before
assembling a system from imported OBJ geometry.

Repair establishes the checked topology. It does not certify physical scale,
watertight intent, outward normals, geometric fidelity, or electromagnetic
resolution.

## 1. Import with a resource policy

```julia
using DiffMoM

mesh = read_obj_mesh(
    "input.obj";
    max_vertices=5_000_000,
    max_triangles=10_000_000,
    max_raw_bytes=536_870_912,
    max_input_bytes=1_073_741_824,
    max_line_bytes=1_048_576,
)
```

The defaults shown above bound file size, declared mesh growth, and line length
before unbounded input can consume memory. Lower them when the expected model is
smaller. OBJ files do not encode physical units; scale coordinates explicitly
before repair when the source unit is not metres.

```julia
scale_to_metres = 1.0e-3
mesh = TriMesh(mesh.xyz .* scale_to_metres, copy(mesh.tri))
```

## 2. Inspect the raw mesh

```julia
report = mesh_quality_report(mesh; area_tol_rel=1e-12)

println(report.n_vertices)
println(report.n_triangles)
println(report.n_invalid_triangles)
println(report.n_degenerate_triangles)
println(report.n_duplicate_triangles)
println(report.n_boundary_edges)
println(report.n_nonmanifold_edges)
println(report.n_orientation_conflicts)
```

The report counts structural problems; it does not find disconnected
components or decide whether a boundary is intentional. `mesh_quality_ok`
applies a boundary policy to an existing report:

```julia
open_surface_ok = mesh_quality_ok(
    report;
    allow_boundary=true,
    require_closed=false,
)
```

Typical interpretations are:

| Finding | Effect on the normal solver path |
|---|---|
| Invalid vertex index or repeated index within a face | Rejected |
| Degenerate triangle | Rejected |
| Duplicate face in any winding | Rejected |
| Edge shared by more than two faces | Rejected |
| Neighboring faces traversing a shared edge in the same direction | Rejected |
| Boundary edge | Allowed only when the selected boundary policy permits it |

## 3. Repair in memory

For an intentionally open surface:

```julia
result = repair_mesh_for_simulation(
    mesh;
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=true,
    auto_drop_nonmanifold=true,
    strict_nonmanifold=true,
)
mesh_repaired = result.mesh
```

The function performs these operations in order:

1. report the original mesh;
2. remove invalid and degenerate triangles when enabled;
3. remove duplicate faces independent of winding;
4. compact unreferenced vertices;
5. optionally remove faces attached to non-manifold edges;
6. optionally make neighboring winding consistent; and
7. apply the requested mesh-quality gate.

Its returned named tuple includes:

- `mesh`, `before`, `cleaned`, and `after`;
- vectors `removed_invalid`, `removed_degenerate`, `removed_duplicate`, and
  `removed_vertices`;
- `vertex_old_to_new`;
- integer `removed_nonmanifold`;
- vector `flipped_triangles`; and
- absolute area tolerance `area_tol_abs`.

For a closed scatterer, use:

```julia
closed = repair_mesh_for_simulation(
    mesh;
    allow_boundary=false,
    require_closed=true,
)
```

That gate rejects any remaining boundary edge. It does not infer absolute
outward orientation. Inspect normals or compare against a known signed-volume
convention separately.

### Fail instead of deleting non-manifold regions

Set `auto_drop_nonmanifold=false` and keep `strict_nonmanifold=true` when
removing faces would be an unacceptable implicit geometry change:

```julia
checked = repair_mesh_for_simulation(
    mesh;
    allow_boundary=true,
    auto_drop_nonmanifold=false,
    strict_nonmanifold=true,
)
```

The function then throws if non-manifold edges remain. Setting
`strict_nonmanifold=false` can retain them for inspection, but normal RWG
prechecks still reject that topology.

## 4. Repair and write an OBJ

`repair_obj_mesh` combines import, repair, and export:

```julia
result = repair_obj_mesh(
    "input.obj",
    "output_repaired.obj";
    reader_kwargs=(
        max_input_bytes=100_000_000,
        max_vertices=1_000_000,
        max_triangles=2_000_000,
        max_raw_bytes=100_000_000,
    ),
    allow_boundary=true,
    require_closed=false,
)
println(result.output_path)
```

The output is written only after the repaired mesh passes its selected quality
gate.

## 5. Assert the result before use

```julia
assert_mesh_quality(
    mesh_repaired;
    allow_boundary=true,
    require_closed=false,
    area_tol_rel=1e-12,
)

rwg = build_rwg(
    mesh_repaired;
    precheck=true,
    allow_boundary=true,
    require_closed=false,
)
```

Keep `precheck=true` even after a separate repair step. It prevents a later
mutation or wrong mesh variable from bypassing the boundary.

## 6. Decide whether to coarsen

```julia
matrix_gib = estimate_dense_matrix_gib(rwg.nedges)
println(matrix_gib)
```

This value is the payload of one dense `ComplexF64` matrix. It is not the peak
memory of assembly, factorization, or far-field postprocessing.

When a bounded coarsening study is appropriate:

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

println(coarse.rwg_count)
println(coarse.best_gap)
println(coarse.iterations)
mesh_sim = coarse.mesh
```

The algorithm uses voxel clustering, not feature-aware edge collapse. It may
remove narrow gaps, sharp tips, or small disconnected features. Inspect the
returned geometry and compare the intended observable across several actual RWG
counts.

## 7. Render a comparison

```julia
preview = save_mesh_preview(
    mesh_repaired,
    mesh_sim,
    "figures/mesh_repair_comparison";
    title_a="Repaired",
    title_b="Coarsened",
    color_a=:steelblue,
    color_b=:darkorange,
)

println(preview.png_path)
println(preview.pdf_path)
```

The preview uses matched axes. Check it for scale, lost components, holes, and
feature loss. A rendered preview cannot prove manifold topology or resolution.

## Diagnosing failures

### Cleanup removes every usable face

Inspect `mesh_quality_report` on the input and reduce the scope of automatic
deletion. Repair the source geometry externally when the valid surface cannot
be recovered without reconstructing faces.

### Non-manifold edges remain

Run with `auto_drop_nonmanifold=false` to expose the original problem, then
inspect or separate the touching surfaces. `drop_nonmanifold_triangles` is
aggressive: it removes every triangle attached to an over-shared edge.

### Neighboring winding is consistent but normals point inward

This is outside the orientation repair contract. Reverse the component in the
source model or use a separately verified outward-orientation procedure.

### Coarsening misses the requested count

Use the returned `rwg_count` and `best_gap`. Increase `max_iters` only after
confirming that additional search work fits the budget. Increase the target or
use a feature-aware external simplifier when the valid candidate space cannot
meet the requested count.

## Source map

| Task | Source |
|---|---|
| OBJ reader and writer | `src/geometry/Mesh.jl` |
| Quality reports and assertions | `src/geometry/Mesh.jl` |
| Repair and coarsening | `src/geometry/Mesh.jl` |
| RWG construction | `src/basis/RWG.jl` |
| Wireframe rendering | `src/postprocessing/Visualization.jl` |
| Imported-platform example | `examples/06_aircraft_rcs.jl` |
