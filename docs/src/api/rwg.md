# API: RWG Utilities

## Purpose

Reference for RWG (Rao-Wilton-Glisson) basis function construction and evaluation. RWG functions are the standard divergence-conforming basis for surface Method of Moments: each basis function is associated with one interior edge of the triangle mesh and has support on the two triangles sharing that edge.

**Why RWG?** The EFIE requires basis functions whose normal component is continuous across triangle edges (to properly represent surface current). RWG functions satisfy this by construction: the current flows smoothly across interior edges while naturally enforcing zero normal current at boundary edges.

The number of RWG basis functions (`rwg.nedges`) equals the number of interior edges in the mesh, and this determines the dimension N of the MoM system matrix.

---

## `build_rwg(mesh; precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12)`

Constructs `RWGData` from the interior edges of a triangle mesh. Each interior edge (shared by exactly two triangles) defines one RWG basis function.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh. Must pass quality checks (no degenerate triangles, no non-manifold edges, consistent orientation). |
| `precheck` | `Bool` | `true` | Run mesh quality checks before basis construction. **Recommended for imported meshes.** Set to `false` only if you have already verified the mesh quality and want to skip redundant checks. |
| `allow_boundary` | `Bool` | `true` | Allow boundary edges (edges belonging to only one triangle). These edges do not generate RWG basis functions. Set to `true` for open surfaces (plates, reflectors). |
| `require_closed` | `Bool` | `false` | Require that the surface is closed (zero boundary edges). Set to `true` for enclosed PEC bodies where boundary edges indicate a mesh defect. |
| `area_tol_rel` | `Float64` | `1e-12` | Relative tolerance for degenerate triangle detection during precheck. |

**Returns:** `RWGData` containing edge-to-triangle connectivity and geometric factors. See [types.md](types.md) for detailed field descriptions.

**Relationship between mesh size and system size:**

| Mesh | Vertices | Triangles | Interior edges (N) | System matrix |
|------|----------|-----------|-------------------|---------------|
| 3x3 plate | 16 | 18 | 24 | 24 x 24 |
| 5x5 plate | 36 | 50 | 84 | 84 x 84 |
| 10x10 plate | 121 | 200 | 319 | 319 x 319 |

For a regular triangulation: `N ~ 3 * Nt / 2` (interior edges scale with triangle count).

---

## `eval_rwg(rwg, n, r, t)`

Evaluates RWG basis function `n` at a physical point `r` on triangle `t`. This gives the vector-valued basis function used in the EFIE inner products.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `rwg` | `RWGData` | RWG basis data from `build_rwg`. |
| `n` | `Int` | Basis function index (1-based, in `1:rwg.nedges`). |
| `r` | `Vec3` | Evaluation point in meters. Must lie on (or near) triangle `t`. |
| `t` | `Int` | Triangle index. Must be either `rwg.tplus[n]` or `rwg.tminus[n]` (one of the two support triangles of basis `n`). Returns zero if `t` is not in the support. |

**Returns:** `Vec3` -- the basis function vector value at point `r`.

**Formula:**

The RWG basis function is a piecewise-linear vector field defined on its two support triangles:

- On T+ (`t == rwg.tplus[n]`): `f_n(r) = (l_n / (2 A+)) * (r - r_opp+)`
  The vector points **away from** the opposite vertex `r_opp+`.

- On T- (`t == rwg.tminus[n]`): `f_n(r) = (l_n / (2 A-)) * (r_opp- - r)`
  The vector points **toward** the opposite vertex `r_opp-`.

where `l_n` is the edge length, `A+`/`A-` are the triangle areas, and `r_opp+`/`r_opp-` are the positions of the vertices opposite the shared edge.

**Physical interpretation:** The basis function represents a localized surface current that flows across the shared edge from T+ to T-. The current magnitude is proportional to the edge length and inversely proportional to the triangle area, ensuring proper normalization.

---

## `div_rwg(rwg, n, t)`

Returns the piecewise-constant surface divergence of basis function `n` on triangle `t`. The divergence is constant within each support triangle (a property of the linear RWG basis).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `rwg` | `RWGData` | RWG basis data. |
| `n` | `Int` | Basis function index. |
| `t` | `Int` | Triangle index (must be `rwg.tplus[n]` or `rwg.tminus[n]`). |

**Returns:** `Float64` divergence value (1/m).

**Formula:**
- On T+ (`t == rwg.tplus[n]`): `div f_n = +l_n / A+`
- On T- (`t == rwg.tminus[n]`): `div f_n = -l_n / A-`

**Role in EFIE:** The divergence appears in the scalar (charge) part of the EFIE kernel. The full EFIE entry is:

```
Z_mn = -j*omega*mu0 * [ integral{ f_m . f_n G dS dS' } - (1/k^2) * integral{ (div f_m)(div f_n) G dS dS' } ]
```

where `G = exp(-jkR)/(4*pi*R)` and `omega*mu0 = k*eta0`.

The opposite signs on T+ and T- ensure charge conservation: the net charge produced by basis `n` sums to zero (current flows in on one triangle, out on the other).

---

## `basis_triangles(rwg, n)`

Returns the two triangle indices forming the support of basis function `n`.

**Parameters:**
- `rwg::RWGData`: RWG basis data.
- `n::Int`: Basis function index.

**Returns:** Tuple `(tplus, tminus)` where `tplus = rwg.tplus[n]` and `tminus = rwg.tminus[n]`.

This is a convenience accessor; you can also access `rwg.tplus[n]` and `rwg.tminus[n]` directly.

---

## Typical Usage

```julia
# Build RWG basis from a mesh
mesh = make_rect_plate(0.1, 0.1, 5, 5)   # 0.1m plate, 5x5 cells
rwg = build_rwg(mesh; precheck=true)
println("System dimension N = ", rwg.nedges)

# Inspect the first basis function
tp, tm = basis_triangles(rwg, 1)
println("Basis 1: edge length = ", rwg.len[1], " m")
println("  T+ = ", tp, " (area = ", rwg.area_plus[1], " m^2)")
println("  T- = ", tm, " (area = ", rwg.area_minus[1], " m^2)")

# Evaluate divergence (constant on each triangle)
println("  div on T+ = ", div_rwg(rwg, 1, tp), " (1/m)")
println("  div on T- = ", div_rwg(rwg, 1, tm), " (1/m)")
# Note: these have opposite signs (charge conservation)

# Evaluate the vector basis at the centroid of T+
r_center = triangle_center(mesh, tp)
f_val = eval_rwg(rwg, 1, r_center, tp)
println("  f_1 at T+ centroid = ", f_val)
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/basis/RWG.jl` | `build_rwg`, `eval_rwg`, `div_rwg`, `basis_triangles` |
| `src/geometry/Mesh.jl` | Geometry helpers used internally (`triangle_area`, `triangle_center`, etc.) |

---

## Exercises

- **Basic:** Print support triangles and edge length for the first 5 RWG bases of a 3x3 plate. Verify that `div_rwg` values on T+ and T- have opposite signs for each basis.
- **Practical:** For a given basis function, evaluate `eval_rwg` at all quadrature points on T+ and T-. Verify that the dot product of the basis value with the edge normal is continuous across the shared edge.
- **Advanced:** Count the total number of unique edges (`mesh_unique_edges`) and interior edges (`rwg.nedges`) for several plate sizes. Verify the relationship: `n_boundary_edges = n_total_edges - n_interior_edges`.
