# API: Assembly and Solve

## Purpose

Reference for EFIE system assembly, impedance loading, and linear solvers (direct and iterative). This page covers the core computational pipeline: building the MoM system matrix, applying surface impedance, and solving for surface currents.

---

## Kernel Functions

These low-level functions evaluate the free-space Green's function and its derivatives. They are used internally by `assemble_Z_efie` but are also available for custom kernel implementations.

### `greens(r, rp, k)`

Scalar free-space Green's function:

```
G(r, r') = exp(-ikR) / (4*pi*R),   R = |r - r'|
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `r` | `Vec3` | Observation point (meters). |
| `rp` | `Vec3` | Source point (meters). |
| `k` | Real or Complex | Wavenumber (rad/m). Accepts complex `k` for complex-step differentiation. |

**Returns:** `ComplexF64` value of G.

**Convention:** Uses `exp(+iwt)` time convention, hence `exp(-ikR)` in the numerator.

---

### `greens_smooth(r, rp, k)`

Smooth part of the Green's function after singularity extraction:

```
G_smooth(r, r') = [exp(-ikR) - 1] / (4*pi*R)
```

with the well-defined limit `-ik/(4*pi)` as `R -> 0`.

Used in self-cell integration: the `1/R` singularity is separated out and handled analytically (via `analytical_integral_1overR`), while `G_smooth` is integrated numerically without any singularity.

**Parameters:** Same as `greens`.

**Returns:** `ComplexF64` value of G_smooth.

---

### `grad_greens(r, rp, k)`

Gradient of `G` with respect to the observation point `r`:

```
nabla G = [(-ik - 1/R) * G] * R_hat
```

where `R_hat = (r - r') / |r - r'|`.

**Parameters:** Same as `greens`.

**Returns:** `CVec3` (complex 3-vector).

---

## Quadrature

### `tri_quad_rule(order)`

Return Gaussian quadrature points and weights on the reference triangle with vertices `(0,0)`, `(1,0)`, `(0,1)`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `order` | `Int` | Quadrature order. Supported values: **1** (1 point), **3** (3 points), **4** (4 points), **7** (7 points). |

**Returns:** Tuple `(xi, w)` where:
- `xi::Vector{SVector{2,Float64}}`: Barycentric coordinates `(xi_1, xi_2)` on the reference triangle.
- `w::Vector{Float64}`: Weights (already include the Jacobian factor of `1/2` for the unit reference triangle).

**Choosing quadrature order:**
- `order=3` (3 points): Default. Sufficient for most EFIE assembly and excitation integration.
- `order=7` (7 points): Higher accuracy for curved surfaces or when high precision is needed.
- `order=1` (1 point): Centroid rule. Fast but low accuracy; use only for rough estimates.

**Integration formula:** To integrate `f` over a physical triangle of area `A`:

```
integral_triangle f dA = sum_q w_q * f(r_q) * 2A
```

---

### `tri_quad_points(mesh, t, xi)`

Map reference-triangle quadrature points `xi` to physical coordinates on triangle `t` of the mesh.

**Parameters:**
- `mesh::TriMesh`: Triangle mesh.
- `t::Int`: Triangle index (1-based).
- `xi::Vector{SVector{2,Float64}}`: Barycentric coordinates from `tri_quad_rule`.

**Returns:** `Vector{Vec3}` of physical coordinates.

---

## Singular Integration

These handle the `1/R` singularity that arises when source and test triangles overlap (self-cell terms). Without proper singular treatment, the EFIE matrix would have infinite diagonal entries.

### `analytical_integral_1overR(P, V1, V2, V3)`

Analytical integral `integral{ 1/|r - P| dS }` over a flat triangle with vertices `V1`, `V2`, `V3`. This closed-form result is exact (no numerical quadrature error).

**Parameters:**
- `P::Vec3`: Source point (typically on or near the triangle).
- `V1, V2, V3::Vec3`: Triangle vertices (counter-clockwise order).

**Returns:** `Float64` value of the integral.

---

### `self_cell_contribution(...)`

Compute the EFIE self-cell integral for basis functions `m` and `n` on the same triangle using singularity extraction. The integral splits into:
- **Smooth part**: Standard product quadrature with `G_smooth` (no singularity).
- **Singular part**: Outer quadrature point with analytical inner `integral{ 1/R dS' }`.

This is a low-level internal helper; most users should call `assemble_Z_efie` which handles self-cells automatically.

---

## EFIE Assembly

### `assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=376.730313668, mesh_precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12)`

Build the dense N x N EFIE impedance matrix. This is the core MoM system matrix: for a PEC scatterer with no impedance loading, the MoM equation is `Z_efie * I = v`.

Assembly is O(N^2) in both time and memory. Each entry `Z[m,n]` involves a double surface integral of the Green's function weighted by RWG basis functions.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `k` | Real or Complex | -- | Wavenumber `k = 2*pi/lambda` in rad/m. Can be complex for complex-step gradient verification. |
| `quad_order` | `Int` | `3` | Quadrature order on the reference triangle. Use `3` for standard accuracy; `7` for high-precision validation. |
| `eta0` | `Real` | `376.730313668` | Free-space impedance in Ohms. The default is `mu_0 * c_0 = 376.73...` The EFIE matrix scales linearly with `eta0`. |
| `mesh_precheck` | `Bool` | `true` | Run mesh quality checks before assembly. Disable only for performance when you are certain the mesh is valid. |
| `allow_boundary` | `Bool` | `true` | Allow boundary edges during precheck. |
| `require_closed` | `Bool` | `false` | Require closed surface during precheck. |
| `area_tol_rel` | `Float64` | `1e-12` | Relative tolerance for degenerate triangle detection. |

**Returns:** `Matrix{ComplexF64}` of size `N x N` where `N = rwg.nedges`.

**Performance:** Assembly time scales as O(N^2 * Nq^2) where Nq is the number of quadrature points. For N = 500, assembly takes seconds; for N = 5000, it takes minutes.

---

## Matrix-Free EFIE Operators

For problems where the dense N x N matrix would exceed available memory, or when only matrix-vector products are needed (GMRES, ACA), the matrix-free EFIE operators provide the same physics without allocating the full matrix. See [types.md](types.md) for the type definitions.

### `matrixfree_efie_operator(mesh, rwg, k; kwargs...)`

Create a `MatrixFreeEFIEOperator` that behaves like the dense EFIE matrix but computes entries on demand from a precomputed `EFIEApplyCache`.

**Parameters:** Same as `assemble_Z_efie` (mesh, rwg, k, quad_order, eta0, mesh_precheck, allow_boundary, require_closed, area_tol_rel).

**Returns:** `MatrixFreeEFIEOperator{ComplexF64}` -- an `AbstractMatrix{ComplexF64}` of size `(N, N)`.

**Supported operations:**

| Operation | Syntax | Cost | Description |
|-----------|--------|------|-------------|
| Single entry | `A[i, j]` or `efie_entry(A, i, j)` | O(Nq^2) | Compute one EFIE entry on the fly. |
| Matrix-vector product | `A * x` or `mul!(y, A, x)` | O(N^2 * Nq^2) | Full matvec, row by row. |
| Adjoint | `A'` or `adjoint(A)` | Free | Returns `MatrixFreeEFIEAdjointOperator`. |
| Adjoint matvec | `A' * x` | O(N^2 * Nq^2) | Adjoint matvec for adjoint sensitivity solves. |

---

### `matrixfree_efie_adjoint_operator(A)`

Return the adjoint operator `A'` for Krylov adjoint solves. Equivalent to `adjoint(A)`.

---

### `efie_entry(A, m, n)`

Compute a single EFIE matrix entry `Z[m,n]` from a `MatrixFreeEFIEOperator`. Used by the ACA algorithm and the near-field preconditioner builder to access individual matrix elements without dense storage.

**Parameters:**
- `A::MatrixFreeEFIEOperator`: The matrix-free operator.
- `m::Int`, `n::Int`: Row and column indices (1-based).

**Returns:** `ComplexF64` matrix entry.

---

### When to use matrix-free operators

| Scenario | Recommended approach |
|----------|---------------------|
| N < ~5000 | Dense `assemble_Z_efie` (simpler, enables direct LU) |
| N > ~5000, memory-limited | `matrixfree_efie_operator` + GMRES with NF preconditioner |
| ACA H-matrix | Used internally by `build_aca_operator` (see [aca-workflow.md](aca-workflow.md)) |

**Example:**

```julia
# Pure matrix-free GMRES workflow (no dense matrix allocated)
A = matrixfree_efie_operator(mesh, rwg, k)
P_nf = build_nearfield_preconditioner(A, 1.0 * lambda0)
I, stats = solve_gmres(A, v; preconditioner=P_nf)
println("Solved with $(stats.niter) GMRES iterations, no dense matrix")
```

---

## Impedance Assembly

These functions add surface impedance loading to the EFIE matrix. In optimization, the impedance parameters `theta` are the design variables.

### `precompute_patch_mass(mesh, rwg, partition; quad_order=3)`

Precompute patch mass matrices `Mp[p]` where:

```
Mp[p][m,n] = integral_{Gamma_p} f_m(r) . f_n(r) dS
```

This is the overlap integral of two RWG basis functions restricted to patch `p`. These matrices are sparse (only nonzero when both basis `m` and `n` have support on triangles in patch `p`) and are precomputed once before optimization.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh` | `TriMesh` | Triangle mesh. |
| `rwg` | `RWGData` | RWG basis data. |
| `partition` | `PatchPartition` | Mapping of triangles to patches. |
| `quad_order` | `Int` | Quadrature order (default 3). |

**Returns:** `Vector{SparseMatrixCSC{Float64,Int}}` of length `P` (one sparse matrix per patch).

---

### `assemble_Z_impedance(Mp, theta)`

Build the impedance contribution from patch mass matrices and parameter vector:

```
Z_imp = -sum_p theta_p * Mp[p]
```

The negative sign follows the convention that positive `theta_p` reduces the total impedance (the surface impedance opposes the EFIE impedance).

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: Patch mass matrices from `precompute_patch_mass`.
- `theta::AbstractVector`: Parameter vector (real or complex, length `P`).

**Returns:** `Matrix{ComplexF64}` of size `N x N`.

**For reactive loading:** Pass complex coefficients: `theta_complex = 1im .* theta_real`.

---

### `assemble_dZ_dtheta(Mp, p)`

Returns the exact derivative matrix `dZ/d(theta_p) = -Mp[p]`. This is used internally by the adjoint gradient computation but is available for custom sensitivity analysis.

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: Patch mass matrices.
- `p::Int`: Patch index (1-based).

**Returns:** Sparse matrix (same type as `Mp[p]`).

---

### `assemble_full_Z(Z_efie, Mp, theta; reactive=false)`

Convenience function to assemble the full system matrix combining EFIE and impedance loading:

```
Z = Z_efie - sum_p c_p * Mp[p]
```

where the coefficients depend on the loading mode:
- **Resistive** (`reactive=false`, default): `c_p = theta_p`. The impedance parameters represent real-valued surface resistance.
- **Reactive** (`reactive=true`): `c_p = i * theta_p`. The impedance parameters represent imaginary-valued surface reactance (lossless loading).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z_efie` | `Matrix{<:Number}` | -- | EFIE matrix from `assemble_Z_efie`. |
| `Mp` | `Vector{<:AbstractMatrix}` | -- | Patch mass matrices. |
| `theta` | `AbstractVector` | -- | Parameter vector (always real-valued; the `reactive` flag controls the mapping). |
| `reactive` | `Bool` | `false` | If `true`, treat `theta` as reactive parameters (multiplied by `im` internally). |

**Returns:** `Matrix{ComplexF64}`.

---

## Linear Solves

### `solve_forward(Z, v; solver=:direct, preconditioner=nothing, gmres_tol=1e-8, gmres_maxiter=200, verbose_gmres=false)`

Solve the MoM system `Z * I = v` for the surface current coefficients `I`. This is the central solve step: given the system matrix and excitation, compute the induced currents.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `Matrix{<:Number}` | -- | System matrix (N x N). Typically from `assemble_full_Z` or `assemble_Z_efie`. |
| `v` | `Vector{<:Number}` | -- | Excitation vector (length N). From `assemble_excitation` or `assemble_v_plane_wave`. |
| `solver` | `Symbol` | `:direct` | **`:direct`**: LU factorization (`Z \ v`). Exact, O(N^3). Best for N < ~2000. **`:gmres`**: Iterative GMRES. O(N^2 * n_iter). Best for large N with a good preconditioner. |
| `preconditioner` | `Nothing` or `AbstractPreconditionerData` | `nothing` | Near-field preconditioner for GMRES. Ignored when `solver=:direct`. Build with `build_nearfield_preconditioner`. |
| `gmres_tol` | `Float64` | `1e-8` | Relative convergence tolerance for GMRES. Smaller = more accurate but more iterations. `1e-8` is conservative; `1e-6` is often sufficient. |
| `gmres_maxiter` | `Int` | `200` | Maximum GMRES iterations. With a good preconditioner, convergence typically occurs in 10--50 iterations. Set higher (300--500) for difficult problems or tight tolerances. |
| `verbose_gmres` | `Bool` | `false` | Print GMRES convergence information (iteration count, residual). |

**Returns:** `Vector{ComplexF64}` solution `I` (surface current coefficients).

**Choosing a solver:**

| Criterion | Direct (`:direct`) | GMRES (`:gmres`) |
|-----------|-------------------|-------------------|
| Best for | N < ~2000 | N > ~1000 with preconditioner |
| Time complexity | O(N^3) | O(N^2 * n_iter) |
| Memory | O(N^2) for factorization | O(N^2) for matrix + O(N * n_iter) for Krylov |
| Accuracy | Machine precision | Controlled by `gmres_tol` |
| Requires preconditioner? | No | Strongly recommended |

---

### `solve_system(Z, rhs; solver=:direct, preconditioner=nothing, gmres_tol=1e-8, gmres_maxiter=200)`

General linear solve `Z * x = rhs` with the same solver dispatch as `solve_forward`. This is an alias that forwards to `solve_forward`.

---

## Iterative Solves (GMRES)

These are the low-level GMRES interfaces using Krylov.jl. Most users should use `solve_forward(...; solver=:gmres)` instead, which wraps these.

### `solve_gmres(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8, maxiter=200, memory=20, verbose=false)`

Solve `Z * x = rhs` using GMRES from Krylov.jl, with optional near-field preconditioning.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `AbstractMatrix{<:Number}` | -- | System matrix. |
| `rhs` | `Vector{ComplexF64}` | -- | Right-hand side. |
| `preconditioner` | `Nothing` or `AbstractPreconditionerData` | `nothing` | Near-field preconditioner. When provided, applies `Z_nf^{-1}` as a preconditioner to reduce GMRES iterations. |
| `precond_side` | `Symbol` | `:left` | `:left` or `:right` preconditioning. Both give the same iteration count for EFIE matrices. Left is the default. |
| `tol` | `Float64` | `1e-8` | Relative convergence tolerance. |
| `maxiter` | `Int` | `200` | Maximum GMRES iterations. |
| `memory` | `Int` | `20` | GMRES restart length (number of Krylov vectors stored). Larger values may improve convergence for difficult problems at the cost of O(N * memory) storage. |
| `verbose` | `Bool` | `false` | Print convergence info. |

**Returns:** Tuple `(x, stats)` where `x` is the solution and `stats` is the Krylov.jl convergence info. Access iteration count with `stats.niter`.

---

### `solve_gmres_adjoint(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8, maxiter=200, verbose=false)`

Solve the adjoint system `Z' * x = rhs` using GMRES with the adjoint preconditioner `Z_nf^{-H}` (inverse conjugate transpose of the near-field matrix). Used internally by `solve_adjoint` for sensitivity analysis.

**Parameters:** Same as `solve_gmres` except `memory` (GMRES restart length) is not exposed; the Krylov.jl default is used for adjoint solves.

**Returns:** Tuple `(x, stats)`.

**Note:** When a preconditioner is provided, the adjoint preconditioner `NearFieldAdjointOperator` is automatically applied. The `precond_side` parameter (`:left` or `:right`) is respected for adjoint solves, matching the behavior of `solve_gmres`.

---

## Near-Field Sparse Preconditioner

The near-field preconditioner retains entries `Z[m,n]` where basis functions `m` and `n` are within a cutoff distance, then factorizes the resulting sparse matrix. This is the recommended preconditioning strategy for dense EFIE systems. See [types.md](types.md) for the `AbstractPreconditionerData` type hierarchy.

Multiple overloads of `build_nearfield_preconditioner` are available, depending on what data you have:

### Overload 1: From a dense matrix

```julia
build_nearfield_preconditioner(Z::Matrix, mesh, rwg, cutoff; neighbor_search=:spatial, factorization=:lu, ilu_tau=1e-3)
```

Build a preconditioner by extracting near-field entries from a pre-assembled dense N x N matrix `Z`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `Matrix{<:Number}` | -- | The full N x N MoM matrix. |
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `cutoff` | `Float64` | -- | Distance cutoff in meters. Typical: `0.5 * lambda` to `2.0 * lambda`. |
| `neighbor_search` | `Symbol` | `:spatial` | **`:spatial`** (default): O(N) spatial hashing for neighbor finding. **`:bruteforce`**: O(N^2) all-pairs reference mode. Use `:bruteforce` only for testing/validation. |
| `factorization` | `Symbol` | `:lu` | **`:lu`** (default): Sparse LU factorization. Returns `NearFieldPreconditionerData`. **`:ilu`**: Incomplete LU with drop tolerance `ilu_tau`. Returns `ILUPreconditionerData`. **`:diag`**: Jacobi/diagonal preconditioner (only retains `Z[i,i]`). Returns `DiagonalPreconditionerData`. |
| `ilu_tau` | `Float64` | `1e-3` | Drop tolerance for ILU factorization (only used when `factorization=:ilu`). |

**Returns:** `NearFieldPreconditionerData` (for `:lu`), `ILUPreconditionerData` (for `:ilu`), or `DiagonalPreconditionerData` (for `:diag`).

---

### Overload 2: From an abstract matrix or operator

```julia
build_nearfield_preconditioner(A::AbstractMatrix, mesh, rwg, cutoff; neighbor_search=:spatial, factorization=:lu, ilu_tau=1e-3)
```

Same as Overload 1, but accepts any `AbstractMatrix{<:Number}` including custom matrix types. Entries are accessed via `A[m, n]`.

---

### Overload 3: From a `MatrixFreeEFIEOperator`

```julia
build_nearfield_preconditioner(A::MatrixFreeEFIEOperator, cutoff; neighbor_search=:spatial, factorization=:lu, ilu_tau=1e-3)
```

Build the preconditioner directly from a matrix-free EFIE operator without allocating a full dense matrix. The mesh and RWG data are extracted from the operator's internal cache. This is the most memory-efficient path for large problems.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `MatrixFreeEFIEOperator` | -- | Matrix-free EFIE operator from `matrixfree_efie_operator`. |
| `cutoff` | `Float64` | -- | Distance cutoff in meters. |
| `neighbor_search` | `Symbol` | `:spatial` | Neighbor search method. |
| `factorization` | `Symbol` | `:lu` | Factorization type. |

---

### Overload 4: From geometry/physics inputs directly

```julia
build_nearfield_preconditioner(mesh, rwg, k, cutoff; quad_order=3, eta0=376.730313668, neighbor_search=:spatial, factorization=:lu, ...)
```

Build the preconditioner directly from mesh, basis, and wavenumber — without requiring a pre-assembled matrix or explicit operator. Internally creates a `MatrixFreeEFIEOperator` and extracts the needed entries.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `k` | Real or Complex | -- | Wavenumber (rad/m). |
| `cutoff` | `Float64` | -- | Distance cutoff in meters. |
| `quad_order` | `Int` | `3` | Quadrature order for EFIE entry evaluation. |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance. |
| `mesh_precheck` | `Bool` | `true` | Run mesh quality checks. |
| `allow_boundary` | `Bool` | `true` | Allow boundary edges. |
| `require_closed` | `Bool` | `false` | Require closed surface. |
| `area_tol_rel` | `Float64` | `1e-12` | Degenerate triangle tolerance. |
| `neighbor_search` | `Symbol` | `:spatial` | Neighbor search method. |
| `factorization` | `Symbol` | `:lu` | Factorization type. |

**Use case:** When you want a preconditioner but have not (or will not) assemble the full dense matrix. For example, in a pure matrix-free GMRES workflow.

---

### Common keyword arguments

| Keyword | Values | Description |
|---------|--------|-------------|
| `neighbor_search` | `:spatial` (default), `:bruteforce` | **`:spatial`**: Uses spatial hashing (cell size = cutoff) for O(N) neighbor finding. Each basis function is hashed into a 3D grid cell, and only the 27 neighboring cells are searched. **`:bruteforce`**: O(N^2) all-pairs distance check. Gives identical results; use only for validation. |
| `factorization` | `:lu` (default), `:ilu`, `:diag` | **`:lu`**: Sparse LU factorization of the near-field matrix. Best GMRES convergence. **`:ilu`**: Incomplete LU with drop tolerance `ilu_tau`. Less fill than full LU, feasible for large N. **`:diag`**: Jacobi preconditioner using only diagonal entries. Cheapest to build and apply, but weaker convergence. |
| `ilu_tau` | `Float64`, default `1e-3` | Drop tolerance for ILU factorization (only used when `factorization=:ilu`). Smaller values = more fill = better preconditioner. |

### Performance benchmarks

| Scenario | Cutoff | GMRES iters | Speedup vs direct LU |
|----------|--------|-------------|---------------------|
| Impedance-loaded, N=96--736 | 1.0 lambda | ~10 (constant) | 3--6x at N=736 |
| PEC, N=736 | 0.5 lambda | 28--45 | ~5.5x |
| PEC, N=736, unpreconditioned | -- | ~194 | (baseline) |

**Key insight:** In tested impedance-loaded cases, NF preconditioning keeps
iteration growth weak as N increases. This keeps GMRES+NF close to O(N^2) per
solve for dense-matvec workflows, versus O(N^3) for direct LU.

---

### `rwg_centers(mesh, rwg)`

Compute the center point of each RWG basis function, defined as the average of the centroids of its two supporting triangles. Used internally by the preconditioner builder for distance calculations.

**Parameters:** `mesh::TriMesh`, `rwg::RWGData`.

**Returns:** `Vector{Vec3}` of length `N`.

---

### `build_block_diag_preconditioner(A_mlfma)`

Build a block-diagonal (block-Jacobi) preconditioner from MLFMA leaf boxes. Each leaf box's diagonal block in `Z_near` is LU-factorized independently. Much faster than ILU for large N.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A_mlfma` | `MLFMAOperator` | The MLFMA operator (must have an octree and Z_near). |

**Returns:** `BlockDiagPrecondData`. See [types.md](types.md) for field details.

**Complexity:** `O(n_boxes * n_bf^3)` where `n_bf` is the average BFs per leaf box (typically 100--500). Memory: `n_boxes * n_bf^2 * 16` bytes (typically < 100 MB).

**Example:**

```julia
A_mlfma = build_mlfma_operator(mesh, rwg, k; leaf_lambda=1.0)
P_bd = build_block_diag_preconditioner(A_mlfma)
I, stats = solve_gmres(A_mlfma, v; preconditioner=P_bd)
```

---

### `build_mlfma_preconditioner(A_mlfma; factorization=:ilu, ilu_tau=1e-2)`

Build a preconditioner for MLFMA by reordering `Z_near` to MLFMA BF ordering (block-banded structure) before factorization. The block-banded structure makes ILU dramatically faster and more effective than operating on the original scattered BF ordering. The resulting `PermutedPrecondData` handles the permutation automatically on each preconditioner application.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A_mlfma` | `MLFMAOperator` | -- | The MLFMA operator. |
| `factorization` | `Symbol` | `:ilu` | `:ilu` for incomplete LU (recommended), `:lu` for full sparse LU. |
| `ilu_tau` | `Float64` | `1e-2` | Drop tolerance for ILU. |

**Returns:** `PermutedPrecondData` wrapping an `ILUPreconditionerData` (or `NearFieldPreconditionerData` for `:lu`). See [types.md](types.md) for field details.

**Example:**

```julia
A_mlfma = build_mlfma_operator(mesh, rwg, k; leaf_lambda=1.0)
P_mlfma = build_mlfma_preconditioner(A_mlfma; ilu_tau=1e-2)
I, stats = solve_gmres(A_mlfma, v; preconditioner=P_mlfma)
```

---

## Conditioning Helpers

These advanced functions implement mass-based preconditioning and regularization. They are used internally by the optimizers but can also be called directly for custom workflows.

### `make_mass_regularizer(Mp)`

Build a Hermitian positive-semidefinite mass-based regularizer: `R = sum_p Mp[p]`.

Adding `alpha * R` to the system matrix improves conditioning at the cost of introducing a small perturbation.

**Parameters:** `Mp::Vector{<:AbstractMatrix}`: Patch mass matrices.

**Returns:** Dense `Matrix{ComplexF64}` `R`.

---

### `make_left_preconditioner(Mp; eps_rel=1e-8)`

Build a simple mass-based left preconditioner: `M = R + eps * I`, where `R = sum_p Mp[p]` and `eps = eps_rel * max(tr(R)/N, 1.0)`.

The small diagonal shift ensures M is invertible even if R is rank-deficient.

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: Patch mass matrices.
- `eps_rel::Float64=1e-8`: Relative diagonal shift. Larger values improve numerical stability but reduce preconditioning effectiveness.

**Returns:** `Matrix{ComplexF64}` `M`.

---

### `select_preconditioner(Mp; mode=:off, preconditioner_M=nothing, n_threshold=256, iterative_solver=false, eps_rel=1e-6)`

Select the effective left preconditioner matrix used by the solver. This is the decision logic used internally by the optimizers.

**Modes:**

| Mode | Behavior |
|------|----------|
| `:off` | Disable mass-based preconditioning (unless `preconditioner_M` is explicitly provided). |
| `:on` | Always build and use a mass-based preconditioner from `Mp`. |
| `:auto` | Enable preconditioning when `iterative_solver=true` OR `N >= n_threshold`. |

If `preconditioner_M` is explicitly provided, it takes precedence over the `mode` setting.

**Returns:** Tuple `(M_eff, enabled, reason)`:
- `M_eff`: Dense `Matrix{ComplexF64}` or `nothing`.
- `enabled::Bool`: Whether preconditioning is active.
- `reason::String`: Human-readable status for logging.

---

### `transform_patch_matrices(Mp; preconditioner_M=nothing, preconditioner_factor=nothing)`

Transform derivative blocks under left preconditioning: `Mp_tilde[p] = M^{-1} * Mp[p]`.

When no preconditioner is active (`preconditioner_M === nothing`), returns `Mp` unchanged. If `preconditioner_factor` is provided (a pre-computed LU factorization of M), it is reused to avoid redundant factorization.

**Returns:** Tuple `(Mp_tilde, factor)`.

---

### `prepare_conditioned_system(Z_raw, rhs; regularization_alpha=0.0, regularization_R=nothing, preconditioner_M=nothing, preconditioner_factor=nothing)`

Build the conditioned linear system used by forward and adjoint solves:

```
Z_reg = Z_raw + alpha * R          (regularization)
Z_eff = M^{-1} * Z_reg             (left preconditioning)
rhs_eff = M^{-1} * rhs
```

If no regularization or preconditioning is requested, returns `(Z_raw, rhs, nothing)` unchanged.

**Returns:** Tuple `(Z_eff, rhs_eff, factor)` where `factor` is the LU factorization of M (or `nothing`).

---

## Minimal Patterns

### Direct solve (default)
```julia
Z_efie = assemble_Z_efie(mesh, rwg, k)
Mp = precompute_patch_mass(mesh, rwg, partition)
Z = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
I = solve_forward(Z, v)
```

### GMRES with near-field preconditioner (from dense matrix)
```julia
Z_efie = assemble_Z_efie(mesh, rwg, k)
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, 1.0 * lambda0)
Z = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
I, stats = solve_gmres(Matrix{ComplexF64}(Z), v; preconditioner=P_nf)
println("GMRES iterations: ", stats.niter)
```

### GMRES with preconditioner built from geometry directly (no dense matrix)
```julia
P_nf = build_nearfield_preconditioner(mesh, rwg, k, 1.0 * lambda0)
I = solve_forward(Z, v; solver=:gmres, preconditioner=P_nf)
```

### GMRES via solve_forward dispatch
```julia
I = solve_forward(Z, v; solver=:gmres, preconditioner=P_nf,
                   gmres_tol=1e-8, gmres_maxiter=300)
```

### Matrix-free GMRES (no dense matrix)
```julia
A = matrixfree_efie_operator(mesh, rwg, k)
P_nf = build_nearfield_preconditioner(A, 1.0 * lambda0)
I, stats = solve_gmres(A, v; preconditioner=P_nf)
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/assembly/EFIE.jl` | Dense assembly (`assemble_Z_efie`), matrix-free operators (`MatrixFreeEFIEOperator`, `matrixfree_efie_operator`, `efie_entry`) |
| `src/assembly/Impedance.jl` | Impedance blocks (`precompute_patch_mass`, `assemble_Z_impedance`) |
| `src/solver/Solve.jl` | `solve_forward`, `solve_system`, `assemble_full_Z`, conditioning helpers |
| `src/solver/NearFieldPreconditioner.jl` | `build_nearfield_preconditioner`, `build_block_diag_preconditioner`, `build_mlfma_preconditioner`, `rwg_centers`, operator wrappers |
| `src/solver/IterativeSolve.jl` | `solve_gmres`, `solve_gmres_adjoint` |

---

## Exercises

- **Basic:** Confirm that `assemble_full_Z(Z_efie, Mp, zeros(P))` equals `Z_efie` (zero impedance = PEC).
- **Practical:** Solve the same system with `:direct` and `:gmres` (with NF preconditioner). Compare the solutions and measure the relative error.
- **Challenge:** Sweep the NF preconditioner cutoff from 0.25-lambda to 2.0-lambda and plot GMRES iteration count vs cutoff. What is the optimal balance point for your problem?
