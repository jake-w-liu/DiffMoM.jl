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

### `grad_analytical_integral_1overR(P, V1, V2, V3)`

Analytical gradient with respect to the observation point `P` of the static potential integral computed by `analytical_integral_1overR`:

```
nabla_P S(P) = -integral_T (P - r') / |P - r'|^3 dS'
```

This closed-form result (Graglia 1993; Wilton et al. 1984) splits into an in-plane part and a part along the triangle normal. It is the gradient counterpart of `analytical_integral_1overR`, used to subtract the `1/R^2` singularity of the mixed-potential scalar term near the surface.

**Parameters:**
- `P::Vec3`: Observation point.
- `V1, V2, V3::Vec3`: Triangle vertices.

**Returns:** `SVector{3,Float64}` (the gradient vector). Returns the zero vector for degenerate (zero-area) triangles.

---

### `self_cell_contribution(...)`

Compute the EFIE self-cell integral for basis functions `m` and `n` on the same triangle using singularity extraction. The integral splits into:
- **Smooth part**: Standard product quadrature with `G_smooth` (no singularity).
- **Singular part**: Outer quadrature point with analytical inner `integral{ 1/R dS' }`.

This is a low-level internal helper; most users should call `assemble_Z_efie` which handles self-cells automatically.

---

## EFIE Assembly

### `assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=376.730313668, mesh_precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12, max_output_bytes=2_000_000_000, max_cache_bytes=2_000_000_000, max_adjacency_pairs=20_000_000)`

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
| `max_output_bytes` | `Integer` | `2_000_000_000` | Raw-payload ceiling for the returned dense matrix, enforced before mesh/cache work. Use a matrix-free operator when exceeded. |
| `max_cache_bytes` | `Integer` | `2_000_000_000` | Estimated peak ceiling for EFIE quadrature, RWG-value, and adjacency storage/workspace. |
| `max_adjacency_pairs` | `Integer` | `20_000_000` | Maximum edge-derived triangle-pair records before deduplication. |

**Returns:** `Matrix{ComplexF64}` of size `N x N` where `N = rwg.nedges`.

**Performance:** Dense storage scales as O(N^2). Regular entry assembly scales
as O(N^2 * Nq^2), where Nq is the number of quadrature points; singular and
near-singular entries use different kernels. Measure wall time on the target
mesh and machine, and use the byte limits or a matrix-free operator to make the
resource decision.

---

## Matrix-Free EFIE Operators

For problems where the dense N x N matrix would exceed available memory, or when only matrix-vector products are needed (GMRES, ACA), the matrix-free EFIE operators provide the same physics without allocating the full matrix. See [types.md](types.md) for the type definitions.

### `matrixfree_efie_operator(mesh, rwg, k; kwargs...)`

Create a `MatrixFreeEFIEOperator` that behaves like the dense EFIE matrix but computes entries on demand from a precomputed `EFIEApplyCache`.

The cache stores triangle edge-adjacency in compact-row form (`offsets` plus
contiguous `neighbors`), so self/adjacent singular-integration metadata remains
linear in mesh size rather than allocating an `N_t × N_t` pair matrix.

**Parameters:** Same physical, validation, `max_cache_bytes`, and
`max_adjacency_pairs` parameters as `assemble_Z_efie`. The dense-only
`max_output_bytes` keyword does not apply.

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

### Choosing an operator representation

| Requirement | Available approach |
|-------------|--------------------|
| Dense entries and direct factorization | `assemble_Z_efie` |
| Avoid storing an `N x N` matrix | `matrixfree_efie_operator` + GMRES |
| Compress admissible far-field blocks | `build_aca_operator` (see [aca-workflow.md](aca-workflow.md)) |

The crossover depends on the geometry, quadrature order, solver behavior, and
available memory. Compare peak memory, setup time, solve time, and checked true
residual on the target workload.

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

### `precompute_patch_mass(mesh, rwg, partition; quad_order=3, max_work_bytes=536_870_912, max_terms=200_000_000)`

Precompute patch mass matrices `Mp[p]` where:

```
Mp[p][m,n] = integral_{Gamma_p} f_m(r) . f_n(r) dS
```

This is the overlap integral of two RWG basis functions restricted to patch `p`. These matrices are stored as compact `LocalMassMatrix` objects (an `AbstractMatrix{T}` triplet form; only nonzero when both basis `m` and `n` have support on triangles in patch `p`) and are precomputed once before optimization.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh` | `TriMesh` | Triangle mesh. |
| `rwg` | `RWGData` | RWG basis data. |
| `partition` | `PatchPartition` | Mapping of triangles to patches. |
| `quad_order` | `Int` | Quadrature order (default 3). |
| `max_work_bytes` | `Integer` | Raw-payload ceiling for quadrature/support workspaces, triplet builders, compact results, and constructor transients (default 512 MiB). |
| `max_terms` | `Integer` | Ceiling for local basis-pair/quadrature evaluations (default 200,000,000). |

**Returns:** `Vector{LocalMassMatrix{T}}` of length `P` (one compact mass matrix per patch). `T` is `Float64` for real RWG coefficients, or `ComplexF64` for complex (Bloch) coefficients.

---

### `assemble_Z_impedance(Mp, theta; max_output_bytes=2_000_000_000)`

Build the impedance contribution from patch mass matrices and parameter vector:

```
Z_imp = -sum_p theta_p * Mp[p]
```

The negative sign follows the convention that positive `theta_p` reduces the total impedance (the surface impedance opposes the EFIE impedance).

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: Patch mass matrices from `precompute_patch_mass`.
- `theta::AbstractVector`: Parameter vector (real or complex, length `P`).

**Returns:** Matrix of size `N x N` with element type `ComplexF64` if `theta` is real-valued, or `eltype(theta)` if `theta` is complex-valued.

**For reactive loading:** Pass complex coefficients: `theta_complex = 1im .* theta_real`.

`max_output_bytes` bounds the raw payload of the dense returned matrix before allocation.

---

### `assemble_dZ_dtheta(Mp, p)`

Returns the exact derivative matrix `dZ/d(theta_p) = -Mp[p]`. This is used internally by the adjoint gradient computation but is available for custom sensitivity analysis.

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: Patch mass matrices.
- `p::Int`: Patch index (1-based).

**Returns:** `-Mp[p]`, the negated patch mass matrix (same `LocalMassMatrix` type as `Mp[p]`).

---

### `assemble_full_Z(Z_efie, Mp, theta; reactive=false, max_output_bytes=2_000_000_000)`

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
| `max_output_bytes` | `Integer` | `2_000_000_000` | Raw-payload ceiling for the returned dense matrix, checked before copying `Z_efie`. |

**Returns:** `Matrix{ComplexF64}`.

---

### `assemble_full_Z!(Z, Z_efie, Mp, theta; reactive=false)`

In-place variant of `assemble_full_Z`. Writes `Z(theta) = Z_efie - sum_p c_p * Mp[p]` into the pre-allocated destination matrix `Z` (it first copies `Z_efie` into `Z` via `copyto!`, then subtracts the scaled patch mass matrices). Use this to avoid reallocating the full matrix on every objective/gradient evaluation in an optimization loop.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `Matrix{<:Number}` | -- | Pre-allocated destination matrix (N x N). Overwritten with the result. |
| `Z_efie` | `Matrix{<:Number}` | -- | EFIE matrix from `assemble_Z_efie`. |
| `Mp` | `Vector{<:AbstractMatrix}` | -- | Patch mass matrices. |
| `theta` | `AbstractVector` | -- | Parameter vector. |
| `reactive` | `Bool` | `false` | If `true`, treat `theta` as reactive (coefficient `i * theta_p`); otherwise resistive (coefficient `theta_p`). |

**Returns:** The destination matrix `Z` (also modified in place).

---

## Linear Solves

### `solve_forward(Z, v; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, verbose_gmres=false, check_gmres_convergence=true, check_true_residual=true, true_residual_factor=100.0)`

Solve the MoM system `Z * I = v` for the surface current coefficients `I`. This is the central solve step: given the system matrix and excitation, compute the induced currents.

For `solver=:direct`, `Z` must be a dense `Matrix` (an error is raised otherwise); use `solver=:gmres` for operator-based systems.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `AbstractMatrix{<:Number}` | -- | System matrix (N x N). Typically from `assemble_full_Z` or `assemble_Z_efie`. For `:direct`, must be a dense `Matrix`. |
| `v` | `AbstractVector{<:Number}` | -- | Excitation vector (length N). From `assemble_excitation` or `assemble_v_plane_wave`. |
| `solver` | `Symbol` | `:direct` | **`:direct`**: verified dense factorization with automatic equilibration and a high-precision fallback for exceptional ranges. **`:gmres`**: iterative GMRES for dense or matrix-free operators. Choose from measured memory, factorization cost, iteration count, and true residual on the target problem. |
| `preconditioner` | `Nothing` or `AbstractPreconditionerData` | `nothing` | Near-field preconditioner for GMRES. Ignored when `solver=:direct`. Build with `build_nearfield_preconditioner`. |
| `gmres_precond_side` | `Symbol` | `:left` | `:left` or `:right` preconditioner application side (forwarded to `solve_gmres` as `precond_side`). |
| `gmres_tol` | `Float64` | `1e-8` | Relative convergence tolerance for GMRES. Select it from the required true residual and observable accuracy; a smaller value can require more iterations. |
| `gmres_maxiter` | `Int` | `200` | Maximum GMRES iterations. Increase it only after inspecting convergence history, preconditioner quality, and the checked true residual. |
| `gmres_memory` | `Int` | `20` | GMRES restart length / Krylov memory (forwarded to `solve_gmres` as `memory`). |
| `verbose_gmres` | `Bool` | `false` | Print GMRES convergence information (iteration count, residual). |
| `check_gmres_convergence` | `Bool` | `true` | If `true`, raise an error when GMRES returns an unconverged solve. |
| `check_true_residual` | `Bool` | `true` | Verify the true relative residual without an absolute denominator floor. |
| `true_residual_factor` | `Float64` | `100.0` | Allowed true-residual multiple of `gmres_tol` (only used when `check_true_residual=true`). |

**Returns:** `Vector{ComplexF64}` solution `I` (surface current coefficients).

**Choosing a solver:**

| Criterion | Direct (`:direct`) | GMRES (`:gmres`) |
|-----------|-------------------|-------------------|
| Best for | Problems whose dense factorization fits the time and memory budget | Problems with a verified convergent operator/preconditioner combination |
| Time complexity | O(N^3) | O(N^2 * n_iter) |
| Memory | O(N^2) for factorization | O(N^2) for matrix + O(N * n_iter) for Krylov |
| Accuracy | Conditioning- and residual-dependent | Controlled by convergence and true-residual checks |
| Preconditioner | Not used | Problem-dependent; compare iterations and true residual |

---

### `solve_system(Z, rhs; solver=:direct, preconditioner=nothing, gmres_precond_side=:left, gmres_tol=1e-8, gmres_maxiter=200, gmres_memory=20, check_gmres_convergence=true, check_true_residual=true, true_residual_factor=100.0)`

General linear solve `Z * x = rhs` with the same solver dispatch as `solve_forward`. This is an alias that forwards to `solve_forward` (it accepts the same keyword arguments except `verbose_gmres`, which it does not forward).

---

## Iterative Solves (GMRES)

These are the low-level GMRES interfaces using Krylov.jl. Most users should use `solve_forward(...; solver=:gmres)` instead, which wraps these.

### `solve_gmres(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8, maxiter=200, memory=20, max_workspace_bytes=536_870_912, verbose=false, check_gmres_convergence=true)`

Solve `Z * x = rhs` using GMRES from Krylov.jl, with optional near-field preconditioning.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `AbstractMatrix{<:Number}` | -- | System matrix. |
| `rhs` | `Vector{ComplexF64}` | -- | Right-hand side. |
| `preconditioner` | `Nothing` or `AbstractPreconditionerData` | `nothing` | Near-field preconditioner. When provided, applies `Z_nf^{-1}` as a preconditioner to reduce GMRES iterations. |
| `precond_side` | `Symbol` | `:left` | `:left` or `:right` preconditioning. The side changes the residual minimized by GMRES and can change convergence; compare the checked true residual on the target system. |
| `tol` | `Float64` | `1e-8` | Relative convergence tolerance. |
| `maxiter` | `Int` | `200` | Maximum GMRES iterations. |
| `memory` | `Int` | `20` | GMRES restart length (number of Krylov vectors stored). Larger values may improve convergence for difficult problems at the cost of O(N * memory) storage. |
| `max_workspace_bytes` | `Integer` | `536_870_912` | Maximum raw payload of the Krylov vectors and Hessenberg/rotation workspace, checked before Krylov allocation. |
| `verbose` | `Bool` | `false` | Print convergence info. |
| `check_gmres_convergence` | `Bool` | `true` | Reject an unconverged, inconsistent, or non-finite result. Set to `false` only to inspect a partial iterate and its stats. |

**Returns:** Tuple `(x, stats)` where `x` is the verified solution and `stats` is the Krylov.jl convergence info. Access iteration count with `stats.niter`.

Stored dense and sparse systems with extreme global scale are normalized by
exact powers of two before iteration. This prevents absolute Krylov breakdown
thresholds from misclassifying a globally tiny, well-conditioned system.

---

### `solve_gmres_adjoint(Z, rhs; preconditioner=nothing, precond_side=:left, tol=1e-8, maxiter=200, memory=20, max_workspace_bytes=536_870_912, verbose=false, check_gmres_convergence=true)`

Solve the adjoint system `Z' * x = rhs` using GMRES with the adjoint preconditioner `Z_nf^{-H}` (inverse conjugate transpose of the near-field matrix). Used internally by `solve_adjoint` for sensitivity analysis.

**Parameters:** Same as `solve_gmres`, including the default fail-closed
`check_gmres_convergence=true` behavior.

**Returns:** Tuple `(x, stats)`.

**Note:** When a preconditioner is provided, the adjoint preconditioner `NearFieldAdjointOperator` is automatically applied. The `precond_side` parameter (`:left` or `:right`) is respected for adjoint solves, matching the behavior of `solve_gmres`.

---

## Near-Field Sparse Preconditioner

The near-field preconditioner retains entries `Z[m,n]` where basis functions `m` and `n` are within a cutoff distance, then factorizes the resulting sparse matrix. Its effectiveness and storage are problem-dependent. See [types.md](types.md) for the `AbstractPreconditionerData` type hierarchy.

Multiple overloads of `build_nearfield_preconditioner` are available, depending on what data you have:

### Overload 1: From a dense matrix

```julia
build_nearfield_preconditioner(Z::Matrix, mesh, rwg, cutoff; neighbor_search=:spatial, factorization=:lu, ilu_tau=1e-3, max_triplet_bytes=536_870_912)
```

Build a preconditioner by extracting near-field entries from a pre-assembled dense N x N matrix `Z`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `Matrix{<:Number}` | -- | The full N x N MoM matrix. |
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `cutoff` | `Float64` | -- | Nonnegative distance cutoff in meters. Select it with a storage/convergence sweep for the target problem. |
| `neighbor_search` | `Symbol` | `:spatial` | **`:spatial`** (default): O(N) spatial hashing for neighbor finding. **`:bruteforce`**: O(N^2) all-pairs reference mode. Use `:bruteforce` only for testing/validation. |
| `factorization` | `Symbol` | `:lu` | **`:lu`** (default): Sparse LU factorization. Returns `NearFieldPreconditionerData`. **`:ilu`**: Incomplete LU with drop tolerance `ilu_tau`. Returns `ILUPreconditionerData`. **`:diag`**: Jacobi/diagonal preconditioner (only retains `Z[i,i]`). Entries smaller than `1e-10` times the largest diagonal magnitude are regularized at that relative floor; an all-zero diagonal is rejected. Returns `DiagonalPreconditionerData`. |
| `ilu_tau` | `Float64` | `1e-3` | Drop tolerance for ILU factorization (only used when `factorization=:ilu`). |
| `max_triplet_bytes` | `Integer` | `536_870_912` | Maximum raw payload of the three temporary sparse-triplet arrays. The limit is checked incrementally and before predictable all-pairs allocation. |

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
build_nearfield_preconditioner(A::MatrixFreeEFIEOperator, cutoff; neighbor_search=:spatial, factorization=:lu, ilu_tau=1e-3, max_triplet_bytes=536_870_912, max_green_cache_bytes=268_435_456, max_green_cache_entries=250_000)
```

Build the preconditioner directly from a matrix-free EFIE operator without allocating a full dense matrix. The mesh and RWG data are extracted from the operator's internal cache. This is the most memory-efficient path for large problems.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `MatrixFreeEFIEOperator` | -- | Matrix-free EFIE operator from `matrixfree_efie_operator`. |
| `cutoff` | `Float64` | -- | Distance cutoff in meters. |
| `neighbor_search` | `Symbol` | `:spatial` | Neighbor search method. |
| `factorization` | `Symbol` | `:lu` | Factorization type. |
| `max_triplet_bytes` | `Integer` | `536_870_912` | Maximum raw payload of temporary near-field triplets. |
| `max_green_cache_bytes` | `Integer` | `268_435_456` | Maximum raw payload retained by cached and scratch quadrature Green matrices. After the cache fills, one bounded matrix is reused. |
| `max_green_cache_entries` | `Integer` | `250_000` | Maximum number of cached triangle-pair Green matrices; an independent count limit also bounds dictionary overhead. |

---

### Overload 4: From geometry/physics inputs directly

```julia
build_nearfield_preconditioner(mesh, rwg, k, cutoff; quad_order=3, eta0=376.730313668, mesh_precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12, factorization=:lu, ilu_tau=1e-3, max_triplet_bytes=536_870_912, max_cache_bytes=2_000_000_000, max_adjacency_pairs=20_000_000, max_green_cache_bytes=268_435_456, max_green_cache_entries=250_000)
```

Build the preconditioner directly from mesh, basis, and wavenumber — without requiring a pre-assembled matrix or explicit operator. Internally creates a `MatrixFreeEFIEOperator` and delegates to Overload 3 (spatial neighbor search, batched Green's evaluation).

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
| `factorization` | `Symbol` | `:lu` | Factorization type (`:lu`, `:ilu`, or `:diag`). |
| `ilu_tau` | `Float64` | `1e-3` | Drop tolerance for ILU (only used when `factorization=:ilu`). |
| `max_triplet_bytes` | `Integer` | `536_870_912` | Maximum raw temporary triplet payload. |
| `max_cache_bytes` | `Integer` | `2_000_000_000` | Estimated EFIE cache/workspace ceiling. |
| `max_adjacency_pairs` | `Integer` | `20_000_000` | Maximum triangle-adjacency pair records. |
| `max_green_cache_bytes` | `Integer` | `268_435_456` | Maximum cached/scratch Green-matrix raw payload. |
| `max_green_cache_entries` | `Integer` | `250_000` | Maximum number of retained triangle-pair Green matrices. |

This overload does **not** accept a `neighbor_search` keyword; it uses the batched spatial path of Overload 3 internally.

**Use case:** When you want a preconditioner but have not (or will not) assemble the full dense matrix. For example, in a pure matrix-free GMRES workflow.

---

### Overload 5: From an `ACAOperator`

```julia
build_nearfield_preconditioner(A_aca::ACAOperator; factorization=:lu, ilu_tau=1e-3, max_triplet_bytes=536_870_912)
```

Build a preconditioner by extracting the dense (inadmissible) blocks already computed inside the ACA H-matrix operator, using the cluster-tree permutation. No EFIE entries are recomputed — the near-field matrix is assembled directly from the stored block data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A_aca` | `ACAOperator` | -- | ACA H-matrix operator (see [aca-workflow.md](aca-workflow.md)). |
| `factorization` | `Symbol` | `:lu` | Factorization type (`:lu`, `:ilu`, or `:diag`). |
| `ilu_tau` | `Float64` | `1e-3` | Drop tolerance for ILU (only used when `factorization=:ilu`). |
| `max_triplet_bytes` | `Integer` | `536_870_912` | Maximum raw payload of triplets extracted from dense ACA blocks; checked before allocation. |

There is no `cutoff` argument: the near-field sparsity is defined by the ACA dense blocks themselves. This overload delegates to Overload 6 with the assembled sparse matrix.

---

### Overload 6: From a pre-assembled sparse near-field matrix

```julia
build_nearfield_preconditioner(Z_nf::SparseMatrixCSC{ComplexF64,Int}; factorization=:lu, ilu_tau=1e-3)
```

Build a preconditioner directly from an already-assembled sparse near-field matrix, skipping the distance-based neighbor search entirely. Useful for MLFMA, where `Z_near` is already assembled with the correct sparsity pattern from the octree.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z_nf` | `SparseMatrixCSC{ComplexF64,Int}` | -- | Pre-assembled sparse near-field matrix. |
| `factorization` | `Symbol` | `:lu` | Factorization type (`:lu`, `:ilu`, or `:diag`). |
| `ilu_tau` | `Float64` | `1e-3` | Drop tolerance for ILU (only used when `factorization=:ilu`). |

**Returns:** `NearFieldPreconditionerData` (`:lu`), `ILUPreconditionerData` (`:ilu`), or `DiagonalPreconditionerData` (`:diag`). The stored `cutoff` field is set to `Inf` for these factorizations since no distance cutoff is used.

---

### Common keyword arguments

| Keyword | Values | Description |
|---------|--------|-------------|
| `neighbor_search` | `:spatial` (default), `:bruteforce` | **`:spatial`**: Uses spatial hashing (cell size = cutoff) for O(N) neighbor finding. Each basis function is hashed into a 3D grid cell, and only the 27 neighboring cells are searched. **`:bruteforce`**: O(N^2) all-pairs distance check. Gives identical results; use only for validation. |
| `factorization` | `:lu` (default), `:ilu`, `:diag` | **`:lu`**: sparse LU of the retained matrix. **`:ilu`**: incomplete LU with drop tolerance `ilu_tau`. **`:diag`**: Jacobi preconditioning from the diagonal. Compare setup, factor storage, iterations, and true residual. |
| `ilu_tau` | `Float64`, default `1e-3` | Nonnegative ILU drop tolerance. Changing it alters fill, setup cost, and convergence; measure all three. |

### Performance evaluation

Preconditioner effectiveness depends on geometry, frequency, mesh density,
loading, cutoff, and factorization. Compare setup time, factor storage, GMRES
iterations, solve time, and the checked true residual on the target problem.
`examples/05_solver_methods.jl` and `examples/05b_aca_scaling.jl` provide
runnable comparisons without treating one machine's measurements as defaults.

---

### `rwg_centers(mesh, rwg)`

Compute the center point of each RWG basis function, defined as the average of the centroids of its two supporting triangles. Used internally by the preconditioner builder for distance calculations.

**Parameters:** `mesh::TriMesh`, `rwg::RWGData`.

**Returns:** `Vector{Vec3}` of length `N`.

---

### `build_block_diag_preconditioner(A_mlfma; max_storage_bytes=536_870_912)`

Build a block-diagonal (block-Jacobi) preconditioner from MLFMA leaf boxes.
Each leaf box's diagonal block in `Z_near` is LU-factorized independently,
avoiding a global sparse factorization. Compare build time, storage, iteration
count, and true residual against ILU for the target problem.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A_mlfma` | `MLFMAOperator` | The MLFMA operator (must have an octree and Z_near). |
| `max_storage_bytes` | `Integer` | Maximum raw payload of dense LU factors, leaf/pivot indices, and the reusable work vector. The complete estimate is checked before block allocation. |

**Returns:** `BlockDiagPrecondData`. See [types.md](types.md) for field details.

**Complexity:** If box `b` contains `n_b` basis functions, factorization
cost is `O(sum_b n_b^3)` and stored dense factors require
`O(sum_b n_b^2)` entries. `max_storage_bytes` bounds the estimated raw payload
before allocation.

The loaded overload
`build_block_diag_preconditioner(A_mlfma, Mp, theta; reactive=false,
max_storage_bytes=536_870_912, max_exact_work=20_000_000)` assembles each leaf
directly into its factor storage. It does not materialize one dense temporary
per patch, and it uses bounded high-precision accumulation for range-sensitive
entries.

**Example:**

```julia
A_mlfma = build_mlfma_operator(mesh, rwg, k; leaf_lambda=1.0)
P_bd = build_block_diag_preconditioner(A_mlfma)
I, stats = solve_gmres(A_mlfma, v; preconditioner=P_bd)
```

---

### `build_mlfma_preconditioner(A_mlfma; factorization=:ilu, ilu_tau=1e-2)`

Build a preconditioner for MLFMA by reordering `Z_near` to MLFMA BF ordering
before factorization. This exposes the block-banded near-field structure to
ILU. The resulting `PermutedPrecondData` applies and reverses the permutation
automatically. Measure factorization cost, iteration count, and true residual
for the target problem.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A_mlfma` | `MLFMAOperator` | -- | The MLFMA operator. |
| `factorization` | `Symbol` | `:ilu` | `:ilu` for incomplete LU or `:lu` for full sparse LU. Compare factor storage, setup cost, iterations, and true residual. |
| `ilu_tau` | `Float64` | `1e-2` | Drop tolerance for ILU. |

**Returns:** `PermutedPrecondData` wrapping an `ILUPreconditionerData` (or `NearFieldPreconditionerData` for `:lu`). It retains one reusable length-`N` permutation vector, avoiding two transient full-vector allocations per application. See [types.md](types.md) for field details.

**Example:**

```julia
A_mlfma = build_mlfma_operator(mesh, rwg, k; leaf_lambda=1.0)
P_mlfma = build_mlfma_preconditioner(A_mlfma; ilu_tau=1e-2)
I, stats = solve_gmres(A_mlfma, v; preconditioner=P_mlfma)
```

---

## Conditioning Helpers

These advanced functions implement mass-based preconditioning and regularization. They are used internally by the optimizers but can also be called directly for custom workflows.

### `make_mass_regularizer(Mp; max_output_bytes=2_000_000_000)`

Build a Hermitian positive-semidefinite mass-based regularizer: `R = sum_p Mp[p]`.

Adding `alpha * R` to the system matrix improves conditioning at the cost of introducing a small perturbation.

**Parameters:** `Mp::Vector{<:AbstractMatrix}`: Patch mass matrices. `max_output_bytes` bounds the raw payload of the dense result before allocation.

**Returns:** Dense `Matrix{ComplexF64}` `R`.

---

### `make_left_preconditioner(Mp; eps_rel=1e-8, max_output_bytes=2_000_000_000)`

Build a simple mass-based left preconditioner: `M = R + eps * I`, where `R = sum_p Mp[p]` and `eps = eps_rel * max(tr(R)/N, 1.0)`.

The small diagonal shift ensures M is invertible even if R is rank-deficient.

**Parameters:**
- `Mp::Vector{<:AbstractMatrix}`: Patch mass matrices.
- `eps_rel::Float64=1e-8`: Relative diagonal shift. Larger values improve numerical stability but reduce preconditioning effectiveness.
- `max_output_bytes::Integer=2_000_000_000`: Raw-payload ceiling for the dense result, checked before allocation.

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

### `transform_patch_matrices(Mp; preconditioner_M=nothing, preconditioner_factor=nothing, max_output_bytes=2_000_000_000)`

Transform derivative blocks under left preconditioning: `Mp_tilde[p] = M^{-1} * Mp[p]`.

When no preconditioner is active (`preconditioner_M === nothing`), returns `Mp` unchanged. A factor created by this API retains the physical matrix and can be reused alone. To reuse an externally constructed factor, also pass its original `preconditioner_M`; the original matrix is required for residual verification.

`max_output_bytes` bounds the combined raw payload of all returned dense
transformed patch matrices and is checked before the first matrix allocation.

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

Package-created factors returned by the conditioning APIs retain the original
preconditioner matrix and may be passed alone. An externally constructed factor
must be paired with its original `preconditioner_M` so the transformed solves
can be verified against the physical matrix.

**Returns:** Tuple `(Z_eff, rhs_eff, factor)` where `factor` is the reusable verified factorization for M (or `nothing`).

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
| `src/assembly/Impedance.jl` | Impedance blocks (`precompute_patch_mass`, `assemble_Z_impedance`, `assemble_dZ_dtheta`) |
| `src/assembly/SingularIntegrals.jl` | Singularity extraction (`analytical_integral_1overR`, `grad_analytical_integral_1overR`, `self_cell_contribution`, `adjacent_cell_contribution`) |
| `src/solver/Solve.jl` | `solve_forward`, `solve_system`, `assemble_full_Z`, `assemble_full_Z!`, conditioning helpers |
| `src/solver/NearFieldPreconditioner.jl` | `build_nearfield_preconditioner`, `build_block_diag_preconditioner`, `build_mlfma_preconditioner`, `rwg_centers`, operator wrappers |
| `src/solver/IterativeSolve.jl` | `solve_gmres`, `solve_gmres_adjoint` |

---

## Exercises

- **Basic:** Confirm that `assemble_full_Z(Z_efie, Mp, zeros(P))` equals `Z_efie` (zero impedance = PEC).
- **Practical:** Solve the same system with `:direct` and `:gmres` (with NF preconditioner). Compare the solutions and measure the relative error.
- **Challenge:** Sweep the NF preconditioner cutoff from 0.25-lambda to 2.0-lambda and plot GMRES iteration count vs cutoff. What is the optimal balance point for your problem?
