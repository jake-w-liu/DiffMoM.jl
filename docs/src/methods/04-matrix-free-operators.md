# Matrix-Free EFIE Operators

## Purpose

Dense EFIE matrices require quadratic storage, yet many
algorithms (GMRES, ACA, near-field preconditioning) only need matrix-vector
products or individual entry evaluations. This chapter introduces the
matrix-free operator infrastructure in `DiffMoM.jl`, which stores the
kernel evaluation "recipe" rather than the full $N \times N$ matrix, enabling
memory-efficient iterative solves and on-demand entry queries.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why matrix-free operators exist and quantify the memory savings.
2. Describe the role of `EFIEApplyCache` in precomputing geometry for fast entry evaluation.
3. Construct a `MatrixFreeEFIEOperator` and use it for matvecs and entry queries.
4. Understand how the adjoint operator is formed and where it is used.
5. Choose between dense, matrix-free, and ACA representations for a given problem size.

---

## 1. Why Matrix-Free?

A dense EFIE matrix $\mathbf{Z} \in \mathbb{C}^{N \times N}$ stores $N^2$ `ComplexF64` entries at 16 bytes each:

```math
\text{Memory} = 16 N^2 \;\text{bytes}.
```

| $N$ (unknowns) | Dense storage |
|-----------------|---------------|
| 1 000           | 15.3 MiB      |
| 5 000           | 381 MiB       |
| 10 000          | 1.49 GiB      |
| 20 000          | 5.96 GiB      |
| 50 000          | 37.3 GiB      |

These values cover only the matrix payload; factorization and assembly need
additional memory. GMRES needs the matvec $\mathbf{y}=\mathbf{Z}\mathbf{x}$,
ACA and near-field construction need selected entries $Z_{mn}$, and the adjoint
solve needs $\mathbf{y}=\mathbf{Z}^\dagger\mathbf{x}$. Those operations do not
require all $N^2$ entries to remain stored simultaneously.

A matrix-free operator stores the *recipe* for computing any entry $Z_{mn}$ on demand, bundled into an `EFIEApplyCache`. The trade-off:

| Aspect | Dense matrix | Matrix-free operator |
|--------|-------------|----------------------|
| Storage | $O(N^2)$ | $O(N)$ (cache only) |
| Matvec cost | $O(N^2)$ (memory read) | $O(N^2 N_q^2)$ (recomputation) |
| Single entry | $O(1)$ (lookup) | $O(N_q^2)$ (compute) |
| Setup cost | $O(N^2 N_q^2)$ (full assembly) | $O(N N_q + N_t\log N_t)$ (cache build) |

The matvec has the same asymptotic $O(N^2)$ complexity, while the matrix-free
version recomputes Green-function evaluations instead of reading stored matrix
entries. Compare peak memory and total solve time for the target workload.

---

## 2. The EFIEApplyCache

The internal struct `EFIEApplyCache` precomputes all geometry and basis function data needed for entry evaluation. It is built once and shared by all subsequent calls.

| Field | Type | Description |
|-------|------|-------------|
| `mesh` | `TriMesh` | Triangle mesh reference |
| `rwg` | `RWGData` | RWG basis function data |
| `k` | scalar | Wavenumber |
| `omega_mu0` | scalar | $\omega\mu_0 = k \eta_0$ |
| `wq` | `Vector{Float64}` | Quadrature weights ($N_q$ values) |
| `Nq` | `Int` | Number of quadrature points per triangle |
| `quad_pts` | `Vector{Vector{Vec3}}` | Quadrature points for each triangle |
| `areas` | `Vector{Float64}` | Area of each triangle |
| `tri_ids` | `Matrix{Int}` $(2 \times N)$ | Two support triangles per RWG edge |
| `div_vals` | `Matrix{Float64}` $(2 \times N)$ | Precomputed $\nabla \cdot \mathbf{f}_n$ per support triangle |
| `rwg_vals` | `Vector{NTuple{2,Vector{Vec3}}}` | RWG values at all quad points |
| `adjacent` | `TriangleAdjacency` | Compact-row edge-neighbor table (`offsets` + `neighbors`) |
| `wq_hi`, `quad_pts_hi`, `rwg_vals_hi` | vectors | High-order data for self/adjacent near-singular integration |

Each RWG basis function $\mathbf{f}_n$ lives on two triangles $T_n^+$ and $T_n^-$. The cache precomputes quadrature points on every triangle, RWG basis values at those points, and divergence values -- so that evaluating $Z_{mn}$ reduces to looking up cached values and computing $G(\mathbf{r}, \mathbf{r}')$ at the quadrature point pairs.

Triangle edge records are sorted once to build the compact adjacency, adding
$O(N_t\log N_t)$ setup work. Resident cache storage remains
$O(N_tN_q + NN_q + N_{\mathrm{adj}})$; it does not contain an
$N_t\times N_t$ adjacency matrix.

---

## 3. Single Entry Evaluation

For a given pair $(m, n)$, `_efie_entry(cache, m, n)` computes $Z_{mn}$ by:

1. Looping over the 2 support triangles of basis $m$ times 2 of basis $n$ (4 triangle pairs).
2. **Self-cell** ($T_m^{(i)} = T_n^{(j)}$): calls `self_cell_contribution` for singular integration.
3. **Edge-adjacent**: uses the high-order near-singular integration path.
4. **Other pairs**: double loop over $N_q \times N_q$ quadrature points, accumulating:

```math
\text{vec} = \bigl(\mathbf{f}_m(\mathbf{r}) \cdot \mathbf{f}_n(\mathbf{r}')\bigr) G(\mathbf{r}, \mathbf{r}'), \quad
\text{scl} = \frac{(\nabla \cdot \mathbf{f}_m)(\nabla' \cdot \mathbf{f}_n)}{k^2} G(\mathbf{r}, \mathbf{r}')
```

5. Final multiply by $-i\omega\mu_0$.

Each entry costs at most $4 \times N_q^2$ Green's function evaluations:

| `quad_order` | $N_q$ | Max evaluations per entry |
|-------------|--------|---------------------------|
| 1           | 1      | 4                         |
| 3           | 3      | 36                        |
| 4           | 4      | 64                        |
| 7           | 7      | 196                       |

---

## 4. MatrixFreeEFIEOperator

The user-facing type wraps an `EFIEApplyCache` and presents the Julia `AbstractMatrix` interface:

```julia
struct MatrixFreeEFIEOperator{T, TC<:EFIEApplyCache} <: AbstractMatrix{T}
    cache::TC
end
```

It implements:

- **`size(A)`** -- returns $(N, N)$.
- **`A[i, j]`** -- delegates to `_efie_entry(cache, i, j)`.
- **`mul!(y, A, x)`** -- in-place matvec, row-by-row: $y_m = \sum_n Z_{mn} x_n$.
- **`A * x`** -- allocating matvec.
- **`adjoint(A)`** -- returns `MatrixFreeEFIEAdjointOperator` (zero cost).

The matvec recomputes all $N^2$ entries per call at a total cost of $O(N^2 \cdot N_q^2)$. Physically, the operator maps a current coefficient vector to tested tangential electric fields, computed directly from the Green's function without materializing the full interaction matrix.

---

## 5. MatrixFreeEFIEAdjointOperator

```julia
struct MatrixFreeEFIEAdjointOperator{T, TO<:MatrixFreeEFIEOperator{T}} <: AbstractMatrix{T}
    op::TO
end
```

The adjoint entry is the conjugate transpose:

```math
[\mathbf{Z}^\dagger]_{ij} = \overline{Z_{ji}}
```

The adjoint matvec computes $y_n = \sum_m \overline{Z_{mn}} \, x_m$ and is used by `solve_gmres_adjoint` for the adjoint system:

```math
\mathbf{Z}^\dagger \boldsymbol{\lambda} = \mathbf{q}
```

Because `adjoint(A)` returns a lightweight wrapper (no entries are copied or conjugated), forming the adjoint is a zero-cost operation.

---

## 6. Constructor API

```julia
A = matrixfree_efie_operator(mesh, rwg, k;
        quad_order=3, eta0=376.730313668,
        mesh_precheck=true, allow_boundary=true,
        require_closed=false, area_tol_rel=1e-12)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mesh` | required | `TriMesh` geometry |
| `rwg` | required | `RWGData` basis functions |
| `k` | required | Wavenumber (can be complex for complex-step differentiation) |
| `quad_order` | 3 | Triangle quadrature order |
| `eta0` | 376.73 | Free-space impedance $\eta_0$ |
| `mesh_precheck` | `true` | Run mesh quality assertions |
| `allow_boundary` | `true` | Allow boundary (non-closed) meshes |
| `require_closed` | `false` | Require a closed (watertight) mesh |
| `area_tol_rel` | `1e-12` | Relative tolerance for degenerate-triangle area check |

Returns a `MatrixFreeEFIEOperator{ComplexF64}` that can be passed directly to `solve_gmres`, `build_nearfield_preconditioner`, or used with standard Julia linear algebra.

---

## 7. Where Matrix-Free Operators Are Used

**Pure matrix-free GMRES** -- solve without any dense matrix:

```julia
A = matrixfree_efie_operator(mesh, rwg, k)
P_nf = build_nearfield_preconditioner(A, 1.0 * lambda0)
I, stats = solve_gmres(A, v; preconditioner=P_nf)
```

**ACA block assembly** -- `build_aca_operator` internally creates an `EFIEApplyCache` and uses `_efie_entry` for both pivot selection and dense near-block fill.

**Near-field preconditioner** -- `build_nearfield_preconditioner(A, cutoff)` reads entries via `A[i, j]` to build a sparse approximation whose LU factorization serves as the preconditioner.

**Individual entry queries** -- for debugging or custom algorithms:

```julia
z_mn = efie_entry(A, m, n)   # explicit function call
z_mn = A[m, n]                # standard indexing
```

---

## 8. Choosing a representation

| Requirement | Approach | Storage model | Matvec model |
|-------------|----------|---------------|--------------|
| Materialized entries or direct LU | `assemble_Z_efie` | $O(N^2)$ | $O(N^2)$ |
| Avoid dense storage while retaining exact on-demand entries | `matrixfree_efie_operator` + GMRES | Geometry/cache data | $O(N^2N_q^2)$ per iteration |
| Compress admissible far-field blocks | `build_aca_operator` | Problem- and rank-dependent | Problem- and rank-dependent |
| Hierarchical multipole evaluation | `build_mlfma_operator` | Sparse near field plus hierarchical data | Expected $O(N\log N)$ under implementation assumptions |
| Adjoint solve | All four representations support adjoint application | --- | --- |

With its current defaults, `solve_scattering(method=:auto)` dispatches to
dense-direct through `N=2_000`, dense-GMRES through `N=10_000`, ACA-GMRES
through `N=50_000`, and MLFMA above that. These configurable thresholds select
code paths; benchmark before treating them as performance crossovers.

---

## 9. Complete Usage Example

```julia
using DiffMoM

# --- Geometry ---
freq = 1e9
c0 = 299_792_458.0
lambda0 = c0 / freq
k = 2pi / lambda0

mesh = read_obj_mesh("antenna.obj")
rwg = build_rwg(mesh)
N = rwg.nedges
println("Unknowns: N = $N")

# --- Matrix-free operator (no dense Z allocated) ---
A = matrixfree_efie_operator(mesh, rwg, k; quad_order=3)

# --- Query individual entries ---
println("Z[1,1] = ", A[1, 1])
println("Z[1,2] = ", efie_entry(A, 1, 2))

# --- Build near-field preconditioner ---
P_nf = build_nearfield_preconditioner(A, 1.0 * lambda0)

# --- Excitation ---
k_hat = Vec3(0.0, 0.0, -1.0)
pol = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k * k_hat, 1.0, pol)

# --- Forward solve ---
I_sol, stats = solve_gmres(A, v; preconditioner=P_nf)
println("GMRES converged in $(stats.niter) iterations")

# --- Adjoint solve ---
rhs_adj = rand(ComplexF64, N)
lambda, adj_stats = solve_gmres_adjoint(A, rhs_adj; preconditioner=P_nf)
println("Adjoint GMRES: $(adj_stats.niter) iterations")
```

---

## Code Mapping

| Source file | Key types / functions | Role |
|-------------|----------------------|------|
| `src/assembly/EFIE.jl` | `EFIEApplyCache`, `_efie_entry`, `_build_efie_cache` | Internal cache and entry evaluator |
| `src/assembly/EFIE.jl` | `MatrixFreeEFIEOperator`, `MatrixFreeEFIEAdjointOperator` | User-facing operator types |
| `src/assembly/EFIE.jl` | `matrixfree_efie_operator`, `efie_entry` | Public constructors and entry API |
| `src/assembly/EFIE.jl` | `assemble_Z_efie` | Dense assembly (uses same cache internally) |
| `src/solver/NearFieldPreconditioner.jl` | `build_nearfield_preconditioner` | Sparse preconditioner from operator entries |
| `src/solver/IterativeSolve.jl` | `solve_gmres`, `solve_gmres_adjoint` | GMRES wrappers accepting any `AbstractMatrix` |
| `src/fast/ACA.jl` | `build_aca_operator` | ACA H-matrix using `_efie_entry` |
| `src/Workflow.jl` | `solve_scattering` | Auto method selection |

---

## Exercises

1. **Memory calculation**: For $N = 15\,000$, compute the dense matrix storage in GiB. Then estimate the `EFIEApplyCache` storage (hint: dominant terms are `quad_pts` at $O(N_t \cdot N_q)$ and `rwg_vals` at $O(N \cdot N_q)$).

2. **Matvec timing**: Create a matrix-free operator and a dense matrix for $N \approx 500$. Time `A * x` for both and compute the ratio. Relate the result to stored-entry reads and on-demand kernel evaluation.

3. **Adjoint verification**: Verify `adjoint(A)[i,j] == conj(A[j,i])` for random index pairs, then check $\langle \mathbf{Z}^\dagger \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{Z}\mathbf{y} \rangle$ for random vectors.

4. **Preconditioner from operator**: Build a near-field preconditioner from a matrix-free operator and from a dense matrix for the same problem. Verify the resulting sparse matrices are identical entry-wise.

5. **Crossover point**: Run a sequence of increasing meshes using both dense-direct and matrix-free-GMRES. Plot total time and peak memory versus $N$ and identify the crossover on your hardware.

---

## Chapter Checklist

- [ ] Quantify the memory cost of a dense EFIE matrix and explain when it becomes prohibitive.
- [ ] Describe what `EFIEApplyCache` stores and why precomputing geometry accelerates entry evaluation.
- [ ] Construct a `MatrixFreeEFIEOperator` using `matrixfree_efie_operator`.
- [ ] Use the operator for matvecs (`A * x`), entry queries (`A[m, n]`), and adjoint formation (`adjoint(A)`).
- [ ] Explain the cost per entry ($O(N_q^2)$) and per matvec ($O(N^2 N_q^2)$).
- [ ] Choose between dense, matrix-free, and ACA from measured memory, time, and accuracy requirements.
- [ ] Integrate the matrix-free operator with `solve_gmres` and `build_nearfield_preconditioner`.

---

## Further Reading

- Barrett, R. et al., *Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods* (1994) -- comprehensive overview of matrix-free iterative methods and their implementation.
- Saad, Y., *Iterative Methods for Sparse Linear Systems* (2003), Ch. 9 -- matrix-free Krylov methods and practical considerations for operator-only access patterns.
- Harrington, R. F., *Field Computation by Moment Methods* (1968) -- the original MoM reference; discusses impedance matrix structure motivating on-demand evaluation.

---

Next: [The `solve_scattering` Workflow](05-solve-scattering-workflow.md) introduces the high-level one-call API with automatic method selection and mesh validation.
