# API: MLFMA (Multi-Level Fast Multipole Algorithm)

## Purpose

The MLFMA module provides an O(N log N) matrix-vector product for the EFIE system matrix, enabling iterative solution of large scattering problems that are too expensive for dense O(N^2) assembly. The MLFMA operator integrates with the existing GMRES/preconditioner infrastructure and is used automatically by `solve_scattering` for N > 50,000.

---

## Types

### `SphereSampling`

Spherical sampling grid for MLFMA plane-wave representation at a given truncation order.

```julia
struct SphereSampling
    L::Int                    # truncation order
    ntheta::Int
    nphi::Int
    npts::Int                 # = ntheta * nphi
    theta::Vector{Float64}
    phi::Vector{Float64}
    weights::Vector{Float64}
    khat::Matrix{Float64}     # (3, npts) unit direction vectors
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `L` | `Int` | Truncation order for the multipole expansion. |
| `ntheta` | `Int` | Number of theta (polar) sampling points (`L + 1`). |
| `nphi` | `Int` | Number of phi (azimuthal) sampling points (`2L + 2`). |
| `npts` | `Int` | Total sample points (`ntheta * nphi`). |
| `theta` | `Vector{Float64}` | Polar angles in radians (from Gauss-Legendre nodes). |
| `phi` | `Vector{Float64}` | Azimuthal angles in radians (uniform grid). |
| `weights` | `Vector{Float64}` | Quadrature weights (GL weight * dphi). |
| `khat` | `Matrix{Float64}` | `(3, npts)` unit direction vectors `[sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]`. |

---

### `MLFMAOperator <: AbstractMatrix{ComplexF64}`

Matrix-free operator that computes EFIE matrix-vector products using the MLFMA algorithm. Supports `size`, `eltype`, `mul!`, `*`, `adjoint`, and element access `A[i,j]` (falls back to `Z_near[i,j]`).

```julia
struct MLFMAOperator <: AbstractMatrix{ComplexF64}
    octree::Octree
    Z_near::SparseMatrixCSC{ComplexF64,Int}
    k::Float64
    eta0::Float64
    prefactor::ComplexF64
    samplings::Vector{SphereSampling}
    trans_factors::Vector{Dict{NTuple{3,Int}, Vector{ComplexF64}}}
    bf_patterns::Array{ComplexF64,3}
    interp_theta::Vector{Matrix{Float64}}
    interp_phi::Vector{Matrix{Float64}}
    agg_filters::Vector{Vector{Matrix{Float64}}}
    disagg_filters::Vector{Vector{Matrix{Float64}}}
    N::Int
end
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `octree` | Octree spatial decomposition. See [octree.md](octree.md). |
| `Z_near` | Sparse near-field matrix (neighbor interactions). |
| `k` | Wavenumber (rad/m). |
| `samplings` | Per-level spherical sampling grids (indexed by `level - 1`). |
| `trans_factors` | Per-level precomputed translation factors. |
| `bf_patterns` | `(4, npts_leaf, N)` radiation patterns per BF: components 1:3 = vector, 4 = scalar (div/k). |
| `agg_filters` | Per-level per-m theta filters for aggregation (child to parent). |
| `disagg_filters` | Per-level per-m theta filters for disaggregation (parent to child). |
| `N` | Number of RWG unknowns. |

**Supported operations:**

- `size(A)` returns `(N, N)`
- `A * x` and `mul!(y, A, x)` compute the MLFMA matvec in O(N log N)
- `adjoint(A)` returns an `MLFMAAdjointOperator`
- `A[i, j]` returns `Z_near[i, j]` (near-field entry or zero)

---

### `MLFMAAdjointOperator <: AbstractMatrix{ComplexF64}`

Hermitian adjoint of `MLFMAOperator`. Obtained via `adjoint(A)`. Uses conjugated translation factors and swapped aggregation/disaggregation roles.

```julia
struct MLFMAAdjointOperator <: AbstractMatrix{ComplexF64}
    op::MLFMAOperator
end
```

---

## Functions

### `build_mlfma_operator(mesh, rwg, k; kwargs...)`

Build an MLFMA operator for the EFIE system. This is the main entry point for constructing the fast solver.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `k` | `Float64` | -- | Wavenumber (rad/m). |
| `leaf_lambda` | `Float64` | `0.25` | Leaf box edge length in wavelengths. Controls octree depth. |
| `quad_order` | `Int` | `3` | Surface quadrature order for radiation patterns and near-field. |
| `precision` | `Int` | `3` | Translation truncation precision parameter (digits of accuracy). |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance. |
| `verbose` | `Bool` | `false` | Print progress messages. |

**Returns:** `MLFMAOperator`.

**Choosing `leaf_lambda`:**

| `leaf_lambda` | Typical levels | Matvec error | Recommendation |
|---------------|---------------|--------------|----------------|
| 3.0 | 4 | ~0.0007% | Maximum accuracy |
| 2.0 | 4 | ~0.004% | High accuracy |
| 1.0 | 5 | ~0.15% | **Recommended minimum for production** |
| 0.75 | 6 | ~11% | Unstable (L > kr in translation) |

Use `leaf_lambda >= 1.0` for production runs. Smaller values increase octree depth but cause translation operator instability when the truncation order exceeds `k * r` for the box separation distance.

**Example:**

```julia
k = 2pi * freq / c0
A_mlfma = build_mlfma_operator(mesh, rwg, k; leaf_lambda=1.0, verbose=true)

# Build preconditioner and solve
P = build_mlfma_preconditioner(A_mlfma; ilu_tau=1e-2)
I, stats = solve_gmres(A_mlfma, v; preconditioner=P)
println("GMRES iters: ", stats.niter)
```

---

### `assemble_mlfma_nearfield(octree, mesh, rwg, k; quad_order=3, eta0=376.730313668)`

Assemble the near-field (neighbor interaction) sparse matrix for MLFMA. Only computes EFIE entries `Z[m,n]` for BF pairs `(m, n)` that belong to neighboring leaf boxes in the octree. Returns a CSC sparse matrix in the original BF ordering.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `octree` | `Octree` | -- | Octree spatial decomposition. |
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `k` | `Float64` | -- | Wavenumber (rad/m). |
| `quad_order` | `Int` | `3` | Quadrature order for EFIE entry evaluation. |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance. |

**Returns:** `SparseMatrixCSC{ComplexF64, Int}` of size `(N, N)`.

**Note:** This is called internally by `build_mlfma_operator`. You typically do not need to call it directly unless building a custom workflow.

---

## MLFMA Algorithm Overview

The MLFMA matvec `y = A * x` computes:

1. **Near-field:** `y = Z_near * x` (sparse matvec)
2. **Leaf aggregation:** Compute plane-wave radiation patterns weighted by `x`
3. **Bottom-up aggregation:** Interpolate child-level patterns to parent sampling using per-m spectral filters (associated Legendre `P_l^m` band-limiting), then apply phase shift for center translation
4. **Translation:** At each level, multiply aggregated patterns by precomputed translation operators for interaction-list box pairs
5. **Top-down disaggregation:** Phase shift parent incoming fields to child centers, then apply per-m spectral filter to child sampling
6. **Leaf disaggregation:** Project incoming fields onto BF receiving patterns and accumulate into `y`

### Radiation patterns

Each BF has a 4-component radiation pattern `(fx, fy, fz, div/k)` at the leaf-level sampling points. The vector components represent the surface-current contribution; the scalar component (divergence/k) represents the charge contribution. This 4-component representation is equivalent to the standard 2-component `(theta-hat, phi-hat)` representation but avoids coordinate singularities at the poles.

### Per-m spectral filters

Aggregation and disaggregation use per-azimuthal-mode (`m`) spectral filters based on associated Legendre functions `P_l^m`. This correctly band-limits all azimuthal modes, unlike separated Lagrange interpolation which fails for odd `m` near the poles.

---

## Preconditioners for MLFMA

See [assembly-solve.md](assembly-solve.md) for the full preconditioner API. The recommended options for MLFMA:

| Builder | Type | When to use |
|---------|------|-------------|
| `build_mlfma_preconditioner(A; ilu_tau=1e-2)` | `PermutedPrecondData` wrapping `ILUPreconditionerData` | **Recommended.** Reorders Z_near for block-banded ILU. |
| `build_block_diag_preconditioner(A)` | `BlockDiagPrecondData` | Fast fallback when ILU is too slow. |
| `build_nearfield_preconditioner(A.Z_near; factorization=:ilu)` | `ILUPreconditionerData` | Direct ILU on Z_near without reordering. |

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/fast/MLFMA.jl` | `MLFMAOperator`, `MLFMAAdjointOperator`, `build_mlfma_operator`, `SphereSampling`, `assemble_mlfma_nearfield`, translation operators, spectral filters, matvec |
| `src/fast/Octree.jl` | Octree data structures used by MLFMA. See [octree.md](octree.md). |
| `src/solver/NearFieldPreconditioner.jl` | `build_mlfma_preconditioner`, `build_block_diag_preconditioner` |
