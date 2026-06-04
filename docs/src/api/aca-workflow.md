# API: ACA H-Matrix and High-Level Workflow

## Purpose

Reference for the Adaptive Cross Approximation (ACA) H-matrix acceleration and the `solve_scattering` high-level workflow. These subsystems extend the dense EFIE solver to larger problems:

- **ACA H-matrix** compresses the N x N EFIE matrix into O(N log^2 N) storage by exploiting the low-rank structure of far-field interactions, enabling iterative solves without allocating the full dense matrix.
- **`solve_scattering`** is a one-call workflow that automatically selects the best solver method (dense direct, dense GMRES, or ACA GMRES) based on problem size, validates mesh resolution, and handles preconditioner setup.

---

## Cluster Tree

The cluster tree partitions RWG basis functions into a spatial hierarchy by recursive bisection of their bounding box along the longest axis. It determines which matrix blocks are "admissible" (well-separated, hence low-rank) versus "inadmissible" (near-field, stored dense).

### `ClusterNode`

One node in the binary cluster tree.

```julia
struct ClusterNode
    indices::UnitRange{Int}   # range into the permutation array
    bbox_min::Vec3            # bounding box minimum corner
    bbox_max::Vec3            # bounding box maximum corner
    left::Int                 # left child index (0 = leaf)
    right::Int                # right child index (0 = leaf)
    level::Int                # depth in tree (root = 0)
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `indices` | `UnitRange{Int}` | Range of basis function indices (in tree-permuted order) belonging to this cluster. |
| `bbox_min`, `bbox_max` | `Vec3` | Axis-aligned bounding box corners. |
| `left`, `right` | `Int` | Child node indices in the `nodes` array. `0` means this node is a leaf (no children). |
| `level` | `Int` | Depth in the tree. Root is level 0; leaves are at the deepest level. |

---

### `ClusterTree`

Binary cluster tree with flat node storage.

```julia
struct ClusterTree
    nodes::Vector{ClusterNode}  # flat node array, root is nodes[1]
    perm::Vector{Int}           # tree-order -> original index mapping
    iperm::Vector{Int}          # original index -> tree-order mapping
    leaf_size::Int              # maximum cluster size at leaves
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | `Vector{ClusterNode}` | Flat array of tree nodes. The root is `nodes[1]`. |
| `perm` | `Vector{Int}` | Permutation mapping: `perm[k]` is the original basis index of tree-order position `k`. |
| `iperm` | `Vector{Int}` | Inverse permutation: `iperm[i]` is the tree-order position of original basis `i`. |
| `leaf_size` | `Int` | Target leaf size. Clusters with `<= leaf_size` basis functions are not subdivided further. |

---

### `build_cluster_tree(centers; leaf_size=64)`

Build a binary cluster tree by recursive bisection along the longest bounding-box axis.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `centers` | `Vector{Vec3}` | -- | RWG basis function center positions (from `rwg_centers`). |
| `leaf_size` | `Int` | `64` | Maximum number of basis functions per leaf node. Smaller values create deeper trees with finer blocking; larger values create fewer, bigger leaf blocks. |

**Returns:** `ClusterTree`

**Choosing `leaf_size`:**
- `64` (default): Good balance for most problems. Typical for N = 1000--50000.
- `32`: Finer blocking, more low-rank blocks. May improve compression ratio at the cost of more block overhead.
- `128`: Coarser blocking, fewer blocks. Useful when N is very large and block management overhead matters.

---

### Tree Query Functions

#### `cluster_diameter(tree, node_idx)`

Maximum dimension of the bounding box of cluster `node_idx`.

**Returns:** `Float64` diameter in meters.

---

#### `cluster_distance(tree, i, j)`

Minimum distance between the bounding boxes of clusters `i` and `j`. Returns `0.0` if the boxes overlap.

**Returns:** `Float64` distance in meters.

---

#### `is_admissible(tree, i, j; eta=1.5)`

Test the standard H-matrix admissibility condition:

```
min(diam(i), diam(j)) <= eta * dist(i, j)
```

If true, the block (i, j) is well-separated and can be approximated as low-rank via ACA. If false (including overlapping/touching clusters where `dist = 0`), the block must be stored dense.

**Parameters:**
- `tree::ClusterTree`: The cluster tree.
- `i, j::Int`: Node indices.
- `eta::Float64=1.5`: Admissibility parameter. Larger values accept more blocks as admissible (more compression, potentially less accuracy). `eta = 1.5` is a standard choice.

**Returns:** `Bool`

---

#### `is_leaf(tree, node_idx)`

Return `true` if the node has no children.

---

#### `leaf_nodes(tree)`

Return indices of all leaf nodes in the tree.

**Returns:** `Vector{Int}`

---

## ACA Low-Rank Approximation

### `aca_lowrank(cache, row_indices, col_indices; tol=1e-6, max_rank=50)`

Compute a low-rank approximation of a sub-block `Z[row_indices, col_indices]` using partially-pivoted Adaptive Cross Approximation.

The algorithm iteratively selects rows and columns to build a rank-k factorization `U * V'` that approximates the block to within the specified tolerance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache` | `EFIEApplyCache` | -- | Precomputed EFIE data (Green's function, basis evaluations, etc.). Created internally by `build_aca_operator`. |
| `row_indices` | `Vector{Int}` | -- | Original (unpermuted) row basis function indices. |
| `col_indices` | `Vector{Int}` | -- | Original (unpermuted) column basis function indices. |
| `tol` | `Float64` | `1e-6` | Convergence tolerance. ACA stops when the Frobenius norm of the latest rank-1 update falls below `tol * ||accumulated approximation||_F`. |
| `max_rank` | `Int` | `50` | Maximum rank. ACA stops after `max_rank` iterations even if tolerance is not met. |

**Returns:** Tuple `(U, V)` where:
- `U::Matrix{ComplexF64}` of size `(m, k)` -- left factor.
- `V::Matrix{ComplexF64}` of size `(n, k)` -- right factor (the approximation is `U * V'`).

---

## ACA Operator

### `ACAOperator{TC} <: AbstractMatrix{ComplexF64}`

H-matrix operator assembled via ACA. Supports `mul!` for GMRES and `getindex` for preconditioner construction. It stores the EFIE matrix in compressed form: near-field blocks as dense sub-matrices, far-field blocks as low-rank factorizations.

```julia
struct ACAOperator{TC<:EFIEApplyCache} <: AbstractMatrix{ComplexF64}
    cache::TC
    tree::ClusterTree
    dense_blocks::Vector{DenseBlock}
    lowrank_blocks::Vector{LowRankBlock}
    N::Int
end
```

**Key properties:**
- **`size(A)` = `(N, N)`**: Same dimensions as the full EFIE matrix.
- **`A[i, j]`**: Falls back to `_efie_entry(cache, i, j)` for element access. This allows the near-field preconditioner to be built from an `ACAOperator` without forming the dense matrix.
- **`mul!(y, A, x)`**: O(N log^2 N) matvec via dense blocks (BLAS `gemv`) and low-rank blocks (`U * (V' * x)`).
- **`adjoint(A)`**: Returns `ACAAdjointOperator` for adjoint GMRES solves.

---

### `ACAAdjointOperator{TA} <: AbstractMatrix{ComplexF64}`

Adjoint wrapper for an `ACAOperator`. Implements `mul!(y, A', x)` by transposing the block structure: dense blocks use `data'`, low-rank blocks swap `U` and `V`.

---

### `DenseBlock` and `LowRankBlock`

Internal storage types for H-matrix blocks:

```julia
struct DenseBlock
    row_range::UnitRange{Int}   # rows in tree-permuted order
    col_range::UnitRange{Int}   # columns in tree-permuted order
    data::Matrix{ComplexF64}    # dense sub-matrix
end

struct LowRankBlock
    row_range::UnitRange{Int}
    col_range::UnitRange{Int}
    U::Matrix{ComplexF64}       # (m, k) left factor
    V::Matrix{ComplexF64}       # (n, k) right factor; block approx = U * V'
end
```

---

### `build_aca_operator(mesh, rwg, k; kwargs...)`

Build an H-matrix EFIE operator using Adaptive Cross Approximation. This is the main entry point for compressed EFIE assembly.

Dense near-field blocks use triangle-pair batched Green's function evaluation for fewer redundant kernel calls. Block construction is parallelized with `Threads.@threads`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `k` | Real or Complex | -- | Wavenumber (rad/m). |
| `leaf_size` | `Int` | `64` | Cluster tree leaf size. |
| `eta` | `Float64` | `1.5` | Admissibility parameter. |
| `aca_tol` | `Float64` | `1e-6` | ACA convergence tolerance for low-rank blocks. |
| `max_rank` | `Int` | `50` | Maximum rank per low-rank block. |
| `quad_order` | `Int` | `3` | Quadrature order for EFIE entry evaluation. |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance. |
| `mesh_precheck` | `Bool` | `true` | Run mesh quality checks. |

**Returns:** `ACAOperator`

**Usage with GMRES:**
```julia
A_aca = build_aca_operator(mesh, rwg, k; leaf_size=64, aca_tol=1e-6)
P_nf = build_nearfield_preconditioner(mesh, rwg, k, 1.0 * lambda0)
I_coeffs, stats = solve_gmres(A_aca, v; preconditioner=P_nf)
println("GMRES iterations: ", stats.niter)
```

**Performance characteristics:**
- Assembly: O(N log^2 N) time and memory (vs O(N^2) for dense).
- Matvec: O(N log^2 N) per iteration (vs O(N^2) for dense).
- The ACA operator plugs directly into all existing GMRES and preconditioner infrastructure.

**Current limitation:** Impedance-loaded surfaces are not supported with ACA in v1. Use ACA for PEC-only problems; for impedance optimization, use dense assembly.

---

## High-Level Workflow

### `solve_scattering(mesh, freq_hz, excitation; kwargs...)`

One-call scattering solve that automatically selects the solver method, validates mesh resolution, builds preconditioners, and returns a comprehensive result.

**Required parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mesh` | `TriMesh` | Triangle surface mesh. |
| `freq_hz` | `Real` | Frequency in Hz (must be > 0). |
| `excitation` | `AbstractExcitation` or `Vector{ComplexF64}` | Either an excitation object (e.g., `PlaneWaveExcitation`) or a pre-assembled RHS vector. |

**Method selection keywords:**

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `method` | `Symbol` | `:auto` | `:auto`, `:dense_direct`, `:dense_gmres`, `:aca_gmres`, or `:mlfma`. |
| `dense_direct_limit` | `Int` | `2000` | N threshold for dense direct (auto mode). |
| `dense_gmres_limit` | `Int` | `10000` | N threshold for dense GMRES vs ACA (auto mode). |
| `mlfma_threshold` | `Int` | `50000` | N threshold for ACA GMRES vs MLFMA (auto mode). |

**Auto method selection logic:**
- `N <= 2000`: Dense EFIE assembly + LU direct solve.
- `2000 < N <= 10000`: Dense EFIE + NF-preconditioned GMRES.
- `10000 < N <= 50000`: ACA H-matrix + NF-preconditioned GMRES.
- `N > 50000`: MLFMA + NF-preconditioned GMRES. See [mlfma.md](mlfma.md).

**Mesh validation keywords:**

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `check_resolution` | `Bool` | `true` | Run mesh resolution check against frequency. |
| `points_per_wavelength` | `Real` | `10.0` | Target mesh density (edges per wavelength). |
| `error_on_underresolved` | `Bool` | `false` | Throw error instead of warning for under-resolved meshes. |

**Solver settings:**

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `gmres_tol` | `Float64` | `1e-6` | GMRES relative tolerance. |
| `gmres_maxiter` | `Int` | `300` | Maximum GMRES iterations. |

**NF preconditioner settings:**

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `nf_cutoff_lambda` | `Float64` | `1.0` | Near-field cutoff in wavelengths. |
| `preconditioner` | `Symbol` | `:auto` | `:auto` (LU for GMRES methods), `:lu`, `:diag`, or `:none`. |

**ACA settings:**

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `aca_tol` | `Float64` | `1e-6` | ACA low-rank approximation tolerance. |
| `aca_leaf_size` | `Int` | `64` | Cluster tree leaf size. |
| `aca_eta` | `Float64` | `1.5` | Admissibility parameter. |
| `aca_max_rank` | `Int` | `50` | Maximum rank per low-rank block. |

**General settings:**

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `verbose` | `Bool` | `true` | Print progress info (N, method, timing). |
| `quad_order` | `Int` | `3` | Quadrature order for EFIE entry evaluation. |
| `c0` | `Real` | `299792458.0` | Speed of light (m/s). |

**Returns:** `ScatteringResult` (see [types.md](types.md)).

**Example:**
```julia
freq = 1e9
k_vec = Vec3(0, 0, -2pi * freq / 3e8)
pw = make_plane_wave(k_vec, 1.0, Vec3(1, 0, 0))

result = solve_scattering(mesh, freq, pw; verbose=true)
println("Method: ", result.method, ", N = ", result.N)
println("Assembly: ", round(result.assembly_time_s, digits=2), " s")
println("Solve: ", round(result.solve_time_s, digits=2), " s")
if result.gmres_iters >= 0
    println("GMRES: ", result.gmres_iters, " iterations")
end
I_coeffs = result.I_coeffs
```

**Forcing a specific method:**
```julia
# Force ACA even for small problems (e.g., for testing)
result = solve_scattering(mesh, freq, pw; method=:aca_gmres)
```

---

### `ScatteringResult`

Result type returned by `solve_scattering`. See [types.md](types.md) for full field documentation.

**Quick access pattern:**
```julia
result = solve_scattering(mesh, freq, pw)
I = result.I_coeffs            # current coefficients
method = result.method          # :dense_direct, :dense_gmres, :aca_gmres, or :mlfma
N = result.N                    # number of unknowns
```

---

## Typical Workflows

### Small problem (N < 2000)
```julia
result = solve_scattering(mesh, freq, pw)
# Auto-selects dense_direct: assembles full Z, solves Z \ v
```

### Medium problem (2000 < N < 10000)
```julia
result = solve_scattering(mesh, freq, pw)
# Auto-selects dense_gmres: assembles full Z, builds NF preconditioner, GMRES
```

### Large problem (N > 10000)
```julia
result = solve_scattering(mesh, freq, pw)
# Auto-selects aca_gmres: ACA H-matrix, NF preconditioner, GMRES
```

### Manual ACA + post-processing
```julia
# Build ACA operator for reuse across multiple RHS
rwg = build_rwg(mesh)
k = 2pi * freq / 3e8
A_aca = build_aca_operator(mesh, rwg, k)
P_nf = build_nearfield_preconditioner(mesh, rwg, k, 1.0 * lambda0)

# Solve for multiple excitations
for pw in plane_waves
    v = assemble_excitation(mesh, rwg, pw)
    I, stats = solve_gmres(A_aca, v; preconditioner=P_nf)
    # ... post-processing ...
end
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/fast/ClusterTree.jl` | `ClusterNode`, `ClusterTree`, `build_cluster_tree`, admissibility queries |
| `src/fast/ACA.jl` | `ACAOperator`, `build_aca_operator`, `aca_lowrank`, H-matrix matvec |
| `src/Workflow.jl` | `solve_scattering`, auto method selection, mesh validation |
| `src/Types.jl` | `ScatteringResult` |

---

## Exercises

- **Basic:** Build a cluster tree for a 5x5 plate mesh. Print the number of nodes, leaves, and the tree depth. Verify that `sum(length(node.indices) for node in leaf_nodes)` equals N.
- **Practical:** For a medium-sized mesh (N ~ 1000), compare `assemble_Z_efie` + `Z \ v` against `build_aca_operator` + `solve_gmres`. Verify the solutions agree to within GMRES tolerance.
- **Challenge:** Sweep `aca_tol` from `1e-4` to `1e-8` and measure both the total low-rank storage (sum of ranks across all blocks) and the matvec accuracy vs the dense matrix. Plot compression ratio vs accuracy.
