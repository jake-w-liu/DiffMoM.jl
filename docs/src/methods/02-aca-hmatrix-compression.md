# ACA and H-Matrix Compression

## Purpose

Dense EFIE matrices scale as $O(N^2)$ in storage and $O(N^3)$ for direct
factorization, limiting practical problem sizes to a few thousand unknowns on
workstation hardware. This chapter explains how `DiffMoM.jl` uses
Hierarchical Matrices (H-matrices) with Adaptive Cross Approximation (ACA) to
compress the EFIE operator into a mixture of dense near-field blocks and
low-rank far-field blocks, reducing both storage and matrix-vector product cost
to $O(N \log^2 N)$.

We cover the full pipeline: constructing a binary cluster tree from RWG basis
function centers, partitioning the matrix into admissible and inadmissible
blocks, compressing admissible blocks via partially-pivoted ACA, assembling the
resulting H-matrix operator, and using it with GMRES and the near-field
preconditioner.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why far-field blocks of the EFIE matrix are numerically low-rank and why near-field blocks are not.
2. Describe the binary cluster tree construction algorithm and its role in block partitioning.
3. State the standard H-matrix admissibility condition and its geometric meaning.
4. Walk through the partially-pivoted ACA algorithm step by step.
5. Explain how `ACAOperator` performs a matrix-vector product using dense and low-rank blocks.
6. Use `build_aca_operator` and `solve_scattering` with ACA in practice.
7. Analyze the complexity trade-offs between dense and H-matrix approaches.

---

## 1. The Dense Matrix Problem

### 1.1 Cost of Dense EFIE Assembly

The EFIE discretization with $N$ RWG basis functions produces an impedance matrix $\mathbf{Z} \in \mathbb{C}^{N \times N}$. Each entry requires a double surface integral over the supports of two basis functions:

```math
Z_{mn} = -i\omega\mu_0 \left[ \iint \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{f}_n(\mathbf{r}') \, G(\mathbf{r},\mathbf{r}') \, dS \, dS' - \frac{1}{k^2} \iint (\nabla_s \cdot \mathbf{f}_m)(\nabla_s' \cdot \mathbf{f}_n) \, G(\mathbf{r},\mathbf{r}') \, dS \, dS' \right].
```

For each of the $N^2$ entries, the double integral is evaluated with $N_q^2$ quadrature point pairs (where $N_q$ is the number of points per triangle), giving an assembly cost of $O(N^2 N_q^2)$. Storing the matrix requires $N^2$ complex numbers:

```math
\text{Storage} = N^2 \times 16 \;\text{bytes} \quad (\texttt{ComplexF64}).
```

### 1.2 Practical Scaling

| $N$ | Storage | LU Factor Time | Dense Matvec |
|-----|---------|----------------|--------------|
| 1,000 | 15 MiB | < 1 s | trivial |
| 5,000 | 381 MiB | ~10 s | ~ms |
| 10,000 | 1.5 GiB | ~1 min | ~10 ms |
| 50,000 | 37.3 GiB | hours | ~0.2 s |
| 100,000 | 149 GiB | infeasible | ~1 s |

For moderate-to-large problems ($N > 10{,}000$), dense assembly and direct factorization become impractical. The matvec cost alone ($O(N^2)$ per iteration) makes GMRES slow, and the storage cost exceeds available memory on typical workstations.

### 1.3 The Key Physical Observation

Not all entries of $\mathbf{Z}$ are equally important. The Green's function $G(\mathbf{r},\mathbf{r}') = e^{-ikR}/(4\pi R)$ decays as $1/R$ with increasing separation. When two groups of basis functions are **far apart**, their mutual interactions vary smoothly as a function of position within each group. This smoothness implies that the corresponding sub-block of $\mathbf{Z}$ is **numerically low-rank**: it can be approximated accurately by a product $\mathbf{U}\mathbf{V}'$ where $\mathbf{U}$ and $\mathbf{V}$ have far fewer columns than the block dimension.

Conversely, when basis functions are **close together**, the Green's function varies rapidly (and may even be singular for self-interactions), so these blocks must be stored in full.

This observation is the foundation of H-matrix compression: partition $\mathbf{Z}$ into blocks, store near-field blocks dense, and compress far-field blocks as low-rank factorizations.

---

## 2. Hierarchical Matrix (H-Matrix) Concept

### 2.1 Block Partition of the Matrix

An H-matrix partitions the index set $\{1, \dots, N\}$ hierarchically using a **cluster tree** and then classifies each block pair as either:

- **Admissible** (far-field): the block can be approximated as low-rank $\mathbf{U}\mathbf{V}'$.
- **Inadmissible** (near-field): the block must be stored as a dense sub-matrix.

Schematically, the matrix looks like:

```
       ┌─────────────────────────────┐
       │  Dense   │  U₁V₁'  │ U₂V₂' │
       │  (near)  │  (far)   │ (far)  │
       ├──────────┼──────────┼────────┤
       │  U₃V₃'  │  Dense   │ U₄V₄' │
       │  (far)   │  (near)  │ (far)  │
       ├──────────┼──────────┼────────┤
       │  U₅V₅'  │  U₆V₆'  │ Dense  │
       │  (far)   │  (far)   │ (near) │
       └──────────┴──────────┴────────┘
```

The near-field blocks lie along the block diagonal (where source and test clusters are close), while far-field blocks populate the off-diagonal regions.

### 2.2 The Admissibility Condition

The standard admissibility criterion for determining whether a block $(i, j)$ between cluster $i$ (row) and cluster $j$ (column) can be treated as low-rank is:

```math
\min\bigl(\operatorname{diam}(i),\; \operatorname{diam}(j)\bigr) \le \eta \cdot \operatorname{dist}(i, j),
```

where:

- $\operatorname{diam}(i)$ is the diameter (maximum bounding-box span) of cluster $i$.
- $\operatorname{dist}(i, j)$ is the minimum distance between the bounding boxes of clusters $i$ and $j$.
- $\eta$ is the admissibility parameter (default $\eta = 1.5$ in the package).

**Geometric interpretation**: a block is admissible when the two clusters are separated by at least $1/\eta$ times the size of the smaller cluster. Larger $\eta$ makes more blocks admissible (more compression) but may require higher ranks for accurate approximation. Smaller $\eta$ is more conservative (more dense blocks) but yields tighter low-rank approximations.

**Physical interpretation**: when two groups of basis functions are well-separated relative to their size, the Green's function varies slowly across the block. The number of significant degrees of freedom in the interaction is much smaller than the block dimensions, enabling low-rank compression.

### 2.3 Complexity Improvement

For surface meshes embedded in 3D, the H-matrix block partition yields:

```math
\text{Storage} = O(N \log^2 N), \qquad \text{Matvec} = O(N \log^2 N).
```

This is a dramatic improvement over the $O(N^2)$ scaling of the dense approach. The $\log^2 N$ factor arises from the depth of the cluster tree (approximately $\log N$ levels) and the number of blocks at each level.

---

## 3. Binary Cluster Tree

### 3.1 Purpose

The cluster tree recursively partitions the set of RWG basis function centers into groups of geometrically nearby indices. It determines which matrix blocks are "near" (inadmissible, stored dense) and which are "far" (admissible, compressed to low-rank).

### 3.2 Data Structures

The cluster tree is built from two structs defined in `src/fast/ClusterTree.jl`:

```julia
struct ClusterNode
    indices::UnitRange{Int}    # range into the permutation array
    bbox_min::Vec3             # axis-aligned bounding box minimum
    bbox_max::Vec3             # axis-aligned bounding box maximum
    left::Int                  # left child index (0 = leaf)
    right::Int                 # right child index (0 = leaf)
    level::Int                 # depth in the tree (root = 0)
end

struct ClusterTree
    nodes::Vector{ClusterNode} # flat storage, root = nodes[1]
    perm::Vector{Int}          # tree-order -> original index
    iperm::Vector{Int}         # original index -> tree-order
    leaf_size::Int             # maximum cluster size at leaves
end
```

Key design choices:

- **Flat node storage**: All nodes are stored in a single `Vector{ClusterNode}`, with children referenced by integer index. This avoids pointer-chasing and is cache-friendly.
- **Permutation arrays**: `perm[k]` gives the original RWG index for tree position `k`. The inverse `iperm` maps the other direction. These permutations enable efficient reordering of vectors during matvec.
- **Contiguous index ranges**: Each node's `indices` field is a `UnitRange{Int}` into the permutation array, so the indices belonging to any cluster are always contiguous in memory.

### 3.3 Construction Algorithm

The function `build_cluster_tree(centers; leaf_size=64)` takes a `Vector{Vec3}` of RWG basis function centers and builds the tree via recursive bisection:

1. **Compute bounding box**: For the current set of indices, find the axis-aligned bounding box (min and max coordinates along each axis).
2. **Check leaf condition**: If the number of points $\le$ `leaf_size`, create a leaf node and return.
3. **Find longest axis**: Determine which bounding-box dimension (x, y, or z) has the largest span.
4. **Median split**: Sort the indices along that axis and split at the median, producing two roughly equal-sized child groups.
5. **Recurse**: Build left and right subtrees on the two halves.
6. **Build inverse permutation**: After the tree is complete, compute `iperm` from `perm`.

```julia
# Build a cluster tree from RWG centers
centers = rwg_centers(mesh, rwg)
tree = build_cluster_tree(centers; leaf_size=64)
```

### 3.4 Choosing the Leaf Size

The `leaf_size` parameter controls the granularity of the block partition:

- **Smaller leaf size** (e.g., 16): More blocks, finer partition, potentially better compression, but more overhead from block bookkeeping.
- **Larger leaf size** (e.g., 128): Fewer blocks, coarser partition, less overhead, but dense blocks are larger and some compressible regions may be lumped with near-field.
- **Default** (`leaf_size=64`): A good balance for typical electromagnetic problems.

The tree depth is approximately $\log_2(N/\text{leaf\_size})$ levels.

### 3.5 Helper Functions

The cluster tree module provides several utility functions:

| Function | Description |
|----------|-------------|
| `cluster_diameter(tree, i)` | Maximum bounding-box span of cluster `i` |
| `cluster_distance(tree, i, j)` | Minimum bounding-box gap between clusters `i` and `j` |
| `is_admissible(tree, i, j; eta=1.5)` | Test the admissibility condition |
| `is_leaf(tree, i)` | Check if node `i` is a leaf |
| `leaf_nodes(tree)` | Return indices of all leaf nodes |

The admissibility test is implemented as:

```julia
function is_admissible(tree, i, j; eta=1.5)
    d = cluster_distance(tree, i, j)
    d <= 0.0 && return false  # overlapping or touching boxes
    diam_min = min(cluster_diameter(tree, i), cluster_diameter(tree, j))
    return diam_min <= eta * d
end
```

Note that overlapping or touching bounding boxes ($d = 0$) are always inadmissible, regardless of $\eta$. This ensures that self-interactions and near-neighbor interactions, where the Green's function is strongly varying or singular, are always treated with full dense blocks.

---

## 4. Adaptive Cross Approximation (ACA)

### 4.1 The Low-Rank Approximation Problem

For an admissible block with row indices $\mathcal{I}$ (size $m$) and column indices $\mathcal{J}$ (size $n$), we seek a rank-$k$ approximation:

```math
\mathbf{Z}[\mathcal{I}, \mathcal{J}] \approx \mathbf{U} \mathbf{V}', \qquad \mathbf{U} \in \mathbb{C}^{m \times k}, \quad \mathbf{V} \in \mathbb{C}^{n \times k}.
```

A naive approach would compute the full $m \times n$ block and then apply a truncated SVD. This requires $O(mn)$ entry evaluations and $O(mn \min(m,n))$ factorization work, negating the benefit of compression.

ACA achieves the same approximation quality using only $O(k(m+n))$ entry evaluations by adaptively selecting rows and columns that capture the dominant interactions.

### 4.2 Physical Intuition for Low Rank

Why are far-field blocks low-rank? Consider two well-separated groups of basis functions. The Green's function interaction between them can be expanded in a multipole series:

```math
G(\mathbf{r}, \mathbf{r}') = \sum_{\ell=0}^{\infty} \sum_{m=-\ell}^{\ell} \alpha_{\ell m}(\mathbf{r}) \, \beta_{\ell m}(\mathbf{r}'),
```

where $\alpha_{\ell m}$ depends only on the observation point and $\beta_{\ell m}$ depends only on the source point. For well-separated groups, truncating this series at $\ell = L$ gives an approximation with rank $(L+1)^2$ that converges exponentially in $L$. In practice, ranks of 5--30 suffice for engineering accuracy ($10^{-6}$ relative error), regardless of the block size.

### 4.3 The Partially-Pivoted ACA Algorithm

The function `aca_lowrank` in `src/fast/ACA.jl` implements partially-pivoted ACA. The algorithm builds the low-rank factorization one rank at a time:

**Input**: Entry evaluation function, row indices $\mathcal{I}$, column indices $\mathcal{J}$, tolerance `tol`, maximum rank `max_rank`.

**Output**: Matrices $\mathbf{U}$ and $\mathbf{V}$ such that $\mathbf{Z}[\mathcal{I}, \mathcal{J}] \approx \mathbf{U}\mathbf{V}'$.

**Algorithm**:

For $k = 1, 2, \dots, \texttt{max\_rank}$:

1. **Compute residual row** at the pivot row $i_k$:
```math
\tilde{\mathbf{r}}_k = \mathbf{Z}[i_k, \mathcal{J}] - \sum_{j=1}^{k-1} u_j[i_k] \, \mathbf{v}_j^*
```

2. **Select column pivot**: $j_k = \arg\max_{j \notin \text{used}} |\tilde{r}_k[j]|$

3. **Compute residual column** at the pivot column $j_k$:
```math
\tilde{\mathbf{c}}_k = \mathbf{Z}[\mathcal{I}, j_k] - \sum_{j=1}^{k-1} \mathbf{u}_j \, v_j^*[j_k]
```

4. **Form rank-1 update**:
```math
\mathbf{u}_k = \frac{\tilde{\mathbf{c}}_k}{\tilde{r}_k[j_k]}, \qquad \mathbf{v}_k = \overline{\tilde{\mathbf{r}}_k}
```

5. **Check convergence**:
```math
\|\mathbf{u}_k\| \cdot \|\mathbf{v}_k\| < \texttt{tol} \cdot \|\mathbf{U}_k \mathbf{V}_k'\|_F
```

6. **Select next pivot row**: $i_{k+1} = \arg\max_{i \notin \text{used}} |u_k[i]|$

The Frobenius norm $\|\mathbf{U}_k\mathbf{V}_k'\|_F$ is updated incrementally without forming the full product:

```math
\|\mathbf{U}_k \mathbf{V}_k'\|_F^2 = \|\mathbf{U}_{k-1} \mathbf{V}_{k-1}'\|_F^2 + 2\operatorname{Re}\!\sum_{j<k} (\mathbf{u}_j^H \mathbf{u}_k)(\mathbf{v}_j^H \mathbf{v}_k) + \|\mathbf{u}_k\|^2 \|\mathbf{v}_k\|^2.
```

### 4.4 Why Partial Pivoting Works

The column pivot selection (step 2) chooses the column where the current residual row has the largest magnitude. This greedy strategy tends to capture the most significant interactions first, so the approximation converges rapidly.

The row pivot selection (step 6) exploits the observation that the column with the largest new contribution $|u_k[i]|$ is likely to reveal additional structure when used as the next pivot row.

Together, these heuristics make ACA converge in $O(k)$ steps for a rank-$k$ block, with each step requiring $O(m + n)$ entry evaluations. The total cost per block is therefore $O(k(m + n))$.

### 4.5 Entry Evaluation via `_efie_entry`

ACA never forms the full block. Instead, it evaluates individual entries on demand using `_efie_entry` from `EFIEApplyCache`. This function computes a single EFIE matrix element $Z_{mn}$ by looping over the triangle pairs of the two RWG basis functions and evaluating the double surface integral with Gaussian quadrature (including singularity extraction for self-cell pairs).

```julia
function aca_lowrank(cache::EFIEApplyCache,
                     row_indices, col_indices;
                     tol=1e-6, max_rank=50)
    # ... partially-pivoted ACA using _efie_entry(cache, i, j) ...
    return U, V
end
```

The `V` matrix stores the **conjugate** of the residual rows, so the approximation is $\mathbf{U}\mathbf{V}'$ (Hermitian adjoint).

### 4.6 Convergence and Accuracy

The ACA tolerance `tol` controls the approximation accuracy. For typical electromagnetic problems:

| `tol` | Typical Rank | Relative Error | Use Case |
|-------|-------------|----------------|----------|
| $10^{-3}$ | 3--10 | ~0.1% | Quick estimates, visualization |
| $10^{-6}$ | 10--25 | ~$10^{-4}$% | Engineering accuracy (default) |
| $10^{-9}$ | 20--40 | ~$10^{-7}$% | High-precision, gradient computation |

The maximum rank `max_rank` (default 50) is a safety bound that prevents runaway iterations if the block is not truly low-rank (which should not happen for properly admissible blocks).

---

## 5. H-Matrix Assembly

### 5.1 Two-Phase Block Assembly

The function `build_aca_operator` in `src/fast/ACA.jl` assembles the H-matrix operator in two phases:

**Phase 1 (Serial): Block task enumeration**

The recursive function `_enum_block_tasks!` traverses all pairs of cluster tree nodes and classifies each block:

```julia
function _enum_block_tasks!(tasks, tree, row_node, col_node; eta)
    if is_admissible(tree, row_node, col_node; eta=eta)
        push!(tasks, BlockTask(row_node, col_node, :lowrank))
        return
    end
    if is_leaf(tree, row_node) && is_leaf(tree, col_node)
        push!(tasks, BlockTask(row_node, col_node, :dense))
        return
    end
    # Recurse into children (2-way or 4-way split)
    ...
end
```

The recursion follows these rules:

1. **Admissible pair**: Create a `:lowrank` task. Stop recursing.
2. **Both leaves, inadmissible**: Create a `:dense` task. Cannot recurse further.
3. **One is a leaf, the other is not**: Recurse into the non-leaf node's children (2-way split).
4. **Neither is a leaf**: Recurse into all four child combinations (4-way split).

**Phase 2 (Parallel): Block computation**

All block tasks are processed in parallel using `Threads.@threads`:

```julia
Threads.@threads for i in eachindex(tasks)
    task = tasks[i]
    if task.kind == :dense
        _fill_dense_block_batched!(data, cache, row_indices, col_indices)
    else
        U, V = aca_lowrank(cache, row_indices, col_indices;
                           tol=aca_tol, max_rank=max_rank)
    end
end
```

### 5.2 Batched Dense Block Fill

Dense blocks use a specialized routine `_fill_dense_block_batched!` that exploits the structure of the EFIE to reduce redundant Green's function evaluations.

The key optimization: multiple RWG basis functions share the same supporting triangles. Instead of computing $G(\mathbf{r}_q, \mathbf{r}_q')$ separately for each basis function pair, the routine:

1. **Collects unique triangle pairs** referenced by the block's row and column RWG indices.
2. **Precomputes the Green's function** for all $(T_m, T_n)$ triangle pairs at all quadrature point combinations, storing them in a dictionary.
3. **Fills block entries** by looking up the precomputed values.

This triangle-pair caching yields approximately 8 times fewer Green's function evaluations compared to the naive approach, because each triangle typically supports 3 RWG basis functions.

### 5.3 Block Data Structures

The assembled blocks are stored in two simple structs:

```julia
struct DenseBlock
    row_range::UnitRange{Int}   # in tree-permuted order
    col_range::UnitRange{Int}
    data::Matrix{ComplexF64}    # full dense sub-matrix
end

struct LowRankBlock
    row_range::UnitRange{Int}   # in tree-permuted order
    col_range::UnitRange{Int}
    U::Matrix{ComplexF64}       # m x k
    V::Matrix{ComplexF64}       # n x k (stores conjugate)
end
```

The block approximation for a `LowRankBlock` is $\mathbf{U}\mathbf{V}'$, where the conjugate is already folded into $\mathbf{V}$ during ACA.

---

## 6. ACAOperator and Matrix-Vector Product

### 6.1 The Operator Type

The assembled H-matrix is wrapped in an `ACAOperator` that implements the `AbstractMatrix` interface:

```julia
struct ACAOperator{TC<:EFIEApplyCache} <: AbstractMatrix{ComplexF64}
    cache::TC                          # EFIE entry evaluation cache
    tree::ClusterTree                  # cluster tree with permutations
    dense_blocks::Vector{DenseBlock}   # near-field blocks
    lowrank_blocks::Vector{LowRankBlock}  # far-field blocks
    N::Int                             # matrix dimension
end
```

Because `ACAOperator <: AbstractMatrix{ComplexF64}`, it can be passed directly to `solve_gmres`, `build_nearfield_preconditioner`, and other functions that accept `AbstractMatrix` arguments.

### 6.2 Matvec Algorithm

The matrix-vector product $\mathbf{y} = \mathbf{A}\mathbf{x}$ is implemented in `LinearAlgebra.mul!` and proceeds in four steps:

**Step 1: Permute input to tree order**

```math
x_{\text{perm}}[k] = x[\text{perm}[k]], \qquad k = 1, \dots, N.
```

This reorders $\mathbf{x}$ so that indices within each cluster are contiguous, enabling efficient sub-vector extraction.

**Step 2: Dense block contributions (BLAS `gemv`)**

For each dense block with row range $\mathcal{R}$ and column range $\mathcal{C}$:

```math
y_{\text{perm}}[\mathcal{R}] \mathrel{+}= \mathbf{D} \cdot x_{\text{perm}}[\mathcal{C}],
```

where $\mathbf{D}$ is the stored dense sub-matrix. This is a standard BLAS level-2 operation.

**Step 3: Low-rank block contributions (two small `gemv`s)**

For each low-rank block with $\mathbf{U} \in \mathbb{C}^{m \times k}$ and $\mathbf{V} \in \mathbb{C}^{n \times k}$:

```math
\mathbf{t} = \mathbf{V}' \cdot x_{\text{perm}}[\mathcal{C}], \qquad y_{\text{perm}}[\mathcal{R}] \mathrel{+}= \mathbf{U} \cdot \mathbf{t}.
```

The intermediate vector $\mathbf{t} \in \mathbb{C}^k$ is small (typically $k \le 30$), so this requires $O(kn + km)$ operations instead of $O(mn)$.

**Step 4: Un-permute output**

```math
y[\text{perm}[k]] = y_{\text{perm}}[k], \qquad k = 1, \dots, N.
```

### 6.3 Adjoint Matvec

The adjoint operator $\mathbf{A}^H$ is needed for adjoint GMRES solves in gradient computation. It is represented by `ACAAdjointOperator`:

```julia
struct ACAAdjointOperator{TA<:ACAOperator} <: AbstractMatrix{ComplexF64}
    op::TA
end
```

The adjoint matvec transposes each block operation:

- **Dense blocks**: $y_{\text{perm}}[\mathcal{C}] \mathrel{+}= \mathbf{D}^H \cdot x_{\text{perm}}[\mathcal{R}]$ (rows and columns swapped, data conjugate-transposed).
- **Low-rank blocks**: The adjoint of $\mathbf{U}\mathbf{V}'$ is $\mathbf{V}\mathbf{U}'$, so: $\mathbf{t} = \mathbf{U}' \cdot x_{\text{perm}}[\mathcal{R}]$, then $y_{\text{perm}}[\mathcal{C}] \mathrel{+}= \mathbf{V} \cdot \mathbf{t}$.

### 6.4 Fallback `getindex`

For compatibility with the near-field preconditioner (which needs to extract individual matrix entries), `ACAOperator` provides a `getindex` fallback:

```julia
function Base.getindex(A::ACAOperator, i::Int, j::Int)
    return _efie_entry(A.cache, i, j)
end
```

This evaluates the exact EFIE entry on demand using the cached quadrature data. While $O(N_q^2)$ per call (not fast for forming the full matrix), it is efficient for the sparse near-field extraction used by `build_nearfield_preconditioner`.

---

## 7. Complexity Analysis

### 7.1 Operation Costs

| Operation | Dense | ACA H-Matrix |
|-----------|-------|--------------|
| Storage | $O(N^2)$ | $O(N \log^2 N)$ |
| Assembly | $O(N^2 N_q^2)$ | $O(N \log^2 N \cdot k \cdot N_q^2)$ |
| Matvec | $O(N^2)$ | $O(N \log^2 N)$ |
| Direct solve (LU) | $O(N^3)$ | Not supported |
| GMRES (per iteration) | $O(N^2)$ | $O(N \log^2 N)$ |

Here $k$ is the typical rank of low-rank blocks (usually 10--30) and $N_q^2$ is the cost per entry evaluation.

### 7.2 Where the $\log^2 N$ Comes From

The cluster tree has $O(\log N)$ levels. At each level, the total number of index pairs covered by admissible blocks is $O(N)$ (each index appears in a bounded number of interactions at each level). The rank of each low-rank block is bounded by a constant $k$ (independent of $N$ for a fixed tolerance and smooth kernel). Summing over all $O(\log N)$ levels and accounting for the tree traversal overhead gives $O(N \log^2 N)$.

In practice, for typical MoM problems, the $\log^2 N$ factor is small. The crossover point where ACA becomes faster than dense assembly is typically around $N \approx 3{,}000$--$5{,}000$, depending on the geometry and hardware.

### 7.3 Memory Breakdown

For an H-matrix with $B_d$ dense blocks of average size $s_d \times s_d$ and $B_r$ low-rank blocks of average size $s_r \times s_r$ with rank $k$:

```math
\text{Memory} = B_d \cdot s_d^2 \cdot 16 + B_r \cdot 2 k s_r \cdot 16 \quad \text{bytes}.
```

The dense blocks dominate near the diagonal (where clusters are close), while the low-rank blocks dominate in the far field (where $2ks_r \ll s_r^2$). The total is $O(N \log^2 N)$.

### 7.4 Limitation: No Direct Solve

Unlike the dense matrix, the H-matrix does not support direct $LU$ factorization. Solving $\mathbf{A}\mathbf{x} = \mathbf{b}$ requires an iterative solver (GMRES). This is not a significant limitation in practice because:

1. GMRES with a near-field preconditioner converges in $O(1)$ iterations (independent of $N$).
2. Each GMRES iteration costs $O(N \log^2 N)$, so the total solve time is $O(N \log^2 N)$.

For the adjoint solve needed in gradient computation, the `ACAAdjointOperator` provides the necessary adjoint matvec.

---

## 8. Integration with GMRES and Near-Field Preconditioner

### 8.1 Using ACA with GMRES

Because `ACAOperator <: AbstractMatrix`, it plugs directly into the existing GMRES infrastructure. The solve call is identical to the dense case:

```julia
using DiffMoM

# Setup
mesh = read_obj_mesh("antenna.obj")
rwg = build_rwg(mesh)
freq = 3e9
k = 2pi * freq / 299792458.0
lambda0 = 299792458.0 / freq

# Build ACA operator
A_aca = build_aca_operator(mesh, rwg, k;
                            leaf_size=64, eta=1.5, aca_tol=1e-6)

println("Dense blocks: ", length(A_aca.dense_blocks))
println("Low-rank blocks: ", length(A_aca.lowrank_blocks))

# Build near-field preconditioner
P_nf = build_nearfield_preconditioner(A_aca, mesh, rwg, 1.0 * lambda0)

# Assemble excitation
pw = make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(1,0,0))
v = assemble_excitation(mesh, rwg, pw)

# Solve
I_coeffs, stats = solve_gmres(A_aca, v; preconditioner=P_nf)
println("GMRES converged in $(stats.niter) iterations")
```

### 8.2 Near-Field Preconditioner Construction from ACAOperator

The `build_nearfield_preconditioner` function accepts any `AbstractMatrix`, including `ACAOperator`. It extracts individual entries $A[m, n]$ via the `getindex` fallback (which calls `_efie_entry`), keeping only entries where the RWG centers are within the cutoff distance. The extracted sparse matrix is then LU-factorized.

This is efficient because:

- The number of near-field entries is $O(N)$ (each basis function has a bounded number of neighbors within the cutoff).
- Each entry evaluation costs $O(N_q^2)$ (independent of $N$).
- The sparse LU factorization cost depends on the sparsity pattern, not on $N^2$.

### 8.3 The `solve_scattering` High-Level API

For convenience, `solve_scattering` automatically selects ACA for large problems:

```julia
result = solve_scattering(mesh, freq_hz, excitation; method=:auto)
```

The auto-selection logic is:

| Problem Size | Method | Description |
|-------------|--------|-------------|
| $N \le 2{,}000$ | `:dense_direct` | Dense assembly + LU solve |
| $2{,}000 < N \le 10{,}000$ | `:dense_gmres` | Dense + NF-preconditioned GMRES |
| $10{,}000 < N \le 50{,}000$ | `:aca_gmres` | ACA H-matrix + NF-preconditioned GMRES |
| $N > 50{,}000$ | `:mlfma` | MLFMA + NF-preconditioned GMRES |

You can also force ACA explicitly:

```julia
result = solve_scattering(mesh, freq_hz, excitation;
                           method=:aca_gmres,
                           aca_tol=1e-6,
                           aca_leaf_size=64,
                           aca_eta=1.5,
                           aca_max_rank=50,
                           nf_cutoff_lambda=1.0)
```

### 8.4 Preconditioner Effectiveness with ACA

In tested regimes, the near-field preconditioner often gives weak iteration growth with problem size. This is because the preconditioner captures the dominant near-field coupling structure of the Green's function, and the far-field interactions (handled by the low-rank blocks) are better conditioned after preconditioning.

Typical iteration counts with the near-field preconditioner:

| $N$ | Without Preconditioner | With NF Preconditioner |
|-----|----------------------|----------------------|
| 1,000 | 50--100 | 15--25 |
| 5,000 | 150--300 | 15--25 |
| 10,000 | 300--500+ | 15--25 |
| 50,000 | diverges | 15--25 |

The cutoff distance (default $1\lambda$) controls the density of the preconditioner. Larger cutoffs give better preconditioning but cost more to factorize.

---

## 9. Worked Example: ACA on a Rectangular Plate

### 9.1 Problem Setup

Consider a $1\lambda \times 1\lambda$ PEC plate meshed with increasing resolution:

```julia
using DiffMoM

freq = 3e9
c0 = 299792458.0
lambda = c0 / freq
k = 2pi / lambda

# Create mesh with ~5000 unknowns
mesh = make_rect_plate(lambda, lambda, 50, 50)
rwg = build_rwg(mesh)
println("N = ", rwg.nedges, " unknowns")
```

### 9.2 Dense vs. ACA Assembly

```julia
# Dense assembly
t_dense = @elapsed Z_dense = assemble_Z_efie(mesh, rwg, k)
println("Dense assembly: $(round(t_dense, digits=2)) s")
println("Dense storage: $(round(sizeof(Z_dense) / 2^20, digits=1)) MiB")

# ACA assembly
t_aca = @elapsed A_aca = build_aca_operator(mesh, rwg, k)
println("ACA assembly: $(round(t_aca, digits=2)) s")
println("Dense blocks: ", length(A_aca.dense_blocks))
println("Low-rank blocks: ", length(A_aca.lowrank_blocks))
```

### 9.3 Verifying Accuracy

```julia
# Compare matvec results
x = randn(ComplexF64, rwg.nedges)

y_dense = Z_dense * x
y_aca = A_aca * x

rel_error = norm(y_dense - y_aca) / norm(y_dense)
println("Relative matvec error: ", rel_error)
# Expected: ~1e-6 (matching aca_tol)
```

### 9.4 Full Solve Comparison

```julia
pw = make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(1,0,0))
v = assemble_excitation(mesh, rwg, pw)

# Dense direct solve
I_dense = Z_dense \ v

# ACA + GMRES solve
P_nf = build_nearfield_preconditioner(A_aca, mesh, rwg, lambda)
I_aca, stats = solve_gmres(A_aca, v; preconditioner=P_nf)

rel_solution_error = norm(I_dense - I_aca) / norm(I_dense)
println("Relative solution error: ", rel_solution_error)
println("GMRES iterations: ", stats.niter)
```

---

## 10. Code Mapping

This section maps the concepts discussed in this chapter to the source files in
`DiffMoM.jl`.

### 10.1 Source File Overview

| File | Purpose | Key Exports |
|------|---------|-------------|
| `src/fast/ClusterTree.jl` | Binary space-partitioning tree | `ClusterNode`, `ClusterTree`, `build_cluster_tree`, `is_admissible` |
| `src/fast/ACA.jl` | ACA low-rank approximation, H-matrix operator | `ACAOperator`, `ACAAdjointOperator`, `build_aca_operator`, `aca_lowrank` |
| `src/assembly/EFIE.jl` | Entry evaluation cache used by ACA | `EFIEApplyCache`, `_efie_entry` |
| `src/solver/NearFieldPreconditioner.jl` | Sparse preconditioner (works with ACAOperator) | `build_nearfield_preconditioner`, `NearFieldOperator` |
| `src/solver/IterativeSolve.jl` | GMRES wrapper (accepts ACAOperator) | `solve_gmres` |
| `src/Workflow.jl` | High-level API with auto method selection | `solve_scattering` |

### 10.2 Data Flow

The complete ACA pipeline follows this flow:

1. **`rwg_centers(mesh, rwg)`** computes the geometric center of each RWG basis function.
2. **`build_cluster_tree(centers)`** partitions the centers into a binary tree.
3. **`_enum_block_tasks!(tasks, tree, 1, 1)`** recursively classifies all block pairs as `:dense` or `:lowrank`.
4. **`Threads.@threads` loop** processes all blocks in parallel:
   - Dense blocks: `_fill_dense_block_batched!` with triangle-pair caching.
   - Low-rank blocks: `aca_lowrank` with `_efie_entry` for on-demand entry evaluation.
5. **`ACAOperator`** wraps the blocks for use as an `AbstractMatrix`.
6. **`mul!(y, A, x)`** performs the H-matrix matvec during GMRES iterations.

### 10.3 Key Internal Functions

| Function | Location | Role |
|----------|----------|------|
| `_efie_entry(cache, m, n)` | `src/assembly/EFIE.jl` | Evaluate single EFIE matrix entry |
| `_build_efie_cache(mesh, rwg, k)` | `src/assembly/EFIE.jl` | Precompute quadrature points, areas, RWG values |
| `_enum_block_tasks!(tasks, tree, i, j)` | `src/fast/ACA.jl` | Recursive block classification |
| `_fill_dense_block_batched!(data, cache, rows, cols)` | `src/fast/ACA.jl` | Batched dense block assembly |
| `aca_lowrank(cache, rows, cols)` | `src/fast/ACA.jl` | Partially-pivoted ACA for one block |

---

## 11. Exercises

### Conceptual

1. **Why not compress everything?** Explain why near-field blocks cannot be approximated as low-rank. What physical property of the Green's function prevents compression for nearby interactions?

2. **Admissibility parameter sensitivity**: How does increasing $\eta$ from 1.5 to 3.0 affect the number of dense vs. low-rank blocks? What happens to the ACA ranks? What is the trade-off?

3. **Leaf size effects**: Sketch what the block partition looks like for `leaf_size=16` vs. `leaf_size=256` on the same mesh. Which has more blocks? Which has larger dense blocks?

4. **ACA pivot selection**: In the ACA algorithm, why is the first pivot row chosen as row 1 (rather than, say, the row with the largest diagonal entry)? Does the choice of initial pivot affect convergence?

### Computational

5. **Compression ratio**: For a $2\lambda \times 2\lambda$ plate with $N = 5{,}000$ unknowns, build the ACA operator and compute the compression ratio (total H-matrix storage / dense storage). Vary `aca_tol` from $10^{-3}$ to $10^{-9}$ and plot the compression ratio vs. tolerance.

6. **Matvec accuracy**: Generate a random vector $\mathbf{x}$ and compare $\mathbf{Z}\mathbf{x}$ (dense) with $\mathbf{A}_{\text{ACA}}\mathbf{x}$. Plot the relative error as a function of `aca_tol`. Verify that the error is approximately proportional to the tolerance.

7. **Scaling study**: Create meshes of increasing size ($N = 1{,}000$ to $50{,}000$) and measure the ACA assembly time and matvec time. Plot both on a log-log scale and verify the $O(N \log^2 N)$ scaling.

8. **Preconditioner interaction**: Solve the same scattering problem with and without the near-field preconditioner when using ACA. Record the iteration counts and total solve times. At what problem size does the preconditioner become essential?

### Advanced

9. **Block rank distribution**: After building an ACA operator, extract the rank of each low-rank block (from `size(blk.U, 2)`). Plot a histogram of ranks. How does the distribution change with frequency (i.e., electrical size of the object)?

10. **Adjoint consistency**: Verify that the adjoint matvec $\mathbf{A}^H \mathbf{x}$ computed by `ACAAdjointOperator` satisfies $\langle \mathbf{A}\mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{A}^H\mathbf{y} \rangle$ for random vectors $\mathbf{x}$ and $\mathbf{y}$. This is essential for correct adjoint gradient computation.

---

## 12. Chapter Checklist

Before proceeding, ensure you understand:

- [ ] Why far-field EFIE interactions are numerically low-rank (smooth Green's function variation for well-separated groups).
- [ ] The cluster tree construction: recursive bisection along the longest bounding-box axis, median splitting, `leaf_size` as the termination criterion.
- [ ] The admissibility condition: $\min(\operatorname{diam}(i), \operatorname{diam}(j)) \le \eta \cdot \operatorname{dist}(i, j)$ with $\eta = 1.5$.
- [ ] The ACA algorithm: partial pivoting, rank-1 updates, convergence criterion $\|\mathbf{u}_k\|\|\mathbf{v}_k\| < \texttt{tol} \cdot \|\text{approx}\|_F$.
- [ ] How `build_aca_operator` uses two-phase assembly: serial block enumeration followed by parallel block computation.
- [ ] The batched Green's function optimization in dense blocks ($\sim 8\times$ fewer evaluations).
- [ ] How the ACA matvec works: permute, dense `gemv`, low-rank two-step `gemv`, un-permute.
- [ ] Why `ACAOperator` provides a `getindex` fallback (for near-field preconditioner extraction via `_efie_entry`).
- [ ] The complexity: $O(N \log^2 N)$ storage and matvec, compared to $O(N^2)$ dense.
- [ ] How `solve_scattering` auto-selects ACA for $10{,}000 < N \le 50{,}000$ (default thresholds).

---

## 13. Further Reading

1. **H-matrix theory**:
   - Hackbusch, W. (1999). *A sparse matrix arithmetic based on H-matrices. Part I: Introduction to H-matrices*. Computing, 62(2), 89--108. (Foundational paper introducing hierarchical matrices.)
   - Bebendorf, M. (2008). *Hierarchical Matrices: A Means to Efficiently Solve Elliptic Boundary Value Problems*. Springer. (Comprehensive monograph covering theory, algorithms, and applications.)

2. **ACA algorithm**:
   - Bebendorf, M. (2000). *Approximation of boundary element matrices*. Numerische Mathematik, 86(4), 565--589. (Original ACA paper proving convergence for asymptotically smooth kernels.)
   - Bebendorf, M., & Rjasanow, S. (2003). *Adaptive low-rank approximation of collocation matrices*. Computing, 70(1), 1--24. (Partially-pivoted ACA variant used in practice.)

3. **ACA in electromagnetics**:
   - Zhao, K., Vouvakis, M. N., & Lee, J.-F. (2005). *The adaptive cross approximation algorithm for accelerated method of moments computations of EMC problems*. IEEE Transactions on Electromagnetic Compatibility, 47(4), 763--773. (Application of ACA to MoM, demonstrates practical performance.)
   - Tamayo, J. M., Heldring, A., & Rius, J. M. (2011). *Multilevel adaptive cross approximation (MLACA)*. IEEE Transactions on Antennas and Propagation, 59(12), 4600--4608. (Multilevel extension for further compression.)

4. **Fast algorithms for integral equations**:
   - Chew, W. C., Jin, J. M., Michielssen, E., & Song, J. (2001). *Fast and Efficient Algorithms in Computational Electromagnetics*. Artech House. (Covers FMM, ACA, and other fast algorithms in context.)

---

*Next: [Physical Optics Approximation](03-physical-optics.md) introduces the high-frequency PO method for fast RCS estimation on electrically large objects.*
