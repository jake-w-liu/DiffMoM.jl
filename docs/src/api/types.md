# API: Types

## Purpose

Reference for the core data structures used across `DiffMoM.jl`. Understanding these types is essential for using the package effectively: they carry the geometry, basis-function connectivity, far-field sampling, and preconditioner data that flow through every stage of the simulation pipeline.

---

## `TriMesh` -- Triangle Mesh

A `TriMesh` holds the geometry of a triangulated surface. It is the starting point for every MoM simulation: you create or import a `TriMesh`, then build RWG basis functions on it.

```julia
struct TriMesh
    xyz::Matrix{Float64}   # (3, Nv) vertex coordinates [m]
    tri::Matrix{Int}       # (3, Nt) 1-based vertex indices per triangle
end
```

**Fields:**

| Field | Type | Dimensions | Description |
|-------|------|------------|-------------|
| `xyz` | `Matrix{Float64}` | `(3, Nv)` | Vertex coordinates: `xyz[:, i]` is `[x, y, z]` of vertex `i` in meters. Column-major layout for cache-efficient access. |
| `tri` | `Matrix{Int}` | `(3, Nt)` | Triangle connectivity: `tri[:, j]` contains three 1-based vertex indices forming triangle `j`. Vertices should be ordered counter-clockwise when viewed from the outward-normal side. |

**Helper functions:**

- `nvertices(mesh::TriMesh) -> Int`: Number of vertices (`size(mesh.xyz, 2)`).
- `ntriangles(mesh::TriMesh) -> Int`: Number of triangles (`size(mesh.tri, 2)`).

**How to create a TriMesh:**

| Method | Use case |
|--------|----------|
| `make_rect_plate(Lx, Ly, Nx, Ny)` | Programmatic flat plate in the xy-plane |
| `make_parabolic_reflector(D, f, Nr, Nphi)` | Programmatic parabolic dish |
| `read_obj_mesh(path)` | Import from Wavefront OBJ file (CAD exports, external meshers) |
| Manual construction | `TriMesh(xyz, tri)` from your own arrays |

**Example:**

```julia
# Two triangles forming a unit square in the xy-plane
xyz = [0.0 1.0 1.0 0.0;   # x coordinates of 4 vertices
       0.0 0.0 1.0 1.0;   # y coordinates
       0.0 0.0 0.0 0.0]   # z coordinates (all zero = flat plate)
tri = [1 1;               # first vertex of triangle 1 and 2
       2 3;               # second vertex
       3 4]               # third vertex
mesh = TriMesh(xyz, tri)
println("Vertices: ", nvertices(mesh), ", triangles: ", ntriangles(mesh))
# Output: Vertices: 4, triangles: 2
```

---

## `RWGData` -- RWG Basis Data

Stores edge-to-triangle connectivity and geometric factors for Rao-Wilton-Glisson (RWG) basis functions. Each RWG basis function is associated with one interior edge of the mesh -- an edge shared by exactly two triangles. The basis function has support on these two triangles and is zero everywhere else.

RWG functions are the standard basis for surface MoM: they ensure continuity of the normal component of surface current across edges, which is required by the EFIE formulation.

```julia
struct RWGData
    mesh::TriMesh
    nedges::Int
    tplus::Vector{Int}          # T+ triangle index for each edge
    tminus::Vector{Int}         # T- triangle index for each edge
    evert::Matrix{Int}          # (2, nedges) edge vertex indices
    vplus_opp::Vector{Int}      # opposite vertex in T+
    vminus_opp::Vector{Int}     # opposite vertex in T-
    len::Vector{Float64}        # edge length [m]
    area_plus::Vector{Float64}  # area of T+ [m^2]
    area_minus::Vector{Float64} # area of T- [m^2]
end
```

**Fields:**

| Field | Type | Dimensions | Description |
|-------|------|------------|-------------|
| `mesh` | `TriMesh` | -- | Reference to the original mesh. Do not mutate the mesh after building RWG data. |
| `nedges` | `Int` | -- | Number of interior edges = number of RWG basis functions = the dimension N of the MoM system. |
| `tplus`, `tminus` | `Vector{Int}` | `nedges` | The two triangle indices sharing each edge. Convention: on `tplus`, the basis vector points away from the opposite vertex; on `tminus`, it points toward the opposite vertex. |
| `evert` | `Matrix{Int}` | `(2, nedges)` | Edge endpoint vertex indices: `evert[1, n]` and `evert[2, n]` are the two vertices of edge `n`. |
| `vplus_opp`, `vminus_opp` | `Vector{Int}` | `nedges` | Vertex index opposite the edge in triangle `tplus` / `tminus`. These "free vertices" define the direction of the RWG basis vector. |
| `len` | `Vector{Float64}` | `nedges` | Edge length in meters. Appears as a scaling factor in the RWG basis formula. |
| `area_plus`, `area_minus` | `Vector{Float64}` | `nedges` | Areas of the two support triangles in m^2. Used to normalize the basis function amplitude. |

**Constructor:**

```julia
rwg = build_rwg(mesh; precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12)
```

- `precheck=true`: Run mesh quality checks before building. Recommended for imported meshes.
- `allow_boundary=true`: Allow boundary edges (edges with only one triangle). Set to `false` for closed surfaces where every edge must be interior.
- `require_closed=false`: If `true`, throw an error if any boundary edges exist.
- `area_tol_rel=1e-12`: Relative tolerance for degenerate triangle detection.

**Example:**

```julia
mesh = make_rect_plate(0.1, 0.1, 3, 3)   # 0.1m x 0.1m plate, 3x3 cells
rwg = build_rwg(mesh; precheck=true)
println("RWG basis count (N): ", rwg.nedges)      # = system matrix dimension
println("Edge length of first basis: ", rwg.len[1], " m")
println("Support triangles: T+ = ", rwg.tplus[1], ", T- = ", rwg.tminus[1])
```

---

## `PatchPartition` -- Impedance Patch Mapping

Maps each triangle to a "design patch" for impedance parameterization in optimization. During optimization, each patch has one scalar impedance parameter `theta_p`, and all triangles in that patch share the same impedance value.

```julia
struct PatchPartition
    tri_patch::Vector{Int}      # length Nt: patch id for each triangle
    P::Int                      # number of patches
end
```

**Fields:**

| Field | Type | Dimensions | Description |
|-------|------|------------|-------------|
| `tri_patch` | `Vector{Int}` | `Nt` | `tri_patch[t]` is the 1-based patch index of triangle `t`. All triangles with the same patch index share the same impedance parameter. |
| `P` | `Int` | -- | Total number of patches (= number of design parameters in optimization). |

**Design choices:**

- **One patch per triangle** (`P = Nt`): Maximum design freedom. Each triangle has an independent impedance. This is the typical choice for fine-grained optimization.
- **Coarser grouping** (`P < Nt`): Reduced design space. Group triangles into regions (e.g., by quadrant, by ring, etc.) to reduce the number of optimization variables when fine-grained control is not needed.

**Example:**

```julia
Nt = ntriangles(mesh)

# One patch per triangle (typical for optimization)
tri_patch = collect(1:Nt)
partition = PatchPartition(tri_patch, Nt)

# Or: two patches (left half vs right half)
centers = [triangle_center(mesh, t) for t in 1:Nt]
tri_patch_lr = [c[1] < 0 ? 1 : 2 for c in centers]
partition_lr = PatchPartition(tri_patch_lr, 2)
```

---

## `SphGrid` -- Spherical Far-Field Grid

Defines sampling directions and quadrature weights for far-field pattern computation. The grid covers the unit sphere (or a portion of it) and provides weights for numerical integration of far-field quantities like radiated power and directivity.

```julia
struct SphGrid
    rhat::Matrix{Float64}       # (3, N_omega) unit direction vectors
    theta::Vector{Float64}      # polar angles [rad]
    phi::Vector{Float64}        # azimuthal angles [rad]
    w::Vector{Float64}          # quadrature weights [sr]
end
```

**Fields:**

| Field | Type | Dimensions | Description |
|-------|------|------------|-------------|
| `rhat` | `Matrix{Float64}` | `(3, N_omega)` | Unit direction vectors: `rhat[:, q] = [sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)]`. |
| `theta` | `Vector{Float64}` | `N_omega` | Polar angle in [0, pi], measured from the +z axis. theta = 0 is broadside (+z), theta = pi/2 is the horizon. |
| `phi` | `Vector{Float64}` | `N_omega` | Azimuthal angle in [0, 2pi), measured from the +x axis in the xy-plane. |
| `w` | `Vector{Float64}` | `N_omega` | Quadrature weights in steradians. For a full sphere, `sum(w) ~ 4pi`. To integrate a function over the sphere: `integral = sum(f.(directions) .* w)`. |

**Constructor:**

```julia
grid = make_sph_grid(Ntheta, Nphi)
```

Creates a uniform midpoint grid with `N_omega = Ntheta * Nphi` total directions.

**Choosing resolution:**
- `Ntheta=180, Nphi=72` gives 1-degree theta resolution and 5-degree phi resolution (12,960 directions). Good for smooth patterns.
- `Ntheta=90, Nphi=36` gives 2-degree / 10-degree resolution (3,240 directions). Adequate for most optimization and RCS computations.
- Higher resolution increases the cost of `radiation_vectors` and `build_Q` but improves integration accuracy.

**Example:**

```julia
grid = make_sph_grid(90, 36)   # 2-degree theta, 10-degree phi
println("Number of directions: ", length(grid.w))          # 3240
println("Total solid angle: ", sum(grid.w), " (expect ~4pi = ", 4pi, ")")
println("First direction: theta = ", rad2deg(grid.theta[1]), " deg")
```

---

## Preconditioner Types

The package provides a type hierarchy for GMRES preconditioners. All preconditioners inherit from `AbstractPreconditionerData`, which allows solvers and optimizers to accept any preconditioner variant through a single interface.

### `AbstractPreconditionerData` -- Base Type

```julia
abstract type AbstractPreconditionerData end
```

The common supertype for all preconditioner data. Functions that accept preconditioners (e.g., `solve_forward`, `solve_gmres`, `optimize_lbfgs`) are typed against this abstract type, so any concrete subtype works interchangeably.

**Subtypes:**

| Type | Strategy | When to use |
|------|----------|-------------|
| `NearFieldPreconditionerData` | Sparse LU of near-field entries | Default choice. Best convergence, moderate memory. |
| `ILUPreconditionerData` | Incomplete LU (ILU) of near-field entries | Large N with moderate nnz%. Less fill than full LU, slightly more GMRES iters. |
| `DiagonalPreconditionerData` | Jacobi (inverse diagonal) | Minimal memory and setup cost. Weaker convergence. |
| `BlockDiagPrecondData` | Block-Jacobi from MLFMA leaf boxes | Fast to build, good for MLFMA. Each leaf box LU-factorized independently. |
| `PermutedPrecondData` | Wrapped preconditioner with reordered BFs | MLFMA: reorder Z_near to block-banded form before ILU for better quality. |

---

### `NearFieldPreconditionerData` -- Near-Field Sparse Preconditioner

Stores a factorized sparse approximation of the MoM matrix, used as a preconditioner for GMRES. The near-field preconditioner retains only entries `Z[m,n]` where basis functions `m` and `n` are within a cutoff distance, then LU-factorizes the resulting sparse matrix.

This is the recommended approach for preconditioning dense EFIE systems: near-field interactions dominate the matrix structure, so a sparse near-field approximation captures the essential conditioning information.

```julia
struct NearFieldPreconditionerData <: AbstractPreconditionerData
    Z_nf_fac::SparseArrays.UMFPACK.UmfpackLU{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Z_nf_fac` | `UmfpackLU{ComplexF64, Int64}` | LU factorization of the sparse near-field matrix. Applying the preconditioner solves a sparse system via forward/back substitution. |
| `cutoff` | `Float64` | Distance cutoff in meters used to build the sparse matrix. Basis pairs farther than `cutoff` apart are set to zero. |
| `nnz_ratio` | `Float64` | Fraction of nonzeros retained: `nnz(Z_nf) / N^2`. A smaller ratio means a sparser (cheaper) preconditioner, but potentially less effective. |

**Constructor:**

```julia
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, cutoff)
```

See [assembly-solve.md](assembly-solve.md) for all `build_nearfield_preconditioner` overloads (dense matrix, abstract matrix, matrix-free operator, and geometry-direct).

**Choosing the cutoff distance:**

The cutoff is specified in meters. It is typically expressed in wavelengths for physical intuition:

| Cutoff | Sparsity | GMRES iters (impedance-loaded) | GMRES iters (PEC) | When to use |
|--------|----------|------|------|------|
| `0.5 * lambda0` | Very sparse | ~15 | 28--45 | Large problems, memory-limited |
| `1.0 * lambda0` | Moderate | ~10 | 15--25 | Best balance of cost vs convergence |
| `2.0 * lambda0` | Denser | ~8 | 10--15 | Aggressive convergence, smaller problems |

Key properties:
- Impedance-loaded EFIE: ~10 iterations with 1.0-lambda cutoff in tested cases, with weak growth over N = 96 to 736.
- PEC EFIE: 28--45 iterations with 0.5-lambda cutoff (vs ~194 unpreconditioned at N = 736).
- The preconditioner is built once and reused across all GMRES solves (forward, adjoint, line search in optimization).

---

### `DiagonalPreconditionerData` -- Jacobi/Diagonal Preconditioner

Stores the inverse diagonal entries of the MoM matrix as a lightweight preconditioner. Each application is O(N) element-wise multiplication — no sparse solve needed.

```julia
struct DiagonalPreconditionerData <: AbstractPreconditionerData
    dinv::Vector{ComplexF64}
    cutoff::Float64
    nnz_ratio::Float64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `dinv` | `Vector{ComplexF64}` | Inverse diagonal entries: `dinv[i] = 1 / Z[i,i]`. A floor is applied to avoid division by near-zero diagonals. |
| `cutoff` | `Float64` | The cutoff value passed at construction (informational; `0.0` for pure diagonal). |
| `nnz_ratio` | `Float64` | Always `1/N` (only diagonal entries retained). |

**Constructor:**

```julia
P_diag = build_nearfield_preconditioner(Z_efie, mesh, rwg, 0.0; factorization=:diag)
```

**When to use:** The diagonal preconditioner is useful as a baseline comparison or when memory for sparse LU is limited. It provides modest iteration reduction compared to unpreconditioned GMRES but is much weaker than the near-field sparse preconditioner.

---

### `ILUPreconditionerData` -- Incomplete LU Preconditioner

Stores an incomplete LU (ILU) factorization of the near-field sparse matrix, using IncompleteLU.jl's Crout ILU with drop tolerance `tau`. Compared to full sparse LU (`NearFieldPreconditionerData`), ILU has much less fill-in and is feasible for large N, at the cost of slightly more GMRES iterations.

```julia
struct ILUPreconditionerData <: AbstractPreconditionerData
    ilu_fac::IncompleteLU.ILUFactorization{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
    tau::Float64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `ilu_fac` | `ILUFactorization{ComplexF64, Int64}` | Incomplete LU factorization. Applying the preconditioner solves the approximate system via forward/back substitution. |
| `cutoff` | `Float64` | Distance cutoff in meters (or `Inf` when built from a pre-assembled sparse matrix). |
| `nnz_ratio` | `Float64` | Fraction of nonzeros retained in the sparse near-field matrix. |
| `tau` | `Float64` | Drop tolerance for ILU. Smaller `tau` = more fill = better preconditioner. |

**Constructor:**

```julia
P_ilu = build_nearfield_preconditioner(Z_efie, mesh, rwg, cutoff; factorization=:ilu, ilu_tau=1e-3)
# Or from pre-assembled sparse matrix (e.g., MLFMA Z_near):
P_ilu = build_nearfield_preconditioner(Z_near_sparse; factorization=:ilu, ilu_tau=1e-3)
```

**When to use:** ILU is the recommended factorization for MLFMA-scale problems (N > 10k) where full sparse LU is too slow or memory-intensive. The drop tolerance `tau` controls the memory/quality trade-off (typical: `1e-3` for distance-based, `1e-2` for MLFMA-reordered).

---

### `BlockDiagPrecondData` -- Block-Diagonal Preconditioner

Block-diagonal (block-Jacobi) preconditioner built from MLFMA leaf boxes. Each leaf box contributes a small dense diagonal block from `Z_near`, which is LU-factorized independently. Very fast to build and memory-efficient.

```julia
struct BlockDiagPrecondData <: AbstractPreconditionerData
    lu_blocks::Vector{LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}}
    box_bf_indices::Vector{Vector{Int}}
    N::Int
    nnz_ratio::Float64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `lu_blocks` | `Vector{LU{...}}` | Independent LU factorizations of diagonal blocks, one per leaf box. |
| `box_bf_indices` | `Vector{Vector{Int}}` | Original BF indices belonging to each leaf box. |
| `N` | `Int` | Total number of basis functions. |
| `nnz_ratio` | `Float64` | Fraction of nonzeros retained (`sum(block_size^2) / N^2`). |

**Constructor:**

```julia
P_bd = build_block_diag_preconditioner(A_mlfma)
```

**When to use:** When full ILU factorization of `Z_near` is too slow or memory-intensive. Block-diagonal builds in `O(n_boxes * n_bf^3)` where `n_bf` is the average BFs per leaf box (typically 100--500). Weaker than ILU but much faster to construct.

---

### `PermutedPrecondData` -- Permuted Preconditioner Wrapper

Wraps an inner preconditioner (ILU or LU) that operates in a permuted BF ordering. On application: permute input to reordered space, apply inner preconditioner, unpermute output. Used for MLFMA where reordering `Z_near` to block-banded form before ILU dramatically improves factorization speed and quality.

```julia
struct PermutedPrecondData{T<:AbstractPreconditionerData} <: AbstractPreconditionerData
    inner::T
    perm::Vector{Int}     # perm[new] = old (original → permuted)
    iperm::Vector{Int}    # iperm[old] = new (permuted → original)
    N::Int
    nnz_ratio::Float64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `inner` | `AbstractPreconditionerData` | The inner preconditioner operating in permuted ordering. |
| `perm` | `Vector{Int}` | Forward permutation: `perm[new_index] = old_index`. |
| `iperm` | `Vector{Int}` | Inverse permutation: `iperm[old_index] = new_index`. |
| `N` | `Int` | Total number of basis functions. |
| `nnz_ratio` | `Float64` | Fraction of nonzeros (from the inner preconditioner). |

**Constructor:**

```julia
P_mlfma = build_mlfma_preconditioner(A_mlfma; factorization=:ilu, ilu_tau=1e-2)
```

**When to use:** This is the recommended preconditioner for MLFMA. `build_mlfma_preconditioner` handles the reordering automatically; you do not need to construct `PermutedPrecondData` directly.

---

### Usage (all preconditioner types)

All preconditioner subtypes work interchangeably wherever a preconditioner is accepted:

```julia
# Build any type
P_nf   = build_nearfield_preconditioner(Z_efie, mesh, rwg, 1.0 * lambda0)
P_ilu  = build_nearfield_preconditioner(Z_efie, mesh, rwg, 1.0 * lambda0; factorization=:ilu)
P_diag = build_nearfield_preconditioner(Z_efie, mesh, rwg, 0.0; factorization=:diag)
P_bd   = build_block_diag_preconditioner(A_mlfma)       # MLFMA block-diagonal
P_mlfma = build_mlfma_preconditioner(A_mlfma)            # MLFMA reordered ILU

# Use with solve_gmres directly
I, stats = solve_gmres(Z_efie, v; preconditioner=P_nf)

# Use with solve_forward dispatch
I = solve_forward(Z, v; solver=:gmres, preconditioner=P_diag)

# Use in optimization
theta_opt, trace = optimize_lbfgs(Z_efie, Mp, v, Q, theta0;
    solver=:gmres, nf_preconditioner=P_nf)
```

**Operator wrappers:** `NearFieldOperator(P)` and `NearFieldAdjointOperator(P)` wrap any `AbstractPreconditionerData` subtype for Krylov.jl's `mul!` interface. These are parameterized on the preconditioner type (`NearFieldOperator{PType}`) and dispatch to the appropriate application method (sparse LU solve for NF, element-wise multiply for diagonal). They are used internally by `solve_gmres` and `solve_gmres_adjoint`; you typically do not need to construct them directly.

---

## `ScatteringResult` -- Workflow Output

Returned by `solve_scattering` (see [aca-workflow.md](aca-workflow.md)), this struct bundles the MoM solution with performance metadata and diagnostics. It is the primary output of the high-level workflow API.

```julia
struct ScatteringResult
    I_coeffs::Vector{ComplexF64}
    method::Symbol
    N::Int
    assembly_time_s::Float64
    solve_time_s::Float64
    preconditioner_time_s::Float64
    gmres_iters::Int
    gmres_residual::Float64
    mesh_report::NamedTuple
    warnings::Vector{String}
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `I_coeffs` | `Vector{ComplexF64}` | Surface current coefficients (length N). Pass to `compute_farfield`, `bistatic_rcs`, etc. |
| `method` | `Symbol` | Solver method used: `:dense_direct`, `:dense_gmres`, `:aca_gmres`, or `:mlfma`. |
| `N` | `Int` | Number of RWG unknowns (= system matrix dimension). |
| `assembly_time_s` | `Float64` | Wall-clock time for impedance matrix assembly (seconds). |
| `solve_time_s` | `Float64` | Wall-clock time for the linear solve (seconds). |
| `preconditioner_time_s` | `Float64` | Wall-clock time for preconditioner construction (0.0 for direct solves). |
| `gmres_iters` | `Int` | GMRES iteration count (-1 for direct solves). |
| `gmres_residual` | `Float64` | Final GMRES relative residual (NaN for direct solves). |
| `mesh_report` | `NamedTuple` | Output of `mesh_resolution_report` for the mesh at the given frequency. |
| `warnings` | `Vector{String}` | Any warnings generated during the solve (e.g., under-resolved mesh). |

**Example:**

```julia
result = solve_scattering(mesh, freq_hz, excitation)
I = result.I_coeffs
println("Method: ", result.method, ", N=", result.N)
println("Assembly: ", result.assembly_time_s, "s, Solve: ", result.solve_time_s, "s")
if result.method in (:dense_gmres, :aca_gmres)
    println("GMRES iters: ", result.gmres_iters, ", residual: ", result.gmres_residual)
end
```

---

## `ImpedanceLoadedOperator` -- Composite Impedance Operator

A matrix-free operator that wraps any `AbstractMatrix{ComplexF64}` base operator (MLFMA, ACA, dense, matrix-free EFIE) with sparse impedance loading. This enables GMRES-based optimization with fast operators without forming the full dense impedance-loaded system.

```julia
struct ImpedanceLoadedOperator{T<:AbstractMatrix{ComplexF64},
                                S<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    Z_base::T
    Mp::Vector{S}
    theta::Vector{Float64}
    reactive::Bool
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Z_base` | `AbstractMatrix{ComplexF64}` | Base EFIE operator (MLFMAOperator, ACAOperator, MatrixFreeEFIEOperator, or dense Matrix). |
| `Mp` | `Vector{<:AbstractMatrix}` | Sparse patch mass matrices from `precompute_patch_mass`. |
| `theta` | `Vector{Float64}` | Current impedance parameter vector (length P). |
| `reactive` | `Bool` | `false` = resistive (`Z_s = theta`), `true` = reactive (`Z_s = i*theta`). |

**Constructor:**

```julia
Z_op = ImpedanceLoadedOperator(Z_base, Mp, theta)           # resistive
Z_op = ImpedanceLoadedOperator(Z_base, Mp, theta, true)     # reactive
```

**Supported operations:** `size`, `eltype`, `mul!`, `*`, `adjoint`. The adjoint returns an `ImpedanceLoadedAdjointOperator` for adjoint sensitivity solves. See [composite-operators.md](composite-operators.md) for full API details.

**When to use:**

| Scenario | Base operator | Use `ImpedanceLoadedOperator`? |
|----------|---------------|-------------------------------|
| Small problem, dense Z | `Matrix{ComplexF64}` | Optional (can use `assemble_full_Z` instead) |
| ACA-compressed | `ACAOperator` | Yes |
| MLFMA fast multipole | `MLFMAOperator` | Yes |
| Multi-angle RCS optimization | Any | Yes (used internally by `optimize_multiangle_rcs`) |

---

## `AngleConfig` -- Multi-Angle RCS Configuration

Configuration for one incidence angle in a multi-angle RCS optimization. Created by `build_multiangle_configs` and consumed by `optimize_multiangle_rcs`.

```julia
struct AngleConfig
    k_vec::Vec3
    pol::Vec3
    v::Vector{ComplexF64}
    Q::Matrix{ComplexF64}
    weight::Float64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `k_vec` | `Vec3` | Incidence wave vector `k * k_hat` (rad/m). |
| `pol` | `Vec3` | Polarization unit vector (perpendicular to `k_vec`). |
| `v` | `Vector{ComplexF64}` | Pre-assembled excitation vector for this angle. |
| `Q` | `Matrix{ComplexF64}` | Backscatter Q-matrix targeting direction `-k_hat` with the specified cone angle. |
| `weight` | `Float64` | Weight `w_a` in the composite objective `J = sum_a w_a (I_a' Q_a I_a)`. |

**Constructor:**

```julia
configs = build_multiangle_configs(mesh, rwg, k, angles; grid=grid, backscatter_cone=10.0)
```

See [adjoint-optimize.md](adjoint-optimize.md) and the [Multi-Angle RCS chapter](../differentiable-design/05-multiangle-rcs.md) for usage details.

---

## Matrix-Free EFIE Operators

These types wrap the EFIE kernel evaluation as `AbstractMatrix` subtypes, enabling matrix-vector products without allocating a dense N x N matrix. They are used internally by the ACA H-matrix builder and can be used directly with Krylov.jl for pure matrix-free GMRES solves.

### `MatrixFreeEFIEOperator` -- Forward Operator

```julia
struct MatrixFreeEFIEOperator{T, TC<:EFIEApplyCache} <: AbstractMatrix{T}
    cache::TC
end
```

Behaves as an N x N `AbstractMatrix{ComplexF64}`. Supports:
- `size(A)` returns `(N, N)`
- `A[i, j]` computes EFIE entry Z[i,j] on the fly via `efie_entry(A, i, j)`
- `A * x` and `mul!(y, A, x)` compute the EFIE matrix-vector product row by row (O(N^2) per matvec)
- `adjoint(A)` returns a `MatrixFreeEFIEAdjointOperator`

**Constructor:**

```julia
A = matrixfree_efie_operator(mesh, rwg, k; quad_order=3, eta0=376.730313668,
                              mesh_precheck=true, allow_boundary=true, require_closed=false)
```

Parameters are the same as `assemble_Z_efie`. The difference is that no dense matrix is allocated; instead, the returned operator computes entries on demand from the internal `EFIEApplyCache`.

### `MatrixFreeEFIEAdjointOperator` -- Adjoint Operator

```julia
struct MatrixFreeEFIEAdjointOperator{T, TO<:MatrixFreeEFIEOperator{T}} <: AbstractMatrix{T}
    op::TO
end
```

The Hermitian adjoint of `MatrixFreeEFIEOperator`: `A_adj[i, j] = conj(A[j, i])`. Obtained via `adjoint(A)` or `matrixfree_efie_adjoint_operator(A)`. Used by `solve_gmres_adjoint` for adjoint sensitivity solves without a dense matrix.

### `efie_entry(A, m, n)`

Compute a single EFIE matrix entry Z[m,n] from a `MatrixFreeEFIEOperator`. This is the entry point used by the ACA algorithm and the near-field preconditioner builder to access individual matrix elements without dense storage.

**When to use matrix-free operators:**

| Scenario | Recommended approach |
|----------|---------------------|
| N < ~5000 | Dense `assemble_Z_efie` (simpler, enables direct LU) |
| N > ~5000, memory-limited | `matrixfree_efie_operator` + GMRES with NF preconditioner |
| ACA H-matrix solve | Used internally by `build_aca_operator` |

**Example:**

```julia
A = matrixfree_efie_operator(mesh, rwg, k)
P_nf = build_nearfield_preconditioner(A, 1.0 * lambda0)
I, stats = solve_gmres(A, v; preconditioner=P_nf)
println("GMRES iters: ", stats.niter)
```

---

## Vector Type Aliases

| Alias | Definition | Purpose |
|-------|------------|---------|
| `Vec3` | `SVector{3,Float64}` | 3-component real vector for positions, directions, and geometric quantities. |
| `CVec3` | `SVector{3,ComplexF64}` | 3-component complex vector for phasor fields (E-field, H-field, far-field). |

These are `StaticArrays` types: stack-allocated, fixed-size, and fast. They are used throughout kernel evaluations and field representations.

**Example:**

```julia
using StaticArrays
v = Vec3(1.0, 2.0, 3.0)             # position vector
cv = CVec3(1.0 + 2.0im, 0.0, 0.0)   # complex electric field phasor
```

---

## Internal Details

### Memory Layout

- **`TriMesh`** stores vertices and triangles as dense column-major matrices. Accessing all coordinates of one vertex (`xyz[:, i]`) is a contiguous memory read.
- **`RWGData`** fields are `Vector` or `Matrix` with `nedges` elements/columns. All lengths are in meters, areas in m^2.
- **`SphGrid`** weights `w` are scaled so that `sum(w) ~ 4pi` for a full sphere or `2pi` for a hemisphere.

### Immutability

- All types are defined as `struct` (not `mutable struct`), so the struct itself is immutable. However, the array contents they hold can be mutated in-place.
- Functions that modify meshes (repair, coarsening) return **new** `TriMesh` instances rather than modifying the input.
- `RWGData` holds a reference to the original `mesh`. Do not mutate the mesh's `xyz` or `tri` arrays after RWG construction, or the precomputed geometric data will become inconsistent.

---

## Code Mapping

| Type | Source File | Primary Users |
|------|-------------|---------------|
| `TriMesh`, `RWGData`, `PatchPartition`, `SphGrid`, `ScatteringResult` | `src/Types.jl` | All assembly, solve, and post-processing modules |
| `MatrixFreeEFIEOperator`, `MatrixFreeEFIEAdjointOperator` | `src/assembly/EFIE.jl` | `src/fast/ACA.jl`, `src/solver/NearFieldPreconditioner.jl`, `src/Workflow.jl` |
| `AbstractPreconditionerData`, `NearFieldPreconditionerData`, `ILUPreconditionerData`, `DiagonalPreconditionerData`, `BlockDiagPrecondData`, `PermutedPrecondData` | `src/solver/NearFieldPreconditioner.jl` | `src/solver/IterativeSolve.jl`, `src/solver/Solve.jl`, `src/optimization/Optimize.jl` |
| `NearFieldOperator`, `NearFieldAdjointOperator` | `src/solver/NearFieldPreconditioner.jl` | `src/solver/IterativeSolve.jl` (internal) |
| `ClusterNode`, `ClusterTree` | `src/fast/ClusterTree.jl` | `src/fast/ACA.jl` |
| `ACAOperator`, `ACAAdjointOperator` | `src/fast/ACA.jl` | `src/Workflow.jl`, `src/solver/IterativeSolve.jl` |
| `ImpedanceLoadedOperator`, `ImpedanceLoadedAdjointOperator` | `src/assembly/CompositeOperator.jl` | `src/optimization/MultiAngleRCS.jl` |
| `AngleConfig` | `src/optimization/MultiAngleRCS.jl` | `optimize_multiangle_rcs` |
| Vector aliases (`Vec3`, `CVec3`) | `src/Types.jl` | `src/geometry/Mesh.jl`, `src/basis/Greens.jl`, `src/postprocessing/FarField.jl`, `src/assembly/Excitation.jl` |

---

## Exercises

### Basic

1. Create a `TriMesh` for a single right-triangle with vertices (0,0,0), (1,0,0), (0,1,0). Verify `ntriangles` returns 1.
2. Build `RWGData` for a 2x2 plate. Print `tplus`, `tminus`, and `len` for each edge. Confirm each edge is interior (both `tplus` and `tminus` are valid triangle indices).

### Practical

1. Generate a `SphGrid` with 90 theta points and 36 phi points. Compute `sum(grid.w)` and compare with 4pi (full sphere).
2. Create a `PatchPartition` that groups triangles into 4 patches based on the quadrant of their centroid. Use `triangle_center` from `Mesh.jl`.

### Advanced

1. Write a function that extracts all edges of a `TriMesh` (including boundary edges using `mesh_unique_edges`) and compares with `RWGData`'s interior edges. How many boundary edges does your mesh have?
2. Build a `NearFieldPreconditionerData` for a plate mesh at several cutoff distances (0.25-lambda, 0.5-lambda, 1.0-lambda). Compare `nnz_ratio` and GMRES iteration counts.
