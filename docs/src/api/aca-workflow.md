# ACA and high-level workflow API

This page identifies the public entry points for ACA-compressed PEC solves and
the `solve_scattering` workflow. Exact signatures and defaults come from the
[fast-method](exported-fast.md) and [core](exported-core.md) source docstrings.
For algorithm details, see [ACA H-matrix compression](../methods/02-aca-hmatrix-compression.md).

## Build an ACA operator

```julia
using DiffMoM

rwg = build_rwg(mesh)
A = build_aca_operator(
    mesh,
    rwg,
    k;
    leaf_size=64,
    eta=1.5,
    aca_tol=1e-6,
    max_rank=50,
    quad_order=3,
)
```

`build_aca_operator` stores inadmissible blocks densely and approximates
admissible blocks as low-rank factors. The returned `ACAOperator` implements
`size`, indexed access, forward `mul!`, and adjoint products, so it can be used
with `solve_gmres` and the near-field preconditioner API.

Indexed access returns the compressed operator entry and is consistent with a
basis-vector product. Use `efie_entry(A, i, j)` only when the uncompressed EFIE
entry is explicitly required.

Forward and adjoint products track componentwise reduction bounds through
dense blocks and both low-rank stages. Cancellation-sensitive products retry
the compressed operator in bounded high-precision output chunks.

The main accuracy controls are:

| Keyword | Effect |
|:--|:--|
| `leaf_size` | Sets the target leaf size in the cluster tree |
| `eta` | Controls geometric block admissibility |
| `aca_tol` | Sets the low-rank stopping tolerance |
| `max_rank` | Caps the rank retained in one admissible block |
| `quad_order` | Selects the triangle quadrature used for EFIE entries |

Resource keywords bound the cluster tree, block enumeration, persistent block
storage, EFIE cache, adjacency records, and per-worker Green-function cache.
Use the source docstring for their exact names and defaults. A low-rank block
that becomes non-finite during construction is rebuilt densely; resource caps
still apply to that replacement.

The operator owns reusable product workspace protected by a lock. Concurrent
calls on one operator are safe but serialize access to that workspace. Use
separate operators when independent products must run simultaneously.

## Solve with GMRES

```julia
P = build_nearfield_preconditioner(A; factorization=:lu)
current, stats = solve_gmres(
    A,
    rhs;
    preconditioner=P,
    tol=1e-6,
    maxiter=300,
)
```

Check the returned convergence status and the true residual against `A`. The
approximation itself needs a separate comparison on a tractable problem: hold
the mesh, excitation, quadrature, and observable fixed, then compare ACA with a
dense operator while tightening `aca_tol` and `max_rank`.

`build_aca_operator` constructs the PEC EFIE base operator. Supported sparse
impedance loading can be applied with `ImpedanceLoadedOperator`; use dense
impedance assembly when the complete loaded matrix is required.

## Cluster-tree queries

`build_cluster_tree` constructs the spatial hierarchy used by ACA. The public
query functions are:

- `cluster_diameter(tree, node_index)`;
- `cluster_distance(tree, first_index, second_index)`;
- `is_admissible(tree, first_index, second_index; eta=...)`;
- `is_leaf(tree, node_index)`; and
- `leaf_nodes(tree)`.

`ClusterNode` and `ClusterTree` expose the resulting hierarchy. Treat
`DenseBlock` and `LowRankBlock` as implementation storage rather than calling
interfaces.

## High-level `solve_scattering` workflow

```julia
result = solve_scattering(
    mesh,
    frequency_hz,
    excitation;
    method=:auto,
    check_resolution=true,
    check_gmres_convergence=true,
    check_true_residual=true,
)
```

In automatic mode, the RWG count is compared with
`dense_direct_limit`, `dense_gmres_limit`, and `mlfma_threshold`:

| Condition | Selected method |
|:--|:--|
| `N <= dense_direct_limit` | `:dense_direct` |
| `dense_direct_limit < N <= dense_gmres_limit` | `:dense_gmres` |
| `dense_gmres_limit < N <= mlfma_threshold` | `:aca_gmres` |
| `N > mlfma_threshold` | `:mlfma` |

These thresholds select an implementation; they do not establish that the
mesh, approximation, memory use, or observable is accurate. Override `method`
only after measuring the intended problem.

For iterative methods, `check_gmres_convergence` rejects failed Krylov status
and `check_true_residual` checks the returned vector against the selected
operator. `true_residual_factor` sets the allowed multiple of `gmres_tol`.
Disabling either check is appropriate only when the caller deliberately wants
to inspect a partial iterate.

`preconditioner=:auto` selects sparse LU for dense and ACA GMRES and incomplete
LU for MLFMA. `:none` disables preconditioning. The dense path uses
`nf_cutoff_lambda`; the ACA path reuses its dense inadmissible blocks; the MLFMA
path uses its stored near-field matrix.

## Result fields

`solve_scattering` returns `ScatteringResult`:

| Field | Meaning |
|:--|:--|
| `I_coeffs` | Solved RWG coefficients |
| `method` | Method actually selected |
| `N` | Number of RWG unknowns |
| `assembly_time_s` | Operator construction time |
| `solve_time_s` | Linear-solve time |
| `preconditioner_time_s` | Preconditioner construction time |
| `gmres_iters` | Krylov iteration count, or `-1` for a direct solve |
| `gmres_residual` | Unpreconditioned true relative residual against the selected operator, or `NaN` for a direct solve |
| `mesh_report` | Electrical-resolution report |
| `warnings` | Workflow warnings retained for the caller |

Use `result.method`, rather than the requested `:auto`, when recording a run.
For iterative results, `result.gmres_residual` is the same true residual used by
the default acceptance gate.

## Source map

| Area | Source |
|:--|:--|
| Cluster tree | `src/fast/ClusterTree.jl` |
| ACA construction and products | `src/fast/ACA.jl` |
| Automatic workflow | `src/Workflow.jl` |
| Result type | `src/Types.jl` |
| GMRES and true-residual checks | `src/solver/IterativeSolve.jl` |
| Near-field preconditioning | `src/solver/NearFieldPreconditioner.jl` |
