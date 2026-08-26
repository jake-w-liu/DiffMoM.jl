# GMRES Iterative Solving with Near-Field Sparse Preconditioning

## Purpose

Dense Method of Moments (MoM) systems produce impedance matrices with $O(N^2)$
storage and $O(N^3)$ direct-factorization cost. This chapter introduces the
GMRES iterative solver with near-field sparse preconditioning as implemented
in `DiffMoM.jl`. A dense GMRES matvec remains $O(N^2)$; whether iteration
reduces total cost depends on measured convergence and preconditioner setup for
the target problem. The treatment covers the algorithm, preconditioner, adjoint
extension, and package API.

---

## Learning Goals

After this chapter, you should be able to:

1. Compare direct and iterative resource costs and measure their crossover for a target problem.
2. Describe the GMRES algorithm for non-symmetric complex linear systems and the role of the Krylov subspace.
3. Construct a near-field sparse preconditioner and measure its effect on iteration growth.
4. Distinguish left and right preconditioning and their implications for residual monitoring.
5. Use the `solve_gmres`, `build_nearfield_preconditioner`, and `solve_gmres_adjoint` functions from the package API.
6. Set up preconditioned adjoint solves for gradient-based impedance optimization.

---

## 1. Why Iterative Solving for MoM

### 1.1 The Scaling Wall of Direct Solvers

The MoM discretization of the EFIE produces a dense impedance matrix $\mathbf{Z} \in \mathbb{C}^{N \times N}$, where $N$ is the number of RWG basis functions (interior edges). A direct solve via LU factorization requires:

- **Storage**: $O(N^2)$ complex entries $\approx 16N^2$ bytes.
- **Factorization**: $O(\tfrac{2}{3}N^3)$ floating-point operations.
- **Back-substitution**: $O(N^2)$ per right-hand side.

A dense `ComplexF64` matrix alone requires $16N^2$ payload bytes, before
factorization storage and workspaces. Use the package's resource estimates and
measure factorization time on the target machine; a universal wall-time
crossover does not follow from asymptotic complexity.

### 1.2 Iterative Solvers: Trading Exact Factorization for Matvecs

Iterative Krylov methods replace the single $O(N^3)$ factorization with a sequence of matrix-vector products (matvecs), each costing $O(N^2)$ for a dense matrix. If the solver converges in $m$ iterations, the total cost is $O(mN^2)$. The critical question is: **how does $m$ depend on $N$?**

The iteration count $m$ depends on the spectrum, restart, tolerance, geometry,
frequency, mesh, and preconditioner. Record it across the intended problem
sequence; the implementation does not assume an `N`-independent value.

### 1.3 Selecting the solve method

`solve_scattering(method=:auto)` selects among dense direct, dense GMRES, ACA,
and MLFMA branches from the unknown count and its configurable thresholds. The
`solve_scattering` docstring owns the exact branch conditions and defaults;
they are dispatch rules, not performance guarantees. Override them after
measuring memory, setup time, solve time, observable accuracy, and the checked
true residual for the target workload.

### 1.4 Physical Insight: Spatial Decay of Interactions

The MoM impedance matrix has a distinctive structure rooted in the physics of the Green's function. The free-space Green's function in the $e^{+i\omega t}$ convention is

```math
G(\mathbf{r}, \mathbf{r}') = \frac{e^{-ikR}}{4\pi R}, \qquad R = |\mathbf{r} - \mathbf{r}'|,
```

whose scalar magnitude decays as $1/R$ with distance. A near-field
preconditioner retains local interactions exactly and omits longer-range ones.
That structure can improve conditioning, but oscillatory phase, geometry, and
basis coupling prevent distance alone from guaranteeing entry ordering or
convergence.

---

## 2. The GMRES Algorithm for Complex Systems

### 2.1 Problem Setting

We seek $\mathbf{x} \in \mathbb{C}^N$ satisfying

```math
\mathbf{Z}\mathbf{x} = \mathbf{b},
```

where $\mathbf{Z}$ is a dense, non-Hermitian, complex matrix (the EFIE impedance matrix) and $\mathbf{b}$ is the excitation vector.

### 2.2 Krylov Subspace Construction

Starting from an initial residual $\mathbf{r}_0 = \mathbf{b} - \mathbf{Z}\mathbf{x}_0$ (typically $\mathbf{x}_0 = \mathbf{0}$), GMRES builds the Krylov subspace of dimension $m$:

```math
\mathcal{K}_m(\mathbf{Z}, \mathbf{r}_0) = \operatorname{span}\{\mathbf{r}_0,\, \mathbf{Z}\mathbf{r}_0,\, \mathbf{Z}^2\mathbf{r}_0,\, \ldots,\, \mathbf{Z}^{m-1}\mathbf{r}_0\}.
```

At iteration $m$, GMRES finds the approximate solution

```math
\mathbf{x}_m = \mathbf{x}_0 + \mathbf{y}_m, \qquad \mathbf{y}_m \in \mathcal{K}_m(\mathbf{Z}, \mathbf{r}_0),
```

that **minimizes the residual norm** over the Krylov subspace:

```math
\mathbf{x}_m = \arg\min_{\mathbf{x}_0 + \mathbf{y} \in \mathcal{K}_m} \|\mathbf{b} - \mathbf{Z}(\mathbf{x}_0 + \mathbf{y})\|_2.
```

### 2.3 Arnoldi Orthogonalization

In practice, GMRES constructs an orthonormal basis $\{\mathbf{v}_1, \ldots, \mathbf{v}_m\}$ for $\mathcal{K}_m$ via the Arnoldi process, which is a modified Gram-Schmidt procedure applied to the sequence $\mathbf{Z}\mathbf{v}_1, \mathbf{Z}\mathbf{v}_2, \ldots$. This produces the Arnoldi relation

```math
\mathbf{Z} \mathbf{V}_m = \mathbf{V}_{m+1} \bar{\mathbf{H}}_m,
```

where $\mathbf{V}_m \in \mathbb{C}^{N \times m}$ has orthonormal columns and $\bar{\mathbf{H}}_m \in \mathbb{C}^{(m+1)\times m}$ is an upper Hessenberg matrix. The least-squares minimization then reduces to an $(m+1)\times m$ least-squares problem, which is solved cheaply via Givens rotations.

### 2.4 Convergence and Eigenvalue Clustering

The convergence rate of GMRES depends on the **distribution of eigenvalues** of $\mathbf{Z}$. Formally, if the eigenvalues of $\mathbf{Z}$ can be enclosed in a set $\mathcal{S} \subset \mathbb{C}$ that excludes the origin, then after $m$ iterations

```math
\frac{\|\mathbf{r}_m\|}{\|\mathbf{r}_0\|} \le \min_{p \in \mathcal{P}_m,\, p(0)=1} \max_{\lambda \in \mathcal{S}} |p(\lambda)|,
```

where $\mathcal{P}_m$ is the set of polynomials of degree $m$. The key implication is:

- **Clustered eigenvalues**: If the eigenvalues cluster around a single point $c \ne 0$, a low-degree polynomial can make $|p(\lambda)|$ small on the cluster, leading to fast convergence.
- **Scattered eigenvalues**: If eigenvalues are spread widely (including near the origin), high-degree polynomials are needed, resulting in slow convergence.

This motivates preconditioning: we want to transform $\mathbf{Z}$ into a matrix whose eigenvalues cluster tightly around 1.

### 2.5 Left vs. Right Preconditioning

Given a nonsingular preconditioner $\mathbf{M} \approx \mathbf{Z}$, there are two natural ways to precondition:

**Left preconditioning** transforms the system to

```math
\mathbf{M}^{-1}\mathbf{Z}\mathbf{x} = \mathbf{M}^{-1}\mathbf{b}.
```

GMRES minimizes $\|\mathbf{M}^{-1}\mathbf{b} - \mathbf{M}^{-1}\mathbf{Z}\mathbf{x}_m\| = \|\mathbf{M}^{-1}\mathbf{r}_m\|$, the **preconditioned residual**. The true residual $\|\mathbf{r}_m\|$ is not directly minimized and may differ from the reported residual.

**Right preconditioning** introduces a substitution $\mathbf{x} = \mathbf{M}^{-1}\mathbf{y}$ and solves

```math
\mathbf{Z}\mathbf{M}^{-1}\mathbf{y} = \mathbf{b}, \qquad \mathbf{x} = \mathbf{M}^{-1}\mathbf{y}.
```

GMRES minimizes $\|\mathbf{b} - \mathbf{Z}\mathbf{M}^{-1}\mathbf{y}_m\| = \|\mathbf{r}_m\|$, the **true residual**. This makes convergence monitoring straightforward.

In Krylov.jl (the iterative solver library used by `DiffMoM.jl`), the left preconditioner is passed via the `M` keyword and the right preconditioner via the `N` keyword:

```julia
# Left preconditioning
x, stats = Krylov.gmres(Z, b; M=M_left, rtol=1e-8)

# Right preconditioning
x, stats = Krylov.gmres(Z, b; N=M_right, rtol=1e-8)
```

The default is left preconditioning (`precond_side=:left`). Left and right
preconditioning minimize different residuals, so their iteration counts need
not match. Compare convergence history and the true physical residual for the
target system.

---

## 3. Near-Field Sparse Preconditioner

### 3.1 Physical Motivation

The free-space Green's function

```math
G(\mathbf{r}, \mathbf{r}') = \frac{e^{-ikR}}{4\pi R}
```

decays as $1/R$ with separation distance $R = |\mathbf{r} - \mathbf{r}'|$. As a result, the impedance matrix entry $Z_{mn}$ between two RWG basis functions $m$ and $n$ has magnitude that decreases with the distance between their spatial supports. Entries corresponding to **near-field** interactions (small $R$) dominate the matrix structure, while **far-field** entries (large $R$) contribute relatively little to the conditioning.

This observation suggests a natural preconditioner: extract only the near-field portion of $\mathbf{Z}$ into a sparse matrix, and use its factorization as an approximate inverse.

### 3.2 Construction: Spatial Thresholding

Define the center of each RWG basis function as the average of the centroids of its two supporting triangles. Given a distance cutoff $d_{\mathrm{cut}}$, the near-field sparse matrix is

```math
[\mathbf{Z}_{\mathrm{nf}}]_{mn} =
\begin{cases}
Z_{mn}, & \text{if } |\mathbf{c}_m - \mathbf{c}_n| \le d_{\mathrm{cut}} \text{ or } m = n, \\
0, & \text{otherwise},
\end{cases}
```

where $\mathbf{c}_m$ and $\mathbf{c}_n$ are the RWG centers. The diagonal is always retained.

The number of nonzero entries in $\mathbf{Z}_{\mathrm{nf}}$ scales as
$O(N \cdot n_{\mathrm{local}})$, where $n_{\mathrm{local}}$ is the average
number of neighbors within the cutoff radius. It remains linear in `N` only
when $n_{\mathrm{local}}$ stays bounded, such as growth by adding similarly
resolved area. Refining a fixed physical region increases the number of basis
functions inside a fixed-wavelength cutoff and can increase fill faster.

### 3.3 Factorization Strategies

Once the sparse matrix $\mathbf{Z}_{\mathrm{nf}}$ is assembled, three factorization options are available:

**Sparse LU factorization** (`factorization=:lu`): The default and most effective option. Uses UMFPACK (via Julia's `SparseArrays.lu`) to compute a sparse LU factorization of $\mathbf{Z}_{\mathrm{nf}}$. Applying the preconditioner costs $O(\mathrm{nnz})$ per solve, where $\mathrm{nnz}$ is the number of nonzero entries plus fill-in from the factorization.

**Diagonal (Jacobi) preconditioner** (`factorization=:diag`): Retains only the diagonal entries of $\mathbf{Z}$:

```math
\mathbf{M}_{\mathrm{diag}} = \operatorname{diag}(\mathbf{Z}).
```

Application cost is $O(N)$. This is much cheaper than sparse LU but less effective at clustering eigenvalues. Use it when $N$ is very large and the sparse LU factorization itself becomes expensive.

**Incomplete LU (ILU)** (`factorization=:ilu`): Uses `IncompleteLU.jl` with drop tolerance `ilu_tau`. This trades preconditioner strength for lower setup cost and memory, and is commonly useful in very large MLFMA problems.

### 3.4 Cutoff Selection and Sparsity

The cutoff distance $d_{\mathrm{cut}}$ controls the trade-off between preconditioner quality and cost:

| Cutoff change | Stored near-field entries | Likely tradeoff to measure |
|---|---|---|
| Decrease | Fewer | Cheaper setup/apply, potentially more iterations |
| Increase | More | Costlier setup/apply, potentially fewer iterations |
| $\infty$ | Full matrix | Exact factorization when supported; dense-like cost |

Choose $d_{\mathrm{cut}}$ by measuring factorization cost, fill, iteration count,
and the true residual on the target problem. In `solve_scattering`,
`nf_cutoff_lambda` controls this distance only for `method=:dense_gmres`; ACA
and MLFMA reuse near-field structures stored by their operators. The
[`solve_scattering` docstring](../api/exported-core.md) owns the exact default.

### 3.5 What the Preconditioner Retains

The preconditioner $\mathbf{M} = \mathbf{Z}_{\mathrm{nf}}$ retains every
interaction inside $d_{\mathrm{cut}}$ plus the diagonal. The preconditioned
operator is
$\mathbf{M}^{-1}\mathbf{Z} = \mathbf{I} +
\mathbf{M}^{-1}(\mathbf{Z}-\mathbf{M})$.
This decomposition motivates the method without asserting a spectral bound.
Measure fill, factor stability, iterations, and true residual as the mesh or
frequency changes.

### 3.6 Spatial Hashing for Efficient Neighbor Search

A naive search for all basis-function pairs within the cutoff distance costs $O(N^2)$. The package uses **spatial hashing** to reduce this to $O(N)$ for problems with bounded cutoff:

1. Partition 3D space into cubic cells of side length $d_{\mathrm{cut}}$.
2. Hash each RWG center into its cell using $\texttt{key} = (\lfloor x/d_{\mathrm{cut}} \rfloor, \lfloor y/d_{\mathrm{cut}} \rfloor, \lfloor z/d_{\mathrm{cut}} \rfloor)$.
3. For each basis function $m$, check only the 27 neighboring cells (the $3\times3\times3$ stencil) for candidates within the cutoff.

If occupancy and candidate checks per cell remain bounded, the neighbor search
is O(N). Local refinement or a large cutoff can increase occupancy, so the
implementation also enforces candidate-work and storage limits.

---

## 4. Implementation in DiffMoM.jl

### 4.1 Source File Overview

The iterative solver and preconditioner are implemented in two dedicated source files:

- **`src/solver/NearFieldPreconditioner.jl`**: Preconditioner construction, operator wrappers, spatial hashing.
- **`src/solver/IterativeSolve.jl`**: GMRES wrapper functions using Krylov.jl.

These files depend on earlier definitions (types, mesh, RWG, EFIE) and must be included in the correct order in `src/DiffMoM.jl`. Specifically, `NearFieldPreconditioner.jl` must precede `IterativeSolve.jl` because the operator types `NearFieldOperator` and `NearFieldAdjointOperator` are used as preconditioners in the GMRES calls.

### 4.2 Preconditioner Data Types

The type hierarchy for preconditioner data is:

```julia
abstract type AbstractPreconditionerData end

struct NearFieldPreconditionerData <: AbstractPreconditionerData
    Z_nf_fac::SparseArrays.UMFPACK.UmfpackLU{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
end

struct DiagonalPreconditionerData <: AbstractPreconditionerData
    dinv::Vector{ComplexF64}   # inverse diagonal entries
    cutoff::Float64
    nnz_ratio::Float64
end

struct ILUPreconditionerData <: AbstractPreconditionerData
    ilu_fac::IncompleteLU.ILUFactorization{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
    tau::Float64
end
```

The `nnz_ratio` field records the fraction of nonzero entries relative to the full $N \times N$ matrix, which is useful for diagnostics.

### 4.3 Building the Preconditioner: Four Overloads

The function `build_nearfield_preconditioner` has four method signatures, each suited to a different workflow:

**Overload 1: From a dense matrix**

```julia
P_nf = build_nearfield_preconditioner(Z::Matrix, mesh, rwg, cutoff;
                                       factorization=:lu)
```

Extracts entries from the pre-assembled dense matrix `Z`. Useful when `Z` is already available (e.g., for moderate-size problems solved with dense GMRES).

**Overload 2: From an abstract operator**

```julia
P_nf = build_nearfield_preconditioner(A::AbstractMatrix, mesh, rwg, cutoff;
                                       factorization=:lu)
```

Extracts entries element-by-element via `A[m,n]`. Works with any operator that supports `getindex`, including the `ACAOperator` from the H-matrix module.

**Overload 3: From a matrix-free EFIE operator**

```julia
A = matrixfree_efie_operator(mesh, rwg, k)
P_nf = build_nearfield_preconditioner(A::MatrixFreeEFIEOperator, cutoff;
                                       factorization=:lu)
```

Convenience overload that extracts `mesh` and `rwg` from the operator's cache. No dense matrix is ever formed.

**Overload 4: From geometry and physics directly**

```julia
P_nf = build_nearfield_preconditioner(mesh, rwg, k, cutoff;
                                       quad_order=3, factorization=:lu)
```

Internally creates a temporary `MatrixFreeEFIEOperator` to compute the required entries, then discards it. This is the most self-contained overload, requiring no pre-assembled operator.

### 4.4 Krylov.jl Operator Wrappers

Krylov.jl expects preconditioners to support the `mul!(y, M, x)` interface (in-place matrix-vector multiply). The package provides two wrapper types:

```julia
struct NearFieldOperator{PType<:AbstractPreconditionerData}
    P::PType
end

struct NearFieldAdjointOperator{PType<:AbstractPreconditionerData}
    P::PType
end
```

`NearFieldOperator(P)` applies $\mathbf{Z}_{\mathrm{nf}}^{-1}\mathbf{v}$ (forward preconditioner). `NearFieldAdjointOperator(P)` applies $\mathbf{Z}_{\mathrm{nf}}^{-\dagger}\mathbf{v}$ (adjoint preconditioner). Both dispatch internally to either sparse LU solves (`ldiv!`) or diagonal scaling, depending on the concrete type of `P`.

### 4.5 The `solve_gmres` Function

The main iterative solve entry point is:

```julia
function solve_gmres(Z, rhs;
                     preconditioner=nothing,
                     precond_side=:left,
                     tol=1e-8,
                     maxiter=200,
                     memory=20,
                     verbose=false,
                     check_gmres_convergence=true,
                     check_true_residual=true,
                     true_residual_factor=100.0)
```

Key behavior:

- If `preconditioner === nothing`, calls `Krylov.gmres(Z, rhs)` without preconditioning.
- If `precond_side == :left`, wraps the preconditioner as a `NearFieldOperator` and passes it as the `M` keyword to `Krylov.gmres`.
- If `precond_side == :right`, passes it as the `N` keyword instead.
- The `tol` parameter maps to Krylov.jl's `rtol` (relative tolerance) with `atol=0.0`.
- The returned vector must also satisfy the unpreconditioned true-residual gate
  by default; Krylov's preconditioned status alone is not sufficient.
- Returns `(x, stats)` where `stats.niter` gives the iteration count and `stats.residuals` gives the residual history.

### 4.6 The `solve_gmres_adjoint` Function

For adjoint systems arising in sensitivity analysis:

```julia
function solve_gmres_adjoint(Z, rhs;
                              preconditioner=nothing,
                              precond_side=:left,
                              tol=1e-8,
                              maxiter=200,
                              verbose=false)
```

Solves $\mathbf{Z}^\dagger \mathbf{x} = \mathbf{rhs}$ by calling `Krylov.gmres(adjoint(Z), rhs)` with the **adjoint preconditioner** `NearFieldAdjointOperator(P)`. The adjoint operator applies $\mathbf{Z}_{\mathrm{nf}}^{-\dagger}$, which is the conjugate transpose of the forward preconditioner's inverse.

### 4.7 The `rwg_centers` Utility

```julia
function rwg_centers(mesh, rwg) -> Vector{Vec3}
```

Computes the center of each RWG basis function as the average of the centroids of its two supporting triangles:

```math
\mathbf{c}_n = \frac{1}{2}\bigl(\mathbf{c}_{T_n^+} + \mathbf{c}_{T_n^-}\bigr),
```

where $\mathbf{c}_{T_n^\pm}$ are the triangle centroids. These centers are used for distance computations in the near-field neighbor search.

---

## 5. Interpreting Iteration Growth

### 5.1 Operator Decomposition

Consider the preconditioned matrix $\tilde{\mathbf{Z}} = \mathbf{Z}_{\mathrm{nf}}^{-1}\mathbf{Z}$. We can decompose it as

```math
\tilde{\mathbf{Z}} = \mathbf{I} + \mathbf{Z}_{\mathrm{nf}}^{-1}(\mathbf{Z} - \mathbf{Z}_{\mathrm{nf}}) = \mathbf{I} + \mathbf{Z}_{\mathrm{nf}}^{-1}\mathbf{Z}_{\mathrm{ff}},
```

where $\mathbf{Z}_{\mathrm{ff}} = \mathbf{Z} - \mathbf{Z}_{\mathrm{nf}}$
contains the omitted interactions. This identity explains what the
preconditioner removes, but it does not prove an `N`-independent spectrum:
that would also require bounds on the inverse and omitted operator for the
specific geometry, frequency, mesh family, and cutoff.

### 5.2 Measure the Target Sequence

For a mesh sequence, record the unknown count, mesh-resolution rule, cutoff,
near-field fill and factor storage, setup time, iteration count, solve time,
and true residual. Weak iteration growth on one sequence is evidence for that
sequence only; a frequency sweep, local refinement, or changed geometry can
produce different conditioning.

### 5.3 Compare Left and Right Preconditioning

Left and right preconditioning solve related but different Krylov systems.
Either can converge faster on a particular EFIE matrix. Use the same restart,
tolerance, maximum iteration count, and preconditioner when comparing them,
then verify both returned currents against the original unpreconditioned
equation.

---

## 6. Adjoint Preconditioning for Gradient-Based Optimization

### 6.1 The Adjoint Linear System

In impedance optimization, the adjoint method computes gradients of an objective $\Phi(\mathbf{I})$ with respect to impedance parameters $\boldsymbol{\theta}$ by solving the adjoint equation

```math
\mathbf{Z}^\dagger \boldsymbol{\lambda} = \frac{\partial \Phi}{\partial \mathbf{I}^*} = \mathbf{Q}\mathbf{I},
```

where $\mathbf{Q}$ is a quadratic objective matrix and $\mathbf{I}$ is the forward solution (see `src/optimization/Adjoint.jl`). Once $\boldsymbol{\lambda}$ is known, the gradient is

```math
\frac{\partial \Phi}{\partial \theta_p} = 2\,\operatorname{Re}\!\bigl\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\bigr\},
```

where $\mathbf{M}_p$ is the patch mass matrix for patch $p$.

### 6.2 Adjoint Preconditioner: $\mathbf{Z}_{\mathrm{nf}}^{-\dagger}$

If the forward system is preconditioned by $\mathbf{Z}_{\mathrm{nf}}^{-1}$ (left preconditioning), the adjoint system must be preconditioned by the **conjugate transpose** of the forward preconditioner:

```math
(\mathbf{Z}_{\mathrm{nf}}^{-1})^\dagger = \mathbf{Z}_{\mathrm{nf}}^{-\dagger}.
```

This is automatically handled by `solve_gmres_adjoint`, which wraps the preconditioner data in a `NearFieldAdjointOperator`. The adjoint operator applies the conjugate-transposed LU solve:

```julia
# Inside NearFieldAdjointOperator mul!:
ldiv!(adjoint(P.Z_nf_fac), y)   # applies Z_nf^{-H} to y in-place
```

For the diagonal preconditioner, the adjoint is simply the conjugate of the inverse diagonal: $\bar{d}_i^{-1}$.

### 6.3 Using the Same Preconditioner for Forward and Adjoint

A single call to `build_nearfield_preconditioner` produces a `P_nf` object that is reused for both forward and adjoint solves. This ensures **adjoint consistency**: the same near-field factorization is used in both directions, yielding correct gradients.

```julia
# Build preconditioner once
P_nf = build_nearfield_preconditioner(Z, mesh, rwg, cutoff)

# Forward solve
I, stats_fwd = solve_gmres(Z, v; preconditioner=P_nf)

# Adjoint solve (uses the same P_nf, with adjoint operator automatically)
lambda, stats_adj = solve_gmres_adjoint(Z, Q * I; preconditioner=P_nf)
```

### 6.4 Integration with the Optimization Loop

The optimizer in `src/optimization/Optimize.jl` accepts a near-field preconditioner via the `nf_preconditioner` keyword argument. When provided, both `solve_forward` and `solve_adjoint` dispatch to the GMRES path with the given preconditioner:

```julia
result = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, zeros(P);
    solver = :gmres,
    nf_preconditioner = P_nf,
    gmres_tol = 1e-8,
    gmres_maxiter = 200,
)
```

This avoids rebuilding the preconditioner at each optimization iteration, which is valid as long as the EFIE matrix $\mathbf{Z}_{\mathrm{EFIE}}$ does not change (only the impedance terms $\theta_p \mathbf{M}_p$ change).

---

## 7. Practical Guidelines

### 7.1 Cutoff Selection

The near-field cutoff $d_{\mathrm{cut}}$ should be expressed in wavelengths:

```math
d_{\mathrm{cut}} = \alpha \cdot \lambda_0, \qquad \lambda_0 = \frac{2\pi}{k}.
```

Start with the workflow's source-owned default when using
`method=:dense_gmres`, then sweep $\alpha$ for the target problem. A larger
cutoff retains more entries and can strengthen the preconditioner, but it also
increases assembly, factorization, and storage cost. Report the effective
cutoff with any timing or iteration result.

### 7.2 Factorization Choice

| Factorization | Build cost | Apply cost | Selection condition |
|---|---|---|---|
| `:lu` (sparse LU) | Fill- and ordering-dependent | Sparse triangular solves | Exact near-field factorization when its factors fit |
| `:ilu` (incomplete LU) | Drop-tolerance-dependent | Sparse triangular solves | Lower-fill approximation to compare with LU |
| `:diag` (Jacobi) | $O(N)$ | $O(N)$ | Lowest-storage baseline |

Sparse LU uses UMFPACK. Compare factor storage, setup time, application time,
iterations, and true residual among LU, ILU, and diagonal modes; problem size
alone does not select the best mode.

### 7.3 Operator Compatibility

The near-field preconditioner works with any system matrix that supports element access `A[m,n]`:

| Matrix type | Source | Notes |
|---|---|---|
| `Matrix{ComplexF64}` | Dense EFIE assembly | Standard workflow |
| `MatrixFreeEFIEOperator` | `matrixfree_efie_operator(mesh, rwg, k)` | No dense matrix needed |
| `ACAOperator` | `build_aca_operator(mesh, rwg, k; ...)` | H-matrix acceleration |
| Geometry inputs | `build_nearfield_preconditioner(mesh, rwg, k, cutoff)` | From scratch |

### 7.4 Convergence Monitoring

Successful calls return convergence statistics:

```julia
I, stats = solve_gmres(Z, v; preconditioner=P_nf)

println("Converged: ", stats.solved)
println("Iterations: ", stats.niter)
println("Final residual: ", stats.residuals[end])
```

The public `solve_gmres` and `solve_gmres_adjoint` entry points fail closed by
default when Krylov reports failure, the returned vector is non-finite, or the
unpreconditioned true residual exceeds its gate. To inspect a partial iterate,
pass both `check_gmres_convergence=false` and `check_true_residual=false`, then
inspect the returned statistics and independently evaluate the residual needed
by the study.

If `stats.niter` equals `maxiter` and the residual is above tolerance, the solver has **stagnated**. Common remedies:

1. Increase the cutoff (e.g., from $1\lambda$ to $1.5\lambda$).
2. Switch from `:diag` to `:lu` factorization.
3. Increase `maxiter`.
4. Check mesh quality (degenerate elements cause localized ill-conditioning).

### 7.5 Memory Considerations

The sparse near-field matrix requires approximately $16 \cdot \mathrm{nnz}$ bytes (complex double entries). For $N = 10{,}000$ and $d_{\mathrm{cut}} = 1\lambda$ with 10% fill, this is roughly $16 \times 10^7 = 160$ MB---much less than the $16 \times 10^8 = 1.6$ GB required for the full dense matrix.

LU factorization introduces fill-in whose storage depends on the sparsity
pattern and ordering. Measure the factors for the target matrix before choosing
a memory budget.

---

## 8. Complete Worked Example

This section presents a self-contained example that demonstrates the full workflow: mesh creation, EFIE assembly, preconditioner construction, GMRES solve, and adjoint gradient computation.

### 8.1 Problem Setup

```julia
using DiffMoM
using LinearAlgebra

# Geometry: 1λ x 1λ PEC plate
freq = 300e6                          # 300 MHz
c0 = 299792458.0
lambda0 = c0 / freq                   # 1.0 m
k = 2pi / lambda0

# Mesh with ~10 points per wavelength
mesh = make_rect_plate(lambda0, lambda0, 10, 10)
rwg = build_rwg(mesh)
N = rwg.nedges
println("N = $N unknowns")
```

### 8.2 EFIE Assembly and Excitation

```julia
# Assemble dense EFIE matrix
Z = assemble_Z_efie(mesh, rwg, k)

# Plane-wave excitation: z-propagating, x-polarized
v = assemble_v_plane_wave(mesh, rwg,
    Vec3(0.0, 0.0, -k),    # propagation direction
    1.0,                     # amplitude
    Vec3(1.0, 0.0, 0.0);    # polarization
    quad_order=3)
```

### 8.3 Direct Solve (Baseline)

```julia
t_direct = @elapsed I_direct = Z \ v
println("Direct solve: $(round(t_direct, digits=3)) s")
```

### 8.4 Preconditioned GMRES Solve

```julia
# Build near-field preconditioner with 1λ cutoff
cutoff = 1.0 * lambda0
P_nf = build_nearfield_preconditioner(Z, mesh, rwg, cutoff)
println("NF preconditioner: nnz = $(round(P_nf.nnz_ratio * 100, digits=1))%")

# Solve with GMRES
t_gmres = @elapsed begin
    I_gmres, stats = solve_gmres(Z, v; preconditioner=P_nf, tol=1e-8)
end
println("GMRES solve: $(round(t_gmres, digits=3)) s, $(stats.niter) iterations")

# Verify agreement with direct solve
rel_err = norm(I_gmres - I_direct) / norm(I_direct)
println("Relative error vs direct: ", rel_err)
```

### 8.5 Matrix-Free Solve (No Dense Matrix)

```julia
# Create matrix-free operator
A = matrixfree_efie_operator(mesh, rwg, k)

# Build preconditioner from the operator
P_nf_mf = build_nearfield_preconditioner(A, cutoff)

# Solve using only matvecs
I_mf, stats_mf = solve_gmres(A, v; preconditioner=P_nf_mf, tol=1e-8)
println("Matrix-free GMRES: $(stats_mf.niter) iterations")
```

### 8.6 Preconditioner from Geometry Alone

```julia
# Build preconditioner without any pre-assembled operator
P_nf_geo = build_nearfield_preconditioner(mesh, rwg, k, cutoff; quad_order=3)

# Use with any compatible system matrix
I_geo, stats_geo = solve_gmres(Z, v; preconditioner=P_nf_geo, tol=1e-8)
println("Geometry-based preconditioner: $(stats_geo.niter) iterations")
```

### 8.7 Adjoint Solve for Optimization

```julia
# Build a directivity objective matrix
G_mat = radiation_vectors(mesh, rwg, grid, k)
pol_ff = pol_linear_x(grid)
Q = build_Q(G_mat, grid, pol_ff)   # broadside directivity

# Forward solve
I, _ = solve_gmres(Z, v; preconditioner=P_nf)

# Adjoint solve: Z† λ = Q I
lambda, stats_adj = solve_gmres_adjoint(Z, Q * I; preconditioner=P_nf)
println("Adjoint solve: $(stats_adj.niter) iterations")

# Compute gradient (if impedance parameters are present)
# g = gradient_impedance(Mp, I, lambda)
```

### 8.8 Using the High-Level API

For the simplest workflow, `solve_scattering` handles everything automatically:

```julia
result = solve_scattering(mesh, freq,
    make_plane_wave(Vec3(0,0,-k), 1.0, Vec3(1,0,0));
    method = :auto,
    verbose = true)

println("Method: ", result.method)
println("GMRES iterations: ", result.gmres_iters)
I_auto = result.I_coeffs
```

Set `nf_cutoff_lambda` when forcing `method=:dense_gmres`. ACA uses its dense
inadmissible blocks, and MLFMA uses its stored near-field matrix.

### 8.9 Comparing Factorization Types

```julia
# Sparse LU preconditioner (default)
P_lu = build_nearfield_preconditioner(Z, mesh, rwg, cutoff; factorization=:lu)
I_lu, stats_lu = solve_gmres(Z, v; preconditioner=P_lu)

# ILU preconditioner (memory/cost trade-off)
P_ilu = build_nearfield_preconditioner(Z, mesh, rwg, cutoff;
                                       factorization=:ilu, ilu_tau=1e-3)
I_ilu, stats_ilu = solve_gmres(Z, v; preconditioner=P_ilu)

# Diagonal (Jacobi) preconditioner
P_diag = build_nearfield_preconditioner(Z, mesh, rwg, cutoff; factorization=:diag)
I_diag, stats_diag = solve_gmres(Z, v; preconditioner=P_diag)

println("Sparse LU:  $(stats_lu.niter) iterations")
println("ILU:        $(stats_ilu.niter) iterations")
println("Diagonal:   $(stats_diag.niter) iterations")
```

---

## 9. Code Mapping

This section maps the concepts presented in this chapter to the source files and functions in `DiffMoM.jl`.

### 9.1 Source Files

| File | Purpose |
|---|---|
| `src/solver/NearFieldPreconditioner.jl` | Near-field sparse preconditioner: type definitions, spatial hashing, sparse matrix assembly, LU factorization, operator wrappers |
| `src/solver/IterativeSolve.jl` | GMRES solver wrappers via Krylov.jl: forward and adjoint solves with preconditioner dispatch |
| `src/solver/Solve.jl` | High-level solver dispatch (`solve_forward`, `solve_system`) that routes to direct or GMRES |
| `src/optimization/Adjoint.jl` | Adjoint equation solve (`solve_adjoint`) and gradient computation (`gradient_impedance`) |
| `src/assembly/EFIE.jl` | `MatrixFreeEFIEOperator` definition and `matrixfree_efie_operator` constructor |
| `src/Workflow.jl` | `solve_scattering` high-level API with automatic method selection and preconditioner setup |

### 9.2 Key Functions

| Function | File | Description |
|---|---|---|
| `build_nearfield_preconditioner` | `NearFieldPreconditioner.jl` | Build sparse NF preconditioner (4 overloads) |
| `rwg_centers` | `NearFieldPreconditioner.jl` | Compute RWG basis function centers |
| `solve_gmres` | `IterativeSolve.jl` | Forward GMRES solve with optional preconditioning |
| `solve_gmres_adjoint` | `IterativeSolve.jl` | Adjoint GMRES solve with adjoint preconditioner |
| `solve_forward` | `Solve.jl` | Dispatch to direct or GMRES based on `solver` kwarg |
| `solve_adjoint` | `Adjoint.jl` | Adjoint solve: $\mathbf{Z}^\dagger \boldsymbol{\lambda} = \mathbf{Q}\mathbf{I}$ |
| `gradient_impedance` | `Adjoint.jl` | Compute $\partial\Phi/\partial\theta_p$ from forward and adjoint solutions |
| `solve_scattering` | `Workflow.jl` | End-to-end scattering solve with auto method selection |

### 9.3 Key Types

| Type | File | Description |
|---|---|---|
| `AbstractPreconditionerData` | `NearFieldPreconditioner.jl` | Abstract base for all preconditioner data |
| `NearFieldPreconditionerData` | `NearFieldPreconditioner.jl` | Sparse LU preconditioner (stores UMFPACK factorization) |
| `ILUPreconditionerData` | `NearFieldPreconditioner.jl` | Incomplete LU preconditioner (stores ILU factors) |
| `DiagonalPreconditionerData` | `NearFieldPreconditioner.jl` | Jacobi preconditioner (stores inverse diagonal) |
| `NearFieldOperator` | `NearFieldPreconditioner.jl` | Krylov.jl-compatible wrapper applying $\mathbf{Z}_{\mathrm{nf}}^{-1}$ |
| `NearFieldAdjointOperator` | `NearFieldPreconditioner.jl` | Krylov.jl-compatible wrapper applying $\mathbf{Z}_{\mathrm{nf}}^{-\dagger}$ |
| `MatrixFreeEFIEOperator` | `EFIE.jl` | Matrix-free EFIE matvec operator |

### 9.4 Data Flow Diagram

```
mesh, rwg, k
     |
     v
assemble_Z_efie  or  matrixfree_efie_operator
     |                        |
     v                        v
 Z (dense)        A (MatrixFreeEFIEOperator)
     |                        |
     +------------------------+
     |
     v
build_nearfield_preconditioner(... , cutoff)
     |
     +---> rwg_centers() ---> spatial hash ---> sparse Z_nf ---> lu(Z_nf)
     |
     v
P_nf :: AbstractPreconditionerData
     |
     +-------------------+
     |                   |
     v                   v
solve_gmres          solve_gmres_adjoint
(NearFieldOperator)  (NearFieldAdjointOperator)
     |                   |
     v                   v
  I (forward)        lambda (adjoint)
     |                   |
     +-------------------+
     |
     v
gradient_impedance(Mp, I, lambda)
     |
     v
   g (gradient vector)
```

---

## 10. Exercises

### Conceptual

1. Explain why retaining near-field interactions can improve conditioning over
   a diagonal (Jacobi) preconditioner. Which additional coupling does it retain,
   and what measurements would establish how iteration count scales?

2. Suppose you double the operating frequency while keeping the physical mesh fixed. What happens to (a) the number of unknowns $N$, (b) the near-field cutoff in meters for $\alpha = 1\lambda$, (c) the sparsity of $\mathbf{Z}_{\mathrm{nf}}$, and (d) the expected GMRES iteration count?

3. Why does right preconditioning minimize the true residual $\|\mathbf{b} - \mathbf{Z}\mathbf{x}_m\|$ while left preconditioning minimizes the preconditioned residual $\|\mathbf{M}^{-1}(\mathbf{b} - \mathbf{Z}\mathbf{x}_m)\|$? Derive this from the GMRES minimization property.

4. In the adjoint solve $\mathbf{Z}^\dagger \boldsymbol{\lambda} = \mathbf{Q}\mathbf{I}$, why must the preconditioner be $\mathbf{Z}_{\mathrm{nf}}^{-\dagger}$ rather than $\mathbf{Z}_{\mathrm{nf}}^{-1}$? What would go wrong if the forward preconditioner were used directly?

### Coding

5. Write a script that solves a scattering problem using both direct LU and preconditioned GMRES for mesh sizes $N = 200, 500, 1000, 2000$. Plot wall-clock time vs. $N$ for both methods and identify the crossover point.

6. Implement a cutoff sweep: for a fixed problem ($N \approx 1000$), vary $d_{\mathrm{cut}}$ from $0.25\lambda$ to $3\lambda$ in steps of $0.25\lambda$. Record the number of GMRES iterations and the preconditioner build time. Plot both quantities and find the optimal cutoff that minimizes total solve time (build + iterations $\times$ matvec cost).

7. Compare the convergence behavior of `:lu` and `:diag` factorizations by plotting the residual history (available in `stats.residuals`) for both preconditioners on the same problem. How many additional iterations does the diagonal preconditioner require?

8. Verify adjoint consistency: for a small problem ($N \approx 100$) with impedance parameters, compute the adjoint gradient and compare with a centered finite-difference approximation. Use the `fd_grad` utility from `src/optimization/Verification.jl`. The relative error should be below $10^{-5}$.

### Advanced

9. Implement a **block-diagonal preconditioner** that partitions the RWG basis functions into spatial clusters (e.g., using the cluster tree from `src/fast/ClusterTree.jl`) and factorizes each diagonal block independently. Compare its iteration count with the standard near-field preconditioner.

10. Investigate the effect of mesh non-uniformity on preconditioner performance. Create a mesh with locally refined regions and measure GMRES iteration counts with a fixed cutoff. Does the cutoff need to be adapted to the local mesh density?

---

## 11. Chapter Checklist

- [ ] Compare the measured direct and iterative costs against the $O(N^3)$ and $O(mN^2)$ models.
- [ ] Describe the GMRES algorithm: Krylov subspace, Arnoldi process, least-squares minimization, and convergence dependence on eigenvalue clustering.
- [ ] Construct a near-field sparse preconditioner using `build_nearfield_preconditioner` with an appropriate cutoff in wavelengths.
- [ ] Distinguish left preconditioning ($\mathbf{M}^{-1}\mathbf{Z}\mathbf{x} = \mathbf{M}^{-1}\mathbf{b}$) from right preconditioning ($\mathbf{Z}\mathbf{M}^{-1}\mathbf{y} = \mathbf{b}$) and their residual-monitoring implications.
- [ ] Explain which interactions the near-field preconditioner retains and measure iteration growth over a defined problem sequence.
- [ ] Use `solve_gmres` and `solve_gmres_adjoint` with a `NearFieldPreconditionerData` object for forward and adjoint solves.
- [ ] Set up the adjoint preconditioner $\mathbf{Z}_{\mathrm{nf}}^{-\dagger}$ and verify that the same `P_nf` object is reused for both forward and adjoint directions.
- [ ] Choose between `:lu`, `:ilu`, and `:diag` from measured setup, storage, iteration, and residual results.
- [ ] Diagnose GMRES convergence issues using `stats.niter` and `stats.residuals`, and apply appropriate remedies (increase cutoff, switch factorization, check mesh).
- [ ] Locate the relevant source files: `src/solver/NearFieldPreconditioner.jl`, `src/solver/IterativeSolve.jl`, `src/solver/Solve.jl`, `src/optimization/Adjoint.jl`, `src/Workflow.jl`.

---

## 12. Further Reading

### Iterative Methods

- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM. The standard reference on Krylov subspace methods, including GMRES, preconditioning strategies, and convergence theory.
- Greenbaum, A. (1997). *Iterative Methods for Solving Linear Systems*. SIAM. Rigorous treatment of GMRES convergence bounds and polynomial approximation.

### Preconditioning for Integral Equations

- Chew, W. C., Jin, J. M., Michielssen, E., & Song, J. (2001). *Fast and Efficient Algorithms in Computational Electromagnetics*. Artech House. Covers near-field preconditioning, fast multipole methods, and their interplay with iterative solvers.
- Vipiana, F., Pirinoli, P., & Vecchi, G. (2005). A multiresolution method of moments for triangular meshes. *IEEE Trans. Antennas Propag.*, 53(7), 2247--2258. Discusses hierarchical preconditioners for MoM.

### EFIE and MoM Foundations

- Rao, S. M., Wilton, D. R., & Glisson, A. W. (1982). Electromagnetic scattering by surfaces of arbitrary shape. *IEEE Trans. Antennas Propag.*, 30(3), 409--418. The original RWG paper establishing the basis functions used throughout this package.
- Harrington, R. F. (1993). *Field Computation by Moment Methods*. IEEE Press. Classic MoM text.

### Software

- Krylov.jl documentation: [https://juliasmoothoptimizers.github.io/Krylov.jl/](https://juliasmoothoptimizers.github.io/Krylov.jl/). Reference for the `gmres` function signatures, preconditioner interface (`M`/`N` keywords), and convergence statistics.

---

*Next: [ACA and H-Matrix Compression](02-aca-hmatrix-compression.md) introduces hierarchical low-rank approximation for $O(N \log^2 N)$ matvecs, enabling preconditioned GMRES to scale beyond $N = 10{,}000$ without dense matrix storage.*
