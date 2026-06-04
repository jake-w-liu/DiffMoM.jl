# Impedance Sensitivities

## Purpose

This chapter explains why impedance parameters are particularly well‑suited for gradient‑based inverse design in the EFIE–MoM framework. The key insight is that the derivative of the system matrix with respect to an impedance parameter, $\partial \mathbf{Z}/\partial \theta_p$, can be expressed **analytically** using precomputed patch mass matrices $\mathbf{M}_p$. This eliminates the need for numerical differentiation of the singular EFIE kernel, ensures exact gradients, and decouples the expensive geometry‑dependent assembly from the parameter‑dependent optimization loop.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the closed‑form expression for $\partial \mathbf{Z}/\partial \theta_p$ for both resistive and reactive impedance sheets.
2. Explain how patch mass matrices $\mathbf{M}_p$ capture the geometry of the impedance distribution.
3. Implement the gradient formulas $\partial J/\partial \theta_p = 2\Re\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$ (resistive) and $-2\Im\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$ (reactive).
4. Understand why impedance sensitivities are numerically attractive compared to shape or material derivatives.
5. Use `precompute_patch_mass` and `gradient_impedance` in practice.

---

## 1. Impedance Boundary Condition and Its Discretization

### 1.1 Continuous Formulation

Consider a metasurface occupying a surface $\Gamma$. The impedance boundary condition (IBC) relates the tangential total electric field to the surface current density:

```math
\mathbf{E}^{\mathrm{tot}}_t(\mathbf{r}) = Z_s(\mathbf{r};\boldsymbol{\theta}) \mathbf{J}(\mathbf{r}), \qquad \mathbf{r} \in \Gamma,
\label{eq:IBC_continuous}
```

where $Z_s(\mathbf{r};\boldsymbol{\theta})$ is the surface impedance (units: $\Omega$). In this package, the impedance is parameterized as a **piecewise‑constant** function over a partition of $\Gamma$ into $P$ patches $\{\Gamma_p\}_{p=1}^P$:

```math
Z_s(\mathbf{r};\boldsymbol{\theta}) = \sum_{p=1}^P \theta_p \,\chi_{\Gamma_p}(\mathbf{r}),
\qquad
\chi_{\Gamma_p}(\mathbf{r}) =
\begin{cases}
1, & \mathbf{r} \in \Gamma_p,\\
0, & \text{otherwise}.
\end{cases}
\label{eq:impedance_param}
```

Here $\boldsymbol{\theta} \in \mathbb{R}^P$ are the design variables. Two physical interpretations are supported:

- **Resistive sheet**: $\theta_p$ is a real surface resistance ($\theta_p \ge 0$ for passive sheets).
- **Reactive sheet**: $\theta_p$ is a real surface reactance, and the impedance is purely imaginary: $Z_s = i\theta_p$ (inductive if $\theta_p>0$, capacitive if $\theta_p<0$).

### 1.2 Galerkin Discretization

Substituting the IBC into the EFIE and applying Galerkin testing with RWG basis functions $\{\mathbf{f}_m\}$ yields the MoM matrix

```math
\mathbf{Z}(\boldsymbol{\theta}) = \mathbf{Z}_{\mathrm{EFIE}} + \mathbf{Z}_{\mathrm{imp}}(\boldsymbol{\theta}),
\label{eq:Z_total}
```

where $\mathbf{Z}_{\mathrm{EFIE}}$ is the standard EFIE matrix (independent of $\boldsymbol{\theta}$) and $\mathbf{Z}_{\mathrm{imp}}$ is the impedance contribution. Using the piecewise‑constant expansion $\eqref{eq:impedance_param}$, the impedance matrix entries are

```math
[\mathbf{Z}_{\mathrm{imp}}(\boldsymbol{\theta})]_{mn}
=
-\int_\Gamma \mathbf{f}_m(\mathbf{r}) \cdot
\left[ \sum_{p=1}^P \theta_p \chi_{\Gamma_p}(\mathbf{r}) \right]
\mathbf{f}_n(\mathbf{r}) \, dS.
```

Because the impedance is constant on each patch, the integral separates:

```math
\mathbf{Z}_{\mathrm{imp}}(\boldsymbol{\theta})
=
-\sum_{p=1}^P \theta_p \underbrace{
\int_{\Gamma_p} \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{f}_n(\mathbf{r}) \, dS
}_{\displaystyle [\mathbf{M}_p]_{mn}}.
```

Thus,

```math
\boxed{
\mathbf{Z}_{\mathrm{imp}}(\boldsymbol{\theta})
=
-\sum_{p=1}^P \theta_p \mathbf{M}_p
\qquad \text{(resistive)}.
}
\label{eq:Zimp_resistive}
```

For reactive sheets, replace $\theta_p$ by $i\theta_p$:

```math
\boxed{
\mathbf{Z}_{\mathrm{imp}}(\boldsymbol{\theta})
=
-\sum_{p=1}^P i\theta_p \mathbf{M}_p
\qquad \text{(reactive)}.
}
\label{eq:Zimp_reactive}
```

The matrices $\mathbf{M}_p \in \mathbb{C}^{N\times N}$ are the **patch mass matrices**—Gram matrices of the RWG basis functions restricted to patch $\Gamma_p$. They are Hermitian positive semidefinite and depend only on geometry, not on frequency or design variables.

---

## 2. Derivative Blocks $\partial \mathbf{Z}/\partial \theta_p$

### 2.1 Closed‑Form Expressions

Differentiating $\eqref{eq:Z_total}$ with respect to $\theta_p$ is straightforward because $\mathbf{Z}_{\mathrm{EFIE}}$ is independent of $\boldsymbol{\theta}$ and $\mathbf{Z}_{\mathrm{imp}}$ is linear in $\theta_p$. From $\eqref{eq:Zimp_resistive}$ and $\eqref{eq:Zimp_reactive}$ we obtain the exact derivatives

```math
\boxed{
\frac{\partial \mathbf{Z}}{\partial \theta_p}
=
-\mathbf{M}_p
\qquad \text{(resistive)},
}
\label{eq:dZ_dtheta_resistive}
```

```math
\boxed{
\frac{\partial \mathbf{Z}}{\partial \theta_p}
=
-i\mathbf{M}_p
\qquad \text{(reactive)}.
}
\label{eq:dZ_dtheta_reactive}
```

These formulas are **exact** for the chosen discretization and patch parameterization. No numerical approximation or differentiation of singular kernels is required.

### 2.2 Geometric Interpretation

The patch mass matrix $\mathbf{M}_p$ measures the overlap of RWG basis functions on patch $\Gamma_p$. Its entries are

```math
[\mathbf{M}_p]_{mn}
=
\int_{\Gamma_p} \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{f}_n(\mathbf{r}) \, dS.
```

If basis functions $\mathbf{f}_m$ and $\mathbf{f}_n$ have no support on $\Gamma_p$, then $[\mathbf{M}_p]_{mn}=0$. Consequently, $\partial \mathbf{Z}/\partial \theta_p$ is a **sparse** matrix (or at least has many zeros) when patches are small relative to the mesh size. This sparsity can be exploited for efficient gradient computation.

### 2.3 Connection to the Adjoint Gradient

Substituting $\eqref{eq:dZ_dtheta_resistive}$ into the general adjoint gradient formula (Chapter 1, Eq. $\eqref{eq:adjoint_grad}$) gives

```math
\frac{\partial J}{\partial \theta_p}
=
-2\,\Re\!\left\{
\boldsymbol{\lambda}^\dagger (-\mathbf{M}_p) \mathbf{I}
\right\}
=
+2\,\Re\!\left\{
\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}
\right\}.
\label{eq:grad_resistive}
```

The algebraic steps are:
1. Substitute $\partial \mathbf{Z}/\partial \theta_p = -\mathbf{M}_p$ into $\partial J/\partial \theta_p = -2\Re\{\boldsymbol{\lambda}^\dagger (\partial \mathbf{Z}/\partial \theta_p) \mathbf{I}\}$
2. Distribute the minus sign: $-2\Re\{\boldsymbol{\lambda}^\dagger (-\mathbf{M}_p) \mathbf{I}\} = -2\Re\{-\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$
3. Use linearity of real part: $-2\Re\{-\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\} = +2\Re\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$

Similarly, for reactive sheets, substitute $\partial \mathbf{Z}/\partial \theta_p = -i\mathbf{M}_p$:

```math
\frac{\partial J}{\partial \theta_p}
=
-2\,\Re\!\left\{
\boldsymbol{\lambda}^\dagger (-i\mathbf{M}_p) \mathbf{I}
\right\}
=
+2\,\Re\!\left\{
i\,\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}
\right\}
=
-2\,\Im\!\left\{
\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}
\right\}.
\label{eq:grad_reactive}
```

The key step uses the identity $\Re\{i z\} = -\Im\{z\}$ for any complex number $z$. The full derivation:
1. Substitute $\partial \mathbf{Z}/\partial \theta_p = -i\mathbf{M}_p$: $-2\Re\{\boldsymbol{\lambda}^\dagger (-i\mathbf{M}_p) \mathbf{I}\}$
2. Distribute the minus sign: $-2\Re\{-i\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\} = +2\Re\{i\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$
3. Apply $\Re\{i z\} = -\Im\{z\}$: $+2\Re\{i\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\} = -2\Im\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$

Defining the scalar overlap $l_p = \boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I} \in \mathbb{C}$, we obtain the compact forms used in the code:

| Parameterization | Gradient formula | Code flag |
|-----------------|-----------------|-----------|
| Resistive | $\partial J/\partial \theta_p = 2\Re\{l_p\}$ | `reactive=false` |
| Reactive | $\partial J/\partial \theta_p = -2\Im\{l_p\}$ | `reactive=true` |

These formulas are implemented in `gradient_impedance` (see Section 5).

---

## 3. Why Impedance Sensitivities Are Numerically Attractive

### 3.1 Decoupling of Geometry and Parameters

The expensive part of EFIE–MoM assembly—computing the singular integrals in $\mathbf{Z}_{\mathrm{EFIE}}$—is **independent** of $\boldsymbol{\theta}$. It can be performed once at the beginning of an optimization and reused for all iterations. Only the inexpensive linear combination $\sum_p \theta_p \mathbf{M}_p$ needs updating when $\boldsymbol{\theta}$ changes.

### 3.2 No Numerical Differentiation of Singular Kernels

Shape derivatives (sensitivity to vertex positions) require differentiating the $1/R$ singular kernel, which is both mathematically delicate and computationally expensive. Impedance derivatives avoid this complexity entirely because $\mathbf{M}_p$ involves only non‑singular integrals of basis‑function dot products.

### 3.3 Reusability of Derivative Blocks

The matrices $\mathbf{M}_p$ depend solely on the mesh and patch partition. They can be precomputed and stored, then reused across:
- Multiple optimization iterations,
- Different frequency points,
- Various incident angles,
- Different objective functions (different $\mathbf{Q}$ matrices).

This precomputation amortizes the assembly cost over the entire design workflow.

### 3.4 Exact Gradients

Since $\partial \mathbf{Z}/\partial \theta_p$ is exact, the adjoint gradient is also exact (up to floating‑point roundoff). There is no approximation error from finite‑difference step size or iterative linear‑solver tolerances.

### 3.5 Scalability

For large problems, $\mathbf{M}_p$ can be stored in a sparse format (most entries are zero). The gradient computation then scales as $O(P \cdot \mathrm{nnz}(\mathbf{M}_p))$, which is often linear in $N$ for localized patches.

---

## 4. Patch Mass Matrices: Construction and Properties

### 4.1 Definition

Given a mesh $\mathcal{M}$ and a patch partition $\{\Gamma_p\}$, the patch mass matrix for patch $p$ is

```math
\mathbf{M}_p = \int_{\Gamma_p} \mathbf{f}_m(\mathbf{r}) \cdot \mathbf{f}_n(\mathbf{r}) \, dS.
```

In practice, $\Gamma_p$ is a union of mesh triangles. The integral is computed by summing contributions from each triangle in the patch using Gaussian quadrature.

### 4.2 Implementation via `precompute_patch_mass`

The function `precompute_patch_mass` (in `src/assembly/Impedance.jl`) computes all $\mathbf{M}_p$ matrices:

```julia
function precompute_patch_mass(mesh, rwg, partition; quad_order=3)
    # Returns a vector of matrices Mp[p] for p = 1:P
    ...
end
```

**Arguments**:
- `mesh`: triangular mesh (`Mesh` type).
- `rwg`: RWG basis structure (`RWG` type).
- `partition`: `PatchPartition` object mapping each triangle to a patch index.
- `quad_order`: quadrature order for Gaussian integration (default 3).

**Output**: A vector `Mp` of length $P$, where `Mp[p]` is an $N\times N$ sparse matrices (`SparseMatrixCSC`), since the code uses `spzeros` internally.

### 4.3 Example: Creating a Patch Partition

A simple partition assigns each triangle to its own patch ($P = N_{\text{tri}}$):

```julia
using DiffMoM

mesh = make_rect_plate(0.1, 0.1, 10, 10)
rwg = build_rwg(mesh)
ntri = ntriangles(mesh)
partition = PatchPartition(collect(1:ntri), ntri)  # each triangle separate
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
```

More practically, patches are grouped into larger regions (e.g., $2\times2$ blocks of triangles) to reduce the number of design variables.

### 4.4 Properties of $\mathbf{M}_p$

1. **Hermitian symmetry**: $\mathbf{M}_p = \mathbf{M}_p^\dagger$.
2. **Positive semidefiniteness**: $\mathbf{v}^\dagger \mathbf{M}_p \mathbf{v} \ge 0$ for any $\mathbf{v} \in \mathbb{C}^N$.
3. **Additivity**: The global mass matrix $\mathbf{M}_{\text{global}} = \sum_{p=1}^P \mathbf{M}_p$ is the standard RWG Gram matrix.
4. **Sparsity**: $\mathbf{M}_p$ is non‑zero only for basis pairs that have support on patch $\Gamma_p$.

---

## 5. Gradient Computation in Practice

### 5.1 Function `gradient_impedance`

The gradient formulas $\eqref{eq:grad_resistive}$ and $\eqref{eq:grad_reactive}$ are implemented in `gradient_impedance` (`src/optimization/Adjoint.jl`):

```julia
function gradient_impedance(Mp, I, lambda; reactive=false)
    P = length(Mp)
    g = zeros(Float64, P)
    for p in 1:P
        # Compute l_p = λ† M_p I
        l_p = dot(lambda, Mp[p] * I)
        if reactive
            g[p] = -2 * imag(l_p)   # -2 Im{l_p}
        else
            g[p] =  2 * real(l_p)   # +2 Re{l_p}
        end
    end
    return g
end
```

**Arguments**:
- `Mp`: vector of precomputed patch mass matrices.
- `I`: forward solution vector ($\mathbf{I}$).
- `lambda`: adjoint solution vector ($\boldsymbol{\lambda}$).
- `reactive`: Boolean flag; `false` for resistive, `true` for reactive.

**Returns**: A real vector $\mathbf{g} \in \mathbb{R}^P$ with $\mathbf{g}[p] = \partial J/\partial \theta_p$.

### 5.2 Complete Gradient Pipeline

A typical gradient computation within an optimization loop looks like:

```julia
# Precomputation (once)
mesh, rwg = ...  # geometry
partition = ...  # patch definition
Mp = precompute_patch_mass(mesh, rwg, partition)
Z_efie = assemble_Z_efie(mesh, rwg, k)

# Inside each iteration
theta = ...  # current design parameters
Z = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
I = solve_forward(Z, v)
λ = solve_adjoint(Z, Q, I)
g = gradient_impedance(Mp, I, λ; reactive=true)
```

### 5.3 Gradient Scaling and Units

- **Resistive gradient**: $\partial J/\partial \theta_p$ has units of $\text{W}/\Omega$ (watts per ohm). Increasing $\theta_p$ (adding resistance) generally **dissipates** power, so $\partial J/\partial \theta_p$ is often negative for radiation‑maximization objectives.
- **Reactive gradient**: $\partial J/\partial \theta_p$ has units of $\text{W}/\Omega$ as well, but $\theta_p$ is a reactance. Positive $\theta_p$ (inductive) stores magnetic energy, negative $\theta_p$ (capacitive) stores electric energy.

These sign conventions are consistent with the gradient formulas above.

---

## 6. Verification and Debugging

### 6.1 Finite‑Difference Check for a Single Patch

To verify that `gradient_impedance` returns the correct derivative for patch $p$, perturb $\theta_p$ by a small $\epsilon$ and compare with a central‑difference approximation:

```julia
function check_single_patch(p, epsilon=1e-8)
    # Current parameters
    theta0 = copy(theta)
    
    # Adjoint gradient
    g_adj = gradient_impedance(Mp, I, λ; reactive=true)[p]
    
    # Finite‑difference
    theta_plus = copy(theta0); theta_plus[p] += epsilon
    theta_minus = copy(theta0); theta_minus[p] -= epsilon
    
    Z_plus = assemble_full_Z(Z_efie, Mp, theta_plus; reactive=true)
    Z_minus = assemble_full_Z(Z_efie, Mp, theta_minus; reactive=true)
    
    I_plus = solve_forward(Z_plus, v)
    I_minus = solve_forward(Z_minus, v)
    
    J_plus = compute_objective(I_plus, Q)
    J_minus = compute_objective(I_minus, Q)
    
    g_fd = (J_plus - J_minus) / (2epsilon)
    
    rel_err = abs(g_adj - g_fd) / (abs(g_fd) + 1e-12)
    println("Patch $p: adjoint = $g_adj, FD = $g_fd, rel error = $rel_err")
    return rel_err
end
```

The relative error should be $\lesssim 10^{-6}$ for well‑conditioned problems.

### 6.2 Common Mistakes

1. **Wrong `reactive` flag**: Using `reactive=false` for reactive design yields gradients with the wrong sign and magnitude.
2. **Incorrect patch partition**: If the partition does not match the parameter vector length, `gradient_impedance` will throw a dimension mismatch.
3. **Missing factor of 2**: Forgetting the factor 2 in $\eqref{eq:grad_resistive}$ or $\eqref{eq:grad_reactive}$ leads to gradients that are half the correct size.
4. **Using $\mathbf{Z}_{\mathrm{EFIE}}$ instead of $\mathbf{Z}$**: The gradient requires the **full** matrix $\mathbf{Z} = \mathbf{Z}_{\mathrm{EFIE}} + \mathbf{Z}_{\mathrm{imp}}$; using only $\mathbf{Z}_{\mathrm{EFIE}}$ gives incorrect $\mathbf{I}$ and $\boldsymbol{\lambda}$.

---

## 7. Advanced Topics

### 7.1 Mixed Resistive‑Reactive Design

The current implementation supports either purely resistive or purely reactive sheets. A mixed design with complex impedance $Z_s = R + iX$ would require two parameters per patch: $\theta_p^{\mathrm{res}}$ and $\theta_p^{\mathrm{react}}$. The gradient would then have two components:

```math
\frac{\partial J}{\partial R_p} = 2\Re\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}, \qquad
\frac{\partial J}{\partial X_p} = -2\Im\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}.
```

This extension is straightforward but not yet implemented.

### 7.2 Spatially Varying Impedance Profiles

The piecewise‑constant model can approximate any continuous impedance profile $Z_s(\mathbf{r})$ by using many small patches. For smooth profiles, one could introduce higher‑order basis functions (e.g., linear or bilinear) on the patches, leading to more complex derivative blocks. The current implementation uses the simplest (constant‑per‑patch) model, which is sufficient for many metasurface applications.

### 7.3 Connection to Surface Susceptibility Models

For thin dielectric layers, the impedance is related to surface susceptibility tensors $\bar{\bar{\chi}}$. The patch‑mass approach can be extended to anisotropic impedance by replacing the scalar $\theta_p$ with a $2\times2$ tensor and modifying $\mathbf{M}_p$ accordingly. This is beyond the scope of the current package.

---

## 8. Summary of Key Formulas

| Quantity | Resistive sheet | Reactive sheet |
|----------|----------------|----------------|
| Impedance contribution | $\displaystyle \mathbf{Z}_{\mathrm{imp}} = -\sum_p \theta_p \mathbf{M}_p$ | $\displaystyle \mathbf{Z}_{\mathrm{imp}} = -\sum_p i\theta_p \mathbf{M}_p$ |
| Derivative block | $\displaystyle \frac{\partial \mathbf{Z}}{\partial \theta_p} = -\mathbf{M}_p$ | $\displaystyle \frac{\partial \mathbf{Z}}{\partial \theta_p} = -i\mathbf{M}_p$ |
| Gradient component | $\displaystyle \frac{\partial J}{\partial \theta_p} = +2\Re\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$ | $\displaystyle \frac{\partial J}{\partial \theta_p} = -2\Im\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$ |
| Code flag | `reactive=false` | `reactive=true` |

---

## 9. Code Mapping

- **`src/Types.jl`** – Patch partition definition (`PatchPartition`).
- **`src/assembly/Impedance.jl`** – Patch mass matrix assembly (`precompute_patch_mass`).
- **`src/optimization/Adjoint.jl`** – Gradient computation (`gradient_impedance`).
- **`src/solver/Solve.jl`** – Assembly of full matrix $\mathbf{Z}(\boldsymbol{\theta})$ (`assemble_full_Z`).
- **`src/assembly/EFIE.jl`** – Assembly of $\mathbf{Z}_{\mathrm{EFIE}}$.
- **`test/runtests.jl`** – Verification script for impedance sensitivities.

---

## 10. Exercises

### 10.1 Conceptual Questions

1. **Physical interpretation**: Explain why $\partial \mathbf{Z}/\partial \theta_p = -\mathbf{M}_p$ for a resistive sheet. What does the negative sign signify physically?
2. **Patch size effect**: Suppose you double the size of each patch (merge adjacent patches). How does this affect the sparsity of $\mathbf{M}_p$ and the computational cost of `gradient_impedance`?
3. **Reactive gradient sign**: Why does the reactive gradient formula have a minus sign in front of the imaginary part, while the resistive formula has a plus sign in front of the real part?

### 10.2 Derivation Tasks

1. **Derive the gradient formulas**: Starting from the general adjoint gradient formula and $\partial \mathbf{Z}/\partial \theta_p = -i\mathbf{M}_p$, derive $\eqref{eq:grad_reactive}$ step by step.
2. **Mass matrix properties**: Prove that $\mathbf{M}_p$ is Hermitian positive semidefinite. Show that $\mathbf{v}^\dagger \mathbf{M}_p \mathbf{v} = \int_{\Gamma_p} |\mathbf{v}(\mathbf{r})|^2 dS$, where $\mathbf{v}(\mathbf{r}) = \sum_n v_n \mathbf{f}_n(\mathbf{r})$.

### 10.3 Coding Exercises

1. **Basic verification**: Reproduce the minimal example from Section 5.2 for a small plate with 10×10 triangles. Verify that the gradient vector length equals the number of patches.
2. **Sign change test**: Compute gradients for the same problem with `reactive=false` and `reactive=true`. Compare the signs and magnitudes; explain the differences using the formulas in Section 8.
3. **Single‑patch finite‑difference check**: Implement the verification function from Section 6.1 and test it for several randomly chosen patches. Ensure the relative error is below $10^{-6}$.

### 10.4 Advanced Challenges

1. **Sparse storage benchmark**: The current implementation already stores $\mathbf{M}_p$ as sparse matrices (`SparseMatrixCSC`, via `spzeros` in `precompute_patch_mass`). Benchmark the memory savings compared to dense storage for a large mesh (e.g., $N > 2000$) by comparing `sizeof(Matrix(Mp[p]))` vs. the sparse representation. Also profile `gradient_impedance` to measure the speed‑up from sparse matrix‑vector products.
2. **Gradient with respect to patch geometry**: Suppose patch boundaries can move (changing which triangles belong to which patch). Derive the gradient of $J$ with respect to patch‑boundary positions. This is a shape‑derivative problem that combines impedance and geometry sensitivities.

---

## 11. Chapter Checklist

After studying this chapter, you should be able to:

- [ ] **Derive** the closed‑form expressions $\partial \mathbf{Z}/\partial \theta_p = -\mathbf{M}_p$ (resistive) and $-i\mathbf{M}_p$ (reactive).
- [ ] **Explain** why impedance sensitivities are numerically attractive: no singular‑kernel differentiation, decoupling of geometry and parameters, reusability of derivative blocks.
- [ ] **Construct** patch mass matrices using `precompute_patch_mass` and a `PatchPartition`.
- [ ] **Compute** gradients using `gradient_impedance` with the correct `reactive` flag.
- [ ] **Verify** gradient accuracy with a single‑patch finite‑difference check.
- [ ] **Interpret** the physical meaning of the gradient sign for resistive vs. reactive sheets.

If you can confidently check all items, you have mastered impedance sensitivities in `DiffMoM.jl` and are ready to proceed to Chapter 3 (Ratio Objectives).

---

## 12. Further Reading

1. **Impedance boundary conditions in electromagnetics**:
   - Senior, T. B. A., & Volakis, J. L. (1995). *Approximate boundary conditions in electromagnetics*. IET.
   - Holloway, C. L., & Kuester, E. F. (2012). *Generalized sheet transition conditions*. IEEE Transactions on Antennas and Propagation, 60(2), 517–528.

2. **Metasurface modeling and design**:
   - Pfeiffer, C., & Grbic, A. (2013). *Metamaterial Huygens’ surfaces: tailoring wave fronts with reflectionless sheets*. Physical Review Letters, 110(19), 197401.
   - Epstein, A., & Eleftheriades, G. V. (2016). *Huygens’ metasurfaces via the equivalence principle: design and applications*. Journal of the Optical Society of America B, 33(2), A31–A50.

3. **Mass matrices in finite elements**:
   - Jin, J.‑M. (2014). *The finite element method in electromagnetics* (3rd ed.). Wiley. (Chapter 6 covers mass matrices in vector finite elements.)
   - Peterson, A. F., Ray, S. L., & Mittra, R. (1998). *Computational methods for electromagnetics*. IEEE Press.

4. **Gradient‑based optimization of impedance surfaces**:
   - Díaz‑Rubio, A., et al. (2017). *From the generalized reflection law to the realization of perfect anomalous reflectors*. Science Advances, 3(8), e1602714.
   - Chen, M., et al. (2020). *Inverse design of a single‑layer wide‑angle metasurface based on gradient optimization*. Optics Express, 28(18), 26536–26550.

---

*Next: Chapter 3, “Ratio Objectives,” explains why directivity‑style ratio objectives are preferred for beam steering and how the package computes stable gradients using two separate adjoint solves.*
