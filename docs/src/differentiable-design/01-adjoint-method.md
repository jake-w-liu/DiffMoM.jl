# Adjoint Method

## Purpose

This chapter explains the adjoint method for computing gradients of electromagnetic objectives with respect to impedance parameters. The adjoint method is the mathematical foundation that enables high‑dimensional inverse design in `DiffMoM.jl` by providing gradients at a computational cost that is **essentially independent** of the number of design variables. Unlike finite‑difference approximations that require $O(P)$ forward solves, the adjoint method requires only **one additional linear solve** per iteration (two for ratio objectives), regardless of $P$.

**What makes this approach "differentiable"?** The key is that we compute **exact analytical derivatives** of the system matrix with respect to design parameters, not numerical approximations. By combining these analytical derivatives with the solution of an auxiliary (adjoint) linear system, we obtain gradients that are accurate to machine precision—enabling reliable gradient‑based optimization even with thousands of design variables.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the adjoint equation from the forward EFIE–MoM system.
2. Explain the connection between the adjoint variable $\boldsymbol{\lambda}$ and the objective gradient.
3. Implement the adjoint gradient formula in code using precomputed derivative blocks.
4. Compare the computational complexity of adjoint vs. finite‑difference gradients.
5. Understand when one vs. two adjoint solves are required (quadratic vs. ratio objectives).

---

## 1. Mathematical Foundation

### 1.1 Forward Problem

Consider a metasurface described by a surface impedance $Z_s(\mathbf{r};\boldsymbol{\theta})$ parameterized by $P$ real design variables $\boldsymbol{\theta} \in \mathbb{R}^P$. The EFIE with impedance boundary condition (IBC) is discretized via Galerkin MoM using RWG basis functions $\{\mathbf{f}_n\}_{n=1}^N$, yielding the linear system

```math
\mathbf{Z}(\boldsymbol{\theta})\,\mathbf{I}(\boldsymbol{\theta}) = \mathbf{v},
\qquad
\mathbf{Z}(\boldsymbol{\theta}) \in \mathbb{C}^{N\times N},\;
\mathbf{I},\mathbf{v} \in \mathbb{C}^{N},

```

where $\mathbf{Z}(\boldsymbol{\theta}) = \mathbf{Z}_{\mathrm{EFIE}} - \sum_{p=1}^P \theta_p \mathbf{M}_p$ for resistive sheets, or $\mathbf{Z}(\boldsymbol{\theta}) = \mathbf{Z}_{\mathrm{EFIE}} - \sum_{p=1}^P i\theta_p \mathbf{M}_p$ for reactive sheets. The right‑hand side $\mathbf{v}$ is the tested incident field, independent of $\boldsymbol{\theta}$.

### 1.2 Quadratic Objective

The radiation performance is quantified by a real‑valued quadratic objective

```math
J(\boldsymbol{\theta}) = \Phi(\mathbf{I}(\boldsymbol{\theta})) = \mathbf{I}^\dagger \mathbf{Q} \mathbf{I},
\qquad
\mathbf{Q} = \mathbf{Q}^\dagger \succeq \mathbf{0},

```

where $\mathbf{Q}$ is a Hermitian positive‑semidefinite matrix assembled from far‑field projection operators (see Chapter 3 of Part II). Common examples include power radiated into a target angular region, cross‑polarization suppression, or sidelobe level minimization.

### 1.3 Gradient via the Chain Rule

To optimize $J$ with gradient‑based methods (e.g., L‑BFGS), we need $\partial J/\partial \theta_p$. Because $\mathbf{I}$ depends implicitly on $\boldsymbol{\theta}$ through $\eqref{eq:forward}$, the total derivative is

```math
\frac{d J}{d \theta_p}
=
\frac{\partial \Phi}{\partial \mathbf{I}^*} \cdot \frac{\partial \mathbf{I}}{\partial \theta_p}
+
\frac{\partial \Phi}{\partial \mathbf{I}} \cdot \frac{\partial \mathbf{I}^*}{\partial \theta_p}.

```

For real‑valued $J$ and complex $\mathbf{I}$, the Wirtinger calculus gives $\partial \Phi/\partial \mathbf{I}^* = \mathbf{Q}\mathbf{I}$ and $\partial \Phi/\partial \mathbf{I} = (\mathbf{Q}\mathbf{I})^*$.

---

## 2. Derivation of the Adjoint Equation

### 2.1 Differentiating the Forward Constraint

Differentiate $\eqref{eq:forward}$ with respect to $\theta_p$:

```math
\mathbf{Z} \frac{\partial \mathbf{I}}{\partial \theta_p}
=
-
\frac{\partial \mathbf{Z}}{\partial \theta_p} \mathbf{I},
\qquad
p=1,\dots,P.

```

This is a linear system for $\partial \mathbf{I}/\partial \theta_p$ with the same matrix $\mathbf{Z}$ but different right‑hand sides. A direct approach would require solving $\eqref{eq:forward_deriv}$ for each $p$, leading to $P$ forward solves—prohibitively expensive for large $P$.

### 2.2 Introducing the Adjoint Variable

The adjoint method circumvents this cost by introducing an auxiliary variable $\boldsymbol{\lambda} \in \mathbb{C}^N$ that satisfies the **adjoint equation**

```math
\mathbf{Z}^\dagger \boldsymbol{\lambda}
=
\frac{\partial \Phi}{\partial \mathbf{I}^*}
=
\mathbf{Q}\mathbf{I}.

```

Physically, $\boldsymbol{\lambda}$ represents the sensitivity of the objective to perturbations in the right‑hand side $\mathbf{v}$. The adjoint equation is a single linear system with the Hermitian‑transposed operator $\mathbf{Z}^\dagger$; its solution cost is comparable to one forward solve.

### 2.3 Gradient Formula

Multiply $\eqref{eq:forward_deriv}$ by $\boldsymbol{\lambda}^\dagger$ from the left and use $\eqref{eq:adjoint_eq}$:

```math
\boldsymbol{\lambda}^\dagger \mathbf{Z} \frac{\partial \mathbf{I}}{\partial \theta_p}
=
-
\boldsymbol{\lambda}^\dagger \frac{\partial \mathbf{Z}}{\partial \theta_p} \mathbf{I}.
```

But $\boldsymbol{\lambda}^\dagger \mathbf{Z} = (\mathbf{Z}^\dagger \boldsymbol{\lambda})^\dagger = (\mathbf{Q}\mathbf{I})^\dagger$. Therefore,

```math
\boldsymbol{\lambda}^\dagger \mathbf{Z} \frac{\partial \mathbf{I}}{\partial \theta_p}
=
(\mathbf{Q}\mathbf{I})^\dagger \frac{\partial \mathbf{I}}{\partial \theta_p}.
```

Now recall the chain rule $\eqref{eq:chain_rule}$:

```math
\frac{d J}{d \theta_p}
=
\frac{\partial \Phi}{\partial \mathbf{I}^*} \cdot \frac{\partial \mathbf{I}}{\partial \theta_p}
+
\frac{\partial \Phi}{\partial \mathbf{I}} \cdot \frac{\partial \mathbf{I}^*}{\partial \theta_p}.
```

With $\partial \Phi/\partial \mathbf{I}^* = \mathbf{Q}\mathbf{I}$ and $\partial \Phi/\partial \mathbf{I} = (\mathbf{Q}\mathbf{I})^*$, we have

```math
\frac{d J}{d \theta_p}
=
(\mathbf{Q}\mathbf{I})^\dagger \frac{\partial \mathbf{I}}{\partial \theta_p}
+
\bigl[(\mathbf{Q}\mathbf{I})^\dagger \frac{\partial \mathbf{I}}{\partial \theta_p}\bigr]^*,
```

because for any complex vectors $\mathbf{a},\mathbf{b}$, $\mathbf{a}^* \cdot \mathbf{b} = (\mathbf{a}^\dagger \mathbf{b})^*$.

From the earlier equation $\boldsymbol{\lambda}^\dagger \mathbf{Z} \frac{\partial \mathbf{I}}{\partial \theta_p} = -\boldsymbol{\lambda}^\dagger \frac{\partial \mathbf{Z}}{\partial \theta_p} \mathbf{I}$, and using $\boldsymbol{\lambda}^\dagger \mathbf{Z} = (\mathbf{Q}\mathbf{I})^\dagger$, we have

```math
(\mathbf{Q}\mathbf{I})^\dagger \frac{\partial \mathbf{I}}{\partial \theta_p}
=
-\boldsymbol{\lambda}^\dagger \frac{\partial \mathbf{Z}}{\partial \theta_p} \mathbf{I}.
```

Substituting this into the chain rule expression gives

```math
\frac{d J}{d \theta_p}
=
-\boldsymbol{\lambda}^\dagger \frac{\partial \mathbf{Z}}{\partial \theta_p} \mathbf{I}
+
\Bigl[-\boldsymbol{\lambda}^\dagger \frac{\partial \mathbf{Z}}{\partial \theta_p} \mathbf{I}\Bigr]^*.
```

Using the identity $z + z^* = 2\Re\{z\}$ for any complex number $z$, we obtain the compact gradient expression

```math
\boxed{
\frac{\partial J}{\partial \theta_p}
=
-2\,\Re\!\left\{
\boldsymbol{\lambda}^\dagger
\left(
\frac{\partial \mathbf{Z}}{\partial \theta_p}
\right)
\mathbf{I}
\right\}},
\qquad p=1,\dots,P.
```

For impedance parameters, $\partial \mathbf{Z}/\partial \theta_p = -\mathbf{M}_p$ (resistive) or $-i\mathbf{M}_p$ (reactive), where $\mathbf{M}_p$ are the precomputed patch mass matrices (Chapter 2). Thus, once $\mathbf{I}$ and $\boldsymbol{\lambda}$ are known, each gradient component reduces to a cheap inner product involving $\mathbf{M}_p$.

---

## 3. Computational Cost Analysis

### 3.1 Cost Breakdown

For a quadratic objective, each optimization iteration requires:

1. **One forward solve** for $\mathbf{I}$ (already needed to evaluate $J$).
2. **One adjoint solve** for $\boldsymbol{\lambda}$ (same cost as a forward solve).
3. **$P$ matrix‑vector contractions** of the form $\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}$, each costing $O(N^2)$ operations (or less if $\mathbf{M}_p$ is sparse).

The total cost is therefore **$O(1)$ linear solves + $O(P)$ inner products**, independent of $P$ in the number of solves.

### 3.2 Comparison with Finite Differences

| Method | Forward solves | Adjoint solves | Inner products | Scaling in $P$ |
|--------|----------------|----------------|----------------|----------------|
| Finite difference (central) | $2P$ | $0$ | $0$ | $O(P)$ |
| Adjoint method | $1$ | $1$ | $P$ | $O(1)$ in solves |

For typical metasurface problems with $P \sim 10^2–10^3$ patches, the adjoint method reduces the per‑iteration cost by **two to three orders of magnitude**, making gradient‑based optimization feasible.

### 3.3 Diagram: Adjoint Gradient Pipeline

```
    Forward solve (1×)           Adjoint solve (1×)
          ↓                            ↓
       Z I = v                    Z† λ = Q I
          ↓                            ↓
       I computed                λ computed
          └──────────────┬──────────────┘
                         ↓
              Gradient assembly (P×)
                         ↓
       ∂J/∂θ_p = -2 Re{ λ† (∂Z/∂θ_p) I }
```

---

## 4. Physical Interpretation of the Adjoint Variable

The adjoint variable $\boldsymbol{\lambda}$ is not merely a mathematical convenience; it has a clear physical meaning. Consider perturbing the right‑hand side $\mathbf{v}$ by a small amount $\delta \mathbf{v}$. The resulting change in the objective is, to first order,

```math
\delta J \approx \Re\{ \boldsymbol{\lambda}^\dagger \delta \mathbf{v} \}.
```

Thus, $\boldsymbol{\lambda}$ acts as a **Green’s function** that maps perturbations in the incident field to changes in the objective. In the context of impedance optimization, $\boldsymbol{\lambda}$ quantifies how sensitive the radiation pattern is to local changes in the surface current.

For a quadratic objective $\mathbf{I}^\dagger \mathbf{Q} \mathbf{I}$, the adjoint source $\mathbf{Q}\mathbf{I}$ is the “desired” current distribution that would maximize $J$. The adjoint equation $\mathbf{Z}^\dagger \boldsymbol{\lambda} = \mathbf{Q}\mathbf{I}$ finds the excitation $\boldsymbol{\lambda}$ that would produce this desired current in a **reciprocal** system (hence the Hermitian transpose).

---

## 5. Implementation in `DiffMoM.jl`

### 5.1 Core Functions (`src/optimization/Adjoint.jl`)

The package provides three essential functions that implement the adjoint pipeline:

- **`compute_objective(I, Q)`** – evaluates $J = \Re\{\mathbf{I}^\dagger \mathbf{Q} \mathbf{I}\}$.
- **`solve_adjoint(Z, Q, I; solver=:direct, preconditioner=nothing, gmres_tol=1e-8, gmres_maxiter=200)`** – solves $\mathbf{Z}^\dagger \boldsymbol{\lambda} = \mathbf{Q}\mathbf{I}$ and returns $\boldsymbol{\lambda}$. Supports both direct (LU) and iterative (GMRES) solvers via the `solver` keyword, with optional preconditioning for GMRES.
- **`gradient_impedance(Mp, I, λ; reactive=false)`** – computes $\partial J/\partial \theta_p$ using $\eqref{eq:adjoint_grad}$ with the appropriate $\partial \mathbf{Z}/\partial \theta_p$.

### 5.2 Code Walkthrough

A minimal working example that reproduces the core of an optimization iteration is:

```julia
using DiffMoM

# Assemble forward operator Z(θ) and right‑hand side v
Z = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
v = assemble_v_plane_wave(...)

# Forward solve
I = solve_forward(Z, v)

# Objective matrix Q (e.g., target angular region)
grid = make_sph_grid(64, 128)
G_mat = radiation_vectors(mesh, rwg, grid, k)
pol_ff = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=deg2rad(10.0))   # broadside cone ±10°
Q = build_Q(G_mat, grid, pol_ff; mask=mask)

# Compute objective value
J = compute_objective(I, Q)

# Adjoint solve
λ = solve_adjoint(Z, Q, I)

# Gradient with respect to impedance parameters
g = gradient_impedance(Mp, I, λ; reactive=true)

println("Objective J = ", J)
println("Gradient norm = ", norm(g))
```

### 5.3 Integration with Optimization

The function `optimize_lbfgs` in `src/optimization/Optimize.jl` wraps this pipeline into a projected L‑BFGS loop. At each iteration, it:

1. Assembles $\mathbf{Z}(\boldsymbol{\theta}^{(k)})$,
2. Calls `solve_forward` and `compute_objective`,
3. Calls `solve_adjoint` and `gradient_impedance`,
4. Updates $\boldsymbol{\theta}$ using the L‑BFGS two‑loop recursion with box‑constraint projection.

All conditioning and preconditioning options (Chapter 5) are respected in both forward and adjoint solves to ensure gradient consistency.

---

## 6. Special Considerations

### 6.1 Adjoint Consistency with Conditioning

When regularization or left preconditioning is applied to the forward solve, the **same conditioned operator must be used in the adjoint solve**. Otherwise, the gradient formula $\eqref{eq:adjoint_grad}$ is invalid. The function `prepare_conditioned_system` (Chapter 5) returns a conditioned matrix $\tilde{\mathbf{Z}}$ that should be passed to both `solve_forward` and `solve_adjoint`.

### 6.2 Complex‑ versus Real‑Valued Parameters

The derivation assumes $\theta_p \in \mathbb{R}$. If parameters were complex (e.g., complex impedance), the gradient formula would involve both $\partial J/\partial \theta_p$ and $\partial J/\partial \theta_p^*$. The package currently supports only real resistive or reactive parameters, for which $\eqref{eq:adjoint_grad}$ is exact.

### 6.3 Multiple Right‑Hand Sides

For multi‑port or multi‑frequency problems with several incident fields $\mathbf{v}_1,\dots,\mathbf{v}_M$, the adjoint method generalizes straightforwardly: each right‑hand side requires its own forward and adjoint solve, increasing the cost linearly with $M$. The gradient is then a sum over contributions from each excitation.

---

## 7. Verification and Debugging

### 7.1 Finite‑Difference Gradient Check

The gold‑standard test for adjoint implementation correctness is comparison with finite differences. For a small random perturbation $\delta \boldsymbol{\theta}$,

```julia
# Adjoint gradient
g_adj = gradient_impedance(Mp, I, λ; reactive=true)

# Finite‑difference approximation (central difference)
ϵ = 1e-8
g_fd = zeros(length(theta))
for p in 1:length(theta)
    θ_plus = copy(theta); θ_plus[p] += ϵ
    θ_minus = copy(theta); θ_minus[p] -= ϵ
    
    Z_plus = assemble_full_Z(Z_efie, Mp, θ_plus; reactive=true)
    Z_minus = assemble_full_Z(Z_efie, Mp, θ_minus; reactive=true)
    
    I_plus = solve_forward(Z_plus, v)
    I_minus = solve_forward(Z_minus, v)
    
    J_plus = compute_objective(I_plus, Q)
    J_minus = compute_objective(I_minus, Q)
    
    g_fd[p] = (J_plus - J_minus) / (2ϵ)
end

rel_err = norm(g_adj - g_fd) / norm(g_fd)
println("Relative error = ", rel_err)   # Should be < 1e-6
```

A relative error below $10^{-6}$ confirms that the adjoint gradient matches the finite‑difference approximation to high precision.

### 7.2 Common Pitfalls

- **Inconsistent conditioning** between forward and adjoint solves → gradient errors $\sim 10^{-2}$–$10^{-1}$.
- **Wrong `reactive` flag** in `gradient_impedance` → gradient sign errors.
- **Missing factor of 2** in $\eqref{eq:adjoint_grad}$ → gradients half as large as they should be.
- **Using $\mathbf{Z}$ instead of $\mathbf{Z}^\dagger$ in adjoint solve** → nonsense gradients.

---

## 8. Extensions and Advanced Topics

### 8.1 Ratio Objectives (Two Adjoint Solves)

For ratio objectives $J = f/g$ (Chapter 3), the gradient requires derivatives of both numerator $f$ and denominator $g$. Each derivative term is of the form $\eqref{eq:adjoint_grad}$ with its own adjoint variable, leading to **two adjoint solves** per iteration. This is still independent of $P$ and vastly more efficient than finite differences.

### 8.2 Second‑Order Sensitivities (Hessian‑Vector Products)

The adjoint framework can be extended to compute Hessian‑vector products $\mathbf{H} \mathbf{d}$ via a second‑order adjoint solve, enabling Newton‑type optimization or uncertainty quantification. This is not currently implemented in the package but could be added following the same pattern.

### 8.3 Shape Derivatives

While this package focuses on impedance parameters, the adjoint method also applies to shape optimization (moving mesh vertices). The main additional complexity is computing $\partial \mathbf{Z}/\partial \mathbf{v}_i$, which requires differentiating the singular EFIE kernel with respect to vertex coordinates—a topic beyond the current scope.

---

## 9. Summary of Key Formulas

| Quantity | Formula |
|----------|---------|
| Forward equation | $\mathbf{Z}(\boldsymbol{\theta})\mathbf{I} = \mathbf{v}$ |
| Quadratic objective | $J = \mathbf{I}^\dagger \mathbf{Q} \mathbf{I}$ |
| Adjoint equation | $\mathbf{Z}^\dagger \boldsymbol{\lambda} = \mathbf{Q}\mathbf{I}$ |
| Gradient (general) | $\displaystyle \frac{\partial J}{\partial \theta_p} = -2\Re\{\boldsymbol{\lambda}^\dagger (\partial \mathbf{Z}/\partial \theta_p) \mathbf{I}\}$ |
| Gradient (resistive) | $\partial J/\partial \theta_p = +2\Re\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$ |
| Gradient (reactive) | $\partial J/\partial \theta_p = -2\Im\{\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}\}$ |

---

## 10. Code Mapping

- **`src/optimization/Adjoint.jl`** – Core adjoint functions: `compute_objective`, `solve_adjoint`, `gradient_impedance`.
- **`src/optimization/Optimize.jl`** – Optimization loops that call the adjoint pipeline.
- **`src/solver/Solve.jl`** – Forward solve (`solve_forward`) and conditioned system preparation.
- **`src/solver/Solve.jl`** – Assembly of $\mathbf{Z}(\boldsymbol{\theta})$ via `assemble_full_Z`.
- **`src/assembly/Impedance.jl`** – Assembly of impedance contribution `assemble_Z_impedance` and patch mass matrices $\mathbf{M}_p$.
- **`src/assembly/EFIE.jl`** – Assembly of $\mathbf{Z}_{\mathrm{EFIE}}$.
- **`test/runtests.jl`** – Verification script comparing adjoint and finite‑difference gradients.

---

## 11. Exercises

### 11.1 Conceptual Questions

1. **Physical interpretation**: Explain in your own words what the adjoint variable $\boldsymbol{\lambda}$ represents physically. How would you measure $\boldsymbol{\lambda}$ in a thought experiment?
2. **Cost scaling**: Suppose you have $P=500$ impedance patches and $M=3$ different incident angles. How many linear solves per iteration would be required using (a) finite differences (central), (b) the adjoint method? Compare the ratios.
3. **Adjoint consistency**: Why must the same conditioned operator be used in forward and adjoint solves? What would happen if you used $\mathbf{Z}_{\mathrm{raw}}$ in the adjoint solve but $\mathbf{Z}_{\mathrm{conditioned}}$ in the forward solve?

### 11.2 Derivation Tasks

1. **Derive the gradient formula**: Starting from $\eqref{eq:forward_deriv}$ and $\eqref{eq:adjoint_eq}$, fill in the algebraic steps that lead to $\eqref{eq:adjoint_grad}$. Pay special attention to the factor of 2 and the real‑part operator.
2. **Wirtinger calculus**: Show that for a real‑valued function $J = \mathbf{I}^\dagger \mathbf{Q} \mathbf{I}$ with Hermitian $\mathbf{Q}$, the Wirtinger derivatives are $\partial J/\partial \mathbf{I}^* = \mathbf{Q}\mathbf{I}$ and $\partial J/\partial \mathbf{I} = (\mathbf{Q}\mathbf{I})^*$.

### 11.3 Coding Exercises

1. **Basic verification**: Write a script that reproduces the minimal code walkthrough from Section 5.2 for a small rectangular plate. Verify that `length(g)` equals the number of patches.
2. **Gradient check**: Implement the finite‑difference gradient check from Section 7.1 for both resistive (`reactive=false`) and reactive (`reactive=true`) parameterizations. Confirm that the relative error is below $10^{-6}$.
3. **Cost comparison**: For $P = 10, 50, 100$ (adjust mesh size accordingly), measure the wall‑clock time to compute gradients using (a) the adjoint method and (b) central finite differences. Plot the time ratio vs. $P$.

### 11.4 Advanced Challenges

1. **Custom objective**: Implement a custom objective function that penalizes cross‑polarization: $J = \mathbf{I}^\dagger \mathbf{Q}_{\mathrm{co}} \mathbf{I} - \alpha \mathbf{I}^\dagger \mathbf{Q}_{\mathrm{cross}} \mathbf{I}$, where $\mathbf{Q}_{\mathrm{co}}$ and $\mathbf{Q}_{\mathrm{cross}}$ are co‑ and cross‑polarization power matrices. Derive the corresponding adjoint source term and verify the gradient with finite differences.
2. **Memory‑efficient gradient**: The current `gradient_impedance` loops over patches, computing $\boldsymbol{\lambda}^\dagger \mathbf{M}_p \mathbf{I}$ for each $p$. For large $P$, this can be memory‑intensive if all $\mathbf{M}_p$ are stored densely. Design a batched or matrix‑free version that computes the gradient without storing all $\mathbf{M}_p$ simultaneously.

---

## 12. Chapter Checklist

After studying this chapter, you should be able to:

- [ ] **Derive** the adjoint equation from the forward EFIE–MoM system and the chain rule.
- [ ] **Explain** why the adjoint method scales as $O(1)$ in the number of design parameters $P$, while finite differences scale as $O(P)$.
- [ ] **Implement** the adjoint gradient formula using `compute_objective`, `solve_adjoint`, and `gradient_impedance`.
- [ ] **Verify** gradient correctness with a finite‑difference check to tolerance $<10^{-6}$.
- [ ] **Ensure** adjoint consistency when regularization or preconditioning is used.
- [ ] **Interpret** the physical meaning of the adjoint variable $\boldsymbol{\lambda}$ as a sensitivity to incident‑field perturbations.
- [ ] **Extend** the adjoint method to ratio objectives (two adjoint solves) and understand why this is still efficient.

If you can confidently check all items, you have mastered the adjoint method as implemented in `DiffMoM.jl` and are ready to proceed to Chapter 2 (Impedance Sensitivities).

---

## 13. Further Reading

1. **Classical adjoint method references**:
   - Nikolova, N. K., & Bakr, M. H. (2005). *Adjoint techniques for sensitivity analysis in high‑frequency structure CAD*. IEEE Transactions on Microwave Theory and Techniques, 53(1), 30–41.
   - Georgieva, N. K., & Glavic, S. (2002). *Microwave circuit design by sensitivity analysis*. IEEE Transactions on Microwave Theory and Techniques, 50(9), 2105–2113.

2. **Adjoint methods in electromagnetics**:
   - Toivanen, J. I., et al. (2010). *Adjoint variable method for time‑domain integral equations with full‑wave Maxwell’s equations*. IEEE Transactions on Antennas and Propagation, 58(1), 87–95.
   - Hassan, E., et al. (2014). *Topology optimization of metallic antennas*. IEEE Transactions on Antennas and Propagation, 62(5), 2488–2500.

3. **Differentiable programming**:
   - Baydin, A. G., et al. (2018). *Automatic differentiation in machine learning: a survey*. Journal of Machine Learning Research, 18(153), 1–43.
   - Innes, M., et al. (2019). *A differentiable programming system to bridge machine learning and scientific computing*. arXiv:1907.07587.

4. **Julia resources**:
   - `LinearAlgebra` – Standard library for linear solves and adjoint operations.
   - `Zygote.jl` – Automatic differentiation library that could be integrated with the adjoint pipeline.

---

*Next: Chapter 2, "Impedance Sensitivities," details why impedance parameters yield especially simple derivative blocks $\partial \mathbf{Z}/\partial \theta_p$ and how these blocks are assembled from precomputed patch mass matrices.*