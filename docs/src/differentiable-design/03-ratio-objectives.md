# Ratio Objectives

## Purpose

This chapter explains why ratio objectives of the form $J = f/g$—where $f$ is power radiated into a target angular region and $g$ is total radiated power—are preferred for beam‑steering and directivity optimization. Ratio objectives are scale‑invariant, avoid trivial solutions (e.g., simply increasing input power), and naturally trade off beam concentration against sidelobe levels. The chapter derives the quotient‑rule gradient formula and explains why the package uses **two separate adjoint solves** for numerical stability, avoiding the cancellation errors that plague a single‑adjoint approach near convergence.

---

## Learning Goals

After this chapter, you should be able to:

1. Write the ratio objective $J = f/g$ in terms of Hermitian positive‑semidefinite matrices $\mathbf{Q}_t$ and $\mathbf{Q}_{\mathrm{tot}}$.
2. Derive the quotient‑rule gradient $\partial J/\partial \theta_p = (g \partial f/\partial \theta_p - f \partial g/\partial \theta_p)/g^2$.
3. Explain why two adjoint solves are numerically more stable than a single adjoint solve with an effective matrix $\mathbf{Q}_t - J\mathbf{Q}_{\mathrm{tot}}$.
4. Use `optimize_directivity` for beam‑steering design with box constraints.
5. Interpret optimization traces and adjust target‑region parameters to control sidelobes.

---

## 1. Motivation for Ratio Objectives

### 1.1 Limitations of Absolute Power Objectives

Consider optimizing the power radiated into a target angular region $\mathcal{D}_t$:

```math
J_{\mathrm{abs}}(\boldsymbol{\theta}) = \mathbf{I}^\dagger \mathbf{Q}_t \mathbf{I}.
```

Maximizing $J_{\mathrm{abs}}$ alone can lead to trivial solutions: simply **increasing the input power** (e.g., by reducing impedance everywhere) raises $J_{\mathrm{abs}}$ without improving directivity. Moreover, $J_{\mathrm{abs}}$ depends on the absolute scale of the incident field, making comparisons across different excitations difficult.

### 1.2 Directivity as a Ratio

The directivity toward a direction $\hat{\mathbf{r}}_0$ is defined as

```math
D(\hat{\mathbf{r}}_0) = \frac{4\pi\,U(\hat{\mathbf{r}}_0)}{P_{\mathrm{rad}}},
```

where $U$ is radiation intensity and $P_{\mathrm{rad}}$ is total radiated power. This is intrinsically a **ratio** of target power to total power. Generalizing to an angular region $\mathcal{D}_t$, we define

```math
J(\boldsymbol{\theta}) = \frac{f(\boldsymbol{\theta})}{g(\boldsymbol{\theta})},
\qquad
f(\boldsymbol{\theta}) = \mathbf{I}^\dagger \mathbf{Q}_t \mathbf{I},\;
g(\boldsymbol{\theta}) = \mathbf{I}^\dagger \mathbf{Q}_{\mathrm{tot}} \mathbf{I},
\label{eq:ratio_obj}
```

where $\mathbf{Q}_t$ and $\mathbf{Q}_{\mathrm{tot}}$ are Hermitian positive‑semidefinite matrices that project onto the target region and the whole radiation sphere (or a selected polarization channel), respectively.

### 1.3 Benefits of Ratio Objectives

1. **Scale invariance**: Multiplying $\mathbf{I}$ by a constant leaves $J$ unchanged. This eliminates trivial scaling solutions.
2. **Physical meaning**: $J$ approximates the fraction of total radiated power that goes into the target region—a direct measure of beam concentration.
3. **Automatic sidelobe suppression**: Maximizing $J$ implicitly penalizes power radiated outside $\mathcal{D}_t$, because increasing $g$ (total power) decreases $J$.
4. **Robustness to incident‑field magnitude**: $J$ is independent of the incident‑field amplitude, facilitating comparisons across different excitation scenarios.

---

## 2. Mathematical Formulation

### 2.1 Matrix Definitions

Let $\mathcal{D}_t \subset \mathbb{S}^2$ be the target angular region (e.g., a conical sector around a desired steering angle) and $\mathbf{p}(\hat{\mathbf{r}})$ a unit polarization vector. The target power matrix $\mathbf{Q}_t$ has entries (see Chapter 3 of Part II)

```math
[\mathbf{Q}_t]_{mn}
=
\int_{\mathcal{D}_t}
\bigl[ \mathbf{p}^\dagger(\hat{\mathbf{r}}) \mathbf{g}_m(\hat{\mathbf{r}}) \bigr]^*
\bigl[ \mathbf{p}^\dagger(\hat{\mathbf{r}}) \mathbf{g}_n(\hat{\mathbf{r}}) \bigr]
\, d\Omega,
\label{eq:Q_target}
```

where $\mathbf{g}_n$ are the radiation patterns of the RWG basis functions. The total‑power matrix $\mathbf{Q}_{\mathrm{tot}}$ is defined similarly but integrated over the entire sphere (or over a prescribed solid angle that defines the "total" channel, e.g., the forward hemisphere).

By construction, $\mathbf{Q}_t, \mathbf{Q}_{\mathrm{tot}} \succeq 0$ and $\mathbf{Q}_t \preceq \mathbf{Q}_{\mathrm{tot}}$ (element‑wise) if $\mathcal{D}_t$ is a subset of the total region.

### 2.2 Gradient via the Quotient Rule

Differentiate $J = f/g$ using the quotient rule:

```math
\frac{\partial J}{\partial \theta_p}
=
\frac{
g \frac{\partial f}{\partial \theta_p}
- f \frac{\partial g}{\partial \theta_p}
}{g^2},
\qquad p = 1,\dots,P.
\label{eq:quotient_rule}
```

Both $f$ and $g$ are quadratic forms, so their derivatives follow the adjoint formula derived in Chapter 1:

```math
\frac{\partial f}{\partial \theta_p}
=
-2\,\Re\!\left\{
\boldsymbol{\lambda}_f^\dagger
\left(
\frac{\partial \mathbf{Z}}{\partial \theta_p}
\right)
\mathbf{I}
\right\},
\qquad
\frac{\partial g}{\partial \theta_p}
=
-2\,\Re\!\left\{
\boldsymbol{\lambda}_g^\dagger
\left(
\frac{\partial \mathbf{Z}}{\partial \theta_p}
\right)
\mathbf{I}
\right\},
\label{eq:deriv_fg}
```

where $\boldsymbol{\lambda}_f$ and $\boldsymbol{\lambda}_g$ satisfy the respective adjoint equations

```math
\mathbf{Z}^\dagger \boldsymbol{\lambda}_f = \mathbf{Q}_t \mathbf{I},
\qquad
\mathbf{Z}^\dagger \boldsymbol{\lambda}_g = \mathbf{Q}_{\mathrm{tot}} \mathbf{I}.
\label{eq:adjoints_fg}
```

Substituting $\eqref{eq:deriv_fg}$ into $\eqref{eq:quotient_rule}$ yields the complete ratio gradient.

### 2.3 Efficient Gradient Assembly

Because $\partial \mathbf{Z}/\partial \theta_p = -\mathbf{M}_p$ (resistive) or $-i\mathbf{M}_p$ (reactive), the contractions in $\eqref{eq:deriv_fg}$ reduce to inner products involving $\mathbf{M}_p$. Let's derive these expressions step by step.

First, recall the adjoint gradient formula from Chapter 1:
```math
\frac{\partial f}{\partial \theta_p} = -2\,\Re\!\left\{
\boldsymbol{\lambda}_f^\dagger
\left(
\frac{\partial \mathbf{Z}}{\partial \theta_p}
\right)
\mathbf{I}
\right\}.
```

**Resistive case** ($\partial \mathbf{Z}/\partial \theta_p = -\mathbf{M}_p$):
```math
\frac{\partial f}{\partial \theta_p}
= -2\,\Re\!\left\{
\boldsymbol{\lambda}_f^\dagger (-\mathbf{M}_p) \mathbf{I}
\right\}
= +2\,\Re\!\left\{
\boldsymbol{\lambda}_f^\dagger \mathbf{M}_p \mathbf{I}
\right\}.
```

**Reactive case** ($\partial \mathbf{Z}/\partial \theta_p = -i\mathbf{M}_p$):
```math
\frac{\partial f}{\partial \theta_p}
= -2\,\Re\!\left\{
\boldsymbol{\lambda}_f^\dagger (-i\mathbf{M}_p) \mathbf{I}
\right\}
= +2\,\Re\!\left\{
i\,\boldsymbol{\lambda}_f^\dagger \mathbf{M}_p \mathbf{I}
\right\}
= -2\,\Im\!\left\{
\boldsymbol{\lambda}_f^\dagger \mathbf{M}_p \mathbf{I}
\right\},
```
where we used the identity $\Re\{i z\} = -\Im\{z\}$. The same steps apply to $\partial g/\partial \theta_p$ with $\boldsymbol{\lambda}_g$.

Define the scalar overlaps

```math
l_p^{(f)} = \boldsymbol{\lambda}_f^\dagger \mathbf{M}_p \mathbf{I},
\qquad
l_p^{(g)} = \boldsymbol{\lambda}_g^\dagger \mathbf{M}_p \mathbf{I}.
```

Then we can summarize:

**Resistive sheets**:
```math
\frac{\partial f}{\partial \theta_p} = +2\,\Re\{l_p^{(f)}\},\quad
\frac{\partial g}{\partial \theta_p} = +2\,\Re\{l_p^{(g)}\}.
```

**Reactive sheets**:
```math
\frac{\partial f}{\partial \theta_p} = -2\,\Im\{l_p^{(f)}\},\quad
\frac{\partial g}{\partial \theta_p} = -2\,\Im\{l_p^{(g)}\}.
```

Now substitute these into the quotient rule $\eqref{eq:quotient_rule}$:

**Resistive case**:
```math
\frac{\partial J}{\partial \theta_p}
=
\frac{1}{g^2}
\Bigl(
g \bigl[+2\Re\{l_p^{(f)}\}\bigr]
- f \bigl[+2\Re\{l_p^{(g)}\}\bigr]
\Bigr)
=
\frac{2}{g^2}
\Bigl(
g \,\Re\{l_p^{(f)}\} - f \,\Re\{l_p^{(g)}\}
\Bigr).
```

**Reactive case**:
```math
\frac{\partial J}{\partial \theta_p}
=
\frac{1}{g^2}
\Bigl(
g \bigl[-2\Im\{l_p^{(f)}\}\bigr]
- f \bigl[-2\Im\{l_p^{(g)}\}\bigr]
\Bigr)
=
-\frac{2}{g^2}
\Bigl(
g \,\Im\{l_p^{(f)}\} - f \,\Im\{l_p^{(g)}\}
\Bigr).
```

These formulas are implemented in `optimize_directivity` (Section 5).

---

## 3. Why Two Adjoint Solves Are Necessary

### 3.1 The Tempting Single‑Adjoint Approach

At first glance, one might try to combine the two adjoint systems into a single solve. Define an "effective" matrix

```math
\mathbf{Q}_{\mathrm{eff}} = \mathbf{Q}_t - J \mathbf{Q}_{\mathrm{tot}}.
```

Then, using the linearity of the adjoint equation, it appears that

```math
\mathbf{Z}^\dagger \boldsymbol{\lambda}_{\mathrm{eff}} = \mathbf{Q}_{\mathrm{eff}} \mathbf{I}
```

would yield a gradient satisfying

```math
\frac{\partial J}{\partial \theta_p}
=
-2\,\Re\!\left\{
\boldsymbol{\lambda}_{\mathrm{eff}}^\dagger
\left(
\frac{\partial \mathbf{Z}}{\partial \theta_p}
\right)
\mathbf{I}
\right\}.
```

This approach would require only **one** adjoint solve per iteration instead of two.

### 3.2 Numerical Cancellation Problem

The issue is that near convergence, $J \approx f/g$, so

```math
\mathbf{Q}_{\mathrm{eff}} \mathbf{I}
\approx
\mathbf{Q}_t \mathbf{I} - \frac{f}{g} \mathbf{Q}_{\mathrm{tot}} \mathbf{I}.
```

But $\mathbf{Q}_t \mathbf{I}$ and $(f/g) \mathbf{Q}_{\mathrm{tot}} \mathbf{I}$ can be nearly equal in magnitude, leading to **catastrophic cancellation** when forming the right‑hand side. This cancellation amplifies round‑off errors, making the gradient noisy and potentially destabilizing the optimization.

### 3.3 Two‑Solve Approach Avoids Cancellation

By solving two separate adjoint systems, we compute $\boldsymbol{\lambda}_f$ and $\boldsymbol{\lambda}_g$ from well‑conditioned right‑hand sides $\mathbf{Q}_t \mathbf{I}$ and $\mathbf{Q}_{\mathrm{tot}} \mathbf{I}$, each of which is numerically stable. The cancellation occurs only in the final linear combination

```math
\frac{\partial J}{\partial \theta_p}
\propto
g \frac{\partial f}{\partial \theta_p} - f \frac{\partial g}{\partial \theta_p},
```

which involves **scalar** numbers $f$ and $g$ rather than large vectors. Scalar cancellation is much less harmful and can be controlled with standard floating‑point precautions.

### 3.4 Cost‑Benefit Trade‑Off

The two‑adjoint approach doubles the linear‑solve cost per iteration compared to a single quadratic objective. However, this is still **independent of $P$** and vastly cheaper than finite differences. The added stability is well worth the extra solve, especially for challenging beam‑steering problems where high directivity requires precise gradient information.

---

## 4. Implementation in `optimize_directivity`

### 4.1 Function Signature

The main ratio‑optimization entry point is

```julia
function optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta0;
    reactive=false,
    maxiter=100,
    tol=1e-6,
    m_lbfgs=10,
    alpha0=0.01,
    lb=nothing,
    ub=nothing,
    regularization_alpha=0.0,
    regularization_R=nothing,
    preconditioner_M=nothing,
    preconditioning=:off,
    auto_precondition_n_threshold=256,
    iterative_solver=false,
    auto_precondition_eps_rel=1e-6,
    solver=:direct,
    nf_preconditioner=nothing,
    gmres_tol=1e-8,
    gmres_maxiter=200,
    verbose=true
)
```

**Arguments**:
- `Z_efie`: The EFIE matrix $\mathbf{Z}_{\mathrm{EFIE}}$ (independent of $\boldsymbol{\theta}$).
- `Mp`: Precomputed patch mass matrices.
- `v`: Right‑hand side vector (tested incident field).
- `Q_target`, `Q_total`: Target and total power matrices $\mathbf{Q}_t$, $\mathbf{Q}_{\mathrm{tot}}$.
- `theta0`: Initial design vector $\boldsymbol{\theta}^{(0)}$.
- `reactive`: `false` for resistive design (default), `true` for reactive.
- `maxiter`: Maximum iterations (default 100).
- `tol`: Gradient‑norm tolerance (default 1e-6).
- `m_lbfgs`: L‑BFGS memory size (default 10).
- `alpha0`: Initial inverse‑Hessian scaling for L‑BFGS (default 0.01). Used as the diagonal scaling `gamma` in the two‑loop recursion before any curvature pairs are available.
- `lb`, `ub`: Box constraints (lower and upper bounds); `nothing` for unconstrained (default).
- `regularization_alpha`: Tikhonov regularization parameter (default 0.0). Non‑zero values add $\alpha \mathbf{R}$ to the system matrix.
- `regularization_R`: Custom regularization matrix; if `nothing` (default), a mass‑based regularizer is used when `regularization_alpha > 0`.
- `preconditioner_M`: Explicit preconditioner matrix for parameter‑space conditioning (default `nothing`).
- `preconditioning`: `:off` (default), `:on`, or `:auto`. Controls parameter‑space preconditioning (see Chapter 5).
- `auto_precondition_n_threshold`: Patch‑count threshold for auto‑preconditioning (default 256).
- `iterative_solver`: Boolean flag that influences auto‑preconditioning decisions in `select_preconditioner` (default `false`). Does **not** switch the solver; use `solver` for that.
- `auto_precondition_eps_rel`: Relative tolerance for auto‑preconditioning (default 1e-6).
- `solver`: Solver dispatch -- `:direct` (LU, default) or `:gmres`.
- `nf_preconditioner`: Near‑field preconditioner for GMRES (default `nothing`).
- `gmres_tol`: GMRES convergence tolerance (default 1e-8).
- `gmres_maxiter`: Maximum GMRES iterations (default 200).
- `verbose`: Print progress information.

**Returns**: A tuple `(theta_opt, trace)`, where `theta_opt` is the optimized parameter vector and `trace` is a `Vector{NamedTuple{(:iter, :J, :gnorm)}}` recording iteration number, objective value, and gradient norm at each iteration.

### 4.2 Internal Steps per Iteration

For each L‑BFGS iteration $k$, `optimize_directivity` performs:

1. **Assemble** $\mathbf{Z}^{(k)} = \mathbf{Z}_{\mathrm{EFIE}} + \mathbf{Z}_{\mathrm{imp}}(\boldsymbol{\theta}^{(k)})$.
2. **Forward solve** $\mathbf{Z}^{(k)} \mathbf{I}^{(k)} = \mathbf{v}$.
3. **Compute** $f^{(k)} = (\mathbf{I}^{(k)})^\dagger \mathbf{Q}_t \mathbf{I}^{(k)}$, $g^{(k)} = (\mathbf{I}^{(k)})^\dagger \mathbf{Q}_{\mathrm{tot}} \mathbf{I}^{(k)}$, $J^{(k)} = f^{(k)}/g^{(k)}$.
4. **Adjoint solves** $\mathbf{Z}^{(k)\dagger} \boldsymbol{\lambda}_f = \mathbf{Q}_t \mathbf{I}^{(k)}$ and $\mathbf{Z}^{(k)\dagger} \boldsymbol{\lambda}_g = \mathbf{Q}_{\mathrm{tot}} \mathbf{I}^{(k)}$.
5. **Gradient assembly** using $\eqref{eq:quotient_rule}$ and the formulas in Section 2.3.
6. **L‑BFGS update** with projection onto $[\mathtt{lb}, \mathtt{ub}]$.

### 4.3 Box Constraints and Projection

The optimizers enforce simple bounds $\theta_p \in [\mathtt{lb}_p, \mathtt{ub}_p]$ via **projected L‑BFGS**: after each L‑BFGS step, parameters that violate bounds are clipped to the nearest bound. This is crucial for physical realizability (e.g., passive sheets require $\theta_p \ge 0$ for resistive sheets) and to prevent runaway numerical values.

---

## 5. Practical Usage Example

### 5.1 Setting Up a Beam‑Steering Problem

The following script sets up and solves a reactive beam‑steering problem for a $4\lambda \times 4\lambda$ plate:

```julia
using DiffMoM

# ------------------------------
# Geometry and discretization
# ------------------------------
λ = 1.0
k = 2π / λ
mesh = make_rect_plate(4λ, 4λ, 40, 40)   # 40×40 triangles
rwg = build_rwg(mesh)
N = rwg.nedges
println("Number of unknowns N = ", N)

# ------------------------------
# Patch partition (one patch per triangle for fine control)
# ------------------------------
ntri = ntriangles(mesh)
partition = PatchPartition(collect(1:ntri), ntri)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
P = length(Mp)
println("Number of patches P = ", P)

# ------------------------------
# Incident plane wave (broadside)
# ------------------------------
E0 = 1.0
k_vec = Vec3(0.0, 0.0, -k)
pol = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)

# ------------------------------
# EFIE matrix (parameter‑independent part)
# ------------------------------
Z_efie = assemble_Z_efie(mesh, rwg, k)

# ------------------------------
# Q matrices: target cone ±10° around steering angle θ_steer = 30°
# ------------------------------
θ_steer = 30.0   # degrees
cone_halfwidth = 10.0   # degrees
grid = make_sph_grid(64, 128)
G = radiation_vectors(mesh, rwg, grid, k; quad_order=3)
pol_ff = pol_linear_x(grid)
θ0 = deg2rad(θ_steer)
Δθ = deg2rad(cone_halfwidth)
mask_target = abs.(grid.theta .- θ0) .<= Δθ
Q_target = build_Q(G, grid, pol_ff; mask=mask_target)
Q_total = build_Q(G, grid, pol_ff)   # whole sphere

# ------------------------------
# Initial parameters (zero reactance → PEC reference)
# ------------------------------
theta0 = zeros(P)

# ------------------------------
# Run optimization
# ------------------------------
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta0;
    reactive=true,
    maxiter=100,
    tol=1e-6,
    lb=fill(-500.0, P),   # capacitive/inductive limits
    ub=fill( 500.0, P),
    verbose=true
)

# ------------------------------
# Post‑processing
# ------------------------------
println("Final directivity ratio J = ", trace[end].J)
println("Final gradient norm = ", trace[end].gnorm)
```

### 5.2 Interpreting Results

- **Optimization trace**: The `trace` is a `Vector{NamedTuple{(:iter, :J, :gnorm)}}`. Access objective values as `[t.J for t in trace]` and gradient norms as `[t.gnorm for t in trace]`. Plotting `J` vs. iteration shows whether convergence is monotonic or oscillatory.
- **Final pattern**: Re‑assemble $\mathbf{Z}$ with `theta_opt`, solve for $\mathbf{I}$, and compute the far‑field pattern to verify beam steering and sidelobe levels.
- **Parameter distribution**: Visualize `theta_opt` on the mesh to see the resulting reactance distribution (periodic phase‑gradient patterns are typical for beam steering).

---

## 6. Numerical Stability Considerations

### 6.1 Regularization for Ill‑Conditioned Systems

When the EFIE matrix is ill‑conditioned (low frequencies, fine meshes), both forward and adjoint solves benefit from the regularization and preconditioning techniques described in Chapter 5. The same conditioned operator must be used for all three solves (forward + two adjoint) to maintain gradient consistency.

### 6.2 Avoiding Division by Zero

The denominator $g$ is total radiated power, which is positive for any non‑zero current. The code computes the ratio directly as `J_ratio = f_val / g_val` without an explicit epsilon guard. In practice, $g > 0$ whenever the current vector is non‑zero (which is guaranteed after a successful forward solve). If numerical issues are suspected (e.g., extremely lossy sheets), monitor $g$ in the optimization trace and consider adding regularization.

### 6.3 Scaling of $f$ and $g$

The implementation computes $f$ and $g$ directly as quadratic forms (`real(dot(I, Q * I))`) and forms the ratio $J = f/g$ without any additional scaling. For typical metasurface problems, the magnitudes of $f$ and $g$ are well within double‑precision range. If very large problems lead to overflow concerns, the user can pre‑scale the $\mathbf{Q}$ matrices externally before passing them to the optimizer.

---

## 7. Advanced Topics

### 7.1 Multi‑Objective Ratio Goals

More complex design targets can be expressed as weighted sums of ratios:

```math
J_{\text{multi}} = \alpha \frac{f_1}{g_1} + \beta \frac{f_2}{g_2},
```

where $f_1, g_1$ might correspond to a main‑beam region and $f_2, g_2$ to a sidelobe region (with negative $\beta$ to suppress sidelobes). Each ratio requires its own pair of adjoint solves, increasing the cost linearly with the number of ratio terms.

### 7.2 Constrained Ratio Optimization

Instead of a pure ratio, one may want to maximize $f$ subject to $g \le G_{\max}$ (total‑power constraint). This can be solved via the method of Lagrange multipliers, which again leads to two adjoint solves per iteration.

### 7.3 Relation to Generalized Eigenvalue Problems

Maximizing $J = f/g$ is equivalent to maximizing the Rayleigh quotient

```math
\frac{\mathbf{I}^\dagger \mathbf{Q}_t \mathbf{I}}{\mathbf{I}^\dagger \mathbf{Q}_{\mathrm{tot}} \mathbf{I}},
```

which is a generalized eigenvalue problem. The gradient‑based approach iteratively improves the current distribution $\mathbf{I}$ while respecting the EFIE constraint, effectively solving a constrained generalized eigenvalue problem.

---

## 8. Summary of Key Formulas

| Quantity | Expression |
|----------|------------|
| Ratio objective | $\displaystyle J = \frac{\mathbf{I}^\dagger \mathbf{Q}_t \mathbf{I}}{\mathbf{I}^\dagger \mathbf{Q}_{\mathrm{tot}} \mathbf{I}}$ |
| Quotient‑rule gradient | $\displaystyle \frac{\partial J}{\partial \theta_p} = \frac{g \partial f/\partial \theta_p - f \partial g/\partial \theta_p}{g^2}$ |
| Adjoint equations | $\mathbf{Z}^\dagger \boldsymbol{\lambda}_f = \mathbf{Q}_t \mathbf{I}$, $\quad \mathbf{Z}^\dagger \boldsymbol{\lambda}_g = \mathbf{Q}_{\mathrm{tot}} \mathbf{I}$ |
| Gradient (resistive) | $\displaystyle \frac{\partial J}{\partial \theta_p} = \frac{2}{g^2}\bigl(g \Re\{l_p^{(f)}\} - f \Re\{l_p^{(g)}\}\bigr)$ |
| Gradient (reactive) | $\displaystyle \frac{\partial J}{\partial \theta_p} = -\frac{2}{g^2}\bigl(g \Im\{l_p^{(f)}\} - f \Im\{l_p^{(g)}\}\bigr)$ |

---

## 9. Code Mapping

- **`src/optimization/Optimize.jl`** – Ratio optimizer `optimize_directivity`.
- **`src/optimization/Adjoint.jl`** – Adjoint solves `solve_adjoint` and gradient assembly utilities.
- **`src/optimization/QMatrix.jl`** – Construction of $\mathbf{Q}_t$ and $\mathbf{Q}_{\mathrm{tot}}$ matrices (`build_Q`, masks via `cap_mask` or custom logical masks).
- **`src/solver/Solve.jl`** – Forward solve and conditioned system preparation.
- **`examples/03_beamsteering_physical_unitcell.jl`** – Complete beam‑steering example.
- **`test/runtests.jl`** – Verification script for ratio‑objective gradients.

---

## 10. Exercises

### 10.1 Conceptual Questions

1. **Scale invariance**: Prove that the ratio objective $J = f/g$ is invariant under scaling $\mathbf{I} \to \alpha \mathbf{I}$ for any non‑zero complex scalar $\alpha$. Why is this property desirable for beam‑steering optimization?
2. **Cancellation analysis**: Suppose $f = 1.0$ and $g = 2.0$, so $J = 0.5$. If $\mathbf{Q}_t \mathbf{I}$ and $\mathbf{Q}_{\mathrm{tot}} \mathbf{I}$ are computed with relative error $10^{-8}$, estimate the error in $\mathbf{Q}_{\mathrm{eff}} \mathbf{I} = \mathbf{Q}_t \mathbf{I} - J \mathbf{Q}_{\mathrm{tot}} \mathbf{I}$. Compare with the error in the two‑solve approach.
3. **Physical interpretation**: What does a negative gradient component $\partial J/\partial \theta_p$ indicate for a reactive sheet? Should you increase or decrease $\theta_p$ to improve directivity?

### 10.2 Derivation Tasks

1. **Derive the quotient‑rule gradient**: Start from $J = f/g$ and apply the chain rule together with the adjoint gradient formula for $f$ and $g$. Obtain the final expressions for resistive and reactive sheets.
2. **Generalized eigenvalue connection**: Show that maximizing $J = \mathbf{I}^\dagger \mathbf{Q}_t \mathbf{I} / \mathbf{I}^\dagger \mathbf{Q}_{\mathrm{tot}} \mathbf{I}$ subject to $\mathbf{Z} \mathbf{I} = \mathbf{v}$ is equivalent to solving a constrained generalized eigenvalue problem. What is the Lagrangian?

### 10.3 Coding Exercises

1. **Basic ratio optimization**: Run the example from Section 5.1 for a smaller plate ($2\lambda \times 2\lambda$) and visualize the optimization trace. Does the directivity ratio converge monotonically?
2. **Target‑region sweep**: Vary the target cone half‑width (5°, 10°, 20°) and run optimizations for each. Compare the final directivity values and far‑field patterns. Explain the trade‑off between beam width and peak directivity.
3. **Gradient verification**: Write a finite‑difference check for the ratio gradient. Perturb each parameter by $\epsilon = 10^{-8}$ and compare with the adjoint gradient from `optimize_directivity`. Ensure relative error $< 10^{-5}$.

### 10.4 Advanced Challenges

1. **Sidelobe suppression**: Design a ratio objective that maximizes power in a main‑beam cone while suppressing power in a sidelobe region. Implement this as $J = f_{\mathrm{main}}/g_{\mathrm{total}} - \beta f_{\mathrm{sidelobe}}/g_{\mathrm{total}}$ and optimize for $\beta = 0.1, 0.5, 1.0$. Compare the resulting patterns.
2. **Multi‑frequency ratio optimization**: Extend the ratio objective to average performance over a band of frequencies: $J_{\mathrm{band}} = \frac{1}{K} \sum_{k=1}^K J(\omega_k)$. Implement frequency‑looping inside `optimize_directivity` and test on a small bandwidth (e.g., $0.95f_0$ to $1.05f_0$).

---

## 11. Chapter Checklist

After studying this chapter, you should be able to:

- [ ] **Write** the ratio objective $J = f/g$ in terms of target‑ and total‑power matrices $\mathbf{Q}_t$, $\mathbf{Q}_{\mathrm{tot}}$.
- [ ] **Derive** the quotient‑rule gradient formula and explain why it requires derivatives of both $f$ and $g$.
- [ ] **Explain** why two separate adjoint solves are numerically more stable than a single adjoint solve with $\mathbf{Q}_{\mathrm{eff}} = \mathbf{Q}_t - J\mathbf{Q}_{\mathrm{tot}}$.
- [ ] **Use** `optimize_directivity` with appropriate box constraints and conditioning options.
- [ ] **Interpret** optimization traces and adjust target‑region parameters to control beam width and sidelobe levels.
- [ ] **Verify** ratio‑objective gradients with finite‑difference checks.

If you can confidently check all items, you have mastered ratio‑objective optimization in `DiffMoM.jl` and are ready to proceed to Chapter 4 (Optimization Workflow).

---

## 12. Further Reading

1. **Directivity and ratio objectives in antenna design**:
   - Balanis, C. A. (2016). *Antenna theory: analysis and design* (4th ed.). Wiley. (Chapter 2 covers directivity definitions.)
   - Haupt, R. L., & Werner, D. H. (2007). *Genetic algorithms in electromagnetics*. Wiley. (Includes examples of ratio‑based fitness functions.)

2. **Quotient‑rule gradients in adjoint optimization**:
   - Nomura, S., et al. (2007). *Shape optimization for steady‑state heat conduction problems using a topology optimization technique*. International Journal of Heat and Mass Transfer, 50(13–14), 2853–2865. (Early application of quotient‑rule gradients in PDE‑constrained optimization.)
   - Deng, Y., & Liu, Z. (2020). *Adjoint‑based optimization of photonic devices with robustness considerations*. Optics Express, 28(18), 26632–26644.

3. **Beam‑steering metasurfaces**:
   - Yu, N., et al. (2011). *Light propagation with phase discontinuities: generalized laws of reflection and refraction*. Science, 334(6054), 333–337.
   - Pfeiffer, C., & Grbic, A. (2013). *Metamaterial Huygens' surfaces: tailoring wave fronts with reflectionless sheets*. Physical Review Letters, 110(19), 197401.

4. **Numerical stability of adjoint methods**:
   - Giles, M. B., & Pierce, N. A. (2000). *An introduction to the adjoint approach to design*. Flow, Turbulence and Combustion, 65(3–4), 393–415. (Discusses cancellation issues in adjoint right‑hand sides.)
   - Nadarajah, S. K., & Jameson, A. (2000). *A comparison of the continuous and discrete adjoint approach to automatic aerodynamic optimization*. AIAA Paper 2000–0667.

---

*Next: Chapter 4, "Optimization Workflow," provides an end‑to‑end practical guide for inverse design, covering objective setup, solver calls, line‑search behavior, diagnostic checks, and post‑optimization validation.*
