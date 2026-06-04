# API: Adjoint and Optimization

## Purpose

Reference for adjoint sensitivity analysis and optimization. This page covers:
1. **Adjoint primitives** -- the building blocks for computing gradients of far-field objectives with respect to impedance parameters.
2. **Optimizers** -- projected L-BFGS routines that use the adjoint gradient to optimize surface impedance distributions.

The key idea: instead of computing N finite-difference solves (one per parameter), the adjoint method computes the exact gradient in just **one additional linear solve** (the adjoint system), regardless of the number of parameters.

---

## Adjoint Primitives

### `compute_objective(I, Q)`

Compute the quadratic far-field objective:

```
J = Re(I' * Q * I)
```

where `I` is the current coefficient vector and `Q` is a Hermitian positive-semidefinite matrix built from far-field operators (see `build_Q` in [farfield-rcs.md](farfield-rcs.md)).

**Parameters:**
- `I::Vector{<:Number}`: MoM current coefficients (length N).
- `Q::Matrix{<:Number}`: Hermitian PSD objective matrix (N x N).

**Returns:** `Float64` objective value `J`.

**Physical meaning:** `J = Re(I' Q I)` represents a weighted integral of the far-field pattern. For example:
- `Q = Q_total` (all directions, all polarizations): `J` is proportional to total radiated power.
- `Q = Q_target` (mask to broadside cone, x-polarization): `J` is proportional to power radiated in the target direction with the desired polarization.

---

### `solve_adjoint(Z, Q, I; solver=:direct, preconditioner=nothing, gmres_tol=1e-8, gmres_maxiter=200)`

Solve the adjoint system:

```
Z' * lambda = Q * I
```

where `Z'` is the conjugate transpose of the system matrix. The adjoint variable `lambda` is used to compute the gradient without forming the full Jacobian.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `AbstractMatrix{<:Number}` | -- | System matrix (same Z used in the forward solve). |
| `Q` | `Matrix{<:Number}` | -- | Objective matrix. |
| `I` | `Vector{<:Number}` | -- | Current coefficients from the forward solve. |
| `solver` | `Symbol` | `:direct` | `:direct` for LU factorization, `:gmres` for GMRES. Same choice as the forward solve. |
| `preconditioner` | `Nothing` or `AbstractPreconditionerData` | `nothing` | Near-field preconditioner for GMRES. When provided, the **adjoint** preconditioner `Z_nf^{-H}` is automatically applied. |
| `gmres_tol` | `Float64` | `1e-8` | GMRES relative tolerance. |
| `gmres_maxiter` | `Int` | `200` | Maximum GMRES iterations. |

**Returns:** `Vector{ComplexF64}` adjoint variable `lambda`.

**Note on GMRES:** When `solver=:gmres`, uses `solve_gmres_adjoint` internally, which applies the adjoint (conjugate-transpose) preconditioner `Z_nf^{-H}`. This ensures the preconditioned adjoint system is consistent with the preconditioned forward system.

---

### `solve_adjoint_rhs(Z, rhs; solver=:direct, preconditioner=nothing, gmres_tol=1e-8, gmres_maxiter=200)`

Solve the adjoint system with a pre-computed right-hand side:

```
Z' * lambda = rhs
```

Unlike `solve_adjoint(Z, Q, I)` which internally computes `rhs = Q * I`, this function accepts the RHS directly. This is useful when:
- Using `apply_Q` for matrix-free Q application (avoids forming the dense N x N Q matrix).
- Computing the adjoint RHS from multiple sources (e.g., multi-angle objectives).
- Working with `ImpedanceLoadedOperator` where Q*I is computed per-angle in the outer loop.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z` | `AbstractMatrix{<:Number}` | -- | System matrix (same Z used in the forward solve). |
| `rhs` | `AbstractVector{<:Number}` | -- | Pre-computed right-hand side vector (e.g., `Q * I` or output of `apply_Q`). |
| `solver` | `Symbol` | `:direct` | `:direct` for LU factorization, `:gmres` for GMRES. |
| `preconditioner` | `Nothing` or `AbstractPreconditionerData` | `nothing` | Near-field preconditioner for GMRES. |
| `gmres_tol` | `Float64` | `1e-8` | GMRES relative tolerance. |
| `gmres_maxiter` | `Int` | `200` | Maximum GMRES iterations. |

**Returns:** `Vector{ComplexF64}` adjoint variable `lambda`.

**Example:**

```julia
# Standard usage: equivalent to solve_adjoint(Z, Q, I)
rhs = Q * I
lambda = solve_adjoint_rhs(Z, rhs)

# Matrix-free Q application (large N)
rhs = apply_Q(G_mat, grid, pol, I; mask=mask)
lambda = solve_adjoint_rhs(Z, rhs; solver=:gmres, preconditioner=P_nf)

# With ImpedanceLoadedOperator
Z_op = ImpedanceLoadedOperator(Z_base, Mp, theta)
lambda = solve_adjoint_rhs(Z_op, rhs; solver=:gmres, preconditioner=P_nf)
```

---

### `gradient_impedance(Mp, I, lambda; reactive=false)`

Compute the adjoint gradient of the objective with respect to impedance parameters:

```
g[p] = -2 Re{ lambda' * (dZ/d(theta_p)) * I }
```

This is the key formula: the gradient for **all** P parameters is computed from a single inner product per parameter, using the already-computed forward solution `I` and adjoint variable `lambda`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Mp` | `Vector{<:AbstractMatrix}` | -- | Patch mass matrices from `precompute_patch_mass`. |
| `I` | `Vector{<:Number}` | -- | Current coefficients from forward solve. |
| `lambda` | `Vector{<:Number}` | -- | Adjoint variable from `solve_adjoint`. |
| `reactive` | `Bool` | `false` | Must match the `reactive` flag used in `assemble_full_Z`. |

**Returns:** `Vector{Float64}` gradient vector `g` of length `P`.

**Formulas by loading mode:**

| Mode | Derivative | Gradient formula |
|------|-----------|-----------------|
| Resistive (`reactive=false`) | `dZ/d(theta_p) = -Mp[p]` | `g[p] = +2 Re{ lambda' * Mp[p] * I }` |
| Reactive (`reactive=true`) | `dZ/d(theta_p) = -i*Mp[p]` | `g[p] = -2 Im{ lambda' * Mp[p] * I }` |

---

## Optimizers

### `optimize_lbfgs(Z_efie, Mp, v, Q, theta0; kwargs...)`

Projected L-BFGS optimization for a single quadratic objective `J = Re(I' Q I)`. This is the main optimization entry point for impedance design problems.

**Required parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Z_efie` | `Matrix{ComplexF64}` | EFIE matrix (PEC part, without impedance loading). |
| `Mp` | `Vector{<:AbstractMatrix}` | Patch mass matrices from `precompute_patch_mass`. |
| `v` | `Vector{ComplexF64}` | Excitation vector (from any excitation type). |
| `Q` | `Matrix{ComplexF64}` | Objective matrix (Hermitian PSD, from `build_Q`). |
| `theta0` | `Vector{Float64}` | Initial parameter vector (length P). |

**Optimization options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `reactive` | `Bool` | `false` | If `true`, impedance is `Z_s = i*theta` (reactive/lossless). If `false`, `Z_s = theta` (resistive/lossy). |
| `maximize` | `Bool` | `false` | If `true`, maximize `J` instead of minimizing. Internally minimizes `-J`. |
| `lb` | `Vector` or `nothing` | `nothing` | Lower bounds on `theta` (projected L-BFGS-B). `nothing` = no lower bound. |
| `ub` | `Vector` or `nothing` | `nothing` | Upper bounds on `theta`. `nothing` = no upper bound. |
| `maxiter` | `Int` | `100` | Maximum L-BFGS iterations. Each iteration requires one forward solve + one adjoint solve + one line-search solve. |
| `tol` | `Float64` | `1e-6` | Gradient-norm convergence tolerance. The optimizer stops when `||g|| < tol`. |
| `m_lbfgs` | `Int` | `10` | L-BFGS memory length (number of past gradient pairs stored). Higher values give better Hessian approximation but use more memory. 5--20 is typical. |
| `alpha0` | `Float64` | `0.01` | Initial step-size scaling for the first iteration. Subsequent step sizes are adapted by L-BFGS. |
| `verbose` | `Bool` | `true` | Print iteration progress (iteration number, objective value, gradient norm). |

**Solver options (controls how the forward and adjoint systems are solved):**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `solver` | `Symbol` | `:direct` | `:direct` for LU factorization (exact, O(N^3) per solve), `:gmres` for iterative GMRES. |
| `nf_preconditioner` | `Nothing` or `AbstractPreconditionerData` | `nothing` | Near-field preconditioner for GMRES. Build once with `build_nearfield_preconditioner` and pass here. Ignored when `solver=:direct`. |
| `gmres_tol` | `Float64` | `1e-8` | GMRES relative tolerance. |
| `gmres_maxiter` | `Int` | `200` | Maximum GMRES iterations per solve. |

**Mass-based conditioning options (advanced):**

These options apply a mass-based left preconditioner `M` to the system `Z_eff = M^{-1} Z` for better conditioning. This is separate from the GMRES near-field preconditioner and addresses the conditioning of the optimization problem itself.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `regularization_alpha` | `Float64` | `0.0` | Regularization coefficient. Adds `alpha * R` to the system matrix where `R = sum(Mp)`. |
| `regularization_R` | Matrix or `nothing` | `nothing` | Custom regularization matrix. Default: `sum(Mp)` when alpha > 0. |
| `preconditioner_M` | Matrix or `nothing` | `nothing` | Explicit left preconditioner matrix. Takes precedence over `preconditioning` mode. |
| `preconditioning` | `Symbol` | `:off` | `:off` (disabled), `:on` (always), or `:auto` (enable when N >= threshold). |
| `auto_precondition_n_threshold` | `Int` | `256` | System size threshold for `:auto` mode. |
| `iterative_solver` | `Bool` | `false` | If `true`, enables preconditioning in `:auto` mode. |
| `auto_precondition_eps_rel` | `Float64` | `1e-6` | Relative diagonal shift for auto-built preconditioner. |

**Returns:** Tuple `(theta_opt, trace)` where:
- `theta_opt::Vector{Float64}`: Optimized parameter vector.
- `trace::Vector{NamedTuple}`: Iteration records with fields `(iter, J, gnorm)` -- iteration number, objective value, and gradient norm.

---

### `optimize_directivity(Z_efie, Mp, v, Q_target, Q_total, theta0; kwargs...)`

Maximize the directivity ratio:

```
J = (I' * Q_target * I) / (I' * Q_total * I)
```

using projected L-BFGS. This is the standard formulation for maximizing directivity in a target direction: `Q_target` selects the far-field power in the desired region, and `Q_total` represents total radiated power.

**Required parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Z_efie` | `Matrix{ComplexF64}` | EFIE matrix. |
| `Mp` | `Vector{<:AbstractMatrix}` | Patch mass matrices. |
| `v` | `Vector{ComplexF64}` | Excitation vector. |
| `Q_target` | `Matrix{ComplexF64}` | Objective matrix for numerator (target region). |
| `Q_total` | `Matrix{ComplexF64}` | Objective matrix for denominator (total radiated power). |
| `theta0` | `Vector{Float64}` | Initial parameter vector. |

**Options:** Same as `optimize_lbfgs` (including all solver, NF preconditioner, and conditioning options), except `maximize` is implicitly true for the ratio. The routine internally minimizes `-J`.

**Returns:** Same structure as `optimize_lbfgs`.

**How to build Q_target and Q_total:**
```julia
grid = make_sph_grid(90, 36)
G_mat = radiation_vectors(mesh, rwg, grid, k)
pol = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=pi/18)   # 10-degree cone around broadside

Q_target = build_Q(G_mat, grid, pol; mask=mask)   # power in target cone
Q_total  = build_Q(G_mat, grid, pol)               # total radiated power
```

---

### `optimize_multiangle_rcs(Z_base, Mp, configs, theta0; kwargs...)`

Minimize total weighted backscatter RCS over multiple incidence angles using projected L-BFGS. This optimizer supports any `AbstractMatrix{ComplexF64}` as the base operator (MLFMA, ACA, dense) via `ImpedanceLoadedOperator`.

**Required parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `Z_base` | `AbstractMatrix{ComplexF64}` | Base EFIE operator (MLFMAOperator, ACAOperator, or dense Matrix). |
| `Mp` | `Vector{<:AbstractMatrix}` | Patch mass matrices from `precompute_patch_mass`. |
| `configs` | `Vector{AngleConfig}` | Per-angle configurations from `build_multiangle_configs`. |
| `theta0` | `Vector{Float64}` | Initial parameter vector (length P). |

**Keyword arguments:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `maxiter` | `Int` | `100` | Maximum L-BFGS iterations. |
| `tol` | `Float64` | `1e-6` | Gradient-norm convergence tolerance. |
| `m_lbfgs` | `Int` | `10` | L-BFGS memory length. |
| `alpha0` | `Float64` | `0.01` | Initial inverse-Hessian scaling. |
| `verbose` | `Bool` | `true` | Print iteration progress. |
| `reactive` | `Bool` | `false` | Impedance mode: `false` = resistive, `true` = reactive. |
| `lb` | `Vector` or `nothing` | `nothing` | Lower bounds on theta. |
| `ub` | `Vector` or `nothing` | `nothing` | Upper bounds on theta. |
| `preconditioner` | `AbstractPreconditionerData` or `nothing` | `nothing` | GMRES preconditioner (strongly recommended). |
| `gmres_tol` | `Float64` | `1e-6` | GMRES relative tolerance. |
| `gmres_maxiter` | `Int` | `300` | Maximum GMRES iterations per solve. |

**Returns:** Tuple `(theta_opt, trace)` where:
- `theta_opt::Vector{Float64}`: Optimized parameter vector.
- `trace::Vector{NamedTuple}`: Iteration records with fields `(iter, J, gnorm)`.

**Per-iteration cost:** M forward solves + M adjoint solves + line-search forward solves (M per trial step). Always uses GMRES internally (the composite `ImpedanceLoadedOperator` is matrix-free).

**Objective:** `J(theta) = sum_a w_a * Re(I_a' Q_a I_a)` where `I_a = Z(theta)^{-1} v_a`.

**Gradient:** `g[p] = sum_a w_a * gradient_impedance(Mp, I_a, lambda_a)` where `Z(theta)' lambda_a = Q_a I_a`.

See the [Multi-Angle RCS chapter](../differentiable-design/05-multiangle-rcs.md) for a detailed walkthrough and examples.

---

## Solver and Conditioning Options

Both optimizers support two independent preconditioning mechanisms:

### 1. GMRES with near-field preconditioner (for the linear solves)

When `solver=:gmres`, each forward and adjoint solve uses GMRES instead of LU factorization. The `nf_preconditioner` provides a sparse near-field approximation that dramatically reduces iteration counts.

```julia
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, 1.0 * lambda0)
theta_opt, trace = optimize_lbfgs(Z_efie, Mp, v, Q, theta0;
    solver=:gmres, nf_preconditioner=P_nf,
    gmres_tol=1e-8, gmres_maxiter=300)
```

**Note:** The preconditioner is built from the PEC EFIE matrix `Z_efie` and reused throughout the optimization, even as the impedance loading changes. This works because the near-field structure of the EFIE matrix is the dominant conditioning factor.

### 2. Mass-based conditioning (for the optimization problem)

The conditioning options (`regularization_alpha`, `preconditioning`, `preconditioner_M`) transform the system `Z_eff = M^{-1} Z` to improve the conditioning of the optimization landscape. This is a different concern from GMRES convergence: it affects gradient quality and optimizer convergence rate.

These two mechanisms are independent and can be combined.

---

## Minimal Usage

### Direct solver (default)
```julia
theta_opt, trace = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta0;
    reactive=true, maximize=true, maxiter=100
)
```

### GMRES with near-field preconditioner
```julia
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, 1.0 * lambda0)
theta_opt, trace = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta0;
    solver=:gmres, nf_preconditioner=P_nf,
    gmres_tol=1e-8, gmres_maxiter=300,
    reactive=true, maximize=true, maxiter=100
)
```

### Directivity optimization
```julia
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta0;
    reactive=true, maxiter=300
)
```

### Inspecting convergence
```julia
# Plot objective vs iteration
using Plots
plot([t.iter for t in trace], [t.J for t in trace],
     xlabel="Iteration", ylabel="Objective J", title="Convergence")

# Check final gradient norm
println("Final gradient norm: ", trace[end].gnorm)
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/optimization/Adjoint.jl` | `compute_objective`, `solve_adjoint`, `solve_adjoint_rhs`, `gradient_impedance` |
| `src/optimization/Optimize.jl` | `optimize_lbfgs`, `optimize_directivity` |
| `src/optimization/MultiAngleRCS.jl` | `AngleConfig`, `build_multiangle_configs`, `optimize_multiangle_rcs` |
| `src/assembly/CompositeOperator.jl` | `ImpedanceLoadedOperator` (composite operator for fast-operator optimization) |
| `src/solver/Solve.jl` | Conditioning helpers used by optimizers |

---

## Exercises

- **Basic:** Run one `optimize_lbfgs` case and inspect the trace length and final objective value.
- **Practical:** Compare optimization convergence with `:direct` vs `:gmres` + NF preconditioner for the same problem. Verify the optimal `theta` vectors agree.
- **Challenge:** Compare `:off` vs `:auto` mass-based preconditioning traces for the same problem. Does preconditioning improve convergence rate?
