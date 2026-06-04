# Tutorial: Adjoint Gradient Check

## Purpose

Gradient accuracy is the foundation of reliable optimization. Before attempting any inverse design, you must verify that the adjoint gradient matches independent finite‑difference approximations. This tutorial walks through a complete gradient‑verification workflow for resistive and reactive impedance parameters, teaching you to:

- Compute adjoint gradients using the built‑in `Adjoint.jl` routines
- Validate them against central finite differences and complex‑step derivatives
- Interpret relative‑error magnitudes and diagnose common discrepancies
- Understand the mathematical foundations of the adjoint method (paper Eq. 25)

**Why verification matters:** A single incorrect gradient component can stall an optimization or converge to a sub‑optimal design. Gradient verification is a mandatory prerequisite for any serious optimization work.

---

## Learning Goals

After this tutorial, you should be able to:

1. **Understand the adjoint gradient formula** for impedance parameters (paper Eq. 25) and its finite‑difference approximation.
2. **Compute adjoint gradients** for a small test problem using `solve_adjoint` and `gradient_impedance`.
3. **Validate gradient components** with central finite differences (`verify_gradient`) and complex‑step derivatives.
4. **Interpret relative‑error magnitudes** and distinguish acceptable numerical noise from implementation bugs.
5. **Diagnose common gradient‑verification failures** (sign errors, preconditioning inconsistencies, step‑size sensitivity).
6. **Apply verification to custom objectives** (far‑field directivity, power ratio, etc.).

---

## Mathematical Background

### Adjoint Gradient for Impedance Parameters

The objective function $J(\theta) = \Re\{I^\dagger Q I\}$ depends on the surface‑impedance parameters $\theta_p$ through the MoM matrix

\[
Z(\theta) = Z_\text{EFIE} + Z_\text{imp}(\theta),\qquad
Z_\text{imp}(\theta) = -\sum_{p=1}^P \theta_p M_p.
\]

For **resistive impedance** ($Z_s = \theta_p$), the derivative is $\partial Z/\partial\theta_p = -M_p$. Using the adjoint variable $\lambda$ that solves $Z^\dagger \lambda = Q I$, the gradient is (paper Eq. 25)

\[
\frac{\partial J}{\partial\theta_p} = -2\,\Re\{\lambda^\dagger (\partial Z/\partial\theta_p) I\}
      = +2\,\Re\{\lambda^\dagger M_p I\}.
\]

For **reactive impedance** ($Z_s = i\theta_p$), $\partial Z/\partial\theta_p = -iM_p$ and

\[
\frac{\partial J}{\partial\theta_p} = -2\,\Re\{\lambda^\dagger (-iM_p) I\}
      = -2\,\Im\{\lambda^\dagger M_p I\}.
\]

These closed‑form expressions are implemented in `gradient_impedance` (`src/optimization/Adjoint.jl:45`).

### Finite‑Difference Verification

Central finite differences approximate the gradient as

\[
g_p^{\text{FD}} = \frac{J(\theta + h e_p) - J(\theta - h e_p)}{2h},
\]

with step size $h \sim 10^{-5}–10^{-8}$. The relative error

\[
\varepsilon_p = \frac{|g_p^{\text{adj}} - g_p^{\text{FD}}|}{\max(|g_p^{\text{FD}}|, \epsilon_{\text{tiny}})}
\]

should be $\lesssim 10^{-6}$ for well‑conditioned problems. Larger errors indicate implementation mistakes (sign errors, incorrect `Mp` scaling, preconditioning inconsistency).

### Complex‑Step Derivatives

When the objective is holomorphic in $\theta$, the complex‑step method

\[
g_p^{\text{CS}} = \frac{\Im\{J(\theta + i\epsilon e_p)\}}{\epsilon},\qquad \epsilon \sim 10^{-30}
\]

provides machine‑precision gradients without subtractive cancellation. The function `verify_gradient` (`src/optimization/Verification.jl:42`) automatically compares adjoint, finite‑difference, and complex‑step results.

---

## Step‑by‑Step Workflow

### 1) Build a Small Test Problem

Use a modest plate mesh so repeated solves are fast (≈1 s for the whole verification). The same workflow scales to larger problems once verified.

```julia
using DiffMoM
using Printf

# 10 cm × 10 cm plate, 3×3 triangles (N=12 edges)
mesh = make_rect_plate(0.1, 0.1, 3, 3)
rwg  = build_rwg(mesh)
N    = rwg.nedges
Nt   = ntriangles(mesh)

# EFIE matrix at 3 GHz
freq = 3e9
c0   = 299792458.0
lambda0 = c0 / freq
k    = 2π / lambda0
eta0 = 376.730313668
Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)

# Impedance partition: one patch per triangle (P = Nt)
partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

# Plane‑wave excitation (normal incidence, x‑polarized)
k_vec = Vec3(0.0, 0.0, -k)
E0    = 1.0
pol_inc = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol_inc; quad_order=3)

# Simple objective: total radiated power (Q = identity)
Q = Matrix{ComplexF64}(I, N, N)

# Initial impedance parameters (small resistive values)
theta0 = randn(Nt) .* 10.0   # ~10 Ω standard deviation
```

### 2) Compute Adjoint Gradient

Assemble the full MoM matrix, solve the forward and adjoint systems, and evaluate the gradient.

```julia
# Reactive impedance (Z_s = iθ) – change to reactive=false for resistive
reactive_flag = true

Z = assemble_full_Z(Z_efie, Mp, theta0; reactive=reactive_flag)
I = solve_forward(Z, v)
λ = solve_adjoint(Z, Q, I)
g_adj = gradient_impedance(Mp, I, λ; reactive=reactive_flag)

println("Gradient vector length P = $(length(g_adj))")
println("‖g‖₂ = $(norm(g_adj))")
println("min/max g = $(extrema(g_adj))")
```

### 3) Define the Objective Function for Verification

The verification routine needs a function `f_obj(θ)` that returns $J(\theta)$. It must accept complex arguments for complex‑step derivatives.

```julia
function f_obj(theta_vec::AbstractVector)
    Z_local = assemble_full_Z(Z_efie, Mp, theta_vec; reactive=reactive_flag)
    I_local = Z_local \ v
    return real(dot(I_local, Q * I_local))
end

# Quick sanity check: value at theta0
J0 = f_obj(theta0)
println("J(θ₀) = $J0")
```

### 4) Run Gradient Verification

Use `verify_gradient` to compare adjoint, finite‑difference, and complex‑step gradients for the first 10 parameters (checking all $P$ parameters is expensive but possible).

```julia
# `verify_gradient` is exported by DiffMoM; no extra import needed

results = verify_gradient(f_obj, g_adj, theta0;
                         indices=1:10,          # check only first 10 parameters
                         h_fd=1e-5,             # finite‑difference step
                         eps_cs=1e-30)          # complex‑step epsilon

println("\nGradient verification (first 10 parameters):")
println("p | adjoint        FD            CS            rel_err_fd   rel_err_cs")
println("-"^78)
for r in results
    @printf("%2d | %12.6f  %12.6f  %12.6f  %10.2e  %10.2e\n",
            r.p, r.adj, r.fd, r.cs, r.rel_err_fd, r.rel_err_cs)
end

# Summarise
max_fd_err = maximum(r.rel_err_fd for r in results)
max_cs_err = maximum(r.rel_err_cs for r in results)
println("\nMaximum relative error vs FD: $max_fd_err")
println("Maximum relative error vs CS: $max_cs_err")
```

### 5) Full Verification (All Parameters)

For production‑level confidence, verify all parameters. This requires $2P$ forward solves (expensive for large $P$), so restrict to small test problems.

```julia
# Full verification (optional – slow for P > 100)
if Nt ≤ 50
    results_all = verify_gradient(f_obj, g_adj, theta0; indices=nothing, h_fd=1e-5)
    max_err_all = maximum(r.rel_err_fd for r in results_all)
    println("Full verification (all $Nt parameters): max relative error = $max_err_all")
    if max_err_all < 1e-6
        println("✓ Gradient passes verification.")
    else
        println("⚠ Gradient verification failed – investigate!")
    end
end
```

---

## Interpretation Guidelines

### Expected Error Magnitudes

| Condition | Relative Error (vs FD) | Meaning |
|-----------|------------------------|---------|
| Well‑conditioned, unpreconditioned | $10^{-8} – 10^{-12}$ | Machine‑precision agreement |
| With left preconditioning | $10^{-6} – 10^{-8}$ | Slight inconsistency in `Mp` transformation |
| Ill‑conditioned ($\kappa(Z) > 10^{10}$) | $10^{-4} – 10^{-6}$ | Finite‑difference truncation amplified |
| **Sign error** | $\approx 2$ | Gradient formula has wrong sign |
| **Scaling error** | Constant factor (e.g., 0.5, 2) | Missing factor in `gradient_impedance` |

### Step‑Size Sensitivity

Plot `max_rel_error` vs `h` to distinguish truncation error ($\propto h^2$) from round‑off error ($\propto h^{-1}$). The optimal $h$ balances the two; for double precision, $h \approx 10^{-5}$ is usually safe.

```julia
hs = 10.0 .^ (-10:-2)   # 1e‑10 … 1e‑2
errs = Float64[]
for h in hs
    rs = verify_gradient(f_obj, g_adj, theta0; indices=1:5, h_fd=h)
    push!(errs, maximum(r.rel_err_fd for r in rs))
end

# Plot with your favorite package (Plots, PyPlot, etc.)
# plot(log10.(hs), log10.(errs), xlabel="log10(h)", ylabel="log10(max rel error)")
```

The curve should have a V‑shape; a flat plateau indicates numerical noise dominates.

### Preconditioning Consistency

If you use left preconditioning (`preconditioner_M`), the derivative blocks must be transformed as $M_p \to M^{-1} M_p$. The function `transform_patch_matrices` (`src/solver/Solve.jl:135`) handles this automatically when `preconditioner_M` is passed to `gradient_impedance`. Verify that the same preconditioner is used in forward, adjoint, and gradient computations.

---

## Troubleshooting

### Error 1: Relative Errors ≈ 2 (Sign Error)

If all relative errors are close to 2, the adjoint gradient has the wrong sign. Check:

- **`reactive` flag**: `gradient_impedance(…; reactive=true)` for $Z_s = i\theta$, `false` for $Z_s = \theta$.
- **Objective sign**: $J = \Re\{I^\dagger Q I\}$ uses `real(dot(I, Q*I))`, not `abs2`.
- **Adjoint RHS**: `solve_adjoint(Z, Q, I)` solves $Z^\dagger \lambda = Q I$, not $(Q I)^*$.

### Error 2: Constant Scaling Factor (e.g., 0.5)

If errors show a consistent factor (e.g., 0.5, 2, 4), verify the gradient formula prefactor:

- Resistive: $+2\Re\{\lambda^\dagger M_p I\}$ (factor 2)
- Reactive: $-2\Im\{\lambda^\dagger M_p I\}$ (factor −2)

The `gradient_impedance` implementation (`src/optimization/Adjoint.jl:45`) already includes these factors; a scaling discrepancy suggests `Mp` is incorrectly normalized.

### Error 3: Large Errors Only for Ill‑Conditioned Parameters

Parameters that affect nearly singular modes produce large finite‑difference truncation errors. Solutions:

- Use complex‑step derivatives (`eps_cs=1e‑30`) which are immune to truncation.
- Add a small regularisation (`regularization_alpha=1e‑8`) to improve conditioning.
- Accept larger errors ($<10^{-4}$) for those parameters.

### Error 4: Inconsistent Results with Preconditioning

If verification passes without preconditioning but fails with `preconditioner_M`, ensure:

1. **Same preconditioner in forward/adjoint solves**: `prepare_conditioned_system` returns `(Z_eff, rhs_eff, factor)`; use `factor` for both solves.
2. **Transformed derivative blocks**: Call `gradient_impedance` with `preconditioner_factor=factor` (or `preconditioner_M`).
3. **Consistent `reactive` flag**: Preconditioner transformation does not change the `reactive`/`resistive` distinction.

### Error 5: Complex‑Step Fails (NaN/Infinite Errors)

Complex‑step requires the objective to be holomorphic. If $J(\theta)$ involves `conj`, `real`, or `abs` applied to intermediate complex variables, complex‑step will fail. In that case, rely on finite‑difference verification only.

---

## Code Mapping

| Task | Function | Source File | Key Lines |
|------|----------|-------------|-----------|
| **Adjoint solve** | `solve_adjoint(Z, Q, I)` | `src/optimization/Adjoint.jl` | 25–29 |
| **Gradient computation** | `gradient_impedance(Mp, I, λ; reactive)` | `src/optimization/Adjoint.jl` | 45–64 |
| **Finite‑difference derivative** | `fd_grad(f, theta, p; h, scheme)` | `src/optimization/Verification.jl` | 27–39 |
| **Complex‑step derivative** | `complex_step_grad(f, theta, p; eps)` | `src/optimization/Verification.jl` | 15–20 |
| **Gradient verification** | `verify_gradient(f_obj, g_adj, theta; …)` | `src/optimization/Verification.jl` | 42–86 |
| **Full system assembly** | `assemble_full_Z(Z_efie, Mp, theta; reactive)` | `src/solver/Solve.jl` | 34–44 |
| **Forward solve** | `solve_forward(Z, v)` | `src/solver/Solve.jl` | 13–15 |
| **Patch mass matrices** | `precompute_patch_mass(mesh, rwg, partition)` | `src/assembly/Impedance.jl` | 15–64 |

**Example from the beam‑steering tutorial** (`examples/03_beamsteering_physical_unitcell.jl:210‑226`):

```julia
function J_of_theta_reactive(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= (1im * theta_vec[p]) .* Mp[p]
    end
    I_t = Z_t \ v
    f_t = real(dot(I_t, Q_target * I_t))
    g_t = real(dot(I_t, Q_total * I_t))
    return f_t / g_t   # directivity ratio
end

# Finite‑difference check at optimum
for p in 1:5
    g_fd = fd_grad(J_of_theta_reactive, theta_opt, p; h=1e-5)
    rel_err = abs(g_opt[p] - g_fd) / max(abs(g_opt[p]), 1e-30)
    println("p=$p: adj=$(g_opt[p])  fd=$g_fd  rel_err=$rel_err")
end
```

---

## Exercises

### Basic (30 minutes)

1. **Run the verification workflow** above for resistive impedance (`reactive=false`). Confirm that relative errors are $<10^{-6}$.
2. **Change the objective** to $J = \Im\{I^\dagger Q I\}$ (imaginary part). Modify `f_obj` and verify the gradient.
3. **Sweep step size** $h$ from $10^{-2}$ to $10^{-10}$ for the first parameter. Plot relative error vs $h$ and identify the optimal $h$.

### Practical (60 minutes)

1. **Add left preconditioning** with `make_left_preconditioner(Mp; eps_rel=1e-6)`. Use `prepare_conditioned_system` for forward/adjoint solves and `transform_patch_matrices` for gradient computation. Verify that gradients still match.
2. **Test a far‑field objective** using `build_Q` from `src/optimization/QMatrix.jl`. Compute the gradient for maximising power in a $10^\circ$ cone and verify against finite differences.
3. **Compare complex‑step and finite‑difference** errors for ill‑conditioned systems. Increase the frequency to 30 GHz (smaller mesh relative to wavelength) and observe how conditioning affects verification accuracy.

### Advanced (90 minutes)

1. **Implement your own gradient routine** for the objective $J = |S_{11}|^2$ (reflection coefficient). Derive the adjoint formula, code it, and verify against finite differences.
2. **Profile the verification cost** for $P$ parameters. Measure the time of $2P$ forward solves vs one adjoint solve. At what $P$ does the adjoint method break even?
3. **Extend verification to shape parameters** (vertex displacements). Use the shape derivative formulas from the paper and compare with finite‑difference perturbations of `mesh.vertices`.

---

## Tutorial Checklist

Before moving to optimization, complete these verification steps:

- [ ] **Run the basic verification** with resistive and reactive impedance.
- [ ] **Confirm relative errors** $<10^{-6}$ for all checked parameters.
- [ ] **Test with your actual objective** (e.g., far‑field directivity, S‑parameters).
- [ ] **Validate preconditioning consistency** if using left preconditioning.
- [ ] **Document the verification** in your notebook or script, including max error and step‑size sensitivity plot.

---

## Further Reading

- **Paper Section 3.2** – Adjoint gradient derivation for impedance parameters (Eq. 25).
- **`src/optimization/Adjoint.jl`** – Full implementation of `solve_adjoint` and `gradient_impedance`.
- **`src/optimization/Verification.jl`** – Gradient verification utilities with complex‑step and finite‑difference.
- **Tutorial 3: Beam‑Steering Design** – Applies verified gradients to a realistic inverse‑design problem.
- **Martins et al., *ACM Trans. Math. Softw.*, 2019** – Comprehensive review of adjoint methods and verification techniques.
