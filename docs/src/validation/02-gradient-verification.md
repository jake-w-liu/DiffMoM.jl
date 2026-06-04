# Chapter 2: Gradient Verification

## Purpose

Ensure that adjoint gradients are mathematically correct by comparing them against finite-difference and complex-step references. Gradient verification is essential before trusting optimization results, as it confirms that the sensitivity analysis correctly encodes the derivative of the discretized objective with respect to design parameters.

---

## Learning Goals

After this chapter, you should be able to:

1. Run finite-difference and complex-step gradient checks for impedance parameters.
2. Interpret relative-error trends versus step size and identify optimal FD step regions.
3. Understand the limitations of complex-step for real-valued objectives with conjugation.
4. Apply systematic verification workflows used in the paper's validation.
5. Diagnose common gradient implementation errors through verification patterns.

---

## 1) Mathematical Foundations

### 1.1 Adjoint Gradient Formula

For impedance parameters $\theta_p$, the adjoint gradient of a quadratic objective $J = \Re\{\mathbf{I}^\dagger \mathbf{Q} \mathbf{I}\}$ is:

```math
g_p^{\mathrm{adj}} = -2\,\Re\!\left\{
\boldsymbol{\lambda}^\dagger
\left(\frac{\partial\mathbf{Z}}{\partial\theta_p}\right)
\mathbf{I}
\right\}
```

where $\boldsymbol{\lambda}$ solves the adjoint system $\mathbf{Z}^\dagger \boldsymbol{\lambda} = \mathbf{Q}\mathbf{I}$. The impedance derivative is simply $\partial\mathbf{Z}/\partial\theta_p = -\mathbf{M}_p$ (resistive) or $-i\mathbf{M}_p$ (reactive), where $\mathbf{M}_p$ is the patch mass matrix.

### 1.2 Finite-Difference Reference

Central finite differences provide a reliable reference:

```math
g_p^{\mathrm{FD}} \approx \frac{J(\theta_p + h) - J(\theta_p - h)}{2h}
```

The truncation error is $O(h^2)$, while round-off error grows as $\epsilon_{\text{mach}}/h$. The optimal step $h^*$ balances these errors.

### 1.3 Complex-Step Method

For holomorphic functions $f_{\text{hol}}(\theta)$, complex-step gives machine-precision derivatives:

```math
g_p^{\mathrm{CS}} \approx \frac{\Im\{f_{\text{hol}}(\theta + i\varepsilon)\}}{\varepsilon}, \quad \varepsilon \sim 10^{-30}
```

**Important limitation**: Real-valued objectives involving complex conjugation (e.g., $\mathbf{I}^\dagger\mathbf{Q}\mathbf{I}$) are not holomorphic end-to-end. Complex-step can only verify holomorphic subcomponents.

### 1.4 Relative Error Metric

Compare using relative error:

```math
\text{rel.err.} = \frac{|g^{\mathrm{adj}} - g^{\mathrm{ref}}|}{\max(|g^{\mathrm{ref}}|, \epsilon)}
```

where $\epsilon \sim 10^{-30}$ prevents division by zero.

---

## 2) Implementation in `DiffMoM.jl`

### 2.1 Verification Functions (`src/optimization/Verification.jl`)

- **`fd_grad(f, theta, p; h=1e-6, scheme=:central)`**: Finite-difference gradient
- **`complex_step_grad(f, theta, p; eps=1e-30)`**: Complex-step gradient (holomorphic `f` only)
- **`verify_gradient(f_objective, adjoint_grad, theta; indices=nothing, eps_cs=1e-30, h_fd=1e-6)`**: Comprehensive verification

### 2.2 Typical Verification Workflow

```julia
# 1. Define objective function
function J_of_theta(theta_vec)
    Z_total = assemble_full_Z(Z_efie, Mp, theta_vec)
    I = Z_total \ v
    return real(dot(I, Q * I))
end

# 2. Compute adjoint gradient (impedance parameters)
Z_total = assemble_full_Z(Z_efie, Mp, theta0)
I = solve_forward(Z_total, v)
lambda = solve_adjoint(Z_total, Q, I)
g_adj = gradient_impedance(Mp, I, lambda; reactive=true)

# 3. Verify against references
results = verify_gradient(J_of_theta, g_adj, theta0;
                          indices=1:10,   # Check first 10 parameters
                          h_fd=1e-5)

# 4. Analyze results
max_rel_err = maximum(r.rel_err_fd for r in results)
println("Maximum relative error: $max_rel_err")
```

### 2.3 Expected Accuracy from Paper

The validation in `bare_jrnl.tex` (Section IV, Table 1) reports:

- **Relative errors below $3 \times 10^{-7}$** for all tested parameters
- **Optimal step size $h = 10^{-5}$** for typical objective magnitudes
- **Adjoint-FD agreement** confirmed across mesh refinement levels

Example from paper (first 10 parameters on 3×3 mesh):

| $p$ | Adjoint | FD (central) | Rel. error |
|-----|---------|--------------|------------|
| 1 | $-1.44\times 10^{-7}$ | $-1.44\times 10^{-7}$ | $2.4\times 10^{-8}$ |
| 2 | $-4.58\times 10^{-8}$ | $-4.58\times 10^{-8}$ | $5.2\times 10^{-8}$ |
| 3 | $-1.74\times 10^{-7}$ | $-1.74\times 10^{-7}$ | $2.6\times 10^{-8}$ |
| ... | ... | ... | ... |
| 10 | $-1.26\times 10^{-7}$ | $-1.26\times 10^{-7}$ | $6.3\times 10^{-8}$ |

All relative errors are below $3\times 10^{-7}$, confirming adjoint correctness to the expected accuracy of central finite differences.

---

## 3) Practical Verification Workflows

### 3.1 Complete Verification Script

The convergence study (`validation/paper/run_convergence_study.jl`) includes gradient verification:

```julia
# Gradient check: impedance at one patch
partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
theta_test = fill(200.0, Nt)
Z_full = assemble_full_Z(Z_efie, Mp, theta_test)
I_imp  = Z_full \ v
lambda = Z_full' \ (Q * I_imp)
g_adj  = gradient_impedance(Mp, I_imp, lambda)

# FD verification on a few parameters
function J_of_theta(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= theta_vec[p] .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

n_check = min(5, Nt)
max_gerr = 0.0
for p in 1:n_check
    g_fd = fd_grad(J_of_theta, theta_test, p; h=1e-5)
    rel_err = abs(g_adj[p] - g_fd) / max(abs(g_adj[p]), 1e-30)
    max_gerr = max(max_gerr, rel_err)
end
```

### 3.2 Step Size Sweep Analysis

To identify the optimal FD step $h^*$, perform a sweep:

```julia
hs = 10.0 .^ (-10:-2)  # 1e-10 to 1e-2
errors = Float64[]

for h in hs
    g_fd = fd_grad(J_of_theta, theta0, p; h=h)
    err = abs(g_adj[p] - g_fd) / max(abs(g_adj[p]), 1e-30)
    push!(errors, err)
end

# Plot log-log: error should show V-shaped minimum
# Truncation error ∝ h² (decreasing left to right)
# Round-off error ∝ 1/h (increasing left to right)
```

**Expected pattern**: Error decreases as $O(h^2)$ until round-off dominates, then increases as $O(1/h)$. The minimum typically occurs at $h \sim 10^{-5}$ to $10^{-6}$ for double precision.

### 3.3 Complex-Step for Holomorphic Components

While the full objective isn't holomorphic, individual components may be:

```julia
# Check impedance contribution to matrix assembly
function Z_contrib(theta)
    return theta * Mp[p]  # Linear in theta, holomorphic
end

# Complex-step works perfectly for this component
g_cs = complex_step_grad(Z_contrib, theta0[p]; eps=1e-30)
```

---

## 4) Interpreting Results and Diagnosing Issues

### 4.1 Success Patterns

- **Small, stable relative errors ($< 10^{-6}$)**: Adjoint implementation correct
- **Error follows expected $h^2$ convergence**: Gradient consistent with discretized objective
- **Complex-step matches for holomorphic parts**: Subcomponent implementations correct

### 4.2 Failure Patterns and Diagnosis

1. **Large bias insensitive to $h$**
   - Likely cause: Adjoint formula or $\partial\mathbf{Z}/\partial\theta_p$ implementation error
   - Check: Patch mass matrix sign, adjoint system solve, gradient assembly

2. **Error decreases but plateaus above $10^{-4}$**
   - Likely cause: Inconsistent objective evaluation between adjoint and FD
   - Check: $\mathbf{Q}$ matrix reuse, excitation vector $\mathbf{v}$, linear solver tolerance

3. **Error increases monotonically as $h$ decreases**
   - Likely cause: Excessive round-off in FD due to noisy objective
   - Check: Linear solver convergence, objective conditioning

4. **Intermittent failures across parameters**
   - Likely cause: Parameter scaling issues or patch ordering mismatch
   - Check: $\theta_p$ units (Ω), patch indexing, mass matrix normalization

### 4.3 Paper's Verification Strategy

The paper's validation (Section IV) employs a multi-tier approach:

1. **Complex-step on holomorphic subproblems** to verify matrix derivative kernels
2. **Central finite differences ($h=10^{-5}$)** for end-to-end objective verification
3. **Convergence study** across mesh refinement to confirm gradient consistency
4. **Reciprocity and energy checks** to ensure physical correctness

This layered strategy catches implementation errors at different levels.

---

## 5) Advanced Topics

### 5.1 Ratio Objective Gradient Verification

For ratio objectives $J = N/D$, the quotient rule gradient requires verification of both numerator and denominator gradients:

```julia
# Verify numerator gradient
gN_adj = compute_numerator_gradient(...)
results_N = verify_gradient(J_numerator, gN_adj, theta0)

# Verify denominator gradient  
gD_adj = compute_denominator_gradient(...)
results_D = verify_gradient(J_denominator, gD_adj, theta0)

# Combined ratio gradient uses both via quotient rule
```

### 5.2 Mesh Refinement Consistency

Gradient accuracy should remain stable under mesh refinement:

```julia
for Nx in [2, 3, 4, 5, 6, 8]
    mesh = make_rect_plate(Lx, Ly, Nx, Nx)
    # ... assemble and solve
    max_err = verify_gradient(...) |> maximum
    println("Nx=$Nx: max gradient error = $max_err")
end
```

**Expected**: Maximum relative error $< 10^{-5}$ across all mesh levels.

### 5.3 Automated Regression Testing

The test suite (`test/runtests.jl`) includes gradient verification tests that should pass before any changes:

```julia
@testset "Gradient verification" begin
    results = verify_gradient(J_of_theta, g_adj, theta0; indices=1:5)
    for r in results
        @test r.rel_err_fd < 1e-5
    end
end
```

---

## 6) Code Mapping

### 6.1 Primary Implementation Files

- **Verification utilities**: `src/optimization/Verification.jl`
  - `fd_grad`, `complex_step_grad`, `verify_gradient`
  
- **Adjoint gradient computation**: `src/optimization/Adjoint.jl`
  - `gradient_impedance`, `solve_adjoint`

- **Optimization interface**: `src/optimization/Optimize.jl`
  - `optimize_directivity`, `optimize_lbfgs`

### 6.2 Example Scripts

- **Convergence study**: `validation/paper/run_convergence_study.jl` (includes gradient verification)
- **Beam steering optimization**: `examples/03_beamsteering_physical_unitcell.jl` (uses verified gradients)
- **Test suite**: `test/runtests.jl` (regression tests)

---

## 7) Exercises

### 7.1 Basic Level

1. **Run 10-parameter verification**:
   - Use the `verify_gradient` function on a small plate case
   - Report maximum relative error
   - Verify it's below $10^{-5}$ (paper threshold: $3\times 10^{-7}$)

2. **Step size sweep**:
   - Perform $h$-sweep from $10^{-3}$ to $10^{-8}$ for one parameter
   - Plot error vs. $h$ on log-log scale
   - Identify optimal $h^*$ and explain the V-shaped curve

### 7.2 Intermediate Level

3. **Mesh refinement consistency**:
   - Reproduce gradient verification across mesh sizes (2×2 to 8×8)
   - Confirm error remains below $10^{-4}$ for all meshes
   - Explain any trends with mesh density

4. **Complex-step limitations**:
   - Attempt complex-step on full objective $J = \mathbf{I}^\dagger\mathbf{Q}\mathbf{I}$
   - Observe the failure and explain why conjugation breaks holomorphicity
   - Verify complex-step works on linear subcomponent $\theta_p \mathbf{M}_p$

### 7.3 Advanced Level

5. **Ratio objective verification**:
   - Implement verification for numerator and denominator gradients separately
   - Verify the quotient rule combination matches finite differences
   - Compare computational cost: 2 adjoint solves vs. $P$ FD evaluations

6. **Failure diagnosis**:
   - Intentionally introduce a sign error in `gradient_impedance`
   - Document the verification failure pattern
   - Propose debugging steps to locate the error

---

## 8) Chapter Checklist

Before trusting optimization results, ensure you can:

- [ ] Run finite-difference verification with $h=10^{-5}$
- [ ] Achieve relative errors $< 10^{-5}$ (paper: $< 3\times 10^{-7}$)
- [ ] Perform $h$-sweep and identify optimal step size
- [ ] Understand complex-step limitations for real-valued objectives
- [ ] Verify gradient consistency across mesh refinement
- [ ] Interpret failure patterns and diagnose implementation errors

---

## 9) Further Reading

- **Finite-difference error analysis**: Nocedal & Wright, *Numerical Optimization* (2006), Section 8.1
- **Complex-step method**: Martins et al., *The Complex-Step Derivative Approximation* (2003)
- **Adjoint verification in EM**: M. B. Giles & N. A. Pierce, *An Introduction to the Adjoint Approach to Design* (2000)
- **Paper reference**: `bare_jrnl.tex`, Section IV (gradient verification results, Table 1)
- **Julia AD ecosystem**: `ForwardDiff.jl`, `ReverseDiff.jl`, `Zygote.jl` (alternative verification approaches)
