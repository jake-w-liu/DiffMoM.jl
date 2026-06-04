# Density Topology Optimization Implementation

## Purpose

This chapter documents the density-based topology implementation added to `DiffMoM.jl` and its exact code-formulation mapping:

1. SIMP-like per-triangle impedance penalty (`DensityConfig`, `precompute_triangle_mass`, `assemble_Z_penalty`),
2. conic filtering + smooth Heaviside projection (`build_filter_weights`, `filter_and_project`),
3. adjoint gradients through the full density chain (`gradient_density`, `gradient_density_full`).

The objective is technical correctness: equations, defaults, and signs match source implementation in `src/assembly/DensityInterpolation.jl`, `src/optimization/DensityFiltering.jl`, and `src/optimization/DensityAdjoint.jl`.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain the role of raw, filtered, and projected densities (`rho`, `rho_tilde`, `rho_bar`).
2. Derive the penalty matrix and its derivative with respect to projected densities.
3. Build and apply the conic filter and projection stages used in the code.
4. Derive the adjoint gradient and chain-rule backpropagation to raw density variables.
5. Reproduce the implementation pipeline and verify gradients against finite differences.
6. Map every formula to a specific source file and regression test.

---

## 1. Parameterization and SIMP Penalty

### 1.1 Triangle-Wise Design Variables

Each triangle `t` carries a density variable `rho_t in [0,1]`:

- `rho_t = 1`: metal-like (low penalty),
- `rho_t = 0`: void-like (high penalty).

### 1.2 Configuration Type

`DensityConfig` stores:

```julia
struct DensityConfig
    p::Float64
    Z_max::Float64
    vf_target::Float64
end
```

with constructor defaults:

```julia
DensityConfig(; p=3.0, Z_max_factor=1000.0, eta0=376.730313668, vf_target=0.5)
```

and `Z_max = Z_max_factor * eta0` in code.

### 1.3 Penalty Matrix

For projected density `rho_bar`, the penalty matrix is:

```math
\mathbf{Z}_{\text{pen}} = \sum_{t=1}^{N_t}\left(1 - \rho_{\!bar,t}^{\,p}\right) Z_{\max}\,\mathbf{M}_t,
```

where `M_t` is the per-triangle mass matrix.

This is implemented by `assemble_Z_penalty(Mt, rho_bar, config)` and returns a dense `Matrix{ComplexF64}`.

---

## 2. Triangle Mass Matrices

`precompute_triangle_mass(mesh, rwg; quad_order=3)` computes:

```math
[\mathbf{M}_t]_{mn} = \int_{T_t}\mathbf{f}_m(\mathbf{r})\cdot\mathbf{f}_n(\mathbf{r})\,dS.
```

Implementation details:

1. uses `tri_quad_rule(quad_order)` and mapped points `tri_quad_points(...)`,
2. includes only basis functions with support on triangle `t`,
3. multiplies quadrature sum by `2*A_t` (reference-to-physical Jacobian),
4. stores each `M_t` as sparse (`spzeros(Float64, N, N)` then fill).

---

## 3. Density Filtering and Projection

### 3.1 Conic Filter

`build_filter_weights(mesh, r_min)` builds:

```math
W_{ts} = \max(0, r_{\min} - \|\mathbf{c}_t - \mathbf{c}_s\|),
```

with centroids `c_t`, and row sums:

```math
w_{\text{sum},t} = \sum_s W_{ts}.
```

Filtered density:

```math
\rho_{\tilde{t}} = \frac{\sum_s W_{ts}\rho_s}{w_{\text{sum},t}}.
```

Code path:
- forward map: `apply_filter(W, w_sum, rho)`,
- adjoint transpose map: `apply_filter_transpose(W, w_sum, g_rho_tilde) = W' * (g_rho_tilde ./ w_sum)`.

### 3.2 Smooth Heaviside Projection

`heaviside_project(rho_tilde, beta, eta=0.5)` applies:

```math
\rho_{\!bar} =
\frac{\tanh(\beta\eta) + \tanh\!\left(\beta(\rho_{\tilde{}}-\eta)\right)}
     {\tanh(\beta\eta) + \tanh\!\left(\beta(1-\eta)\right)}.
```

Derivative used in backpropagation:

```math
\frac{d\rho_{\!bar}}{d\rho_{\tilde{}}}
=
\frac{\beta\left(1-\tanh^2(\beta(\rho_{\tilde{}}-\eta))\right)}
     {\tanh(\beta\eta)+\tanh(\beta(1-\eta))}.
```

Implemented by `heaviside_derivative(...)`.

---

## 4. Sensitivity Matrices and Adjoint Gradient

### 4.1 Matrix Derivative

`assemble_dZ_drhobar(Mt, rho_bar, config, t)` returns:

```math
\frac{\partial \mathbf{Z}_{\text{pen}}}{\partial \rho_{\!bar,t}}
= -p\,\rho_{\!bar,t}^{\,p-1}\,Z_{\max}\,\mathbf{M}_t.
```

### 4.2 Gradient w.r.t. Projected Density

Given forward current `I` and adjoint variable `lambda`, the implemented gradient is:

```math
g_t
=
\frac{\partial J}{\partial \rho_{\!bar,t}}
=
-2\,\Re\!\left\{\lambda^\dagger
\left(\frac{\partial \mathbf{Z}}{\partial \rho_{\!bar,t}}\right)I\right\}
=
2p\,Z_{\max}\,\rho_{\!bar,t}^{\,p-1}\,\Re\!\left\{\lambda^\dagger \mathbf{M}_t I\right\}.
```

`gradient_density(...)` implements this formula directly.

### 4.3 Chain Rule to Raw Density

`gradient_density_full(...)` computes:

1. `g_rho_bar = gradient_density(...)`,
2. `g_rho = gradient_chain_rule(g_rho_bar, rho_tilde, W, w_sum, beta; eta=...)`.

This gives `dJ/drho` for raw optimization variables.

---

## 5. End-to-End Implementation Pattern

```julia
cfg = DensityConfig(; p=3.0, Z_max_factor=1000.0, vf_target=0.5)

# Precompute structural terms
Mt = precompute_triangle_mass(mesh, rwg; quad_order=3)
W, w_sum = build_filter_weights(mesh, r_min)

# Forward density pipeline
rho_tilde, rho_bar = filter_and_project(W, w_sum, rho, beta, eta)
Z_pen = assemble_Z_penalty(Mt, rho_bar, cfg)
Z = Z_efie + Z_pen

# Forward + adjoint
I = solve_forward(Z, v)
lambda = solve_adjoint(Z, Q, I)

# Raw-density gradient
g_rho = gradient_density_full(Mt, I, lambda, rho_tilde, rho_bar, cfg, W, w_sum, beta; eta=eta)
```

---

## 6. Complexity and Practical Notes

1. `build_filter_weights` is O(`Nt^2`) due to all-pairs centroid distances.
2. `precompute_triangle_mass` is expensive for large `N`/`Nt` because each `M_t` is local but assembled explicitly.
3. Large `Z_max` improves material contrast but can worsen linear-system conditioning.
4. `gradient_density_full` is only correct when `rho_tilde/rho_bar` come from the same `(W, w_sum, beta, eta)` used in the forward pass.

---

## 7. Validation and Correspondence Checks

Density implementation is directly regression-tested in `test/test_periodic_topology.jl`:

- Test 38: `DensityInterpolation` (`DensityConfig`, mass matrices, `assemble_Z_penalty`, `assemble_dZ_drhobar`),
- Test 39: `DensityFiltering`,
- Test 40: `DensityAdjoint`, including FD-vs-adjoint gradient checks.

The test enforces adjoint-vs-FD agreement with relative error `< 1e-4` on sampled variables, providing a direct formulation-to-code regression check.

---

## 8. Code Mapping

| Concept | Source file | Key function / type |
|---------|-------------|---------------------|
| Density config and SIMP penalty | `src/assembly/DensityInterpolation.jl` | `DensityConfig`, `assemble_Z_penalty`, `assemble_dZ_drhobar` |
| Triangle mass matrices | `src/assembly/DensityInterpolation.jl` | `precompute_triangle_mass` |
| Conic filtering | `src/optimization/DensityFiltering.jl` | `build_filter_weights`, `apply_filter`, `apply_filter_transpose` |
| Heaviside projection | `src/optimization/DensityFiltering.jl` | `heaviside_project`, `heaviside_derivative` |
| Full density pipeline | `src/optimization/DensityFiltering.jl` | `filter_and_project`, `gradient_chain_rule` |
| Adjoint density gradients | `src/optimization/DensityAdjoint.jl` | `gradient_density`, `gradient_density_full` |

---

## 9. Exercises

### Conceptual

1. Explain why filtering and projection are separated into two stages instead of one map.
2. Show how increasing `beta` affects `d rho_bar / d rho_tilde` near `rho_tilde = eta`.
3. Discuss conditioning trade-offs when `Z_max_factor` is increased from 100 to 5000.

### Coding

1. Verify `assemble_dZ_drhobar` by finite-difference perturbation of one `rho_bar[t]`.
2. Run a `beta` continuation (`1 -> 4 -> 16 -> 64`) and track binarization metrics for `rho_bar`.
3. Reproduce `gradient_density_full` FD checks on a random subset of design variables and report max relative error.

---

## 10. Chapter Checklist

- [ ] Construct and interpret `DensityConfig` defaults (`p`, `Z_max`, `vf_target`).
- [ ] Build per-triangle mass matrices and explain their sparsity pattern.
- [ ] Apply conic filter + Heaviside projection and identify all intermediate variables.
- [ ] Derive and evaluate `dZ/drho_bar[t]`.
- [ ] Compute adjoint gradient w.r.t. projected density and chain it to raw density.
- [ ] Verify gradient correctness against finite differences.
- [ ] Map each step to its corresponding source file and test block.

---

## 11. Further Reading

- Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications*.
- Lazarov, B. S., & Sigmund, O. (2011). Filters in topology optimization.
- Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence, and robust formulations.
- Tucek, P., Capek, M., & Jelinek, L. (2023). Topology optimization formulations for electromagnetic surface design.

---

*Next: [Internal Consistency Validation](../validation/01-internal-consistency.md) for end-to-end solver and gradient verification workflows.*
