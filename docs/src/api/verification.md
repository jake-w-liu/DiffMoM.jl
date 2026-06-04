# API: Verification

## Purpose

Reference for gradient verification tools. These functions compare the adjoint gradient against independent numerical references (complex-step and finite-difference) to confirm correctness. Gradient verification is essential whenever you modify the adjoint code, add a new objective function, or change the impedance parameterization.

---

## Why Verify Gradients?

The adjoint method gives exact gradients in theory, but implementation errors (sign mistakes, missing conjugates, wrong reactive/resistive mode) can produce gradients that are subtly wrong. A wrong gradient will cause the optimizer to converge to the wrong solution or diverge, often without any obvious error message.

Two independent numerical references are available:

| Method | Accuracy | Applicability | Cost |
|--------|----------|---------------|------|
| Complex-step | Machine precision (~1e-15) | Only for holomorphic parameter paths (no conjugation in objective) | 1 solve per parameter |
| Central finite-difference | ~h^2 accuracy (~1e-8 for h=1e-6) | Always works, including objectives with conjugation (`I' Q I`) | 2 solves per parameter |

For the standard MoM objective `J = Re(I' Q I)`, which involves conjugation, **central finite-difference is the primary reference**. Complex-step requires a holomorphic formulation and is mainly useful for verifying intermediate quantities.

---

## Gradient Check Helpers

### `complex_step_grad(f, theta, p; eps=1e-30)`

Complex-step derivative of a scalar function `f(theta)` with respect to parameter `theta_p`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `f` | `Function` | -- | Scalar-valued function `f(theta)` that must accept `ComplexF64` input. The function must be holomorphic in the perturbed parameter for the result to be valid. |
| `theta` | `Vector{Float64}` | -- | Parameter vector at which to evaluate the derivative. |
| `p` | `Int` | -- | Index of the parameter to differentiate (1-based). |
| `eps` | `Float64` | `1e-30` | Complex-step perturbation magnitude. The default `1e-30` is well below machine precision and introduces no round-off error (unlike finite differences). |

**Returns:** `Float64` approximation of `df/d(theta_p)`.

**Formula:** `df/d(theta_p) = Im[f(theta + i*eps*e_p)] / eps`

**When to use:** Complex-step gives machine-precision derivatives, but only when `f` is holomorphic in the perturbed parameter. For `J = Re(I' Q I)`, the conjugation makes the objective non-holomorphic, so complex-step is not directly applicable to the full objective. Use `fd_grad` with central differences instead.

---

### `fd_grad(f, theta, p; h=1e-6, scheme=:central)`

Finite-difference derivative of `f(theta)` with respect to `theta_p`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `f` | `Function` | -- | Scalar-valued function `f(theta)` (must return a real value). |
| `theta` | `Vector{Float64}` | -- | Parameter vector. |
| `p` | `Int` | -- | Parameter index to differentiate. |
| `h` | `Float64` | `1e-6` | Step size. Central differences have error ~h^2, so `h=1e-6` gives ~1e-12 truncation error. Forward differences have error ~h, so accuracy is much lower. |
| `scheme` | `Symbol` | `:central` | `:central` (recommended) or `:forward`. |

**Returns:** `Float64` FD approximation of `df/d(theta_p)`.

**Formulas:**
- `:central`: `(f(theta + h*e_p) - f(theta - h*e_p)) / (2h)` -- O(h^2) accurate
- `:forward`: `(f(theta + h*e_p) - f(theta)) / h` -- O(h) accurate

**Choosing `h`:**
- Too small: round-off error dominates (the numerator becomes the difference of nearly equal numbers).
- Too large: truncation error dominates.
- `h = 1e-6` is a good default for central differences.
- `h = 1e-4` to `1e-5` for forward differences.

---

### `verify_gradient(f_objective, adjoint_grad, theta; indices=nothing, eps_cs=1e-30, h_fd=1e-6)`

Compare an adjoint gradient against both complex-step and finite-difference references. This is the main verification entry point.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `f_objective` | `Function` | -- | Objective function `theta -> J(theta)`. Must accept `ComplexF64` input for the complex-step check. |
| `adjoint_grad` | `Vector{Float64}` | -- | Adjoint gradient vector `g` of length P. |
| `theta` | `Vector{Float64}` | -- | Parameter vector at which the gradient was computed. |
| `indices` | Collection or `nothing` | `nothing` | Which parameters to check. Default: all parameters. For large P, check a subset (e.g., `1:10`) to save time. |
| `eps_cs` | `Float64` | `1e-30` | Complex-step perturbation magnitude. |
| `h_fd` | `Float64` | `1e-6` | Finite-difference step size. |

**Returns:** `Vector{NamedTuple}` with elements `(p, adj, cs, fd, rel_err_cs, rel_err_fd)`:
- `p::Int`: Parameter index.
- `adj::Float64`: Adjoint gradient value.
- `cs::Float64`: Complex-step derivative.
- `fd::Float64`: Finite-difference derivative.
- `rel_err_cs::Float64`: `|adj - cs| / max(|cs|, 1e-30)`.
- `rel_err_fd::Float64`: `|adj - fd| / max(|cs|, 1e-30)`.

**Interpreting results:**

| `rel_err_fd` | Interpretation |
|-------------|----------------|
| < 1e-4 | Gradient is correct. The residual error is from FD truncation. |
| 1e-4 to 1e-2 | Suspicious. Check `h_fd` choice, or there may be a subtle bug. |
| > 1e-2 | Gradient is likely wrong. Debug the adjoint code. |

---

## Typical Pattern

```julia
# 1. Define the objective function (must support ComplexF64 for complex-step)
function f_obj(theta)
    Z = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
    I = Z \ v
    return real(dot(I, Q * I))
end

# 2. Compute adjoint gradient at theta0
Z0 = assemble_full_Z(Z_efie, Mp, theta0; reactive=true)
I0 = Z0 \ v
lambda0 = solve_adjoint(Z0, Q, I0)
g_adj = gradient_impedance(Mp, I0, lambda0; reactive=true)

# 3. Verify
res = verify_gradient(f_obj, g_adj, theta0; indices=1:10, h_fd=1e-5)
max_err = maximum(r.rel_err_fd for r in res)
println("Max relative error vs FD: ", max_err)

# 4. Inspect individual parameters
for r in res
    println("p=$(r.p): adj=$(r.adj), fd=$(r.fd), rel_err=$(r.rel_err_fd)")
end
```

---

## Notes

- **Complex-step** is the gold standard for holomorphic functions but is not directly applicable to the standard MoM objective `J = Re(I' Q I)` due to conjugation. Use it for intermediate checks (e.g., verifying that `assemble_Z_efie` supports complex `k`).
- **Central finite-difference** is the primary reference for the full adjoint gradient. The `1e-6` default step size gives ~1e-8 accuracy, which is more than sufficient to catch sign errors or missing terms.
- For large P, check a random subset of parameters rather than all P (each FD check requires 2 extra solves).

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/optimization/Verification.jl` | `complex_step_grad`, `fd_grad`, `verify_gradient` |
| `src/optimization/Adjoint.jl` | Adjoint gradient code being verified |

---

## Exercises

- **Basic:** Compute `fd_grad` for one parameter and compare with the adjoint gradient value.
- **Practical:** Run `verify_gradient` for 10 parameters with both `h_fd=1e-5` and `h_fd=1e-7`. Compare the relative errors to understand the effect of step size.
- **Challenge:** Verify gradients for both resistive (`reactive=false`) and reactive (`reactive=true`) modes. Confirm that using the wrong `reactive` flag produces a large gradient error.
