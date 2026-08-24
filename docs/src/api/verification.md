# API: gradient verification

Use these functions to compare an analytical or adjoint gradient with a
numerical derivative. The finite-difference path applies to real scalar
objectives. The complex-step path has a narrower holomorphic contract.

## `fd_grad`

```julia
fd_grad(f, theta, p; h=1e-6, scheme=:central)
```

`theta` must be a `Vector{Float64}`, `p` must index it, and `h` must be finite
and positive. `f(theta)` must return a finite scalar number.

Supported schemes are:

- `:central`, which evaluates `f` at the nearest representable values for
  `theta[p] + h` and `theta[p] - h`; and
- `:forward`, which evaluates the positive perturbation and the baseline.

The implementation rejects a step that does not change `theta[p]` in
`Float64` arithmetic. Central differences have second-order truncation error
for a sufficiently smooth function, but the useful step size also depends on
rounding, objective scale, conditioning, and solve accuracy. Check a range of
steps instead of treating the default as an accuracy guarantee.

```julia
function symmetric_error(first, second)
    scale = max(abs(first), abs(second))
    return iszero(scale) ? 0.0 : abs(first - second) / scale
end

steps = (1e-4, 1e-5, 1e-6, 1e-7)
rows = [begin
    derivative = fd_grad(objective, theta, p; h=step)
    (; step, derivative,
       error=symmetric_error(adjoint_gradient[p], derivative))
end for step in steps]
```

## `complex_step_grad`

```julia
complex_step_grad(f, theta, p; eps=1e-30)
```

This function evaluates

```math
\frac{\operatorname{Im} f(\theta+i\epsilon e_p)}{\epsilon}.
```

Its input contract is stricter than the finite-difference contract:

1. `f` must accept a complex-valued parameter vector.
2. `f` must be holomorphic in the perturbed parameter.
3. `f` must be real at the supplied real parameter vector.
4. Every evaluated value must be finite.

If the requested perturbation produces an exactly zero imaginary response,
the implementation increases the step by exact powers of two. It returns after
two successive nonzero derivative estimates agree exactly, or throws when the
response remains inconclusive.

This self-contained example satisfies the contract:

```julia
theta = [2.0, -1.0]
f = x -> x[1]^2 + 3x[2]

@assert isapprox(complex_step_grad(f, theta, 1), 4.0; rtol=1e-12)
@assert isapprox(complex_step_grad(f, theta, 2), 3.0; rtol=1e-12)
```

Do not apply complex-step directly to an objective such as
`real(dot(I, Q * I))`. Its conjugation makes the end-to-end function
nonholomorphic. Use `fd_grad` for that objective.

## `verify_gradient`

```julia
verify_gradient(
    f_objective,
    adjoint_grad,
    theta;
    indices=nothing,
    eps_cs=1e-30,
    h_fd=1e-6,
)
```

`adjoint_grad` and `theta` must be finite `Vector{Float64}` values of equal
length. `indices=nothing` checks every component; otherwise, `indices` must be
an iterable of valid integer indices.

The function always runs both complex-step and central finite differences. Use
it only when `f_objective` satisfies the complex-step contract. It returns one
named tuple per checked index with fields:

| Field | Meaning |
|:--|:--|
| `p` | Checked parameter index |
| `adj` | Supplied gradient component |
| `cs` | Complex-step derivative |
| `fd` | Central finite-difference derivative |
| `rel_err_cs` | Error relative to the complex-step reference |
| `rel_err_fd` | Symmetric adjoint versus finite-difference error |

```julia
theta = [2.0, -1.0]
gradient = [4.0, 3.0]
f = x -> x[1]^2 + 3x[2]

result = verify_gradient(f, gradient, theta)
@assert maximum(row.rel_err_fd for row in result) < 1e-8
@assert maximum(row.rel_err_cs for row in result) < 1e-12
```

For the standard quadratic MoM objective, run only the finite-difference
comparison:

```julia
indices = 1:min(10, length(theta0))
rows = map(indices) do p
    derivative = fd_grad(objective, theta0, p; h=1e-5)
    scale = max(abs(g_adjoint[p]), abs(derivative))
    error = iszero(scale) ? 0.0 :
        abs(g_adjoint[p] - derivative) / scale
    (; p, adjoint=g_adjoint[p], finite_difference=derivative, error)
end
```

## Interpreting a comparison

Declare the acceptance threshold for the specific validation case. Preserve
the checked indices, step sizes, derivative values, forward and adjoint true
residuals, and the relative-error definition. A stable low-error interval over
several steps is stronger evidence than one selected step.

When the values disagree, first compare:

- resistive versus reactive parameterization;
- the sign and scaling of each derivative block;
- the forward and adjoint equations and their true residuals;
- the exact Q matrix, excitation, and parameter vector used by both paths; and
- any algebraic conditioning applied to the system and derivative blocks.

## Source

| Area | File |
|:--|:--|
| Numerical derivative helpers | `src/optimization/Verification.jl` |
| Adjoint solve and impedance gradient | `src/optimization/Adjoint.jl` |
| System conditioning and derivative transforms | `src/solver/Solve.jl` |

See [Tutorial 2](../tutorials/02-adjoint-gradient-check.md) for an end-to-end
impedance-gradient check.
