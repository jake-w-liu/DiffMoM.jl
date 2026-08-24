# Gradient verification

Gradient validation in this repository compares the adjoint derivative of a
real quadratic objective with central finite differences. Complex-step is used
only for scalar functions that satisfy its holomorphic contract.

## Quantities being compared

For impedance parameters,

```math
J=\operatorname{Re}(I^\dagger QI),\qquad Z(\theta)I=v,
```

and the adjoint equation is

```math
Z^\dagger\lambda=QI.
```

The package evaluates

```math
g_p^{\mathrm{adj}}=-2\operatorname{Re}\left\{
\lambda^\dagger\frac{\partial Z}{\partial\theta_p}I
\right\}.
```

`gradient_impedance` uses $\partial Z/\partial\theta_p=-M_p$ for resistive
parameters and $-iM_p$ for reactive parameters.

The central finite-difference reference is

```math
g_p^{\mathrm{FD}}\approx
\frac{J(\theta+he_p)-J(\theta-he_p)}{2h}.
```

The executable convergence driver compares the two with the symmetric error

```math
\epsilon_p=
\frac{|g_p^{\mathrm{adj}}-g_p^{\mathrm{FD}}|}
     {\max(|g_p^{\mathrm{adj}}|,|g_p^{\mathrm{FD}}|)},
```

where the error is zero when both values are zero.

## Run the repository validation

From the repository root:

```bash
julia --project=. --startup-file=no validation/paper/run_convergence_study.jl
```

The driver writes:

- `data/convergence_study.csv`, containing mesh-level objective, gradient, and
  energy data; and
- `data/gradient_verification.csv`, containing the fixed reference gradient
  comparison.

The executable gates cover the maximum mesh-sweep gradient error, the maximum
fixed-reference gradient error, and the maximum absolute PEC energy-ratio
error. Their authoritative values are the `CONVERGENCE_MAX_*` constants in
`validation/paper/run_convergence_study.jl`; the verification labels are built
from those constants.

The command exits nonzero if a value is non-finite or a gate fails. Preserve
the generated CSV files and command output when reporting the result.

## Check a custom objective

For a real objective that contains conjugation, use `fd_grad` directly:

```julia
function objective(theta::Vector{Float64})
    Z = assemble_full_Z(Z_efie, Mp, theta; reactive=true)
    current = solve_forward(Z, v; solver=:direct)
    return compute_objective(current, Q)
end

Z = assemble_full_Z(Z_efie, Mp, theta0; reactive=true)
current = solve_forward(Z, v; solver=:direct)
adjoint = solve_adjoint(Z, Q, current; solver=:direct)
gradient = gradient_impedance(
    Mp, current, adjoint; reactive=true)

indices = 1:min(5, length(theta0))
for h in (1e-4, 1e-5, 1e-6, 1e-7)
    errors = map(indices) do p
        derivative = fd_grad(objective, theta0, p; h=h)
        scale = max(abs(gradient[p]), abs(derivative))
        iszero(scale) ? 0.0 : abs(gradient[p] - derivative) / scale
    end
    println((h=h, maximum_error=maximum(errors)))
end
```

Record the forward and adjoint true residuals along with this table. The useful
finite-difference step depends on the parameter scale, objective scale,
conditioning, and solve accuracy.

## Complex-step boundary

`complex_step_grad` computes

```math
\operatorname{Im}f(\theta+i\epsilon e_p)/\epsilon.
```

The supplied function must accept complex parameters, be holomorphic in the
perturbed parameter, return a real value at the real baseline, and remain
finite. The standard objective `real(dot(I, Q * I))` contains conjugation and
does not satisfy that contract.

`verify_gradient` always runs both complex-step and finite differences. Use it
for a holomorphic scalar test such as:

```julia
theta = [2.0, -1.0]
f = x -> x[1]^2 + 3x[2]
result = verify_gradient(f, [4.0, 3.0], theta)
```

Use the explicit `fd_grad` workflow for the end-to-end MoM quadratic.

## Evidence to retain

For each gradient result, record:

1. the parameterization and `reactive` flag;
2. the objective definition and Q-matrix construction;
3. the checked indices and their derivative magnitudes;
4. every finite-difference step and its error;
5. forward and adjoint true residuals;
6. mesh, quadrature, frequency, and excitation; and
7. any mass conditioning or GMRES preconditioner.

A low error at one step does not establish a stable interval. A persistent
sign or scale difference is evidence to inspect the derivative definition, but
it is not by itself proof of a specific implementation defect.

## Conditioning

Mass-based left conditioning changes the system and its derivative blocks:

```math
Z_{\mathrm{eff}}=M^{-1}Z,
\quad v_{\mathrm{eff}}=M^{-1}v,
\quad M_{p,\mathrm{eff}}=M^{-1}M_p.
```

Use `prepare_conditioned_system` and `transform_patch_matrices` with the same
factorization. A near-field GMRES preconditioner is a solver aid and does not
require this manual derivative-block transformation. In both cases, validate
the original-system true residual.

## Code map

| Task | Source |
|:--|:--|
| Numerical derivative helpers | `src/optimization/Verification.jl` |
| Adjoint solve and impedance gradient | `src/optimization/Adjoint.jl` |
| Mass conditioning | `src/solver/Solve.jl` |
| Executable convergence study | `validation/paper/run_convergence_study.jl` |
| Regression coverage | `test/runtests.jl` |

See [Tutorial 2](../tutorials/02-adjoint-gradient-check.md) for the complete
small-plate setup.
