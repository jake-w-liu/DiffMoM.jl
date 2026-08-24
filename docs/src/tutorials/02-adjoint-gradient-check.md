# Tutorial 2: Check an impedance adjoint gradient

This tutorial compares `gradient_impedance` with central finite differences on
a small plate. Run this check before using a new objective, parameterization,
or conditioning path in optimization.

## Gradient used by the package

For

```math
J=\operatorname{Re}(I^\dagger QI),\qquad Z(\theta)I=v,
```

the adjoint solves

```math
Z^\dagger\lambda=QI,
```

and each component is

```math
\frac{\partial J}{\partial\theta_p}
=-2\operatorname{Re}\left\{
\lambda^\dagger\frac{\partial Z}{\partial\theta_p}I
\right\}.
```

The implemented impedance matrices are

```math
Z_{\mathrm{res}}=Z_{\mathrm{EFIE}}-\sum_p\theta_pM_p,
\qquad
Z_{\mathrm{react}}=Z_{\mathrm{EFIE}}-i\sum_p\theta_pM_p.
```

Therefore, `gradient_impedance` returns

```math
g_p=2\operatorname{Re}(\lambda^\dagger M_pI)
```

for `reactive=false`, and

```math
g_p=-2\operatorname{Im}(\lambda^\dagger M_pI)
```

for `reactive=true`.

## Build a bounded test case

```julia
using DiffMoM
using LinearAlgebra
using Printf

frequency = 3.0e9
c0 = 299792458.0
k = 2π * frequency / c0
eta0 = 376.730313668

mesh = make_rect_plate(0.1, 0.1, 3, 3)
rwg = build_rwg(mesh)
N = rwg.nedges
Nt = ntriangles(mesh)

Z_efie = assemble_Z_efie(
    mesh, rwg, k; quad_order=3, eta0=eta0)

partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

k_vec = Vec3(0.0, 0.0, -k)
v = assemble_v_plane_wave(
    mesh,
    rwg,
    k_vec,
    1.0,
    Vec3(1.0, 0.0, 0.0);
    quad_order=3,
)

Q = Matrix{ComplexF64}(I, N, N)
theta0 = fill(10.0, Nt)
```

The identity Q matrix makes the objective the squared coefficient norm. It is
a numerical smoke objective, not a radiated-power definition.

## Compute the adjoint gradient

Choose one parameterization and use the same flag everywhere:

```julia
reactive = true

Z = assemble_full_Z(Z_efie, Mp, theta0; reactive=reactive)
current = solve_forward(Z, v; solver=:direct)
adjoint = solve_adjoint(Z, Q, current; solver=:direct)
g_adjoint = gradient_impedance(
    Mp, current, adjoint; reactive=reactive)

forward_residual = norm(Z * current - v) / norm(v)
adjoint_rhs = Q * current
adjoint_residual =
    norm(Z' * adjoint - adjoint_rhs) / norm(adjoint_rhs)

println((forward_residual, adjoint_residual, gradient_norm=norm(g_adjoint)))
```

Preserve both residuals with the gradient result. A gradient discrepancy cannot
be interpreted cleanly when either solve misses its declared gate.

## Define the independent objective

```julia
function objective(theta::Vector{Float64})
    local_Z = assemble_full_Z(
        Z_efie, Mp, theta; reactive=reactive)
    local_current = solve_forward(local_Z, v; solver=:direct)
    return compute_objective(local_current, Q)
end
```

`compute_objective` evaluates the same real quadratic form as the adjoint
derivation. It contains conjugation, so this end-to-end objective is not valid
for `complex_step_grad`.

## Compare several components

Use the symmetric error

```math
\epsilon_p=
\frac{|g_p^{\mathrm{adj}}-g_p^{\mathrm{FD}}|}
     {\max(|g_p^{\mathrm{adj}}|,|g_p^{\mathrm{FD}}|)},
```

with zero error when both values are zero:

```julia
function finite_difference_rows(f, gradient, theta, indices; h)
    return map(indices) do p
        g_fd = fd_grad(f, theta, p; h=h, scheme=:central)
        scale = max(abs(gradient[p]), abs(g_fd))
        error = iszero(scale) ? 0.0 :
            abs(gradient[p] - g_fd) / scale
        (; p, adjoint=gradient[p], finite_difference=g_fd, error)
    end
end

indices = 1:min(10, length(theta0))
rows = finite_difference_rows(
    objective, g_adjoint, theta0, indices; h=1e-5)

for row in rows
    @printf(
        "%2d  adj=% .6e  fd=% .6e  error=%.3e\n",
        row.p,
        row.adjoint,
        row.finite_difference,
        row.error,
    )
end
```

Do not select an acceptance threshold from one step. Repeat the comparison over
a range:

```julia
for h in (1e-4, 1e-5, 1e-6, 1e-7)
    rows = finite_difference_rows(
        objective, g_adjoint, theta0, indices; h=h)
    println((h=h, maximum_error=maximum(row.error for row in rows)))
end
```

For a smooth objective, central-difference truncation error is second order in
an asymptotic interval. Very small steps expose rounding and solve error. Use
the measured low-error interval and record every tested step.

## Resistive check

Repeat the workflow after setting `reactive=false`. Rebuild `Z`, the current,
the adjoint, the gradient, and the objective closure. Mixing the reactive flag
between those stages changes the derivative being tested.

## Conditioned systems

Mass-based conditioning changes the algebraic system to
$M^{-1}ZI=M^{-1}v$. Transform the derivative blocks by the same left solve:

```julia
M = make_left_preconditioner(Mp; eps_rel=1e-8)
Z_effective, v_effective, factor = prepare_conditioned_system(
    Z, v; preconditioner_M=M)
Mp_effective, _ = transform_patch_matrices(
    Mp;
    preconditioner_M=M,
    preconditioner_factor=factor,
)
```

Use `Z_effective`, `v_effective`, and `Mp_effective` together in the forward,
adjoint, and gradient calculations. The high-level optimizers perform this
transformation for their mass-conditioning options.

A near-field preconditioner passed to GMRES is different. It changes the
iterative solve, not the physical derivative blocks. Keep the original-system
true-residual checks enabled and do not transform `Mp` solely because GMRES is
preconditioned.

## Diagnose disagreement

Collect evidence in this order:

1. Confirm the same `reactive` flag, Q matrix, excitation, and parameter vector.
2. Record forward and adjoint true residuals.
3. Run the finite-difference step sweep on a small direct-solve case.
4. Check whether the discrepancy is a sign or constant scaling factor.
5. Compare the unconditioned path before adding one conditioning mechanism.
6. Recheck several parameters, including any near zero.

Complex-step is not a fallback for this quadratic objective. It is available
for a scalar function that is holomorphic in the perturbed parameter and real
at real inputs; see the [verification API](../api/verification.md).

## Repository check

`validation/paper/run_convergence_study.jl` runs the package's mesh-level and
reference gradient checks and applies executable symmetric-error gates:

```bash
julia --project=. --startup-file=no validation/paper/run_convergence_study.jl
```

The regression suite includes resistive, reactive, conditioning, and
verification-helper cases:

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
```

## Code map

| Task | Source |
|:--|:--|
| Adjoint solve and impedance gradient | `src/optimization/Adjoint.jl` |
| Finite-difference and complex-step helpers | `src/optimization/Verification.jl` |
| Impedance matrices | `src/assembly/Impedance.jl` |
| Full system and mass conditioning | `src/solver/Solve.jl` |
| Executable convergence gates | `validation/paper/run_convergence_study.jl` |
