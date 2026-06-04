# Conditioning and Preconditioning

## Purpose

This chapter documents the conditioning and preconditioning workflow that is
actually implemented in `DiffMoM.jl`.

It focuses on the current APIs in:

- `src/postprocessing/Diagnostics.jl`
- `src/solver/Solve.jl`
- `src/solver/NearFieldPreconditioner.jl`
- `src/optimization/Adjoint.jl`
- `src/optimization/Optimize.jl`

---

## Learning Goals

After this chapter, you should be able to:

1. Diagnose EFIE conditioning with `condition_diagnostics`.
2. Apply matrix-level regularization and left preconditioning correctly.
3. Use `select_preconditioner` modes (`:off`, `:on`, `:auto`) correctly.
4. Keep forward/adjoint operators consistent for correct gradients.
5. Distinguish matrix-level conditioning from GMRES near-field preconditioning.

---

## 1. Conditioning Diagnostics

The package provides:

```julia
stats = condition_diagnostics(Z)
```

which returns

- `stats.cond = sigma_max / sigma_min`
- `stats.sv_max`
- `stats.sv_min`

Implementation note: `condition_diagnostics` uses `svdvals(Z)` (full SVD), so it
is practical for moderate matrix sizes but expensive for large `N`.

---

## 2. Matrix-Level Conditioning in `Solve.jl`

The matrix assembled for impedance optimization is

```math
Z_{\text{raw}} = Z_{\text{EFIE}} - \sum_{p=1}^{P} \theta_p M_p,
```

from `assemble_full_Z(Z_efie, Mp, theta; reactive=false)`.

### 2.1 Regularization

`make_mass_regularizer(Mp)` builds

```math
R = \sum_{p=1}^{P} M_p,
```

as a dense Hermitian `ComplexF64` matrix.

`prepare_conditioned_system` applies optional regularization:

```math
Z_{\text{reg}} = Z_{\text{raw}} + \alpha R,
```

when `regularization_alpha != 0` and `regularization_R` is provided.

### 2.2 Left Preconditioning

`make_left_preconditioner(Mp; eps_rel=1e-8)` builds

```math
M = R + \epsilon I,
\quad
\epsilon = \texttt{eps_rel} \cdot \max(\operatorname{tr}(R)/N, 1).
```

If enabled, `prepare_conditioned_system` applies:

```math
Z_{\text{eff}} = M^{-1} Z_{\text{reg}},
\qquad
\text{rhs}_{\text{eff}} = M^{-1}\,\text{rhs}.
```

The LU factorization of `M` is returned for reuse.

### 2.3 Preconditioner Mode Selection

`select_preconditioner(Mp; mode=:off|:on|:auto, ...)` returns
`(M_eff, enabled, reason)`.

Behavior:

- `mode=:off`: no matrix-level preconditioner (unless user supplies one).
- `mode=:on`: always build mass-based `M`.
- `mode=:auto`: enable if `iterative_solver=true` or `N >= n_threshold`.

Important default detail:

- `make_left_preconditioner` default `eps_rel` is `1e-8`.
- `select_preconditioner` default `eps_rel` is `1e-6`.

---

## 3. Forward/Adjoint Consistency

For gradient correctness, forward and adjoint must use the same conditioned
operator.

Forward solve:

```math
Z_{\text{eff}} I = \text{rhs}_{\text{eff}}.
```

Adjoint solve (`solve_adjoint`):

```math
Z_{\text{eff}}^{\dagger} \lambda = Q I.
```

For impedance gradients, if left preconditioning is active, use transformed
patch blocks:

```julia
Mp_eff, _ = transform_patch_matrices(Mp; preconditioner_factor=fac)
g = gradient_impedance(Mp_eff, I, lambda; reactive=false)
```

This matches the implementation pattern in `optimize_lbfgs`.

---

## 4. GMRES Near-Field Preconditioning (Separate Layer)

`src/solver/NearFieldPreconditioner.jl` provides Krylov preconditioners for
`solve_forward(...; solver=:gmres)` and `solve_adjoint(...; solver=:gmres)`.

Key builder:

```julia
nf = build_nearfield_preconditioner(Z, mesh, rwg, cutoff;
    factorization=:lu,   # :lu | :ilu | :diag
    ilu_tau=1e-3,
    neighbor_search=:spatial)
```

Available data types include:

- `NearFieldPreconditionerData` (sparse LU)
- `ILUPreconditionerData`
- `DiagonalPreconditionerData`

This is distinct from matrix-level `M^{-1}` conditioning in `Solve.jl`; both can
be used together.

---

## 5. End-to-End Example (Code-Faithful)

```julia
using DiffMoM
using LinearAlgebra

# Geometry / basis
mesh = make_rect_plate(0.1, 0.1, 8, 8)
rwg = build_rwg(mesh)
partition = assign_patches_grid(mesh; nx=2, ny=2, nz=1)
Mp = precompute_patch_mass(mesh, rwg, partition)

# Physics
k = 2pi / 0.1
theta = zeros(ComplexF64, partition.P)
Z_efie = assemble_Z_efie(mesh, rwg, k)
Z_raw = assemble_full_Z(Z_efie, Mp, theta)

pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
rhs = assemble_excitation(mesh, rwg, pw)

# Diagnostics before conditioning
stats_raw = condition_diagnostics(Z_raw)
println("kappa raw = ", stats_raw.cond)

# Matrix-level conditioning
R = make_mass_regularizer(Mp)
M_eff, enabled, reason = select_preconditioner(
    Mp; mode=:auto, iterative_solver=true, n_threshold=256
)
println("preconditioning: ", enabled, " (", reason, ")")

Z_eff, rhs_eff, fac = prepare_conditioned_system(
    Z_raw,
    rhs;
    regularization_alpha=1e-10,
    regularization_R=R,
    preconditioner_M=M_eff,
)

# Forward / adjoint
I_coeffs = solve_forward(Z_eff, rhs_eff; solver=:direct)
Q = Matrix{ComplexF64}(LinearAlgebra.I, rwg.nedges, rwg.nedges)
lambda = solve_adjoint(Z_eff, Q, I_coeffs; solver=:direct)

# Gradient with transformed patch matrices (if left preconditioning active)
Mp_eff, _ = transform_patch_matrices(Mp; preconditioner_factor=fac)
g = gradient_impedance(Mp_eff, I_coeffs, lambda; reactive=false)
```

### GMRES variant with near-field preconditioner

```julia
nf = build_nearfield_preconditioner(Z_eff, mesh, rwg, 0.05;
    factorization=:ilu, ilu_tau=1e-2)

I_gmres = solve_forward(Z_eff, rhs_eff;
    solver=:gmres, preconditioner=nf, gmres_tol=1e-8, gmres_maxiter=300)

lambda_gmres = solve_adjoint(Z_eff, Q, I_gmres;
    solver=:gmres, preconditioner=nf, gmres_tol=1e-8, gmres_maxiter=300)
```

---

## 6. Common Failure Modes

- `regularization_alpha != 0` with `regularization_R === nothing`
  - `prepare_conditioned_system` throws by design.

- Using conditioned forward solve but unconditioned adjoint solve
  - causes gradient mismatch.

- Using raw `Mp` with preconditioned `Z_eff`
  - gradient should use `Mp_eff = M^{-1} Mp` via `transform_patch_matrices`.

- Confusing matrix-level and Krylov preconditioners
  - `preconditioner_M` in `prepare_conditioned_system` is not the same as
    `preconditioner=nf` in GMRES.

---

## 7. Code Map

| Topic | File | API |
|---|---|---|
| Conditioning diagnostics | `src/postprocessing/Diagnostics.jl` | `condition_diagnostics` |
| Matrix assembly | `src/solver/Solve.jl` | `assemble_full_Z` |
| Regularizer/preconditioner matrices | `src/solver/Solve.jl` | `make_mass_regularizer`, `make_left_preconditioner`, `select_preconditioner` |
| Conditioned operator build | `src/solver/Solve.jl` | `prepare_conditioned_system`, `transform_patch_matrices` |
| Forward linear solves | `src/solver/Solve.jl` | `solve_forward`, `solve_system` |
| GMRES near-field preconditioners | `src/solver/NearFieldPreconditioner.jl` | `build_nearfield_preconditioner` |
| Adjoint/gradient | `src/optimization/Adjoint.jl` | `solve_adjoint`, `solve_adjoint_rhs`, `gradient_impedance` |
| Optimization loop usage | `src/optimization/Optimize.jl` | `optimize_lbfgs`, `optimize_directivity` |

---

## 8. Chapter Checklist

- [ ] Compute `condition_diagnostics(Z)` and interpret `cond`, `sv_max`, `sv_min`.
- [ ] Build `R` with `make_mass_regularizer(Mp)`.
- [ ] Select matrix-level preconditioning via `select_preconditioner`.
- [ ] Build `(Z_eff, rhs_eff, fac)` with `prepare_conditioned_system`.
- [ ] Use the same `Z_eff` in `solve_forward` and `solve_adjoint`.
- [ ] Use `transform_patch_matrices` before `gradient_impedance` when left preconditioning is active.
- [ ] Distinguish matrix-level conditioning from GMRES near-field preconditioning.

---

*Next: [Excitation Theory and Implementation](06-excitation-theory-and-usage.md).* 
