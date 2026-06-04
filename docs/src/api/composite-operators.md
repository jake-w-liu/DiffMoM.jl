# API: Composite Operators

## Purpose

Reference for the `ImpedanceLoadedOperator`, a matrix-free composite operator that wraps any `AbstractMatrix{ComplexF64}` base operator (MLFMA, ACA, or dense) with sparse impedance perturbation. This enables GMRES-based optimization with fast operators without ever forming the full dense impedance-loaded system matrix.

The key equation:

```
Z(theta) = Z_base - Sigma_p coeff_p * M_p
```

where `coeff_p = theta_p` (resistive) or `coeff_p = i*theta_p` (reactive), and `M_p` are sparse patch mass matrices from `precompute_patch_mass`.

---

## `ImpedanceLoadedOperator`

### Type Definition

```julia
struct ImpedanceLoadedOperator{T<:AbstractMatrix{ComplexF64},
                                S<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    Z_base::T
    Mp::Vector{S}
    theta::Vector{Float64}
    reactive::Bool
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `Z_base` | `AbstractMatrix{ComplexF64}` | Base EFIE operator. Can be `MLFMAOperator`, `ACAOperator`, `MatrixFreeEFIEOperator`, or dense `Matrix{ComplexF64}`. |
| `Mp` | `Vector{<:AbstractMatrix}` | Sparse patch mass matrices from `precompute_patch_mass`. Length P (number of design patches). |
| `theta` | `Vector{Float64}` | Current impedance parameter vector (length P). |
| `reactive` | `Bool` | `false` = resistive loading (`Z_s = theta`), `true` = reactive loading (`Z_s = i*theta`). |

### Constructor

```julia
ImpedanceLoadedOperator(Z_base, Mp, theta, reactive=false)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Z_base` | `AbstractMatrix{ComplexF64}` | -- | Any EFIE operator (MLFMA, ACA, dense, matrix-free). |
| `Mp` | `Vector{<:AbstractMatrix}` | -- | Patch mass matrices. |
| `theta` | `Vector{Float64}` | -- | Impedance parameters. |
| `reactive` | `Bool` | `false` | Loading mode. |

**Example:**

```julia
# Dense base operator
Z_efie = assemble_Z_efie(mesh, rwg, k)
Mp = precompute_patch_mass(mesh, rwg, partition)
theta = zeros(partition.P)
Z_op = ImpedanceLoadedOperator(Z_efie, Mp, theta)

# MLFMA base operator
mlfma = build_mlfma_operator(mesh, rwg, k; leaf_lambda=1.0)
Z_op = ImpedanceLoadedOperator(mlfma, Mp, theta)

# ACA base operator
aca = build_aca_operator(mesh, rwg, k)
Z_op = ImpedanceLoadedOperator(aca, Mp, theta)
```

---

### AbstractMatrix Interface

`ImpedanceLoadedOperator` implements the full `AbstractMatrix{ComplexF64}` interface required by Krylov.jl:

| Method | Description |
|--------|-------------|
| `size(A)` | Returns `size(A.Z_base)` — the N x N system dimension. |
| `eltype(A)` | Returns `ComplexF64`. |
| `A * x` | Forward matvec: computes `Z_base * x - Sigma_p coeff_p * (M_p * x)`. |
| `mul!(y, A, x)` | In-place forward matvec (preferred by Krylov.jl). |
| `adjoint(A)` | Returns `ImpedanceLoadedAdjointOperator` for the adjoint system. |

---

### Forward Matvec

```
y = Z(theta) * x = Z_base * x - Sigma_p coeff_p * (M_p * x)
```

The impedance contribution is O(nnz) per patch — negligible compared to the base operator's cost (O(N log N) for MLFMA, O(N log^2 N) for ACA).

For resistive loading (`reactive=false`): `coeff_p = theta_p`
For reactive loading (`reactive=true`): `coeff_p = i * theta_p`

---

### Adjoint Matvec

```
y = Z(theta)' * x = Z_base' * x - Sigma_p conj(coeff_p) * (M_p' * x)
```

For resistive loading: `conj(coeff_p) = theta_p` (real, unchanged)
For reactive loading: `conj(coeff_p) = -i * theta_p`

The adjoint operator is obtained via `adjoint(Z_op)` and is used internally by `solve_adjoint_rhs` and `solve_gmres_adjoint` for adjoint sensitivity solves.

---

## `ImpedanceLoadedAdjointOperator`

```julia
struct ImpedanceLoadedAdjointOperator{T<:AbstractMatrix{ComplexF64},
                                       S<:AbstractMatrix} <: AbstractMatrix{ComplexF64}
    parent::ImpedanceLoadedOperator{T,S}
end
```

The adjoint wrapper. Obtained via `adjoint(Z_op)`, not constructed directly. Implements `size`, `eltype`, `mul!`, `*`, and `adjoint` (which returns the original operator).

---

## Relationship to Existing Functions

`ImpedanceLoadedOperator` replaces `assemble_full_Z` for operator-based (non-dense) systems:

| Approach | Base operator | Impedance loading | Use case |
|----------|---------------|-------------------|----------|
| `assemble_full_Z(Z_efie, Mp, theta)` | Dense `Matrix{ComplexF64}` | Forms dense Z_efie + Z_imp | N < ~5000, direct solver |
| `ImpedanceLoadedOperator(Z_base, Mp, theta)` | Any `AbstractMatrix{ComplexF64}` | Matrix-free sparse perturbation | Any N, GMRES solver |

The composite operator is designed for the optimization loop: at each iteration, a new `ImpedanceLoadedOperator` is created with updated `theta`, and GMRES solves the forward/adjoint systems without re-assembling any dense matrices.

---

## Usage with GMRES

The composite operator works directly with `solve_forward` and `solve_adjoint_rhs`:

```julia
Z_op = ImpedanceLoadedOperator(Z_base, Mp, theta, reactive)

# Forward solve
I = solve_forward(Z_op, v; solver=:gmres,
                  preconditioner=P_nf,
                  gmres_tol=1e-6, gmres_maxiter=300)

# Adjoint solve (with pre-computed RHS)
rhs = Q * I
lambda = solve_adjoint_rhs(Z_op, rhs; solver=:gmres,
                            preconditioner=P_nf)
```

**Note:** Direct solver (`:direct`) is not supported with `ImpedanceLoadedOperator` since the operator is matrix-free. Always use `solver=:gmres`.

**Preconditioner reuse:** The near-field preconditioner is built from the base PEC EFIE matrix and reused throughout optimization, even as `theta` changes. This works because the near-field structure of the EFIE dominates the conditioning.

---

## Usage in Multi-Angle Optimization

`ImpedanceLoadedOperator` is used internally by `optimize_multiangle_rcs`:

```julia
# The optimizer creates a new composite operator each iteration:
Z_op = ImpedanceLoadedOperator(Z_base, Mp, theta_current, reactive)

# Then solves M forward + M adjoint systems (one per angle)
for a in 1:M
    I_a = solve_forward(Z_op, configs[a].v; solver=:gmres, ...)
end
for a in 1:M
    lambda_a = solve_adjoint_rhs(Z_op, Q_a * I_a; solver=:gmres, ...)
end
```

---

## Performance Characteristics

| Component | Cost per matvec | Memory |
|-----------|----------------|--------|
| MLFMA base | O(N log N) | O(N log N) |
| ACA base | O(N log^2 N) | O(N log^2 N) |
| Dense base | O(N^2) | O(N^2) |
| Impedance perturbation | O(nnz(M_p) * P) | O(nnz(M_p) * P) |

The impedance perturbation cost is typically negligible: each `M_p` is sparse with O(N/P) nonzeros per patch, and there are P patches, giving O(N) total work per matvec for the impedance term.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/assembly/CompositeOperator.jl` | `ImpedanceLoadedOperator`, `ImpedanceLoadedAdjointOperator`, `mul!`, `adjoint` |
| `src/assembly/Impedance.jl` | `precompute_patch_mass`, `assemble_Z_impedance` |
| `src/solver/Solve.jl` | `assemble_full_Z` (dense alternative) |
| `src/optimization/MultiAngleRCS.jl` | Primary consumer: `optimize_multiangle_rcs` |

---

## Exercises

- **Basic:** Build an `ImpedanceLoadedOperator` from a dense EFIE matrix and verify that `Z_op * x` matches `assemble_full_Z(Z_efie, Mp, theta) * x` for a random `x`.
- **Practical:** Verify the adjoint: check that `dot(Z_op * x, y) ≈ dot(x, Z_op' * y)` for random vectors `x` and `y`.
- **Challenge:** Compare GMRES iteration counts for `ImpedanceLoadedOperator` wrapping a dense matrix vs. wrapping an `ACAOperator`, with and without near-field preconditioning.
