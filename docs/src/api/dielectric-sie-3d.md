# API: Dielectric Surface Integral Equation (3D)

## Purpose

Reference for the closed-surface dielectric surface integral equation (SIE) solver for homogeneous isotropic bodies. Two formulations are supported: the first-kind **PMCHWT** (Poggio-Miller-Chang-Harrington-Wu-Tsai) system and the second-kind **Müller** system. Both equivalent-current formulations solve for the tangential electric and magnetic surface currents `[J; M]` on the closed boundary between an exterior and an interior medium, using RWG basis functions on a closed triangle mesh.

The electric-current blocks reuse the package's singularity-treated EFIE assembly (see [assembly-solve.md](assembly-solve.md)); the magnetic-field `K` operator is assembled by principal-value product quadrature with higher-order quadrature on near-singular panel pairs (pairs sharing at least one vertex). The Müller formulation is a validated alternative to PMCHWT: on a dielectric sphere the PMCHWT and Müller surface currents agree to better than 1%, tightening under mesh refinement.

All routines require a **closed** mesh: build the RWG basis with `build_rwg(mesh; allow_boundary=false, require_closed=true)` (see [rwg.md](rwg.md) and [types.md](types.md)). Both dense and matrix-free assembly are available; the matrix-free path is used by `solver=:gmres`.

This subsystem lives in `src/mom3d/SurfaceIE3D.jl`.

---

## Media

### `struct DielectricMedium3D`

Holds the constitutive parameters and derived wave constants for one homogeneous isotropic region.

```julia
struct DielectricMedium3D
    eps_r::ComplexF64
    mu_r::ComplexF64
    k::ComplexF64
    eta::ComplexF64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `eps_r` | `ComplexF64` | Relative permittivity of the medium. |
| `mu_r` | `ComplexF64` | Relative permeability of the medium. |
| `k` | `ComplexF64` | Wavenumber in the medium, `k = k0 * sqrt(eps_r * mu_r)`. |
| `eta` | `ComplexF64` | Wave impedance in the medium, `eta = eta0 * sqrt(mu_r / eps_r)`. |

Construct with `dielectric_medium_3d` rather than by calling the constructor directly, so that `k` and `eta` are computed consistently.

---

### `dielectric_medium_3d(k0, eps_r=1.0 + 0im, mu_r=1.0 + 0im; eta0=376.730313668)`

Build a `DielectricMedium3D` from the free-space wavenumber and relative material parameters. The medium wavenumber and impedance are derived as `k = k0 * sqrt(eps_r * mu_r)` and `eta = eta0 * sqrt(mu_r / eps_r)`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k0` | `Real` | -- | Free-space wavenumber (rad/m). Must be positive. |
| `eps_r` | Real or Complex | `1.0 + 0im` | Relative permittivity. Must be finite and nonzero. |
| `mu_r` | Real or Complex | `1.0 + 0im` | Relative permeability. Must be finite and nonzero. |
| `eta0` | `Real` | `376.730313668` | Free-space impedance (Ohm). Must be positive. |

**Returns:** `DielectricMedium3D` with computed `k` and `eta`.

**Example:**

```julia
k0 = 1.0
ext = dielectric_medium_3d(k0)                 # vacuum (eps_r = mu_r = 1)
int = dielectric_medium_3d(k0, 2.5 + 0im)      # dielectric, eps_r = 2.5
```

---

## Magnetic-Field (K) Operator

### `assemble_magnetic_field_operator_3d(mesh, rwg, k; quad_order=3, singular_quad_order=7, mesh_precheck=true, area_tol_rel=1e-12)`

Assemble the dense magnetic-field principal-value operator

```
K[m,n] = <f_m, PV integral{ grad(G(r,r')) x f_n(r') dS' }>
```

where `G` is the homogeneous-medium Green function at wavenumber `k`. This is the off-diagonal surface-current coupling block used in the PMCHWT/Müller systems. Triangle pairs that share at least one vertex (self, edge-adjacent, and vertex-touching) are integrated with the higher-order `singular_quad_order` rule; all other pairs use `quad_order`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Closed triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data (must be non-periodic, built from `mesh`). |
| `k` | Real or Complex | -- | Wavenumber in the medium (rad/m). |
| `quad_order` | `Int` | `3` | Quadrature order for far (non-touching) triangle pairs. |
| `singular_quad_order` | `Int` | `7` | Quadrature order for near-singular pairs (sharing >= 1 vertex). |
| `mesh_precheck` | `Bool` | `true` | If `true`, assert mesh quality with `allow_boundary=false, require_closed=true`. |
| `area_tol_rel` | `Float64` | `1e-12` | Relative tolerance for degenerate-triangle detection in the precheck. |

**Returns:** `Matrix{ComplexF64}` `K` of size `N x N`, where `N = rwg.nedges`.

**Example:**

```julia
rwg = build_rwg(mesh; allow_boundary=false, require_closed=true)
K = assemble_magnetic_field_operator_3d(mesh, rwg, k0)
```

---

### `matrixfree_magnetic_field_operator_3d(mesh, rwg, k; quad_order=3, singular_quad_order=7, mesh_precheck=true, area_tol_rel=1e-12)`

Build a matrix-free version of the magnetic-field `K` operator. Parameters are identical to `assemble_magnetic_field_operator_3d`; no dense `N x N` matrix is allocated. Instead, the returned `MatrixFreeMagneticFieldOperator3D` caches the per-triangle quadrature points/weights and the near-pair mask, and computes entries on demand.

**Returns:** `MatrixFreeMagneticFieldOperator3D`.

**Example:**

```julia
K_mf = matrixfree_magnetic_field_operator_3d(mesh, rwg, k0)
y = K_mf * x                # matrix-vector product, length-N x and y
```

---

### `struct MatrixFreeMagneticFieldOperator3D`

Matrix-free wrapper for the magnetic-field `K` operator. It is an `AbstractMatrix{ComplexF64}` of size `(N, N)` and supports `size`, `eltype`, `getindex` (single entry via `A[i,j]`), `mul!`, and `*`.

```julia
struct MatrixFreeMagneticFieldOperator3D <: AbstractMatrix{ComplexF64}
    mesh::TriMesh
    rwg::RWGData
    k::ComplexF64
    wq::Vector{Float64}
    pts::Vector{Vector{Vec3}}
    areas::Vector{Float64}
    wq_hi::Vector{Float64}
    pts_hi::Vector{Vector{Vec3}}
    areas_hi::Vector{Float64}
    near_pairs::BitMatrix
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mesh` | `TriMesh` | The closed mesh. |
| `rwg` | `RWGData` | RWG basis data. |
| `k` | `ComplexF64` | Wavenumber in the medium. |
| `wq` | `Vector{Float64}` | Quadrature weights for the far (coarse) rule. |
| `pts` | `Vector{Vector{Vec3}}` | Coarse-rule quadrature points per triangle. |
| `areas` | `Vector{Float64}` | Triangle areas (m^2). |
| `wq_hi` | `Vector{Float64}` | Quadrature weights for the high-order (near-singular) rule. |
| `pts_hi` | `Vector{Vector{Vec3}}` | High-order quadrature points per triangle. |
| `areas_hi` | `Vector{Float64}` | Triangle areas for the high-order rule (m^2). |
| `near_pairs` | `BitMatrix` | `near_pairs[ti, tj] = true` if triangles `ti`, `tj` share at least one vertex. |

Construct with `matrixfree_magnetic_field_operator_3d`.

---

## RHS Assembly

### `assemble_dielectric_sie_rhs_3d(mesh, rwg, excitation, exterior; quad_order=3, formulation=:pmchwt, interior=nothing)`

Assemble the stacked right-hand side `[v_E; v_H]` for a plane-wave incident field in the **exterior** medium. The incident field lives only in the exterior region, so `v_E` tests the incident electric field and `v_H` tests the incident magnetic field `H_inc = (k_hat x E_inc) / eta_ext`.

For `formulation=:pmchwt` the RHS is `[v_E; v_H]`. For `formulation=:muller` the exterior equations are scaled by the exterior row weights, giving `[c_ze_ext * v_E; c_zh_ext * v_H]`; this requires the `interior` medium so the Müller weights can be computed.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Closed triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `excitation` | `PlaneWaveExcitation` | -- | Incident plane wave (see [excitation.md](excitation.md)). |
| `exterior` | `DielectricMedium3D` | -- | The exterior (background) medium. |
| `quad_order` | `Int` | `3` | Quadrature order for the testing integrals. |
| `formulation` | `Symbol` | `:pmchwt` | `:pmchwt` or `:muller`. |
| `interior` | `Nothing` or `DielectricMedium3D` | `nothing` | Interior medium. Required when `formulation=:muller`. |

**Returns:** `Vector{ComplexF64}` of length `2N`.

**Example:**

```julia
ext = dielectric_medium_3d(k0)
int = dielectric_medium_3d(k0, 2.5 + 0im)
pw  = make_plane_wave(Vec3(0.0, 0.0, k0), 1.0, Vec3(1.0, 0.0, 0.0))
rhs = assemble_dielectric_sie_rhs_3d(mesh, rwg, pw, ext)               # PMCHWT
rhs_mu = assemble_dielectric_sie_rhs_3d(mesh, rwg, pw, ext;
                                        formulation=:muller, interior=int)
```

---

## Dense System Assembly

### `assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in=1.0 + 0im; mur_in=1.0 + 0im, epsr_ext=1.0 + 0im, mur_ext=1.0 + 0im, formulation=:pmchwt, quad_order=3, singular_quad_order=7, eta0=376.730313668, mesh_precheck=true, area_tol_rel=1e-12)`

Assemble the dense `2N x 2N` dielectric SIE matrix for an isotropic homogeneous body. Unknowns are stacked RWG coefficients `[J; M]` (electric current `J`, then magnetic current `M`). The block structure is

```
A = [ A11  A12 ;
      A21  A22 ]
```

where `A11`/`A22` are the (row-weighted) EFIE `T` blocks for the electric/magnetic rows and `A12`/`A21` carry the magnetic-field `K` operator (with sign `-` on the E-row, `+` on the H-row). For `:muller`, the diagonal `T` and off-diagonal `K` blocks are mu/eps-weighted and the off-diagonal additionally includes the second-kind `nhat x` Gram identity residue that the principal-value `K` operator omits; this term cancels identically in PMCHWT.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Closed triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data (non-periodic, closed). |
| `k0` | `Real` | -- | Free-space wavenumber (rad/m). |
| `epsr_in` | Real or Complex | `1.0 + 0im` | Interior relative permittivity. |
| `mur_in` | Real or Complex | `1.0 + 0im` | Interior relative permeability. |
| `epsr_ext` | Real or Complex | `1.0 + 0im` | Exterior relative permittivity. |
| `mur_ext` | Real or Complex | `1.0 + 0im` | Exterior relative permeability. |
| `formulation` | `Symbol` | `:pmchwt` | `:pmchwt` or `:muller`. |
| `quad_order` | `Int` | `3` | Quadrature order for far panel pairs. |
| `singular_quad_order` | `Int` | `7` | Quadrature order for near-singular panel pairs. |
| `eta0` | `Real` | `376.730313668` | Free-space impedance (Ohm). |
| `mesh_precheck` | `Bool` | `true` | Assert closed-surface mesh quality before assembly. |
| `area_tol_rel` | `Float64` | `1e-12` | Relative degenerate-triangle tolerance. |

**Returns:** `Matrix{ComplexF64}` `A` of size `(2N, 2N)`.

**Example:**

```julia
A = assemble_dielectric_sie_3d(mesh, rwg, k0, 2.5 + 0im; mur_in=1.6 + 0im)
```

---

### `assemble_pmchwt_3d(mesh, rwg, k0, epsr_in=1.0 + 0im; kwargs...)`

Assemble the dense first-kind **PMCHWT** system. Equivalent to `assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in; formulation=:pmchwt, kwargs...)`. All keyword arguments of `assemble_dielectric_sie_3d` (except `formulation`) are forwarded.

PMCHWT uses unit row weights, so its diagonal `T` blocks combine the exterior and interior EFIE blocks with equal weight and the `nhat x` Gram identity term cancels.

**Returns:** `Matrix{ComplexF64}` `A` of size `(2N, 2N)`.

**Example:**

```julia
rwg_closed = build_rwg(mesh_closed; allow_boundary=false, require_closed=true)
A = assemble_pmchwt_3d(mesh_closed, rwg_closed, k, 2.5 + 0im)
```

---

### `assemble_muller_3d(mesh, rwg, k0, epsr_in=1.0 + 0im; kwargs...)`

Assemble the dense second-kind **Müller** system. Equivalent to `assemble_dielectric_sie_3d(mesh, rwg, k0, epsr_in; formulation=:muller, kwargs...)`. All keyword arguments of `assemble_dielectric_sie_3d` (except `formulation`) are forwarded.

The Müller formulation applies mu/eps-weighted row coefficients to both the diagonal `T` blocks and the off-diagonal `K` blocks, and includes the `nhat x` Gram identity residue on the off-diagonal that the principal-value `K` operator omits. This is a validated alternative to PMCHWT: on a dielectric sphere the Müller and PMCHWT currents agree to better than 1%, tightening under mesh refinement. The Müller system is distinct from PMCHWT (a different matrix), but discretizes the same boundary value problem, so the solved `J`, `M` match.

**Returns:** `Matrix{ComplexF64}` `A` of size `(2N, 2N)`.

**Example:**

```julia
A_mu = assemble_muller_3d(mesh, rwg, k0, 2.5 + 0im; mur_in=1.6 + 0im)
```

---

## Matrix-Free System Operator

### `matrixfree_dielectric_sie_operator_3d(mesh, rwg, k0, epsr_in=1.0 + 0im; mur_in=1.0 + 0im, epsr_ext=1.0 + 0im, mur_ext=1.0 + 0im, formulation=:pmchwt, quad_order=3, singular_quad_order=7, eta0=376.730313668, mesh_precheck=true, area_tol_rel=1e-12)`

Build a matrix-free `2N x 2N` dielectric SIE operator without forming any dense block. The returned `MatrixFreeDielectricSIE3D` wraps matrix-free EFIE operators (`Ze`/`Zh` for exterior/interior) and matrix-free magnetic-field operators (`K`), applies the formulation-specific row weights, and (for Müller) precomputes the dense `nhat x` Gram matrix. Parameters match `assemble_dielectric_sie_3d`.

**Returns:** `MatrixFreeDielectricSIE3D`.

**Example:**

```julia
A = matrixfree_dielectric_sie_operator_3d(mesh, rwg, k0, 2.5 + 0im;
                                          mur_in=1.6 + 0im, formulation=:pmchwt)
x, stats = Krylov.gmres(A, rhs)
```

---

### `struct MatrixFreeDielectricSIE3D`

Matrix-free `2N x 2N` dielectric SIE operator. It is a `mutable struct <: AbstractMatrix{ComplexF64}` and supports `size`, `eltype`, `getindex` (single entry via `A[row,col]`), `mul!` (including the 5-argument `mul!(y, A, x, alpha, beta)` form), and `*`. It carries preallocated work buffers, so it is intended to be reused across matvecs.

```julia
mutable struct MatrixFreeDielectricSIE3D{TZe,TZh,TK} <: AbstractMatrix{ComplexF64}
    formulation::Symbol
    exterior::DielectricMedium3D
    interior::DielectricMedium3D
    Ze_ext::TZe
    Ze_int::TZe
    Zh_ext::TZh
    Zh_int::TZh
    K_ext::TK
    K_int::TK
    c_ze_ext::ComplexF64
    c_ze_int::ComplexF64
    c_zh_ext::ComplexF64
    c_zh_int::ComplexF64
    Gram::Matrix{ComplexF64}
    c_g_e::ComplexF64
    c_g_h::ComplexF64
    work_J::Vector{ComplexF64}
    work_M::Vector{ComplexF64}
    tmp1::Vector{ComplexF64}
    tmp2::Vector{ComplexF64}
    tmp3::Vector{ComplexF64}
    tmp4::Vector{ComplexF64}
    tmp5::Vector{ComplexF64}
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `formulation` | `Symbol` | `:pmchwt` or `:muller`. |
| `exterior` | `DielectricMedium3D` | Exterior medium. |
| `interior` | `DielectricMedium3D` | Interior medium. |
| `Ze_ext`, `Ze_int` | `MatrixFreeEFIEOperator` | Electric-row EFIE `T` operators (exterior/interior). |
| `Zh_ext`, `Zh_int` | `MatrixFreeEFIEOperator` | Magnetic-row EFIE `T` operators (exterior/interior). |
| `K_ext`, `K_int` | `MatrixFreeMagneticFieldOperator3D` | Magnetic-field `K` operators (exterior/interior). |
| `c_ze_ext`, `c_ze_int` | `ComplexF64` | Electric-row exterior/interior block weights. |
| `c_zh_ext`, `c_zh_int` | `ComplexF64` | Magnetic-row exterior/interior block weights. |
| `Gram` | `Matrix{ComplexF64}` | `nhat x` Gram identity matrix (empty `0 x 0` when not needed, i.e. for PMCHWT). |
| `c_g_e` | `ComplexF64` | E-row off-diagonal Gram coefficient `-(c_ze_ext - c_ze_int) * 0.5`. |
| `c_g_h` | `ComplexF64` | H-row off-diagonal Gram coefficient `(c_zh_ext - c_zh_int) * 0.5`. |
| `work_J`, `work_M` | `Vector{ComplexF64}` | Length-`N` input buffers for the `J`/`M` sub-blocks. |
| `tmp1`-`tmp5` | `Vector{ComplexF64}` | Length-`N` scratch buffers for the matvec. |

Construct with `matrixfree_dielectric_sie_operator_3d`. For PMCHWT the weights are all `1` and the Gram term is skipped; for Müller they are the mu/eps weights and the Gram term is active.

---

## Solve

### `solve_dielectric_sie_3d(mesh, rwg, k0, epsr_in, rhs; mur_in=1.0 + 0im, epsr_ext=1.0 + 0im, mur_ext=1.0 + 0im, formulation=:pmchwt, solver=:direct, quad_order=3, singular_quad_order=7, eta0=376.730313668, mesh_precheck=true, area_tol_rel=1e-12, tol=1e-8, maxiter=200, memory=20, verbose=false)`

Solve a closed-surface PMCHWT/Müller dielectric SIE system and return the split surface currents plus solver metadata. The `rhs` argument may be either a length-`2N` vector or a `PlaneWaveExcitation`; in the latter case the RHS is assembled internally via `assemble_dielectric_sie_rhs_3d` for the given formulation (with the exterior/interior media built from the supplied parameters).

With `solver=:direct` the dense `2N x 2N` matrix is assembled and LU-factorized. With `solver=:gmres` the matrix-free operator is used and the system is solved with `Krylov.gmres` (no dense matrix is formed).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Closed triangle mesh. |
| `rwg` | `RWGData` | -- | RWG basis data (non-periodic, closed). |
| `k0` | `Real` | -- | Free-space wavenumber (rad/m). |
| `epsr_in` | Real or Complex | -- | Interior relative permittivity. |
| `rhs` | `AbstractVector{<:Number}` or `PlaneWaveExcitation` | -- | Length-`2N` RHS vector, or a plane-wave excitation. |
| `mur_in` | Real or Complex | `1.0 + 0im` | Interior relative permeability. |
| `epsr_ext` | Real or Complex | `1.0 + 0im` | Exterior relative permittivity. |
| `mur_ext` | Real or Complex | `1.0 + 0im` | Exterior relative permeability. |
| `formulation` | `Symbol` | `:pmchwt` | `:pmchwt` or `:muller`. |
| `solver` | `Symbol` | `:direct` | `:direct` (dense LU) or `:gmres` (matrix-free Krylov). |
| `quad_order` | `Int` | `3` | Quadrature order for far panel pairs. |
| `singular_quad_order` | `Int` | `7` | Quadrature order for near-singular panel pairs. |
| `eta0` | `Real` | `376.730313668` | Free-space impedance (Ohm). |
| `mesh_precheck` | `Bool` | `true` | Assert closed-surface mesh quality before assembly. |
| `area_tol_rel` | `Float64` | `1e-12` | Relative degenerate-triangle tolerance. |
| `tol` | `Float64` | `1e-8` | GMRES relative tolerance (`rtol`); used only for `solver=:gmres`. |
| `maxiter` | `Int` | `200` | GMRES maximum iterations (`itmax`); used only for `solver=:gmres`. |
| `memory` | `Int` | `20` | GMRES restart memory; used only for `solver=:gmres`. |
| `verbose` | `Bool` | `false` | If `true`, print GMRES progress; used only for `solver=:gmres`. |

**Returns:** `DielectricSIEResult3D` with the solved currents `J`, `M` and solver metadata.

**Example:**

```julia
rwg_closed = build_rwg(mesh_closed; allow_boundary=false, require_closed=true)

# Solve from a precomputed RHS vector (length 2N) with GMRES
res_sie = solve_dielectric_sie_3d(mesh_closed, rwg_closed, k, 2.5 + 0im, rhs;
                                  solver=:gmres)

# Or solve directly from a plane-wave excitation
pw  = make_plane_wave(Vec3(0.0, 0.0, k), 1.0, Vec3(1.0, 0.0, 0.0))
res = solve_dielectric_sie_3d(mesh_closed, rwg_closed, k, 2.5 + 0im, pw;
                              mur_in=1.6 + 0im, formulation=:muller)
J, M = res.J, res.M
```

---

### `struct DielectricSIEResult3D`

Output of `solve_dielectric_sie_3d`. Bundles the split surface currents with the system operator, RHS, solver choice, and per-solver metadata.

```julia
struct DielectricSIEResult3D{TA<:AbstractMatrix{ComplexF64},TLU,TStats}
    J::Vector{ComplexF64}
    M::Vector{ComplexF64}
    A::TA
    rhs::Vector{ComplexF64}
    A_LU::TLU
    solver::Symbol
    stats::TStats
    formulation::Symbol
    exterior::DielectricMedium3D
    interior::DielectricMedium3D
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `J` | `Vector{ComplexF64}` | Electric surface-current RWG coefficients (length `N`). |
| `M` | `Vector{ComplexF64}` | Magnetic surface-current RWG coefficients (length `N`). |
| `A` | `AbstractMatrix{ComplexF64}` | The system operator: a dense `Matrix{ComplexF64}` for `solver=:direct`, or a `MatrixFreeDielectricSIE3D` for `solver=:gmres`. |
| `rhs` | `Vector{ComplexF64}` | The right-hand side used (length `2N`). |
| `A_LU` | LU factorization or `Nothing` | LU factorization of `A` for `solver=:direct`; `nothing` for `solver=:gmres`. |
| `solver` | `Symbol` | `:direct` or `:gmres`. |
| `stats` | Krylov stats or `Nothing` | `Krylov.gmres` statistics for `solver=:gmres`; `nothing` for `solver=:direct`. |
| `formulation` | `Symbol` | `:pmchwt` or `:muller`. |
| `exterior` | `DielectricMedium3D` | Exterior medium used in the solve. |
| `interior` | `DielectricMedium3D` | Interior medium used in the solve. |

The stacked solution is `vcat(res.J, res.M)`, which satisfies `res.A * vcat(res.J, res.M) ~ res.rhs`.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/mom3d/SurfaceIE3D.jl` | `DielectricMedium3D`, `DielectricSIEResult3D`, `dielectric_medium_3d`, `assemble_magnetic_field_operator_3d`, `MatrixFreeMagneticFieldOperator3D`, `matrixfree_magnetic_field_operator_3d`, `assemble_dielectric_sie_rhs_3d`, `assemble_dielectric_sie_3d`, `assemble_pmchwt_3d`, `assemble_muller_3d`, `matrixfree_dielectric_sie_operator_3d`, `MatrixFreeDielectricSIE3D`, `solve_dielectric_sie_3d` |

See also [assembly-solve.md](assembly-solve.md) (EFIE `T` blocks and matrix-free EFIE operators), [excitation.md](excitation.md) (`PlaneWaveExcitation`, `make_plane_wave`), [rwg.md](rwg.md) (closed-surface RWG construction), and [types.md](types.md) (`TriMesh`, `RWGData`, `Vec3`).
