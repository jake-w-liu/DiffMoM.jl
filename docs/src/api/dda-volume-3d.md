# API: 3D Volume Material Solver (DDA / VIE-style)

## Purpose

Reference for the 3D material volume scattering subsystem. Unlike the surface
EFIE path (which solves for RWG surface currents on a triangulated boundary),
this subsystem discretizes a material volume into a uniform Cartesian grid of
voxels and solves a vector discrete-dipole / volume-integral-equation (DDA /
VIE-style) system. The unknowns are the three components of the total electric
field (or six components of the total electric and magnetic fields) at each
voxel center.

The package convention is `exp(+i omega t)`, so the scalar Green phase is
`exp(-i k R)`. Each voxel is assigned a normalized polarizability
`alpha = p / (eps0 * E)` (in cubic meters) from a Clausius-Mossotti model, and
the induced normalized dipole is `q = alpha * E`. The coupled system

```
E_i - sum_{j != i} G_EE(r_i, r_j) alpha_j E_j = E_inc_i
```

is solved either with a dense direct factorization, a matrix-free GMRES
operator, or an FFT-accelerated GMRES operator that exploits the block-Toeplitz
structure of the uniform grid.

For the constitutive material inputs (relative permittivity, permeability, and
bianisotropic constitutive tensors) consumed by these solvers, see
[material-models-3d.md](material-models-3d.md). For surface-current far-field
and RCS post-processing on the EFIE path, see [farfield-rcs.md](farfield-rcs.md).

This page is organized into five sections:

1. Voxel grid and types
2. Electric DDA (scalar / tensor `eps_r`)
3. Coupled electric-magnetic DDA
4. FFT-accelerated operators
5. Adjoint / gradient

---

## 1. Voxel Grid and Types

### `VoxelGrid3D(x_range, y_range, z_range, nx, ny, nz)`

Create a uniform Cartesian voxel grid over
`[x_range[1], x_range[2]] x [y_range[1], y_range[2]] x [z_range[1], z_range[2]]`.
Each voxel stores a center and a (uniform) volume; material properties are
supplied separately as one value per voxel, so the same grid can be reused for
parameter sweeps.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_range` | `Tuple{<:Real,<:Real}` | -- | Increasing `(x_min, x_max)` bounds in meters. |
| `y_range` | `Tuple{<:Real,<:Real}` | -- | Increasing `(y_min, y_max)` bounds in meters. |
| `z_range` | `Tuple{<:Real,<:Real}` | -- | Increasing `(z_min, z_max)` bounds in meters. |
| `nx` | `Int` | -- | Number of voxels along x (must be `>= 1`). |
| `ny` | `Int` | -- | Number of voxels along y (must be `>= 1`). |
| `nz` | `Int` | -- | Number of voxels along z (must be `>= 1`). |

**Returns:** `VoxelGrid3D` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `centers` | `Vector{Vec3}` | Voxel center coordinates, length `nvoxels`. |
| `volumes` | `Vector{Float64}` | Per-voxel volume in m^3 (uniform `dx*dy*dz`), length `nvoxels`. |
| `nvoxels` | `Int` | Total number of voxels `= nx * ny * nz`. |
| `nx`, `ny`, `nz` | `Int` | Number of voxels along each axis. |
| `dx`, `dy`, `dz` | `Float64` | Voxel edge lengths in meters. |
| `x0`, `y0`, `z0` | `Float64` | Lower-corner coordinates (`x_range[1]`, etc.). |

Voxel centers are placed at midpoints: the linear index runs fastest over `ix`,
then `iy`, then `iz`, with center `[x0 + (ix-0.5)*dx, y0 + (iy-0.5)*dy, z0 + (iz-0.5)*dz]`.

**Example:**

```julia
k = 2pi / 0.1
grid = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 7, 7, 7)
println("Voxels: ", grid.nvoxels)        # 343
println("Edge length dx: ", grid.dx, " m")
```

---

### `make_voxel_grid_3d(x_range, y_range, z_range, nx, ny, nz)`

Functional alias for the `VoxelGrid3D` constructor. Identical arguments and
return value; provided for callers that prefer a `make_*` factory style
consistent with the rest of the package.

**Returns:** `VoxelGrid3D`.

**Example:**

```julia
grid = make_voxel_grid_3d((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 7, 7, 7)
```

---

### `DDAOperator3D`

Matrix-free coupled-dipole operator for 3D electric material scattering. It
represents the same `3N x 3N` linear system as `assemble_dda_3d` without storing
the dense matrix, and behaves as an `AbstractMatrix{ComplexF64}` (supports
`size`, `eltype`, `getindex`, `mul!`, and `adjoint`). Construct with
`dda_operator_3d`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `VoxelGrid3D` | The voxel grid. |
| `k0` | `Float64` | Background wavenumber (rad/m). |
| `eps_r` | `AbstractVector` | Per-voxel relative permittivity (scalar or `3x3` tensor entries). |
| `alpha` | `AbstractVector` | Per-voxel normalized polarizability (scalar or `3x3` tensor). |
| `radiative_correction` | `Bool` | Whether the radiation-reaction correction was applied to `alpha`. |

**Size:** `(3*nvoxels, 3*nvoxels)`. The matrix-free `mul!` performs an
all-pairs dipole sum (O(N^2) per matvec) with zero heap allocation.

---

### `DDAAdjointOperator3D`

Hermitian-adjoint wrapper for `DDAOperator3D`, obtained via `adjoint(A)` where
`A::DDAOperator3D`. Behaves as an `AbstractMatrix{ComplexF64}` with
`A_adj[i, j] = conj(A[j, i])`. It is used internally by `solve_dda_adjoint_3d`
for adjoint sensitivity solves without forming a dense matrix.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `parent` | `DDAOperator3D` | The forward operator being adjointed. |

---

### `DDAResult3D`

Result from a 3D electric discrete-dipole material solve, returned by
`solve_dda_3d`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `E_total` | `Vector{CVec3}` | Total electric field at each voxel center. |
| `E_inc` | `Vector{CVec3}` | Incident electric field at each voxel center (the right-hand side). |
| `eps_r` | `AbstractVector` | Per-voxel relative permittivity (scalar or tensor). |
| `alpha` | `AbstractVector` | Per-voxel normalized polarizability `p / (eps0 * E)` in m^3, so induced dipoles are `alpha[j] * E_total[j]`. |
| `A` | `AbstractMatrix{ComplexF64}` | The solved operator: a dense `Matrix` for `:direct`, or a `DDAOperator3D` for `:gmres`. |
| `A_LU` | -- | Stored LU factorization for `:direct` (reused by the adjoint solve), or `nothing` for `:gmres`. |
| `solver` | `Symbol` | `:direct` or `:gmres`. |
| `stats` | -- | Krylov solver statistics for `:gmres`, or `nothing` for `:direct`. |
| `grid` | `VoxelGrid3D` | The voxel grid. |
| `k0` | `Float64` | Background wavenumber (rad/m). |
| `radiative_correction` | `Bool` | Whether the radiation-reaction correction was applied. |

---

### `EMDDAOperator3D`

Matrix-free coupled electric-magnetic DDA operator. Unknowns are the total
electric and magnetic fields at each voxel, stored as six components per voxel
in the order `(Ex, Ey, Ez, Hx, Hy, Hz)`. Behaves as an
`AbstractMatrix{ComplexF64}` of size `(6*nvoxels, 6*nvoxels)`. Construct with
`em_dda_operator_3d`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `VoxelGrid3D` | The voxel grid. |
| `k0` | `Float64` | Background wavenumber (rad/m). |
| `alpha` | `AbstractVector` | Per-voxel `6x6` polarizability mapping `[E; H]` to `[q; m]`. |
| `radiative_correction` | `Bool` | Whether the radiation-reaction correction was applied. |

---

### `EMDDAResult3D`

Result from a coupled electric-magnetic DDA solve, returned by
`solve_em_dda_3d`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `E_total` | `Vector{CVec3}` | Total electric field at each voxel center. |
| `H_total` | `Vector{CVec3}` | Total magnetic field at each voxel center. |
| `E_inc` | `Vector{CVec3}` | Incident electric field at each voxel center. |
| `H_inc` | `Vector{CVec3}` | Incident magnetic field at each voxel center. |
| `alpha` | `AbstractVector` | Per-voxel `6x6` polarizability. |
| `A` | `AbstractMatrix{ComplexF64}` | The solved operator (dense `Matrix` for `:direct`; matrix-free operator otherwise). |
| `A_LU` | -- | Stored LU factorization for `:direct`, or `nothing` otherwise. |
| `solver` | `Symbol` | Reported solver: `:direct`, `:gmres`, or `:fft_gmres`. |
| `stats` | -- | Krylov statistics for iterative solves, or `nothing` for `:direct`. |
| `grid` | `VoxelGrid3D` | The voxel grid. |
| `k0` | `Float64` | Background wavenumber (rad/m). |
| `radiative_correction` | `Bool` | Whether the radiation-reaction correction was applied. |

---

## 2. Electric DDA (scalar / tensor `eps_r`)

### `clausius_mossotti_polarizability(eps_r, volume; k0=0.0, radiative_correction=false)`

Return the normalized electric polarizability `alpha = p / (eps0 * E)` for an
isotropic (`eps_r::Number`) or tensor (`eps_r::AbstractMatrix`, `3x3`) voxel of
volume `volume`. The default is the Clausius-Mossotti polarizability

```
alpha0 = 3V * (eps_r - 1) / (eps_r + 2)
```

(the exact electrostatic polarizability of a sphere of the same volume). With
`radiative_correction=true`, a leading-order radiation-reaction correction
consistent with `exp(+i omega t)` is applied: `alpha0 / (1 + i k^3 alpha0 / (6 pi))`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_r` | `Number` or `AbstractMatrix` | -- | Relative permittivity (scalar or `3x3` tensor). |
| `volume` | `Real` | -- | Voxel volume in m^3 (must be positive). |
| `k0` | `Real` | `0.0` | Wavenumber (rad/m), used only when `radiative_correction=true`. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** `ComplexF64` (scalar input) or `SMatrix{3,3,ComplexF64}` (tensor
input). Errors if `eps_r` is near `-2` (singular denominator).

**Example:**

```julia
alpha = clausius_mossotti_polarizability(2.5 + 0im, grid.volumes[1])
```

---

### `dda_polarizabilities(grid, k0, eps_r; radiative_correction=false)`

Compute the normalized electric polarizability for every voxel in `grid`. The
`eps_r` argument may be a single scalar/tensor (broadcast to all voxels), a
`3x3` tensor, a diagonal tensor tuple, or a `Vector` of per-voxel
scalars/tensors (see [material-models-3d.md](material-models-3d.md) for the
accepted material specifications).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m). |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** `Vector` of per-voxel polarizabilities (`ComplexF64` or
`SMatrix{3,3,ComplexF64}` elements), length `grid.nvoxels`.

---

### `dda_operator_3d(grid, k0, eps_r; radiative_correction=false)`

Construct the matrix-free `DDAOperator3D`. This is the memory-efficient
counterpart to `assemble_dda_3d`: it stores O(N) material/geometric data instead
of the O(N^2) dense interaction matrix and produces an identical matvec.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m), must be positive. |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** `DDAOperator3D`.

**Example:**

```julia
A_op = dda_operator_3d(grid, k, fill(2.5 + 0.1im, grid.nvoxels))
y = zeros(ComplexF64, size(A_op, 1))
mul!(y, A_op, x)
```

---

### `electric_dipole_dyadic_3d(r, rp, k0)`

Free-space electric dipole dyadic `G_EE(r, rp)` for the `exp(+i omega t)`
convention. It maps a normalized dipole moment `q = p / eps0` at `rp` to the
electric field at `r`. The singular self term (`r == rp`) is undefined and must
be handled by the chosen polarizability model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | `Vec3` | -- | Observation point (m). |
| `rp` | `Vec3` | -- | Source dipole location (m). |
| `k0` | `Real` | -- | Wavenumber (rad/m), must be nonnegative. |

**Returns:** `SMatrix{3,3,ComplexF64}`. Errors for coincident points (`r == rp`).

**Formula:**

```
G_EE = exp(-i k R)/(4 pi) * [ (k^2/R)(I - r_hat r_hat')
                              + (1/R^3 + i k/R^2)(3 r_hat r_hat' - I) ]
```

where `R = |r - rp|` and `r_hat = (r - rp)/R`.

---

### `assemble_dda_3d(grid, k0, eps_r; radiative_correction=false)`

Assemble the dense coupled-dipole system

```
E_i - sum_{j != i} G_EE(r_i, r_j) alpha_j E_j = E_inc_i
```

for complex (scalar or tensor) relative permittivity `eps_r`. The dense build is
threaded over source voxels when worker threads are available. Prefer
`dda_operator_3d` for larger grids to avoid O(N^2) dense storage.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m), must be positive. |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** Tuple `(A, alpha, epsv)`:
- `A::Matrix{ComplexF64}`: dense `3N x 3N` system matrix.
- `alpha::Vector`: per-voxel polarizabilities.
- `epsv::Vector`: coerced per-voxel relative permittivities.

---

### `planewave_dda_3d(grid, k_vec, E0, pol)`

Evaluate a transverse plane wave at all voxel centers:

```
E_inc(r) = pol * E0 * exp(-i k_vec . r)
```

The polarization must be transverse to `k_vec` (the normalized dot product must
be `<= 1e-10`, else an error is raised).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k_vec` | `Vec3` | -- | Wave vector `k * k_hat` (rad/m), must be nonzero. |
| `E0` | `Number` | -- | Complex amplitude. |
| `pol` | length-3 vector | -- | Polarization vector (transverse to `k_vec`). |

**Returns:** `Vector{CVec3}` of length `grid.nvoxels`.

---

### `solve_dda_3d(grid, k0, eps_r, E_inc; radiative_correction=false, solver=:direct, tol=1e-8, maxiter=200, memory=20, verbose=false)`

Solve the 3D vector electric material scattering problem for the total electric
field at voxel centers.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m). |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `E_inc` | `AbstractVector` (of `CVec3`) | -- | Incident E-field per voxel (e.g. from `planewave_dda_3d`). |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |
| `solver` | `Symbol` | `:direct` | `:direct` (dense LU) or `:gmres` (matrix-free). |
| `tol` | `Float64` | `1e-8` | GMRES relative tolerance (`rtol`). |
| `maxiter` | `Int` | `200` | Maximum GMRES iterations. |
| `memory` | `Int` | `20` | GMRES restart memory. |
| `verbose` | `Bool` | `false` | Print GMRES progress. |

**Returns:** `DDAResult3D`.

**Example:**

```julia
k = 2pi / 0.1
grid = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 7, 7, 7)
epsr = fill(2.5 + 0im, grid.nvoxels)
Einc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
res = solve_dda_3d(grid, k, epsr, Einc; solver=:gmres)
Ffar = farfield_dda_3d(res, Vec3(0.0, 1.0, 0.0))
```

---

### `induced_dipoles_dda_3d(result)`

Return the normalized induced electric dipoles `q_j = p_j / eps0 = alpha_j E_j`
for each voxel of a `DDAResult3D`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `DDAResult3D` | -- | A solved electric DDA result. |

**Returns:** `Vector{CVec3}` of length `grid.nvoxels`.

---

### `scattered_field_dda_3d(result, r_obs)`

Compute the scattered electric field at observation points by summing the
radiated field of all induced dipoles. Observation points must not coincide with
voxel centers.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `DDAResult3D` | -- | A solved electric DDA result. |
| `r_obs` | `AbstractVector{Vec3}` | -- | Observation points (m). |

**Returns:** `Vector{CVec3}` of length `length(r_obs)`.

---

### `farfield_dda_3d(result, rhat)`

Return the far-field amplitude `F(rhat)` such that

```
E_scat(r) ~= exp(-i k r) / r * F(rhat)
```

for a unit observation direction `rhat`. A `Vector{Vec3}` of directions may be
passed to obtain a `Vector{CVec3}` of amplitudes.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `DDAResult3D` | -- | A solved electric DDA result. |
| `rhat` | `Vec3` or `AbstractVector{Vec3}` | -- | Observation direction(s); each is normalized internally. |

**Returns:** `CVec3` (single direction) or `Vector{CVec3}` (multiple directions).

**Example:**

```julia
F = farfield_dda_3d(res, Vec3(0.0, 1.0, 0.0))
sigma = 4pi * real(dot(F, F))   # bistatic RCS (m^2)
```

---

## 3. Coupled Electric-Magnetic DDA

### `BianisotropicPolarizability3D(alpha6)`

Validated per-voxel `6 x 6` polarizability mapping total fields
`(Ex, Ey, Ez, Hx, Hy, Hz)` to normalized electric and magnetic dipoles
`(qx, qy, qz, mx, my, mz)`. The single field `alpha` stores the validated
`SMatrix{6,6,ComplexF64}`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha6` | `6x6` matrix | -- | The `6x6` polarizability (validated for finiteness). |

**Returns:** `BianisotropicPolarizability3D`. Accepted directly by
`em_dda_operator_3d` and `solve_em_dda_3d`. For starting from a normalized
constitutive tensor, use `BianisotropicMaterial3D` (see
[material-models-3d.md](material-models-3d.md)).

---

### `clausius_mossotti_polarizability` (magnetic and tensor forms)

The electric `clausius_mossotti_polarizability` (Section 2) is reused for the
magnetic path; the following wrappers and helpers supply the coupled EM
polarizabilities.

### `magnetic_clausius_mossotti_polarizability(mu_r, volume; k0=0.0, radiative_correction=false)`

Return the magnetic polarizability `alpha_m` for an isotropic
(`mu_r::Number`) or tensor (`mu_r::AbstractMatrix`) relative permeability voxel.
It uses the same Clausius-Mossotti form as the electric path with `mu_r`
replacing `eps_r`, so the induced magnetic dipole is `m = alpha_m * H`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu_r` | `Number` or `AbstractMatrix` | -- | Relative permeability (scalar or `3x3` tensor). |
| `volume` | `Real` | -- | Voxel volume in m^3. |
| `k0` | `Real` | `0.0` | Wavenumber (rad/m), used only for the correction. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** `ComplexF64` or `SMatrix{3,3,ComplexF64}`.

---

### `bianisotropic_clausius_mossotti_polarizability(C6, volume; k0=0.0, radiative_correction=false, eta0=376.730313668)`

Return a `6 x 6` coupled electric-magnetic polarizability from a normalized
bianisotropic relative material matrix `C6`. `C6` acts on `[E; eta0*H]`; the
returned polarizability acts on the solver fields `[E; H]` and returns `[q; m]`.
`C6` may be a raw `6x6` matrix or a `BianisotropicMaterial3D`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C6` | `6x6` matrix or `BianisotropicMaterial3D` | -- | Normalized bianisotropic constitutive matrix. |
| `volume` | `Real` | -- | Voxel volume in m^3 (must be positive). |
| `k0` | `Real` | `0.0` | Wavenumber (rad/m), used only for the correction. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |
| `eta0` | `Real` | `376.730313668` | Free-space impedance (Ohm). |

**Returns:** `SMatrix{6,6,ComplexF64}`. Errors if `C6 + 2I` is singular.

---

### `em_dda_polarizabilities(grid, k0, eps_r, mu_r; radiative_correction=false)`

Compute per-voxel coupled electric-magnetic polarizability matrices. Several
methods are available:

- `em_dda_polarizabilities(grid, k0, eps_r, mu_r; radiative_correction=false)`:
  block-diagonal `6x6` matrices for magnetodielectric voxels from separate
  `eps_r` and `mu_r` specifications.
- `em_dda_polarizabilities(grid, k0, alpha6; radiative_correction=false)`:
  coerce explicit per-voxel `6x6` polarizabilities (a single matrix, a
  `BianisotropicPolarizability3D`, or a vector of either).
- `em_dda_polarizabilities(grid, k0, material; radiative_correction=false, eta0=376.730313668)`:
  build from a `BianisotropicMaterial3D` (or a vector of them); see
  [material-models-3d.md](material-models-3d.md).

**Parameters (magnetodielectric method):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m). |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `mu_r` | scalar / tensor / vector | -- | Relative permeability specification. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** `Vector{SMatrix{6,6,ComplexF64}}` of length `grid.nvoxels`.

---

### `em_dda_operator_3d(grid, k0, eps_r, mu_r; radiative_correction=false)`

Construct a matrix-free `EMDDAOperator3D`. Several methods are available:

- `em_dda_operator_3d(grid, k0, eps_r, mu_r; radiative_correction=false)`:
  magnetodielectric voxels from `eps_r` and `mu_r`.
- `em_dda_operator_3d(grid, k0, alpha6; radiative_correction=false)`: from
  explicit per-voxel `6x6` polarizabilities (matrix, `BianisotropicPolarizability3D`,
  or vector).
- `em_dda_operator_3d(grid, k0, material; radiative_correction=false, eta0=376.730313668)`:
  from a `BianisotropicMaterial3D` (or vector of them).

**Parameters (magnetodielectric method):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m), must be positive. |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `mu_r` | scalar / tensor / vector | -- | Relative permeability specification. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** `EMDDAOperator3D`.

---

### `assemble_em_dda_3d(grid, k0, eps_r, mu_r; radiative_correction=false)`

Assemble the dense coupled electric-magnetic DDA system. Prefer
`em_dda_operator_3d` (or `fft_em_dda_operator_3d`) for larger grids to avoid
O(N^2) dense storage. A second method
`assemble_em_dda_3d(grid, k0, alpha6; ...)` accepts explicit per-voxel `6x6`
polarizabilities.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m). |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `mu_r` | scalar / tensor / vector | -- | Relative permeability specification. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** Tuple `(A, alpha)`:
- `A::Matrix{ComplexF64}`: dense `6N x 6N` system matrix.
- `alpha::Vector{SMatrix{6,6,ComplexF64}}`: per-voxel `6x6` polarizabilities.

---

### `planewave_em_dda_3d(grid, k_vec, E0, pol; eta0=376.730313668)`

Evaluate the transverse plane-wave incident electric and magnetic fields at
voxel centers, with `H = k_hat x E / eta0`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k_vec` | `Vec3` | -- | Wave vector `k * k_hat` (rad/m). |
| `E0` | `Number` | -- | Complex amplitude. |
| `pol` | length-3 vector | -- | Polarization (transverse to `k_vec`). |
| `eta0` | `Real` | `376.730313668` | Free-space impedance (Ohm). |

**Returns:** Tuple `(E_inc, H_inc)`, each a `Vector{CVec3}` of length `grid.nvoxels`.

---

### `solve_em_dda_3d(grid, k0, eps_r, mu_r, E_inc, H_inc; radiative_correction=false, solver=:direct, tol=1e-8, maxiter=200, memory=20, verbose=false)`

Solve the coupled electric-magnetic volume DDA for magnetodielectric voxels.
Additional methods accept explicit per-voxel `6x6` polarizabilities
(`solve_em_dda_3d(grid, k0, alpha6, E_inc, H_inc; ...)`) or a
`BianisotropicMaterial3D` (with an extra `eta0` keyword).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m). |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `mu_r` | scalar / tensor / vector | -- | Relative permeability specification. |
| `E_inc` | `AbstractVector` (of `CVec3`) | -- | Incident E-field per voxel. |
| `H_inc` | `AbstractVector` (of `CVec3`) | -- | Incident H-field per voxel. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |
| `solver` | `Symbol` | `:direct` | `:direct`, `:gmres`, or `:fft_gmres` (uses the FFT operator). |
| `tol` | `Float64` | `1e-8` | GMRES relative tolerance. |
| `maxiter` | `Int` | `200` | Maximum GMRES iterations. |
| `memory` | `Int` | `20` | GMRES restart memory. |
| `verbose` | `Bool` | `false` | Print GMRES progress. |

**Returns:** `EMDDAResult3D`.

**Example:**

```julia
Einc, Hinc = planewave_em_dda_3d(grid, Vec3(0.0, 0.0, k), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
res_em = solve_em_dda_3d(grid, k, 2.5 + 0im, 1.2 + 0im, Einc, Hinc; solver=:gmres)
```

---

### `induced_dipoles_em_dda_3d(result)`

Return `(q, m)`: the normalized induced electric dipoles and magnetic dipoles
from a coupled EM DDA result, computed as `alpha_j * [E_j; H_j]` split into the
electric and magnetic halves.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `EMDDAResult3D` | -- | A solved coupled EM DDA result. |

**Returns:** Tuple `(q, m)`, each a `Vector{CVec3}` of length `grid.nvoxels`.

---

### `scattered_fields_em_dda_3d(result, r_obs)`

Compute the scattered electric and magnetic fields at observation points by
summing the induced electric and magnetic dipoles. Observation points must not
coincide with voxel centers.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `EMDDAResult3D` | -- | A solved coupled EM DDA result. |
| `r_obs` | `AbstractVector{Vec3}` | -- | Observation points (m). |

**Returns:** Tuple `(E_scat, H_scat)`, each a `Vector{CVec3}` of length `length(r_obs)`.

---

### `farfield_em_dda_3d(result, rhat; eta0=376.730313668)`

Return `(F_E, F_H)` such that `E_scat ~= exp(-i k r) F_E / r` and
`H_scat ~= exp(-i k r) F_H / r` in observation direction `rhat`. A
`Vector{Vec3}` may be passed to obtain `Vector{CVec3}` outputs.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `EMDDAResult3D` | -- | A solved coupled EM DDA result. |
| `rhat` | `Vec3` or `AbstractVector{Vec3}` | -- | Observation direction(s). |
| `eta0` | `Real` | `376.730313668` | Free-space impedance (Ohm). |

**Returns:** Tuple `(F_E, F_H)` of `CVec3` (single direction) or
`Vector{CVec3}` (multiple directions). In the deep far zone these satisfy the
radiation condition `F_E = -eta0 * (n_hat x F_H)`.

---

## 4. FFT-Accelerated Operators

The uniform Cartesian grid gives the dipole interaction matrix a block-Toeplitz
structure, so the dense all-pairs sum can be replaced by a zero-padded
convolution evaluated with FFTs. These operators produce matvecs that match the
dense / direct operators to machine precision while scaling far better with
voxel count. They are used by `solve_dda_3d`/`solve_em_dda_3d` via GMRES or via
the `:fft_gmres` solver mode.

### `FFTDDAKernel3D`

Fourier-space block-Toeplitz embedding of the free-space electric DDA dyadic for
a uniform `VoxelGrid3D`. The stored kernel excludes the singular self offset.
Constructed by `fft_dda_kernel_3d`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `pad_dims` | `NTuple{3,Int}` | Padded FFT dimensions `(2nx-1, 2ny-1, 2nz-1)`. |
| `kernel_hat` | `Array{ComplexF64,5}` | FFT of the spatial dyadic kernel, indexed `[ix, iy, iz, a, b]`. |

---

### `fft_dda_kernel_3d(grid, k0)`

Build the `FFTDDAKernel3D` for the electric DDA operator: it sweeps Cartesian
grid offsets `(ox, oy, oz)` (excluding the origin), fills the `3x3` dyadic
`electric_dipole_dyadic_3d` into a zero-padded array, and stores the FFT of each
of the 9 component arrays.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m), must be positive. |

**Returns:** `FFTDDAKernel3D`.

---

### `FFTDDAOperator3D`

FFT-accelerated coupled-dipole operator for a uniform `VoxelGrid3D`. It applies
`y = x - G * (alpha .* x)` with the self interaction excluded, matching
`DDAOperator3D`. Behaves as an `AbstractMatrix{ComplexF64}` of size
`(3*nvoxels, 3*nvoxels)`. Constructed by `fft_dda_operator_3d`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `VoxelGrid3D` | The voxel grid. |
| `k0` | `Float64` | Wavenumber (rad/m). |
| `eps_r` | `AbstractVector` | Per-voxel relative permittivity. |
| `alpha` | `AbstractVector` | Per-voxel polarizability. |
| `radiative_correction` | `Bool` | Whether the radiation-reaction correction was applied. |
| `kernel` | `FFTDDAKernel3D` | The precomputed Fourier kernel. |
| `qhat`, `conv` | `Array{ComplexF64}` | Preallocated FFT workspaces reused across matvecs. |

---

### `fft_dda_operator_3d(grid, k0, eps_r; radiative_correction=false)`

Construct an `FFTDDAOperator3D`. The matvec matches `dda_operator_3d` while
replacing the dense all-pairs sum by the zero-padded block-Toeplitz convolution.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m), must be positive. |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** `FFTDDAOperator3D`.

**Example:**

```julia
A_fft = fft_dda_operator_3d(grid, k, fill(2.2 + 0.03im, grid.nvoxels))
y = zeros(ComplexF64, size(A_fft, 1))
mul!(y, A_fft, x)   # matches dda_operator_3d matvec to machine precision
```

---

### `FFTEMDDAKernel3D`

Fourier-space block-Toeplitz embedding of the coupled electric-magnetic DDA
interaction. The stored kernel maps induced `[q; m]` dipoles to scattered
`[E; H]` fields and excludes the singular self offset. Constructed by
`fft_em_dda_kernel_3d`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `pad_dims` | `NTuple{3,Int}` | Padded FFT dimensions `(2nx-1, 2ny-1, 2nz-1)`. |
| `kernel_hat` | `Array{ComplexF64,5}` | FFT of the `6x6` spatial kernel, indexed `[ix, iy, iz, a, b]`. |

---

### `fft_em_dda_kernel_3d(grid, k0)`

Build the `FFTEMDDAKernel3D` for the coupled EM DDA operator by sweeping
Cartesian grid offsets, evaluating the `6x6` electromagnetic interaction at each
offset, and storing the FFT of each component array.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m), must be positive. |

**Returns:** `FFTEMDDAKernel3D`.

---

### `FFTEMDDAOperator3D`

FFT-accelerated coupled electric-magnetic DDA operator. It applies
`y = x - G_em * (alpha6 * x)` with the self interaction excluded, matching
`EMDDAOperator3D`. Behaves as an `AbstractMatrix{ComplexF64}` of size
`(6*nvoxels, 6*nvoxels)`. Constructed by `fft_em_dda_operator_3d`.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `VoxelGrid3D` | The voxel grid. |
| `k0` | `Float64` | Wavenumber (rad/m). |
| `alpha` | `AbstractVector` | Per-voxel `6x6` polarizability. |
| `radiative_correction` | `Bool` | Whether the radiation-reaction correction was applied. |
| `kernel` | `FFTEMDDAKernel3D` | The precomputed Fourier kernel. |
| `qhat`, `conv` | `Array{ComplexF64}` | Preallocated FFT workspaces reused across matvecs. |

---

### `fft_em_dda_operator_3d(grid, k0, eps_r, mu_r; radiative_correction=false)`

Construct an `FFTEMDDAOperator3D`. Additional methods accept explicit per-voxel
`6x6` polarizabilities (`fft_em_dda_operator_3d(grid, k0, alpha6; ...)`) or a
`BianisotropicMaterial3D` (with an extra `eta0` keyword). This operator backs the
`:fft_gmres` solver mode of `solve_em_dda_3d`.

**Parameters (magnetodielectric method):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `VoxelGrid3D` | -- | The voxel grid. |
| `k0` | `Real` | -- | Wavenumber (rad/m), must be positive. |
| `eps_r` | scalar / tensor / vector | -- | Relative permittivity specification. |
| `mu_r` | scalar / tensor / vector | -- | Relative permeability specification. |
| `radiative_correction` | `Bool` | `false` | Apply the radiation-reaction correction. |

**Returns:** `FFTEMDDAOperator3D`.

**Example:**

```julia
Einc, Hinc = planewave_em_dda_3d(grid, Vec3(0.0, 0.0, k), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
res = solve_em_dda_3d(grid, k, 2.3 + 0im, 1.4 + 0im, Einc, Hinc; solver=:fft_gmres)
```

---

## 5. Adjoint / Gradient

These functions provide material design sensitivities for the electric DDA path
via the adjoint method, mirroring the surface-EFIE adjoint workflow in
[adjoint-optimize.md](adjoint-optimize.md).

### `solve_dda_adjoint_3d(result, grad_E_flat; solver=:direct, tol=1e-8, maxiter=200, memory=20, verbose=false)`

Solve the 3D DDA adjoint system `A' * lambda = grad_E_flat` for an existing
`DDAResult3D`. For an objective `J = real(E' * Q * E)`, pass
`grad_E_flat = Q * E`; `gradient_epsr_dda_3d` applies the corresponding factor
of two for real design parameters. The `:direct` mode reuses the stored LU
factorization from the forward solve (O(N^2) per call) when available.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `DDAResult3D` | -- | A solved electric DDA result. |
| `grad_E_flat` | `AbstractVector` | -- | Adjoint RHS, length `nvoxels` (of `CVec3`) or `3*nvoxels` (flat). |
| `solver` | `Symbol` | `:direct` | `:direct` (LU/adjoint solve) or `:gmres`. |
| `tol` | `Float64` | `1e-8` | GMRES relative tolerance. |
| `maxiter` | `Int` | `200` | Maximum GMRES iterations. |
| `memory` | `Int` | `20` | GMRES restart memory. |
| `verbose` | `Bool` | `false` | Print GMRES progress. |

**Returns:** `Vector{ComplexF64}` adjoint variable `lambda` of length `3*nvoxels`.

---

### `gradient_epsr_dda_3d(result, lambda)`

Return the real gradient with respect to one real scalar `eps_r` design
parameter per voxel, using

```
alpha = 3V * (eps_r - 1) / (eps_r + 2)
d alpha / d eps_r = 9V / (eps_r + 2)^2
```

and the DDA system convention `A_ij = delta_ij - G_ij * alpha_j`. This currently
supports only the uncorrected Clausius-Mossotti polarizability; it raises an
error if `result.radiative_correction` is `true`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `DDAResult3D` | -- | A solved electric DDA result. |
| `lambda` | `AbstractVector` | -- | Adjoint variable from `solve_dda_adjoint_3d` (length `nvoxels` or `3*nvoxels`). |

**Returns:** `Vector{Float64}` real gradient of length `nvoxels`.

**Example:**

```julia
res = solve_dda_3d(grid, k0, epsr, E_inc)
E = reduce(vcat, res.E_total)
lambda = solve_dda_adjoint_3d(res, weights .* E)
grad = gradient_epsr_dda_3d(res, lambda)   # d J / d eps_r per voxel
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/mom3d/Types3D.jl` | `VoxelGrid3D`, `make_voxel_grid_3d`, `DDAOperator3D`, `DDAAdjointOperator3D`, `DDAResult3D`, `EMDDAOperator3D`, `EMDDAResult3D` |
| `src/mom3d/DDA3D.jl` | `clausius_mossotti_polarizability`, `dda_polarizabilities`, `dda_operator_3d`, `electric_dipole_dyadic_3d`, `assemble_dda_3d`, `solve_dda_3d`, `planewave_dda_3d`, `induced_dipoles_dda_3d`, `scattered_field_dda_3d`, `farfield_dda_3d` |
| `src/mom3d/EMDDA3D.jl` | `BianisotropicPolarizability3D`, `magnetic_clausius_mossotti_polarizability`, `bianisotropic_clausius_mossotti_polarizability`, `em_dda_polarizabilities`, `em_dda_operator_3d`, `assemble_em_dda_3d`, `solve_em_dda_3d`, `planewave_em_dda_3d`, `induced_dipoles_em_dda_3d`, `scattered_fields_em_dda_3d`, `farfield_em_dda_3d` |
| `src/mom3d/FFTDDA3D.jl` | `FFTDDAKernel3D`, `FFTDDAOperator3D`, `fft_dda_kernel_3d`, `fft_dda_operator_3d`, `FFTEMDDAKernel3D`, `FFTEMDDAOperator3D`, `fft_em_dda_kernel_3d`, `fft_em_dda_operator_3d` |
| `src/mom3d/Adjoint3D.jl` | `solve_dda_adjoint_3d`, `gradient_epsr_dda_3d` |

For the constitutive material models (`MagneticMaterial3D`,
`BianisotropicMaterial3D`, and the constitutive-tensor builders) consumed by the
coupled-EM constructors, see [material-models-3d.md](material-models-3d.md).
