# API: Periodic Methods

## Purpose

Reference for 2D-periodic scattering APIs:

- Ewald-accelerated periodic Green's correction.
- Periodic EFIE assembly.
- Floquet-mode post-processing for periodic unit cells.

These APIs are intended for unit-cell metasurface analyses with Bloch-periodic phase shift.

---

## `PeriodicLattice`

Lattice/incident-wave parameters used by periodic Green's and periodic EFIE.

```julia
struct PeriodicLattice
    dx::Float64
    dy::Float64
    kx_bloch::Float64
    ky_bloch::Float64
    k::Float64
    E::Float64
    N_spatial::Int
    N_spectral::Int
end
```

### Constructor

```julia
PeriodicLattice(dx, dy, theta_inc, phi_inc, k; N_spatial=4, N_spectral=4)
```

Inputs:

- `dx`, `dy`: unit-cell periods (meters).
- `theta_inc`, `phi_inc`: incident angles (radians).
- `k`: free-space wavenumber.
- `N_spatial`, `N_spectral`: minimum truncation orders for Ewald sums.

Constructor behavior:
- Computes Bloch transverse wavenumbers `kx_bloch`, `ky_bloch`.
- Auto-selects Ewald splitting `E` for numerical stability.
- Auto-enlarges spectral truncation when needed to include propagating Floquet modes.

---

## Periodic Green's Correction

### `greens_periodic_correction(r, rp, k, lattice)`

Compute periodic correction:

```
DeltaG(r,rp) = G_per(r,rp) - G_0(r,rp)
```

using Helmholtz-Ewald decomposition (self-correction + spatial images + spectral sum).

Current implementation scope:
- Intended for coplanar periodic unit-cell surfaces (`z = const`).
- Non-coplanar point pairs (`|z-z'| > 1e-12`) are rejected at runtime.
- For periodic EFIE/Floquet postprocessing with boundary-touching conductors, use
  `build_rwg_periodic(mesh, lattice; ...)`; non-Bloch RWG input is rejected.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `r` | `SVector{3}` | Observation point. |
| `rp` | `SVector{3}` | Source point. |
| `k` | Real | Wavenumber (rad/m). |
| `lattice` | `PeriodicLattice` | Periodic lattice setup. |

**Returns:** `ComplexF64` correction value.

---

## Periodic EFIE Assembly

### `assemble_Z_efie_periodic(mesh, rwg, k, lattice; quad_order=3, eta0=376.730313668)`

Assemble periodic EFIE matrix:

```
Z_per = Z_free + Z_corr
```

where:
- `Z_free`: free-space EFIE from `assemble_Z_efie`.
- `Z_corr`: periodic image correction using `greens_periodic_correction`.

`Z_corr` uses standard product quadrature for all triangle pairs because `DeltaG` is smooth (no `1/R` singularity).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Geometry mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `k` | Real | -- | Wavenumber (rad/m). |
| `lattice` | `PeriodicLattice` | -- | Unit-cell periodic setup. |
| `quad_order` | `Int` | `3` | Triangle quadrature order. |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance. |

**Returns:** Dense periodic matrix `Matrix{ComplexF64}`.

---

## Floquet Post-Processing Types

### `FloquetMode`

One diffraction order `(m,n)` in Floquet decomposition.

```julia
struct FloquetMode
    m::Int
    n::Int
    kx::Float64
    ky::Float64
    kz::ComplexF64
    propagating::Bool
    theta_r::Float64
    phi_r::Float64
end
```

Fields:
- `propagating == true`: `kz` is real and the mode carries power.
- `propagating == false`: evanescent mode (`kz` imaginary), `theta_r` and `phi_r` are `NaN`.

---

## Floquet Metrics

### `floquet_modes(k, lattice; N_orders=3)`

Enumerate modes `(m,n)` for `m,n in [-N_orders, N_orders]` and classify each as propagating or evanescent.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `Real` | -- | Free-space wavenumber. |
| `lattice` | `PeriodicLattice` | -- | Periodic lattice and Bloch setup. |
| `N_orders` | `Int` | `3` | Truncation order in each lattice direction. |

**Returns:** `Vector{FloquetMode}`.

---

### `reflection_coefficients(mesh, rwg, I_coeffs, k, lattice; quad_order=3, N_orders=3, E0=1.0, pol=SVector(1.0, 0.0, 0.0), eta0=376.730313668)`

Compute complex reflection coefficients for propagating Floquet modes by integrating the current Fourier coefficient over the unit cell.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Unit-cell mesh. |
| `rwg` | `RWGData` | -- | RWG basis data. |
| `I_coeffs` | `Vector{<:Number}` | -- | Solved current coefficients. |
| `k` | `Real` | -- | Wavenumber. |
| `lattice` | `PeriodicLattice` | -- | Periodic setup. |
| `quad_order` | `Int` | `3` | Quadrature order for current integration. |
| `N_orders` | `Int` | `3` | Floquet order truncation. |
| `E0` | `Float64` | `1.0` | Incident field amplitude. |
| `pol` | `SVector{3,Float64}` | `SVector(1.0, 0.0, 0.0)` | Incident polarization vector used in coefficient projection. |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance. |

**Returns:** `(modes, R_coeffs)`:
- `modes::Vector{FloquetMode}`
- `R_coeffs::Vector{ComplexF64}` with the same length/order as `modes` (evanescent-mode entries remain zero)

---

### `transmission_coefficients(modes, R_coeffs; incident_order=(0, 0))`

Convert reflection amplitudes to transmitted Floquet amplitudes for the free-standing sheet model used by this module.

Convention:
- Incident order `(m,n) = incident_order`: evaluates both `1 + R` and `1 - R`
  and keeps the lower-amplitude branch (passive convention under sign/phase ambiguity).
- Other orders: `T = R`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modes` | `Vector{FloquetMode}` | -- | Floquet mode list. |
| `R_coeffs` | `Vector{ComplexF64}` | -- | Reflection amplitudes aligned with `modes`. |
| `incident_order` | `Tuple{Int,Int}` | `(0, 0)` | Which order carries the incident field. |

**Returns:** `Vector{ComplexF64}` transmission amplitudes with the same ordering as `modes`.

---

### `specular_rcs_objective(mesh, rwg, grid, k, lattice; quad_order=3, half_angle=pi/18, polarization=:x)`

Build a quadratic objective matrix `Q` that integrates scattered power inside a cone around the specular reflection direction.

Implementation details:
- Specular direction is derived from Bloch wavevector in `lattice`.
- Uses `direction_mask` to define the cone.
- Uses `radiation_vectors` + `build_Q`.
- Supported symbol values:
  - `:x`, `:theta`, `:tm` → `pol_linear_x(grid)` (`theta_hat` basis)
  - `:y`, `:phi`, `:te` → `pol_linear_y(grid)` (`phi_hat` basis)
- You may also pass a custom polarization matrix of size `(3, NΩ)`.

**Returns:** `Matrix{ComplexF64}` objective matrix.

---

### `power_balance(I_coeffs, Z_pen, A_cell, k, modes, R_coeffs; eta0=376.730313668, E0=1.0, transmission=:none, T_coeffs=nothing, incident_order=(0, 0))`

Compute periodic-cell power accounting:

- `P_inc`: incident power through cell area.
- `P_refl`: sum of propagating reflected Floquet powers.
- `P_abs`: absorbed penalty power `0.5 * Re(I' * Z_pen * I)`.
- `P_trans`: transmitted power (mode-dependent, via `transmission` option).
- `P_resid = P_inc - P_refl - P_abs - P_trans`.

**Returns:** Named tuple:

`(P_inc, P_refl, P_abs, P_trans, P_resid, refl_frac, abs_frac, trans_frac, resid_frac)`.

`transmission` options:
- `:none`: set `P_trans = 0`.
- `:closure`: infer `P_trans` from conservation residual (clamped to `[0, P_inc]`).
- `:floquet`: compute `P_trans` from transmitted Floquet amplitudes (`T_coeffs` or inferred via `transmission_coefficients`).

---

## Typical Pattern

```julia
lattice = PeriodicLattice(dx, dy, theta_inc, phi_inc, k)
rwg = build_rwg_periodic(mesh, lattice;
                         precheck=true, allow_boundary=true, require_closed=false)

Z_per = assemble_Z_efie_periodic(mesh, rwg, k, lattice)
I = solve_forward(Z_per, v)

modes, R = reflection_coefficients(mesh, rwg, I, k, lattice; N_orders=3)
pb = power_balance(I, Z_pen, lattice.dx * lattice.dy, k, modes, R)

Q_spec = specular_rcs_objective(mesh, rwg, grid, k, lattice; half_angle=pi/18)
```

---

## Notes

- `assemble_Z_efie_periodic` forms a dense matrix; for large RWG counts, memory scales as O(`N^2`) like dense free-space EFIE.
- `reflection_coefficients`/`power_balance` assume consistent normalization (`E0`, `eta0`, and unit-cell area) across all calls.
- `transmission_coefficients` applies the free-standing sheet convention used in this module; use explicit `T_coeffs` in `power_balance(...; transmission=:floquet)` when a different transmission model is required.
- In `power_balance`, `transmission=:closure` is an energy-closure estimate, not a mode-resolved transmission decomposition.
- For meshes with conductor edges on unit-cell boundaries, periodic EFIE and reflection extraction require Bloch-paired RWG data from `build_rwg_periodic(mesh, lattice; ...)`; non-Bloch RWG input is rejected with `ArgumentError`.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/basis/PeriodicGreens.jl` | `PeriodicLattice`, `greens_periodic_correction` |
| `src/assembly/PeriodicEFIE.jl` | `assemble_Z_efie_periodic` |
| `src/postprocessing/PeriodicMetrics.jl` | `FloquetMode`, `floquet_modes`, `reflection_coefficients`, `transmission_coefficients`, `specular_rcs_objective`, `power_balance` |

---

## Exercises

- **Basic:** Construct `PeriodicLattice(dx, dy, theta_inc, phi_inc, k)` and inspect how `N_spectral` changes as `dx` and `dy` increase.
- **Practical:** Solve one periodic case, compute `(modes, R_coeffs)`, then compare `power_balance` with `transmission=:none` and `transmission=:floquet`.
- **Challenge:** Build `Q = specular_rcs_objective(...)` for two incidence angles and verify the selected specular cone shifts consistently with Bloch phase (`kx_bloch`, `ky_bloch`).
