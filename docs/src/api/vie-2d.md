# API: 2D Volume Integral Equation (TM)

## Purpose

Reference for the 2D TM (transverse-magnetic, `E_z`-only) volume integral equation
(VIE) Method of Moments subsystem. This solver handles scattering from an
inhomogeneous dielectric domain discretized on a uniform rectangular grid with a
pulse basis and point matching. The governing equation is

```
E_z(r) = E_z_inc(r) + k0^2 integral_D chi(r') G_2D(r, r') E_z(r') dA'
```

where `chi(r) = eps_r(r) - 1` is the dielectric contrast and `G_2D` is the 2D
scalar Green's function. The subsystem covers mesh construction, Green's function
evaluation, system assembly and solve, plane-wave / line-source excitation,
scattered-field evaluation, the contrast Jacobian for differentiable design, and
a 2D Mie-series reference for circular cylinders.

The package uses the `exp(+i omega t)` time convention, so outgoing waves are
represented by the Hankel function of the second kind `H0(2)`. For the 3D surface
MoM types and pipeline, see [types.md](types.md) and
[assembly-solve.md](assembly-solve.md).

**See also:** for the theory and a worked walkthrough, see
[2D Volume Integral Equation (TM Polarization)](../formulations/01-2d-volume-ie.md).

---

## Types and Mesh

### `Vec2`

Type alias `SVector{2,Float64}`: a stack-allocated 2-component real vector used
for positions and directions in the 2D plane.

**Usage:**

```julia
using StaticArrays
r = Vec2(0.5, -0.25)   # (x, y) position in meters
```

---

### `CVec2`

Type alias `SVector{2,ComplexF64}`: a stack-allocated 2-component complex vector
(complex phasor counterpart of `Vec2`).

**Usage:**

```julia
using StaticArrays
cv = CVec2(1.0 + 0.0im, 0.0 + 1.0im)
```

---

### `Mesh2D`

Rectangular grid discretization of a 2D domain for the VIE. Each cell has constant
material properties and field values (pulse basis). Construct with the
`Mesh2D(x_range, y_range, nx, ny)` constructor described below.

```julia
struct Mesh2D
    centers::Vector{Vec2}   # cell center coordinates
    cell_area::Float64      # uniform cell area (dx * dy)
    ncells::Int
    nx::Int                 # cells in x
    ny::Int                 # cells in y
    dx::Float64             # cell width
    dy::Float64             # cell height
    x0::Float64             # domain lower-left x
    y0::Float64             # domain lower-left y
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `centers` | `Vector{Vec2}` | Cell center coordinates, length `ncells`. Ordered row-major: `ix` varies fastest, then `iy`. |
| `cell_area` | `Float64` | Uniform area of each cell, `dx * dy` (m^2). |
| `ncells` | `Int` | Total number of cells, `nx * ny` (= dimension N of the VIE system). |
| `nx` | `Int` | Number of cells in the x direction. |
| `ny` | `Int` | Number of cells in the y direction. |
| `dx` | `Float64` | Cell width in x (m). |
| `dy` | `Float64` | Cell height in y (m). |
| `x0` | `Float64` | Domain lower-left x coordinate (m). |
| `y0` | `Float64` | Domain lower-left y coordinate (m). |

---

### `Mesh2D(x_range, y_range, nx, ny)`

Create a uniform rectangular grid over `[x_range[1], x_range[2]] x [y_range[1], y_range[2]]`
with `nx x ny` cells. Cell centers are placed at the midpoint of each cell.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_range` | `Tuple{Float64,Float64}` | -- | Lower and upper x bounds of the domain (m). Must be increasing. |
| `y_range` | `Tuple{Float64,Float64}` | -- | Lower and upper y bounds of the domain (m). Must be increasing. |
| `nx` | `Int` | -- | Number of cells in x. Must be `>= 1`. |
| `ny` | `Int` | -- | Number of cells in y. Must be `>= 1`. |

**Returns:** `Mesh2D` with `ncells = nx * ny` cells.

**Example:**

```julia
mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 5, 5)
println("Cells: ", mesh.ncells, ", dx = ", mesh.dx, ", area = ", mesh.cell_area)
# Output: Cells: 25, dx = 0.2, area = 0.04
```

---

### `VIEResult2D`

Result bundle from a 2D VIE forward solve, returned by `solve_vie_2d`. It carries
the total and incident fields, the contrast profile, the assembled matrices, the
cached LU factorization, the mesh, and the wavenumber, so downstream routines
(`scattered_field_2d`, `jacobian_scattered_field_2d`) can reuse them.

```julia
struct VIEResult2D
    E_total::Vector{ComplexF64}   # total field at cell centers
    E_inc::Vector{ComplexF64}     # incident field at cell centers
    chi::Vector{Float64}          # contrast profile (eps_r - 1)
    D::Matrix{ComplexF64}         # Green's function integral matrix
    Z::Matrix{ComplexF64}         # system matrix (I - k0^2 diag(chi) D)
    Z_LU::LinearAlgebra.LU{ComplexF64, Matrix{ComplexF64}, Vector{Int64}}
    mesh::Mesh2D
    k0::Float64                   # free-space wavenumber
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `E_total` | `Vector{ComplexF64}` | Total field `E_z` at the cell centers (length `ncells`). |
| `E_inc` | `Vector{ComplexF64}` | Incident field `E_z_inc` at the cell centers (length `ncells`). |
| `chi` | `Vector{Float64}` | Dielectric contrast `eps_r - 1` per cell (length `ncells`). |
| `D` | `Matrix{ComplexF64}` | Green's function integral matrix `D[m,n]` (size `ncells x ncells`). |
| `Z` | `Matrix{ComplexF64}` | System matrix `Z = I - k0^2 diag(chi) D` (size `ncells x ncells`). |
| `Z_LU` | `LU{ComplexF64, ...}` | Cached LU factorization of `Z` (reused by the Jacobian). |
| `mesh` | `Mesh2D` | The mesh used for the solve. |
| `k0` | `Float64` | Free-space wavenumber (rad/m). |

---

### `equivalent_radius(mesh)`

Equivalent circular-cell radius used by the self-cell integration: the radius `a`
of a disk whose area equals one grid cell, i.e. `pi * a^2 = cell_area`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh2D` | -- | The 2D grid. |

**Returns:** `Float64` equivalent radius `sqrt(cell_area / pi)` (m).

**Example:**

```julia
mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 5, 5)
a_eq = equivalent_radius(mesh)
println(pi * a_eq^2 ≈ mesh.cell_area)   # true
```

---

## Green's Function

### `greens_2d(r, rp, k)`

2D scalar free-space Green's function for the Helmholtz equation, satisfying
`(laplacian + k^2) G = -delta(r - rp)`:

```
G_2D(r, rp) = (-i/4) H0(2)(k |r - rp|)
```

with the `exp(+i omega t)` convention and outgoing Hankel function `H0(2)`. As a
guard against the singularity at coincident points, the function returns `0` when
the separation is below `1e-30`; the self-cell contribution is supplied separately
by `self_cell_integral_2d`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | `Vec2` | -- | Observation point (m). |
| `rp` | `Vec2` | -- | Source point (m). |
| `k` | `Float64` | -- | Wavenumber (rad/m). |

**Returns:** `ComplexF64` value of `G_2D(r, rp)` (or `0 + 0im` for coincident points).

**Example:**

```julia
k = 2pi
G = greens_2d(Vec2(0.0, 0.0), Vec2(1.0, 0.0), k)
```

---

### `self_cell_integral_2d(k, a_eq)`

Analytical integral of `G_2D` over a circular cell of radius `a_eq`, used to fill
the diagonal of the Green's integral matrix where the midpoint rule is singular:

```
integral_{|rp| <= a_eq} G_2D(0, rp) dA' = (-i pi / (2 k^2)) [k a_eq H1(2)(k a_eq) - 2i/pi]
```

derived from `d/du [u H1(2)(u)] = u H0(2)(u)`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `Float64` | -- | Wavenumber (rad/m). Must be `> 0`. |
| `a_eq` | `Float64` | -- | Equivalent cell radius (m). Must be `> 0`. Typically `equivalent_radius(mesh)`. |

**Returns:** `ComplexF64` value of the self-cell integral.

**Example:**

```julia
k = 2pi
mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 5, 5)
D_self = self_cell_integral_2d(k, equivalent_radius(mesh))
```

---

## Assembly and Solve

### `assemble_vie_2d(mesh, k0, chi)`

Assemble the VIE system matrix using pulse basis / point matching:

```
Z[m,n] = delta[m,n] - k0^2 chi[n] D[m,n]
```

where `D[m,n] = integral_{cell_n} G_2D(r_m, rp) dA'` is the Green's integral matrix
(off-diagonal via midpoint rule `G_2D(r_m, r_n) * cell_area`, diagonal via
`self_cell_integral_2d`). For `chi = 0` (free space) the result is the identity.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh2D` | -- | The 2D grid. |
| `k0` | `Float64` | -- | Free-space wavenumber (rad/m). |
| `chi` | `AbstractVector{Float64}` | -- | Per-cell dielectric contrast `eps_r - 1` (length `ncells`). |

**Returns:** Tuple `(Z, D)` of `Matrix{ComplexF64}`, each of size `ncells x ncells`:
the system matrix `Z` and the Green's integral matrix `D`.

**Example:**

```julia
k0 = 2pi
mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 5, 5)
chi = fill(1.0, mesh.ncells)        # eps_r = 2
Z, D = assemble_vie_2d(mesh, k0, chi)
```

---

### `solve_vie_2d(mesh, k0, chi, E_inc)`

Solve the 2D VIE for the internal total field by assembling `Z` (via
`assemble_vie_2d`), LU-factorizing it, and solving `Z * E_total = E_inc`. Bundles
all computed quantities into a `VIEResult2D` for downstream scattered-field and
Jacobian evaluation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh2D` | -- | The 2D grid. |
| `k0` | `Float64` | -- | Free-space wavenumber (rad/m). |
| `chi` | `AbstractVector{Float64}` | -- | Per-cell dielectric contrast (length `ncells`). |
| `E_inc` | `AbstractVector{ComplexF64}` | -- | Incident field at cell centers (length `ncells`), e.g. from `planewave_2d` or `linesource_2d`. |

**Returns:** `VIEResult2D` with the total field `E_total`, the incident field, the
contrast profile, the matrices `D` and `Z`, the LU factorization, the mesh, and
`k0`.

**Example:**

```julia
k0 = 2pi
mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 5, 5)
chi = fill(1.0, mesh.ncells)
E_inc = planewave_2d(mesh, k0, 0.0)
vr = solve_vie_2d(mesh, k0, chi, E_inc)
```

---

## Excitation

### `planewave_2d(mesh, k0, phi_inc; E0=1.0)`

Generate an incident TM plane wave sampled at cell centers. The propagation
direction is `k_hat = (cos(phi_inc), sin(phi_inc))` and

```
E_z_inc(r) = E0 exp(-i k0 k_hat . r)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh2D` | -- | The 2D grid. |
| `k0` | `Float64` | -- | Free-space wavenumber (rad/m). |
| `phi_inc` | `Float64` | -- | Incidence angle (rad); `0` is propagation along +x. |
| `E0` | `Float64` | `1.0` | Plane-wave amplitude. |

**Returns:** `Vector{ComplexF64}` of incident-field values at cell centers (length `ncells`).

**Example:**

```julia
mesh = Mesh2D((-1.0, 1.0), (-1.0, 1.0), 4, 4)
E_inc = planewave_2d(mesh, 2pi, pi/4)
```

---

### `linesource_2d(mesh, k0, r_src)`

Generate the incident field of a unit-amplitude 2D line source located at `r_src`,
sampled at cell centers:

```
E_z_inc(r) = (-i/4) H0(2)(k0 |r - r_src|)
```

This equals `greens_2d(r, r_src, k0)`. The source should lie outside the
scattering domain.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh2D` | -- | The 2D grid. |
| `k0` | `Float64` | -- | Free-space wavenumber (rad/m). |
| `r_src` | `Vec2` | -- | Line-source position (m). |

**Returns:** `Vector{ComplexF64}` of incident-field values at cell centers (length `ncells`).

**Example:**

```julia
mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 6, 6)
E_inc = linesource_2d(mesh, 2pi, Vec2(3.0, 0.0))
```

---

## Scattered Field and Jacobian

### `green_obs_matrix(r_obs, mesh, k0)`

Compute the observation Green's function matrix `G_obs[m,n] = G_2D(r_obs[m], r_n)`,
mapping each cell center `r_n` to each observation point. Observation points must
be outside the scattering domain.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r_obs` | `AbstractVector{Vec2}` | -- | Observation points (m), length M. |
| `mesh` | `Mesh2D` | -- | The 2D grid. |
| `k0` | `Float64` | -- | Free-space wavenumber (rad/m). |

**Returns:** `Matrix{ComplexF64}` of size `M x ncells`.

**Example:**

```julia
mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 6, 6)
r_obs = [Vec2(2.0, 0.0), Vec2(0.0, 2.0)]
G_obs = green_obs_matrix(r_obs, mesh, 2pi)
```

---

### `scattered_field_2d(vie_result, r_obs)`

Compute the scattered field at observation points from a solved VIE result:

```
E_scat(r_obs) = k0^2 sum_n chi_n E_n G_2D(r_obs, r_n) A_n
```

where `A_n = cell_area`, `E_n = vr.E_total`, and the Green's values come from
`green_obs_matrix`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vie_result` | `VIEResult2D` | -- | Solved result from `solve_vie_2d`. |
| `r_obs` | `AbstractVector{Vec2}` | -- | Observation points (m), length M (outside the domain). |

**Returns:** `Vector{ComplexF64}` of scattered-field values, length M.

**Example:**

```julia
vr = solve_vie_2d(mesh, k0, chi, planewave_2d(mesh, k0, 0.0))
r_obs = [Vec2(1.5*cos(p), 1.5*sin(p)) for p in range(0, 2pi, length=37)[1:36]]
E_scat = scattered_field_2d(vr, r_obs)
```

---

### `jacobian_scattered_field_2d(vie_result, r_obs)`

Compute the Jacobian of the scattered field with respect to the per-cell contrast,
`J[m,p] = d E_scat(r_obs[m]) / d chi_p`, via implicit differentiation of the VIE
system. Since `Z E = E_inc`, the field sensitivity is
`dE/dchi_p = k0^2 E_p Z^-1 D[:,p]`; the routine precomputes `S = Z^-1 D` (reusing
the cached `Z_LU`) for efficiency.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vie_result` | `VIEResult2D` | -- | Solved result from `solve_vie_2d`. |
| `r_obs` | `AbstractVector{Vec2}` | -- | Observation points (m), length M (outside the domain). |

**Returns:** Tuple `(J, G_obs)`:
- `J::Matrix{ComplexF64}` of size `M x ncells` (the Jacobian).
- `G_obs::Matrix{ComplexF64}` of size `M x ncells` (the observation Green's matrix).

**Example:**

```julia
vr = solve_vie_2d(mesh, k0, chi, E_inc)
J, G_obs = jacobian_scattered_field_2d(vr, r_obs)
```

---

## Mie-Series Reference

These functions provide the exact 2D TM solution for a homogeneous circular
cylinder, used to validate the VIE solver (analogous to the 3D
`mie_bistatic_rcs_pec` reference in [farfield-rcs.md](farfield-rcs.md)).

### `mie_coefficients_2d(k0, a, eps_r; nmax=nothing, pec=false)`

Compute the 2D Mie scattering coefficients `c_n` for a circular cylinder. For a
PEC cylinder, `c_n = -J_n(k0 a) / H_n(2)(k0 a)`; for a dielectric cylinder the
coefficients enforce field/derivative continuity at `rho = a`. The coefficients
are returned indexed from `-N:N`, stored at `c[n + N + 1]`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k0` | `Float64` | -- | Free-space wavenumber (rad/m). Must be `> 0`. |
| `a` | `Float64` | -- | Cylinder radius (m). Must be `> 0`. |
| `eps_r` | `Float64` | -- | Relative permittivity (ignored when `pec=true`). |
| `nmax` | `Union{Nothing,Int}` | `nothing` | Maximum order N. Auto-determined from `k0 a` when `nothing`. |
| `pec` | `Bool` | `false` | If `true`, compute PEC cylinder coefficients. |

**Returns:** Tuple `(c, N)`:
- `c::Vector{ComplexF64}` of length `2N + 1` (coefficient for order `n` at index `n + N + 1`).
- `N::Int` the maximum order used.

**Example:**

```julia
c, N = mie_coefficients_2d(2pi, 0.5, 1.0; pec=true)
c0 = c[N + 1]   # n = 0 coefficient
```

---

### `mie_scattered_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0, nmax=nothing, pec=false)`

Compute the exact scattered field at observation points for a circular cylinder:

```
E_z_scat(rho, phi) = E0 sum_n (-i)^n c_n H_n(2)(k0 rho) exp(i n (phi - phi_inc))
```

evaluated with unit incident amplitude.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k0` | `Float64` | -- | Free-space wavenumber (rad/m). |
| `a` | `Float64` | -- | Cylinder radius (m). |
| `eps_r` | `Float64` | -- | Relative permittivity (ignored when `pec=true`). |
| `r_obs` | `AbstractVector{Vec2}` | -- | Observation positions (m), length M. |
| `phi_inc` | `Float64` | `0.0` | Incidence angle (rad); `0` is +x. |
| `nmax` | `Nothing` or `Int` | `nothing` | Maximum Mie order (auto if `nothing`). |
| `pec` | `Bool` | `false` | If `true`, treat as a PEC cylinder. |

**Returns:** `Vector{ComplexF64}` of scattered-field values, length M.

**Example:**

```julia
a = 0.1
r_obs = [Vec2(3a*cos(p), 3a*sin(p)) for p in range(0, 2pi, length=37)[1:36]]
E_scat_mie = mie_scattered_field_2d(2pi, a, 4.0, r_obs; phi_inc=0.0)
```

---

### `mie_total_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0, nmax=nothing, pec=false)`

Compute the exact total field (incident plus scattered) at observation points
outside the cylinder (`rho > a`). The incident term is the plane wave
`exp(-i k0 k_hat . r)` with `k_hat = (cos(phi_inc), sin(phi_inc))`, and the
scattered term is `mie_scattered_field_2d`.

**Parameters:** Same as `mie_scattered_field_2d` (`k0`, `a`, `eps_r`, `r_obs`;
keywords `phi_inc=0.0`, `nmax=nothing`, `pec=false`).

**Returns:** `Vector{ComplexF64}` of total-field values, length M.

**Example:**

```julia
# Total field just outside a PEC cylinder surface (should be near zero)
r_surf = [Vec2(0.5*cos(p), 0.5*sin(p)) for p in range(0, 2pi, length=37)[1:36]]
E_tot = mie_total_field_2d(2pi, 0.5, 1.0, r_surf; pec=true)
```

---

## End-to-End Example

The following mirrors the MoM-vs-Mie convergence check in `test/test_mom2d.jl`: a
circular dielectric cylinder is approximated on a rectangular grid, and the VIE
scattered field is compared against the 2D Mie series.

```julia
using DiffMoM, LinearAlgebra

freq = 1e9; c0 = 3e8; lambda = c0 / freq; k0 = 2pi / lambda
a = 0.1 * lambda; eps_r = 4.0; chi_val = eps_r - 1.0

# Observation ring outside the cylinder
r_obs = [Vec2(3a*cos(p), 3a*sin(p)) for p in range(0, 2pi, length=37)[1:36]]
E_scat_mie = mie_scattered_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0)

# VIE solve on a 40x40 grid bounding the cylinder
mesh = Mesh2D((-a, a), (-a, a), 40, 40)
chi = zeros(mesh.ncells)
for i in 1:mesh.ncells
    r = sqrt(mesh.centers[i][1]^2 + mesh.centers[i][2]^2)
    r <= a && (chi[i] = chi_val)
end

E_inc = planewave_2d(mesh, k0, 0.0)
vr = solve_vie_2d(mesh, k0, chi, E_inc)
E_scat_mom = scattered_field_2d(vr, r_obs)

rel_err = norm(E_scat_mom - E_scat_mie) / norm(E_scat_mie)
println("Relative error: ", rel_err)

# Contrast Jacobian for differentiable design
J, _ = jacobian_scattered_field_2d(vr, r_obs)   # size (length(r_obs), mesh.ncells)
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/mom2d/Types2D.jl` | `Vec2`, `CVec2`, `Mesh2D`, `VIEResult2D`, `equivalent_radius` |
| `src/mom2d/Greens2D.jl` | `greens_2d`, `self_cell_integral_2d` |
| `src/mom2d/Assembly2D.jl` | `assemble_vie_2d`, `solve_vie_2d` |
| `src/mom2d/Excitation2D.jl` | `planewave_2d`, `linesource_2d` |
| `src/mom2d/Scatter2D.jl` | `scattered_field_2d`, `green_obs_matrix`, `jacobian_scattered_field_2d` |
| `src/mom2d/Mie2D.jl` | `mie_coefficients_2d`, `mie_scattered_field_2d`, `mie_total_field_2d` |
