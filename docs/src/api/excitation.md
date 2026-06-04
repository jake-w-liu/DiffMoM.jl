# API: Excitation System

## Purpose

Reference for incident-field excitation assembly. The excitation system provides a unified interface for assembling the right-hand-side (RHS) vector `v` in the MoM equation `Z * I = v`. Different physical excitation types (plane waves, ports, dipoles, imported fields, pattern feeds) are all handled through a single `assemble_excitation` interface.

For full physics derivations and guidance on choosing between excitation types, see `fundamentals/06-excitation-theory-and-usage.md`.

If you plan to call `compute_total_field`, prefer constructing an excitation
object with `make_plane_wave`, `make_dipole`, `make_loop`, `make_pattern_feed`,
or `make_imported_excitation(...; kind=:electric_field)` and assembling via
`assemble_excitation(...)`. The legacy `assemble_v_plane_wave(...)` helper
remains valid for RHS assembly, but it does not by itself retain the excitation
object needed for total-field observation.

---

## Excitation Types

All excitation types inherit from `AbstractExcitation`. You create an excitation using a constructor function (e.g., `make_plane_wave`), then pass it to `assemble_excitation` to get the RHS vector.

### `PlaneWaveExcitation`

A uniform plane wave illumination -- the most common excitation for scattering problems and RCS computation.

**Fields:**
- `k_vec::Vec3`: Wave vector (rad/m). Direction of propagation is along `k_vec`; magnitude is `|k_vec| = k = 2*pi/lambda`.
- `E0::Float64`: Amplitude of the electric field (V/m).
- `pol::Vec3`: Polarization direction (unit vector, must be orthogonal to `k_vec`).

**Constructor:**
```julia
pw = make_plane_wave(k_vec, E0, pol)
```

**Example:**
```julia
# z-propagating, x-polarized plane wave at 1 GHz
freq = 1e9
lambda0 = 3e8 / freq
k = 2pi / lambda0
pw = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
v = assemble_excitation(mesh, rwg, pw)
```

---

### `DeltaGapExcitation`

A voltage source applied across a single RWG edge. This is the simplest antenna feed model: a uniform electric field is applied across the gap, driving current into the structure.

**Fields:**
- `edge::Int`: RWG edge index where the gap is placed.
- `voltage::ComplexF64`: Gap voltage (V).
- `gap_length::Float64`: Physical gap length (m). The excitation field is `V/gap_length` applied across the edge.

**Constructor:**
```julia
gap = make_delta_gap(edge, voltage, gap_length)
```

**Choosing the edge:** The delta-gap edge should be at the antenna feed point. For a center-fed dipole, place it at the middle edge.

---

### `PortExcitation`

Port excitation defined by a set of RWG edges with applied voltage. This generalizes `DeltaGapExcitation` to multi-edge ports.

**Fields:**
- `port_edges::Vector{Int}`: RWG edge indices forming the port.
- `voltage::ComplexF64`: Port voltage (V).
- `impedance::ComplexF64`: Port impedance (stored for model completeness; the present RHS assembly uses only `voltage` and `port_edges`).

**Note:** The RHS contribution is edge-localized: `v_m = V / l_m` for edges in the port, zero otherwise.

---

### `DipoleExcitation`

Electric or magnetic dipole source. The incident field from the dipole is computed at each quadrature point on the mesh and integrated against the RWG basis functions.

**Fields:**
- `position::Vec3`: Dipole position (m).
- `moment::CVec3`: Dipole moment (electric: C*m, magnetic: A*m^2).
- `orientation::Vec3`: Dipole orientation metadata (unit vector).
- `type::Symbol`: `:electric` or `:magnetic`.
- `frequency::Float64`: Source frequency (Hz).

**Constructor:**
```julia
dip = make_dipole(position, moment, orientation, type, frequency=1e9)
```

**Placement:** The dipole must not be placed on the mesh surface itself (the field is singular at the source). Place it at least a fraction of a wavelength away from the scatterer.

---

### `LoopExcitation`

Circular current loop source. Models a small magnetic source.

**Fields:**
- `center::Vec3`: Loop center (m).
- `normal::Vec3`: Loop normal direction (unit vector).
- `radius::Float64`: Loop radius (m).
- `current::ComplexF64`: Loop current (A).
- `frequency::Float64`: Source frequency (Hz).

**Constructor:**
```julia
lp = make_loop(center, normal, radius, current, frequency=1e9)
```

---

### `ImportedExcitation`

A general imported or distributed source model. You provide a function `source(r) -> CVec3` that returns the incident field (or surface current density) at any point `r` on the mesh. This is the most flexible excitation type, supporting externally computed fields from other solvers or measurements.

**Fields:**
- `source_func::Function`: Spatial function `source(r) -> CVec3`.
- `kind::Symbol`: How to interpret the source function:
  - `:electric_field` -- `source(r)` is the incident electric field `E_inc(r)` directly.
  - `:surface_current_density` -- `source(r)` is an impressed surface current density `J_s(r)`, mapped via `E_inc(r) = eta_equiv * J_s(r)` (local equivalent-sheet approximation).
- `eta_equiv::ComplexF64`: Equivalent sheet impedance, used only when `kind=:surface_current_density`.
- `min_quad_order::Int`: Minimum quadrature order for assembly (mapped to supported orders 1, 3, 4, 7).

**Constructors:**
```julia
# Recommended
exc = make_imported_excitation(source_func; kind=:electric_field,
                               eta_equiv=376.730313668+0im, min_quad_order=3)

# Direct constructor
exc = ImportedExcitation(source_func; kind=:electric_field, ...)
```

**When to use each `kind`:**

| `kind` | Use when... | Example |
|--------|-------------|---------|
| `:electric_field` | You know `E_inc(r)` at the scatterer (from external solver, analytical formula, etc.) | Horn antenna near-field import, external full-wave simulation |
| `:surface_current_density` | You have an impressed surface current `J_s(r)` and want a local equivalent-sheet mapping `E = eta * J_s` | Aperture-based feed models, sheet-like sources |

**Scope and limitations:**

`ImportedExcitation` is an **RHS field model** only. It does not solve an auxiliary radiation problem for the source. For `:surface_current_density`, the mapping `E = eta * J_s` is a local approximation valid near sheet-like or aperture sources; it is **not** a rigorous dyadic Green's function radiation integral.

For rigorous imported sources, compute `E_inc(r)` externally and use `kind=:electric_field`.

For total-field observation via `compute_total_field`, only
`ImportedExcitation(kind=:electric_field)` is supported. The
`:surface_current_density` mode remains a local RHS approximation and is
intentionally excluded from v1 total-field evaluation.

---

### `PatternFeedExcitation`

Incident field synthesized from imported spherical far-field coefficients. Use this when you have radiation pattern data (e.g., from a horn antenna measurement or simulation) and want to use it as a feed illuminating a reflector or scatterer.

**Fields:**
- `theta::Vector{Float64}`: Polar-angle grid (rad), strictly increasing in [0, pi].
- `phi::Vector{Float64}`: Azimuth grid (rad), strictly increasing over one open 2*pi period.
- `Ftheta::Matrix{ComplexF64}`: Far-field coefficient F_theta(theta, phi).
- `Fphi::Matrix{ComplexF64}`: Far-field coefficient F_phi(theta, phi).
- `frequency::Float64`: Frequency (Hz).
- `phase_center::Vec3`: Pattern phase-center location (m).
- `convention::Symbol`: `:exp_plus_iwt` or `:exp_minus_iwt` for the imported data.

**The field model is:**

```
E(r) = (exp(-ikR)/R) * [F_theta * theta_hat + F_phi * phi_hat]
```

where `R = |r - r_c|` is the distance from the phase center.

**Constructors:**
```julia
# From arrays
pf = make_pattern_feed(theta, phi, Ftheta, Fphi, frequency;
                        phase_center=Vec3(0,0,0), convention=:exp_plus_iwt)

# From pattern objects (RadiationPatterns.jl compatible)
pf = make_pattern_feed(Etheta_pattern, Ephi_pattern, frequency; ...)
```

The second form accepts pattern objects with `.x`, `.y`, `.U` fields (e.g., `RadiationPatterns.jl` `Pattern` objects).

**Important:** Use two complex patterns (E_theta and E_phi), not a power-only pattern, to preserve polarization and phase.

---

### `MultiExcitation`

Combination of multiple excitations with complex weights. The resulting RHS is a weighted sum of individual excitation vectors.

**Fields:**
- `excitations::Vector{AbstractExcitation}`: List of excitations.
- `weights::Vector{ComplexF64}`: Weight for each excitation.

**Constructor:**
```julia
multi = make_multi_excitation(excitations, weights=nothing)
```
If `weights` is not provided, equal weights (all 1.0) are used.

---

## Core Functions

### `plane_wave_field(r, k_vec, E0, pol)`

Evaluate a plane wave `E_inc(r) = pol * E0 * exp(-i k_vec . r)` at point `r`. Useful for computing incident fields at arbitrary points (e.g., for PO surface currents or field probes).

**Parameters:**
- `r::Vec3`: Observation point (meters).
- `k_vec::Vec3`: Wave vector (rad/m).
- `E0`: Amplitude (V/m).
- `pol::Vec3`: Polarization (unit vector).

**Returns:** `CVec3` electric field phasor.

**Convention:** Uses `exp(+iwt)` time convention, hence `exp(-i k_vec . r)` spatial phase.

---

### `pattern_feed_field(r, pat)`

Evaluate the incident electric field at position `r` generated by a `PatternFeedExcitation`. Interpolates the stored far-field pattern to the direction from the phase center to `r`, and applies the appropriate spherical-wave phase and amplitude decay.

**Parameters:**
- `r::Vec3`: Observation point (meters).
- `pat::PatternFeedExcitation`: Pattern feed excitation source.

**Returns:** `CVec3` electric field phasor at `r`.

**Details:** Computes the spherical angles `(theta, phi)` from `pat.phase_center` to `r`, bilinearly interpolates `F_theta` and `F_phi` from the stored pattern grid, applies time-convention correction if needed, and multiplies by `exp(-jkR)/R`.

---

### `assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)`

Assemble the excitation vector for a plane wave directly. This is a convenience function equivalent to `assemble_excitation(mesh, rwg, make_plane_wave(k_vec, E0, pol))`.

**Parameters:**
- `mesh::TriMesh`, `rwg::RWGData`: Mesh and basis data.
- `k_vec::Vec3`: Wave vector (rad/m).
- `E0`: Amplitude (V/m).
- `pol::Vec3`: Polarization (unit vector).
- `quad_order::Int=3`: Quadrature order on reference triangle.

**Returns:** `Vector{ComplexF64}` of length `N = rwg.nedges`.

---

### `assemble_excitation(mesh, rwg, excitation; quad_order=3)`

**The unified excitation assembly function.** Assembles the RHS vector `v` for any `AbstractExcitation` subtype.

**Parameters:**
- `mesh::TriMesh`: Triangle mesh.
- `rwg::RWGData`: RWG basis data.
- `excitation::AbstractExcitation`: Any excitation type.
- `quad_order::Int=3`: Quadrature order on reference triangle.

**Returns:** `Vector{ComplexF64}` excitation vector `v` of length `N`, where:

```
v[m] = -integral_{S_m} f_m(r) . E_inc_t(r) dS
```

The integral is over the support of basis `m`, and `E_inc_t` is the tangential component of the incident electric field.

**Dispatch:** Automatically selects the correct assembly method based on the excitation type:
- `PlaneWaveExcitation`: Quadrature integration of `E_inc = pol * E0 * exp(-ik.r)`
- `PortExcitation`: Delta-gap approximation across port edges
- `DeltaGapExcitation`: Single-edge delta gap
- `DipoleExcitation`: Field from electric/magnetic dipole formula
- `LoopExcitation`: Field from circular current loop
- `ImportedExcitation`: Quadrature integration of user-provided field function
- `PatternFeedExcitation`: Bilinear interpolation of spherical pattern data
- `MultiExcitation`: Weighted combination of component excitations

---

### `assemble_multiple_excitations(mesh, rwg, excitations; quad_order=3)`

Assemble RHS vectors for multiple excitations at once (multiple right-hand sides).

**Parameters:**
- `mesh::TriMesh`, `rwg::RWGData`: Mesh and basis data.
- `excitations::Vector{<:AbstractExcitation}`: List of excitations.
- `quad_order::Int=3`: Quadrature order.

**Returns:** `Matrix{ComplexF64}` of size `N x M` where each column is the RHS for one excitation.

---

## Assembly Details

### Plane Wave
The excitation vector is computed via quadrature:

```
v_m = -integral_{S_m} f_m(r) . E_inc(r) dS
```

where `E_inc(r) = pol * E0 * exp(-i k . r)` and `S_m` is the union of the two support triangles of basis `m`.

### Port and Delta-Gap
Port excitations use edge-localized voltage injection:

```
v_m = (V / l_m) * delta_{m in port}
```

where `l_m` is the edge length. For `DeltaGapExcitation`, the gap length `g` is used instead:

```
v_m = (V / g) * delta_{m = gap_edge}
```

### Dipole and Loop Sources

The incident field from the dipole or loop is evaluated at quadrature points on each support triangle and integrated numerically via `dipole_incident_field(r, dipole)` (internal). The explicit field formulas (exp(+iwt) convention) are:

**Electric dipole** (moment `p`, position `r_0`):

```
E(r) = [exp(-ikR) / (4*pi*eps0*R)] * { k^2 (R_hat x p) x R_hat
        + (3 R_hat (R_hat . p) - p) * (1/R^2 - ik/R) }
```

where `R = |r - r_0|` and `R_hat = (r - r_0) / R`. This is the full near-field + far-field electric dipole radiation formula (Balanis, Ch. 4). It includes the 1/R^3 quasi-static term, the 1/R^2 induction term, and the 1/R radiation term.

**Magnetic dipole** (moment `m`, position `r_0`):

```
E(r) = (eta0 / 4*pi) * (k/R^2 + ik^2/R) * exp(-ikR) * (m x R_hat)
```

This follows from Balanis's small circular loop result (Ch. 5): `E_phi = +(eta*k^2*m*sin(theta))/(4*pi*r) * (i + 1/(kr)) * exp(-ikr)`, generalized to 3D vector form. It includes both the 1/R^2 induction term and the 1/R radiation (far-field) term. The quasi-static 1/R^3 term for the magnetic dipole is in H, not E.

**Loop excitation** maps to a magnetic dipole with moment `m = I * pi * a^2 * n_hat`, where `I` is the loop current, `a` is the loop radius, and `n_hat` is the loop normal.

**Far-field pattern coefficients** (used by `make_analytic_dipole_pattern_feed`):

- Electric: `F(r_hat) = k^2 / (4*pi*eps0) * (r_hat x (p x r_hat))`
- Magnetic: `F(r_hat) = i * eta0 * k^2 / (4*pi) * (m x r_hat)`

### Imported/Distributed source (`ImportedExcitation`)
The same quadrature integral is used, with the source function evaluated at each quadrature point. For `kind=:electric_field`, `E_inc = source(r)` directly. For `kind=:surface_current_density`, `E_inc = eta_equiv * source(r)`.

Implementation guards: the source function output is validated as a finite 3-component vector; tuple/array/SVector returns are accepted and converted to complex 3-vectors; malformed outputs throw explicit errors.

### Pattern Feed
Pattern feeds interpolate complex `(F_theta, F_phi)` data on the spherical grid, reconstruct `E_inc` at quadrature points using the phase-center geometry, and assemble via the standard MoM RHS integral.

---

## Examples

### Basic Plane Wave
```julia
k = 2pi / 0.1   # lambda = 0.1m
k_vec = Vec3(0.0, 0.0, -k)    # propagating in -z
E0 = 1.0
pol = Vec3(1.0, 0.0, 0.0)     # x-polarized

# Legacy method
v_old = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol)

# Unified method (recommended)
pw = make_plane_wave(k_vec, E0, pol)
v_new = assemble_excitation(mesh, rwg, pw)
```

### Delta-Gap Antenna Feed
```julia
# 1V source across edge 10 with 1mm physical gap
gap = make_delta_gap(10, 1.0, 0.001)
v_gap = assemble_excitation(mesh, rwg, gap)
```

### Multi-Port Excitation
```julia
# Two ports with different voltages
port1 = PortExcitation([1, 2, 3], 1.0 + 0.0im, 50.0)
port2 = PortExcitation([4, 5, 6], 0.5 + 0.0im, 50.0)
V = assemble_multiple_excitations(mesh, rwg, [port1, port2])
# V[:,1] is RHS from port1, V[:,2] from port2
```

### Near-Field Dipole Source
```julia
# Electric dipole at z=0.1m, oriented in x-direction
dipole = make_dipole(Vec3(0, 0, 0.1),
                     CVec3(1e-9, 0, 0),
                     Vec3(1, 0, 0), :electric, 1e9)
v_dip = assemble_excitation(mesh, rwg, dipole)
```

### Imported Electric Field
```julia
# Custom incident field: x-polarized wave propagating in z
Efun(r) = CVec3(exp(-1im * k * r[3]), 0.0 + 0im, 0.0 + 0im)
exc = make_imported_excitation(Efun; kind=:electric_field, min_quad_order=3)
v = assemble_excitation(mesh, rwg, exc)
```

### Imported Surface Current Density
```julia
# Impressed surface current with cosine distribution
Lx = 0.1
Jsfun(r) = CVec3(cos(2pi * r[1] / Lx) + 0im, 0.0 + 0im, 0.0 + 0im)
exc = make_imported_excitation(Jsfun;
    kind=:surface_current_density, eta_equiv=120 + 30im, min_quad_order=4)
v = assemble_excitation(mesh, rwg, exc)
```

### Pattern Feed from Analytical Dipole
```julia
freq = 1.0e9
dip = make_dipole(Vec3(0,0,0), CVec3(0, 0, 1e-12+0im),
                  Vec3(0,0,1), :electric, freq)
theta_deg = collect(0.0:2.0:180.0)
phi_deg = collect(0.0:5.0:355.0)
pat = make_analytic_dipole_pattern_feed(dip, theta_deg, phi_deg;
                                        angles_in_degrees=true)
v_pat = assemble_excitation(mesh, rwg, pat; quad_order=3)
```

### Pattern Feed from External CSV
```julia
# See example scripts for full workflow:
#   julia --project=. examples/07_pattern_feed.jl
#   julia --project=. examples/07_pattern_feed.jl
```

### Combined Excitation
```julia
pw = make_plane_wave(k_vec, E0, pol)
gap = make_delta_gap(10, 1.0, 0.001)
multi = make_multi_excitation([pw, gap], [0.7, 0.3])
v_multi = assemble_excitation(mesh, rwg, multi)
```

---

## Integration with Optimization

All excitations are compatible with the adjoint gradient computation. The excitation vector `v` enters the adjoint equation as a fixed right-hand side; the gradient depends on `Z` and `Q`, not on `v`. This means you can optimize impedance parameters with any excitation type without modifying the adjoint code.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/assembly/Excitation.jl` | All excitation types and assembly methods |
| `src/basis/Quadrature.jl` | Quadrature rules used for integration |
| `examples/07_pattern_feed.jl` | Pattern import from CSV and end-to-end horn pattern feed demo |

---

## Notes

- For plane waves, `quad_order=3` is sufficient for typical accuracy.
- Port and delta-gap excitations are local (affect only the specified edges, no quadrature needed).
- Near-field sources (dipole, loop) must be placed away from the mesh surface to avoid singular fields.
- For imported fields, prefer `make_imported_excitation(...)` so `kind` and scaling are explicit.
- If you need a rigorously radiating impressed-current model, use `kind=:electric_field` with externally computed `E_inc`, rather than assuming `E = eta * J` outside sheet-like cases.

---

## Exercises

### Basic
1. Assemble plane-wave excitation with both `assemble_v_plane_wave(...)` and `assemble_excitation(..., make_plane_wave(...))`. Verify they match element-by-element.
2. Create a delta-gap excitation on edge 5 and inspect which entries of `v` are non-zero.

### Practical
1. Set up a two-port excitation and solve for currents. Compute the input impedance at each port.
2. Combine plane wave and dipole excitations with weights 0.6 and 0.4 using `MultiExcitation`.

### `compute_total_field` compatibility
- Supported: `PlaneWaveExcitation`, `DipoleExcitation`, `LoopExcitation`, `PatternFeedExcitation`, `ImportedExcitation(kind=:electric_field)`, and `MultiExcitation` composed only of those supported source types.
- Not supported: `PortExcitation`, `DeltaGapExcitation`, and `ImportedExcitation(kind=:surface_current_density)`.

### Advanced
1. Write a function that imports field data from a CSV file and creates an `ImportedExcitation`. Compare the RHS against a plane-wave reference.
2. Use `make_analytic_dipole_pattern_feed` to create a pattern feed, then compare its RHS against the direct `DipoleExcitation` RHS for the same dipole. Quantify the agreement.
