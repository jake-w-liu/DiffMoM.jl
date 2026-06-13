# API: Grounded (Half-Space) EFIE

## Purpose

Reference for the periodic EFIE subsystem that models a coplanar metasurface a
distance `h` above an infinite PEC ground plane via image theory. For horizontal
(coplanar) electric currents above a PEC ground, the image of a current `J` at
height `h` is `-J` at depth `-h`, with charges imaged by the same `-1` factor.
Both the vector- and scalar-potential kernels therefore acquire the image with a
single `-1`, so the grounded EFIE is obtained from the free-standing periodic
EFIE by replacing the scalar Green's function `G_per(d_rho, 0)` with
`G_grounded = G_per(d_rho, 0) - G_per(d_rho, 2h)` in both the `f.f` and
`(div f)(div f)` integrals:

```
Z_grounded = Z_direct - Z_image
```

where `Z_direct` is the free-standing coplanar periodic EFIE and `Z_image` is the
interaction with the mirror currents at depth `2h` (full periodic Green's
function, no singularity).

These functions build on the free-standing periodic EFIE and Floquet
post-processing. See [periodic-methods.md](periodic-methods.md) for
`PeriodicLattice`, `assemble_Z_efie_periodic`, `reflection_coefficients`, and
`reflection_coefficient_vectors`, and [excitation.md](excitation.md) for the
plane-wave excitation model. For a complete end-to-end optimization workflow, see
`examples/21_grounded_rcs_demo.jl` and the `examples/grounded_rcs/` directory.

**See also:** for the theory and a worked walkthrough, see
[Grounded (Half-Space) EFIE via Image Theory](../formulations/05-grounded-efie.md).

---

### `assemble_Z_efie_grounded(mesh, rwg, k, lattice; height, quad_order=3, eta0=376.730313668)`

Assemble the periodic EFIE impedance matrix for a coplanar metasurface a distance
`height` (h) above an infinite PEC ground plane, via image theory. Internally it
assembles the free-standing periodic EFIE `Z_direct` with
`assemble_Z_efie_periodic` and subtracts the image-current block `Z_image`
evaluated with the full periodic Green's function at vertical separation `2*height`
(smooth, no singularity extraction).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Geometry mesh of the unit cell (coplanar, `z = const`). |
| `rwg` | `RWGData` | -- | Bloch-paired RWG basis data from `build_rwg_periodic`. |
| `k` | Real | -- | Free-space wavenumber (rad/m). |
| `lattice` | `PeriodicLattice` | -- | Unit-cell periodic setup. |
| `height` | `Real` | -- | Distance `h` of the metasurface above the PEC ground plane (meters). Must be positive. |
| `quad_order` | `Int` | `3` | Triangle quadrature order. |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance (Ohm). |

**Returns:** Dense `Matrix{ComplexF64}` `Z_grounded = Z_direct - Z_image` of size `N x N`.

A non-positive `height` raises `ArgumentError`.

```julia
lam = 2.99792458e8 / 10e9
k = 2pi / lam
dxc = 1.2 * lam; h = lam / 4
mesh = make_rect_plate(dxc, dxc, 14, 14)
lat = PeriodicLattice(dxc, dxc, 0.0, 0.0, k)
rwg = build_rwg_periodic(mesh, lat; precheck=true, allow_boundary=true, require_closed=false)
Zg = assemble_Z_efie_grounded(mesh, rwg, k, lat; height=h)
```

---

### `assemble_excitation_grounded(mesh, rwg, pw, k, lattice; height, quad_order=3)`

Assemble the excitation vector for the grounded problem. The metasurface is
illuminated by the incident plane wave plus its bare-ground reflection. The
free-standing excitation vector (from `assemble_excitation`) is scaled by the
factor `1 - exp(-2i kz_inc h)`, where `kz_inc` is the incident vertical wavenumber
of the specular order (`kz_inc = k cos(theta_inc)`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Geometry mesh of the unit cell. |
| `rwg` | `RWGData` | -- | Bloch-paired RWG basis data from `build_rwg_periodic`. |
| `k` | Real | -- | Free-space wavenumber (rad/m). |
| `lattice` | `PeriodicLattice` | -- | Unit-cell periodic setup (carries the Bloch wavenumbers). |
| `pw` | -- | -- | Plane-wave excitation model (e.g. from `make_plane_wave`), passed through to `assemble_excitation`. |
| `height` | `Real` | -- | Distance `h` of the metasurface above the PEC ground plane (meters). |
| `quad_order` | `Int` | `3` | Triangle quadrature order. |

**Returns:** `Vector{ComplexF64}` of length `N` (the free-standing excitation scaled by `1 - exp(-2i kz_inc h)`).

```julia
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
v = Vector{ComplexF64}(assemble_excitation_grounded(mesh, rwg, pw, k, lat; height=h))
I = Zg \ v
```

---

### `reflection_coefficients_grounded(mesh, rwg, I, k, lattice; height, kwargs...)`

Compute the scalar (co-polar) Floquet reflection coefficients for a metasurface a
height `h` above a PEC ground. It calls `reflection_coefficients` for the
free-standing per-mode coefficients `R_cur`, then adds the image-current
contribution and the bare-ground specular background:

```
R_mn^grounded = R_mn^cur * (1 - exp(-2i kz_mn h)) - delta_{mn,(0,0)} * exp(-2i kz_inc h)
```

The image phase delay uses `real(m.kz)`: evanescent orders store `kz = i*beta`, for
which `exp(-2i kz h)` would overflow, so only the real vertical wavenumber drives
the image phase. Limiting behavior: an empty cell gives the bare-ground
`R_00 = -exp(-2i kz_inc h)` (`|R| = 1`); a full PEC sheet at `z = 0` gives
`R_00 = -1` for any `h`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Geometry mesh of the unit cell. |
| `rwg` | `RWGData` | -- | Bloch-paired RWG basis data from `build_rwg_periodic`. |
| `I` | `Vector{<:Number}` | -- | Solved RWG current coefficients (length `N`). |
| `k` | Real | -- | Free-space wavenumber (rad/m). |
| `lattice` | `PeriodicLattice` | -- | Unit-cell periodic setup. |
| `height` | `Real` | -- | Distance `h` of the metasurface above the PEC ground plane (meters). |
| `kwargs...` | -- | -- | Forwarded to `reflection_coefficients` (e.g. `N_orders`, `E0`, `pol`, `quad_order`, `eta0`). |

**Returns:** Tuple `(modes, R_g)`, where `modes` is the `Vector{FloquetMode}` of
diffraction orders and `R_g` is the `Vector{ComplexF64}` of grounded co-polar
reflection coefficients (one per mode, same length as `modes`).

```julia
modes, R_g = reflection_coefficients_grounded(mesh, rwg, I, k, lat;
                                              height=h, N_orders=3, E0=1.0,
                                              pol=SVector(1.0, 0.0, 0.0))
i00 = findfirst(m -> m.m == 0 && m.n == 0, modes)
specular_mag = abs(R_g[i00])
```

---

### `reflection_coefficient_vectors_grounded(mesh, rwg, I, k, lattice; height, pol=SVector(1.0, 0.0, 0.0), kwargs...)`

Compute the full vector Floquet reflection amplitudes for a grounded metasurface.
This is the energy-budget counterpart to `reflection_coefficients_grounded`: it
calls `reflection_coefficient_vectors` to retain both transverse polarizations in
every propagating order, then applies the same image-current phase factor and
bare-ground background. The `(0,0)` background is subtracted along the
mode-transverse projection of `pol` (skipped when that projection is undefined).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Geometry mesh of the unit cell. |
| `rwg` | `RWGData` | -- | Bloch-paired RWG basis data from `build_rwg_periodic`. |
| `I` | `Vector{<:Number}` | -- | Solved RWG current coefficients (length `N`). |
| `k` | Real | -- | Free-space wavenumber (rad/m). |
| `lattice` | `PeriodicLattice` | -- | Unit-cell periodic setup. |
| `height` | `Real` | -- | Distance `h` of the metasurface above the PEC ground plane (meters). |
| `pol` | `SVector{3,Float64}` | `SVector(1.0, 0.0, 0.0)` | Incident polarization used to project the bare-ground `(0,0)` background onto the mode-transverse plane. |
| `kwargs...` | -- | -- | Forwarded to `reflection_coefficient_vectors` (e.g. `N_orders`, `E0`, `quad_order`, `eta0`). |

**Returns:** Tuple `(modes, R_g)`, where `modes` is the `Vector{FloquetMode}` of
diffraction orders and `R_g` is the `Vector{SVector{3,ComplexF64}}` of grounded
vector reflection amplitudes (one three-component vector per mode). Pass `R_g` to
`reflected_power_fractions` (see [periodic-methods.md](periodic-methods.md)) for a
per-order reflected-power budget.

```julia
modes, R_g = reflection_coefficient_vectors_grounded(mesh, rwg, I, k, lat;
                                                     height=h, N_orders=3, E0=1.0,
                                                     pol=SVector(1.0, 0.0, 0.0))
budget = sum(reflected_power_fractions(modes, R_g, k))   # lossless-check total
```

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/assembly/GroundedEFIE.jl` | `assemble_Z_efie_grounded`, `assemble_excitation_grounded`, `reflection_coefficients_grounded`, `reflection_coefficient_vectors_grounded` |
| `src/assembly/PeriodicEFIE.jl` | `assemble_Z_efie_periodic` (free-standing `Z_direct`) |
| `src/postprocessing/PeriodicMetrics.jl` | `reflection_coefficients`, `reflection_coefficient_vectors`, `reflected_power_fractions` |

---

## Exercises

- **Basic:** For an empty unit cell (zero current vector), confirm
  `reflection_coefficients_grounded` returns the bare-ground specular
  `R_00 = -exp(-2i kz_inc h)` with `|R_00| = 1`.
- **Practical:** Assemble `Zg` and `v`, solve `I = Zg \ v`, then compare the
  specular `|R_00|` from `reflection_coefficients_grounded` against the bare-ground
  level (1.0) to read off the reflection suppression in dB.
- **Challenge:** Use `reflection_coefficient_vectors_grounded` with
  `reflected_power_fractions` to verify that the total reflected power budget is
  close to 1 for a lossless (reactive) loaded metasurface.
