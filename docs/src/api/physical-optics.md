# API: Physical Optics (PO)

## Purpose

The Physical Optics module provides a high-frequency approximate solver for PEC scattering. It computes surface currents and far-field scattering using the tangential magnetic field approximation (`J_s = 2(n-hat x H_inc)` on illuminated faces, zero on shadow faces). PO works directly on triangle meshes without RWG basis functions and uses analytical phase integration over each triangle, matching the POFacets 4.5 algorithm.

PO is useful for electrically large problems where full MoM is too expensive, and as a fast reference for validating MoM results at high frequencies.

---

## Types

### `POResult`

Result container for the PO solver.

```julia
struct POResult
    E_ff::Matrix{ComplexF64}     # (3, N_omega) scattered far-field
    J_s::Vector{CVec3}           # (Nt,) PO surface current per triangle centroid
    illuminated::BitVector       # (Nt,) which triangles are illuminated
    grid::SphGrid
    freq_hz::Float64
    k::Float64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `E_ff` | `Matrix{ComplexF64}` | `(3, N_omega)` scattered electric far-field at each observation direction. |
| `J_s` | `Vector{CVec3}` | `(Nt,)` PO surface current density at each triangle centroid. Zero on shadow faces. |
| `illuminated` | `BitVector` | `(Nt,)` mask: `true` for illuminated triangles (`k_hat . n_hat <= 0`). |
| `grid` | `SphGrid` | Spherical observation grid used for far-field computation. |
| `freq_hz` | `Float64` | Frequency in Hz. |
| `k` | `Float64` | Wavenumber (rad/m). |

**Computing RCS from POResult:**

```julia
result = solve_po(mesh, freq_hz, excitation)
# Bistatic RCS at each observation angle
rcs_vals = bistatic_rcs(result.E_ff)
```

---

## Functions

### `solve_po(mesh, freq_hz, excitation; grid, c0=299792458.0, eta0=376.730313668)`

Compute the Physical Optics scattered far-field for a PEC body.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh of the scatterer. |
| `freq_hz` | `Real` | -- | Frequency in Hz (must be > 0). |
| `excitation` | `PlaneWaveExcitation` | -- | Incident plane wave. |
| `grid` | `SphGrid` | `make_sph_grid(36, 72)` | Spherical observation grid. |
| `c0` | `Float64` | `299792458.0` | Speed of light (m/s). |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance (ohms). |

**Returns:** `POResult`.

**Physics:**

For a plane wave `E_inc = E0 * pol * exp(-jk . r)`:
1. **Illumination test:** Triangle `t` is illuminated if `k_hat . n_hat <= 0` (wave impinges on the outward-normal side).
2. **PO currents:** On illuminated faces, `J_s = 2(n_hat x H_inc)` where `H_inc = (k_hat x E_inc) / eta0`.
3. **Far-field integral:** `E_scat(r_hat) = (-jk E0 / 2pi) * sum_t [r_hat x (r_hat x V_t)] * I_t`, where `I_t` is the analytical phase integral over triangle `t` for the phase `exp(jk (r_hat - k_hat) . r')`.

The analytical phase integral handles all special cases (small phase differences, co-linear configurations) using Taylor-series expansions, avoiding numerical singularities.

**Example:**

```julia
mesh = read_obj_mesh("sphere.obj")
freq_hz = 3e9
k = 2π * freq_hz / 299792458.0
k_vec = Vec3(0.0, 0.0, -k)                       # +z-propagating wave
excitation = make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0))  # x-polarized
grid = make_sph_grid(90, 36)

result = solve_po(mesh, freq_hz, excitation; grid=grid)

println("Illuminated triangles: ", count(result.illuminated), " / ", ntriangles(mesh))
println("Far-field shape: ", size(result.E_ff))
```

---

## Comparison with MoM

| Aspect | MoM (`solve_scattering`) | PO (`solve_po`) |
|--------|-------------------------|-----------------|
| Accuracy | Exact (within discretization) | High-frequency approximation |
| Complexity | O(N^2) to O(N log N) | O(Nt * N_omega) |
| Requires RWG | Yes | No |
| Handles diffraction | Yes | No (shadow boundary artifacts) |
| Handles creeping waves | Yes | No |
| Best for | lambda-scale to moderate objects | Electrically large objects (D >> lambda) |

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/postprocessing/PhysicalOptics.jl` | `POResult`, `solve_po`, analytical phase integrals |
