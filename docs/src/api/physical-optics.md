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

### `solve_po(mesh, freq_hz, excitation; grid, c0=299792458.0, eta0=376.730313668, max_work_bytes=536_870_912)`

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
| `max_work_bytes` | `Integer` | `536_870_912` | Maximum raw payload of solver-owned outputs and construction workspaces, checked before allocation. |

**Returns:** `POResult`.

The frequency, `c0`, and `eta0` must be finite and positive. The excitation
must have a finite amplitude, a finite nonzero wavevector whose norm matches
`2π*freq_hz/c0`, and a finite nonzero polarization transverse to that
wavevector. The spherical-grid arrays must have consistent nonempty shapes.

**Physics:**

For a plane wave `E_inc = E0 * pol * exp(-jk . r)`:
1. **Illumination test:** Triangle `t` is illuminated if `k_hat . n_hat <= 0` (wave impinges on the outward-normal side).
2. **PO currents:** On illuminated faces, `J_s = 2(n_hat x H_inc)` where `H_inc = (k_hat x E_inc) / eta0`.
3. **Far-field integral:** `E_scat(r_hat) = (+jk E0 / 2pi) * sum_t [r_hat x (r_hat x V_t)] * I_t`, where `I_t` is the analytical phase integral over triangle `t` for the phase `exp(jk (r_hat - k_hat) . r')`. The positive prefactor follows the package-wide `exp(+j omega t)` convention.

The analytical phase integral handles all special cases (small phase differences, co-linear configurations) using Taylor-series expansions, avoiding numerical singularities.

**Example:**

```julia
mesh = read_obj_mesh("sphere.obj")
freq_hz = 3e9
k = 2π * freq_hz / 299792458.0
k_vec = Vec3(0.0, 0.0, -k)                       # wave propagating toward -z
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
| Model | Full-wave PEC surface-current integral equation | High-frequency illuminated-surface approximation |
| Error controls | Mesh, quadrature, operator, and solve convergence | Mesh, observation sampling, and high-frequency model validity |
| Cost model | Dense or accelerated operator, depending on `method` | O(Nt * N_omega) for Nt triangles and N_omega directions |
| Requires RWG | Yes | No |
| Diffraction | Emerges from the solved current when the discretization resolves it | Not included in the base PO term; `solve_ptd` adds supported edge corrections |

Choose between them from the required observable accuracy, electrical size,
available memory, and a convergence or reference comparison for the target
geometry.

---

## Physical Theory of Diffraction (PTD)

The PTD module adds Ufimtsev edge-diffraction (fringe) corrections on top of the
PO solution. The fringe current `= exact_edge - PO_edge` recovers the
diffracted field that PO misses at shadow/reflection boundaries, improving
side-lobe and wide-angle RCS prediction. PTD calls `solve_po` internally for the
PO contribution, extracts diffraction edges from the mesh, and adds the edge
fringe far-field using the Sáez de Adana et al. formulation.

!!! warning "Validity"
    `solve_ptd` accepts **boundary half-plane edges only** (`n = 2`, `α =
    2π`). It rejects an interior wedge that passes `min_dihedral_deg` because
    the coefficient branch has no mesh-label-independent illuminated-side
    convention. Use `solve_po` for that mesh, or use
    `extract_diffraction_edges` to inspect the wedge geometry.

    When mixed-scale inputs require 8704-bit direction accumulation,
    `solve_ptd` retains at most 12,288 high-precision field values (4,096
    observation directions) and then fails with `ArgumentError`. This bounds
    exceptional memory independently of the ordinary `ComplexF64` output.

### `DiffractionEdge`

A diffraction edge extracted from a triangle mesh, storing the local wedge
geometry needed for PTD computations. For interior edges, `face_o` and `face_n`
are the two adjacent faces and `alpha` is the exterior wedge angle. For boundary
edges, `face_n == 0` and `alpha == 2π` (half-plane).

```julia
struct DiffractionEdge
    v1::Int          # vertex index 1
    v2::Int          # vertex index 2
    p1::Vec3         # vertex 1 position
    p2::Vec3         # vertex 2 position
    center::Vec3     # edge midpoint
    tangent::Vec3    # unit tangent (p2-p1)/|p2-p1|
    length::Float64  # edge length
    face_o::Int      # canonical first wedge-face index
    face_n::Int      # "inner" face index (0 for boundary)
    normal_o::Vec3   # unit normal of face_o
    normal_n::Vec3   # unit normal of face_n (zero for boundary)
    alpha::Float64   # exterior wedge angle (radians), in (0, 2π]
    uo::Vec3         # outward unit vector in o-face plane, perp to tangent
end
```

### `PTDResult`

Result from the PTD solver: combined PO+PTD far-field, individual components for
diagnostics, and the diffraction edge data.

```julia
struct PTDResult
    E_ff::Matrix{ComplexF64}       # (3, NΩ) combined PO+PTD far-field
    E_ff_po::Matrix{ComplexF64}    # (3, NΩ) PO-only far-field
    E_ff_ptd::Matrix{ComplexF64}   # (3, NΩ) PTD edge correction only
    J_s::Vector{CVec3}             # PO surface currents
    illuminated::BitVector         # PO illumination mask
    edges::Vector{DiffractionEdge} # diffraction edges found
    grid::SphGrid
    freq_hz::Float64
    k::Float64
end
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `E_ff` | `Matrix{ComplexF64}` | `(3, NΩ)` combined PO+PTD scattered far-field. |
| `E_ff_po` | `Matrix{ComplexF64}` | `(3, NΩ)` PO-only far-field (same as `solve_po`). |
| `E_ff_ptd` | `Matrix{ComplexF64}` | `(3, NΩ)` PTD edge fringe correction only. |
| `J_s` | `Vector{CVec3}` | PO surface current per triangle centroid. |
| `illuminated` | `BitVector` | PO illumination mask. |
| `edges` | `Vector{DiffractionEdge}` | Diffraction edges used for the correction. |
| `grid` | `SphGrid` | Spherical observation grid. |
| `freq_hz` | `Float64` | Frequency in Hz. |
| `k` | `Float64` | Wavenumber (rad/m). |

### `extract_diffraction_edges(mesh; min_dihedral_deg=5.0, include_boundary=true)`

Extract diffraction-feature edges from a triangle mesh. Interior edges whose
dihedral angle exceeds `min_dihedral_deg` are kept; edges with a single adjacent
face (boundary edges) are treated as half-planes (`α = 2π`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh of the scatterer. |
| `min_dihedral_deg` | `Float64` | `5.0` | Minimum dihedral angle (degrees) for an interior edge to be kept. |
| `include_boundary` | `Bool` | `true` | If `true`, keep boundary (open) edges as half-planes. |

**Returns:** `Vector{DiffractionEdge}`.

### `solve_ptd(mesh, freq_hz, excitation; grid, c0=299792458.0, eta0=376.730313668, min_dihedral_deg=5.0, include_boundary=true, max_work_bytes=536_870_912)`

Compute the PO+PTD scattered far-field for a PEC body.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `TriMesh` | -- | Triangle mesh of the scatterer. |
| `freq_hz` | `Real` | -- | Frequency in Hz. |
| `excitation` | `PlaneWaveExcitation` | -- | Incident plane wave. |
| `grid` | `SphGrid` | `make_sph_grid(36, 72)` | Spherical observation grid. |
| `c0` | `Float64` | `299792458.0` | Speed of light (m/s). |
| `eta0` | `Float64` | `376.730313668` | Free-space impedance (ohms). |
| `min_dihedral_deg` | `Float64` | `5.0` | Passed to `extract_diffraction_edges`. |
| `include_boundary` | `Bool` | `true` | Passed to `extract_diffraction_edges`. |
| `max_work_bytes` | `Integer` | `536_870_912` | Maximum combined raw payload of the PO and PTD outputs and construction workspaces, checked before field allocation. |

**Returns:** `PTDResult`.

**Example:**

```julia
L = 1.0    # plate side length (m)
Ns = 20    # cells per side
mesh = make_rect_plate(L, L, Ns, Ns)
freq_hz = 3e9
k = 2π * freq_hz / 299792458.0
pw = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))  # +z, x-pol

grid = make_sph_grid(90, 180)   # (Ntheta, Nphi) observation grid
ptd = solve_ptd(mesh, freq_hz, pw; grid=grid)

println("Diffraction edges: ", length(ptd.edges))

# Bistatic RCS from the combined PO+PTD field, and the edge-only contribution
rcs_total = bistatic_rcs(ptd.E_ff; E0=1.0)
rcs_edge  = bistatic_rcs(ptd.E_ff_ptd; E0=1.0)
```

See `examples/22_po_ptd_comparison.jl` (flat plates) and
`examples/23_circular_plate_ptd.jl` (circular plate) for full MoM vs PO vs
PO+PTD comparisons.

---

## Code Mapping

| File | Contents |
|------|----------|
| `src/postprocessing/PhysicalOptics.jl` | `POResult`, `solve_po`, analytical phase integrals |
| `src/postprocessing/PTD.jl` | `DiffractionEdge`, `PTDResult`, `extract_diffraction_edges`, `solve_ptd` |
