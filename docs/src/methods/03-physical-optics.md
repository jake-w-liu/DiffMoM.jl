# Physical Optics approximation

`solve_po` evaluates a local tangent-plane current on illuminated triangles and
integrates that current into the scattered far field. It does not construct RWG
basis functions or solve a coupled surface-current system.

Physical Optics (PO) is an approximation. Its error depends on electrical
size, geometry, observation direction, and the importance of diffraction,
creeping waves, cavities, and multiple interactions. Establish its useful
regime with a reference or convergence study for the target observable.

## Current model

For a plane wave under the package's $e^{+i\omega t}$ convention,

```math
\mathbf E^{\mathrm{inc}}(\mathbf r)
=E_0\mathbf p\,e^{-i\mathbf k\cdot\mathbf r},
\qquad
\mathbf H^{\mathrm{inc}}(\mathbf r)
=\frac{1}{\eta_0}\hat{\mathbf k}\times
\mathbf E^{\mathrm{inc}}(\mathbf r).
```

A triangle with unit normal $\hat{\mathbf n}_t$ is classified as illuminated
when

```math
\hat{\mathbf k}\cdot\hat{\mathbf n}_t\leq0.
```

The implemented current is

```math
\mathbf J_s(\mathbf r)=
\begin{cases}
2\hat{\mathbf n}_t\times\mathbf H^{\mathrm{inc}}(\mathbf r),
& \text{illuminated},\\
0,& \text{otherwise}.
\end{cases}
```

This classification depends on the supplied winding. Mesh-quality checks can
make neighboring windings consistent, but they do not infer the absolute
outward direction of each component.

## Far-field integral

Define the triangle-constant direction

```math
\mathbf V_t=
\hat{\mathbf n}_t\times(\hat{\mathbf k}\times\mathbf p).
```

For observation direction $\hat{\mathbf r}$, the implementation accumulates

```math
\mathbf E_\infty(\hat{\mathbf r})=
\frac{ikE_0}{2\pi}
\sum_{t\in\mathrm{illum}}
\left[\hat{\mathbf r}\times
      (\hat{\mathbf r}\times\mathbf V_t)\right]I_t,
```

where

```math
I_t=\int_{T_t}
\exp\!\left(ik(\hat{\mathbf r}-\hat{\mathbf k})
\mathbin{\cdot}\mathbf r'\right)\,\mathrm dS'.
```

`_phase_integral_analytical` evaluates the triangle phase integral with a
closed-form expression and small-phase series branches. It is an internal
helper. The positive $ik$ prefactor matches the far-field convention used by
`radiation_vectors`, so PO, MoM, and PTD amplitudes can be combined coherently.

The double cross product removes radial content. The implementation evaluates
it directly and normalizes supplied observation directions before use.

## `solve_po`

```julia
solve_po(
    mesh,
    freq_hz,
    excitation;
    grid=make_sph_grid(36, 72),
    c0=299792458.0,
    eta0=376.730313668,
    max_work_bytes=536_870_912,
)
```

Requirements:

- `mesh` must pass `assert_mesh_quality` with boundaries allowed;
- `freq_hz`, `c0`, and `eta0` must be finite and positive;
- `excitation` must be a valid `PlaneWaveExcitation`; and
- `norm(excitation.k_vec)` must match `2π * freq_hz / c0` within the
  implemented tolerance.

`max_work_bytes` bounds the raw payload of solver-owned outputs and
workspaces before they are allocated.

The result is:

```julia
struct POResult
    E_ff::Matrix{ComplexF64}  # size (3, number of directions)
    J_s::Vector{CVec3}        # centroid current for each triangle
    illuminated::BitVector
    grid::SphGrid
    freq_hz::Float64
    k::Float64
end
```

`J_s[t]` is the model current evaluated at triangle `t`'s centroid. The
far-field calculation integrates the spatial phase over the complete triangle;
it does not approximate the integral by that centroid value.

## Bounded example

```julia
using DiffMoM

frequency = 3.0e9
c0 = 299792458.0
wavelength = c0 / frequency
k = 2π / wavelength

mesh = make_rect_plate(wavelength, wavelength, 10, 10)
wave = make_plane_wave(
    Vec3(0.0, 0.0, -k),
    1.0,
    Vec3(1.0, 0.0, 0.0),
)
grid = make_sph_grid(36, 72)

po = solve_po(mesh, frequency, wave; grid=grid)
sigma_po = bistatic_rcs(po.E_ff; E0=wave.E0)

println((
    illuminated=count(po.illuminated),
    triangles=ntriangles(mesh),
    rcs_range=extrema(sigma_po),
))
```

## Compare PO with MoM

Use identical geometry, excitation, observation grid, constants, and RCS
normalization:

```julia
rwg = build_rwg(mesh)
Z = assemble_Z_efie(mesh, rwg, k)
v = assemble_excitation(mesh, rwg, wave)
current = solve_forward(Z, v; solver=:direct)

G = radiation_vectors(mesh, rwg, grid, k)
E_mom = compute_farfield(G, current, length(grid.w))
sigma_mom = bistatic_rcs(E_mom; E0=wave.E0)

sigma_floor = 1e-30
difference_db =
    10 .* log10.(max.(sigma_po, sigma_floor)) .-
    10 .* log10.(max.(sigma_mom, sigma_floor))
```

This comparison measures disagreement between two models; it does not make
either result a truth reference. Report the dB floor and inspect linear-scale
differences near nulls.

Before interpreting the discrepancy:

1. check mesh quality and absolute normal direction;
2. check electrical resolution of the shared mesh;
3. verify the MoM true residual;
4. refine the observation grid;
5. compare supported MoM triangle quadrature orders; and
6. compare with an analytical or external reference where available.

## Cost and limitations

The main PO loop visits each observation direction and each illuminated
triangle, so its work scales with their product. Output storage is linear in
the number of triangles and directions. Measure runtime and peak memory for the
actual mesh and grid.

The local current rule does not solve for:

- edge-diffracted current;
- creeping waves around smooth shadowed surfaces;
- repeated interactions between separated surface regions;
- cavity fields; or
- global resonant current distributions.

The package also provides `solve_ptd`, which adds its implemented edge
correction to a PO result. See the
[Physical Optics and PTD API](../api/physical-optics.md) for its geometry and
resource controls.

## Verification

Run the repository PO validator from the project root:

```bash
julia --project=. --startup-file=no validation/po/validate_po_vs_pofacets.jl
```

That script requires a user-supplied `examples/demo_aircraft.obj`; the file is
not included in the repository. Set `DMOM_AIRCRAFT_OBJ` to use another path.
Record the effective geometry path with any reported result.

The regression suite contains bounded PO and PTD checks:

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
```

## Code map

| Task | Source |
|:--|:--|
| PO result, illumination, phase integral, and far field | `src/postprocessing/PhysicalOptics.jl` |
| PTD edge extraction and correction | `src/postprocessing/PTD.jl` |
| Plane-wave construction | `src/assembly/Excitation.jl` |
| RCS conversion | `src/postprocessing/Diagnostics.jl` |
| PO/PTD API reference | `docs/src/api/physical-optics.md` |
