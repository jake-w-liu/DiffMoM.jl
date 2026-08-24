# Excitation API

Excitations produce the right-hand side in the MoM system

\[
\mathbf Z\mathbf I=\mathbf v,\qquad
v_m=-\int_{S_m}\mathbf f_m(\mathbf r)\mathbin{\cdot}
\mathbf E^{\mathrm{inc}}(\mathbf r)\,\mathrm dS.
\]

Use a source constructor, then pass the source to `assemble_excitation`:

```julia
using DiffMoM

freq = 1.0e9
k = 2pi * freq / 299792458.0
source = make_plane_wave(
    Vec3(0.0, 0.0, -k),
    1.0,
    Vec3(1.0, 0.0, 0.0),
)
v = assemble_excitation(mesh, rwg, source; quad_order=3)
```

The public concrete source types are exported. Their common abstract base is an
implementation detail, so application code should use the constructors and
`assemble_excitation` instead of extending that base.

## Source models

| Model | Construction | RHS model |
|---|---|---|
| Plane wave | `make_plane_wave(k_vec, E0, pol)` | Quadrature of `E0 * pol * exp(-im * dot(k_vec, r))` |
| Delta gap | `make_delta_gap(edge, voltage, gap_length)` | One entry, `voltage / gap_length` |
| Port | `PortExcitation(edges, voltage, impedance)` | `voltage / rwg.len[e]` on each listed edge |
| Electric or magnetic dipole | `make_dipole(position, moment, orientation, type, frequency)` | Dipole electric field evaluated at quadrature points |
| Current loop | `make_loop(center, normal, radius, current, frequency)` | Magnetic-dipole field with moment `current * pi * radius^2 * normal` |
| Monopole | `make_monopole(position, axis, height, amplitude, frequency; include_image=true)` | Simpson-integrated wire field |
| Imported field | `make_imported_excitation(source_func; ...)` | User function evaluated at quadrature points |
| Pattern feed | `make_pattern_feed(...)` | Interpolated complex spherical far-field coefficients |
| Weighted sum | `make_multi_excitation(sources, weights)` | Weighted sum of child right-hand sides |

Constructors validate finite values, dimensions, supported symbols, and
source-specific invariants before assembly. The
[assembly docstrings](exported-assembly-1.md) own their exact signatures and
defaults.

### Plane waves

`k_vec` points in the propagation direction and has magnitude in rad/m. `pol`
must be transverse to `k_vec`. The package uses the `exp(+iwt)` convention, so
the spatial phase is `exp(-im * dot(k_vec, r))`.

Evaluate the same field away from assembly points with:

```julia
Einc = plane_wave_field(r, source.k_vec, source.E0, source.pol)
```

`assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)` remains a
supported convenience wrapper. Constructing a source object is preferable when
the same incident field will later be passed to `compute_total_field`.

### Delta gaps and ports

A delta gap addresses one RWG edge. A port addresses several edges and stores
an impedance value, but current RHS assembly uses only its voltage and edge
indices. These are localized voltage models; they do not represent an incident
electric field away from the mesh.

```julia
gap = make_delta_gap(10, 1.0 + 0im, 1.0e-3)
v_gap = assemble_excitation(mesh, rwg, gap)

port = PortExcitation([10, 11], 1.0 + 0im, 50.0 + 0im)
v_port = assemble_excitation(mesh, rwg, port)
```

### Dipoles and loops

`make_dipole` accepts `type=:electric` or `type=:magnetic`. The moment units are
C m for an electric dipole and A m^2 for a magnetic dipole. `orientation` is
validated and stored as source metadata; the complex moment determines the
field.

```julia
dipole = make_dipole(
    Vec3(0.0, 0.0, 0.1),
    CVec3(1.0e-9 + 0im, 0im, 0im),
    Vec3(1.0, 0.0, 0.0),
    :electric,
    1.0e9,
)
v_dipole = assemble_excitation(mesh, rwg, dipole)
```

Dipole and loop fields are singular at their source location. Assembly rejects
non-finite results, but the caller must choose source positions that do not
coincide with mesh quadrature points.

### Monopoles

With `include_image=true`, the monopole uses a wire-plus-image equivalent for an
infinite PEC ground and returns zero below the source ground plane. With
`include_image=false`, it uses the physical half-wire and radiates into free
space; use that mode when a finite ground plane is explicitly meshed.

```julia
monopole = make_monopole(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.075,
    1.0 + 0im,
    1.0e9;
    include_image=false,
)
Einc = monopole_incident_field(observation_point, monopole)
```

`monopole_incident_field` accepts `max_exact_work` to bound exceptional
high-precision phase evaluation. Its exact default is rendered from the source
in the [assembly docstrings](exported-assembly-1.md).

### Imported fields

The source function must return a finite three-component vector.

```julia
field(r) = CVec3(exp(-1im * k * r[3]), 0im, 0im)
imported = make_imported_excitation(
    field;
    kind=:electric_field,
    min_quad_order=3,
)
v_imported = assemble_excitation(mesh, rwg, imported)
```

Two interpretations are available:

- `kind=:electric_field` uses `source_func(r)` as the incident electric field.
- `kind=:surface_current_density` applies the local map
  `E_inc(r) = eta_equiv * source_func(r)`.

The surface-current mode is an equivalent-sheet RHS approximation, not a
Green-function radiation solve. If the incident field is available from an
external solver, import it as `kind=:electric_field`.

`min_quad_order` must be from 1 through 7. Assembly selects the smallest
supported rule in `(1, 3, 4, 7)` that is no lower than both the caller's
`quad_order` and `min_quad_order`.

### Pattern feeds

Pattern feeds store complex `Ftheta` and `Fphi` coefficients on strictly
increasing spherical grids. For the package convention,

\[
\mathbf E(\mathbf r)=\frac{e^{-ikR}}{R}
\left(F_\theta\hat{\boldsymbol\theta}+
F_\phi\hat{\boldsymbol\phi}\right).
\]

The theta grid must lie in `[0, pi]`. The phi grid must span less than one full
period and must not repeat the endpoint. Both grids need at least two points.

```julia
theta_deg = collect(0.0:2.0:180.0)
phi_deg = collect(0.0:5.0:355.0)
Ftheta = zeros(ComplexF64, length(theta_deg), length(phi_deg))
Fphi = similar(Ftheta)

pattern = make_pattern_feed(
    theta_deg,
    phi_deg,
    Ftheta,
    Fphi,
    1.0e9;
    angles_in_degrees=true,
    convention=:exp_plus_iwt,
)
Einc = pattern_feed_field(observation_point, pattern)
```

Use `convention=:exp_minus_iwt` when imported coefficients use the opposite
time convention. The constructor converts angle arrays and coefficient matrices
to owned `Float64` and `ComplexF64` storage. `max_storage_bytes` bounds their raw
payload before allocation.

The pattern-object overload accepts two objects with matching `.x`, `.y`, and
`.U` fields. `make_analytic_dipole_pattern_feed` samples a `DipoleExcitation`
onto a pattern grid. In either case, retain both complex polarizations; a power
pattern does not contain enough phase or polarization information.

The field is undefined at the phase center. `pattern_feed_field` returns the
zero vector there, matching the behavior used during assembly.

### Weighted sources

`MultiExcitation` assembles a complex weighted sum. If `weights` is omitted,
each child has weight `1 + 0im`.

```julia
combined = make_multi_excitation(
    [source, gap],
    ComplexF64[0.7, 0.3],
)
v_combined = assemble_excitation(mesh, rwg, combined)
```

Nested graphs must be acyclic and no deeper than 64 levels. Shared children in
separate branches are allowed. `max_exact_bytes` on `assemble_excitation`
bounds the high-precision accumulator used only for exceptional range or
cancellation recovery.

## Assembly functions

### `assemble_excitation`

```julia
v = assemble_excitation(
    mesh,
    rwg,
    source;
    quad_order=3,
    max_exact_bytes=536_870_912,
)
```

Supported quadrature orders are 1, 3, 4, and 7. Port and delta-gap models do not
use surface quadrature. The function validates that `mesh` and `rwg` match and
that the returned vector is finite.

### `assemble_multiple_excitations`

```julia
V = assemble_multiple_excitations(
    mesh,
    rwg,
    [source_a, source_b];
    quad_order=3,
    max_output_bytes=2_000_000_000,
    max_work_bytes=536_870_912,
    max_terms=200_000_000,
)
```

The result has size `rwg.nedges x length(sources)`, with one RHS per column.
The batch path shares cached mesh quadrature between compatible sources.
Resource limits are checked before the dense output and retained quadrature
caches are allocated.

## Total-field compatibility

`compute_total_field` can add incident fields from plane waves, dipoles, loops,
monopoles, pattern feeds, electric-field imports, and weighted combinations of
those models. It rejects ports, delta gaps, and surface-current-density imports
because those models do not define an incident electric field at arbitrary
observation points.

## Source map

| API | Source |
|---|---|
| Source types, constructors, field evaluation, and RHS assembly | `src/assembly/Excitation.jl` |
| Triangle quadrature rules | `src/basis/Quadrature.jl` |
| Total-field support | `src/postprocessing/NearField.jl` |
| Pattern-feed example | `examples/07_pattern_feed.jl` |
