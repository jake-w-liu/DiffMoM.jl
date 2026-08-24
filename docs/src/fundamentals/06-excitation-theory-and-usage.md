# Excitation theory and usage

An excitation either defines an incident electric field that is projected onto
the RWG basis or writes a lumped right-hand side directly. Use
`assemble_excitation` as the common entry point.

## From incident field to right-hand side

For an incident electric field $\mathbf E^{\mathrm{inc}}$, the assembled entry is

```math
v_m=-\int_\Gamma
\mathbf f_m(\mathbf r)\mathbin{\cdot}
\mathbf E^{\mathrm{inc}}(\mathbf r)\,\mathrm dS.
```

The implementation evaluates this surface integral on the two triangles that
support each RWG basis function. Supported triangle quadrature orders are 1, 3,
4, and 7.

The package uses the $e^{+i\omega t}$ convention. Its outward plane-wave phase
is

```math
\mathbf E^{\mathrm{inc}}(\mathbf r)
=E_0\mathbf p\,e^{-i\mathbf k\cdot\mathbf r}.
```

Use metres, hertz, and the corresponding wavenumber in radians per metre in the
documented workflows. More generally, geometry and wavenumber must use
reciprocal units.

## Supported excitation models

| Model | Construction | RHS source | Pointwise incident field for `compute_total_field` |
|:--|:--|:--|:--|
| Plane wave | `make_plane_wave` | Surface projection | Yes |
| Electric or magnetic dipole | `make_dipole` | Surface projection | Yes |
| Small loop model | `make_loop` | Surface projection | Yes |
| Monopole model | `make_monopole` | Surface projection | Yes |
| Imported field | `make_imported_excitation` | Surface projection | Only for `kind=:electric_field` |
| Pattern feed | `make_pattern_feed` | Surface projection | Yes |
| Delta gap | `make_delta_gap` | Direct RHS entry | No |
| Port | `PortExcitation` | Direct RHS entries | No |
| Weighted combination | `make_multi_excitation` | Weighted child sum | Only when every child supports pointwise fields |

`AbstractExcitation` is an internal dispatch base, not the supported extension
interface. Use `ImportedExcitation` for a custom pointwise field.

## Plane wave

`make_plane_wave(k_vec, E0, pol)` validates that `k_vec` is finite and nonzero,
`pol` is a unit vector, and the two vectors are transverse.

```julia
using DiffMoM

frequency = 3.0e9
c0 = 299792458.0
k = 2π * frequency / c0

plane_wave = make_plane_wave(
    Vec3(0.0, 0.0, -k),
    1.0,
    Vec3(1.0, 0.0, 0.0),
)
v = assemble_excitation(mesh, rwg, plane_wave; quad_order=3)

sample = plane_wave_field(
    Vec3(0.0, 0.0, 0.1),
    plane_wave.k_vec,
    plane_wave.E0,
    plane_wave.pol,
)
```

The compatibility function `assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol)`
constructs the same model and delegates to `assemble_excitation`.

## Dipole, loop, and monopole sources

Each local-source constructor stores its own frequency. Use the same frequency
as the forward problem when the source is later passed to
`compute_total_field`.

```julia
electric_dipole = make_dipole(
    Vec3(0.0, 0.0, 0.15),
    CVec3(1e-9 + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0),
    :electric,
    frequency,
)

loop = make_loop(
    Vec3(0.0, 0.0, 0.15),
    Vec3(0.0, 0.0, 1.0),
    0.01,
    1.0 + 0im,
    frequency,
)

monopole = make_monopole(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.025,
    1.0 + 0im,
    frequency;
    include_image=true,
)

v_dipole = assemble_excitation(mesh, rwg, electric_dipole; quad_order=3)
v_loop = assemble_excitation(mesh, rwg, loop; quad_order=3)
v_monopole = assemble_excitation(mesh, rwg, monopole; quad_order=3)
```

For `DipoleExcitation`, `moment` sets the source moment. `orientation` is
validated as unit-vector metadata and does not replace the moment vector.
`type` must be `:electric` or `:magnetic`.

`LoopExcitation` converts the loop radius, current, and unit normal into the
implemented equivalent magnetic moment. It is a small-loop source model.

For `MonopoleExcitation`, `include_image=true` represents the infinite-PEC-ground
equivalent and returns zero below that ground plane. Use `include_image=false`
when a finite ground plane is explicitly meshed and only the physical half-wire
source is wanted. Monopole field evaluation has a bounded numerical integration
work limit and throws when the requested electrical length exceeds it.

## Delta-gap and port surrogates

These models write RHS values without defining a pointwise incident field.

```julia
gap = make_delta_gap(5, 1.0 + 0im, 1e-3)
v_gap = assemble_excitation(mesh, rwg, gap)

port = PortExcitation([5, 6, 7], 1.0 + 0im, 50.0 + 0im)
v_port = assemble_excitation(mesh, rwg, port)
```

The delta-gap assembler sets `v[edge] = voltage / gap_length`. The port
assembler sets `v[edge] = voltage / rwg.len[edge]` for every selected edge.
`PortExcitation.impedance` is validated and stored, but the current RHS assembler
does not use it. Neither model can be passed to `compute_total_field` as an
incident-field source.

## Imported field

Use `kind=:electric_field` when the callback returns the incident electric field
phasor directly:

```julia
source_field = r -> CVec3(
    exp(-1im * k * r[3]),
    0.0 + 0im,
    0.0 + 0im,
)

imported = make_imported_excitation(
    source_field;
    kind=:electric_field,
    min_quad_order=3,
)
v_imported = assemble_excitation(mesh, rwg, imported; quad_order=3)
```

The callback may return a `CVec3`, `Vec3`, numeric 3-tuple, or numeric vector of
length three. Every returned component must be finite.

With `kind=:surface_current_density`, the callback returns $\mathbf J_s$ and the
assembler maps it locally as
$\mathbf E^{\mathrm{inc}}=\eta_{\mathrm{equiv}}\mathbf J_s$. This variant is an
RHS convention, not a source-radiation solver, and is not accepted by
`compute_total_field`.

`min_quad_order` raises the requested order to the first supported order that is
at least as large. For example, a requested order of 2 with a minimum of 3 uses
order 3. Targets above 7 are rejected.

## Pattern feed

A pattern feed stores complex spherical coefficients such that

```math
\mathbf E(\mathbf r)=\frac{e^{-ikR}}{R}
\left(F_\theta(\theta,\phi)\hat{\boldsymbol\theta}
     +F_\phi(\theta,\phi)\hat{\boldsymbol\phi}\right),
```

where $\mathbf r-\mathbf r_0$ defines $R,\theta,\phi$ and `phase_center` is
$\mathbf r_0$.

```julia
theta = collect(range(0.0, π; length=91))
phi = collect(range(0.0, 2π; length=181))[1:end-1]
Ftheta = zeros(ComplexF64, length(theta), length(phi))
Fphi = zeros(ComplexF64, length(theta), length(phi))

for i in eachindex(theta), j in eachindex(phi)
    Ftheta[i, j] = sin(theta[i])
end

feed = make_pattern_feed(
    theta,
    phi,
    Ftheta,
    Fphi,
    frequency;
    phase_center=Vec3(0.0, 0.0, 0.0),
    convention=:exp_plus_iwt,
)

v_feed = assemble_excitation(mesh, rwg, feed; quad_order=3)
E_feed = pattern_feed_field(Vec3(0.0, 0.0, 1.0), feed)
```

The theta and phi vectors must each contain at least two finite, strictly
increasing samples. Theta must lie in $[0,\pi]$. Phi must span less than one full
$2\pi$ period, so omit the duplicate endpoint. `Ftheta` and `Fphi` normally have
shape `(length(theta), length(phi))`; a transposed input is detected and copied
with a warning.

Set `convention=:exp_minus_iwt` only when the imported coefficient matrices use
that convention. Field evaluation conjugates those coefficients before applying
the package's $e^{+i\omega t}$ propagation phase. At the phase center, the
implemented field value is zero.

`make_analytic_dipole_pattern_feed` creates coefficient matrices from an
existing electric or magnetic `DipoleExcitation` on a supplied grid.

## Weighted and batched excitations

`MultiExcitation` assembles a coherent weighted sum:

```julia
wave_x = make_plane_wave(
    Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
wave_y = make_plane_wave(
    Vec3(0.0, 0.0, -k), 1.0, Vec3(0.0, 1.0, 0.0))

combined = make_multi_excitation(
    [wave_x, wave_y],
    ComplexF64[0.7 + 0im, 0.0 + 0.3im],
)
v_combined = assemble_excitation(mesh, rwg, combined; quad_order=3)
```

With no weights argument, `make_multi_excitation` uses a weight of one for every
child. It rejects empty, cyclic, over-deep, mismatched, or non-finite
combinations. Nested models are supported up to the implemented depth limit.

To assemble independent columns rather than one coherent sum, use:

```julia
V = assemble_multiple_excitations(
    mesh,
    rwg,
    [wave_x, wave_y];
    quad_order=3,
)
```

Column `j` equals the RHS for excitation `j`. Batch assembly shares cached mesh
quadrature across compatible children. `max_output_bytes`, `max_work_bytes`, and
`max_terms` reject requests that exceed their declared resource limits before
the main allocation or loop.

## Verification checklist

For a new excitation setup:

1. Check that every frequency-derived source matches the forward wavenumber.
2. Compare at least two supported triangle quadrature orders.
3. Verify plane-wave polarization with
   `abs(dot(normalize(k_vec), pol))`.
4. Compare a `MultiExcitation` RHS with the explicit weighted sum of its child
   RHS vectors.
5. For imported or pattern data, compare complex vector components, not only
   power.
6. If total fields are needed, confirm that every excitation defines a
   pointwise incident electric field.
7. Record phase convention, phase center, units, callback definition or data
   hash, quadrature order, and resource limits.

Run the regression suite from the repository root after changing excitation
code:

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
```

## Code map

All models, constructors, pointwise field evaluators, validation, resource
preflights, and RHS dispatch are in `src/assembly/Excitation.jl`. Near and total
field recovery is in `src/postprocessing/NearField.jl`.

See [Excitation API](../api/excitation.md) for constructor signatures and
canonical docstrings. `examples/07_pattern_feed.jl` contains executable dipole,
loop, and imported-pattern demonstrations.
