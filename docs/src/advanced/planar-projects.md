# Declarative planar projects

Projects store units, expressions, layers, named polygons/vias, ports and
component models in TOML. The executable example is
`examples/planar_project_line.toml`.

```julia
using DiffMoM
project = load_planar_project("examples/planar_project_line.toml")
model = planar_project_layout(project; freq=1e9)
result = solve_planar_project(project, 1e9; mx=16, my=12)
maps = planar_current_maps(result; incident_waves=[1., 0.])
data = planar_project_sweep(project; mx=16, my=12)
planar_write_touchstone("line.s2p", data)
write_planar_project("copy.toml", project)
```

## Units, expressions and layer order

Ports accept frequency-dependent Kurokawa references. Use `resistance`,
`reactance`, `inductance`, `capacitance`, or an `impedance` table with
`r,x,l,c`; a scalar complex `impedance` expression is also accepted.
The table's default `topology="series"` gives `R+jX+jωL+1/(jωC)`;
`topology="parallel"` places C across R+jX+jωL, matching native Sonnet.
Signed L/C define references rather than passive loads. Series C=0 is
omitted; nonzero series C at DC is infinite and rejects. Every evaluated
reference must be finite with positive real part.

Physical peak-phasor waves obey `a=(V+Z*I)/(2sqrt(real(Z)))` and
`b=(V-conj(Z)*I)/(2sqrt(real(Z)))`. Thus accepted mean power is
`0.5*(sum(abs2,a)-sum(abs2,b))`. [`planar_power_waves`](@ref) and
[`planar_wave_voltages`](@ref) perform these conversions. Solved results
retain frequency-specific `z0`; current and radiation excitation use it.
Discrete project sweeps retain `reference_series`. ABS outputs are
renormalized to fixed references at the start frequency and report those
in `PlanarSweep.z0`. Touchstone export converts every sample to its
declared fixed real references.

Complex-reference launch embedding connects physical voltages/currents.
TRL standard inputs can name complex measured `z0`, but the line's
characteristic impedance remains an independent calibration ambiguity.
Provide `line_impedance` for physical Kurokawa output; otherwise results
retain intrinsic travelling-wave coordinates and cannot be renormalized
to a claimed physical reference. Reference-plane and coupled ABCD/Y
calibration retain the full transfer back to raw EM gap voltages.

Bare length numbers use `project.unit`; suffixed quantities include
`mm`, `um`, `mil`, `GHz`, `ohm`, `nH`, `pF` and `S/m`. Expression results
are SI: `width="1 mm"` followed by `"=2*width"` produces 0.002 m.
Numeric variables are dimensionless. The interpreter permits arithmetic,
trigonometric functions, `sqrt`, logarithms, `complex`, `real`, `imag`,
`min` and `max`. Undefined variables, cycles, field access, statements and
unlisted function calls reject. Expressions never use Julia `eval`.

`freq` is in Hz and is available for materials, components and reference
planes. Geometry and layer thickness stay fixed with frequency. Each
sweep point reevaluates the dielectric stack and conductor/load models.

Layers and interface indices default to bottom to top. Interface 0 is the
bottom cover, and interface `length(layers)` is the top cover. Legacy
ASCENT projects require `load_planar_project(path; ascent=true)`, which
explicitly converts their top-to-bottom layers and level zero below the
top dielectric. This convention survives serialization as
`stackup.order="ascent"`.

## Geometry, materials and technology

`polygons` use `name`, `level`, `metal`, `net` and `vertices`. Vias use
`from_level`, `to_level`, `via_type` and polygon vertices, or a circular
`center`/`diameter` footprint. `to_level="gnd"` means the bottom cover.
Technology tables map a named drawing layer to a metal interface or via
contract; geometry may refer to `tech_layer` instead of repeating it.
Explicit `stream_layer`/`datatype` metadata survives roundtrip.

The box has `width` and `length`, or `auto_margin`, which translates the
complete layout into a positive box. `[mesh]` sets `nx,ny`; `grid=(nx,ny)`
overrides them. The default is 64 by 64. Features and terminal planes that
cannot be represented on that grid reject.

Dielectrics accept separate transverse/axial permittivity, permeability,
loss tangent and conductivity. Named custom `[materials]` provide defaults.
Relative permittivity/permeability may be complex passive expressions,
including relaxation models. The positive-frequency convention requires
positive real parts and nonpositive imaginary parts. Additional loss tangent
and conductivity add to the imaginary permittivity without changing its real
part.
Nominal presets include air, RO4003C, silicon and GaAs. RO4003C uses the
manufacturer's design permittivity 3.55 and typical 10 GHz loss tangent
0.0027; semiconductor presets provide static permittivity only and need
explicit doping/conductivity and loss models. These are starting values,
with source and scope returned by `planar_material_preset`, rather than
dispersion models. Sources: [Rogers RO4003C](https://www.rogerscorp.com/advanced-electronics-solutions/ro4000-series-laminates/ro4003c-laminates),
[Ioffe silicon](https://www.ioffe.ru/SVA/NSM/Semicond/Si/basic.html),
[Ioffe GaAs](https://www.ioffe.ru/SVA/NSM/Semicond/GaAs/basic.html).

Metal types are `lossless`, `surface_impedance`, `sheet`, `thick`,
`volume` and `bulk`. Declared surface impedance is `rs+i(xs+ωls)`. A bare sheet is
`1/(σt)`; finite films/plating use the analytic conductor cascade, with
Hammerstad or Huray roughness. Project `thick` inserts the two physical
faces, splits the host dielectric stack and remaps every interface. Each
face carries half of every declared film/plating thickness; PEC perimeter
ties connect the faces. This two-sheet approximation omits finite-resistance
lateral side currents. `direction="up"` (default) or `"down"` selects
which side of the declared interface contains the conductor.

`volume` inserts a prism with finite bulk resistivity and all three current
components: horizontal volume rooftops plus uniform/tapered axial profiles.
It adds no PEC or impedance sheet across the prism. `axial_subdivisions`
refines its thickness while retaining common physical wall voltages and
power-conjugate summed currents. Uniform volume plating/roughness and local
magnetic materials require separate resolved material models and reject.
Thick conductors must fit within their host dielectric layer.

## Physical ports, calibration and loaded circuits

Ports identify a polygon edge and have a unique `name` and `number`.
`box_wall` requires a physical box wall. Open interior edges receive
explicit PEC-cover return vias; the returned layout retains their EM lead
effects. `terminal_ground=:below/:above/:auto` chooses a clear cover.
These raw lead effects require standards/calibration for calibrated SMD
comparisons. The solver contracts layered return voltages before solving.
Volume prisms support box-wall ports directly; interior volume terminals
require explicit conductive contact geometry and currently reject.

`type="floating"` identifies two facing pad edges on the same sheet:
`polygon`/`edge` selects the signal and `ref_polygon`/`ref_edge` selects its
finite reference conductor. The factory fills the empty aligned gap with
a declared conductive bridge and places an impressed differential voltage
at an internal current cut. `source_edge` optionally chooses that grid cut;
`bridge_metal` defaults to the signal metal. The reference remains floating.
Bridge geometry and its EM lead effects remain in the raw response. Sister
standards must retain those leads for joint calibration. Native Sonnet
floating/co-calibration comparison remains unavailable with the installed
Lite license; these source contracts have independent charge, power and
response regressions.

For a uniform mirror-symmetric sheet fixture,
`planar_floating_line_standards(problem,bridges;left,right)` constructs the
line, doubled line and a coupled signal-open reflect. Bridge metadata needs
its source `port` index; project layouts retain it in `terminal_paths`.
The finite reference strips remain continuous in the reflect. Geometry,
material and source contracts must be retained when solving these standards.
The solved line responses can feed `planar_group_double_delay_calibrate`,
whose reciprocal/cascade checks reject inconsistent launch identification.
These standards preserve local reference physics; their generation and
checked launch reconstruction do not certify the native GLG algorithm.
`planar_cocalibrate(raw,[left,right],[cal.launch,cal.launch])` applies those
full coupled error boxes to a solved result and retains its full voltage
transfer to the raw source gaps. Current maps and radiation then use the
physical fields behind the calibrated planes.

An axial port declares `type="via"`, a named `via`, physical dielectric
`layer` (numbered bottom to top), and `cells=[first,last]` using column-major
grid cell indices. Its span must lie inside the named via footprint.
`polarity=1` or `-1` applies to any port. Layer splitting distributes an
axial voltage in proportion to each slice's physical thickness.

`refplane={length=...,zc=...,gamma=...}` defines a physical per-frequency
line removal. Results retain a calibrated wrapper and the transfer to raw
gap voltages, so current maps still reconstruct the physical solved fields.

Each EM port can declare `nodes=["S","G"]`, its positive and negative
circuit nodes. Node `0`/`"gnd"` denotes the physical global reference.
Ordinary ports default to their named signal and ground. Floating ports
default to their named signal and `"ref:" * ref_polygon`. Two ports can
share a node without discarding their supplied source modes. The EM
admittance stamps through the signed node/port incidence matrix `D` as
`D*Y*transpose(D)`; physical port voltages are `transpose(D)*node_voltages`.

`planar_project_source_modes(model)` reports the incidence matrix, its rank
and unexcited node-voltage modes. A lone S/G source supplies a differential
response and lacks an independent G/global response. An additional physical
reference-to-global source supplies that information. Connecting node names
does not infer it. Floating circuit components select a voltage coordinate
anchor; `result.circuit.gauge_nodes` records these anchors without adding
a physical ground path.

Components declare either legacy `ports` (named or numbered EM anchors) or
explicit `nodes`. An R/L/C branch uses `nodes=["S","G"]`; a network or
transmission line uses pairs such as `nodes=[["S","G"],["T","G"]]`.
An ideal transformer uses two such pairs or four node names. Legacy R/L/C
`ports=["port",0]` loads that port's actual positive/negative pair, including
a floating return; other legacy anchors retain their physical node wiring.
R/L/C series or parallel branches, transmission lines, ideal
transformers, Touchstone file blocks and inline complex S series use MNA.
File paths are relative to the loaded project's file. File and inline
responses use the public databank interpolation and reference conversion.
Component anchor ports are hidden by default; `external=true/false`
overrides this choice. An unloaded hidden port needs an explicit
termination. Named linear SPICE subcircuits use ordered explicit `nodes`
matching their formal pins and a `subckt` definition name. For example,
given a library definition `.subckt ladder p1 p2 ref rval=10`:

```toml
[[components]]
name = "LadderLoad"
type = "subckt"
path = "models/ladder.lib"
subckt = "ladder"
nodes = ["input", "output", "gnd"]

[components.parameters]
rval = "10 ohm"
```

The reader supports confined relative includes, nested X instances,
numeric defaults/overrides and arithmetic parameters; R/L/C, all four
linear controlled-source families E/G/F/H, named K coupled inductors,
lossless T and scoped O/LTRA distributed lines,
and independent V/I sources
with zero incremental AC drive. It retains source hashes, line origins,
literal global zero and private instance nodes. MNA attachment preserves
exact shorts and ideal voltage constraints. Loaded currents and radiation
use the physical EM nodes, excluding private model coordinates.

K coefficients may be signed, including perfect coupling. The first
terminal of each L is its dotted terminal. Joint PSD energy is checked
exactly for each active group; a K record creates no galvanic connection
between isolated returns. Zero L and zero K values decouple. Matrix
storage, integer bit growth, group size and pair count are bounded
before attachment; scaled branch equations retain physical currents.

SPICE T uses four ordered terminals with `Z0` and either `TD` or `F`/`NL`;
omitted `NL` means one quarter wavelength. O resolves a local or inherited
`.model LTRA` with nonnegative per-length R/L/G/C and positive `LEN`.
The verified model combinations are RLC, RC, LC and RG. Known transient
controls remain provenance and do not change frequency-domain equations.
Each line retains two physical inward currents and separate return gauges.
Exact DC and small electrical-length transfer equations preserve total series
resistance near DC; travelling-wave equations handle greater attenuation.
General RLCG, transient initial-condition fields and other line families reject.

Readable Sonnet linear Spectre exports use the explicit adapter:

```toml
[[components]]
name = "ExportedInductor"
type = "subckt"
dialect = "spectre"
path = "models/1nH_oct_inductor.scs"
subckt = "_1nH_oct_inductor"
nodes = ["input", "output", "gnd"] # native formal pins: 1, 4, REF
```

`planar_read_spectre` accepts the retained export grammar: a
`simulator lang=spectre` header, named `subckt`/`ends` definitions,
literal scalar R/L/C and `mutual_inductor` records. Names are case
sensitive; SI `M` is mega and `m` milli. Native formal node `REF` and
`gnd` remain ordinary nodes, while literal `0` is global ground.
Project `nodes` selects each formal pin's physical circuit connection.
Includes, parameters, expressions, nested Spectre instances and other
device/analysis records reject with source diagnostics. This bounded
export subset does not claim arbitrary Spectre or vendor compatibility.

The standalone APIs are `planar_read_spice`, `planar_read_spectre`, `planar_spice_model`,
`circuit_add_spice!` and `planar_spice_sparams`. Parsing, expansion and
attachment have aggregate storage and record/node/element/depth limits.
Nonlinear devices, behavioural/POLY/TABLE/LAPLACE sources, other model and
transmission-line classes reject. No
nonlinear operating point, transient or noise response is synthesized.
Vendor and other unsupported model contracts require
`component_response(project,component,freq)` returning
`(response=matrix,format=:s|:y|:z,z0=...)`; unsupported models reject.

Native `TYPE SPARAM` uses its literal `SMDFILES` index automatically for
the represented PEC box AUTO return and explicit per-pin FEED width.
`sonnet_component_files(project_or_path, freq)` provides bounded staging
before EM geometry allocation; `solve_sonnet_project(files; raw=true, ...)`
uses its retained snapshots. Each model's final SMDP pin index determines
device order independently of geometry labels. The original file basis is
used for interpolation, followed by optional output renormalization.
Extrapolation is rejected. Exact source/model SHA256 snapshots and a separate
effective project/grid/variable configuration hash remain in
`result.model_files`. In-memory project edits are permitted and retained as
the effective configuration; the raw source hash does not claim those edited
fields came from the current on-disk bytes. Existing staged objects keep their
responses when files change; a fresh staging call reads fresh dependencies.

`circuit_add_sonnet_files!` maps ordered model pins to physical circuit nodes
transactionally. Device common-mode shunts retain the physical box return
node zero. Unsupported CELL/CUST widths, finite-loss AUTO selection,
PolygonPlane/Edge, ordinary floating device references and component pin
reference planes reject explicitly. Raw reference-via leads remain physical;
coupled pin calibration must be supplied independently. `SonnetLinkedProject`
staging materializes only the proved static STF subset and retains the
original SON/STF identity alongside model dependencies. Native data-file and
project-component engine acceptance is separately blocked by the installed
Lite license; automatic raw attachment does not certify that renderer or its
native co-calibration.

Loaded current maps use the actual solved component-node voltages.
`planar_farfield(result;incident_waves=...,radiation_stack=...)` and
`planar_radiated_power(result;...)` use those same physical coefficients.
An explicit radiation stack selects open propagation boundaries without
changing the solved current geometry; default boundaries remain the EM box.
`sweep.adaptive=true` uses the ABS adapter; explicit/logarithmic sweeps
return `PlanarNetworkData`. Connectivity operates on the physical geometry
and declared nets. `max_bytes` reserves derived layout, retained EM data,
circuit workspace and sweep output across the operation.

## Radiation at normal observation angles

At `theta=0` and `theta=pi`, `planar_farfield` evaluates the physical
transverse wavevector as zero and supplies the polarization basis from the
requested `phi`. A purely vertical current therefore has zero radiated
field along its axis. The spherical components of a transverse current
rotate with `phi`, including the sign of the south-pole theta basis.

For nonzero angles near the axis, the transverse component uses
`sin(theta)` directly. This preserves separately representable small
fields even when `cos(theta)` rounds to one; no minimum transverse angle
is imposed. The vertical-current projection retains the linear transverse
norm with `hypot`; a separately representable field is preserved when the
squared wave number underflows. Observation directions retain their physical
sine and cosine at grazing incidence. Matched-layer receiving moments retain
the axial wave number, and PEC images use their antisymmetric exponential
relation. The spherical-wave factor weights the receiving fields before
rounding; binary scaling preserves representable fields from amplified
currents when a unit via reaction would become zero or subnormal. Colocated
uniform and tapered via currents are combined before axial moments round,
preserving the smaller field of a zero-mean current. Axial field integrals
use a convergent series in the unit disk, retaining the working precision. The receiving integrals stay analytic in the squared axial wave number at cutoff, preserving its parameter derivatives.


On macOS, the default raw-payload budget includes free, inactive, and
purgeable pages using the OS page size, bounded by physical memory. Julia's
bundled libuv free-page query can understate the reclaimable memory and reject
a supported solve. This estimate follows [libuv's Darwin availability
calculation](https://github.com/libuv/libuv/blob/v1.x/src/unix/darwin.c).
Explicit caller byte budgets keep the same checks and accounting.
