# Shielded Planar Spectral-Domain MoM

## Purpose

Reference for the shielded multilayer planar solver under `src/planar/`. This
is the formulation for printed-circuit and RFIC-style problems: infinitely
thin metal sheets on the interfaces of a stratified dielectric stack inside a
rectangular shielding box with PEC (default) or PMC sidewalls.

## Model

* Box `0 <= x <= a`, `0 <= y <= b` with PEC (`WALL_PEC`, default) or PMC
  (`WALL_PMC`) sidewalls, selected by the `walls` keyword of `CellGrid`.
* `L` dielectric layers fill `z = 0 .. sum(thickness)`, ordered bottom to top.
* Interface `i` (`i = 0..L`) is the top face of layer `i`; metal sheets carry
  a cell mask (`SheetLevel`) on one interface each.
* The vertical terminations are PEC (`TERM_GND`/`TERM_PEC`), PMC
  (`TERM_PMC`), a surface impedance (`TERM_SURFACE`), or an open half-space
  (`TERM_OPEN`/`TERM_SPACE`).

## Method

Fields expand in the discrete TE/TM modes of the box
(`kx = m*pi/a`, `ky = n*pi/b`). For every mode the layered stack collapses to
a transmission-line cascade in `z`:

```
gamma_l = sqrt(kc^2 - k_l^2),   Re gamma >= 0   (e^{+i w t} convention)
Zc_TE   = i*omega*mu_l / gamma_l
Zc_TM   = gamma_l / (i*omega*eps_l)
```

The lateral eigenvalue lattice `kx = m*pi/a`, `ky = n*pi/b` is the same for
both sidewall kinds, but the mode parity changes: on `WALL_PEC` walls the
transverse modal E field is
`(e_TE, e_TM) ~ (ky cos(kx x) sin(ky y), -kx sin(kx x) cos(ky y))` /
`(kx cos sin, ky sin cos)`, while on `WALL_PMC` walls every sin/cos factor
swaps.  Consequently the PMC box loses the TE modes with `m = 0` or
`n = 0` (their `H_z = sin*sin` needs both indices) but gains the TM modes
on the `m = 0` / `n = 0` axes (`E_z = cos*cos`, only the uniform `(0,0)`
mode — which has no transverse E — is excluded).  The modal norms, basis
transforms, and the wall half-rooftop transforms all follow the wall
parity; wall-port half rooftops on `WALL_PMC` transform with the one-sided
sin kernel `Hs(k,h) = (k h - sin(k h)) / (k^2 h)` instead of `Hc`.

A layer may be uniaxial with optic axis along `z` (common for laminated
substrates and woven dielectrics): `PlanarLayer(epsr, mur, d; epsr_z, mur_z)`
stores the transverse constants in `epsr`/`mur` and the axial ones in
`epsr_z`/`mur_z` (defaulting to the transverse values). Each polarization
then sees its own decay constant while the characteristic impedances keep
the transverse constants:

```
gamma_TE^2 = (mur/mur_z)*kc^2 - k_l^2      (ordinary wave)
gamma_TM^2 = (epsr/epsr_z)*kc^2 - k_l^2    (extraordinary wave)
k_l^2      = (omega/c0)^2 * epsr * mur     (transverse product)
```

so a layer with `epsr_z > epsr` (typical laminate anisotropy) raises the
cutoff of TM modes but leaves TE modes unchanged when `mur_z = mur`.

All cascade arithmetic uses the `exp(-2*gamma*d)` transfer form so deeply
evanescent modes cannot overflow. Analytic small-argument ABCD coefficients
retain the finite TE series impedance `i*omega*mu*d` and TM shunt
admittance `i*omega*eps*d` at exact axial cutoff (`gamma*d = 0`).
A unit modal sheet current on
interface `s` produces the modal voltage `V(f,s) = (Zdn[s] || Zup[s])`
propagated through the cascade transfer factors.

Currents are expanded in one-cell-wide subsection rooftops
(`build_planar_basis`): a rooftop exists on a grid edge only when both
neighbour cells are metal, and wall half-rooftops appear where the sheet is
electrically connected to a sidewall (galvanic short or port). The Galerkin
impedance

```
Z[p,q] = -sum_mn V_pol(f,s; mn) <f_p, e_pol> <e_pol, f_q>
```

is accumulated per interface pair through real `dgemm` on mode blocks, so the
assembly is `O(nb * nmode)` in workspace and `O(nb^2 * nmode)` in flops with
BLAS utilization.

## Via columns

`ViaLevel` adds z-directed volume-current unknowns on the same cell grid: a
via column on stackup `layer` spans from interface `layer-1` (its bottom
face) to interface `layer` (its top). Two independent axial profiles exist
per cell — `uni[i,j]` marks a uniform (constant in z) column and `tap[i,j]`
an up-tapered one (linearly zero at the bottom face to maximum at the top);
a downward taper is realized by the MoM solution as `uniform - up-taper`,
so a cell may carry both marks. Build one with `via_level(layer, nx, ny)`,
mark the masks, and pass `vias=[...]` to `build_planar_basis`,
`build_planar_problem`, or `assemble_planar_z`.

Via columns couple only through TM modes — TE modes have no `E_z`. Their
lateral transforms are the raw cell-pulse transforms `fx*fy` (including the
PMC parity swap), without the transverse `kx`/`ky` factor carried by sheet
rooftops. The axial coupling uses the TM relation `E_z = beta * dV/dz` with
`beta = kc^2 * eps_t / (gamma^2 * eps_z)`, evaluated coefficientwise through
the same `exp(-2*gamma*d)` cascade, so exact-cutoff and deeply evanescent
modes stay finite; open/infinite endpoint loads are handled by the
coefficientwise limits.

A volume-current via basis resolves the axial electric field, end-face
charge and inductive coupling. A connected via recovers a galvanic path
through its solved current, including the physical inductance. Both
grounded and interlevel vias are checked against live Sonnet fixtures.

## Volume rooftops (thick metal)

`VolLevel` adds the Rautio-Thelen volume-rooftop basis family: x- and
y-directed volume currents that occupy the full thickness of one stackup
`layer` — a sheet rooftop extended uniformly through the conductor
thickness. Build one with `vol_level(layer, nx, ny)`, mark `mask` and the
`connect_*` wall flags exactly as for `sheet_level`, and pass `vols=[...]`
to `build_planar_basis`, `build_planar_problem`, or `assemble_planar_z`.
Ports still attach only to sheets; drive a thick conductor through a
co-located sheet on a bounding interface, or place the volume level on a
feed-free interior conductor.

Where a sheet rooftop samples the interface modal voltage `V(f,s)`, a
volume rooftop sees the layer's full voltage profile. Its axial kernels are
evaluated analytically from the per-layer endpoint voltages `Vt`, `Vb` and
the uniform-source self moment `mself = (1/h^2) int int G(z,u)` of the
modal Green's function `G(z,u) = Zc*phi_d(z<)*phi_u(z>)/Dz`:

* vol field <- sheet/vol source on another layer: the source's endpoint
  voltages propagate through the cascade and the field basis takes the
  smooth-profile moment `wbar*(Vt + Vb)` with
  `wbar = tanh(gamma*h/2)/(gamma*h)`;
* vol self term: `mself` (the full double integral, not a sheet value);
* via field <- vol source: the via's profile moment of the volume's
  voltage profile — `Vt - Vb` for a uniform via, `Vt - <V>` for a taper —
  and reciprocally for a volume field driven by a via.

Volume bases carry horizontal current, so they couple through both TE and
TM modes with the same `kx`/`ky` lateral weights as sheets; via-involving
pairs stay TM-only. All axial formulas use the `exp(-2*gamma*d)`
coefficient forms and even-power cutoff limits, retaining the finite TE
series impedance and TM shunt admittance at zero axial decay. In the
thin-layer limit (mid-stack, `V` ~ const) a volume
rooftop approaches the equivalent sheet rooftop; against a nearby ground
return it instead recovers the classical `h/3` uniform-current internal
inductance correction — the two models are only identical where the axial
field is uniform.

## Genuine triangular subsections

`PlanarConformalMesh` preserves physical triangle coordinates and permits
different triangle sizes at sheet edges and interiors. Shared edges carry
normal-continuous affine RWG currents; open metal boundaries carry no normal
current. `PlanarConformalPort` adds inward half functions on complete box-wall
edges or drives complete shared interior edges, including diagonal cuts.
For an oriented interior segment A→B, positive current flows into its left
side. The conductor Gram is integrated exactly as an affine polynomial.
`assemble_planar_conformal_z` evaluates the same multilayer TE/TM box modes
using analytic simplex exponential moments, including repeated and zero
phase limits. It uses no Green or surface quadrature.

`planar_conformal_mesh(polygon; edge_size, interior_size)` triangulates a
simple concave or convex polygon, then bisects shared edges conformingly.
Sloped physical boundaries remain straight polygon segments. Boundary
triangles and a configurable `edge_band` use the smaller edge size. A
provided constrained triangle mesh can represent holes or multiple sheet
interfaces and be refined with `planar_refine_conformal`.

`solve_planar_conformal` returns the physical coefficient columns, Y/S, a
two-sided edge-length equilibrated LU, and recomputed voltage residuals.
`planar_conformal_current_maps` evaluates the piecewise affine vector current
at all triangle vertices and returns its exact triangle integral.

For lattice-aligned triangles, `method=:ufft, nx=..., ny=...` uses exact
translated triangle Fourier families. Congruent triangles use a fixed number
of families; arbitrary distinct triangle shapes increase the family count.
All physical high modes are multiplied by their analytic transforms and
multilayer kernels before alias folding. Real triangle geometry gives
opposite Fourier-sign coefficients as conjugate pairs, so two sign planes
are stored and the other two are reconstructed exactly. This halves the
coefficient arrays without changing the complex, lossy Green kernels.
The exact FFT diagonal includes
cross terms between both triangles of every RWG function. GMRES retains
bounded restart storage and gates a freshly recomputed voltage residual.
Coordinates outside the declared lattice reject instead of changing geometry.
`solve_planar_conformal_defect(prob,f; modes,nx,ny,...)` also accepts genuine
nonuniform triangles on multiple interior sheets in an isotropic stack with
PEC covers/sidewalls and wall ports. A projected FFT operator supplies the
inner solves; bounded outer corrections use independently streamed original
analytic modal actions. A result is returned only when the original voltage
equation passes `rtol`. `relative_residuals` and
`approximate_relative_residuals` report the original and final projected
equations separately. The preconditioner does not replace the original
physical equation.

`surface_zs` accepts a finite complex impedance in ohms/square, a vector with
one value per physical triangle, or a frequency provider returning either.
The exact weighted RWG Gram enters both equations without projection or
quadrature. Evaluated values are retained in
`diagnostics.surface_impedances`; providers run once after resource rejection.
Passive and purely reactive sheets retain the physical current and power
conventions of the dense solver.

The constructor uses bounded spatial bins and exact analytic local modal
entries without a dense oracle. Aggregate array reservations include
construction, restarted Krylov work, physical source/result columns and
original residual work before material/reference providers are evaluated. Explicit
limits cover local pairs, bin references/visits, projection patches/stencils,
modal terms and pair-times-mode work. Original actions still cost
O((triangles + unknowns) × modal terms × source columns × sheet interfaces);
this path establishes bounded
storage, without general fast exact nonuniform FFT acceptance. The optimized
single-interface path remains available. Multiple interfaces retain all
original interlevel kernels, with three projected FFT kernels per ordered
interface pair and bounded original modal blocks. Via/volume/contact geometry
and bulk coupling for this path remain separate work.

`PlanarHybridProblem(conformal,bulk_grid;vias,vols)` couples genuine sheet
triangles to the established axial via and transverse volume bases.
`assemble_planar_hybrid_z` retains exact TE/TM cross reactions and axial
moments. `solve_planar_hybrid(...;method=:ufft)` uses the two geometries'
analytic Fourier families on the same physical lattice, including all
cross kernels before alias folding. It stores no dense cross block.
The exact diagonal and power-conjugate trace scaling precondition GMRES;
bounded iterative refinement handles charge/current cancellation while
retaining the unchanged complete voltage-residual gate and iteration cap.
`planar_hybrid_current_maps` returns both physical representations.
Contact footprints should constrain the triangle mesh so sheet divergence
resolves the galvanic footprint. Independent grounded/interlevel contact
and finite-conductivity via resistance tests exercise actual solves.
Native mixed conformal geometry lowering and mesh convergence remain
acceptance work.

`build_planar_conformal_layout` preserves the exact union of normalized
polygons, explicit void holes, material seams and interior source constraints.
Its vertical slab partition uses actual vertices and line intersections;
the resulting genuine triangles retain sloped boundaries. Rectangular bulk
contact footprints add physical constraints before refinement. Generated
contact coordinates preserve a unique polygon or source coordinate within
one neighboring Float64 value of the lattice calculation. Distinct physical
coordinates in that interval reject; polygon vertices remain unchanged.
Per-triangle
material models retain names and frequency callbacks.
`sonnet_conformal_layout` uses this path directly from native sheet polygons
and the native stack helper, without first rasterizing tiny sheet features.
`solve_sonnet_conformal` preserves gap-plane calibration and shared terminal
number contraction. Native via footprints retain their explicitly declared
uniform bulk grid and meshing flags. `wall_contacts` adds undriven PEC
wall connections, while driven source spans override their zero-voltage
contact. The native adapter preserves actual sheet-to-wall connections.
Nonuniform acceleration, licensed feature acceptance and complete native
mesh/modal convergence remain work items.

## Axial conductor refinement

`planar_refine_axial(problem,[1,64,1])` splits the middle layer into 64
equal physical slices and preserves the original dielectric boundaries,
metal footprints and terminal meaning. Each volume slice gains its own
current coefficient; via slices retain the original uniform or global
linear profile and gain finer axial degrees of freedom. No extra metal
sheets are inserted between slices.

Use `solve_planar_axial(refinement,f;volume_sigma=...,via_sigma=...)` with
material arrays indexed by the original conductor levels. It returns a
`PlanarContractedResult`: via-terminal voltage is distributed over slices
in proportion to thickness, and current is contracted by the transpose
of that same mapping, preserving terminal power. The retained raw solve
supplies current maps on every physical slice.

An independent Maxwell conductor-slab regression at 10 GHz resolves the
skin current of 8.6 µm copper. Relative integrated-current profile errors
at 16, 64 and 128 slices are 5.14%, 0.343% and 0.086%. This verifies axial
profile convergence for that slab; layout and lateral-current convergence
must also be checked for the device being solved.

## Fast frequency sweeps and box resonances

`planar_sweep_abs(prob, fmin, fmax; n_eval=..., max_points=...)` is the
ABS-style adaptive rational sweep: each frequency solves `solve_planar`
once, and a Thiele continued-fraction interpolant predicts the S-parameter
curves between analysis points. The error is estimated by comparing the
highest-order interpolant against the next lower order on a probe set; new
analyses are inserted where the estimate is largest until the predicted
error falls about 40 dB below the data or `max_points` is reached — so the
sweep cost is the small set of adaptively placed full solves, not the
dense evaluation grid. It returns a `PlanarSweep` with the analysis
frequencies plus dense frequency/S samples.

`planar_box_resonances(stack, grid, fspan; mx=..., my=...) ` scans the
modal cascade for box resonances without solving a system: for every
existing TE/TM mode it tracks the parallel impedance `Zdn + Zup` at each
interface and reports a `PlanarResonance` (mode indices, polarization,
interface, frequency, `|Zdn + Zup|` minimum) wherever the pole condition
`Zdn + Zup -> 0` is bracketed over the span. Results agree with the
analytic cavity eigenfrequencies on empty boxes and locate the resonances
a frequency sweep must resolve or step over.

## Ports and network extraction

A `PlanarPort` claims a contiguous range of wall-connected edges on one sheet
level. Each port is an infinitesimal gap-voltage source between the sheet
edge and the sidewall; with unit gap voltage `V`,

```
rhs_b   = -s_p * w_b * V
I_p     =  s_p * sum_b w_b * i_b        (current into the network)
s_p     = +1 for :west/:south, -1 for :east/:north
```

Internal zero-width delta gaps use
`PlanarPort(sheet_level, :x, edge, rows, z0)` or
`PlanarPort(sheet_level, :y, edge, columns, z0)`. Each claimed edge must
have metal on both sides. Positive voltage drives current in the positive
coordinate direction; `polarity=-1` reverses the terminal convention.
Via gap ports use `PlanarPort(via_level, :via, linear_cells, z0)`, with
column-major cell indices. Their axial voltage is distributed through the
layer; the current trace uses the average axial profile (1 for uniform,
1/2 for up-tapered columns). Calibration is applied separately to these
raw gap-port results.

`solve_planar` assembles `Z`, solves one right-hand side per port, and
returns the short-circuit admittance `Y` plus S-parameters normalized to the
per-port reference impedances. Complex references require finite positive
real parts and use Kurokawa waves
`a=(V+Z0 I)/(2sqrt(real(Z0)))`,
`b=(V-conj(Z0) I)/(2sqrt(real(Z0)))`. A port may supply a frequency [Hz]
provider, including `PlanarPortImpedance`; each solved result retains its
evaluated `z0`. References affect wave normalization and terminations;
the physical gap-voltage/current basis is preserved. `retain_matrix=false` factors
the assembled buffer in place, returning `z_mom=nothing` while retaining
the LU and currents; this saves one dense complex matrix. `planar_sparams`
uses that mode by default. `write_touchstone` writes `.sNp` files after
validating the full dataset.

### Current maps

`planar_current_maps(result; port=1)` reconstructs the solved sheet currents
in A/m and volume/via current densities in A/m² at cell centres. The default
is a one-volt excitation of the selected port, with other gap voltages zero.
`voltages` specifies all gap voltages; `incident_waves` specifies Kurokawa
power waves, with unexcited ports terminated in their retained references.
For peak phasors, accepted average power is
`0.5*(sum(abs2,a)-sum(abs2,b))`.
`z_fraction` selects the sampling height within via and volume layers.
`write_planar_current_csv(path, maps)` exports coordinates, level extents,
units, and all three complex current components without smoothing.

## Conductor loss

`surface_zs` (scalar or per-sheet vector) adds the analytic Gram term
`-Zs * int f_p . f_q dA` to `Z`, nonzero only for a rooftop and its immediate
neighbour on the same level. `planar_surface_zs(f, sigma)` builds the value
from bulk conductivity as `Zs = Rs*(1 + i)` with
`Rs = sqrt(omega*mu/(2*sigma))` and skin depth
`delta = sqrt(2/(omega*mu*sigma))` (`planar_skin_depth`). Published
roughness corrections enter through `roughness_factor`:
`HammerstadRoughness(rms; rf=2)` gives the Hammerstad-Jensen factor
`1 + (rf - 1)*(2/pi)*atan(1.4*(rms/delta)^2)` saturating at `rf`, and
`HurayRoughness(radius, density)` gives the snowball factor
`1 + (3/2)*(4*pi*r^2*rho)/(1 + delta/r + delta^2/(2*r^2))`, which approaches
1 at low frequency and saturates at the nodule-area factor when the skin
depth falls below the nodule radius. The factor scales the full complex
`Zs` by default; `loss_only=true` restricts it to `Re(Zs)`.

`volume_sigma` and `via_sigma` specify bulk conductivity in S/m, either
as scalars or vectors ordered by conductor level. The default `Inf` gives
PEC conductors. Finite values must be nonzero with nonnegative real part;
purely imaginary values give lossless local reactance. Values and their
reciprocals must remain finite in solver arithmetic. These add the volume
constitutive reaction from the exact basis
overlap; uniform volume-rooftop profiles carry `rho/h` loss and via
profiles carry `rho*area*int(w_a*w_b dz)`. Layer-thickness adjoints include
these loss terms. Subdivide thick conductor layers to resolve variation
of the axial current profile.

## Differentiability

`PlanarLayer`/`PlanarTerminator`/`PlanarStackup` are generic in the scalar
type, and `planar_mode_cascade` promotes to `Complex{typeof(omega)}`, so
complex-step perturbations of `epsr`, `mur`, `thickness`, `zs`, or `freq`
propagate through the cascade and the real/imag split in the impedance
assembly preserves the perturbation exactly (both parts are kept). Solve
with `solve_planar(prob, ComplexF64(f0, tiny))` to complex-step the
frequency.

`planar_objective_gradient(prob, freq, f)` differentiates a real port-space
objective `f(Y)` with respect to `PlanarParam` descriptors covering every
layer `epsr`/`mur`/`thickness`/`epsr_z`/`mur_z` component and surface/open
terminator parameters. The port-space adjoint uses the already-factored LU
(`dY = -transpose(Λ) dZ X` with `Λ = Z^{-T} E`), and the Wirtinger
contraction is evaluated per mode so that no parameter-dependent dense
`dZ` is ever materialized. Modal-voltage derivatives come from a minimal
forward-mode `_PlanarDual` scalar propagated through the same generic
cascade — complex-stepping cannot be used there because the modal
voltages (and `Y` itself) are already complex, so a real
`imag(v)/eps` extraction would lose the derivative to cancellation.

## Port de-embedding

A gap port carries a parasitic chain between the EM reference plane and
the device plane: in the shielded box it is, to leading order, a pure
shunt admittance `yd` (gap fringe) followed by a length of the port's
connecting line. `deembed_ports(Y, chains)` removes arbitrary per-port
2x2 ABCD chains from a port-admittance matrix — with `chains[p]` mapping
`(V_B, I_B) -> (V_A, I_A)` the result is
`Y_B = (Y_A*B - D)^{-1}*(C - Y_A*A)` where `A,B,C,D` are the diagonal
per-port ABCD entries. `deembed_port_extension(Y, zc, gl)` is the
special case removing the same `planar_line_abcd(zc, gl)` line section at
every port (`gl` may be negative to add length).

`deembed_double_delay_calibrate(Y_l, Y_2l; len)` implements the
double-delay extraction: for thru standards of length `len` and `2*len`
built with the same port geometry, `P = T_l*T_2l^{-1}*T_l` cancels the
line and leaves the doubled shunt discontinuity `[1 0; 2*yd 1]`. The
pure-shunt form is a fail-closed self-diagnostic (`tol`) — the solver's
gap ports satisfy it to machine precision. Removing `yd` from the `len`
standard then yields the connecting line's TEM-equivalent `zc` and
`gamma*len` (electrical-length branch resolved so that
`planar_line_abcd(zc, gamma_l)` reconstructs the measured section; keep
standards under roughly a quarter wave so `acosh` does not wrap).
`deembed_double_delay_apply(Y, cal)` then removes `Md*L(len)` per port —
moving each reference plane `len` into the DUT — or just `Md` with
`line=false`.

Circuit extraction turns de-embedded port data into element values:
`planar_line_params(Y)` recovers a uniform line's `zc` and `gamma*len`;
`planar_rlgc(Y, len, f)` (or `planar_rlgc(cal, f)`) reports the
per-unit-length `z = gamma*zc` and `y = gamma/zc` as R, L, G, C;
`planar_pi_model(Y)` gives the pi-equivalent admittances of a 2-port; and
`planar_inductor(Y, f)` reports the series-branch `r`, `l`, and
`q = Im(Z)/Re(Z)` — for a 1x1 `Y` the branch is the port impedance, for a
2x2 the pi-model series branch `-1/Y12`. For the air stripline the
TEM identity `L*C = mu0*eps0` holds to the extracted line's accuracy.

### General calibration and port reference planes

`planar_double_delay_calibrate(Y_l,Y_2l;length)` recovers a full symmetric
reciprocal launch box `E`, using `E^2=T_l*(T_2l\\T_l)`. Its series and
shunt terms are retained. The principal matrix logarithm selects the
launch near identity; symmetry, reciprocity and reconstructed standards
are checked. Different left/right launches require
`planar_line_calibrate(thru,line;delta_length,reflect_standard)`, which
uses the two isolated equal-reflect responses to fix eigenvector scaling.
The common reflection may be known or recovered with its open/short sign.
The conventional TRL output reference is the line impedance. Independent
impedance information is required to resolve that calibration ambiguity.

`planar_line_standards(problem)` creates uniform thru/line and isolated
open-stub raster problems for aligned opposite-wall sheet launches.
It preserves the transverse launch and grid spacing. These are ordinary
solver problems; use consistent physical modal truncation across their
different box lengths. Other launch geometries require explicit standards.
`planar_group_line_standards(problem;left,right)` generates coupled
length/double-length lines and geometric reflect stubs for uniform
multiconductor cross-sections. Every sheet, including unported floating
reference strips, is retained. Its `ordering` places responses in
`[left;right]` order for group double-delay extraction. A four-port EM
fixture reconstructs and removes its coupled launches with residual
4.44e-15; this test does not establish native GLG calibration accuracy.
`deembed_cocal_group(Y,ports,chain)` removes a full coupled ABCD group,
including mutual launch terms. `planar_mixed_mode(S,pairs;z0)` changes to
differential/common power waves and returns the corresponding references.

Store broadband plane shifts in each `PlanarPort` as
`refplane=PlanarReferencePlane(distance,zc,f -> gamma(f))`. The distance
is physical; providers are evaluated at each solved frequency.
`planar_reference_planes(raw)` returns calibrated S/Y and retains the raw
result. Calibrated current reconstruction transfers terminal voltages
through the launch back to the raw gap; the raw current/adjoint meaning
remains intact.

Open-edge `:terminal_x/:terminal_y` halves represent mathematical gap-field
sources and retain their endpoint charge. They do not create a conductive
ground return. `planar_terminal_returns` instead adds physical reference
via columns from pad-edge cells to a clear PEC cover. It returns a voltage
contract matrix: raw voltages are `C*V`, terminal currents are `Cᵀ*I`.
Multilayer sources use layer-thickness voltage weights, preserving work
and distributed axial field. `planar_contract_ports` retains raw data and
provides contracted coefficient columns, Y/S and current reconstruction.

`planar_floating_bridge(problem,signal,reference;source_edge)` provides an
explicit local two-conductor source for aligned facing pad endpoints on
the same sheet. Physical metal fills the empty strip between them; a full
shared-rooftop source cut drives `Vsignal-Vreference`. The reference pad
remains floating, and the complete strip delay, loss and mutual coupling
remain in the raw response. Existing endpoint half sources are removed.
Metadata identifies new cells and the full cut for material assignment
and coupled sister standards. Sheet paths bypassing the cut reject.
Independent finite-volume current divergence verifies equal and opposite
pad charges and zero net charge. Automatic local-ground sister-standard
construction and licensed native floating-port comparison remain open;
the installed Lite engine rejects the native floating coupons.

Native IDEAL R/L/C components use these physical returns and circuit
loading. The added lead effects remain until a `pin_calibration` group
ABCD box is supplied. General automatic local-ground grouping, arbitrary
floating/polygon return calibration, and the vendor/project device model
must be provided explicitly. This matches Sonnet's documented requirement
to represent a return path and remove its effects during calibration:
[Sonnet port ground references](https://www.sonnetsoftware.com/support/doc_19.52/Topics/Ports/Co-calibrated_Internal_Ports.htm).

### Circuit subdivision

`planar_subdivide(problem,:x,edge)` (or `:y`) cuts between raster cells.
Each contiguous sheet crossing becomes a connected pair of circuit ports;
the original ports remain the external circuit ports. Use
`solve_planar_subdivision(plan,f;chains=...,retain_results=true)` to solve
each part and combine native power-wave network equations with any lumped
or measured elements in `plan.circuit`. Launch chains remove the extra
discontinuities introduced at the cut. `planar_subdivision_currents(result)`
uses the combined circuit node voltages to reconstruct each part's current
and translates maps to the original layout coordinates.

The regression independently solves a 24-cell shielded line, calibrates
8/16-cell sister standards, and recombines 8+16-cell parts. At 1, 5 and
10 GHz its absolute S-matrix difference is below `5e-5`. This gate covers
uniform line cuts. Separate boxes omit electromagnetic coupling between
parts; arbitrary filter/radiating cuts need their own comparison against a
whole solve. Cuts through volume conductors or existing via-port terminals,
and cuts through a transverse port span, require explicit parts and maps.

## Worked example

```julia
stack = PlanarStackup(
    [PlanarLayer(1.0, 1.0, 0.5e-3), PlanarLayer(1.0, 1.0, 0.5e-3)],
    TERM_GND, TERM_GND, 4.99654097e-3, 10e-3)
grid  = CellGrid(stack.a, stack.b, 32, 42)
s     = sheet_level(1, grid.nx, grid.ny)
rasterize_rect!(s, grid, 0.0, stack.a,
                stack.b/2 - 0.7211948e-3, stack.b/2 + 0.7211948e-3)
prow  = findall(j -> any(s.mask[:, j]), 1:grid.ny)
s.connect_west[prow] .= true
s.connect_east[prow] .= true
ports = [PlanarPort(1, :west, prow[1]:prow[end], 50.0),
         PlanarPort(1, :east, prow[1]:prow[end], 50.0)]
prob  = build_planar_problem(stack, grid, [s], ports)
r     = solve_planar(prob, 15e9)     # 50-Ohm stripline: |S21| ~ 1
```

See `examples/planar_stripline_benchmark.jl` for the canonical
50-Ohm stripline check (ground spacing 1 mm, width 1.4423896 mm,
quarter-wave at 15 GHz).

## Layout and RFIC workflow

`PlanarShape` primitives provide lines, bends, tapers, steps, junctions,
edge and broadside coupled lines, polygonal/circular spirals, MIM and
interdigital capacitors, via arrays, radial stubs, and air bridges. Named
pins support placement and attachment. Interface numbers increase from
bottom to top. `build_planar_layout` preserves named metal and via models
and lowers physical polygons to the cell grid. `solve_planar(layout,f)`
evaluates material callbacks at each frequency; mixed sheet metals use
exact cell-weighted vector Gram overlaps. `planar_connectivity(layout)`
reports disconnected declared nets, merged different nets, grounded
components, and missing via contacts on that grid.

`solve_planar(prob,f; method=:ufft)` applies the same finite analytic modal
sum through FFTs and restarted GMRES. It retains no dense basis matrix,
checks the recomputed physical residual for each port, and supports
current maps and stackup gradients. Repeated matrix-vector products reuse
operator workspace; simultaneous operations require separate operators.
The reported raw-array limits exclude opaque FFTW plan storage and Julia
container overhead. Dense solving can omit the additional unfactored
matrix with `retain_matrix=false`.

`assemble_planar_z_ufft(prob,f)` and
`solve_planar(prob,f;method=:dense_fft)` accelerate retained dense assembly.
For each ordered subsection-family pair, four signed sum/difference
lattice kernels retain the complete analytic modal spectra before folding.
Inverse FFTs then supply every matrix entry by lookup. This uses the same
sheet/via/volume kernels and weighted conductor Gram as the independent
modal assembly; no sampled Green function or numerical quadrature is used.
The warm 1008-unknown, 96×48-mode benchmark measures 2.6603 s for modal
assembly and 0.04132 s for FFT assembly, with relative matrix error
`3.9845e-16`. This measurement excludes factorization time.

## Radiation from solved currents

`planar_farfield` evaluates continuous-angle polarized spherical-wave
amplitudes through Lorentz reciprocity. Unit TE/TM waves incident from the
observation medium are propagated through the analytic layered cascade.
Their overlap with each physical current basis uses exact lateral Fourier
integrals and exact axial moments. Sheet, uniform/tapered via, volume and
genuine triangle currents are supported. The output fields are
`r*E*exp(+ik*r)` in volts; intensity is in W/sr.

The radiation environment is an infinite lateral layered substrate, as in
Sonnet's current-based far-field viewer. `planar_radiation_stack` keeps
infinite PEC ground planes and lossless open exterior media. To postprocess
currents from a native constant-resistance Free Space cover, explicitly
replace that artificial cover with `TERM_SPACE`. Generic `TERM_OPEN`
is an exact modal half-space termination, while Sonnet's native Free Space
cover is a constant 376.73 Ω surface impedance. The installed guide and
[Sonnet's antenna analysis paper](https://www.sonnetsoftware.com/support/downloads/publications/EuCAP_SonnetAntennaExamples_Nov2007.pdf)
describe the finite-box/cover convergence required for radiating designs.

```julia
radiation = planar_radiation_stack(result.problem.stack;top=TERM_SPACE)
pattern = planar_farfield(result;incident_waves=[1.,0.],
    theta=range(0,pi/2;length=91),phi=[0.,pi/2],radiation_stack=radiation)
write_planar_radiation_csv("pattern.csv",pattern)
save_planar_plot("pattern.html",plot_planar_radiation(pattern;quantity=:gain))
```

`planar_radiated_power` integrates without duplicate poles or azimuths and
reports the change when both angular rules are doubled.
`planar_radiation_metrics` provides accepted-power gain, independently
normalized directivity/efficiency, circular components and polarization
ellipse axial ratio. Efficiencies above one remain visible diagnostics.
Grazing incidence uses a one-sided axial-cosine limit of `1e-7`.
The independent regression ladder includes a homogeneous dyadic Green
oracle, PEC images, short-dipole total power, a separate layered 4×4 wave
boundary system, axial quadrature and dense/FFT triangle result mappings.
This does not certify that an arbitrary finite-box current solve has
converged to its open-environment solution.

Interactive response/current/radiation exports encode true logarithmic
nulls and conductor-free cells as JSON nulls. No finite dB floor is inserted.
Complex response databanks and CSVs retain the actual numerical data.

`PlanarCircuit` connects EM/data blocks to ideal RLC, transmission lines,
and transformers through modified nodal analysis. Exact open, short and
through S blocks use voltage/current constraints. `PlanarRationalModel`
fits real shared-pole N-port admittance models, assesses global passivity
separately from sample passivity, and exports a real controlled-source
SPICE realization. Multiconductor RLGC extraction retains coupled matrix
terms. Native Sonnet and GDSII/DXF import APIs retain units and reject
unsupported semantics; successful parsing alone does not prove solving
or accuracy parity.

### Linear SPICE subcircuits

`planar_read_spice` reads bounded libraries with relative includes and
SHA256 provenance. `planar_spice_model` compiles a named `.subckt`, numeric
parameters and nested X instances. `circuit_add_spice!` attaches its pins
in their literal definition order, retaining internal nodes and global
`0`/`gnd`. Scalar R/L/C, named K couplings, E/G/F/H controlled sources and independent DC V/I
use direct MNA equations; ideal constraints need no singular Y conversion.
The retained DC source values have zero incremental AC excitation. This
path does not compute a nonlinear operating point or discard unsupported
devices. Grammar and equations follow the
[ngspice manual](https://ngspice.sourceforge.io/docs/ngspice-manual.pdf),
sections 2.1, 2.6, 2.8, 2.11 and 4.2.

The first terminal of each L is dotted. For a coupled group,
`Mᵢⱼ = kᵢⱼ √Lᵢ √Lⱼ`; signed K and perfect coupling are retained.
The finite Float64 correlation entries are dyadic rationals, so their
joint PSD property is checked exactly by bounded fraction-free
elimination. A Hadamard minor bound reserves integer workspace before
matrix allocation. Independently driven winding branch equations are
scaled by `1/√Lᵢ`, preserving physical KCL currents and avoiding an
unnecessary tiny matrix row. Uncoupled arithmetic is preserved. K is
metadata and does not join isolated galvanic gauge components.

`planar_read_spectre` separately ingests the proved readable Sonnet
linear export subset: literal scalar RLC and named mutual inductors.
It preserves case-sensitive names, native formal-pin order, source
records and SHA256 provenance. Spectre SI `M` means mega, `m` milli;
literal `0` is global ground, while `gnd` and `REF` are ordinary nodes.
The explicit project `dialect="spectre"` selects this grammar. The
retained three native exports were evaluated through independently
translated ngspice equations; no native Spectre execution is claimed.

Projects declare a named library definition and ordered circuit nodes:

```toml
[[components]]
name = "matching_network"
type = "subckt"
path = "models/matching.lib"
subckt = "matching"
nodes = ["input", "output", "gnd"] # literal .subckt pin order
parameters = { rval = "10 ohm" }
```

Includes remain within the library root; include cycles, subcircuit
recursion, excessive records/nodes/elements, unsafe expressions and
storage limits reject before attachment mutation. Private internal nodes
are retained in circuit voltages; current and radiation outputs project
the original physical EM nodes. Automatic floating gauges retain isolated
allocated nodes as arbitrary coordinates. Exact uniform-null Y blocks
with a shared return also retain their floating common coordinate; finite
common-mode conductance is not thresholded away.

Archived actual ngspice47 responses cover nested passive networks, all
four controlled-source classes, exact shorts, singular-Y voltage gain,
grounded/floating references and an independent physical common mode.
Real-pole and conjugate-pole exported models also round-trip through this
reader. These checks do not certify arbitrary vendor libraries: nonlinear
devices, behavioural/POLY/TABLE/LAPLACE sources, `.model`/`.lib`/`.global`,
conditional/string/function parameters, SPICE
transmission lines, nonzero AC sources, noise and transient analysis
reject explicitly. Vendor/native project model records still require a
response adapter unless their supported linear library is declared above.

## Limitations and remaining parity work

The [Sonnet parity ledger](../advanced/sonnet-parity.md) records each plan
item, its evidence, and its remaining implementation and acceptance gates.

* A `VolLevel` rooftop constrains its current profile uniform in z
  across the layer, so skin-effect redistribution inside one thick
  conductor is approximated, not resolved (subdivide the layer and place
  a `VolLevel` on each sublayer for a finer axial profile).
* Volume rooftops have power-conjugate wall sources
  `:volume_west/:volume_east/:volume_south/:volume_north`, with lateral
  width traces independent of their normalized axial profile. An
  independent bulk conductor recovers `ρL/(w h)` without PEC face sheets.
  `volume_sigma` supplies their bulk loss; no sheet `surface_zs` Gram
  applies to a volume basis. Alternatively,
  `planar_two_sheet_zs(f,sigma,t)` supplies the published two-sheet model:
  place matching footprints at the physical faces, separated by `t`, and
  give each sheet the open-back impedance for `t/2`. The Green function
  supplies their mutual electromagnetic coupling. Use the returned diagonal
  matrix as `sheet_coupling_zs` with `surface_zs=0`, or pass its diagonal as
  `surface_zs`. Connect both face terminals to the same external node. This
  approximation omits lateral side currents; do not also apply bulk metal
  conductivity or volume loss in the intervening host dielectric.
* `PlanarConductorLayer(sigma,t;mur)` and
  `planar_layered_surface_zs(f,layers;substrate_sigma=...)` model plating
  stacks such as Au/Ni over copper, ordered from exposed face inward.
  Each finite layer uses a scaled conductor transmission-line cascade;
  thin and thick limits avoid cancellation and hyperbolic overflow.
  An exposed-face roughness correction can be applied to the resulting
  impedance. These one-face models require the physical geometry and
  metallurgical layer properties supplied by the caller.
* Via columns are z-directed volume-current bases spanning exactly one
  layer each — multi-layer via stacks are built by marking the same cells
  on consecutive `ViaLevel`s. They include their electromagnetic inductance;
  `via_sigma` supplies bulk loss. Nonresonant exact axial cutoff uses
  current-based analytic limits; true lossless resonances still reject.
* Sidewalls are uniform per box (`CellGrid(...; walls=...)`): either all
  four PEC or all four PMC.  A gap port on a `WALL_PMC` boundary is a
  mathematical port — it drives the sheet edge against the boundary
  surface, and with no wall conductor to return current the line end is
  electrically open.
* The gap port carries capacitance and feed-line parasitics. The original
  `deembed_double_delay_*` helpers assume a pure shunt discontinuity;
  `planar_double_delay_calibrate` identifies full symmetric reciprocal
  launch boxes. General local-ground standards still require physical
  return geometry and independent calibration acceptance.
* Wall ports on :south/:north require the same `connect_*` flags and claim
  cell *columns* rather than rows.
* The rectangular native route uses cell-centre rasterization. The
  `sonnet_conformal_layout` / `solve_sonnet_conformal` route preserves
  actual sheet polygons, material seams and shared diagonal sources on
  genuine triangles with independent edge and interior sizes. Via
  footprints retain their declared bulk grid, whose contact boundaries
  constrain the sheet mesh. `PlanarHybridProblem` includes analytic
  triangle/via/volume reactions and an exact lattice FFT operator.
  General nonuniform acceleration, broader native source/calibration
  lowering and complete native mesh convergence remain acceptance work.
* Current-based infinite-layer radiation output is implemented. Native
  pattern comparison and the full finite-box/cover convergence ladder
  remain required for antenna accuracy parity.
* Native thick/rough/plated and general layout-component projects need
  additional lowering coverage. The checked native corpus and coarse RFIC
  comparisons do not establish the fine-grid 0.1% absolute accuracy target,
  100 dB dynamic range, or arbitrary-layout FEM cross-validation.

## References

* J. C. Rautio and R. F. Harrington, "An electromagnetic time-harmonic
  analysis of shielded microstrip circuits," IEEE Trans. MTT-35, 1987.
* J. C. Rautio and M. A. Thelen, "Method of moments analysis of arbitrary
  structures in shielded layered media," IEEE Trans. MTT-69(1), 2021
  (volume-current rooftop and uniform/tapered via basis functions).
* J. C. Rautio, "An experimental investigation of the microwave properties
  of a roughly etched stripline," IEEE Trans. MTT-42, 1994 (shielded
  stripline standard problem).
* J. C. Rautio and V. Demir, ["Microstrip conductor loss models for
  electromagnetic analysis"](https://www.sonnetsoftware.com/support/downloads/techdocs/MicCondLoss_Mar03.pdf),
  IEEE Trans. MTT-51(3), March 2003. The two-sheet model assigns the
  finite-film impedance for half the total thickness to each physical face;
  its convergence and lateral-current limits must be checked for the layout.
* J. C. Rautio and V. I. Okhmatovski, "Unification of double-delay and
  SOC electromagnetic deembedding," IEEE Trans. MTT-53, Sep. 2005
  (shunt-discontinuity form of the gap port and the `T_l T_2l^{-1} T_l`
  extraction).
