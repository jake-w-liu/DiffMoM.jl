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
evanescent modes cannot overflow, and exact-cutoff modes
(`gamma*d = 0`) reduce to transparent sections. A unit modal sheet current on
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

A volume-current via basis is a Galerkin current unknown, not an ideal
galvanic clamp: its z-directed self-impedance is dominated by end-face
charge and remains capacitive at low frequency, so a via "to ground" loads
the structure in the inductive direction without driving the voltage to
zero. Expect a partial loading effect rather than a perfect short.

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
coefficient forms, so exact-cutoff and evanescent modes stay finite, and
the gamma -> 0 limit reduces every layer state to the parallel impedance
`zdn || zup`. In the thin-layer limit (mid-stack, `V` ~ const) a volume
rooftop approaches the equivalent sheet rooftop; against a nearby ground
return it instead recovers the classical `h/3` uniform-current internal
inductance correction — the two models are only identical where the axial
field is uniform.

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

`solve_planar` assembles `Z`, solves one right-hand side per port, and
returns the short-circuit admittance `Y` plus S-parameters normalized to the
per-port reference impedances (`planar_y_to_s` applies the
`D^{-1/2} (I - Z0 Y) (I + Z0 Y)^{-1} D^{1/2}` power-wave sandwich, which only
cancels when all ports share the same `z0`). `planar_sparams` sweeps a
frequency vector; `write_touchstone` writes `.sNp` files.

## Conductor loss

`surface_zs` (scalar or per-sheet vector) adds the analytic Gram term
`Zs * int f_p . f_q dA` to `Z`, nonzero only for a rooftop and its immediate
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

## Limitations

* Conductor thickness is modelled only through the uniform-current
  `VolLevel` rooftop: the current profile is constrained uniform in z
  across the layer, so skin-effect redistribution inside one thick
  conductor is approximated, not resolved (subdivide the layer and place
  a `VolLevel` on each sublayer for a finer axial profile).
* Volume rooftops carry no port and no `surface_zs` Gram term — drive
  them through a co-located sheet and apply sheet-level loss models on
  the bounding interfaces.
* Via columns are z-directed volume-current bases spanning exactly one
  layer each — multi-layer via stacks are built by marking the same cells
  on consecutive `ViaLevel`s. They are not ideal shorts: see the partial
  clamp discussion above.
* Sidewalls are uniform per box (`CellGrid(...; walls=...)`): either all
  four PEC or all four PMC.  A gap port on a `WALL_PMC` boundary is a
  mathematical port — it drives the sheet edge against the boundary
  surface, and with no wall conductor to return current the line end is
  electrically open.
* The gap port carries a shunt capacitance and feed-line parasitic; use
  `deembed_ports` / `deembed_double_delay_*` to move the reference plane
  (double-delay assumes the discontinuity is a pure shunt — checked
  fail-closed at calibration).
* Wall ports on :south/:north require the same `connect_*` flags and claim
  cell *columns* rather than rows.

## References

* J. C. Rautio and R. F. Harrington, "An electromagnetic time-harmonic
  analysis of shielded microstrip circuits," IEEE Trans. MTT-35, 1987.
* J. C. Rautio and M. A. Thelen, "Method of moments analysis of arbitrary
  structures in shielded layered media," IEEE Trans. MTT-69(1), 2021
  (volume-current rooftop and uniform/tapered via basis functions).
* J. C. Rautio, "An experimental investigation of the microwave properties
  of a roughly etched stripline," IEEE Trans. MTT-42, 1994 (shielded
  stripline standard problem).
* J. C. Rautio and V. I. Okhmatovski, "Unification of double-delay and
  SOC electromagnetic deembedding," IEEE Trans. MTT-53, Sep. 2005
  (shunt-discontinuity form of the gap port and the `T_l T_2l^{-1} T_l`
  extraction).
