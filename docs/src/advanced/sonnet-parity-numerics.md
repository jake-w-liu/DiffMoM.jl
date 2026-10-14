# Sonnet parity — material and numerical evidence

This page continues the [Sonnet parity ledger](sonnet-parity.md)
with the material-law, wall-refinement, surface-impedance, skin-depth,
VOL-metal, modal-assembly, low-frequency and spiral reference evidence.
It is the same audit scope — unfinished items remain work items, not
completion claims.

## Axial refinement of bulk wall sources

`planar_refine_axial` applies a volume wall port's full voltage to every
refined slice of that conductor. Its terminal current adds across the
physical thickness. Volume ordinals are remapped after earlier volumes
are subdivided; original port impedance providers, polarity and reference
planes are retained. Via voltages keep their thickness-distributed contract.

The independent 3 mm by 1 mm by 20 micrometre conductor with conductivity
`1e4 S/m` retains its analytical 15 ohm DC resistance after one, two or four
axial slices in both x and y directions, under the original `5e-6` relative
gate. This source-mapping check does not establish Sonnet's native RF volume
loss law or general device convergence.


## Conductor surface impedance range recovery

`planar_layered_surface_zs` and the two-sheet half-film relation preserve
representable impedance components across gamma-square, thickness-square
and intermediate product range loss. General cascades recover in scoped
wide precision, retaining higher caller/input precision. Positive Float64
single open films recover through exponent-balanced products. Results
outside the scalar output range reject explicitly. The two-sheet wrapper
widens before halving when a subnormal thickness component would be lost.

Julia 1.12.7 and 1.13.1 pass 958 focused checks, including 906 new material
regression assertions. The public grid qualifies 244 representable cases
and 10 range rejections; independent transfer matrices qualify 32
loaded/plated/complex/bare cases. Sixty-six ordinary controls retain
impedance bits and warmed allocation minima. Three rare single-film calls
allocate zero Julia bytes in five warmed samples per case/version.
Concurrent two-layer wide calls preserve precision scopes. These scoped
results do not establish unrestricted numerical range, peak-memory limits,
native frequency-dependent Volume Loss or complete Sonnet parity.


## Skin depth and roughness range recovery

`planar_skin_depth`, `planar_surface_zs` and the Hammerstad/Huray correction
factors preserve documented material equations after intermediate product
range loss. Positive Float64 skin/resistance recovery uses exponent-balanced
products; rare roughness and mixed-type impedance recovery uses scoped wide
precision. The complete mixed impedance stays within the precision scope.
Zero roughness retains its identity factor after input validation.

Both tested Julia versions pass 1076 focused checks, including 91 new
surface/roughness assertions and the existing conductor/Gram tests.
Independent high-precision references qualify the seven reproduced failure
cases. Sixty-six ordinary controls retain response bits and allocation
minima. Cumulative source, standalone resource limits, unrestricted range,
native Volume Loss and full Sonnet parity remain separate acceptance work.

The native NOR material adapter retains finite thick-film RF limits after
intermediate skin-depth overflow; its regression uses independent component
truth at the existing 2e-12 gate and retains genuine output-range rejection.
This material-equation limit is separate from native engine acceptance.


## Native VOL material loss and sheet-resistance selectors

Native `VOL` lowering supports conductivity (`CDVY`, including its default
form), resistivity (`RSVY`, ohm-centimetres) and DC sheet resistance (`SRVY`,
ohms/square) in the implemented SOLID, axis-aligned rectangular HOLLOW and
simple thin polygonal HOLLOW domains. Existing VOL/ARR `RPV` behavior retains
its constant axial-resistance contract. Scoped VOL endpoint pads are
described below. Non-RPV ARR and general thick polygonal wall/topology
adapters remain separate work.

The effective axial conductivity uses the physical cross-sectional metal
area, the retained mesh area and the complete DC/RF slab transition. SOLID
uses the complete polygon area. Its equivalent bounding-rectangle wall is

```math
t_{\rm eq}=\frac{A}{w+b+\sqrt{(w+b)^2-4A}}.
```

Native SOLID sheet-resistance controls independently qualify conversion
`sigma = 1/(Rs*t_eq)`; the retained wall field is inactive. HOLLOW instead
uses the original declared physical wall in `sigma = 1/(Rs*t_declared)`.
Rectangular overfill caps its cross-sectional/RF depth at half the shorter
side while retaining the declared wall for sheet-to-conductivity conversion.
The two operations are distinct. Native 100-by-100 and 200-by-100 micrometre
controls at 50, 100 and 200 micrometre walls support that rectangular limit.
The retained `volume_hollow_saturated_native_pads` capture under
`native_via_endpoints` extends this to a complete RF case: a 0.25 millimetre
hollow wall on the 0.2-by-0.2 millimetre square via is bit-for-bit identical
to the SOLID control at all five native frequencies, and DiffMoM meets the
native RF gates. The collapse is rectangle-specific: a triangular-via wall
sweep beyond the inscribed-circle radius departs from both SOLID and the
thin-wall answer, so general polygonal thick walls keep their "topology
adapter required" rejection.

Thin nonrectangular walls use an inward mitered offset and the original
physical wall as the RF depth. The scalar fast path normalizes coordinates
and checks conditioning and boundary separation. Uncertain or extreme
inputs use the original vertices/loss/wall/unit values in scoped wide
precision. Both the outer and inset boundaries must remain simple, and
offset edges must keep their direction. A closing nonlocal neck rejects
before applying the thin-wall material law. This guard protects its domain;
it does not implement the missing thick-wall topology adapter.

The material calculation preserves representable real and imaginary
conductivity components after reciprocal overflow or an underflowing wall
times length-unit product. Independent 2400-bit equations use the stored
Float64 input domain and retain the original `2e-12` component gate. The
shared raster/conformal import preflight reserves precision-dependent owned
wide scalar scratch for every VOL material path, including endpoint-film
recovery, plus geometry
workspace for general hollow polygons. A fixed 1 MB budget accepts the qualified 32/8192-bit
caller cases and rejects the 1048576-bit case before that oversized material
workspace is allocated. Precision scopes and borrowed project inputs remain
unchanged. This accounting bounds declared raw workspace, not opaque memory
or peak process RSS.

`test/fixtures/native_volume_sheet_selector/manifest.json` records twelve
original Sonnet 18.53-Lite source/native/process snapshots and SHA-256 hashes.
All five captured frequencies and the original engine logs are retained.
Two additional unchanged rectangle source files exercise expanded resource
limits; those in-memory loss variants do not claim new native EM acceptance.
Three SRVY/CDVY SOLID pairs agree under the preregistered `1e-9` complete
matrix comparison; inactive SOLID-wall controls have identical native
matrices. References are not phase-aligned, retagged or fitted.

The new durable module `test_planar_sonnet_volume_material.jl` passes 489
assertions on Julia 1.13.1 with four threads and Julia 1.12.7 with one thread.
It includes provenance, independent material/range/ownership/PEC equations,
both importers' general-HOLLOW/SOLID/rectangular precision budgets, the nonlocal-neck guard, 24 native raster
RF matrices and two genuine-triangle RF consumers. The RF tests retain the
original `0.06` complete-S, `0.05` real-series-loss and `1e-9` physical-equation
gates. A separate source-pinned audit qualifies twelve conformal matrices
across triangle, concave L and diamond SOLID/HOLLOW SRVY cases at 1/10 GHz.

The earlier source-pinned native material audit compares 130 raster matrices per Julia
version: 60 selector controls and 70 rectangular saturation controls. Their
S/loss/reciprocity/passivity gates pass, with 52 high-frequency physical
equation passes. That earlier snapshot has 78 physical residual failures at 1/10/100 MHz.
The later checked-current repeat below closes those failures for these same
130 controls; passing S/loss alone remains insufficient to certify currents.
Native mode/mesh convergence, broader geometry and measured RFIC acceptance
remain separate from these captured controls. Earlier failed full-volume
Maxwell-current candidates and the grounded ARR discrepancy remain preserved.

Five earlier source-pinned warmed scalar allocation samples per case/version record
CDVY thin-wall values: 448/608/512 bytes for triangle/L/diamond on Julia 1.13,
and 768/928/832 bytes in the tested Julia 1.12 context. This is at least a
99.49% reduction against the matched original wide-only scalar protocol.
Rare wide cases retain their larger allocations. SOLID ordinary/tiny-height
scalar controls allocate 5696 bytes after cold fallback closure removal,
with the tested old RPV controls unchanged. These measurements exclude full
model construction, SRVY-specific allocation contexts, timings, opaque
allocations and peak memory; global optimality is not established.


## Native VOL endpoint currents and shared contact films

A via contacted by a narrower sheet needs tangential endpoint currents to
reach its retained axial cells. VOL and PEC vias therefore add endpoint
geometry at both physical interfaces. Original sheet conductors retain
priority. Without `COVERS`, generated conductors follow the declared via
subsection footprint; `COVERS` uses the original physical polygon. Box-cover
interfaces use the existing cover boundary condition.

In the captured Ring controls, SOLID finite-conductivity endpoints use the
open-back half-height film, with conductivity from the native selector.
The captured HOLLOW controls without `COVERS` use lossless tangential
endpoint geometry; HOLLOW pads with `COVERS` use their physical conductivity
and the half-height film. These controlled results do not establish the
same law for every fill or topology. VOL RPV retains its constant axial
resistance and uses endpoint conductivity `2/(RPV*height)`; zero RPV and PEC
endpoints have zero sheet impedance.

Generated finite films sharing an interface combine in parallel; PEC takes
priority. The raster importer tracks generated cells separately from
original physical sheets. The conformal importer decomposes generated
backgrounds by their active via IDs, subtracts original sheets and keeps
genuine sheet polygons. Reversing stacked source-via order preserves the
response in the captured controls. Finite-film error checks use the full
Hermitian admittance matrix for the grounded stacked case. Lossless PEC
checks use the real part of full ABCD B because `100*S11/S21` can indicate
spurious resistance for an asymmetric lossless network.

`test/fixtures/native_via_endpoints/manifest.json` retains sixteen original
Sonnet 18.53-Lite source/reference/process archives. Native Ring licensing
limits these captures. The endpoint test module exercises actual public
raster and conformal solves, partial contacts, finite pads, shared-film
order and grounded dissipation. Its original gates are 0.06 complete-S,
0.05 finite loss, 1e-9 physical residual/reciprocity and passive S. Native
reference bytes remain unchanged. A finer grounded conformal check uses
matching 40 actual native bulk cells and independently sized sheet
triangles at 1 GHz. The coarser grounded native loss changed substantially
under grid refinement; these controls do not certify continuum convergence
or the corresponding fine conformal result at 10 GHz.

Native `ARR` endpoint pads are films of the record's converted
conductivity at the via height, not perfect conductors: a Lite 18.53
conductivity sweep from 1e-6 to 1e6 S/m on a gap-bridging RING+COVERS
footprint reproduces the full native transition (s21 0.000125+0.00148i
through 0.767-0.203i at 10 GHz) with the two-sheet film law. At exactly
zero conductivity the entire record is inert — `cdvy_loss0` agrees with
the absent-via control to 1e-6 — so zero-sigma array polygons generate
neither axial elements nor endpoint pads on either importer, and a via
port on one rejects with "lacks current bases". `RSVY` in the sixth field
converts `sigma = 100/rho` bit-identically (rho=0.01, 1 and 1e-4 controls
equal CDVY 1e4, 1e2 and 1e6 S/m respectively), infinite resistivity or
zero conductivity runs open, and zero resistivity or `INF` conductivity
leaves perfect-conductor pads and column. A selector token in the density
slot is ignored as before, so `ARR 0 .01 1 RSVY` resolves CDVY and opens.
`ARR RPV` records carry no conductivity field; their pads stay perfect
conductors. Evidence: `test/fixtures/native_arr_open_via` (15 source,
response and log triples) and the `test_planar_sonnet_array_loss.jl`
replay testset.

`ARR` `SRVY` is anomalous: Lite 18.53 logs "Array via metal type
converted from Sheet Resistance at DC to Conductivity" and substitutes a
fixed 3.72e7 S/m conductivity. Every finite entered loss — zero and
negatives included — produces bit-identical output, as do changes to the
fill factor and via height: the record replays bit-identically against a
`CDVY` σ=3.72e7 control at fill percent 0.01 and 1 and at two stack
heights. `INF` still selects a perfect conductor. The axial element and
generated endpoint pads both carry the converted conductivity. Evidence:
`test/fixtures/native_arr_srvy`.

The `VOL` RPV wall field is fully inert: Lite accepts five decades of
hollow and SOLID wall values, including negatives, with bit-identical
output and no warning, so the importer ignores it after a finiteness
check. Via RPV records on non-`VOL`/`ARR` metals fail the native parser
outright (`Project File Error`); the importer now rejects that whole
class rather than silently solving a NOCOVERS one. Evidence:
`test/fixtures/native_vol_rpv_wall`.

`NAT` records select the Rdc/Rrf sheet-loss model: two literal numeric
entries, the DC sheet resistance and the skin-effect coefficient with
Rrf*sqrt(f) high-frequency resistance, joined by the same conductor-slab
loss transition the SUP general-loss path uses. The entries do not
evaluate variables — Lite logs "Bad Argument" warnings and substitutes
zero, as it does for absent fields — trailing tokens are inert, signed
entries normalize to their magnitudes (the signed-coth law is
rrf-symmetric and a negative rdc tracks the magnitude within the parity
gate), and `INF` marks an open film: an INF sheet contributes no metal
cells and an INF cover terminates to free space. `NAT` on a via record
fails the native parser ("wrong material type"), matching the importer's
rejection. Verified against Lite 18.53 replays of the installed
`mems_switch` example's `NAT 0.035 1.287e-05` Nickel record — where
Rrf = sqrt(pi*mu0*mur*rho) for mur≈600 — across the DC/RF transition,
pure-DC and lossless limits, cover and signed edges. Evidence:
`test/fixtures/native_nat_loss` and the "native NAT Rdc/Rrf sheet loss"
testset in `test_planar_sonnet_scalar_units.jl`.

Float64 conformal geometry uses scoped 8192-bit intersection and
orientation fallback arithmetic. Degree-two/three numerators retain the
stored coordinate domain before final division; caller BigFloat precision
and original vertices remain unchanged. Independent 32768-bit primitive
oracles and public triangle meshes at caller precisions 32/256/8192 cover
rounding, exact zeros, near-collinearity and extreme exponents. Unrepresentable
nonzero orientation rejects. The importer reserves owned scalar workspace
before construction.

Endpoint rectangle payload is counted before allocation, and retained
background/overlay geometry is included in subsequent mesh reservations.
The measured 10000-record rejection path fell from 1282495 allocated bytes
to 1280 on Julia 1.13.1 and 1728 on Julia 1.12.7. The row-run counter itself
allocates 64 bytes in the measured 2000-run control. These measurements
bound these specific warmed calls; opaque memory, peak RSS and global
optimization remain unverified.

General thick HOLLOW topology, non-CDVY/RSVY/SRVY selectors, SFC
horizontal endpoint behavior, BAR lowering, native non-Ring endpoint
physics, low-frequency physical residuals, mode/mesh convergence and
measured spiral/coupled RFIC acceptance remain open. A supported input or
passing algebraic residual alone does not establish complete native
physics or complete planar/RFIC design coverage.


### Translated volume area and wide rectangle recovery

Physical VOL area is evaluated after translating vertices to a local
origin, avoiding cancellation between large absolute-coordinate products.
A verified axis-aligned rectangle fills its bounding box exactly. Its
normalized fill is retained before the square-root equivalent-depth law,
and wide recovery reforms its physical area from the owned dimensions.
This rectangle flag also follows cold sheet-resistance, resistivity and
HOLLOW recovery. Other polygon shapes retain their separate geometry laws.

The earlier public importer accepted a 10-mm translated SRVY square with
axial component error 7.63e-7 and endpoint-film error 4.18e-7. A private
1-m rectangle exposed 1.05e-8 depth error, and two accepted 1-mm public
SRVY cases at 1e-308/1e-320 exposed 9.85e-9 axial wide-recovery error.
Those negative results remain retained. Independent 32768-bit equations
for actual stored polygon area, slab conductivity and open-back half-height
film now pass eighteen public imports and sixteen private translation
controls across Julia 1.13.1/1.12.7. The maximum public component error is
8.89e-16 under the original 2e-12 gate. Native archives and borrowed
coordinates remain unchanged. These are material/import controls; no
translated native EM or continuum-convergence acceptance is inferred.


### Bounded conformal modal assembly and refinement

Conformal mesh refinement avoids captured boundary-loop coordinates. In
matched warmed public fine-ground imports, Julia allocations fell from
16.34/16.40 GB to 2.069 GB on Julia 1.13.1/1.12.7, at least 87.3 percent,
with identical vertex, triangle, interface and area bytes. Eighteen
adaptive/uniform/interface/band/interior/budget controls matched exactly
on each version. This is Julia allocation telemetry for these calls;
per-call opaque memory and global optimization remain unverified. A
first process-level acceptance now exists separately: the fresh-process
probe `test/peak_rss_probe.jl` drives a dense native solve, the
de-embedding UFFT plus feed-standard cascade, and a repeated solve at a
second frequency, printing `Sys.maxrss()` high-water marks per stage.
`test_planar_process_rss.jl` gates each stage against a 6 GB increment
over the loaded baseline, a 1 GB repeat-solve retention delta, and a
2 GB feed-standard increment (7 checks). On the reference machine the
measured envelope was 357 MB baseline and 1373 MB peak across all
stages — the feed-standard cascade added no peak RSS over the dense
solve itself. This bounds aggregate process memory for those probed
workloads only; per-call attribution inside a stage remains open.

Dense conformal assembly shares triangle phase factors and barycentric
Fourier weights across RWG halves. Cached weight filling allocates zero
bytes after warm-up and matches scalar TE/TM weights bit for bit for PEC
and PMC walls, zero/high modes and multiple interfaces. The phase cache
fits the existing sequential per-triangle workspace reservation.

When remaining `max_bytes` allows at least two columns, up to 64 modal
contributions use grouped, nonconjugate matrix multiplication. Otherwise
the established rank-one accumulation remains. Owned batch arrays are
counted before allocation. Actual two-, three-, seven- and 64-column
partial batches, finite triangle losses, tight-budget rejection and
fallback, reciprocity and finite matrix controls pass on both versions.
The 152 durable weight/cache/batch assertions retain the existing
2e-14 matrix gate. The existing independent hybrid cross-reaction and
triangle-integral tests remain required in the complete suite.

A matched 2,468-unknown fine-ground sheet assembly with 64 by 64 modes
fell from 68.2 to 17.7 seconds on Julia 1.13.1 and from 69.7 to 20.2
seconds on Julia 1.12.7. Relative matrix error was 2.92e-15 on both.
Those timings compare the phase-cached rank-one and batched assemblers
on the same geometry, mode counts and process configuration. The full
320-mode native grounded RF test, complete package timing, hosted CI,
physical low-frequency limits and mode/mesh convergence remain separate
qualification requirements; this benchmark does not certify them.


### Checked low-frequency dense currents

The checked dense-current solver, owned-precision consumers, resource bounds,
and their current validation evidence are described in
[Checked dense currents and numerical bounds](planar-wide-currents.md).


## Spiral reference equations and empirical scope

`planar_spiral_inductance` evaluates Mohan's empirical **DC inductance**
references. The API's `:rectangular` coefficients are the publication's
square coefficients; its geometry generator uses equal-sided square turns.
The paper validates square, hexagonal, octagonal and circular spirals while
neglecting conductor thickness in these reference formulas.

Its simulation family uses outer diameters 100–480 μm, inductances
0.5–100 nH, widths 2 μm–0.3 outer diameter, spacings 2 μm–3 widths,
and inner diameters 0.1–0.9 outer diameter. Its current-sheet error rises
with spacing/width; the reported maximum 8% applies for spacing ≤3 widths.
These are published validation conditions, not new runtime rejection limits.

Positive stored inputs outside that family evaluate an extrapolation. This
reference alone does not predict RF loss, self-resonance, substrate coupling
or measured RFIC performance; qualify those using the full EM workflow.

Source: [Mohan et al., 1999, pp. 1419–1421](https://web.stanford.edu/~boyd/papers/pdf/inductance_expressions.pdf).

### Derived bounds for material and triangle formulas

The positive open-film conductor recovery uses its exponential slab relation
for every finite propagation product. Float64 underflow supplies the vanishing
decay naturally. An infinite propagation product uses the thick-film limit.
This removes the former unexplained finite cutoff; the small-argument series
and other material precision fallbacks retain their existing scope.

Clustered triangle Fourier moments choose Taylor order from the actual
centered imaginary-phase radius `r` and Float64 unit roundoff
`u = eps(Float64)/2`. The
[Genocchi-Hermite simplex identity](https://ftp.cs.wisc.edu/Approx/dvdsurvey.pdf)
represents the divided difference as an exponential integral. For `r <= 1`,
its normalized magnitude is at least `cos(r) >= 1-r^2/2`. After order `m`,
the normalized omitted Taylor tail is bounded by

```math
\frac{r^{m+1}}{(m+1)!\left(1-r/(m+2)\right)}.
```

The chosen order makes this bound no larger than `u*(1-r^2/2)`. Outward
rounding of the selection arithmetic preserves the inequality. The iteration count
also retains the first coefficient-space orders of both sine and cosine
components. Clustered affine fields contract each Taylor coefficient before
rounding accumulated vertex weights, preserving small representable moments.
Error-free three-term additions retain a small surviving constant coefficient
and coefficient-sum roundoff. When a finite coefficient sum or product
overflows before its simplex factorial division, the formula divides the
coefficients first to retain the representable normalized moment.
Scalar, paired, cached and pulse consumers use
the same contraction, with the existing cache storage budget. Zero centered phase has only a constant moment.
This controls series truncation;
recurrence and summation rounding, modal and mesh convergence, and physical
current-solve accuracy remain separate checks. Existing independent simplex,
quadrature, modal-action, native and allocation tests retain their gates.
