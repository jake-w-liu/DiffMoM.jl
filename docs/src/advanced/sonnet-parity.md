# Sonnet parity implementation and validation ledger

Full implementation and workflow parity is the active requirement. This
ledger maps the `ASCENT.jl/SONNET_UPGRADE.md` plan to DiffMoM's current
sources. An implementation, a synthetic regression, a native comparison,
and an absolute accuracy certificate are different evidence levels.
Unfinished rows remain work items; this document is not a completion claim.

Historical scratch logs and source captures cited below are preserved locally
under ignored `data/prior_worker_validation_20261004`, with original relative
paths and a SHA-256 relocation manifest. They are historical evidence; tracked
`test/fixtures` archives retain the durable native references. Maintained audit
scripts remain in `validation/planar_audit` and write new reports to ignored
`data/planar_audit` by default.

The fresh October 4 follow-up also preserves finite RLC constitutive equations
when intermediate frequency products or reciprocals exceed Float64 range.
The independent 5000-case mixed RLC sweep has maximum complete-S error
`8.89e-16`; bounded coefficient evaluation allocates zero bytes. Circuit
results omit an unrepresentable admittance while retaining finite power waves
and physical currents. Native CKT literals now preserve nonzero SI values,
including extreme decimal exponents, independently of caller precision and
rounding. Native SUP sheet loss retains its DC/RF limits and sums signed
reactances before the final exponent scaling. These range controls do not
establish general continuum, measured-device or native workflow completeness.

## Plan coverage

| Plan | Implemented path | Current evidence | Remaining acceptance or implementation |
|---|---|---|---|
| M0 layered Green functions | `PlanarTypes`, `PlanarImmittance`, `PlanarGreens` | independent TE/TM cascades, exact nonresonant axial cutoff, anisotropy and loss regressions | native dispersive material schema; full radiation boundary validation |
| M1 subsections | `PlanarBasis`, `PlanarConformal`, `PlanarConformalUFFT`, `PlanarConformalProjection`, `PlanarConformalMultiProjection`, `PlanarConformalDefect`, `PlanarHybrid`, `PlanarHybridUFFT`, `PlanarConformalLayout`, `PlanarSonnetConformal` | genuine normal-continuous triangle currents; exact polygon unions, material seams and independent edge/interior sizes; analytic triangle/via/volume coupling and exact lattice FFT; scoped bounded nonuniform multilevel sheet defect solve with exact spatial sheet impedances, gated by the original analytic modal equation; native sloped sheets and shared axial terminals; general physical diagonal sources; independent quadrature, galvanic contact and native trapezoid checks | general nonuniform fast projection/correction and bulk/contact extensions; complete native mesh and mode convergence; broader native component, calibration and geometry lowering |
| M2/M3 matrix and direct solve | `PlanarGreens`, `PlanarFFTAssembly`, `PlanarSolve` | independent modal versus exact FFT retained matrix, all rooftop/via/volume families and losses; native complex S comparisons; 1008-unknown fill 2.6603 s→0.04132 s, relative error 3.9845e-16 | broad geometry acceptance; large-layout factorization resources |
| M4 FFT solve | `PlanarUFFT` | exact finite modal operator versus dense; zero repeated matvec allocations; latest 8192-unknown operator uses 2,212,664 bytes of Julia payload versus 1,073,741,824 bytes for one dense matrix | fine-grid iterative convergence; measured sparse preconditioner prototype did not improve convergence and is not enabled |
| M5 metal/dielectric loss | `PlanarSurface`, `PlanarConductorLoss`, `PlanarMetalModel`, `PlanarSonnetIO` | weighted Gram quadrature; published two-sheet line DC resistance oracle; native mixed films, NOR general current ratio, SEN reactance; physical two-face TMM expansion | licensed native TMM comparison, additional face counts, roughness/plating lowering and refined plated-line agreement |
| M6 vias and volume current | `PlanarVias`, `PlanarVolumes`, `PlanarAxialRefinement` | independent Maxwell BVP and cutoff tests; native interlevel/ground vias and constant VOL/ARR resistance-per-via bridges; automated slab subdivision with source contraction; axial conductor profile error 0.08595% at 128 slices; finite bulk wall-source DC resistance; scoped native VOL CDVY/RSVY/SRVY material laws and RF matrix/loss controls | general thick-wall topology, broader VOL contact convergence, native ARR skin/ground loss, arbitrary bulk contacts and broader material coverage |
| M7 adaptive sweep | `PlanarSweep` | full matrix ABS versus independent analytic cascades, zero/constant cases, callback/resource guards; barycentric fitting resolves three delayed resonances in 14 analyses with maximum full S error 4.217e-7; 12180 independent resonance, off-grid, passivity and resource checks, including a 107.96 dB notch | physical EM and measured broadband/resonance acceptance; finite candidate grids do not certify undiscovered narrow resonances |
| M8 ports/calibration | `PlanarPowerWaves`, `PlanarBasis`, `PlanarTerminalReturns`, `PlanarPortContraction`, `PlanarCalibration`, `PlanarDeembed` | frequency-dependent complex Kurokawa references across six raw solvers, circuits, projects, calibration, retained current/field mappings and fixed-reference ABS; native R/X/L/C Graph oracle; general launch-box identification, coupled/mixed groups, physical return-path DC oracle, native IDEAL SMD and native four-port full complex matrix error 0.000851 | arbitrary nonuniform local-ground standards and calibrated native accuracy; TRL needs known line impedance to identify physical output references; installed Lite rejects non-50 Ω EM references, calibration groups and ports-only components |
| M9 currents/radiation | `PlanarCurrents`, `PlanarRadiation` | physical current maps; exact continuous-angle sheet/via/volume/triangle spectra, receive-layer reciprocity, PEC image/short-dipole power oracles, calibrated/FFT mappings, polarization/gain/CSV/interactive outputs; installed RHP antenna raw S passes at refined modes with fresh native currents | native pattern comparison and finite-box/cover antenna convergence; installed far field viewer is disabled with fresh antenna current data, and its guide requires an optional license; substrate-edge/surface-wave scope remains infinite lateral layers |
| C1 circuits | `PlanarCircuit`, `PlanarSpiceIO`, `PlanarSpiceLines`, `PlanarSpectreIO`, `PlanarSonnetIO`, `PlanarSonnetModelFiles`, `PlanarSonnetProjectFiles`, `PlanarSonnetComponents`, `PlanarNetworkIO` | native attenuator netlists; bounded native SPARAM and literal linear CKT SPROJ snapshots with physical attachment, recursive project/data dependencies and continuous parent-frequency evaluation; archived native PRJ full S at nine frequencies; exact open/short/thru constraints; Touchstone 1/2 S/Y/Z/H/G and mixed-mode tests; native TERM/FTERM references; bounded linear SPICE includes/hierarchy/parameters with SHA256 provenance and direct R/L/C/K/E/G/F/H/V/I MNA; T and scoped O/LTRA RLC/RC/LC/RG with exact DC and stable near-DC/high-attenuation equations; exact grouped coupled-inductor PSD, winding orientation and isolated-return gauges; explicit case-sensitive native linear Spectre export reader; archived actual ngspice47 full Y/S for three installed exports, nested networks, controlled sources, exact constraints and grounded/floating/common-mode circuits; exported affine/conjugate rational-model ingestion; project loaded current/radiation node projection; exact-null Y and isolated-node gauge fixes | nonlinear/behavioural/vendor library classes and their native model-record adapters; native SMD SPARAM/SPROJ acceptance blocked by Lite; broader Spectre grammar, other line classes, geometry project traversal, other GEOVAR modes and vendor acceptance |
| C2 extraction/models | `PlanarExtract`, `PlanarVectorFit` | coupled RLGC oracle, real SPICE restamping, positive-real certification and repair tests; actual ngspice47 complete two-port matrices at 37 frequencies for real-pole/affine-capacitance and conjugate-pole models, grounded and floating references; 444 external and 753 durable reference checks pass, maximum relative matrix error 1.025e-15 | complete measured RFIC model ladder; broader model and engine acceptance |
| C3 subdivision | `PlanarSubdivision` | independently calibrated whole-line versus 8+16-cell circuit recombination | arbitrary filter acceptance and quantified inter-part EM coupling error |
| C4 layout I/O | `PlanarLayoutIO`, `PlanarArtworkIO`, `PlanarArtworkAffine`, `PlanarGerberIO`, `PlanarODBIO`, `PlanarODBSymbolsExtra`, `PlanarODBFonts`, `PlanarODBLayers`, `PlanarODBCompression` | installed GDSII/DXF imports and independent hand geometry; GDS hierarchy/array transforms; GDS M4 native complex S error 0.000452; 19185 Gerber checks including legacy rectangle/ellipse strokes, all MI/SF/OF/IR orders, device-only AS and unchanged aperture/repeat semantics; official corpus and four previews; 62001 ODB++ checks on Windows including nested contours, clockwise regions, squared/rounded/line thermals, home-plate symbols, ordered numeric fillet parameters, legacy U unit records, bounded conductive font text, full-buildup FLIP, canonical matrix references, bounded JSON metadata, transport and an independent libarchive `.Z` exact SHA oracle; rounded square/rectangular annuli with selectable corners, independent signed-distance/area and swept-line raster checks; 9826 shared composite geometry/allocation checks; 27-point independent Gerber/ODB++ native clear-window coupon passes with maximum complex S error 0.000119 | matched-grid DXF complex S gate remains FAIL; full CT spiral process; remaining ODB++ thermal/stencil families, barcodes and dimensional resizing; font renderer/coordinate-format compatibility and broader vendor manufacturing corpus/native EM acceptance |
| C5 native project I/O | `PlanarSonnetIO`, `PlanarSonnetTechnology` | 126/128 installed projects decoded as ordinary geometry/circuit projects; bounded linked-STF source/dependency snapshots retain the other two; all three installed public STFs validate against the captured official XSD; proven scalar/default and exact-node providers; 24 ordinary projects lower at 32×32 actual cells and 50 with aspect-preserving refinement to 256; BOX half-cell decoding verified in live GUI; constant VOL/ARR resistance-per-via loss preserved; scoped VOL conductivity/resistivity/sheet-resistance lowering | applied STF interpolation/etch/rho/RPV/technology geometry and encrypted materials; native linked-STF EM rejected by Lite; further TMM semantics, general VOL thick-wall/contact and ARR conductivity/skin loss, bricks, CUP and unsupported port/project semantics; intake and lowering counts do not certify actual solving |
| P0/P0' RFIC workflow | `PlanarProject`, `PlanarProjectModel`, `PlanarProjectSolve`, `PlanarLibrary`, `PlanarLayout`, `PlanarConnectivity` | safe SI expressions and units, TOML round trips, technology/material callbacks, physical thick/volume stack adapters, wall/interior and axial ports, loaded circuits/current maps, sweeps, bulk-aware gradients, declared opens/shorts; scoped native IDC/MIM full-matrix coupons and independent low-frequency MIM overlap reference | arbitrary volume interior contacts, broader technology presets and complete spiral/coupled-line/measured RFIC acceptance |
| P3/P6 outputs/accuracy | `PlanarOutputs`, `PlanarPlots`, `PlanarRadiation`, `validation/planar_audit` | complex response comparison, equation curves, scoped convergence/report/DC model output, Cartesian/Smith/layout/current/radiation views; stripline width refinement; independent analytic 107.96 dB notch | independent modal/grid convergence, microstrip/coupled-line/measured ladder, physical EM notch acceptance and FEM cross-validation |

The M4 payload measurement excludes opaque FFTW plan allocations and Julia
container overhead. Its dense comparison is one matrix, not a complete
direct solve. Native comparisons use the locally installed Sonnet Lite
18.53 and archive project, version, command, logs and complete complex S.
Lite's native subsection coalescing differs from the uniform cell basis.

## Scoped adaptive sweep evidence

The independent sweep oracle cascades analytic shunt series-RLC branches
and matched transmission-line sections. It does not call production
circuit stamps or rational fitting to compute the expected response.
The declared band is 1–6 GHz with 1001 candidate points, a 64-analysis
limit and an unchanged `1e-5` estimated interpolation tolerance.
The three-resonator case at 2.2, 3.7 and 5.1 GHz previously exhausted
64 analyses with full S error `0.0114012`. Greedy barycentric fitting now
converges in 14 analyses with maximum full S error `4.217e-7`.
The independent single-resonance fixture reaches `−107.9588 dB` and
passes its unchanged `1e-5 dB` depth gate. Dense candidate points and
50 additional off-grid frequencies per case are checked against newly
computed analytic truth, together with reciprocity, passivity and finite
values. These 12180 checks also cover preflight rejection before callbacks
and allocation-free scalar interpolation.

Public bands are validated in the stored Float64 domain before response,
reference or technology callbacks. Overflow, underflow, collapsed
midpoints and duplicated candidate frequencies reject explicitly; finite
responses must remain finite after conversion to ComplexF64. Barycentric
SVD storage is included in the aggregate sweep budget. Current reproducers
are `validation/planar_audit/sweep_frequency_conversion_audit.jl` and
`validation/planar_audit/abs_resonant_ladder_audit.jl`; new reports default to
ignored `data/planar_audit`. Original before/after evidence, including
`sweep_resonant_barycentric_gate.log`, remains in the local historical archive
`data/prior_worker_validation_20261004/validation/planar_audit`.

The algorithm follows the Loewner-matrix barycentric construction in
[Nakatsukasa, Sète and Trefethen, 2018](https://people.maths.ox.ac.uk/trefethen/AAAfinal.pdf).
Fresh actual Sonnet Lite circuit-engine runs also pass 1016 checks at
101 native frequencies per fixture. Three delayed resonators agree with
the independent cascade within `6.876e-13` in complete S; a 107.96 dB
notch preceded by a 20 ps matched launch agrees within `3.923e-11`.
Both pass the unchanged `1e-9` native matrix gate and the notch passes
its `1e-5 dB` depth gate. Their adaptive sweeps use 14 and 10 analyses;
1067 durable checks replay raw native outputs without an installed engine.
Projects, dependency hashes, complete native matrices, logs and unchanged
production-source hashes are retained in
`validation/sonnet_stripline/abs_reference`.

Earlier zero-delay/shared-node/exact-thru native notch variants fail the
same `1e-9` complete-matrix gate by up to `6.450e-7`; their raw runs and
independent falsifiers remain archived in `abs_resonance_C2mgTH`,
`native_notch_thru_yyMjqc` and `native_notch_conditioning_BlghEF`.
Removing the zero-resistance through path did not close that difference.
The finite-launch acceptance does not certify those variants.

An accepted sweep estimates interpolation error on its declared candidate
grid. Physical EM and measured RFIC acceptance, and discovery of resonances
narrower than that grid, remain open; these analytic results do not close
those requirements.

## Scoped network and ingestion evidence

Bounded linear SPICE/Spectre ingestion and direct MNA attachment pass
3877 targeted checks, including 2163 coupled-inductor/export-subset
checks. Original ngspice source decks, libraries, logs, complete
complex responses and SHA256 manifests are retained in
`test/fixtures/spice_linear_ngspice47` and
`test/fixtures/spice_export_ngspice47`. Nested include/parameter networks,
E/G/F/H sources, exact shorts and ideal gains, grounded/floating/common-mode
references, project loads and physical current/radiation mappings are
covered. The additional coupled-inductor and explicit Spectre-export
tests retain signed/perfect K, forward/hierarchical coil names, exact
joint PSD, isolated winding returns, common-mode invariance, zero L/K,
extreme scales and bounded sequential group workspace. Complete actual
ngspice Y/S responses for all three installed readable Sonnet Spectre
exports are retained under `test/fixtures/spice_spectre_ngspice47`, with
the signed coupling decks under `test/fixtures/spice_coupling_ngspice47`.
Both preserve original source hashes, engine logs and SHA256 manifests.
Scalar and coupled project loads also verify physical current/radiation
node projection. This is an explicit linear export subset; arbitrary
Spectre/vendor execution is not claimed. Nonlinear/behavioural models,
other line classes and broader native vendor-library adapters remain implementation
items. Manufactured native SMD SPARAM and SPROJ fixtures each hit an
explicit Sonnet Lite license rejection, retained separately from their
successful IDEAL/child-netlist controls.

The bounded SPICE T/O subset adds 5023 durable checks, retaining 272 exact
ngspice47 source/deck/log/voltage/current files and hashes under
`test/fixtures/spice_lines_ngspice47`. Thirteen line/model cases and three
mixed F/H/K networks use 23 frequencies, unequal references and independently
shifted return coordinates. Actual DC output, 512-bit transfer equations,
near-DC series resistance, scale-safe coefficient products, complex references,
transactional budgets and loaded physical EM current/radiation projection are
covered. Full-S engine errors remain below the unchanged 1e-10 gate.
Native RG's default `(1+gmin)` stamp is separately reproduced: it perturbs
reciprocity at large attenuation. Default output and its independent exact
oracle remain retained; the physical comparison declares `gmin=0` and has
maximum error 9.23e-13. No tolerance or physical model was changed to hide it.

Native SPARAM/SMDFILES now stages exact bounded dependencies automatically
for the proved PEC AUTO+FEED pin contract. Ordered model pins remain separate
from geometry labels; interpolation uses the declared file basis before
optional output renormalization, and extrapolation rejects. Loaded results
retain both raw source hashes and a distinct effective configuration hash,
owned model responses and physical circuit-node current projection. The
213 public checks include independent hand-wired raw EM+Touchstone full
matrix/current checks, complex/dynamic references, Y/Z records, stale source
snapshots, shared files, root/resource guards and transactional attachment.
Installed AUTO/FEED amp binding succeeds; the actual CUST40 example fails
precisely before attachment. Linked static STF snapshots remain retained.
Ordinary FLOAT model references, loss-aware AUTO, other width/reference-plane
contracts, geometry SPROJ traversal, other GEOVAR modes and licensed native pin-group
calibration remain open implementation/acceptance items.

Independent native ANC/SYM XDIR/YDIR dimensions with NSCD/SCUNI/SCXY point sets now
resolve into effective geometry before thick-metal expansion. Sparse polygon
identities, implicit reference points and attached sheet wall-port coordinates
are preserved without modifying the supplied project. Sixteen fresh native
parameter/literal pairs produce bit-identical complete S matrices; the retained
raw coupon solves also pass the declared 0.005 full-S and 1e-9 residual gates.
Sixteen additional fresh SCUNI parameter/literal pairs cover one-axis scaling
in both axes and reference directions, for expansion and contraction of a
notched polygon. Their complete native S matrices are bit-identical; incorrect
translation hypotheses remain retained and differ by more than 0.006.
Current public scaled solves have maximum full-S error 2.19e-5 and original
voltage residual below 5.34e-11 under the same fixed gates. The normal scaling
loop allocates zero bytes in a warmed 100,000-point measurement; rare range
fallbacks use bounded precision and preserve caller rounding and task scope.
These cumulative Julia allocation checks do not establish peak process RSS.
Scaling arithmetic also retains midpoint/subtraction roundoff and selects
the ratio or relative change to preserve strong contractions and small changes.
Four archived failures now agree with an independent 4608-bit affine oracle.
Exact subnormal fixed points and native rounded-center behavior are retained;
two fresh decimal SCUNI controls have bit-identical native literal matrices.
The original helper and failed outputs remain in
`test/fixtures/scaled_geovar_affine_boundaries`, with unchanged native physical
and zero-allocation gates.
Exact nominal passthrough preserves all twenty archived installed examples,
including radial and zero-offset metadata, while invalid dimension headers
reject even at their nominal value, as confirmed by an actual native error.
Fresh native grammar controls also confirm rejection of nonnumeric display
coordinates and negative point-set counts at nominal value. An unused unknown
polygon identity retains native nominal compatibility. An out-of-range point
of a known polygon changes the native response despite nominal equality;
saved-coordinate passthrough has full-S error 1.00014 in the retained repeat.
Current lowering rejects that unsupported point convention explicitly. The
valid nominal and independent literal controls pass the unchanged 0.005 gate
with error 2.78e-5. Exact evidence is retained in
`test/fixtures/native_nominal_geovar_grammar`.
Sixteen additional SCXY native parameter/literal pairs cover both axes, signs,
expansion and contraction, with attached ports on their original wall. All
complete native matrices are bit-identical; axis-only hypotheses differ by
more than 0.014. Current public parameter solves pass the same fixed 0.005
full-S and 1e-9 voltage-residual gates. Exact sources, responses, logs, source
snapshots and invalid diagonal-port evidence are retained in
`test/fixtures/native_two_axis_geovar`. Both coordinate components resolve
without changing the supplied project; an invalid physical wall attachment
rejects as it does in the actual native engine.
Independent native radial dimensions now move each selected point by the same
signed radial distance from its anchor. Twenty-four actual native controls
cover expansion/contraction and all axis, direction and scaling headers; their
complete matrices match the independent literals bit for bit. Proportional
hypotheses differ by 0.000508–0.00355. Current public raw solves pass the unchanged
0.005 full-S and 1e-9 original voltage-residual gates. The point helper uses no
allocation for ordinary coordinates and bounded precision for range or
cancellation cases, checked against an independent 4608-bit formula.
Exact evidence is retained in `test/fixtures/native_radial_geovar`, and
`validation/planar_audit/run_radial_geovar_reference.jl` replays the native pairs.
Native repair of crossing-edge polygons remains unsupported; a retained
native-accepted control rejects explicitly in the public simple-polygon model.
Repeated ordinary adjustable vertices now retain every recorded movement.
Ten actual native ANC/SYM/RAD controls match independent sequential literals
bit for bit. Removing repetitions previously produced radial full-S error
above 1.0 and two symmetric SCXY errors above the existing 0.005 gate.
Current public matrices pass the unchanged 0.005 full-S and 1e-9 voltage-residual
gates, with maximum full-S error 0.0001712. The earlier implementation executes
as an independent regression control, and warmed geometry allocations are
checked against it. Exact sources and failed reference hypotheses remain in
`test/fixtures/native_geovar_point_multiplicity`.
NSCD anchored translations follow the reference-coordinate order. Their
direction field chooses the initial direction when the saved offset is zero.
Sixteen native controls with a conflicting direction field expose the old
125–250 micrometre geometry error and full-S errors above 0.016; independent
literals pass the original physical gates.
NSCD ANC and RAD moving references retain every explicit movement plus the
implicit reference movement. With `r` explicit occurrences, anchored movement
uses the common displacement factor `r(r-1)+1` for every adjustable entry.
That reduction also applies to radial movement when every ray stays away from
its anchor throughout the native adjustments.
Fresh controls cover counts 0–5, 8, 16 and 32, both axes and coordinate orders,
zero offsets, reordered groups, ordinary duplicate points, repeated whole
polygons and all RAD scaling headers. Ninety-four retained native
parameter/literal pairs match their complete matrices bit for bit. Linear,
triangular, exponential and simple sequential hypotheses retain their original
failures. Other explicitly listed references, including scaled ANC and SYM
references, still require separate adapters.
The repeated-reference displacement subtracts quantities before SI conversion
and uses bounded precision for range cases. Ordinary displacement calculations
allocate no memory; aggregate point/storage guards remain in place. Exact
evidence is in `test/fixtures/native_geovar_reference_count_law`, replayed by
`validation/planar_audit/run_geovar_reference_count_law.jl`.
RAD applies three successive adjustments, recomputing the reference radius
between passes. Strong contractions can reverse a ray, so they require the
actual sequence rather than the constant-factor reduction. Thirty-six further
native parameter/literal pairs cover reference counts 0–5, anchor crossings,
diagonal reference rays, repeated ordinary points, reversed group order and
conflicting header fields. Their complete matrices and native subsection
supports match; public geometry, masks, full-S and original voltage-residual
gates are checked. The smooth-path reduction avoids amplifying radius roundoff
at larger counts. Movement accumulates in project units before SI conversion.
A separate case reaches the anchor before another movement and is rejected
because its next ray is undefined; its native full S remains unverified after
an engine memory-allocation failure. Evidence is in
`test/fixtures/native_radial_adjustment_sequence`, replayed by
`validation/planar_audit/run_radial_adjustment_sequence.jl`.
Raster imports validate complete literal and moved box-port edges in the
project's native cell coordinates. Native Sonnet allows half an actual cell
plus 0.0001 cell at a wall; an exact SI bound incorrectly rejects valid inputs.
Six independent literal rectangles cover both axes and both outside ends:
native Sonnet rejects four cases that the raster importer previously accepted.
The two valid corner controls retain the 0.005 public complex S and 1e-9
original-voltage residual gates using independent direct modal assembly.
Sixty-eight additional native controls cover both axes/ends, three native grids,
MM/IN units, rectangular unequal grids and tiny excursions. Three active ANC
parameter/literal pairs check movement across the same boundary. Native
acceptance remains tied to the project grid when a caller selects another
raster resolution. The extent helper allocates zero warmed bytes. Ordinary
unported metal can still be clipped at the box. Evidence is retained in
`test/fixtures/native_plain_box_port_extent`, with native replay in
`validation/planar_audit/run_box_port_extent_reference.jl`.
The boundary and moved controls are retained in
`test/fixtures/native_box_port_boundary_allowance` and
`test/fixtures/native_moved_box_port_boundary_allowance`.
BOX/STD/GAP sheet ports attach to their referenced logical polygon edge. Stored
position fields are annotations; changing them does not select another wall
or reject an otherwise valid attachment. Active geometry updates reset those
fields to the new referenced-edge midpoint. A closing duplicate vertex does
not introduce another edge. Contiguous collinear wall subdivisions share a
physical excitation span. Both native sheet backends project vertices in the
native half-cell wall band using the source BOX counts. Undriven sheet contacts
impose the native ground connection. Both backends clip sheet metal extending
beyond the physical box. The conformal backend intersects the exact polygon
union before triangulation, retaining disconnected regions and material seams.
Twenty independent controls and the original failures are retained in
`test/fixtures/native_box_port_attachment`, replayed by
`validation/planar_audit/run_port_attachment_reference.jl`. Five one-port
short/open controls retain the 0.005 full-S and 1e-9 voltage-residual gates.
Genuine conformal sheet geometry preserves interior coordinates and annotation
metadata. Source BOX counts determine wall contact even with a caller bulk
grid or different triangle sizes. Original conformal false rejections and
one-port short/open errors near 2.0 are retained in
`test/fixtures/native_conformal_wall_ownership`; seven conformal full-S controls
use the same 0.005 and 1e-9 residual gates. A wall edge supplies a wall source;
an interior edge supplies a source on its shared segment with one adjacent
sheet edge. Native BOX/STD/GAP labels all accept these two attachment forms.
Partial shared edges use their actual overlap. Isolated interior edges and
edges shared by three polygons reject with the native return requirements.
Internal conformal sources preserve the native phase convention when the
other adjacent polygon is referenced. Thirteen independent native controls,
plus two-port orientation controls on both axes, retain the 0.005 full-S and 1e-9 residual
gates in `test/fixtures/native_internal_port_attachment` and
`test/fixtures/native_internal_port_orientation` and
`test/fixtures/native_internal_y_port_orientation`.
Native Sonnet rejects diagonal internal ports. The native conformal adapter
enforces that restriction, backed by four independent BOX/STD/GAP rejection
controls in `test/fixtures/native_internal_diagonal_rejection`. General
physical diagonal sources remain available through `PlanarConformalPort`
and `build_planar_conformal_layout`.
Ten independent native box-clipping controls cover all walls, concave and
oblique geometry, a corner cut, outside undriven metal, a shared interior
source, and rejection of an outside driven edge. The nine accepted conformal
controls retain the 0.005 full complex S and 1e-9 original voltage-residual
gates. Original failures and native matrices are sealed in
`test/fixtures/native_conformal_box_clipping`; rerun the native engine with
`validation/planar_audit/run_conformal_clipping_reference.jl`.
The general physical layout builder keeps strict box bounds by default;
`clip_to_box=true` explicitly enables sheet clipping. `planar_conformal_mesh`
accepts `clip_box=(a,b)`. A clipped-away sheet can leave a bulk-only layout
when no driven sheet ports remain. These sheet controls do not establish
native via or volume clipping parity.
A repeated whole-polygon control confirms the native rejection when its port edge extends
outside the box; `test/fixtures/native_geovar_box_port_extent` retains the
engine error and independent native-normalizer stages.
Native `POLY id 0` selectors expand every logical polygon vertex, excluding
the closing duplicate. Sixteen NSCD ANC/RAD controls cover both axes, both
directions and expansion/contraction; native zero-count, explicit-list and
independent-literal matrices agree bit for bit. Public solves pass the unchanged
0.005 full-S and 1e-9 voltage-residual gates, with maximum full-S error 0.0000191.
Expanded selectors consume aggregate point and storage budgets before their
lists are allocated. Source ownership and closing vertices remain intact;
scaled/reference and dependent adapters retain their stated limits. Exact
evidence is in `test/fixtures/native_geovar_whole_polygon`, replayed by
`validation/planar_audit/run_whole_geovar_reference.jl`.
Twenty-four RAD whole-polygon controls additionally establish identical native
matrices under NSCD, SCUNI and SCXY headers. The public adapter retains radial
movement for each header, including one explicit moving reference; the old
rejection, independent literal geometry, full-S and voltage-residual checks,
ownership and budget boundaries are preserved in
`test/fixtures/native_rad_reference_headers`, replayed by
`validation/planar_audit/run_radial_geovar_reference.jl`.
Zero saved ANC/NSCD offsets are accepted when the reference coordinates coincide
on the selected axis. Eight native X/Y, direction and target-size controls match
independent literal geometry bit for bit; the prior adapter rejects all eight.
Six literal public solves passed the unchanged 0.005 full-S and 1e-9 voltage-residual
gates in the preserved baseline; the negative-direction 0.0625 target failed
full-S on both axes (approximately 1.00016 and 1.00014), despite small residuals.
Independent native current/subsection supports exposed two missing X cells and
one spurious Y wall cell from horizontal-ray boundary ownership. Vertical-ray
half-open rasterization with diagonal roundoff handling now passes all eight
original physical gates, with maximum full-S error 0.0000340. Axis-aligned
ownership and zero warmed raster allocation are preserved. The old failures,
causal wall-contact control and rejected inclusive-diagonal hypothesis remain
immutable in `test/fixtures/native_diagonal_boundary_raster`. Exact sources, old implementation,
complete six-PASS/two-FAIL baseline and hashes are retained in
`test/fixtures/native_zero_nominal_geovar`, replayed by
`validation/planar_audit/run_zero_nominal_geovar_reference.jl`.
Scaled zero dimensions, other dimension kinds and distinct-coordinate zero
references remain guarded. Positive source values lost in SI remain rejected.
Overlapping/dependent dimensions and moved component/interior/via attachments
still require explicit adapters. Point/parameter/storage budgets and stored
coordinate cancellation reject before emitting a changed project.
Exact native sources, matrices, metadata, rejected hypotheses, implementation
snapshots and checksums are retained in `test/fixtures/native_scaled_geovar`;
`validation/planar_audit/run_scaled_geovar_reference.jl` replays the source
pairs into a fresh directory with source-stability checks.

Literal linear CKT SPROJ children now stage through the same physical pin
contract. R/L/C, earlier DEF invocations, Touchstone data and recursive PRJ
children retain exact source bytes and hashes, with shared dependency reuse,
root/cycle/depth/node/element guards and aggregate nested-solve storage bounds.
INHSWP N/Y is retained while the child evaluates at the requested frequency.
Nine archived actual native PRJ controls, including inheritance flags 0/1
and coarse/fine child sweeps, agree within `5.261e-15` in complete S.
Independent analytic reactive responses, resistor-loaded geometry/current
maps, changed-file snapshot isolation and transactional attachment pass.
The native source/log/complete-SID evidence and hashes are retained under
`test/fixtures/native_sproj_linear`. Native SMD SPROJ renderer and coupled
pin-calibration acceptance remains blocked by the Lite license; arbitrary
geometry children and parameter bindings still require explicit adapters.

Native `PRJ` records also accept an explicit common-return node after the
signal pins. Each child pin maps to `(signal, REF)`; omission retains return
zero. Six actual native CKT controls cover implicit/explicit zero, internal,
external and shared returns at three frequencies. Independent nodal
incidence agrees in all eighteen complete S matrices within `5.63e-13`
under the unchanged `1e-10` gate. The source/log/GUI/manual evidence and
135 public feature, rejection, node/current and ownership checks are
retained in `test/fixtures/native_prj_common_return`. The trailing native
0/1 field is preserved; its meaning remains unidentified. This closes the
common-return record and physical node mapping, while automatic geometry
calibration and licensed SMD model acceptance remain open.

Touchstone export selects standard 1.0/1.1/2.0/2.1 syntax; actual Sonnet
18.53 requires legacy syntax for the declared delay fixture. Unequal
legacy Y/Z/H/G references now use normalized terminal coordinates,
with 726 independent physical-source, 256-bit literal-input and archived
ngspice checks. A near-open Z-to-DB encoding loses `1.704e-10` in full S
relative to its source; the reader differs from the exact stored-data
oracle by `1.582e-11`. The earlier external round-trip failure remains
archived and is not attributed to a parser error.

Valid complete 224-port inputs exposed numeric allocation before budget
rejection. Token-count preflight and direct numeric append reduce warmed
512-byte-budget public rejection from 16,245,814 bytes to 1344 bytes.
The same inputs are accepted at 32 MiB; arbitrary data-line lengths remain
valid. Option references and native TERM/FTERM numeric records are bounded,
and empty explicit reference blocks reject rather than invent defaults.

Project layout, metal and solve entry points now normalize and validate the
stored positive Float64 frequency before schema traversal, material
evaluation or component callbacks. Finite BigFloat underflow previously
reached zero-frequency material evaluation, and lossless metal's early return
accepted overflow. Invalid wide frequencies now reject immediately;
representable BigFloat frequencies follow the same evaluated workflow as
their stored Float64 values, covered by 62 new checks.
Resource and empty-block gates have 79 checks; export version and legacy
matrix-order gates have 245 checks. Reproduction logs are in
`data/prior_worker_validation_20261004/validation/planar_audit`; original valid inputs and before/after evidence
are retained in `data/prior_worker_validation_20261004/validation/touchstone_resource`.

### Circuit stored-value domain

The public network databank had a separate positive BigFloat frequency
underflow: `1e-500` became stored DC and its reference provider received
`0.0`. Constructor and response frequency guards now reject overflow,
nonzero underflow and collapsed stored knots before provider evaluation.
Explicit DC and representable wide frequencies remain accepted. The 44
additional checks include 17 existing Touchstone text-domain controls:
its Float64 parser already rejects nonzero underflow, so that candidate
was discarded and its parser was left unchanged. Exact before/after
source identities are retained as `network_frequency_domain_*`.
The durable `test/fixtures/network_frequency_domain` archive retains twelve
hashed proof files, including byte-exact original/final sources and the
discarded-candidate replay. Its 34 integrity/disposition checks and the
44 domain checks pass; the targeted network/circuit/output/model-file
neighbor gate passed 1999 checks before the later scalar-table batch.

An independent public-entry audit found that finite BigFloat R/L values
could become stored infinity, a positive C or nonzero transformer ratio
could become zero, and a valid-looking input frequency could return a
result labelled infinity or DC after two callbacks. Public scalar
constructors now validate Float64 representability before mutation.
Circuit frequency is validated and stored before any response/reference
callback, and callbacks receive that finite stored frequency. Copied
network responses are checked in ComplexF64 before local references are
evaluated. All 257 new domain/transaction/provider checks and 24 previous
circuit checks pass; the current SPICE/Spectre/project regate also passes.
Before evidence is `circuit_scalar_conversion_before.log` and
`circuit_frequency_conversion_before.log`; the after gate is
`circuit_stored_domain_gate.log` in `data/prior_worker_validation_20261004/validation/planar_audit`.

### Transmission-line numerical stability

The circuit's previous ABCD line stamp destroyed reciprocity through
hyperbolic cancellation: a matched line with electrical length
`40 + 0.37im` returned reverse transmission magnitude 4 instead of
`exp(-40)`, and attenuation 710 or greater rejected despite finite S.
The new terminal travelling-wave equations retain only an exponential
with magnitude at most one, for either attenuation or gain, with bounded
row coefficients. Constant inputs validate before circuit mutation;
provider results validate in the stored complex domain. All 908 new
checks pass against independent original ABCD boundary equations solved
at 4096-bit precision, including unequal complex references, floating
returns, reversed polarity, cascades, zero/half-wave limits and attenuation
1000. The scalar numeric stamp has zero warmed allocations. These checks
establish circuit constitutive correctness for the tested domain; native
physical-line/material convergence remains a separate requirement.
Reproductions are `circuit_line_conditioning_before.log`,
`circuit_line_conditioning_after.log` and `circuit_line_production_gate.log`
in `data/prior_worker_validation_20261004/validation/planar_audit`.

An independent allocation profile also found missing internal-network
workspace in circuit preflight. A 64-local-port/two-external-port block
passed a 300000-byte limit and invoked three providers despite requiring
at least 416336 reserved numeric bytes. It now rejects before every
provider. Direct native equation stamps reduce 64×64 complex plane
allocations from eight to two; a function barrier removes boxed scalar
loop allocations. Warmed cumulative allocation falls from 1609576 to
423784 bytes in that scoped solve. Response snapshots remain owned before
reference callbacks, and 82 new resource/ownership/allocation checks cover
S/Y/Z blocks. Evidence is `circuit_network_workspace_before.toml` and
`circuit_network_workspace_after.toml`. These are Julia allocation and
numeric-payload measurements, not a process RSS or arbitrary callback
allocation guarantee.

### Scoped native technology intake

`PlanarSonnetTechnology` uses EzXML syntax parsing and bounded offline
snapshots. Source bytes are bounded before parsing; streaming preflight
counts elements, depth and estimated retained storage before tree intake.
DTD/custom entities, processing instructions, external schema dependencies
and unknown physical namespace semantics reject. An explicit local XSD can
validate the exact snapshot, with its SHA256 retained; schema-location URLs
are metadata and are never fetched. A tiny 18-byte source allocates 2480 bytes
under 1 KiB, 1 MiB and 8 MiB ceilings, preserving the prior ceiling-allocation
failure evidence. Machine-maximum ceilings have separate regressions.

All three installed public STFs validate against the captured official
`matl-1.3.xsd`. Their complete XML, model/material/mesh identities, tables,
GDS maps and top-to-bottom stack order remain retained. Proven providers
evaluate scalar/default isotropic materials and exact native-unit table
nodes. They reject off-node interpolation/extrapolation, applied etch/rho/RPV
geometry semantics, encrypted physical data and unknown models. No literal
SCAL record was found in the installed source/help corpus, so no inferred
SCAL grammar is implemented. The two linked installed SON files have exact
source/dependency envelopes; their inherited technology geometry remains
unsupported and they are not counted as fully decoded ordinary projects.

A separately hand-authored scalar two-layer STF/PEC coupon is retained in
`validation/sonnet_stripline/native_technology_reference`, including exact
native command/version logs, dependency hashes and complete complex S.
Static materialization reproduces its independent inline SON stack and
layout. The actual native inline response agrees within 0.000108 under the
unchanged 0.005 full-matrix gate; original source residuals are below 1e-9.
Both actual linked runs explicitly report: “Sonnet Lite does not allow use
of linked STF files.” Native stdout does resolve 100 UM STF variables to 0.1 mm
before that rejection. This proves unit intake and the license restriction;
it does not establish native linked-STF EM equivalence.

Primary source/schema/help snapshots and 30 corpus checks are recorded in
`data/sonnet_validation/sonnet_stf_primary_rd055bdl` and
`data/prior_worker_validation_20261004/validation/planar_audit/sonnet_technology_corpus.toml`. Static XML intake is
a scoped C5 implementation; general technology geometry remains open.

Actual technology-editor maintain-physical exports in
`test/fixtures/native_stf_unit_gui` prove `70000 OHUM = 0.07 OHMM` and
`2 OHSQ = 2000 MOSQ`. The incorrect OHMM factor previously produced 1000 times
the physical conductivity; that failed raw reproduction remains archived.
The reader now converts OHMM as ohm metres and MOSQ/MOHSQ as milliohms per
square. The editor accepts the schema's MOHSQ spelling and saves MOSQ with
unchanged values. Official matl-1.4 only lists MOHSQ, so strict optional XSD
validation still rejects the exact native MOSQ exports. No schema rewrite
or geometry/table interpolation is implied. Schema-defined `writeable`
Boolean authoring metadata is preserved without changing the static stack.

The same actual technology-editor load writes a dielectric `RSVY` field of
7 for a 0.07 ohm metre substrate. Native stdout identifies this as
`7 Ohm-cm`. Six raw-R50 runs at 1/5/10 GHz show full-S agreement within
1.01e-13 with independently authored conductivity controls of 100/7 S/m.
Ignoring the marker previously produced full-S errors of 0.166–0.175.
The corrected reader and independent manual stack have maximum error
0.000546 against native under the unchanged 0.005 gate, with original
source-equation residual below 1.84e-13. Exact sources, GUI/installed-help
evidence, outputs and the original failures are retained in
`test/fixtures/native_dielectric_resistivity_gui`. Licensed STF EM parity
remains unverified.

### Native normal-metal selectors and frequency expressions

Twenty-four independently authored native coupons provide 48 raw-R50
responses at 1 and 10 GHz. Their 166 hashed source/output/proof files and
separate manifest are retained in `test/fixtures/native_sonnet_scalar_units`.
Eight selector coupons establish that `NOR` uses `CDVY` (or no selector)
for S/m, `RSVY` for ohm centimetres and `SRVY` for ohms per square; thickness
uses `DIM LNG`. Physically equivalent selector controls produce bit-identical
native full matrices. Three zero-loss controls similarly establish
`RSVY0`/`SRVY0` equivalence with native PEC.

Nine frequency-expression coupons establish `FREQ` as SI hertz even when
`DIM FREQ` is GHz or MHz. The previous project-unit interpretation failed
actual full-S comparisons by up to 0.988; those failures remain archived.
Four additional native coupons prove `h2p`/`p2h` frequency conversion and
`m2p`/`p2m` length conversion against independently authored literal controls.
The corrected finite-film provider and independent hand geometry agree
within 0.000255 under the unchanged 0.005 full-matrix gate. Original source
equations, physical current maps, passivity, output selection/reference
guards, SHA256 and independent CRC32 checks are retained. The new registered
scalar/selector gate passes 1,251 checks. Finite current ratios as large as
1e308 now reach the finite one-face limit rather than overflowing to NaN.

A separate native technology-loader discrepancy remains explicit: an STF
variable `Rs=2000` with `MOHSQ` exports `SRVY0.002`, while the same literal
2000 exports `SRVY2`; using variable `Rs=2` with `OHSQ` also exports `SRVY2`.
Exact GUI/source exports preserve this native 1000-fold variable/milliunit
disagreement. The physical static provider returns 2 ohms per square in all
three cases; it does not reproduce the native loader discrepancy. General
technology geometry, bias/interpolation and licensed linked-STF solving
remain open.

### Native output selection and primitive units

Native CSV scalar dependency intake (`PlanarSonnetScalarFiles`) preserves
literal-file `table1`/`table2` sources and an independently owned effective
project. Source/configuration identities cover detached effective component,
sweep and port records as well as original records. Bytes, files, numeric
nodes, line lengths and estimated storage are bounded; canonical paths stay
within the selected dependency root. Scalar consumers share staged snapshots
instead of rereading mutable files during physical evaluation.
Intake compares the original decoded records and diagnostic line positions
against captured parent bytes; stale parsed inputs reject before CSV reads.
Detached effective edits retain a distinct configuration identity. Linked
static-STF snapshots use their explicit captured SON/XML provenance, with
decoded technology identities checked against the retained XML. A bounded
scalar expression now rejects above 128 levels before parser/evaluator stack
overflow; the original 14,023-byte nested-expression failure is preserved.

Actual native controls in `test/fixtures/native_sonnet_scalar_tables` retain
97 hashed proof/source/output files plus their separate manifest. `table1`
endpoints and midpoint agree exactly with independent literal materials;
non-affine `table2` corners yield bilinear midpoint 7.5 and quarter 4.375 ohms
per square, with bit-identical native full matrices. An unescaped nested-quote
control evaluates native loss as zero and fails full-S by 0.499; it remains
archived. Faithful escaped-quote decoding fixes the previous lost filename
quotes. Out-of-table native runs warn and hold endpoints, while installed
help says extrapolation is unavailable. Default providers therefore reject
outside keys; explicit `outside=:hold` records the chosen endpoint policy.
This CSV evidence does not certify STF XML bias interpolation.

Actual logarithm controls in `test/fixtures/native_sonnet_scalar_logarithms`
retain 41 hashed source/proof/output files plus their manifest. Four native
coupons (`ln(exp(2))`, `ln(-exp(2))`, `log10(100)` and `log10(-100)`) give
2 ohms per square and bit-identical full matrices to the independently
declared finite-conductivity film at 1 and 10 GHz. Providers implement the
primary documented magnitude semantics with exactly one argument. The
previously accepted `log(exp(2))` returns 2 in the old reader, while native
Sonnet warns of an unknown function and substitutes zero, changing full S
by 0.0909–0.0918. The reader now rejects that invalid native syntax.

The complete installed Functions and Operators page is mapped in the
validation inventory. `native_sonnet_scalar_functions` and
`native_sonnet_hyperbolic_phase` retain 222 actual jobs and 340 raw R50
matrices in 1,361 SHA/CRC checked files plus two manifests. Independent
literal materials establish radian trigonometry under both DEG/RAD settings,
inverse/hyperbolic/general functions, complex intermediate arithmetic,
magnitude extractors and real-axis branch defaults. Native unparenthesized
powers associate left; explicit parentheses and a signed exponent preserve
their distinct semantics. The old wrong exponent response and all wrong
phase/branch hypotheses remain in their original reports.

Physical replay uses the unchanged full-S `0.005` gate, independent film
constitutive law and original source/current equations. A separately captured native SRES control stores
`Inner=cmplx(3,4)` as 3: `abs(Inner)` is bit-identical to literal resistance
3 and differs from literal 5. Current named-quantity storage follows this
finite real projection. Twelve further actual controls in
`native_sonnet_final_complex_material` retain 75 hashed files plus a manifest:
quoted/bare inline NOR `cmplx(3,4)`, `sqrt(-1)` and their quotient are
bit-identical to independently declared resistances 3, 0 and 4. Final
quantity evaluation preserves that projection and rejects nonfinite complex
expressions before storage. A further 73 actual jobs in
`native_sonnet_complex_binary_functions` retain 72 raw R50 matrices and
448 SHA/CRC checked files plus a manifest. Literal controls establish
complex `cmplx`/`hypot` operands, real-part `int`, signed-magnitude
`fmod`/`min`/`max`, second-operand ties and exact-zero phase resets. Original
wrong sign/ordering/remainder hypotheses remain intact. One complex `atan2`
pole produces an actual fatal IMSL error and has no response matrix.
Two further archives, `native_sonnet_complex_atan2_units_keys` and
`native_sonnet_complex_function_corrected`, retain 121 actual jobs,
119 raw R50 matrices and 904 SHA/CRC checked files plus two manifests.
Independent literal materials establish complex `atan2` quadrant branches,
its unusual zero-denominator value `pi/2 + im*real(y)`, finite real
projection for unit conversions and `table1` keys, and signed-magnitude
row/real-projected column keys for `table2`. Both-zero `atan2` logs NaN
and produces an empty high-precision export; this remains rejected along
with nonfinite poles. Original wrong conversion/degenerate/key hypotheses
and native fallback outputs remain archived. Corrected controls compare
every full-S entry to independently declared literal materials.
Native comparisons/conditionals and a positive signed exponent chain
warn and fall back to zero; those actual failures remain rejected. This
scope does not certify graph-only FORMAT or broader unresolved native
material and geometry models.

Native component and FLOAT consumers retain the same owned CSV source
contract in `scalar_files`, with numeric `scalar_overrides` recorded in
results. Automatic SPARAM staging counts CSV paths with SON, Touchstone and
STF dependencies and includes their identities in the effective configuration
hash. Scalar source limits and the aggregate geometry/model storage limit
are checked before optional component callbacks. Independent literal-load
and literal-reference controls compare full S, loaded node voltages and EM
current maps at the unchanged `1e-10` gate; retained snapshots keep their
response after CSV edits. Mixed cover/FLOAT lowering preserves its derived
geometry without reintroducing filtered components. These checks establish
raw physical loading and provenance; coupled pin calibration and licensed
native SPARAM/FLOAT acceptance remain separate.

For an explicitly materialized static linked technology, CSV staging retains
the original SON and STF bytes and hashes with a `static-stf` conversion
identity. Its four-way SON/STF/CSV/Touchstone dependency count is checked
before response providers. The 129 RF scalar/provenance checks include a
public linked-project staging/solve compared with independent literal
reference values; they do not extend the supported STF materialization subset.

A mixed FLOAT/ordinary-SPARAM result also retains the ordinary component's
`model_files`. A fresh audit showed the previous FLOAT wrapper discarded
those staged dependency snapshots while preserving only the numeric load.
The retained file storage is now reserved during bridge construction and
solve; independent callback wiring reproduces its full S, loaded node
voltages and EM currents at `1e-10`, and later file edits leave the recorded
bytes and hashes unchanged.

Native external Touchstone files are checked against the explicitly selected
raw or calibrated response log before comparisons consume them. Every full
matrix entry must lie within the independent magnitude/angle printing
intervals, and every frequency and evaluated wave reference must match.
Reference contracts come from the final native CKT `DEF` or geometry-port
R/X/L/C fields; sources without a derivable contract require explicit
`expected_z0`. Data is not renormalized by this guard. An ideal matched thru
with identical numerical S at R50 and R75 still rejects the wrong R75 basis;
per-port TERM and frequency-dependent FTERM mismatches also reject. OPTIONS,
the intended state, references, hashes and each accepted or rejected check
remain in metadata. These consistency checks do not replace physics,
convergence or solver-residual gates.

There are 68 durable guard checks. Fresh actual raw/calibrated controls pass
19 assertions, including rejection of the observed 18.53 `ND`/`OPTIONS -d`
selection mismatch. The calibrated calculation contains only a calibrated
response section, so its external ND data is compared against a separately
run raw sister coupon with identical physical geometry. A derived replay of
nine four-port D-Pack outputs and eight raw/calibrated SFC outputs passes all
17 contracts and 53 assertions while preserving the original archive hashes.
The deliberately unguarded output-selection falsifier remains separate.
Scripts are `validate_sonnet_output_guard.jl` and
`recheck_sonnet_native_outputs.jl` in `validation/sonnet_stripline`.

Native primitive resistors now honor `DIM RES OH`, `OHMS`, `KOH` and `MOH`,
with factors 1, 1, 1000 and 1000000. Omitted resistance units default to ohms;
unknown tokens reject. The final CKT `DEF` reference remains in fixed ohms,
as do the native port R/X fields. Four actual three-frequency series-resistor
controls agree with the independent 50 Ω matrix within `3.342e-14`; 36 durable
checks preserve source unit records and verify the imported circuits against
those native outputs. The raw sources, logs, outputs and provenance are in
`validation/sonnet_stripline/native_circuit_units_reference`. The fresh
combined native parser/reference suite passes 314 checks.

Native SMD `TYPE IDEAL` loads also honor the project's `DIM RES`, `CAP`
and `IND` units; native port R/X/L/C reference fields keep their separately
defined fixed units. A confirmed missing RES conversion previously treated
KOH 16.77 as 16.77 Ω. The corrected 16,770 Ω load reduces that raw coupon's
full complex S error from 0.8505 to 0.00165. Nine actual Sonnet Lite controls
independently prove OH 16.77 = KOH .01677, PF 1 = NF .001 and NH 1 = UH .001, and
distinguish the same literal values in the larger units. Exact sources,
raw R50 outputs, engine logs and selection guards are retained as 116
hashed files in `test/fixtures/native_ideal_component_units`, including the
original failure log. All 355 public unit/domain checks pass and retain the prior 0.03 raw
physical return fixture gate and the unchanged 1e-10 equivalent unit current,
node-voltage and response gates. Finite nonzero values must survive SI
conversion; invalid loads, frequencies and override values reject before
geometry and optional model callbacks. This raw fixture evidence does not
establish coupled pin-group calibration equivalence.

Component lowering now reserves geometry and staged model storage together
for IDEAL, NONE, supplied callbacks and automatic SPARAM files. A confirmed
20,000-byte acceptance retained a 20,480-byte material plane; the original
larger-grid rejection allocated 745,532 bytes before its error. Construction
preflight now rejects before geometry or callbacks, then uses alias-aware
live storage plus terminal-return workspace reservations. Cover-ground models
retain their numeric `payload` for subsequent solve reservations. The same
FIXED/FLOAT R/L/C conversions use the shared stored-domain checks, including
SI-Hz frequency expressions and nonzero wide-value preservation. The 213
resource/FLOAT checks retain the unchanged `1e-10` response, node and current
gates. Native FLOAT/GLG calibration remains unverified under Lite licensing.

The shared circuit network constructor also checks matrix dimensions before
copying a callback result or storing reference providers. The preserved
invalid 1024×1024 two-pin response used 16,964,254 warmed allocation bytes
before its late size error; the same public component reproduction now uses
186,967 bytes. The 27 transactional constructor/provider checks reject
without reading malformed matrix entries or evaluating local references.
Original failures, exact pre-fix sources and after observations are retained
as 19 hashed files in `test/fixtures/native_component_budget_float`.

## Scoped D-Pack artwork evidence

Standard straight/rounded D-Pack arrays now import through the public
ODB++ reader. An independent explicit pad enumeration checks the primary
side/gap/count identity across unequal rows/columns, units and radii.
The production geometry retains 72 bytes of field payload, independent
of pad count, with allocation-free scalar membership. A billion-by-billion
zero-gap array imports within an 8 KiB aggregate fixture budget.
The 55604 durable checks cover geometry, all ten orientations, transparent
gaps under clear polarity, resource rejection and representation boundaries.
Nine fresh actual Sonnet Lite runs independently declare six pad
protrusions and two buses, rather than generating native geometry from
the imported mask. The complete four-port matrices pass the unchanged
`0.06` error gate at 16/32/64 cells and 1/5/10 GHz, with maximum error
`6.0982e-5` and original voltage residual `1.0687e-10` below `1e-9`.
Raw projects, outputs, selected-log checks and source hashes are retained
in `validation/sonnet_stripline/dpack_reference` for durable replay.
This closes the symbol implementation and declared physical workflow;
vendor translator and independent convergence acceptance remain open.
Prototype and production logs
are `odb_dpack_prototype.log` and `odb_dpack_production_gate.log` in
`data/prior_worker_validation_20261004/validation/planar_audit`; the initial rounded-unit endpoint harness failure
is retained separately.

## Scoped ODB polygon strokes and reader resources

An independent public `symbol_resolver` U-aperture exposed a line-stroke
bug: convex-hull filling closed the aperture's notch. Exact segment/boundary
membership now retains concavity, reversed directions, either polygon
winding and zero-length paths. Independent parameter-interval clipping of
the original three rectangles verifies the swept shape. Typical membership
allocates zero bytes; 194738 focused geometry, resource and source-ownership
checks pass. The bounded composite user-symbol upgrade is described below.

The public imported coupon passes 39 fresh actual Sonnet Lite checks on
16/32/64-cell grids at 1/5/10 GHz. Native geometry consists of independently
declared rectangular conductor pieces. The largest complete complex-S
difference is `7.635576817052e-5`; original residuals are at most
`1.835000216829e-10`, under the unchanged `0.06` and `1e-9` gates.
The raw sources, logs, checked output basis and matrices are retained in
`validation/sonnet_stripline/odb_concave_stroke_reference`; 397 durable
checks re-solve against those exact archived outputs. A fresh repeat after
source snapshotting also passes. These are declared coupon checks; native
ODB translation, arbitrary vendor layouts and continuum convergence remain
open.

Unused built-in stencil definitions also bypassed lookup accounting:
2000 accepted definitions retained at least 1280000 numeric segment bytes
under a 207721-byte budget. Lookup entries/geometry, attribute/text tables,
ID metadata and emitted attributes now reserve their payload before
retention; callback-entry overhead reserves before invocation. Warmed
rejected-read allocation falls from about 18873025 bytes to 74183–74215
bytes, including the owned 68899-byte input snapshot. Source bytes
are bounded using the open descriptor and snapshotted before callbacks.
An appended caller-owned 2 MiB comment previously caused 4733824 cumulative
allocation bytes under a 4096-byte budget; subsequent file mutations now
leave the original snapshot unchanged, with 14256 measured bytes including
the callback's append IO. Measurements are scoped Julia
allocations and retained-payload estimates, not process RSS or arbitrary
callback guarantees. Original failures and current results are retained as
`odb_stroke_lookup_domain_*`, `odb_unused_symbols_*` and
`odb_input_growth_*` under `data/prior_worker_validation_20261004/validation/planar_audit`.

A further valid comb surface exposed allocation before contour rejection:
20002 segments retained at least 1280128 numeric bytes under a 595544-byte
cap, with about 24139000 cumulative allocation bytes. The reader now
counts contour slots in the owned UTF-8 snapshot, reserves their numeric
and union-tag storage before allocation, and transfers the exact-sized
buffer into the retained region. It no longer grows and copies that
geometry. Warmed rejection allocates 208423–208455 bytes, including the
197832-byte input snapshot. Separate contours keep distinct buffers;
Unicode whitespace, CRLF, analytic arcs, exact aggregate budgets and
failure paths are covered. Original and current evidence is
`odb_surface_retention_before.*` and `odb_surface_retention_final_after.*`.
These remain scoped allocation and payload checks, without a process RSS
or arbitrary callback allocation guarantee.

Exact axis-aligned polygon/region boundaries and even their nonmember
extensions previously allocated 1200 bytes per query through the general
high-precision orientation fallback. Shared boundary membership now checks
the segment bounds first and uses literal coordinate equality on aligned
edges; general edges keep the robust predicate. The declared boundary
probes allocate zero bytes, with independent rectangle and high-precision
general-edge parity checks. This removes a measured hot path in ordered
composite sweeps: the declared 400-by-400 nested-aperture raster falls
from 31121085 to 28781 cumulative allocation bytes, including the mask.
Before/after evidence is `artwork_axis_boundary_allocations_*`; initial
prototype geometry/allocation failures are retained separately.

### Bounded ordered composite user-symbol lines

The closed dyadic triangle tangent at `t=2/3` now passes exact event replay;
the immediately exterior control remains outside. Shared certified boundary
predicates and bounded exact fallbacks also close stored endpoint, ordered
clear/redraw, extreme-scale and affine clipping failures. The recorded 4738
boundary checks and original 189653 composite checks pass. The broader
original 532002 neighboring checks also pass after restoring the Region
literal-axis zero-allocation certificate. These proofs supplement the finite-clearance native
coupon below; specialized unsupported leaves remain implementation work.

`read_odb_features` and complete enclosing-product `read_odb` now sweep
ordered analytic apertures along finite lines. At each translation, clear
holes and redrawn islands follow the original aperture membership. A
query segment is partitioned at primitive boundary candidates, then the
original ordered geometry decides every interval and candidate. Sweeping
the outside and subtracting the swept hole would erase valid material;
the independent interval oracle checks the original local construction.

The supported leaf set is circles, polygons, capsule and circular-arc
strokes, polygon line strokes, analytic line/arc regions, ordinary rounded
or chamfered rectangles, butterflies, drill and null symbols, affine
transforms and nested composites. Nonround ODB arc symbols still reject,
as required by the primary format specification. Specialized periodic
thermals, half ovals, D-Pack and curved polygon-stroke leaves remain
separate implementation work; this is not arbitrary-symbol acceptance.

`max_stroke_boundaries=256` bounds composite aperture work, with a 64-level
nesting guard and checked bounds. Empty leaves and wrapper nodes count
toward its conservative work bound. Ordinary queries use fixed-size interval
storage; exceptional queries reserve bounded exact temporary workspace
before arbitrary-precision work. Separator work and transform depth are
bounded explicitly. Ordinary circle and
polygon lines retain their existing dedicated algorithms. The public
geometry/resource/transport checks include enclosing boards, archives,
compressed entities, clear image operations, cycles and work-limit errors.
Independent arc-distance and convex-support oracles check the additional
stroke leaves. Typical typed membership allocates zero bytes. The declared
400-by-400 hplate raster uses 92285 cumulative bytes including its mask,
under the unchanged 100000-byte gate. Three independent exact-event stress
processes each pass 27648 checks, with about 2.54 GB cumulative allocation;
this is not a peak-memory or zero-allocation claim for exceptional queries.
Point fallback uses a separate default workspace cap; the renderer's mask
payload cap does not promise an aggregate mask-plus-query budget.

Callback geometry is now reserved, depth-checked and snapshotted before
lookup retention. Mutating caller-owned polygon arrays or appending a
cycle no longer changes an accepted document's membership or cached bounds;
the same cyclic input rejects on a fresh read. This also covers flashes and
dedicated polygon strokes. A 10000-empty-leaf aperture previously bypassed
a work limit of one; it now rejects. Independent before/after evidence is
`artwork_sweep_independent_audit.*` and `artwork_sweep_independent_after.*`.

Scale stress exposed raw edge-determinant underflow/overflow: the original
new sweep mismatched 3 of 1763 probes at `1e-200` and 43 at each of `1e160`
and `1e200`. Normalizing the edge equation before its determinant gives
zero mismatches at all seven declared scales from `1e-200` to `1e200`.
The original and corrected evidence is `odb_composite_stroke_scale_*`;
fixed-decimal ODB inputs retain the primary numeric grammar.

Canonical affine membership uses the stored inverse transform. Bounds now
outward-enclose that same geometry, so an accepted point cannot be clipped
by an independently rounded forward box. Normalized and diagonal inversion
support representable large, small and anisotropic scales; zero or infinite
stored inverses reject. Arc radii retain the stored Float64 hypot convention
and literal endpoints are closed, without an epsilon dilation band.

The full enclosing-board coupon passes nine fresh actual native runs at
16/32/64 cells and 1/5/10 GHz, plus scale, canonical and final literal-axis
allocation source repeats. Maximum complete
complex-S difference is `6.728376042513382e-5`; maximum original voltage
residual is `1.408007846257492e-10`, under unchanged `0.06` and `1e-9` gates.
The native five-rectangle geometry is independently declared. Registered
`odb_composite_stroke_reference` preserves the byte-exact native sources,
logs, output guards, full matrices and earlier failed boundary fixture.
The final literal-axis repeat retains another thirteen byte-exact source
snapshots and passes all original native gates. The archive now has 639 hashed
files; interrupted validation attempts remain separately recorded. A fresh
combined package/docs checkpoint is still required for the latest source batch.
Its original exact-boundary mask failure remains recorded; the accepted
fixture has finite clearance without a geometry-tolerance change. Native
ODB translation, arbitrary layouts and continuum convergence remain open.

A separate public callback circle with a nonzero local center exposed
line/arc placement errors of 0.2 mm. Both paths now preserve that literal
center; clockwise and counterclockwise arc distance/bounds checks pass.
Evidence is `odb_offset_circle_before.*` and `odb_offset_circle_after.*`.

## Scoped conformal refinement evidence

The genuine trapezoid mesh at refinement level 6 has 16,384 triangles,
24,576 current unknowns and a 128×512 vertex lattice. The following runs
retain the same physical geometry and unchanged `1e-10` full voltage
residual gate. The bounded charge-graph preconditioner remains a validation
prototype; it changes the iterative solve, not the Galerkin operator.

| Source and solve | Modal counts | Iterations per port | Full voltage residuals | Elapsed time |
|---|---|---|---|---|
| Original four-sign coefficients, Jacobi | 512×512 | 10000 / 10000 | 9.762e-11 / 3.249e-11 | 9552.75 s |
| Exact two-sign coefficients, bounded charge graph | 512×512 | 1564 / 1572 | 9.956e-12 / 9.865e-12 | 1509.27 s |
| Exact two-sign coefficients, bounded charge graph | 512×1024 | 1363 / 1386 | 9.811e-12 / 9.944e-12 | 2187.30 s |

The original and two-sign 512×512 responses differ by `4.920e-12` in
complete complex S. The original retained operator payload is 836,539,272
bytes. A fresh two-sign constructor on the same geometry and modes stores
402,653,184 coefficient bytes and 433,886,088 total numerical operator
bytes, within its independently checked 600 MB constructor budget. The
original solve remains anchored to the earlier source copies and hashes
in `data/prior_worker_validation_20261004/validation/planar_audit/conformal_level6_original`; it does not certify
the later constructor. All three solves used a 1 GB aggregate numerical
payload budget. FFTW plans and Julia overhead are excluded from these
array-payload measurements; the timings are separate measured runs.

The level-5→6 spatial step is `0.01096849` in complete S; doubling the
level-6 transverse modal count changes S by `0.001018746`. The native
128→256→512 staircase mesh steps are `0.00955531` and `0.00926412`.
Neither spatial ladder is converged. The installed Lite engine rejects
its true conformal option and its 1024-cell reference exceeds the license
memory limit. These results are scoped numerical evidence, not complete
M1 or native conformal accuracy acceptance. Logs are
`conformal_uniform_level6_mode512.log`,
`conformal_two_sign_level6_mode512_constructor.log`,
`conformal_hodge_banded_level6_mode512.log` and
`conformal_hodge_banded_level6_mode512x1024.log` in
`data/prior_worker_validation_20261004/validation/planar_audit`.

## Bounded nonuniform original-equation solve

`solve_planar_conformal_defect` supports multiple interior sheets with
isotropic layers, PEC covers/sidewalls and wall sources. Its projected FFT
operator is an approximate preconditioner. Bounded defect corrections use
the original analytic modal action, streamed without dense matrices or
complete coefficient planes. Only the original voltage equation accepts
a result; final projected and original residuals are recorded separately.
Scalar, per-triangle and frequency-dependent `surface_zs` use an exact sparse
RWG Gram in both equations. Independent triangle integration and accepted
power checks cover resistive, reactive and spatially varying impedances;
dense Y/S, physical currents and radiation remain independent oracles.
The optimized single-interface path is preserved. The multilevel path
retains three projected FFT kernels per ordered interface pair, exact
cross-interface local entries, and bounded original modal kernel blocks.
Compensated vector and physical triangle-pulse contractions preserve
charge-free loops without projecting charge onto vector RWG points.

Validation on the pre-material source (projection `487a90e5`, solve `ecdc5482`)
on the genuine nonuniform trapezoid
with 78 triangles, 112 unknowns, 256×256 modes and a 128×128 projection
grid reached original residuals `3.493e-14 / 3.331e-14`, after two corrections
per port. The final projected residuals remained `1.240e-5 / 1.242e-5`.
The aggregate numeric payload bound was 12,248,000 bytes under a 128 MB
cap; construction took 10.85 s and solving 30.49 s, including 25.18 s
of original modal actions. A loaded four-layer complex-reference fixture
also agrees with the independently assembled original dense Y/S and
physical currents, preserves rotation and nonnegative accepted power, and
reuses retained references in current/radiation consumers. These tests
include resource/provider rejection and mirror/periodic neighbor falsifiers.

Original actions still cost O(triangles × modal terms × source columns),
and local construction has explicit pair-times-mode work limits. General
fast nonuniform acceleration, bulk/contact extensions and
native continuum convergence remain open. The production-source evidence
is `conformal_defect_production_gate.log` and
`conformal_defect_production_medium.log` in `data/prior_worker_validation_20261004/validation/planar_audit`, with
source hashes in the latter. Earlier prototype and level-6 storage results
retain their own source anchors and do not certify this constructor.

The multilevel production regressions cover two loaded interfaces and
three lossless interfaces, exact sheet Gram integration, passive and reactive
power, dense original action/Y/S/currents, rotation, provider/preflight
rejection, coefficient aliases and mirror/periodic neighbors. Warmed
multilevel matvecs allocate zero bytes. These scoped capabilities preserve
the original `1e-10` voltage residual gate; projected residuals alone cannot
accept results. The reusable workspace replaces the prototype's fixed
16 MB modal allowance. Source-bound production evidence is retained in
`conformal_multilevel_production_gate.log` in `data/prior_worker_validation_20261004/validation/planar_audit`.

Independent native coupling references use two shortened overlapping PEC
rectangles on a three-layer stack (εr 2.5/7.5/1, heights 0.3/0.2/0.5 mm).
A genuine nonuniform 690-triangle/1003-unknown mesh with 128×128 original
modes meets the unchanged `0.005` complete complex-S gate against the
native 32-cell raw reference at 5 and 10 GHz: errors `0.003331/0.003980`,
original residuals at most `9.54e-12`. These are validation-prototype
results and retain that source anchor; their memory/timing does not certify
the production constructor. Native 32→64→128 refinement remains
unconverged, with complete-S steps `0.04989/0.04689` at 5 GHz and
`0.05912/0.05464` at 10 GHz. The finer genuine mesh was independently
rejected at the configured 200,000-pair cap. No native continuum or general
fast acceleration acceptance follows from these coarse comparisons.
Actual engine outputs and source hashes are retained in
`data/sonnet_validation/conformal_multilevel_native_S8MlTz/native.toml`;
the independent runner and rejected attempts retain their own source hashes.

## Active validation failures

Frequency-dependent complex references now use Kurokawa power waves across
EM, circuit, project, current and radiation workflows. The generic reference
model supports series C; native Sonnet ports instead place C in parallel
with the R+jX+jωL branch and use fixed Ω/Ω/nH/pF field units. Twelve actual
GUI Graph normalization comparisons at 1, 5 and 10 GHz agree within
`6.60e-12` in complete complex S. The installed Lite engine refuses
non-50 Ω electromagnetic references, so native complex-reference EM
accuracy remains unverified. Durable native exports and provenance are in
`validation/sonnet_stripline/native_reference`; failed engine attempts are
retained in `complex_reference_oKpoAC`.

Physical paired FLOAT sources, uniform floating sister geometries and
full coupled result calibration are implemented with charge, resistance,
power, current and field regressions. Native CUP/component FLOAT lowering
retains finite bridges and signed circuit incidence. Native GLG accuracy
remains unverified: the installed Lite engine rejects calibration groups
and ports-only components. CELL/custom widths and arbitrary local-ground
geometry require explicit resolved contracts.

The emitted SPICE models have also been evaluated by actual ngspice 47.
Two declared two-port models, with real and conjugate poles and affine
capacitance, pass the unchanged `1e-9` complete admittance-matrix gate at
37 frequencies from 1 MHz to 30 GHz. Grounded and complex common-mode
floating references give a maximum relative matrix error of `1.025e-15`.
The last formal pin is `dm_ref`; private auxiliary states use an internal
zero reference while every external voltage and current is relative to
that formal pin. Pure common-mode excitation produces zero port current.
The four raw engine runs, generated subcircuits, source hashes and logs
are retained in `validation/spice_reference/ngspice47`; 753 durable checks
read those outputs independently. This accepts these linear exported
models. Arbitrary vendor or nonlinear models and a complete measured RFIC
ladder remain open.

* **Independent fabrication coupon:** plain dark/clear Gerber regions,
  legacy MI/SF/OF/IR Gerber coordinates and a complete minimal ODB++ board
  job each lower and solve the same declared PEC strip with a central clear
  window. Every imported cell agrees with independent analytic membership.
  The native GEO uses four separately hand-authored physical rectangles;
  it is never generated from imported objects or masks. All 27 comparisons
  at 16/32/64 actual cells and 1/5/10 GHz pass the unchanged 0.06 full complex
  S gate, with maximum error `0.00011889` and recomputed voltage residuals
  below `1e-9`. Native 10 GHz grid steps are `0.01471` and `0.01346`, with
  matching DiffMoM steps; this is agreement at the declared grids, not a
  continuum convergence certificate. This accepts the declared coupon
  workflow, while wider
  vendor fabrication geometry/material/technology acceptance remains open.
  Reproduction: `validation/sonnet_stripline/validate_sonnet_artwork_coupon.jl`;
  actual projects, full matrices, module/job-file hashes and grid steps:
  `artwork_native_3hcgXi` (fresh provenance-complete repeat of
  `artwork_native_yoK3Zq`).
  The initial residual-reporting harness failure is retained separately in
  `artwork_native_jUPlRS` and supplies no acceptance credit.
* **Translated DXF:** a live GUI independently confirmed that native BOX
  counts are half-cell counts. Earlier approximately 0.99 errors compared
  different actual grids and are diagnosed import failures, not matched-grid
  physics evidence. At matching 96×64 actual cells, all sheet/via masks agree;
  native subsection count is 908 and DiffMoM has 2759 unknowns. At 5 GHz,
  128→256 modal refinement reduces maximum raw complex S error from 0.10994
  to 0.08001, still above the unchanged 0.06 gate. Independent line-standard
  calibration gives 0.13175→0.10422 and does not close the discrepancy.
  At 192×128 cells, native has 2074 subsections and DiffMoM has 12577 unknowns;
  the FFT solve fails the strict `1e-9` residual gate after 10000 iterations
  with restart 300. Restart 1200 converges in 2977 iterations per port with
  recomputed residuals approximately `1e-10`, a 2 GB payload budget, and 1014 s
  elapsed time. Raw/calibrated complex S errors remain 0.07730/0.07872.
  Exact FFT retained dense solves extend the fixed-grid modal ladder:
  512→768→1024→1536 reduces raw error to 0.07350→0.06826→0.06669→0.06571,
  which still fails. The former dense FFT route refused mode 2048 because its
  matrix and iterative-only modal workspace exceeded the declared 4 GB budget.
  A compact retained-assembly workspace now closes that resource gate under
  the same 4 GB limit: mode 2048 completes in 47.78 s with independent original
  FFT-action residuals 1.765e-12/1.775e-12. Raw complex S error remains
  0.06533993 above 0.06, so physical acceptance remains **FAIL**. The replay
  uses 139 unchanged copied source/configuration files and retained actual
  native raw-log matrices (`dxf_compact_fft_hMG0qJ`); it is not a fresh engine
  run or a continuum certificate. Bounded dense modal blocks now permit mode
  4096 under the same 4 GB raw-payload limit, with original unfactorized-matrix
  residuals below 2.15e-12. The current 2048/4096 raw complex S errors are
  0.06533993/0.06492637: both still **FAIL** the unchanged 0.06 gate.
  Their cumulative Julia solve allocations, including returned results, are
  approximately 2.818/2.782 GB; these do not establish peak process memory.
  The first public matrix-free folding implementation folds bounded modal blocks into
  family-pair spectra when they require less storage than retained kernels
  and mode-by-element work arrays. On the same 4096-mode DXF case it fits
  the unchanged 4 GB payload budget, retaining 240037168 bytes of Julia
  arrays. Its action matches the full finite-modal matrix within 1.66e-15,
  its diagonal matches exactly, and warmed matvecs allocate zero bytes.
  This resolves the former 8060904784-byte preflight rejection, while the
  accuracy and fine-grid iterative convergence checks remain open.
  A fresh public solve at this same 4096-mode grid converges with restart
  1200 and maxiter5000 in 3354 iterations per port. Recomputed original
  residuals are 9.986e-11/9.986e-11, and its full complex S differs from the
  retained dense reference by 2.504e-9. Native raw error is 0.06492637,
  still **FAIL** at 0.06. This establishes convergence for these declared
  settings; default restart choices and broader spatial convergence remain
  separate checks. Evidence: `dxf_folded_public_solve_3zDFGO`, anchored to
  source `a028459b`, under local `data/planar_audit`.
  Evidence: `dxf_folded_iterative_operator_audit_ZRgNgt` under local
  `data/planar_audit`, anchored to source `a028459b`.
  Source images combine the four signed source terms into one spectrum
  per family pair for box-wall halves. Translated interior terminal halves
  retain two separate cosine/sine source channels, including families that
  contain both wall and interior halves; both channels are reserved before
  allocation. The `07aa67c4` 4096-mode operator retains 70167892 bytes of Julia arrays;
  its spectrum payload falls from 226492416 to 56623104 bytes. Against the
  full finite-modal matrix, action and diagonal relative errors are
  1.76e-15 and 1.61e-16, with zero warmed matvec allocation. Evidence:
  `dxf_source_images_operator_audit_OExCoI`. A preceding paired prototype
  measured median actions of 0.0537792 versus 0.016238 seconds across
  15 samples (`dxf_source_images_prototype_mwnZLo`); this is a scoped timing
  measurement. Independent modal equations cover low and aliased modes,
  PEC/PMC walls, every wall half, sheets/vias/volumes and coupled loss.
  Its complete public solve also converges in 3354 iterations per port
  under the same 4 GB budget, restart1200 and maxiter5000. Original
  residuals are below 9.987e-11, and full complex S differs from the dense
  reference by 2.5044e-9. Native error remains 0.06492637, **FAIL** at 0.06.
  Evidence: `dxf_source_images_public_solve_zpuQlw`; this run began with
  uncommitted edits. Before the correction, its source content was checked
  against `07aa67c4` with Git's text line-ending normalization; its captured
  raw-byte hashes remain recorded separately.
  After the translated-terminal correction, a fresh 4096-mode DXF action
  and diagonal replay retains 70167952 bytes, with relative errors
  1.76e-15/1.61e-16 and zero warmed allocation
  (`dxf_translated_source_images_operator_audit_nbpnaQ`). The additional
  descriptor fields explain the 60-byte increase; the DXF spectrum
  payload is unchanged. This replay compares the full finite matrix and
  does not replace the source-specific native accuracy result above.
  Public dense assembly retains its original bit-identical summation order;
  materializing an iterative image operator changes floating-point grouping
  and uses the same 2e-12 independent equation gate.
  Array payloads exclude opaque FFTW plans and process
  overhead. Folding increases construction work; subsequent actions use
  grid-sized convolutions. Hybrid cross reactions continue using individual
  modal fields, so this storage reduction applies to the uniform-grid
  operator rather than the hybrid cross kernels.
  Exact current source hashes and retained native inputs are recorded in
  `dxf_folded_dense_bounds_modal_ladder_xU521C`. Replacing all shapes by exact
  occupied-cell rectangles gives the identical native response; partial
  polygon rastering does not explain this discrepancy. Uniform-only via
  profiles give 0.06466 at mode 1536, also fail, and are kept as a diagnostic
  rather than changing the importer. Evidence: `layout_15fGqE`,
  `layout_cs7xRq`, `cell_geometry_vXCwHi`, `via_profile_SolY1C`,
  `layout_LuVtPU`, `layout_ZDXjQs`, `layout_FNzUCX`, and the retained GUI screenshot
  `layout_native_q7Tfgp/sonnet_box.png` under local `data/sonnet_validation`.
* **Fine stripline convergence:** 128 width cells, 32 length cells and modal
  factor 2 give `Zc=50.044527081 Ω`, `v/c=1.000001385`, and absolute
  `eT=0.089193%`. The 256-cell solve with GMRES restart 100 fails the
  unchanged `1e-9` recomputed residual gate after 5000 iterations. More
  restart space 500 subsequently closes the 256-cell solve at unchanged
  residual tolerance: `Zc=50.012496197 Ω`, `v/c=1.000001388`, and
  `eT=0.025131%`, with 16608 unknowns. Independent modal factor 4 at 128 cells
  gives `Zc=50.048010646 Ω`, `v/c=1.000004035`, and `eT=0.096425%`,
  also passes the absolute gate. Independent length refinement to 64 cells
  at 128 width cells gives `eT=0.118641%`, above the 0.1% absolute gate.
  Increasing width resolution to 256 at that length closes this gate:
  `Zc=50.027248023 Ω`, `v/c=0.999999364`, `eT=0.054560%`, with 32960
  unknowns, restart 500 and recomputed residual below `1e-10`.
  The same 64×256 grid at modal factor 4 also passes: `Zc=50.028847258 Ω`,
  `v/c=0.999999719`, `eT=0.057723%`, recomputed residual `9.993e-11`,
  3697 iterations and 4572 s. The factor 2→4 absolute-error change is
  0.003163 percentage points; this is a modal sensitivity check, not a
  remaining continuum error bound. Evidence: `stripline_accuracy_nx64_m4_memory500.toml`.
  Absolute error is not monotone across independent grid axes. Final
  mode/grid sensitivity must still be included in the certificate.
* **Calibrated local terminals:** physical box-cover returns and corrected
  actual-cell decoding reduce native IDEAL SMD maximum complex S errors to
  0.00669, 0.01000 and 0.01329 at 200, 300 and 400 MHz. This passes the
  predeclared coarse 0.06 gate on the unchanged installed geometry; automatic
  native pin lead/local-ground calibration remains unverified. Evidence:
  `mixed_metal_FgLQuv`; the original disconnected-terminal failures remain.
* **Licensed thick metal:** the physical two-face importer passes independent
  up/down geometry, material and contraction regressions. Actual native TMM
  runs reject with “Sonnet Lite does not allow thick metal polygons.” These
  runs remain UNVERIFIED; a published approximation test does not replace
  acceptance against the licensed native feature.
* **Native sloped geometry reference:** the actual trapezoid refinement uses
  Staircase meshing, confirmed in the live polygon properties: edge mesh is
  enabled, both current directions are included, and minimum/maximum
  subsection dimensions are 1/100 cells. Native 128, 256 and 512 actual-cell
  runs have 1004, 1858 and 3532 subsections; complete complex S steps remain
  0.009555 and 0.009264. The reference has not converged. Native 1024 cells
  require 180 MB and fail Lite's 64 MB limit; the 512-cell run already uses
  63 MB total. Selecting native Conformal meshing produces the literal GUI
  error “Conformal exceeds your Sonnet Lite license.” Genuine triangle
  backend checks and Staircase comparisons therefore do not certify the
  licensed native Conformal feature. Evidence: `trapezoid_refinement_JD0CIp`,
  `trapezoid_refinement_JvaUtY`, and `trapezoid_gui_conformal_20261003`.
* **Native current CSV:** fresh offset X/Y strips expose a native 18.53
  export convention: Y headers reflect the `.son` coordinates and raw
  complex current phasors reverse the physical generator current. An
  explicitly selected validation adapter preserves the raw failure and
  matches the driven X and rotated Y components within 0.792% complex L2.
  Their tiny transverse components still fail the original component gate.
  On the PEC Ring fixture, corrected X current passes at 2.428% L2, while
  Y current fails at 9.360% L2 and 15.815% peak error against unchanged
  5%/10% gates. A passing S matrix does not establish complete current-map
  accuracy. Evidence: `current_axes_C0cilD`, `current_csv_aZ0wVa`,
  `current_csv_rTpYSD`, `current_csv_CrBRWx`, and `current_csv_u17Xzw`.
* **Native antenna and viewer:** the unchanged installed RHP two-feed geometry
  has 426 native subsections versus 428 DiffMoM bases at 256×256 actual cells.
  Mode 256 fails (complex S error 0.16859); 512, 1024, 1536 and 2048 give
  0.04071, 0.000684, 0.008940 and 0.011874. The last full-matrix modal step is
  0.002935 and both last modes pass the unchanged 0.06 gate. Native generated
  fresh current data. The live GUI shows View Current enabled and Plot Far
  Field Pattern disabled for that exact project; its installed manual states
  that the viewer requires a separately purchased license. This is S/current
  evidence, with no native pattern claim. Evidence: `antenna_pv8YUr`,
  `antenna_hgKyrM`, and `farfield_Lite18_53_20261003/availability.json`.
* **Native via loss scope:** VOL RPV defines a constant total axial resistance,
  split by physical layer height; ARR RPV additionally uses the original
  via density and polygon area. Independent mesh/height regressions and two
  actual 5-ohm bridge fixtures pass (complex S error 0.00006254).
  Two distinct 2-ohm/5-ohm models sharing both dielectric layers and driven
  by axial ports also pass at 1, 5 and 10 GHz (errors 0.000200, 0.000997 and
  0.001914), verifying per-material level grouping and height contraction.
  These RPV controls certify constant loss. Scoped VOL conductivity/skin
  lowering is documented separately below; general ARR and thick
  nonrectangular wall semantics remain open; scoped VOL endpoint pads are
  described below. Evidence:
  `rfic_xLa8AB`, `multilevel_rpv_SnyNQq`, and `import_PkFr4U`.
* **Earlier full-volume conductivity candidate:** the diagnostic candidate uses physical finite-
  conductivity Maxwell currents over the full 400 µm square cross section,
  rather than substituting constant RPV. With conductivity 6250 S/m and
  height 500 µm, the actual native wall-source fixture approaches the 0.5 Ω
  DC resistance. At 10 GHz, however, native real common-mode impedance is
  1.71368 Ω, while the three-component volume candidate gives 1.15449,
  1.15588 and 1.15591 Ω with 1, 4 and 8 axial slices. Recomputed voltage
  residuals are below `7e-13`; refinement preserves a 32.55% loss mismatch.
  The full complex S errors of 0.04349–0.04362 pass the 0.06 S gate but the
  unchanged 5% real-impedance gate fails, so these comparisons are **FAIL**.
  Increasing conductivity to 625000 S/m at 1 GHz likewise leaves roughly
  40% loss error. The native volume editor identifies its last SOLID field
  as an inactive wall thickness; changing it 1000-fold gives bit-identical
  native responses. No fitted skin formula or general VOL/ARR adapter credit
  is inferred. Evidence: `volume_skin_XykaO6`, `volume_skin_LrybRO`,
  `volume_skin_69G7fZ`, and `volume_fields_CAZRHW`. Strict iterative failures
  in `volume_skin_aR8Fb9` and earlier harness failures remain archived.
  Fixed-area aspect tests at 10 GHz also fail the loss gate: native/candidate
  real impedances are 0.86262/0.99882 Ω for 200×800 µm and 0.99150/1.33433 Ω
  for 800×200 µm (`volume_skin_ucMuZ6`). The high precision PEC controls pass
  the original `1e-8` real-impedance gate and full S gate for all three shapes
  (`volume_skin_VMzQQi`), distinguishing the unresolved loss law from the
  earlier rounded-log artifact. A fine 14240-unknown attempt rejects its
  6.49 GB requirement before allocation under the 4 GB cap.
  Actual Ring-footprint PEC controls agree within 0.00010–0.00034 in full S
  with axial currents; adding horizontal Ring currents gives 0.00203–0.00642.
  Applying only DC-area conductivity scaling to the Ring does not resolve RF
  loss: the square gives 0.59861/0.59872 Ω at 1/4 slices versus native
  1.71372 Ω, and both the original S and loss gates fail. These diagnostic
  branches preserve the genuine full-volume candidate and assert no native
  material equivalence (`volume_skin_hsIVRQ`, `volume_skin_yUVwVv`).
* **Native surface-via diagnostic:** the actual GUI saves the Surface model
  as `SFC RDC XDC RRF`; the installed Lite engine accepts it. A nonzero
  field-order GUI probe independently confirms this differs from planar
  `SUP RDC RRF XDC LS`; both RF coefficients use Hz. The documented
  infinitesimal tube resistance `Rs*h/perimeter` is not an accepted native
  adapter: the 500 µm tall, 400 µm square specimen with `RDC=1 Ω/square`
  gives native wall-source real impedance `0.075911 Ω` at 100 MHz, versus
  `0.312537 Ω` from the declared Ring axial candidate. An independently
  specified native `RPV=0.3125 Ω` instead agrees with that candidate.
  Zero-resistance SFC and PEC controls also have different reactances and
  native subsection counts (124 sheet plus 16 via unknowns for SFC versus
  290 plus 36 for PEC on the same 40-cell specimen), so the mismatch cannot
  yet be assigned solely to a local material law. The actual native
  1 MHz resistance with `RDC=1` falls from 0.075605 to 0.042583 to
  0.022221 Ω on 40/80/160-cell grids, rather than approaching the documented
  0.3125 Ω tube resistance. Lossless SFC and PEC reactances remain separated
  throughout this ladder. Fresh surface-current and subsection views also
  show different native sheet coalescing. The Lite engine rejects explicit
  polygon subsection controls, and the GUI rejects Advanced Subsectioning
  with a license error; identical forced native subdivisions remain
  unavailable; Lite also rejects Full, Vertices and Center via fills.
  Independently authored interlevel series bridges give 0.312500007 Ω at
  1 MHz on all three grids. A common-voltage grounded strip whose via
  instead ends on a numeric full-box PEC sheet also gives 0.312500004 Ω
  on 40/80-cell grids. The remaining cover-target branch is therefore a
  topology-specific issue, rather than evidence for a general scalar
  material correction. MM/UM/MIL controls agree within `1.64e-12` in S.
  A separate physical tube candidate retains side/corner wall lengths
  and passes the declared 0.005 complex S and 5% loss gates at 1 GHz,
  with errors 0.002028/0.002080 at 40/80 cells. Its 100 MHz solves miss
  the unchanged `1e-9` voltage-residual gate. At 10 GHz, S errors
  0.019821/0.020438 fail; the zero-loss control already gives
  0.019547/0.020326. Fixed-grid modes 160→320→480→640 leave the finite-loss
  S error at 0.019575, so modal truncation does not close this gap.
  Uniform-only axial controls instead match the lossless interlevel native
  response within 0.000095 in full complex S at 1/10 GHz on both grids.
  Finite-resistance uniform-only S errors are below 0.000530, but the
  40-cell 10 GHz resistance still fails the unchanged 5% loss gate
  (6.52% error); the 80-cell error is 3.38%. The 100 MHz voltage-residual
  gate remains unmet. Uniform-only grounded-cover controls still fail S.
  These observations identify a scoped axial-profile difference, without
  proving Sonnet's general profile or ground-contact contract. An earlier
  nonzero RF/reactance run emitted the wrong SFC field order; its failure
  is preserved as a diagnosed fixture error and earns no physics evidence.
  Corrected 80-cell interlevel controls with nonzero RF resistance and
  reactance pass at 1/10 GHz; a stronger RF coefficient also passes with
  resistance error below 0.005%. Four further aspect/height specimens
  (400×200 µm, 200×400 µm, and 400 µm square with 100/1000 µm height)
  pass both frequencies, with worst S error 0.000590 and resistance error
  4.47%. Pure-reactance controls pass at 1/10 GHz using the independently
  checked lossless complex-conductivity domain. Their 1/100 MHz solves
  remain unverified: residuals 2.63e-7/1.36e-9 exceed the original gate,
  and the 1 MHz spurious resistance also exceeds the lossless bound.
  Actual native pure-reactance controls reproduce the same cover-target
  grid dependence as RDC, while interlevel material reactance agrees with
  `Xdc*h/perimeter`. These passes validate the explicitly declared uniform
  tube candidate on those specimens; they do not promote a general native
  Surface adapter.
  The finer 160-cell candidate has 28,700 unknowns and remains unverified:
  12,000 iterations took 2327 seconds under a 2 GB numeric payload cap,
  with physical voltage residual 3.33e-9 above the unchanged 1e-9 gate.
  Actual native 80/160/320/640 raw full-S steps remain
  0.010518/0.009822/0.009757. Separate engine jobs with native de-embedding
  reduce those steps to 0.001279/0.000649/0.000328, identifying a launch
  contribution without certifying continuum agreement of the candidate.
  The original simultaneous ND/D output attempt wrote calibrated numbers
  to both external files despite distinct internal raw/calibrated SID data;
  that export inconsistency and its invalid raw labeling remain preserved.
  Corrected evidence uses separate raw and calibrated jobs.
  Source/mesh interpretation remains under investigation; no SFC or
  general VOL skin-law fidelity is claimed.
  Evidence: `via_surface_rdc_73s3SX`, `surface_native_controls_OcdGtB`,
  `surface_native_refinement_AWSD2s`, `surface_current_controls_EYl5Dz`,
  `surface_subsection_controls_3o1Jy6`,
  `surface_subsection_frequency_gui_20261003`, and the saved-model GUI
  screenshot in `via_surface_model_VxDwee`. Topology and physical candidate
  controls: `surface_series_bridge_yHCYeA`, `surface_series_bridge_EZdGc9`,
  `surface_series_bridge_Rd3lZ7`, `surface_length_units_4G2yCw`,
  `surface_physical_bridge_8exhBl`, `surface_physical_bridge_b9kEVr`,
  `surface_physical_bridge_BfGhND`, `surface_physical_bridge_HuyXxo`,
  `surface_physical_bridge_mBBxdI`, `surface_physical_bridge_bQZGCG`,
  `surface_physical_bridge_kwXkfN`, `surface_physical_bridge_szpnnP`,
  `surface_physical_bridge_cDgBd6`, `surface_physical_bridge_Jrt9t2`,
  `surface_calibration_ladder_Qzn5ae`, `surface_calibration_ladder_gqRZqd`,
  and `surface_field_order_20261004`. The diagnosed field-order failure remains
  in `surface_physical_bridge_McTbg2`. Fresh corrected and extended proofs:
  `surface_physical_bridge_KxYrcq`, `surface_physical_bridge_hAWKlB`,
  `surface_physical_bridge_iae13d`, `surface_physical_bridge_4aP1ka`,
  `surface_physical_bridge_gLlE7L`, `surface_physical_bridge_phiBUL`,
  `surface_physical_bridge_mnH3pI`, and `surface_reactive_controls_Nhf9O7`.
* **Negative native port labels:** an asymmetric positive/negative wall
  coupon exposed the fixed signed-voltage contraction: complex S errors
  0.02435, 0.12147 and 0.24104 at 1, 5 and 10 GHz. Joint elimination of the
  floating common potential enforces equal/opposite summed currents and
  reduces these errors to 0.00001559, 0.00004595 and 0.00010538. Production
  and independently derived voltage transfers agree, with current balance
  residual below `1.2e-16`. Coupled multi-label and height-weighted terminal
  regressions preserve this contract. Evidence: `balanced_wall_xgdJvT` and
  the fresh corrected production run `balanced_wall_0VjVAt`.

## Confirmed bug ledger

| Candidate / dimension | Verification | Fix and recheck |
|---|---|---|
| Retained FFT assembly keeps unused iterative buffers / measured resource use | multilayer 273-unknown fixture retains four mode-by-element arrays and an unused output vector totaling 3936528 B; warmed assembly allocates 15281904 B | separate compact assembly workspace preserves bit-identical finite-modal matrix and complete public iterative operator; warmed allocation 11344885 B (25.76% reduction), 42 new + 116 neighboring + 112 independent checks; raw payload budget still excludes opaque FFTW and process overhead |
| Dense FFT retains every modal layer-pair kernel / measured resource use | 273-unknown mixed fixture allocates 11.344 MB and rejects an 8 MB payload limit; DXF mode4096 requires a 4.839 GB FFT workspace before its dense matrix | bounded blocks fold each mode in its original order into smaller family-pair lattices; mixed PEC/PMC matrices match the frozen source byte for byte, allocations fall to 3.695 MB and the 8 MB gate passes; DXF mode4096 fits the original 4 GB limit, while its native accuracy gate remains FAIL |
| Matrix-free FFT retains high-mode kernels and four mode-by-element buffers / measured resource use | DXF mode4096 rejects the 4 GB limit because its estimate is 8.061 GB | choose smaller family-pair spectra plus reusable grid fields; the same public operator fits the unchanged budget and retains 240 MB of Julia arrays, with full-matrix action error 1.66e-15 and zero warmed matvec allocation; independent mixed PEC/PMC modal matrices, all wall halves, local/coupled losses and port solves gate correctness; peak process memory and DXF physical acceptance remain unverified |
| Folded iterative FFT stores four signed spectra per family pair / measured resource use | preceding 4096-mode DXF operator retains 240037168 bytes, including 226492416 spectrum bytes | reflected source images reduce spectrum storage to one quarter and total retained arrays to 70167892 bytes; full finite-matrix action/diagonal errors 1.76e-15/1.61e-16, zero warmed allocation, independent low/aliased-mode equations and exact preflight boundary tests; public dense summation order is unchanged |
| Uniform FFT accepts vectors borrowed from its own scratch arrays / correctness | input aliases of output/field buffers silently change the equation; an output alias with nonzero beta loses its initial values, producing relative errors up to 1.0 | reject owned workspace aliases before mutation for retained and folded operators; ordinary input/output aliasing and strided views remain supported, with zero warmed allocation |
| Source-image folding combines translated half-rooftop coefficients as if on a box wall / correctness | full package tests catch six failed action, diagonal and port-response checks for interior terminals in `07aa67c4` | preserve separate cosine/sine channels for translated halves, including mixed wall/interior families; independent original equations, materialized matrices, currents, complete port responses and preflight limits gate the correction |
| Folded accumulation checks owned indices and public results erase FFT plan types / measured CPU and allocations | paired mixed PEC/PMC prototypes spend 17–32% less time with the owned accumulation ranges unchecked; real physical result matvec calls allocate48 bytes despite zero allocation through concrete operators | remove checks only inside the constructor-sized accumulation loop and retain the concrete operator type in `PlanarUFFTResult`; original vector/input indexing and preflight guards remain checked; outputs stay bit-identical in scoped prototypes, and public result/legacy-constructor matvec allocation is gated at zero; timings are workload-specific and do not certify global optimization |
| Finite wide circuit scalars lose stored invariants / numerical and state correctness | finite BigFloat R/L become infinity; positive C and nonzero transformer ratios become zero in owned elements | validate stored Float64 values before mutation; invalid inputs leave elements/nodes/ports unchanged, explicit zero and representable wide values remain supported |
| Circuit frequency overflow/underflow after callbacks / public workflow correctness | pure network input frequencies `1e1000`/`1e-1000` return result frequencies infinity/zero after two callbacks | preflight the stored frequency before providers; invalid frequencies invoke zero callbacks, valid wide inputs use finite Float64 provider/result frequencies |
| Attenuating circuit line loses reciprocity / numerical and physical correctness | matched `40+0.37im` line gives reverse transmission magnitude 4 instead of `exp(-40)`; attenuation 710 rejects finite S | bounded travelling-wave MNA, stored complex-domain validation and no ABCD temporary; 908 independent high-precision/domain/allocation checks |
| Internal circuit network workspace omitted / resources | 64 local ports and two external ports pass 300000 bytes and invoke three providers; eight 64×64 complex planes allocated | reserve largest sequential network snapshot before providers; direct S/Y/Z stamps and typed scattering loop reduce planes to two and warmed cumulative allocation to 423784 bytes |
| Component geometry/model reservations applied only to SPARAM / resources | IDEAL accepts a 20000-byte cap while a material plane alone retains 20480 bytes; a larger grid allocates 745532 bytes before rejection | aggregate all-branch construction preflight, alias-aware live geometry/model reservation and retained model payload; IDEAL/NONE/callback/SPARAM reject before callbacks and preserve physical projections |
| FLOAT IDEAL units and wide overrides lose physical values / import and stored-domain correctness | KOH 10 becomes 10 Ω; KOH .01 becomes .01 Ω; a direct nonzero wide override becomes an exact short | share native DIM R/C/L conversion and stored scalar/frequency/override guards before geometry; constant and SI-Hz provider controls agree in complete S, node voltages and current maps |
| Oversized circuit network copied before shape rejection / resources and transactional state | a preexisting 1024×1024 two-pin callback result allocates 16964254 bytes before its size error | shared constructor rejects dimensions before conversion and reference storage; the same component reproduction allocates 186967 bytes, malformed constant/provider regressions read zero matrix entries |
| Loss sign and crossed vector Gram / physics | passive native metal and independent overlap integrals | negative impedance reaction sign; zero orthogonal sheet Gram; passive native mixed films and quadrature regressions |
| Via modal sign / physics | independent Maxwell via BVP and native ground/interlevel structures | corrected coupling signs; BVP, reciprocity and native RFIC regressions |
| Exact axial cutoff singularity / numerical robustness | nonresonant Maxwell BVP and approach from both sides | current-based analytic limits, inverse-transfer workspace; true resonances still reject |
| Premature constant-response ABS termination / workflow completeness | zero and constant full-matrix callbacks | complete fit validation and finite-data guards; adaptive/dense regressions |
| Public Real sweep band reaches Float64-only fitting after callbacks / input and resource correctness | representable BigFloat bands fail after three callbacks; overflow/underflow bands and collapsed candidate points reach analysis | validate the stored Float64 domain and candidate grid before callbacks; representable BigFloat bands solve, invalid bands invoke zero callbacks |
| Finite wide complex sweep responses become infinite after storage conversion / numerical robustness | finite Complex{BigFloat} values convert to infinite ComplexF64 samples | validate copied samples after conversion and reject at the first invalid response; no nonfinite sample is retained |
| Gradient memory rejection after objective callbacks / resources | 1000-parameter 20 kB reproduction invoked objective 17 times before error | aggregate preflight and forward-budget reservation; same reproduction invokes objective zero times |
| Cutoff/workspace allocations / performance | blocked fill allocation profile | reused cascade storage; measured small fill 2.82 MB→18.4 kB |
| Unsafe wall half index / memory correctness | one-row port connectivity reproduction | map stored boundary edge to occupied cell; narrow wall regression |
| Exact slab face impedance used as current-jump impedance / physics | independently derived boundary equations and published two-sheet formulation | half-thickness sheet model; independent line resistance acceptance |
| Open terminal without physical return / end-to-end physics | native SMD response approximately disconnected despite exact spectra | physical reference-via sources and contraction; DC and native three-point gates |
| Native via meshing flags discarded / import fidelity | installed DXF via masks differed in two cells | preserve SOLID/RING/CENTER/VERTICES; independent masks agree exactly; response discrepancy remains open |
| Native BOX half-cell counts interpreted as cells / import fidelity | live Sonnet GUI: BOX 96×64 displays 48×32 cells and 0.009375 in spacing | divide stored counts by two; doubled fixture writers; matched native stripline, six RFIC primitives and nine mixed-film/SMD points pass; matched DXF remains under investigation |
| Native negative terminals imposed fixed signed voltages / port semantics | official equal/opposite summed-current contract and asymmetric live coupon: old full complex S error 0.24104 at 10 GHz | jointly solve floating common potentials; corrected native error 0.00010538, coupled independent nodal-MNA and current-transfer regressions |
| Native reference capacitor treated as series C / port semantics | actual installed Graph exports and primary port equivalent circuit; incorrect topology fails six comparisons by 0.52–1.30 in complex S | native parallel C evaluator; all twelve R/X/L/C GUI comparisons pass within 6.60e-12; fixed port/material ohms verified by bit-identical OH/KOH/MOH controls |
| Native STF OHMM interpreted as ohm millimetres / material units | actual editor maintain-physical export converts 70000 OHUM to 0.07 OHMM; old public reader reports 0.00007 ohm metres and 1000 times the conductivity | convert OHMM as ohm metres; exact source/hash, numeric, dielectric/conductor and stack regressions preserve the original failure; independently proved MOSQ/MOHSQ milliohm-per-square conversions retain the native/XSD disagreement |
| Native dielectric RSVY silently treated as conductivity / native layer loss | native technology load exports 0.07 ohm metres as RSVY7; actual engine reports 7 Ohm-cm, while the old reader uses 7 S/m and fails full-S by 0.166–0.175 | convert positive dielectric RSVY ohm centimetres to SI conductivity and reject unknown loss/anisotropy flags; six actual native controls, exact hashes, current/source-equation and domain regressions preserve the original failure |
| Native NOR general current-ratio crossover / material semantics | primary Rautio–Demir equation 11 retains full-thickness DC resistance and scales RF resistance | use `k*Zskin*coth(k*gamma*t)`; independent general-ratio regression and native ratio 0.5 complex S gate pass |
| Native FREQ interpreted in project frequency units / physical providers | actual GHz/MHz controls evaluate FREQ in Hz; old physical full-S error reaches 0.988 | retain SI Hz and explicit native unit-conversion functions; literal/node controls, currents, source equations and stored-frequency/provider guards pass |
| Finite NOR current ratio overflows to NaN / numerical domain | finite ratio 1e308 squares to infinity and returns nonfinite sheet impedance | evaluate the symmetric face weight using the reciprocal for ratios above one; independent high-frequency and finite one-face limit tests pass |
| Native escaped expression quotes discarded / source fidelity | an actual native table1 expression gives literal-equivalent full-S, while the reader removes filename quotes and cannot parse it | decode native escaped quotes; bounded owned CSV dependencies and explicit linear/bilinear providers retain exact source identities |
| Native scalar parent source rebound / identity | editing the SON after parsing stages its new bytes alongside the old table expression, returning 2 where the fresh project returns 4 | compare decoded records and their diagnostic line positions before CSV reads; linked snapshots use explicit owned SON/STF conversion provenance |
| Bounded scalar expression stack overflow / resources | a supported 14,023-byte expression with 3,500 nested subtractions stages but raises StackOverflowError when evaluated | reject nesting, AST and combined evaluation depth above 128; preflight expression scratch with effective project storage |
| Native logarithm contract / physical material | valid magnitude ln/log10 expressions reject, while accepted log(exp(2)) disagrees with native's invalid-equation fallback by full-S 0.091 | add exactly-one-argument magnitude ln/log10; explicitly reject undocumented log; retain five actual controls and the old physical failure |
| Native powers parsed with Julia association / physical material | actual `2^3^2/256` gives 0.25 Ω/square, while the old reader gives 2 and changes full S by 0.07855 | bounded native grammar retains explicit parentheses, left power chains and signed-exponent precedence; independent literal and actual native controls retained |
| Native documented functions and real-axis branches missing / source fidelity | inverse/hyperbolic/general/complex functions reject; direct library substitution disagrees at real-axis branch cuts and signed-zero phase | implement native-confirmed functions, real physical quantity boundaries and measured branch defaults; retain 222 actual jobs/340 raw matrices with original false hypotheses, warnings and failures |
| Native complex atan2, unit conversions and CSV keys reject valid operands / source and provider fidelity | 121 actual controls establish complex quadrant/zero-denominator branches, real conversion/table1 projection and signed-magnitude table2 rows; two zero operands give NaN and an empty export | preserve every failed hypothesis and invalid export; retain 119 full matrices in 904 SHA/CRC checked files, compare all valid nonnegative materials to independent native literal controls, and verify original EM/current equations under unchanged gates |
| Native binary complex operands rejected / material and component expressions | actual `hypot`, `int`, `fmod`, `min`, `max` and nested `cmplx` controls evaluate finite quantities that the reader rejects; naive complex ordering hypotheses fail by up to 0.263 in full S | preserve complex operands and measured signed-magnitude/tie/zero rules; retain 73 actual jobs, 72 matrices and 448 hashed files; complex `atan2` was outside that earlier checkpoint and is covered by the later controls above |
| Decimal scalar literal silently underflows to PEC / stored units | source literal 1e-500 becomes zero in Meta.parse and SRVY lowering returns PEC before the stored-value guard sees a nonzero input | validate decimal literals before AST conversion and table numbers before storage; explicit zero remains supported and original reproduction is archived |
| Mixed FLOAT drops ordinary SPARAM snapshots / response provenance | the public physical solve succeeds but returned model/result has no staged Touchstone source, binding or basis snapshot | retain optional model_files in FLOAT model/result, reserve its storage, and compare full S/node/current projection with independent callback wiring; source edits cannot change retained bytes/hashes |
| Native NOR loss selectors rejected / native intake completeness | actual SRVY/RSVY/CDVY and absent-selector controls give identical full-S at matched physical loss | preserve exact selector units, zero-PEC proof and explicit invalid-domain rejection before geometry; general STF geometry and native loader discrepancy remain separate |
| Native FREESPACE treated as perfectly matched modal half-space / boundary semantics | installed radiation manual Fourth Condition and literal 376.7303136-ohm native cover record | preserve constant cover impedance as TERM_SURFACE; generic modal TERM_OPEN remains available; native constant-cover line complex S error 0.001049 passes the original gate (`rfic_yWwbk7`) |
| Parser exception leaves file open / resource lifetime | Windows EBUSY after deliberate budget rejection | `open do` scopes close on every exit; parser rejection and immediate file reuse regressions |
| Later Touchstone option line overwrites first / format correctness | independent IBIS rule and 1 Hz→1 MHz reproduction | ignore later option lines; regression before and after numeric data |
| Touchstone numeric buffers allocate before resource rejection / resources | complete valid 224-port files allocate 16.25 MB before rejecting a 512-byte budget | token-count/aggregate preflight, direct numeric append and early workspace bound; warmed rejection allocates 1344 bytes, the same inputs remain accepted at 32 MiB |
| Empty explicit Touchstone Reference invents default impedances / format correctness | empty blocks in valid 2.0/2.1 headers silently become option-line references | require a complete positive per-port vector when the block is present; 48 malformed/absent/wrapped/sentinel checks pass |
| Sparse native circuit labels create unused singular coordinates / interoperability | actual Sonnet accepts an 11-label circuit with no node 5; default imported solve is singular | densely remap native labels, preserve literal zero and original records; 17 native lowering checks and all 101 actual resonant matrices pass |
| Floating generic Y block falsely connects its null common mode to ground / circuit correctness | 100 ohm two-terminal Y represented on two ground-labelled terminals becomes singular in auto mode | detect exact row/column common-mode nulls and retain the floating coordinate; equivalent-resistor, tiny nonzero common-mode, control and disconnected-pin falsifiers pass |
| Native TERM/FTERM wave references discarded / databank correctness | actual Graph exports contain complex references and noncomment numeric continuations; ignoring them mislabels S or breaks record alignment | retain constant/per-frequency Kurokawa references, native parallel C and strict arity/continuation checks; all twelve Graph exports and circuit-file adapters pass |
| Parametric databank breaks refinement-vector dispatch / API regression | fresh output gate rejects `Vector{PlanarNetworkData{Float64}}` | accept `AbstractVector{<:PlanarNetworkData}`; unchanged three refinement-certificate regressions pass |
| Real-reference voltage conversion changes cancellation / numerical regression | unchanged radiation fixture fails its `1e-13` field gate after the complex-wave upgrade | preserve `sqrt(R)*(a+b)` arithmetic when references are real; original radiation and new physical complex-wave regressions pass |
| Invalid CSV port metadata truncates existing file / failure cleanup | independent sentinel-file reproduction | validate all names before opening; rejection leaves sentinel unchanged |
| Case-colliding SPICE pins alias in the external engine / export correctness | actual ngspice merges `input` and `INPUT`, producing an incorrect first admittance column | case-insensitive uniqueness and reserved-node checks before opening the destination; sentinel-file regressions and actual engine matrices pass |
| SPICE formal `gnd` pin becomes global zero / floating-reference semantics | actual ngspice produces 15 mA and 25 mA under pure common-mode excitation | use the explicit `dm_ref` formal pin for external currents and voltage controls; grounded, floating and zero differential-voltage regressions pass |
| Floating auxiliary state subtraction loses precision / numerical conditioning | actual floating-reference model fails the unchanged `1e-9` matrix gate with error `3.2e-7` | reference computational states to internal zero and retain physical pin-relative controls; all 444 external checks pass, maximum full-matrix error `1.025e-15`; failed raw outputs retained |
| Logarithmic null or masked current crashes HTML export / output workflow | actual Plotly JSON serializer rejects -Inf/NaN | transport true nulls as JSON null without a finite floor; response/current/radiation HTML regressions pass |
| Gerber aperture-block clear objects treated as transparent holes / format fidelity | primary AB rule and independent older-pad/clear-block/polarity-toggle fixtures | preserve ordered block image operations independently of macro holes; nested official corpus and foreground/background regressions |
| Gerber single-quadrant coincident endpoints rejected / legacy geometry | primary G74 zero-length versus G75 full-circle contract and independent point-aperture oracle | retain the zero-length swept aperture at the shared endpoint; 5180 rectangle line/arc checks including both modes |
| ODB nested conductive island erased / format fidelity | primary I/H/I containment order and independent central-island reproduction | preserve stated contour order; deep nested, negative, transparent-hole and user-symbol regressions pass |
| ODB polygon line stroke fills concave notches / analytic geometry | a valid public U-aperture retains an open notch before stroking but the old convex hull fills it | exact endpoint and boundary-segment membership; independent rectangle-interval oracles, reversed/zero-length paths and actual nine-run native coupon pass |
| ODB composite user-symbol line is missing / completeness | enclosing product frame with transparent hole and redraw island rejects before lowering | ordered boundary-partition sweep, supported analytic leaves and bounded nesting/work; independent interval/arc/support oracles plus nine native cases and current-source repeat pass |
| Conformal pulse and polynomial projection erase valid triangles / numerical correctness | accepted nearly collinear triangle has exact nonzero area but rounded determinant gives zero pulse and polynomial moments | reuse the existing robust orientation for both area evaluations; 373 independent rational/simplex-series checks and 4246 conformal/hybrid neighboring checks pass with unchanged residual/allocation gates; earlier external hybrid proof remains source-scoped |
| ODB composite sweep loses scale covariance / numerical robustness | independent normalized interval oracle finds 3 mismatches at 1e-200 and 43 at each of 1e160/1e200 | normalize boundary equations before edge determinants; zero mismatches at all seven declared scales, with unchanged native gates |
| ODB composite stroke misses a closed tangent / analytic correctness | exact dyadic triangle tangent at t=2/3 returns false while robust polygon-segment oracle returns true | certified intervals and bounded exact event replay; closed tangent and immediately exterior controls, current4738 checks and original composite/resource gates pass; final literal-axis eight-helper native repeat retained |
| ODB caller geometry mutation invalidates accepted bounds / ownership | changing callback polygon vertices changes membership with stale bounds; caller graph cycle produces stack overflow after construction | reserve/depth-check and snapshot callback geometry at intake; independent alias/cycle/work replay5 and registered ownership/resource17 checks pass |
| ODB empty aperture leaves bypass work bound / resources | 10000 empty leaves accepted with boundary cap1 and count0 | charge empty leaves and wrappers in conservative traversal/boundary preflight; reject before membership queries |
| ODB circle strokes discard local center / format fidelity | public nonzero-center circle line/arc bounds move by 0.2 mm | preserve literal local center in line endpoints and arc center/endpoints; independent distance, winding and bounds regressions pass |
| Databank wide frequency underflows to DC / stored domain | positive1e-500 accepted as stored0 and dynamic reference callback receives0 | constructor/response guards reject unrepresentable or collapsed frequencies before providers; explicit DC and representable controls pass; separate text-underflow candidate discarded after original parser rejects it |
| RFIC library reference scalars lose stored domain / numerical correctness | positive BigFloat turns/area become zero or infinity; finite spiral arguments lose representable results through intermediate squares/sums | stored-value guards and binary-exponent reference products; 256-bit independent shape/method/range oracles pass with zero warmed reference-product allocation |
| RFIC placement rejects valid rotation and emits invalid polygons / API correctness | representable BigFloat angle throws MethodError; wide offsets emit infinity and finite translation collapses accepted geometry | validate stored placement inputs, convert the angle before trigonometry and validate placed polygons without changing emitted order or pin indices; original reproductions and current native IDC repeat pass |
| RFIC placement retains contradictory pins and invalid via footprints / state correctness | large translation retains original pin width/direction/midpoint despite changed emitted edges; nearly half-pad via margin emits two-vertex vias | rebuild placed pins from emitted edges with range-safe midpoint/width; validate emitted air-bridge via polygons; independent original witnesses and current replay are retained |
| RFIC broadside offsets and capacitance ratios lose representable values / numerical correctness | nonzero wide offsets become zero; valid thickness/permittivity ratios become zero/infinity although final capacitance is finite | stored broadside offset guard; common-exponent positive dielectric-ratio sum and scaled final reference product; independent wide-range equations and controls pass |
| ODB unused lookups bypass memory preflight / resources | 2000 valid unused stencils retain at least 1280000 numeric segment bytes under a 207721-byte limit | reserve lookup geometry/entries, attribute/text tables and output attributes; reject before later callbacks and preserve current allocation gate |
| ODB callbacks grow live input beyond budget / source ownership | appending a caller-owned 2 MiB comment to a 19-byte source causes 4733824 reader allocation bytes under a 4096-byte budget | bounded descriptor snapshot before callbacks; file append/truncate/replace/remove no longer alter the read, and owned byte storage avoids an extra string copy |
| ODB live contours bypass memory preflight / resources | valid 20002-segment comb retains at least 1280128 numeric bytes under a 595544-byte cap before late emit rejection, allocating about 24.14 MB | bounded allocation-free slot counting from the owned snapshot; reserve before exact allocation and transfer each distinct contour buffer into its region; original source/failures and scoped rejection/allocation regressions retained |
| Clockwise circular region loses horizontal radius / analytic geometry | independent disk/ring oracles expose closure alias and rounded trigonometric endpoints | full-circle disk parity and literal partial-arc endpoints; 1286 translated/directed disk, quarter-arc and cap checks pass |
| Elliptical stroke raster allocates per cell / memory efficiency | new 80×80 legacy SF raster allocates 334597096 Julia bytes despite correct geometry | fixed quartic tuple types and homogeneous angle loop; zero membership/helper allocation, 400×400 raster 28781 bytes including returned mask; unchanged geometry and allocation gates pass |
| UNIX-compress decoder allocates arbitrary-precision sums per code / memory efficiency | 837120-byte independently packed fixture allocated 72220604 Julia bytes to a null destination | subtraction before addition enforces the same expansion cap without overflow; allocation falls to 331204 bytes, exact SHA and expansion-cap regression pass |
| Mixed vector text and apertures allocate per raster point / memory efficiency | eight glyphs allocated 149580973 bytes; square pad with circular hole allocated 15350365 bytes on a 400×400 grid | concrete stroke references and bounded small-composite tuple storage reduce both to approximately 28.8 KB including the returned mask; every mask cell unchanged, zero typed-membership allocation and 9826 shared composite regressions pass |
| Valid gzip tar alias treated as plain tar / archive fidelity | installed Sonnet `odb++_trans.tgz` fails before matrix parsing despite independently valid GNU tar members | `.tgz` and `.TGZ` use the bounded gzip route; Windows transport77 checks pass; logical uppercase matrix references also resolve to legal lowercase paths while raw metadata and duplicate/path guards remain intact |
| JSON serializer buffers before resource rejection / memory preflight | rejected300k string under257-byte limit allocated1134559 bytes | exact encoded-size preflight and domain-limited streaming encoder reject before output storage;2464-byte rejection allocation and escaped-control/Unicode decoding checks pass |
| Accepted radiation power becomes infinite or zero after conversion / numerical correctness | finite BigFloat inputs overflow/underflow Float64 storage and silently corrupt gain | finite-positive representability check before angular samples and checks for manually constructed pattern metadata; radiation and complex-reference consumer304 checks pass |
| Bulk conductivity omitted by layout gradient / optimization correctness | a finite-conductivity 15 Ω bar solved correctly forward but the gradient's forward objective and derivative were zero | evaluate the same frequency-dependent bulk models and explicit conductivity override in both gradient paths; analytic `dR/dh=-750000 Ω/m` agrees within `1.5e-9` relative; raw/contracted callback and budget regressions |

## Completion gates

The suspected physical-return epsr adjoint defect was ruled out. Independent
192-bit modal assembly, solve and central differences converge to
`dJ=-8.0443535450e-7`; the production adjoint differs by `5.3e-8` relative.
The Float64 objective is near-insensitive (`J≈0.35343` with a derivative
of order `8e-7`) and differed from the high precision objective by
`4.83e-13`, explaining small-step finite-difference cancellation. Evidence:
`validation/planar_audit/layout_modal_derivative.jl`. The library regression
uses a capacitive objective that resolves the same material sensitivity.

Completion requires closing implementation gaps above, repairing every
confirmed failure, rechecking resources and physical invariants, and running
the full package/docs/native/absolute-accuracy ladder on the final sources.
Parsing coverage or synthetic round trips alone cannot establish full
Sonnet or RFIC design parity.

The library MIM solve includes explicit physical reference returns, per-cell
sheet loss and capacitance extraction. Its low-frequency result is
`40.2464 pF` versus the `38.8477 pF` parallel-plate overlap reference
(3.60% difference). Added lead effects and fringing remain in that response;
the low-frequency 10% approximation gate passes. The 1 GHz extracted value
is dispersive and is not described as a DC capacitance. Reproduction:
`validation/planar_audit/rfic_library.jl`.

The public IDC library has a separate actual native acceptance proof. Two
fingers per electrode and explicit library wall leads match eight independent
literal native rectangles on 16/32/64-cell grids at 1/10/20 GHz, for both PEC
and a constant 0.1 Ω/square film. All eighteen raw full-matrix comparisons
pass the 0.005 complex S gate, with maximum difference `0.000188252` and
maximum independently reconstructed original-voltage residual `2.27e-14`.
The initial nine-case PEC proof and two diagnosed validation harness failures
are retained. The 941-file native/source archive includes fresh eighteen-case
repeats after the Library domain and placement/ratio fixes, with bit-identical
parsed native matrices. It has 1377 current-production replay checks. Reproduction:
`validate_sonnet_library_idc.jl`; fixtures:
`test/fixtures/rfic_library_idc_native`. This closes the declared finite-grid
IDC coupon gate; measured-device accuracy, other library geometries, general
loss and continuum convergence remain separate completion work.

The Library's separate stored-domain audit repairs positive BigFloat inputs
that previously became zero/infinity, intermediate reference arithmetic that
lost finite results, a valid BigFloat rotation method error, and placements
that emitted nonfinite or collapsed polygons. All seven declared spiral
shape/method combinations are checked against independent 256-bit equations
over the recorded range, including subnormal diameters and unrepresentable
output rejection. The binary-exponent product has zero warmed allocation.
The 1004 numerical/domain/allocation checks and 50 durable proof checks pass;
ordinary placement, mirror, pin-edge, layout/current/gradient/connectivity and
actual native IDC controls retain their existing gates. Before/after sources
and public reproductions are in `test/fixtures/library_stored_domain`.

The independent follow-up also closes broadside offset underflow, contradictory
placed pin widths/midpoints/directions, collapsed air-bridge via footprints,
and capacitance-denominator range loss. Placed pins now describe their actual
emitted edges, including tiny midpoint and large translation controls.
Common-exponent dielectric ratios retain representable final capacitances;
the warmed extreme-ratio reference allocates zero bytes. Its 1109 independent
checks and 110 durable original/source checks pass in
`test/fixtures/library_remaining_domain`. The registered numerical/domain
suite has 1602 checks, including 548 new placement/ratio controls.

The public two-level MIM library has a separate eighteen-case actual native
proof on 16/32/64-cell grids at 1/10/20 GHz, for PEC and constant
0.1 Ω/square film. Independent literal rectangles declare both plates and
wall leads across a 1 μm, ε=7.5 insulator, above ε=4 substrate and below air.
All raw full matrices and independently reconstructed original-voltage
residuals pass the unchanged 0.06 and 1e-9 gates. The final repeat follows
the Library placement/ratio and scalar grammar fixes; its 554-file archive
retains twelve source snapshots, earlier eighteen-case evidence and the
diagnosed initial mask-order oracle failure. The current replay has 947
checks. Reproduction: `validate_sonnet_library_mim.jl`; fixtures:
`test/fixtures/rfic_library_mim_native`. This closes this declared finite-grid
wall-port coupon; internal calibration, measured-device accuracy, broader
technology/loss and continuum convergence remain completion work.

The same complete public MIM coupon also has an independent low-frequency
overlap reference: plate plus lead overlap is `7.03125e-8 m²`, giving
`4.669200607 pF` across the 1 μm, ε=7.5 gap. At 100/500 MHz on
16/32/64-cell grids, extracted mutual capacitance differs by 1.14–4.19%,
passing the declared 5% approximation gate. Independently reconstructed
original-voltage residuals remain below `1.55e-12` against `1e-9`.
The archive and current solve regression are in
`test/fixtures/library_mim_low_frequency` and
`test_planar_library_mim_low_frequency.jl`. Finite-grid lead/fringing
effects are retained; this approximate electrostatic check does not
establish general continuum or measured-device accuracy.


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
It does not establish the same RF limit for other shapes.

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
equation passes. The 78 physical residual failures at 1/10/100 MHz remain
open; passing S/loss alone does not certify those low-frequency currents.
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

General thick HOLLOW topology, ARR grounded loss and conductivity, SFC
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
peak RSS, opaque memory and global optimization remain unverified.

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
