# Native volume loss, endpoint currents and conformal geometry

Native VOL conductivity, resistivity and sheet resistance use the physical
cross-sectional metal area and the retained axial mesh. SOLID retains an
inactive wall field; guarded thin HOLLOW geometry and rectangular overfill
preserve their different sheet-to-conductivity and RF-depth contracts.
General thick HOLLOW topology and grounded ARR skin behavior remain open.

The audit reproduced a missing tangential current path for a via contacted
by a narrower sheet. Passing the retained algebraic equation had not
certified the omitted endpoint geometry: published RPV controls and held
VOL/PEC controls exceeded their native complete-S gates. The corrected
importers add endpoint conductors at both interfaces, preserve original
physical-sheet priority and use the captured VOL/RPV/PEC film laws. COVERS
uses physical polygon pads. Generated finite films on a shared interface
combine in parallel, eliminating source-via ordering dependence. Genuine
conformal sheets preserve physical coordinates and material seams.

The audit also reproduced caller-dependent Float64 conformal meshes at
BigFloat precision 32 versus 256/8192. Scoped 8192-bit intersection and
orientation fallback arithmetic preserves the stored Float64 coordinate
domain and caller precision. Independent 32768-bit primitive controls cover
exact zeros, rounding, extreme exponents and near-collinearity.

Endpoint row-run payload is checked before constructing rectangles. The
interface overlay rejects oversized borrowed input before building its
reference/filter arrays, and retains background/overlay/normalized geometry
in subsequent meshing reservations. The 10000-record rejection allocation
fell from 1282495 bytes to 1280 on Julia 1.13.1 and 1728 on Julia 1.12.7.
The 2000-row counter allocates 64 bytes in the measured warmed control.
These are specific allocation measurements, not peak RSS or optimality.

Earlier V13 source-pinned 920 volume and endpoint assertions passed on both
Julia versions. The current endpoint module adds 437 meaningful assertions:
233 provenance/schema controls, 116 actual partial-contact RF/ownership
checks, 16 shared-film/grounded checks, eight precision/ownership checks,
three budget checks, 47 public translated/range material checks and eight
private rectangle-depth checks and six actual refinement allocation/hash/
ownership checks. Current V18 combined endpoint/volume 926 assertions
require complete-suite qualification; the earlier 920 result predates the
solver and refinement changes.

A separate source-pinned public audit independently passes all 52 RF
responses after the geometry-precision fix: 13 original native cases,
raster/conformal, 1/10 GHz, under the original 0.06 full-S, 0.05 finite-loss,
1e-9 physical/reciprocity/passivity gates. PEC uses full ABCD B real loss
with the unchanged 1e-8-ohm gate; an independent lossless L-section proves
that the symmetric 100*S11/S21 proxy can falsely indicate PEC resistance.
The legacy diagnostics and failed candidates remain retained.

The new native fixture archive contains sixteen original Sonnet 18.53-Lite
source/reference/process captures and four original global capture reports.
Two stacked per-case receipt files serialize exact original global capture
rows because no original per-case receipt existed. Native project, S-matrix
and engine-output bytes remain unchanged. References are not fitted,
phase-aligned or retagged. Installed Lite licensing restricts native via
captures to Ring fill; the rejected non-Ring capture remains evidence of
that license limitation, not an accepted native case.

The finer grounded conformal control uses matching 40 actual native bulk
cells and independently sized 25/50-micrometre sheet triangles at 1 GHz.
Grounded finite loss is checked against full Hermitian admittance. Native
40-half-cell loss was not converged; finer-grid controls do not establish
continuum convergence or fine conformal 10-GHz acceptance.

The 78 earlier low-frequency physical residual failures per Julia version
remain open. General thick/topology adapters, ARR ground/skin loss, SFC
horizontal endpoint behavior, BAR lowering, native non-Ring contact physics,
mode/mesh convergence, complete project semantics and measured RFIC
spiral/coupled acceptance remain incomplete. No complete Sonnet parity,
absence of all bugs or global memory/performance optimization is claimed.

Current complete suites, documentation, exact committed source equivalence,
Python validation, pinned quality accounting and all hosted jobs must pass
before publication. Record measured code growth in a separate accounting
commit without relaxing numerical or duplication gates. Preserve the old
V12 publication hold and use a fresh candidate and publisher for this fix.

See [Sonnet Volume Loss Model](https://www.sonnetsoftware.com/support/help-18/users_guide/VolumeLossModel_ug.html)
and [Via Properties](https://www.sonnetsoftware.com/support/help-18/Sonnet_suites/PolygonViaPropertiesWin.html).


The current physical-area calculation anchors polygon coordinates locally.
Verified rectangles preserve exact bounding-box fill before equivalent-depth
calculation, and retain that geometry identity through cold wide recovery.
A valid public 10-mm translated SRVY square formerly had 7.63e-7 axial and
4.18e-7 film component errors. A private 1-m control exposed 1.05e-8 depth
error; two public 1-mm extreme SRVY cases exposed 9.85e-9 wide-recovery error.
All eighteen public imports and sixteen private translation controls now
pass independent 32768-bit physical-area/slab/half-film equations on both
Julia versions, with maximum public component error below 8.89e-16 under
the unchanged 2e-12 gate. This is material/public-import evidence, separate
from native EM or complete mesh/mode convergence. Failed initial fixes and
superseded qualification runs remain retained; no stale full result is
credited to this current source.


The verified refinement fix removes captured boundary-loop coordinate boxes.
Matched warmed imports reduce Julia allocations by at least 87.3 percent,
from 16.34/16.40 GB to 2.069 GB, with exact original vertex/triangle/interface/
area bytes. Both Julia versions match eighteen independent adaptive/uniform
refinement controls and pass six actual durable allocation/geometry checks.

The phase cache shares triangle Fourier weights across RWG halves, matches
the original scalar weights and prior assembly bit for bit, and fills
without warmed allocations. The modal assembler optionally batches up to
64 nonconjugate reactions within the remaining declared payload. A budget
below two columns retains the original scalar accumulation. Both versions
pass 152 durable scalar/cache/batch assertions, including finite triangle
loss, payload boundaries, partial batches, reciprocity and zero allocation.

Matched 2,468-unknown 64-by-64-mode assembly falls from 68.2 to 17.7 seconds
on Julia 1.13.1 and from 69.7 to 20.2 seconds on Julia 1.12.7. Relative
matrix error is 2.92e-15 under the existing 2e-14 gate. This is a bounded
performance measurement; the complete package/native 320-mode grounded RF
and unchanged 3600-second quality command remain required before adoption.

The original held V12 publication remains held. Current V18 source, frozen
committed suites, exact source equivalence, Python validation, current docs,
pinned quality accounting and hosted CI require their own concrete evidence.


The first current V18 full suites stopped at an older pure-bulk expectation
for a sheet-free axial via. The endpoint adapter now represents its physical
end conductor in the hybrid problem. The corrected legacy module preserves
all earlier shape/reference/ground checks and adds independent endpoint area,
axial-only source assignment, coordinate ownership, passivity, reciprocity
and the original 1e-9 residual gate. All 56 assertions pass on both versions.
The failed full reports and initial test-field typo remain retained. Fresh
complete suites with this exact corrected module are required before adoption.

Current V18 low-frequency diagnostics reproduce six physical failures per
version on the captured triangle HOLLOW at 1/10/100 MHz, despite passing
native S/loss. Independent 256-bit residuals prove the issue. Unscaled/scaled
Float64 LU, existing compensated refinement and Float64 casting of a full
wide solution remain above the original 1e-9 gate. A standalone correction
using the existing Float64 LU and wide current coefficients passes eight
128/256-bit controls with independent 512-bit checks. This is a prototype;
public result, ownership, retained-factor, range and full qualification are
still required. Production low-frequency acceptance remains open.
