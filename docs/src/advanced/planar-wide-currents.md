# Checked dense currents and numerical bounds

This page complements the [Sonnet parity ledger](sonnet-parity.md).


Dense and dense-FFT raw solves check physical voltage residuals against
original assembled entries, dividing each Galerkin row by its physical
trace measure. The acceptance limit remains 1e-9. When Float64 coefficients
cannot meet it, the solve derives initial wide precision from the input
significands, system dimension and machine limbs, reuses its Float64 LU
factors, and retains owned coefficients for iterative correction. The
correction loop stops on nonprogress and has a dimension-derived cubic work
bound. It then factors the original physical matrix at the derived precision
if needed, retaining that owned LU with the recovered currents. The original
coarse native attachment fixture exercises this rare path. An earlier
snapshot's independent 512-bit residual was below 7e-24; the current focused
regression independently enforces the same original 1e-9 physical target. Failure to satisfy the
physical limit still raises an error. Ordinary accepted solves retain ComplexF64
coefficients; recovered `PlanarResult.currents` can be a matrix of
`Complex{BigFloat}`. Port admittance, S-parameters and current-map samples
remain ComplexF64.

Port contraction and calibrated coefficient transfer preserve the retained
current precision. Current maps accumulate before narrowing their samples,
and network far-field excitation forms its current reaction before
narrowing. These operations use an owned precision and nearest rounding
scope, preserving the caller's precision and rounding context. Declared
wide array and scalar workspace is checked before construction. The MPFR
library is a declared `MPFR_jll` dependency; import-discipline gates keep
their existing reviewed boundaries.

The dense compact path factors its complex assembly buffer in place and
reuses the existing real and imaginary assembly arrays for exact residual
checks, including the final local loss terms. A single component owner
avoids additional tuple and view overhead. The original allocation test
requires twelve solves to save at least twelve full matrix payloads and
passes on both Julia 1.13.1 and 1.12.7. The checked dense-FFT compact path
still needs its original complex matrix during the solve. Both compact
paths return `z_mom=nothing` and retain the factorization afterward.

The current registered module passes 329 assertions on both versions, plus a
loaded-library rounding-mode control and the unchanged Aqua/import checks.
It compares dense and dense-FFT native triangle HOLLOW responses at
1, 10 and 100 MHz with the original 0.06 complete-S, 0.05 loss and 1e-9
physical gates. Independent 256-bit original-matrix residuals verify both
retained and compact current columns. Constructor compatibility,
cancellation, exact workspace boundaries, port contraction against
512-bit arithmetic and concurrent caller precision/rounding are also
checked. The original 29 compact-resource, admittance and file-preservation
assertions pass separately. A current repeat of all 130 original material and rectangular
overfill controls passes on both versions, with the same native projects,
references, five frequencies, grids, modes and methods. The earlier
78 physical-residual failures per version become zero under independent
256-bit original-matrix checks. The maximum current physical residual is
9.534e-10; all original S, loss, passivity and reciprocity gates also pass.
These are scoped controls. The complete package, other physical-source
solvers, native convergence and broad planar/RFIC completeness require
their own current evidence.

Wide-current derivative contractions and adjoint factor operations also use
the current columns' owned precision and nearest rounding. The original
coarse fixture's gradient is identical across 32, 256 and 8192-bit caller
contexts after this correction; callbacks run before the owned derivative
scope. Nineteen additional coarse-fixture controls check the retained wide
factor, original-matrix residual, compact result and caller-context
restoration. A separate low-frequency triangle finite-difference experiment
does not converge as the perturbation shrinks. Its derivative accuracy and
forward assembly rounding sensitivity remain open qualification items.


The dense gradient's wide adjoint and modal contractions retain owned wide
coefficients instead of writing the adjoint into Float64. On the coarse
fixture, an independent 512-bit transpose-equation check gives 0.0417 for
the narrowed adjoint and below 7e-24 for the wide adjoint. Sixteen retained
and compact, dense and dense-FFT, caller-precision controls pass the
original 1e-9 physical limit on both Julia versions. A compact wide solve
with Float64 factors reconstructs its original matrix for this rare
adjoint check; it releases that matrix before modal contraction. Recovery
and its allocation checks finish before invoking objective callbacks.

Reusable MPFR arithmetic avoids repeated wide complex-product temporaries
inside derivative contractions. Matched warmed calls on the unchanged
native triangle at 1 MHz, a 20 by 20 grid and 40 by 40 modes allocate
10.4 MB on both versions, compared with 4.559 GB before this change: a
99.77 percent reduction in cumulative Julia allocations. Median elapsed
time improves by about 4.00 times on Julia 1.13 and 3.77 times on Julia
1.12 in these concurrent audit-process conditions. An independent
512-bit full derivative-matrix contraction returns the same coarse-fixture
gradient; its modal derivatives share the original dual cascade formulas.
These measurements exclude peak RSS and external allocator bookkeeping.

The warmed public allocation regression retains its 32 MiB limit. Earlier
snapshot measurements reported an 8 MiB rejection based on conservative
guessed padding; that historical rejection is not a current requirement.
Current representation-based estimates accept the 8 MiB request and return
the same objective and gradient as the 16 MiB request. A request limited to
the required MoM matrix alone rejects before either callback. The new
registered checks preserve this early resource failure and both supported
requests, rather than retaining unnecessary padding to fit an old test.
The original 319 gradient, bulk-loss and integration assertions are retained
in the complete test graph. Their previously reported passes refer to the
earlier source snapshot; the current full-source suites are being rerun.
Small earlier reusable-workspace controls reported 352 Julia-allocated bytes.
Whole-solve finite-difference convergence and general planar/RFIC
acceptance remain open. Contracted-source accuracy is qualified separately
below; these earlier allocation measurements concern the raw dense and
gradient paths.


The current wide-current implementation derives its initial precision from
Float64 significands, system dimension and machine limbs. Summation guard
bits apply to similarly scaled terms; acceptance still uses the original
physical equation. Stored coefficients carry their own precision through
current maps, port contractions, calibrated transfers, adjoints and project
radiation. Mixed corrections stop when the physical residual ceases to
decrease and use a dimension-derived work bound of the same cubic order as
direct factorization. The earlier fixed eight-correction limit is removed.

MPFR reports significand storage directly through `mpfr_custom_get_size`.
The remaining scalar object size is derived from Julia's public
`Base.summarysize` representation measure on the actual host. Complex
reference storage follows the concrete type sizes and live arrays. Named
arithmetic roles account for four current-product scalars, eight modal
gradient scalars, and eight factorization/solve scalars; these replace
guessed byte and temporary allowances. Current-map cells receive independent
scalar storage before modification, while untouched cells share a zero
value. A two-scalar weight/product workspace prevents repeated map arithmetic
allocation. These are operation-owned representation bounds; they exclude
compiler memory, garbage awaiting collection and external allocator peaks.

Exact rational stored-input checks verify partial pivoting, `P*A=L*U`,
forward solves and nonconjugated transpose solves for nonsymmetric and
extremely scaled systems. The controls use 128/256-bit owned values under
32/8192-bit caller settings, with singular and dimension error cases.
Matched in-place solve measurements use 80 Julia-allocated bytes instead of
26,080 for the ordinary nonsymmetric 128-bit reference case. An 80-byte
allocation remains; zero-allocation behavior is not claimed.

The complete-project radiation consumer now preserves cancellation in
owned coefficients: `2^100+1-2^100` produces a unit coefficient and the same
nonzero unit-source far field at caller precisions 32, 256 and 8192. Earlier
code returned zero at 32 bits. The probe uses the project's actual source
geometry with manufactured coefficient columns; it does not establish
native/model-solve radiation accuracy. It also checks caller-state restoration,
input ownership and rejection below a measured owned-output requirement.

Matched native-triangle map calculations retain exactly equal stored
`jx`, `jy` and `jz` values while reducing warmed Julia allocations from about
468.9 kB to 391.1 kB on both versions, approximately 16.6 percent. Real and
complex port matrices and a complex voltage vector match an independent
512-bit stored-input oracle and its operation-count rounding bound; their
allocations fall from about 650 kB, 1.42 MB and 633 kB to 72.4 kB, 72.4 kB
and 36.7 kB in the earlier qualified consumer snapshot. Current tests retain
those mathematical controls and add the factorization/transpose cases.

The registered module contains 329 assertions: 255 native/current/precision/
project/gradient/resource controls and 74 exact factor/solve checks. Both
four-thread focused suites pass with Aqua and the unchanged public-import
allowlist. Complete current-source, original native material families,
documentation and hosted CI qualification remain separate checks. The
older 99.77 percent gradient allocation reduction above is an earlier
matched measurement, not a universal optimum or a current peak-memory claim.
The broader audit of fallback precision, error allowances, operational limits,
empirical coefficients, contracted-source accuracy and general planar/RFIC
completeness remains open. A named constant or a configurable value alone
does not justify an arbitrary numerical choice.

## Direct physical voltage contracts

`solve_planar_contracted` solves the physical source columns defined by
`C`, avoiding independently driven raw layer sources. Both dense methods
check the stored original matrix equations after dividing each Galerkin
row by its trace measure. The caller's positive `rtol` is an acceptance
requirement. Failure to meet it raises an error instead of returning a
dense source whose residual exceeds the requested target. FFT retains its
independently recomputed full-operator acceptance check.

The original explicit `PlanarSourceResult{F,O}` constructor retains its
`ComplexF64` current-storage contract. Inferred construction and a fully
specified current type can retain owned wider coefficients.

Ordinary source currents remain `ComplexF64`. Cancellation-sensitive
recovery retains `Complex{BigFloat}` coefficients in `PlanarSourceResult`
and its contracted wrapper. Correction of the initial equilibrated LU
uses its two-sided basis scaling; direct wide recovery factors the original
physical matrix and retains unit `basis_scale`. Terminal admittance is
accumulated with reusable owned scalar arithmetic before narrowing to
`ComplexF64`. Current maps, nested voltage contracts and radiation
excitation form their coefficient product using the stored current
precision. The direct far-field API then converts coefficients to
`ComplexF64` under the caller's conversion rounding mode; wrapper/direct
comparisons use the same caller context.

Compact dense source assembly reuses its real and imaginary reaction
accumulators while factoring the complex buffer. Compact dense FFT keeps
the original matrix through checking. Both return `z_mom=nothing`, retain
their factor and report physical voltage and unscaled Galerkin residuals.
Declared payload accounting includes both matrix roles, wide current
storage, a retained wide factor and its pivots when present, residual
reference arrays and the explicit scalar workspaces. It describes owned
numeric payload; it does not measure peak resident memory.

Exact rational arithmetic on the original native triangle's stored
entries verifies 24 combinations of 1, 10 and 100 MHz, dense/dense FFT,
retained/compact storage and preconditioning on each Julia version.
The earlier Float64-only source exceeded the existing `1e-9` physical
target in 20 combinations per version. The checked source passes all 24,
with maximum exact relative residual `3.662530174199651e-12` on both
versions. The native grid and modal controls are unchanged.

The original 144 contracted-source assertions and 85 caller/consumer
controls pass on Julia 1.13.1 and 1.12.7. The latter include the original
physical target, its square and Float64 machine resolution, original
equations, low caller precision, maps, radiation, nested contracts,
source-amplitude variation, ownership and error paths. Separate measured
output-budget controls pass for the original triangle and the coarse
attachment fixture requiring a wide factor. Their warmed allocation
samples show approximately 1–2 KB of added accounting overhead compared
with the immediately preceding prototype; this is not an allocation
reduction claim. Full current-source suites, documentation and hosted
verification remain separate qualifications before publication. Physical
model accuracy, native convergence and broad RFIC completeness still
require their own evidence.

## Radiation output range

Far-field intensity divides each electric-field component by
`sqrt(2eta)` before squaring and combining polarizations. This is the
peak-phasor power convention, evaluated in an order that avoids overflowing
the unnormalized squares. A representable intensity near
`1.1311572183803282e306` previously returned infinity despite finite fields;
the normalized calculation returns a finite value consistent with exact
rational arithmetic on the stored fields. No amplitude cutoff is applied.

Public field samples require finite electric fields and finite intensity.
When an output cannot fit its declared Float64 representation, the call
raises `ArgumentError`. Ordinary samples and the original independent
analytic radiation, reciprocity, volume, power-wave, metrics and plotting
controls retain their numerical gates. On both Julia versions, all 106
original radiation assertions and 13 new range, energy, ownership, budget
and zero-field controls pass. These controls describe postprocessing range
and stored-field energy; they do not certify the input current solution,
native radiation accuracy or continuum convergence.

## Legacy coefficient input types

`PlanarResult` converts ordinary numeric coefficient arrays to
`ComplexF64`, preserving the original behavior for integer-complex,
Float32-complex and real arrays. Supported Float64 matrices retain their
identity; matrix views receive an owned container. `Complex{BigFloat}`
matrices and their views retain the wider current representation needed
by checked recovery and consumers. Borrowed coefficient values remain
unchanged during construction, maps, contractions and radiation.

An earlier wide-current prototype retained integer-complex and
Float32-complex matrix types, which its MPFR consumers could not use.
Current ordinary conversion restores those public workflows. All 42
datatype, consumer, ownership and view controls pass on both Julia
versions, and the registered current-source suite passes 520 focused
assertions with the existing Aqua and import checks retained. Complete
current-source and hosted qualification remain separate requirements.

## Saved result compatibility

Checked physical currents require changes to the result representation:
`PlanarResult` retains a current/factor type and physical residuals, while
`PlanarSourceResult` carries its current element type. Constructor compatibility
does not imply binary compatibility for prior Julia `Serialization` caches.
Actual caches written with the previous definitions fail to deserialize under
these types on both supported Julia versions.

Recover a cache in its original compatible code/dependency environment, then
export supported network data through Touchstone or rebuild and solve from
the original model inputs. Touchstone excludes geometry, currents, factors and
accuracy evidence. The representative common-real-reference recovery/import
checks preserve the stored network values exactly; opaque object graphs have
no automatic cross-revision upgrader. See the [saved-object policy](../api/types.md#Saved-Julia-objects-and-portable-data)
for the environment records to retain and the explicit migration boundary.
