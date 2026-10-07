# Checked physical currents and verified consumer compatibility

Low-frequency dense and dense-FFT currents must satisfy the existing original
physical voltage requirement. Recovery derives its working precision from
stored significands and dimensions, stops corrections on nonprogress and
retains an owned wide factor when needed. Current maps, contractions,
calibration, project radiation and nonconjugated adjoints preserve owned
precision until their declared outputs. Workspace accounting uses concrete
representation sizes and arithmetic roles.

Contracted-source solves now check the caller's target against original
equations for retained and compact dense results. The original explicit
two-parameter SourceResult constructor retains ComplexF64 storage. Ordinary
PlanarResult integer-complex/Float32-complex inputs again normalize to that
legacy representation, while supported wide matrices and views retain their
precision and ownership. Radiation intensity normalizes fields before
squaring and rejects outputs outside its finite declared representation.

The actual current source passes complete bounds-checked Pkg.test on Julia
1.13.1 with four threads and 1.12.7 with one thread, including 520 focused
assertions and every original native/import/resource test. Fresh 130 archived
native material cases per version pass all original matrix, loss, passivity,
reciprocity and physical-equation gates; maximum physical residual is
9.533142475665256e-10. Exact stored-equation, ownership, caller precision,
budget and legacy constructor controls remain registered.

Actual current-source documentation passes with the original build/navigation
and page-size controls, including the primary DC/empirical applicability of
Mohan reference formulas. Coefficients, original references, scientific
targets and required quality commands remain unchanged.

Earlier matched allocation measurements remain limited to their measured
fixtures. New source-residual accounting adds roughly 1–2 KB in its matched
comparison; it is not an allocation reduction claim. Global optimality, peak
RSS, broader native convergence, wall/ARR/SFC/BAR/serialization and measured
RFIC completeness remain open. Two geometry integer thresholds have exact
Float64 representation derivations; other precision, power, series and
resource policies still require separate justification. Naming a number or
making it configurable supplies no mathematical or measured basis.

This candidate supersedes the archived V40 candidate after fresh constructor
regressions were confirmed and fixed. Exact accounting, committed static and
Python controls, source/test/native/docs equivalence, all-checkout cleanliness
and actual hosted jobs remain distinct required qualifications. Main is not
declared green from a partial local quality run or an older hosted head.
