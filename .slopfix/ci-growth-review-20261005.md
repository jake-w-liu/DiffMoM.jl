# CI integrity and intentional growth review — 2026-10-05

The current Sonnet/planar implementation and its independent native evidence
have grown beyond the previous aggregate limits. The repository policy in
`scripts/slopfix.py` explicitly permits intentional functionality growth through
a separate reviewed ceiling commit. This review records that adjustment; it is
not a source-reduction or optimization claim.

The original `.slopfix/baseline.json`, measurement scope, Julia 1.12.7 counter,
source classification and 400-group census limit remain unchanged. Every
historical source copy and installed HTML manual remains counted. No fixture,
manifest, reference matrix or archived failure is edited or removed.

| Measurement | Previous limit | New audited limit |
| --- | ---: | ---: |
| Counted code lines | 151,500 | 212,017 |
| Census removable-line estimate | 7,900 | 58,394 |

The historical measurement baseline remains 74,755 lines. The preceding
conformal-clipping accounting attributed 70,502 counted lines to immutable or
fixture source, 71,289 to live implementation, 44,169 to live tests, and 25,781
to other code and validation/quality tooling. The current small increase adds
CI repairs, exact integrity reviews and their negative regressions. These are
aggregate growth limits, not thresholds for runtime correctness or memory.

The clone census still reports truncation at 400 groups. Its estimate is the
existing ratchet statistic, not a complete duplication inventory. No narrower
file scope or archive subtraction is used to meet either limit.

## Cleared detector findings

`.slopfix/integrity-reviews.json` binds 55 immutable historical source/manual
files to exact raw SHA256 values for the long-line detector. Installed Sonnet
manual paragraphs and retained historical producer dictionaries/scope strings
are evidence, not newly collapsed implementation blocks. Long lines in eight
maintained test/validation files were instead expanded with identical literal
values. Those maintained files have no long-line exception.

Three exact placeholder-detector findings are retained visibly as advisory:
the live ODB parser and two historical copies reject dimensional pad resize
before parsing ordinary indexed symbols. The public `read_odb_features` contract
explicitly lists that variant as unsupported. This review does not claim resize
support. The current source review normalizes Git line endings; archived reviews
use exact raw bytes. Each finding requires its own path, line and full-file hash.

A changed/missing reviewed file, invalid review, new path, unsupported rule,
new finding or unreviewed live long line still fails. Reviews affect integrity
interpretation only; line/clone counts, original baseline and numerical tests
remain intact. The six new regression tests cover these failure cases, exact
advisory visibility, unchanged counting and explicit line-ending modes.

The local Python validation graph passed all 29 tests. The precise original
voltage residual and conformal iteration limits remain unchanged. Hosted CI
must still pass its full platform matrix, documentation and executable quality
contract before this review can be cited as a green workflow result.

## Follow-up charge precision repair

The subsequent conformal repair adds 40 counted lines for compensated sparse
RWG charge multiplication and independent 256-bit cancellation regressions.
The reviewed line ceiling is now 212,057; the unchanged 400-group census
estimate remains 58,394. This additional ceiling adjustment is a separate
commit from the implementation, with the same original scope and counter.
The existing field workspace supplies the compensation scratch. Focused
single/multilevel tests passed on Julia 1.12.7 and 1.13.1, including zero warm
allocations. The original `1e-10` voltage gate and numerical fixtures remain
unchanged. Hosted macOS and the complete workflow still require verification.

The uniform dense trace-scaling repair and its low-frequency current
regression add nine further counted lines, raising the exact reviewed ceiling
to 212,066. The unchanged duplication estimate remains 58,394. This removes
an unnecessary matrix rescaling when every physical trace measure is equal;
mixed-trace equilibration and FFT preconditioning retain their existing
contracts. The current-map tolerance remains `2e-12`. Focused contracted and
power-wave suites passed on Julia 1.12.7 and 1.13.1. The quality workflow now
pins Julia 1.12.7 exactly, matching the unchanged baseline counter identity
instead of allowing an automatic patch upgrade to break that comparison.

## Native raster resource preflight

Commit f90bd880 adds 248 counted lines for public raster allocation preflight,
aggregate solve ownership, budget forwarding and 52 independent resource
assertions. The reviewed line ceiling is now 212,314 in a separate commit.
The original 74,755-line baseline, counter identity, counted scope and archival
evidence remain unchanged. The unchanged 400-group truncated census reports
58,169 estimated removable lines; the existing ceiling stays 58,394. Its
selected-group estimate is not evidence of a measured optimization or removal.
Blocking smells, integrity findings and measurement warnings remain zero.

The repair and CRC evidence are recorded in
`.slopfix/sonnet-raster-resource-review-20261005.md`. No original numerical
reference, native gate or tolerance was modified. Full package and fresh
hosted CI checks remain mandatory before claiming this commit is verified.

## Unused default PEC material maps

The isolated optimization 0b259e9b adds 60 counted lines for selective
material-map storage, documentation and 68 independent allocation/geometry/
via/material assertions. The reviewed ceiling is now 212,374 in a separate
commit. The original baseline, counter, scope, archives and numerical gates
remain unchanged. The unchanged 400-group truncated census reports 58,118
estimated removable lines; its ceiling remains 58,394. The selected-group
change is not a source-reduction claim. Dedicated checks passed both Julia
versions; full package, strict quality and hosted verification remain required.
