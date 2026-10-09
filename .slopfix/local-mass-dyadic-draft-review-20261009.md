# Unqualified local-mass dyadic precision draft

Confirmed on both Julia versions: exact BigFloat inputs (2^6710+1,-2^6710)
times a valid local operator should produce a finite one but the fixed6656
fallback drops it and produces zero. The gap follows existingfallbackprecision
plus IEEEprecision and one carry, rather than a selected new default.

This draft derives binary input exponent/precision ranges separately for
alpha, matrixvalues, x, beta and prior y. Triple and double product degree,
four/two real terms per complex product and reduction carry bound the exact
dyadic sum. Beta0 ignores prior output. Unsupported nondyadic types retain
the separate legacy path. This does not remove every selected fallback.

All52 new direct/adjoint/complex/BigInt/callerprecision/rounding/ownership/beta0
tests and104 original Test5 testassertions pass on both versions (156each),
with original ordinary/finite/nonfinite/underflow/overflow/gradient/alloc gates
and plain asserts unchanged. A first subsectionrunner omitted an original
allocationhelper; its155PASS1ERROR evidence is preserved and corrected by
including that exact helper, without changing production/tests/thresholds.
V3 separates factor ranges from V2's conservative global range. The tested
correct rare case drops209556 to81824 allocatedbytes on13; V3 is81824 on12.
Ordinary multiplication remains zero allocatedbytes. Earlier main71368bytes
returned the wrong zero, so it is not a correct optimization baseline.

This is a DRAFT. Raw-memory/resource guard/ownership design, nondyadic input
acceptance, the other fixed6656 callers (including Impedance and NearField),
interval2304, additional IEEE/state/robustness controls and full qualification
are still open. Code-growth measurement/ratchet has not been updated. No
native/docs/static/import/proper/full/currentmain/hosted PASS is claimed.
Do not merge this draft into main. Its branch and sources exist to support
the next agent without losing the confirmed bug or tested prototype.
