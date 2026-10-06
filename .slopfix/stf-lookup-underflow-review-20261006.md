# STF lookup coordinate range guard

The public exact-node provider converted finite Real coordinates to Float64
without preserving nonzero values. A positive BigFloat or Rational value
of 1e-400 therefore became zero and selected a valid zero-key material node.
The scalar literal and variable providers already reject that underflow.
The new guard applies the same requirement to lookup coordinates before
accessing the node dictionary. Ordinary stored-key rounding, signed-zero
keys, representable subnormals and returned-value ownership are preserved.

The initial reproduction captured the wrong zero-node result for positive
BigFloat and Rational inputs on Julia 1.13.1 and 1.12.7. Negative inputs
became -0.0 and missed the initial table's +0.0 key, so its all-cases summary
was too strict. Those original rows and logs remain; an independent review
confirms the two positive cases per version. New controls also include
actual signed-zero nodes and both coordinate positions of two-axis tables.

The isolated fix passes 66 new controls and all 229 original technology
assertions on both versions. The combined registered patch preserves the
original test file as an exact byte prefix and all 35 copied fixture inputs.
Original native full-S and physical voltage-residual gates remain unchanged.

The first guard introduced 64-160 extra allocation bytes in two-coordinate
calls. A coordinate-count specialization and tuple map remove that overhead.
Five warmed samples per healthy case retain complete output bits. Ordinary
two-coordinate samples fall from 256/496 to 64 bytes on Julia 1.13.1/1.12.7;
mixed BigFloat/Float64 calls fall from 288/384 to 64 bytes. One-coordinate
ordinary/BigFloat samples remain 64 bytes and subnormal samples remain 96.
These are measured public lookup calls, including their returned vector;
they do not establish runtime speed, peak memory or universal optimization.

Invalid counts 0, 3, 100 and 1000 reject on both versions. Their captured
warmed allocations decrease. First-call compiler costs are preserved for
both implementations and include the diagnostic wrapper; no large new
compilation/resource regression was confirmed by that probe.

All prototype sources, staged tests, original probes and allocation captures
are retained with actual source/input hashes. Actual candidate tests, quality
measurements and publication still must validate this adopted source. The
fix does not implement STF interpolation or physical etch/rho/RPV bias
application, and those broader requirements remain open.

The actual adopted candidate passes all 295 registered assertions on both
versions. The pinned Julia 1.12.7 counter measures 213056 code lines, exactly
31 above the published parent's 213025 ceiling. The original strict
measurement failure is retained. Blocking smells pass and the bounded
400-group duplication estimate remains 58086, below the unchanged 58394
ceiling. The implementation and its exact ceiling adjustment are separate
commits; full-package and hosted verification remain required.
