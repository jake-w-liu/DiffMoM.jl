# Scaling arithmetic range failures

The independent 4608-bit affine oracle uses the exact represented Float64
inputs. The pre-fix helper moves two symmetric endpoints to 1 and 2 instead
of approximately .5 and 1.5, produces 1.11022 instead of approximately 2 for
strong contraction, and changes a stationary subnormal coordinate.

The exact old helper excerpt, original producer, source SHA-256 identity,
inputs and failed outputs remain retained. The current regression executes
that excerpt as an old-behavior control, checks current results against the
independent oracle, and verifies exact fixed-point classification. These
are arithmetic checks, not native EM or continuum accuracy claims.

All payloads, including this readme, are covered by `sha256.toml`. Existing
native SCUNI tests separately retain full-S, original-voltage, geometry,
port, ownership and zero-allocation gates for the current implementation.

The retained post-repair resource probe measures zero cumulative Julia GC
allocation for 100,000 normal coordinates and 6,608 bytes for one exceptional
coordinate. It measures cumulative allocation, not peak process RSS. Its
implementation source hashes are stable before and after measurement.

`decimal_native/` retains two actual Sonnet 18.53 SCUNI parameter/literal
pairs with decimal references .4 and .6 and a center .5. The candidate exact
SI-center guard incorrectly rejected both; native matrices are bit-identical.
Current lowering preserves the native rounded center and exact subnormal
fixed points. The candidate source and failed reader output remain retained.
The independent literal physical controls pass the unchanged .005 full-S and
1e-9 voltage-residual gates with errors .00017173 and 3.11e-11, respectively.
