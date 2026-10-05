# Conformal GEOVAR resource forwarding

The public conformal lowerer passed its geometry transform the default budget
instead of the caller's max_bytes. On unchanged d7c84421, valid whole-polygon
GEOVAR inputs with 5, 513 and 4,097 vertices allocated 90,520, 431,304 and
1,694,683 warm cumulative bytes before rejecting max_bytes=1 at the later
metadata guard. The original capture is retained at
`data/sonnet_conformal_transform_budget_before_v1_20261005.toml`.

A one-keyword process-only prototype forwards max_bytes to the existing
geometry guard. The same inputs reject before transforming or copying source
vertices, allocating 2,784, 2,816 and 2,816 bytes. This measures cumulative
allocation rather than peak resident memory. The implementation makes that
same one-keyword change; it does not alter transforms, meshing or formulas.

Twenty new assertions establish valid transformed inputs, rejection at the
geometry workspace guard, bounded warm rejection and unchanged source vertices
and port coordinates. Both Julia 1.12.7 and 1.13.1 pass those assertions plus
all 484 existing native port-attachment, undriven-wall, provenance and metadata
resource assertions, with bounds checks enabled. Existing physical full-S
reference gates and numerical tolerances remain unchanged.

The preceding combined PEC/radiation commit is undergoing immutable full
package validation. This local change is checked separately on both versions;
the next GitHub run must validate its exact final head on every platform,
documentation and the strict quality contract before any green CI verdict.
