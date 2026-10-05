# Native raster resource repair

The ordinary native geometry wrapper enforced `max_bytes` only at the
numerical solver. With a one-byte budget, the unchanged before capture
allocated 463,313 / 7,348,306 / 26,894,066 bytes on 64 / 256 / 512 square
grids before rejecting. These are warm cumulative allocations, not peak RSS.
The current public solve rejects using 2,080 bytes at all three sizes; the
direct lowerer rejects using 2,096 bytes. The numerical-only 8,699,952-byte
budget also accepted by the old wrapper is now rejected because the wrapper
must retain its geometry and external terminal-response storage.

The lowerer validates source and grid budgets before transformations, checks
actual physical sheet levels and via spans before cell arrays, checks raw
source/contraction storage before ports, and counts basis storage before
construction. The wrapper reserves retained geometry, scalar snapshots,
balanced terminal response and calibration workspace from the numerical
solver's budget. Borrowed project arrays and shared owned arrays are counted
once. Component/floating callers forward their remaining budget; model files
loaded during a component call are included in that remaining amount.

The new 52 assertions cover tiny/invalid/overflow budgets, grid-size allocation
independence, caller immutability, copied wall vertices, numerical-only versus
aggregate budgets, and unchanged S-parameters/currents with adequate budgets.
Focused bounds-checked suites passed on Julia 1.12.7 and 1.13.1, guarding 17,148
input paths before/after, including scalar tables, physical component returns,
floating sources, native geometry dimensions and common-return S/nodes. The
final model-file forwarding check additionally passed 483 assertions on 1.13.1.
Full package tests and a fresh hosted workflow are still required for this
commit; the preceding all-seven-success hosted run applies only to 34758863.

Evidence remains under ignored `data/`: the immutable plain-raster before v2
capture, aggregate-budget before v2, raster-budget after v1, focused v2 reports,
and model-file focused v1 log. Failed earlier smoke/capture/focused producers
are preserved, including the missing test-helper error; they are not relabeled.
Original numerical fixtures, tolerances, archival failures, counter identity,
line-count scope and the original reduction baseline remain unchanged.
