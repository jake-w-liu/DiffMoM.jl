# Bound native TMM expansion before cloning

On unchanged 92eea542, a valid driven conformal two-face TMM project builds
its expanded physical source before max_bytes=1 reaches the later stack guard.
For 2, 128 and 1,024 native layers, warm cumulative rejection allocations are
25,712, 423,141 and 3,282,681 bytes. The default-budget control succeeds with
three physical layers and external labels [1,2]. Sources and caller inputs
remain unchanged in data/sonnet_tmm_budget_before_v1_20261005.toml.

The verified process-only prototype checks for TMM without constructing its
polygon list. It rejects an unaffordable base stack before layer-scaled work,
then reserves the existing conservative source workspace before cloning.
The same cases allocate 3,008, 3,056 and 3,104 warm bytes. These are cumulative
allocation measurements, not peak RSS. The production guard and shared stack
forwarding match this prototype, retaining the original expansion arithmetic.
FLOAT passes its budget after reserving owned variables. The component path
passes the remaining budget after the current staged-file and variable
payloads; its existing earlier geometry guard remains. The latter path is
covered for consistency, not claimed as a separately reproduced late clone.

Forty-seven new assertions cover supported conformal, FLOAT and component
controls, 128/1,024-layer rejection, source-workspace rejection, non-TMM
identity and unchanged input layers, materials, vertices and port values.
Both Julia 1.12.7 and 1.13.1 pass these and all 82 existing native geometry
assertions, including physical TMM face positions, impedances, source
contraction, upward/downward expansion and unsupported sheet-count rejection.
Original physical gates, source archives and tolerances are unchanged.

The pinned Julia 1.12.7 counter measures 212,584 code lines, a reviewed
increase of 82 over the 212,502 ceiling. Integrity/warnings are empty and
blocking smells are zero. The unchanged 400-group census estimates 58,090
removable lines, below its 58,394 ceiling. The repair and exact line-ceiling
adjustment are separate commits. Immutable full-package results and a new
exact-main all-seven-job CI result remain required before a green verdict.
