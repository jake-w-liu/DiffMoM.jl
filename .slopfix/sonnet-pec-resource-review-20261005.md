# Default PEC raster allocation repair

The public PEC lowerer allocated a dense zero material map for each sheet,
even though the returned PlanarProblem does not contain those maps. A
process-only prototype measured warm cumulative allocations on the unchanged
d7c84421 source. Omitting those maps reduced bytes from 156,584 to 139,713
at 32 square cells, 462,577 to 396,618 at 64, 1,809,174 to 1,546,431 at 128,
and 7,347,570 to 6,298,619 at 256. These are allocation measurements, not
peak RSS or portable speed claims. All problem arrays matched; the 32-grid
full S and currents were exactly equal. Source and caller fingerprints stayed
unchanged during that process-only experiment.

The production lowerer now materializes cell maps only when material
semantics or detailed adapter output require them. PEC overlap, wall
connections, via-cover masks and source bases remain unchanged. Material
enabled paths retain their maps and reject conflicting sheet impedances.
The preflight estimate follows the actual retained map choice.

Both Julia 1.12.7 and 1.13.1 passed 68 dedicated assertions, including warm
allocation separation between plain and detailed lowering, identical masks
and bases, finite PEC via cover pads, full S/currents, caller immutability,
and the exact mixed-material overlap rejection. Their 52 preexisting raster
budget assertions also passed. Full package, quality and hosted checks are
still required. No original numerical gate, fixture, counter or archive changed.

Evidence: ignored data/sonnet_pec_material_map_probe_v1_20261005.toml,
the independent validator, and PEC focused v2/v3 logs in the main worktree.
The initial fresh-worktree startup failures are preserved; after dependency
initialization the corrected producers passed. The current main commit remains
d7c84421 while this work is isolated, and its GitHub run must finish before
this optimization is advanced and pushed to main.
