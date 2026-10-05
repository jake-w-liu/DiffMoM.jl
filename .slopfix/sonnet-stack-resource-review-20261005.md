# Native layer-stack budget enforcement

On unchanged bb3f81be, supported FLOAT CUP and driven conformal projects with
2, 128 and 1,024 layers construct the physical stack before a later tiny-budget
rejection. With 1,024 layers the warm rejection allocates 12,016,979 bytes for
FLOAT at max_bytes=2048 and 15,770,171 bytes for conformal at max_bytes=1.
The unchanged-source capture is retained in
`data/sonnet_layer_budget_before_v2_20261005.toml`.

A process-only prototype reserves the original and converted ComplexF64 layer
payloads plus promotion references and vector growth before layer construction.
FLOAT forwards its remaining budget after reserving owned scalar variables.
The same large inputs reject at the stack guard with 3,144 and 3,008 warm
allocated bytes. These figures measure cumulative allocations, not peak RSS.
The production patch uses the same guard and forwarding. It changes no material
evaluation, stack ordering, physical formulas, geometry transforms or mesh gates.
This guard runs after GEOVAR and TMM geometry lowering; it does not establish a
complete memory bound for those preceding transformations.

Twenty-nine new assertions cover supported default-budget controls, the exact
stack-budget boundary, 128/1,024-layer rejection, bounded warm allocation and
unchanged source records, vertices and ports. The existing one-byte wall-copy
test now expects the earlier stack error. Its original allocation and source
integrity gates remain, and an added 384-byte case still exercises the metadata
guard before wall normalization. Julia 1.12.7 and 1.13.1 pass all 534 focused
assertions with bounds checks enabled.

The pinned Julia 1.12.7 quality counter measures 212,502 code lines, a reviewed
increase of 57 over the existing 212,445 ceiling. Blocking smells remain zero;
the unchanged 400-group duplication census estimates 58,106 removable lines,
below its 58,394 ceiling. The code fix and the exact line-ceiling adjustment
are separate commits. Exact-head full-package and hosted CI verification are
required before publishing a green verdict for this repair.
