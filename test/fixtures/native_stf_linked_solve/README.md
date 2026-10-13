# Linked-STF materialized literal — native solve parity (D5)

Sonnet Lite 18.53 rejects solving *linked* STF projects outright
("Sonnet Lite does not allow use of linked STF files"), so linked-STF
physics parity is verified on the materialized-equivalent literal project.

- `rich.son` + `rich.stf`: Lite-legal linked pair — Normal metal models,
  Volume via model, Ring via mesh fill. `rich.son` carries `TLAYNAM ... INH`
  / `NOH` records resolved through `rich.stf`.
- `literal_lite.son`: hand-written literal equal to
  `sonnet_materialize_project(read_sonnet_linked_project("rich.son"))`
  record for record — same layer records, MET rows (modulo display ids)
  and per-polygon level/material/target/flags/vertices. The regression
  testset asserts this equivalence so the literal cannot drift from the
  materializer.
- `literal_lite.native.s2p`: actual `em literal_lite.son` output on Lite
  18.53 (raw, not de-embedded), logged in `literal_lite.native.log`.
  Replays inside a 1e-3 complex-S gate at 1 and 2 GHz
  (observed |ΔS| ≤ 6.4e-4, dominated by a common-mode port-plane phase
  slope; magnitude agreement ~2e-5).

Native license gates decoded on the same literal (kept as evidence):

- `literal_tmm_vertices.son` (VOL via + TMM metal + VERTICES fill):
  rejected at solve — "via polygon which is not set to Ring meshing fill".
  Lite restricts *solved* vias to Ring fill even though `em -N` meshes
  every fill flag.
- `literal_tmm_ring.son` (TMM metal + RING fill): rejected at solve —
  "thick metal polygon(s)". Lite solves no thick-metal models.

These solve-time gates mean TMM-containing and non-Ring-fill materialized
projects have no Lite-verifiable solve oracle; their materialization
remains byte-verified against `soncmd -LoadTechnology` and their physics
is a DocOnly residual under the Lite license.
