# Native port-zero (grounded terminal) probes

Fixtures captured from Sonnet Lite 18.53 (`C:\Program Files\Sonnet
Software\18.53\bin\em.exe`) for the documented port-number-zero grounding
rule (`doc/users_guide/UsingPortNumberZeroforGroundConn.html`): numbered-0
ports simulate as normal ports, are de-embedded like any other port, and are
then shorted out of the reported S-parameter network.

## Files

| File | Case | Lite 18.53 verdict |
| --- | --- | --- |
| `interior.son` | port 0 on a shared interior edge + wall port 1 | **License-blocked**: "Sonnet Lite does not allow use of grounded ports with number 0" (`native_lite_port0_license.log`) — grammar accepted before the license wall |
| `merged.son` | same geometry with the two pads merged into one polygon | solves; the port-zero interior short must match this circuit |
| `open_gap.son` | same geometry with the port-0 record removed | solves identically in the raster model (shared edges already conduct) |
| `via.son` | port 0 on a via polygon + wall port 1 | license-blocked intake (same limitation class) |
| `via_open.son` | identical via polygon with the port record removed | solves; a shorted via port equals a portless via exactly |
| `wall.son` | port 0 on a west box-wall edge + wall port 1 | license-blocked intake |
| `all_zero.son` | every port numbered 0 | license-blocked intake; engines must reject the empty external network loudly |
| `component_grounded.son` | `component_lpf` example, first capacitor's second `SMDP` terminal renumbered to 0, remaining terminals resequenced | **Lite solves** — the port-zero license check does not cover component terminals; S-parameters differ from baseline (the capacitor leg grounds to box) |
| `component_baseline.son` / `component_grounded_nodrp.son` / `component_baseline_nodrp.son` | same component variants with `DRP1` records stripped for raw comparisons | Lite solves; `*_native_response.log` capture the raw de-embedding-free S-parameters |

## Native evidence highlights

- `native_lite_port0_license.log`: Lite intake accepts the records and then
  hits its license limitation on numbered-0 ports — the documented grammar is
  legal, the *solver feature* is Lite-gated.
- `component_grounded_nodrp_native_response.log`: Lite *solved* an
  `SMDP`-terminal-0 component, so component-pin grounding is a native
  capability that our component path reproduces (port shorted to node 0).
- Sequence rule (Lite-verified): component `SMDP` terminal numbers share the
  project port sequence, but a terminal `0` is extra — it does not consume a
  sequence number (`{3,0,4,5,...}` validates where `{0,4}` fails with
  "non-sequential port numbers").
- Pin index (`SMDP` field 7) remains strictly positive/sequential; `0` is
  rejected natively with "non-sequential terminal numbers".
