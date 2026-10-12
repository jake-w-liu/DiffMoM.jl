# Native DRP1 reference-plane grammar probes (Sonnet Lite 18.53)

All `.son` files were driven through the installed `em.exe` (Sonnet Lite
18.53) on 2026-10-12. Native output lives in per-case `*/em_out.log`
(engine stdout) and `*/log_response.log` (solved S-parameters).

## Decoded grammar

`DRP1 <edge> <type> [field]` appears at GEO top level (before `NUM`) and
declares the automatic de-embedding reference plane for every box-wall
port on that edge. Keywords are case-insensitive (`drp1 left fix 1000`
solves). The first `DRP1` record per edge wins (`d_dup`: `FIX 1000` then
`FIX 2000` is bit-identical to `FIX 1000` alone).

| type  | form                                      | native behaviour |
|-------|-------------------------------------------|------------------|
| FIX   | `DRP1 <edge> FIX <len>`                   | len<=0 inert (bit-identical to no DRP); 0<len<=5 half-cells → "Reference plane for port N on <edge> box wall is too short."; larger → reference plane shifted by len (real S transform through the port feed line; port Eeff/Z0 appear in log_response.log) |
| NONE  | `DRP1 <edge> NONE [callen]`               | bare or callen<=0 inert (bit-identical); callen>0 changes the port-discontinuity calibration (~2% delta at 2.5 GHz for callen=1000 mil) |
| LINK  | `DRP1 <edge> LINK` + `POLY <id> 1` + `<v>` | continuation records are mandatory; bare LINK → "Missing Parameter: Ref. Plane" intake error; with continuation the plane links to the polygon vertex (moves with reshape) |

Reject grammar (all intake "Project File Error"): `FIX` non-numeric field
(`d_fixabc`), unknown type (`d_bad`: `BAD 100`), missing LINK continuation
(`d_linkbare`). `DRP2`, `REFPLANE`, `TWTYPE`, `TWVALUE` never occur in any
installed example's GEO section.

## Physics (not implemented — loud adapter rejection)

FIX>5 half-cells embeds a length-`len` section of the port's feed
transmission line between the box wall and the new reference plane. S11
magnitude changes (the feed line is not 50 Ω — log_response.log prints
per-port `Eeff`/`Z0`, e.g. 87.46−j4.29 Ω), S21 rotates ≈ β·len. The
transform needs Sonnet's solved per-port feed-line parameters, which
require a port-characterization solve this engine does not perform.
Positive `FIX`, positive `NONE` callen, and `LINK` therefore raise a
loud `ArgumentError` at `sonnet_planar_problem`; nonpositive values are
accepted as verified inert.

## Files

- `drp_base.son` — 2-port PEC-strip project, box 20000×20000 mil,
  160×160 half-cells (250 mil cells), ports on left/right walls.
- `drp_fixm500/d_FIX_0/d_NONE/d_NONE_4.son` — verified-inert variants.
- `drp_fix500/600`, `d_f625.son` — natively "too short".
- `d_f630/650`, `drp_fix1000.son` — active FIX (adapter rejection here).
- `drp_none500/1000.son` — active NONE calibration length.
- `d_dup.son` — duplicate edge, first record wins.
- `d_lc.son` — lowercase keywords accepted.
- `d_link.son` (LINK + POLY/id/vertex), `d_linkbare.son` (bare LINK →
  intake error), `d_fixabc.son`, `d_bad.son`, `d_drp2.son`.
- `drp_base/`, `drp_fix1000/`, `drp_none1000/`, `drp_fixm500/`,
  `drp_dup/` — native `log_response.log`/`em_out.log`.
- `r_f600/`, `r_x625/` — native too-short errors.
- `r_LINK/`, `r_FIX_abc/`, `r_BAD_100/` — native intake errors.
- `r_lc/`, `r_x630/` — native accepted runs.
