# Sonnet Parity — Open Items Tracker

Authoritative work-item ledger for completing the Sonnet/RFIC parity audit.
Companion evidence: `docs/src/advanced/sonnet-parity.md` (+ `-numerics.md`),
`data/HANDOFF-20261009.md` (session journal), native fixture archives under
`test/fixtures/`.

## Rules of engagement (mandatory, per item)

1. **Deep-debug gate** — every item must pass a fresh deep-debug loop before it
   is marked done: domain correctness (does the semantics match the oracle?),
   implementation correctness, numerical robustness (range/precision/overflow),
   memory/resource behavior (payload budgets, owned snapshots), performance
   where applicable, regression tests, and a recheck after each fix.
2. **Oracle choice** — each item names its verification oracle:
   - `Lite` — Sonnet Lite 18.53 `em` engine (full solve) or `em -N`
     (license-free structural resolution).
   - `HFSS` — installed Ansys EM 2025.1 via `ansys.aedt.core` (PyAEDT 1.9.0,
     verified launching headless). Independent physics oracle; cannot decode
     `.son` semantics, so it validates *physical equivalence* of a materialized
     structure, not grammar.
   - `Nexxim` — `nexxim.exe` circuit solver (S-domain) for network-attach and
     calibration law checks.
   - `Analytic` — closed-form/published reference.
   - `DocOnly` — truly blocked by license/undecoded semantics; keep honest
     explicit rejection + ledger entry. HFSS substitutes wherever physics is
     transferable.
3. **Honesty** — an item is `done` only with evidence (test module + fixture
   archive + ledger update). Approximations and residual error stay visible.
4. **Compatibility** — Julia 1.12 + 1.13, clean worktree, push to `main`,
   hosted CI green after each landed item.

Status marks: `[ ]` open · `[~]` in progress · `[x]` done · `[D]` DocOnly
(blocked — explicit rejection retained).

---

## A. Component / circuit grammar

- [ ] A1 component REFPLANE contracts — DRP1/DRP2/TWTYPE/TWVALUE/REFPLANE
      records currently reject ("reference-plane/terminal adapter"). Oracle:
      Lite (probes on coupled calibration), Nexxim for de-embedding law.
- [ ] A2 zero/ground SMD return terminals (`PlanarSonnetComponents.jl:270`)
      and interior port-zero ground-return adapters (IO:1967/1999, Conformal:365).
      Oracle: Lite.
- [ ] A3 ordered model pins + coupled pin-group calibration
      (ModelFiles:369, parity lines 148/159/445/726). Lite-blocked (GLG license)
      → Nexxim circuit equivalence + DocOnly on the licensed part.
- [ ] A4 signed file-model pin terminals — degenerate-port adapter for `-pin`
      SPARAM/SPROJ attaches. Oracle: analytic (port-reduction identity), Lite
      if a file comp ever runs (license-blocked) → Nexxim.
- [x] A5 multi-pin (>2 unpaired) FLOAT common — DONE. Model-bearing comps
      with ≥3 unpaired positive pins attach each pin as an open-edge port
      on the shared local ground (per-pin star; EM edge == model port).
      Lite-verified that native rejects multi-pin ordinary components, so
      the star is scoped to model-bearing comps; staged↔callback agree at
      ~1e-18. Regression: `test_planar_sonnet_model_files.jl` (3-pin
      SPARAM + via SPROJ mixed project, signed-label and IDEAL rejects).
- [x] A6 diagonal native gap port — DONE as parity-verified rejection.
      Lite 18.53 itself rejects diagonal ports ("Illegal circuit
      configuration. Port is diagonal.") at both 45° and shallow slopes;
      DiffMoM throws the equivalent ArgumentError. Probes archived in
      `test/fixtures/native_diag_port_probes` + replay testset in
      `test_planar_sonnet_io.jl` (complements the conformal-route fixture
      `native_internal_diagonal_rejection`).
- [ ] A7 component-direction co-calibration residual ~0.01 (parity line 187/1167).
      Oracle: HFSS independent solve of materialized coupon.
- [x] A8 unsupported IDEAL component kinds/units — DONE. Lite probe corpus
      (~30 value-bearing projects) decoded the full native unit tables:
      RES GOH=1e9/TOH=1e12, CAP MF=1e-3, trailing-S plural aliases
      (KOHMS/UFS/MHS...), case-insensitive spelling. Implemented in
      `_sonnet_component_scale`. Known divergence (documented): native
      silently defaults unrecognized units (AF→PF, GH→NH, K→OH); the
      adapter keeps explicit rejection. Native rejects non-RLC IDEAL
      kinds — parity. Fixtures: `test/fixtures/native_ideal_units` (12
      value-bearing .son+.native.s2p) + replay testset.
- [x] A9 unsupported TYPE models — DONE as parity-verified rejection.
      Probed 12 candidate kinds (VENDOR/CKT/MODEL/S2P/TLINE/RLGC/CTL/SML/
      NETLIST/MDLX/COIL/CAPACITOR) on Lite 18.53: all parse-reject as
      Project File Error. Native grammar carries no other TYPE kinds in
      Lite; any undiscovered kind still rejects loudly in our parser.

## B. GEOVAR

- [ ] B1 GEOVAR header beyond `ANC/SYM/RAD XDIR/YDIR` point sets (`GeoVars:165`).
      Oracle: Lite probe sweeps.
- [ ] B2 explicit reference repetitions — scaled ANC moving-reference law
      (`GeoVars:228`, undecoded; flat 1.6e-4 residual basins documented).
      Unscaled SYM XDIR/YDIR/wildcard/literal already done bit-identically.
      Oracle: Lite `em -N` + raster probes; may stay DocOnly if law remains
      undecoded.
- [x] B3 moved GEOVAR interior/via port attachment — DONE. Lite probes
      (`fixtures/native_geovar_moved_port/`): POR1 STD and GAP on a
      shared edge whose polygon translates under an ANC YDIR GEOVAR
      solve with bit-identical S rows — the token is descriptive on
      moved interior ports too, and the port rides its polygon. The
      resolver re-anchors `values[6:7]` to the resolved edge midpoint
      and `solve_sonnet_project` reproduces the native matrix to ~6e-6.
      Removed the stale `:gap` exclusion in `GeoVars:599-604` (false
      rejection); moved CUP ports still route to the calibration-group
      adapter (E2 residual). The TMM `:gap` face-duplication gate
      (`IO:1781`) stays — Lite license-blocks all thick-metal polygons,
      so that path is DocOnly-verified conservative, not parity-testable.
- [x] B4 overlapping/dependent GEOVAR dims ordering (`GeoVars:617,628`).
      **Done** — Lite 18.53 probe matrix fully decoded the structural rules
      (`test/fixtures/native_geovar_dim_interactions/manifest.json`):
      moved-point sharing is rejected only for same-axis same-PS-side pairs
      of non-sequential dims (even when fully inactive — intake-level);
      cross-axis and opposite-side sharing compose freely; non-sequential
      same-axis ref-on-moved dependencies reject one-way and mutual alike
      while sequential anchored dims may depend on anything (resolve last).
      Inactivity is exact `target == NOM`, not `iszero(delta)` — ±1e-17
      SI-roundtrip phantoms no longer register false owners. Structural
      checks now run before the nominal-geometry early return, and the
      sequential-order edges are axis-aware. 13-assert regression testset
      `Native GEOVAR dimension interactions are axis- and side-aware`.
      Literal end-to-end parity: cross-axis sharing 8.3e-5 (within the
      SYM raster-law band). Residual: active ANC + non-sequential
      same-axis moved-point sharing is natively accepted but its joint
      application ordering stays undecoded after 15+ ordering probes —
      kept as a loud rejection (DocOnly), not a silent acceptance.
- [x] B5 zero saved-offset non-coincident references (partially handled; extend).
      **Done** — Lite 18.53 probe campaign
      (`test/fixtures/native_geovar_zero_offset/manifest.json`): NOM 0 is
      legal for every kind and header (ANC/SYM/RAD, NSCD + scaled), the
      coincident-REF gate and the ANC-only zero-NOM restriction were false
      rejections. The delta baseline is the geometric span, never NOM
      (bit-exact literal discrimination, ±350 each side landing). Inactive
      dimensions still resolve real displacements — an inconsistent
      inactive ANC collapses to its target (2.1e-6 vs collapse literal).
      So `return p` now requires every dimension to be a stored no-op:
      sub-cell deviations (~1e-9 relative) from file-write quantization
      and SI round-trips are phantoms under a `sqrt(eps)`-relative bound.
      Zero/negative targets also intake natively; degenerate resolved
      geometry still rejects loudly at emission. Residual: Sonnet
      tolerates zero-area polygons we cannot emit (loud gap, e.g.
      full-collapse targets).

## C. Material loss / vias

- [ ] C1 non-rectangular thick-wall VOL topology adapter — inset
      crossing/saturation (`IO:1175-1179`). Needs real offset-with-topology
      algorithm. Oracle: Lite RF controls + analytic rectangle saturation
      limits already archived.
- [x] C2 ARR sheet-resistance conversion (`IO:1218`). **Done** — Lite
      18.53 logs "Array via metal type converted from Sheet Resistance at
      DC to Conductivity": every finite entered loss — zero and negatives
      included — converts to a fixed σ=3.72e7 S/m independent of value,
      fill factor, and via height (bit-identical against a CDVY σ=3.72e7
      control at two fill levels and two stack heights); INF still selects
      a perfect conductor. Selector is position 6; a token there that is
      not CDVY/RSVY/SRVY is inert density data. Axial element and endpoint
      pads both use the converted σ. Selector tokens are matched
      case-insensitively on ARR, VOL and NOR records, and an unrecognized
      token in the VOL/NOR selector slot resolves direct conductivity
      rather than rejecting (VOL `X` ≡ `CDVY` bit-identically). Evidence:
      `fixtures/native_arr_srvy` + `fixtures/native_metal_selector_case`
      + replay testsets (41 + 22 + 17 tests).
- [x] C3 ARR zero-conductivity open-via. **Done** — Lite 18.53: a
      zero-conductivity record is completely inert (no axial element, no
      COVERS/RING endpoint pads); `cdvy_loss0` is bit-equivalent to the
      absent-via control (~1e-6). Both raster and conformal import paths
      skip the polygon; a via port on one throws "lacks current bases".
      Evidence: `fixtures/native_arr_open_via` + replay testset.
- [x] C4 ARR conductivity/skin loss, non-RPV ARR frequency dependence.
      **Done for CDVY/RSVY** — native σ-sweep (σ=1e-6…1e6) showed endpoint
      pads carry the converted conductivity as a `two_sheet_zs` film at
      the via height (earlier "PEC pads" assumption was wrong); RSVY
      converts σ=100/ρ bit-identically; INF→PEC pads+column; RPV records
      carry no σ field so their pads stay PEC. SRVY residual stays in C2.
- [x] C5 VOL RPV hollow wall thickness. **Done** — Lite 18.53: the field
      is fully inert for RPV records (hollow and SOLID, five decades and
      negative values all bit-identical, zero warnings). The
      "dedicated adapter" rejection was removed; only non-finite values
      still reject. Evidence: `fixtures/native_vol_rpv_wall` (8 captures)
      + replay testset in `test_planar_sonnet_endpoints.jl`.
- [~] C6 TMM selectors/flags/face counts. **License-blocked** — Lite
      rejects any thick-metal polygon before solving, so flag semantics
      (RSVY/SRVY/unknown — all parse silently) and >2-face expansion are
      undecodable natively. Verified boundary: `nsheets=1` fails at
      config level ("must be 2 or more"), nsheets≥2 parses. Rejections
      stay; HFSS cannot decode Sonnet flag meaning.
- [x] C7 lossy via-cover horizontal pad loss. **Done as verified
      rejection** — Lite parse-rejects RPV via records on non-VOL/ARR
      metals (SUP/NOR/RES probes all `Project File Error`, COVERS or
      not), so no legal record can reach the pad-loss adapter. Also
      plugged a NOCOVERS silent-solve hole: the importer now rejects the
      class outright instead of computing an unverified number.
      Evidence: `fixtures/native_vol_rpv_wall/rpv_{SUP,NOR,RES,SUPnc}`.
- [x] C8 dielectric loss/anisotropy flags (`IO:1702`). **Done** — full
      positional grammar and selector law decoded bit-identically in Lite:
      the record is `t er mur dtan mtan eloss f7 name flags...` where the
      eloss slot doubles as the short-form name (five-field + name or six
      bare numerics close the record), `f7` is a parsed-but-inert
      anisotropy slot that must stay a numeric expression, and exactly the
      first token after the name is the selector — `RSVY` reads ohm-cm
      (`sigma=100/rho`), `SRVY` reads ohms-per-square through the layer
      thickness (`sigma=1/(Rs*t)`, same law as metal SRVY), `CDVY` and any
      unknown token select plain conductivity, later tokens inert, all
      case-insensitive. `INF` is a literal: plain-conductivity INF is the
      degenerate all-conductor layer (reject), while reciprocal selectors
      map it to sigma=0. Trailing anisotropy numerics are parsed but inert
      in Lite (bit-identical with ErelZ=5, full rows, sigma_z=100), so the
      selector position stays `row[9]` regardless of numeric count.
      Negative fields reject natively; zero under RSVY/SRVY is the same
      degenerate conductor layer and also rejects. Evidence:
      `fixtures/native_dielectric_srvy` (30 captures, five bit-identical
      SRVY/sigma pairs + selector-order + parse-boundary rejects) and the
      replay/selector testset in `test_planar_sonnet_dielectric_scaling.jl`.
      Remaining sub-parity: sub-six-token legacy short records still
      reject at the reader floor.
- [x] C9 NOR remaining semantics: general current ratio, SEN reactance,
      plated/roughness lowering, face counts (parity M5 row). Oracle:
      Lite probes + HFSS plated-line solve.
      **Done** — current-ratio r↔1/r symmetry and SEN reactance
      (positive/zero/negative) verified against Lite within the .06 gate.
      Covers correctly force one-sided current (native strings: "Current
      ratio set to infinite/zero for metal on top/bottom box cover").
      Plated/roughness lowering is the `RUF` (RoughMetal) record family,
      a thick-metal class: Lite rejects it on sheets ("thick metal polygon
      touching the top of the box") and is license-blocked like TMM/TVM.
      **Bonus decode — `NAT` (Rdc/Rrf) sheet metal implemented**: two
      literal numerics (Rdc Ω/sq, Rrf Ω·√s) joined by the conductor-slab
      loss transition; verified against the installed `mems_switch`
      example's `NAT 0.035 1.287e-05` Nickel record. Missing/unparseable
      entries warn→0, trailing tokens inert, signed entries normalize to
      magnitudes, INF is an open film/cover, and `NAT`-on-via parse-rejects
      natively like ours. Evidence: `fixtures/native_nat_loss` (15
      captures) + "native NAT Rdc/Rrf sheet loss" testset (140 tests).
- [x] C10 `surface_zs`/`via_sigma` override coexistence policy (`IO:2191-2192`).
      **Done as verified contract** — `solve_sonnet_project` is the
      project-faithful path; caller `surface_zs`/`via_sigma` keywords
      reject with a clean ArgumentError (no defined coexistence for two
      impedance sources on the same cells). The sanctioned override route
      is `sonnet_planar_problem` + `solve_planar`: passing
      `model.sheet_zs`/`model.via_sigma` reproduces the full-path solve
      bit-exactly, and a custom `via_sigma` field measurably changes the
      response (contract testset in `test_planar_sonnet_io.jl`, 4 tests).
      Policy documented in the `solve_sonnet_project` docstring.

## D. Linked STF

- [ ] D1 applied etch-bias / rho / RPV tables on inherited layers
      (`Technology:822`). Oracle: `em -N` structural where resolvable.
- [ ] D2 STF metal model classes: TrueVolume, Surface, + unknowns
      (`Technology:686`); roughness (`:689`); conductor class (`:410`).
      Oracle: `em -N`.
- [ ] D3 STF via model classes: Surface, Array, volume fill-factor/density,
      sheet-resistance (`Technology:742-753`); via NOH inheritance (`:844`).
      Oracle: `em -N`.
- [ ] D4 STF mesh controls: anisotropic subsectioning, bounds, edge-mesh,
      pads, maxlength (`Technology:656-666`); metal mesh fill kinds (`:713`),
      via mesh fills (`:792`). Oracle: `em -N` subsection counts.
- [ ] D5 native linked-STF EM solving — Lite `-S` gate blocked; HFSS cannot
      decode STF. Verify materialized project physics vs HFSS where a coupon
      is buildable. Partly DocOnly.
- [ ] D6 encrypted material entries — DocOnly (cannot decrypt).

## E. Ports / calibration / project intake

- [x] E1 non-(:box,:std,:gap,:via) port kinds in raster (`IO:1965`) and
      conformal (`Conformal:341`) — DONE. Lite probes
      (`fixtures/native_port_kind_token`): a token after `POR1` is required
      but descriptive — `XYZ`, `VIA`, `AGI`, `gauto`, `all`, `XYZ123` all
      type by placement and solve as "Standard port" (log annotation) on
      sheet-edge, box-wall and via-polygon ports; `VIA`/`STD`/`XYZ` on the
      same via produce bit-identical output including an interior-
      terminating via (`v3_*`, 3-layer `TOLEVEL 1`). Bare `POR1` and
      group-less `CUP` are native parse rejections. The group structure,
      not the token, is what carries calibration semantics: a `CUPGRP`
      block after ANY token is accepted and de-embeds the port, and a
      single-port group solves natively in Lite (`g2_cup`,`xg2_cup`);
      a two-port group with only a trailing CUPGRP rejects natively
      (`mp_cup`). Fixed: geometry-keyed routing in raster+conformal port
      loops, the raster port-workspace estimator, the GEOVAR moved-port
      re-anchor, the TMM face-duplication gate; removed a false "standard
      via port must terminate on a box cover" rejection; every grouped
      port still reaches the co-calibrated adapter rejection (E2).
- [ ] E2 CUP calibration-group solving — records parse. Refined boundary
      (E1 probes, `fixtures/native_port_kind_token`): a CUPGRP block after
      ANY POR1 token is accepted and de-embeds the port; a single-port
      group SOLVES in Lite (not license-blocked). A two-port group with a
      trailing shared CUPGRP rejects natively ("Calibration group
      information not defined") — the correct multi-port group grammar
      still needs decoding. Remaining: group de-embed implementation.
- [x] E3 non-PEC metal material index path — DONE. Lite probes
      (`fixtures/native_material_index`, 35 cases): sheet↔via metal-type
      cross-references natively reject as "Material ID refers to the wrong
      material type" (sheet+VOL/ARR; via+NOR/RES/SEN/SUP/NAT/TMM/RUF/TVM);
      indices outside `0:nmetal-1` including `-2` reject; `-1` is the PEC
      default on sheets AND vias (replay bit-level `.06`-gate agreement);
      cover-row index fields are inert byte-identically; cover rows carrying
      via-metal records are degenerate-accepted natively and stay a loud
      local rejection. Fixed a BoundsError→ArgumentError on out-of-range
      sheet indices (`IO:1851`).
- [x] E4 unsupported project types (`IO:231`), SMD-components-in-plain-lowering
      path (`IO:1818`) — DONE. Lite probes: the `FTYP` token is
      content-routed, not type-routed — `SONNETPRJ` on a GEO file runs
      the EM solve and `SONPROJ` on a CKT file runs the netlist solve
      (both verified end-to-end; only `SONPROJ`/`SONNETPRJ` exact-case
      are valid tokens). Fixed `IO:243` to route on section presence
      (GEO → geometry; else CKT → netlist) instead of the token,
      removing false rejections in both directions. SMD audit:
      `sonnet_planar_problem` (`IO:1960`) and `sonnet_conformal_layout`
      (`Conformal:204`) reject components loudly while
      `solve_sonnet_project` (`IO:2354`) routes to the verified
      `_solve_sonnet_components` path — complete, no changes needed.
- [ ] E5 dielectric bricks (`IO:1829`, `RasterResources:44`,
      `Conformal:181-185` incl. BAR). Oracle: Lite/HFSS.
- [x] E6 scalar sign grammar — DONE. Lite probes (60-case accept/reject
      matrix): `^` operand allows one `-` but never `+`, and a sign's operand
      cannot itself begin with a sign even behind `(`/whitespace (`1^--2`,
      `1^-(-2)`, `-(-2)`, `--x`, `1^-((-2))` all natively reject while
      `1^(-2)`, `1^((-2))`, `1^-abs(-2)`, `2*(-3)`, `2--3`, `2-(-3)` parse).
      Unary `+` is not a grammar token at all: `+2`/`" +2"`/`+2e3`/`+.5`
      evaluate as whole-string decimal literals only (`+x`, `+2+3`, `+ 2`,
      `(+2)`, `2++3`, `abs(+2)` reject). Implemented: shared sign-operand
      head scan + leading-literal fast path in `ScalarFiles:369-475`;
      regressions in `test_planar_sonnet_functions.jl`.

## F. Spectre / SPICE / netlists

- [ ] F1 other SPICE line classes (parity line 128) — T/O done; L/K/W/etc.
      Oracle: published line-model equations + Nexxim.
- [ ] F2 broader Spectre grammar + vendor-library adapters. Oracle: Nexxim
      netlist equivalence.
- [ ] F3 nonlinear/vendor model adapters. Oracle: Nexxim (linear subset),
      DocOnly nonlinear residual.

## G. Layout translation

- [ ] G1 DXF matched-grid parity (parity line 1460 mode4096 resource estimate +
      physical acceptance). Oracle: HFSS import of same DXF.
- [ ] G2 ODB++: vendor translator coverage, arbitrary layouts, continuum
      convergence (lines 833/858/985). Oracle: generated coupons + HFSS.
- [ ] G3 Sonnet artwork leaves: thermals, half-ovals, D-Pack, curved
      polygon-stroke (lines 913/928). Oracle: Lite GUI/BIT archives.
- [ ] G4 GDS maps / top-to-bottom stack order residual gaps (line 554).
      Oracle: generated coupons.

## H. FLOAT / ground-reference residual

- [ ] H1 FLOAT file-backed comps — implementation done; native acceptance is
      Lite-license-blocked extrapolation (parity line 180). Upgrade evidence:
      Nexxim circuit equivalence (EM+network attach law), HFSS EM of the
      stripped structure. Residual license claim stays DocOnly.
- [ ] H2 REFPLANE on FLOAT comps — tied to A1.
- [x] H3 multi-pin FLOAT file models — DONE via A5 (positive-pin star).
      Signed-pin variants remain under A4.

## I. Continuum / numerical acceptance

- [ ] I1 native continuum convergence beyond captured grids (lines 104/1064/
      1191-1222/1282; 80/160/320/640 raw full-S steps). Oracle: HFSS converged
      reference as independent continuum check.
- [ ] I2 conformal >1024-cell license limit (lines 1286); trapezoid
      refinement partial. HFSS substitute + DocOnly residual.
- [ ] I3 S/current native viewer license (line 1308) — DocOnly.
- [ ] I4 DXF mode4096 8 GB estimate (line 1460) — resource path decision.
- [ ] I5 peak-process-RSS / opaque-memory acceptance (lines 889/258-numerics).
- [ ] I6 iterative-solver restart defaults + broad spatial convergence
      (lines 1197-1222).

## J. License-blocked documentation ledger (DocOnly candidates)

- pin-group GLG calibration (A3 residual), SMD renderer compatibility,
  linked-STF native `-S` solve (D5), Conformal license (I2), S/current viewer
  (I3), encrypted STF materials (D6), undecoded scaled GEOVAR law (B2 if it
  resists probing).

---

## Execution order (current)

1. A5 multi-pin FLOAT common — analytic, self-contained, adjacent to fresh code.
2. A4 signed file-pin degenerate-port adapter — depends on A5 port-reduction
   machinery.
3. A6 diagonal gap port + E1/E6 intake items — small, Lite-verifiable.
4. A8/A9 component kinds/model types — Lite probe corpus.
5. C-group material items (C2/C3/C5 first — bounded), then C1 thick-wall.
6. B-group GEOVAR residue (B3 done; B4/B5 tractable; B2 probe campaign).
7. A1 REFPLANE contracts — needs calibration machinery, larger.
8. D-group STF residue — `em -N` structural corpus.
9. F/G-group netlist/layout breadth.
10. I-group HFSS-based independent convergence certification.

Each landed item: implementation + deep-debug loop + tests + fixture evidence
+ ledger/docs update + full suite (1.13) + focused (1.12) + push + CI green.
