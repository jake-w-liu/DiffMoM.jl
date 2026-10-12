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
- [ ] B3 moved GEOVAR interior/via port attachment (`GeoVars:601`). Oracle: Lite.
- [ ] B4 overlapping/dependent GEOVAR dims ordering (`GeoVars:617,628`). Oracle:
      Lite probe corpus.
- [ ] B5 zero saved-offset non-coincident references (partially handled; extend).
      Oracle: Lite.

## C. Material loss / vias

- [ ] C1 non-rectangular thick-wall VOL topology adapter — inset
      crossing/saturation (`IO:1175-1179`). Needs real offset-with-topology
      algorithm. Oracle: Lite RF controls + analytic rectangle saturation
      limits already archived.
- [ ] C2 ARR sheet-resistance conversion (`IO:1218`). Oracle: Lite.
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
- [ ] C5 VOL RPV hollow wall thickness (`IO:1522`). Oracle: Lite.
- [ ] C6 TMM resistivity/sheet-resistance + unknown flags (`IO:1612`);
      further TMM semantics (face counts). Oracle: Lite.
- [ ] C7 lossy via-cover horizontal pad loss (`IO:1514`). Oracle: Lite.
- [ ] C8 dielectric loss/anisotropy flags (`IO:1702`). Oracle: Lite/HFSS.
- [ ] C9 NOR remaining semantics: general current ratio, SEN reactance,
      plated/roughness lowering, face counts (parity M5 row). Oracle:
      Lite probes + HFSS plated-line solve.
- [ ] C10 `surface_zs`/`via_sigma` override coexistence policy (`IO:2191-2192`).
      Oracle: analytic contract.

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

- [ ] E1 non-(:box,:std,:gap,:via) port kinds in raster (`IO:1965`) and
      conformal (`Conformal:341`). Oracle: Lite `-N` + solve.
- [ ] E2 CUP calibration-group solving — records parse; coupled pin-group
      calibration license-blocked. Nexxim-equivalent + DocOnly residual.
- [ ] E3 non-PEC metal material index path (`IO:1849` partial). Oracle: Lite.
- [ ] E4 unsupported project types (`IO:231`), SMD-components-in-plain-lowering
      path (`IO:1818`). Oracle: Lite probes.
- [ ] E5 dielectric bricks (`IO:1829`, `RasterResources:44`,
      `Conformal:181-185` incl. BAR). Oracle: Lite/HFSS.
- [ ] E6 scalar `+` exponent sign (`ScalarFiles:373`). Oracle: Lite parse.

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
6. B-group GEOVAR residue (B3/B4/B5 tractable; B2 probe campaign).
7. A1 REFPLANE contracts — needs calibration machinery, larger.
8. D-group STF residue — `em -N` structural corpus.
9. F/G-group netlist/layout breadth.
10. I-group HFSS-based independent convergence certification.

Each landed item: implementation + deep-debug loop + tests + fixture evidence
+ ledger/docs update + full suite (1.13) + focused (1.12) + push + CI green.
