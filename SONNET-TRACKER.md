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

- [~] A1 component REFPLANE contracts — DRP1/DRP2/TWTYPE/TWVALUE/REFPLANE.
      **Partially done (2026-10-12)** — GEO-level `DRP1` grammar decoded via
      Lite 18.53 probes and now fully validated at intake (case-insensitive
      keywords; `FIX <len>`, `NONE [callen]`, `LINK` + `POLY <id> 1` + vertex
      continuation; first record per edge wins; malformed records intake-fail
      like Lite). Verified-inert forms (`FIX<=0`, bare/`NONE<=0`) solve
      unchanged; active forms (`FIX` above the 5-half-cell floor with the
      native too-short diagnostic preserved below it, positive `NONE`
      callen, `LINK`) now fail loudly at problem build naming the missing
      feed-line de-embedding adapter — previously they were *silently
      dropped*, producing wrong physics. The active physics (embed `len`
      of the solved port feed line: log-exposed Eeff/Z0, e.g. 87.46−j4.29 Ω)
      needs a per-port characterization solve the engine lacks → DocOnly
      residual with full evidence archived. DRP2/REFPLANE/TWTYPE/TWVALUE
      at GEO level reject (unprobed); component/floating/model-file sites
      keep their existing loud rejections. Fixture+manifest:
      `test/fixtures/native_drp_reference_planes/`; regression: 36 tests in
      `test_planar_sonnet_io.jl`. Remaining: the actual feed-line de-embed
      transform (needs port characterization machinery), DRP2 semantics,
      SUBDIV-level REFPLANE.
- [x] A2 zero/ground SMD return terminals - DONE. Documented em semantics
     ("Using Port Number Zero for Ground Connections"): numbered-0 ports
     simulate as normal ports, de-embed like any port, then short out of
     the reported network. Implemented: wall zeros short their edge to the
     box (undriven, physically connected); via-polygon zeros reduce to the
     portless via (bit-identical S); interior zeros are solved normally
     with no contraction label - the all-zero row clamps terminal voltage
     to zero in the balanced solve. Conformal path mirrors all three.
     Component SMDP terminal 0 grounds the model port: ideal R/L/C pass
     circuit node 0, SPARAM/SPROJ/callback responses reduce via
     `_sonnet_ground_network` (Y row/col drop, Z Schur complement, S
     per-port power-wave short Gamma=-z0*/z0 - exact for complex refs;
     singular terminations reject). File-backed model pins accept label 0
     (Lite-verified solvable); FLOAT pin 0 stays a loud rejection pending
     the floating-common adapter (its common is not box ground).
     Lite evidence: POR port-0 grammar accepted then license-blocked
     ("Sonnet Lite does not allow use of grounded ports with number 0");
     SMDP terminal 0 *solves* natively (the license check does not cover
     component terminals) - component_lpf with a grounded capacitor leg
     produced shifted S-parameters. Fixture+manifest:
     `test/fixtures/native_port_zero/`; regressions: port-zero testset in
     `test_planar_sonnet_port_attachment.jl` (20), grounded-pin testset in
     `test_planar_component_returns.jl` (18), transactional test in
     `test_planar_sonnet_model_files.jl` (+3), FLOAT rejection testset (2).
     Remaining: wall port-0 + active DRP1 stays gated to the de-embedding
     adapter (documented short-shift combination), port-0 pin calibration
     interactions untested natively (Lite-gated).
- [x] A3 ordered model pins + coupled pin-group calibration
      (ModelFiles:369, parity lines 148/159/445/726). Ordered pins are
      implemented and Lite-verified at intake. The licensed multi-port
      CUPGRP solve stays Lite-gated ("Sonnet Lite does not allow
      calibration groups"), so the coupled-group kernel
      `deembed_cocal_group` is verified against an independent Nexxim
      2025 R1 oracle instead: a cross-coupled 4-port launch lattice, a
      3-port DUT, and their cascade were solved natively at 1/2 GHz;
      the launch's voltage/current chain built from the measured
      launch admittance recovers the DUT admittance from the measured
      composite to <=7.4e-5 relative error (fixture
      `test/fixtures/nexxim_cocal_group/`, testset "Nexxim coupled
      launch-group oracle" in `test_planar_calibration.jl`). Residual
      DocOnly: Sonnet's own GLG extraction path and licensed
      multi-port de-embed output remain unverifiable under Lite.
- [x] A4 signed file-model pin terminals — DONE. Signed labels are legal
      intake on FLOAT-resolving SPARAM/SPROJ components only (via-mode file
      comps keep the nonnegative gate). Native anchors: the installed
      rfid_tag_pro FLOAT component declares SMDP labels {+1,-1} on pin
      indices {1,2}; a signed FLOAT amp.son SPARAM variant passes Lite
      intake and halts only at the data-file license wall (fixture
      `test/fixtures/native_signed_model_pins`). A signed pin marks the
      return side of its project differential port; its file port still
      attaches between the pin's own bridge node and the component's
      floating common — verified bit-identical to the unsigned two-pin
      FLOAT attach in solved S-parameters. Signed pins without a
      same-magnitude positive partner stage but fail loudly at model
      build; label zero on FLOAT keeps its floating-common adapter
      rejection; `circuit_add_sonnet_files!` fails signed labels on
      lookup. Regressions: `test_planar_sonnet_model_files.jl` (signed
      FLOAT staging/attach, solve identity, via-mode signed rejection,
      unpaired and zero-label failures, fixture-hash testset).
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
- [x] A7 via-mode pin-feed residual — DONE-char (evidence + bounded
      regression, no speculative model change). Forensic sweep of
      `data/a7_viacontrol` against Lite 18.53 ND solves falsified every
      structural hypothesis: mesh refinement converged, literal
      VIA-POLYGON controls on the same box/stack match Lite to ~7e-5,
      via length/cover direction irrelevant, FEED/CELL/CUST widths and
      uni/tap basis variants do not close the gap, and the fixture's
      Touchstone header is explicitly raw ("Data is NOT De-embedded"),
      ruling out a calibration difference. The residual is bounded,
      imaginary-dominant and exactly frequency-linear: an independent
      per-pin fit yields a constant ~0.13 nH series-inductance gap at
      all four probe frequencies — consistent with the
      edge-face-to-column junction jog a cell-column via return cannot
      represent (Lite meshes the pin as its own flap element; its own
      mesh accounting, Via:4 for two 2-cell pins, matches the column
      structure). Lite's raw 4-port geo response shows the residual
      concentrated in pin self/coupling terms. Resolution path is a
      dedicated pin-flap/junction basis or validated co-calibration
      (E2 machinery), not a tuned constant. Fixture
      `test/fixtures/native_pin_feed_residual` (projects + native s2p/sid
      + mesh logs + hash manifest); regression testset "Native via-mode
      pin-feed residual (A7)" in `test_planar_component_returns.jl`
      bounds the residual at 0.03 and pins its frequency-linear
      imaginary-dominant character across 0.5–2 GHz.
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

- [x] B1 GEOVAR header beyond `ANC/SYM/RAD XDIR/YDIR` point sets (`GeoVars:165`).
      **Done** — Lite 18.53 55-probe campaign
      (`test/fixtures/native_geovar_header_grammar/manifest.json`):
      kind/axis stay positional and case-insensitive (position swaps
      reject "Type Keyword not understood"), while the tail is scanned —
      the first NSCD/SCUNI/SCXY sets scaling, the first finite numeric
      token sets the sign-normalized display direction; unknown tokens
      reject only while scaling is unset (ignored after it); missing
      fields default to NSCD/+1 including a bare four-token header, and
      `+1`, `-1.7`, `1e2`, `0`, `99` all parse. The direction field is
      ignored for application whenever the saved reference span is
      nonzero — every dimension resolves `sign(span)*target` (bit-exact
      verified on ANC/SYM/SCXY across -2..3); only a zero span falls
      back to the field's sign. Structural record keywords are
      case-insensitive end-to-end (`ftyp`, `geo`/`end geo`, `valvar`,
      `geovar`, `pos`, `nom`, `eqn`, `ref1/2`, `ps1/2`, `poly`, `end`,
      `box`, `tmet`, `bmet`, `met`, `num`, `via polygon`, `bri poly`,
      `tlaynam`, `tolevel`, `por1`, `cupgrp`, `smd`, `id`, `type`,
      `gndref`, `simple`, `smdp`, `sbox`, `stf`, `options`, `termw`,
      `exsmdp`, `inhswp`, `cupgrp`); an all-lowercase project solves
      bit-identically natively and resolves identically here. DIM unit
      entries (`ANG`, `CAP`...) and VARSWP block contents (`SWPVARS`,
      `ENABLED`, `FREQ`...) are case-SENSITIVE natively and remain
      strict. EQN is optional after NOM or after REF2 (both verified).
      SCUNI grammar path verified bit-identical: impl literal for
      `1 SCUNI 5 NSCD` matches the native parameterized s2p
      (0.999167/-22.89deg @3.315GHz). 90-assert regression testset
      `Native GEOVAR header grammar scans and folds case`.
      Residuals: CKT netlist contents not probed for case; value-position
      tokens (SONPROJ, INH/NOH, LOCAL, Y, STD, L/R) kept strict pending
      dedicated probes.
- [x] B2 explicit reference repetitions — scaled ANC moving-reference law
      (`GeoVars:228`). **Done as DocOnly loud rejection** — the law is not
      literal-expressible. `native_geovar_scaled_mixed_reference/README.md`
      archives the decode campaign: forward/reversed/refonly literals are
      bit-identical only at cell-coincident width 0.28125; the second-width
      experiment falsified composed scaled passes (1.4e-4), additive deltas
      (7.6e-4) and deduplicated forms; an exact cell match exposes a mixed
      law (REF takes two additive σ movements, the ordinary point takes σ
      plus an undecoded ~3σ/16 share) that no two-width or asymmetric probe
      completes. Official Lite docs pin only the single-application affine
      law (`target/span` per-point scaling); repetition composition with
      scaling has no native export channel — `em` emits S-responses and
      SUBS counts but never resolved geometry. Regression:
      `test_planar_sonnet_scaled_mixed_reference.jl` (45 assertions) pins
      the rejection and the falsified-hypothesis suite.
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

- [x] C1 non-rectangular thick-wall VOL topology — **Done**. Lite does not
      resolve offset topology geometrically at all: it applies the
      algebraic tube formula t·(P−t·Σcot(θ/2)) while any offset edge
      survives, and the via is fully plated (cross-section = footprint
      area) once every edge is consumed. Verified 18.53 on four series-via
      probes (`fixtures/native_volume_sheet_topology`): dumbbell_neck
      (waist closes, algebraic regime, 2.6e-6 at 1 MHz),
      dumbbell_saturated (formula overshoots → clamp), notch_consumed
      (remnant empty with formula still at 0.90·A → footprint, 2.8e-6),
      dumbbell_negative (formula negative, no surviving edge → footprint,
      1.1e-5). Surviving-edge test uses convex corner terms only — reflex
      vertices join offsets with an arc instead of trimming. Residual:
      the per-edge survival check is local; a remnant can in principle
      die nonlocally (opposing walls <2t) while an edge still reports a
      span — unobserved in native probes, auto-corrected to footprint
      whenever the tube formula overshoots area. The inset matrix and its
      crossing checks are gone, so the path is also strictly cheaper.
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

- [x] D1 applied etch-bias / rho / RPV tables on inherited layers — DONE as
      a verified gated rejection. Both `soncmd -LoadTechnology` and `em -N`
      hang (>10 min, killed) on bias-carrying STFs on Lite 18.53, so the
      applied tables cannot be structurally decoded and Lite cannot batch-
      solve them either. The materializer keeps the loud "requires its
      physical adapter" rejection rather than silently dropping etch/rho/RPV
      bias (`Technology:822`).
- [x] D2 STF metal model classes + conductor class — DONE for every class
      whose emitted SON row has a verified evaluator. The full
      model/condspec matrix was decoded with `soncmd -LoadTechnology` on
      Lite 18.53: Normal and ThickMetal accept cond/rsvy/shres and emit
      NOR/TMM with the CDVY/RSVY/SRVY loss selector (RSVY in ohm-cm; TMM
      always serializes its selector slot and clamps num_sheets=1 to the
      two-face minimum); Resistor accepts res/shres and emits RES rdc
      (field liveness reads rdc=0 under shres); Rdc/Rrf requires rdcrrf
      and emits NAT rdc rrf; General accepts general/rdcrrf and emits
      SUP rdc rrf xdc ls (xdc/ls read as 0 under rdcrrf); Sense requires
      sense and emits SEN xdc. A Down direction is native-inert outside
      ThickMetal (accepted and ignored; TMM emits TDWN). RoughMetal
      (RUF), TrueVolume (TVM), nonzero roughness, non-Thin cross-section,
      non-3LYR true-volume mode and 3+ TMM sheets remain gated rejections:
      the generated rows have no verified evaluator and Lite's thick-metal
      solve is license-blocked, so no oracle exists. `rpv` is verified
      runit-invariant (raw ohms) and `ls` serializes verbatim.
      Regressions: `test_planar_sonnet_technology.jl` model/spec pairing
      testset.
- [x] D3 STF via model classes — DONE. Volume accepts cond/rsvy/shres/rpv
      and emits `VOL v wall [RSVY|SRVY|RPV]` on both Solid (inert 0 wall
      slot) and Wall forms; fill_factor/via_density are native-verified
      inert on Volume. Array accepts cond/rsvy (`ARR v fill`) and rpv
      (`ARR r 100 RPV density`, fill forced to 100, density after RPV —
      load-bearing: it divides resistance per via); Array+shres is a
      native "invalid combination" rejection and rejects identically.
      Surface accepts every condspec and emits `SFC rdc xdc rrf` reading
      only live fields; SFC solves as a hollow surface tube with
      sigma=perimeter/(Zs*mask_area) on vias and as the planar surface
      impedance Zs=transition(rdc,rrf*sqrt(f))+jXdc on sheets (SFC is
      verified legal on sheets — not the VOL/ARR wrong-type rejection).
      NOH via inheritance is verified: endpoints move to the techlayer's
      from/to while literal material and flags (incl. RING) stay; via
      mesh flags never inherit under NOH.
      Regressions: `test_planar_sonnet_technology.jl` pairing + SFC
      testsets, `test_planar_sonnet_io.jl` SFC field-order contract.
- [x] D4 STF mesh controls — DONE for the representable subset. Oracle:
      `soncmd -LoadTechnology` (which drops every mesh control from the
      serialized output except the via fill/cover tokens, materialized into
      the polygon flags) plus `em -N` subsection counts on Lite 18.53.
      Implemented: the full via mesh-fill matrix under INH is authoritative
      over literal polygon flags — Ring→RING, Vertices→VERTICES,
      Center→CENTER, Full→SOLID, Solid→SOLID, Bar→BAR, Default→RING
      (verified across 7 fills × conflicting literals; the earlier
      "explicit flag wins" rule was wrong and is fixed) — and the via pad
      control, where pads="on" emits COVERS over any literal cover flag and
      everything else emits NOCOVERS (COVERS is the existing endpoint-pad
      footprint model in both raster and conformal paths). Diagonal,
      Conformal and Staircase on a via mesh are native "Invalid fill type
      in mesh definition" errors and reject identically; pads="on" on a
      metal mesh is verified native-inert and accepted-inert.
      Kept rejecting with evidence: iso_type (X Only changes subsections
      114→66, Y Only is a native error), xmin/xmax/ymin/ymax bounds
      (114→93), use_edge_mesh=off and usemaxlength/maxlen (114→114 inert on
      the tested staircase geometry but semantically real controls).
      Regressions: "Linked STF via mesh flag precedence and pads" (179
      checks) plus the INH flag assertions in the techlayer testset.
- [x] D5 native linked-STF EM solving — DONE to the Lite-verifiable scope.
      Linked projects cannot solve on Lite at all, so parity is checked on
      the materialized-equivalent literal: fixture
      `test/fixtures/native_stf_linked_solve` pins
      `rich.son`/`rich.stf` → `literal_lite.son` record-for-record
      (asserted in the "Linked STF materialized literal native solve
      parity" testset) and Lite's actual `em` solve of the literal
      replays inside a 1e-3 complex-S gate at 1/2 GHz (observed |ΔS|
      ≤6.4e-4, common-mode port-plane phase slope; magnitude ~2e-5).
      Newly decoded Lite *solve-time* gates on the same literal: vias
      solve only under Ring fill ("via polygon which is not set to Ring
      meshing fill" — `em -N` still meshes every flag) and no
      thick-metal polygon may solve ("thick metal polygon(s)"), so
      TMM/non-Ring materialized solves have no Lite oracle. HFSS 2025R1
      batch mode is verified working as a fallback oracle. Residual:
      linked-STF native `-S` and TMM/non-Ring solve parity stay DocOnly.
- [x] D6 private/encrypted sections — DONE for materialization. Probed
      `soncmd -LoadTechnology` on STFs carrying declared `private` and
      `privenc` root sections (including a schema-valid divergent private
      materials/stackup): native output is byte-identical to the public
      baseline — private sections are ignored during public
      materialization. `_stf_static_guard` now treats declared root
      private/privenc as opaque (public materialization unchanged);
      undeclared private sections, malformed `has_private` and nested
      private payloads still reject (name-keyed resolution can never
      reach them). Fixture `test/fixtures/native_stf_private` (forged
      STF, both native outputs, README); regression testsets in
      `test_planar_sonnet_technology.jl`. Residual DocOnly: real
      foundry `privenc` parameter blobs referenced by the stack cannot
      be decrypted — they reject as unresolved material names.

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
      via port must terminate on a box cover" rejection; CUP ports now
      solve through the group de-embed (E2). Lite-verified refinement
      (18.53): a `CUPGRP` block on a non-CUP token is inert — wall
      STD+CUPGRP replays bit-identically to the group-less port and an
      interior non-CUP port fails "no ground reference" either way —
      and a wall-attached CUP token degrades to the standard wall port
      (g2_cup/xg2_cup both log "Standard port"). Grouped non-CUP ports
      now parse with structural validation and solve by placement;
      wall-CUP degrades likewise and only non-wall CUP ports count
      toward the `-d` de-embed gate.
- [x] E2 CUP calibration-group solving — DONE (group de-embed +
      bounded residual). Interior open-edge `POR1 CUP` ports stage as
      terminal ports with generated cover-return vias and a per-group
      junction standard; `solve_sonnet_project` subtracts the standard's
      contracted admittance block from the group's labels unless
      `raw=true`. Grouping follows `sonnet_calibration_groups`:
      `SIMPLE Y` ports are independent singletons, other CUP ports
      couple per (CUPGRP name, resolved ground reference) — Lite keys
      groups on the resolved cover, so a named group whose members
      resolve to different covers rejects natively and `Auto`-named
      members sub-group per resolved cover. Standard form: Lite's
      port discontinuity is a lumped launch-junction capacitance — pure
      susceptance exactly ∝f (~0.95fF on the archive fixture, quantized
      per contact cell) — so each member's removal is measured in situ
      as the port cells' self-capacitance marginal: the full host-pad
      solve through a separate cover-contact column minus the same pad
      stripped of that member's port cells (the column's own parasitic
      cancels exactly; mutual junction terms fall out of the same
      contact set). Contacts must also clear every launch cell's
      neighborhood — a contact touching a launch measures column-
      lateral coupling, not junction capacitance — and launch cells
      shared between members on one pad (corner cells where edges meet)
      are stripped exactly once. Pads leaving no non-junction contact
      cell, including same-pad groups where every free cell touches a
      launch, degenerate to a coarser feed-column standard. Wall-attached CUP ports still
      route to the A1 feed-line adapter rejection; FLOAT/polygon-plane
      GNDREF, port-zero CUP, diagonal and non-open edges reject loudly.
      Residual: our junction measurement runs ~1.5x Lite's value —
      a basis-resolution gap in the A7 pin-feed family (Lite's ideal
      lumped contact vs the cell-column junction), bounded by the
      `fixtures/native_cup_junction` archive: calibrated-vs-D |dS| <=
      2e-4 at 1GHz / <= 2e-3 at 10GHz on the CUP-driven terms (S22 is
      excluded — Lite's D stream also removes the wall port's feed-line
      discontinuity, the separate A1 item). Tests in
      `test_planar_sonnet_io.jl` "Native CUP calibration-group solve
      (E2)". Resolved sub-item: the multi-port CUPGRP grammar
      (`POR1 CUP Auto` + `CUPGRP "Auto" LOCAL`, per installed example
      `ind_cap_ports.son`) parses and passes Lite's physical validation
      — aligned launches, shared resolved cover, clear restricted space
      — but the analysis stops at the license boundary ("Sonnet Lite
      does not allow calibration groups"), so no native multi-port
      numerical oracle exists under Lite and coupled-group behavior is
      verified internally. Resolved sub-item: a CUPGRP block after ANY
      POR1 token parses natively, but Lite keys co-calibration on the
      CUP token AND placement — on non-CUP ports the group is inert
      (wall STD+CUPGRP replays bit-identically to the group-less port;
      an interior non-CUP port fails "no ground reference" either way),
      and a wall-attached CUP token degrades to the standard wall port
      (g2_cup/xg2_cup both log "Standard port"). Our intake now parses
      grouped non-CUP ports with structural validation, degrades
      wall-CUP to the wall path, and counts only non-wall CUP ports
      toward the `-d` de-embed gate. Lite's trailing-shared-CUPGRP form
      rejects natively.
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
- [x] E5 dielectric bricks (`IO:1829`, `RasterResources:44`,
      `Conformal:181-185`) — DONE (zparts=1 `BRI` bricks; BAR via
      fill closed under E5d below). Native
      `BRI` material declarations, `TECHLAY BRICK` records and
      `TLAYNAM`-bound `BRI POLY` polygons are parsed and validated
      (level/mode/material binding, Z-parts==1 loud rejection of
      unsupported partitions); geometry is rasterized onto the
      analysis grid, host-layer-mapped (native level `lev` ↔ solver
      layer `L-lev`), and merged into `BrickLevel`s with
      per-direction complex contrast against the host layer.
      Solver: new VIE basis family — divergence-conforming
      transverse rooftops (full interior + half boundary) plus
      axial z pulses — reusing the vol-elem/via-elem modal kernels;
      the constitutive law enters as a shared Gram block
      `-G/(iωΔε)` assembled into the dense modal Z (FFT/UFFT paths
      reject bricks loudly). Bricks thread through
      `PlanarProblem`/`build_planar_problem` (incl. brick-brick,
      brick-via and brick-volume overlap rejection),
      `planar_current_maps` (`:brick` maps, wide-precision too),
      radiation/moments, adjoint gradients (constitutive entries
      differentiated against host thickness and per-direction
      `epsr`), axial refinement (per-slice `BrickLevel` split),
      terminal-return/CUP problem adapters, and resource
      estimators. Oracle: Lite is license-blocked for brick tech
      layers ("Circuit contains brick tech layers"), so validation
      used physics oracles: full-layer brick ≡ equivalent host
      stack (converged equivalence), Δε→∞ metal limit ≡ identical
      volume-conductor element (5 decimals), loss/passivity, plus
      adjoint gradients vs finite differences (exact). BAR fill
      mode (conformal `Vertices` fill per Lite doc) is split out
      as E5d below. Regressions in
      `test/test_planar_dielectric_bricks.jl` and
      `test/test_planar_sonnet_io.jl` ("Sonnet native dielectric
      bricks").
- [x] E5d BAR via meshing — DONE. Per the Lite 18.53 "Simplifying Bar
      Vias" doc, `TOLEVEL ... BAR` polygons sharing an endpoint span are
      grouped by raster connectivity; each connected component meshes as
      the outline-corner cells of its union (the raster analogue of the
      `Vertices` fill on the merged outline). Uniform-material components
      evaluate one loss model on the merged union outline; constant-RPV
      metals instead sum per-member conductances so the merged column
      preserves the parallel DC resistance (verified: two 2-ohm bars
      merge to a 1-ohm group). Mixed-material components, lone bars,
      disjoint bars and pinched/holed unions keep their per-polygon
      vertex fills and vertex-exact loss. Implemented in
      `PlanarSonnetIO.jl` (`_sonnet_bar_resolve`,
      `_planar_region_corner_cells`, `_planar_region_components`,
      `_planar_region_outline`) for both the raster and conformal
      adapters; mask-workspace estimators account BAR temporaries.
      Regressions in `test/test_planar_sonnet_io.jl` ("Sonnet native BAR
      vias").
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

- [x] F1 other SPICE line classes (parity line 128) — T/O done earlier;
      W N-conductor coupled line implemented: `.model x W MODELTYPE=RLGC`
      constant `Lo/Co` (+optional `Ro/Go`) lower-triangular symmetric PSD
      matrices, `N<=64`, exact uniform-RLGC 2N-port admittance via modal
      decomposition (analytic `gamma*coth`/`gamma*csch` coefficients),
      independent return nodes as physical floating coordinates, Nexxim
      2025R1 W-element oracle to ~7e-7 (fixture
      `test/fixtures/nexxim_wline_coupled`, 51 checks). `TABLEMODEL`,
      `FSMODEL`, `SMODEL`, `RLGCFILE` and frequency-dependent `Rs/Gd`
      reject loudly; U/other line classes remain DocOnly scope.
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

- [x] H1 FLOAT file-backed comps — DONE for the attach law. Lite cannot
      solve a FLOAT SPARAM component, so the multi-reference wiring was
      replayed through Ansys Nexxim 2025.1: the lowered 5-node circuit
      (EM 3-port on [(1,0),(2,0),(3,4)] + file-model 2-port on
      [(3,5),(4,5)] sharing floating common node 5) was rebuilt as
      Nexxim multi-reference S-elements plus 1e9-ohm bleeders matching
      the `:auto` floating gauge. Nexxim LNA agrees with
      `solve_sonnet_project` to ~1e-7 at 0.1/0.15 GHz (gate 1e-3 for
      the ~6-digit sdf2csv output). Fixture
      `test/fixtures/nexxim_float_attach` (netlist, CSV, run log,
      stored block matrices, generator, README); regression testset
      in `test_planar_sonnet_floating.jl` (+48) asserts wiring
      topology, netlist↔block-matrix integrity, stored-vs-live drift,
      `PlanarCircuit` replay, and Nexxim agreement. The residual
      license claim (native Sonnet solve of a FLOAT file model) stays
      DocOnly.
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

- licensed GLG extraction + multi-port de-embed output (A3 residual; the
  coupled-group kernel itself is Nexxim-verified), SMD renderer compatibility,
  linked-STF native `-S` solve (D5), Conformal license (I2), S/current viewer
  (I3), encrypted STF parameter blobs referenced by the stack (D6 residual;
  root private sections themselves materialize like native), undecoded
  scaled GEOVAR law (B2 if it
  resists probing).

---

## Execution order (current)

Done so far: B1/B2/B3/B4/B5 (GEOVAR), A1 (DRP1 intake+gating), A2
(port-zero grounding), A3 (coupled pin-group kernel, Nexxim oracle),
C1–C5 (VOL topology and material laws), C10,
E1–E5 (E5 dielectric bricks: VIE rooftops + z pulses, dense assembly,
adjoint/currents/radiation/axial-refinement/CUP threading; E5d BAR
merged-via grouping + Vertices fill), plus
prior ledgered items.

Remaining (updated order):

1. ~~D-group STF residue (D1–D6)~~ — DONE: `em -N`/materialization corpus
   complete; privenc parameter blobs stay DocOnly (undecryptable, reject
   as unresolved materials).
2. F/G-group netlist/layout breadth + H1/H2 FLOAT/REFPLANE components.
3. A1 physical de-embedding adapters (FIX/NONE/LINK) — needs port feed-line
   characterization, larger.
4. I-group HFSS-based independent convergence certification.

Each landed item: implementation + deep-debug loop + tests + fixture evidence
+ ledger/docs update + full suite (1.13) + focused (1.12) + push + CI green.
