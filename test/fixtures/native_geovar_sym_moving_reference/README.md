# Native unscaled SYM moving-reference dimensions — decoded and implemented

Fresh Sonnet 18.53 runs probe `GEOVAR <var> SYM XDIR/YDIR ±1 NSCD` records
whose `PS2` point set contains the moving `REF2` point explicitly (0-based
vertex indices, `REF2 POLY 1 1 / 3` on the nominal rectangle
`(0,0.375)-(1,0.375)-(1,0.625)-(0,0.625)` in a 1×1 mm box rasterized to
32×32 cells, cell = 31.25 µm; the `x_*` cases use the transposed
`(0.375,0)-(0.625,0)-(0.625,1)-(0.375,1)` rectangle under XDIR). The adapter
implements the verified law for explicit literal point lists with
`r ∈ {1,2}` moving-reference repetitions on both axes, plus crossing sets:
`PS2` lists (literal or `POLY id 0` whole-polygon selectors) that contain a
vertex strictly on REF1's side of the `POS` symmetry plane, provided REF1's
own vertex is among them and `r = 1`.

## Verified movement law

With `r` explicit `REF2` occurrences in `PS2`, `δ = target − NOM`,
`cell = box/ncells` in the movement axis, `quantum = min(cell, box/32)`, and
`step = box/64`:

- **REF2 (repeated reference)**: moves `3·r·δ/4` — in mixed lists floored to
  `max(3·r·δ/4, quantum)` once `|δ| ≥ quantum`.
- **Ordinary PS2 points**: deduplicate, then move
  `min(⌈c·r/2⌉·δ/2, c·r·δ/2·ramp)` with `ramp = clamp(|δ|/quantum − 0.5, 0.5, 1)`
  for `c` occurrences; on coarse rasters (`cell > box/32`) additionally capped
  at `3·c·r·δ/8`.
- **REF1 / PS1 side**: gated by `3·r·δ/4 > cell`. Gated REF1 takes `−δ/2` and
  ordinary PS1 points take `−step·max(1, ⌊|δ|/(box/32)⌋)`; ungated PS1 points
  still drop one `step`, sign-independently.
- **Order-independence**: `PS2 = 3 2` ≡ `2 3` bit-identically.
- **PS1 duplicates** deduplicate; the closing duplicate vertex follows REF1.
- **Contraction / direction −1** mirror the law with the sign of `δ`.
- **Crossing sets**: when `PS2` contains any vertex strictly on REF1's side
  of `POS` (the admission rule requires REF1 itself among them and `r = 1`),
  REF2 takes the **full** `δ` instead of `3·r·δ/4`; ordinary shares are
  unchanged, so `REF1` (−δ/2 gated + δ/2 ordinary) nets to zero and dual
  PS1∩PS2 members cancel likewise. `POLY id 0` expands to every vertex and
  always crosses through REF1 — `wild2`, `wild2_neg`, `x_wild2` are
  bit-identical to their literal equivalents `litall`, `lit03`, `x_lit01`.

Every admitted case resolves to a `resolved_<case>` literal whose native
engine output is **bit-identical** to the parameter's own native output —
covering δ sweeps, r ∈ {1, 2}, mixed/ref-only lists, duplicates, reversed
order, negative and −1-direction deltas, pentagon point sets, and
32/48/64/128-half-count grids (the sweep corpus lives under
`data/planar_audit`).

## Still rejected

- `r ≥ 3` repetitions (`o1r3`, `ref3_only`): the native basin flattens below
  the raster resolution; nearest literals miss by `err ≈ 2–4e-4`. Unconfirmed.
- Crossing sets whose only anchor-side members are ordinary points
  (`PS2 = 1 2 3`), crossing sets with `r ≥ 2` (`PS2 = 0 3 3`), and `POLY id 0`
  in `PS1`: each keeps a residual floor (~8e-6–8e-5) after sweeping every
  vertex, port and rasterization degree of freedom — not literal-expressible.
- Scaled (`SCUNI`/`SCXY`) headers: outside the proved domain.

`native/` preserves exact input sources, native output Touchstone matrices,
the adapter-resolved literals and their native outputs, and engine metadata.
`comparison.toml` records every pair status. `sha256.toml` covers every
payload. Exact sources are preserved byte for byte.

Run the portable regression with:

```powershell
julia --project=. --startup-file=no -e 'include("test/test_planar_sonnet_sym_moving_reference.jl")'
```
