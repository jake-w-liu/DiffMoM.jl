# Native unscaled SYM moving-reference dimensions — decoded and implemented

Fresh Sonnet 18.53 runs probe `GEOVAR <var> SYM YDIR ±1 NSCD` records whose
`PS2` point set contains the moving `REF2` point explicitly (0-based vertex
indices, `REF2 POLY 1 1 / 3` on the nominal rectangle
`(0,0.375)-(1,0.375)-(1,0.625)-(0,0.625)` in a 1×1 mm box rasterized to
32×32 cells, cell = 31.25 µm). The adapter implements the verified law for
explicit literal point lists with `r ∈ {1,2}` moving-reference repetitions.

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

Every admitted case resolves to a `resolved_<case>` literal whose native
engine output is **bit-identical** to the parameter's own native output —
covering δ sweeps, r ∈ {1, 2}, mixed/ref-only lists, duplicates, reversed
order, negative and −1-direction deltas, pentagon point sets, and
32/48/64/128-half-count grids (the sweep corpus lives under
`data/planar_audit`).

## Still rejected

- `r ≥ 3` repetitions (`o1r3`, `ref3_only`): the native basin flattens below
  the raster resolution; nearest literals miss by `err ≈ 2–4e-4`. Unconfirmed.
- `XDIR` moving references, scaled (`SCUNI`/`SCXY`) headers, and `POLY id 0`
  whole-polygon selectors: outside the proved domain.

`native/` preserves exact input sources, native output Touchstone matrices,
the adapter-resolved literals and their native outputs, and engine metadata.
`comparison.toml` records every pair status. `sha256.toml` covers every
payload. Exact sources are preserved byte for byte.

Run the portable regression with:

```powershell
julia --project=. --startup-file=no -e 'include("test/test_planar_sonnet_sym_moving_reference.jl")'
```
