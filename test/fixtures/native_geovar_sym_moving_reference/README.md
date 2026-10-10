# Native unscaled SYM moving-reference dimensions — investigated, still rejected

Fresh Sonnet 18.53 runs probe `GEOVAR <var> SYM YDIR ±1 NSCD` records whose
`PS2` point set contains the moving `REF2` point explicitly (0-based vertex
indices, `REF2 POLY 1 1 / 3` on the nominal rectangle
`(0,0.375)-(1,0.375)-(1,0.625)-(0,0.625)` in a 1×1 mm box rasterized to
32×32 cells, cell = 31.25 µm). The parser continues to reject every unscaled
SYM explicit moving-reference form.

## Decoded movement law (confirmed)

With `r` explicit `REF2` occurrences in `PS2` and `δ = target − NOM`:

- **REF2 (repeated reference)**: moves `3·r·δ/4` — verified bit-identically at
  `r = 1, 2` for `δ ∈ {±0.0625, 0.046875, 0.0390625, 0.03125}` and ambiguously
  consistent at `δ = 0.0234375`.
- **Ordinary PS2 points**: deduplicate, then move `⌈c·r/2⌉·(δ/2)` for `c`
  occurrences — verified at `c·r ∈ {(1,1),(2,1),(3,1),(1,2),(2,2)}` for
  `δ = 0.0625`.
- **Order-independence**: `PS2 = 3 2` ≡ `2 3` bit-identically.
- **PS1 duplicates**: deduplicate identically.
- **Contraction / direction −1**: mirror the law with the sign of `δ`.

## Blocking counter-evidence (PS1-side trigger undecoded)

The PS1-side ordinary-point movement is **non-monotonic in δ** and cannot be
reduced to a threshold on `δ`, `δ/2`, cell fractions, or the REF-side
displacement. For the ref-only probe (`PS2 = 3`, `PS1 = 1` implicit `REF1`):

| δ | δ/2 in cells | resolved v1 (bit-identical literal) |
|---|---|---|
| 0.0625 | 1.0 | `0.375 − δ/2 = 0.34375` — moved |
| 0.046875 | 0.75 | `0.375 − δ/2 = 0.3515625` — moved (sub-cell, exact) |
| 0.0390625 | 0.625 | `0.375` — unmoved |
| 0.03125 | 0.5 | `0.375` — unmoved |
| 0.0234375 | 0.375 | ambiguous (both literals match) |

Sub-cell resolved positions are exact when the point moves (the `δ = 0.046875`
literal lands at `0.3515625`, not on a cell edge), so the movement is not
raster-quantized; the unmoved cases are likewise position-exact (a moved
hypothesis fails by `3.9e-3`). No tested rule — cell/half-cell thresholds,
floored or rounded cell counts, ref-side coupling — reproduces this pattern.

`r ≥ 3` cases (`o1r3`, `ref3_only`) additionally retain only nearest-basin
literals (`err ≈ 2–4e-4`); the `r = 3` law is unconfirmed.

Because the PS1-side semantics remain undecoded, emitting the confirmed
portions would silently resolve wrong coordinates for `δ` values inside the
unmoved band. The adapter therefore keeps rejecting explicit `REF2`-in-`PS2`
for unscaled `SYM`; the retained pairs document both the confirmed law and the
blocking anomaly for any future decode attempt.

`native/` preserves exact input sources, native output Touchstone matrices and
engine metadata. `comparison.toml` records every pair status. `sha256.toml`
covers every payload. Exact sources are preserved byte for byte.

Run the portable regression with:

```powershell
julia --project=. --startup-file=no -e 'include("test/test_planar_sonnet_sym_moving_reference.jl")'
```
