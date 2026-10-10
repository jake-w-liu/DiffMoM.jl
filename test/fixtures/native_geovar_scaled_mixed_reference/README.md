# Native scaled ANC mixed-reference dimensions — investigated, still rejected

Fresh Sonnet 18.53 runs compare `SCUNI` parameter records whose `PS2` point set
contains the moving `REF2` point among ordinary adjustable points. The retained
pairs document an investigation that did **not** reach an accepted adapter: the
parser continues to reject every scaled moving-reference form.

## Evidence retained

The `forward` (`PS2: 2 3`), `reversed` (`3 2`) and `refonly` (`3`) literals each
produce a complete native S matrix bit-identical to their parameter's — at the
retained target width `0.28125`, where the hypothesized coordinates rasterize to
the same cells as the true resolved coordinates. `count2` (`PS2: 2 3 3`) retains
its native parameter response without a confirmed literal.

## Falsification (second-width decode experiment)

Repeating the `forward` shape at target width `0.3125` (Δ = 0.0625, cell =
15.625 µm) falsified the composed-movement hypothesis that had matched at
`0.28125` purely by cell coincidence:

- composed scaled passes (ref moved twice through `scaled_coordinate`):
  `(v3=0.6875, v4=0.765625)` → full-S error `1.41e-4`, not bit-identical;
- scaled + absolute delta (`v4=v+2Δ`): `(0.6875, 0.75)` → `7.6e-4`;
- exact cell match found at `(v3=0.69921875, v4=0.75)` → error `0.0`, i.e. the
  reference takes two additive σ=(v−anchor)(r−1) movements while the ordinary
  point takes σ plus a further ~3σ/16 share whose law is undecoded;
- a two-width fit (`0.34375`, `0.21875`) and an asymmetric-offset polygon
  confirmed no simple compositional or additive rule reproduces both points.

`native/` preserves exact input sources, native output, Touchstone matrices and
engine metadata. `comparison.toml` records the pair statuses. `sha256.toml`
covers every payload. Exact sources are preserved byte for byte.

Run the portable regression with:

```powershell
julia --project=. --startup-file=no -e 'include("test/test_planar_sonnet_scaled_mixed_reference.jl")'
```

With a licensed installed native engine, replay the unchanged sources:

```powershell
julia --project=. --startup-file=no validation/planar_audit/run_scaled_mixed_reference.jl
```
