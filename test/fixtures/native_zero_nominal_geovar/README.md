# Zero saved anchored offsets

Sonnet18.53 Lite accepts the eight ANC/NSCD parameter projects with NOM0.
Their reference coordinates coincide on the selected X/Y axis. Both directions
and two positive targets are covered. Each complete native S matrix is
bit-identical to its independently hand-shifted literal control. DiffMoM at
7b907f59 rejected all eight inputs before geometry emission.

The original capture is retained at
`data/sonnet_validation/geovar_zero_nominal_axes_GAt2Wu`. The portable subset
here preserves exact input bytes, full complex matrices, native response logs,
stdout/stderr, metadata, producer and comparison report. Native cache and timing
files remain in that original capture. `source_before/` retains the old adapter
and the complete independent literal public-solver baseline before the repair.
Historical source hashes identify those captures, not today's implementation.
`sha256.toml` covers all portable payloads and this readme.

Six literal coupons pass the existing .005 full-S and1e-9 original-voltage
residual gates. The negative-direction .0625 target fails full-S on both axes
(approximately1.00016 and1.00014), despite small residuals. This is an existing
literal EM discrepancy, independent of parameter lowering. It remains FAIL;
neither its data nor its gate is replaced. These eight cases establish geometry
semantics; the physical regression asserts the six passing baseline coupons.
They do not establish complete native mesh or continuum accuracy.

The adapter preserves source identity, owned output geometry and input
immutability. Scaled zero dimensions, zero saved offsets with distinct reference
coordinates, other dimension kinds, dependent movements and moved non-wall
attachments require their own adapters. Positive source values lost during SI
conversion remain rejected.

Portable regression:

```powershell
julia --project=. --startup-file=no -e 'include("test/test_planar_sonnet_zero_nominal_geometry_variables.jl")'
```

Fresh native replay, with output in a new ignored directory:

```powershell
julia --project=. --startup-file=no validation/planar_audit/run_zero_nominal_geovar_reference.jl
```
