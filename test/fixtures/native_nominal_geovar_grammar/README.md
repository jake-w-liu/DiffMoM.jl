# Native nominal geometry metadata validation

Fresh Sonnet 18.53 runs use the same notched polygon and `SCUNI` dimension,
with the effective quantity exactly equal to saved NOM .25. Native Sonnet
rejects a nonnumeric POS coordinate and a negative PS2 group count. Before
this repair the current geometry adapter accepted both through nominal
passthrough. Exact failed inputs, errors, logs and metadata are retained.

The native valid control and a control referencing an unknown polygon are
accepted at nominal value and have identical complete S matrices. Validating
this unused polygon identity against live polygons would incorrectly reject
the saved nominal input. Effective geometry changes continue to require
valid physical reference identities.

An out-of-range point of a known polygon is also accepted by native Sonnet,
but changes its complete S matrix by more than 1 despite the nominal value.
A fresh repeat and independent saved-coordinate literal control under
`reference_effect/` establish the old adapter's full-S error 1.0001367; the
literal and valid nominal controls pass the previously declared .005 gate
with error 2.78e-5. This point convention requires a separate adapter.
Current lowering rejects it explicitly instead of silently using saved
coordinates. The failed identity oracle and pre-guard physical results
remain retained; no electrical gate was relaxed.

`comparison.toml` and `source_snapshot/` retain the source identity used for
the capture and the original probe. `sha256.toml` covers every payload,
including this readme. Historical code hashes are not required to equal the
current implementation. The current regression checks the public lowering
path and input ownership for both confirmed rejections and accepted controls.

```powershell
julia --project=. --startup-file=no -e 'include("test/test_planar_sonnet_nominal_geometry_grammar.jl")'
```

To replay a native case, pass its exact top-level SON source to the maintained
`SonnetReference.reference_run` helper with a fresh output directory. Existing
native evidence and historical source snapshots must remain byte exact.
