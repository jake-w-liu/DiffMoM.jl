# Native one-axis scaled geometry dimensions

Fresh Sonnet 18.53 runs compare `SCUNI` parameter records with independently
declared literal notched polygons. Sixteen pairs cover ANC/SYM, XDIR/YDIR,
positive/negative reference directions, expansion from NOM .25 to .375 and
contraction to .125. Every pair has a bit-identical complete native S matrix.
Four retained translation hypotheses disagree by more than .006 in full S.

`native/`, `native_x/` and `native_contraction/` preserve exact input sources,
native output, complete Touchstone matrices, logs, errors, engine metadata,
and comparison records. Their source-before/source-after hashes describe the
code used for each historical capture, not a requirement that today's source
equal a historical version. Source snapshots preserve the relevant producer
and implementation bytes. `native/literal_physical_before.toml` records the
independent literal public-solver check performed before the adapter existed.
`sha256.toml` covers every payload and this readme.

The regression resolves actual parameter records through the current public
path, compares owned vertices, attached ports and cell masks to the literal
controls, and uses the retained complete native S matrices. Its previously
declared electrical gates are absolute full-S error .005 and original voltage
residual 1e-9. These are fixed-grid coupon checks, not continuum convergence.
Separate affine oracles cover small changes, representable range boundaries,
caller precision/rounding and concurrent task isolation. Resource rejection
must preserve the input project.

Run the portable regression with:

```powershell
julia --project=. --startup-file=no -e 'include("test/test_planar_sonnet_scaled_geometry_variables.jl")'
```

With a licensed installed native engine, replay the unchanged source pairs:

```powershell
julia --project=. --startup-file=no validation/planar_audit/run_scaled_geovar_reference.jl
```

Fresh replay output defaults to an ignored directory under `data/planar_audit`;
an optional first argument selects another output root. Existing evidence is
never overwritten. Active SCXY/RAD, ordered dependent or overlapping dimensions,
and moved component/interior/via attachments remain separate acceptance work.
