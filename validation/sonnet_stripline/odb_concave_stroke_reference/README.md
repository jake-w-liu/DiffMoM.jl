# Actual Sonnet Lite concave polygon line-stroke reference

The 120 original files were copied byte-for-byte from
`data/sonnet_validation/odb_concave_stroke_native_gKltsm`. They retain exact
native sources, engine commands/version/logs, the selected raw-output basis
checks, complete four-port S matrices and immutable source hashes.
`custom_u.toml` preserves the ordered aperture vertices and units used by
the public `symbol_resolver` adapter. The native geometry is independently
declared rectangular conductor pieces, never an exported importer mask.

Nine actual native runs (16/32/64 cells, 1/5/10 GHz) passed 39 checks under
the existing 0.06 complete complex-S and 1e-9 original-equation gates.
The maximum observed full-matrix difference was 7.635576817052e-5;
the largest original residual was 1.835000216829e-10. Production and
validator hashes stayed unchanged throughout the run.

`final_snapshot_repeat/` retains another 120 original files from
`data/sonnet_validation/odb_concave_stroke_native_4nQIuU`, generated after
bounded byte-snapshot input and all lookup-accounting fixes. Its 39 native
checks pass with the same numerical errors/residuals and unchanged final
production source hashes. Both runs retain their original native outputs;
timestamps differ and their file hashes are kept separately.

This covers the declared public polygon-resolver line stroke and physical
coupon. Native ODB translator availability, composite user-symbol strokes,
arbitrary manufacturing layouts and continuum convergence remain separate
acceptance requirements. Sonnet Lite's translator license rejection is
retained elsewhere in the parity ledger.
