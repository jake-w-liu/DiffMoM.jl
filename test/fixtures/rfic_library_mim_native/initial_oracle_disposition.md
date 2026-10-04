# Initial MIM mask oracle correction

`rfic_library_mim_native_initial.log` and
`data/sonnet_validation/library_mim_native_jrI9Ko/` retain the first run.
Both actual native physical matrix comparisons passed the declared gates,
but two importer mask assertions failed. Those assertions incorrectly
compared array positions: the library sorts sheets by interface, while the
native importer retains polygon encounter order. Matching the masks by
their physical interface fixes the validation oracle. No production code
was changed for this candidate. The original validator and geometry helper
are retained as `library_mim_*_initial_before.jl`.
