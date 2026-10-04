# Coupled-inductor actual-engine reference

These 90 original proof artifacts retain ngspice 47 source libraries,
loaded full-S decks, complete complex voltage/current samples and engine
logs. `comparison.toml` records the engine executable SHA256 and original
validation results; its original validation-only scope text is preserved.
`sha256.json` hashes every original artifact. The production regressions
recompile each library through the public APIs against these untouched
engine responses.

Cases include signed/perfect coupling, reversed winding, forward and
nested names, equal multi-coil K, global zero and two galvanically isolated
returns with independent common-potential shifts. Invalid input fixtures
and initial failed validation logs are deliberately retained. No Sonnet
UserModel or native vendor execution is represented by these files.
