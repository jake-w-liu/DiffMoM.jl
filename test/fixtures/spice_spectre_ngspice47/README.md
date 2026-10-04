# Installed Spectre linear-export reference

These 55 original artifacts retain three exact readable Sonnet example
`.scs` exports, independently translated linear ngspice decks, complete
complex Y and loaded S samples, and actual ngspice 47 logs. `comparison.json`
contains source/deck/engine SHA256 values and original validation results;
`sha256.json` hashes every original artifact.

The native sources are `1nH_oct_inductor`, `ct_inductor_noshield_IMEnet`
and `rectangular_spiral_IMEnet`. Their literal scalar R/L/C and named
mutual-inductor grammar forms the explicit `planar_read_spectre` subset.
Tests evaluate those original sources through the public reader/compiler
against the retained actual engine outputs. No native Spectre execution,
arbitrary Spectre grammar or encrypted/vendor model support is claimed.
