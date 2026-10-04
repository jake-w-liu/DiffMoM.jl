# Actual native artwork workflow reference

These are unmodified outputs from installed Sonnet Lite 18.53 on 2026-10-03.
The Gerber and ODB++ source files declare a PEC strip across a 1 mm square
box, with a central clear window. The native GEO is independently authored
as four literal rectangles. Native GEO was never derived from imported masks
or objects. Sonnet's artwork translator is not used or claimed.

`plain_clear_window.gbr` uses ordered dark and clear regions.
`legacy_clear_window.gbr` uses MI, SF, OF and IR to yield the same geometry.
`odb_clear_window` is a complete minimal board job with its matrix, step header
and ordered positive/negative surface features.

The analysis has two 100 µm air layers, PEC covers and two real 50 Ω wall
ports. Native BOX counts are twice actual cell counts. All fixtures preserve
native default subsection controls and raw `FILEOUT ND` output with 15
digits in real/imaginary form.

All 27 comparisons (3 fabrication inputs × 16/32/64 cells × 1/5/10 GHz)
passed the declared full complex S gate of 0.06. Maximum error was
0.0001188864; maximum recomputed voltage residual was 6.190464e-11.
Native/DiffMoM unknown counts were 126/208, 308/864 and 664/3520.
The native 10 GHz refinement steps were 0.01471 and 0.01346, so this is
agreement at the declared grids, not a continuum convergence certificate.
It does not establish broader vendor manufacturing process acceptance.

Run `julia --project=. validation/sonnet_stripline/validate_sonnet_artwork_coupon.jl`
from the repository root to create a new real native comparison. The runner
never uses this archived response as fresh native output. The original
provenance-complete run is `data/sonnet_validation/artwork_native_3hcgXi`.
`comparison.toml` records all full S matrices, source/module hashes and grid
steps; `archive_manifest.json` gives the exact SHA-256 of every copied file.

The first runner attempt, `artwork_native_jUPlRS`, had a residual-reporting
harness error. It is retained as unverified and contributes no acceptance.
