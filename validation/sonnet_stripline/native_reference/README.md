# Native complex reference oracle

These are actual Sonnet 18.53-Lite outputs, captured on 2026-10-03. The 64 × 64 cell stripline was simulated with real 50 Ω references. The GUI Graph postprocessor then normalized that unchanged data with four R/X/L/C reference sets at 1, 5, and 10 GHz. `graph_comparison.json` records the original paths, source hashes, helper hash, and all twelve comparisons.

`baseline.son` and `baseline.s2p` are the original source and high precision engine output. `native_graph_*.s2p` contain the exact text emitted by the GUI export dialog. The original screenshots and UI Automation control captures remain in `data/sonnet_validation/complex_graph_7bhIy2`.

The native reference is a series R + jX + jωL branch in parallel with C. Project port fields have fixed Ω, Ω, nH, and pF units. Export `! TERM` records contain R and X in SI units per port; `! FTERM` records contain R, X, L, and C in SI units per port, with an `&` continuation onto a numeric line.

The independent Kurokawa power wave calculation agrees with all twelve native exports within 6.60 × 10⁻¹² in complex S. This validates native Graph normalization and the reference convention. Sonnet Lite refuses electromagnetic runs with non-50 Ω port references, so this corpus does not establish an electromagnetic solve with such references. The separate failed engine attempts are retained in `data/sonnet_validation/complex_reference_oKpoAC`.
