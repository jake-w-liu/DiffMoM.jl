Actual Sonnet 18.53 Lite controls captured on 2026-10-04 show that repeated
ordinary adjustable vertices retain their movement multiplicity. Four ANC/RAD
and six SYM controls match independent sequential literals bit for bit.
The original reader removed repeats with `unique!`, producing coordinate
errors above 1e-5 m. Its radial full-S error exceeds 1.0, and both repeated SYM
SCXY controls exceed the unchanged 0.005 full-S gate.

The exact previous implementation executes as a regression oracle. Independent
literal coupons passed the declared 0.005 full-S and 1e-9 original voltage
residual gates at mx=my=128 before the fix. Current lowering preserves every
ordinary movement, without modifying the source project. Warmed cumulative
GC allocations are checked against the previous implementation, including the
returned geometry; this does not establish peak process memory.

Explicitly listed dimension references have separate native movement behavior.
All four such native controls differ from deduplicated literals. They now reject
before physical output while their movement adapter remains unfinished.
The SCUNI/SCXY sequential reference hypotheses also fail, and their exact
failed responses remain retained here. Dependency/propagation investigations
and additional failed reference hypotheses remain under ignored `data`.

Every payload has a SHA256 entry. Reproduce the ten ordinary native comparisons
with `julia --project=. validation/planar_audit/run_geovar_multiplicity_reference.jl`.
It creates a fresh directory under `data/planar_audit` or an optional output
root supplied as its first argument and checks source stability.
