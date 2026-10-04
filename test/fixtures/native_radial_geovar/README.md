These are actual Sonnet 18.53 Lite RAD runs from 2026-10-04. The asymmetric
polygon uses a reference radius of 0.3125 mm and another selected radius of
0.25 mm, so fixed-distance radial movement and proportional scaling differ.
Twenty-four expansion/contraction cases cover XDIR/YDIR, both direction signs,
and NSCD/SCUNI/SCXY. Their complete native matrices match the independent
fixed-distance literals bit for bit. Proportional hypotheses differ by
0.00354743 on expansion and 0.000508359 on contraction.

The independent literal coupons passed the declared 0.005 full-S and 1e-9
original voltage-residual gates at mx=my=128 before implementing RAD.
`source_before` retains the producer, native helper, previous implementation,
installed primary manual, template, public failure reproduction, and physical
gate. `native` retains exact sources, responses, metadata and logs. Checksums
cover every payload, including this file.

`edges` retains native collapse, off-wall and crossing-edge controls. Sonnet
rejects the first two but accepts the crossing-edge geometry. DiffMoM's simple
polygon model rejects all three; native polygon repair remains unsupported.
The initial equal-radius probe was nondiscriminating, so it was superseded
before implementation; its untouched capture and exact producer remain under
the ignored `data` tree as recorded in the preimplementation ledger.

Reproduce the native parameter/literal comparisons with
`julia --project=. validation/planar_audit/run_radial_geovar_reference.jl`.
The runner retains fresh output under `data/planar_audit` or an optional output
root supplied as its first argument. It checks source hashes before and after.
