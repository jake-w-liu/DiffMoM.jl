# F3 Spectre nport/transformer oracle fixture

`xfmr.sp` runs an independent Nexxim 2025 R1 LNA solve of an ideal 2:1
transformer realized as a unity-coupled inductor pair (K=1, L1/L2=4, so
V1/V2=2 exactly). `xfmr.csv` is the sdf2csv conversion of `xfmr.sp.sdf`;
`xfmr.run.log` is the batch run log. Nexxim reports S11=+0.6, S12=S21=0.8,
S22=-0.6 at 1 GHz, matching `circuit_add_transformer!` stamped with
`ratio=2` (`V1/V2 = ratio`, `ratio*I1 + I2 = 0`).

`databank.s2p` is the two-port Touchstone databank consumed by the
Spectre `nport` regression decks (`test_planar_spectre.jl`); the nport
attach is verified to reproduce the file samples exactly at knots and to
linearly interpolate between them, and the `nport` -> `_CircuitNetwork`
plumbing reuses the file-model machinery already verified against Nexxim
file-model attachment (`test/fixtures/nexxim_float_attach`).

Nexxim's `s_element` accepted neither `FILE=`/`TSFILE=`/`DATA='file'`
file attach nor a flat inline `DATA=[f, re11, im11, ...]` vector in this
build, so the oracle for the transformer convention is the K-coupled
inductor equivalent (unity coupling is the exact ideal-transformer
limit) and the nport databank is exercised through file-roundtrip plus
the analytic transformer->databank cascade check in the testset.
