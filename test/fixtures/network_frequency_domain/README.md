# Network frequency storage proof

The public databank constructor previously accepted a positive `BigFloat("1e-500")` frequency, stored it as Float64 zero, and called its reference provider at DC. The final constructor and response query reject nonzero values that cannot be preserved in Float64 before calling providers. Explicit DC and representable positive frequencies remain accepted. Distinct input knots that collapse to one stored Float64 frequency are rejected.

`frequency_before.toml` identifies the byte-exact original source; `frequency_after.toml` identifies the final source. The matching source snapshots, commands, original and final outputs, and completed targeted neighbor gate are retained and hashed. The audit scripts preserve their original repository-relative context; run the live copies in `validation/planar_audit` for new measurements.

The suspected Touchstone numeric text underflow bug was discarded: replaying the isolated byte-exact original reader showed that Julia's existing numeric parser already rejects all three nonzero underflow fields. No text-parser change is part of this fix. Its independent before/after evidence and explicit-zero control are retained separately.

These fixtures certify the frequency guard and its bounded checks. They do not certify full Sonnet parity or all network behavior.
