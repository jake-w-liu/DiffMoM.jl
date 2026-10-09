# Finite power, pole ordering and rational evaluation

Radiation now rejects nonfinite accepted power before gain omission; the
existing infinity comparison incorrectly accepted negative infinite power.
Exact real/imaginary pole grouping and in-place interleaving preserve repeated
and adjacent conjugate eigenmodes without decimal rounding or extra vectors.
Stored incident waves are borrowed read-only, keeping returned voltage and
reflection buffers owned. Caller payload budgets lose exactly that buffer.
The boundary-first retained selector and its per-coordinate range regression
remain unchanged from the parent.

Rational evaluation promotes an unrepresentable angular-frequency intermediate
when its finite response still fits. Two stored Float64 significands determine
the fallback precision; caller MPFR precision and rounding are restored. The
direct promoted formula rejects MPFR's own exponent boundary without recursion.
Ordinary evaluation uses the original formula and measured96bytes in three
controls on both Julia versions. Corrected extreme controls allocate up to
3200bytes; this is a correctness cost, not an overall optimization claim.

Isolated combined properPkg2157 assertions pass both versions, including the
new15 finite-power,436 pole-order and56 evaluation tests and all original gates.
Current native130 cases on each version and documentation240 inputs pass.
The original SPICE1e-9, physical1e-9, field5e-11 and ordinary1e-13 gates remain.
The allocation comparison for incident borrowing is against the corrected
selector:80 through144bytes saved for1 through5ports. Seven controls allocate
an extra consistency vector compared with the earlier unsafe eager selector.
No total-fit runtime, allocator peak, universal cancellation or extreme RFIC
design guarantee is claimed. Complete committed suites, exact static accounting,
parent own CI and this head's own hosted CI remain required for publication.
