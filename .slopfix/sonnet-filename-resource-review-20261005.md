# Reject invalid filename solve budgets before input work

The filename overload decoded every native record before its parsed-project
solver validated max_bytes. Valid 2, 128 and 1,024-layer files rejected a zero
budget only after 52,183, 253,619 and 1,644,689 warm cumulative allocation
bytes. Calling the parsed geometry overload rejected in 1,424 bytes. The
retained v2 reproduction includes exact input hashes and an accepted geometry
control. The first v1 capture was automatically cleaned up; the resulting
prototype harness failure is preserved alongside the corrected retained run.

The process-only prototype calls the existing positive Int-sized budget
validator before reading the filename. All three cases reject in 512 warm
bytes. Complete geometry S, Y and currents match bit for bit; native circuit
DC and ordinary-frequency controls pass. These are cumulative allocation
measurements, not peak process memory. The production forwarding method
matches the verified prototype exactly and does not change frequency rules.

Twenty-eight new assertions cover invalid zero, negative, out-of-range and
noninteger budgets, bounded rejection across source sizes, rejection before
opening a missing source, unchanged geometry results and accepted native
DC/AC resistor responses at a 624-byte numeric budget. The latter preserves
valid circuit budgets below the geometry solver's separate 1 KB minimum.
Both Julia 1.12.7 and 1.13.1 also pass all 52 original raster resource assertions;
143 hashed source/test inputs remain unchanged during each focused run.

This repair concerns invalid budget values. Positive-budget file parsing and
aggregate peak-memory admission are not established by these measurements.
Original numerical/allocation gates and native archives are unchanged. The
pinned Julia 1.12.7 counter measures 212,635 code lines, an increase of 51 over
the 212,584 ceiling. Integrity/warnings are empty and blocking smells are zero.
The unchanged 400-group census estimates 58,090 removable lines under its 58,394
ceiling. The exact line-counter adjustment is committed separately; immutable
full-package checks and exact-main hosted CI remain required before a green
publication verdict.
