# Bulk reciprocal component representability review

Finite mixed conductivities can retain both components during ComplexF64 input conversion but lose a nonzero component during inversion. A finite thin-volume fixture then scales that component back into the representable range after it has already been discarded. Four cases on both supported Julia versions were independently checked with 520-digit Decimal arithmetic from the exact stored Float64 inputs and the full-rooftop integral.

The constitutive adapter now rejects a nonzero reciprocal component that cannot survive ComplexF64 storage before any physical reaction uses it. PEC +Inf, pure real and pure imaginary conductivities, representable subnormal reciprocal components, the inverse algorithm and physical Gram integration retain their behavior.

Both versions pass 120 new assertions, including scalar/vector input rejection, error context, all three public solve methods and five large representable reciprocal controls. All 180 existing focused bulk and volume-port assertions retain their gates. Four healthy paths preserve every resistivity bit and add no measured allocation in five warmed samples. The first harness wrongly passed the UFFT memory keyword to dense methods; those failed logs remain preserved and the unchanged implementation passes the corrected calls.

The reproduction uses an extreme numeric-range fixture. This arithmetic guard does not resolve native volume skin-loss differences or certify realistic RFIC/native EM agreement. Full committed-source package suites and hosted CI remain separate requirements.
