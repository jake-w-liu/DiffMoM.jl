# Remaining rational ranges and sampled passivity

Rebase the archived exact pole grouping, finite angular-frequency
evaluation and read-only incident-wave borrowing onto the current
rational-fit and certificate parent. Preserve all newer current,
physical-power, selector and certificate changes.

Exact pole coordinates replace eight-significant-digit grouping, and
repeated conjugate modes interleave in place. Finite frequencies whose
angular intermediate overflows use two Float64 significands in scoped
MPFR arithmetic. Ordinary evaluation retains its original path.
Stored ComplexF64 incident waves are borrowed read-only. Retained
excitation payload counts follow the remaining three owned vectors.

Fresh public sampled-passivity probes on both Julia versions also show
that finite positive and negative maximum constant admittances throw:
the Hermitian sum overflows before its physical half is applied. Average
each pair in one owned matrix. Retain ordinary sum-then-half arithmetic
for subnormals and only split the operands when finite inputs overflow.
No chosen range threshold or new scientific tolerance is introduced.

All 819 scoped controls pass on both versions: 69 original rational
model/SPICE, 148 fit ranges, 49 certificates, 436 exact pole grouping,
56 finite evaluation and 61 sampled means. An initial negative two-port
fixture had a genuinely unrepresentable eigenvalue; its failed run is
preserved. Corrected fixtures use independent finite quadratic spectra
and the existing 1e-9 gate. Maximum and subnormal scalar margins remain
exact checks. Caller storage, precision and rounding are covered.

Proper package, native, docs, static, complete suites, allocations,
main publication and own hosted CI require their separate current
qualification. Full numeric, resource, performance and RFIC acceptance
remain open; scoped regression passes do not close the broader audit.
