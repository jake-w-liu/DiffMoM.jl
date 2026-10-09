# Rational fit and certificate finite scaling

Rebase the archived rational-fit scale/storage/RMS/decay corrections onto
the current terminal-current and physical-power parent. Only VectorFit
changes; earlier current, power, selector and documentation fixes stay.

Both Julia versions reproduce finite-input failures in the positive-real
certificate: a maximum finite positive feedthrough overflows its symmetric
sum, and scaling the existing narrow active-band fixture by 2^970 produces
Infs in its Hamiltonian. Small amplitudes hit an unrelated 1e-30 floor.

Check the existing positive-residue certificate before the unsafe sum.
Normalize Hamiltonian admittance and frequency by actual stored component
maxima. Binary product/ratio arithmetic preserves finite intermediate
ranges. The imaginary-axis test uses the caller's relative tolerance in
those dimensionless units. Remove the 1e-30 balancing/feedthrough floors
and the dimensioned one-radian-per-second axis floor. Unrepresentable
normalized components produce an uncertified result rather than a false
positive. The existing payload policy requires its separate resource audit.

Independent quadratic roots of Re(Y(jw)) verify the narrow-band crossings.
All 49 new range/frequency/inference/ownership/state/resource checks and
217 unchanged rational-fit/SPICE/certificate controls pass on both versions.
One normalized residue matrix is reused across poles: the observed warmed
one-port active certificate drops from 9392 to 9296 bytes on Julia 1.13.
No whole-workload, peak-memory or universal mathematical proof is inferred.

The new 148 scale/storage/decay regression file is preserved from its
archived qualified source. Proper package, complete suites, native/docs,
static gates, main publication and actual hosted CI remain separate.
Local LLVM out-of-memory failures are preserved and publication queues
held; they are not scientific assertion failures or billing exceptions.
Remaining exact-pole/evaluation/borrowing and planar/RFIC work stays open.
