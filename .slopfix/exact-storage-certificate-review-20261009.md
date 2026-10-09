# Exact stored-energy positive-real certificates

Four confirmed finite-input false certificates were reproduced on both
Julia versions before these changes. LAPACK scaling erases an indefinite
small block beside a large conductance. Halving a minimum subnormal
off-diagonal erases a negative Hermitian energy direction. A rounded
spectral norm understates a residue bound. Finally, absence of floating
Hamiltonian imaginary-axis eigenvalues certifies a model whose exact
stored DC energy is negative, even after a rigorous norm bound rejects it.

Check original stored Hermitian sums with exact dyadic fraction-free PSD
elimination, including singular zero-pivot rows. Complex realification
preserves the Hermitian quadratic form. Five fixed GMP scratch roles
reuse capacity derived from entry ranges, dimensions and minor bounds.
Do not round the original energy matrix before checking its sign.

The uniform sufficient proof uses a rigorously upward-rounded Frobenius
bound. Exact IEEE squares use the existing derived product lattice;
sqrt, division, sum and Float64 conversion round upward. Five owned
MPFR roles preserve caller precision and rounding. Exact shifted
feedthrough PSD proves the resulting lower bound.

Hamiltonian roots retain potential crossing diagnostics. A positive
certificate additionally needs an exact positive-real storage inequality
for the original model, following the positive-real lemma (Boyd et al.,
Linear Matrix Inequalities in System and Control Theory, section 2.7.2).
An ordered stable Schur subspace proposes storage; a Lyapunov increment
moves into its inequality interior with magnitude derived from curvature.
The normalization corresponds to original storage a*f*P. Exact integer
verification clears its positive dyadic denominator. A failed numerical
proposal stays uncertified. No grid assertion replaces the global proof.
Original fitting accuracy and SPICE contracts are unchanged.

KYP assembly borrows the original pole/residue/feedthrough coefficients;
it owns one symmetric storage and one symmetric inequality integer matrix
and seven named scalar buffers. Sparse block structure removes temporary
dense exact A/B/C arrays. Entry polynomial degree and principal-minor
bounds derive limb capacity and raw memory preflight. Named dense stages
replace the old unexplained 16*states^2 reserve. Raw payload excludes
Julia object headers and opaque LAPACK workspace; no peak RSS claim.

All original 865 scoped controls plus 478 independent exact principal-
minor, real/complex energy, shifted PSD, mixed-residue storage, allocation
budget, ownership, caller-state and concurrency checks pass on both
Julia versions. The independent storage oracle uses exact rational matrix
products and Laplace principal minors, rather than the production integer
builder or Bareiss kernel. Earlier prototype failures stay archived.

This candidate descends from held 33af and keeps publication base 364e6,
including all earlier retained-current, power, wave-borrowing, rational,
pole grouping, finite evaluation and sampled-Hermitian fixes. Current
proper, native, docs, static, allocation, complete-package and own hosted
CI qualification remains required before main publication. Broad numeric,
resource/performance and planar/RFIC acceptance audit remains active.
