# Bounded retained-source refinement

The retained dense source path previously solved its equilibrated matrix once
and reported the ordinary voltage residual. On the archived 7180-unknown,
80-cell reactive tube at 100 MHz, that residual is 1.3033e-9, above the original
1e-9 physical limit. At 1 MHz it is 2.4926e-7. The material, matrix, source
contracts, reference impedances and native output remain fixed.

The source solve now tries at most eight LU corrections when a retained
column's ordinary residual exceeds its requested target. A compensated
matrix-vector residual drives the existing factorization. The calculation
supports general complex matrices and has a faster route for certified pure
imaginary matrices with imaginary currents and real sources. Two reusable
vectors fit the existing four-vector dense-source payload reservation.

The returned column changes only when both compensated and ordinary voltage
residuals improve. Corrections stop at the requested ordinary residual or when
the compensated residual stops improving. The original matrix and factor stay
unchanged. Dense results can still exceed the target and report that outcome;
the iterative full-operator residual gate and original acceptance limits are
unchanged. Matrix retention is needed to evaluate these corrections.

An independently checked diagnostic evaluates 73 selected rows in Decimal
against 128/256-bit sums, and 584 selected rows across sixteen correction
steps. The isolated implementation returns bitwise-equal S/Y/current data
to the checked first-correction result. Its 100 MHz residual is 9.0867e-10,
complex-S error 5.0659e-6 against the unchanged .005 gate, and real-impedance
error 4.2046e-13 ohm against the unchanged 1e-8 loss limit. At 1 MHz its
1.7761e-7 residual remains unverified. These are scoped archived-reference
checks on a uniform axial-tube candidate, not general SFC parity.

The isolated source copy passes six neighboring suites on Julia 1.13.1 and
1.12.7. Both versions pass 36 compensated-kernel checks spanning general
complex and imaginary matrices, complex sources, cancellation and scales
1e-200 to 1e200, with zero repeated Julia allocations in the kernel. An early
control runner had a global-status scoping error after all 36 checks passed;
its failed logs remain. The corrected runner passes without a source change.

Four healthy public dense/UFFT fixtures retain every S/Y/current and residual
bit. Five warmed whole-call allocation samples have overlapping ranges;
their minimum deltas are +80, +64, +32 and -144 bytes. This does not establish
zero whole-call allocations, a speed improvement or peak resident memory.

The original registered test file remains an exact prefix. New tests compare
the compensated kernel with a 256-bit oracle, preserve inputs and buffer reuse,
and improve independently computed physical residuals for perturbed pure
imaginary and complex retained systems. Candidate focused/full suites and
quality measurements must validate the actual committed source before push.

The actual candidate passes all six focused suites on both Julia versions,
including the 50 appended assertions. The pinned Julia 1.12.7 counter measures
213025 code lines, 112 above the published parent's 212913 ceiling. The
implementation and regressions are committed separately from that exact
measured ceiling adjustment. The original measurement failure is preserved;
blocking smells pass and the bounded 400-group duplication estimate is 58086,
below the unchanged 58394 duplication ceiling. Full-package and hosted CI
verification remain required before publication.
