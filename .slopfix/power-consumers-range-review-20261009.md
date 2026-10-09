# Finite physical power through public consumers

Both supported Julia versions reproduce a coherent project radiation
failure where the terminal current, coefficients, fields and physical
half-power fit, while the unscaled dot overflows. The original source
equation residual is exactly zero. A rational reference independently
checks the final power. These are controlled numerical API regressions,
not assembled device accuracy references.

Extend the archived public power-wave recovery to EM/project physical
accepted-power consumers. Ordinary dot arithmetic remains unchanged.
Only a nonfinite result uses four owned MPFR scalar roles, full IEEE
product-lattice precision shared with the terminal-current kernel, and
the caller's existing retained workspace budget. Divide by the physical
peak-phasor factor two before narrowing. No selected numeric cutoff or
precision is introduced. Circuit accepted-power dots use the same helper;
circuit matrix products and S-only energy subtraction remain separate.

The archived finite-power guard is included here: true nonfinite negative
power cannot pass an infinite roundoff allowance and silently omit gain.
Its existing 15 regressions reproduce 14 passes/one failure on both
versions before the guard and all 15 afterward. Existing public-wave
58 and new coherent EM/project half-power 15 regressions also pass both.
Original physical, wave, field and source-equation requirements stay.

This extracts the relevant archived power and finite-pin fixes onto the
current terminal-current parent. Remaining rational/pole/evaluation/
borrowing fixes require their own rebase. Proper/full/native/docs/static,
ordinary allocation measurements, main publication and hosted CI require
separate evidence. Historical failures and frozen inputs are preserved.
