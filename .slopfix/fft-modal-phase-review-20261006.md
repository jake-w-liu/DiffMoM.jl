# Retained FFT modal phase

The retained FFT assembly can add tiny real entries to a purely imaginary
modal matrix. A closed, reactive 124-unknown specimen at 1 MHz has exactly
zero real entries with independent direct modal assembly. The previous FFT
matrix instead has a maximum real entry of 2.2823e-17; its solved current has
a real component up to 0.8099 and its power unitarity error is 1.8752e-10.
The corresponding direct solution has zero real current and a 7.3474e-16
unitarity error. These are finite-matrix numerical comparisons, not native
continuum acceptance.

The assembly now records whether every actual modal TE/TM coefficient has
exactly zero real part. In that case it rotates the known imaginary factor
out of the existing Fourier buffer, performs one backward FFT, and restores
that factor when gathering the real spatial sin/cos kernel. General complex
modal data take the original path. Conductor loss is added afterward and
retains every real entry. No tolerance or small-value cutoff is used.

The change covers retained modal kernels, bounded folded kernels, and dense
extraction from an operator with folded source images. The former three-field
private folded-workspace constructor remains available with an uncertified
phase flag. No extra Fourier buffer or dense matrix is introduced. This does
not change the iterative operator's matvec algorithm.

An isolated source-copy prototype passed 108 checks over 48 comparisons with
independent direct assembly, including reactive, resistive/reactive conductor
and lossy dielectric controls, both retained and folded routes. The corrected
specimen has zero real current and power unitarity error 4.4504e-16 at 1 MHz
and 6.8574e-16 at 100 MHz. The original voltage residual gate is not claimed
to pass at 1 MHz; the separate low-frequency conditioning gap remains open.

The appended registered fixture is self-contained. Its 124 new assertions
cover both public assembly routes, folded/unfolded operator extraction,
imaginary phase, preservation of physical real loss and the original matrix
comparison limit. On the published parent the new assertions produce 35
failures with zero errors. All 124 pass on the candidate with Julia 1.13.1
and 1.12.7; the original test file is retained as an exact prefix.

The first broader focused runner omitted its dense-workspace fixture producer;
its failed captures remain in `data/fft_modal_phase_after_v1_*`. The corrected
five-suite runner passes on both Julia versions. Audit artifacts are retained
under `data/fft_modal_phase_*` and `data/surface_phase_preserving_*`.

The pinned Julia 1.12.7 quality counter measures 212913 code lines, 78 above
the parent's 212835 ceiling. The implementation and regression are committed
separately from the exact measured ceiling adjustment. The original measurement
failure is retained; blocking smell and bounded duplication checks pass.

No native reference, acceptance limit, duplication ceiling or CI workflow is
changed. Full Sonnet parity, peak resident memory and universal speed or
allocation improvements remain unproved.
