# Reviewed contracted residual workspace reuse

The dense and FFT source solvers allocated weighted source and residual
vectors for each retained port residual. The shared calculation now uses
the existing residual buffer for both weighted norms. Galerkin error is
computed before weighting that buffer. Source vectors, coefficient columns,
the original voltage gate and owned-payload reservation are unchanged.

Actual public before/after captures use 26- and 488-unknown fixtures, both
dense and FFT solvers, and five warmed samples each. Every after sample
allocates less than every before sample. At 488 unknowns the minimum drops
by 31548 bytes for dense solves and 31516 bytes for FFT solves. Complete S/Y,
all current coefficients, both residual vectors and iteration counts retain
identical stored Float64 bits. This is allocation-byte evidence for these
fixtures, not peak RSS or a universal performance claim.

Both supported Julia versions pass the eight focused suites. Eight new
assertions verify zero repeated allocation of the residual operation and
unchanged source/current inputs for the actual dense and FFT input forms,
using the existing mixed sheet/via fixture. Original registered tests remain
unchanged. The initial comparison script's raw-byte-only source assertion
encountered CRLF/LF worktree differences; that script is retained. The corrected
comparison independently rechecks every actual raw source hash before
normalizing only line endings and verifies a single changed source module.

The pinned line counter measures 212807 code lines, an increase of 26 from
212781 for the shared calculation and registered allocation checks. The
ceiling adjustment is recorded separately. The unchanged duplication ceiling
is 58394; the bounded 400-group census estimate remains 58094.

Full frozen package tests, pinned quality checks and all seven GitHub checks
remain required before publication. Native Sonnet parity remains a separate
open requirement.
