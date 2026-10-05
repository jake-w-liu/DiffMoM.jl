# Radiation accumulator allocation repair

GitHub run 37275116099 at d7c84421 failed its original macOS ARM64 radiation
allocation assertion: 5,028,128 total bytes exceeded 2,506,752 output bytes
plus the unchanged 2,250,000-byte auxiliary allowance. This was a test failure,
not an insufficient-balance failure. The original job log is retained as
`data/github_ci_audit_20261005/job_111650327443.log`, SHA-256
`2bc7ac2a72120e9e0cbe8c27a80f3f039674e60ee34c217554495781a1d3b858`.

The radiation loop used two mutable three-element static magnitude vectors
per observation/basis pair. Their elimination depended on compiler inlining
and escape analysis. A process-only probe on Julia 1.13.1 reproduced this
sensitivity by marking the existing reduction classifier out of line:
2,606,203 bytes normally, 5,948,555 bytes with mutable vectors across that
boundary, and 2,606,283 bytes with immutable vectors across the same boundary.
The complete 128-direction, 408-basis radiation matrix matched bit for bit.
These are warm cumulative allocation measurements, not peak resident memory.

The repair uses immutable Vec3 magnitude bounds in radiation_vectors and
allows both immutable and existing mutable bounds in the shared classifier.
Component accumulation order, classifier arithmetic, exact retry threshold,
quadrature and radiation formulas remain unchanged. Physical-optics and PTD
callers continue using their existing mutable bounds. No allocation ceiling,
numerical tolerance, original fixture or immutable evidence was modified.

Eight regression assertions cover ordinary and cancelling reductions through
an explicit out-of-line boundary, mutable/immutable classification agreement,
zero warm allocations, zero bounds and non-finite bounds. Focused checks also
rerun the original cancellation, transverse-null, translated exact-geometry
and allocation regressions on Julia 1.12.7 and Julia 1.13.1 with bounds enabled.
Both pass; complete package runs and fresh hosted macOS verification remain
mandatory before claiming the repair has passed the full workflow.
