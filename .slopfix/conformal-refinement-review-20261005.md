# Low-frequency conformal residual refinement

Windows CI run 37319573889, attempt 1, failed the existing 1 MHz resistive
rectangle fixture: port 2's recomputed projected voltage residual was
1.3538556616417363e-10, above the unchanged 1e-10 limit after eight iterations.
This was a numerical failure, with no billing indication. The original log
is retained in data/github_ci_audit_20261005/job_111794887167.log.

Process-only tests on the unchanged a532fa5b source reproduced the same
plateau at 11 of 65 representable frequencies within 32 Float64 steps of
1 MHz. Eight warm retries and eight unscaled correction retries retained
all 11 failures. Krylov 0.9.10's gmres.jl also accepts an absolute recurrence
residual at machine precision, even when its relative target is smaller.
Normalizing the recomputed correction source, then scaling its solution
back, passed all 65 cases with the existing two-retry limit. The largest
initial iteration count was nine; both projected and original voltage
residuals remained below 1e-10, and the original DC resistance gate passed.
Raw experiment reports and unsuccessful harness attempts remain in data/.

The production change reuses defect_rhs and inner, adds no numeric arrays,
retains the total maxiter budget, and leaves the independent original modal
equation acceptance unchanged. New regression tests retain both residual
gates and the physical resistance gate across those 65 frequencies. Existing
tests, fixtures, workflows, allocation limits and accuracy limits are unchanged.

This evidence supports the correction algorithm and the nearby-frequency
reproducer. The exact GitHub Windows case still requires verification on the
patched commit; local passes alone do not prove all platform behavior or
complete native Sonnet parity.

Focused checks passed 992 assertions on each of Julia 1.13.1 and 1.12.7,
including all existing conformal defect tests and the new 260 assertions.
All 142 hashed inputs stayed unchanged during each run. The code counter
increased from 212635 to 212654 (+19); the original strict failure is retained.
The separate line-ceiling commit records exactly that reviewed increase.
