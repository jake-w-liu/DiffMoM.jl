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
back, passed all 65 cases on the native CPU target with two retries. The largest
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

A generic CPU-target check then exposed four remaining failures among the
same 65 frequencies with only two normalized retries. That unsuccessful
candidate and its log are retained. A process-only comparison with up to
eight normalized corrections passed all 65 cases on both CPU targets;
the generic target needed at most three corrections (12 initial iterations).
The final implementation allows up to eight retries and exits as soon as
the original residual gate passes, still within the original total maxiter.
Neither extra unscaled retries nor tolerance relaxation is used to accept
the result. Final focused and full checks must use this updated source.
Updated focused checks passed 992 assertions on each Julia version; strict
measure, blocking smells and the 400-group census passed at ceiling 212654.

Hosted f9a68370 then exposed a remaining ARM64 failure at the frequency eight
Float64 steps below 1 MHz: port 1's projected residual stayed at
1.1525489566043346e-10 after 27 iterations. Its original failed macOS log
is retained (job 111841929908, SHA c04142fee73e262587593f51091d33bfd374e96bad61bb63b0119392cc024617).
This is numerical; the runner-capacity notice is not a billing explanation.

ARM64 diagnostic run 37337785316 reproduced that exact failure with unchanged
production arithmetic. Fusing the scaled correction into the addition also
failed at the same frequency. Preserving both product and addition roundoff
between updates passed all 65 cases (maximum 24 initial iterations), with
the unchanged projected/original/DC/iteration gates. Native, generic and
Haswell CPU-target comparisons also passed all cases. The diagnostic producer
recorded the baseline failures rather than certifying them as passing tests.

The production correction now reuses the initial Krylov solution buffer as
the update carry, clearing it only when a retry is actually needed. Every
acceptance residual is still computed from the stored Float64 currents.
No new scratch arrays or numeric payload reservations are added. A separate
256-bit oracle verifies product and sum cancellation, including ordinary
update failure and zero helper allocations. Updated focused tests passed
999 assertions on each Julia version; exact production ARM64 and full-suite
verification remain required before publishing this update.
The counter increased from 212654 to 212690 (+36) for the helper and oracle;
the original strict failure is retained. A separate ceiling commit records
exactly this reviewed increase, with the duplication ceiling unchanged.
