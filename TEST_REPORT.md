# PN-MoM test record

## Coverage and independent oracles

The original full Pkg.test run passed at
61f3bd6af7dde5017226457d5c8b2ababe765f7f with one thread and bounds checking.
The original documentation build passed with HTML-size warnings.

The integrated test/test_discretization_error.jl run passed these groups
with one and four threads before the latest last_solve initialization change:

| Group | Passed assertions |
| --- | --- |
| Retained checked solves | 29 |
| Workflow/backend integration and shared-state adjoints | 72 |
| Fixed-facet RWG nesting | 877 |
| Galerkin block and complete-field identities | 5 |

Manufactured complex currents supply RHS expectations independently of the
solver. BigFloat calculations check extreme dense solves. Pointwise fields,
divergence, boundary traces, and orientation changes check the actual RWG
injection. An independently solved enriched complex system checks block
reconstruction and both far-field error terms. All four workflow backends
are exercised, including concurrent calls serialized through one state.

## Tolerance justification

The small manufactured systems are diagonally dominant; 1e-12 relative
differences allow ordinary Float64 rounding. Iterative comparisons use
1e-10 to 1e-9 against a configured 1e-11 Krylov tolerance. Shape-regular
affine reconstruction uses 1e-11 relative tolerance and explicit absolute
scales for zeros. These are not electromagnetic continuum-error thresholds.

## Anti-False-Test analysis

Complex manufactured solutions reject use of a transpose instead of an
adjoint. Token identity checks reject unnecessary reassembly/refactorization.
Geometry/operator/frequency/quadrature mutation checks reject stale reuse.
Field and divergence reconstruction reject copied coefficients or dropped
signs. Boundary checks include the slot. A deliberately omitted direct
unresolved field contribution fails the enriched-output oracle.

## Current limits

The latest initial-report clarification passed all five new assertions in a
four-thread rerun: 30 retained-solve, 76 integration, 877 nesting, and 5 block
identity assertions, for 988 total. The latest documentation build also passed.
The strict line check passed after the separate a6d4ac5 ceiling commit.
Full updated Pkg.test, development harness, reviewed quality contract, and
final change review remain pending.
The posterior, calibration, scientific campaign, and final code push are not
complete. Existing focused passes do not establish those later requirements.

Completed focused logs: /tmp/pn3d-integrated-pr12-1t.log and
/tmp/pn3d-integrated-pr12-4t.log. Baseline log:
/tmp/pn3d-baseline.MMllkK. Documentation log: /tmp/pn3d-docs-pr12.log.

Latest focused log: /tmp/pn3d-pr12-final-focused.log. Latest documentation:
/tmp/pn3d-docs-final-pr12.log. Active full four-thread test log:
/tmp/pn3d-pkg-pr12-4t.log. Active one-thread research-harness log:
/tmp/pn3d-development-pr12.log. The latter recorded false positives from a
now-corrected project audit script and cannot count as a passing harness run.

## Current PN3D verification: 2026-09-11

This checkpoint updates the earlier feature status. It does not declare the
engineering study or manuscript complete.

- The ACA dense-block ordering defect was repaired in
  `src/fast/ACA.jl::_fill_dense_block_batched!` and covered by the
  `batched ACA canonical entry order` regression (3 checks) in
  `test/test_workflow_aca_budget.jl`. A focused bounds-checked rerun of the
  discretization/ACA/workflow groups passed 26,155 assertions.
- `test/test_pn_study.jl` passes 40 checks covering prediction record
  contracts, Richardson fitting, whole-case scores, calibration quantiles,
  per-mask decision-cost accounting, numerical-budget assembly, and the
  dense/ACA algebraic-correction paths.
- `Project.toml` now bounds `SHA` at `0.7` (Julia 1.12 ships SHA 0.7.x as a
  sysimage stdlib) and adds SHA to the test target. A bounds-checked
  one-thread `Pkg.test()` passed after the fix (log
  `/tmp/pn3d-pkg-full-1t-v4.log`); it ran while study-driver edits were in
  flight, so a final clean one/four-thread rerun remains required.
- Formal backend comparison for `train-0001` of the registered population:
  ACA vs dense relative field difference 2.4748839317291588e-11 (passed at
  the declared 1e-4 tolerance); MLFMA precision 9 failed at
  2.0710541089026974e-4 with a 5.44e-4 sampled action error despite a
  converged 4.53e-10 linear residual. MLFMA precision 12 was still running
  at this writing. Output:
  `.workflow/backend_comparison_formal_after_aca_v2/comparison.json`.
- The earlier level-3 reference GMRES stagnation is explained by the
  trial-era driver omitting `gmres_memory` (default 20). A same-source-hash
  diagnostic with memory 80 converged in 164 iterations at
  1.82e-10 relative residual for 34,872 unknowns. Current drivers pass
  `gmres_memory=80`.
- Three resumable reference-population processes are running over the
  registered train/calibration/test splits into
  `.workflow/reference_population_formal` (Julia 4 threads, bounds checks
  on, one OpenBLAS thread). No complete campaign result exists yet.
- `slopfix smells --severity blocking --strict` reports zero hits and
  `git diff --check` is clean. Current code measures 94,368 lines against
  the committed 92,327 ceiling; a deliberate ceiling update for the study
  evaluation/checks code is required in its own commit before push.
- Prediction generation, numerical checks, calibration, held-out
  evaluation, manuscript completion, the development harness, quality
  contract, final review, commit, and push all remain pending.

Earlier verified results are retained below.
- `examples/experiments.jl` passed mesh, patch-mass, and independent
  full-information field recovery checks. Current documentation built with
  export checks. Existing page/search-size warnings remain below hard page
  limits. Logs: `/tmp/pn3d-experiments-expansion.log`,
  `/tmp/pn3d-docs-expansion.log`.
- Focused repair checks passed: residual precision 62, expansion saturation 6,
  batched products 43, residual-work budgets 31, and mesh aspect 7. The
  original sphere input required 17 expansion components; its 16-slot failure
  now uses the same exact MPFR row calculation without dropping terms or
  enlarging an unsupported mathematical bound. The regression fixture stores
  inputs, not expected results. Log: `/tmp/pn3d-expansion-fix-tests-v2.log`.
- The 21-case-configuration sphere validation completed with unchanged source
  `2effc8671fda2ba08124869d495da2a9af7e694f074c4f6563b7c4fcd5b85db3`.
  At 1920 RWGs and 28-point quadrature, relative complex-field differences
  from Mie were 0.0083607664, 0.0053667914, and 0.0076169735 at ka=0.2, 1, and 2.
  All declared field/RCS/phase/residual checks passed. Fixed-facet results are
  separate from projected-sphere results. Log:
  `/tmp/pn3d-sphere-validation-v2.log`.
- Both Python 3.9 and Python 3.14 passed all 23 validation-report tests after
  repairing Python 3.9 angular-key alignment and the PEC validator's actual
  project-root path handling. The latter also passed a real Bempp field case
  with project-relative paths and the unchanged 2-percent tolerance. Logs:
  `/tmp/pn3d-validation-reports-py39-fixed.log`,
  `/tmp/pn3d-validation-reports-py314-fixed.log`,
  `/tmp/pn3d-bempp-project-root-integration.log`.
- All 332 exported names were observed as defined. Log:
  `/tmp/pn3d-current-export-inventory.log`.
- Official Gitleaks 8.30.1 binary SHA256:
  `b40ab0ae55c505963e365f271a8d3846efbc170aa17f2607f13df610a9aeb6a5`.
  Its history scan covered 581 commits with no leaks. Four working-tree
  findings were inspected as generated HTML heading fragment IDs on the
  verification, density-topology, adjoint-optimize, and assembly-solve API
  pages. They are page anchors derived from the corresponding headings.
  No allowlist was changed. Logs: `/tmp/pn3d-gitleaks-history.log`,
  `/tmp/pn3d-gitleaks-working-tree.log`; redacted report:
  `/tmp/pn3d-gitleaks-working-tree.json`.

Current aggregate source coverage was not measured; the earlier percentage is
historical. Remote CI, cross-version serialization, and optional advisory/SBOM
coverage remain unverified. Full current quality-contract and development
harness runs, the reviewed line-ceiling update, final review, and push remain
pending. No main-study cost, coverage, or practical-usefulness verdict is made.
