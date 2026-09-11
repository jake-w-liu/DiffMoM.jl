# PN-MoM implementation notes

## Development Ledger

| Item | Requirement and evidence |
| --- | --- |
| Objective | Implement the 3D Galerkin-consistent error and RCS decision method specified in project 2026_171/material/pn_mom_plan.md |
| Current slice | Retained checked solves; fixed-facet RWG nesting; consistent enriched restriction; inverse weighted-mass prior; row-adjoint output propagation; residual-conditioned conditioning; whole-case split-conformal calibration; pass/fail/unresolved screening with numerical budgets |
| Canonical owners | Workflow.jl prepares once; solver/RetainedSolve.jl reuses Solve.jl and IterativeSolve.jl; basis/NestedRWG.jl reuses Mesh.jl and RWG.jl; error_estimation/{GalerkinError,Conditioning,Calibration}.jl own the model |
| Compatibility | The default solve_scattering result is unchanged; return_state=true adds an opt-in named tuple |
| Contract | Finite physical RHS, Hermitian adjoints, unchanged factors/operators, invalidation after configuration mutation, all-column success before updating forward state, and bounded work |
| Independent oracles | Manufactured complex solutions; BigFloat solves; both-side field/divergence reconstruction; zero boundary flux; reversed edge orientation; dense enriched block identities; split-conformal rank checks |
| Baseline | Starting revision 61f3bd6af7dde5017226457d5c8b2ababe765f7f passed the full bounds-checked one-thread Pkg.test command and documentation build |
| Regeneration trigger | Source changes rerun focused tests, complete package checks, experiments, documentation, and any dependent scientific artifacts |
| Harness | bash /Users/jake/EMPIRE/projects/ongoing/2026_171/.agents/scripts/dev-harness-audit.sh . from this package root |
| Risks | Stale state, complex-conjugation mistakes, omitted unresolved output terms, complement rank, sparse-factor permutation, resource accounting, and overstated uncertainty claims |

## Repaired defects and dispositions

- Batched ACA dense blocks evaluated lower-triangular non-Bloch entries in
  reversed quadrature/index order, unlike canonical `_efie_entry`. Entry
  (65,17) differed at the 1e-3 level in the imaginary part. The batched fill
  now applies the canonical ordering; the regression test reproduces the
  mismatch and passes post-repair. Focused rerun: 26,155 checks.
- The trial-era population driver omitted `gmres_memory`, so reference level-3
  GMRES used the `solve_scattering` default of 20 and stagnated at 1500
  iterations. The diagnostic with `memory=80` converged in 164 iterations at
  the same source hash. The current drivers pass `gmres_memory=80`.
- `Project.toml` restricted the `SHA` stdlib to version 1, which does not
  exist for Julia 1.12 (SHA is a sysimage stdlib at 0.7.x). The bound is now
  `0.7`, and SHA was added to the test target for record-digest use.
- `MLFMA` at precision 9 produced a 2.07e-4 relative far-field difference and
  a 5.4e-4 sampled action error at 2130 unknowns, outside the declared 1e-4
  field tolerance. Its linear residual was 4.5e-10, so the operator solve
  converged while the compressed operator itself differed. Precision 9 is an
  explicit lower-precision comparison point; the declared gate requires ACA
  and precision 12. The precision-12 outcome is pending at this writing.

## Study workflow

`validation/pn3d/` contains the registered-population drivers:

- `generate_population.jl` writes the immutable 60/199/300 split, looks, and
  masks under a content-hashed manifest.
- `reference_population.jl` produces per-case level 0-3 dense/ACA reference
  records with per-level true residuals, measured algebraic output changes,
  and full cost fields. It is resumable per case and fails stale protocols.
- `reference_backend_comparison.jl` compares dense/ACA/MLFMA fields on one
  training case under a declared tolerance.
- `reference_solver_diagnostic.jl` isolates level-3 solver behavior.
- `study_predictions.jl` writes per-case prediction records covering levels
  0 and 1, probe counts 0/2/8/24, residual-only and global-covariance scales,
  and measured algebraic, restriction, and cost fields.
- `study_numerical_checks.jl` measures the shared quadrature (7 vs 28) and
  ACA-compression output changes on a designated training case.
- `study_analysis.jl` calibrates every configuration over the 199-case split
  and evaluates held-out test decisions over all registered masks with
  per-case numerical budgets, level escalation, and deployment-cost records.

Records are atomic, hash-guarded against protocol and source drift, and mark
incomplete runs rather than fabricating results.

## Remaining implementation

The 559-case reference and prediction campaigns are in progress. Held-out
evidence does not exist yet: no coverage, error-rate, or cost benefit is
established. Do not infer engineering benefit from the finite-system tests.

The paper and research records reside in
/Users/jake/EMPIRE/projects/ongoing/2026_171. Production code remains here.
Pre-existing edits to scripts/slopfix_lib/manifest.py and smells.py are not
part of the PN changes and must not be staged with them.
