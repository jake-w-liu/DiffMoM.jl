# Slop ledger — DiffMoM.jl

Census taken at commit `2c31d91064b07758eac5debdf9e710934e96bc17` on
2026-08-24. Estimates are candidates, not permission to edit or delete.

## Ledger

| ID | Cat | Description | Sites | Est. net lines | Risk | Inventory | Status | Commit |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| SL-001 | A | Seven exceptional ComplexF64 `α*x + β*y` BigFloat reducers repeat the same conversion, overwrite, finite-check, and row-error behaviour while using subsystem-specific precision/labels | `src/assembly/CompositeOperator.jl:194`, `src/fast/ACA.jl:30`, `src/fast/MLFMA.jl:2282`, `src/mom3d/DDA3D.jl:79`, `src/mom3d/SurfaceIE3D.jl:1312`, `src/optimization/QMatrix.jl:413`, plus one census site to resolve | 85–110 | R2 | INV-010, 012, 017, 018, 027–029, 040–043 | open; behavioural diff/approval required | |
| SL-002 | A | Two large MLFMA sections are token clones and may be forward/adjoint variants with different ordering and conjugation | `src/fast/MLFMA.jl:3038`, `:3355` | 0–45 | R3 | INV-018, 040–043 | open; read both completely | |
| SL-003 | A | FFT electric and coupled EM kernel/operator setup sections repeat structure but may encode different tensor/block semantics | `src/mom3d/FFTDDA3D.jl:292`, `:600` | 0–40 | R3 | INV-028, 040–043 | open; behavioural diff required | |
| SL-004 | A | Repeated example bootstrap blocks activate/instantiate DiffMoM and create output directories | seven example files beginning near lines 8–35 | 45–65 | R1 | INV-031, INV-033 | open; example command compatibility approval required | |
| SL-005 | A | `ensure_gmsh_on_path` is duplicated in two Bempp drivers | `validation/bempp/run_impedance_cross_validation.py:17`, `run_pec_cross_validation.py:29` | 8–14 | R1 | INV-035, INV-036 | open; external workflow approval required | |
| SL-006 | C | `Tuple` is imported but has no static reference | `validation/bempp/sweep_impedance_conventions.py:14` | 1 | R1 | INV-035 | audited across imports, annotations, dynamic lookup, docs, and history; deletion awaits explicit approval | |
| SL-007 | G | Scanner flags abstract incident-field fallbacks as placeholders | `src/assembly/Excitation.jl:2484`, `:2561` | 0 | R1 | INV-007, INV-014 | false positive: intentional explicit unsupported-type errors | |
| SL-008 | G | Scanner flags numeric conversion probe as a swallowed error | `src/postprocessing/Diagnostics.jl:32` | 0 | R1 | INV-013, INV-040 | false positive: conservative precision fallback | |
| SL-009 | H | Aqua was unreachable from the normal test graph; package tests included source directly instead of loading the installed package | `test/runtests.jl`, `Project.toml`, `.slopfix/quality.json` | negative | R2 | INV-001, INV-045, INV-048 | complete: tests load the package and run Aqua in the normal graph | pending final audit commit |
| SL-010 | H/J | Allocation assertions existed without a consolidated representative latency/allocation baseline | Q-matrix regression and `.slopfix/quality.json` | negative | R2 | INV-013, INV-041 | complete: warm dense-Q allocation and latency evidence recorded; allocation ceiling enforced by test | pending final audit commit |
| SL-011 | K | Documenter disabled public-doc coverage checking with `checkdocs=:none` | `docs/make.jl`, generated exported-docstring pages | negative | R2 | INV-032, INV-045, INV-049 | complete: `checkdocs=:exports`; every export is defined and rendered | pending final audit commit |
| SL-012 | D | `test/runtests.jl` is a 14,319-line sequential test driver; source god files include Mesh/MLFMA/Excitation/DDA | `test/runtests.jl`, largest source files in `.slopfix/triage.md` | 0 | R3 | all | deferred unless a behaviour-preserving split is separately approved | |
| SL-013 | C/E | Internal definitions with one textual occurrence may be dead or public/dynamic extension points | whole `src/` public/internal symbol census | 15 | R3 | INV-045, INV-047 | 15 private helpers audited with no callers, exports, docs, reflective use, or extension contract; deletion awaits explicit approval | |
| SL-014 | J | One cancellation-sensitive dense-Q entry forced every entry through BigFloat; the replacement BLAS path initially missed cancellation in one complex component | `src/optimization/QMatrix.jl`, Q regressions in `test/runtests.jl` | negative | R3 | INV-013, INV-040–042 | complete: local checked recomputation, component-specific regression, exact Hermitian completion, and measured allocation/latency | pending final audit commit |
| SL-015 | J | The high-level iterative workflow trusted solver status without checking the returned vector against the selected operator | `src/Workflow.jl`, workflow regressions | negative | R3 | INV-011, INV-019, INV-040 | complete: configurable true-residual gate is enabled by default and partial-iterate retrieval is explicit | pending final audit commit |
| SL-016 | J/K | Examples and validation scripts could print completion or write plausible results without a nonzero acceptance failure; one Mie run overwrote a tracked STL fixture | `examples/01_pec_plate_basics.jl`, `examples/04_pec_sphere_mie.jl`, `examples/13_sphere_rcs_optimization.jl`, `validation/` | negative | R3 | INV-031, INV-033, INV-034, INV-040, INV-048 | complete for the touched bounded workflows: named gates derive from executable constants, non-finite metrics fail, completion follows verification, the README plate mesh meets its stated resolution target, and STL round-trips use temporary files | pending final audit commit |
| SL-017 | K | User guides duplicated mutable defaults, mixed tasks, and retained unsupported or misleading workflow claims | `README.md`, changed `docs/src/` guides, Bempp help/readme | negative | R2 | INV-031, INV-032, INV-049 | complete for the edited surface: canonical source pointers, task-focused pages, effective-value output, actionable errors, and link/build checks | pending final audit commit |
| SL-018 | F | Regression tests wrote scratch mesh files into the repository `data/` directory | `test/runtests.jl` | negative | R2 | INV-003, INV-048 | complete: scratch files use one test-owned temporary directory | pending final audit commit |
| SL-019 | L | No repository ratchets guarded line growth, blocking smells, duplication, or the wider quality contract | `.github/workflows/ci.yml`, `AGENTS.md`, `CLAUDE.md`, `.slopfix/`, `scripts/` | negative | R2 | INV-041–046, INV-048 | complete: CI and local commands enforce the committed ceilings and quality contract | pending final audit commit |

## Behavioural diff — SL-001 (pending full site resolution)

| Condition | Current copies | Proposed shared primitive | Decision |
| --- | --- | --- | --- |
| `overwrite=true` | Convert `α*value`; throw subsystem-labelled `OverflowError` if non-finite | Same, with precision and label supplied by caller | pending approval |
| `overwrite=false` | Compute exact BigFloat `α*value + β*previous`, convert once | Same | pending approval |
| fallback precision | Subsystem constants differ | Preserve caller-provided precision exactly | pending approval |
| fast-path trigger | Subsystems use different certificates | Remain subsystem-local; only exceptional reducer is shared | pending approval |
| error wording | Subsystem-specific | Preserve exact label text through caller argument | pending approval |

## Deferred/rejected findings

| ID | Reason | Evidence |
| --- | --- | --- |
| SL-007 | Explicit abstract fallback is required for unsupported `AbstractExcitation` subtypes and is exercised through public total-field validation | `src/assembly/Excitation.jl:2483-2561` |
| SL-008 | `nothing` is a capability result consumed by `_projected_power_fallback_precision`, not an ignored operational failure | `src/postprocessing/Diagnostics.jl:9-69` |

## Audited deletion set awaiting approval

The audit found no static caller, export, documentation reference, reflective
lookup, or repository extension contract for these private definitions:

- `_aca_internal_product_requires_fallback`
- `_filter_2step`, `_interp_2step`, `_anterp_2step`
- `_assert_gmres_result`
- `_scale6_matrix_3d`, `_inv_scale6_matrix_3d`
- `_jacobian_scattered_field_high_precision_2d`
- `_local_mass_component_scale`, `_local_mass_entry_bigfloat`
- `_mie_rcs_from_amplitude`
- `_planewave_phase_bigfloat_dda_3d`
- `_po_farfield_contribution_geometry_exact`
- `_surface_sie_block_sum_bigfloat_3d`
- `_build_q_checked_into!`

The unused `Tuple` import in
`validation/bempp/sweep_impedance_conventions.py` is the sixteenth candidate.
No deletion is recorded until the user explicitly approves this set.

## Running totals

| | Lines |
| --- | ---: |
| Baseline code lines | 74,755 |
| Raw census estimate (unverified) | 7,738 |
| Promised net reduction | 150 |
| Gross removed so far | 239 |
| Gross added so far | 6,586 |
| Net removed so far | -6,346 |
| Remaining to target | 6,496 |
