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
| SL-006 | C | `Tuple` was imported but had no static reference | `validation/bempp/sweep_impedance_conventions.py` | 1 | R1 | INV-035 | complete: removed after the user approved the audited deletion set | `dea2628` |
| SL-007 | G | Scanner flags abstract incident-field fallbacks as placeholders | `src/assembly/Excitation.jl:2484`, `:2561` | 0 | R1 | INV-007, INV-014 | false positive: intentional explicit unsupported-type errors | |
| SL-008 | G | Scanner flags numeric conversion probe as a swallowed error | `src/postprocessing/Diagnostics.jl:32` | 0 | R1 | INV-013, INV-040 | false positive: conservative precision fallback | |
| SL-009 | H | Aqua was unreachable from the normal test graph; package tests included source directly instead of loading the installed package | `test/runtests.jl`, `Project.toml`, `.slopfix/quality.json` | negative | R2 | INV-001, INV-045, INV-048 | complete: tests load the package and run Aqua in the normal graph | `62f59e6` |
| SL-010 | H/J | Allocation assertions existed without a consolidated representative latency/allocation baseline | Q-matrix regression and `.slopfix/quality.json` | negative | R2 | INV-013, INV-041 | complete: warm dense-Q allocation and latency evidence recorded; allocation ceiling enforced by test | `62f59e6` |
| SL-011 | K | Documenter disabled public-doc coverage checking with `checkdocs=:none` | `docs/make.jl`, generated exported-docstring pages | negative | R2 | INV-032, INV-045, INV-049 | complete: `checkdocs=:exports`; every export is defined and rendered | `62f59e6` |
| SL-012 | D | `test/runtests.jl` is a 14,480-line sequential test driver; source god files include Mesh/MLFMA/Excitation/DDA | `test/runtests.jl`, largest source files in `.slopfix/triage.md` | 0 | R3 | all | deferred unless a behaviour-preserving split is separately approved | |
| SL-013 | C/E | Internal definitions with one textual occurrence may be dead or public/dynamic extension points | whole `src/` public/internal symbol census | 15 | R3 | INV-045, INV-047 | complete: 15 private helpers with no callers, exports, docs, reflective use, or extension contract were removed after explicit approval | `dea2628` |
| SL-014 | J | One cancellation-sensitive dense-Q entry forced every entry through BigFloat; the replacement BLAS path initially missed cancellation in one complex component | `src/optimization/QMatrix.jl`, Q regressions in `test/runtests.jl` | negative | R3 | INV-013, INV-040–042 | complete: local checked recomputation, component-specific regression, exact Hermitian completion, and measured allocation/latency | `62f59e6` |
| SL-015 | J | The high-level iterative workflow trusted solver status without checking the returned vector against the selected operator | `src/Workflow.jl`, workflow regressions | negative | R3 | INV-011, INV-019, INV-040 | complete: configurable true-residual gate is enabled by default and partial-iterate retrieval is explicit | `62f59e6` |
| SL-016 | J/K | Examples and validation scripts could print completion or write plausible results without a nonzero acceptance failure; one Mie run overwrote a tracked STL fixture | `examples/01_pec_plate_basics.jl`, `examples/04_pec_sphere_mie.jl`, `examples/13_sphere_rcs_optimization.jl`, `validation/` | negative | R3 | INV-031, INV-033, INV-034, INV-040, INV-048 | complete for the touched bounded workflows: named gates derive from executable constants, non-finite metrics fail, completion follows verification, the README plate mesh meets its stated resolution target, and STL round-trips use temporary files | `62f59e6` |
| SL-017 | K | User guides duplicated mutable defaults, mixed tasks, and retained unsupported or misleading workflow claims | `README.md`, `docs/src/`, Bempp help/readme, user-facing diagnostics | negative | R2 | INV-031, INV-032, INV-049 | complete: canonical source pointers, task-focused pages, effective-value output, actionable recovery, direct capability language, evidence-scoped performance guidance, and link/build checks | `62f59e6`, `e6a2afa` |
| SL-018 | F | Regression tests wrote scratch mesh files into the repository `data/` directory | `test/runtests.jl` | negative | R2 | INV-003, INV-048 | complete: scratch files use one test-owned temporary directory | `62f59e6` |
| SL-019 | L | No repository ratchets guarded line growth, blocking smells, duplication, or the wider quality contract | `.github/workflows/ci.yml`, `AGENTS.md`, `CLAUDE.md`, `.slopfix/`, `scripts/` | negative | R2 | INV-041–046, INV-048 | complete: CI and local commands enforce the committed ceilings and quality contract | `62f59e6` |
| SL-020 | B | Broad dependency imports obscured the package's actual namespace requirements and produced 59 potential implicit-import findings, including four local-binding misattributions | `src/DiffMoM.jl`, four redundant file-local imports, `test/runtests.jl` | negative | R2 | INV-045 | complete: dependency names are imported explicitly; all import/owner checks pass; the 11 reviewed non-public qualified names are pinned to an exact allowlist | `89a24f9` |
| SL-021 | J/K | Advanced examples used unchecked or mislabeled validation paths: four topology optimizers could apply an unevaluated line-search step, grounded postprocessing repeated one solve per basis vector, several studies overstated relaxed-density or discretized-reference results, and artifact failures hid the effective path | `examples/12_plate_rcs_stl_roundtrip.jl`, `14_periodic_to_validation.jl` through `23_circular_plate_ptd.jl`, `examples/grounded_rcs/`, aircraft/PO validators and their guides | negative | R3 | INV-033, INV-034, INV-040, INV-041, INV-048, INV-049 | complete for executable paths: projected accepted-step line searches, one-pass grounded linear maps, bounded defaults, finite/objective/power/gradient gates, honest result labels, and actionable artifact errors | `b318e51` |
| SL-022 | G/H | Aqua's 10-second persistent-task default classified a clean bounds-checked package precompile as a task leak | `test/runtests.jl` | negative | R2 | INV-001, INV-048 | complete: a 300-second diagnostic returned `persistent=false` after 16.733 seconds; the test now retains Aqua's check with a 60-second window and passes in one- and four-thread suites | `b318e51` |
| SL-023 | H/J | A single fresh-resolution radiation-vector allocation sample included one-time JIT work at a new Julia call site | `test/runtests.jl` | negative | R2 | INV-041, INV-048 | complete: the helper takes the minimum of three warm samples, retains the original ceiling, and passes the isolated and bounds-checked suites | `89a24f9` |
| SL-024 | K | The public `planewave_dda_3d` docstring was attached to the intervening stale private helper instead of the exported function | `src/mom3d/DDA3D.jl`, `test/runtests.jl` | negative | R2 | INV-032, INV-045, INV-049 | complete: removing the stale helper restored the public binding; `Docs.hasdoc` now guards it | `dea2628` |
| SL-025 | K | The CAD-conversion availability probe described every Gmsh execution failure as "not found" | `src/geometry/MeshIO.jl`, `test/runtests.jl` | negative | R2 | INV-031, INV-036, INV-049 | complete: the diagnostic preserves the executable path and underlying failure, distinguishes a nonzero version probe, and gives a recovery action | `551d73b` |
| SL-026 | C/E | The private-symbol census found five additional definition-only symbols | `src/mom3d/DDA3D.jl`, `src/mom2d/Mie2D.jl`, `src/mom2d/Assembly2D.jl` | 20 | R2 | INV-045, INV-047 | complete: removed after the user approved the exact audited set | `09f6b7e` |
| SL-027 | C/E | Removing the approved allocating VIE wrapper exposed its private in-place solver and six support definitions as one unreachable cluster | `src/mom2d/Assembly2D.jl` | 50–65 | R2 | INV-045, INV-047 | audited; deletion awaits explicit approval | |
| SL-028 | C/E/J | Five unexported MLFMA interpolation and Legendre routines are outside the production and test reachability graph | `src/fast/MLFMA.jl` | 80–100 | R2 | INV-018, INV-041, INV-045, INV-047 | audited; deletion awaits explicit approval | |
| SL-029 | C/E | One unused grounded-design `Problem` workflow remains in an included helper file; the two helpers consumed by active examples are separate | `examples/grounded_rcs/framework_pixel_design.jl` | 60–75 | R2 | INV-033, INV-045, INV-047 | audited; example compatibility decision awaits explicit approval | |
| SL-030 | C | Two figure constants have no use in their validation script | `validation/meep/meep_validation_figure.jl` | 2 | R1 | INV-034, INV-045 | audited; deletion awaits explicit approval | |
| SL-031 | C/E | Four definitions in the repository-local slopfix tooling have no static or reflective consumer | `scripts/slopfix_lib/{scope,langs,quality,manifest}.py` | 20–30 | R2 | INV-041, INV-045, INV-047 | audited; deletion awaits explicit approval | |

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

## Approved stale-code removal

The user approved this exact set on 2026-08-25. Commit `dea2628` removes the
15 private definitions after the audit found no static caller, export,
documentation reference, reflective lookup, or repository extension contract:

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

The same commit removes the unused `Tuple` import from
`validation/bempp/sweep_impedance_conventions.py`. Removing
`_planewave_phase_bigfloat_dda_3d` also places the adjacent public docstring on
`planewave_dda_3d`, where it belongs; the package test graph now checks that
binding directly.

## Approved definition-only removal

The user approved the following exact set on 2026-08-25; commit `09f6b7e`
removes it:

- `_alpha_adjoint_apply` (two methods)
- `_mie2d_besselj_values_miller_float`
- `_solve_vie_factored_2d`
- `_I3_DDA`
- `_MIE2D_FALLBACK_PRECISION`

The two allocating wrappers had active in-place counterparts, and the DDA
adjoint path used its checked scalar/tensor helpers. The focused 2D and 3D MoM
suites and the full one-thread bounds-checked package suite passed after the
deletion.

## Audited reachability-closure set awaiting approval

The next static reachability closure contains these exact definitions:

- SL-027 — `_VIE_RHS_SCALE_LOWER_2D`, `_VIE_RHS_SCALE_UPPER_2D`,
  `_VIE_SOLVE_MIN_FALLBACK_PRECISION_2D`,
  `_VIE_SOLVE_FALLBACK_GUARD_BITS_2D`,
  `_vie_solve_fallback_precision_2d`, `_solve_vie_high_precision_2d`, and
  `_solve_vie_factored_2d!`.
- SL-028 — `_build_spectral_phi`, `_build_spectral_theta`, `legendre_all`,
  `associated_legendre_m_all`, and `build_interp_matrices`.
- SL-029 — `Problem`, `make_problem`, `objgrad(::Problem, ...)`,
  `evaluate(::Problem, ...)`, and `optimize(::Problem, ...)`.
- SL-030 — `DASHES` and `IEEE_SINGLE_COL_H` in
  `validation/meep/meep_validation_figure.jl`.
- SL-031 — `COUNTED`, `NON_SOURCE_LANGUAGES`, `_file_output_evidence`, and
  `source_language_share`.

Repository-wide token searches found no references outside each listed
cluster. None of the Julia package symbols is exported or documented as public,
and the reflective-use sweep found no lookup by name. `solve_vie_2d` uses
`_factor_vie_system_2d` with `_solve_factored_linear_system`; MLFMA builds the
active per-mode filters with `_build_disagg_filters_all_m`; the grounded
examples use `svec_fast` and `conic_filter_matrix`, which remain. The three
unprefixed MLFMA functions and the included example workflow carry a higher
compatibility risk than private definition-only leaves, so this exact set is
not removed without explicit approval.

## Running totals

| | Lines |
| --- | ---: |
| Baseline code lines | 74,755 |
| Raw census estimate (unverified) | 7,668 |
| Promised net reduction | 150 |
| Gross removed so far | 1,174 |
| Gross added so far | 8,320 |
| Net removed so far | -7,145 |
| Remaining to target | 7,295 |
