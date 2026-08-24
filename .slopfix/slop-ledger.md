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
| SL-006 | C | `Tuple` is imported but has no static reference | `validation/bempp/sweep_impedance_conventions.py:14` | 1 | R1 | INV-035 | open; deletion approval required | |
| SL-007 | G | Scanner flags abstract incident-field fallbacks as placeholders | `src/assembly/Excitation.jl:2484`, `:2561` | 0 | R1 | INV-007, INV-014 | false positive: intentional explicit unsupported-type errors | |
| SL-008 | G | Scanner flags numeric conversion probe as a swallowed error | `src/postprocessing/Diagnostics.jl:32` | 0 | R1 | INV-013, INV-040 | false positive: conservative precision fallback | |
| SL-009 | H | Aqua is not reachable from the normal test graph; package tests include source directly instead of loading the installed package | `test/runtests.jl:5`, `:16`; `.slopfix/quality.json` | negative | R2 | INV-001, INV-045, INV-048 | open; test-dependency/API decision required | |
| SL-010 | H/J | Allocation assertions exist but there is no consolidated benchmark report or named representative hot-path latency baseline | test allocation helpers and assertions | negative | R2 | INV-041 | open | |
| SL-011 | K | Documenter disables public-doc coverage checking with `checkdocs=:none` | `docs/make.jl:13` | 0 | R2 | INV-032, INV-045, INV-049 | open; measure actual undocumented surface before deciding | |
| SL-012 | D | `test/runtests.jl` is a 14,319-line sequential test driver; source god files include Mesh/MLFMA/Excitation/DDA | `test/runtests.jl`, largest source files in `.slopfix/triage.md` | 0 | R3 | all | deferred unless a behaviour-preserving split is separately approved | |
| SL-013 | C/E | Internal definitions with one textual occurrence may be dead or public/dynamic extension points | whole `src/` public/internal symbol census | unknown | R3 | INV-045, INV-047 | open; callers/exports/docs/history/dynamic dispatch audit required | |

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

## Running totals

| | Lines |
| --- | ---: |
| Baseline code lines | 74,755 |
| Raw census estimate (unverified) | 7,738 |
| Promised net reduction | 150 |
| Removed so far | 0 |
| Remaining to target | 150 |

