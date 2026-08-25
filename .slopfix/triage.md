# Slopfix triage — DiffMoM.jl

Derived from commit `2c31d91064b07758eac5debdf9e710934e96bc17` on
2026-08-24, before production edits.

## Verdict

**Go for a conservative, verification-led cleanup.** DiffMoM has a coherent
scientific behaviour surface, a runnable Julia package, a 52-section regression
suite, documentation and examples, and multi-platform CI. The codebase is large
mainly because it implements many distinct numerical methods, so the census's raw
clone estimate is not a safe deletion estimate. Consolidation is limited to
groups whose behavioural union can be characterised and approved.

## Frozen baseline

- Counter: `slopfix-builtin/2+julia/1.12.7`
- Definition: non-blank, non-comment source lines; Julia files classified by the
  Julia parser, other source languages by the bundled scanner
- Scope: 129 source files; tests included; prose/config/data excluded from the
  reduction number but included in the deep-debug audit
- Baseline: 74,755 code lines
- Target: 74,605 code lines, a conservative 150-line (0.2%) net reduction
- Contract: `.slopfix/baseline.json`
- Known counter warning: `validation/po/compare_po_aircraft.m:118` uses MATLAB's
  transpose/string syntax that the fallback Objective-C/MATLAB scanner cannot
  disambiguate; the warning is retained in the baseline rather than hidden.

## Target derivation

The initial census estimated 7,738 removable clone lines across at least 400
groups, but that is a lower-bound clone count and an upper-bound reduction
opportunity. The 150-line commitment is based only on three bounded candidates:

- seven repeated exceptional scaled-output reducers in matrix-free operators;
- repeated example bootstrap blocks;
- two repeated Gmsh-path helpers and one statically unused Python type import.

The target discounts these candidates for new characterisation tests, a shared
implementation, caller adapters, numerical-risk review, and the possibility that
the user rejects a consolidation after seeing its behavioural diff.

## Census summary

- 400 clone groups collected before the configured cap
- 108 files involved
- 7,738 removable lines estimated by token clones (not yet verified as redundant)
- Largest bounded core candidate: seven exceptional complex scaled-output
  reducers in `CompositeOperator.jl`, `ACA.jl`, `MLFMA.jl`, `DDA3D.jl`,
  `SurfaceIE3D.jl`, `QMatrix.jl`, and another operator implementation
- Other candidates: mirrored MLFMA sections, FFT DDA electric/magnetic sections,
  repeated example setup, and validation-script utilities

## Smell triage

The bundled scanner reported three blocking patterns. Manual inspection classified
all three as scanner false positives, not confirmed defects:

- `src/assembly/Excitation.jl:2484` and `:2561` are deliberate abstract fallback
  methods that reject unsupported pointwise incident-field models with explicit
  errors; concrete supported excitation types have methods.
- `src/postprocessing/Diagnostics.jl:32` is a numeric capability probe: failure
  to convert an unfamiliar component intentionally returns `nothing`, causing the
  caller to select a conservative 65,536-bit fallback. No result or operational
  error is discarded.

The advisory scan additionally found one statically unused `Tuple` import in
`validation/bempp/sweep_impedance_conventions.py:14`; deletion remains pending
human approval under the unused-is-not-unwanted rule.

## Verification capability

- Julia regression suite: `test/runtests.jl` plus eight included focused files
- Scientific references: Mie, finite differences, reciprocity, power balance,
  dense/matrix-free equivalence, and external-validation scripts
- Allocation/resource assertions: embedded throughout the Julia tests
- Platform/concurrency matrix: Ubuntu (1 and 4 threads), macOS, and Windows in
  `.github/workflows/ci.yml`
- Documentation: Documenter build in `docs/make.jl`
- Gaps: optional external Bempp/Meep workflows require separate Python packages;
  no committed coverage percentage, SBOM/advisory scan, pinned JET run, or
  serialized-data compatibility fixture yet

## Explicit exclusions

- No feature, public export, example, validation workflow, or error contract is
  removed merely because repository-local static search finds no caller.
- Generated data/figures, dependency trees, and build output are not reduction
  targets.
- Domain algorithms are not merged on token similarity alone.
- Documentation and tests may grow; deleting them is forbidden.
