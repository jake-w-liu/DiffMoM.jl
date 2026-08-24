# DiffMoM.jl - instructions for coding agents

## Search before writing

This repository has been through a duplication and stale-code audit. Before
adding a helper, validator, numerical fallback, operator wrapper, or workflow,
search for the concept and for nearby definitions:

```bash
rg -n -i '<concept>' src test docs examples validation
rg -n 'function .*<concept>|<concept>.*\(.*\)\s*=' src
rg --files src test docs examples validation
```

Extend the canonical implementation when one exists. Do not create a second
implementation next to the caller that happens to need it.

## Canonical modules

| Concept | Canonical location | Notes |
| --- | --- | --- |
| Shared types and resource-limit helpers | `src/Types.jl` | Keep common validation and byte-count logic here. |
| Mesh construction, repair, quality, and core I/O | `src/geometry/Mesh.jl` | Format dispatch and CAD conversion live in `src/geometry/MeshIO.jl`. |
| RWG bases and triangle quadrature | `src/basis/RWG.jl`, `src/basis/Quadrature.jl` | Preserve orientation and allocation contracts. |
| Green functions and periodic Green functions | `src/basis/Greens.jl`, `src/basis/PeriodicGreens.jl` | Reuse checked exceptional-precision paths. |
| Incident fields and excitation assembly | `src/assembly/Excitation.jl` | Source conventions and phase signs are centralized here. |
| PEC EFIE and composite operators | `src/assembly/EFIE.jl`, `src/assembly/CompositeOperator.jl` | BLAS `mul!` semantics include aliasing and `alpha`/`beta` cases. |
| Impedance and density assembly | `src/assembly/Impedance.jl`, `src/assembly/DensityInterpolation.jl` | Keep patch and interpolation conventions aligned with adjoints. |
| Direct and iterative solves | `src/solver/Solve.jl`, `src/solver/IterativeSolve.jl` | Returned iterative results require checked convergence and true residuals. |
| Preconditioners | `src/solver/NearFieldPreconditioner.jl` | Preserve forward/adjoint and left/right semantics. |
| Far fields, power, and RCS | `src/postprocessing/FarField.jl`, `src/postprocessing/Diagnostics.jl` | Validate dimensions, finite values, weights, and physical invariants. |
| Far-field objective matrices | `src/optimization/QMatrix.jl` | Keep dense, matrix-free, and one-shot paths numerically consistent. |
| Adjoint gradients and optimizers | `src/optimization/` | Check gradients against an independent finite-difference oracle. |
| ACA and MLFMA acceleration | `src/fast/` | Forward and adjoint orderings are not interchangeable clones. |
| 2D and 3D material solvers | `src/mom2d/`, `src/mom3d/` | Preserve units, tensor layout, and exceptional-value behavior. |
| High-level solver selection | `src/Workflow.jl` | Keep method choice, memory estimates, and result metadata together. |
| Public documentation | `docs/src/`, `docs/make.jl` | User-facing claims must match an implementation, test, or measured run. |
| Scientific validation | `validation/` | A validator must fail with a nonzero exit when a stated gate fails. |

Add a row when a new shared concept is introduced.

## Use established dependencies

| Instead of hand-rolling | Use |
| --- | --- |
| Dense/sparse linear algebra or BLAS contracts | `LinearAlgebra`, `SparseArrays` |
| Fixed-size 2D/3D vectors | `StaticArrays` and the package aliases |
| Krylov solvers | `Krylov` through `src/solver/IterativeSolve.jl` |
| Incomplete LU | `IncompleteLU` through the preconditioner layer |
| FFT kernels and plans | `FFTW` through the existing DDA implementation |
| Special functions | `SpecialFunctions` |
| Documentation generation | `Documenter` through `docs/make.jl` |

Discuss a new dependency or a new numerical framework before adding it.

## Correctness and error handling

- Validate by running the relevant code. Do not report a value, cause, or
  guarantee from memory.
- Keep scientific conventions explicit: units, time convention, phase sign,
  polarization, normal orientation, and solver tolerance.
- Catch only exceptions that the code can handle. Preserve contextual error
  messages and rethrow unexpected exceptions.
- Do not discard errors, return fake results, or add `TODO`, `FIXME`, stubs, or
  commented-out fallbacks.
- Keep ordinary hot paths allocation-conscious. Exceptional BigFloat paths must
  be bounded and must not replace an entire ordinary workload because one local
  reduction is ill-conditioned.
- Resource limits cover every simultaneous operation-owned raw workspace, not
  only the returned array.
- Preserve exact BLAS `mul!(y, A, x, alpha, beta)` behavior, alias safety,
  Hermitian/adjoint identities, and thread-safe workspace locks.

## Repository layout

| Directory | Contains |
| --- | --- |
| `src/` | Package implementation, grouped by numerical subsystem. |
| `test/` | Regression, numerical-boundary, allocation, and concurrency tests. |
| `docs/` | Documenter project and user documentation. |
| `examples/` | Runnable workflows; defaults must be practical and fail closed. |
| `validation/` | Internal and optional external scientific comparisons. |
| `data/` | Generated example and validation outputs, never committed fixtures or test scratch space. |
| `.slopfix/` | Frozen behavior, quality, measurement, and audit evidence. |
| `scripts/` | Repository-wide quality tooling only. |

Tests create temporary artifacts under `mktempdir()` rather than the repository.

## Verification and definition of done

- Add tests for changed behavior and realistic boundaries, including invalid
  dimensions, non-finite values, cancellation, overflow/underflow, and resource
  limits when applicable.
- Run the smallest focused reproducer first, then the full bounds-checked suite
  with one and four Julia threads.
- Build documentation with
  `julia --project=docs --startup-file=no docs/make.jl` after public API or docs
  changes.
- Run `git diff --check`; parse every changed Julia file; keep examples and
  validators fail-closed.
- Do not weaken a tolerance or allocation ceiling merely to make a regression
  pass. Record the measurement that justifies a changed threshold.
- Run `python3 scripts/slopfix.py smells --severity blocking --strict` and the
  reviewed quality contract before handoff.
- Missing external tools or environments remain `UNVERIFIED`, never pass.

## Line-count ratchet

CI reads the deliberate ceiling from `.slopfix/line-ceiling.txt`:

```bash
python3 scripts/slopfix.py measure --strict \
  --ceiling "$(tr -d '[:space:]' < .slopfix/line-ceiling.txt)"
```

Raise the ceiling only for reviewed functionality, in its own commit, with a
reason. Do not evade it by relocating source, collapsing lines, or deleting
tests or documentation.

## Behavior inventory

`.slopfix/behaviour-inventory.md` is the regression contract. Update it when a
supported behavior changes, and record deliberate behavior changes in its
approved-changes table.

## Process ownership

Never terminate or signal a process unless the current session started that
exact process and retained its session or verified PID. Treat pre-existing and
resumed-session processes as user-owned; report them instead of stopping them.
