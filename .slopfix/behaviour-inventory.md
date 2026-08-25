# Behaviour inventory — DiffMoM.jl

Frozen from the codebase at commit
`2c31d91064b07758eac5debdf9e710934e96bc17` on 2026-08-24, before production
edits. Final evidence and every unverified boundary are recorded in
`.slopfix/report.md` and `.slopfix/quality-report.json`.

**This document is the contract.** Rows may be added when a missed behaviour is
found, or removed only as an explicitly dated and approved product decision. It
must never be rewritten to match a later implementation.

## Verification methods

| Code | Method |
| --- | --- |
| `T` | Automated test with an outcome assertion |
| `C` | Reproducible command with an expected result |
| `M` | Documented manual check |
| `R` | Code reading only; unverified at the final gate |

Criticality is `C1` for scientific/data-integrity behaviour, `C2` for important
user-visible tooling and compatibility, and `C3` for minor conveniences.

## Package and scientific behaviours

| ID | Behaviour and boundaries | Implemented in | Verify | Crit | Baseline status |
| --- | --- | --- | --- | --- | --- |
| INV-001 | An environment allowed by the `julia` compatibility entry resolves, precompiles, and loads `DiffMoM` without a persistent task | `Project.toml`, `src/DiffMoM.jl` | `C` clean-resolution and Aqua gates | C1 | pending |
| INV-002 | `TriMesh` and geometry factories construct finite, consistently indexed meshes; invalid/empty/resource-excess inputs fail explicitly | `src/Types.jl`, `src/geometry/Mesh.jl` | `T` Tests 1, 1c–1e, 43 | C1 | pending |
| INV-003 | OBJ/STL/MSH reading and writing preserve supported geometry and reject malformed, oversized, or unsupported input | `src/geometry/Mesh.jl`, `src/geometry/MeshIO.jl` | `T` Tests 1b, 32 | C1 | pending |
| INV-004 | Mesh repair, clustering, coarsening, refinement, edge extraction, quality, and resolution checks preserve valid topology and enforce work limits | `src/geometry/Mesh.jl` | `T` Tests 1c–1e, 32 | C1 | pending |
| INV-005 | RWG/periodic-RWG construction, evaluation, divergence, support lookup, and triangle quadrature agree with mesh topology and remain allocation-bounded on hot paths | `src/basis/RWG.jl`, `src/basis/Quadrature.jl` | `T` Tests 1, 3, 37–42 | C1 | pending |
| INV-006 | Free-space Green kernels, gradients, smooth splits, and analytical self/adjacent singular integrals remain finite, symmetric where required, and match numerical/scaling references | `src/basis/Greens.jl`, `src/assembly/SingularIntegrals.jl` | `T` Tests 2, 44, 45 | C1 | pending |
| INV-007 | Plane-wave, port, delta-gap, electric/magnetic dipole, loop, monopole, pattern, imported, and multi-source assembly preserve source conventions, phase, linearity, and explicit unsupported-field errors | `src/assembly/Excitation.jl` | `T` Tests 4, 14–16 | C1 | pending |
| INV-008 | Dense PEC EFIE assembly handles self, adjacent, and regular cells; is reciprocal/symmetric within stated tolerances; and enforces cache/work limits | `src/assembly/EFIE.jl`, `src/assembly/SingularIntegrals.jl` | `T` Tests 3, 9, 45 | C1 | pending |
| INV-009 | Surface-impedance loading, patch mass matrices, and derivatives match finite differences for resistive and reactive parameters | `src/assembly/Impedance.jl` | `T` Tests 5, 7, 18 | C1 | pending |
| INV-010 | Matrix-free EFIE and impedance-composite operators implement dense-equivalent forward, adjoint, indexed, and BLAS-scaled products, including cancellation-safe exceptional values | `src/assembly/EFIE.jl`, `src/assembly/CompositeOperator.jl` | `T` Tests 17, 34 | C1 | pending |
| INV-011 | Direct and GMRES forward/adjoint solves validate dimensions/tolerances and achieve checked true residuals for supported operators and preconditioning sides | `src/solver/Solve.jl`, `src/solver/IterativeSolve.jl` | `T` Tests 4, 12, 17–23 | C1 | pending |
| INV-012 | Diagonal, sparse near-field, block, ILU/LU, permuted, left/right, forward/adjoint preconditioners are equivalent to reference solves and reuse workspaces safely | `src/solver/NearFieldPreconditioner.jl` | `T` Tests 12, 19, 20, 23, 30, 31 | C1 | pending |
| INV-013 | Spherical grids, radiation vectors, far fields, polarization masks, Q operators, projected/radiated power, and RCS preserve transversality, Hermitian/PSD identities, scaling, and cancellation | `src/postprocessing/FarField.jl`, `src/postprocessing/Diagnostics.jl`, `src/optimization/QMatrix.jl` | `T` Tests 6, 13–16 | C1 | pending |
| INV-014 | Scattered and total near fields preserve linearity, far-zone agreement, source addition, singularity handling, surface rejection, and resource limits | `src/postprocessing/NearField.jl` | `T` Tests 6b, 6c | C1 | pending |
| INV-015 | Adjoint objectives and impedance/density/material gradients agree with central finite differences and complex-step checks within recorded tolerances | `src/optimization/Adjoint.jl`, `src/optimization/Verification.jl`, `src/mom3d/Adjoint3D.jl` | `T` Tests 7, 8, 18, 40 and 3D adjoint testset | C1 | pending |
| INV-016 | L-BFGS, directivity, and multi-angle optimization honour bounds, line-search semantics, solver choice, filter chains, and reduce checked objectives on smoke problems | `src/optimization/Optimize.jl`, `src/optimization/MultiAngleRCS.jl` | `T` Tests 10, 21–23, 35, 36 | C1 | pending |
| INV-017 | Cluster trees and ACA low-rank operators preserve ordering, admissibility, dense/reference products, indexed access, adjoints, solves, and storage limits | `src/fast/ClusterTree.jl`, `src/fast/ACA.jl` | `T` Tests 24–27 | C1 | pending |
| INV-018 | Octree/MLFMA construction, near/far separation, translation, forward/adjoint products, GMRES, and optimizer integration meet recorded error/allocation bounds | `src/fast/Octree.jl`, `src/fast/MLFMA.jl` | `T` Tests 31, 36 | C1 | pending |
| INV-019 | `solve_scattering` selects or honours dense/GMRES/ACA/MLFMA methods, validates mesh resolution and excitation, and returns a consistent `ScatteringResult` | `src/Workflow.jl` | `T` Test 28 | C1 | pending |
| INV-020 | Physical-optics illumination/current/far-field and PTD edge diffraction preserve translation, polarization, phase, reduction, and analytical/reference RCS behaviour | `src/postprocessing/PhysicalOptics.jl`, `src/postprocessing/PTD.jl` | `T` Test 29 and PO/PTD coverage inside `runtests.jl` | C1 | pending |
| INV-021 | PEC and dielectric spherical Mie amplitudes/RCS remain stable across small/large size, weak contrast, exceptional truncation, direction, and material inputs | `src/postprocessing/Mie.jl` | `T` Tests 2b, 13 | C1 | pending |
| INV-022 | Periodic Green/Ewald sums, periodic EFIE, Floquet modes, reflection/transmission coefficients, and power balance enforce lattice/Wood-anomaly contracts and trusted identities | `src/basis/PeriodicGreens.jl`, `src/assembly/PeriodicEFIE.jl`, `src/postprocessing/PeriodicMetrics.jl` | `T` Tests 37, 41, 42 | C1 | pending |
| INV-023 | Density interpolation, filtering/projection, penalties, transpose chains, and density adjoints preserve mass, topology, symmetry, and FD gradients | `src/assembly/DensityInterpolation.jl`, `src/optimization/DensityFiltering.jl`, `src/optimization/DensityAdjoint.jl` | `T` Tests 38–40 | C1 | pending |
| INV-024 | Grounded-image reflection coefficients, excitation, and EFIE assembly follow supported PEC/dielectric image-theory conventions and limits | `src/assembly/GroundedEFIE.jl` | `T` grounded coverage in `runtests.jl` | C1 | pending |
| INV-025 | 2D TM grids, Green/self kernels, plane/line sources, dense solve, fields/Jacobians, and cylindrical Mie references preserve checked analytic and differential identities | `src/mom2d/` | `T` `2D TM MoM` testset | C1 | pending |
| INV-026 | Isotropic, anisotropic, dispersive, magnetic, and bianisotropic 3D material models validate passivity/shape/frequency contracts and produce checked tensors | `src/mom3d/MaterialModels3D.jl` | `T` `3D material model helpers` | C1 | pending |
| INV-027 | Electric DDA polarizability, dense/matrix-free assembly, solve, incident/induced/scattered/far fields, adjoints, and mixed-scale fallbacks agree with references | `src/mom3d/DDA3D.jl`, `src/mom3d/Adjoint3D.jl` | `T` Tests 46 and 3D adjoint testset | C1 | pending |
| INV-028 | Coupled electric-magnetic and FFT DDA operators preserve dense equivalence, forward/adjoint identities, shared-operator concurrency, fields, and allocation/resource limits | `src/mom3d/EMDDA3D.jl`, `src/mom3d/FFTDDA3D.jl` | `T` Tests 47, 48 | C1 | pending |
| INV-029 | Dense/matrix-free PMCHWT and Müller dielectric SIE assembly/solve preserve closed-surface validation, block conventions, adjoints, and sphere-current agreement | `src/mom3d/SurfaceIE3D.jl` | `T` Test 49 and PMCHWT/Müller testset | C1 | pending |
| INV-030 | Mesh wireframe/comparison plotting and PNG/PDF export produce nonempty artifacts with finite layout limits and reject invalid meshes/work limits | `src/postprocessing/Visualization.jl` | `C` visualization smoke command below | C2 | pending |

## Documentation, examples, validation, and integration behaviours

| ID | Behaviour and boundaries | Implemented in | Verify | Crit | Baseline status |
| --- | --- | --- | --- | --- | --- |
| INV-031 | README installation, test, package-load, and quick-start commands work from a fresh checkout as documented | `README.md`, `Project.toml` | `C` README command replay | C2 | pending |
| INV-032 | The complete Documenter navigation builds with doctests/cross-references and covers the public subsystem/API guide | `docs/make.jl`, `docs/src/` | `C` docs gate | C2 | pending |
| INV-033 | Runnable examples preserve their stated setup and output contracts without silently requiring undeclared packages | `examples/` | `C` example smoke matrix | C2 | pending |
| INV-034 | Internal Mie, PO, grounded, consistency, scaling, robustness, and paper validation scripts either reproduce recorded gates or fail with explicit missing-artifact guidance | `validation/` | `C` internal-validation matrix | C1 | pending |
| INV-035 | Optional Bempp-cl and Meep cross-validations reproduce external-solver comparisons when their separately documented Python dependencies and tools are installed | `validation/bempp/`, `validation/meep/` | `R` external environment unavailable until provisioned | C1 | unverified |
| INV-036 | CAD conversion validates paths/formats/Gmsh availability, reports nonzero conversion exits, and imports the produced supported mesh format | `src/geometry/MeshIO.jl` | `C` Gmsh/error-path smoke | C2 | pending |

## Non-functional and domain contracts

| ID | Characteristic | Contract and threshold | Verify | Crit | Baseline status |
| --- | --- | --- | --- | --- | --- |
| INV-040 | Numerical safety | NaN/Inf, empty/degenerate, subnormal/huge, cancellation, overflow, precision, units, tolerances, and seeded-random paths either match BigFloat/trusted references or fail explicitly | `T` full bounds-checked numerical suite | C1 | pending |
| INV-041 | Allocation efficiency | Existing hot-path `@allocated` ceilings and zero-allocation contracts remain at or below their test thresholds after warm-up | `T` allocation assertions in test suite | C1 | pending |
| INV-042 | Memory/resource bounds | User-sized mesh, cache, adjacency, quadrature, matrix, FFT, MLFMA, DDA, and validation work is preflighted against explicit count/byte limits before large allocation | `T` resource-limit assertions | C1 | pending |
| INV-043 | Concurrency/reliability | Shared FFT/operator/cache/preconditioner paths are race-free under four Julia threads; locks release through errors; BigFloat precision remains task-local | `T` four-thread suite and runtime contract | C1 | pending |
| INV-044 | Platform compatibility | The Julia versions, operating systems, and thread counts configured in CI complete their test jobs; the documentation job completes on its configured platform | `C` GitHub Actions matrix | C2 | pending |
| INV-045 | Public API/dispatch | Every exported name is defined; supported adjoint/index/mul! extensions have no ambiguities or piracy; documented aliases remain callable | `C` Aqua/export/API smoke gates | C1 | pending |
| INV-046 | Dependency/security boundary | A clean isolated environment resolves declared dependencies and precompiles; no credentials are committed; dependency provenance/licenses/advisories are recorded to the available tool limit | `C` clean-resolution, secret, SBOM/advisory gates | C1 | pending |
| INV-047 | Serialization compatibility | Public result/operator structs used with Julia `Serialization` retain field/ordering compatibility or publish an explicit migration/breaking policy | `R` no committed compatibility fixture | C2 | unverified |
| INV-048 | Resource lifetime | Package load leaves no persistent tasks; file I/O uses scoped handles; subprocess failures are surfaced; no current-session process is terminated without ownership | `C` Aqua plus resource-path audit | C1 | pending |
| INV-049 | Documentation/API consistency | README/docs commands, signatures, units, conventions, solver choices, limitations, and generated navigation agree with current exports and implementations | `C` docs build plus source/docs diff audit | C2 | pending |

## Reproducible command checks

### INV-001 — clean resolution and load

Use an isolated temporary Julia environment/depot, `Pkg.develop(path=repo)`,
`Pkg.instantiate()`, `Pkg.precompile()`, then `using DiffMoM`. Expected: exit 0,
no mutation of repository project files, and no persistent package task.

### INV-030 — visualization export

Create a 1×1 rectangular plate inside `mktempdir()`, call
`save_mesh_preview`, and assert that both returned PNG/PDF paths exist and have
nonzero byte sizes. Expected: exit 0 and no repository artifact.

### INV-031 — README replay

Run the exact documented instantiate/load/test commands from the package root in
a clean environment. Expected: load succeeds and the suite prints
`ALL 52 TESTS PASSED`.

### INV-032 — documentation

`julia --project=docs --startup-file=no docs/make.jl`. Expected: exit 0; warnings
must be inspected rather than silently classified as pass.

### INV-033/034 — examples and internal validation

Parse every script first. Execute bounded examples/validations that do not require
large external artifacts, recording each skipped script and its concrete external
requirement. Expected outputs are the script's own assertions/gates.

### INV-036 — Gmsh conversion

Exercise missing CAD path, unsupported extensions, missing executable, and (when a
small CAD fixture exists) successful conversion. Expected: explicit contextual
errors for invalid paths; successful output file and mesh import otherwise.

### INV-044 — platform matrix

Record the latest CI run for the audited commit. The required jobs and effective
runtime versions are owned by `.github/workflows/ci.yml`.

### INV-045/046/048 — package-quality gates

Run Aqua, ambiguity/export/API smoke, isolated resolution, available secret and
advisory scans, and the resource-lifetime audit. Any unavailable scanner remains
`UNVERIFIED` rather than passing by omission.

## Coverage summary (pre-execution classification)

| | Count |
| --- | ---: |
| Total behaviours | 46 |
| Intended automated-test verification (`T`) | 32 |
| Intended reproducible-command verification (`C`) | 12 |
| Intended manual verification (`M`) | 0 |
| Unverified/code-reading-only (`R`) | 2 |

Unverified at baseline:

- INV-035: optional Bempp-cl/Meep environment is not provisioned yet.
- INV-047: no serialized-data compatibility fixture or versioned schema policy.

## Approval questions

Before any production consolidation:

1. Is a required package behaviour, validation workflow, or supported downstream
   use missing from this inventory?
2. Is anything listed here intentionally dead and approved for removal?
3. Which rows are critical enough that their current tests are insufficient?
4. Is any current behaviour known to be wrong but required to remain compatible?

## Approved behaviour changes

| Date | Inventory | Change | Approval basis |
| --- | --- | --- | --- |
| 2026-08-24 | INV-013, INV-040–042 | Dense far-field Q construction uses BLAS for ordinary entries and local checked recomputation for cancellation-sensitive components; exceptional results retain the BigFloat reference behavior. | User requested a fresh correctness, memory, and optimization pass. |
| 2026-08-24 | INV-019, INV-040 | Iterative `solve_scattering` paths verify the returned vector's true residual by default; callers retrieving a partial iterate must disable that gate explicitly. | User requested confirmed bugs to be fixed rather than preserved. |
| 2026-08-24 | INV-031, INV-033, INV-034, INV-049 | Touched examples and validators use practical bounded setups, report effective values, apply named finite acceptance gates, and exit nonzero before reporting completion when a gate fails. | User requested examples, validation, README, and docs to be audited and completed. |
| 2026-08-24 | INV-003, INV-048 | Regression mesh artifacts are written to a test-owned temporary directory; the Mie validator no longer overwrites its repository STL fixture. | User requested stale artifacts and resource behavior to be cleaned up. |
| 2026-08-24 | INV-007, INV-019, INV-031, INV-032, INV-049 | Changed errors, status output, help, README, and guides identify effective behavior and recovery actions; mutable defaults point to source-owned docstrings or constants. | User explicitly required the `ux-writing` skill throughout and a full copy pass. |
| 2026-08-25 | INV-033, INV-034, INV-040, INV-041, INV-048, INV-049 | Advanced periodic, topology-optimization, grounded, aircraft, and PO workflows use accepted projected steps, bounded defaults, objective cross-checks, named gates, effective paths, and explicit artifact recovery. | User requested a fresh deep-debug pass and explicitly required `ux-writing` throughout. |
| 2026-08-25 | INV-001, INV-048 | Aqua's persistent-task test keeps the subprocess check but allows 60 seconds for a clean bounds-checked precompile to finish. | A 300-second diagnostic completed without a persistent task after 16.733 seconds, exceeding Aqua's 10-second default. |
| 2026-08-25 | INV-032, INV-034, INV-035, INV-040, INV-048, INV-049 | Bempp and Meep report readers reject malformed, empty, duplicate, overflowed, and non-finite data; generated JSON uses finite numbers or `null`; matrix gates stop when required metrics are unavailable. | User requested validation correctness and a full `ux-writing` pass. |
| 2026-08-25 | INV-034, INV-035, INV-040, INV-041, INV-048, INV-049 | Workflow-local helpers own shared Bempp/Meep CLI, artifact-reading, grid, and subprocess contracts; all numeric options fail during parsing when outside their declared domain; optional solver and plotting imports do not block `--help`. | User approved the exact consolidation, input-guard, and stale-value removal set. |
| 2026-08-25 | INV-032, INV-040, INV-048, INV-049 | Optimizer docs match the absolute gradient stopping rule, trace fields, and algebraically equivalent ratio-gradient paths; guidance names measurements instead of unsupported method, mesh, or conditioning preferences; the multi-angle stagnation diagnostic states the effective criterion and recovery action. | User explicitly required `ux-writing` throughout and requested a final full copy pass. |
| 2026-08-25 | INV-018, INV-041, INV-048 | The ACA build keeps its threaded rank limit immutable and computes multiplication-workspace rank in a separate post-build binding; operator behavior is unchanged. | The final JET and four-thread review found the captured keyword rebinding, and the user requested a complete optimization and reliability pass. |
