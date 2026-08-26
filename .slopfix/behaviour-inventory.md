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
| 2026-08-26 | INV-026, INV-048 | Dispersive material objects retain their constructor's passivity policy, so `passive=false` remains effective when the stored model is evaluated. | A fresh runtime reproducer showed every dispersive model re-enabled passivity during evaluation; the user requested confirmed bugs to be fixed and reverified. |
| 2026-08-26 | INV-026, INV-049 | Magnetodielectric permeability evaluation accepts the documented Drude, Lorentz, and Debye response objects and delegates them through the stored `mu_model`. | A fresh runtime reproducer showed all three documented dispersive permeability paths raised `MethodError`; the user requested confirmed bugs to be fixed and reverified. |
| 2026-08-26 | INV-007, INV-019, INV-049 | `solve_scattering` rejects every frequency-bearing excitation whose stored spectral model conflicts with the operator, including nested multi-excitations and custom-`c0` misuse of vacuum analytic sources. | Fresh runtime reproducers returned finite currents for five inconsistent source paths; the user requested confirmed bugs to be fixed and reverified. |
| 2026-08-26 | INV-019, INV-040 | True-residual gates recompute cancellation-sensitive stored dense and sparse products before deciding whether an iterative result is acceptable. | Fresh dense and sparse reproducers reported zero residual for solutions whose exact relative residual was 0.25; the user requested confirmed bugs to be fixed and reverified. |
| 2026-08-26 | INV-019, INV-048 | Low-level forward and adjoint GMRES reject a misleading preconditioned-convergence status unless the returned vector also passes the unpreconditioned true-residual gate; both checks require explicit opt-out for partial iterates. | A public diagonal-preconditioner reproducer returned `stats.solved=true` with a true relative residual of 0.707; the user requested confirmed bugs to be fixed and reverified. |
| 2026-08-26 | INV-011, INV-048 | Forward and adjoint GMRES use restarted Arnoldi so `memory` is an enforced basis bound and `max_workspace_bytes` covers every retained Krylov vector. | Krylov's unrestarted default grew past the accepted one-vector estimate through twelve iterations; the user requested confirmed resource bugs to be fixed and reverified. |
| 2026-08-26 | INV-019, INV-049 | Every iterative `ScatteringResult.gmres_residual` records the unpreconditioned true relative residual against the selected dense, ACA, or MLFMA operator; only direct results use `NaN`. | Krylov history was disabled, so all successful iterative workflows previously stored `NaN`; the user requested confirmed metadata and copy bugs to be fixed and reverified. |
| 2026-08-26 | INV-011, INV-040 | Conditioned-system regularization uses checked scaled accumulation so finite stored-input cancellation is retained. | A one-entry reproducer rounded the representable exact regularized value `4.930380657631324e-32` to zero; the user requested confirmed numerical bugs to be fixed and reverified. |
| 2026-08-26 | INV-012, INV-048 | Impedance-loaded MLFMA near-field preconditioners build a bounded union sparsity pattern and evaluate each entry with bounded local exact fallback when ordinary accumulation is ambiguous. | A sparse one-entry reproducer dropped the exact base value after opposing extreme mass terms cancelled; the user requested confirmed numerical and resource bugs to be fixed and reverified. |
| 2026-08-26 | INV-010, INV-040, INV-048 | Matrix-free EFIE forward and adjoint operators implement alias-safe five-argument `mul!` with exact `alpha`/`beta` overwrite, scaling, and cancellation behavior. | The generic fallback corrupted aliased input and bypassed checked output reductions; the user requested confirmed BLAS-contract bugs to be fixed and reverified. |
| 2026-08-26 | INV-017, INV-049 | ACA indexed access represents the same compressed matrix as `mul!`, double adjoint returns the parent operator, and `efie_entry` is the explicit uncompressed EFIE query. | A normally compressed operator disagreed with its own basis-vector columns and double adjoint products; the user requested confirmed matrix-contract bugs to be fixed and reverified. |
| 2026-08-26 | INV-007, INV-040, INV-048 | Surface-excitation assembly uses checked component products and a reusable bounded quadrature-term reduction for every field-based source model. | Public imported-field reproducers lost representable cancellation both within one vector dot product and across quadrature terms; the user requested confirmed source-assembly bugs to be fixed and reverified. |
| 2026-08-26 | INV-023, INV-040 | Density-derivative scaling preserves representable complex products for dense, sparse, and compact local mass matrices. | A finite stored-input product near the ComplexF64 component limit threw after an overflowed rounded intermediate; the user requested confirmed numerical bugs to be fixed and reverified. |
| 2026-08-26 | INV-022, INV-040 | Periodic spatial-image sums omit images whose Float64 shift is infinite or whose Ewald kernel is exactly zero before evaluating an irrelevant Bloch phase; cached and direct paths agree. | A finite wide-period correction produced `NaN` through `0*Inf` phase arithmetic on a mathematically zero image term; the user requested confirmed numerical bugs to be fixed and reverified. |
| 2026-08-26 | INV-024, INV-049 | Grounded excitation accepts only the down-going plane wave whose magnitude and Bloch components match the periodic lattice. | Runtime reproducers showed that port, wrong-frequency, wrong-scan, and up-going sources were silently scaled by a lattice-derived ground phase; the user requested confirmed scientific-contract bugs to be fixed and reverified. |
| 2026-08-26 | INV-013, INV-040, INV-049 | Spherical grids require coherent canonical angles and directions; arbitrary-direction cone masks normalize both operands and clamp the computed cosine. | Runtime reproducers accepted a +z direction labelled as theta=π and selected a measurably off-axis, nearly unit direction for a zero-width cone; the user requested confirmed geometry bugs to be fixed and reverified. |
| 2026-08-26 | INV-017 | ACA indexed-access regression coverage compares against the represented compressed operator, while `efie_entry` remains the explicit uncompressed oracle. | The bounds-checked suite retained a stale exact-dense assertion after the indexed-access contract changed and failed on a valid batched dense-block rounding difference. |
| 2026-08-26 | INV-019 | Workflow residual metadata is checked against an operator rebuilt with the workflow's canonical frequency-to-wavenumber conversion. | The bounds-checked suite compared against `2π/(c0/frequency)`, whose one-ulp wavenumber difference changed an already roundoff-scale residual. |
| 2026-08-26 | INV-015, INV-016, INV-049 | Quadratic adjoint and optimizer objectives reject non-Hermitian Q matrices, including per-angle matrix-free configurations. | A finite-difference reproducer differed from the reported adjoint gradient by 171% because `Re(I'QI)` depends on the Hermitian part of Q while the adjoint used Q directly. |
| 2026-08-26 | INV-016, INV-049 | Bound-constrained optimizers stop on the box-projected KKT gradient and append a final evaluated state after the last accepted update so returned parameters, objectives, gradients, and solve counters agree. | A lower-bound optimum reported `gnorm=2` and stopped as a line-search failure; a one-step run returned `theta=0.5` while its final trace still described `theta=0` and omitted two forward/one adjoint solves. |
| 2026-08-26 | INV-026, INV-049 | Dispersive permeability evaluation reports passivity failures against `mu_r`; permittivity evaluation retains the `eps_r` label. | A passive Lorentz object whose evaluated response became active reported an `eps_r` failure when used as `MagneticMaterial3D.mu_model`. |
| 2026-08-26 | INV-034, INV-049 | The dielectric Mie validator parses its boolean environment override from an explicit true/false vocabulary and rejects misspellings. | `DDA_MIE_EFFECTIVE_A=treu` silently selected the false branch and ran the full validator instead of reporting the invalid configuration. |
| 2026-08-26 | INV-034, INV-048, INV-049 | PEC and dielectric Mie validators write generated CSV and figure artifacts under ignored `data/` storage, with an explicit output-directory override; generated dielectric CSV snapshots are no longer tracked as source. | The dielectric validator overwrote three committed CSV files under `validation/mie`, and the PEC validator wrote ignored outputs beside its source despite the repository layout contract. |
| 2026-08-26 | INV-035, INV-047, INV-049 | Meep validation binds every result and report to matching Julia geometry/reference identity fields and the SHA-256 digests of the exact source artifacts. | Synthetic mismatched geometry/reference files reached the optional solver, and a stale Meep result remained comparable after its Julia reference changed. |
| 2026-08-26 | INV-035, INV-048, INV-049 | Reflectance-curve reuse reruns the comparator for every existing case and exits nonzero on `CHECK`; subprocess failures surface without a Python traceback. | A synthetic reused case with `abs_diff_refl=0.8` and verdict `CHECK` wrote a new curve summary and returned success. |
| 2026-08-26 | INV-035, INV-047, INV-049 | Reflectance-curve reuse requires every Julia identity and Meep runtime control to match the current command, and summary geometry fields come from the validated artifacts. | A self-consistent prior 1x1 case was reused under requested 14x14 controls and the new summary falsely recorded 14x14. |
| 2026-08-26 | INV-035, INV-047, INV-049 | Meep curve case IDs preserve exact Float64 slot widths beyond the readable three-decimal prefix and reject duplicate widths before work. | Distinct widths `0.2001` and `0.2004` both mapped to `wx0p200`, causing artifact overwrite or wrong reuse. |
| 2026-08-26 | INV-035, INV-047, INV-049 | Detailed Meep heuristic analysis validates the same geometry/reference identity and exact source hashes while remaining able to diagnose current `CHECK` cases. | The analyzer paired a Julia reference with an arbitrary hashless Meep result and reported metrics without provenance validation. |
| 2026-08-26 | INV-025, INV-027, INV-028, INV-042, INV-049 | Direct 2D VIE, electric DDA, and coupled EM-DDA byte limits cover all simultaneously retained dense matrices, LU factors/pivots, result material arrays, and flat/structured field buffers. | Each solver accepted a ceiling sized for only its assembled matrix or matrices while returning a second dense LU copy and additional retained arrays beyond that stated limit. |
| 2026-08-26 | INV-025, INV-027, INV-028, INV-042 | Direct volume-solver limits dynamically preflight 4352-bit factor matrices, exact RHS/solution buffers, and coexisting IEEE retry plans before entering the bounded exact path. | An ill-conditioned two-cell VIE returned a 5,544-byte result under a claimed 288-byte combined ceiling because its `Complex{BigFloat}` LU was charged as `ComplexF64`. |
| 2026-08-26 | INV-011, INV-025, INV-027, INV-028 | Dense direct factorization rejects a nominally successful LU whose stored factors are non-finite and caches the finite equilibrated or exact replacement for reuse. | LAPACK reported success for a finite 2x2 matrix while retaining `Inf` factors; direct reuse returned `[0,0]` instead of `[-0.5,0.5]`. |
| 2026-08-26 | INV-011, INV-025, INV-027, INV-028, INV-029 | Direct volume and surface results store physical-matrix-aware factorization wrappers, so every later `A_LU`/`Z_LU` solve retains backward-error verification and exceptional retry without duplicating the matrix payload. | A finite successful LU for a 55x55 Wilkinson matrix had backward error 0.025 and 11% forward error when reused directly, despite the checked solve recovering machine-precision accuracy. |
| 2026-08-26 | INV-025, INV-042 | The 2D scattered-field Jacobian skips the rounded block solve for an exact cached VIE factor and preflights both 4352-bit rectangular solve arrays with its returned matrices. | A one-observation, two-cell exact-factor Jacobian accepted a 96-byte ceiling while its two BigFloat rectangular arrays alone required about 4,864 bytes. |
| 2026-08-26 | INV-017, INV-040 | ACA forward and adjoint products detect cancellation in ordinary dense blocks and both stages of low-rank products, retrying the represented compressed operator in bounded chunks. | A rank-three block indexed as `3` but returned `4` for the corresponding basis-vector product because only extreme exponents triggered exact multiplication. |
| 2026-08-26 | INV-013, INV-040 | Backscatter sample selection and angular error normalize every validated near-unit grid direction before comparing it with the requested direction. | Norm drift within the accepted grid tolerance made a sample one microradian farther away win the raw dot-product ranking and report zero angular error. |
| 2026-08-26 | INV-007, INV-013, INV-049 | Multi-excitation incident far fields validate the complete bounded excitation graph before recursive evaluation. | A cyclic graph overflowed the stack and nesting beyond the supported depth was evaluated despite the shared excitation validator rejecting both models. |
| 2026-08-26 | INV-022, INV-040 | Periodic power closure and residuals preserve representable differences when incident, reflected, absorbed, and transmitted powers cancel. | A finite public power-balance case returned `P_resid=-2` although exact reduction of its stored power components is `-2.9999999999999996`. |
| 2026-08-26 | INV-024, INV-040 | Grounded excitation and scalar/vector reflection paths evaluate the image-interference factor and its downstream products without losing representable phase cancellation. | At `k=1` and `h=2^-30`, ordinary subtraction returned a zero real factor although the exact stored-input factor has real part `1.734723475976807e-18`. |
| 2026-08-26 | INV-007, INV-040 | Surface-excitation cancellation retry reduces retained basis, field, area, and quadrature primitives instead of already-rounded per-sample terms. | A public three-sample imported field returned `3` from rounded terms while the 4352-bit stored-primitive oracle is `2.333333333333335`. |
| 2026-08-26 | INV-017, INV-048 | Ordinary ACA forward and adjoint products compute their conservative term-count bound with allocation-free saturating integer arithmetic. | The bounds-checked allocation gate measured 152 bytes in `mul!`; allocation profiling traced 96 bytes to three per-call `BigInt` temporaries in the term-count estimate. |
| 2026-08-26 | INV-019, INV-048 | The dense-direct workflow limit covers the simultaneous EFIE matrix, verified LU factors and pivots, RHS, and solution, with a larger conditional gate for exact fallback. | A public direct solve succeeded with a ceiling equal to one matrix although the LU copy and field vectors were live during the solve. |
| 2026-08-26 | INV-011, INV-048 | `prepare_conditioned_system` returns the original matrix and RHS without allocation when regularization and preconditioning are disabled. | The exported API documented a no-op identity, but a 1x1 probe returned distinct matrix and vector copies. |
| 2026-08-26 | INV-014, INV-040, INV-048 | Near-field preprocessing and vector/scalar potential reductions detect finite primitive cancellation and use the shared bounded 8704-bit point retry. | Public reproducers reduced three RWG current terms to `2` instead of `3.3628384223565857`, and three disconnected field contributions to a result that disagreed with the stored-geometry oracle. |
| 2026-08-26 | INV-022, INV-040 | Direct and cached periodic spatial sums retain integer image orders whose Cartesian shifts overflow Float64, evaluating their distance, phase, and representable kernel through a bounded exceptional path. | With `dx=dy=floatmax`, `E=5e-309`, and spatial order 2, both paths skipped nonzero image kernels and shifted the correction's real component by `1.2875353693968e-310`. |
| 2026-08-26 | INV-016, INV-018, INV-048 | Single-objective, directivity, and dense multi-angle optimizer limits cover simultaneous setup, accepted/trial factors and pivots, solution/objective buffers, parameter work vectors, and conditional exact solves. | All three public optimizers accepted a one-matrix ceiling while multiple matrices, factors, and vectors were simultaneously operation-owned; 1x1 L-BFGS completed under 16 bytes. |
| 2026-08-26 | INV-044, INV-048 | Required cross-platform test jobs have a 120-minute budget so the full bounds-checked Windows suite can finish before the quality gate. | Windows was cancelled at both the former 30-minute and 60-minute job timeouts while Ubuntu, macOS, documentation, and both local thread-count suites passed. |
| 2026-08-26 | INV-014, INV-040, INV-048 | Near-field evaluation routes extreme coupled `k`, impedance, and current scales through the bounded full-point primitive retry even when an early current/basis product rounds to zero. | With a minimum-subnormal current and `k=1e300`, the public field was zero while the stored-geometry oracle had representable O(1e-26) components; a one-unit exact-work limit was also bypassed. |
| 2026-08-26 | INV-007, INV-040 | Surface-excitation aggregation retries retained primitives when nonzero extreme inputs make every individual exact term round to zero. | Six individually zero terms assembled to zero although their joint 4352-bit primitive reduction rounds to the minimum ComplexF64 subnormal. |
| 2026-08-26 | INV-024, INV-040 | Grounded excitation and reflection multiply the unrounded BigFloat image-interference factor before converting the complete result to ComplexF64. | At minimum-subnormal `k`, `h=0.25`, and `E0=1e308`, the standalone factor rounded to zero while its three RHS products remained representable. |
| 2026-08-26 | INV-016, INV-018, INV-048 | Aggregate optimizer budgets include retained L-BFGS step/gradient history, the transient pre-eviction pair peak, and the two-loop coefficient vector in ordinary, GMRES, and exact paths. | With `P=1000`, `maxiter=2`, and `m_lbfgs=1`, all three estimators omitted 16,008 bytes of live history payload and accepted their one-history-pair runs below the actual lower bound. |
| 2026-08-26 | INV-007, INV-048 | Batch excitation work limits include both the rounded quadrature-term buffer and the simultaneously retained primitive-retry buffer at the largest effective quadrature order. | An order-3 one-RWG plane-wave batch accepted a 376-byte boundary while its missing primitive buffer alone owned another 720 raw bytes. |
| 2026-08-26 | INV-013, INV-044, INV-048 | The radiation-vector allocation ratchet includes the measured macOS ARM64 auxiliary payload while remaining far below the removed per-basis phase-matrix growth. | GitHub macOS 26 ARM64 measured 4,552,992 total bytes for a 2,506,752-byte output; the former fixed 512,000-byte allowance failed despite the bounded implementation and all numerical gates passing locally and on Ubuntu. |
| 2026-08-26 | INV-044, INV-048 | Cross-platform test jobs disable `julia-runtest`'s unused default coverage instrumentation; the separate reviewed coverage ratchet remains recorded by the quality contract. | Julia 1.12 Windows spent the full 58-minute test step in coverage-instrumented precompilation and never entered the suite, while no CI step consumed or uploaded its coverage artifact. |
| 2026-08-26 | INV-044, INV-048 | CI uses a fresh versioned Julia cache namespace and publishes caches only from successful jobs, so interrupted precompile state is not restored by later platform runs. | After coverage was disabled, Windows again remained in precompilation until the 120-minute timeout while restoring a cache saved unconditionally by the preceding cancelled job. |
| 2026-08-26 | INV-034, INV-044, INV-049 | Mie validator subprocess tests build `JULIA_LOAD_PATH` with the platform's list separator while retaining the active project and standard libraries. | A fresh Windows run entered the suite after its cache reset, then failed because the Unix `@:@stdlib` spelling made `LinearAlgebra` unavailable; Windows requires `@;@stdlib`. |
| 2026-08-26 | INV-044, INV-048 | The reviewed quality contract runs the same bounds-checked root project test command as the passing platform jobs; clean temporary resolution remains a separate required gate, and ignored docs manifests are not treated as protected source. | CI platform tests passed on Ubuntu, macOS, and Windows, but the redundant nested-environment test gate exited nonzero without retained output, while the docs gate flagged creation of the intentionally ignored `docs/Manifest.toml`. |
