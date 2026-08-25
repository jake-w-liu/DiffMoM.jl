# Slopfix report — DiffMoM.jl

> Status: the source and UX-writing audit has reached `49c78f5`. The approved
> SL-026 definitions were removed in `09f6b7e`; the exact SL-027–SL-031
> reachability-closure set awaits a separate deletion decision. Full final
> gates will be replayed after that decision.

| | |
| --- | --- |
| Period | `2026-08-24` → `2026-08-25` |
| Baseline commit | `2c31d91064b07758eac5debdf9e710934e96bc17` |
| Audited source commit | `49c78f5` |
| Counter | `slopfix-builtin/2+julia/1.12.7` |
| Definition | non-blank, non-comment source lines |
| Scope | frozen in `.slopfix/baseline.json` |

## Result

| | Lines |
| --- | ---: |
| Baseline code | 74,755 |
| Audited code | 81,900 |
| Gross removed | 1,174 |
| Gross added | 8,320 |
| Gross method | `slopfix-builtin/2-line-fingerprint-diff` |
| **Net change** | **+7,145** |
| **Reduction** | **-9.56%** |

| | |
| --- | ---: |
| Promised reduction | 0.2% (150 lines) |
| Shortfall from target | 7,295 lines |
| **Target attained** | **No** |

Reproduce the current number:

```bash
git checkout 49c78f5
python3 scripts/slopfix.py measure --strict
```

The reduction target was not met. The original clone census included distinct
scientific algorithms whose ordering, conjugation, precision, or external
workflow contracts were not interchangeable. Correctness fixes,
characterisation tests, public-documentation coverage, fail-closed example
gates, explicit-import regression checks, and the executable quality tooling
added more source than the audit safely removed. After explicit approval, the
15 private helpers and one unused Python import that passed the static
dead-code audit were removed in `dea2628`; the next approved five-symbol set
was removed in `09f6b7e`. No tests or documentation were removed, and no
source was parked or compressed to manufacture a reduction.

## Verification

| | Count |
| --- | ---: |
| Behaviours inventoried | 46 |
| Verified by automated test | 32 |
| Verified by reproducible command | 7 |
| Verified by documented manual check | 0 |
| **Unverified** | **7** |

Unverified items and the evidence needed to close them:

| ID | Behaviour | Why not verified | What would close it |
| --- | --- | --- | --- |
| INV-031 | Every README installation and quick-start route | Local instantiate/load, package tests, and Examples 01 and 08 passed; the remote `Pkg.add(url=...)` route was not replayed against the unpushed audited commit | Push the audited commit and install its Git URL in a clean depot |
| INV-033 | Complete example matrix | Examples 01, 04, 08, 12–23 and the grounded gradient/small-assembly workflows passed their available gates; aircraft and full 24/36-mesh grounded workflows require absent geometry or serialized artifacts | Provision the named artifacts and run every remaining example in a clean bounded environment |
| INV-034 | Complete internal-validation matrix | Bounded Mie, beam-steering, convergence, robustness, periodic, and grounded checks passed; external PO/MATLAB and artifact-dependent paper/scaling workflows were not all executable | Provision their declared artifacts and MATLAB/Octave, then run the remaining validation gates |
| INV-035 | Bempp-cl and Meep comparisons | Their separate Python solver environments were not provisioned | Install the pinned external requirements and run each comparison gate |
| INV-036 | Successful CAD conversion | Missing-path, unsupported-format, executable-presence, and mesh-I/O paths passed; the repository has no small CAD fixture | Add a licensed test CAD fixture, convert it with Gmsh, and import the result |
| INV-044 | Current remote CI result | The platform matrix is configured, but the audited source has not been observed in GitHub Actions | Push the final commit and retain green Ubuntu, macOS, Windows, one-thread, four-thread, and docs jobs |
| INV-047 | Serialized-struct compatibility | There is no committed serialized fixture or schema/migration policy | Add versioned fixtures and round-trip/migration tests for supported public structs |

Final-gate checks:

| Check | Result |
| --- | --- |
| One-thread bounds-checked project suite | PASS; all 52 sections (52 at baseline) |
| Four-thread bounds-checked project suite | PASS; all 52 sections |
| Isolated package test gate | PASS; exit 0 |
| Documentation and doctests | PASS; export coverage and cross-references checked |
| Modified Julia parser sweep | PASS; all 56 changed Julia files |
| Python bytecode compilation | PASS; all 13 changed Python files |
| Aqua | PASS through the installed-package test graph |
| JET | PASS; zero reports on the selected concrete entrypoints |
| Source coverage | 21,079 / 23,031 executable lines (91.52%) |
| Blocking smells | 0 |
| Duplication ratchet | 7,668 estimated removable lines, ceiling 7,900 |
| Line ratchet | 81,854 code lines, ceiling 82,648 |
| Secret scan | Gitleaks 8.30.1: 442-commit history and audited source files passed |
| `slopfix measure --strict` | Exit 0; no warnings or integrity findings |

The one-thread, four-thread, and strict quality runs in this table completed at
`e6a2afa`. The approved SL-026 deletion passed its focused suites and a full
one-thread bounds-checked suite. The validation-copy changes in `49c78f5`
passed their focused Python regression, parser checks, and documentation build.
The full matrix will be replayed after the SL-027–SL-031 deletion decision so
the final report describes one exact source state.

The bounds-checked commands were:

```bash
JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=. --startup-file=no -e \
  'using Pkg; Pkg.test(; julia_args=`--check-bounds=yes --startup-file=no`)'

JULIA_NUM_THREADS=4 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=. --startup-file=no -e \
  'using Pkg; Pkg.test(; julia_args=`--check-bounds=yes --startup-file=no`)'
```

Focused numerical evidence:

- The cancellation-sensitive 32-column dense-Q case allocated 20,144 bytes
  and had a 17.625 microsecond median over 100 warm runs on this machine. The
  previous all-BigFloat implementation measured 249,783,504 bytes and
  45.418 milliseconds on the same case. Wall time is machine-specific; the
  regression suite enforces allocation ceilings.
- A fresh-resolution replay of the radiation-vector allocation gate measured
  4,552,992 bytes on its first sample at a new Julia call site, then 2,619,680
  bytes on each of four repeats. The gate now takes the minimum of three warm
  samples and retains its original 3,018,752-byte ceiling.
- One hundred seeded dense-Q cases agreed with a 512-bit BigFloat reference to
  a maximum relative error of `4.73455569118545e-16`; every result was exactly
  Hermitian.
- The visualization smoke wrote a 75,665-byte PNG and a 142,736-byte PDF in a
  temporary directory.
- The bounded Mie validator had residual `2.05e-14`, E-plane mean/maximum error
  `0.050/0.095 dB`, H-plane `0.041/0.099 dB`, backscatter error `0.09 dB`, and
  energy ratio `1.0002`; every declared gate passed.
- The beam-steering gradient maximum error was `2.766e-6` against its `1e-5`
  gate. The convergence study's mesh/reference maxima were `2.81e-6` and
  `3.89e-7` against `3e-6`; every declared gate passed.
- Example 19's weakest-case reduction was `13.367 dB`; its weakest checkerboard
  advantage was `9.513 dB`; direct and GMRES currents differed by
  `7.603e-11` relative.
- Example 20 reached `|R00| = 0.08655` (`-21.25 dB`), closed its power
  diagnostic to `5.13%`, and exceeded the feasible checkerboard by `16.91 dB`.
- The grounded postprocessor replaced 588 unit-current passes with one linear
  map: the measured path changed from `0.925092 s` and `484,276,608` bytes to
  `0.113160 s` and `17,609,680` bytes, with maximum coefficient difference
  `2.22e-16`. Timing is machine-specific; the coefficient comparison is the
  correctness gate.
- Aqua's default 10-second persistent-task window failed under a fresh
  bounds-checked compile. The same check completed with `persistent=false`
  after `16.733 s`; a 60-second test window then passed in both one-thread
  (`13.6 s`) and four-thread (`15.0 s`) suites.
- Before `dea2628`, `Docs.hasdoc(DiffMoM, :planewave_dda_3d)` returned `false`
  while the adjacent stale private helper returned `true`. After the approved
  deletion, the public function returns `true`; `test/runtests.jl` now protects
  that binding.
- The final UX diagnostic replay exercised rejected radiative-correction DDA
  gradients, non-coplanar periodic meshes, and interior PTD wedges. Each error
  named the rejected state and a usable recovery path; the focused command
  returned `ux_diagnostics=PASS`. The documentation build then passed its
  doctest, export-coverage, and cross-reference checks.
- The Gmsh availability diagnostic replay used a missing executable path and
  returned `gmsh_diagnostic=PASS`. The error retained that exact path and the
  operating-system failure, then directed the caller to install Gmsh, update
  `PATH`, or pass a working `gmsh_exe`.

The test suite skipped the optional paper-consistency comparisons because their
generated artifacts were absent. Test 32 verified the Gmsh executable and error
paths but skipped successful CAD conversion because no CAD fixture exists.

## Quality assurance

Reproduce:

```bash
python3 scripts/slopfix.py quality-check --run --strict
```

The final matrix is recorded in `.slopfix/quality-report.json`.

| | |
| --- | --- |
| Quality model | `ISO/IEC 25010:2023` |
| Profile | `julia` |
| Config SHA-256 | `680eca22432a9339880ddf0d51b7e1aa273d19cbd4d34b74f47392599133ad6c` |
| Run scope | full; all 16 gates selected |
| Totals | 14 PASS, 0 FAIL, 1 UNVERIFIED, 1 NOT_APPLICABLE |
| Strict verdict | PASS; no required gate failed or remained unverified |

| Characteristic | PASS | FAIL | UNVERIFIED | NOT_APPLICABLE | Evidence or limit |
| --- | ---: | ---: | ---: | ---: | --- |
| Functional suitability | 3 | 0 | 0 | 0 | tests, coverage, numerical contracts |
| Performance efficiency | 1 | 0 | 0 | 0 | allocation and warm latency baseline |
| Compatibility | 1 | 0 | 0 | 1 | CI configuration checked; no committed library manifest |
| Interaction capability | 1 | 0 | 0 | 0 | docs/doctest command |
| Reliability | 1 | 0 | 0 | 0 | one/four-thread and resource-lifecycle evidence |
| Security | 2 | 0 | 1 | 0 | clean resolution and secrets pass; SBOM/advisories unverified |
| Maintainability | 3 | 0 | 0 | 0 | Aqua, JET, and ExplicitImports pass |
| Flexibility | 1 | 0 | 0 | 0 | exports, dispatch, and public docs |
| Safety | 1 | 0 | 0 | 0 | finite, bounds, residual, power, and gradient gates |

Optional unverified quality gate:

- `julia-sbom-advisories`: the library commits no manifest and the environment
  has no Julia-aware advisory scanner, so no SBOM/advisory result exists.

ExplicitImports v1.15 now runs in the normal test graph. Its seven checks report
no implicit or stale imports, no non-owning or non-public explicit imports, and
no self-qualified accesses. All qualified accesses use the owning module. The
exact non-public qualified allowlist remains a compatibility boundary and fails
if a new name appears.

The 11 reviewed non-public integrations are `Base.RefValue`, `Base.decompose`,
`Base.mightalias`, `Base.promote_op`, `Base.setindex`,
`Base.Checked.checked_add`, `Base.Checked.checked_mul`,
`LinearAlgebra.LAPACK.gecon!`, `SparseArrays.AbstractSparseMatrixCSC`,
`SparseArrays.UMFPACK`, and `IncompleteLU.ILUFactorization`. They are deliberate
implementation dependencies, but remain a compatibility boundary.

## Approved behaviour changes

| Date | Inventory | Change | Reason | Approved by |
| --- | --- | --- | --- | --- |
| 2026-08-24 | INV-013, 040–042 | Dense Q construction uses BLAS for ordinary entries and local compensated/BigFloat recomputation for cancellation-sensitive components | Preserve exceptional numerical behaviour without making one local cancellation force an all-BigFloat workload | User's deep-debug, correctness, memory, and optimization request |
| 2026-08-24 | INV-011, 019, 040 | Iterative `solve_scattering` paths check the returned vector's true residual by default | A solver status alone did not prove the selected operator equation was satisfied | User's request to fix confirmed bugs |
| 2026-08-24 | INV-031, 033, 034, 049 | Touched examples and validators use bounded setups, named finite gates, effective-value output, and nonzero failure exits | Prevent plausible-looking output from being reported as completed validation | User's examples, validation, README, and docs audit request |
| 2026-08-24 | INV-003, 048 | Tests and STL round-trips use temporary paths instead of repository artifacts | Prevent validation from overwriting tracked fixtures or leaving test scratch files | User's stale-artifact and resource-lifecycle request |
| 2026-08-24 | INV-007, 019, 031, 032, 049 | Errors, help, status output, README, and guides state the effective behaviour and next action; mutable defaults point to canonical source where practical | Full `ux-writing` pass requested by the user | User's explicit `$ux-writing` reminder |
| 2026-08-25 | INV-033, 034, 040, 041, 048, 049 | Advanced periodic, topology-optimization, grounded, aircraft, and PO workflows use bounded defaults, projected-step line searches, objective cross-checks, named gates, effective paths, and explicit artifact recovery | Prevent unchecked or mislabeled example output and avoid repeated grounded postprocessing solves | User's deep-debug and full `$ux-writing` pass request |
| 2026-08-25 | INV-001, 048 | Aqua's persistent-task gate uses a 60-second completion window while retaining the same subprocess-based task check | A measured 16.733-second clean bounds-checked compile exceeded Aqua's 10-second default without leaving a persistent task | User's request to verify and fix the full test graph |
| 2026-08-25 | INV-031, 032, 049 | Diagnostics state the rejected input, observed value, and recovery action; permanent docs state capabilities directly and limit performance guidance to measured comparisons | Keep user-facing copy useful after the implementation and runtime environment change | User's explicit full `$ux-writing` pass request |
| 2026-08-25 | INV-031, 036, 049 | CAD conversion distinguishes an unavailable executable from a nonzero Gmsh version probe and preserves the effective path and recovery action | Prevent an executable or permission failure from being mislabeled as a missing installation | User's explicit full `$ux-writing` pass request |

No confirmed bug was deliberately preserved.

## What was done

| Ledger | Change | Commit | Inventory verified |
| --- | --- | --- | --- |
| SL-009 | Installed-package tests now reach Aqua | `62f59e6` | INV-001, 045, 048 |
| SL-010, SL-014 | Added dense-Q performance evidence and local cancellation fallback | `62f59e6` | INV-013, 040–042 |
| SL-011 | Enabled Documenter export checking and rendered all 301 exports | `62f59e6` | INV-032, 045, 049 |
| SL-015 | Added configurable high-level true-residual verification | `62f59e6` | INV-011, 019, 040 |
| SL-016 | Made the touched examples and validators fail closed after named gates | `62f59e6` | INV-031, 033, 034, 040, 048 |
| SL-017 | Reworked README/docs and user-facing diagnostics, then swept temporal language, recovery guidance, and unsupported performance claims | `62f59e6`, `e6a2afa` | INV-031, 032, 049 |
| SL-018 | Moved test scratch files to a test-owned temporary directory | `62f59e6` | INV-003, 048 |
| SL-019 | Added line, duplication, smell, secret, and ISO/IEC 25010 quality ratchets | `62f59e6` | INV-041–046, 048 |
| SL-020 | Replaced broad dependency imports with the exact package namespace and added executable import/owner ratchets | `89a24f9` | INV-045, 048 |
| SL-021 | Hardened and executed the advanced periodic, optimization, grounded, and PO example gates; artifact-dependent scripts now fail with the effective path and recovery command | `b318e51` | INV-033, 034, 040, 041, 048, 049 |
| SL-022 | Replaced Aqua's too-short default completion window with a measured 60-second window | `b318e51` | INV-001, 048 |
| SL-023 | Excluded one-time JIT allocations from the steady-state radiation-vector allocation gate without raising its ceiling | `89a24f9` | INV-041, 048 |
| SL-006, SL-013 | Removed the explicitly approved set of 15 unreachable private helpers and one unused Python import | `dea2628` | INV-035, 045, 047 |
| SL-024 | Restored the exported `planewave_dda_3d` docstring binding and added a direct regression | `dea2628` | INV-032, 045, 049 |
| SL-025 | Made Gmsh availability failures specific, path-preserving, and actionable | `551d73b` | INV-031, 036, 049 |
| SL-026 | Removed the explicitly approved set of five additional definition-only symbols | `09f6b7e` | INV-045, 047 |

No production module was wholesale rewritten.

## What was not done

| Ledger | Candidate | Why not | What it needs |
| --- | --- | --- | --- |
| SL-001 | Seven exceptional scaled-output reducers | Precision, trigger, and error contracts differ by subsystem | Complete the site-by-site behavioural diff, add shared-oracle tests, and obtain approval |
| SL-002 | Mirrored MLFMA sections | Forward/adjoint ordering and conjugation make token similarity unsafe | Extract and prove a shared invariant before editing |
| SL-003 | Electric and coupled FFT-DDA setup | Tensor/block semantics differ | Differential dense/operator tests for every material mode, then approval |
| SL-004 | Example bootstrap blocks | Invocation and environment behaviour are user-visible | Define one supported example launcher contract and replay the example matrix |
| SL-005 | Bempp Gmsh-path helpers | External environments were unavailable | Provision Bempp/Gmsh and characterize both drivers first |
| SL-012 | Large test/source files | A split is high-risk churn without a module-level characterization spec | Separate approved restructuring task |
| SL-027–SL-031 | VIE, MLFMA, grounded-example, Meep-figure, and slopfix-tool reachability closures | The static audit found no users, but deletion requires an explicit compatibility decision | Approve or reject the exact set in `.slopfix/slop-ledger.md` |

## Integrity findings

| Finding | Detail | Disposition |
| --- | --- | --- |
| MATLAB scanner warning | The fallback scanner previously misread MATLAB's non-conjugating transpose apostrophe as a string delimiter | Replaced the expression with the equivalent `transpose(Ri)` form; the current strict measurement reports no warning |
| Generated-doc secret false positives | A worktree-wide scan matched `API:...` HTML anchor IDs in ignored `docs/build/` files | Confirmed generated anchors, not credentials; the 442-commit history and audited source files passed |
| Parked/compressed/deleted verification code | None reported by `measure --strict` | No action required |

## Remaining known problems

- Remote platform results, external Bempp/Meep/MATLAB comparisons, successful
  CAD conversion, artifact-dependent example/validation paths, and
  serialization compatibility remain unverified as itemized above.
- The SL-027–SL-031 reachability-closure definitions remain pending explicit
  deletion approval; the exact set and evidence are in the slop ledger.
- The docs build passes but warns that two API pages exceed 100 KiB and the
  generated search index is about 1.4 MiB.
- `test/runtests.jl` remains a 14,480-line sequential driver. Large source files
  include `src/geometry/Mesh.jl`, `src/fast/MLFMA.jl`, and
  `src/assembly/Excitation.jl`; splitting them without behavioural specs was
  deliberately deferred.
- The source coverage result is 91.52%, not complete coverage. External and
  artifact-dependent workflows are the clearest remaining evidence gap.

## Guardrails installed

| Artefact | What it does | What fails the build |
| --- | --- | --- |
| `AGENTS.md`, `CLAUDE.md` | Records canonical modules, search-first rules, scientific contracts, and definition of done | Advisory |
| `.gitleaks.toml` | Extends the default secret rules with one reviewed prose allowlist | A committed secret finding |
| `.slopfix/line-ceiling.txt` | Caps counted source at 82,648 lines | Any unexplained increase above the ceiling |
| `.slopfix/duplication-ceiling.txt` | Caps the census estimate at 7,900 lines | A higher clone estimate |
| `.slopfix/quality.json` | Replays the reviewed ISO/IEC 25010 contract | Any failed or required-unverified gate |
| `.github/workflows/ci.yml` | Runs the platform matrix, docs, ratchets, smells, and quality contract | Any configured job or gate failure |

The line ceiling is 794 lines above the audited count. Raising either ceiling
requires a deliberate reviewed commit. These checks expose re-accumulation; they
do not prove that every future change is non-duplicative.

## Artefacts

- `.slopfix/baseline.json` — frozen measurement contract
- `.slopfix/behaviour-inventory.md` — regression inventory
- `.slopfix/slop-ledger.md` — findings, decisions, and backlog
- `.slopfix/quality.json` — reviewed executable quality contract
- `.slopfix/quality-report.json` — last full quality run
- `.slopfix/report.md` — this report
- `AGENTS.md`, `CLAUDE.md`, `.gitleaks.toml`, and `.github/workflows/ci.yml`

No `.slopfix/specs/` directory is included because no production module was
wholesale rewritten.
