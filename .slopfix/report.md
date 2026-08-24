# Slopfix report — DiffMoM.jl

> Status: implementation audit complete at `62f59e6`; removal of the audited
> private dead-code set still requires the user's explicit approval. Amend the
> final commit and measurements if that set is approved.

| | |
| --- | --- |
| Period | `2026-08-24` → `2026-08-24` |
| Baseline commit | `2c31d91064b07758eac5debdf9e710934e96bc17` |
| Audited source commit | `62f59e6` |
| Counter | `slopfix-builtin/2+julia/1.12.7` |
| Definition | non-blank, non-comment source lines |
| Scope | frozen in `.slopfix/baseline.json` |

## Result

| | Lines |
| --- | ---: |
| Baseline code | 74,755 |
| Audited code | 81,101 |
| Gross removed | 239 |
| Gross added | 6,586 |
| Gross method | `slopfix-builtin/2-line-fingerprint-diff` |
| **Net removed** | **-6,346** |
| **Reduction** | **-8.49%** |

| | |
| --- | ---: |
| Promised reduction | 0.2% (150 lines) |
| Delivered | -6,346 lines removed (a net addition of 6,346) |
| **Attainment** | **0% of the committed goal** |

Reproduce the current number:

```bash
git checkout 62f59e6
python3 scripts/slopfix.py measure --strict
```

The reduction target was not met. The original clone census included distinct
scientific algorithms whose ordering, conjugation, precision, or external
workflow contracts were not interchangeable. Correctness fixes,
characterisation tests, public-documentation coverage, and the executable
quality tooling added more source than the audit safely removed. Fifteen private
helpers and one Python import passed the static dead-code audit, but the skill's
unused-is-not-unwanted rule prohibits deleting them without explicit approval.
No tests or documentation were removed, and no source was parked or compressed
to manufacture a reduction.

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
| INV-033 | Complete example matrix | Examples 01, 04, 08, and 13 passed; the other examples include larger or artifact-dependent workflows and were parsed but not executed | Run every example in clean bounded environments and retain each asserted output |
| INV-034 | Complete internal-validation matrix | Bounded Mie, beam-steering, convergence, and robustness validators passed; artifact-dependent paper/PO/grounded/scaling workflows were not all executed | Provision their declared artifacts and run the complete validation matrix |
| INV-035 | Bempp-cl and Meep comparisons | Their separate Python solver environments were not provisioned | Install the pinned external requirements and run each comparison gate |
| INV-036 | Successful CAD conversion | Missing-path, unsupported-format, executable-presence, and mesh-I/O paths passed; the repository has no small CAD fixture | Add a licensed test CAD fixture, convert it with Gmsh, and import the result |
| INV-044 | Current remote CI result | The platform matrix is configured, but commit `62f59e6` has not been observed in GitHub Actions | Push the commit and retain green Ubuntu, macOS, Windows, one-thread, four-thread, and docs jobs |
| INV-047 | Serialized-struct compatibility | There is no committed serialized fixture or schema/migration policy | Add versioned fixtures and round-trip/migration tests for supported public structs |

Final-gate checks:

| Check | Result |
| --- | --- |
| One-thread bounds-checked project suite | PASS; all 52 sections (52 at baseline) |
| Four-thread bounds-checked project suite | PASS; all 52 sections |
| Isolated package test gate | PASS; exit 0 |
| Documentation and doctests | PASS; export coverage and cross-references checked |
| Julia parser sweep | PASS; 115 files |
| Python bytecode compilation | PASS for `validation/` and `scripts/` |
| Aqua | PASS through the installed-package test graph |
| JET | PASS; zero reports on the selected concrete entrypoints |
| Source coverage | 21,079 / 23,031 executable lines (91.52%) |
| Blocking smells | 0 |
| Duplication ratchet | 7,749 estimated removable lines, ceiling 7,900 |
| Line ratchet | 81,101 code lines, ceiling 82,648 |
| Secret scan | Gitleaks 8.30.1: 442-commit history and audited source files passed |
| `slopfix measure --strict` | Exit 0; no integrity findings; one retained MATLAB scanner warning |

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
| Config SHA-256 | `a08d537c2fd8890130372aa7aeaec5914c7fe0a7ab0cca8ba02c9d1ac660bf16` |
| Run scope | full; all 16 gates selected |
| Totals | 13 PASS, 0 FAIL, 2 UNVERIFIED, 1 NOT_APPLICABLE |
| Strict verdict | PASS; no required gate failed or remained unverified |

| Characteristic | PASS | FAIL | UNVERIFIED | NOT_APPLICABLE | Evidence or limit |
| --- | ---: | ---: | ---: | ---: | --- |
| Functional suitability | 3 | 0 | 0 | 0 | tests, coverage, numerical contracts |
| Performance efficiency | 1 | 0 | 0 | 0 | allocation and warm latency baseline |
| Compatibility | 1 | 0 | 0 | 1 | CI configuration checked; no committed library manifest |
| Interaction capability | 1 | 0 | 0 | 0 | docs/doctest command |
| Reliability | 1 | 0 | 0 | 0 | one/four-thread and resource-lifecycle evidence |
| Security | 2 | 0 | 1 | 0 | clean resolution and secrets pass; SBOM/advisories unverified |
| Maintainability | 2 | 0 | 1 | 0 | Aqua/JET pass; ExplicitImports triage incomplete |
| Flexibility | 1 | 0 | 0 | 0 | exports, dispatch, and public docs |
| Safety | 1 | 0 | 0 | 0 | finite, bounds, residual, power, and gradient gates |

Optional unverified quality gates:

- `julia-sbom-advisories`: the library commits no manifest and the environment
  has no Julia-aware advisory scanner, so no SBOM/advisory result exists.
- `julia-explicit-imports`: ExplicitImports v1.15.0 found no stale explicit
  imports but reported 59 potential implicit imports and 11 non-public qualified
  accesses. The potential set includes analyzer false positives for local
  bindings such as `path` and `volume` and still needs per-symbol triage.

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

No confirmed bug was deliberately preserved.

## What was done

| Ledger | Change | Commit | Inventory verified |
| --- | --- | --- | --- |
| SL-009 | Installed-package tests now reach Aqua | `62f59e6` | INV-001, 045, 048 |
| SL-010, SL-014 | Added dense-Q performance evidence and local cancellation fallback | `62f59e6` | INV-013, 040–042 |
| SL-011 | Enabled Documenter export checking and rendered all 301 exports | `62f59e6` | INV-032, 045, 049 |
| SL-015 | Added configurable high-level true-residual verification | `62f59e6` | INV-011, 019, 040 |
| SL-016 | Made the touched examples and validators fail closed after named gates | `62f59e6` | INV-031, 033, 034, 040, 048 |
| SL-017 | Reworked README/docs and user-facing diagnostics under the UX-writing contract | `62f59e6` | INV-031, 032, 049 |
| SL-018 | Moved test scratch files to a test-owned temporary directory | `62f59e6` | INV-003, 048 |
| SL-019 | Added line, duplication, smell, secret, and ISO/IEC 25010 quality ratchets | `62f59e6` | INV-041–046, 048 |

No production module was wholesale rewritten.

## What was not done

| Ledger | Candidate | Why not | What it needs |
| --- | --- | --- | --- |
| SL-001 | Seven exceptional scaled-output reducers | Precision, trigger, and error contracts differ by subsystem | Complete the site-by-site behavioural diff, add shared-oracle tests, and obtain approval |
| SL-002 | Mirrored MLFMA sections | Forward/adjoint ordering and conjugation make token similarity unsafe | Extract and prove a shared invariant before editing |
| SL-003 | Electric and coupled FFT-DDA setup | Tensor/block semantics differ | Differential dense/operator tests for every material mode, then approval |
| SL-004 | Example bootstrap blocks | Invocation and environment behaviour are user-visible | Define one supported example launcher contract and replay the example matrix |
| SL-005 | Bempp Gmsh-path helpers | External environments were unavailable | Provision Bempp/Gmsh and characterize both drivers first |
| SL-006, SL-013 | Fifteen private helpers and one Python import | Static audit found no users, but deletion requires explicit human approval | Approve or reject the exact set in `.slopfix/slop-ledger.md` |
| SL-012 | Large test/source files | A split is high-risk churn without a module-level characterization spec | Separate approved restructuring task |
| SL-020 | Potential implicit imports | Analyzer output includes local-binding false positives | Per-symbol triage before any namespace rewrite |

## Integrity findings

| Finding | Detail | Disposition |
| --- | --- | --- |
| MATLAB scanner warning | `validation/po/compare_po_aircraft.m:118` uses MATLAB's non-conjugating transpose `Ri.'`; the fallback scanner treats the apostrophe as a string delimiter | Retained and reported; Julia/parser-backed counts and the frozen baseline are unchanged |
| Generated-doc secret false positives | A worktree-wide scan matched `API:...` HTML anchor IDs in ignored `docs/build/` files | Confirmed generated anchors, not credentials; the 442-commit history and audited source files passed |
| Parked/compressed/deleted verification code | None reported by `measure --strict` | No action required |

## Remaining known problems

- The exact dead-code set is still present pending explicit approval.
- Remote platform results, external Bempp/Meep comparisons, successful CAD
  conversion, complete example/validation matrices, and serialization
  compatibility remain unverified as itemized above.
- ExplicitImports' 59 potential implicit imports need per-symbol triage.
- The docs build passes but warns that two API pages exceed 100 KiB and the
  generated search index is about 1.4 MiB.
- `test/runtests.jl` remains a 14,415-line sequential driver. Large source files
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

The line ceiling is 1,547 lines above the audited count. Raising either ceiling
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
