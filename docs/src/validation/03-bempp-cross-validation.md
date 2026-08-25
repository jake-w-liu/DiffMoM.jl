# Bempp cross-validation

The scripts in `validation/bempp/` compare Julia far-field results with
independent `bempp-cl` calculations. They write their artifacts under `data/`.
This is an optional validation environment; the repository does not include a
committed result table.

## Environment

Create a separate Python environment, then install the declared packages:

```bash
python -m pip install -r validation/bempp/requirements.txt
```

The requirements file declares `bempp-cl` and `numpy`. Gmsh must also be
available when a script uses `--mesh-mode gmsh_screen`. Run commands from the
project root so the Julia project and `data/` directory resolve consistently.

## PEC comparison

The PEC workflow uses the beam-steering driver's PEC field as the Julia
reference:

```bash
julia --project=. validation/paper/run_beam_steering_case.jl
python validation/bempp/run_pec_cross_validation.py
python validation/bempp/compare_pec_to_julia.py
```

The comparison joins samples on `(theta_deg, phi_deg)` after rounding each key
to `1e-6` degrees and uses the matched angular cut with the smallest periodic
distance to phi zero. Empty, duplicate, malformed, or non-finite samples stop
the comparison before a report is written. It reports global, near-target, and
sampled-cut differences in:

- `data/bempp_cross_validation_report.json`
- `data/bempp_cross_validation_report.md`

`compare_pec_to_julia.py` produces metrics but does not impose an acceptance
threshold. Apply a gate in the calling study and record its basis.

## One impedance-loaded case

Generate matched Julia and Bempp results with the same frequency, sheet
reactance, incidence, angular grid, and output prefix:

```bash
julia --project=. validation/bempp/run_impedance_case_julia_reference.jl \
  --freq-ghz 3.0 \
  --theta-ohm 100 \
  --theta-inc-deg 0 \
  --phi-inc-deg 0 \
  --n-theta 180 \
  --n-phi 72 \
  --output-prefix z100

python validation/bempp/run_impedance_cross_validation.py \
  --freq-ghz 3.0 \
  --zs-imag-ohm 100 \
  --theta-inc-deg 0 \
  --phi-inc-deg 0 \
  --n-theta 180 \
  --n-phi 72 \
  --mesh-mode structured \
  --nx 12 \
  --ny 12 \
  --output-prefix z100

python validation/bempp/compare_impedance_to_julia.py \
  --output-prefix z100 \
  --target-theta-deg 30
```

The comparator extracts main-beam and sidelobe features from the sampled cut
nearest phi zero. When a pattern has no eligible sidelobe outside the declared
main-beam exclusion window, the optional sidelobe fields are JSON `null` and
the Markdown report says that they are unavailable. It writes:

- `data/bempp_z100_cross_validation_report.json`
- `data/bempp_z100_cross_validation_report.md`

The single-case comparator reports metrics without applying matrix-level gates.
All generated JSON uses standard finite numbers; the scripts reject non-standard
`NaN` and infinity constants. The matrix requires finite main-beam and sidelobe
metrics, so an unavailable sidelobe stops the run with a regeneration or
feature-window action instead of being counted as a passing case.

## Validation matrix

Run the cases declared in the matrix driver's `CASES` constant:

```bash
python validation/bempp/run_impedance_validation_matrix.py
```

Use `--help` to inspect the matrix driver's mesh and sampling defaults. To use a
structured mesh explicitly:

```bash
python validation/bempp/run_impedance_validation_matrix.py \
  --mesh-mode structured \
  --nx 12 \
  --ny 12
```

The driver runs the Julia reference, Bempp solve, and comparison for each case.
It exits with status 2 unless every case meets its main-beam angle, main-beam
level, and sidelobe-suppression gates. `MAX_MAIN_THETA_DIFF_DEG`,
`MAX_MAIN_LEVEL_DIFF_DB`, and `MAX_SLL_DIFF_DB` in
`validation/bempp/run_impedance_validation_matrix.py` own the executable limits
and the generated report labels.

The aggregate outputs are:

- `data/impedance_validation_matrix_summary.csv`
- `data/impedance_validation_matrix_summary.json`
- `data/impedance_validation_matrix_summary.md`

The JSON records the effective convention profile and mesh configuration.
Preserve it with any reported result.

### Preview and reuse

Use `--dry-run` to print the subprocess commands without running them:

```bash
python validation/bempp/run_impedance_validation_matrix.py --dry-run
```

`--skip-julia`, `--skip-bempp`, and `--skip-compare` reuse existing artifacts.
The driver requires each expected comparison JSON before building its
summary, so stale or mismatched prefixes can invalidate a reused run.

## Convention checks

Use `--help` to inspect the available convention profiles and selected default.
The effective operator sign, RHS cross-product order and sign, far-field phase
sign, and surface-impedance scale are written to the summary JSON.

To inspect the alternate declared profile:

```bash
python validation/bempp/run_impedance_validation_matrix.py \
  --convention-profile case03_sweep_best \
  --dry-run
```

Individual convention fields can be overridden with
`--bempp-op-sign`, `--bempp-rhs-cross`, `--bempp-rhs-sign`,
`--bempp-phase-sign`, and `--bempp-zs-scale`. Treat a convention sweep as a
diagnostic experiment; record every override and validate it against the weak
forms before adopting it.

The dedicated sweep driver runs a bounded set of variants:

```bash
python validation/bempp/sweep_impedance_conventions.py --run-julia
```

## Current and residual diagnostics

Use the operator-aligned workflow when far-field differences need a lower-level
comparison:

```bash
python validation/bempp/run_impedance_operator_aligned_benchmark.py \
  --output-prefix opalign_z100 \
  --freq-ghz 3.0 \
  --zs-imag-ohm 100 \
  --theta-inc-deg 0 \
  --phi-inc-deg 0 \
  --mesh-mode structured \
  --nx 12 \
  --ny 12
```

It combines far-field features with element-center current, phase, coherence,
and forward-residual reports. The current comparator accepts optional
`--max-vector-rms-rel` and `--min-coherence` gates; without those options it
only reports the metrics. Missing residual metadata is recorded as JSON `null`;
malformed or non-finite metadata stops the comparison and identifies the file
that must be regenerated.

For existing far-field artifacts, generate diagnostic plots with:

```bash
python validation/bempp/plot_impedance_comparison.py \
  --julia-prefix z100 \
  --bempp-prefix z100 \
  --output-prefix z100_diag
```

## Reading a discrepancy

A failed comparison does not identify its cause. Check, in order:

1. frequency, geometry dimensions, mesh mode, and angular sample keys;
2. propagation, polarization, time, phase, RHS, and impedance conventions;
3. forward true residuals and finite current values;
4. main-beam and sidelobe feature windows; and
5. mesh and angular convergence in both implementations.

Deep nulls can amplify dB differences. Retain linear-scale comparisons when a
reported discrepancy is concentrated near a null.

## Source map

| Task | Script |
|---|---|
| PEC Bempp solve | `validation/bempp/run_pec_cross_validation.py` |
| PEC comparison | `validation/bempp/compare_pec_to_julia.py` |
| Julia impedance reference | `validation/bempp/run_impedance_case_julia_reference.jl` |
| Bempp impedance solve | `validation/bempp/run_impedance_cross_validation.py` |
| Impedance comparison | `validation/bempp/compare_impedance_to_julia.py` |
| Seven-case matrix and gates | `validation/bempp/run_impedance_validation_matrix.py` |
| Convention sweep | `validation/bempp/sweep_impedance_conventions.py` |
| Operator-aligned benchmark | `validation/bempp/run_impedance_operator_aligned_benchmark.py` |
| Diagnostic plots | `validation/bempp/plot_impedance_comparison.py` |
