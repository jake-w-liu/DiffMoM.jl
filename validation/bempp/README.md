# Bempp-cl cross-validation

These scripts compare DiffMoM Julia reference fields with independent
`bempp-cl` calculations. Run them from the repository root. Generated files are
written under `data/`.

## Setup

Use a separate Python environment:

```bash
python -m pip install -r validation/bempp/requirements.txt
```

The requirements file declares `bempp-cl` and `numpy`. Gmsh must be available
when using `--mesh-mode gmsh_screen`.

## PEC workflow

```bash
julia --project=. validation/paper/run_beam_steering_case.jl
python validation/bempp/run_pec_cross_validation.py
python validation/bempp/compare_pec_to_julia.py
```

The comparison writes `data/bempp_cross_validation_report.json` and
`data/bempp_cross_validation_report.md`. It reports differences but does not
apply an acceptance gate.

## One impedance-loaded case

Use the same settings and prefix for the Julia and Bempp runs:

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

This writes `data/bempp_z100_cross_validation_report.{json,md}`. The comparator
extracts main-beam and sidelobe features from the sampled cut nearest phi zero.

## Validation matrix

```bash
python validation/bempp/run_impedance_validation_matrix.py
```

The driver runs the cases declared in its `CASES` constant. Use `--help` to
inspect its mesh and sampling defaults. A structured run is:

```bash
python validation/bempp/run_impedance_validation_matrix.py \
  --mesh-mode structured \
  --nx 12 \
  --ny 12
```

Every case must meet the declared main-beam angle, main-beam level, and
sidelobe-suppression gates. `MAX_MAIN_THETA_DIFF_DEG`,
`MAX_MAIN_LEVEL_DIFF_DB`, and `MAX_SLL_DIFF_DB` in
`run_impedance_validation_matrix.py` own the executable limits and the generated
report labels.

The process exits with status 2 when any gate fails. Aggregate outputs are:

- `data/impedance_validation_matrix_summary.csv`
- `data/impedance_validation_matrix_summary.json`
- `data/impedance_validation_matrix_summary.md`

Preview the commands without executing them:

```bash
python validation/bempp/run_impedance_validation_matrix.py --dry-run
```

The matrix also accepts `--skip-julia`, `--skip-bempp`, and `--skip-compare` to
reuse existing artifacts. Missing comparison JSON files still cause failure.

## Convention diagnostics

Use `--help` to inspect the available convention profiles and the selected
default. The effective operator sign, RHS cross-product order and sign, phase
sign, impedance scale, and mesh settings are stored in the matrix summary JSON.

```bash
python validation/bempp/sweep_impedance_conventions.py --run-julia
```

The sweep writes:

- `data/impedance_convention_sweep.csv`
- `data/impedance_convention_sweep.json`
- `data/impedance_convention_sweep.md`

Record the full convention configuration with any result. A better numerical
match from a sweep is not, by itself, evidence that the selected weak-form
convention is correct.

## Current, phase, and residual checks

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

The benchmark combines far-field metrics with element-center current, phase,
coherence, and forward-residual reports. The standalone current comparator is:

```bash
python validation/bempp/compare_impedance_operator_aligned.py \
  --output-prefix opalign_z100 \
  --mag-floor-db -20
```

Optional `--max-vector-rms-rel` and `--min-coherence` values turn those current
metrics into executable gates.

## Diagnostic plots

```bash
python validation/bempp/plot_impedance_comparison.py \
  --julia-prefix z100 \
  --bempp-prefix z100 \
  --output-prefix z100_diag
```

The plotter writes a PNG and a text summary under `data/`.

## Matching rules

- Keep frequency, plate size, incidence, polarization, angular grid, and sheet
  impedance identical between solvers.
- Use the same output prefix unless deliberately comparing different artifacts.
- Sample keys are matched after rounding `(theta_deg, phi_deg)` to `1e-6`
  degrees.
- Preserve the generated metadata and convention configuration with the
  comparison report.
- Inspect linear-scale values when dB differences are concentrated near nulls.
