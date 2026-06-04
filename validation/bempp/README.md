# Bempp-cl Cross-Validation Scripts

This folder provides independent `bempp-cl` cross-validation workflows for the
package beam-steering study, using Julia-generated reference artifacts in
`data/`.

## Scope

- Implemented here: PEC far-field cross-validation, single-case impedance
  cross-validation, impedance validation matrices, convention sweeps, and
  operator-aligned impedance benchmarks.

The goal is to add independent-solver checks for the public beam-steering
artifacts used by the paper consistency workflow.

## Prerequisites

- Python 3.10+
- `bempp-cl`
- `numpy`
- Gmsh available on the system path (used by BEM mesh generation)

Example install:

```bash
pip install -r validation/bempp/requirements.txt
```

## Run

From project root (`DiffMoM.jl/`):

1) Generate the Julia beam-steering reference artifacts used by the PEC and
impedance comparisons:

```bash
julia --project=. validation/paper/run_beam_steering_case.jl
```

2) Generate Bempp PEC reference:

```bash
python validation/bempp/run_pec_cross_validation.py
```

3) Compare Bempp vs Julia PEC far field:

```bash
python validation/bempp/compare_pec_to_julia.py
```

### Impedance-Loaded (Single Case)

Generate Julia and Bempp fields for a chosen impedance case:

```bash
julia --project=. validation/bempp/run_impedance_case_julia_reference.jl \
  --freq-ghz 3.0 --theta-ohm 200 --theta-inc-deg 0 --phi-inc-deg 0 \
  --output-prefix impedance

python validation/bempp/run_impedance_cross_validation.py \
  --freq-ghz 3.0 --zs-imag-ohm 200 --theta-inc-deg 0 --phi-inc-deg 0 \
  --output-prefix impedance

python validation/bempp/compare_impedance_to_julia.py \
  --output-prefix impedance --target-theta-deg 30
```

Use a Julia-matching structured plate mesh in Bempp (recommended for apples-to-apples checks):

```bash
python validation/bempp/run_impedance_cross_validation.py \
  --freq-ghz 3.0 --zs-imag-ohm 200 --theta-inc-deg 0 --phi-inc-deg 0 \
  --mesh-mode structured --nx 12 --ny 12 \
  --output-prefix impedance
```

### Impedance-Loaded Validation Matrix (Acceptance Guide)

Run the 6-case impedance matrix and produce aggregate summary artifacts:

```bash
python validation/bempp/run_impedance_validation_matrix.py
```

The matrix runner defaults to a refined Bempp screen mesh
`--mesh-step-lambda 0.2` for stronger impedance cross-validation.
The default convention profile is `paper_default` (the profile used for
manuscript-reported beam-centric matrix values).
Matrix acceptance is beam-centric (`|Δθ_main|`, `|ΔD_main|`, `|ΔSLL|`).

Matrix run with structured Bempp mesh:

```bash
python validation/bempp/run_impedance_validation_matrix.py \
  --mesh-mode structured --nx 12 --ny 12
```

Select a named convention profile:

```bash
python validation/bempp/run_impedance_validation_matrix.py \
  --convention-profile paper_default

python validation/bempp/run_impedance_validation_matrix.py \
  --convention-profile case03_sweep_best
```

Dry-run command preview:

```bash
python validation/bempp/run_impedance_validation_matrix.py --dry-run
```

Optional convention overrides (for reconciliation experiments):

```bash
python validation/bempp/run_impedance_validation_matrix.py \
  --bempp-op-sign minus \
  --bempp-rhs-cross e_cross_n \
  --bempp-rhs-sign 1.0 \
  --bempp-phase-sign plus \
  --bempp-zs-scale 1.0
```

### Impedance Convention Sweep

Sweep Bempp convention variants against a fixed Julia impedance reference:

```bash
python validation/bempp/sweep_impedance_conventions.py --run-julia
```

Sweep against an existing Julia reference (no Julia run), with matched
structured mesh:

```bash
python validation/bempp/sweep_impedance_conventions.py \
  --julia-prefix case03_z200_n0_f3p00 \
  --freq-ghz 3.0 --zs-imag-ohm 200 \
  --theta-inc-deg 0 --phi-inc-deg 0 \
  --n-theta 180 --n-phi 72 \
  --mesh-mode structured --nx 12 --ny 12
```

Outputs:

- `data/impedance_convention_sweep.csv`
- `data/impedance_convention_sweep.md`
- `data/impedance_convention_sweep.json`

### Impedance Diagnostic Plots

Generate heuristic plots to inspect where impedance mismatch is concentrated
(cut comparison, cut error, 2D error map, error-vs-level scatter):

```bash
python validation/bempp/plot_impedance_comparison.py \
  --julia-prefix case03_z200_n0_f3p00 \
  --bempp-prefix case03_z200_n0_f3p00 \
  --output-prefix case03_z200_diag \
  --theta-min 0 --theta-max 90 \
  --title "Impedance case: Zs=i200 Ohm, f=3.0 GHz"
```

### Operator-Aligned Impedance Benchmark (Current/Phase + Residuals)

Run a single-case benchmark that combines:
- far-field beam metrics,
- element-center complex current comparison,
- phase/coherence diagnostics,
- forward-solve residual checks in Julia and Bempp.

```bash
python validation/bempp/run_impedance_operator_aligned_benchmark.py \
  --output-prefix opalign_z200 \
  --freq-ghz 3.0 \
  --zs-imag-ohm 200 \
  --theta-inc-deg 0 \
  --phi-inc-deg 0 \
  --mesh-mode structured --nx 12 --ny 12
```

Standalone current/phase comparator for existing artifacts:

```bash
python validation/bempp/compare_impedance_operator_aligned.py \
  --output-prefix opalign_z200 \
  --mag-floor-db -20
```

## Outputs

- `data/bempp_pec_farfield.csv`
- `data/bempp_pec_cut_phi0.csv`
- `data/bempp_pec_metadata.json`
- `data/bempp_cross_validation_report.json`
- `data/bempp_cross_validation_report.md`
- `data/impedance_validation_matrix_summary.csv` (matrix run)
- `data/impedance_validation_matrix_summary.md` (matrix run)
- `data/impedance_validation_matrix_summary.json` (matrix run)
- `data/julia_*_element_currents.csv` (operator-aligned benchmark inputs)
- `data/julia_*_operator_checks.json` (Julia residual diagnostics)
- `data/bempp_*_element_currents.csv` (Bempp current-center outputs)
- `data/bempp_*_operator_aligned_report.json` (current/phase metrics)
- `data/bempp_*_operator_aligned_report.md` (current/phase metrics)
- `data/bempp_*_operator_aligned_benchmark.md` (combined summary)
- `data/bempp_*_diagnostic.png` (optional impedance diagnostic plots)
- `data/bempp_*_diagnostic_summary.txt` (optional impedance diagnostic stats)

## Notes

- Angular sampling is matched to the Julia grid (`180 x 72`) with centered bins.
- Comparison keys are matched on `(theta_deg, phi_deg)` rounded to `1e-6` deg.
- Small dB-level differences are expected because discretization and kernels differ between implementations.
