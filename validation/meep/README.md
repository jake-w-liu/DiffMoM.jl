# Meep Cross-Validation (Open Source)

This folder adds an open-source periodic cross-validation path using
[Meep](https://meep.readthedocs.io/) for FDTD and the in-repo Julia periodic MoM.

The workflow uses a binary PEC pixel pattern:
- Julia computes Floquet reflection metrics from periodic MoM.
- Meep runs the same unit-cell geometry with periodic boundaries in `x,y`.
- A comparator reports agreement with reflectance as the primary verdict metric.

## Files

- `run_periodic_case_julia_reference.jl`:
  Builds the periodic PEC pattern in Julia, solves periodic MoM, and exports:
  - `data/julia_<prefix>_geometry.json`
  - `data/julia_<prefix>_reference.json`
  - `data/julia_<prefix>_modes.csv`
- `run_periodic_cross_validation.py`:
  Loads the exported geometry and runs a Meep flux simulation, exporting:
  - `data/meep_<prefix>_results.json`
  - `data/meep_<prefix>_results.csv`
- `compare_periodic_to_julia.py`:
  Compares Julia vs Meep totals and writes:
  - `data/meep_<prefix>_cross_validation_report.json`
  - `data/meep_<prefix>_cross_validation_report.md`
- `run_reflectance_curve_comparison.py`:
  Runs a slot-width sweep and saves a heuristic curve match plot:
  - `data/<prefix_base>_curve_summary.csv`
  - `data/<prefix_base>_reflectance_curve.png`
- `analyze_meep_detailed_comparison.py`:
  Builds a detailed heuristic report/plot from existing curve and mesh-convergence
  outputs:
  - `data/<out_base>.png`
  - `data/<out_base>.json`
  - `data/<out_base>.md`

## Setup

From `DiffMoM.jl/validation/meep`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Meep is installed from conda-forge (recommended):
# conda install -c conda-forge pymeep
```

## Run

From `DiffMoM.jl`:

```bash
julia --project=. validation/meep/run_periodic_case_julia_reference.jl \
  --output-prefix meep_periodic \
  --periodic-bc bloch

python validation/meep/run_periodic_cross_validation.py \
  --output-prefix meep_periodic

python validation/meep/compare_periodic_to_julia.py \
  --output-prefix meep_periodic

# Heuristic trend-curve comparison (Julia vs Meep reflectance)
python validation/meep/run_reflectance_curve_comparison.py \
  --prefix-base meep_curve_demo \
  --slot-wx-fracs 0.20,0.30,0.40 \
  --nx 14 --ny 14 \
  --periodic-bc bloch

# Detailed heuristic report from existing outputs
python validation/meep/analyze_meep_detailed_comparison.py \
  --curve-prefix-base meep_curve_bugfix \
  --curve-suffixes wx0p200,wx0p300,wx0p400 \
  --conv-prefixes dbg_jconv_n10,dbg_jconv_n14,dbg_jconv_n20 \
  --out-base meep_detailed_heuristic_check
```

## Notes

- This path is intentionally open-source only (Julia + Meep).
- Julia periodic reference uses `--periodic-bc bloch`, which enables
  Bloch-paired boundary RWG functions for boundary-touching unit-cell conductors.
- The reference case is normal-incidence, `x`-polarized illumination.
- Comparison uses Julia `closure` transmission for a conservative power-bounded
  baseline; Floquet-derived transmission is still exported as a diagnostic.
- For heuristic visual matching, prefer trend curves (e.g., `R` vs slot width)
  rather than single-point scalar agreement.
- Avoid `nx,ny < 14` for cross-validation runs: coarse periodic discretization can
  depress Julia reflectance and exaggerate Julia-vs-Meep gaps.
