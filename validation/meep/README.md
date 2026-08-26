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
  - `data/<prefix_base>_curve_summary.json`
  - `data/<prefix_base>_reflectance_curve.png`
- `analyze_meep_detailed_comparison.py`:
  Builds a detailed heuristic report/plot from existing curve and mesh-convergence
  outputs:
  - `data/<out_base>.png`
  - `data/<out_base>.json`
  - `data/<out_base>.md`

## Setup

Create a conda environment with Meep, then install the remaining Python
dependencies from `DiffMoM.jl/validation/meep`:

```bash
conda create -n diffmom-meep -c conda-forge python pymeep
conda activate diffmom-meep
python -m pip install -r requirements.txt
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

`--reuse-existing` skips matching Julia and Meep solver inputs only. It always
reruns the comparator so current artifact hashes and `--tol-refl` apply. Any
case with a `CHECK` reflectance verdict stops the curve command with a nonzero
exit before it writes a new summary. Reuse also requires every Julia geometry
control and Meep runtime control to equal the current command; otherwise the
error names the mismatched option and asks you to regenerate. Summary geometry
fields come from the validated artifacts, not from unverified CLI labels.

## Notes

- This path is intentionally open-source only (Julia + Meep).
- Julia periodic reference uses `--periodic-bc bloch`, which enables
  Bloch-paired boundary RWG functions for boundary-touching unit-cell conductors.
- The reference case is normal-incidence, `x`-polarized illumination.
- Comparison uses Julia `closure` transmission for a conservative power-bounded
  baseline; Floquet-derived transmission is still exported as a diagnostic.
- Use trend curves (for example, `R` versus slot width) to interpret the
  cross-method comparison; a single point does not establish convergence.
- JSON readers reject `NaN` and infinity constants, and generated JSON contains
  only standard finite numbers or `null`. The detailed report uses `null` when
  fewer than two varying points make Pearson correlation unavailable.
- The Meep result records SHA-256 digests of the exact Julia geometry and
  reference JSON files. Comparisons stop if either source artifact has changed,
  or if the geometry and reference identity fields do not describe the same
  case. Regenerate the Julia and Meep artifacts instead of mixing prefixes or
  stale files.
- The supplied comparison command uses `nx=ny=14`. Repeat the run at finer
  meshes and confirm convergence before interpreting Julia-versus-Meep gaps.
