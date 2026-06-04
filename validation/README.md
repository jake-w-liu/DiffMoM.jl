# Validation Workflows

This directory contains the public reproducibility and cross-validation scripts
for `DiffMoM.jl`.

Recommended run order from the package root:

```bash
# Analytical / standalone validation
julia --project=. validation/mie/validate_mie_rcs.jl
julia --project=. validation/mie/validate_dielectric_mie_dda.jl
julia --project=. validation/po/validate_po_vs_pofacets.jl

# Paper data drivers
julia --project=. validation/paper/run_convergence_study.jl
julia --project=. validation/paper/run_beam_steering_case.jl
julia --project=. validation/scaling/run_cost_scaling.jl
julia --project=. validation/robustness/run_robustness_sweep.jl

# External cross-validation
python validation/bempp/run_pec_cross_validation.py
python validation/bempp/compare_pec_to_julia.py
python validation/bempp/run_impedance_validation_matrix.py

# Aggregated consistency snapshot
julia --project=. validation/paper/generate_consistency_report.jl
```

Outputs are written to `data/`.
