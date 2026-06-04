# Chapter 5: Meep Open-Source Cross-Validation

## Purpose

Provide an additional **open-source external sanity check** for periodic workflows
using [Meep](https://meep.readthedocs.io/) (FDTD) against the Julia periodic-MoM pipeline.

This chapter is complementary to the Bempp cross-validation chapter:
- Bempp is boundary-integral to boundary-integral.
- Meep is a cross-method check (surface-current MoM vs finite-thickness FDTD).

Because of this model mismatch, we treat **reflectance** as the primary metric.

---

## 1) Scope and Modeling Caveat

The Meep workflow in `validation/meep/` compares:
- Julia periodic MoM on a binary metal-mask unit cell.
- Meep FDTD with periodic boundaries in `x,y`, PML in `z`, and voxelized metal blocks.

Not operator-identical:
- Julia model: infinitesimally thin current sheet.
- Meep model: finite-thickness conductor (`metal_thickness_lambda=0.03` default).
- Julia periodic assembly/postprocessing in this workflow uses
  Bloch-paired RWG (`build_rwg_periodic`) for boundary-touching unit-cell conductors.
  Non-Bloch periodic RWG input is rejected.

Therefore:
1. Primary agreement metric: total reflectance difference `|ΔR|`.
2. Transmission is reported using Julia `closure` transmission as a bounded baseline.

---

## 2) Workflow

From package root:

```bash
# 1) Julia reference export (geometry + periodic metrics)
julia --project=. validation/meep/run_periodic_case_julia_reference.jl \
  --output-prefix meep_periodic \
  --periodic-bc bloch

# 2) Meep run on exported geometry
python validation/meep/run_periodic_cross_validation.py \
  --output-prefix meep_periodic

# 3) Comparison report (reflectance-primary verdict)
python validation/meep/compare_periodic_to_julia.py \
  --output-prefix meep_periodic

# 4) Heuristic trend curve (R vs slot width)
python validation/meep/run_reflectance_curve_comparison.py \
  --prefix-base meep_curve_demo \
  --slot-wx-fracs 0.20,0.30,0.40 \
  --nx 14 --ny 14 \
  --periodic-bc bloch
```

Outputs in `data/`:
- `julia_<prefix>_geometry.json`
- `julia_<prefix>_reference.json`
- `meep_<prefix>_results.json`
- `meep_<prefix>_cross_validation_report.json`
- `<prefix_base>_curve_summary.csv`
- `<prefix_base>_reflectance_curve.png`

---

## 3) Verified Compact Matrix

Verified run set (10 GHz, normal incidence, `x`-polarized):

| Case | Mesh | Slot `(wx, wy)` | `|ΔR|` | `|ΔT|` | Verdict |
|---|---:|---:|---:|---:|---:|
| Representative | 8×8 | (0.40, 0.20) | 0.067 | 0.069 | PASS |
| Stress | 8×8 | (0.30, 0.20) | 0.127 | 0.095 | CHECK |
| Stress | 10×10 | (0.30, 0.20) | 0.140 | 0.032 | CHECK |

Notes:
- Coarse periodic meshes (`nx,ny < 14`) can depress Julia reflectance and enlarge
  `|ΔR|`; use `nx,ny >= 14` for primary comparisons.
- The legacy periodic-BC path is removed in this workflow; use Bloch pairing only.
- The previous `R>1` artifact was traced to polarization projection and fixed.
- Matrix CSV artifact: `data/meep_open_source_matrix.csv`.

---

## 4) Practical Interpretation

Use this chapter for:
- confirming open-source cross-solver consistency at the workflow level,
- detecting gross model or convention mismatches,
- documenting expected discrepancy bands for cross-method comparisons.

Avoid over-interpreting Meep-vs-MoM agreement as strict operator equivalence.
