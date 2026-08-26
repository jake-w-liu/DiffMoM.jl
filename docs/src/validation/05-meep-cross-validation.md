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

## 3) Interpret the Generated Report

`compare_periodic_to_julia.py` records the effective boundary model,
transmission reference, tolerances, totals, and verdict in the generated JSON
and Markdown reports. The overall verdict is reflectance-primary: it is `PASS`
when `|ΔR|` does not exceed the effective `--tol-refl` value. `|ΔT|` receives a
separate diagnostic status against `--tol-trans`; it does not change the overall
verdict because the sheet and finite-thickness models are not operator-identical.
The comparator writes both reports and exits nonzero when the reflectance verdict
is not `PASS`; a transmission-only diagnostic does not change the exit status.

The workflow accepts Bloch pairing only. For a mesh-convergence check, repeat
the reference export with increasing `--nx` and `--ny` while holding the
geometry, Meep resolution, and comparison tolerances fixed. Read the measured
values from `<prefix_base>_curve_summary.json` or the per-case comparison JSON
rather than copying one run into this chapter.

The workflow rejects non-standard `NaN` and infinity constants when reading
JSON and writes only standard finite numbers or `null`. A detailed heuristic
report records Pearson correlation as `null` when fewer than two varying points
make that statistic unavailable.

Each Meep result records SHA-256 digests of its Julia geometry and reference
JSON inputs. The runner first requires those two artifacts to carry identical
case fields; the comparator then requires their current digests to match the
ones stored in the Meep result. A mismatch is stale or mixed provenance, not a
comparison result: regenerate the Julia pair and rerun Meep.

For a curve sweep, `--reuse-existing` reuses only matching Julia and Meep
solver inputs. The comparator still runs for every point with the current
tolerance and provenance checks. A reflectance `CHECK` stops the sweep with a
nonzero exit before a new curve summary is written. Reuse requires all Julia
geometry controls and Meep source, monitor, PML, resolution, bandwidth, and
runtime controls to match the current command exactly. Curve rows take their
mesh and slot values from those validated artifacts.

---

## 4) Practical Interpretation

Use this chapter for:
- confirming open-source cross-solver consistency at the workflow level,
- detecting gross model or convention mismatches,
- documenting expected discrepancy bands for cross-method comparisons.

Avoid over-interpreting Meep-vs-MoM agreement as strict operator equivalence.
