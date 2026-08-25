#!/usr/bin/env python3
"""Compare Bempp PEC far-field against Julia reference data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bempp_common import (
    add_project_root_argument,
    common_angular_arrays,
    finite_float,
    load_angular_map,
    nearest_theta_stats,
)


def summary_stats(values: np.ndarray) -> dict:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("summary input must contain finite samples")
    abs_values = np.abs(values)
    return {
        "mean_diff_db": float(np.mean(values)),
        "mean_abs_diff_db": float(np.mean(abs_values)),
    }


def write_markdown(path: Path, metrics: dict) -> None:
    lines = [
        "# Bempp vs Julia PEC Cross-Validation",
        "",
        "## Global Error Metrics",
        f"- Mean delta (Bempp - Julia): {metrics['global']['mean_diff_db']:.4f} dB",
        f"- Mean absolute delta: {metrics['global']['mean_abs_diff_db']:.4f} dB",
        "",
        "## Sampled Cut Nearest Phi Zero",
        f"- Nearest |phi|: {metrics['phi0_cut']['nearest_phi_abs_deg']:.4f} deg",
        f"- Mean absolute delta: {metrics['phi0_cut']['mean_abs_diff_db']:.4f} dB",
        "",
        "## Directional Slices",
        f"- Near 0 deg: nearest theta = {metrics['near_broadside']['nearest_theta_deg']:.1f} deg, "
        f"mean abs delta = {metrics['near_broadside']['mean_abs_diff_db']:.4f} dB",
        f"- Near {metrics['near_target']['target_theta_deg']:g} deg: nearest theta = "
        f"{metrics['near_target']['nearest_theta_deg']:.1f} deg, "
        f"mean abs delta = {metrics['near_target']['mean_abs_diff_db']:.4f} dB",
        "",
        "## Notes",
        "- Julia reference columns: `data/beam_steer_farfield.csv` -> `dir_pec_dBi`",
        "- Bempp reference columns: `data/bempp_pec_farfield.csv` -> `dir_bempp_dBi`",
        "- Grid match uses rounded `(theta_deg, phi_deg)` keys to 1e-6 deg.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_root_argument(parser, __file__)
    parser.add_argument("--target-theta-deg", type=finite_float, default=30.0)
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    julia_csv = data_dir / "beam_steer_farfield.csv"
    bempp_csv = data_dir / "bempp_pec_farfield.csv"
    report_json = data_dir / "bempp_cross_validation_report.json"
    report_md = data_dir / "bempp_cross_validation_report.md"

    if not julia_csv.exists():
        raise SystemExit(
            f"Missing Julia reference file: {julia_csv}. Run `julia --project=. "
            "validation/paper/run_beam_steering_case.jl`, then rerun this comparison."
        )
    if not bempp_csv.exists():
        raise SystemExit(
            f"Missing Bempp file: {bempp_csv}. Run `python "
            "validation/bempp/run_pec_cross_validation.py`, then rerun this comparison."
        )

    julia_map = load_angular_map(julia_csv, "dir_pec_dBi")
    bempp_map = load_angular_map(bempp_csv, "dir_bempp_dBi")
    theta, phi, julia_vals, bempp_vals = common_angular_arrays(julia_map, bempp_map)
    delta = bempp_vals - julia_vals

    phi_dist = np.abs(np.mod(phi + 180.0, 360.0) - 180.0)
    nearest_phi_distance = float(np.min(phi_dist))
    phi0_mask = np.isclose(phi_dist, nearest_phi_distance, atol=1e-9, rtol=0.0)

    metrics = {
        "num_common_points": int(theta.size),
        "global": summary_stats(delta),
        "phi0_cut": {
            "nearest_phi_abs_deg": nearest_phi_distance,
            "num_points": int(np.count_nonzero(phi0_mask)),
            **summary_stats(delta[phi0_mask]),
        },
        "near_broadside": nearest_theta_stats(theta, delta, target_deg=0.0),
        "near_target": nearest_theta_stats(
            theta, delta, target_deg=args.target_theta_deg
        ),
    }

    report_json.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_markdown(report_md, metrics)

    print(f"Compared {metrics['num_common_points']} common angular samples.")
    print(f"Global mean |delta|: {metrics['global']['mean_abs_diff_db']:.4f} dB")
    print(f"Saved {report_json}")
    print(f"Saved {report_md}")


if __name__ == "__main__":
    main()
