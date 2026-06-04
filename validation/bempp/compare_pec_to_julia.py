#!/usr/bin/env python3
"""Compare Bempp PEC far-field against Julia reference data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def keyed_map(rows: Iterable[dict], theta_key: str, phi_key: str, value_key: str) -> Dict[Tuple[float, float], float]:
    out: Dict[Tuple[float, float], float] = {}
    for row in rows:
        theta = round(float(row[theta_key]), 6)
        phi = round(float(row[phi_key]), 6)
        out[(theta, phi)] = float(row[value_key])
    return out


def summary_stats(values: np.ndarray) -> dict:
    abs_values = np.abs(values)
    return {
        "mean_diff_db": float(np.mean(values)),
        "mean_abs_diff_db": float(np.mean(abs_values)),
    }


def nearest_theta_stats(theta: np.ndarray, delta: np.ndarray, target_deg: float) -> dict:
    unique_thetas = np.unique(theta)
    nearest = unique_thetas[np.argmin(np.abs(unique_thetas - target_deg))]
    mask = np.isclose(theta, nearest, atol=1e-9)
    return {
        "target_theta_deg": target_deg,
        "nearest_theta_deg": float(nearest),
        "mean_abs_diff_db": float(np.mean(np.abs(delta[mask]))),
    }


def write_markdown(path: Path, metrics: dict) -> None:
    lines = [
        "# Bempp vs Julia PEC Cross-Validation",
        "",
        "## Global Error Metrics",
        f"- Mean delta (Bempp - Julia): {metrics['global']['mean_diff_db']:.4f} dB",
        f"- Mean absolute delta: {metrics['global']['mean_abs_diff_db']:.4f} dB",
        "",
        "## Phi=0 Cut Metrics",
        f"- Mean absolute delta: {metrics['phi0_cut']['mean_abs_diff_db']:.4f} dB",
        "",
        "## Directional Slices",
        f"- Near 0 deg: nearest theta = {metrics['near_broadside']['nearest_theta_deg']:.1f} deg, "
        f"mean abs delta = {metrics['near_broadside']['mean_abs_diff_db']:.4f} dB",
        f"- Near 30 deg: nearest theta = {metrics['near_target']['nearest_theta_deg']:.1f} deg, "
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
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/.",
    )
    parser.add_argument("--target-theta-deg", type=float, default=30.0)
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    julia_csv = data_dir / "beam_steer_farfield.csv"
    bempp_csv = data_dir / "bempp_pec_farfield.csv"
    report_json = data_dir / "bempp_cross_validation_report.json"
    report_md = data_dir / "bempp_cross_validation_report.md"

    if not julia_csv.exists():
        raise SystemExit(f"Missing Julia reference file: {julia_csv}")
    if not bempp_csv.exists():
        raise SystemExit(f"Missing Bempp file: {bempp_csv}")

    julia_rows = load_csv_rows(julia_csv)
    bempp_rows = load_csv_rows(bempp_csv)

    julia_map = keyed_map(julia_rows, "theta_deg", "phi_deg", "dir_pec_dBi")
    bempp_map = keyed_map(bempp_rows, "theta_deg", "phi_deg", "dir_bempp_dBi")

    common_keys = sorted(set(julia_map.keys()) & set(bempp_map.keys()))
    if not common_keys:
        raise SystemExit("No common (theta_deg, phi_deg) keys were found between files.")

    theta = np.array([k[0] for k in common_keys], dtype=float)
    phi = np.array([k[1] for k in common_keys], dtype=float)
    julia_vals = np.array([julia_map[k] for k in common_keys], dtype=float)
    bempp_vals = np.array([bempp_map[k] for k in common_keys], dtype=float)
    delta = bempp_vals - julia_vals

    phi_dist = np.minimum(phi, 360.0 - phi)
    phi0_mask = phi_dist <= (0.5 * (360.0 / 72.0) + 1e-9)

    metrics = {
        "num_common_points": int(len(common_keys)),
        "global": summary_stats(delta),
        "phi0_cut": summary_stats(delta[phi0_mask]),
        "near_broadside": nearest_theta_stats(theta, delta, target_deg=0.0),
        "near_target": nearest_theta_stats(theta, delta, target_deg=args.target_theta_deg),
    }

    report_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_markdown(report_md, metrics)

    print(f"Compared {metrics['num_common_points']} common angular samples.")
    print(f"Global mean |delta|: {metrics['global']['mean_abs_diff_db']:.4f} dB")
    print(f"Saved {report_json}")
    print(f"Saved {report_md}")


if __name__ == "__main__":
    main()
