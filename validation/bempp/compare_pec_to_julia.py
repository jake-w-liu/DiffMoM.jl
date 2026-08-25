#!/usr/bin/env python3
"""Compare Bempp PEC far-field against Julia reference data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def finite_float(raw: str) -> float:
    """Parse one finite command-line number."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {raw!r}")
    return value


def load_csv_rows(path: Path) -> List[dict]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, UnicodeError) as exc:
        raise SystemExit(
            f"Could not read CSV data from {path}: {exc}. Regenerate the source "
            "artifact, then rerun this comparison."
        ) from exc
    if reader.fieldnames is None or not rows:
        raise SystemExit(
            f"No CSV data rows found in {path}. Regenerate the source artifact, "
            "then rerun this comparison."
        )
    return rows


def keyed_map(
    rows: Iterable[dict],
    theta_key: str,
    phi_key: str,
    value_key: str,
    source: Path,
) -> Dict[Tuple[float, float], float]:
    out: Dict[Tuple[float, float], float] = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            theta_raw = float(row[theta_key])
            phi_raw = float(row[phi_key])
            value = float(row[value_key])
        except KeyError as exc:
            raise SystemExit(
                f"Missing required column {exc.args[0]!r} in {source}. Regenerate "
                "the source artifact with the expected schema."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Invalid numeric value in {source} row {row_number}: {exc}. "
                "Regenerate the source artifact with finite numeric samples."
            ) from exc
        if not all(math.isfinite(value) for value in (theta_raw, phi_raw, value)):
            raise SystemExit(
                f"Non-finite numeric value in {source} row {row_number}. Regenerate "
                "the source artifact with finite angular and directivity samples."
            )
        key = (round(theta_raw, 6), round(phi_raw, 6))
        if key in out:
            raise SystemExit(
                f"Duplicate rounded angular key {key} in {source}. Regenerate the "
                "source artifact with one sample per angular key."
            )
        out[key] = value
    return out


def summary_stats(values: np.ndarray) -> dict:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("summary input must contain finite samples")
    abs_values = np.abs(values)
    return {
        "mean_diff_db": float(np.mean(values)),
        "mean_abs_diff_db": float(np.mean(abs_values)),
    }


def nearest_theta_stats(
    theta: np.ndarray, delta: np.ndarray, target_deg: float
) -> dict:
    if theta.size == 0 or theta.size != delta.size:
        raise ValueError("theta and delta must be nonempty arrays of equal length")
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
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/.",
    )
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

    julia_rows = load_csv_rows(julia_csv)
    bempp_rows = load_csv_rows(bempp_csv)

    julia_map = keyed_map(julia_rows, "theta_deg", "phi_deg", "dir_pec_dBi", julia_csv)
    bempp_map = keyed_map(
        bempp_rows, "theta_deg", "phi_deg", "dir_bempp_dBi", bempp_csv
    )

    common_keys = sorted(set(julia_map.keys()) & set(bempp_map.keys()))
    if not common_keys:
        raise SystemExit(
            "No common rounded (theta_deg, phi_deg) keys were found. Regenerate "
            "both far-field files with the same angular grid, then rerun this "
            "comparison."
        )

    theta = np.array([k[0] for k in common_keys], dtype=float)
    phi = np.array([k[1] for k in common_keys], dtype=float)
    julia_vals = np.array([julia_map[k] for k in common_keys], dtype=float)
    bempp_vals = np.array([bempp_map[k] for k in common_keys], dtype=float)
    delta = bempp_vals - julia_vals

    phi_dist = np.abs(np.mod(phi + 180.0, 360.0) - 180.0)
    nearest_phi_distance = float(np.min(phi_dist))
    phi0_mask = np.isclose(phi_dist, nearest_phi_distance, atol=1e-9, rtol=0.0)

    metrics = {
        "num_common_points": int(len(common_keys)),
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
