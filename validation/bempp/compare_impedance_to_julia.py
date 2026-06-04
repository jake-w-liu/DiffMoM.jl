#!/usr/bin/env python3
"""Compare Bempp impedance-loaded far-field against Julia impedance reference."""

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


def nearest_theta_stats(theta: np.ndarray, delta: np.ndarray, target_deg: float) -> dict:
    unique_thetas = np.unique(theta)
    nearest = unique_thetas[np.argmin(np.abs(unique_thetas - target_deg))]
    mask = np.isclose(theta, nearest, atol=1e-9)
    return {
        "target_theta_deg": target_deg,
        "nearest_theta_deg": float(nearest),
        "mean_abs_diff_db": float(np.mean(np.abs(delta[mask]))),
    }


def collapse_phi0_cut(theta: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average duplicate theta samples in phi≈0 cut."""
    unique_theta = np.unique(theta)
    avg = np.zeros_like(unique_theta, dtype=float)
    for i, t in enumerate(unique_theta):
        mask = np.isclose(theta, t, atol=1e-9)
        avg[i] = float(np.mean(values[mask]))
    return unique_theta, avg


def local_maxima_indices(values: np.ndarray) -> List[int]:
    indices: List[int] = []
    n = values.size
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            indices.append(i)
    return indices


def extract_beam_features(
    theta_deg: np.ndarray,
    values_db: np.ndarray,
    theta_min: float,
    theta_max: float,
    sidelobe_exclusion_deg: float,
) -> dict:
    window = (theta_deg >= theta_min) & (theta_deg <= theta_max)
    tw = theta_deg[window]
    vw = values_db[window]
    if tw.size == 0:
        return {
            "main_theta_deg": float("nan"),
            "main_level_db": float("nan"),
            "sidelobe_theta_deg": float("nan"),
            "sidelobe_level_db": float("nan"),
            "sll_down_db": float("nan"),
        }

    i_main = int(np.argmax(vw))
    main_theta = float(tw[i_main])
    main_level = float(vw[i_main])

    peak_ids = local_maxima_indices(vw)
    side_candidates: List[int] = []
    for i in peak_ids:
        if abs(float(tw[i]) - main_theta) > sidelobe_exclusion_deg:
            side_candidates.append(i)

    if side_candidates:
        i_side = max(side_candidates, key=lambda i: vw[i])
        side_theta = float(tw[i_side])
        side_level = float(vw[i_side])
        sll_down = float(main_level - side_level)
    else:
        side_theta = float("nan")
        side_level = float("nan")
        sll_down = float("nan")

    return {
        "main_theta_deg": main_theta,
        "main_level_db": main_level,
        "sidelobe_theta_deg": side_theta,
        "sidelobe_level_db": side_level,
        "sll_down_db": sll_down,
    }


def nearest_value_at(theta_grid: np.ndarray, values: np.ndarray, theta_query: float) -> Tuple[float, float]:
    idx = int(np.argmin(np.abs(theta_grid - theta_query)))
    return float(theta_grid[idx]), float(values[idx])


def write_markdown(path: Path, metrics: dict) -> None:
    lines = [
        "# Bempp vs Julia Impedance-Loaded Cross-Validation",
        "",
        "## Directional Slices",
        f"- Near 0 deg: nearest theta = {metrics['near_broadside']['nearest_theta_deg']:.1f} deg, "
        f"mean abs delta = {metrics['near_broadside']['mean_abs_diff_db']:.4f} dB",
        f"- Near 30 deg: nearest theta = {metrics['near_target']['nearest_theta_deg']:.1f} deg, "
        f"mean abs delta = {metrics['near_target']['mean_abs_diff_db']:.4f} dB",
        "",
        "## Beam-Centric Feature Metrics (phi≈0 cut)",
        f"- Main-beam angle Julia/Bempp: "
        f"{metrics['pattern_features']['julia_main_theta_deg']:.1f} / "
        f"{metrics['pattern_features']['bempp_main_theta_deg']:.1f} deg "
        f"(abs diff {metrics['pattern_features']['main_theta_abs_diff_deg']:.3f} deg)",
        f"- Main-beam level Julia/Bempp: "
        f"{metrics['pattern_features']['julia_main_level_db']:.3f} / "
        f"{metrics['pattern_features']['bempp_main_level_db']:.3f} dBi "
        f"(diff {metrics['pattern_features']['main_level_diff_db']:.3f} dB)",
        f"- Level diff at Julia-main angle: "
        f"{metrics['pattern_features']['delta_at_julia_main_db']:.3f} dB",
        f"- Side-lobe angle Julia/Bempp: "
        f"{metrics['pattern_features']['julia_sidelobe_theta_deg']:.1f} / "
        f"{metrics['pattern_features']['bempp_sidelobe_theta_deg']:.1f} deg "
        f"(abs diff {metrics['pattern_features']['sidelobe_theta_abs_diff_deg']:.3f} deg)",
        f"- Side-lobe level Julia/Bempp: "
        f"{metrics['pattern_features']['julia_sidelobe_level_db']:.3f} / "
        f"{metrics['pattern_features']['bempp_sidelobe_level_db']:.3f} dBi "
        f"(diff {metrics['pattern_features']['sidelobe_level_diff_db']:.3f} dB)",
        f"- SLL-down Julia/Bempp: "
        f"{metrics['pattern_features']['julia_sll_down_db']:.3f} / "
        f"{metrics['pattern_features']['bempp_sll_down_db']:.3f} dB "
        f"(diff {metrics['pattern_features']['sll_down_diff_db']:.3f} dB)",
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
    parser.add_argument("--output-prefix", type=str, default="impedance")
    parser.add_argument(
        "--julia-prefix",
        type=str,
        default=None,
        help="Prefix for Julia reference file (defaults to output-prefix)",
    )
    parser.add_argument(
        "--bempp-prefix",
        type=str,
        default=None,
        help="Prefix for Bempp file (defaults to output-prefix)",
    )
    parser.add_argument("--target-theta-deg", type=float, default=30.0)
    parser.add_argument("--feature-theta-min", type=float, default=0.0)
    parser.add_argument("--feature-theta-max", type=float, default=90.0)
    parser.add_argument("--sidelobe-exclusion-deg", type=float, default=10.0)
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    julia_prefix = args.julia_prefix if args.julia_prefix is not None else args.output_prefix
    bempp_prefix = args.bempp_prefix if args.bempp_prefix is not None else args.output_prefix

    julia_csv = data_dir / f"julia_{julia_prefix}_farfield.csv"
    bempp_csv = data_dir / f"bempp_{bempp_prefix}_farfield.csv"
    report_prefix = args.output_prefix
    report_json = data_dir / f"bempp_{report_prefix}_cross_validation_report.json"
    report_md = data_dir / f"bempp_{report_prefix}_cross_validation_report.md"

    if not julia_csv.exists():
        raise SystemExit(f"Missing Julia reference file: {julia_csv}")
    if not bempp_csv.exists():
        raise SystemExit(f"Missing Bempp file: {bempp_csv}")

    julia_rows = load_csv_rows(julia_csv)
    bempp_rows = load_csv_rows(bempp_csv)

    julia_map = keyed_map(julia_rows, "theta_deg", "phi_deg", "dir_julia_imp_dBi")
    bempp_map = keyed_map(bempp_rows, "theta_deg", "phi_deg", "dir_bempp_imp_dBi")

    common_keys = sorted(set(julia_map.keys()) & set(bempp_map.keys()))
    if not common_keys:
        raise SystemExit("No common (theta_deg, phi_deg) keys were found between files.")

    theta = np.array([k[0] for k in common_keys], dtype=float)
    phi = np.array([k[1] for k in common_keys], dtype=float)
    julia_vals = np.array([julia_map[k] for k in common_keys], dtype=float)
    bempp_vals = np.array([bempp_map[k] for k in common_keys], dtype=float)
    delta = bempp_vals - julia_vals

    unique_phi = np.unique(phi)
    n_phi = int(unique_phi.size)
    phi_dist = np.minimum(phi, 360.0 - phi)
    phi0_half_bin = 0.5 * (360.0 / n_phi)
    phi0_mask = phi_dist <= (phi0_half_bin + 1e-9)
    if not np.any(phi0_mask):
        nearest_phi = unique_phi[np.argmin(np.minimum(unique_phi, 360.0 - unique_phi))]
        phi0_mask = np.isclose(phi, nearest_phi, atol=1e-9)

    theta_cut, julia_cut = collapse_phi0_cut(theta[phi0_mask], julia_vals[phi0_mask])
    _, bempp_cut = collapse_phi0_cut(theta[phi0_mask], bempp_vals[phi0_mask])
    delta_cut = bempp_cut - julia_cut

    jf = extract_beam_features(
        theta_cut, julia_cut,
        theta_min=args.feature_theta_min,
        theta_max=args.feature_theta_max,
        sidelobe_exclusion_deg=args.sidelobe_exclusion_deg,
    )
    bf = extract_beam_features(
        theta_cut, bempp_cut,
        theta_min=args.feature_theta_min,
        theta_max=args.feature_theta_max,
        sidelobe_exclusion_deg=args.sidelobe_exclusion_deg,
    )
    _, delta_at_julia_main = nearest_value_at(theta_cut, delta_cut, jf["main_theta_deg"])
    _, delta_at_bempp_main = nearest_value_at(theta_cut, delta_cut, bf["main_theta_deg"])

    pattern_features = {
        "feature_theta_min_deg": float(args.feature_theta_min),
        "feature_theta_max_deg": float(args.feature_theta_max),
        "sidelobe_exclusion_deg": float(args.sidelobe_exclusion_deg),
        "julia_main_theta_deg": float(jf["main_theta_deg"]),
        "bempp_main_theta_deg": float(bf["main_theta_deg"]),
        "main_theta_abs_diff_deg": float(abs(jf["main_theta_deg"] - bf["main_theta_deg"])),
        "julia_main_level_db": float(jf["main_level_db"]),
        "bempp_main_level_db": float(bf["main_level_db"]),
        "main_level_diff_db": float(bf["main_level_db"] - jf["main_level_db"]),
        "delta_at_julia_main_db": float(delta_at_julia_main),
        "delta_at_bempp_main_db": float(delta_at_bempp_main),
        "julia_sidelobe_theta_deg": float(jf["sidelobe_theta_deg"]),
        "bempp_sidelobe_theta_deg": float(bf["sidelobe_theta_deg"]),
        "sidelobe_theta_abs_diff_deg": float(abs(jf["sidelobe_theta_deg"] - bf["sidelobe_theta_deg"])),
        "julia_sidelobe_level_db": float(jf["sidelobe_level_db"]),
        "bempp_sidelobe_level_db": float(bf["sidelobe_level_db"]),
        "sidelobe_level_diff_db": float(bf["sidelobe_level_db"] - jf["sidelobe_level_db"]),
        "julia_sll_down_db": float(jf["sll_down_db"]),
        "bempp_sll_down_db": float(bf["sll_down_db"]),
        "sll_down_diff_db": float(bf["sll_down_db"] - jf["sll_down_db"]),
    }

    metrics = {
        "num_common_points": int(len(common_keys)),
        "n_phi_detected": n_phi,
        "near_broadside": nearest_theta_stats(theta, delta, target_deg=0.0),
        "near_target": nearest_theta_stats(theta, delta, target_deg=args.target_theta_deg),
        "pattern_features": pattern_features,
    }

    report_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_markdown(report_md, metrics)

    print(f"Compared {metrics['num_common_points']} common angular samples.")
    print(
        "Beam-feature diffs (|Δθ_main|, |ΔD_main|, |ΔSLL|): "
        f"{metrics['pattern_features']['main_theta_abs_diff_deg']:.3f} deg, "
        f"{abs(metrics['pattern_features']['main_level_diff_db']):.3f} dB, "
        f"{abs(metrics['pattern_features']['sll_down_diff_db']):.3f} dB"
    )
    print(f"Saved {report_json}")
    print(f"Saved {report_md}")


if __name__ == "__main__":
    main()
