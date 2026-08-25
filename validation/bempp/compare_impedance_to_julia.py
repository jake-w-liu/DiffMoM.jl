#!/usr/bin/env python3
"""Compare Bempp impedance-loaded far-field against Julia impedance reference."""

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


def nonnegative_finite_float(raw: str) -> float:
    """Parse one finite, nonnegative command-line number."""
    value = finite_float(raw)
    if value < 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite, nonnegative number, got {raw!r}"
        )
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
            sample = float(row[value_key])
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
        if not all(math.isfinite(number) for number in (theta_raw, phi_raw, sample)):
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
        out[key] = sample
    return out


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


def collapse_phi0_cut(
    theta: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
        raise ValueError("beam-feature window contains no angular samples")

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
        side_theta = None
        side_level = None
        sll_down = None

    return {
        "main_theta_deg": main_theta,
        "main_level_db": main_level,
        "sidelobe_theta_deg": side_theta,
        "sidelobe_level_db": side_level,
        "sll_down_db": sll_down,
    }


def nearest_value_at(
    theta_grid: np.ndarray, values: np.ndarray, theta_query: float
) -> Tuple[float, float]:
    idx = int(np.argmin(np.abs(theta_grid - theta_query)))
    return float(theta_grid[idx]), float(values[idx])


def optional_difference(first: object, second: object, *, absolute: bool = False):
    """Return a finite difference, or None when either optional metric is absent."""
    if first is None or second is None:
        return None
    difference = float(first) - float(second)
    return abs(difference) if absolute else difference


def write_markdown(path: Path, metrics: dict) -> None:
    features = metrics["pattern_features"]
    lines = [
        "# Bempp vs Julia Impedance-Loaded Cross-Validation",
        "",
        "## Directional Slices",
        f"- Near 0 deg: nearest theta = {metrics['near_broadside']['nearest_theta_deg']:.1f} deg, "
        f"mean abs delta = {metrics['near_broadside']['mean_abs_diff_db']:.4f} dB",
        f"- Near {metrics['near_target']['target_theta_deg']:g} deg: nearest theta = "
        f"{metrics['near_target']['nearest_theta_deg']:.1f} deg, "
        f"mean abs delta = {metrics['near_target']['mean_abs_diff_db']:.4f} dB",
        "",
        "## Beam-Centric Feature Metrics",
        f"- Sampled cut nearest |phi|: {metrics['phi0_cut_abs_deg']:.4f} deg",
        f"- Main-beam angle Julia/Bempp: "
        f"{features['julia_main_theta_deg']:.1f} / "
        f"{features['bempp_main_theta_deg']:.1f} deg "
        f"(abs diff {features['main_theta_abs_diff_deg']:.3f} deg)",
        f"- Main-beam level Julia/Bempp: "
        f"{features['julia_main_level_db']:.3f} / "
        f"{features['bempp_main_level_db']:.3f} dBi "
        f"(diff {features['main_level_diff_db']:.3f} dB)",
        f"- Level diff at Julia-main angle: "
        f"{features['delta_at_julia_main_db']:.3f} dB",
    ]
    if features["sll_down_diff_db"] is None:
        lines.append(
            "- Side-lobe metrics: not available because at least one pattern has "
            "no eligible local maximum outside the main-beam exclusion window."
        )
    else:
        lines.extend(
            [
                f"- Side-lobe angle Julia/Bempp: "
                f"{features['julia_sidelobe_theta_deg']:.1f} / "
                f"{features['bempp_sidelobe_theta_deg']:.1f} deg "
                f"(abs diff {features['sidelobe_theta_abs_diff_deg']:.3f} deg)",
                f"- Side-lobe level Julia/Bempp: "
                f"{features['julia_sidelobe_level_db']:.3f} / "
                f"{features['bempp_sidelobe_level_db']:.3f} dBi "
                f"(diff {features['sidelobe_level_diff_db']:.3f} dB)",
                f"- SLL-down Julia/Bempp: "
                f"{features['julia_sll_down_db']:.3f} / "
                f"{features['bempp_sll_down_db']:.3f} dB "
                f"(diff {features['sll_down_diff_db']:.3f} dB)",
            ]
        )
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
    parser.add_argument("--target-theta-deg", type=finite_float, default=30.0)
    parser.add_argument("--feature-theta-min", type=finite_float, default=0.0)
    parser.add_argument("--feature-theta-max", type=finite_float, default=90.0)
    parser.add_argument(
        "--sidelobe-exclusion-deg", type=nonnegative_finite_float, default=10.0
    )
    args = parser.parse_args()
    if args.feature_theta_min > args.feature_theta_max:
        parser.error("--feature-theta-min must not exceed --feature-theta-max")

    data_dir = args.project_root / "data"
    julia_prefix = (
        args.julia_prefix if args.julia_prefix is not None else args.output_prefix
    )
    bempp_prefix = (
        args.bempp_prefix if args.bempp_prefix is not None else args.output_prefix
    )

    julia_csv = data_dir / f"julia_{julia_prefix}_farfield.csv"
    bempp_csv = data_dir / f"bempp_{bempp_prefix}_farfield.csv"
    report_prefix = args.output_prefix
    report_json = data_dir / f"bempp_{report_prefix}_cross_validation_report.json"
    report_md = data_dir / f"bempp_{report_prefix}_cross_validation_report.md"

    if not julia_csv.exists():
        raise SystemExit(
            f"Missing Julia reference file: {julia_csv}. Run `julia --project=. "
            "validation/bempp/run_impedance_case_julia_reference.jl` with the "
            "same prefix, then rerun this comparison."
        )
    if not bempp_csv.exists():
        raise SystemExit(
            f"Missing Bempp file: {bempp_csv}. Run `python "
            "validation/bempp/run_impedance_cross_validation.py` with the same "
            "prefix, then rerun this comparison."
        )

    julia_rows = load_csv_rows(julia_csv)
    bempp_rows = load_csv_rows(bempp_csv)

    julia_map = keyed_map(
        julia_rows, "theta_deg", "phi_deg", "dir_julia_imp_dBi", julia_csv
    )
    bempp_map = keyed_map(
        bempp_rows, "theta_deg", "phi_deg", "dir_bempp_imp_dBi", bempp_csv
    )

    common_keys = sorted(set(julia_map.keys()) & set(bempp_map.keys()))
    if not common_keys:
        raise SystemExit(
            "No common rounded (theta_deg, phi_deg) keys were found. Regenerate "
            "both far-field files with the same angular grid and prefixes, then "
            "rerun this comparison."
        )

    theta = np.array([k[0] for k in common_keys], dtype=float)
    phi = np.array([k[1] for k in common_keys], dtype=float)
    julia_vals = np.array([julia_map[k] for k in common_keys], dtype=float)
    bempp_vals = np.array([bempp_map[k] for k in common_keys], dtype=float)
    delta = bempp_vals - julia_vals

    unique_phi = np.unique(phi)
    n_phi = int(unique_phi.size)
    phi_dist = np.abs(np.mod(phi + 180.0, 360.0) - 180.0)
    nearest_phi_distance = float(np.min(phi_dist))
    phi0_mask = np.isclose(phi_dist, nearest_phi_distance, atol=1e-9, rtol=0.0)

    theta_cut, julia_cut = collapse_phi0_cut(theta[phi0_mask], julia_vals[phi0_mask])
    _, bempp_cut = collapse_phi0_cut(theta[phi0_mask], bempp_vals[phi0_mask])
    delta_cut = bempp_cut - julia_cut

    feature_window = (theta_cut >= args.feature_theta_min) & (
        theta_cut <= args.feature_theta_max
    )
    if not np.any(feature_window):
        raise SystemExit(
            "No samples on the cut nearest phi zero fall inside the requested "
            f"feature window [{args.feature_theta_min:g}, "
            f"{args.feature_theta_max:g}] deg. Choose bounds within the available "
            f"theta range [{float(np.min(theta_cut)):g}, "
            f"{float(np.max(theta_cut)):g}] deg."
        )

    jf = extract_beam_features(
        theta_cut,
        julia_cut,
        theta_min=args.feature_theta_min,
        theta_max=args.feature_theta_max,
        sidelobe_exclusion_deg=args.sidelobe_exclusion_deg,
    )
    bf = extract_beam_features(
        theta_cut,
        bempp_cut,
        theta_min=args.feature_theta_min,
        theta_max=args.feature_theta_max,
        sidelobe_exclusion_deg=args.sidelobe_exclusion_deg,
    )
    _, delta_at_julia_main = nearest_value_at(
        theta_cut, delta_cut, jf["main_theta_deg"]
    )
    _, delta_at_bempp_main = nearest_value_at(
        theta_cut, delta_cut, bf["main_theta_deg"]
    )

    pattern_features = {
        "feature_theta_min_deg": float(args.feature_theta_min),
        "feature_theta_max_deg": float(args.feature_theta_max),
        "sidelobe_exclusion_deg": float(args.sidelobe_exclusion_deg),
        "julia_main_theta_deg": float(jf["main_theta_deg"]),
        "bempp_main_theta_deg": float(bf["main_theta_deg"]),
        "main_theta_abs_diff_deg": float(
            abs(jf["main_theta_deg"] - bf["main_theta_deg"])
        ),
        "julia_main_level_db": float(jf["main_level_db"]),
        "bempp_main_level_db": float(bf["main_level_db"]),
        "main_level_diff_db": float(bf["main_level_db"] - jf["main_level_db"]),
        "delta_at_julia_main_db": float(delta_at_julia_main),
        "delta_at_bempp_main_db": float(delta_at_bempp_main),
        "julia_sidelobe_theta_deg": jf["sidelobe_theta_deg"],
        "bempp_sidelobe_theta_deg": bf["sidelobe_theta_deg"],
        "sidelobe_theta_abs_diff_deg": optional_difference(
            jf["sidelobe_theta_deg"], bf["sidelobe_theta_deg"], absolute=True
        ),
        "julia_sidelobe_level_db": jf["sidelobe_level_db"],
        "bempp_sidelobe_level_db": bf["sidelobe_level_db"],
        "sidelobe_level_diff_db": optional_difference(
            bf["sidelobe_level_db"], jf["sidelobe_level_db"]
        ),
        "julia_sll_down_db": jf["sll_down_db"],
        "bempp_sll_down_db": bf["sll_down_db"],
        "sll_down_diff_db": optional_difference(bf["sll_down_db"], jf["sll_down_db"]),
    }

    metrics = {
        "num_common_points": int(len(common_keys)),
        "n_phi_detected": n_phi,
        "phi0_cut_abs_deg": nearest_phi_distance,
        "near_broadside": nearest_theta_stats(theta, delta, target_deg=0.0),
        "near_target": nearest_theta_stats(
            theta, delta, target_deg=args.target_theta_deg
        ),
        "pattern_features": pattern_features,
    }

    report_json.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_markdown(report_md, metrics)

    print(f"Compared {metrics['num_common_points']} common angular samples.")
    sll_diff = metrics["pattern_features"]["sll_down_diff_db"]
    sll_text = "not available" if sll_diff is None else f"{abs(sll_diff):.3f} dB"
    print(
        "Beam-feature diffs (|Δθ_main|, |ΔD_main|, |ΔSLL|): "
        f"{metrics['pattern_features']['main_theta_abs_diff_deg']:.3f} deg, "
        f"{abs(metrics['pattern_features']['main_level_diff_db']):.3f} dB, "
        f"{sll_text}"
    )
    print(f"Saved {report_json}")
    print(f"Saved {report_md}")


if __name__ == "__main__":
    main()
