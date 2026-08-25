#!/usr/bin/env python3
"""Operator-aligned impedance comparison with current/phase diagnostics."""

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


def positive_finite_float(raw: str) -> float:
    """Parse one finite, positive command-line number."""
    value = finite_float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite, positive number, got {raw!r}"
        )
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


def xyz_key(row: dict, decimals: int) -> Tuple[float, float, float]:
    return (
        round(float(row["x_m"]), decimals),
        round(float(row["y_m"]), decimals),
        round(float(row["z_m"]), decimals),
    )


def current_vec(row: dict) -> np.ndarray:
    return np.array(
        [
            complex(float(row["Jx_re"]), float(row["Jx_im"])),
            complex(float(row["Jy_re"]), float(row["Jy_im"])),
            complex(float(row["Jz_re"]), float(row["Jz_im"])),
        ],
        dtype=np.complex128,
    )


def row_map(
    rows: Iterable[dict], decimals: int, source: Path
) -> Dict[Tuple[float, float, float], dict]:
    out: Dict[Tuple[float, float, float], dict] = {}
    numeric_columns = (
        "x_m",
        "y_m",
        "z_m",
        "Jx_re",
        "Jx_im",
        "Jy_re",
        "Jy_im",
        "Jz_re",
        "Jz_im",
    )
    for row_number, row in enumerate(rows, start=2):
        try:
            values = [float(row[column]) for column in numeric_columns]
        except KeyError as exc:
            raise SystemExit(
                f"Missing required column {exc.args[0]!r} in {source}. Regenerate "
                "the source artifact with the expected schema."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Invalid numeric value in {source} row {row_number}: {exc}. "
                "Regenerate the source artifact with finite current samples."
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(
                f"Non-finite numeric value in {source} row {row_number}. Regenerate "
                "the source artifact with finite coordinates and current samples."
            )
        key = xyz_key(row, decimals)
        if key in out:
            raise SystemExit(
                f"Duplicate rounded element-center key {key} in {source}. Regenerate "
                "the source artifact with one current sample per element center."
            )
        out[key] = row
    return out


def circular_mean_deg(phase_rad: np.ndarray) -> float:
    if phase_rad.size == 0:
        raise ValueError("phase array must not be empty")
    z = np.mean(np.exp(1j * phase_rad))
    return float(np.degrees(np.angle(z)))


def circular_std_deg(phase_rad: np.ndarray) -> float:
    if phase_rad.size == 0:
        raise ValueError("phase array must not be empty")
    r = np.abs(np.mean(np.exp(1j * phase_rad)))
    r = min(max(r, 1e-12), 1.0)
    std = np.sqrt(-2.0 * np.log(r))
    return float(np.degrees(std))


def hypothesis_rms_rel(
    julia_curr: np.ndarray, bempp_curr: np.ndarray, transform: str
) -> float:
    if transform == "direct":
        target = julia_curr
    elif transform == "conjugate":
        target = np.conjugate(julia_curr)
    elif transform == "negated":
        target = -julia_curr
    elif transform == "negated_conjugate":
        target = -np.conjugate(julia_curr)
    else:
        raise ValueError(f"Unknown transform {transform}")
    denom = np.maximum(np.linalg.norm(target, axis=1), 1e-30)
    rel = np.linalg.norm(bempp_curr - target, axis=1) / denom
    return float(np.sqrt(np.mean(rel**2)))


def reject_nonstandard_json_constant(value: str):
    raise ValueError(f"non-standard numeric constant {value}")


def load_optional_residual(path: Path):
    """Load an optional finite residual; return None when no metric is available."""
    if not path.exists():
        return None
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(
            f"Could not read standard JSON metadata from {path}: {exc}. Regenerate "
            "the source artifact, then rerun this comparison."
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(
            f"Invalid metadata in {path}: expected a JSON object. Regenerate the "
            "source artifact, then rerun this comparison."
        )
    value = data.get("solve_residual_l2_rel")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(
            f"Invalid solve_residual_l2_rel in {path}: expected a finite JSON "
            f"number or null, got {value!r}. Regenerate the source artifact."
        )
    residual = float(value)
    if not math.isfinite(residual) or residual < 0.0:
        raise SystemExit(
            f"Invalid solve_residual_l2_rel in {path}: expected a finite, "
            f"nonnegative number, got {value!r}. Regenerate the source artifact."
        )
    return residual


def format_optional_scientific(value: object) -> str:
    return "not available" if value is None else f"{float(value):.6e}"


def write_markdown(path: Path, metrics: dict) -> None:
    h = metrics["hypotheses"]
    lines = [
        "# Operator-Aligned Impedance Benchmark",
        "",
        "## Current Matching",
        f"- Common element-center points: {metrics['num_common_points']}",
        f"- Active points (mask): {metrics['num_active_points']} "
        f"(mag floor {metrics['mag_floor_db']} dB relative to Julia max)",
        f"- Vector RMS relative error: {metrics['vector_rms_rel']:.6f}",
        f"- Vector mean relative error: {metrics['vector_mean_rel']:.6f}",
        f"- Median |J| ratio (Bempp/Julia): {metrics['mag_ratio_median']:.6f}",
        f"- Mean |J| ratio (Bempp/Julia): {metrics['mag_ratio_mean']:.6f}",
        "",
        "## Phase/Coherence",
        f"- Mean coherence: {metrics['coherence_mean']:.6f}",
        f"- Median coherence: {metrics['coherence_median']:.6f}",
        f"- Circular mean phase diff: {metrics['phase_mean_deg']:.4f} deg",
        f"- Circular std phase diff: {metrics['phase_std_deg']:.4f} deg",
        "",
        "## Convention Hypotheses",
        f"- direct: {h['direct']:.6f}",
        f"- conjugate: {h['conjugate']:.6f}",
        f"- negated: {h['negated']:.6f}",
        f"- negated_conjugate: {h['negated_conjugate']:.6f}",
        f"- Best transform: {metrics['best_hypothesis']} ({metrics['best_hypothesis_rms_rel']:.6f})",
        "",
        "## Operator Residuals",
        "- Julia solve residual (rel L2): "
        f"{format_optional_scientific(metrics['julia_residual_rel_l2'])}",
        "- Bempp solve residual (rel L2): "
        f"{format_optional_scientific(metrics['bempp_residual_rel_l2'])}",
        "",
        "## Notes",
        "- This check is element-center current/phase based and complements dBi-only comparisons.",
        "- Hypothesis ranking helps isolate sign/conjugation convention effects.",
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
    parser.add_argument("--julia-prefix", type=str, default=None)
    parser.add_argument("--bempp-prefix", type=str, default=None)
    parser.add_argument("--coord-tol", type=positive_finite_float, default=1e-9)
    parser.add_argument("--mag-floor-db", type=finite_float, default=-20.0)
    parser.add_argument(
        "--max-vector-rms-rel", type=nonnegative_finite_float, default=None
    )
    parser.add_argument("--min-coherence", type=finite_float, default=None)
    args = parser.parse_args()
    if args.min_coherence is not None and not 0.0 <= args.min_coherence <= 1.0:
        parser.error("--min-coherence must be between 0 and 1 inclusive")

    data_dir = args.project_root / "data"
    julia_prefix = (
        args.julia_prefix if args.julia_prefix is not None else args.output_prefix
    )
    bempp_prefix = (
        args.bempp_prefix if args.bempp_prefix is not None else args.output_prefix
    )

    julia_csv = data_dir / f"julia_{julia_prefix}_element_currents.csv"
    bempp_csv = data_dir / f"bempp_{bempp_prefix}_element_currents.csv"
    julia_meta = data_dir / f"julia_{julia_prefix}_operator_checks.json"
    bempp_meta = data_dir / f"bempp_{bempp_prefix}_metadata.json"
    out_json = data_dir / f"bempp_{args.output_prefix}_operator_aligned_report.json"
    out_md = data_dir / f"bempp_{args.output_prefix}_operator_aligned_report.md"

    if not julia_csv.exists():
        raise SystemExit(
            f"Missing Julia current file: {julia_csv}. Run `julia --project=. "
            "validation/bempp/run_impedance_case_julia_reference.jl` with the "
            "same prefix, then rerun this comparison."
        )
    if not bempp_csv.exists():
        raise SystemExit(
            f"Missing Bempp current file: {bempp_csv}. Run `python "
            "validation/bempp/run_impedance_cross_validation.py` with the same "
            "prefix, then rerun this comparison."
        )

    decimals = max(0, int(round(-np.log10(max(args.coord_tol, 1e-15)))))
    julia_rows = load_csv_rows(julia_csv)
    bempp_rows = load_csv_rows(bempp_csv)
    julia_map = row_map(julia_rows, decimals, julia_csv)
    bempp_map = row_map(bempp_rows, decimals, bempp_csv)

    common = sorted(set(julia_map.keys()) & set(bempp_map.keys()))
    if not common:
        raise SystemExit(
            "No common rounded element-center coordinates were found. Regenerate "
            "both current files with matching meshes and coordinate tolerance, then "
            "rerun this comparison."
        )

    julia_curr = np.stack([current_vec(julia_map[k]) for k in common], axis=0)
    bempp_curr = np.stack([current_vec(bempp_map[k]) for k in common], axis=0)

    julia_mag = np.linalg.norm(julia_curr, axis=1)
    bempp_mag = np.linalg.norm(bempp_curr, axis=1)
    max_mag = float(np.max(julia_mag))
    if not math.isfinite(max_mag) or max_mag <= 0.0:
        raise SystemExit(
            f"Julia current data in {julia_csv} has no positive finite magnitude. "
            "Regenerate the reference after checking the excitation and solve residual."
        )
    mag_floor = max_mag * (10.0 ** (args.mag_floor_db / 20.0))
    active = julia_mag >= mag_floor
    if np.count_nonzero(active) < 8:
        order = np.argsort(-julia_mag)
        active = np.zeros_like(julia_mag, dtype=bool)
        active[order[: min(8, len(order))]] = True

    j_act = julia_curr[active]
    b_act = bempp_curr[active]
    jmag_act = np.maximum(np.linalg.norm(j_act, axis=1), 1e-30)
    bmag_act = np.maximum(np.linalg.norm(b_act, axis=1), 1e-30)

    vec_rel = np.linalg.norm(b_act - j_act, axis=1) / jmag_act
    mag_ratio = bmag_act / jmag_act
    inner = np.sum(np.conjugate(j_act) * b_act, axis=1)
    coherence = np.abs(inner) / (jmag_act * bmag_act)
    phase = np.angle(inner)
    for label, values in (
        ("relative current errors", vec_rel),
        ("current-magnitude ratios", mag_ratio),
        ("current coherences", coherence),
        ("phase differences", phase),
    ):
        if not np.all(np.isfinite(values)):
            raise SystemExit(
                f"Computed {label} contain a non-finite value. Check both current "
                "artifacts for numerical overflow, then regenerate them."
            )

    hypotheses = {
        name: hypothesis_rms_rel(j_act, b_act, name)
        for name in ["direct", "conjugate", "negated", "negated_conjugate"]
    }
    best = min(hypotheses, key=hypotheses.get)

    julia_res = load_optional_residual(julia_meta)
    bempp_res = load_optional_residual(bempp_meta)

    metrics = {
        "num_common_points": int(len(common)),
        "num_active_points": int(np.count_nonzero(active)),
        "coord_round_decimals": int(decimals),
        "mag_floor_db": float(args.mag_floor_db),
        "vector_rms_rel": float(np.sqrt(np.mean(vec_rel**2))),
        "vector_mean_rel": float(np.mean(vec_rel)),
        "mag_ratio_mean": float(np.mean(mag_ratio)),
        "mag_ratio_median": float(np.median(mag_ratio)),
        "coherence_mean": float(np.mean(coherence)),
        "coherence_median": float(np.median(coherence)),
        "phase_mean_deg": circular_mean_deg(phase),
        "phase_std_deg": circular_std_deg(phase),
        "hypotheses": hypotheses,
        "best_hypothesis": best,
        "best_hypothesis_rms_rel": float(hypotheses[best]),
        "julia_residual_rel_l2": julia_res,
        "bempp_residual_rel_l2": bempp_res,
    }

    out_json.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_markdown(out_md, metrics)

    print(f"Matched {metrics['num_common_points']} common element centers.")
    print(f"Active points: {metrics['num_active_points']}")
    print(f"Vector RMS relative error: {metrics['vector_rms_rel']:.6f}")
    print(f"Coherence mean: {metrics['coherence_mean']:.6f}")
    print(
        f"Best hypothesis: {metrics['best_hypothesis']} ({metrics['best_hypothesis_rms_rel']:.6f})"
    )
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")

    failed = False
    if (
        args.max_vector_rms_rel is not None
        and metrics["vector_rms_rel"] > args.max_vector_rms_rel
    ):
        print(
            f"FAIL: vector RMS relative error {metrics['vector_rms_rel']:.6f} "
            f"> threshold {args.max_vector_rms_rel:.6f}"
        )
        failed = True
    if (
        args.min_coherence is not None
        and metrics["coherence_mean"] < args.min_coherence
    ):
        print(
            f"FAIL: mean coherence {metrics['coherence_mean']:.6f} "
            f"< threshold {args.min_coherence:.6f}"
        )
        failed = True
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
