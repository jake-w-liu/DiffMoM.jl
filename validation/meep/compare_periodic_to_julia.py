#!/usr/bin/env python3
"""Compare Meep periodic cross-validation totals against Julia periodic reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict


def reject_nonstandard_json_constant(value: str):
    raise ValueError(f"non-standard numeric constant {value}")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(
                f,
                parse_constant=reject_nonstandard_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(
            f"Could not read a JSON result from {path}: {exc}. Regenerate the "
            "source result before comparing it."
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(
            f"Invalid result in {path}: expected a JSON object. Regenerate the "
            "source result before comparing it."
        )
    return data


def nonnegative_finite_float(raw: str) -> float:
    """Parse a finite, nonnegative command-line tolerance."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite, nonnegative value, got {raw!r}"
        )
    return value


def require_finite_metric(source: Path, data: Dict[str, Any], key: str) -> float:
    """Return one numeric result or stop with its source and recovery action."""
    if key not in data:
        raise SystemExit(
            f"Missing required metric {key!r} in {source}. Regenerate the source "
            "result before comparing it."
        )
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(
            f"Invalid {key!r} in {source}: expected a JSON number, got {value!r}. "
            "Regenerate the source result before comparing it."
        )
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid {key!r} in {source}: expected a number, got {value!r}. "
            "Regenerate the source result before comparing it."
        ) from exc
    if not math.isfinite(converted):
        raise SystemExit(
            f"Invalid {key!r} in {source}: expected a finite number, got "
            f"{converted!r}. Regenerate the source result before comparing it."
        )
    return converted


def write_markdown(path: Path, metrics: Dict[str, Any]) -> None:
    lines = [
        "# Meep vs Julia Periodic Cross-Validation",
        "",
        f"- Julia periodic BC model: `{metrics['julia_periodic_bc_model']}`",
        f"- Julia transmission reference model: `{metrics['julia_trans_model']}`",
        f"- Verdict basis: `{metrics['verdict_basis']}`",
        "",
        "## Totals",
        f"- Julia reflected total: {metrics['julia_refl_total']:.6f}",
        f"- Meep reflected total:  {metrics['meep_refl_total']:.6f}",
        f"- |delta R|:             {metrics['abs_diff_refl']:.6f}",
        "",
        f"- Julia transmitted total: {metrics['julia_trans_total']:.6f}",
        f"- Meep transmitted total:  {metrics['meep_trans_total']:.6f}",
        f"- |delta T|:               {metrics['abs_diff_trans']:.6f}",
        f"- Transmittance diagnostic: **{metrics['trans_verdict']}**",
        "",
        f"- Julia absorption estimate: {metrics['julia_abs_total']:.6f}",
        f"- Meep absorption estimate:  {metrics['meep_abs_total']:.6f}",
        f"- |delta A|:                {metrics['abs_diff_abs']:.6f}",
        "",
        "## Status",
        f"- Reflectance tolerance: {metrics['tol_refl']:.6f}",
        f"- Transmittance tolerance: {metrics['tol_trans']:.6f}",
        f"- Verdict: **{metrics['verdict']}**",
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
    parser.add_argument("--output-prefix", type=str, default="meep_periodic")
    parser.add_argument("--tol-refl", type=nonnegative_finite_float, default=0.12)
    parser.add_argument("--tol-trans", type=nonnegative_finite_float, default=0.12)
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    julia_path = data_dir / f"julia_{args.output_prefix}_reference.json"
    meep_path = data_dir / f"meep_{args.output_prefix}_results.json"
    report_json_path = (
        data_dir / f"meep_{args.output_prefix}_cross_validation_report.json"
    )
    report_md_path = data_dir / f"meep_{args.output_prefix}_cross_validation_report.md"

    if not julia_path.exists():
        raise SystemExit(
            f"Missing Julia reference file: {julia_path}. Run "
            "validation/meep/run_periodic_case_julia_reference.jl with the "
            "same --output-prefix, then rerun this comparison."
        )
    if not meep_path.exists():
        raise SystemExit(
            f"Missing Meep results file: {meep_path}. Run "
            "validation/meep/run_periodic_cross_validation.py with the same "
            "--output-prefix, then rerun this comparison."
        )

    julia = load_json(julia_path)
    meep = load_json(meep_path)

    frequency_ghz = require_finite_metric(julia_path, julia, "frequency_ghz")
    j_refl = require_finite_metric(julia_path, julia, "refl_total_fraction")
    if "trans_total_fraction_closure" in julia:
        j_trans = require_finite_metric(
            julia_path, julia, "trans_total_fraction_closure"
        )
        j_trans_model = "closure"
    else:
        j_trans = require_finite_metric(julia_path, julia, "trans_total_fraction")
        j_trans_model = "direct"
    j_abs = require_finite_metric(julia_path, julia, "abs_total_fraction")

    m_refl = require_finite_metric(meep_path, meep, "reflectance_total")
    m_trans = require_finite_metric(meep_path, meep, "transmittance_total")
    m_abs = require_finite_metric(meep_path, meep, "absorption_total")

    diff_refl = abs(m_refl - j_refl)
    diff_trans = abs(m_trans - j_trans)
    diff_abs = abs(m_abs - j_abs)

    verdict = "PASS" if diff_refl <= args.tol_refl else "CHECK"
    trans_verdict = "PASS" if diff_trans <= args.tol_trans else "CHECK"

    metrics = {
        "output_prefix": args.output_prefix,
        "frequency_ghz": frequency_ghz,
        "julia_periodic_bc_model": julia.get("periodic_bc_model", "unknown"),
        "julia_trans_model": j_trans_model,
        "verdict_basis": "reflectance_primary",
        "julia_refl_total": j_refl,
        "meep_refl_total": m_refl,
        "abs_diff_refl": diff_refl,
        "julia_trans_total": j_trans,
        "meep_trans_total": m_trans,
        "abs_diff_trans": diff_trans,
        "julia_abs_total": j_abs,
        "meep_abs_total": m_abs,
        "abs_diff_abs": diff_abs,
        "tol_refl": args.tol_refl,
        "tol_trans": args.tol_trans,
        "verdict": verdict,
        "trans_verdict": trans_verdict,
    }

    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
        f.write("\n")

    write_markdown(report_md_path, metrics)

    print(f"Saved {report_json_path}")
    print(f"Saved {report_md_path}")
    print(
        "Comparison summary: "
        f"|delta R|={diff_refl:.6f}, "
        f"|delta T|={diff_trans:.6f}, "
        f"reflectance verdict={verdict}, "
        f"transmittance diagnostic={trans_verdict}"
    )


if __name__ == "__main__":
    main()
