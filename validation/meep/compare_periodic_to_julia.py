#!/usr/bin/env python3
"""Compare Meep periodic cross-validation totals against Julia periodic reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    parser.add_argument("--tol-refl", type=float, default=0.12)
    parser.add_argument("--tol-trans", type=float, default=0.12)
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    julia_path = data_dir / f"julia_{args.output_prefix}_reference.json"
    meep_path = data_dir / f"meep_{args.output_prefix}_results.json"
    report_json_path = data_dir / f"meep_{args.output_prefix}_cross_validation_report.json"
    report_md_path = data_dir / f"meep_{args.output_prefix}_cross_validation_report.md"

    if not julia_path.exists():
        raise SystemExit(f"Missing Julia reference file: {julia_path}")
    if not meep_path.exists():
        raise SystemExit(f"Missing Meep results file: {meep_path}")

    julia = load_json(julia_path)
    meep = load_json(meep_path)

    j_refl = float(julia["refl_total_fraction"])
    if "trans_total_fraction_closure" in julia:
        j_trans = float(julia["trans_total_fraction_closure"])
        j_trans_model = "closure"
    else:
        j_trans = float(julia["trans_total_fraction"])
        j_trans_model = "direct"
    j_abs = float(julia["abs_total_fraction"])

    m_refl = float(meep["reflectance_total"])
    m_trans = float(meep["transmittance_total"])
    m_abs = float(meep["absorption_total"])

    diff_refl = abs(m_refl - j_refl)
    diff_trans = abs(m_trans - j_trans)
    diff_abs = abs(m_abs - j_abs)

    verdict = "PASS" if diff_refl <= args.tol_refl else "CHECK"

    metrics = {
        "output_prefix": args.output_prefix,
        "frequency_ghz": julia["frequency_ghz"],
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
    }

    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    write_markdown(report_md_path, metrics)

    print(f"Saved {report_json_path}")
    print(f"Saved {report_md_path}")
    print(
        "Comparison summary: "
        f"|delta R|={diff_refl:.6f}, "
        f"|delta T|={diff_trans:.6f}, "
        f"verdict={verdict}"
    )


if __name__ == "__main__":
    main()
