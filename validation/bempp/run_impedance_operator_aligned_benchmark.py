#!/usr/bin/env python3
"""Run operator-aligned impedance benchmark (far-field + current/phase)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

from _bempp_common import (
    ImpedanceValidationConfig,
    add_incident_arguments,
    add_project_root_argument,
    add_sampling_arguments,
    finite_float,
    impedance_comparison_command,
    load_json_object,
    run_command,
)


def require_finite_metric(data: Dict, key: str, report_name: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(
            f"Invalid {key!r} in {report_name}: expected a finite JSON number, "
            f"got {value!r}. Regenerate the benchmark artifacts."
        )
    return float(value)


def format_optional_metric(value: object, spec: str, unit: str = "") -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(
            f"Invalid optional report metric {value!r}. Regenerate the benchmark "
            "artifacts before writing the summary."
        )
    converted = float(value)
    suffix = f" {unit}" if unit else ""
    return f"{converted:{spec}}{suffix}"


def write_summary(path: Path, ff: Dict, op: Dict) -> None:
    pf = ff.get("pattern_features", {})
    if not isinstance(pf, dict):
        raise SystemExit(
            "Invalid far-field report: missing object 'pattern_features'. "
            "Regenerate the benchmark artifacts."
        )
    main_theta = require_finite_metric(
        pf, "main_theta_abs_diff_deg", "far-field report"
    )
    main_level = require_finite_metric(pf, "main_level_diff_db", "far-field report")
    vector_rms = require_finite_metric(op, "vector_rms_rel", "operator report")
    coherence = require_finite_metric(op, "coherence_mean", "operator report")
    phase_mean = require_finite_metric(op, "phase_mean_deg", "operator report")
    phase_std = require_finite_metric(op, "phase_std_deg", "operator report")
    hypothesis_rms = require_finite_metric(
        op, "best_hypothesis_rms_rel", "operator report"
    )
    lines = [
        "# Operator-Aligned Impedance Benchmark Summary",
        "",
        "## Far-Field Beam Metrics",
        f"- Main-beam angle abs diff: {main_theta:.4f} deg",
        f"- Main-beam level diff: {main_level:.4f} dB",
        "- Side-lobe angle abs diff: "
        f"{format_optional_metric(pf.get('sidelobe_theta_abs_diff_deg'), '.4f', 'deg')}",
        "- SLL diff: "
        f"{format_optional_metric(pf.get('sll_down_diff_db'), '.4f', 'dB')}",
        "",
        "## Current/Phase Metrics",
        f"- Vector RMS relative error: {vector_rms:.6f}",
        f"- Mean coherence: {coherence:.6f}",
        f"- Circular phase mean: {phase_mean:.4f} deg",
        f"- Circular phase std: {phase_std:.4f} deg",
        f"- Best transform hypothesis: {op.get('best_hypothesis', 'n/a')} "
        f"({hypothesis_rms:.6f})",
        "",
        "## Operator Residuals",
        "- Julia residual rel L2: "
        f"{format_optional_metric(op.get('julia_residual_rel_l2'), '.6e')}",
        "- Bempp residual rel L2: "
        f"{format_optional_metric(op.get('bempp_residual_rel_l2'), '.6e')}",
        "",
        "## Interpretation",
        "- Beam metrics show pattern-level agreement.",
        "- Current/phase metrics isolate convention or formulation mismatch sources.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_root_argument(parser, __file__)
    parser.add_argument("--output-prefix", type=str, default="impedance_operator")
    add_incident_arguments(parser)
    add_sampling_arguments(
        parser,
        n_theta_default=180,
        n_phi_default=72,
        mesh_mode_default="structured",
        mesh_step_default=0.2,
    )
    parser.add_argument("--target-theta-deg", type=finite_float, default=30.0)
    parser.add_argument("--mag-floor-db", type=finite_float, default=-20.0)
    parser.add_argument("--skip-julia", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix
    validation = ImpedanceValidationConfig.from_namespace(args)

    if not args.skip_julia:
        run_command(
            validation.julia_command(prefix),
            cwd=project_root,
            dry_run=args.dry_run,
        )

    run_command(
        validation.bempp_command(prefix),
        cwd=project_root,
        dry_run=args.dry_run,
    )

    run_command(
        impedance_comparison_command(
            prefix, target_theta_deg=args.target_theta_deg
        ),
        cwd=project_root,
        dry_run=args.dry_run,
    )

    run_command(
        [
            sys.executable,
            "validation/bempp/compare_impedance_operator_aligned.py",
            "--output-prefix",
            prefix,
            "--mag-floor-db",
            str(args.mag_floor_db),
        ],
        cwd=project_root,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run complete.")
        return

    recovery = "Regenerate the benchmark artifacts, then rerun the summary."
    ff_report = load_json_object(
        data_dir / f"bempp_{prefix}_cross_validation_report.json", recovery=recovery
    )
    op_report = load_json_object(
        data_dir / f"bempp_{prefix}_operator_aligned_report.json", recovery=recovery
    )
    summary_md = data_dir / f"bempp_{prefix}_operator_aligned_benchmark.md"
    write_summary(summary_md, ff_report, op_report)
    print(f"Saved {summary_md}")


if __name__ == "__main__":
    main()
