#!/usr/bin/env python3
"""Run operator-aligned impedance benchmark (far-field + current/phase)."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    path = env.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    if venv_bin not in entries:
        env["PATH"] = venv_bin + os.pathsep + path
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def reject_nonstandard_json_constant(value: str):
    raise ValueError(f"non-standard numeric constant {value}")


def load_json(path: Path) -> Dict:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(
            f"Could not read standard JSON from {path}: {exc}. Regenerate the "
            "benchmark artifacts, then rerun the summary."
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(
            f"Invalid report in {path}: expected a JSON object. Regenerate the "
            "benchmark artifacts, then rerun the summary."
        )
    return data


def finite_float(raw: str) -> float:
    """Parse one finite command-line number."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {raw!r}")
    return value


def positive_int(raw: str) -> int:
    """Parse one positive command-line integer."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}")
    return value


def positive_finite_float(raw: str) -> float:
    """Parse one finite, positive command-line number."""
    value = finite_float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite, positive number, got {raw!r}"
        )
    return value


def require_finite_metric(data: Dict, key: str, report_name: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(
            f"Invalid {key!r} in {report_name}: expected a finite JSON number, "
            f"got {value!r}. Regenerate the benchmark artifacts."
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise SystemExit(
            f"Invalid {key!r} in {report_name}: expected a finite JSON number, "
            f"got {value!r}. Regenerate the benchmark artifacts."
        )
    return converted


def format_optional_metric(value: object, spec: str, unit: str = "") -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(
            f"Invalid optional report metric {value!r}. Regenerate the benchmark "
            "artifacts before writing the summary."
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise SystemExit(
            f"Invalid optional report metric {value!r}. Regenerate the benchmark "
            "artifacts before writing the summary."
        )
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
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/ and Project.toml",
    )
    parser.add_argument("--output-prefix", type=str, default="impedance_operator")
    parser.add_argument("--freq-ghz", type=positive_finite_float, default=3.0)
    parser.add_argument("--zs-imag-ohm", type=finite_float, default=200.0)
    parser.add_argument("--theta-inc-deg", type=finite_float, default=0.0)
    parser.add_argument("--phi-inc-deg", type=finite_float, default=0.0)
    parser.add_argument("--n-theta", type=positive_int, default=180)
    parser.add_argument("--n-phi", type=positive_int, default=72)
    parser.add_argument(
        "--mesh-mode", choices=["gmsh_screen", "structured"], default="structured"
    )
    parser.add_argument("--nx", type=positive_int, default=12)
    parser.add_argument("--ny", type=positive_int, default=12)
    parser.add_argument("--mesh-step-lambda", type=positive_finite_float, default=0.2)
    parser.add_argument("--target-theta-deg", type=finite_float, default=30.0)
    parser.add_argument("--mag-floor-db", type=finite_float, default=-20.0)
    parser.add_argument("--skip-julia", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix

    if not args.skip_julia:
        run_cmd(
            [
                "julia",
                "--project=.",
                "validation/bempp/run_impedance_case_julia_reference.jl",
                "--freq-ghz",
                str(args.freq_ghz),
                "--theta-ohm",
                str(args.zs_imag_ohm),
                "--theta-inc-deg",
                str(args.theta_inc_deg),
                "--phi-inc-deg",
                str(args.phi_inc_deg),
                "--n-theta",
                str(args.n_theta),
                "--n-phi",
                str(args.n_phi),
                "--output-prefix",
                prefix,
            ],
            cwd=project_root,
            dry_run=args.dry_run,
        )

    run_cmd(
        [
            sys.executable,
            "validation/bempp/run_impedance_cross_validation.py",
            "--freq-ghz",
            str(args.freq_ghz),
            "--zs-imag-ohm",
            str(args.zs_imag_ohm),
            "--theta-inc-deg",
            str(args.theta_inc_deg),
            "--phi-inc-deg",
            str(args.phi_inc_deg),
            "--n-theta",
            str(args.n_theta),
            "--n-phi",
            str(args.n_phi),
            "--mesh-mode",
            args.mesh_mode,
            "--nx",
            str(args.nx),
            "--ny",
            str(args.ny),
            "--mesh-step-lambda",
            str(args.mesh_step_lambda),
            "--output-prefix",
            prefix,
        ],
        cwd=project_root,
        dry_run=args.dry_run,
    )

    run_cmd(
        [
            sys.executable,
            "validation/bempp/compare_impedance_to_julia.py",
            "--output-prefix",
            prefix,
            "--target-theta-deg",
            str(args.target_theta_deg),
        ],
        cwd=project_root,
        dry_run=args.dry_run,
    )

    run_cmd(
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

    ff_report = load_json(data_dir / f"bempp_{prefix}_cross_validation_report.json")
    op_report = load_json(data_dir / f"bempp_{prefix}_operator_aligned_report.json")
    summary_md = data_dir / f"bempp_{prefix}_operator_aligned_benchmark.md"
    write_summary(summary_md, ff_report, op_report)
    print(f"Saved {summary_md}")


if __name__ == "__main__":
    main()
