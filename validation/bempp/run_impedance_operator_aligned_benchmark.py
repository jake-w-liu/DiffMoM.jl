#!/usr/bin/env python3
"""Run operator-aligned impedance benchmark (far-field + current/phase)."""

from __future__ import annotations

import argparse
import json
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


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(path: Path, ff: Dict, op: Dict) -> None:
    pf = ff.get("pattern_features", {})
    lines = [
        "# Operator-Aligned Impedance Benchmark Summary",
        "",
        "## Far-Field Beam Metrics",
        f"- Main-beam angle abs diff: {pf.get('main_theta_abs_diff_deg', float('nan')):.4f} deg",
        f"- Main-beam level diff: {pf.get('main_level_diff_db', float('nan')):.4f} dB",
        f"- Side-lobe angle abs diff: {pf.get('sidelobe_theta_abs_diff_deg', float('nan')):.4f} deg",
        f"- SLL diff: {pf.get('sll_down_diff_db', float('nan')):.4f} dB",
        "",
        "## Current/Phase Metrics",
        f"- Vector RMS relative error: {op.get('vector_rms_rel', float('nan')):.6f}",
        f"- Mean coherence: {op.get('coherence_mean', float('nan')):.6f}",
        f"- Circular phase mean: {op.get('phase_mean_deg', float('nan')):.4f} deg",
        f"- Circular phase std: {op.get('phase_std_deg', float('nan')):.4f} deg",
        f"- Best transform hypothesis: {op.get('best_hypothesis', 'n/a')} "
        f"({op.get('best_hypothesis_rms_rel', float('nan')):.6f})",
        "",
        "## Operator Residuals",
        f"- Julia residual rel L2: {op.get('julia_residual_rel_l2', float('nan')):.6e}",
        f"- Bempp residual rel L2: {op.get('bempp_residual_rel_l2', float('nan')):.6e}",
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
    parser.add_argument("--freq-ghz", type=float, default=3.0)
    parser.add_argument("--zs-imag-ohm", type=float, default=200.0)
    parser.add_argument("--theta-inc-deg", type=float, default=0.0)
    parser.add_argument("--phi-inc-deg", type=float, default=0.0)
    parser.add_argument("--n-theta", type=int, default=180)
    parser.add_argument("--n-phi", type=int, default=72)
    parser.add_argument("--mesh-mode", choices=["gmsh_screen", "structured"], default="structured")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--mesh-step-lambda", type=float, default=0.2)
    parser.add_argument("--target-theta-deg", type=float, default=30.0)
    parser.add_argument("--mag-floor-db", type=float, default=-20.0)
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
