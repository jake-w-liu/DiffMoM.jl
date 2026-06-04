#!/usr/bin/env python3
"""Sweep Bempp impedance-convention variants against a fixed Julia reference."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


ETA0 = 376.730313668


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


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Impedance Convention Sweep")
    lines.append("")
    lines.append(
        "| Rank | op-sign | rhs-cross | rhs-sign | phase-sign | zs-scale | Main |Δθ| (deg) | Main |ΔL| (dB) | SLL |Δ| (dB) | Beam score |"
    )
    lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---:|")
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {row['op_sign']} | {row['rhs_cross']} | {row['rhs_sign']:.1f} | "
            f"{row['phase_sign']} | {row['zs_scale']:.8f} | {row['main_theta_abs_diff_deg']:.4f} | "
            f"{row['main_level_abs_diff_db']:.4f} | {row['sll_abs_diff_db']:.4f} | {row['score']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/ and Project.toml",
    )
    parser.add_argument("--freq-ghz", type=float, default=3.0)
    parser.add_argument("--zs-imag-ohm", type=float, default=200.0)
    parser.add_argument("--theta-inc-deg", type=float, default=0.0)
    parser.add_argument("--phi-inc-deg", type=float, default=0.0)
    parser.add_argument("--n-theta", type=int, default=60)
    parser.add_argument("--n-phi", type=int, default=24)
    parser.add_argument("--mesh-mode", choices=["gmsh_screen", "structured"], default="structured")
    parser.add_argument("--nx", type=int, default=12)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--mesh-step-lambda", type=float, default=0.2)
    parser.add_argument("--target-theta-deg", type=float, default=30.0)
    parser.add_argument("--julia-prefix", type=str, default="convref")
    parser.add_argument("--tag", type=str, default="convsweep")
    parser.add_argument("--run-julia", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.run_julia:
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
                args.julia_prefix,
            ],
            cwd=project_root,
            dry_run=args.dry_run,
        )

    combos = list(
        itertools.product(
            ("minus", "plus"),
            ("e_cross_n", "n_cross_e"),
            (1.0, -1.0),
            ("plus", "minus"),
            (1.0, 1.0 / ETA0, ETA0),
        )
    )

    rows: List[Dict[str, object]] = []
    for idx, (op_sign, rhs_cross, rhs_sign, phase_sign, zs_scale) in enumerate(combos, start=1):
        prefix = f"{args.tag}_{idx:02d}"
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
                "--op-sign",
                op_sign,
                "--rhs-cross",
                rhs_cross,
                "--rhs-sign",
                str(rhs_sign),
                "--phase-sign",
                phase_sign,
                "--zs-scale",
                str(zs_scale),
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
                "--julia-prefix",
                args.julia_prefix,
                "--bempp-prefix",
                prefix,
                "--target-theta-deg",
                str(args.target_theta_deg),
            ],
            cwd=project_root,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue

        report = load_json(data_dir / f"bempp_{prefix}_cross_validation_report.json")
        pf = report.get("pattern_features", {})
        main_theta_abs_diff = abs(float(pf.get("main_theta_abs_diff_deg", float("inf"))))
        main_level_abs_diff = abs(float(pf.get("main_level_diff_db", float("inf"))))
        sll_abs_diff = abs(float(pf.get("sll_down_diff_db", float("inf"))))
        score = main_theta_abs_diff + main_level_abs_diff + sll_abs_diff
        rows.append(
            {
                "prefix": prefix,
                "op_sign": op_sign,
                "rhs_cross": rhs_cross,
                "rhs_sign": rhs_sign,
                "phase_sign": phase_sign,
                "zs_scale": zs_scale,
                "main_theta_abs_diff_deg": main_theta_abs_diff,
                "main_level_abs_diff_db": main_level_abs_diff,
                "sll_abs_diff_db": sll_abs_diff,
                "score": score,
            }
        )

    if args.dry_run:
        print("\nDry run complete.")
        return

    rows.sort(key=lambda row: row["score"])
    out_csv = data_dir / "impedance_convention_sweep.csv"
    out_md = data_dir / "impedance_convention_sweep.md"
    out_json = data_dir / "impedance_convention_sweep.json"

    write_csv(out_csv, rows)
    write_md(out_md, rows)
    out_json.write_text(
        json.dumps(
            {
                "config": {
                    "freq_ghz": args.freq_ghz,
                    "zs_imag_ohm": args.zs_imag_ohm,
                    "theta_inc_deg": args.theta_inc_deg,
                    "phi_inc_deg": args.phi_inc_deg,
                    "n_theta": args.n_theta,
                    "n_phi": args.n_phi,
                    "mesh_mode": args.mesh_mode,
                    "nx": args.nx,
                    "ny": args.ny,
                    "mesh_step_lambda": args.mesh_step_lambda,
                    "target_theta_deg": args.target_theta_deg,
                    "julia_prefix": args.julia_prefix,
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved {out_csv}")
    print(f"Saved {out_md}")
    print(f"Saved {out_json}")
    if rows:
        best = rows[0]
        print(
            "Best convention: "
            f"op={best['op_sign']}, rhs_cross={best['rhs_cross']}, "
            f"rhs_sign={best['rhs_sign']}, phase={best['phase_sign']}, "
            f"zs_scale={best['zs_scale']}, "
            f"|Δθ_main|={best['main_theta_abs_diff_deg']:.4f}, "
            f"|ΔD_main|={best['main_level_abs_diff_db']:.4f}, "
            f"|ΔSLL|={best['sll_abs_diff_db']:.4f}"
        )


if __name__ == "__main__":
    main()
