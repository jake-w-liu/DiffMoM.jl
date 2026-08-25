#!/usr/bin/env python3
"""Sweep Bempp impedance-convention variants against a fixed Julia reference."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List

from _bempp_common import (
    ImpedanceValidationConfig,
    add_incident_arguments,
    add_project_root_argument,
    add_sampling_arguments,
    finite_float,
    impedance_comparison_command,
    load_json_object,
    require_pattern_metric,
    run_command,
)


ETA0 = 376.730313668


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
    add_project_root_argument(parser, __file__)
    add_incident_arguments(parser)
    add_sampling_arguments(
        parser,
        n_theta_default=60,
        n_phi_default=24,
        mesh_mode_default="structured",
        mesh_step_default=0.2,
    )
    parser.add_argument("--target-theta-deg", type=finite_float, default=30.0)
    parser.add_argument("--julia-prefix", type=str, default="convref")
    parser.add_argument("--tag", type=str, default="convsweep")
    parser.add_argument("--run-julia", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    validation = ImpedanceValidationConfig.from_namespace(args)

    if args.run_julia:
        run_command(
            validation.julia_command(args.julia_prefix),
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
    for idx, (op_sign, rhs_cross, rhs_sign, phase_sign, zs_scale) in enumerate(
        combos, start=1
    ):
        prefix = f"{args.tag}_{idx:02d}"
        run_command(
            validation.bempp_command(
                prefix,
                op_sign=op_sign,
                rhs_cross=rhs_cross,
                rhs_sign=rhs_sign,
                phase_sign=phase_sign,
                zs_scale=zs_scale,
            ),
            cwd=project_root,
            dry_run=args.dry_run,
        )
        run_command(
            impedance_comparison_command(
                prefix,
                target_theta_deg=args.target_theta_deg,
                julia_prefix=args.julia_prefix,
                bempp_prefix=prefix,
            ),
            cwd=project_root,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue

        report_path = data_dir / f"bempp_{prefix}_cross_validation_report.json"
        report = load_json_object(
            report_path,
            recovery="Regenerate the comparison report, then rerun the sweep.",
        )
        try:
            main_theta_abs_diff = abs(
                require_pattern_metric(report, "main_theta_abs_diff_deg")
            )
            main_level_abs_diff = abs(
                require_pattern_metric(report, "main_level_diff_db")
            )
            sll_abs_diff = abs(require_pattern_metric(report, "sll_down_diff_db"))
        except ValueError as exc:
            raise SystemExit(
                f"Invalid report in {report_path}: {exc}. Regenerate the comparison "
                "report, then rerun the sweep."
            ) from exc
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
            allow_nan=False,
        )
        + "\n",
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
