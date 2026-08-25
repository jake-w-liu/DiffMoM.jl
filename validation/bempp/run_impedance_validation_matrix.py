#!/usr/bin/env python3
"""Run Julia and Bempp impedance cases, compare them, and write gate summaries."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from _bempp_common import (
    ImpedanceValidationConfig,
    add_project_root_argument,
    add_sampling_arguments,
    finite_float,
    impedance_comparison_command,
    read_json_object,
    require_pattern_metric,
    run_command,
)


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    freq_ghz: float
    zs_imag_ohm: float
    theta_inc_deg: float
    phi_inc_deg: float


CASES: List[ValidationCase] = [
    ValidationCase("case01_z0_n0_f3p00", 3.00, 0.0, 0.0, 0.0),
    ValidationCase("case02_z25_n0_f3p00", 3.00, 25.0, 0.0, 0.0),
    ValidationCase("case03_z50_n0_f3p00", 3.00, 50.0, 0.0, 0.0),
    ValidationCase("case04_z75_n0_f3p00", 3.00, 75.0, 0.0, 0.0),
    ValidationCase("case05_z100_n0_f3p00", 3.00, 100.0, 0.0, 0.0),
    ValidationCase("case06_z100_n5_f3p00", 3.00, 100.0, 5.0, 0.0),
    ValidationCase("case07_z100_n0_f3p06", 3.06, 100.0, 0.0, 0.0),
]

MAX_MAIN_THETA_DIFF_DEG = 3.0
MAX_MAIN_LEVEL_DIFF_DB = 1.5
MAX_SLL_DIFF_DB = 3.0

CONVENTION_PROFILES: Dict[str, Dict[str, object]] = {
    "paper_default": {
        "op_sign": "minus",
        "rhs_cross": "e_cross_n",
        "rhs_sign": 1.0,
        "phase_sign": "plus",
        "zs_scale": 1.0,
    },
    "case03_sweep_best": {
        "op_sign": "plus",
        "rhs_cross": "e_cross_n",
        "rhs_sign": 1.0,
        "phase_sign": "plus",
        "zs_scale": 1.0 / 376.730313668,
    },
}


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(
    path: Path,
    rows: List[Dict[str, object]],
    gates: Dict[str, object],
    config: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("# Impedance Validation Matrix Summary")
    lines.append("")
    lines.append("## Bempp Convention Configuration")
    lines.append(f"- op-sign: `{config['bempp_op_sign']}`")
    lines.append(f"- rhs-cross: `{config['bempp_rhs_cross']}`")
    lines.append(f"- rhs-sign: `{config['bempp_rhs_sign']}`")
    lines.append(f"- phase-sign: `{config['bempp_phase_sign']}`")
    lines.append(f"- zs-scale: `{config['bempp_zs_scale']}`")
    lines.append(f"- mesh-mode: `{config['mesh_mode']}`")
    lines.append(f"- nx: `{config['nx']}`")
    lines.append(f"- ny: `{config['ny']}`")
    lines.append(f"- mesh-step-lambda: `{config['mesh_step_lambda']}`")
    lines.append("")
    lines.append("## Case Results")
    lines.append(
        "| Case | f (GHz) | Zs imag (ohm) | theta_inc (deg) | "
        "Main |Δθ| (deg) | Main |ΔL| (dB) | SLL |Δ| (dB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['freq_ghz']:.2f} | {row['zs_imag_ohm']:.1f} | "
            f"{row['theta_inc_deg']:.1f} | {row['main_theta_abs_diff_deg']:.3f} | "
            f"{row['main_level_abs_diff_db']:.3f} | {row['sll_abs_diff_db']:.3f} |"
        )
    lines.append("")
    lines.append("## Acceptance Gates")
    lines.append(
        f"- Cases with main-beam |Δθ| <= {MAX_MAIN_THETA_DIFF_DEG:g} "
        f"deg: {gates['count_main_theta_le_3deg']}/{gates['num_cases']}"
    )
    lines.append(
        f"- Cases with main-beam |ΔL| <= {MAX_MAIN_LEVEL_DIFF_DB:g} "
        f"dB: {gates['count_main_level_le_1p5db']}/{gates['num_cases']}"
    )
    lines.append(
        f"- Cases with |ΔSLL| <= {MAX_SLL_DIFF_DB:g} "
        f"dB: {gates['count_sll_le_3db']}/{gates['num_cases']}"
    )
    lines.append("")
    lines.append(
        f"- Beam-centric gate status (main angle/level/SLL all pass in all cases): "
        f"{'PASS' if gates['beam_gate_pass'] else 'FAIL'}"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_root_argument(parser, __file__)
    add_sampling_arguments(
        parser,
        n_theta_default=180,
        n_phi_default=72,
        mesh_mode_default="gmsh_screen",
        mesh_step_default=0.2,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print subprocess commands without running or writing summaries",
    )
    parser.add_argument(
        "--skip-julia",
        action="store_true",
        help="Reuse existing Julia reference artifacts",
    )
    parser.add_argument(
        "--skip-bempp",
        action="store_true",
        help="Reuse existing Bempp artifacts",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Reuse existing comparison reports",
    )
    parser.add_argument(
        "--convention-profile",
        choices=sorted(CONVENTION_PROFILES.keys()),
        default="paper_default",
        help=(
            "Base convention profile; explicit --bempp-* options override "
            "individual fields (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--bempp-op-sign",
        choices=["minus", "plus"],
        default=None,
        help="Override the profile's Bempp operator sign",
    )
    parser.add_argument(
        "--bempp-rhs-cross",
        choices=["e_cross_n", "n_cross_e"],
        default=None,
        help="Override the profile's RHS cross-product order",
    )
    parser.add_argument(
        "--bempp-rhs-sign",
        type=finite_float,
        default=None,
        help="Override the profile's RHS scalar sign",
    )
    parser.add_argument(
        "--bempp-phase-sign",
        choices=["plus", "minus"],
        default=None,
        help="Override the profile's far-field phase sign",
    )
    parser.add_argument(
        "--bempp-zs-scale",
        type=finite_float,
        default=None,
        help="Override the profile's sheet-impedance scale",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    profile = CONVENTION_PROFILES[args.convention_profile]

    effective_op_sign = (
        args.bempp_op_sign if args.bempp_op_sign is not None else profile["op_sign"]
    )
    effective_rhs_cross = (
        args.bempp_rhs_cross
        if args.bempp_rhs_cross is not None
        else profile["rhs_cross"]
    )
    effective_rhs_sign = (
        args.bempp_rhs_sign if args.bempp_rhs_sign is not None else profile["rhs_sign"]
    )
    effective_phase_sign = (
        args.bempp_phase_sign
        if args.bempp_phase_sign is not None
        else profile["phase_sign"]
    )
    effective_zs_scale = (
        args.bempp_zs_scale if args.bempp_zs_scale is not None else profile["zs_scale"]
    )

    summary_rows: List[Dict[str, object]] = []

    for case in CASES:
        prefix = case.case_id
        print(f"\n=== {case.case_id} ===")
        validation = ImpedanceValidationConfig.from_namespace(
            args,
            freq_ghz=case.freq_ghz,
            zs_imag_ohm=case.zs_imag_ohm,
            theta_inc_deg=case.theta_inc_deg,
            phi_inc_deg=case.phi_inc_deg,
        )

        if not args.skip_julia:
            run_command(
                validation.julia_command(prefix),
                cwd=project_root,
                dry_run=args.dry_run,
            )

        if not args.skip_bempp:
            run_command(
                validation.bempp_command(
                    prefix,
                    op_sign=str(effective_op_sign),
                    rhs_cross=str(effective_rhs_cross),
                    rhs_sign=float(effective_rhs_sign),
                    phase_sign=str(effective_phase_sign),
                    zs_scale=float(effective_zs_scale),
                ),
                cwd=project_root,
                dry_run=args.dry_run,
            )

        if not args.skip_compare:
            run_command(
                impedance_comparison_command(prefix),
                cwd=project_root,
                dry_run=args.dry_run,
            )

        if args.dry_run:
            continue

        report_json = data_dir / f"bempp_{prefix}_cross_validation_report.json"
        if not report_json.exists():
            raise SystemExit(
                f"Missing comparison report for {case.case_id}: {report_json}. "
                "Rerun without --skip-compare; also regenerate either solver "
                "artifact if its corresponding skip option reused stale data."
            )

        try:
            metrics = read_json_object(report_json)
            main_theta_diff = require_pattern_metric(metrics, "main_theta_abs_diff_deg")
            main_level_diff = abs(require_pattern_metric(metrics, "main_level_diff_db"))
            sll_diff = abs(require_pattern_metric(metrics, "sll_down_diff_db"))
            flags = {
                "pass_main_theta_le_3deg": main_theta_diff <= MAX_MAIN_THETA_DIFF_DEG,
                "pass_main_level_le_1p5db": main_level_diff <= MAX_MAIN_LEVEL_DIFF_DB,
                "pass_sll_le_3db": sll_diff <= MAX_SLL_DIFF_DB,
            }
        except ValueError as exc:
            raise SystemExit(
                f"Invalid comparison report for {case.case_id}: {exc}. Rerun "
                "without --skip-compare; regenerate either solver artifact if a "
                "skip option reused stale or incomplete data."
            ) from exc
        row: Dict[str, object] = {
            "case_id": case.case_id,
            "freq_ghz": case.freq_ghz,
            "zs_imag_ohm": case.zs_imag_ohm,
            "theta_inc_deg": case.theta_inc_deg,
            "phi_inc_deg": case.phi_inc_deg,
            "main_theta_abs_diff_deg": main_theta_diff,
            "main_level_abs_diff_db": main_level_diff,
            "sll_abs_diff_db": sll_diff,
            **flags,
        }
        summary_rows.append(row)

    if args.dry_run:
        print("\nDry run complete.")
        return

    summary_csv = data_dir / "impedance_validation_matrix_summary.csv"
    summary_md = data_dir / "impedance_validation_matrix_summary.md"
    summary_json = data_dir / "impedance_validation_matrix_summary.json"

    write_summary_csv(summary_csv, summary_rows)

    gates = {
        "num_cases": len(summary_rows),
        "count_main_theta_le_3deg": sum(
            bool(r["pass_main_theta_le_3deg"]) for r in summary_rows
        ),
        "count_main_level_le_1p5db": sum(
            bool(r["pass_main_level_le_1p5db"]) for r in summary_rows
        ),
        "count_sll_le_3db": sum(bool(r["pass_sll_le_3db"]) for r in summary_rows),
    }
    gates["beam_gate_pass"] = (
        gates["count_main_theta_le_3deg"] == gates["num_cases"]
        and gates["count_main_level_le_1p5db"] == gates["num_cases"]
        and gates["count_sll_le_3db"] == gates["num_cases"]
    )
    gates["matrix_gate_pass"] = gates["beam_gate_pass"]

    config = {
        "convention_profile": args.convention_profile,
        "bempp_op_sign": effective_op_sign,
        "bempp_rhs_cross": effective_rhs_cross,
        "bempp_rhs_sign": effective_rhs_sign,
        "bempp_phase_sign": effective_phase_sign,
        "bempp_zs_scale": effective_zs_scale,
        "mesh_mode": args.mesh_mode,
        "nx": args.nx,
        "ny": args.ny,
        "mesh_step_lambda": args.mesh_step_lambda,
    }
    write_summary_md(summary_md, summary_rows, gates, config)
    summary_json.write_text(
        json.dumps(
            {"config": config, "cases": summary_rows, "gates": gates},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved {summary_csv}")
    print(f"Saved {summary_md}")
    print(f"Saved {summary_json}")
    print(
        f"Beam-centric gate status: {'PASS' if gates['beam_gate_pass'] else 'FAIL'} "
        f"(main_theta<={MAX_MAIN_THETA_DIFF_DEG:g}deg: "
        f"{gates['count_main_theta_le_3deg']}/{gates['num_cases']}, "
        f"main_level<={MAX_MAIN_LEVEL_DIFF_DB:g}dB: "
        f"{gates['count_main_level_le_1p5db']}/{gates['num_cases']}, "
        f"sll<={MAX_SLL_DIFF_DB:g}dB: "
        f"{gates['count_sll_le_3db']}/{gates['num_cases']})"
    )
    if not gates["beam_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
