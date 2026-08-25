#!/usr/bin/env python3
"""Run a heuristic Julia-vs-Meep reflectance curve comparison over slot width."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from _meep_common import (
    add_project_root_argument,
    add_runtime_arguments,
    cli_options,
    load_json_object,
    nonnegative_finite_float,
    parse_unit_interval_list,
    positive_finite_float,
    positive_int,
    run_command,
    unit_interval_float,
    validate_runtime_geometry,
)


def slug_float(x: float) -> str:
    return f"{x:.3f}".replace("-", "m").replace(".", "p")


def make_plot(rows: List[Dict[str, Any]], out_png: Path, tol_refl: float) -> None:
    import matplotlib.pyplot as plt

    x = [float(r["slot_wx_frac"]) for r in rows]
    j_r = [float(r["julia_refl_total"]) for r in rows]
    m_r = [float(r["meep_refl_total"]) for r in rows]
    d_r = [float(r["abs_diff_refl"]) for r in rows]

    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True, constrained_layout=True
    )

    axes[0].plot(x, j_r, "o-", label="Julia (MoM)", linewidth=1.6)
    axes[0].plot(x, m_r, "s-", label="Meep (FDTD)", linewidth=1.6)
    axes[0].set_ylabel("Total Reflectance R")
    axes[0].set_title("Heuristic Reflectance Curve Match: Julia vs Meep")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, d_r, "d-", color="tab:red", linewidth=1.6, label="|ΔR|")
    axes[1].axhline(
        tol_refl,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Tolerance ({tol_refl:.3f})",
    )
    axes[1].set_xlabel("Slot width fraction wx")
    axes[1].set_ylabel("|ΔR|")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_root_argument(parser, __file__)
    parser.add_argument("--prefix-base", type=str, default="meep_curve")
    parser.add_argument("--freq-ghz", type=positive_finite_float, default=10.0)
    parser.add_argument("--dx-lambda", type=positive_finite_float, default=1.2)
    parser.add_argument("--dy-lambda", type=positive_finite_float, default=1.2)
    parser.add_argument("--nx", type=positive_int, default=14)
    parser.add_argument("--ny", type=positive_int, default=14)
    parser.add_argument(
        "--slot-wx-fracs",
        type=parse_unit_interval_list,
        default="0.20,0.30,0.40",
    )
    parser.add_argument("--slot-wy-frac", type=unit_interval_float, default=0.20)
    parser.add_argument("--tol-refl", type=nonnegative_finite_float, default=0.12)
    parser.add_argument("--periodic-bc", type=str, default="bloch", choices=["bloch"])
    parser.add_argument("--reuse-existing", action="store_true")

    add_runtime_arguments(
        parser, resolution_default=30, after_sources_default=180.0
    )
    args = parser.parse_args()
    validate_runtime_geometry(parser, args)
    if args.nx < 14 or args.ny < 14:
        print(
            "WARNING: nx/ny below 14 is outside this workflow's comparison "
            "configuration. Confirm mesh convergence before interpreting the "
            "solver difference."
        )
        print(f"         received nx={args.nx}, ny={args.ny}")

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    meep_dir = Path(__file__).resolve().parent
    julia_script = meep_dir / "run_periodic_case_julia_reference.jl"
    meep_script = meep_dir / "run_periodic_cross_validation.py"
    compare_script = meep_dir / "compare_periodic_to_julia.py"

    rows: List[Dict[str, Any]] = []

    for wx in args.slot_wx_fracs:
        case_prefix = f"{args.prefix_base}_wx{slug_float(wx)}"
        print(f"\n=== Curve Case wx={wx:.3f} ({case_prefix}) ===")

        julia_ref = data_dir / f"julia_{case_prefix}_reference.json"
        meep_ref = data_dir / f"meep_{case_prefix}_results.json"
        cmp_ref = data_dir / f"meep_{case_prefix}_cross_validation_report.json"

        if not (
            args.reuse_existing
            and julia_ref.exists()
            and meep_ref.exists()
            and cmp_ref.exists()
        ):
            run_command(
                [
                    "julia",
                    f"--project={project_root}",
                    str(julia_script),
                ]
                + cli_options(
                    ("--output-prefix", case_prefix),
                    ("--freq-ghz", args.freq_ghz),
                    ("--dx-lambda", args.dx_lambda),
                    ("--dy-lambda", args.dy_lambda),
                    ("--nx", args.nx),
                    ("--ny", args.ny),
                    ("--slot-wx-frac", wx),
                    ("--slot-wy-frac", args.slot_wy_frac),
                    ("--periodic-bc", args.periodic_bc),
                ),
                cwd=project_root,
            )

            run_command(
                [
                    sys.executable,
                    str(meep_script),
                ]
                + cli_options(
                    ("--project-root", project_root),
                    ("--output-prefix", case_prefix),
                    ("--resolution", args.resolution),
                    ("--pml-lambda", args.pml_lambda),
                    ("--sz-lambda", args.sz_lambda),
                    ("--metal-thickness-lambda", args.metal_thickness_lambda),
                    ("--source-offset-lambda", args.source_offset_lambda),
                    ("--refl-offset-lambda", args.refl_offset_lambda),
                    ("--tran-offset-lambda", args.tran_offset_lambda),
                    ("--fwidth", args.fwidth),
                    ("--after-sources-time", args.after_sources_time),
                ),
                cwd=project_root,
            )

            run_command(
                [
                    sys.executable,
                    str(compare_script),
                ]
                + cli_options(
                    ("--project-root", project_root),
                    ("--output-prefix", case_prefix),
                    ("--tol-refl", args.tol_refl),
                ),
                cwd=project_root,
            )

        recovery = "Regenerate the case artifacts before building the curve summary."
        julia = load_json_object(julia_ref, recovery=recovery)
        meep = load_json_object(meep_ref, recovery=recovery)
        rep = load_json_object(cmp_ref, recovery=recovery)

        rows.append(
            {
                "case_prefix": case_prefix,
                "slot_wx_frac": wx,
                "slot_wy_frac": float(args.slot_wy_frac),
                "nx": int(args.nx),
                "ny": int(args.ny),
                "periodic_bc_model": str(julia.get("periodic_bc_model", "unknown")),
                "julia_refl_total": float(julia["refl_total_fraction"]),
                "meep_refl_total": float(meep["reflectance_total"]),
                "abs_diff_refl": float(rep["abs_diff_refl"]),
                "julia_trans_total": float(
                    julia.get(
                        "trans_total_fraction_closure", julia["trans_total_fraction"]
                    )
                ),
                "meep_trans_total": float(meep["transmittance_total"]),
                "abs_diff_trans": float(rep["abs_diff_trans"]),
                "verdict": rep["verdict"],
            }
        )

    rows.sort(key=lambda r: float(r["slot_wx_frac"]))

    csv_path = data_dir / f"{args.prefix_base}_curve_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = data_dir / f"{args.prefix_base}_curve_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, allow_nan=False)
        f.write("\n")

    plot_path = data_dir / f"{args.prefix_base}_reflectance_curve.png"
    make_plot(rows, plot_path, tol_refl=float(args.tol_refl))

    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {plot_path}")

    print("\nCurve summary:")
    for r in rows:
        print(
            f"  wx={r['slot_wx_frac']:.3f}: "
            f"R_julia={r['julia_refl_total']:.4f}, "
            f"R_meep={r['meep_refl_total']:.4f}, "
            f"|ΔR|={r['abs_diff_refl']:.4f}, verdict={r['verdict']}"
        )


if __name__ == "__main__":
    main()
