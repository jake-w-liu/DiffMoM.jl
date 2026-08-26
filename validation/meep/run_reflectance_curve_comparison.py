#!/usr/bin/env python3
"""Run a heuristic Julia-vs-Meep reflectance curve comparison over slot width."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct
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
    validate_meep_result_provenance,
    validate_runtime_geometry,
)


def slug_float(x: float) -> str:
    rounded_text = f"{x:.3f}"
    base = rounded_text.replace("-", "m").replace(".", "p")
    if x == float(rounded_text):
        return base
    return f"{base}_{struct.pack('>d', x).hex()}"


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


def require_reuse_control(
    source: Path,
    payload: Dict[str, Any],
    field: str,
    expected: Any,
    option: str,
) -> None:
    value = payload.get(field)
    if type(value) is not type(expected) or value != expected:
        raise SystemExit(
            f"Cannot reuse {source}: stored {field}={value!r} does not match "
            f"current {option}={expected!r}. Rerun without --reuse-existing "
            "to regenerate this case."
        )


def validate_reuse_controls(
    args: Any,
    geometry_path: Path,
    geometry: Dict[str, Any],
    reference_path: Path,
    reference: Dict[str, Any],
    meep_path: Path,
    meep: Dict[str, Any],
    case_prefix: str,
    slot_wx_frac: float,
) -> None:
    validate_meep_result_provenance(
        case_prefix,
        geometry_path,
        geometry,
        reference_path,
        reference,
        meep_path,
        meep,
    )
    julia_controls = (
        ("frequency_ghz", args.freq_ghz, "--freq-ghz"),
        ("dx_lambda", args.dx_lambda, "--dx-lambda"),
        ("dy_lambda", args.dy_lambda, "--dy-lambda"),
        ("nx", args.nx, "--nx"),
        ("ny", args.ny, "--ny"),
        ("slot_wx_frac", slot_wx_frac, "--slot-wx-fracs"),
        ("slot_wy_frac", args.slot_wy_frac, "--slot-wy-frac"),
        ("periodic_bc_model", args.periodic_bc, "--periodic-bc"),
    )
    for field, expected, option in julia_controls:
        require_reuse_control(reference_path, reference, field, expected, option)

    meep_controls = (
        ("resolution_px_per_lambda", args.resolution, "--resolution"),
        ("pml_lambda", args.pml_lambda, "--pml-lambda"),
        ("sz_lambda", args.sz_lambda, "--sz-lambda"),
        (
            "requested_metal_thickness_lambda",
            args.metal_thickness_lambda,
            "--metal-thickness-lambda",
        ),
        ("source_offset_lambda", args.source_offset_lambda, "--source-offset-lambda"),
        ("refl_offset_lambda", args.refl_offset_lambda, "--refl-offset-lambda"),
        ("tran_offset_lambda", args.tran_offset_lambda, "--tran-offset-lambda"),
        ("fwidth", args.fwidth, "--fwidth"),
        ("after_sources_time", args.after_sources_time, "--after-sources-time"),
    )
    for field, expected, option in meep_controls:
        require_reuse_control(meep_path, meep, field, expected, option)


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
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse matching Julia and Meep inputs, but rerun provenance and "
            "tolerance comparison for every case."
        ),
    )

    add_runtime_arguments(
        parser, resolution_default=30, after_sources_default=180.0
    )
    args = parser.parse_args()
    validate_runtime_geometry(parser, args)
    case_prefixes = [
        f"{args.prefix_base}_wx{slug_float(width)}"
        for width in args.slot_wx_fracs
    ]
    if len(set(case_prefixes)) != len(case_prefixes):
        parser.error(
            "--slot-wx-fracs contains duplicate widths; every curve point "
            "must have a distinct exact Float64 case identifier"
        )
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

    for wx, case_prefix in zip(args.slot_wx_fracs, case_prefixes):
        print(f"\n=== Curve Case wx={wx:.3f} ({case_prefix}) ===")

        geometry_ref = data_dir / f"julia_{case_prefix}_geometry.json"
        julia_ref = data_dir / f"julia_{case_prefix}_reference.json"
        meep_ref = data_dir / f"meep_{case_prefix}_results.json"
        cmp_ref = data_dir / f"meep_{case_prefix}_cross_validation_report.json"

        reuse_inputs = (
            args.reuse_existing
            and geometry_ref.exists()
            and julia_ref.exists()
            and meep_ref.exists()
        )
        if reuse_inputs:
            recovery = (
                "Regenerate the case artifacts before reusing this curve point."
            )
            reuse_geometry = load_json_object(geometry_ref, recovery=recovery)
            reuse_julia = load_json_object(julia_ref, recovery=recovery)
            reuse_meep = load_json_object(meep_ref, recovery=recovery)
            validate_reuse_controls(
                args,
                geometry_ref,
                reuse_geometry,
                julia_ref,
                reuse_julia,
                meep_ref,
                reuse_meep,
                case_prefix,
                wx,
            )
        if not reuse_inputs:
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

        # Always regenerate the comparison report. This validates artifact
        # provenance against the current Julia inputs, applies the requested
        # tolerance, and exits nonzero when the primary reflectance gate fails.
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
        verdict = rep.get("verdict")
        if verdict not in ("PASS", "CHECK"):
            raise SystemExit(
                f"Invalid verdict in {cmp_ref}: expected 'PASS' or 'CHECK', "
                f"got {verdict!r}. Regenerate the comparison report."
            )
        if verdict != "PASS":
            raise SystemExit(
                f"Reflectance curve case {case_prefix!r} is not PASS: "
                f"verdict={verdict}. Inspect {cmp_ref} and rerun the case "
                "before building a curve summary."
            )
        if "trans_total_fraction_closure" in julia:
            julia_trans_total = float(julia["trans_total_fraction_closure"])
        else:
            julia_trans_total = float(julia["trans_total_fraction"])

        rows.append(
            {
                "case_prefix": case_prefix,
                "slot_wx_frac": float(julia["slot_wx_frac"]),
                "slot_wy_frac": float(julia["slot_wy_frac"]),
                "nx": int(julia["nx"]),
                "ny": int(julia["ny"]),
                "periodic_bc_model": str(julia.get("periodic_bc_model", "unknown")),
                "julia_refl_total": float(julia["refl_total_fraction"]),
                "meep_refl_total": float(meep["reflectance_total"]),
                "abs_diff_refl": float(rep["abs_diff_refl"]),
                "julia_trans_total": julia_trans_total,
                "meep_trans_total": float(meep["transmittance_total"]),
                "abs_diff_trans": float(rep["abs_diff_trans"]),
                "verdict": verdict,
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
