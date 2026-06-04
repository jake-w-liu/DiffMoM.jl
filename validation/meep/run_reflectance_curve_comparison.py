#!/usr/bin/env python3
"""Run a heuristic Julia-vs-Meep reflectance curve comparison over slot width."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List


def parse_float_list(raw: str) -> List[float]:
    vals: List[float] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("No values parsed from --slot-wx-fracs.")
    return vals


def slug_float(x: float) -> str:
    return f"{x:.3f}".replace("-", "m").replace(".", "p")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def make_plot(rows: List[Dict[str, Any]], out_png: Path, tol_refl: float) -> None:
    import matplotlib.pyplot as plt

    x = [float(r["slot_wx_frac"]) for r in rows]
    j_r = [float(r["julia_refl_total"]) for r in rows]
    m_r = [float(r["meep_refl_total"]) for r in rows]
    d_r = [float(r["abs_diff_refl"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, constrained_layout=True)

    axes[0].plot(x, j_r, "o-", label="Julia (MoM)", linewidth=1.6)
    axes[0].plot(x, m_r, "s-", label="Meep (FDTD)", linewidth=1.6)
    axes[0].set_ylabel("Total Reflectance R")
    axes[0].set_title("Heuristic Reflectance Curve Match: Julia vs Meep")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, d_r, "d-", color="tab:red", linewidth=1.6, label="|ΔR|")
    axes[1].axhline(tol_refl, color="gray", linestyle="--", linewidth=1.2, label=f"Tolerance ({tol_refl:.3f})")
    axes[1].set_xlabel("Slot width fraction wx")
    axes[1].set_ylabel("|ΔR|")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/.",
    )
    parser.add_argument("--prefix-base", type=str, default="meep_curve")
    parser.add_argument("--freq-ghz", type=float, default=10.0)
    parser.add_argument("--dx-lambda", type=float, default=1.2)
    parser.add_argument("--dy-lambda", type=float, default=1.2)
    parser.add_argument("--nx", type=int, default=14)
    parser.add_argument("--ny", type=int, default=14)
    parser.add_argument("--slot-wx-fracs", type=str, default="0.20,0.30,0.40")
    parser.add_argument("--slot-wy-frac", type=float, default=0.20)
    parser.add_argument("--tol-refl", type=float, default=0.12)
    parser.add_argument("--periodic-bc", type=str, default="bloch", choices=["bloch"])
    parser.add_argument("--reuse-existing", action="store_true")

    # Forwarded to Meep run script
    parser.add_argument("--resolution", type=int, default=30)
    parser.add_argument("--pml-lambda", type=float, default=1.0)
    parser.add_argument("--sz-lambda", type=float, default=6.0)
    parser.add_argument("--metal-thickness-lambda", type=float, default=0.03)
    parser.add_argument("--source-offset-lambda", type=float, default=0.35)
    parser.add_argument("--refl-offset-lambda", type=float, default=0.25)
    parser.add_argument("--tran-offset-lambda", type=float, default=0.35)
    parser.add_argument("--fwidth", type=float, default=0.2)
    parser.add_argument("--after-sources-time", type=float, default=180.0)
    args = parser.parse_args()
    if args.nx < 14 or args.ny < 14:
        print(
            "WARNING: nx/ny below 14 can under-resolve the periodic current basis and "
            "bias Julia reflectance low versus Meep."
        )
        print(f"         received nx={args.nx}, ny={args.ny}")

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    meep_dir = Path(__file__).resolve().parent
    julia_script = meep_dir / "run_periodic_case_julia_reference.jl"
    meep_script = meep_dir / "run_periodic_cross_validation.py"
    compare_script = meep_dir / "compare_periodic_to_julia.py"

    wx_list = parse_float_list(args.slot_wx_fracs)
    rows: List[Dict[str, Any]] = []

    for wx in wx_list:
        case_prefix = f"{args.prefix_base}_wx{slug_float(wx)}"
        print(f"\n=== Curve Case wx={wx:.3f} ({case_prefix}) ===")

        julia_ref = data_dir / f"julia_{case_prefix}_reference.json"
        meep_ref = data_dir / f"meep_{case_prefix}_results.json"
        cmp_ref = data_dir / f"meep_{case_prefix}_cross_validation_report.json"

        if not (args.reuse_existing and julia_ref.exists() and meep_ref.exists() and cmp_ref.exists()):
            run_cmd(
                [
                    "julia",
                    f"--project={project_root}",
                    str(julia_script),
                    "--output-prefix",
                    case_prefix,
                    "--freq-ghz",
                    str(args.freq_ghz),
                    "--dx-lambda",
                    str(args.dx_lambda),
                    "--dy-lambda",
                    str(args.dy_lambda),
                    "--nx",
                    str(args.nx),
                    "--ny",
                    str(args.ny),
                    "--slot-wx-frac",
                    str(wx),
                    "--slot-wy-frac",
                    str(args.slot_wy_frac),
                    "--periodic-bc",
                    str(args.periodic_bc),
                ],
                cwd=project_root,
            )

            run_cmd(
                [
                    sys.executable,
                    str(meep_script),
                    "--project-root",
                    str(project_root),
                    "--output-prefix",
                    case_prefix,
                    "--resolution",
                    str(args.resolution),
                    "--pml-lambda",
                    str(args.pml_lambda),
                    "--sz-lambda",
                    str(args.sz_lambda),
                    "--metal-thickness-lambda",
                    str(args.metal_thickness_lambda),
                    "--source-offset-lambda",
                    str(args.source_offset_lambda),
                    "--refl-offset-lambda",
                    str(args.refl_offset_lambda),
                    "--tran-offset-lambda",
                    str(args.tran_offset_lambda),
                    "--fwidth",
                    str(args.fwidth),
                    "--after-sources-time",
                    str(args.after_sources_time),
                ],
                cwd=project_root,
            )

            run_cmd(
                [
                    sys.executable,
                    str(compare_script),
                    "--project-root",
                    str(project_root),
                    "--output-prefix",
                    case_prefix,
                    "--tol-refl",
                    str(args.tol_refl),
                ],
                cwd=project_root,
            )

        julia = load_json(julia_ref)
        meep = load_json(meep_ref)
        rep = load_json(cmp_ref)

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
                "julia_trans_total": float(julia.get("trans_total_fraction_closure", julia["trans_total_fraction"])),
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
        json.dump({"rows": rows}, f, indent=2)
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
