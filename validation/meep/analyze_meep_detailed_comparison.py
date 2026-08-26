#!/usr/bin/env python3
"""Build a detailed heuristic Julia-vs-Meep comparison from existing validation outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _meep_common import (
    add_project_root_argument,
    load_json_object,
    validate_meep_result_provenance,
)


JSON_RECOVERY = "Regenerate the source result before building this comparison."


def load_cases(
    data_dir: Path, names: List[str], *, prefix_base: str = ""
) -> List[Dict[str, Any]]:
    """Load matched Julia and Meep artifacts for each requested prefix."""
    rows: List[Dict[str, Any]] = []
    for name in names:
        prefix = f"{prefix_base}_{name}" if prefix_base else name
        gpath = data_dir / f"julia_{prefix}_geometry.json"
        jpath = data_dir / f"julia_{prefix}_reference.json"
        mpath = data_dir / f"meep_{prefix}_results.json"
        if not gpath.exists() or not jpath.exists() or not mpath.exists():
            raise SystemExit(
                f"Missing validation artifacts for prefix {prefix!r}: expected "
                f"{gpath}, {jpath}, and {mpath}. Generate the Julia artifact "
                "pair and Meep result for that prefix, then rerun this analysis."
            )
        g = load_json_object(gpath, recovery=JSON_RECOVERY)
        j = load_json_object(jpath, recovery=JSON_RECOVERY)
        m = load_json_object(mpath, recovery=JSON_RECOVERY)
        geometry_sha256, reference_sha256 = validate_meep_result_provenance(
            prefix, gpath, g, jpath, j, mpath, m)
        rows.append(
            {
                "prefix": prefix,
                "julia_geometry_sha256": geometry_sha256,
                "julia_reference_sha256": reference_sha256,
                "wx": float(j["slot_wx_frac"]),
                "nx": int(j["nx"]),
                "ny": int(j["ny"]),
                "julia_refl": float(j["refl_total_fraction"]),
                "meep_refl": float(m["reflectance_total"]),
            }
        )
    return rows


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("at least one comparison row is required")
    j = np.array([r["julia_refl"] for r in rows], dtype=float)
    m = np.array([r["meep_refl"] for r in rows], dtype=float)
    if not np.all(np.isfinite(j)) or not np.all(np.isfinite(m)):
        raise ValueError("comparison rows must contain finite reflectance values")
    d = j - m
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    bias = float(np.mean(d))
    max_abs = float(np.max(np.abs(d)))
    corr = None
    if len(rows) >= 2 and np.std(j) > 0.0 and np.std(m) > 0.0:
        candidate = float(np.corrcoef(j, m)[0, 1])
        corr = candidate if math.isfinite(candidate) else None
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "max_abs_diff": max_abs,
        "corr": corr,
    }


def make_plot(
    curve_rows: List[Dict[str, Any]], conv_rows: List[Dict[str, Any]], out_png: Path
) -> None:
    import matplotlib.pyplot as plt

    wx = np.array([r["wx"] for r in curve_rows], dtype=float)
    jr = np.array([r["julia_refl"] for r in curve_rows], dtype=float)
    mr = np.array([r["meep_refl"] for r in curve_rows], dtype=float)
    dr = jr - mr

    nx = np.array([r["nx"] for r in conv_rows], dtype=float)
    jr_n = np.array([r["julia_refl"] for r in conv_rows], dtype=float)
    mr_n = np.array([r["meep_refl"] for r in conv_rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)

    axes[0].plot(wx, jr, "o-", linewidth=1.8, label="Julia (MoM)")
    axes[0].plot(wx, mr, "s-", linewidth=1.8, label="Meep (FDTD)")
    axes[0].set_title("Reflectance Curve")
    axes[0].set_xlabel("Slot width fraction wx")
    axes[0].set_ylabel("Total reflectance R")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(wx, dr, "d-", color="tab:red", linewidth=1.8, label="R_julia - R_meep")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.2)
    axes[1].set_title("Reflectance Bias vs wx")
    axes[1].set_xlabel("Slot width fraction wx")
    axes[1].set_ylabel("Bias ΔR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(nx, jr_n, "o-", linewidth=1.8, label="Julia (MoM)")
    axes[2].plot(nx, mr_n, "s-", linewidth=1.8, label="Meep (FDTD)")
    axes[2].set_title("Mesh Convergence (wx fixed)")
    axes[2].set_xlabel("nx (=ny)")
    axes[2].set_ylabel("Total reflectance R")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_markdown(
    out_md: Path,
    curve_rows: List[Dict[str, Any]],
    conv_rows: List[Dict[str, Any]],
    curve_metrics: Dict[str, Any],
    conv_metrics: Dict[str, Any],
    out_png: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Detailed Meep vs Julia Heuristic Comparison")
    lines.append("")
    lines.append("## Curve-Sweep Metrics")
    lines.append(f"- MAE(|ΔR|): {curve_metrics['mae']:.6f}")
    lines.append(f"- RMSE(ΔR): {curve_metrics['rmse']:.6f}")
    lines.append(f"- Mean bias (Julia - Meep): {curve_metrics['bias']:.6f}")
    lines.append(f"- Max |ΔR|: {curve_metrics['max_abs_diff']:.6f}")
    curve_corr = curve_metrics["corr"]
    if curve_corr is None:
        lines.append(
            "- Pearson corr(R_julia, R_meep): not available (fewer than two "
            "varying samples)"
        )
    else:
        lines.append(f"- Pearson corr(R_julia, R_meep): {curve_corr:.6f}")
    lines.append("")
    lines.append("## Curve-Sweep Points")
    lines.append("| wx | Julia R | Meep R | ΔR |")
    lines.append("|---:|--------:|-------:|---:|")
    for r in curve_rows:
        d = r["julia_refl"] - r["meep_refl"]
        lines.append(
            f"| {r['wx']:.3f} | {r['julia_refl']:.6f} | {r['meep_refl']:.6f} | {d:.6f} |"
        )
    lines.append("")
    lines.append("## Mesh-Convergence Points")
    lines.append("| nx | Julia R | Meep R | ΔR |")
    lines.append("|---:|--------:|-------:|---:|")
    for r in conv_rows:
        d = r["julia_refl"] - r["meep_refl"]
        lines.append(
            f"| {r['nx']} | {r['julia_refl']:.6f} | {r['meep_refl']:.6f} | {d:.6f} |"
        )
    lines.append("")
    lines.append("## Mesh-Convergence Metrics")
    lines.append(f"- MAE(|ΔR|): {conv_metrics['mae']:.6f}")
    lines.append(f"- RMSE(ΔR): {conv_metrics['rmse']:.6f}")
    lines.append(f"- Mean bias (Julia - Meep): {conv_metrics['bias']:.6f}")
    lines.append(f"- Max |ΔR|: {conv_metrics['max_abs_diff']:.6f}")
    lines.append("")
    lines.append("## Plot")
    lines.append(f"- `{out_png.name}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_root_argument(parser, __file__)
    parser.add_argument("--curve-prefix-base", type=str, default="meep_curve_bugfix")
    parser.add_argument("--curve-suffixes", type=str, default="wx0p200,wx0p300,wx0p400")
    parser.add_argument(
        "--conv-prefixes", type=str, default="dbg_jconv_n10,dbg_jconv_n14,dbg_jconv_n20"
    )
    parser.add_argument("--out-base", type=str, default="meep_detailed_heuristic_check")
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    curve_suffixes = [s.strip() for s in args.curve_suffixes.split(",") if s.strip()]
    conv_prefixes = [s.strip() for s in args.conv_prefixes.split(",") if s.strip()]
    if not curve_suffixes:
        parser.error("--curve-suffixes must contain at least one nonempty suffix")
    if not conv_prefixes:
        parser.error("--conv-prefixes must contain at least one nonempty prefix")

    curve_rows = load_cases(
        data_dir, curve_suffixes, prefix_base=args.curve_prefix_base
    )
    conv_rows = load_cases(data_dir, conv_prefixes)
    curve_rows.sort(key=lambda row: row["wx"])
    conv_rows.sort(key=lambda row: row["nx"])

    curve_metrics = compute_metrics(curve_rows)
    conv_metrics = compute_metrics(conv_rows)

    out_png = data_dir / f"{args.out_base}.png"
    out_json = data_dir / f"{args.out_base}.json"
    out_md = data_dir / f"{args.out_base}.md"

    make_plot(curve_rows, conv_rows, out_png)

    payload = {
        "curve_prefix_base": args.curve_prefix_base,
        "curve_suffixes": curve_suffixes,
        "conv_prefixes": conv_prefixes,
        "curve_metrics": curve_metrics,
        "conv_metrics": conv_metrics,
        "curve_rows": curve_rows,
        "conv_rows": conv_rows,
        "plot_file": out_png.name,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")

    write_markdown(out_md, curve_rows, conv_rows, curve_metrics, conv_metrics, out_png)

    print(f"Saved {out_png}")
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
    print(
        "Curve MAE={:.6f}, Curve bias={:.6f}, Conv MAE={:.6f}".format(
            curve_metrics["mae"], curve_metrics["bias"], conv_metrics["mae"]
        )
    )


if __name__ == "__main__":
    main()
