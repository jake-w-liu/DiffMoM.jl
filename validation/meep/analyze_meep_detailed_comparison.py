#!/usr/bin/env python3
"""Build a detailed heuristic Julia-vs-Meep comparison from existing validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_curve_cases(data_dir: Path, curve_prefix_base: str, suffixes: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sfx in suffixes:
        prefix = f"{curve_prefix_base}_{sfx}"
        jpath = data_dir / f"julia_{prefix}_reference.json"
        mpath = data_dir / f"meep_{prefix}_results.json"
        if not jpath.exists() or not mpath.exists():
            raise SystemExit(f"Missing curve files for prefix '{prefix}': {jpath} or {mpath}")
        j = load_json(jpath)
        m = load_json(mpath)
        rows.append(
            {
                "prefix": prefix,
                "wx": float(j["slot_wx_frac"]),
                "nx": int(j["nx"]),
                "ny": int(j["ny"]),
                "julia_refl": float(j["refl_total_fraction"]),
                "meep_refl": float(m["reflectance_total"]),
            }
        )
    rows.sort(key=lambda r: r["wx"])
    return rows


def load_convergence_cases(data_dir: Path, prefixes: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prefix in prefixes:
        jpath = data_dir / f"julia_{prefix}_reference.json"
        mpath = data_dir / f"meep_{prefix}_results.json"
        if not jpath.exists() or not mpath.exists():
            raise SystemExit(f"Missing convergence files for prefix '{prefix}': {jpath} or {mpath}")
        j = load_json(jpath)
        m = load_json(mpath)
        rows.append(
            {
                "prefix": prefix,
                "nx": int(j["nx"]),
                "ny": int(j["ny"]),
                "wx": float(j["slot_wx_frac"]),
                "julia_refl": float(j["refl_total_fraction"]),
                "meep_refl": float(m["reflectance_total"]),
            }
        )
    rows.sort(key=lambda r: r["nx"])
    return rows


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    j = np.array([r["julia_refl"] for r in rows], dtype=float)
    m = np.array([r["meep_refl"] for r in rows], dtype=float)
    d = j - m
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    bias = float(np.mean(d))
    max_abs = float(np.max(np.abs(d)))
    if len(rows) >= 2:
        corr = float(np.corrcoef(j, m)[0, 1])
    else:
        corr = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "max_abs_diff": max_abs,
        "corr": corr,
    }


def make_plot(curve_rows: List[Dict[str, Any]], conv_rows: List[Dict[str, Any]], out_png: Path) -> None:
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
    curve_metrics: Dict[str, float],
    conv_metrics: Dict[str, float],
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
    lines.append(f"- Pearson corr(R_julia, R_meep): {curve_metrics['corr']:.6f}")
    lines.append("")
    lines.append("## Curve-Sweep Points")
    lines.append("| wx | Julia R | Meep R | ΔR |")
    lines.append("|---:|--------:|-------:|---:|")
    for r in curve_rows:
        d = r["julia_refl"] - r["meep_refl"]
        lines.append(f"| {r['wx']:.3f} | {r['julia_refl']:.6f} | {r['meep_refl']:.6f} | {d:.6f} |")
    lines.append("")
    lines.append("## Mesh-Convergence Points")
    lines.append("| nx | Julia R | Meep R | ΔR |")
    lines.append("|---:|--------:|-------:|---:|")
    for r in conv_rows:
        d = r["julia_refl"] - r["meep_refl"]
        lines.append(f"| {r['nx']} | {r['julia_refl']:.6f} | {r['meep_refl']:.6f} | {d:.6f} |")
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
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/.",
    )
    parser.add_argument("--curve-prefix-base", type=str, default="meep_curve_bugfix")
    parser.add_argument("--curve-suffixes", type=str, default="wx0p200,wx0p300,wx0p400")
    parser.add_argument("--conv-prefixes", type=str, default="dbg_jconv_n10,dbg_jconv_n14,dbg_jconv_n20")
    parser.add_argument("--out-base", type=str, default="meep_detailed_heuristic_check")
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    curve_suffixes = [s.strip() for s in args.curve_suffixes.split(",") if s.strip()]
    conv_prefixes = [s.strip() for s in args.conv_prefixes.split(",") if s.strip()]

    curve_rows = load_curve_cases(data_dir, args.curve_prefix_base, curve_suffixes)
    conv_rows = load_convergence_cases(data_dir, conv_prefixes)

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
        json.dump(payload, f, indent=2)
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
