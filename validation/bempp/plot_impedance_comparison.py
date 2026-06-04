#!/usr/bin/env python3
"""Plot diagnostic comparisons between Julia and Bempp impedance far fields."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_farfield_map(path: Path, value_key: str) -> Dict[Tuple[float, float], float]:
    data: Dict[Tuple[float, float], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (float(row["theta_deg"]), float(row["phi_deg"]))
            data[key] = float(row[value_key])
    return data


def build_common_arrays(
    julia_map: Dict[Tuple[float, float], float],
    bempp_map: Dict[Tuple[float, float], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(set(julia_map).intersection(bempp_map))
    if not keys:
        raise RuntimeError("No common angular samples found between Julia and Bempp data.")

    theta = np.array([k[0] for k in keys], dtype=float)
    phi = np.array([k[1] for k in keys], dtype=float)
    julia_vals = np.array([julia_map[k] for k in keys], dtype=float)
    bempp_vals = np.array([bempp_map[k] for k in keys], dtype=float)
    return theta, phi, julia_vals, bempp_vals


def to_grid(
    theta: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_u = np.unique(theta)
    phi_u = np.unique(phi)
    grid = np.full((theta_u.size, phi_u.size), np.nan, dtype=float)

    theta_index = {v: i for i, v in enumerate(theta_u.tolist())}
    phi_index = {v: i for i, v in enumerate(phi_u.tolist())}

    for t, p, v in zip(theta, phi, values):
        grid[theta_index[t], phi_index[p]] = v
    return theta_u, phi_u, grid


def nearest_phi_cut(
    theta: np.ndarray,
    phi: np.ndarray,
    julia_vals: np.ndarray,
    bempp_vals: np.ndarray,
    n_phi: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_bin = 0.5 * (360.0 / n_phi)
    phi_dist = np.minimum(phi, 360.0 - phi)
    mask = phi_dist <= half_bin + 1e-12
    idx = np.argsort(theta[mask])
    return theta[mask][idx], julia_vals[mask][idx], bempp_vals[mask][idx]


def summarize_delta(delta: np.ndarray, julia_vals: np.ndarray) -> List[str]:
    abs_delta = np.abs(delta)
    main_lobe_mask = julia_vals >= (np.max(julia_vals) - 10.0)
    deep_null_mask = julia_vals <= -20.0

    def stat_line(name: str, mask: np.ndarray) -> str:
        if not np.any(mask):
            return f"{name}: N=0"
        q95 = float(np.quantile(abs_delta[mask], 0.95))
        return (
            f"{name}: N={int(np.sum(mask))}, "
            f"mean|Δ|={float(np.mean(abs_delta[mask])):.3f} dB, "
            f"p95|Δ|={q95:.3f} dB"
        )

    lines = [
        stat_line("Global", np.ones_like(delta, dtype=bool)),
        stat_line("Main-lobe (Julia >= peak-10 dB)", main_lobe_mask),
        stat_line("Deep-null (Julia <= -20 dBi)", deep_null_mask),
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/.",
    )
    parser.add_argument("--julia-prefix", required=True)
    parser.add_argument("--bempp-prefix", required=True)
    parser.add_argument("--output-prefix", default="impedance_diag")
    parser.add_argument("--title", default="")
    parser.add_argument("--delta-clim", type=float, default=12.0)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=90.0)
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    julia_csv = data_dir / f"julia_{args.julia_prefix}_farfield.csv"
    bempp_csv = data_dir / f"bempp_{args.bempp_prefix}_farfield.csv"

    if not julia_csv.exists():
        raise FileNotFoundError(f"Missing Julia far-field file: {julia_csv}")
    if not bempp_csv.exists():
        raise FileNotFoundError(f"Missing Bempp far-field file: {bempp_csv}")

    julia_map = load_farfield_map(julia_csv, "dir_julia_imp_dBi")
    bempp_map = load_farfield_map(bempp_csv, "dir_bempp_imp_dBi")

    theta, phi, julia_vals, bempp_vals = build_common_arrays(julia_map, bempp_map)
    delta = bempp_vals - julia_vals

    theta_u, phi_u, julia_grid = to_grid(theta, phi, julia_vals)
    _, _, bempp_grid = to_grid(theta, phi, bempp_vals)
    _, _, delta_grid = to_grid(theta, phi, delta)

    n_phi = phi_u.size
    cut_theta, cut_julia, cut_bempp = nearest_phi_cut(theta, phi, julia_vals, bempp_vals, n_phi=n_phi)
    cut_delta = cut_bempp - cut_julia

    summary_lines = summarize_delta(delta, julia_vals)
    for line in summary_lines:
        print(line)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(cut_theta, cut_julia, label="Julia", linewidth=2.0)
    ax.plot(cut_theta, cut_bempp, label="Bempp", linewidth=1.8, linestyle="--")
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel("Directivity (dBi)")
    ax.set_title(r"E-plane cut ($\phi \approx 0^\circ$)")
    ax.set_xlim(args.theta_min, args.theta_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(cut_theta, cut_delta, color="tab:red", linewidth=1.8)
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$\Delta D$ (dB)")
    ax.set_title(r"Cut error: Bempp$-$Julia")
    ax.set_xlim(args.theta_min, args.theta_max)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    im = ax.imshow(
        delta_grid,
        origin="lower",
        aspect="auto",
        extent=[float(phi_u.min()), float(phi_u.max()), float(theta_u.min()), float(theta_u.max())],
        cmap="coolwarm",
        vmin=-args.delta_clim,
        vmax=args.delta_clim,
    )
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title(r"2D error map $\Delta D$ (dB)")
    ax.set_ylim(args.theta_min, args.theta_max)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(r"$\Delta D$ (dB)")

    ax = axes[1, 1]
    ax.scatter(julia_vals, delta, s=10, alpha=0.35, edgecolors="none")
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Julia directivity (dBi)")
    ax.set_ylabel(r"$\Delta D$ (dB)")
    ax.set_title("Error vs Julia level")
    ax.grid(True, alpha=0.25)

    title = args.title if args.title else f"Impedance comparison: Julia={args.julia_prefix}, Bempp={args.bempp_prefix}"
    fig.suptitle(title, fontsize=12)

    out_png = data_dir / f"bempp_{args.output_prefix}_diagnostic.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    summary_txt = data_dir / f"bempp_{args.output_prefix}_diagnostic_summary.txt"
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved {out_png}")
    print(f"Saved {summary_txt}")


if __name__ == "__main__":
    main()
