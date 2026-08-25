#!/usr/bin/env python3
"""Plot diagnostic comparisons between Julia and Bempp impedance far fields."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np


def load_farfield_map(path: Path, value_key: str) -> Dict[Tuple[float, float], float]:
    data: Dict[Tuple[float, float], float] = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (OSError, UnicodeError) as exc:
        raise SystemExit(
            f"Could not read far-field CSV {path}: {exc}. Regenerate the source "
            "artifact, then rerun this plotter."
        ) from exc
    if reader.fieldnames is None or not rows:
        raise SystemExit(
            f"No far-field rows found in {path}. Regenerate the source artifact, "
            "then rerun this plotter."
        )
    for row_number, row in enumerate(rows, start=2):
        try:
            theta = float(row["theta_deg"])
            phi = float(row["phi_deg"])
            value = float(row[value_key])
        except KeyError as exc:
            raise SystemExit(
                f"Missing required column {exc.args[0]!r} in {path}. Regenerate "
                "the source artifact with the expected schema."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Invalid numeric value in {path} row {row_number}: {exc}. "
                "Regenerate the source artifact with finite samples."
            ) from exc
        if not all(math.isfinite(number) for number in (theta, phi, value)):
            raise SystemExit(
                f"Non-finite numeric value in {path} row {row_number}. Regenerate "
                "the source artifact with finite angular and directivity samples."
            )
        key = (round(theta, 6), round(phi, 6))
        if key in data:
            raise SystemExit(
                f"Duplicate rounded angular key {key} in {path}. Regenerate the "
                "source artifact with one sample per angular key."
            )
        data[key] = value
    return data


def build_common_arrays(
    julia_map: Dict[Tuple[float, float], float],
    bempp_map: Dict[Tuple[float, float], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(set(julia_map).intersection(bempp_map))
    if not keys:
        raise SystemExit(
            "No common rounded angular samples were found between the Julia and "
            "Bempp data. Regenerate both files on the same angular grid."
        )

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
    if (
        theta.ndim != 1
        or phi.ndim != 1
        or values.ndim != 1
        or theta.shape != phi.shape
        or theta.shape != values.shape
    ):
        raise ValueError(
            "theta, phi, and values arrays must be one-dimensional and have "
            "equal shapes"
        )
    theta_u = np.unique(theta)
    phi_u = np.unique(phi)
    grid = np.full((theta_u.size, phi_u.size), np.nan, dtype=float)

    theta_index = {v: i for i, v in enumerate(theta_u.tolist())}
    phi_index = {v: i for i, v in enumerate(phi_u.tolist())}

    for index in range(theta.size):
        grid[theta_index[theta[index]], phi_index[phi[index]]] = values[index]
    return theta_u, phi_u, grid


def nearest_phi_cut(
    theta: np.ndarray,
    phi: np.ndarray,
    julia_vals: np.ndarray,
    bempp_vals: np.ndarray,
    n_phi: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_phi <= 0:
        raise ValueError("n_phi must be positive")
    if (
        theta.ndim != 1
        or phi.ndim != 1
        or julia_vals.ndim != 1
        or bempp_vals.ndim != 1
        or theta.shape != phi.shape
        or theta.shape != julia_vals.shape
        or theta.shape != bempp_vals.shape
        or theta.size == 0
    ):
        raise ValueError("cut inputs must be nonempty one-dimensional arrays of equal size")
    if not all(
        np.all(np.isfinite(values))
        for values in (theta, phi, julia_vals, bempp_vals)
    ):
        raise ValueError("cut inputs must contain only finite values")
    phi_dist = np.abs(np.mod(phi + 180.0, 360.0) - 180.0)
    nearest_distance = float(np.min(phi_dist))
    mask = np.isclose(phi_dist, nearest_distance, atol=1e-9, rtol=0.0)
    selected_theta = theta[mask]
    selected_julia = julia_vals[mask]
    selected_bempp = bempp_vals[mask]
    unique_theta = np.unique(selected_theta)
    cut_julia = np.empty_like(unique_theta, dtype=float)
    cut_bempp = np.empty_like(unique_theta, dtype=float)
    for index, angle in enumerate(unique_theta):
        angle_mask = np.isclose(selected_theta, angle, atol=1e-9, rtol=0.0)
        cut_julia[index] = float(np.mean(selected_julia[angle_mask]))
        cut_bempp[index] = float(np.mean(selected_bempp[angle_mask]))
    return unique_theta, cut_julia, cut_bempp


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

    return [
        stat_line("Global", np.ones_like(delta, dtype=bool)),
        stat_line("Main-lobe (Julia >= peak-10 dB)", main_lobe_mask),
        stat_line("Deep-null (Julia <= -20 dBi)", deep_null_mask),
    ]


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
        raise SystemExit(
            f"Missing Julia far-field file: {julia_csv}. Run "
            "validation/bempp/run_impedance_case_julia_reference.jl with the same "
            "prefix, then rerun this plotter."
        )
    if not bempp_csv.exists():
        raise SystemExit(
            f"Missing Bempp far-field file: {bempp_csv}. Run "
            "validation/bempp/run_impedance_cross_validation.py with the same "
            "prefix, then rerun this plotter."
        )

    julia_map = load_farfield_map(julia_csv, "dir_julia_imp_dBi")
    bempp_map = load_farfield_map(bempp_csv, "dir_bempp_imp_dBi")

    theta, phi, julia_vals, bempp_vals = build_common_arrays(julia_map, bempp_map)
    delta = bempp_vals - julia_vals

    theta_u, phi_u, julia_grid = to_grid(theta, phi, julia_vals)
    _, _, bempp_grid = to_grid(theta, phi, bempp_vals)
    _, _, delta_grid = to_grid(theta, phi, delta)

    n_phi = phi_u.size
    cut_theta, cut_julia, cut_bempp = nearest_phi_cut(
        theta, phi, julia_vals, bempp_vals, n_phi=n_phi
    )
    cut_delta = cut_bempp - cut_julia

    summary_lines = summarize_delta(delta, julia_vals)
    for line in summary_lines:
        print(line)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required to render this diagnostic. Install the "
            "validation requirements, then rerun the plotter."
        ) from exc

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
        extent=[
            float(phi_u.min()),
            float(phi_u.max()),
            float(theta_u.min()),
            float(theta_u.max()),
        ],
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

    title = (
        args.title
        if args.title
        else f"Impedance comparison: Julia={args.julia_prefix}, Bempp={args.bempp_prefix}"
    )
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
