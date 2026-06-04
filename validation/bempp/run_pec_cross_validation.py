#!/usr/bin/env python3
"""Run a Bempp-cl PEC far-field reference for cross-validation.

This script reproduces the PEC baseline used in the Julia project:
- Frequency: 3 GHz (default)
- Aperture: 4 lambda x 4 lambda square plate
- Illumination: normally incident x-polarized plane wave
- Sampling grid: theta/phi centers matching make_sph_grid(180, 72)

Outputs:
- data/bempp_pec_farfield.csv
- data/bempp_pec_cut_phi0.csv
- data/bempp_pec_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Tuple

import numpy as np


def ensure_gmsh_on_path() -> None:
    venv_bin = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    if venv_bin not in entries:
        os.environ["PATH"] = venv_bin + os.pathsep + path


ensure_gmsh_on_path()


try:
    import bempp_cl.api as bempp
    from bempp_cl.api.linalg import lu
except ImportError as exc:
    raise SystemExit(
        "Bempp-cl is required. Install it first, then rerun this script.\n"
        "Example: pip install bempp-cl"
    ) from exc


C0 = 299_792_458.0


def spherical_sampling_grid(
    n_theta: int, n_phi: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtheta = np.pi / n_theta
    dphi = 2.0 * np.pi / n_phi

    theta = (np.arange(n_theta) + 0.5) * dtheta
    phi = (np.arange(n_phi) + 0.5) * dphi

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()

    directions = np.vstack(
        (
            np.sin(theta_flat) * np.cos(phi_flat),
            np.sin(theta_flat) * np.sin(phi_flat),
            np.cos(theta_flat),
        )
    )
    weights = np.sin(theta_flat) * dtheta * dphi
    return theta_flat, phi_flat, directions, weights, np.array([dtheta, dphi])


def run_bempp_pec(
    freq_hz: float,
    aperture_lambda: float,
    mesh_step_lambda: float,
    n_theta: int,
    n_phi: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    lambda0 = C0 / freq_hz
    k0 = 2.0 * np.pi / lambda0
    side = aperture_lambda * lambda0
    mesh_h = mesh_step_lambda * lambda0

    corners = np.array(
        [
            [-side / 2.0, -side / 2.0, 0.0],
            [side / 2.0, -side / 2.0, 0.0],
            [side / 2.0, side / 2.0, 0.0],
            [-side / 2.0, side / 2.0, 0.0],
        ]
    )
    grid = bempp.shapes.screen(corners=corners, h=mesh_h)

    rwg = bempp.function_space(grid, "RWG", 0)
    snc = bempp.function_space(grid, "SNC", 0)
    electric = bempp.operators.boundary.maxwell.electric_field(rwg, rwg, snc, k0)

    direction = np.array([0.0, 0.0, -1.0])
    polarization = np.array([1.0, 0.0, 0.0], dtype=np.complex128)

    @bempp.complex_callable
    def tangential_trace(x, n, domain_index, result):
        phase = np.exp(1j * k0 * np.dot(direction, x))
        e_inc = polarization * phase
        result[:] = np.cross(e_inc, n)

    rhs = bempp.GridFunction(rwg, fun=tangential_trace, dual_space=snc)
    surface_current = lu(electric, rhs)

    theta_flat, phi_flat, directions, weights, steps = spherical_sampling_grid(
        n_theta=n_theta, n_phi=n_phi
    )
    far_field = bempp.operators.far_field.maxwell.electric_field(rwg, directions, k0)
    e_ff = -(far_field * surface_current)

    power = np.real(np.sum(np.conjugate(e_ff) * e_ff, axis=0))
    power = np.maximum(power, 1e-30)
    total = float(np.sum(power * weights))

    directivity = 4.0 * np.pi * power / total
    dir_dbi = 10.0 * np.log10(np.maximum(directivity, 1e-30))

    metadata = {
        "frequency_hz": freq_hz,
        "lambda_m": lambda0,
        "wavenumber_rad_per_m": k0,
        "aperture_lambda": aperture_lambda,
        "aperture_side_m": side,
        "mesh_step_lambda": mesh_step_lambda,
        "mesh_step_m": mesh_h,
        "n_theta": n_theta,
        "n_phi": n_phi,
        "dtheta_deg": float(np.rad2deg(steps[0])),
        "dphi_deg": float(np.rad2deg(steps[1])),
        "num_points": int(theta_flat.size),
        "num_vertices": int(grid.number_of_vertices),
        "num_elements": int(grid.number_of_elements),
    }

    return theta_flat, phi_flat, dir_dbi, e_ff, metadata


def write_farfield_csv(path: Path, theta: np.ndarray, phi: np.ndarray, dir_dbi: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "phi_deg", "dir_bempp_dBi"])
        for t, p, d in zip(theta, phi, dir_dbi):
            writer.writerow([float(np.rad2deg(t)), float(np.rad2deg(p)), float(d)])


def write_phi0_cut_csv(
    path: Path, theta: np.ndarray, phi: np.ndarray, dir_dbi: np.ndarray, n_phi: int
) -> None:
    half_bin = 0.5 * (2.0 * np.pi / n_phi)
    phi_dist = np.minimum(phi, 2.0 * np.pi - phi)
    indices = np.where(phi_dist <= half_bin + 1e-12)[0]
    order = indices[np.argsort(theta[indices])]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "dir_bempp_dBi"])
        for i in order:
            writer.writerow([float(np.rad2deg(theta[i])), float(dir_dbi[i])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq-ghz", type=float, default=3.0)
    parser.add_argument("--aperture-lambda", type=float, default=4.0)
    parser.add_argument("--mesh-step-lambda", type=float, default=1.0 / 3.0)
    parser.add_argument("--n-theta", type=int, default=180)
    parser.add_argument("--n-phi", type=int, default=72)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/.",
    )
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    theta, phi, dir_dbi, _, meta = run_bempp_pec(
        freq_hz=args.freq_ghz * 1e9,
        aperture_lambda=args.aperture_lambda,
        mesh_step_lambda=args.mesh_step_lambda,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
    )

    farfield_csv = data_dir / "bempp_pec_farfield.csv"
    cut_csv = data_dir / "bempp_pec_cut_phi0.csv"
    meta_json = data_dir / "bempp_pec_metadata.json"

    write_farfield_csv(farfield_csv, theta, phi, dir_dbi)
    write_phi0_cut_csv(cut_csv, theta, phi, dir_dbi, args.n_phi)
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {farfield_csv}")
    print(f"Saved {cut_csv}")
    print(f"Saved {meta_json}")
    print(
        "Note: compare against Julia with "
        "python validation/bempp/compare_pec_to_julia.py"
    )


if __name__ == "__main__":
    main()
