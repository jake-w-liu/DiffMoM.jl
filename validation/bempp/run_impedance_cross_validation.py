#!/usr/bin/env python3
"""Run a Bempp-cl impedance-loaded far-field reference for cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

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


def write_structured_plate_msh(path: Path, side: float, nx: int, ny: int) -> None:
    """Write a Julia-matching structured rectangular plate mesh in Gmsh v2 format."""
    dx = side / nx
    dy = side / ny

    nodes = []
    node_id = 0
    for jy in range(ny + 1):
        y = -0.5 * side + jy * dy
        for jx in range(nx + 1):
            x = -0.5 * side + jx * dx
            node_id += 1
            nodes.append((node_id, x, y, 0.0))

    def vid(ix: int, iy: int) -> int:
        return iy * (nx + 1) + ix + 1

    elems = []
    elem_id = 0
    for jy in range(ny):
        for jx in range(nx):
            v1 = vid(jx, jy)
            v2 = vid(jx + 1, jy)
            v3 = vid(jx + 1, jy + 1)
            v4 = vid(jx, jy + 1)

            elem_id += 1
            elems.append((elem_id, v1, v2, v3))
            elem_id += 1
            elems.append((elem_id, v1, v3, v4))

    with path.open("w", encoding="utf-8") as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for nid, x, y, z in nodes:
            f.write(f"{nid} {x:.16e} {y:.16e} {z:.16e}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{len(elems)}\n")
        for eid, n1, n2, n3 in elems:
            f.write(f"{eid} 2 0 {n1} {n2} {n3}\n")
        f.write("$EndElements\n")


def run_bempp_impedance(
    freq_hz: float,
    aperture_lambda: float,
    mesh_step_lambda: float,
    mesh_mode: str,
    nx: int,
    ny: int,
    zs_imag_ohm: float,
    theta_inc_deg: float,
    phi_inc_deg: float,
    n_theta: int,
    n_phi: int,
    op_sign: str,
    rhs_cross: str,
    rhs_sign: float,
    phase_sign: str,
    zs_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, Dict[str, np.ndarray]]:
    lambda0 = C0 / freq_hz
    k0 = 2.0 * np.pi / lambda0
    side = aperture_lambda * lambda0
    mesh_h = mesh_step_lambda * lambda0

    if mesh_mode == "gmsh_screen":
        corners = np.array(
            [
                [-side / 2.0, -side / 2.0, 0.0],
                [side / 2.0, -side / 2.0, 0.0],
                [side / 2.0, side / 2.0, 0.0],
                [-side / 2.0, side / 2.0, 0.0],
            ]
        )
        grid = bempp.shapes.screen(corners=corners, h=mesh_h)
        mesh_descriptor = {
            "mesh_mode": mesh_mode,
            "mesh_step_lambda": mesh_step_lambda,
            "mesh_step_m": mesh_h,
        }
    elif mesh_mode == "structured":
        mesh_file = Path(__file__).resolve().parents[2] / "data" / f"structured_plate_nx{nx}_ny{ny}.msh"
        mesh_file.parent.mkdir(parents=True, exist_ok=True)
        write_structured_plate_msh(mesh_file, side=side, nx=nx, ny=ny)
        grid = bempp.import_grid(str(mesh_file))
        mesh_descriptor = {
            "mesh_mode": mesh_mode,
            "structured_nx": nx,
            "structured_ny": ny,
            "mesh_file": str(mesh_file),
            "mesh_step_lambda": side / (nx * lambda0),
            "mesh_step_m": side / nx,
        }
    else:
        raise ValueError(f"Unknown mesh_mode: {mesh_mode}")

    rwg = bempp.function_space(grid, "RWG", 0)
    snc = bempp.function_space(grid, "SNC", 0)

    electric = bempp.operators.boundary.maxwell.electric_field(rwg, rwg, snc, k0)
    identity = bempp.operators.boundary.sparse.identity(rwg, rwg, snc)
    zs = 1j * zs_imag_ohm * zs_scale
    if op_sign == "minus":
        op = electric - zs * identity
    elif op_sign == "plus":
        op = electric + zs * identity
    else:
        raise ValueError(f"Unknown op_sign: {op_sign}")

    theta_inc = np.deg2rad(theta_inc_deg)
    phi_inc = np.deg2rad(phi_inc_deg)

    direction = np.array(
        [
            np.sin(theta_inc) * np.cos(phi_inc),
            np.sin(theta_inc) * np.sin(phi_inc),
            -np.cos(theta_inc),
        ],
        dtype=float,
    )
    # Theta polarization unit vector, perpendicular to propagation direction.
    polarization = np.array(
        [
            np.cos(theta_inc) * np.cos(phi_inc),
            np.cos(theta_inc) * np.sin(phi_inc),
            np.sin(theta_inc),
        ],
        dtype=np.complex128,
    )

    @bempp.complex_callable
    def tangential_trace(x, n, domain_index, result):
        if phase_sign == "plus":
            phase = np.exp(1j * k0 * np.dot(direction, x))
        elif phase_sign == "minus":
            phase = np.exp(-1j * k0 * np.dot(direction, x))
        else:
            raise ValueError(f"Unknown phase_sign: {phase_sign}")
        e_inc = polarization * phase
        if rhs_cross == "e_cross_n":
            trace = np.cross(e_inc, n)
        elif rhs_cross == "n_cross_e":
            trace = np.cross(n, e_inc)
        else:
            raise ValueError(f"Unknown rhs_cross: {rhs_cross}")
        result[:] = rhs_sign * trace

    rhs = bempp.GridFunction(rwg, fun=tangential_trace, dual_space=snc)
    surface_current = lu(op, rhs)
    residual = op * surface_current - rhs
    rhs_l2 = float(rhs.l2_norm())
    residual_l2 = float(residual.l2_norm())
    residual_field_rel = residual_l2 / max(rhs_l2, 1e-30)

    rhs_proj = np.asarray(rhs.projections(snc))
    lhs_proj = np.asarray((op * surface_current).projections(snc))
    residual_proj = lhs_proj - rhs_proj
    rhs_proj_l2 = float(np.linalg.norm(rhs_proj))
    residual_proj_l2 = float(np.linalg.norm(residual_proj))
    residual_proj_rel = residual_proj_l2 / max(rhs_proj_l2, 1e-30)

    op_mat = op.weak_form().A
    x_coeff = surface_current.coefficients
    residual_coeff = op_mat.dot(x_coeff) - rhs_proj
    residual_coeff_l2 = float(np.linalg.norm(residual_coeff))
    residual_coeff_rel = residual_coeff_l2 / max(rhs_proj_l2, 1e-30)

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
        **mesh_descriptor,
        "zs_imag_ohm": zs_imag_ohm,
        "zs_scale": zs_scale,
        "zs_effective_imag_ohm": zs_imag_ohm * zs_scale,
        "theta_inc_deg": theta_inc_deg,
        "phi_inc_deg": phi_inc_deg,
        "op_sign": op_sign,
        "rhs_cross": rhs_cross,
        "rhs_sign": rhs_sign,
        "phase_sign": phase_sign,
        "n_theta": n_theta,
        "n_phi": n_phi,
        "dtheta_deg": float(np.rad2deg(steps[0])),
        "dphi_deg": float(np.rad2deg(steps[1])),
        "num_points": int(theta_flat.size),
        "num_vertices": int(grid.number_of_vertices),
        "num_elements": int(grid.number_of_elements),
        "rhs_l2_norm": rhs_l2,
        "rhs_projection_l2_norm": rhs_proj_l2,
        "solve_residual_field_l2_abs": residual_l2,
        "solve_residual_field_l2_rel": residual_field_rel,
        "solve_residual_projection_l2_abs": residual_proj_l2,
        "solve_residual_projection_l2_rel": residual_proj_rel,
        "solve_residual_coeff_l2_abs": residual_coeff_l2,
        "solve_residual_coeff_l2_rel": residual_coeff_rel,
        "solve_residual_l2_abs": residual_coeff_l2,
        "solve_residual_l2_rel": residual_coeff_rel,
        "surface_current_l2_norm": float(surface_current.l2_norm()),
        "surface_current_coeff_l2_norm": float(np.linalg.norm(surface_current.coefficients)),
    }

    centers = np.asarray(grid.centroids)
    if centers.ndim != 2:
        raise ValueError(f"Unexpected centroid shape: {centers.shape}")
    if centers.shape[0] != 3 and centers.shape[1] == 3:
        centers = centers.T
    if centers.shape[0] != 3:
        raise ValueError(f"Unexpected centroid shape after normalization: {centers.shape}")

    currents = np.asarray(surface_current.evaluate_on_element_centers())
    if currents.ndim != 2:
        raise ValueError(f"Unexpected current shape: {currents.shape}")
    if currents.shape[0] != 3 and currents.shape[1] == 3:
        currents = currents.T
    if currents.shape[0] != 3:
        raise ValueError(f"Unexpected current shape after normalization: {currents.shape}")
    current_mag = np.sqrt(np.sum(np.abs(currents) ** 2, axis=0))
    pol_tan = polarization.copy()
    pol_tan[2] = 0.0
    pol_tan_norm = np.linalg.norm(pol_tan)
    if pol_tan_norm < 1e-12:
        pol_tan = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    else:
        pol_tan = pol_tan / pol_tan_norm
    proj = np.einsum("i,ij->j", np.conjugate(pol_tan), currents)
    phase_deg = np.degrees(np.angle(proj))
    phase_deg[current_mag < 1e-15] = np.nan

    current_data = {
        "element_index": np.arange(currents.shape[1], dtype=int) + 1,
        "x_m": centers[0, :],
        "y_m": centers[1, :],
        "z_m": centers[2, :],
        "Jx_re": np.real(currents[0, :]),
        "Jx_im": np.imag(currents[0, :]),
        "Jy_re": np.real(currents[1, :]),
        "Jy_im": np.imag(currents[1, :]),
        "Jz_re": np.real(currents[2, :]),
        "Jz_im": np.imag(currents[2, :]),
        "J_mag": current_mag,
        "J_phase_deg": phase_deg,
    }
    return theta_flat, phi_flat, dir_dbi, metadata, current_data


def write_farfield_csv(path: Path, theta: np.ndarray, phi: np.ndarray, dir_dbi: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "phi_deg", "dir_bempp_imp_dBi"])
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
        writer.writerow(["theta_deg", "dir_bempp_imp_dBi"])
        for i in order:
            writer.writerow([float(np.rad2deg(theta[i])), float(dir_dbi[i])])


def write_element_currents_csv(path: Path, current_data: Dict[str, np.ndarray]) -> None:
    keys = [
        "element_index",
        "x_m",
        "y_m",
        "z_m",
        "Jx_re",
        "Jx_im",
        "Jy_re",
        "Jy_im",
        "Jz_re",
        "Jz_im",
        "J_mag",
        "J_phase_deg",
    ]
    num = len(current_data["element_index"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(num):
            writer.writerow([current_data[k][i] for k in keys])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq-ghz", type=float, default=3.0)
    parser.add_argument("--aperture-lambda", type=float, default=4.0)
    parser.add_argument("--mesh-step-lambda", type=float, default=1.0 / 3.0)
    parser.add_argument("--mesh-mode", choices=["gmsh_screen", "structured"], default="gmsh_screen")
    parser.add_argument("--nx", type=int, default=12, help="Structured mesh x-cells when --mesh-mode structured")
    parser.add_argument("--ny", type=int, default=12, help="Structured mesh y-cells when --mesh-mode structured")
    parser.add_argument("--zs-imag-ohm", type=float, default=200.0)
    parser.add_argument("--theta-inc-deg", type=float, default=0.0)
    parser.add_argument("--phi-inc-deg", type=float, default=0.0)
    parser.add_argument("--n-theta", type=int, default=180)
    parser.add_argument("--n-phi", type=int, default=72)
    parser.add_argument("--op-sign", choices=["minus", "plus"], default="minus")
    parser.add_argument("--rhs-cross", choices=["e_cross_n", "n_cross_e"], default="e_cross_n")
    parser.add_argument("--rhs-sign", type=float, default=1.0)
    parser.add_argument("--phase-sign", choices=["plus", "minus"], default="plus")
    parser.add_argument("--zs-scale", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=str, default="impedance")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/.",
    )
    args = parser.parse_args()

    data_dir = args.project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    theta, phi, dir_dbi, meta, current_data = run_bempp_impedance(
        freq_hz=args.freq_ghz * 1e9,
        aperture_lambda=args.aperture_lambda,
        mesh_step_lambda=args.mesh_step_lambda,
        mesh_mode=args.mesh_mode,
        nx=args.nx,
        ny=args.ny,
        zs_imag_ohm=args.zs_imag_ohm,
        theta_inc_deg=args.theta_inc_deg,
        phi_inc_deg=args.phi_inc_deg,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        op_sign=args.op_sign,
        rhs_cross=args.rhs_cross,
        rhs_sign=args.rhs_sign,
        phase_sign=args.phase_sign,
        zs_scale=args.zs_scale,
    )

    farfield_csv = data_dir / f"bempp_{args.output_prefix}_farfield.csv"
    cut_csv = data_dir / f"bempp_{args.output_prefix}_cut_phi0.csv"
    meta_json = data_dir / f"bempp_{args.output_prefix}_metadata.json"
    current_csv = data_dir / f"bempp_{args.output_prefix}_element_currents.csv"

    write_farfield_csv(farfield_csv, theta, phi, dir_dbi)
    write_phi0_cut_csv(cut_csv, theta, phi, dir_dbi, args.n_phi)
    write_element_currents_csv(current_csv, current_data)
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {farfield_csv}")
    print(f"Saved {cut_csv}")
    print(f"Saved {current_csv}")
    print(f"Saved {meta_json}")
    print(
        "Note: compare against Julia with "
        "python validation/bempp/compare_impedance_to_julia.py"
    )


if __name__ == "__main__":
    main()
