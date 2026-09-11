"""Compare absolute complex PEC far fields on an explicitly supplied mesh.

The incident wave, mesh, and look directions are transferred without fitting
amplitude or phase. Bempp uses exp(-i*omega*t); its scattered field is conjugated
to compare with DiffMoM's exp(+i*omega*t) convention. The boundary trace and
representation signs follow the Bempp Maxwell screen tutorial.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from _bempp_common import (
    add_project_root_argument,
    load_bempp,
    nonnegative_finite_float,
    positive_finite_float,
    positive_int,
    read_json_object,
)


def _matrix(raw, width, label, *, integer=False):
    values = np.asarray(raw)
    kinds = "iu" if integer else "iuf"
    if values.ndim != 2 or values.shape[1] != width or values.shape[0] == 0:
        raise ValueError(f"{label} must be a nonempty array with {width} columns")
    if values.dtype.kind not in kinds or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} has invalid or non-finite numeric entries")
    return np.asarray(values, dtype=np.int64 if integer else np.float64)


def read_case(path, max_dofs):
    data = read_json_object(path)
    if data.get("schema_version") != 1 or data.get("convention") != "exp(+i*omega*t)":
        raise ValueError("unsupported PEC case schema or phasor convention")
    vertices = _matrix(data["vertices_m"], 3, "vertices_m")
    triangles = _matrix(data["triangles_zero_based"], 3, "triangles_zero_based", integer=True)
    if vertices.shape[0] > 3 * max_dofs or triangles.shape[0] > 2 * max_dofs + 4:
        raise ValueError("input mesh exceeds the dense reference size limit")
    if np.min(triangles) < 0 or np.max(triangles) >= len(vertices):
        raise ValueError("triangle vertex index is out of bounds")
    looks = _matrix(data["looks_unit"], 3, "looks_unit")
    wavevector = _matrix([data["incident_wavevector_rad_m"]], 3, "wavevector")[0]
    polarization = _matrix([data["incident_polarization"]], 3, "polarization")[0]
    scalars = []
    for key in ("frequency_hz", "c0_m_s", "incident_amplitude_v_m"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
        scalars.append(float(value))
    frequency, speed, amplitude = scalars
    k = 2 * math.pi * frequency / speed
    if not math.isfinite(k) or not np.isclose(np.linalg.norm(wavevector), k, rtol=1e-12, atol=0):
        raise ValueError("wavevector does not match frequency and wave speed")
    if not np.isclose(np.linalg.norm(polarization), 1.0, rtol=0, atol=1e-12):
        raise ValueError("polarization must be a unit vector")
    if abs(np.dot(polarization, wavevector / k)) > 1e-12:
        raise ValueError("polarization must be transverse to the incident wave")
    if not np.allclose(np.linalg.norm(looks, axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("look directions must be unit vectors")
    theta = np.asarray(data["theta_rad"], dtype=float)
    phi = np.asarray(data["phi_rad"], dtype=float)
    if theta.shape != (len(looks),) or phi.shape != theta.shape:
        raise ValueError("look angles have the wrong shape")
    reconstructed = np.column_stack((np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi), np.cos(theta)))
    if not np.allclose(reconstructed, looks, rtol=0, atol=1e-12):
        raise ValueError("look angles and direction vectors disagree")
    reference = _matrix(data["field_real"], 2, "field_real") + 1j * _matrix(
        data["field_imag"], 2, "field_imag")
    if reference.shape != (len(looks), 2):
        raise ValueError("reference field has the wrong number of looks")
    return data, vertices, triangles, looks, wavevector, polarization, k, amplitude, theta, phi, reference


def solve_case(args):
    case = read_case(args.input, args.max_dofs)
    data, vertices, triangles, looks, wavevector, polarization, k, amplitude, theta, phi, reference = case
    bempp, lu = load_bempp()
    bempp.DEFAULT_DEVICE_INTERFACE = "numba"
    bempp.DEFAULT_PRECISION = "double"
    parameters = bempp.DefaultParameters()
    parameters.quadrature.regular = args.regular_order
    parameters.quadrature.singular = args.singular_order
    started = time.perf_counter()
    grid = bempp.Grid(vertices.T, triangles.T)
    space = bempp.function_space(grid, "RWG", 0)
    dual = bempp.function_space(grid, "SNC", 0)
    dofs = space.global_dof_count
    if dofs > args.max_dofs or dofs != data["dof_count"]:
        raise ValueError("Bempp RWG count exceeds the limit or differs from the supplied discretization")

    @bempp.complex_callable
    def tangential_trace(x, n, domain_index, result):
        field = amplitude * polarization * np.exp(1j * np.dot(wavevector, x))
        result[:] = np.cross(field, n)

    rhs = bempp.GridFunction(space, dual_space=dual, fun=tangential_trace, parameters=parameters)
    operator = bempp.operators.boundary.maxwell.electric_field(
        space, space, dual, k, parameters=parameters, device_interface="numba", precision="double")
    matrix = operator.weak_form().A
    projected_rhs = rhs.projections(dual)
    assembly_s = time.perf_counter() - started
    solved = time.perf_counter()
    current = lu(operator, rhs)
    solve_s = time.perf_counter() - solved
    coefficients = current.coefficients
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("Bempp returned non-finite coefficients")
    # A scalar contraction avoids spurious floating-status exceptions from
    # some BLAS matmul builds without disabling strict error handling.
    residual_norm = np.linalg.norm(
        np.einsum("ij,j->i", matrix, coefficients, optimize=False) - projected_rhs)
    rhs_norm = np.linalg.norm(projected_rhs)
    relative_residual = float(residual_norm / rhs_norm) if rhs_norm else float(residual_norm)
    if not math.isfinite(relative_residual) or relative_residual > 1e-10:
        raise RuntimeError(f"Bempp linear residual is {relative_residual}, exceeding 1e-10")

    evaluated = time.perf_counter()
    potential = bempp.operators.far_field.maxwell.electric_field(
        space, looks.T, k, parameters=parameters, device_interface="numba", precision="double")
    scattered = -(potential * current)
    if scattered.shape != (3, len(looks)) or not np.all(np.isfinite(scattered)):
        raise RuntimeError("Bempp returned an invalid far field")
    theta_hat = np.column_stack((np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi), -np.sin(theta)))
    phi_hat = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    transverse = np.column_stack((np.sum(theta_hat * scattered.T, axis=1),
        np.sum(phi_hat * scattered.T, axis=1)))
    converted = np.conjugate(transverse)
    farfield_s = time.perf_counter() - evaluated
    differences = np.linalg.norm(converted - reference, axis=1)
    difference_norm = float(np.linalg.norm(converted - reference))
    reference_norm = float(np.linalg.norm(reference))
    tolerance = args.relative_field_tolerance * reference_norm + args.absolute_field_tolerance
    passed = difference_norm <= tolerance
    report = {
        "schema_version": 1, "case_id": data["case_id"], "passed": passed,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "julia_source_sha256": data["julia_source_sha256"],
        "bempp_version": bempp.__version__, "device_interface": "numba", "dof_count": dofs,
        "python_version": sys.version,
        "versions": {name: version(name) for name in ("bempp-cl", "numpy", "scipy", "numba")},
        "precision": "double",
        "regular_order": args.regular_order, "singular_order": args.singular_order,
        "relative_linear_residual": relative_residual,
        "field_difference_norm": difference_norm, "reference_field_norm": reference_norm,
        "relative_field_tolerance": args.relative_field_tolerance,
        "absolute_field_tolerance": args.absolute_field_tolerance,
        "per_look_absolute_field_difference": differences.tolist(),
        "bempp_field_real": transverse.real.tolist(), "bempp_field_imag": transverse.imag.tolist(),
        "julia_convention_field_real": converted.real.tolist(),
        "julia_convention_field_imag": converted.imag.tolist(),
        "rcs_m2": (4 * math.pi * np.sum(np.abs(converted / amplitude) ** 2, axis=1)).tolist(),
        "costs": {"assembly_s": assembly_s, "solve_s": solve_s, "farfield_s": farfield_s},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(f"{data['case_id']}: {'PASS' if passed else 'FAIL'}; field difference {difference_norm:.6g}; limit {tolerance:.6g}")
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_root_argument(parser, __file__)
    parser.add_argument("input", type=Path, help="Case JSON, relative to --project-root unless absolute.")
    parser.add_argument("output", type=Path, help="Report JSON, relative to --project-root unless absolute.")
    parser.add_argument("--relative-field-tolerance", type=positive_finite_float, required=True)
    parser.add_argument("--absolute-field-tolerance", type=nonnegative_finite_float, default=1e-10)
    parser.add_argument("--regular-order", type=positive_int, default=4)
    parser.add_argument("--singular-order", type=positive_int, default=6)
    # Dense cross-solver comparisons are not intended as large production solves.
    parser.add_argument("--max-dofs", type=positive_int, default=3000)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        parser.error("--project-root must name an existing directory")
    args.project_root = root
    for name in ("input", "output"):
        path = getattr(args, name).expanduser()
        setattr(args, name, (path if path.is_absolute() else root / path).resolve())
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        return solve_case(args)


if __name__ == "__main__":
    raise SystemExit(main())
