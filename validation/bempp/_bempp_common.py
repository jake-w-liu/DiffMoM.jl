"""Shared implementation details for the Bempp validation commands."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


class ArtifactReadError(ValueError):
    """A validation artifact is unreadable or violates its required schema."""


def add_project_root_argument(parser: argparse.ArgumentParser, source_file: str) -> None:
    """Add the repository-root option shared by validation commands."""
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(source_file).resolve().parents[2],
        help="Project root containing the validation data directory.",
    )


def _bounded_float(
    raw: str,
    *,
    lower: float | None = None,
    lower_inclusive: bool = True,
    expectation: str = "a finite number",
) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    below = lower is not None and (
        value < lower or (not lower_inclusive and value == lower)
    )
    if not math.isfinite(value) or below:
        raise argparse.ArgumentTypeError(f"expected {expectation}, got {raw!r}")
    return value


def finite_float(raw: str) -> float:
    """Parse one finite command-line number."""
    return _bounded_float(raw)


def positive_finite_float(raw: str) -> float:
    """Parse one finite, positive command-line number."""
    return _bounded_float(
        raw, lower=0.0, lower_inclusive=False,
        expectation="a finite, positive number",
    )


def nonnegative_finite_float(raw: str) -> float:
    """Parse one finite, nonnegative command-line number."""
    return _bounded_float(
        raw, lower=0.0, expectation="a finite, nonnegative number"
    )


def positive_int(raw: str) -> int:
    """Parse one positive command-line integer."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}")
    return value


def add_incident_arguments(
    parser: argparse.ArgumentParser, *, frequency_default: float = 3.0
) -> None:
    """Add the shared impedance-case frequency and incidence options."""
    parser.add_argument(
        "--freq-ghz", type=positive_finite_float, default=frequency_default
    )
    parser.add_argument("--zs-imag-ohm", type=finite_float, default=200.0)
    parser.add_argument("--theta-inc-deg", type=finite_float, default=0.0)
    parser.add_argument("--phi-inc-deg", type=finite_float, default=0.0)


def add_sampling_arguments(
    parser: argparse.ArgumentParser,
    *,
    n_theta_default: int,
    n_phi_default: int,
    mesh_mode_default: str,
    mesh_step_default: float,
) -> None:
    """Add the shared Bempp angular-sampling and mesh options."""
    parser.add_argument("--n-theta", type=positive_int, default=n_theta_default)
    parser.add_argument("--n-phi", type=positive_int, default=n_phi_default)
    parser.add_argument(
        "--mesh-mode",
        choices=["gmsh_screen", "structured"],
        default=mesh_mode_default,
    )
    parser.add_argument(
        "--nx", type=positive_int, default=12,
        help="Structured-mesh cells along x (default: %(default)s).",
    )
    parser.add_argument(
        "--ny", type=positive_int, default=12,
        help="Structured-mesh cells along y (default: %(default)s).",
    )
    parser.add_argument(
        "--mesh-step-lambda", type=positive_finite_float,
        default=mesh_step_default,
        help="Gmsh target edge length in wavelengths (default: %(default)s).",
    )


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value}")


def _reject_duplicate_json_members(
    pairs: List[Tuple[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object member {key!r}")
        result[key] = value
    return result


def _validate_json_numbers(value: Any, location: str = "$") -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"non-finite number at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_numbers(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_numbers(item, f"{location}[{index}]")


def read_json_object(path: Path) -> Dict[str, Any]:
    """Read a standard JSON object and reject every non-finite number."""
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
            object_pairs_hook=_reject_duplicate_json_members,
        )
        _validate_json_numbers(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactReadError(
            f"Could not read standard JSON from {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ArtifactReadError(f"Invalid JSON in {path}: expected an object")
    return data


def load_json_object(path: Path, *, recovery: str) -> Dict[str, Any]:
    """Read a JSON object or stop with a caller-specific recovery action."""
    try:
        return read_json_object(path)
    except ArtifactReadError as exc:
        raise SystemExit(f"{exc}. {recovery}") from exc


def require_pattern_metric(metrics: Dict[str, Any], key: str) -> float:
    """Return one finite number from a comparison report's feature object."""
    pattern = metrics.get("pattern_features")
    if not isinstance(pattern, dict):
        raise ValueError("missing object 'pattern_features'")
    value = pattern.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"pattern_features.{key} must be a finite JSON number, got {value!r}"
        )
    return float(value)


def load_csv_rows(path: Path) -> List[dict]:
    """Read a nonempty CSV artifact or stop with a recovery action."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise SystemExit(
                    f"No CSV header found in {path}. Regenerate the source artifact "
                    "with the expected schema, then rerun this command."
                )

            empty_columns = [
                position
                for position, name in enumerate(fieldnames, start=1)
                if not name.strip()
            ]
            if empty_columns:
                positions = ", ".join(str(position) for position in empty_columns)
                raise SystemExit(
                    f"Invalid CSV header in {path}: empty column name at position(s) "
                    f"{positions}. Regenerate the source artifact with nonempty, "
                    "unique column names, then rerun this command."
                )

            seen = set()
            duplicate_columns = []
            for name in fieldnames:
                if name in seen and name not in duplicate_columns:
                    duplicate_columns.append(name)
                seen.add(name)
            if duplicate_columns:
                names = ", ".join(repr(name) for name in duplicate_columns)
                raise SystemExit(
                    f"Invalid CSV header in {path}: duplicate column name(s) {names}. "
                    "Regenerate the source artifact with nonempty, unique column "
                    "names, then rerun this command."
                )

            rows = []
            for row in reader:
                if None in row:
                    extra_count = len(row[None])
                    raise SystemExit(
                        f"Invalid CSV record ending at line {reader.line_num} in "
                        f"{path}: found {extra_count} field(s) beyond the header. "
                        "Regenerate the source artifact with the expected schema, "
                        "then rerun this command."
                    )
                missing_columns = [
                    name for name in fieldnames if row[name] is None
                ]
                if missing_columns:
                    names = ", ".join(repr(name) for name in missing_columns)
                    raise SystemExit(
                        f"Invalid CSV record ending at line {reader.line_num} in "
                        f"{path}: missing value(s) for column(s) {names}. Regenerate "
                        "the source artifact with the expected schema, then rerun "
                        "this command."
                    )
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SystemExit(
            f"Could not read CSV data from {path}: {exc}. Regenerate the source "
            "artifact, then rerun this command."
        ) from exc
    if not rows:
        raise SystemExit(
            f"No CSV data rows found in {path}. Regenerate the source artifact, "
            "then rerun this command."
        )
    return rows


def load_angular_map(path: Path, value_key: str) -> Dict[Tuple[float, float], float]:
    """Load finite angular samples indexed by rounded theta and phi."""
    samples: Dict[Tuple[float, float], float] = {}
    for row_number, row in enumerate(load_csv_rows(path), start=2):
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
                "Regenerate the source artifact with finite numeric samples."
            ) from exc
        if not all(math.isfinite(number) for number in (theta, phi, value)):
            raise SystemExit(
                f"Non-finite numeric value in {path} row {row_number}. Regenerate "
                "the source artifact with finite angular and directivity samples."
            )
        key = (round(theta, 6), round(phi, 6))
        if key in samples:
            raise SystemExit(
                f"Duplicate rounded angular key {key} in {path}. Regenerate the "
                "source artifact with one sample per angular key."
            )
        samples[key] = value
    return samples


def common_angular_arrays(
    first: Dict[Tuple[float, float], float],
    second: Dict[Tuple[float, float], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align two angular maps and return their common samples."""
    keys = sorted(first.keys() & second.keys())
    if not keys:
        raise SystemExit(
            "No common rounded angular samples were found. Regenerate both "
            "far-field artifacts on the same angular grid, then rerun this command."
        )
    theta, phi = (
        np.array(component, dtype=float) for component in zip(*keys, strict=True)
    )
    return (
        theta,
        phi,
        np.array([first[key] for key in keys], dtype=float),
        np.array([second[key] for key in keys], dtype=float),
    )


def nearest_theta_stats(
    theta: np.ndarray, delta: np.ndarray, target_deg: float
) -> Dict[str, float]:
    """Summarize error at the sampled polar angle nearest a target."""
    if theta.size == 0 or theta.size != delta.size:
        raise ValueError("theta and delta must be nonempty arrays of equal length")
    unique_theta = np.unique(theta)
    nearest = unique_theta[np.argmin(np.abs(unique_theta - target_deg))]
    mask = np.isclose(theta, nearest, atol=1e-9)
    return {
        "target_theta_deg": target_deg,
        "nearest_theta_deg": float(nearest),
        "mean_abs_diff_db": float(np.mean(np.abs(delta[mask]))),
    }


def run_command(cmd: Sequence[str], cwd: Path, dry_run: bool = False) -> None:
    """Run one validation subprocess with the active environment's tools on PATH."""
    print("+", shlex.join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    _prepend_venv_bin(env)
    subprocess.run(list(cmd), cwd=str(cwd), check=True, env=env)


@dataclass(frozen=True)
class ImpedanceValidationConfig:
    """Inputs shared by the Julia and Bempp impedance validation commands."""

    freq_ghz: float
    zs_imag_ohm: float
    theta_inc_deg: float
    phi_inc_deg: float
    n_theta: int
    n_phi: int
    mesh_mode: str
    nx: int
    ny: int
    mesh_step_lambda: float

    @classmethod
    def from_namespace(cls, args: Any, **overrides: Any):
        values = {
            name: getattr(args, name)
            for name in cls.__dataclass_fields__
            if hasattr(args, name)
        }
        return cls(**(values | overrides))

    def julia_command(self, prefix: str) -> List[str]:
        return [
            "julia", "--project=.",
            "validation/bempp/run_impedance_case_julia_reference.jl",
            "--freq-ghz", str(self.freq_ghz),
            "--theta-ohm", str(self.zs_imag_ohm),
            "--theta-inc-deg", str(self.theta_inc_deg),
            "--phi-inc-deg", str(self.phi_inc_deg),
            "--n-theta", str(self.n_theta),
            "--n-phi", str(self.n_phi),
            "--output-prefix", prefix,
        ]

    def bempp_command(
        self,
        prefix: str,
        *,
        op_sign: str = "minus",
        rhs_cross: str = "e_cross_n",
        rhs_sign: float = 1.0,
        phase_sign: str = "plus",
        zs_scale: float = 1.0,
    ) -> List[str]:
        return [
            sys.executable,
            "validation/bempp/run_impedance_cross_validation.py",
            "--freq-ghz", str(self.freq_ghz),
            "--zs-imag-ohm", str(self.zs_imag_ohm),
            "--theta-inc-deg", str(self.theta_inc_deg),
            "--phi-inc-deg", str(self.phi_inc_deg),
            "--n-theta", str(self.n_theta),
            "--n-phi", str(self.n_phi),
            "--mesh-mode", self.mesh_mode,
            "--nx", str(self.nx),
            "--ny", str(self.ny),
            "--mesh-step-lambda", str(self.mesh_step_lambda),
            "--op-sign", op_sign,
            "--rhs-cross", rhs_cross,
            "--rhs-sign", str(rhs_sign),
            "--phase-sign", phase_sign,
            "--zs-scale", str(zs_scale),
            "--output-prefix", prefix,
        ]


def impedance_comparison_command(
    prefix: str,
    *,
    target_theta_deg: float = 30.0,
    julia_prefix: str | None = None,
    bempp_prefix: str | None = None,
) -> List[str]:
    """Build the shared far-field comparison command."""
    command = [
        sys.executable,
        "validation/bempp/compare_impedance_to_julia.py",
        "--output-prefix", prefix,
        "--target-theta-deg", str(target_theta_deg),
    ]
    if julia_prefix is not None:
        command.extend(("--julia-prefix", julia_prefix))
    if bempp_prefix is not None:
        command.extend(("--bempp-prefix", bempp_prefix))
    return command


def _prepend_venv_bin(env: Dict[str, str]) -> None:
    venv_bin = str(Path(sys.executable).parent)
    path = env.get("PATH", "")
    entries = path.split(os.pathsep) if path else []
    if venv_bin not in entries:
        env["PATH"] = venv_bin + (os.pathsep + path if path else "")


def load_bempp():
    """Load Bempp only when a solver run, rather than CLI help, needs it."""
    _prepend_venv_bin(os.environ)
    try:
        import bempp_cl.api as bempp
        from bempp_cl.api.linalg import lu
    except ImportError as exc:
        raise SystemExit(
            f"Could not import Bempp-cl or one of its dependencies: {exc}. Run "
            "python -m pip install bempp-cl, then rerun the command."
        ) from exc
    return bempp, lu


def spherical_sampling_grid(
    n_theta: int, n_phi: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the centered spherical grid shared by the Bempp solvers."""
    if n_theta <= 0 or n_phi <= 0:
        raise ValueError("spherical sample counts must be positive")
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


def _validate_farfield_arrays(
    theta: np.ndarray, phi: np.ndarray, values: np.ndarray
) -> None:
    if (
        theta.ndim != 1
        or phi.ndim != 1
        or values.ndim != 1
        or theta.shape != phi.shape
        or theta.shape != values.shape
        or theta.size == 0
    ):
        raise ValueError(
            "Far-field theta, phi, and value arrays must be nonempty, "
            "one-dimensional, and have equal shapes."
        )
    if not all(np.all(np.isfinite(array)) for array in (theta, phi, values)):
        raise ValueError("Far-field arrays must contain only finite values.")


def write_farfield_csv(
    path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    value_header: str,
) -> None:
    """Write one finite far-field sample per CSV row."""
    _validate_farfield_arrays(theta, phi, values)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["theta_deg", "phi_deg", value_header])
        for index in range(theta.size):
            writer.writerow(
                [
                    float(np.rad2deg(theta[index])),
                    float(np.rad2deg(phi[index])),
                    float(values[index]),
                ]
            )


def write_phi_zero_cut_csv(
    path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    n_phi: int,
    value_header: str,
) -> None:
    """Write samples from the grid columns nearest zero azimuth."""
    _validate_farfield_arrays(theta, phi, values)
    if n_phi <= 0:
        raise ValueError("azimuth sample count must be positive")
    half_bin = np.pi / n_phi
    phi_dist = np.minimum(phi, 2.0 * np.pi - phi)
    indices = np.flatnonzero(phi_dist <= half_bin + 1e-12)
    order = indices[np.argsort(theta[indices])]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["theta_deg", value_header])
        for index in order:
            writer.writerow(
                [float(np.rad2deg(theta[index])), float(values[index])]
            )
