"""Shared implementation details for the Meep validation commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, List, Sequence


JULIA_ARTIFACT_IDENTITY_FIELDS = (
    "output_prefix",
    "periodic_bc_model",
    "frequency_ghz",
    "lambda_m",
    "dx_cell_m",
    "dy_cell_m",
    "dx_lambda",
    "dy_lambda",
    "nx",
    "ny",
    "slot_wx_frac",
    "slot_wy_frac",
    "metal_fill_fraction",
)


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
    upper: float | None = None,
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
    if not math.isfinite(value) or below or (upper is not None and value > upper):
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


def unit_interval_float(raw: str) -> float:
    """Parse one finite command-line number in the closed unit interval."""
    return _bounded_float(
        raw, lower=0.0, upper=1.0,
        expectation="a finite value between 0 and 1",
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


def parse_unit_interval_list(raw: str) -> List[float]:
    """Parse a nonempty comma-separated list of finite unit-interval values."""
    values: List[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            values.append(unit_interval_float(stripped))
        except argparse.ArgumentTypeError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid comma-separated value {stripped!r}: {exc}"
            ) from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def add_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    resolution_default: int,
    after_sources_default: float,
) -> None:
    """Add the runtime controls shared by Meep validation commands."""
    parser.add_argument(
        "--resolution", type=positive_int, default=resolution_default,
        help="Pixels per wavelength (default: %(default)s).",
    )
    parser.add_argument("--pml-lambda", type=positive_finite_float, default=1.0)
    parser.add_argument("--sz-lambda", type=positive_finite_float, default=6.0)
    parser.add_argument(
        "--metal-thickness-lambda", type=positive_finite_float, default=0.03
    )
    parser.add_argument(
        "--source-offset-lambda", type=nonnegative_finite_float, default=0.35
    )
    parser.add_argument(
        "--refl-offset-lambda", type=nonnegative_finite_float, default=0.25
    )
    parser.add_argument(
        "--tran-offset-lambda", type=nonnegative_finite_float, default=0.35
    )
    parser.add_argument("--fwidth", type=positive_finite_float, default=0.2)
    parser.add_argument(
        "--after-sources-time", type=positive_finite_float,
        default=after_sources_default,
    )


def validate_runtime_geometry(parser: argparse.ArgumentParser, args: Any) -> None:
    """Reject source or monitor planes that enter the PML layers."""
    lower_z = -0.5 * args.sz_lambda + args.pml_lambda
    upper_z = 0.5 * args.sz_lambda - args.pml_lambda
    source_z = upper_z - args.source_offset_lambda
    reflection_z = source_z - args.refl_offset_lambda
    transmission_z = lower_z + args.tran_offset_lambda
    if lower_z >= upper_z:
        parser.error("--sz-lambda must exceed twice --pml-lambda")
    if not lower_z <= reflection_z <= source_z <= upper_z:
        parser.error(
            "--source-offset-lambda and --refl-offset-lambda must keep both "
            "planes between the PML layers"
        )
    if not lower_z <= transmission_z <= upper_z:
        parser.error(
            "--tran-offset-lambda must keep the transmission plane between the PML layers"
        )
    if reflection_z <= transmission_z:
        parser.error("the reflection plane must be above the transmission plane")


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value}")


def _reject_duplicate_json_members(
    pairs: List[tuple[str, Any]],
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


def load_json_object(path: Path, *, recovery: str) -> Dict[str, Any]:
    """Read a finite standard JSON object or stop with a recovery action."""
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
            object_pairs_hook=_reject_duplicate_json_members,
        )
        _validate_json_numbers(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(
            f"Could not read standard JSON from {path}: {exc}. {recovery}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid JSON in {path}: expected an object. {recovery}")
    return data


def sha256_file(path: Path, *, recovery: str) -> str:
    """Return the SHA-256 digest of one artifact or stop with recovery text."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SystemExit(f"Could not hash artifact {path}: {exc}. {recovery}") from exc
    return digest.hexdigest()


def validate_julia_artifact_pair(
    requested_prefix: str,
    geometry_path: Path,
    geometry: Dict[str, Any],
    reference_path: Path,
    reference: Dict[str, Any],
) -> None:
    """Require geometry and reference JSON to describe one exact Julia case."""
    for field in JULIA_ARTIFACT_IDENTITY_FIELDS:
        if field not in geometry:
            raise SystemExit(
                f"Missing identity field {field!r} in {geometry_path}. "
                "Regenerate both Julia artifacts with the same output prefix."
            )
        if field not in reference:
            raise SystemExit(
                f"Missing identity field {field!r} in {reference_path}. "
                "Regenerate both Julia artifacts with the same output prefix."
            )
        geometry_value = geometry[field]
        reference_value = reference[field]
        if type(geometry_value) is not type(reference_value) or (
            geometry_value != reference_value
        ):
            raise SystemExit(
                f"Julia artifact identity mismatch for {field!r}: "
                f"{geometry_path} has {geometry_value!r}, while {reference_path} "
                f"has {reference_value!r}. Regenerate both Julia artifacts "
                "with the same output prefix."
            )

    if geometry["output_prefix"] != requested_prefix:
        raise SystemExit(
            f"Julia artifacts identify output_prefix={geometry['output_prefix']!r}, "
            f"but the command requested {requested_prefix!r}. Regenerate or select "
            "the matching artifacts."
        )

    nx = geometry["nx"]
    ny = geometry["ny"]
    if isinstance(nx, bool) or not isinstance(nx, int) or nx <= 0:
        raise SystemExit(f"Invalid positive integer 'nx' in {geometry_path}: {nx!r}.")
    if isinstance(ny, bool) or not isinstance(ny, int) or ny <= 0:
        raise SystemExit(f"Invalid positive integer 'ny' in {geometry_path}: {ny!r}.")
    mask = geometry.get("metal_mask_row_major")
    if not isinstance(mask, list) or len(mask) != ny:
        raise SystemExit(
            f"Invalid metal mask in {geometry_path}: expected {ny} rows. "
            "Regenerate the Julia geometry artifact."
        )
    for row_index, row in enumerate(mask):
        if not isinstance(row, list) or len(row) != nx:
            raise SystemExit(
                f"Invalid metal mask row {row_index} in {geometry_path}: "
                f"expected {nx} entries. Regenerate the Julia geometry artifact."
            )
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
                raise SystemExit(
                    f"Invalid metal mask value at ({row_index}, {column_index}) "
                    f"in {geometry_path}: expected integer 0 or 1, got {value!r}. "
                    "Regenerate the Julia geometry artifact."
                )


def _required_nonempty_text(
    source: Path, data: Dict[str, Any], key: str, recovery: str
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(
            f"Invalid or missing {key!r} in {source}: expected a nonempty JSON "
            f"string. {recovery}"
        )
    return value


def validate_meep_result_provenance(
    requested_prefix: str,
    geometry_path: Path,
    geometry: Dict[str, Any],
    reference_path: Path,
    reference: Dict[str, Any],
    result_path: Path,
    result: Dict[str, Any],
) -> tuple[str, str]:
    """Bind one Meep result to the exact current Julia artifact pair."""
    recovery = "Regenerate the Meep result from the current Julia artifacts."
    validate_julia_artifact_pair(
        requested_prefix,
        geometry_path,
        geometry,
        reference_path,
        reference,
    )
    geometry_sha256 = sha256_file(geometry_path, recovery=recovery)
    reference_sha256 = sha256_file(reference_path, recovery=recovery)
    result_prefix = _required_nonempty_text(
        result_path, result, "output_prefix", recovery
    )
    if result_prefix != requested_prefix:
        raise SystemExit(
            f"Meep result {result_path} identifies output_prefix={result_prefix!r}, "
            f"but the command requested {requested_prefix!r}. {recovery}"
        )
    recorded_geometry_sha256 = _required_nonempty_text(
        result_path, result, "julia_geometry_sha256", recovery
    )
    recorded_reference_sha256 = _required_nonempty_text(
        result_path, result, "julia_reference_sha256", recovery
    )
    if recorded_geometry_sha256 != geometry_sha256:
        raise SystemExit(
            f"Meep result {result_path} was generated from Julia geometry "
            f"SHA-256 {recorded_geometry_sha256}, which does not match current "
            f"artifact {geometry_path} ({geometry_sha256}). {recovery}"
        )
    if recorded_reference_sha256 != reference_sha256:
        raise SystemExit(
            f"Meep result {result_path} was generated from Julia reference "
            f"SHA-256 {recorded_reference_sha256}, which does not match current "
            f"artifact {reference_path} ({reference_sha256}). {recovery}"
        )
    return geometry_sha256, reference_sha256


def cli_options(*pairs: tuple[str, Any]) -> List[str]:
    """Flatten command-line option/value pairs into subprocess arguments."""
    return [str(token) for pair in pairs for token in pair]


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    """Run one validation subprocess and fail if it does not complete."""
    print("$", shlex.join(cmd))
    try:
        subprocess.run(list(cmd), cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Validation subprocess exited with status {exc.returncode}: "
            f"{shlex.join(cmd)}. Review its output above before rerunning."
        ) from exc
