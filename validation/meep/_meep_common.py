"""Shared implementation details for the Meep validation commands."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, List, Sequence


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
        )
        _validate_json_numbers(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(
            f"Could not read standard JSON from {path}: {exc}. {recovery}"
        ) from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid JSON in {path}: expected an object. {recovery}")
    return data


def cli_options(*pairs: tuple[str, Any]) -> List[str]:
    """Flatten command-line option/value pairs into subprocess arguments."""
    return [str(token) for pair in pairs for token in pair]


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    """Run one validation subprocess and fail if it does not complete."""
    print("$", shlex.join(cmd))
    subprocess.run(list(cmd), cwd=str(cwd), check=True)
