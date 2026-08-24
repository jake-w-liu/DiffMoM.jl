#!/usr/bin/env python3
"""Fail when a slopfix census exceeds the committed duplication ceiling."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}")
    return value


def read_ceiling(path: pathlib.Path) -> int:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"cannot read duplication ceiling {path}: {error}") from error
    if not text.isascii() or not text.isdigit():
        raise ValueError(
            f"duplication ceiling {path} must contain one nonnegative integer"
        )
    return int(text)


def read_estimate(path: pathlib.Path) -> int:
    try:
        with path.open(encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read census report {path}: {error}") from error
    if not isinstance(report, dict) or report.get("kind") != "slopfix-census":
        raise ValueError(f"{path} is not a slopfix census report")
    duplication = report.get("duplication")
    if not isinstance(duplication, dict):
        raise ValueError(f"{path} has no duplication result")
    return nonnegative_integer(
        duplication.get("removable_lines_estimate"),
        "duplication.removable_lines_estimate",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("census", type=pathlib.Path)
    parser.add_argument("ceiling", type=pathlib.Path)
    args = parser.parse_args()
    try:
        estimate = read_estimate(args.census)
        ceiling = read_ceiling(args.ceiling)
    except ValueError as error:
        parser.error(str(error))
    print(f"duplication_estimate={estimate} ceiling={ceiling}")
    if estimate > ceiling:
        print(
            "error: duplication estimate exceeds the committed ceiling; "
            "consolidate the new clone or raise the ceiling deliberately",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
