"""Replayable, evidence-bearing quality gates for slopfix engagements.

The reduction scanner deliberately covers only structural slop.  This module
adds the wider assurance contract: every ISO/IEC 25010:2023 product-quality
characteristic must be represented, every executable gate is an argv array
(never a shell string), and missing evidence stays UNVERIFIED.

Quality configurations are executable input.  ``slopfix quality-check`` only
validates them by default; commands run only when the caller passes ``--run``.
"""

from __future__ import annotations

import base64
import datetime as _datetime
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable
from typing import Any, BinaryIO

from . import counting

SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

PASS = "PASS"
FAIL = "FAIL"
UNVERIFIED = "UNVERIFIED"
NOT_APPLICABLE = "NOT_APPLICABLE"
STATUSES = (PASS, FAIL, UNVERIFIED, NOT_APPLICABLE)

QUALITY_CHARACTERISTICS = (
    "functional-suitability",
    "performance-efficiency",
    "compatibility",
    "interaction-capability",
    "reliability",
    "security",
    "maintainability",
    "flexibility",
    "safety",
)

GATE_KINDS = (
    "command",
    "reachable-contains",
    "julia-reachable-contains",
    "review",
)
REVIEW_DISPOSITIONS = ("pass", "fail", "unverified", "not_applicable")
_GATE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_INCLUDE = re.compile(
    r"""\binclude\s*\(\s*(?P<quote>["'])(?P<path>[^"'()\n]+\.jl)(?P=quote)\s*\)"""
)
_JULIA_PROJECT_NAME = re.compile(
    r'^\s*name\s*=\s*"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"\s*(?:#.*)?$'
)
_MAX_TEXT_BYTES = 5 * 1024 * 1024
_MAX_REACHABLE_FILES = 1000
_MAX_CAPTURE_BYTES = 16 * 1024


class QualityConfigError(ValueError):
    """The quality configuration is malformed or unsafe."""


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_join(root: str, relative: str, field: str) -> str:
    if not isinstance(relative, str) or not relative.strip():
        raise QualityConfigError(f"{field} must be a non-empty relative path")
    if "\0" in relative or os.path.isabs(relative):
        raise QualityConfigError(f"{field} must stay inside the repository: {relative!r}")
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, relative))
    try:
        inside = os.path.commonpath((root_real, candidate)) == root_real
    except ValueError:
        inside = False
    if not inside:
        raise QualityConfigError(f"{field} escapes the repository: {relative!r}")
    return candidate


def _review_gate(
    gate_id: str,
    category: str,
    title: str,
    instructions: str,
    *,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "category": category,
        "title": title,
        "required": required,
        "kind": "review",
        "disposition": "unverified",
        "evidence": "",
        "rationale": "",
        "instructions": instructions,
    }


def _command_gate(
    gate_id: str,
    category: str,
    title: str,
    command: list[str],
    *,
    timeout_seconds: int,
    required: bool = True,
    cwd: str = ".",
    isolation: str = "none",
    protect_paths: list[str] | None = None,
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "id": gate_id,
        "category": category,
        "title": title,
        "required": required,
        "kind": "command",
        "command": command,
        "cwd": cwd,
        "timeout_seconds": timeout_seconds,
        "capture": "digest",
    }
    if isolation != "none":
        gate["isolation"] = isolation
    if protect_paths:
        gate["protect_paths"] = protect_paths
    return gate


def _generic_gates() -> list[dict[str, Any]]:
    return [
        _review_gate(
            "project-tests", "functional-suitability", "Project test suite",
            "Replace this review gate with the repository's real test command and "
            "record the intended behavior and edge-case coverage.",
        ),
        _review_gate(
            "coverage-ratchet", "functional-suitability", "Coverage quality ratchet",
            "Configure a changed-line or project coverage threshold. Coverage alone "
            "is not correctness; pair it with characterisation, property, differential, "
            "or mutation tests where risk warrants.",
        ),
        _review_gate(
            "performance-baseline", "performance-efficiency",
            "Performance and resource baseline",
            "Benchmark representative hot paths and pin time, memory, allocation, "
            "latency, or throughput limits that callers rely on.",
        ),
        _review_gate(
            "platform-matrix", "compatibility", "Supported platform matrix",
            "Run the supported runtime, operating-system, architecture, and dependency "
            "matrix, or mark dimensions not applicable with a concrete rationale.",
        ),
        _review_gate(
            "docs-examples", "interaction-capability", "Documentation and examples",
            "Build documentation and execute doctests, examples, CLI help, and error "
            "messages that users depend on.",
        ),
        _review_gate(
            "failure-concurrency-resources", "reliability",
            "Failure, concurrency, and resource behavior",
            "Exercise retries, partial failure, concurrent calls, cleanup, restart, "
            "idempotency, and leak-sensitive paths applicable to the system.",
        ),
        _review_gate(
            "security-scan", "security", "Security and secret scanning",
            "Configure authoritative source, secret, configuration, and dependency "
            "security checks. A passing functional suite is not a security scan.",
        ),
        _review_gate(
            "dependency-provenance", "security", "Dependency provenance and advisories",
            "Resolve dependencies in a clean environment; verify names, identities, "
            "locks/manifests, licenses, SBOM output, and available advisories.",
        ),
        _review_gate(
            "language-maintainability", "maintainability",
            "Language-native maintainability checks",
            "Run the language's authoritative linter/static analysis and resolve or "
            "ratchet existing findings.",
        ),
        _review_gate(
            "public-api-contract", "flexibility", "Public API and extension contract",
            "Snapshot or otherwise verify exported APIs, extension points, schemas, "
            "configuration, migrations, and backward-compatibility commitments.",
        ),
        _review_gate(
            "safety-analysis", "safety", "Safety and domain hazards",
            "For safety- or mission-critical behavior, verify hazard controls and "
            "domain invariants. Otherwise mark this not applicable and state why.",
            required=False,
        ),
    ]


def _julia_package_name(root: str) -> str | None:
    """Return the standard package module name, or None for an application project."""
    for filename in ("Project.toml", "JuliaProject.toml"):
        path = os.path.join(root, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.lstrip().startswith("["):
                        break
                    match = _JULIA_PROJECT_NAME.fullmatch(line.rstrip("\r\n"))
                    if match:
                        name = match["name"]
                        source = os.path.join(root, "src", f"{name}.jl")
                        return name if os.path.isfile(source) else None
        except (OSError, UnicodeError):
            return None
    return None


def _julia_gates(root: str) -> list[dict[str, Any]]:
    project_files = [
        "Project.toml", "JuliaProject.toml", "Manifest.toml", "JuliaManifest.toml"
    ]
    project_files.extend(
        os.path.relpath(path, root).replace(os.sep, "/")
        for path in glob.glob(os.path.join(root, "Manifest-v*.toml"))
    )
    project_files = sorted(set(project_files))

    package_name = _julia_package_name(root)
    if package_name is None:
        clean_resolve = (
            "using Pkg, TOML; "
            "root = abspath(ARGS[1]); "
            "project = let candidates = "
            "[joinpath(root, \"Project.toml\"), joinpath(root, \"JuliaProject.toml\")]; "
            "existing = filter(isfile, candidates); "
            "isempty(existing) && error(\"no Julia project file\"); first(existing); end; "
            "function absolutize_paths!(value, base); "
            "if value isa AbstractDict; "
            "for key in collect(keys(value)); child = value[key]; "
            "if key == \"path\" && child isa AbstractString && !isabspath(child); "
            "value[key] = normpath(joinpath(base, child)); "
            "else; absolutize_paths!(child, base); end; end; "
            "elseif value isa AbstractVector; "
            "foreach(child -> absolutize_paths!(child, base), value); end; "
            "return value; end; "
            "function copy_environment_file(file, env); "
            "data = TOML.parsefile(file); absolutize_paths!(data, root); "
            "open(joinpath(env, basename(file)), \"w\") do io; "
            "TOML.print(io, data); end; end; "
            "mktempdir() do env; "
            "copy_environment_file(project, env); "
            "for file in readdir(root; join=true); "
            "name = basename(file); "
            "if isfile(file) && "
            "(name in (\"Manifest.toml\", \"JuliaManifest.toml\") || "
            "occursin(r\"^Manifest-v[0-9].*\\.toml$\", name)); "
            "copy_environment_file(file, env); "
            "end; end; "
            "Pkg.activate(env); "
            "Pkg.instantiate(); "
            "end"
        )
        clean_title = "Clean Julia application dependency resolution"
    else:
        clean_resolve = (
            "using Pkg; "
            "root = abspath(ARGS[1]); "
            "mktempdir() do env; "
            "Pkg.activate(env); "
            "Pkg.develop(Pkg.PackageSpec(path=root)); "
            "Pkg.instantiate(); "
            "Pkg.precompile(); "
            "end"
        )
        clean_title = "Clean Julia package resolution and precompile"
    package_test = (
        "using Pkg, TOML; "
        "root = abspath(ARGS[1]); "
        "project = let candidates = "
        "[joinpath(root, \"Project.toml\"), joinpath(root, \"JuliaProject.toml\")]; "
        "existing = filter(isfile, candidates); "
        "isempty(existing) && error(\"no Julia project file\"); first(existing); end; "
        "name = get(TOML.parsefile(project), \"name\", nothing); "
        "name isa String || error(\"Julia package project has no string name\"); "
        "mktempdir() do env; "
        "Pkg.activate(env); "
        "Pkg.develop(Pkg.PackageSpec(path=root)); "
        "Pkg.test(Pkg.PackageSpec(name=name)); "
        "end"
    )

    if package_name is None:
        test_gate = _review_gate(
            "julia-tests", "functional-suitability", "Julia project tests",
            "This Julia project is not a standard package with a top-level name and "
            "matching src/<Name>.jl. Replace this review gate with the application's "
            "real test command, isolate its active environment, and protect project "
            "and manifest files from mutation.",
        )
    else:
        test_gate = _command_gate(
            "julia-tests", "functional-suitability", "Julia project tests",
            ["julia", "--startup-file=no", "-e", package_test, "."],
            timeout_seconds=3600,
            protect_paths=project_files,
        )

    gates: list[dict[str, Any]] = [
        test_gate,
        {
            "id": "julia-aqua",
            "category": "maintainability",
            "title": "Aqua package checks are in the reachable Julia test graph",
            "required": True,
            "kind": "julia-reachable-contains",
            "entrypoint": "test/runtests.jl",
            "needles": ["Aqua.test_all"],
            "max_files": 500,
        },
        _command_gate(
            "julia-clean-resolution", "security",
            clean_title,
            [
                "julia", "--startup-file=no", "-e", clean_resolve, ".",
            ],
            timeout_seconds=3600,
            isolation="temporary-julia-depot",
            protect_paths=project_files,
        ),
        _review_gate(
            "julia-manifest-reproducibility", "compatibility",
            "Julia manifest and environment reproducibility",
            "For applications, instantiate the committed Manifest.toml in a clean "
            "environment and prove it is unchanged. For libraries that intentionally "
            "do not commit a manifest, mark this not applicable and cite the supported "
            "compatibility/version-matrix policy.",
        ),
        _review_gate(
            "julia-coverage-ratchet", "functional-suitability",
            "Julia coverage and test-strength ratchet",
            "Set a coverage threshold or changed-line ratchet and add property, "
            "differential, fuzz, or mutation checks where example-based tests can "
            "mirror an incorrect implementation.",
        ),
        _review_gate(
            "julia-numerical-contracts", "functional-suitability",
            "Julia numerical and scientific contracts",
            "Test applicable combinations of NaN, ±Inf, missing, empty/degenerate "
            "shapes, integer overflow, Float32/Float64/BigFloat, units, tolerances, "
            "random seeds, reproducibility, and domain invariants. Mark irrelevant "
            "dimensions not applicable with reasons.",
        ),
        _review_gate(
            "julia-benchmarks", "performance-efficiency",
            "Julia benchmark and allocation baseline",
            "Use BenchmarkTools or PkgBenchmark on representative hot paths; pin "
            "time and memory tolerances and allocation ceilings. Include compilation "
            "or time-to-first-execution when it is user-visible.",
        ),
        _review_gate(
            "julia-version-platform-matrix", "compatibility",
            "Supported Julia and platform matrix",
            "Test every supported Julia minor version and relevant OS/architecture. "
            "Include optional dependencies and package extensions where applicable.",
        ),
        _review_gate(
            "julia-docs-doctests", "interaction-capability",
            "Julia documentation, doctests, and examples",
            "Build Documenter output and execute doctests and user examples. Verify "
            "that documented names are public and examples instantiate cleanly.",
        ),
        _review_gate(
            "julia-tasks-resources", "reliability",
            "Julia tasks, concurrency, and resource lifecycle",
            "Exercise Threads.@spawn/async paths, channels, locks, cancellation, "
            "partial failure, files/sockets/GPU resources, and shutdown. Aqua's "
            "persistent-task check covers package loading, not every runtime path.",
        ),
        _review_gate(
            "julia-secrets-security", "security",
            "Julia source, configuration, and secret security scan",
            "Configure a repository-appropriate secret scanner and security review. "
            "Record any unavailable Julia-aware SAST coverage as a boundary rather "
            "than claiming no vulnerabilities.",
        ),
        _review_gate(
            "julia-sbom-advisories", "security",
            "Julia SBOM, licenses, and available advisories",
            "Generate an SBOM from the manifest when applicable, inspect package "
            "sources/licenses, and run an available Julia-aware advisory scanner. "
            "Record unavailable ecosystem coverage as UNVERIFIED.",
        ),
        _review_gate(
            "julia-explicit-imports", "maintainability",
            "ExplicitImports and public-name checks",
            "Add ExplicitImports checks for implicit, stale, non-owning, or non-public "
            "imports and qualified accesses, allowing documented exceptions only.",
            required=False,
        ),
        _review_gate(
            "julia-jet-entrypoints", "maintainability",
            "Pinned JET analysis on concrete entrypoints",
            "Pin a Julia-compatible JET release and analyze concrete public/hot "
            "entrypoints. Clear or document existing findings before ratcheting.",
            required=False,
        ),
        _review_gate(
            "julia-public-api", "flexibility",
            "Julia public API, dispatch, and extension compatibility",
            "Inventory exported/public names, method-extension contracts, schemas, "
            "preferences, weak dependencies, package extensions, deprecations, and "
            "serialization compatibility. Verify changes against downstream callers.",
        ),
        _review_gate(
            "julia-safety-hazards", "safety",
            "Julia domain safety and hazard controls",
            "For safety-, finance-, control-, or research-critical code, verify hazard "
            "controls, bounds, units, tolerances, deterministic replay, and trusted "
            "reference results. Otherwise mark not applicable with a reason.",
            required=False,
        ),
    ]

    docs_make = os.path.join(root, "docs", "make.jl")
    if os.path.isfile(docs_make):
        for index, gate in enumerate(gates):
            if gate["id"] == "julia-docs-doctests":
                gates[index] = _command_gate(
                    "julia-docs-doctests", "interaction-capability",
                    "Julia documentation, doctests, and examples",
                    [
                        "julia", "--project=docs", "--startup-file=no",
                        "docs/make.jl",
                    ],
                    timeout_seconds=1800,
                    protect_paths=[
                        "docs/Project.toml", "docs/Manifest.toml",
                        "docs/JuliaProject.toml", "docs/JuliaManifest.toml",
                    ],
                )
                break
    return gates


def detect_profile(root: str) -> str:
    """Return ``julia`` for a Julia project, otherwise ``generic``."""
    if any(
        os.path.isfile(os.path.join(root, name))
        for name in ("Project.toml", "JuliaProject.toml")
    ):
        for _base, dirs, files in os.walk(root):
            dirs[:] = [
                name for name in dirs
                if name not in {".git", ".slopfix", "node_modules", "vendor"}
            ]
            if any(name.endswith(".jl") for name in files):
                return "julia"
    return "generic"


def build(root: str, profile: str = "auto") -> dict[str, Any]:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise QualityConfigError(f"repository root is not a directory: {root}")
    if profile == "auto":
        profile = detect_profile(root)
    if profile not in {"generic", "julia"}:
        raise QualityConfigError(f"unknown quality profile: {profile!r}")
    gates = _julia_gates(root) if profile == "julia" else _generic_gates()
    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": "slopfix-quality-config",
        "profile": profile,
        "created_at": _now(),
        "quality_model": "ISO/IEC 25010:2023",
        "gates": gates,
    }
    validate(config, root)
    return config


def _require_keys(mapping: dict[str, Any], keys: Iterable[str], where: str) -> None:
    for key in keys:
        if key not in mapping:
            raise QualityConfigError(f"{where} is missing required field {key!r}")


def _validate_common(gate: dict[str, Any], root: str, index: int) -> None:
    where = f"gate #{index + 1}"
    _require_keys(gate, ("id", "category", "title", "required", "kind"), where)
    gate_id = gate["id"]
    if not isinstance(gate_id, str) or not _GATE_ID.fullmatch(gate_id):
        raise QualityConfigError(
            f"{where} id must be lower-case kebab-case, 1-64 characters"
        )
    if gate["category"] not in QUALITY_CHARACTERISTICS:
        raise QualityConfigError(
            f"gate {gate_id!r} has unknown category {gate['category']!r}"
        )
    if not isinstance(gate["title"], str) or not gate["title"].strip():
        raise QualityConfigError(f"gate {gate_id!r} title must be non-empty")
    if not isinstance(gate["required"], bool):
        raise QualityConfigError(f"gate {gate_id!r} required must be boolean")
    if gate["kind"] not in GATE_KINDS:
        raise QualityConfigError(
            f"gate {gate_id!r} has unknown kind {gate['kind']!r}"
        )

    if gate["kind"] == "command":
        command = gate.get("command")
        if (
            not isinstance(command, list)
            or not command
            or len(command) > 128
            or any(
                not isinstance(part, str)
                or not part
                or len(part) > 4096
                or "\0" in part
                for part in command
            )
        ):
            raise QualityConfigError(
                f"gate {gate_id!r} command must be a non-empty argv string array"
            )
        timeout = gate.get("timeout_seconds")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or not 1 <= timeout <= 86400
        ):
            raise QualityConfigError(
                f"gate {gate_id!r} timeout_seconds must be 1..86400"
            )
        _safe_join(root, gate.get("cwd", "."), f"gate {gate_id!r} cwd")
        capture = gate.get("capture", "digest")
        if capture not in {"digest", "tail"}:
            raise QualityConfigError(
                f"gate {gate_id!r} capture must be 'digest' or 'tail'"
            )
        isolation = gate.get("isolation", "none")
        if isolation not in {"none", "temporary-julia-depot"}:
            raise QualityConfigError(
                f"gate {gate_id!r} has unknown isolation {isolation!r}"
            )
        protect = gate.get("protect_paths", [])
        if not isinstance(protect, list) or any(not isinstance(p, str) for p in protect):
            raise QualityConfigError(
                f"gate {gate_id!r} protect_paths must be a string array"
            )
        for path in protect:
            _safe_join(root, path, f"gate {gate_id!r} protected path")

    elif gate["kind"] in {"reachable-contains", "julia-reachable-contains"}:
        _safe_join(root, gate.get("entrypoint", ""), f"gate {gate_id!r} entrypoint")
        needles = gate.get("needles")
        if (
            not isinstance(needles, list)
            or not needles
            or any(
                not isinstance(needle, str) or not needle or "\0" in needle
                for needle in needles
            )
        ):
            raise QualityConfigError(
                f"gate {gate_id!r} needles must be a non-empty string array"
            )
        max_files = gate.get("max_files", 500)
        if (
            isinstance(max_files, bool)
            or not isinstance(max_files, int)
            or not 1 <= max_files <= _MAX_REACHABLE_FILES
        ):
            raise QualityConfigError(
                f"gate {gate_id!r} max_files must be 1..{_MAX_REACHABLE_FILES}"
            )

    elif gate["kind"] == "review":
        disposition = gate.get("disposition")
        if disposition not in REVIEW_DISPOSITIONS:
            raise QualityConfigError(
                f"gate {gate_id!r} has invalid review disposition {disposition!r}"
            )
        evidence = gate.get("evidence", "")
        rationale = gate.get("rationale", "")
        if not isinstance(evidence, str) or not isinstance(rationale, str):
            raise QualityConfigError(
                f"gate {gate_id!r} evidence and rationale must be strings"
            )
        if disposition in {"pass", "fail"} and not evidence.strip():
            raise QualityConfigError(
                f"gate {gate_id!r} disposition {disposition!r} requires evidence"
            )
        if disposition == "not_applicable" and not rationale.strip():
            raise QualityConfigError(
                f"gate {gate_id!r} not_applicable requires a rationale"
            )


def validate(config: dict[str, Any], root: str) -> None:
    if not isinstance(config, dict):
        raise QualityConfigError("quality config must be a JSON object")
    _require_keys(
        config,
        ("schema_version", "kind", "profile", "quality_model", "gates"),
        "quality config",
    )
    if (
        isinstance(config["schema_version"], bool)
        or config["schema_version"] != SCHEMA_VERSION
    ):
        raise QualityConfigError(
            f"quality config schema_version is {config['schema_version']!r}; "
            f"this tool requires {SCHEMA_VERSION}"
        )
    if config["kind"] != "slopfix-quality-config":
        raise QualityConfigError("quality config kind must be 'slopfix-quality-config'")
    if (
        not isinstance(config["profile"], str)
        or config["profile"] not in {"generic", "julia"}
    ):
        raise QualityConfigError("quality config profile must be 'generic' or 'julia'")
    if config["quality_model"] != "ISO/IEC 25010:2023":
        raise QualityConfigError(
            "quality_model must be the pinned value 'ISO/IEC 25010:2023'"
        )
    gates = config["gates"]
    if not isinstance(gates, list) or not gates:
        raise QualityConfigError("quality config gates must be a non-empty array")
    seen: set[str] = set()
    categories: set[str] = set()
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            raise QualityConfigError(f"gate #{index + 1} must be an object")
        _validate_common(gate, root, index)
        if gate["id"] in seen:
            raise QualityConfigError(f"duplicate gate id: {gate['id']!r}")
        seen.add(gate["id"])
        categories.add(gate["category"])
    missing = sorted(set(QUALITY_CHARACTERISTICS) - categories)
    if missing:
        raise QualityConfigError(
            "quality config does not represent every quality characteristic; "
            f"missing: {', '.join(missing)}"
        )


def read(path: str, root: str) -> tuple[dict[str, Any], str]:
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise QualityConfigError(f"could not read quality config {path}: {exc}") from exc
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityConfigError(f"quality config is not valid UTF-8 JSON: {exc}") from exc
    validate(config, root)
    return config, _sha256_bytes(raw)


def write(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".slopfix-quality-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _protected_state(root: str, paths: list[str]) -> dict[str, str]:
    state: dict[str, str] = {}
    for relative in paths:
        absolute = _safe_join(root, relative, "protected path")
        if os.path.isfile(absolute):
            state[relative] = _sha256_file(absolute)
        elif os.path.exists(absolute):
            state[relative] = "<non-file>"
        else:
            state[relative] = "<missing>"
    return state


def _output_evidence(data: bytes, capture: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
    }
    if capture == "tail":
        result["tail"] = data[-_MAX_CAPTURE_BYTES:].decode("utf-8", errors="replace")
        result["truncated"] = len(data) > _MAX_CAPTURE_BYTES
    return result


def _file_output_evidence(handle: BinaryIO, capture: str) -> dict[str, Any]:
    handle.flush()
    handle.seek(0)
    digest = hashlib.sha256()
    total = 0
    tail = b""
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        total += len(block)
        digest.update(block)
        if capture == "tail":
            tail = (tail + block)[-_MAX_CAPTURE_BYTES:]
    result: dict[str, Any] = {
        "bytes": total,
        "sha256": digest.hexdigest(),
    }
    if capture == "tail":
        result["tail"] = tail.decode("utf-8", errors="replace")
        result["truncated"] = total > _MAX_CAPTURE_BYTES
    return result


def _stream_output_evidence(
    stream: BinaryIO,
    capture: str,
    destination: dict[str, Any],
) -> None:
    """Drain one child pipe while retaining only a digest and bounded tail."""
    digest = hashlib.sha256()
    total = 0
    tail = b""
    try:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            digest.update(block)
            if capture == "tail":
                tail = (tail + block)[-_MAX_CAPTURE_BYTES:]
        destination.update({
            "bytes": total,
            "sha256": digest.hexdigest(),
        })
        if capture == "tail":
            destination["tail"] = tail.decode("utf-8", errors="replace")
            destination["truncated"] = total > _MAX_CAPTURE_BYTES
    finally:
        stream.close()


def _terminate_owned_process_tree(proc: subprocess.Popen[bytes]) -> bool:
    """Stop one gate's exact process group; return whether anything was alive."""
    had_live_process = proc.poll() is None
    if os.name == "posix":
        try:
            os.killpg(proc.pid, 0)
            had_live_process = True
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        # Python's stdlib has no Windows Job Object API. The direct child is
        # still exact and owned; CREATE_NEW_PROCESS_GROUP keeps it isolated.
        try:
            if proc.poll() is None:
                proc.kill()
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"timed-out command process {proc.pid} did not terminate"
            ) from exc
    return had_live_process


def _join_output_threads(
    threads: list[threading.Thread],
    proc: subprocess.Popen[bytes],
) -> None:
    for thread in threads:
        thread.join(timeout=30)
    if any(thread.is_alive() for thread in threads):
        # A surviving Windows descendant can retain an inherited pipe even
        # after the direct child exits. Closing our exact gate pipes prevents
        # an unbounded wait; the gate then fails rather than claiming evidence.
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        for thread in threads:
            thread.join(timeout=5)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("command output collectors did not terminate")


def _run_command(gate: dict[str, Any], root: str) -> dict[str, Any]:
    command: list[str] = gate["command"]
    if os.name == "nt":
        return {
            "status": UNVERIFIED,
            "source": "command",
            "message": (
                "native Windows command gates are not executed because this "
                "runner cannot guarantee descendant-process cleanup; use WSL "
                "or a reviewed external CI runner"
            ),
            "command": command,
        }
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "status": UNVERIFIED,
            "source": "command",
            "message": f"executable not found on PATH: {command[0]}",
            "command": command,
        }
    cwd = _safe_join(root, gate.get("cwd", "."), f"gate {gate['id']!r} cwd")
    if not os.path.isdir(cwd):
        return {
            "status": FAIL,
            "source": "command",
            "message": f"working directory does not exist: {gate.get('cwd', '.')}",
            "command": command,
        }
    protect_paths: list[str] = gate.get("protect_paths", [])
    before = _protected_state(root, protect_paths)
    environment = os.environ.copy()
    environment["SLOPFIX_QUALITY_GATE"] = gate["id"]
    depot: tempfile.TemporaryDirectory[str] | None = None
    if gate.get("isolation", "none") == "temporary-julia-depot":
        depot = tempfile.TemporaryDirectory(prefix="slopfix-julia-depot-")
        environment["JULIA_DEPOT_PATH"] = depot.name
        environment["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    started = time.monotonic()
    proc: subprocess.Popen[bytes] | None = None
    exceptional: dict[str, Any] | None = None
    stdout_evidence: dict[str, Any] | None = None
    stderr_evidence: dict[str, Any] | None = None
    returncode: int | None = None
    surviving_descendants = False
    try:
        try:
            popen_args: dict[str, Any] = {
                "cwd": cwd,
                "env": environment,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if os.name == "posix":
                popen_args["start_new_session"] = True
            elif os.name == "nt":
                popen_args["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(command, **popen_args)
            assert proc.stdout is not None and proc.stderr is not None
            stdout_evidence = {}
            stderr_evidence = {}
            capture = gate.get("capture", "digest")
            threads = [
                threading.Thread(
                    target=_stream_output_evidence,
                    args=(proc.stdout, capture, stdout_evidence),
                    daemon=True,
                ),
                threading.Thread(
                    target=_stream_output_evidence,
                    args=(proc.stderr, capture, stderr_evidence),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            try:
                returncode = proc.wait(timeout=gate["timeout_seconds"])
                # The direct command can exit while daemonized children keep
                # running. They belong to this exact, newly-created process
                # group; clean them before inspecting protected files.
                if os.name == "posix":
                    surviving_descendants = _terminate_owned_process_tree(proc)
            except subprocess.TimeoutExpired:
                _terminate_owned_process_tree(proc)
                exceptional = {
                    "status": FAIL,
                    "source": "command",
                    "message": f"timed out after {gate['timeout_seconds']} seconds",
                    "command": command,
                    "duration_seconds": round(time.monotonic() - started, 3),
                }
            _join_output_threads(threads, proc)
        except OSError as exc:
            exceptional = {
                "status": UNVERIFIED,
                "source": "command",
                "message": f"could not execute command: {exc}",
                "command": command,
                "duration_seconds": round(time.monotonic() - started, 3),
            }
    finally:
        if depot is not None:
            depot.cleanup()

    after = _protected_state(root, protect_paths)
    changed = [
        path for path in protect_paths if before.get(path) != after.get(path)
    ]
    if exceptional is not None:
        exceptional["stdout"] = stdout_evidence or _output_evidence(b"", "digest")
        exceptional["stderr"] = stderr_evidence or _output_evidence(b"", "digest")
        exceptional["protected_paths_changed"] = changed
        if changed:
            exceptional["status"] = FAIL
            exceptional["message"] += "; protected files changed: " + ", ".join(changed)
        return exceptional
    if proc is None or returncode is None:
        raise RuntimeError("command execution produced no result")
    status = (
        PASS
        if returncode == 0 and not changed and not surviving_descendants
        else FAIL
    )
    message = f"exit code {returncode}"
    if surviving_descendants:
        message += "; terminated surviving descendant process(es)"
    if changed:
        message += "; protected files changed: " + ", ".join(changed)
    return {
        "status": status,
        "source": "command",
        "message": message,
        "command": command,
        "executable": executable,
        "cwd": os.path.relpath(cwd, root).replace(os.sep, "/"),
        "duration_seconds": round(time.monotonic() - started, 3),
        "exit_code": returncode,
        "stdout": stdout_evidence,
        "stderr": stderr_evidence,
        "protected_paths_changed": changed,
    }


def _read_small_text(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise QualityConfigError(f"could not stat {path}: {exc}") from exc
    if size > _MAX_TEXT_BYTES:
        raise QualityConfigError(
            f"reachable source is larger than {_MAX_TEXT_BYTES} bytes: {path}"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeError) as exc:
        raise QualityConfigError(f"could not read reachable source {path}: {exc}") from exc


def _run_reachable_contains(gate: dict[str, Any], root: str) -> dict[str, Any]:
    entry = _safe_join(root, gate["entrypoint"], f"gate {gate['id']!r} entrypoint")
    if not os.path.isfile(entry):
        return {
            "status": FAIL,
            "source": "static",
            "message": f"entrypoint does not exist: {gate['entrypoint']}",
            "files_checked": [],
        }
    root_real = os.path.realpath(root)
    pending = [entry]
    visited: set[str] = set()
    found: dict[str, list[str]] = {}
    limit = gate.get("max_files", 500)
    while pending:
        path = os.path.realpath(pending.pop())
        if path in visited:
            continue
        if len(visited) >= limit:
            return {
                "status": UNVERIFIED,
                "source": "static",
                "message": f"reachable include graph exceeds configured limit {limit}",
                "files_checked": sorted(
                    os.path.relpath(item, root_real).replace(os.sep, "/")
                    for item in visited
                ),
            }
        try:
            if os.path.commonpath((root_real, path)) != root_real:
                return {
                    "status": FAIL,
                    "source": "static",
                    "message": f"reachable include escapes repository: {path}",
                    "files_checked": [],
                }
        except ValueError:
            return {
                "status": FAIL,
                "source": "static",
                "message": f"reachable include escapes repository: {path}",
                "files_checked": [],
            }
        if not os.path.isfile(path):
            return {
                "status": FAIL,
                "source": "static",
                "message": "reachable include does not exist: "
                + os.path.relpath(path, root_real).replace(os.sep, "/"),
                "files_checked": [],
            }
        visited.add(path)
        try:
            text = _read_small_text(path)
        except QualityConfigError as exc:
            return {
                "status": UNVERIFIED,
                "source": "static",
                "message": str(exc),
                "files_checked": [],
            }
        relative = os.path.relpath(path, root_real).replace(os.sep, "/")
        for needle in gate["needles"]:
            if needle in text:
                found.setdefault(needle, []).append(relative)
        for match in _INCLUDE.finditer(text):
            child = os.path.realpath(os.path.join(os.path.dirname(path), match["path"]))
            pending.append(child)
    missing = [needle for needle in gate["needles"] if needle not in found]
    return {
        "status": FAIL if missing else PASS,
        "source": "static",
        "message": (
            "missing from reachable test graph: " + ", ".join(missing)
            if missing else "all required markers are reachable"
        ),
        "files_checked": sorted(
            os.path.relpath(item, root_real).replace(os.sep, "/")
            for item in visited
        ),
        "matches": found,
    }


def _run_julia_reachable_contains(
    gate: dict[str, Any], root: str
) -> dict[str, Any]:
    executable = counting.julia_path()
    if executable is None:
        return {
            "status": UNVERIFIED,
            "source": "julia-ast",
            "message": "julia is not on PATH",
            "files_checked": [],
            "matches": {},
        }
    helper = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "julia_lines.jl",
    )
    if not os.path.isfile(helper):
        return {
            "status": UNVERIFIED,
            "source": "julia-ast",
            "message": f"Julia helper is missing: {helper}",
            "files_checked": [],
            "matches": {},
        }
    fields = [
        os.path.abspath(root),
        gate["entrypoint"],
        str(gate.get("max_files", 500)),
        *gate["needles"],
    ]
    try:
        proc = subprocess.run(
            [executable, "--startup-file=no", helper, "--reachable-contains"],
            input="\0".join(fields),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": UNVERIFIED,
            "source": "julia-ast",
            "message": "Julia reachable-graph analysis timed out after 300 seconds",
            "files_checked": [],
            "matches": {},
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": UNVERIFIED,
            "source": "julia-ast",
            "message": f"could not run Julia reachable-graph analysis: {exc}",
            "files_checked": [],
            "matches": {},
        }
    if proc.returncode != 0:
        return {
            "status": UNVERIFIED,
            "source": "julia-ast",
            "message": "Julia reachable-graph helper failed: "
            + (proc.stderr or proc.stdout).strip()[:400],
            "files_checked": [],
            "matches": {},
        }

    status: str | None = None
    message = ""
    files: list[str] = []
    matches: dict[str, list[str]] = {}

    def decode(value: str) -> str:
        try:
            return base64.b64decode(value, validate=True).decode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise QualityConfigError(
                "Julia reachable-graph helper emitted invalid base64"
            ) from exc

    for row in proc.stdout.splitlines():
        parts = row.split("\t")
        if len(parts) == 3 and parts[0] == "STATUS":
            status = parts[1]
            message = decode(parts[2])
        elif len(parts) == 2 and parts[0] == "FILE":
            files.append(decode(parts[1]))
        elif len(parts) == 3 and parts[0] == "MATCH":
            needle = decode(parts[1])
            matches.setdefault(needle, []).append(decode(parts[2]))
        elif row.strip():
            raise QualityConfigError(
                f"Julia reachable-graph helper emitted an invalid row: {row[:200]!r}"
            )
    if status not in STATUSES:
        raise QualityConfigError(
            "Julia reachable-graph helper emitted no valid status"
        )
    return {
        "status": status,
        "source": "julia-ast",
        "message": message,
        "files_checked": sorted(set(files)),
        "matches": {
            needle: sorted(set(paths)) for needle, paths in matches.items()
        },
    }


def _run_review(gate: dict[str, Any]) -> dict[str, Any]:
    disposition = gate["disposition"]
    status = {
        "pass": PASS,
        "fail": FAIL,
        "unverified": UNVERIFIED,
        "not_applicable": NOT_APPLICABLE,
    }[disposition]
    message = gate.get("evidence", "") if status in {PASS, FAIL} else (
        gate.get("rationale", "") if status == NOT_APPLICABLE
        else gate.get("instructions", "no evidence recorded")
    )
    return {
        "status": status,
        "source": "review",
        "message": message,
        "evidence": gate.get("evidence", ""),
        "rationale": gate.get("rationale", ""),
    }


def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = {status: 0 for status in STATUSES}
    by_category = {
        category: {status: 0 for status in STATUSES}
        for category in QUALITY_CHARACTERISTICS
    }
    required_unverified = 0
    failures = 0
    for result in results:
        status = result["status"]
        by_status[status] += 1
        by_category[result["category"]][status] += 1
        if status == FAIL:
            failures += 1
        if result["required"] and status == UNVERIFIED:
            required_unverified += 1
    return {
        "total": len(results),
        "by_status": by_status,
        "by_category": by_category,
        "failures": failures,
        "required_unverified": required_unverified,
        "strict_pass": failures == 0 and required_unverified == 0,
    }


def run(
    config: dict[str, Any],
    root: str,
    config_sha256: str,
    only: set[str] | None = None,
) -> dict[str, Any]:
    """Execute or evaluate selected gates and return an immutable evidence report."""
    root = os.path.abspath(root)
    validate(config, root)
    all_ids = {gate["id"] for gate in config["gates"]}
    if only:
        unknown = sorted(only - all_ids)
        if unknown:
            raise QualityConfigError(f"unknown --only gate id(s): {', '.join(unknown)}")
    selected = [
        gate for gate in config["gates"] if not only or gate["id"] in only
    ]
    results: list[dict[str, Any]] = []
    for gate in selected:
        if gate["kind"] == "command":
            evidence = _run_command(gate, root)
        elif gate["kind"] == "julia-reachable-contains":
            evidence = _run_julia_reachable_contains(gate, root)
        elif gate["kind"] == "reachable-contains":
            evidence = _run_reachable_contains(gate, root)
        else:
            evidence = _run_review(gate)
        results.append({
            "id": gate["id"],
            "category": gate["category"],
            "title": gate["title"],
            "required": gate["required"],
            "kind": gate["kind"],
            **evidence,
        })
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "slopfix-quality-report",
        "generated_at": _now(),
        "quality_model": config["quality_model"],
        "profile": config["profile"],
        "root": root,
        "config_sha256": config_sha256,
        "partial": len(selected) != len(config["gates"]),
        "selected_gate_ids": [gate["id"] for gate in selected],
        "summary": _summarise(results),
        "gates": results,
    }
