"""The frozen baseline, the re-measure, and the integrity checks.

A reduction percentage is a claim about a codebase, so it is only trustworthy if
the measurement was pinned before any work started. `build()` writes that pin:
the counter identity, the exact file scope, and the per-file counts. `compare()`
replays it and refuses to produce a number if anything about the measurement
itself changed.

`compare()` also runs the integrity checks that make the number honest. They
encode the engagement's forbidden moves as detectors:

  `code-parked-outside-scope`  code moved into an excluded directory instead of
                               being deleted
  `code-golf-suspected`        lines got dramatically longer, i.e. statements
                               were packed together to lower the line count
  `comments-stripped`          explanatory comments deleted (they never counted
                               toward the target, so removing them can only cost
                               maintainability)
  `tests-deleted`              test code shrank faster than production code
  `placeholder-introduced`     new stub or swallowed-error code appeared

None of these are automatic verdicts. Each is a finding a human has to clear or
act on before the reduction can be reported.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections import defaultdict

from . import counting, langs, scope, smells

SCHEMA_VERSION = 3

# Integrity thresholds. Each is a detector sensitivity, not a rule about how code
# should look, so they live here rather than being buried in the checks.
GOLF_MIN_LINE_GROWTH_PCT = 25.0   # mean code-line length growth that looks packed
GOLF_MIN_MEAN_CHARS = 80          # ...and only once lines are actually long
GOLF_LONG_LINE_CHARS = 200        # a single line this long is worth a look
# Prose loss is judged per file, not in aggregate. Consolidating twenty
# documented functions into one legitimately deletes nineteen docstrings, and no
# aggregate ratio can tell that apart from stripping docstrings out of code that
# was left alone. The per-file question can: did a file lose its prose *while its
# code stayed put*? Aggregate measures also break under scc, where docstring
# lines count as code.
PROSE_STRIP_MIN_DROP_PCT = 50.0   # a file losing this share of its prose...
PROSE_STRIP_MAX_CODE_MOVE_PCT = 20.0  # ...while its code moved less than this
PROSE_STRIP_MIN_LINES = 5         # per-file floor, to ignore trivial churn
COMMENT_DROP_FLOOR_LINES = 10     # repo-wide floor before reporting at all
TEST_DROP_MARGIN_PCT = 15.0       # test code shrinking faster than prod code by this
COUNTER_DIVERGENCE_PCT = 10.0     # contract counter vs builtin gap worth surfacing


# The counters `counter_identity` knows how to resolve. Kept in one place: a
# second hard-coded copy in `compare` meant a `julia` baseline could be written
# but never replayed.
COUNTERS = ("scc", "builtin", "julia")


class CounterMismatch(RuntimeError):
    """Raised when a baseline and a re-measure used different counters."""


class ScopeMismatch(RuntimeError):
    """Raised when a manifest cannot be replayed as written."""


# Test-file classification lives in `scope`, next to the rest of the path
# bucketing, and is re-exported here because the integrity checks read as
# `manifest.is_test_path`.
is_test_path = scope.is_test_path


# --- taking a measurement ----------------------------------------------------

def counter_identity(counter: str) -> str:
    """Resolve a counter name to its recorded identity string.

    The identity is what makes a reduction reproducible, so a counter that
    behaves differently must read differently. `julia` is the builtin scanner
    with a parser-backed Julia backend, and its identity names both -- a baseline taken
    with Julia present cannot be silently re-measured without it.
    """
    if counter == "scc":
        return counting.scc_identity()
    if counter == "builtin":
        return counting.BUILTIN_COUNTER_ID
    if counter == "julia":
        return f"{counting.BUILTIN_COUNTER_ID}+{counting.julia_identity()}"
    raise ValueError(f"unknown counter {counter!r}; expected one of {COUNTERS}")


def resolve_counter(requested: str) -> str:
    """Turn `auto` into a concrete counter, preferring scc."""
    if requested != "auto":
        return requested
    return "scc" if counting.scc_path() is not None else "builtin"


def _strict_source_hashes(
    root: str, files: list[tuple[str, langs.Language]]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relpath, _ in files:
        digest = hashlib.sha256()
        path = os.path.join(root, relpath)
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise counting.SourceReadError(
                f"{relpath}: cannot hash stable measurement input ({exc})"
            ) from exc
        hashes[relpath] = digest.hexdigest()
    return hashes


def _assert_measurement_stable(
    root: str, sc: scope.Scope, discovery: scope.Discovery
) -> None:
    current = scope.discover(sc)
    before_files = [(path, lang.name) for path, lang in discovery.counted]
    after_files = [(path, lang.name) for path, lang in current.counted]
    if before_files != after_files or discovery.rejected != current.rejected:
        raise counting.SourceReadError(
            "repository file scope changed while it was being measured; "
            "retry against a quiescent worktree"
        )
    if _strict_source_hashes(root, current.counted) != discovery.measurement_hashes:
        raise counting.SourceReadError(
            "in-scope source changed while it was being measured; "
            "retry against a quiescent worktree"
        )


def measure(
    root: str, sc: scope.Scope, counter: str
) -> tuple[dict[str, counting.FileCount], scope.Discovery, list[str]]:
    """Count the in-scope files. Returns (relpath -> count, discovery, warnings)."""
    discovery = scope.discover(sc)
    before_hashes = _strict_source_hashes(root, discovery.counted)
    warnings: list[str] = []
    counts: dict[str, counting.FileCount] = {}

    if counter == "scc":
        rows = counting.run_scc(
            root=root,
            exclude_dirs=list(sc.exclude_dirs),
            respect_gitignore=sc.respect_gitignore,
        )
        in_scope = {relpath for relpath, _ in discovery.counted}
        for row in rows:
            if row.path in in_scope:
                counts[row.path] = row
        # scc has its own file discovery; anything our scope wanted but scc did
        # not report is recorded rather than silently counted as zero.
        missing = sorted(in_scope - set(counts))
        if missing:
            raise counting.SccOutputError(
                f"{len(missing)} in-scope file(s) were not reported by scc "
                f"(first few: {missing[:5]}). Counting them as zero would "
                "understate the baseline; explicitly exclude unsupported files "
                "or use --counter builtin."
            )
    else:
        # `julia` is the builtin scanner with Julia's own parser handling .jl
        # files; every other language still uses the builtin scanner.
        julia_files = (
            [rel for rel, lang in discovery.counted if lang.name == "Julia"]
            if counter == "julia" else []
        )
        exact: dict[str, counting.FileCount] = {}
        if julia_files:
            exact = counting.run_julia(root, julia_files)
        for relpath, lang in discovery.counted:
            if relpath in exact:
                counts[relpath] = exact[relpath]
                continue
            result = counting.count_file(
                os.path.join(root, relpath), lang, sc.max_file_bytes
            )
            result.path = relpath
            counts[relpath] = result
            warnings.extend(result.warnings)

    after_hashes = _strict_source_hashes(root, discovery.counted)
    if before_hashes != after_hashes:
        raise counting.SourceReadError(
            "in-scope source changed while the counter was running; "
            "retry against a quiescent worktree"
        )
    discovery.measurement_hashes = after_hashes
    return counts, discovery, warnings


def _out_of_scope_code(root: str, sc: scope.Scope, discovery: scope.Discovery) -> int:
    """Code lines in source files the scope excluded.

    Always measured with the builtin scanner: it is a diagnostic for the parked-
    code check, not part of the contract number, and scc would have to be re-run
    over a different file set to produce it.
    """
    total = 0
    for relpath, lang in discovery.out_of_scope_source:
        text, _ = counting.read_source(
            os.path.join(root, relpath), sc.max_file_bytes
        )
        # This is a diagnostic outside the frozen denominator. An excluded
        # generated or vendor file can be oversized/binary without making the
        # in-scope baseline unknowable; it simply cannot contribute evidence to
        # the parking check.
        if text is not None:
            total += counting.count_text(text, lang, relpath).code
    return total


def _builtin_metrics(
    root: str, sc: scope.Scope, files: list[tuple[str, langs.Language]]
) -> dict:
    """Diagnostic metrics from the builtin scanner, whatever the active counter.

    Deliberately counter-independent. The headline number belongs to the
    contract counter, but every *check* on that number has to come from a
    scanner whose classification we can trust:

    * `scc` 3.7.0 mis-tracks state when an apostrophe appears inside a Python
      triple-quoted docstring -- the "isn't" case -- which is extremely common.
      It then reports docstring lines as code, inflating `Code`: measured at
      +71% against Python's own tokenizer on `_pydecimal.py`. Because those lines
      count as code to scc, *deleting docstrings lowers the scc number*, which is
      precisely the gaming vector the non-comment definition exists to close.
    * The builtin scanner classifies docstrings as comments and agrees with
      `ast` + `tokenize` to within 0.15% on the same file.

    So `comments-stripped` and the code-golf checks read from here, not from the
    per-counter totals, and stay accurate under either counter.
    """
    total_chars = 0
    code_lines = 0
    comment_lines = 0
    longest = 0
    per_file: dict[str, list[int]] = {}
    for relpath, lang in files:
        result = counting.count_file(
            os.path.join(root, relpath), lang, sc.max_file_bytes
        )
        total_chars += result.code_chars
        code_lines += result.code
        comment_lines += result.comments
        longest = max(longest, result.max_code_line)
        # [code, comments] per file, so the prose-stripping check can ask the
        # per-file question that no aggregate ratio can answer.
        per_file[relpath] = [result.code, result.comments]
    return {
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "mean_code_chars": round(total_chars / code_lines, 2) if code_lines else 0.0,
        "max_code_line": longest,
        "per_file": per_file,
    }


def _git_head(root: str) -> str | None:
    """The commit the baseline was taken at, or None.

    Returning None on failure is deliberate and safe: the HEAD sha is provenance
    recorded in the manifest, not an input to any count. Contrast
    `scope._git_tracked_files`, where a failure *would* change the measured file
    set and therefore raises instead.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None  # provenance only; never affects the measurement
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def _code_line_fingerprints(
    root: str, sc: scope.Scope, files: list[tuple[str, langs.Language]]
) -> dict[str, list[str]]:
    """Ordered hashes of builtin-classified code lines for gross diff metrics.

    The manifest does not retain source text. Ordered fingerprints are enough
    for SequenceMatcher to distinguish deletion, insertion and replacement while
    keeping the contract artefact compact.
    """
    fingerprints: dict[str, list[str]] = {}
    for relpath, lang in files:
        text, _ = counting.read_source(
            os.path.join(root, relpath), sc.max_file_bytes
        )
        if text is None:
            continue
        scanned, _ = counting.scan(text, lang, relpath)
        fingerprints[relpath] = [
            hashlib.sha256(line.stripped.encode("utf-8")).hexdigest()
            for line in scanned
            if line.kind == counting.CODE
        ]
    return fingerprints


def build(
    root: str,
    sc: scope.Scope,
    counter: str = "auto",
    promised_reduction_pct: float = 0.0,
) -> dict:
    """Take a baseline measurement and freeze everything needed to replay it."""
    resolved = resolve_counter(counter)
    identity = counter_identity(resolved)
    counts, discovery, warnings = measure(root, sc, resolved)

    per_language: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "code": 0, "comments": 0, "blanks": 0, "lines": 0}
    )
    totals = {"files": 0, "code": 0, "comments": 0, "blanks": 0, "lines": 0}
    test_code = 0
    for relpath, count in counts.items():
        bucket = per_language[count.language]
        bucket["files"] += 1
        totals["files"] += 1
        for field in ("code", "comments", "blanks", "lines"):
            value = getattr(count, field)
            bucket[field] += value
            totals[field] += value
        if is_test_path(relpath):
            test_code += count.code

    promised = float(promised_reduction_pct)
    if not math.isfinite(promised) or not 0.0 <= promised <= 100.0:
        raise ValueError(
            f"promised reduction must be between 0 and 100 percent, got "
            f"{promised_reduction_pct!r}"
        )
    # A percentage commitment means removing *at least* that share. Rounding the
    # remaining lines can round in the wrong direction: 50% of a 3-line baseline
    # used to produce a target of 2 lines (only 33% removed). Round the required
    # removal upward, then derive the target.
    required_removal = math.ceil(totals["code"] * promised / 100.0)
    target_lines = totals["code"] - required_removal

    rejected_counts: dict[str, int] = defaultdict(int)
    for bucket in discovery.rejected.values():
        rejected_counts[bucket] += 1

    builtin = _builtin_metrics(root, sc, discovery.counted)
    # Cross-check the contract counter against an independent scan. A large gap
    # is not necessarily wrong, but it must be visible before a target is quoted
    # against the number — see _builtin_metrics for the known scc case.
    if resolved != "builtin" and builtin["code_lines"] > 0 and totals["code"] > 0:
        gap_pct = abs(totals["code"] - builtin["code_lines"]) / builtin["code_lines"] * 100.0
        if gap_pct >= COUNTER_DIVERGENCE_PCT:
            warnings.append(
                f"counter cross-check: {identity} reports {totals['code']} code lines, "
                f"the builtin scanner reports {builtin['code_lines']} "
                f"({gap_pct:.1f}% apart). The target is quoted against "
                f"{identity}, so this is not an error — but confirm the "
                "denominator is what you intend. scc 3.7.0 inflates Python "
                "counts when a docstring contains an apostrophe."
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "slopfix-baseline",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "repo": {
            "root": os.path.abspath(root),
            "git_head": _git_head(root),
        },
        "counter": {"name": resolved, "id": identity},
        "scope": sc.to_json(),
        "totals": totals,
        "test_code": test_code,
        "out_of_scope_source_code": _out_of_scope_code(root, sc, discovery),
        "generated_file_count": rejected_counts.get(scope.GENERATED, 0),
        "builtin_metrics": builtin,
        "per_language": {
            name: dict(values) for name, values in sorted(per_language.items())
        },
        "files": {relpath: count.code for relpath, count in sorted(counts.items())},
        "file_hashes": dict(discovery.measurement_hashes),
        "code_line_fingerprints": _code_line_fingerprints(
            root, sc, discovery.counted
        ),
        "file_comments": {
            relpath: count.comments for relpath, count in sorted(counts.items())
        },
        "discovery": {
            "notes": discovery.notes,
            "rejected_counts": dict(sorted(rejected_counts.items())),
        },
        "target": {
            "promised_reduction_pct": promised,
            "target_code_lines": target_lines,
            "lines_to_remove": required_removal,
        },
        "warnings": warnings,
    }
    _assert_measurement_stable(root, sc, discovery)
    return _validate_baseline(payload, "generated baseline")


def write(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".slopfix-baseline-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
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


def _nonnegative_int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScopeMismatch(f"{where} must be a non-negative integer")
    return value


def _string_list(value, where: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ScopeMismatch(f"{where} must be an array of strings")
    return value


def _validate_baseline(payload: object, path: str) -> dict:
    if not isinstance(payload, dict) or payload.get("kind") != "slopfix-baseline":
        raise ScopeMismatch(f"{path} is not a slopfix baseline manifest")
    version = payload.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ScopeMismatch(
            f"{path} has schema_version {version!r}, this tool writes "
            f"{SCHEMA_VERSION}. Re-take the baseline or use the matching version."
        )
    required = (
        "repo", "counter", "scope", "totals", "test_code",
        "out_of_scope_source_code", "generated_file_count", "builtin_metrics",
        "per_language", "files", "file_hashes", "code_line_fingerprints",
        "file_comments", "discovery", "target", "warnings",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ScopeMismatch(
            f"{path} is missing required baseline field(s): {', '.join(missing)}"
        )

    repo = payload["repo"]
    if (
        not isinstance(repo, dict)
        or not isinstance(repo.get("root"), str)
        or not repo["root"]
        or not (repo.get("git_head") is None or isinstance(repo.get("git_head"), str))
    ):
        raise ScopeMismatch(f"{path} repo must contain root and optional git_head strings")

    counter = payload["counter"]
    if (
        not isinstance(counter, dict)
        or counter.get("name") not in COUNTERS
        or not isinstance(counter.get("id"), str)
        or not counter["id"]
    ):
        raise ScopeMismatch(f"{path} counter must contain a supported name and id")

    frozen_scope = payload["scope"]
    if not isinstance(frozen_scope, dict):
        raise ScopeMismatch(f"{path} scope must be an object")
    for key in ("exclude_dirs", "exclude_globs", "generated_globs"):
        _string_list(frozen_scope.get(key), f"{path} scope.{key}")
    if not isinstance(frozen_scope.get("include_non_source"), bool):
        raise ScopeMismatch(f"{path} scope.include_non_source must be boolean")
    if not isinstance(frozen_scope.get("respect_gitignore"), bool):
        raise ScopeMismatch(f"{path} scope.respect_gitignore must be boolean")
    max_bytes = frozen_scope.get("max_file_bytes")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ScopeMismatch(f"{path} scope.max_file_bytes must be a positive integer")

    totals = payload["totals"]
    if not isinstance(totals, dict):
        raise ScopeMismatch(f"{path} totals must be an object")
    for key in ("files", "code", "comments", "blanks", "lines"):
        _nonnegative_int(totals.get(key), f"{path} totals.{key}")
    _nonnegative_int(payload["test_code"], f"{path} test_code")
    _nonnegative_int(
        payload["out_of_scope_source_code"], f"{path} out_of_scope_source_code"
    )
    _nonnegative_int(payload["generated_file_count"], f"{path} generated_file_count")

    target = payload["target"]
    if not isinstance(target, dict):
        raise ScopeMismatch(f"{path} target must be an object")
    promised = target.get("promised_reduction_pct")
    if (
        isinstance(promised, bool)
        or not isinstance(promised, int | float)
        or not math.isfinite(float(promised))
        or not 0.0 <= float(promised) <= 100.0
    ):
        raise ScopeMismatch(f"{path} target.promised_reduction_pct must be 0..100")
    target_lines = _nonnegative_int(
        target.get("target_code_lines"), f"{path} target.target_code_lines"
    )
    lines_to_remove = _nonnegative_int(
        target.get("lines_to_remove"), f"{path} target.lines_to_remove"
    )
    expected_removal = math.ceil(totals["code"] * float(promised) / 100.0)
    if (
        lines_to_remove != expected_removal
        or target_lines != totals["code"] - lines_to_remove
    ):
        raise ScopeMismatch(f"{path} target is inconsistent with baseline code totals")

    for key in ("files", "file_comments"):
        mapping = payload[key]
        if not isinstance(mapping, dict) or any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for name, value in mapping.items()
        ):
            raise ScopeMismatch(
                f"{path} {key} must map path strings to non-negative integers"
            )
    hashes = payload["file_hashes"]
    if not isinstance(hashes, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in hashes.items()
    ):
        raise ScopeMismatch(f"{path} file_hashes must map path strings to hashes")
    fingerprints = payload["code_line_fingerprints"]
    if not isinstance(fingerprints, dict) or any(
        not isinstance(name, str)
        or not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        for name, values in fingerprints.items()
    ):
        raise ScopeMismatch(
            f"{path} code_line_fingerprints must map paths to hash arrays"
        )

    builtin = payload["builtin_metrics"]
    if not isinstance(builtin, dict) or not isinstance(builtin.get("per_file"), dict):
        raise ScopeMismatch(f"{path} builtin_metrics must contain per_file")
    for key in ("code_lines", "comment_lines", "max_code_line"):
        _nonnegative_int(builtin.get(key), f"{path} builtin_metrics.{key}")
    mean_chars = builtin.get("mean_code_chars")
    if (
        isinstance(mean_chars, bool)
        or not isinstance(mean_chars, int | float)
        or not math.isfinite(float(mean_chars))
        or mean_chars < 0
    ):
        raise ScopeMismatch(f"{path} builtin_metrics.mean_code_chars must be non-negative")
    for name, pair in builtin["per_file"].items():
        if (
            not isinstance(name, str)
            or not isinstance(pair, list)
            or len(pair) != 2
        ):
            raise ScopeMismatch(
                f"{path} builtin_metrics.per_file values must be [code, comments]"
            )
        _nonnegative_int(pair[0], f"{path} builtin_metrics.per_file[{name!r}][0]")
        _nonnegative_int(pair[1], f"{path} builtin_metrics.per_file[{name!r}][1]")

    per_language = payload["per_language"]
    if not isinstance(per_language, dict):
        raise ScopeMismatch(f"{path} per_language must be an object")
    language_totals = {
        key: 0 for key in ("files", "code", "comments", "blanks", "lines")
    }
    for name, values in per_language.items():
        if not isinstance(name, str) or not name or not isinstance(values, dict):
            raise ScopeMismatch(
                f"{path} per_language must map language names to metric objects"
            )
        for key in language_totals:
            language_totals[key] += _nonnegative_int(
                values.get(key), f"{path} per_language[{name!r}].{key}"
            )
    if language_totals != {
        key: totals[key] for key in language_totals
    }:
        raise ScopeMismatch(f"{path} per_language metrics do not match totals")
    if totals["lines"] != totals["code"] + totals["comments"] + totals["blanks"]:
        raise ScopeMismatch(f"{path} total lines do not match code/comments/blanks")
    if totals["files"] != len(payload["files"]):
        raise ScopeMismatch(f"{path} totals.files does not match the files map")
    if payload["test_code"] > totals["code"]:
        raise ScopeMismatch(f"{path} test_code exceeds total code")

    discovery = payload["discovery"]
    if (
        not isinstance(discovery, dict)
        or not isinstance(discovery.get("rejected_counts"), dict)
    ):
        raise ScopeMismatch(f"{path} discovery must contain rejected_counts")
    _string_list(discovery.get("notes"), f"{path} discovery.notes")
    for bucket, value in discovery["rejected_counts"].items():
        if not isinstance(bucket, str) or not bucket:
            raise ScopeMismatch(f"{path} discovery bucket names must be strings")
        _nonnegative_int(value, f"{path} discovery.rejected_counts[{bucket!r}]")
    if (
        discovery["rejected_counts"].get(scope.GENERATED, 0)
        != payload["generated_file_count"]
    ):
        raise ScopeMismatch(
            f"{path} generated_file_count does not match discovery"
        )
    _string_list(payload["warnings"], f"{path} warnings")
    return payload


def read(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return _validate_baseline(payload, path)


# --- comparing against the baseline ------------------------------------------

def attainment(baseline: int, current: int, promised_pct: float) -> float:
    """Percentage of the promised reduction actually delivered.

    Mirrors the engagement's payment rule: promise 50% of 100 lines (50 removed),
    deliver 20 removed, and that is 40% of the goal. Clamped to [0, 100]: code
    growth is 0% attainment, and over-delivery is 100%.
    """
    if promised_pct <= 0 or baseline <= 0:
        return 0.0
    promised_lines = baseline * (promised_pct / 100.0)
    if promised_lines <= 0:
        return 0.0
    removed = baseline - current
    if removed <= 0:
        return 0.0
    return round(min(100.0, removed / promised_lines * 100.0), 2)


def _finding(rule: str, message: str, **detail) -> dict:
    return {"rule": rule, "message": message, "detail": detail}


def compare(baseline: dict, root: str, counter: str = "auto") -> dict:
    """Re-measure `root` under the baseline's frozen scope and counter.

    `auto` means "whatever the baseline used", not "whatever is installed". A
    baseline taken with the builtin counter must keep being measured with it even
    once scc is available, or the comparison would be refused for no good reason.
    """
    baseline = _validate_baseline(baseline, "baseline")
    sc = scope.Scope.from_json(root, baseline["scope"])
    recorded = baseline["counter"]["id"]
    if counter == "auto":
        resolved = str(baseline["counter"].get("name") or "").strip()
        if resolved not in COUNTERS:
            raise ScopeMismatch(
                f"baseline records an unusable counter name {resolved!r}; "
                f"pass one of {COUNTERS} explicitly."
            )
    else:
        resolved = counter
    identity = counter_identity(resolved)
    if identity != recorded:
        raise CounterMismatch(
            f"baseline was measured with {recorded!r} but this run resolves to "
            f"{identity!r}. A reduction computed across two counters is not a "
            f"real number. Install the original counter, or re-baseline."
        )

    counts, discovery, warnings = measure(root, sc, resolved)
    current_code = sum(count.code for count in counts.values())
    baseline_code = int(baseline["totals"]["code"])

    baseline_files: dict[str, int] = dict(baseline.get("files", {}))
    current_files = {relpath: count.code for relpath, count in counts.items()}

    before_fingerprints: dict[str, list[str]] = dict(
        baseline.get("code_line_fingerprints", {})
    )
    after_fingerprints = _code_line_fingerprints(root, sc, discovery.counted)
    removed_lines = 0
    added_lines = 0
    for relpath in set(before_fingerprints) | set(after_fingerprints):
        before = before_fingerprints.get(relpath, [])
        after = after_fingerprints.get(relpath, [])
        matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in {"delete", "replace"}:
                removed_lines += i2 - i1
            if tag in {"insert", "replace"}:
                added_lines += j2 - j1

    promised = float(baseline["target"]["promised_reduction_pct"])
    net_removed = baseline_code - current_code
    report = {
        "kind": "slopfix-measure",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "counter": {"name": resolved, "id": identity},
        "baseline_code": baseline_code,
        "current_code": current_code,
        "gross_removed": removed_lines,
        "gross_added": added_lines,
        "gross_method": f"{counting.BUILTIN_COUNTER_ID}-line-fingerprint-diff",
        "net_removed": net_removed,
        "reduction_pct": round(net_removed / baseline_code * 100.0, 2) if baseline_code else 0.0,
        "promised_reduction_pct": promised,
        "target_code_lines": int(baseline["target"]["target_code_lines"]),
        "attainment_pct": attainment(baseline_code, current_code, promised),
        "files_added": sorted(set(current_files) - set(baseline_files)),
        "files_removed": sorted(set(baseline_files) - set(current_files)),
        "warnings": warnings,
        "integrity": _integrity_checks(
            baseline=baseline,
            root=root,
            sc=sc,
            discovery=discovery,
            counts=counts,
            current_code=current_code,
            baseline_code=baseline_code,
        ),
    }
    _assert_measurement_stable(root, sc, discovery)
    return report


def _integrity_checks(
    *,
    baseline: dict,
    root: str,
    sc: scope.Scope,
    discovery: scope.Discovery,
    counts: dict[str, counting.FileCount],
    current_code: int,
    baseline_code: int,
) -> list[dict]:
    findings: list[dict] = []
    code_removed = baseline_code - current_code
    # All comment/length diagnostics come from the builtin scanner regardless of
    # the contract counter. See _builtin_metrics for why that matters.
    before_builtin = baseline.get("builtin_metrics", {})
    after_builtin = _builtin_metrics(root, sc, discovery.counted)

    # 1. Code parked outside the measured scope instead of deleted.
    baseline_oos = int(baseline.get("out_of_scope_source_code", 0))
    current_oos = _out_of_scope_code(root, sc, discovery)
    oos_growth = current_oos - baseline_oos
    if oos_growth > 0 and code_removed > 0 and oos_growth >= 0.10 * code_removed:
        findings.append(_finding(
            "code-parked-outside-scope",
            f"Source code outside the measured scope grew by {oos_growth} lines "
            f"while in-scope code fell by {code_removed}. Moving code into an "
            "excluded directory is not a reduction.",
            baseline_out_of_scope=baseline_oos,
            current_out_of_scope=current_oos,
            growth=oos_growth,
        ))

    # 2. Code golf: lines got materially longer.
    before_golf = before_builtin
    after_golf = after_builtin
    before_mean = float(before_golf.get("mean_code_chars", 0.0))
    after_mean = float(after_golf.get("mean_code_chars", 0.0))
    if before_mean > 0 and after_mean >= GOLF_MIN_MEAN_CHARS:
        growth_pct = (after_mean - before_mean) / before_mean * 100.0
        if growth_pct >= GOLF_MIN_LINE_GROWTH_PCT:
            findings.append(_finding(
                "code-golf-suspected",
                f"Mean code-line length rose {growth_pct:.1f}% "
                f"({before_mean:.1f} -> {after_mean:.1f} chars). Packing "
                "statements onto fewer lines is an excluded reduction.",
                baseline_mean_chars=before_mean,
                current_mean_chars=after_mean,
                growth_pct=round(growth_pct, 2),
            ))
    before_max = int(before_golf.get("max_code_line", 0))
    after_max = int(after_golf.get("max_code_line", 0))
    if after_max >= GOLF_LONG_LINE_CHARS and after_max > before_max:
        findings.append(_finding(
            "long-line-introduced",
            f"Longest code line grew from {before_max} to {after_max} characters. "
            "Confirm this is generated or data content, not a collapsed block.",
            baseline_max=before_max,
            current_max=after_max,
        ))

    # 3. Comments and docstrings stripped. Under the builtin counter these never
    #    counted toward the target, so deleting them buys nothing. Under scc,
    #    Python docstring lines can count as *code*, which means deleting them
    #    does lower the number — so this check matters more, not less, there.
    stripped = _prose_stripped_files(before_builtin, after_builtin)
    total_stripped = sum(item["prose_removed"] for item in stripped)
    if total_stripped >= COMMENT_DROP_FLOOR_LINES:
        worst = ", ".join(
            f"{item['path']} (-{item['prose_removed']} prose, code "
            f"{item['baseline_code']}->{item['current_code']})"
            for item in stripped[:4]
        )
        findings.append(_finding(
            "comments-stripped",
            f"{total_stripped} lines of comments/docstrings were removed from "
            f"{len(stripped)} file(s) whose code barely changed: {worst}"
            f"{' ...' if len(stripped) > 4 else ''}. Consolidation removes prose "
            "along with the code it documented; prose leaving code that stayed "
            "put is not a reduction.",
            files=stripped[:25],
            total_prose_removed=total_stripped,
        ))

    # 4. Tests deleted faster than production code.
    baseline_tests = int(baseline.get("test_code", 0))
    current_tests = sum(
        count.code for relpath, count in counts.items() if is_test_path(relpath)
    )
    if baseline_tests > 0:
        baseline_production = max(0, baseline_code - baseline_tests)
        current_production = max(0, current_code - current_tests)
        production_drop_pct = (
            (baseline_production - current_production)
            / baseline_production
            * 100.0
            if baseline_production
            else 0.0
        )
        test_drop_pct = (baseline_tests - current_tests) / baseline_tests * 100.0
        if (
            test_drop_pct > 0
            and test_drop_pct > production_drop_pct + TEST_DROP_MARGIN_PCT
        ):
            findings.append(_finding(
                "tests-deleted",
                f"Test code fell {test_drop_pct:.1f}% against "
                f"{production_drop_pct:.1f}% production code. Tests are the "
                "safety net for every other change.",
                baseline_test_code=baseline_tests,
                current_test_code=current_tests,
                baseline_production_code=baseline_production,
                current_production_code=current_production,
            ))
    elif current_tests == 0 and current_code > 0:
        findings.append(_finding(
            "no-tests-present",
            "No test files were found at baseline or now. Behaviour-preserving "
            "consolidation cannot be verified by the suite; every claim has to "
            "rest on the behaviour inventory instead.",
        ))

    # 5. New placeholders or swallowed errors in files that changed.
    introduced = _new_blocking_smells(root, sc, discovery, baseline)
    if introduced:
        findings.append(_finding(
            "placeholder-introduced",
            f"{len(introduced)} blocking smell(s) appear in files that changed: "
            "stubs, unimplemented paths or discarded errors. A reduction that "
            "adds these is a regression.",
            hits=introduced[:25],
        ))

    return findings


def _prose_stripped_files(before: dict, after: dict) -> list[dict]:
    """Files that lost their comments/docstrings while their code stood still.

    That combination is the signature of deleting explanatory text to move the
    number, and it is what separates it from honest consolidation: merging twenty
    documented functions into one deletes nineteen docstrings, but it also
    collapses the code, so those files are not reported.
    """
    before_files = before.get("per_file") or {}
    after_files = after.get("per_file") or {}
    stripped: list[dict] = []
    for relpath, before_pair in before_files.items():
        after_pair = after_files.get(relpath)
        if after_pair is None:
            continue  # deleted file: covered by the reduction itself
        base_code, base_prose = int(before_pair[0]), int(before_pair[1])
        cur_code, cur_prose = int(after_pair[0]), int(after_pair[1])
        prose_removed = base_prose - cur_prose
        if base_prose <= 0 or prose_removed < PROSE_STRIP_MIN_LINES:
            continue
        if prose_removed / base_prose * 100.0 < PROSE_STRIP_MIN_DROP_PCT:
            continue
        # Did the code itself actually move? If it shrank (or grew) materially,
        # the prose went with a real change rather than being stripped out.
        code_move_pct = (
            abs(base_code - cur_code) / base_code * 100.0 if base_code > 0 else 100.0
        )
        if code_move_pct >= PROSE_STRIP_MAX_CODE_MOVE_PCT:
            continue
        stripped.append({
            "path": relpath,
            "prose_removed": prose_removed,
            "baseline_prose": base_prose,
            "current_prose": cur_prose,
            "baseline_code": base_code,
            "current_code": cur_code,
        })
    stripped.sort(key=lambda item: -item["prose_removed"])
    return stripped


def _new_blocking_smells(
    root: str, sc: scope.Scope, discovery: scope.Discovery, baseline: dict
) -> list[dict]:
    """Blocking smells in files whose content changed since baseline.

    Restricted to changed files so a pre-existing stub in untouched code is not
    reported as newly introduced. SHA-256, rather than the code-line count,
    establishes whether content changed; equal-sized replacements are a common
    way for a real implementation to become a stub.
    """
    baseline_hashes: dict[str, str] = dict(baseline.get("file_hashes", {}))
    hits: list[dict] = []
    for relpath, lang in discovery.counted:
        abspath = os.path.join(root, relpath)
        try:
            digest = hashlib.sha256()
            with open(abspath, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            current_hash = digest.hexdigest()
        except OSError:
            current_hash = None
        if baseline_hashes.get(relpath) == current_hash:
            continue
        text, _ = counting.read_source(abspath, sc.max_file_bytes)
        if text is None:
            continue
        for hit in smells.scan_text(relpath, text, lang):
            if hit.severity == smells.BLOCKING:
                hits.append(hit.to_json())
    return hits


def summarise_languages(baseline: dict) -> list[tuple[str, int, int]]:
    """(language, files, code) rows sorted by code descending, for display."""
    rows = [
        (name, values.get("files", 0), values.get("code", 0))
        for name, values in baseline.get("per_language", {}).items()
    ]
    rows.sort(key=lambda row: -row[2])
    return rows


def source_language_share(baseline: dict) -> dict[str, int]:
    """Code lines per language, restricted to languages inside the target."""
    return {
        name: values.get("code", 0)
        for name, values in baseline.get("per_language", {}).items()
        if langs.is_source_language(name)
    }
