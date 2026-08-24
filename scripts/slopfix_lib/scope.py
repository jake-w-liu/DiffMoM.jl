"""What counts as "the codebase".

The measurement scope is frozen at baseline time and replayed verbatim at
re-measure time. That is the whole point of this module: a reduction percentage
is only meaningful if the denominator was defined once, in advance, and cannot
drift. `slopfix measure` reads the scope out of the baseline manifest rather
than recomputing defaults, so changing the defaults in this file can never
retroactively change an open engagement's number.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from dataclasses import dataclass, field

from . import langs
from .langs import Language

_GIT_TIMEOUT_SECONDS = 120

# Directories that hold dependencies, build output or tool caches. Excluding
# them is the default because nobody is paid to delete `node_modules`.
DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    ".git", ".hg", ".svn", ".jj",
    "node_modules", "bower_components", "jspm_packages",
    "vendor", "third_party", "3rdparty", "external", "externals",
    "dist", "build", "out", "target", "bin", "obj",
    ".next", ".nuxt", ".svelte-kit", ".astro", ".turbo", ".parcel-cache",
    ".venv", "venv", ".env", "site-packages", "__pycache__", ".eggs",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    ".gradle", ".m2", "Pods", "DerivedData", ".terraform",
    "coverage", "htmlcov", ".nyc_output", ".cache",
    "__snapshots__", ".idea", ".vscode",
)

# Excluded directories that are never even enumerated: dependency trees, VCS
# metadata and tool caches. Walking them costs a lot and tells us nothing —
# nobody parks their own source code in `.git` or `node_modules`, and in a git
# repo they are ignored anyway.
HARD_PRUNE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", ".jj",
    "node_modules", "bower_components", "jspm_packages",
    ".venv", "venv", ".env", "site-packages", "__pycache__", ".eggs",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    ".gradle", ".m2", "Pods", "DerivedData", ".terraform",
    ".next", ".nuxt", ".svelte-kit", ".astro", ".turbo", ".parcel-cache",
    ".cache", "coverage", "htmlcov", ".nyc_output",
})

# Excluded directories whose contents legitimately change on every build. They
# are enumerated (so they can be reported) but not watched for parked code: a
# rebuild would otherwise look like someone hiding source there.
BUILD_OUTPUT_DIRS: frozenset[str] = frozenset({
    "dist", "build", "out", "target", "bin", "obj",
})

# Files nobody wrote by hand. Deleting generated output is not a reduction, so
# these are reported separately and excluded from the target denominator.
DEFAULT_GENERATED_GLOBS: tuple[str, ...] = (
    "*.min.js", "*.min.css", "*.min.mjs",
    "*.lock", "*-lock.json", "*-lock.yaml", "pnpm-lock.yaml", "go.sum",
    "*_pb2.py", "*_pb2_grpc.py", "*_pb.js", "*_pb.d.ts",
    "*.pb.go", "*.pb.cc", "*.pb.h", "*.pb.rs",
    "*.g.dart", "*.freezed.dart", "*.gr.dart", "*.config.dart",
    "*.gen.go", "*.gen.ts", "*.generated.*", "*_generated.*",
    "*.designer.cs", "*.feature.cs",
    "*.snap", "*.snapshot",
    "*.map",
)


_TEST_PATH_MARKERS = (
    "/test/", "/tests/", "/spec/", "/specs/", "/__tests__/", "/e2e/", "/integration/",
)
def is_test_path(relpath: str) -> bool:
    """True for test and spec files.

    Used in two places: the integrity check that watches for tests being deleted,
    and the concept census, where a test named `test_format_date` must not count
    as a fourteenth date formatter.
    """
    lowered = "/" + relpath.replace(os.sep, "/").lower()
    if any(marker in lowered for marker in _TEST_PATH_MARKERS):
        return True
    base = lowered.rsplit("/", 1)[-1]
    return (
        base == "conftest.py"
        or base.startswith("test_")
        or "_test." in base
        or ".test." in base
        or ".spec." in base
        or "_spec." in base
        or base.endswith(("test.java", "tests.cs"))
    )


@dataclass
class Scope:
    """A frozen definition of the measured file set."""

    root: str
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS
    exclude_globs: tuple[str, ...] = ()
    generated_globs: tuple[str, ...] = DEFAULT_GENERATED_GLOBS
    # Config/data/prose languages (YAML, JSON, Markdown...) are reported but not
    # counted toward the target unless the engagement explicitly includes them.
    include_non_source: bool = False
    max_file_bytes: int = 5 * 1024 * 1024
    respect_gitignore: bool = True

    def to_json(self) -> dict:
        return {
            "exclude_dirs": list(self.exclude_dirs),
            "exclude_globs": list(self.exclude_globs),
            "generated_globs": list(self.generated_globs),
            "include_non_source": self.include_non_source,
            "max_file_bytes": self.max_file_bytes,
            "respect_gitignore": self.respect_gitignore,
        }

    @classmethod
    def from_json(cls, root: str, payload: dict) -> Scope:
        """Rebuild a scope from a manifest.

        Every field is read explicitly. A manifest written by an older version
        that lacks a field falls back to that field's current default, and the
        caller is responsible for surfacing the manifest's schema version.
        """
        return cls(
            root=root,
            exclude_dirs=tuple(payload.get("exclude_dirs", DEFAULT_EXCLUDE_DIRS)),
            exclude_globs=tuple(payload.get("exclude_globs", ())),
            generated_globs=tuple(payload.get("generated_globs", DEFAULT_GENERATED_GLOBS)),
            include_non_source=bool(payload.get("include_non_source", False)),
            max_file_bytes=int(payload.get("max_file_bytes", 5 * 1024 * 1024)),
            respect_gitignore=bool(payload.get("respect_gitignore", True)),
        )


# Why a file was not counted. Every discovered file lands in exactly one bucket
# so the manifest can account for the whole tree.
COUNTED = "counted"
EXCLUDED_DIR = "excluded_dir"
EXCLUDED_GLOB = "excluded_glob"
GENERATED = "generated"
NON_SOURCE = "non_source"
UNRECOGNIZED = "unrecognized"
TOO_LARGE = "too_large"
BINARY = "binary"


@dataclass
class Discovery:
    counted: list[tuple[str, Language]] = field(default_factory=list)
    # relpath -> bucket, for every file that was discovered but not counted.
    rejected: dict[str, str] = field(default_factory=dict)
    # Non-counted source files that still hold real code, kept so `measure` can
    # detect code being parked outside the measured scope.
    out_of_scope_source: list[tuple[str, Language]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Populated by manifest.measure after a pre/post counter stability check.
    measurement_hashes: dict[str, str] = field(default_factory=dict)


class GitListingFailed(RuntimeError):
    """Raised when git says this *is* a work tree but listing its files failed.

    Falling back to a filesystem walk in that situation silently changes the
    measured file set -- ignored build output gets pulled in -- which changes the
    denominator of the reduction with no indication. A loud failure is correct:
    the alternative is a wrong number that looks right.
    """


def _git_tracked_files(root: str) -> list[str] | None:
    """Working-tree files git does not ignore, or None when root is not a repo.

    `--cached --others --exclude-standard` is exactly "tracked plus untracked
    but not ignored", which matches what a reviewer would call the codebase.

    Returns None only for the legitimate case: git is unavailable, or `root` is
    not a work tree. If it *is* a work tree and the listing fails, that raises
    rather than silently falling back to a different file set.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", root, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError:
        return None  # git not installed: filesystem walk is the honest fallback
    except subprocess.TimeoutExpired as exc:
        raise GitListingFailed(
            f"`git rev-parse` timed out after {_GIT_TIMEOUT_SECONDS} seconds for "
            f"{root}. Refusing to fall back to a filesystem walk, which could "
            "silently change the measured scope."
        ) from exc
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        return None  # not a work tree
    try:
        proc = subprocess.run(
            ["git", "-C", root, "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitListingFailed(
            f"{root} is a git work tree but `git ls-files` could not be run "
            f"({exc}). Refusing to fall back to a filesystem walk, which would "
            "silently include ignored files and change the measured scope."
        ) from exc
    if proc.returncode != 0:
        raise GitListingFailed(
            f"{root} is a git work tree but `git ls-files` exited "
            f"{proc.returncode}: {(proc.stderr or '').strip()[:200]}. Refusing to "
            "fall back to a filesystem walk, which would silently include ignored "
            "files and change the measured scope."
        )
    names = [name for name in proc.stdout.split("\0") if name]
    # De-duplicate: a path can appear in both --cached and --others listings.
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _matches_any(relpath: str, patterns: tuple[str, ...]) -> bool:
    base = os.path.basename(relpath)
    for pattern in patterns:
        if fnmatch.fnmatchcase(relpath, pattern) or fnmatch.fnmatchcase(base, pattern):
            return True
    return False


def _excluded_component(relpath: str, exclude_dirs: tuple[str, ...]) -> bool:
    parts = relpath.split("/")[:-1]
    return any(part in exclude_dirs for part in parts)


def _in_watched_dir(relpath: str, watched: frozenset[str]) -> bool:
    return any(part in watched for part in relpath.split("/")[:-1])


def parking_watch_dirs(scope_obj: Scope) -> frozenset[str]:
    """Excluded directories where source code appearing later is suspicious.

    Everything the caller added by hand is watched, because a new exclusion is
    exactly how a reduction target gets met by relocation rather than deletion.
    Dependency trees and build output are not watched.
    """
    return frozenset(scope_obj.exclude_dirs) - HARD_PRUNE_DIRS - BUILD_OUTPUT_DIRS


def _contains_nul(path: str) -> bool | None:
    """Return whether a file is binary-like, or None when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if b"\x00" in chunk:
                    return True
    except OSError:
        return None
    return False


def discover(scope: Scope) -> Discovery:
    """Enumerate the tree once and bucket every file."""
    result = Discovery()
    tracked = _git_tracked_files(scope.root) if scope.respect_gitignore else None
    if tracked is not None:
        result.notes.append(
            "file list from `git ls-files --cached --others --exclude-standard`"
        )
        candidates = tracked
    else:
        result.notes.append(
            "file list from filesystem walk (not a git work tree, or git "
            "unavailable); .gitignore is not applied"
        )
        candidates = _walk_filesystem(scope)

    watched = parking_watch_dirs(scope)
    for relpath in sorted(candidates):
        abspath = os.path.join(scope.root, relpath)
        if os.path.islink(abspath) or not os.path.isfile(abspath):
            continue

        bucket: str | None = None
        if _excluded_component(relpath, scope.exclude_dirs):
            bucket = EXCLUDED_DIR
        elif scope.exclude_globs and _matches_any(relpath, scope.exclude_globs):
            bucket = EXCLUDED_GLOB

        lang = langs.detect(relpath)
        if bucket is not None:
            result.rejected[relpath] = bucket
            # Source code sitting in an excluded directory is the classic way to
            # make a reduction target look met, so keep a handle on it.
            if lang is not None and lang.is_source and (
                bucket == EXCLUDED_GLOB or _in_watched_dir(relpath, watched)
            ):
                result.out_of_scope_source.append((relpath, lang))
            continue

        if lang is None:
            result.rejected[relpath] = UNRECOGNIZED
            continue
        if _matches_any(relpath, scope.generated_globs):
            result.rejected[relpath] = GENERATED
            if lang.is_source:
                result.out_of_scope_source.append((relpath, lang))
            continue
        if not lang.is_source and not scope.include_non_source:
            result.rejected[relpath] = NON_SOURCE
            continue
        try:
            size = os.path.getsize(abspath)
        except OSError:
            # Keep the file in scope. The counter will report the read/stat
            # failure instead of silently changing the measured file set.
            pass
        else:
            if size > scope.max_file_bytes:
                result.rejected[relpath] = TOO_LARGE
                continue
        # A recognized extension does not prove a file is source text. Bucket
        # NUL-containing files here so every counter sees the same scope and
        # manifest maps cannot disagree about file membership.
        if _contains_nul(abspath):
            result.rejected[relpath] = BINARY
            continue
        result.counted.append((relpath, lang))

    return result


def _walk_filesystem(scope: Scope) -> list[str]:
    """Enumerate the tree, descending into excluded dirs except the hard-pruned.

    Excluded-but-enumerated directories still have to be listed: `discover` needs
    to see a file in `vendor/` to notice that source code was parked there.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(scope.root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in HARD_PRUNE_DIRS]
        for filename in filenames:
            abspath = os.path.join(dirpath, filename)
            found.append(os.path.relpath(abspath, scope.root).replace(os.sep, "/"))
    return found
