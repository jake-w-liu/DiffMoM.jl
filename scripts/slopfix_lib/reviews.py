"""Exact, byte-bound reviews of integrity detector false positives.

Reviews never change measured files or line/clone counts. A changed or missing
reviewed input fails closed; a new finding needs its own reviewed entry.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import replace

from . import smells


class IntegrityReviews:
    def __init__(self, root: str):
        self.long_lines: set[str] = set()
        self.smells: dict[tuple[str, int, str], str] = {}
        path = pathlib.Path(root) / ".slopfix/integrity-reviews.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or isinstance(data.get("schema_version"), bool)
                or data.get("schema_version") != 1):
            raise ValueError("invalid integrity-review schema")
        entries = data.get("reviews")
        if not isinstance(entries, list):
            raise ValueError("integrity reviews must be a list")
        directory = pathlib.Path(root).resolve()
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("integrity-review entry must be an object")
            name = entry.get("path")
            if not isinstance(name, str) or "\\" in name:
                raise ValueError("review path must be repository-relative POSIX text")
            relative = pathlib.PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("review path must stay within the repository")
            source = (directory / name).resolve()
            if not source.is_relative_to(directory):
                raise ValueError("review source resolves outside the repository")
            reason = entry.get("reason")
            digest = entry.get("sha256")
            mode = entry.get("hash_mode")
            if not isinstance(reason, str) or len(reason.strip()) < 20:
                raise ValueError("review requires a concrete rationale")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("review requires a SHA256 digest")
            if mode not in {"raw", "lf"}:
                raise ValueError("review hash_mode must be raw or lf")
            content = source.read_bytes()
            hashed = content.replace(b"\r\n", b"\n") if mode == "lf" else content
            if hashlib.sha256(hashed).hexdigest() != digest:
                raise ValueError(f"stale integrity review: {name}")
            rule = entry.get("rule")
            line = entry.get("line")
            if line is not None and (isinstance(line, bool) or not isinstance(line, int)):
                raise ValueError("review line must be an integer")
            if not isinstance(rule, str):
                raise ValueError("review requires an integrity rule")
            key = (name, line, rule)
            if key in seen:
                raise ValueError("duplicate integrity review")
            seen.add(key)
            if rule == "long-line-introduced":
                if mode != "raw" or line is not None or not name.startswith("test/fixtures/"):
                    raise ValueError("long-line reviews require an exact archival fixture")
                self.long_lines.add(name)
            elif rule == "placeholder-implementation":
                if isinstance(line, bool) or not isinstance(line, int) or line < 1:
                    raise ValueError("smell review requires one positive source line")
                if line > len(content.splitlines()):
                    raise ValueError("review line is outside the source")
                self.smells[key] = reason
            else:
                raise ValueError("unsupported integrity-review rule")

    def apply(self, hit: smells.Hit) -> smells.Hit:
        reason = self.smells.get((hit.path, hit.lineno, hit.rule))
        if reason is None:
            return hit
        return replace(hit, severity=smells.ADVISORY,
                       message=f"Reviewed capability boundary: {reason}")
