"""Token-level duplicate detection (type-1 and type-2 clones).

Finds the "fourteen date formatters" case: blocks that are structurally
identical but use different identifier and literal names, which plain text or
line-diff tools miss and which is the dominant shape of AI-generated
duplication.

Method: normalise each file to a token stream (identifiers -> `V`, numbers ->
`N`, strings -> `S`, keywords and punctuation kept verbatim), hash every sliding
window of `window` tokens, group windows that share a hash, then extend each
group greedily to its maximal length so one long clone is reported once rather
than as dozens of overlapping windows.

Deliberate limits, stated so the output is not over-trusted:
  * Type-3 clones (duplicated logic with inserted or reordered statements) are
    not detected. A concept census (`concepts.py`) covers that case by name.
  * A hash collision would report a false clone. Windows are verified by
    comparing the actual token slices before a group is emitted, so collisions
    cannot produce a false positive.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field

from . import counting
from .langs import Language

# Multi-character operators kept as single tokens so structure survives
# normalisation. Sorted longest-first at use.
_OPERATORS: tuple[str, ...] = (
    ">>>=", "<<=", ">>=", ">>>", "...", "===", "!==", "**=", "&&=", "||=", "??=",
    "<=>", "->>",
    "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "::", "->", "=>", "++", "--",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "**", "//", "..",
)
_OPERATOR_CHARS = frozenset("+-*/%=<>!&|^~?:.@#$")
_IDENT_START = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$"
)
_IDENT_CONT = _IDENT_START | frozenset("0123456789")
_DIGITS = frozenset("0123456789")


def _is_ident_start(ch: str) -> bool:
    """Identifier start for the languages in the model, including Julia Unicode."""
    return ch in "_$" or ch.isalpha()


def _is_ident_cont(ch: str) -> bool:
    return (
        _is_ident_start(ch)
        or ch.isdigit()
        or unicodedata.category(ch).startswith("M")
    )


@dataclass
class Unit:
    """One file reduced to a normalised token stream."""

    path: str
    language: str
    tokens: list[str] = field(default_factory=list)
    # tokens[i] came from source line lines[i]; used to report clone locations.
    lines: list[int] = field(default_factory=list)


@dataclass
class Occurrence:
    path: str
    start_line: int
    end_line: int


@dataclass
class CloneGroup:
    token_count: int
    occurrences: list[Occurrence]

    @property
    def duplicate_lines(self) -> int:
        """Lines removable by keeping one copy of this block.

        Counts distinct (file, line) positions rather than summing span lengths.
        Two occurrences in the same file can share lines, and summing spans then
        reports more removable lines than the file contains.
        """
        if len(self.occurrences) < 2:
            return 0
        # Keep the largest copy; every line of the rest is a removal candidate.
        ordered = sorted(
            self.occurrences, key=lambda o: o.end_line - o.start_line, reverse=True
        )
        positions: set[tuple[str, int]] = set()
        for occ in ordered[1:]:
            for lineno in range(occ.start_line, occ.end_line + 1):
                positions.add((occ.path, lineno))
        # Lines the retained copy occupies are not removable.
        primary = ordered[0]
        for lineno in range(primary.start_line, primary.end_line + 1):
            positions.discard((primary.path, lineno))
        return len(positions)

    def to_json(self) -> dict:
        return {
            "token_count": self.token_count,
            "copies": len(self.occurrences),
            "removable_lines": self.duplicate_lines,
            "occurrences": [
                {"path": o.path, "start_line": o.start_line, "end_line": o.end_line}
                for o in self.occurrences
            ],
        }


def tokenize(path: str, text: str, lang: Language) -> Unit:
    """Normalise source into a token stream, dropping comments entirely.

    Comments are skipped rather than tokenised: two blocks that differ only in
    their comments are the same duplicate, and AI-generated copies routinely
    carry differently-worded comments.

    The shared source scanner performs comment/string/character classification
    first. This matters for Julia, where `'#'` is a character literal rather than
    the start of a comment and `A'` is an adjoint rather than a quote.
    """
    unit = Unit(path=path, language=lang.name)
    strings = tuple(sorted(lang.strings, key=lambda s: -len(s.open_seq)))
    keywords = lang.keywords

    scanned, _ = counting.scan(text, lang, path)
    for source_line in scanned:
        if source_line.kind != counting.CODE or not source_line.code_text:
            continue
        code = source_line.code_text
        i = 0
        n = len(code)
        while i < n:
            ch = code[i]
            if ch.isspace():
                i += 1
                continue

            string_end = _skip_string(code, i, strings)
            if string_end is not None:
                unit.tokens.append("S")
                unit.lines.append(source_line.lineno)
                i = string_end
                continue

            if _is_ident_start(ch):
                start = i
                i += 1
                while i < n and _is_ident_cont(code[i]):
                    i += 1
                word = code[start:i]
                unit.tokens.append(word if word in keywords else "V")
                unit.lines.append(source_line.lineno)
                continue

            if ch in _DIGITS:
                i += 1
                while i < n and (code[i] in _DIGITS or code[i] in "._aAbBcCdDeEfFxXoO"):
                    i += 1
                unit.tokens.append("N")
                unit.lines.append(source_line.lineno)
                continue

            if ch in _OPERATOR_CHARS:
                for op in _OPERATORS:
                    if code.startswith(op, i):
                        unit.tokens.append(op)
                        unit.lines.append(source_line.lineno)
                        i += len(op)
                        break
                else:
                    unit.tokens.append(ch)
                    unit.lines.append(source_line.lineno)
                    i += 1
                continue

            unit.tokens.append(ch)
            unit.lines.append(source_line.lineno)
            i += 1

    return unit


def _skip_string(text: str, i: int, strings) -> int | None:
    """Index just past a string literal starting at `i`, or None."""
    for spec in strings:
        closer = spec.close_seq
        if spec.raw_hash:
            prefix = spec.open_seq[:-1]
            if not text.startswith(prefix, i):
                continue
            if prefix[0] in _IDENT_CONT and i > 0 and text[i - 1] in _IDENT_CONT:
                continue
            k = i + len(prefix)
            hashes = 0
            while k < len(text) and text[k] == "#":
                hashes += 1
                k += 1
            if k >= len(text) or text[k] != '"':
                continue
            closer = '"' + "#" * hashes
            j = k + 1
        else:
            if not text.startswith(spec.open_seq, i):
                continue
            if spec.open_seq[0] in _IDENT_CONT and i > 0 and text[i - 1] in _IDENT_CONT:
                continue
            j = i + len(spec.open_seq)
        if spec.raw_delim:
            k = i + len(spec.open_seq)
            start = k
            while k < len(text) and text[k] not in "(\n":
                k += 1
            if k >= len(text) or text[k] != "(":
                continue
            closer = ")" + text[start:k] + '"'
            j = k + 1
        n = len(text)
        clen = len(closer)
        while j < n:
            if spec.escape and text[j] == "\\":
                j += 2
                continue
            if not spec.multiline and text[j] == "\n":
                # Unterminated single-line literal: stop at the newline so one
                # malformed line cannot consume the rest of the file.
                return j
            if text.startswith(closer, j):
                if spec.doubling and text.startswith(closer, j + clen):
                    j += 2 * clen
                    continue
                return j + clen
            j += 1
        return n
    return None


def _window_hash(tokens: list[str], start: int, window: int) -> bytes:
    joined = "\x00".join(tokens[start : start + window])
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).digest()


def find_clones(
    units: list[Unit],
    window: int = 60,
    min_tokens: int = 60,
    max_groups: int = 400,
    truncated: list[bool] | None = None,
) -> list[CloneGroup]:
    """Group duplicated token runs, longest first.

    `window` is the detection granularity; `min_tokens` is the smallest run that
    gets reported. Runs shorter than `window` can never be found, so callers
    should keep `window <= min_tokens`.

    Pass `truncated` to learn whether the `max_groups` cap was hit. A capped run
    understates both the group count and the removable-line estimate, and a cap
    that is not reported reads as "this is all the duplication there is".
    """
    if window < 8:
        raise ValueError("window must be at least 8 tokens to avoid noise")
    if min_tokens < window:
        raise ValueError("min_tokens must be >= window")

    # hash -> list of (unit index, token offset)
    buckets: dict[bytes, list[tuple[int, int]]] = {}
    for unit_idx, unit in enumerate(units):
        limit = len(unit.tokens) - window
        for offset in range(limit + 1):
            buckets.setdefault(_window_hash(unit.tokens, offset, window), []).append(
                (unit_idx, offset)
            )

    consumed: list[set[int]] = [set() for _ in units]
    groups: list[CloneGroup] = []

    # Process the most-repeated windows first so the biggest wins are reported
    # even when max_groups truncates the list.
    ordered = sorted(
        (b for b in buckets.values() if len(b) > 1),
        key=lambda positions: -len(positions),
    )

    for positions in ordered:
        if len(groups) >= max_groups:
            if truncated is not None:
                truncated.append(True)
            break
        fresh = [
            (unit_idx, offset)
            for unit_idx, offset in positions
            if consumed[unit_idx].isdisjoint(range(offset, offset + window))
        ]
        if len(fresh) < 2:
            continue
        # Verify the windows really are identical; a hash collision must not be
        # reportable as duplication.
        reference = units[fresh[0][0]].tokens[fresh[0][1] : fresh[0][1] + window]
        verified = [
            (unit_idx, offset)
            for unit_idx, offset in fresh
            if units[unit_idx].tokens[offset : offset + window] == reference
        ]
        if len(verified) < 2:
            continue

        length = _extend(units, verified, window, consumed)
        if length < min_tokens:
            continue

        # Drop occurrences that overlap an already-accepted one in the same file.
        # A long uniform block (a lookup table, a list of constants) matches
        # itself at every offset, which would otherwise be reported as hundreds
        # of "copies" that are really one repetitive region.
        accepted: list[tuple[int, int]] = []
        taken: dict[int, list[tuple[int, int]]] = {}
        for unit_idx, offset in sorted(verified):
            end = offset + length
            ranges = taken.setdefault(unit_idx, [])
            if any(offset < r_end and r_start < end for r_start, r_end in ranges):
                continue
            ranges.append((offset, end))
            accepted.append((unit_idx, offset))
        if len(accepted) < 2:
            continue

        occurrences: list[Occurrence] = []
        for unit_idx, offset in accepted:
            unit = units[unit_idx]
            for pos in range(offset, offset + length):
                consumed[unit_idx].add(pos)
            occurrences.append(
                Occurrence(
                    path=unit.path,
                    start_line=unit.lines[offset],
                    end_line=unit.lines[min(offset + length - 1, len(unit.lines) - 1)],
                )
            )
        groups.append(CloneGroup(token_count=length, occurrences=occurrences))

    groups.sort(key=lambda g: (-g.duplicate_lines, -g.token_count))
    return groups


def _extend(
    units: list[Unit],
    positions: list[tuple[int, int]],
    window: int,
    consumed: list[set[int]],
) -> int:
    """Grow a verified window while it agrees and stays outside prior groups."""
    length = window
    while True:
        next_tokens = set()
        for unit_idx, offset in positions:
            tokens = units[unit_idx].tokens
            index = offset + length
            if index >= len(tokens) or index in consumed[unit_idx]:
                return length
            next_tokens.add(tokens[index])
            if len(next_tokens) > 1:
                return length
        length += 1


def summarise(groups: list[CloneGroup]) -> dict:
    """Totals for the census report.

    The removable-line estimate is a union of distinct (file, line) positions,
    not a sum of per-group figures. Two clone groups can cover different tokens
    on the same physical line, so summing them double-counts and can report more
    removable lines than the codebase has.
    """
    files: set[str] = set()
    removable_positions: set[tuple[str, int]] = set()
    for group in groups:
        for occ in group.occurrences:
            files.add(occ.path)
        if len(group.occurrences) < 2:
            continue
        # Keep the largest copy; every line of the rest is a removal candidate.
        ordered = sorted(
            group.occurrences, key=lambda o: o.end_line - o.start_line, reverse=True
        )
        for occ in ordered[1:]:
            for lineno in range(occ.start_line, occ.end_line + 1):
                removable_positions.add((occ.path, lineno))
    return {
        "clone_groups": len(groups),
        "files_involved": len(files),
        "removable_lines_estimate": len(removable_positions),
    }
