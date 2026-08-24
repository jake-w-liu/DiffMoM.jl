"""Source scanning: line classification plus comment/string stripping.

One scanner serves three consumers — the line counter, the concept census and
the smell scan — so there is a single place where "is this text really code"
is decided.

Two counters are available and they are never mixed:

`scc`      - the canonical counter. Slopfix-style engagements quote reduction
             against scc's `Code` column (non-blank, non-comment lines), so scc
             is the default whenever it is on PATH.
`builtin`  - a stdlib fallback with the same *definition* but its own identity.
             It exists so the method still runs where scc cannot be installed.

Every count records a *counter identity* string. `slopfix measure` refuses to
compare a baseline and a re-measure taken with different identities, because a
reduction number produced by two different counters is meaningless.

Builtin counter rules (exact, so the number is reproducible and auditable):

1. A physical line whose content is entirely whitespace is `blank` — including
   lines inside a multi-line string or block comment.
2. A non-blank line is `code` if any non-whitespace character on it lies outside
   a comment. A trailing comment does not stop a line from being code.
3. A non-blank line is `comment` only if every non-whitespace character on it
   lies inside a comment.
4. String literal contents are code, not comments.
5. Python-style docstrings count as comments, and only in true docstring
   position: a triple-quoted string opening at the first non-whitespace
   character of a line, where either no code has been seen in the file yet
   (module docstring) or the previous code line ended with `:` (def/class body).
   A triple-quoted string anywhere else is a string literal, i.e. code.
6. An unterminated single-line string is reported as a parse warning and the
   scanner resets to normal state at end of line, so one malformed line cannot
   corrupt the classification of the rest of the file.
7. *Every* bare string in docstring position counts as documentation, not only
   the first. Python's `ast` calls the second consecutive module-level string an
   expression rather than a docstring; a differential fuzz against `ast` shows
   this is the one systematic divergence (1 case in 600 generated files). It is
   deliberate: a bare string expression does nothing at runtime, so treating it
   as prose can never hide executable code, and it preserves the property the
   whole definition exists for -- that deleting documentation cannot lower the
   count.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import ClassVar

from .langs import BlockComment, Language, StringSpec

BUILTIN_COUNTER_VERSION = 2
BUILTIN_COUNTER_ID = f"slopfix-builtin/{BUILTIN_COUNTER_VERSION}"
_SCC_RUN_TIMEOUT_SECONDS = 900

BLANK = "blank"
COMMENT = "comment"
CODE = "code"
# Internal only, never returned by `scan`. Marks lines the scanner knows are
# inside a docstring, so the Julia post-pass gets the exact region instead of
# reconstructing it from `COMMENT` -- which cannot tell a `#` code comment from
# a Markdown `# Examples` heading inside the docstring itself.
_DOCLINE = "docstring"

_IDENT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
_PY_STRING_PREFIXES = frozenset("rRbBuUfF")


# Placeholder written into `code_text` in place of a string literal's contents,
# so pattern matching over code cannot match text that only appears in a string.
_STRING_PLACEHOLDER = '""'
# Same idea for regex literals: keep the line as code without exposing its
# contents to the pattern matchers in concepts.py / smells.py.
_REGEX_PLACEHOLDER = "/re/"


@dataclass
class ScannedLine:
    lineno: int
    kind: str
    # The line with comments removed and string contents replaced. Empty for
    # blank lines, comment lines, and continuation lines of a multi-line string.
    code_text: str
    # The original line, stripped of leading/trailing whitespace.
    stripped: str


@dataclass
class FileCount:
    path: str
    language: str
    lines: int = 0
    blanks: int = 0
    comments: int = 0
    code: int = 0
    # Total characters on code lines, and the longest single code line. Both
    # feed the code-golf integrity check.
    code_chars: int = 0
    max_code_line: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "lines": self.lines,
            "blanks": self.blanks,
            "comments": self.comments,
            "code": self.code,
            "code_chars": self.code_chars,
            "max_code_line": self.max_code_line,
        }


# --- the scanner -------------------------------------------------------------


class _State:
    NORMAL = 0
    BLOCK_COMMENT = 1
    STRING = 2
    DOCSTRING = 3


class _Syntax:
    """Per-language scanning tables, computed once per language."""

    _cache: ClassVar[dict[str, _Syntax]] = {}

    def __init__(self, lang: Language) -> None:
        self.lang = lang
        self.blocks = tuple(sorted(lang.block_comments, key=lambda b: -len(b.open_seq)))
        self.line_comments = tuple(sorted(lang.line_comments, key=len, reverse=True))
        self.strings = tuple(sorted(lang.strings, key=lambda s: -len(s.open_seq)))
        interesting: set[str] = set()
        for block in self.blocks:
            interesting.add(block.open_seq[0])
        for prefix in self.line_comments:
            interesting.add(prefix[0])
        for spec in self.strings:
            interesting.add(spec.open_seq[0])
        # Every character that triggers a specialised check must be listed here,
        # or the fast path consumes it first and the check never runs. That is how
        # `'"'` in Julia used to open a phantom string: `'` is not a Julia string
        # delimiter, so it was not "interesting" and the char-literal branch was
        # unreachable.
        if lang.char_literals:
            interesting.add("'")
        if lang.regex_literals:
            interesting.add("/")
        self.interesting = frozenset(interesting)

    @classmethod
    def get(cls, lang: Language) -> _Syntax:
        cached = cls._cache.get(lang.name)
        if cached is None:
            cached = cls(lang)
            cls._cache[lang.name] = cached
        return cached


def _find_string_close(line: str, start: int, closer: str, spec: StringSpec) -> int:
    """Index just past `closer`, or -1 if the literal continues past this line."""
    i = start
    n = len(line)
    clen = len(closer)
    while i < n:
        if spec.escape and line[i] == "\\":
            i += 2
            continue
        if line.startswith(closer, i):
            if spec.doubling and line.startswith(closer, i + clen):
                i += 2 * clen
                continue
            return i + clen
        i += 1
    return -1


def _match_string_opener(
    line: str, i: int, specs: tuple[StringSpec, ...]
) -> tuple[StringSpec, str, int] | None:
    """Match a string opener at `line[i]`.

    Returns (spec, closing delimiter, index just past the opener). The closer is
    computed rather than read from the spec because raw forms embed a
    caller-chosen delimiter in the opener.
    """
    for spec in specs:
        if spec.raw_hash:
            # Rust: r"..." / r#"..."# / br##"..."##. The hashes sit before
            # the quote, so `startswith('r"')` cannot recognise the hashed
            # forms; match the prefix first, then consume the caller's hash run.
            prefix = spec.open_seq[:-1]
            if not line.startswith(prefix, i):
                continue
            if prefix[0] in _IDENT_CHARS and i > 0 and line[i - 1] in _IDENT_CHARS:
                continue
            j = i + len(prefix)
            hashes = 0
            while j < len(line) and line[j] == "#":
                hashes += 1
                j += 1
            if j >= len(line) or line[j] != '"':
                continue
            return spec, '"' + "#" * hashes, j + 1
        if not line.startswith(spec.open_seq, i):
            continue
        # `R"`, `@"` and other prefixed forms must not be the tail of a longer
        # identifier.
        if spec.open_seq[0] in _IDENT_CHARS and i > 0 and line[i - 1] in _IDENT_CHARS:
            continue
        if spec.raw_delim:
            # C++: R"delim(...)delim"
            j = i + len(spec.open_seq)
            delim_start = j
            while j < len(line) and line[j] != "(":
                j += 1
            if j >= len(line):
                continue
            return spec, ")" + line[delim_start:j] + '"', j + 1
        return spec, spec.close_seq, i + len(spec.open_seq)
    return None


# After these, a `/` begins a regex literal rather than a division. This is the
# standard lexer heuristic: division can only follow a value, so anything that
# cannot end an expression must be followed by a regex.
_REGEX_PRECEDERS = frozenset("([{,;:=!&|?+-*%^<>~")
_REGEX_PRECEDING_KEYWORDS = frozenset(
    """return typeof instanceof in of new delete void case do else yield await
    throw""".split()
)


def _starts_regex(parts: list[str]) -> bool:
    """Decide whether a `/` at this point opens a regex literal.

    `parts` is the code text accumulated for the current line so far. A regex is
    only possible where a value is not, so the test is on the last significant
    character: after `(`, `,`, `=`, `:`, `return` and friends it is a regex; after
    an identifier, number, `)` or `]` it is division.
    """
    text = "".join(parts).rstrip()
    if not text:
        return True  # start of a line or statement
    last = text[-1]
    if last in _REGEX_PRECEDERS:
        return True
    if last in _IDENT_CHARS:
        # Could be `x / y` (division) or `return /re/` (regex). Look at the word.
        index = len(text)
        while index > 0 and text[index - 1] in _IDENT_CHARS:
            index -= 1
        return text[index:] in _REGEX_PRECEDING_KEYWORDS
    return False


def _skip_regex(line: str, start: int) -> int:
    """Index just past a `/.../flags` literal, or -1 if it does not close.

    A `/` inside a character class does not terminate the literal, which is the
    case that matters: `/[a-z/]/` and `/[!$&'()]/` both appear in real code.
    """
    i = start
    n = len(line)
    in_class = False
    while i < n:
        ch = line[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "/" and not in_class:
            i += 1
            while i < n and line[i].isalpha():  # trailing flags: gimsuy
                i += 1
            return i
        i += 1
    return -1


def _has_line_continuation(line: str) -> bool:
    """True when `line` ends with an odd number of backslashes.

    A single-quoted or double-quoted literal may then continue onto the next
    physical line -- Julia, C, C++ and others all allow it. Parity matters: a
    literal ending in an escaped backslash (`\\\\`) is *not* continued.
    """
    trailing = len(line) - len(line.rstrip("\\"))
    return trailing % 2 == 1


def _char_literal_possible(parts: list[str]) -> bool:
    """True when a `'` here could open a char literal rather than be an adjoint.

    Julia's `'` is postfix transpose as well as a char delimiter, and shape alone
    cannot separate them: in `(A')'` the middle `')'` looks exactly like a
    one-character literal. Position can. Transpose follows a *value* -- an
    identifier, `)`, `]`, `}`, a digit, or another `'` -- while a literal only
    appears where a value is expected. Same reasoning as regex-versus-division in
    JavaScript.

    Getting this wrong is not cosmetic: consuming the `)` of `(A')'` as literal
    content leaves the bracket depth permanently unbalanced, which silently
    disables docstring detection for the remainder of the file.
    """
    text = "".join(parts).rstrip()
    if not text:
        return True  # start of a line or statement
    return text[-1] not in _IDENT_CHARS and text[-1] not in ")]}.'\""


def _match_julia_char(line: str, i: int) -> int:
    """Index just past a Julia char literal at `line[i]`, or -1 if it is not one.

    Shape check only; the caller must first establish that a literal is possible
    at this position via `_char_literal_possible`. A literal holds exactly one
    character or one escape sequence, which is all Julia allows. Returning -1
    leaves the `'` to be treated as an operator, so a mis-read costs one
    character rather than swallowing the rest of the file as a string.
    """
    n = len(line)
    if i + 2 < n and line[i + 1] not in "\\'" and line[i + 2] == "'":
        return i + 3  # 'a', '#', '"'
    if i + 1 < n and line[i + 1] == "\\":
        # Longest escape is '\UXXXXXXXX'; bound the search so an adjoint followed
        # by a backslash cannot run away.
        limit = min(n, i + 14)
        j = i + 2
        while j < limit:
            if line[j] == "'":
                return j + 1
            j += 1
    return -1


def _match_docstring(
    line: str, first_idx: int, specs: tuple[StringSpec, ...], triple_only: bool = False
) -> tuple[StringSpec, str, int] | None:
    """Match a docstring opener at line start, allowing an r/b/u/f prefix.

    Triple-quoted forms are tried first so `\"\"\"` is not read as an empty `\"`
    followed by more. Single-quoted docstrings are also matched: they are legal
    and common in older code (`def f():` then `\"Short doc.\"`), and Python's own
    `ast` reports them as docstrings, so classifying them as code would disagree
    with the language.
    """
    prefix_end = first_idx
    while (
        prefix_end < len(line)
        and line[prefix_end] in _PY_STRING_PREFIXES
        and prefix_end - first_idx < 2
    ):
        prefix_end += 1
    ordered = sorted(specs, key=lambda s: -len(s.open_seq))
    for start in (first_idx, prefix_end):
        for spec in ordered:
            # `triple_only` languages restrict docstrings to the triple-quoted
            # form. Julia allows `"short doc"` before a definition, but a
            # line-starting single-quoted string is far more often a *value* --
            # `"-all_load"` as an `if` branch result, or a backtick command --
            # and misreading those as prose loses real code lines.
            if triple_only and (len(spec.open_seq) < 3 or spec.open_seq[0] != '"'):
                continue
            if line.startswith(spec.open_seq, start):
                return spec, spec.close_seq, start + len(spec.open_seq)
    return None


def scan(text: str, lang: Language, path: str = "") -> tuple[list[ScannedLine], list[str]]:
    """Classify every physical line and return its code-only text.

    Returns (lines, warnings). Warnings name mis-parses rather than hiding them,
    so a caller can tell when the builtin scanner is unreliable for a file.
    """
    syntax = _Syntax.get(lang)
    warnings: list[str] = []
    scanned: list[ScannedLine] = []

    state = _State.NORMAL
    active_spec: StringSpec | None = None
    active_closer = ""
    active_block: BlockComment | None = None
    block_depth = 0

    seen_code = False
    prev_code_ended_with_colon = False
    # Open brackets carried across lines. A bare string opening a continuation
    # line is an argument or an element, never a docstring, so docstring
    # detection requires depth 0.
    bracket_depth = 0

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.expandtabs(1) if "\t" in raw_line else raw_line
        stripped = line.strip()
        if not stripped:
            scanned.append(ScannedLine(lineno, BLANK, "", ""))
            continue

        # A shebang on line 1 is a comment in every language that allows one,
        # including those where `#` is not otherwise a comment marker (it is a
        # hashbang comment in ES2023). Counting it as code would overstate every
        # executable script by one line.
        if lineno == 1 and state == _State.NORMAL and stripped.startswith("#!"):
            scanned.append(ScannedLine(lineno, COMMENT, "", stripped))
            continue

        saw_code = False
        saw_comment = False
        saw_doc = False
        parts: list[str] = []
        i = 0
        n = len(line)
        first_idx = n - len(line.lstrip())

        while i < n:
            ch = line[i]

            if state in (_State.STRING, _State.DOCSTRING):
                assert active_spec is not None
                end = _find_string_close(line, i, active_closer, active_spec)
                if state == _State.STRING:
                    saw_code = True
                else:
                    saw_comment = True
                    saw_doc = True
                if end == -1:
                    if not active_spec.multiline:
                        if active_spec.escape and _has_line_continuation(line):
                            # `"text \` continues the literal onto the next line.
                            # Keep the state; this is valid, not a mis-parse.
                            pass
                        else:
                            warnings.append(
                                f"{path}:{lineno}: unterminated single-line string; "
                                "scanner reset at end of line"
                            )
                            state = _State.NORMAL
                            active_spec = None
                            # Bracket tracking is no longer trustworthy after a
                            # mis-parse, and a stuck depth would silently disable
                            # docstring detection for the rest of the file.
                            bracket_depth = 0
                    i = n
                else:
                    i = end
                    state = _State.NORMAL
                    active_spec = None
                continue

            if state == _State.BLOCK_COMMENT:
                assert active_block is not None
                saw_comment = True
                if active_block.nestable and line.startswith(active_block.open_seq, i):
                    block_depth += 1
                    i += len(active_block.open_seq)
                    continue
                if line.startswith(active_block.close_seq, i):
                    block_depth -= 1
                    i += len(active_block.close_seq)
                    if block_depth <= 0:
                        state = _State.NORMAL
                        active_block = None
                    continue
                i += 1
                continue

            # --- NORMAL ---
            if ch.isspace():
                parts.append(" ")
                i += 1
                continue

            # The docstring check must precede the fast path. A prefixed
            # docstring (`r"""`, `f"""`) starts with a letter, which the fast path
            # would consume -- leaving `i` past `first_idx`, so this check could
            # never fire and every prefixed docstring would read as plain code.
            if lang.docstring_style and i == first_idx and bracket_depth == 0:
                if lang.docstring_style == "python":
                    # Inside the block: at file top, or after a `def`/`class:`.
                    in_position = not seen_code or prev_code_ended_with_colon
                else:
                    # Julia: the docstring precedes what it documents, so any
                    # statement-position bare string qualifies.
                    #
                    # A tried-and-rejected refinement: also require that the
                    # previous code line did not end on an operator, to catch
                    # `@test f(x) ==` followed by a triple-quoted expectation.
                    # Measured against Julia Base it made things worse -- total
                    # absolute error 83 lines versus 63 -- because Julia
                    # documents bare operators (`const \u2260 = !=`, then a
                    # docstring), and those lines end on operators too. It also
                    # erred toward counting docstrings as code, which is the
                    # unsafe direction: it would let deleting docstrings lower
                    # the number.
                    in_position = True
                if in_position:
                    match = _match_docstring(
                        line, first_idx, syntax.strings,
                        triple_only=lang.docstring_style == "julia",
                    )
                    if match is not None:
                        active_spec, active_closer, i = match
                        state = _State.DOCSTRING
                        # Mark it now. A line holding only the opening delimiter
                        # -- `\"\"\"` alone, which is the common multi-line form in
                        # both Python and Julia -- ends here, so the docstring
                        # branch never runs for it and the line would otherwise
                        # fall through and be counted as code.
                        saw_comment = True
                        saw_doc = True
                        continue

            if ch not in syntax.interesting:
                # Fast path: nothing can start here, so take the whole run.
                start = i
                i += 1
                while i < n and line[i] not in syntax.interesting and not line[i].isspace():
                    i += 1
                chunk = line[start:i]
                parts.append(chunk)
                saw_code = True
                # Brackets are not "interesting" characters, so this run is where
                # most of them are consumed; depth has to be updated here too or
                # it would never move.
                bracket_depth = max(
                    0,
                    bracket_depth
                    + sum(chunk.count(c) for c in "([{")
                    - sum(chunk.count(c) for c in ")]}"),
                )
                continue

            matched = False
            for block in syntax.blocks:
                if line.startswith(block.open_seq, i):
                    state = _State.BLOCK_COMMENT
                    active_block = block
                    block_depth = 1
                    saw_comment = True
                    i += len(block.open_seq)
                    matched = True
                    break
            if matched:
                continue

            for prefix in syntax.line_comments:
                if line.startswith(prefix, i):
                    saw_comment = True
                    i = n
                    matched = True
                    break
            if matched:
                continue

            # Regex literals are checked after the comment openers, so `//` and
            # `/*` stay comments, and before strings, so a quote inside a
            # character class cannot open a phantom string literal.
            if lang.regex_literals and ch == "/" and _starts_regex(parts):
                end = _skip_regex(line, i + 1)
                saw_code = True
                if end == -1:
                    # Unterminated on this line. A regex cannot span lines, so
                    # treat the rest as code rather than leaking scanner state.
                    parts.append(line[i:].strip())
                    i = n
                else:
                    parts.append(_REGEX_PLACEHOLDER)
                    i = end
                continue

            # Char literals are checked before strings so a quote or `#` inside
            # one -- `'"'`, `'#'` -- cannot open a string or a comment.
            if lang.char_literals and ch == "'":
                end = (
                    _match_julia_char(line, i)
                    if _char_literal_possible(parts)
                    else -1
                )
                if end != -1:
                    # Downstream census/smell scans must not see the contents of
                    # a character literal any more than they see string
                    # contents. Using the shared placeholder also lets clone
                    # tokenization treat Julia chars consistently with strings.
                    parts.append(_STRING_PLACEHOLDER)
                    saw_code = True
                    i = end
                    continue
                # Not a literal: it is the adjoint operator, i.e. ordinary code.
                parts.append(ch)
                saw_code = True
                i += 1
                continue

            opener = _match_string_opener(line, i, syntax.strings)
            if opener is not None:
                active_spec, active_closer, i = opener
                state = _State.STRING
                saw_code = True
                parts.append(_STRING_PLACEHOLDER)
                continue

            if ch in "([{":
                bracket_depth += 1
            elif ch in ")]}":
                bracket_depth = max(0, bracket_depth - 1)
            parts.append(ch)
            saw_code = True
            i += 1

        # A non-blank line with neither flag cannot occur, but defaulting to code
        # means an unforeseen case never silently shrinks the count.
        if saw_doc and not saw_code:
            scanned.append(ScannedLine(lineno, _DOCLINE, "", stripped))
            continue
        if saw_code or not saw_comment:
            code_text = "".join(parts).strip()
            scanned.append(ScannedLine(lineno, CODE, code_text, stripped))
            seen_code = True
            # Test the code, not the raw line: a `def f():` carrying a trailing
            # lint-suppression comment still opens a block, and its docstring
            # must still be recognised as one.
            prev_code_ended_with_colon = code_text.endswith(":")
        else:
            scanned.append(ScannedLine(lineno, COMMENT, "", stripped))

    if state == _State.BLOCK_COMMENT:
        warnings.append(f"{path}: unterminated block comment at end of file")
    elif state in (_State.STRING, _State.DOCSTRING):
        warnings.append(f"{path}: unterminated string literal at end of file")

    # `_DOCLINE` is internal; callers only ever see blank/comment/code.
    for pos, line in enumerate(scanned):
        if line.kind == _DOCLINE:
            scanned[pos] = ScannedLine(line.lineno, COMMENT, "", line.stripped)

    return scanned, warnings


def count_text(text: str, lang: Language, path: str = "") -> FileCount:
    """Classify `text` into blank / comment / code line counts."""
    scanned, warnings = scan(text, lang, path)
    result = FileCount(path=path, language=lang.name, warnings=warnings)
    for line in scanned:
        result.lines += 1
        if line.kind == BLANK:
            result.blanks += 1
        elif line.kind == COMMENT:
            result.comments += 1
        else:
            result.code += 1
            result.code_chars += len(line.stripped)
            result.max_code_line = max(result.max_code_line, len(line.stripped))
    return result


def read_source(path: str, max_bytes: int) -> tuple[str | None, str | None]:
    """Read a file as text. Returns (text, skip_reason); exactly one is None."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return None, f"cannot stat ({exc})"
    if size > max_bytes:
        return None, f"skipped, {size} bytes exceeds max-file-bytes ({max_bytes})"
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        return None, f"cannot read ({exc})"
    if b"\x00" in data:
        return None, "skipped, binary"
    return data.decode("utf-8", errors="replace").lstrip("﻿"), None


class SourceReadError(RuntimeError):
    """An in-scope source file could not be counted without guessing."""


def count_file(path: str, lang: Language, max_bytes: int) -> FileCount:
    """Count one file, failing rather than treating unreadable source as zero."""
    text, reason = read_source(path, max_bytes)
    if text is None:
        raise SourceReadError(f"{path}: {reason}")
    return count_text(text, lang, path=path)


# --- scc adapter -------------------------------------------------------------


class SccUnavailable(RuntimeError):
    pass


class SccOutputError(RuntimeError):
    pass


def scc_path() -> str | None:
    return shutil.which("scc")


def scc_identity() -> str:
    """Stable identity string for the installed scc, e.g. `scc/3.6.0`."""
    exe = scc_path()
    if exe is None:
        raise SccUnavailable("scc is not on PATH")
    try:
        proc = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SccUnavailable(f"could not run `scc --version`: {exc}") from exc
    output = (proc.stdout or proc.stderr).strip().splitlines()
    if proc.returncode != 0:
        raise SccUnavailable(
            f"`scc --version` exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()[:300]}"
        )
    label = output[0].strip() if output else ""
    if not label:
        raise SccUnavailable("`scc --version` produced no version identity")
    token = label.split()[-1]
    return f"scc/{token}"


def _pick(mapping: dict, *names: str):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _wire_int(mapping: dict, names: tuple[str, ...], *, required: bool) -> int:
    """Read a non-negative JSON integer without coercing malformed wire data."""
    present = next((name for name in names if name in mapping), None)
    if present is None:
        if not required:
            return 0
        raise SccOutputError(
            f"scc JSON is missing every expected key {names!r}; "
            f"observed keys: {sorted(mapping)!r}. "
            "Re-run with --counter builtin, or report this message upstream."
        )
    value = mapping[present]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SccOutputError(
            f"scc JSON key {present!r} must be a non-negative integer: {value!r}"
        )
    return value


def _require_int(mapping: dict, *names: str) -> int:
    """Read a key the reduction number depends on. Missing is fatal."""
    return _wire_int(mapping, names, required=True)


def _optional_int(mapping: dict, *names: str) -> int:
    """Read a reporting-only key. Absence degrades the report, not the number."""
    return _wire_int(mapping, names, required=False)


def parse_scc_json(payload: object, root: str) -> list[FileCount]:
    """Normalise scc's `--by-file -f json` output into FileCount rows.

    scc's Go structs carry no json tags, so the wire format uses the Go field
    names verbatim: a top-level array of language summaries, each with `Name` and
    `Files`, and each file with `Location`, `Lines`, `Code`, `Comment`, `Blank`.
    Parsing is defensive: if those spellings ever change, this raises
    SccOutputError naming the keys it actually saw instead of silently reporting
    zero lines and therefore a zero reduction.
    """
    if not isinstance(payload, list):
        raise SccOutputError(
            f"expected a JSON array of language summaries, got {type(payload).__name__}"
        )

    counts: list[FileCount] = []
    observed_paths: set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise SccOutputError("scc JSON array contains a non-object element")
        lang_name = str(_pick(entry, "Name", "Language", "name") or "Unknown")
        files = _pick(entry, "Files", "files")
        if not isinstance(files, list):
            raise SccOutputError(
                f"language entry {lang_name!r} has no per-file list; "
                "was scc run without --by-file?"
            )
        for record in files:
            if not isinstance(record, dict):
                raise SccOutputError(f"language {lang_name!r} has a non-object file entry")
            location = _pick(record, "Location", "Filename", "location", "filename")
            if not isinstance(location, str) or not location:
                raise SccOutputError(
                    f"language {lang_name!r} has a file with no string location"
                )
            relpath = os.path.relpath(location, root).replace(os.sep, "/")
            if relpath in observed_paths:
                raise SccOutputError(
                    f"scc JSON reports the same file more than once: {relpath}"
                )
            observed_paths.add(relpath)
            code = _require_int(record, "Code", "code")
            blanks = _optional_int(record, "Blank", "Blanks", "blank")
            comments = _optional_int(record, "Comment", "Comments", "comment")
            lines = code + comments + blanks
            line_keys = ("Lines", "lines")
            if any(name in record for name in line_keys):
                reported_lines = _optional_int(record, *line_keys)
                if reported_lines != lines:
                    raise SccOutputError(
                        f"scc JSON line metrics for {relpath} are inconsistent: "
                        f"Lines={reported_lines}, Code+Comment+Blank={lines}"
                    )
            counts.append(
                FileCount(
                    path=relpath,
                    language=str(_pick(record, "Language", "language") or lang_name),
                    lines=lines,
                    blanks=blanks,
                    comments=comments,
                    # `Code` is the contract number, so its absence is fatal
                    # rather than reported as zero reduction.
                    code=code,
                )
            )
    return counts


def run_scc(
    root: str,
    exclude_dirs: list[str],
    respect_gitignore: bool,
) -> list[FileCount]:
    """Run `scc --by-file -f json` and normalise it into FileCount rows.

    Glob exclusions are deliberately *not* forwarded to scc. Its `--not-match`
    takes a Go regular expression, not an fnmatch glob, so passing a glob either
    fails to compile (`*.min.js`) or silently means something else (`a.py`, where
    `.` matches any character). The caller filters scc's output against the file
    set from `scope.discover`, which is the authority on scope, so forwarding the
    globs bought nothing and could silently drop in-scope files -- each one then
    contributing zero to the total.
    """
    exe = scc_path()
    if exe is None:
        raise SccUnavailable("scc is not on PATH")
    cmd = [exe, "--by-file", "-f", "json", "--no-cocomo"]
    if exclude_dirs:
        cmd += ["--exclude-dir", ",".join(exclude_dirs)]
    if not respect_gitignore:
        cmd += ["--no-gitignore", "--no-ignore"]
    cmd.append(root)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SCC_RUN_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SccOutputError(
            f"scc timed out after {_SCC_RUN_TIMEOUT_SECONDS} seconds"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise SccUnavailable(f"could not run scc: {exc}") from exc
    if proc.returncode != 0:
        raise SccOutputError(
            f"scc exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:500]}"
        )
    # scc logs some failures to stderr and still exits 0. Those must not pass
    # silently: they mean the file set it reported is not the one we asked for.
    stderr = (proc.stderr or "").strip()
    if stderr and "ERROR" in stderr.upper():
        raise SccOutputError(
            f"scc reported an error while exiting 0, so its output cannot be "
            f"trusted: {stderr[:500]}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SccOutputError(f"scc did not emit valid JSON: {exc}") from exc
    return parse_scc_json(payload, root)


# --- Julia docstring resolution ----------------------------------------------

# Rejected design, recorded so it is not retried: resolving a Julia
# statement-position string by looking at what *follows* it. It is the real
# semantic rule -- a docstring documents the next item -- but every
# implementable approximation measured worse than not doing it at all. A
# whitelist of documentable forms rejected almost every real docstring (45%
# exact) because Julia can document nearly any expression, including `kw"module"`
# and bare operators. A blacklist of block keywords, run against exact
# scanner-marked regions, still landed at 92.7% exact and 114 lines of absolute
# error versus 62 without it. Grid-searched over all eight combinations of
# {multiline `"`, triple-quote-only docstrings, lookahead}; the lookahead lost
# in all four pairings. Deciding this correctly needs a parser, not a scanner.


# --- exact Julia counter ------------------------------------------------------

class JuliaUnavailable(RuntimeError):
    pass


class JuliaOutputError(RuntimeError):
    pass


_JULIA_HELPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "julia_lines.jl")
_JULIA_BATCH_TIMEOUT_SECONDS = 900


def julia_path() -> str | None:
    return shutil.which("julia")


def julia_identity() -> str:
    """Stable identity for the installed Julia, e.g. `julia/1.12.6`.

    Comes from the helper script rather than `julia --version` so the identity
    and the classification always originate from the same interpreter.
    """
    exe = julia_path()
    if exe is None:
        raise JuliaUnavailable("julia is not on PATH")
    if not os.path.exists(_JULIA_HELPER):
        raise JuliaUnavailable(f"helper script missing: {_JULIA_HELPER}")
    try:
        proc = subprocess.run(
            [exe, "--startup-file=no", _JULIA_HELPER, "--version-id"],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise JuliaUnavailable(f"could not run julia: {exc}") from exc
    line = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not line:
        raise JuliaUnavailable(
            f"julia helper failed: {(proc.stderr or proc.stdout).strip()[:300]}"
        )
    return line[0].strip()


def run_julia(root: str, relpaths: list[str]) -> dict[str, FileCount]:
    """Classify Julia files with Julia's own tokenizer and parser.

    One batched invocation for the whole file set: the interpreter's start-up
    dominates per-file cost, so classifying 220 files takes about as long as one.

    A file the parser cannot handle raises rather than being recorded as zero
    lines -- silently counting an unparseable file as empty would understate the
    baseline, which is the failure mode this whole layer exists to prevent.
    """
    if not relpaths:
        return {}
    exe = julia_path()
    if exe is None:
        raise JuliaUnavailable("julia is not on PATH")
    cmd = [exe, "--startup-file=no", _JULIA_HELPER, "--stdin0"]
    path_input = "\0".join(os.path.join(root, rel) for rel in relpaths)
    try:
        proc = subprocess.run(
            cmd, input=path_input, capture_output=True, text=True,
            timeout=_JULIA_BATCH_TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise JuliaOutputError(
            "julia helper timed out after "
            f"{_JULIA_BATCH_TIMEOUT_SECONDS} seconds while classifying "
            f"{len(relpaths)} file(s)"
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise JuliaUnavailable(f"could not run julia: {exc}") from exc
    if proc.returncode != 0:
        raise JuliaOutputError(
            f"julia helper exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()[:400]}"
        )

    counts: dict[str, FileCount] = {}
    failures: list[str] = []
    expected = set(relpaths)
    for row in proc.stdout.splitlines():
        parts = row.split("\t")
        if len(parts) < 3:
            if row.strip():
                failures.append(f"malformed helper row: {row[:160]!r}")
            continue
        rel = os.path.relpath(parts[0], root).replace(os.sep, "/")
        if rel not in expected:
            failures.append(f"unexpected helper path: {rel}")
            continue
        if rel in counts:
            failures.append(f"duplicate helper result: {rel}")
            continue
        if parts[1] == "ERR":
            failures.append(f"{rel}: {parts[2] if len(parts) > 2 else 'unknown'}")
            continue
        if len(parts) != 4:
            failures.append(f"{rel}: malformed helper result")
            continue
        try:
            code, comment, blank = (int(parts[1]), int(parts[2]), int(parts[3]))
        except ValueError:
            failures.append(f"{rel}: helper counts are not integers")
            continue
        if min(code, comment, blank) < 0:
            failures.append(f"{rel}: helper counts must be non-negative")
            continue
        counts[rel] = FileCount(
            path=rel, language="Julia", lines=code + comment + blank,
            blanks=blank, comments=comment, code=code,
        )
    if failures:
        raise JuliaOutputError(
            f"{len(failures)} Julia file(s) could not be classified, so the count "
            f"would be wrong: {'; '.join(failures[:5])}"
        )
    missing = sorted(expected - set(counts))
    if missing:
        raise JuliaOutputError(
            f"{len(missing)} Julia file(s) produced no result "
            f"(first few: {missing[:5]})"
        )
    return counts
