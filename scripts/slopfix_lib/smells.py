"""Scan for the slop patterns that block a CRC-clean reduction.

Deliberately narrow. `ruff`, `eslint`, `clippy`, `go vet` and friends remain the
authoritative smell detectors and the skill tells you to run them. This module
covers only what a language-agnostic pass can find reliably and what the method
cannot proceed without:

  * `broad-except` / `swallowed-error` — the single most common AI-introduced
    smell, and a Robustness violation the method forbids leaving in place.
  * `placeholder-implementation` — a Completeness violation. Consolidating code
    that contains a stub is how a reduction becomes a regression.
  * `unused-import` — direct, safe line reduction.
  * `god-function` — the primary consolidation target.
  * `deep-nesting`, `debug-leftover` — advisory.

Every rule reports a location so a human can confirm it. None of them edit code.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass

from . import counting, scope
from .langs import Language

BLOCKING = "blocking"      # must be resolved before the reduction can be claimed
ADVISORY = "advisory"      # a lead worth reviewing


@dataclass
class Hit:
    path: str
    lineno: int
    rule: str
    severity: str
    message: str
    excerpt: str

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "line": self.lineno,
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "excerpt": self.excerpt[:200],
        }


# --- single-line regex rules -------------------------------------------------

@dataclass(frozen=True)
class _Rule:
    rule: str
    severity: str
    message: str
    pattern: re.Pattern[str]
    languages: frozenset[str] | None = None  # None means every language
    # Match the raw source line instead of the comment-stripped, string-blanked
    # text. Required for rules that look *inside* a string literal -- the scanner
    # replaces string contents with an empty placeholder, so
    # `error("not implemented")` reads as `error("")` and can never match. Only
    # code lines are ever tested, so a comment-only line still cannot trigger a
    # raw rule; a trailing comment on a code line can, which is acceptable here.
    on_raw: bool = False
    # Extra condition matched against the *stripped* text, i.e. with string
    # contents blanked. A raw rule needs this to stay precise: matching raw text
    # alone means a message table or a docstring that merely mentions the phrase
    # fires a BLOCKING finding. The guard proves the line really performs the
    # operation, because a keyword survives blanking while prose does not.
    guard: re.Pattern[str] | None = None


# A raw-matching placeholder rule only fires when the line genuinely raises. The
# keyword survives string-blanking; prose inside a literal does not.
_RAISES = re.compile(r"\b(?:throw|raise|panic|error|fail|abort)\b|\bpanic!|\berror\s*\(")

_PY = frozenset({"Python"})
_JSLIKE = frozenset({"JavaScript", "TypeScript", "Vue", "Svelte"})

# Languages whose function bodies are delimited by indentation or by an `end`
# keyword rather than by braces, so function length is measured by indentation.
_INDENT_STRUCTURED = frozenset(
    {"Python", "Ruby", "Elixir", "Shell", "R", "Haskell", "Julia"}
)

_RULES: tuple[_Rule, ...] = (
    _Rule(
        "broad-except", BLOCKING,
        "Catches every exception; the specific failure becomes invisible.",
        re.compile(r"^\s*except\s*(?:\(?\s*(?:BaseException|Exception)\s*\)?)?\s*"
                   r"(?:as\s+\w+\s*)?:"),
        _PY,
    ),
    _Rule(
        "broad-except", BLOCKING,
        "Catches every error without inspecting or re-raising it.",
        re.compile(r"\bcatch\s*\(\s*(?:\w+\s*(?::\s*(?:any|unknown|Error))?)?\s*\)"),
        _JSLIKE,
    ),
    _Rule(
        "broad-except", BLOCKING,
        "Catches Throwable/Exception at the widest possible type.",
        re.compile(r"\bcatch\s*\(\s*(?:java\.lang\.)?(?:Throwable|Exception)\s+\w+\s*\)"),
        frozenset({"Java", "Kotlin", "Scala", "C#"}),
    ),
    _Rule(
        "swallowed-error", BLOCKING,
        "Error handler discards the error and continues.",
        re.compile(r"\.catch\s*\(\s*\(?\s*\w*\s*\)?\s*=>\s*\{\s*\}\s*\)"),
        _JSLIKE,
    ),
    _Rule(
        "swallowed-error", BLOCKING,
        "Empty catch block.",
        re.compile(r"\bcatch\s*(?:\([^)]*\))?\s*\{\s*\}"),
        None,
    ),
    _Rule(
        "swallowed-error", BLOCKING,
        "Error assigned to `_` and dropped.",
        re.compile(r"^\s*_\s*(?::)?=\s*\w+\([^)]*\)\s*$"),
        frozenset({"Go"}),
    ),
    _Rule(
        "placeholder-implementation", BLOCKING,
        "Unimplemented path left in the codebase.",
        re.compile(r"\b(?:NotImplementedError|NotImplementedException|"
                   r"unimplemented!|todo!|MethodNotImplemented)\b"),
        None,
    ),
    _Rule(
        "placeholder-implementation", BLOCKING,
        "Throws a 'not implemented' error at runtime.",
        re.compile(r"""(?:throw|raise|panic!?)\s*.{0,60}?not\s*[-_ ]?implemented""",
                   re.IGNORECASE),
        None,
        on_raw=True,
        guard=_RAISES,
    ),
    _Rule(
        "placeholder-implementation", ADVISORY,
        "Work marker left in code.",
        re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b"),
        None,
    ),
    _Rule(
        "placeholder-implementation", BLOCKING,
        "Unimplemented path signalled with `error(...)`.",
        re.compile(r"""\berror\s*\(\s*["'].{0,60}?(?:not\s*[-_ ]?implemented|unimplemented|"""
                   r"""todo)""", re.IGNORECASE),
        frozenset({"Julia"}),
        on_raw=True,
        guard=_RAISES,
    ),
    _Rule(
        "placeholder-implementation", BLOCKING,
        "`@assert false` marks an unreachable-but-unwritten branch.",
        re.compile(r"@assert\s+false\b"),
        frozenset({"Julia"}),
    ),
    _Rule(
        "broad-except", ADVISORY,
        "Bare `catch` with no binding cannot inspect or rethrow selectively.",
        re.compile(r"^\s*catch\s*$"),
        frozenset({"Julia"}),
    ),
    _Rule(
        "debug-leftover", ADVISORY,
        "Debug output left in shipped code.",
        re.compile(r"\b(?:console\.(?:log|debug|dir)|debugger|System\.out\.print(?:ln)?|"
                   r"fmt\.Print(?:ln|f)?|dbg!|println!)\s*[(!]"),
        None,
    ),
    _Rule(
        "wildcard-import", ADVISORY,
        "Wildcard import hides what the module actually uses.",
        re.compile(r"^\s*from\s+[\w.]+\s+import\s+\*"),
        _PY,
    ),
    _Rule(
        "mutable-default-arg", BLOCKING,
        "Mutable default argument is shared across calls.",
        re.compile(r"\bdef\s+\w+\s*\([^)]*=\s*(?:\[\s*\]|\{\s*\}|set\(\))"),
        _PY,
    ),
    _Rule(
        "open-without-encoding", ADVISORY,
        "`open()` without an explicit encoding behaves differently per platform.",
        re.compile(r"\bopen\s*\((?![^)]*encoding\s*=)(?![^)]*[\"']rb[\"'])"
                   r"(?![^)]*[\"']wb[\"'])(?![^)]*[\"']ab[\"'])[^)]*\)"),
        _PY,
    ),
)

# Python bodies that discard the exception entirely.
_PY_SWALLOW_BODY = re.compile(r"^\s*(?:pass|continue|return(?:\s+(?:None|False|\[\]|\{\}))?)\s*$")
_PY_BROAD_EXCEPT = re.compile(
    r"^\s*except\s*(?:\(?\s*(?:BaseException|Exception)\s*\)?)?"
    r"\s*(?:as\s+\w+\s*)?:"
)

# Julia has no braces, so an empty `catch` shows up as `catch` followed directly by
# `end` -- or by a body that discards the error. Comment-only bodies count as empty
# because `_next_code_line` skips comments.
_JL_CATCH = re.compile(r"^\s*catch\b")
_JL_SWALLOW_BODY = re.compile(
    r"^\s*(?:end|nothing|return(?:\s+(?:nothing|false))?|continue|break)\s*$"
)
_JL_INLINE_SWALLOW = re.compile(
    r"\bcatch(?:\s+[^\s;]+)?\s*;\s*"
    r"(?:(?:nothing|return(?:\s+(?:nothing|false))?|continue|break)\s*;\s*)?"
    r"end\b"
)


def _python_reraising_handlers(text: str) -> set[int]:
    """Lines of broad handlers that unconditionally re-raise after cleanup."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        broad = node.type is None or (
            isinstance(node.type, ast.Name)
            and node.type.id in {"Exception", "BaseException"}
        )
        final_is_bare_raise = bool(node.body) and (
            isinstance(node.body[-1], ast.Raise) and node.body[-1].exc is None
        )
        preceding_control_transfer = any(
            isinstance(child, ast.Return | ast.Break | ast.Continue)
            for statement in node.body[:-1]
            for child in ast.walk(statement)
        )
        if broad and final_is_bare_raise and not preceding_control_transfer:
            lines.add(node.lineno)
    return lines


def _long_function_hits(
    path: str, lang: Language, lines: list[counting.ScannedLine], max_lines: int
) -> list[Hit]:
    """Report function bodies longer than `max_lines` code lines.

    Uses indentation for indent-structured languages and brace depth otherwise.
    Both are heuristics: a misjudged boundary produces a wrong length, never a
    wrong file. The result is a ranked list of consolidation targets.
    """
    hits: list[Hit] = []
    code_lines = [line for line in lines if line.kind == counting.CODE and line.code_text]
    if not code_lines:
        return hits

    starter = re.compile(
        r"\b(?:def|fn|func|function|sub)\b|"
        r"\b(?:const|let|var|val)\s+[\w$]+\s*(?::[^=]+)?=\s*(?:async\s+)?\([^)]*\)\s*=>"
    )

    if lang.name in _INDENT_STRUCTURED:
        # Indent-structured: a function ends when indentation returns to its own
        # level or less.
        for index, line in enumerate(code_lines):
            if not starter.search(line.code_text):
                continue
            base_indent = _leading_spaces(line)
            length = 1
            for follower in code_lines[index + 1:]:
                if _leading_spaces(follower) <= base_indent:
                    break
                length += 1
            if length > max_lines:
                hits.append(Hit(
                    path=path, lineno=line.lineno, rule="god-function", severity=ADVISORY,
                    message=f"Function body spans about {length} code lines "
                            f"(threshold {max_lines}).",
                    excerpt=line.stripped,
                ))
        return hits

    # Brace-structured: track depth from the opening brace of the definition.
    for index, line in enumerate(code_lines):
        if not starter.search(line.code_text):
            continue
        depth = line.code_text.count("{") - line.code_text.count("}")
        if depth <= 0:
            continue
        length = 1
        for follower in code_lines[index + 1:]:
            length += 1
            depth += follower.code_text.count("{") - follower.code_text.count("}")
            if depth <= 0:
                break
        if length > max_lines:
            hits.append(Hit(
                path=path, lineno=line.lineno, rule="god-function", severity=ADVISORY,
                message=f"Function body spans about {length} code lines "
                        f"(threshold {max_lines}).",
                excerpt=line.stripped,
            ))
    return hits


def _leading_spaces(line: counting.ScannedLine) -> int:
    return len(line.code_text) - len(line.code_text.lstrip())


# --- unused imports ----------------------------------------------------------

# Languages whose import syntax this pass models. Anything else is left to that
# language's own linter rather than guessed at.
_UNUSED_IMPORT_LANGUAGES = frozenset({"Python", "JavaScript", "TypeScript"})

_PY_IMPORT = re.compile(r"^\s*import\s+(.+)$")
_PY_FROM = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+(.+)$")
_JS_IMPORT = re.compile(r"^\s*import\s+(?![\"'])(.+?)\s+from\s+[\"']")


def _python_imported_names(code_text: str) -> list[str]:
    names: list[str] = []
    match = _PY_FROM.match(code_text)
    if match:
        module, targets = match.group(1), match.group(2)
        if module == "__future__" or "*" in targets:
            return []
        for target in targets.replace("(", "").replace(")", "").split(","):
            target = target.strip()
            if not target:
                continue
            names.append(target.split(" as ")[-1].strip() if " as " in target else target)
        return names
    match = _PY_IMPORT.match(code_text)
    if match:
        for target in match.group(1).split(","):
            target = target.strip()
            if not target:
                continue
            if " as " in target:
                names.append(target.split(" as ")[-1].strip())
            else:
                # `import a.b.c` binds only `a`.
                names.append(target.split(".")[0].strip())
    return names


def _js_imported_names(code_text: str) -> list[str]:
    match = _JS_IMPORT.match(code_text)
    if not match:
        return []
    clause = match.group(1).strip()
    names: list[str] = []
    brace = re.search(r"\{(.*?)\}", clause, re.DOTALL)
    if brace:
        for target in brace.group(1).split(","):
            target = target.strip()
            if not target:
                continue
            names.append(target.split(" as ")[-1].strip() if " as " in target else target)
        clause = clause[: brace.start()] + clause[brace.end():]
    for token in re.split(r"[,\s]+", clause):
        token = token.strip()
        if not token or token in {"type", "*", "as"}:
            continue
        if re.fullmatch(r"[A-Za-z_$][\w$]*", token):
            names.append(token)
    # `import * as ns from` — capture the namespace binding.
    ns = re.search(r"\*\s+as\s+([A-Za-z_$][\w$]*)", match.group(1))
    if ns:
        names.append(ns.group(1))
    return names


def _unused_import_hits(
    path: str, lang: Language, lines: list[counting.ScannedLine]
) -> list[Hit]:
    """Imported names never referenced elsewhere in the same file.

    Skipped where the pattern is legitimately absent from the body: package
    entry points that re-export, and files declaring an explicit export list.
    """
    if lang.name not in _UNUSED_IMPORT_LANGUAGES:
        return []
    basename = path.rsplit("/", 1)[-1]
    if basename in {"__init__.py", "index.js", "index.ts", "index.mjs", "mod.ts"}:
        return []
    body_text = "\n".join(
        line.code_text for line in lines if line.kind == counting.CODE
    )
    if "__all__" in body_text or re.search(r"\bexport\s+\*", body_text):
        return []

    extract = _python_imported_names if lang.name == "Python" else _js_imported_names
    imports: list[tuple[int, str, str]] = []
    import_linenos: set[int] = set()
    for line in lines:
        if line.kind != counting.CODE or not line.code_text:
            continue
        for name in extract(line.code_text):
            imports.append((line.lineno, name, line.stripped))
            import_linenos.add(line.lineno)

    if not imports:
        return []

    non_import_body = "\n".join(
        line.code_text
        for line in lines
        if line.kind == counting.CODE and line.lineno not in import_linenos
    )
    hits: list[Hit] = []
    for lineno, name, excerpt in imports:
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", name):
            continue
        if re.search(rf"\b{re.escape(name)}\b", non_import_body):
            continue
        hits.append(Hit(
            path=path, lineno=lineno, rule="unused-import", severity=ADVISORY,
            message=f"`{name}` is imported but not referenced in this file.",
            excerpt=excerpt,
        ))
    return hits


# --- entry points ------------------------------------------------------------

def scan_text(
    path: str, text: str, lang: Language, god_function_lines: int = 60
) -> list[Hit]:
    """All smell hits for one file."""
    lines, _ = counting.scan(text, lang, path)
    hits: list[Hit] = []
    is_test = scope.is_test_path(path)
    python_reraising = (
        _python_reraising_handlers(text) if lang.name == "Python" else set()
    )

    for index, line in enumerate(lines):
        if line.kind != counting.CODE or not line.code_text:
            continue
        for rule in _RULES:
            if rule.languages is not None and lang.name not in rule.languages:
                continue
            if rule.guard is not None and not rule.guard.search(line.code_text):
                continue
            subject = line.stripped if rule.on_raw else line.code_text
            if rule.pattern.search(subject):
                if rule.rule == "broad-except" and line.lineno in python_reraising:
                    continue
                severity = rule.severity
                # Catch-all handlers are often deliberate in a test harness:
                # they turn an unexpected exception into a recorded test
                # failure. Keep the lead visible without making a production
                # guardrail impossible to pass.
                if is_test and rule.rule in {"broad-except", "swallowed-error"}:
                    severity = ADVISORY
                hits.append(Hit(
                    path=path, lineno=line.lineno, rule=rule.rule,
                    severity=severity, message=rule.message,
                    excerpt=line.stripped,
                ))
        # Python `except ...:` whose entire body discards the error.
        if lang.name == "Python" and re.match(r"^\s*except\b", line.code_text):
            body = _next_code_line(lines, index)
            if body is not None and _PY_SWALLOW_BODY.match(body.code_text):
                # A specific exception mapped to a sentinel or ignored because
                # the failure is an expected capability probe is not reliably a
                # bug. Only a broad discard blocks; narrow discards stay
                # advisory so a human/language linter can judge the contract.
                severity = (
                    BLOCKING
                    if _PY_BROAD_EXCEPT.match(line.code_text) and not is_test
                    else ADVISORY
                )
                hits.append(Hit(
                    path=path, lineno=line.lineno, rule="swallowed-error",
                    severity=severity,
                    message="Exception handler discards the error "
                            f"(`{body.stripped}`) without logging or re-raising.",
                    excerpt=line.stripped,
                ))
        # Julia `catch` whose body is empty, comment-only, or discards the error.
        if lang.name == "Julia" and _JL_INLINE_SWALLOW.search(line.code_text):
            hits.append(Hit(
                path=path, lineno=line.lineno, rule="swallowed-error",
                severity=ADVISORY if is_test else BLOCKING,
                message="Inline exception handler discards the error without "
                        "logging or rethrowing.",
                excerpt=line.stripped,
            ))
            continue
        if lang.name == "Julia" and _JL_CATCH.match(line.code_text):
            body = _next_code_line(lines, index)
            if body is not None and _JL_SWALLOW_BODY.match(body.code_text):
                detail = "block is empty" if body.code_text.strip() == "end" else (
                    f"body is `{body.stripped}`"
                )
                hits.append(Hit(
                    path=path, lineno=line.lineno, rule="swallowed-error",
                    severity=ADVISORY if is_test else BLOCKING,
                    message=f"Exception handler discards the error ({detail}) "
                            "without logging or rethrowing.",
                    excerpt=line.stripped,
                ))

    hits += _unused_import_hits(path, lang, lines)
    hits += _long_function_hits(path, lang, lines, god_function_lines)

    # Several patterns can describe the same defect on the same line (a bare
    # `raise NotImplementedError` matches two placeholder rules). Report it once.
    deduped: list[Hit] = []
    seen: set[tuple[int, str]] = set()
    for hit in sorted(hits, key=lambda h: (h.lineno, h.rule, h.severity)):
        key = (hit.lineno, hit.rule)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped


def _next_code_line(
    lines: list[counting.ScannedLine], index: int
) -> counting.ScannedLine | None:
    for line in lines[index + 1:]:
        if line.kind == counting.CODE and line.code_text:
            return line
    return None


def summarise(hits: list[Hit]) -> dict:
    """Counts by rule and severity, for the census report."""
    by_rule: dict[str, int] = defaultdict(int)
    by_severity: dict[str, int] = defaultdict(int)
    for hit in hits:
        by_rule[hit.rule] += 1
        by_severity[hit.severity] += 1
    return {
        "total": len(hits),
        "by_severity": dict(sorted(by_severity.items())),
        "by_rule": dict(sorted(by_rule.items(), key=lambda item: -item[1])),
    }
