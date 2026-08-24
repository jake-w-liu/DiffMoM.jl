"""Concept census: how many times did the codebase implement the same idea?

Token-level clone detection (`clones.py`) finds copy-paste. It does *not* find
the case that actually dominates AI-generated codebases: fourteen date
formatters that were each written from scratch, share no tokens, and all do the
same job. Those are found by name.

Matching is on *word sets*, not substrings, because the interesting duplicates
carry a qualifier in the middle: `formatDate`, `formatOrderDate`,
`format_invoice_date` and `formatDueDate` are one concept, and a substring test
for `formatdate` only finds the first. Each symbol is split into words
(snake_case, kebab-case, camelCase and acronym runs) and a concept matches when
either a standalone marker word is present, or one word from `verbs` and one from
`nouns` both are — in any order, so `cloneDeep` and `deepClone` both match.

A concept with definitions in many files is a consolidation *candidate*, not a
confirmed duplicate. The behaviours still have to be diffed by hand before
anything is merged, because the fourteenth formatter is usually the one that
handles a timezone edge case the other thirteen get wrong.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from . import counting, scope
from .langs import Language

# Definition-site patterns. Each must capture the defined symbol in group 1.
# They are matched against comment-stripped, string-blanked code text.
_FUNCTION_DEFINITION = re.compile(
    r"\bfunction\s*\*?\s*([A-Za-z_$][A-Za-z0-9_$]*)"
)
_DEFINITION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        # Python / Ruby / Scala / Elixir / Groovy
        r"\bdef(?:p)?\s+([A-Za-z_][A-Za-z0-9_]*)",
        # Rust / Go / Perl
        r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)",
        r"\bsub\s+([A-Za-z_][A-Za-z0-9_]*)",
        # class / struct / interface / trait / enum / type
        (
            r"\b(?:class|struct|interface|trait|enum|record|protocol|type)\s+"
            r"([A-Za-z_$][A-Za-z0-9_$]*)"
        ),
        # const/let/var/val bound to a function or arrow function
        (
            r"\b(?:const|let|var|val)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*"
            r"(?::[^=]+)?=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|"
            r"[A-Za-z_$][A-Za-z0-9_$]*\s*=>)"
        ),
        # Object/class method shorthand and C-family methods: `name(args) {`
        #
        # Every quantifier here is bounded. The modifier list and the optional
        # return-type group can both match the same words (the type class allows
        # spaces, for `Map<String, Object>`), and with open `*` quantifiers that
        # ambiguity backtracks superlinearly when the match ultimately fails --
        # measured at 4.2 s for a single 800-modifier line. Real declarations use
        # at most a handful of modifiers and a short type, so bounding costs no
        # recall and makes the worst case linear.
        # `:` is in the type class for namespace-qualified C++/Rust return types
        # (`std::string`, `crate::Foo`). It cannot match a Python annotation,
        # because the pattern still requires a trailing `{`.
        (
            r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|"
            r"override|async|export|virtual|inline)\s+){0,8}"
            r"(?:[A-Za-z_$][\w<>,.:\[\]$&* ]{0,120}?\s+)?"
            r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\([^;]{0,400}\)\s*"
            r"(?:->\s*[^{;]{0,120})?\{\s*$"
        ),
    )
)

# Julia uses both `function f(x) ... end` and the very common short form
# `f(x) = ...`. Qualified extension methods (`Dates.format_date`) should report
# the function name, not the module, and Julia identifiers are Unicode.
_JL_IDENT = r"[^\W\d]\w*"
_JULIA_SHORT_DEFINITION = re.compile(
    rf"^\s*(?:@\w+(?:\([^)]{{0,120}}\))?\s+){{0,4}}"
    rf"(?:{_JL_IDENT}\s*\.\s*)*({_JL_IDENT}!?)"
    r"\s*(?:\{[^{}]{0,200}\})?\s*\([^\n]{0,1200}\)"
    r"\s*(?:::[^=]{1,120})?"
    r"\s*(?:where\s+(?:\{[^{}]{0,200}\}|[^\s=]+)\s*)?="
)
_JULIA_DEFINITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\bfunction\s+(?:{_JL_IDENT}\s*\.\s*)*({_JL_IDENT}!?)\s*(?:\{{|\()"
    ),
    re.compile(rf"\bmacro\s+({_JL_IDENT}!?)\b"),
    _JULIA_SHORT_DEFINITION,
)
_JULIA_SHORT_START = re.compile(
    rf"^\s*(?:@\w+(?:\([^)]{{0,120}}\))?\s+){{0,4}}"
    rf"(?:{_JL_IDENT}\s*\.\s*)*({_JL_IDENT}!?)"
    r"\s*(?:\{[^{}]{0,200}\})?\s*\("
)

# Lines longer than this are not scanned for definitions. A definition site is
# never this long in hand-written code; minified or generated output is, and
# feeding it to the patterns above is pure cost. Skips are counted and reported
# rather than dropped silently.
MAX_DEFINITION_LINE_CHARS = 2000

# Symbols that carry no intent and would otherwise pollute every bucket. Includes
# keywords that leak through the definition patterns: `if type in mapping:` makes
# the `type <name>` pattern capture `in`, so keywords are filtered by name rather
# than by trying to make every pattern context-aware.
_IGNORED_SYMBOLS = frozenset(
    """if for while switch catch try else elif return with match case do fn func def
    function class struct enum interface trait type const let var val new delete
    constructor init main setup teardown expect beforeeach aftereach beforeall
    afterall get set in is not and or as from import pass raise assert lambda
    yield await async global nonlocal del self cls this true false none null nil
    void public private protected static final abstract override""".split()
)

# Generic operation names. Many classes legitimately define `close` or `read`;
# that is polymorphism, not duplicated logic. They are excluded from the
# same-name-in-many-files report, where they would otherwise crowd out the
# qualified names (`formatInvoiceDate`) that indicate a real duplicate.
_GENERIC_METHOD_NAMES = frozenset(
    """close open read write flush seek tell reset clear copy clone get set add
    remove insert append extend pop push next prev start stop run send receive
    recv connect disconnect bind listen accept name value size length count keys
    values items update save load delete create list find search filter map
    reduce apply call execute build make parse format encode decode dumps loads
    dump serialize deserialize str repr hash iter enter exit exec compile check
    verify handle process step tick wrap unwrap begin end commit rollback lock
    unlock acquire release join split strip trim replace match compare sort
    reverse render draw paint refresh reload configure dispose destroy register
    unregister subscribe unsubscribe emit publish notify enqueue dequeue peek
    poll put take fileno readline readlines writelines writable readable
    seekable truncate isatty detach abort""".split()
)
# This list is not exhaustive and is not meant to be. It removes the names that
# are unambiguously language or IO protocol methods; beyond that, a single-word
# name repeated across files is usually polymorphism, which is why the report is
# labelled a lead list rather than a defect list.

# A symbol whose *first word* is one of these names a test, not an implementation.
_TEST_FIRST_WORDS = frozenset(
    {"test", "tests", "it", "should", "describe", "spec", "when", "given"}
)

# Dunder / magic methods implement a language protocol. Every class has
# `__init__` and `__repr__`, so counting them as a duplicated concept buries the
# real findings under hundreds of rows.
_DUNDER = re.compile(r"^__[A-Za-z0-9_]+__$")

_WORD_SPLIT = re.compile(r"[^A-Za-z0-9]+")
# camelCase / PascalCase / acronym-run boundaries: `HTTPClient` -> HTTP, Client.
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_words(symbol: str) -> list[str]:
    """Break an identifier into lowercase words.

    `format_invoice_date` -> [format, invoice, date]
    `formatDueDate`       -> [format, due, date]
    `HTTPClient`          -> [http, client]
    """
    words: list[str] = []
    for chunk in _WORD_SPLIT.split(symbol):
        if not chunk:
            continue
        for part in _CAMEL_SPLIT.split(chunk):
            if part:
                words.append(part.lower())
    return words


@dataclass(frozen=True)
class Concept:
    name: str
    # Any one of these words, alone, identifies the concept.
    words: tuple[str, ...] = ()
    # Or: one word from `verbs` and one from `nouns`, in any order.
    verbs: tuple[str, ...] = ()
    nouns: tuple[str, ...] = ()

    def matches(self, symbol_words: frozenset[str]) -> bool:
        if symbol_words & frozenset(self.words):
            return True
        if not self.verbs or not self.nouns:
            return False
        verb_hits = symbol_words & frozenset(self.verbs)
        noun_hits = symbol_words & frozenset(self.nouns)
        if not verb_hits or not noun_hits:
            return False
        # Two *distinct* words must be present. A word listed as both verb and
        # noun (`request`) would otherwise match on its own, so `retry_request`
        # would register as an http client wrapper.
        return len(verb_hits | noun_hits) >= 2


_FORMAT_VERBS = ("format", "fmt", "render", "display", "pretty", "humanize",
                 "humanise", "stringify", "print")

CONCEPTS: tuple[Concept, ...] = (
    Concept(
        "date-time formatting",
        words=("strftime", "strptime", "timeago", "datestring", "isoformat"),
        verbs=(*_FORMAT_VERBS, "parse", "to", "from"),
        nouns=("date", "dates", "time", "times", "datetime", "timestamp",
               "duration", "elapsed"),
    ),
    Concept(
        "number/currency formatting",
        words=("formatbytes", "filesize"),
        verbs=(*_FORMAT_VERBS, "round"),
        nouns=("currency", "money", "price", "amount", "number", "decimal",
               "percent", "percentage", "bytes", "filesize"),
    ),
    Concept(
        "string case / slug",
        words=("slugify", "slug", "camelize", "camelcase", "snakecase", "kebabcase",
               "pascalcase", "titlecase", "capitalize", "capitalise", "titleize",
               "dasherize", "underscore"),
        verbs=("to", "convert", "normalize", "normalise"),
        nouns=("camel", "snake", "kebab", "pascal", "slug", "titlecase"),
    ),
    Concept(
        "validation",
        words=("validate", "validator", "validation", "isvalid"),
        verbs=("validate", "check", "assert", "ensure", "is", "verify"),
        nouns=("email", "phone", "url", "uri", "password", "input", "payload",
               "form", "schema", "required", "valid"),
    ),
    Concept(
        "http client wrapper",
        words=("apiclient", "httpclient", "fetchjson", "apifetch"),
        verbs=("fetch", "request", "call", "send", "do", "make", "get", "post",
               "put", "patch"),
        nouns=("json", "api", "http", "endpoint", "request", "resource"),
    ),
    Concept(
        "retry / backoff",
        words=("retry", "retries", "retryable", "backoff"),
        verbs=("with", "exponential"),
        nouns=("retry", "backoff", "attempts"),
    ),
    Concept(
        "deep clone / merge / equality",
        words=("structuredclone", "isequal", "deepequal"),
        verbs=("deep", "structured"),
        nouns=("clone", "copy", "merge", "equal", "equals", "diff"),
    ),
    Concept(
        "debounce / throttle",
        words=("debounce", "throttle", "ratelimit", "ratelimiter"),
        verbs=("rate",),
        nouns=("limit", "limiter"),
    ),
    Concept(
        "auth / permission check",
        words=("authorize", "authorise", "authenticate", "authguard"),
        verbs=("is", "has", "can", "check", "require", "ensure", "verify", "validate"),
        nouns=("auth", "authenticated", "permission", "permissions", "role",
               "roles", "token", "jwt", "admin", "owner", "access", "scope"),
    ),
    Concept(
        "pagination",
        words=("paginate", "pagination", "paginated"),
        verbs=("build", "get", "make", "apply", "parse"),
        nouns=("page", "pagination", "cursor", "offset"),
    ),
    Concept(
        "error mapping / wrapping",
        verbs=("handle", "map", "wrap", "format", "normalize", "normalise",
               "parse", "to", "get", "convert", "report"),
        nouns=("error", "errors", "err", "exception", "failure"),
    ),
    Concept(
        "logging wrapper",
        words=("loginfo", "logerror", "logwarn", "logdebug", "logevent"),
        verbs=("get", "create", "make", "build", "setup", "init"),
        nouns=("logger", "log", "logging"),
    ),
    Concept(
        "id generation",
        words=("uuid", "nanoid", "shortid", "cuid", "guid", "randomstring"),
        verbs=("generate", "gen", "make", "create", "new", "random", "next"),
        nouns=("id", "ids", "uuid", "uid", "guid", "token", "nonce", "key"),
    ),
    Concept(
        "sleep / delay",
        words=("sleep", "delay"),
        verbs=("wait", "sleep"),
        nouns=("ms", "seconds", "timeout", "for"),
    ),
    Concept(
        "serialization / key conversion",
        words=("serialize", "serialise", "deserialize", "deserialise", "tojson",
               "fromjson", "todict", "fromdict"),
        verbs=("to", "from", "convert", "serialize", "deserialize"),
        nouns=("json", "dict", "dto", "model", "plain", "payload", "entity", "record"),
    ),
    Concept(
        "caching / memoization",
        words=("memoize", "memoise", "memo", "cachekey"),
        verbs=("get", "set", "build", "invalidate", "clear", "with", "read", "write"),
        nouns=("cache", "cached"),
    ),
    Concept(
        "sanitize / escape",
        words=("sanitize", "sanitise", "escape", "unescape", "purify", "striptags"),
        verbs=("strip", "clean", "escape", "sanitize"),
        nouns=("html", "tags", "input", "xss", "markup"),
    ),
    Concept(
        "env / config loading",
        verbs=("load", "get", "read", "parse", "require", "resolve"),
        nouns=("config", "env", "settings", "environment", "options"),
    ),
    Concept(
        "crypto / hashing",
        words=("hmac", "checksum", "sha256", "md5"),
        verbs=("hash", "verify", "compare", "encrypt", "decrypt", "sign", "digest"),
        nouns=("password", "token", "secret", "payload", "signature"),
    ),
    Concept(
        "query building",
        words=("querybuilder", "buildquery", "orderby"),
        verbs=("build", "make", "to", "apply", "compose", "add"),
        nouns=("query", "where", "sql", "filter", "filters", "sort", "order",
               "clause", "predicate"),
    ),
    Concept(
        "text truncation",
        words=("truncate", "ellipsis", "excerpt", "elide"),
        verbs=("truncate", "shorten", "clamp", "trim"),
        nouns=("text", "string", "label", "title"),
    ),
    Concept(
        "collection helpers",
        words=("groupby", "chunk", "flatten", "unique", "uniq", "dedupe", "dedup",
               "partition", "keyby", "sortby", "zip"),
        verbs=("group", "sort", "key", "order"),
        nouns=("by",),
    ),
    Concept(
        "api response envelope",
        words=("successresponse", "errorresponse", "jsonresponse", "okresponse"),
        verbs=("build", "make", "send", "create", "wrap"),
        nouns=("response", "envelope", "reply", "result"),
    ),
    Concept(
        "file / path helpers",
        words=("mkdirp", "ensuredir", "fileexists"),
        verbs=("read", "write", "ensure", "resolve", "normalize", "normalise",
               "join", "safe", "exists"),
        nouns=("file", "dir", "directory", "path", "folder"),
    ),
)


@dataclass
class Hit:
    path: str
    lineno: int
    symbol: str
    concept: str


def normalise_symbol(symbol: str) -> str:
    """Canonical form for comparing two names: words joined, case and separators
    dropped. `format_date` and `formatDate` both become `formatdate`."""
    return "".join(split_words(symbol))


def _concepts_for(symbol: str) -> list[str]:
    words = frozenset(split_words(symbol))
    if not words:
        return []
    return [concept.name for concept in CONCEPTS if concept.matches(words)]


def definitions(
    path: str, text: str, lang: Language, include_tests: bool = False,
    skipped_lines: list[int] | None = None,
) -> list[tuple[int, str]]:
    """Every (line number, symbol) definition site found in `text`.

    Test files and test-named symbols are skipped by default: they borrow the
    vocabulary of the code they exercise without duplicating its logic, so
    `test_format_date` must not count as another date formatter.

    Pass `skipped_lines` to collect the line numbers skipped for exceeding
    `MAX_DEFINITION_LINE_CHARS`, so a caller can report the coverage gap instead
    of the scan quietly covering less than it appears to.
    """
    if not include_tests and scope.is_test_path(path):
        return []
    scanned, _ = counting.scan(text, lang, path)
    patterns = _DEFINITION_PATTERNS + (
        _JULIA_DEFINITION_PATTERNS
        if lang.name == "Julia"
        else (_FUNCTION_DEFINITION,)
    )
    found: list[tuple[int, str]] = []
    # Several patterns can match one definition — `export function foo() {` is
    # both a `function NAME` and a C-family `NAME(args) {`. Report it once.
    seen: set[tuple[int, str]] = set()
    for line in scanned:
        if line.kind != counting.CODE or not line.code_text:
            continue
        if len(line.code_text) > MAX_DEFINITION_LINE_CHARS:
            if skipped_lines is not None:
                skipped_lines.append(line.lineno)
            continue
        for pattern in patterns:
            for match in pattern.finditer(line.code_text):
                symbol = match.group(1)
                if len(symbol) < 2 or _DUNDER.match(symbol):
                    continue
                words = split_words(symbol)
                # Compare the normalised form too, so `__init__` and `_init` are
                # both caught by the `init` entry.
                if symbol.lower() in _IGNORED_SYMBOLS or "".join(words) in _IGNORED_SYMBOLS:
                    continue
                if not include_tests and words and words[0] in _TEST_FIRST_WORDS:
                    continue
                key = (line.lineno, symbol)
                if key in seen:
                    continue
                seen.add(key)
                found.append(key)

    # Short-form Julia signatures are often formatted across several lines,
    # especially when they have keyword arguments. Reconstruct only a bounded
    # parenthesized signature; bodies and arbitrary expressions are never joined.
    if lang.name == "Julia":
        code_lines = [
            line for line in scanned
            if line.kind == counting.CODE and line.code_text
        ]
        for index, line in enumerate(code_lines):
            start = _JULIA_SHORT_START.match(line.code_text)
            if start is None:
                continue
            depth = line.code_text.count("(") - line.code_text.count(")")
            if depth <= 0:
                continue  # the one-line pattern already handled it
            pieces = [line.code_text]
            for follower in code_lines[index + 1:index + 25]:
                pieces.append(follower.code_text)
                depth += follower.code_text.count("(") - follower.code_text.count(")")
                if depth > 0:
                    continue
                signature = " ".join(pieces)
                match = _JULIA_SHORT_DEFINITION.match(signature)
                if match is not None:
                    key = (line.lineno, match.group(1))
                    if key not in seen:
                        seen.add(key)
                        found.append(key)
                break
    found.sort(key=lambda item: (item[0], item[1]))
    return found


def scan_text(path: str, text: str, lang: Language) -> list[Hit]:
    """Concept hits for one file."""
    hits: list[Hit] = []
    seen_julia_methods: set[tuple[str, str]] = set()
    for lineno, symbol in definitions(path, text, lang):
        for concept in _concepts_for(symbol):
            if lang.name == "Julia":
                # Multiple dispatch intentionally defines the same function
                # name for several signatures. Count that concept once per file
                # and symbol; clone detection still sees duplicated bodies, and
                # differently named implementations remain separate leads.
                key = (normalise_symbol(symbol), concept)
                if key in seen_julia_methods:
                    continue
                seen_julia_methods.add(key)
            hits.append(Hit(path=path, lineno=lineno, symbol=symbol, concept=concept))
    return hits


def summarise(hits: list[Hit], min_definitions: int = 2) -> list[dict]:
    """Concepts implemented `min_definitions` or more times, worst first."""
    buckets: dict[str, list[Hit]] = defaultdict(list)
    for hit in hits:
        buckets[hit.concept].append(hit)

    rows: list[dict] = []
    for concept, concept_hits in buckets.items():
        if len(concept_hits) < min_definitions:
            continue
        distinct_symbols = {normalise_symbol(hit.symbol) for hit in concept_hits}
        files = {hit.path for hit in concept_hits}
        rows.append({
            "concept": concept,
            "definitions": len(concept_hits),
            "distinct_names": len(distinct_symbols),
            "files": len(files),
            "sites": [
                {"path": hit.path, "line": hit.lineno, "symbol": hit.symbol}
                for hit in sorted(concept_hits, key=lambda h: (h.path, h.lineno))
            ],
        })
    rows.sort(key=lambda row: (-row["definitions"], -row["files"], row["concept"]))
    return rows


def duplicate_symbols(
    all_definitions: list[tuple[str, int, str]], min_count: int = 2
) -> list[dict]:
    """Symbol names defined in more than one file, after normalisation.

    Single-word generic operations (`close`, `read`, `write`) are excluded: many
    classes implement those by design, and reporting them buries the qualified
    names — `formatInvoiceDate`, `buildOrderQuery` — that actually indicate
    duplicated logic. Overloads and interface implementations still repeat names
    legitimately, so this is a lead list, not a defect list.
    """
    buckets: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for path, lineno, symbol in all_definitions:
        buckets[normalise_symbol(symbol)].append((path, lineno, symbol))

    rows: list[dict] = []
    for normalised, sites in buckets.items():
        distinct_files = {path for path, _, _ in sites}
        if len(sites) < min_count or len(distinct_files) < 2:
            continue
        if normalised in _GENERIC_METHOD_NAMES:
            continue
        rows.append({
            "symbol": normalised,
            "definitions": len(sites),
            "files": len(distinct_files),
            "sites": [
                {"path": path, "line": lineno, "symbol": symbol}
                for path, lineno, symbol in sorted(sites)
            ],
        })
    rows.sort(key=lambda row: (-row["definitions"], row["symbol"]))
    return rows
