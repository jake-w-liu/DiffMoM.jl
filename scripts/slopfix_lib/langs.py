"""Language syntax table used by the line counter and the clone tokenizer.

The table is deliberately explicit rather than clever: every language declares its
comment and string syntax, and the scanners in `counting.py` / `clones.py` consume
that declaration. Adding a language means adding one entry here, not editing a
scanner.

Only syntax that affects *line classification* or *tokenization* is modelled.
Semantics (types, scoping, macros) are out of scope.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StringSpec:
    """One string-literal form.

    open_seq / close_seq: literal delimiter text.
    escape: backslash escaping applies inside the literal.
    doubling: the delimiter is escaped by doubling it (SQL `''`, VB `""`).
    multiline: the literal may span physical lines.
    raw_hash: Rust-style `r#"..."#` — an arbitrary run of `#` between the
        `r` and the quote, repeated after the closing quote.
    raw_delim: C++-style `R"delim(...)delim"` — a user-chosen delimiter.
    """

    open_seq: str
    close_seq: str
    escape: bool = True
    doubling: bool = False
    multiline: bool = False
    raw_hash: bool = False
    raw_delim: bool = False


@dataclass(frozen=True)
class BlockComment:
    open_seq: str
    close_seq: str
    nestable: bool = False


@dataclass(frozen=True)
class Language:
    name: str
    extensions: tuple[str, ...] = ()
    filenames: tuple[str, ...] = ()
    line_comments: tuple[str, ...] = ()
    block_comments: tuple[BlockComment, ...] = ()
    strings: tuple[StringSpec, ...] = ()
    # How this language writes docstrings, so a bare documentation string counts
    # as a comment rather than as code. See counting.py for the exact rules.
    #   ""       - no docstring convention
    #   "python" - inside the block, after a `def`/`class` header or at file top
    #   "julia"  - immediately *before* the definition it documents
    docstring_style: str = ""
    # `'x'` character literals where `'` is also a postfix operator. In Julia `A'`
    # is the adjoint, so a `'` may only open a literal when it encloses exactly
    # one character or escape -- otherwise `A' * B` would open a phantom string.
    char_literals: bool = False
    # Treated as source for reduction accounting by default. Config, data and
    # prose languages are reported but excluded from the default code scope so
    # the reduction denominator cannot be padded with YAML and Markdown.
    is_source: bool = True
    # Token-level clone detection is only meaningful for languages with a
    # C/ALGOL-ish token structure; markup and data files are skipped.
    clone_detectable: bool = True
    # `/.../` regex literals. Without this the scanner reads a quote inside a
    # character class -- `/[!$&'()*+,;=]/` -- as opening a string literal, which
    # then fails to terminate. Real and common: it is how URI-validation regexes
    # are written.
    regex_literals: bool = False
    keywords: frozenset[str] = field(default_factory=frozenset)


# --- shared literal / comment shapes -----------------------------------------

_C_BLOCK = (BlockComment("/*", "*/"),)
_DQ = StringSpec('"', '"')
_SQ = StringSpec("'", "'")
_BACKTICK_ML = StringSpec("`", "`", multiline=True)
_C_STRINGS = (_DQ, _SQ)
_JS_STRINGS = (_DQ, _SQ, _BACKTICK_ML)

# Keyword sets exist so the clone tokenizer can keep control flow visible while
# normalising identifiers. They do not need to be exhaustive: an unlisted
# keyword is normalised like an identifier, which only makes clone detection
# slightly more permissive, never unsound.
_JS_KEYWORDS = frozenset(
    """await async break case catch class const continue debugger default delete do else
    export extends finally for function if import in instanceof let new of return static
    super switch this throw try typeof var void while with yield""".split()
)
_TS_KEYWORDS = _JS_KEYWORDS | frozenset(
    """abstract as declare enum implements interface is keyof namespace never private
    protected public readonly satisfies type unknown""".split()
)
_PY_KEYWORDS = frozenset(
    """and as assert async await break class continue def del elif else except finally for
    from global if import in is lambda match nonlocal not or pass raise return try while
    with yield""".split()
)
_RUST_KEYWORDS = frozenset(
    """as async await break const continue crate dyn else enum extern fn for if impl in let
    loop match mod move mut pub ref return self static struct super trait type unsafe use
    where while""".split()
)
_GO_KEYWORDS = frozenset(
    """break case chan const continue default defer else fallthrough for func go goto if
    import interface map package range return select struct switch type var""".split()
)
_JAVA_KEYWORDS = frozenset(
    """abstract assert break case catch class const continue default do else enum extends
    final finally for goto if implements import instanceof interface native new package
    private protected public return static strictfp super switch synchronized this throw
    throws transient try volatile while""".split()
)
_C_KEYWORDS = frozenset(
    """auto break case const continue default do else enum extern for goto if inline
    register restrict return sizeof static struct switch typedef union volatile while""".split()
)
_CPP_KEYWORDS = _C_KEYWORDS | frozenset(
    """catch class constexpr delete dynamic_cast explicit friend mutable namespace new
    operator private protected public reinterpret_cast static_cast template this throw try
    typeid typename using virtual""".split()
)
_JULIA_KEYWORDS = frozenset(
    """baremodule begin break catch const continue do else elseif end export false
    finally for function global if import in isa let local macro module mutable
    primitive quote return struct true try using where while abstract type""".split()
)
_RUBY_KEYWORDS = frozenset(
    """alias begin break case class def do else elsif end ensure for if in module next nil
    redo rescue retry return self super then unless until when while yield""".split()
)


LANGUAGES: tuple[Language, ...] = (
    Language(
        name="Python",
        extensions=(".py", ".pyi", ".pyw"),
        line_comments=("#",),
        strings=(
            StringSpec('"""', '"""', multiline=True),
            StringSpec("'''", "'''", multiline=True),
            _DQ,
            _SQ,
        ),
        docstring_style="python",
        keywords=_PY_KEYWORDS,
    ),
    Language(
        name="Julia",
        extensions=(".jl",),
        line_comments=("#",),
        # `#= ... =#` nests, unlike most block comments.
        block_comments=(BlockComment("#=", "=#", nestable=True),),
        strings=(
            StringSpec('"""', '"""', multiline=True),
            # Unlike Python, a Julia `"..."` string may contain literal newlines.
            # Base relies on it for multi-line field docs and for string macros
            # such as `KSet"begin while\n if for"`.
            StringSpec('"', '"', multiline=True),
            # Command literals. The triple form must precede the single one so it
            # wins the longest-match sort.
            StringSpec("```", "```", multiline=True),
            StringSpec("`", "`", multiline=True),
        ),
        # Julia docstrings sit above the definition, not inside the block.
        docstring_style="julia",
        # `'` is the adjoint operator as well as a char delimiter.
        char_literals=True,
        keywords=_JULIA_KEYWORDS,
    ),
    Language(
        name="JavaScript",
        extensions=(".js", ".mjs", ".cjs", ".jsx"),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=_JS_STRINGS,
        keywords=_JS_KEYWORDS,
        regex_literals=True,
    ),
    Language(
        name="TypeScript",
        extensions=(".ts", ".mts", ".cts", ".tsx"),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=_JS_STRINGS,
        keywords=_TS_KEYWORDS,
        regex_literals=True,
    ),
    Language(
        name="Rust",
        extensions=(".rs",),
        line_comments=("//",),
        block_comments=(BlockComment("/*", "*/", nestable=True),),
        strings=(
            StringSpec('br"', '"', escape=False, multiline=True, raw_hash=True),
            StringSpec('r"', '"', escape=False, multiline=True, raw_hash=True),
            StringSpec('"', '"', multiline=True),
            _SQ,
        ),
        keywords=_RUST_KEYWORDS,
    ),
    Language(
        name="Go",
        extensions=(".go",),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=(_DQ, _SQ, StringSpec("`", "`", escape=False, multiline=True)),
        keywords=_GO_KEYWORDS,
    ),
    Language(
        name="Java",
        extensions=(".java",),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=(StringSpec('"""', '"""', multiline=True), _DQ, _SQ),
        keywords=_JAVA_KEYWORDS,
    ),
    Language(
        name="Kotlin",
        extensions=(".kt", ".kts"),
        line_comments=("//",),
        block_comments=(BlockComment("/*", "*/", nestable=True),),
        strings=(StringSpec('"""', '"""', multiline=True), _DQ, _SQ),
        keywords=_JAVA_KEYWORDS | frozenset("fun val var when object companion".split()),
    ),
    Language(
        name="Swift",
        extensions=(".swift",),
        line_comments=("//",),
        block_comments=(BlockComment("/*", "*/", nestable=True),),
        strings=(StringSpec('"""', '"""', multiline=True), _DQ),
        keywords=frozenset(
            """associatedtype case class deinit enum extension fileprivate func guard if
            import init inout internal let open operator private protocol public repeat return
            static struct subscript typealias var where while""".split()
        ),
    ),
    Language(
        name="C",
        extensions=(".c", ".h"),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=_C_STRINGS,
        keywords=_C_KEYWORDS,
    ),
    Language(
        name="C++",
        extensions=(".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".ipp", ".inl"),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=(StringSpec('R"', '"', escape=False, multiline=True, raw_delim=True), _DQ, _SQ),
        keywords=_CPP_KEYWORDS,
    ),
    Language(
        name="Objective-C",
        extensions=(".m", ".mm"),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=_C_STRINGS,
        keywords=_C_KEYWORDS,
    ),
    Language(
        name="C#",
        extensions=(".cs",),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=(StringSpec('"""', '"""', multiline=True), StringSpec('@"', '"', escape=False, doubling=True, multiline=True), _DQ, _SQ),
        keywords=_JAVA_KEYWORDS
        | frozenset(
            (
                "async await base bool byte checked decimal delegate event fixed "
                "foreach get in is lock namespace object out override params readonly "
                "ref sbyte set sizeof stackalloc string struct typeof uint ulong "
                "unchecked unsafe ushort using var virtual where yield"
            ).split()
        ),
    ),
    Language(
        name="Ruby",
        extensions=(".rb", ".rake", ".gemspec"),
        filenames=("Rakefile", "Gemfile"),
        line_comments=("#",),
        block_comments=(BlockComment("=begin", "=end"),),
        strings=(_DQ, _SQ),
        keywords=_RUBY_KEYWORDS,
    ),
    Language(
        name="PHP",
        extensions=(".php", ".phtml"),
        line_comments=("//", "#"),
        block_comments=_C_BLOCK,
        strings=(_DQ, _SQ),
        keywords=_JAVA_KEYWORDS | frozenset("echo elseif endforeach endif foreach fn function global require require_once include include_once isset list print unset use".split()),
    ),
    Language(
        name="Shell",
        extensions=(".sh", ".bash", ".zsh", ".ksh"),
        line_comments=("#",),
        strings=(StringSpec('"', '"', multiline=True), StringSpec("'", "'", escape=False, multiline=True)),
        keywords=frozenset("case do done elif else esac fi for function if in local return select then until while".split()),
    ),
    Language(
        name="Dart",
        extensions=(".dart",),
        line_comments=("//",),
        block_comments=(BlockComment("/*", "*/", nestable=True),),
        strings=(StringSpec('"""', '"""', multiline=True), StringSpec("'''", "'''", multiline=True), _DQ, _SQ),
        keywords=_JAVA_KEYWORDS | frozenset("await async factory get late required set var".split()),
    ),
    Language(
        name="Scala",
        extensions=(".scala", ".sc"),
        line_comments=("//",),
        block_comments=(BlockComment("/*", "*/", nestable=True),),
        strings=(StringSpec('"""', '"""', multiline=True), _DQ, _SQ),
        keywords=_JAVA_KEYWORDS | frozenset("def match object trait val var with yield given using".split()),
    ),
    Language(
        name="Elixir",
        extensions=(".ex", ".exs"),
        line_comments=("#",),
        strings=(StringSpec('"""', '"""', multiline=True), _DQ, _SQ),
        keywords=frozenset("case cond def defmacro defmodule defp do else end fn for if import raise receive rescue try unless use when with".split()),
    ),
    Language(
        name="Lua",
        extensions=(".lua",),
        line_comments=("--",),
        block_comments=(BlockComment("--[[", "]]"),),
        strings=(StringSpec("[[", "]]", escape=False, multiline=True), _DQ, _SQ),
        keywords=frozenset("and break do else elseif end for function if in local not or repeat return then until while".split()),
    ),
    Language(
        name="Haskell",
        extensions=(".hs", ".lhs"),
        line_comments=("--",),
        block_comments=(BlockComment("{-", "-}", nestable=True),),
        strings=(_DQ, _SQ),
        keywords=frozenset("case class data deriving do else if import in instance let module newtype of then type where".split()),
    ),
    Language(
        name="Zig",
        extensions=(".zig",),
        line_comments=("//",),
        strings=(_DQ, _SQ),
        keywords=frozenset("break catch comptime const continue defer else enum error export extern fn for if inline or orelse pub return struct switch test try union var while".split()),
    ),
    Language(
        name="R",
        extensions=(".r", ".R"),
        line_comments=("#",),
        strings=(_DQ, _SQ),
        keywords=frozenset("break else for function if in next repeat return while".split()),
    ),
    Language(
        name="Perl",
        extensions=(".pl", ".pm", ".t"),
        line_comments=("#",),
        strings=(_DQ, _SQ),
        keywords=frozenset("do else elsif for foreach if my our package return sub unless until use while".split()),
    ),
    Language(
        name="Solidity",
        extensions=(".sol",),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=(_DQ, _SQ),
        keywords=frozenset("contract else emit enum event external for function if import internal library mapping memory modifier new payable private public pure require return returns revert storage struct using view while".split()),
    ),
    Language(
        name="SQL",
        extensions=(".sql",),
        line_comments=("--",),
        block_comments=_C_BLOCK,
        strings=(StringSpec("'", "'", escape=False, doubling=True, multiline=True), StringSpec('"', '"', escape=False, doubling=True)),
        keywords=frozenset("and as by create delete drop from group having insert into join left not null on or order select set table union update values where".split()),
    ),
    Language(
        name="HCL",
        extensions=(".tf", ".tfvars", ".hcl"),
        line_comments=("#", "//"),
        block_comments=_C_BLOCK,
        strings=(_DQ,),
        clone_detectable=False,
    ),
    Language(
        name="Protobuf",
        extensions=(".proto",),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=(_DQ, _SQ),
        clone_detectable=False,
    ),
    Language(
        name="GraphQL",
        extensions=(".graphql", ".gql"),
        line_comments=("#",),
        strings=(StringSpec('"""', '"""', multiline=True), _DQ),
        clone_detectable=False,
    ),
    Language(
        name="Vue",
        extensions=(".vue",),
        line_comments=("//",),
        block_comments=(BlockComment("/*", "*/"), BlockComment("<!--", "-->")),
        strings=_JS_STRINGS,
        keywords=_TS_KEYWORDS,
        regex_literals=True,
    ),
    Language(
        name="Svelte",
        extensions=(".svelte",),
        line_comments=("//",),
        block_comments=(BlockComment("/*", "*/"), BlockComment("<!--", "-->")),
        strings=_JS_STRINGS,
        keywords=_TS_KEYWORDS,
        regex_literals=True,
    ),
    Language(
        name="CSS",
        extensions=(".css",),
        block_comments=_C_BLOCK,
        strings=(_DQ, _SQ),
        clone_detectable=False,
    ),
    Language(
        name="Sass",
        extensions=(".scss", ".sass", ".less"),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=(_DQ, _SQ),
        clone_detectable=False,
    ),
    Language(
        name="HTML",
        extensions=(".html", ".htm", ".xhtml"),
        block_comments=(BlockComment("<!--", "-->"),),
        strings=(),
        clone_detectable=False,
    ),
    Language(
        name="Makefile",
        extensions=(".mk",),
        filenames=("Makefile", "makefile", "GNUmakefile"),
        line_comments=("#",),
        strings=(),
        is_source=False,
        clone_detectable=False,
    ),
    Language(
        name="Dockerfile",
        extensions=(".dockerfile",),
        filenames=("Dockerfile",),
        line_comments=("#",),
        strings=(_DQ, _SQ),
        is_source=False,
        clone_detectable=False,
    ),
    # --- reported but outside the default code scope --------------------------
    Language(
        name="YAML",
        extensions=(".yml", ".yaml"),
        line_comments=("#",),
        strings=(_DQ, _SQ),
        is_source=False,
        clone_detectable=False,
    ),
    Language(
        name="TOML",
        extensions=(".toml",),
        line_comments=("#",),
        strings=(StringSpec('"""', '"""', multiline=True), _DQ, _SQ),
        is_source=False,
        clone_detectable=False,
    ),
    Language(
        name="JSON",
        extensions=(".json",),
        strings=(_DQ,),
        is_source=False,
        clone_detectable=False,
    ),
    Language(
        name="JSON with comments",
        extensions=(".jsonc", ".json5"),
        line_comments=("//",),
        block_comments=_C_BLOCK,
        strings=(_DQ, _SQ),
        is_source=False,
        clone_detectable=False,
    ),
    Language(
        name="XML",
        extensions=(".xml", ".xsd", ".xsl", ".plist"),
        block_comments=(BlockComment("<!--", "-->"),),
        is_source=False,
        clone_detectable=False,
    ),
    Language(
        name="Markdown",
        extensions=(".md", ".markdown", ".mdx"),
        block_comments=(BlockComment("<!--", "-->"),),
        is_source=False,
        clone_detectable=False,
    ),
    Language(
        name="reStructuredText",
        extensions=(".rst",),
        is_source=False,
        clone_detectable=False,
    ),
    Language(
        name="Plain Text",
        extensions=(".txt",),
        is_source=False,
        clone_detectable=False,
    ),
)


def _build_index() -> tuple[dict[str, Language], dict[str, Language]]:
    by_ext: dict[str, Language] = {}
    by_name: dict[str, Language] = {}
    for lang in LANGUAGES:
        for ext in lang.extensions:
            # First declaration wins so the table order documents precedence
            # for extensions shared between languages (e.g. `.h`, `.m`).
            by_ext.setdefault(ext.lower(), lang)
        for fname in lang.filenames:
            by_name.setdefault(fname, lang)
    return by_ext, by_name


_BY_EXT, _BY_FILENAME = _build_index()

# Language names scc reports that we map onto our own names, so a baseline taken
# with scc and a re-measure taken with scc stay comparable even if the display
# name differs. Only names that actually differ need an entry.
SCC_NAME_ALIASES: dict[str, str] = {
    "C Header": "C",
    "C++ Header": "C++",
    "JSX": "JavaScript",
    "TSX": "TypeScript",
    "TypeScript Typings": "TypeScript",
    "BASH": "Shell",
    "Bourne Shell": "Shell",
    "Zsh": "Shell",
    "Happy": "Haskell",
    "SASS": "Sass",
    "Objective C": "Objective-C",
    "Objective C++": "Objective-C",
    "C#": "C#",
    "Markdown": "Markdown",
    "Plain Text": "Plain Text",
    "JSON5": "JSON with comments",
    "Terraform": "HCL",
    "Docker ignore": "Plain Text",
}

def detect(path: str) -> Language | None:
    """Return the language for `path`, or None when the file is not recognised."""
    base = os.path.basename(path)
    lang = _BY_FILENAME.get(base)
    if lang is not None:
        return lang
    # `.d.ts` and friends: longest matching suffix wins over the plain extension.
    lowered = base.lower()
    for compound in (".d.ts", ".d.mts", ".d.cts"):
        if lowered.endswith(compound):
            return _BY_EXT[".ts"]
    _, ext = os.path.splitext(base)
    if not ext:
        return None
    return _BY_EXT.get(ext.lower())


def by_name(name: str) -> Language | None:
    """Look up a language by our canonical name, resolving scc's aliases."""
    canonical = SCC_NAME_ALIASES.get(name, name)
    for lang in LANGUAGES:
        if lang.name == canonical:
            return lang
    return None


def is_source_language(name: str) -> bool:
    """True when a language name counts toward the reduction target.

    Unknown languages default to True: a language we do not model is more likely
    to be real source than to be config, and silently dropping it from the
    denominator would understate the codebase.
    """
    lang = by_name(name)
    return True if lang is None else lang.is_source
