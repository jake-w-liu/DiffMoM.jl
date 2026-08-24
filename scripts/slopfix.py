#!/usr/bin/env python3
"""slopfix — measurement and census tooling for AI-slop reduction engagements.

    slopfix doctor                     what is installed, and what to install
    slopfix baseline --target 50       freeze the pre-work measurement
    slopfix measure                    re-measure against the frozen baseline
    slopfix census                     find duplication and consolidation targets
    slopfix smells                     find blocking slop patterns
    slopfix quality-init               create the wider quality assurance contract
    slopfix quality-check --run        execute and record that contract

The reduction number this tool reports is only as honest as the baseline it was
pinned against, so `baseline` must run before any code changes and `measure`
replays that exact scope and counter. `measure` refuses to compare across
counters and reports integrity findings (parked code, code golf, stripped
comments, deleted tests) alongside the number.

Measurement, census, and smell scans do not modify source files. ``quality-init``
writes its requested config artifact. ``quality-check --run`` executes the
commands in that config, so inspect them before opting in.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slopfix_lib import (
    clones,
    concepts,
    counting,
    langs,
    manifest,
    quality,
    scope,
    smells,
)

DEFAULT_MANIFEST = ".slopfix/baseline.json"
DEFAULT_QUALITY_CONFIG = ".slopfix/quality.json"
DEFAULT_QUALITY_REPORT = ".slopfix/quality-report.json"


# --- shared plumbing ---------------------------------------------------------

def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed


def _clone_window(value: str) -> int:
    parsed = int(value)
    if parsed < 8:
        raise argparse.ArgumentTypeError("must be at least 8 tokens")
    return parsed


def _percentage(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed


def _add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", default=".", help="repository root to measure (default: .)"
    )
    parser.add_argument(
        "--exclude-dir", action="append", default=[], metavar="NAME",
        help="additional directory name to exclude (repeatable)",
    )
    parser.add_argument(
        "--exclude-glob", action="append", default=[], metavar="GLOB",
        help="additional path glob to exclude (repeatable)",
    )
    parser.add_argument(
        "--include-non-source", action="store_true",
        help="count config/data/prose languages (YAML, JSON, Markdown) toward "
             "the target as well as source code",
    )
    parser.add_argument(
        "--max-file-bytes", type=_positive_int, default=5 * 1024 * 1024,
        help="skip files larger than this (default: 5 MiB)",
    )
    parser.add_argument(
        "--no-gitignore", action="store_true",
        help="ignore .gitignore and walk the filesystem instead",
    )


def _scope_from_args(args: argparse.Namespace) -> scope.Scope:
    return scope.Scope(
        root=os.path.abspath(args.root),
        exclude_dirs=scope.DEFAULT_EXCLUDE_DIRS + tuple(args.exclude_dir),
        exclude_globs=tuple(args.exclude_glob),
        generated_globs=scope.DEFAULT_GENERATED_GLOBS,
        include_non_source=args.include_non_source,
        max_file_bytes=args.max_file_bytes,
        respect_gitignore=not args.no_gitignore,
    )


def _emit(payload: dict, as_json: bool, render) -> int:
    if as_json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        render(payload)
    return 0


def _bar(pct: float, width: int = 28) -> str:
    filled = max(0, min(width, round(pct / 100.0 * width)))
    return "#" * filled + "." * (width - filled)


def _iter_sources(sc: scope.Scope):
    """Yield (relpath, language, text) for every in-scope file that reads."""
    discovery = scope.discover(sc)
    for relpath, lang in discovery.counted:
        text, reason = counting.read_source(
            os.path.join(sc.root, relpath), sc.max_file_bytes
        )
        if text is None:
            raise counting.SourceReadError(f"{relpath}: {reason}")
        yield relpath, lang, text


# --- doctor ------------------------------------------------------------------

_OPTIONAL_TOOLS = (
    ("scc", "brew install scc  |  go install github.com/boyter/scc/v3@latest",
     "canonical line counter; the reduction number is quoted against its Code column"),
    ("git", "install git", "scope discovery via `git ls-files`, and atomic per-step commits"),
    ("julia", "https://julialang.org/downloads (or juliaup)",
     "`--counter julia` classifies .jl files with Julia's own tokenizer/parser"),
    ("ruff", "pipx install ruff", "authoritative Python smell and unused-code detection"),
    ("eslint", "npm i -D eslint", "authoritative JS/TS smell detection, incl. shadowing"),
    ("jscpd", "npm i -g jscpd", "second opinion on duplication, with HTML reports"),
    ("gitleaks", "https://github.com/gitleaks/gitleaks",
     "repository secret scanning for the quality security gate"),
    ("jh", "https://help.juliahub.com/juliahub/stable/tutorials/juliahub_cli/",
     "Julia-aware dependency vulnerability scanning where available"),
    ("cargo", "install Rust toolchain", "clippy for Rust smell detection"),
    ("go", "install Go", "go vet and staticcheck"),
)


def cmd_doctor(args: argparse.Namespace) -> int:
    rows = []
    for name, install, why in _OPTIONAL_TOOLS:
        path = shutil.which(name)
        rows.append({
            "tool": name, "found": path is not None, "path": path,
            "install": install, "why": why,
        })
    counter = manifest.resolve_counter("auto")
    try:
        identity = manifest.counter_identity(counter)
    except counting.SccUnavailable as exc:
        counter, identity = "builtin", f"{counting.BUILTIN_COUNTER_ID} ({exc})"
    try:
        julia_counter = manifest.counter_identity("julia")
    except counting.JuliaUnavailable:
        julia_counter = None
    payload = {
        "python": sys.version.split()[0],
        "counter_that_would_be_used": counter,
        "julia_counter": julia_counter,
        "counter_identity": identity,
        "languages_modelled": len(langs.LANGUAGES),
        "tools": rows,
    }

    def render(data: dict) -> None:
        print("slopfix doctor")
        print(f"  python              {data['python']}")
        print(f"  counter            {data['counter_that_would_be_used']} "
              f"-> {data['counter_identity']}")
        print(f"  languages modelled {data['languages_modelled']}")
        print()
        for row in data["tools"]:
            mark = "ok  " if row["found"] else "MISS"
            print(f"  [{mark}] {row['tool']:<8} {row['why']}")
            if not row["found"]:
                print(f"           install: {row['install']}")
        if data.get("julia_counter"):
            print()
            print("  julia is available, so `--counter julia` can be pinned at a")
            print(f"  baseline: {data['julia_counter']}. It classifies .jl files")
            print("  with Julia's own tokenizer and parser, and uses the")
            print("  builtin scanner for every other language. julia must still be")
            print("  present at measure time, or the comparison is refused.")
        if data["counter_that_would_be_used"] != "scc":
            print()
            print("  scc is absent, so the builtin counter would be used. It applies")
            print("  the same definition (non-blank, non-comment lines) but records a")
            print("  different counter identity. Install scc before taking a baseline")
            print("  if the engagement quotes scc numbers.")

    return _emit(payload, args.json, render)


# --- quality assurance -------------------------------------------------------

def cmd_quality_init(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    out_path = args.out or os.path.join(root, DEFAULT_QUALITY_CONFIG)
    if os.path.exists(out_path) and not args.force:
        print(
            f"error: {out_path} already exists. It may contain reviewed commands "
            "and dispositions; pass --force only when intentionally replacing it.",
            file=sys.stderr,
        )
        return 2
    payload = quality.build(root, profile=args.profile)
    quality.write(out_path, payload)
    payload = dict(payload, config_path=out_path)

    def render(data: dict) -> None:
        required = sum(1 for gate in data["gates"] if gate["required"])
        reviews = sum(1 for gate in data["gates"] if gate["kind"] == "review")
        print(f"quality config written to {data['config_path']}")
        print(f"  profile        {data['profile']}")
        print(f"  quality model  {data['quality_model']}")
        print(f"  gates          {len(data['gates'])} ({required} required)")
        print(f"  unresolved     {reviews} review gate(s)")
        print()
        print("Review every command and resolve each review gate as pass, fail,")
        print("unverified, or not_applicable with evidence. The config is executable")
        print("input; `quality-check` validates only until you explicitly pass --run.")

    return _emit(payload, args.json, render)


def cmd_quality_check(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    path = args.config or os.path.join(root, DEFAULT_QUALITY_CONFIG)
    config, config_sha256 = quality.read(path, root)
    if not args.run:
        if args.strict:
            print("error: --strict requires --run", file=sys.stderr)
            return 2
        if args.only:
            print("error: --only requires --run", file=sys.stderr)
            return 2
        payload = {
            "kind": "slopfix-quality-validation",
            "config_path": path,
            "config_sha256": config_sha256,
            "profile": config["profile"],
            "quality_model": config["quality_model"],
            "gates": len(config["gates"]),
            "commands_executed": False,
        }

        def render_validation(data: dict) -> None:
            print("quality config valid")
            print(f"  config         {data['config_path']}")
            print(f"  sha256         {data['config_sha256']}")
            print(f"  profile        {data['profile']}")
            print(f"  quality model  {data['quality_model']}")
            print(f"  gates          {data['gates']}")
            print()
            print("No commands executed. Inspect the config, then pass --run.")

        return _emit(payload, args.json, render_validation)

    report = quality.run(
        config, root, config_sha256,
        only=set(args.only) if args.only else None,
    )
    out_path = args.out or os.path.join(root, DEFAULT_QUALITY_REPORT)
    if not args.no_write:
        quality.write(out_path, report)
        report = dict(report, report_path=out_path)

    def render_report(data: dict) -> None:
        summary = data["summary"]
        print("slopfix quality-check")
        print(f"  profile        {data['profile']}")
        print(f"  config sha256  {data['config_sha256']}")
        print(f"  gates          {summary['total']}"
              + (" (partial run)" if data["partial"] else ""))
        for status in quality.STATUSES:
            print(f"  {status.lower():<14} {summary['by_status'][status]}")
        print()
        for gate in data["gates"]:
            marker = "*" if gate["required"] else "-"
            print(f"  [{gate['status']:<14}] {marker} {gate['id']}: "
                  f"{gate['message']}")
        print()
        print("  * required gate")
        if "report_path" in data:
            print(f"  report written to {data['report_path']}")
        if data["partial"]:
            print("  This is a partial gate run; it is not evidence for the full model.")
        print(f"  strict verdict: {'PASS' if summary['strict_pass'] else 'FAIL'}")

    _emit(report, args.json, render_report)
    if report["summary"]["failures"]:
        return 1
    if args.strict and report["summary"]["required_unverified"]:
        return 1
    return 0


# --- baseline ----------------------------------------------------------------

def cmd_baseline(args: argparse.Namespace) -> int:
    sc = _scope_from_args(args)
    if not os.path.isdir(sc.root):
        print(f"error: {sc.root} is not a directory", file=sys.stderr)
        return 2
    out_path = args.out or os.path.join(sc.root, DEFAULT_MANIFEST)
    if os.path.exists(out_path) and not args.force:
        print(
            f"error: {out_path} already exists.\n"
            "A baseline is meant to be taken once, before any code changes. "
            "Overwriting it after work has started destroys the number it exists "
            "to protect. Pass --force only if no reduction work has happened yet.",
            file=sys.stderr,
        )
        return 2
    try:
        payload = manifest.build(
            root=sc.root, sc=sc, counter=args.counter,
            promised_reduction_pct=args.target,
        )
    except (counting.SourceReadError,
            counting.SccUnavailable, counting.SccOutputError,
            counting.JuliaUnavailable, counting.JuliaOutputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    manifest.write(out_path, payload)
    payload = dict(payload, manifest_path=out_path)

    def render(data: dict) -> None:
        totals = data["totals"]
        target = data["target"]
        print(f"baseline written to {data['manifest_path']}")
        print(f"  counter        {data['counter']['id']}")
        print(f"  git HEAD       {data['repo']['git_head'] or '(not a git repo)'}")
        print(f"  files counted  {totals['files']}")
        print(f"  code lines     {totals['code']}   "
              f"(comments {totals['comments']}, blanks {totals['blanks']})")
        if target["promised_reduction_pct"] > 0:
            print(f"  promised       {target['promised_reduction_pct']:.1f}% reduction "
                  f"-> target {target['target_code_lines']} code lines "
                  f"({target['lines_to_remove']} to remove)")
        else:
            print("  promised       no target set (pass --target to commit to one)")
        print()
        print("  top languages")
        for name, files, code in manifest.summarise_languages(data)[:12]:
            print(f"    {code:>8}  {files:>4} files  {name}")
        rejected = data["discovery"]["rejected_counts"]
        if rejected:
            print()
            print("  not counted    " + ", ".join(
                f"{count} {bucket}" for bucket, count in rejected.items()
            ))
        if data["warnings"]:
            print()
            print(f"  {len(data['warnings'])} warning(s):")
            for warning in data["warnings"][:10]:
                print(f"    - {warning}")

    return _emit(payload, args.json, render)


# --- measure -----------------------------------------------------------------

def cmd_measure(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    path = args.manifest or os.path.join(root, DEFAULT_MANIFEST)
    if not os.path.exists(path):
        print(
            f"error: no baseline at {path}. Run `slopfix baseline` before any "
            "code changes; a reduction measured against a moving baseline is "
            "not a measurement.",
            file=sys.stderr,
        )
        return 2
    try:
        baseline = manifest.read(path)
        report = manifest.compare(baseline, root, counter=args.counter)
    except (manifest.CounterMismatch, manifest.ScopeMismatch) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (counting.SccUnavailable, counting.SccOutputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # The ratchet: a committed ceiling on in-scope code lines. Growth stays
    # possible but has to be a deliberate edit to whatever file holds the number.
    ceiling_exceeded = False
    if args.ceiling is not None:
        report["ceiling"] = args.ceiling
        report["ceiling_exceeded"] = report["current_code"] > args.ceiling
        ceiling_exceeded = report["ceiling_exceeded"]

    def render(data: dict) -> None:
        print("slopfix measure")
        print(f"  counter          {data['counter']['id']}")
        print(f"  baseline code    {data['baseline_code']}")
        print(f"  current code     {data['current_code']}")
        print(f"  removed / added  {data['gross_removed']} / {data['gross_added']}"
              f"   (net {data['net_removed']})")
        print(f"  reduction        {data['reduction_pct']:.2f}%")
        if data["promised_reduction_pct"] > 0:
            print(f"  promised         {data['promised_reduction_pct']:.1f}% "
                  f"-> target {data['target_code_lines']}")
            print(f"  attainment       {data['attainment_pct']:.2f}%  "
                  f"[{_bar(data['attainment_pct'])}]")
        if data["files_added"]:
            print(f"  files added      {len(data['files_added'])}")
        if data["files_removed"]:
            print(f"  files removed    {len(data['files_removed'])}")
        if "ceiling" in data:
            verdict = "EXCEEDED" if data["ceiling_exceeded"] else "within"
            print(f"  ceiling          {data['ceiling']}  ({verdict})")
        print()
        if data["integrity"]:
            print(f"  {len(data['integrity'])} integrity finding(s) — clear each one "
                  "before reporting the number above:")
            for finding in data["integrity"]:
                print(f"    [{finding['rule']}] {finding['message']}")
        else:
            print("  integrity checks: no findings")
        if data["warnings"]:
            print()
            print(f"  {len(data['warnings'])} warning(s):")
            for warning in data["warnings"][:10]:
                print(f"    - {warning}")

    _emit(report, args.json, render)
    if ceiling_exceeded:
        print(
            f"error: {report['current_code']} code lines exceeds the ceiling of "
            f"{args.ceiling}. Raise the ceiling deliberately, in its own commit, "
            "if real functionality was added — do not park code outside the scope, "
            "collapse lines, or delete tests to get under it.",
            file=sys.stderr,
        )
        return 1
    # A non-zero exit on integrity findings makes this usable as a CI gate.
    return 1 if (args.strict and report["integrity"]) else 0


# --- census ------------------------------------------------------------------

def cmd_census(args: argparse.Namespace) -> int:
    sc = _scope_from_args(args)
    if not os.path.isdir(sc.root):
        print(f"error: {sc.root} is not a directory", file=sys.stderr)
        return 2
    if args.min_tokens < args.window:
        print(
            "error: --min-tokens must be greater than or equal to --window; "
            "otherwise the requested smaller clones cannot be detected",
            file=sys.stderr,
        )
        return 2

    units: list[clones.Unit] = []
    concept_hits: list[concepts.Hit] = []
    all_definitions: list[tuple[str, int, str]] = []
    long_line_skips: dict[str, list[int]] = {}
    files_read = 0

    for relpath, lang, text in _iter_sources(sc):
        files_read += 1
        if lang.clone_detectable:
            units.append(clones.tokenize(relpath, text, lang))
        concept_hits.extend(concepts.scan_text(relpath, text, lang))
        skipped: list[int] = []
        for lineno, symbol in concepts.definitions(
            relpath, text, lang, skipped_lines=skipped
        ):
            all_definitions.append((relpath, lineno, symbol))
        if skipped:
            long_line_skips[relpath] = skipped

    truncated: list[bool] = []
    groups = clones.find_clones(
        units, window=args.window, min_tokens=args.min_tokens,
        max_groups=args.max_groups, truncated=truncated,
    )
    payload = {
        "kind": "slopfix-census",
        "root": sc.root,
        "files_scanned": files_read,
        "clone_groups_truncated": bool(truncated),
        "max_groups": args.max_groups,
        "long_lines_skipped": {p: lines for p, lines in sorted(long_line_skips.items())},
        "duplication": clones.summarise(groups),
        "clone_groups": [group.to_json() for group in groups[: args.top]],
        "concepts": concepts.summarise(concept_hits, min_definitions=args.min_definitions),
        "duplicate_symbols": concepts.duplicate_symbols(all_definitions)[: args.top],
    }

    def render(data: dict) -> None:
        dup = data["duplication"]
        print(f"slopfix census — {data['files_scanned']} files")
        print()
        print("duplication (token-identical after renaming)")
        print(f"  clone groups            {dup['clone_groups']}")
        print(f"  files involved          {dup['files_involved']}")
        print(f"  removable lines (est.)  {dup['removable_lines_estimate']}")
        if data["clone_groups"]:
            print()
            print("  largest clone groups")
            for group in data["clone_groups"][: args.top]:
                print(f"    {group['removable_lines']:>5} removable  "
                      f"{group['copies']} copies  {group['token_count']} tokens")
                for occ in group["occurrences"][:6]:
                    print(f"          {occ['path']}:{occ['start_line']}-{occ['end_line']}")
                if len(group["occurrences"]) > 6:
                    print(f"          ... and {len(group['occurrences']) - 6} more")
        print()
        print("concepts implemented more than once")
        if not data["concepts"]:
            print("  none above the threshold")
        for row in data["concepts"][: args.top]:
            print(f"  {row['definitions']:>3} definitions in {row['files']:>3} files  "
                  f"{row['concept']}  ({row['distinct_names']} distinct names)")
            for site in row["sites"][:6]:
                print(f"        {site['path']}:{site['line']}  {site['symbol']}")
            if len(row["sites"]) > 6:
                print(f"        ... and {len(row['sites']) - 6} more")
        if data["duplicate_symbols"]:
            print()
            print("same name defined in multiple files")
            for row in data["duplicate_symbols"][: args.top]:
                print(f"  {row['definitions']:>3}x  {row['symbol']}  "
                      f"({row['files']} files)")
        if data.get("clone_groups_truncated"):
            print()
            print(f"  NOTE: clone detection stopped at --max-groups "
                  f"{data['max_groups']}. More duplication exists; the group count "
                  "and removable-line estimate above are lower bounds. Raise "
                  "--max-groups for the full picture.")
        if data.get("long_lines_skipped"):
            total = sum(len(v) for v in data["long_lines_skipped"].values())
            print()
            print(f"  NOTE: {total} very long line(s) in "
                  f"{len(data['long_lines_skipped'])} file(s) were skipped by the "
                  "definition scan (likely minified or generated).")
        print()
        print("These are consolidation candidates, not confirmed duplicates.")
        print("Diff the behaviours before merging: the one that differs is usually")
        print("the one handling an edge case the others get wrong.")

    return _emit(payload, args.json, render)


# --- smells ------------------------------------------------------------------

def cmd_smells(args: argparse.Namespace) -> int:
    sc = _scope_from_args(args)
    if not os.path.isdir(sc.root):
        print(f"error: {sc.root} is not a directory", file=sys.stderr)
        return 2
    if args.strict and args.severity == smells.ADVISORY:
        print(
            "error: --strict cannot be combined with --severity advisory "
            "because blocking findings would be hidden",
            file=sys.stderr,
        )
        return 2

    hits: list[smells.Hit] = []
    for relpath, lang, text in _iter_sources(sc):
        hits.extend(smells.scan_text(relpath, text, lang,
                                     god_function_lines=args.god_function_lines))
    if args.severity != "all":
        hits = [hit for hit in hits if hit.severity == args.severity]

    by_file: dict[str, list[smells.Hit]] = {}
    for hit in hits:
        by_file.setdefault(hit.path, []).append(hit)

    payload = {
        "kind": "slopfix-smells",
        "root": sc.root,
        "summary": smells.summarise(hits),
        "hits": [hit.to_json() for hit in hits],
    }

    def render(data: dict) -> None:
        summary = data["summary"]
        print(f"slopfix smells — {summary['total']} hit(s)")
        for severity, count in summary["by_severity"].items():
            print(f"  {severity:<9} {count}")
        print()
        for rule, count in summary["by_rule"].items():
            print(f"  {count:>5}  {rule}")
        print()
        shown = 0
        for path in sorted(by_file):
            if shown >= args.top:
                break
            print(f"{path}")
            for hit in by_file[path]:
                print(f"  {hit.lineno:>5}  [{hit.severity}] {hit.rule}: {hit.message}")
                print(f"         {hit.excerpt[:110]}")
                shown += 1
                if shown >= args.top:
                    print(f"  ... output truncated at --top {args.top}")
                    break
        print()
        print("Language checks (Aqua/JET, ruff, ESLint, Clippy, go vet, etc.)")
        print("remain authoritative. This pass only covers blocking reduction smells.")

    _emit(payload, args.json, render)
    blocking = sum(
        1 for hit in hits if hit.severity == smells.BLOCKING
    )
    return 1 if (args.strict and blocking) else 0


# --- CLI ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slopfix",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")

    # `--json` is accepted on either side of the subcommand, since both read
    # naturally. SUPPRESS on the subparser copy stops its default from
    # overwriting a value the top-level parser already set.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="emit JSON instead of text",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor", parents=[common],
        help="report available tooling and counter identity",
    )
    doctor.set_defaults(func=cmd_doctor)

    baseline = sub.add_parser("baseline", parents=[common],
                             help="freeze the pre-work measurement")
    _add_scope_args(baseline)
    baseline.add_argument(
        "--target", type=_percentage, default=0.0, metavar="PCT",
        help="promised reduction percentage to commit to (e.g. 50)",
    )
    baseline.add_argument(
        "--counter", choices=("auto", "scc", "builtin", "julia"), default="auto",
        help="line counter to pin (default: auto, preferring scc). `julia` is "
             "the builtin scanner with Julia's own parser classifying .jl files "
             "and requires julia on PATH at measure time too.",
    )
    baseline.add_argument("--out", help=f"manifest path (default: <root>/{DEFAULT_MANIFEST})")
    baseline.add_argument(
        "--force", action="store_true",
        help="overwrite an existing baseline (only valid before work starts)",
    )
    baseline.set_defaults(func=cmd_baseline)

    measure = sub.add_parser("measure", parents=[common],
                         help="re-measure against the frozen baseline")
    measure.add_argument("--root", default=".", help="repository root (default: .)")
    measure.add_argument("--manifest", help=f"baseline path (default: <root>/{DEFAULT_MANIFEST})")
    measure.add_argument(
        "--counter", choices=("auto", "scc", "builtin", "julia"), default="auto",
        help="default `auto` replays whichever counter the baseline recorded; "
             "an explicit value must still match it",
    )
    measure.add_argument(
        "--ceiling", type=_nonnegative_int, metavar="N",
        help="fail when in-scope code lines exceed N (the CI ratchet). Keep N in "
             "a version-controlled file so raising it takes a deliberate commit.",
    )
    measure.add_argument(
        "--strict", action="store_true",
        help="exit non-zero when there are integrity findings (for CI)",
    )
    measure.set_defaults(func=cmd_measure)

    census = sub.add_parser("census", parents=[common],
                        help="find duplication and consolidation targets")
    _add_scope_args(census)
    census.add_argument(
        "--window", type=_clone_window, default=60, metavar="N",
        help="clone detection granularity in tokens (default: 60)",
    )
    census.add_argument(
        "--min-tokens", type=_positive_int, default=60, metavar="N",
        help="smallest clone to report, in tokens (default: 60)",
    )
    census.add_argument(
        "--min-definitions", type=_positive_int, default=2, metavar="N",
        help="report a concept once it has this many definitions (default: 2)",
    )
    census.add_argument(
        "--top", type=_nonnegative_int, default=25,
        help="rows per section (default: 25)",
    )
    census.add_argument(
        "--max-groups", type=_positive_int, default=400,
        help="cap on clone groups collected (default: 400)",
    )
    census.set_defaults(func=cmd_census)

    smell = sub.add_parser("smells", parents=[common],
                       help="find blocking slop patterns")
    _add_scope_args(smell)
    smell.add_argument(
        "--severity", choices=("all", smells.BLOCKING, smells.ADVISORY), default="all",
        help="filter by severity (default: all)",
    )
    smell.add_argument(
        "--god-function-lines", type=_positive_int, default=60, metavar="N",
        help="report functions longer than this many code lines (default: 60)",
    )
    smell.add_argument(
        "--top", type=_nonnegative_int, default=80,
        help="max hits to print (default: 80)",
    )
    smell.add_argument(
        "--strict", action="store_true",
        help="exit non-zero when any blocking smell is present (for CI)",
    )
    smell.set_defaults(func=cmd_smells)

    quality_init = sub.add_parser(
        "quality-init", parents=[common],
        help="create the full quality assurance contract",
    )
    quality_init.add_argument("--root", default=".", help="repository root (default: .)")
    quality_init.add_argument(
        "--profile", choices=("auto", "generic", "julia"), default="auto",
        help="quality profile (default: auto-detect Julia)",
    )
    quality_init.add_argument(
        "--out", help=f"config path (default: <root>/{DEFAULT_QUALITY_CONFIG})",
    )
    quality_init.add_argument(
        "--force", action="store_true",
        help="replace an existing quality config intentionally",
    )
    quality_init.set_defaults(func=cmd_quality_init)

    quality_check = sub.add_parser(
        "quality-check", parents=[common],
        help="validate or execute the quality assurance contract",
    )
    quality_check.add_argument("--root", default=".", help="repository root (default: .)")
    quality_check.add_argument(
        "--config", help=f"config path (default: <root>/{DEFAULT_QUALITY_CONFIG})",
    )
    quality_check.add_argument(
        "--run", action="store_true",
        help="execute command gates; without this flag the config is only validated",
    )
    quality_check.add_argument(
        "--strict", action="store_true",
        help="fail on any failed gate or required UNVERIFIED gate (requires --run)",
    )
    quality_check.add_argument(
        "--only", action="append", default=[], metavar="GATE_ID",
        help="run only this gate (repeatable); the report is marked partial",
    )
    quality_check.add_argument(
        "--out", help=f"report path (default: <root>/{DEFAULT_QUALITY_REPORT})",
    )
    quality_check.add_argument(
        "--no-write", action="store_true",
        help="emit results without writing a report artifact",
    )
    quality_check.set_defaults(func=cmd_quality_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except scope.GitListingFailed as exc:
        # Scope discovery could not be trusted. Reporting a number anyway would
        # mean measuring a different file set than the baseline froze.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (counting.SourceReadError,
            counting.SccUnavailable, counting.SccOutputError,
            counting.JuliaUnavailable, counting.JuliaOutputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: filesystem operation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # `slopfix census | head` should not print a traceback.
        try:
            sys.stdout.close()
        except OSError:
            pass
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
