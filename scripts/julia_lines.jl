# Parser-backed line classification for Julia.
#
# Emits one TSV row per file: path <TAB> code <TAB> comment <TAB> blank
# and `path <TAB> ERR <TAB> <reason>` for anything that cannot be classified,
# so a failure is never silently counted as zero.
#
# Used by `slopfix --counter julia`. The heuristic scanner in counting.py reaches
# ~96% per-file agreement with this; this is the same definition computed by the
# language itself, so it agrees by construction.
#
# Definition, matching the builtin counter:
#   blank   - the line is entirely whitespace
#   comment - every non-whitespace character is comment trivia or a docstring
#   code    - anything else
#
# Usage:  julia julia_lines.jl FILE...
#         printf 'FILE1\0FILE2' | julia julia_lines.jl --stdin0
#         julia julia_lines.jl --version-id

const JS = Base.JuliaSyntax

using Base64

"Byte offset (1-based) -> line number, via binary search over line starts."
function line_of(offsets::Vector{Int}, byte::Int)
    lo, hi = 1, length(offsets)
    while lo < hi
        mid = (lo + hi + 1) >> 1
        offsets[mid] <= byte ? (lo = mid) : (hi = mid - 1)
    end
    lo
end

function classify(path::AbstractString)
    src = read(path, String)
    lines = split(src, '\n')
    # `split` yields a trailing empty element for a final newline; drop it so the
    # line count matches what every other counter reports.
    if !isempty(lines) && isempty(lines[end])
        pop!(lines)
    end
    offsets = Int[1]
    for l in lines
        push!(offsets, offsets[end] + sizeof(l) + 1)
    end

    codelines = Set{Int}()
    commentlines = Set{Int}()
    codetokens = UnitRange{Int}[]
    for t in JS.tokenize(src)
        k = string(JS.kind(t))
        a, b = Int(first(t.range)), Int(last(t.range))
        b < a && continue
        for ln in line_of(offsets, a):line_of(offsets, b)
            if k == "Comment"
                push!(commentlines, ln)
            elseif !(k in ("Whitespace", "NewlineWs", "EndMarker"))
                push!(codelines, ln)
            end
        end
        if !(k in ("Comment", "Whitespace", "NewlineWs", "EndMarker"))
            push!(codetokens, a:b)
        end
    end

    # Docstrings: a `doc` node's first child is the documentation string.
    # `ignore_warnings` is essential -- parseall raises ParseError for mere
    # warnings (Base/iterators.jl does), and treating that as "no docstrings"
    # would silently count every docstring in the file as code.
    docl = Set{Int}()
    docranges = UnitRange{Int}[]
    tree = JS.parseall(JS.SyntaxNode, src; filename = String(path),
                       ignore_warnings = true)
    function walk(n)
        if occursin("doc", string(JS.head(n)))
            ch = JS.children(n)
            if !isnothing(ch) && !isempty(ch)
                r = JS.byte_range(ch[1])
                push!(docranges, Int(first(r)):Int(last(r)))
                for ln in line_of(offsets, Int(first(r))):line_of(offsets, Int(last(r)))
                    push!(docl, ln)
                end
            end
        end
        ch = JS.children(n)
        isnothing(ch) || foreach(walk, ch)
    end
    walk(tree)

    blank = Set(ln for (ln, l) in enumerate(lines) if isempty(strip(l)))
    # A docstring line is documentation unless it also carries a real token
    # outside the parser's docstring byte range. Julia permits the compact form
    # `"""docs""" f(x) = x`; that physical line is code under the counter's
    # contract because the definition is observable code.
    sort!(docranges; by=first)
    docstarts = first.(docranges)
    function token_is_in_doc(r::UnitRange{Int})
        i = searchsortedlast(docstarts, first(r))
        i > 0 && last(r) <= last(docranges[i])
    end
    code_outside_doc = Set{Int}()
    for r in codetokens
        token_is_in_doc(r) && continue
        for ln in line_of(offsets, first(r)):line_of(offsets, last(r))
            push!(code_outside_doc, ln)
        end
    end
    doc_only = setdiff(setdiff(docl, blank), code_outside_doc)
    code = setdiff(setdiff(codelines, blank), doc_only)
    comment = setdiff(union(setdiff(commentlines, blank), doc_only), code)
    (code = length(code), comment = length(comment), blank = length(blank))
end

function ast_contains(node, target)
    isequal(node, target) && return true
    node isa Expr || return false
    node.head in (:quote, :inert) && return false
    any(child -> ast_contains(child, target), node.args)
end

function include_path(node)
    node isa Expr || return nothing, false
    node.head in (:quote, :inert) && return nothing, false
    if node.head == :call && !isempty(node.args)
        callee = node.args[1]
        is_include = callee == :include || (
            callee isa Expr && callee.head == :. &&
            !isempty(callee.args) && callee.args[end] == QuoteNode(:include)
        )
        if is_include
            paths = [arg for arg in node.args[2:end] if arg isa String]
            return isempty(paths) ? nothing : paths[end], isempty(paths)
        end
    end
    return nothing, false
end

function collect_includes!(node, paths::Vector{String}, dynamic::Base.RefValue{Bool})
    node isa Expr || return
    node.head in (:quote, :inert) && return
    path, unresolved = include_path(node)
    unresolved && (dynamic[] = true)
    isnothing(path) || push!(paths, path)
    foreach(child -> collect_includes!(child, paths, dynamic), node.args)
end

function inside_root(root::String, path::String)
    relative = relpath(path, root)
    separator = string(Base.Filesystem.path_separator)
    relative != ".." && !startswith(relative, ".." * separator) && !isabspath(relative)
end

function emit_field(kind::String, fields...)
    encoded = base64encode.(string.(fields))
    println(join((kind, encoded...), '\t'))
end

function emit_status(status::String, message::String)
    println("STATUS", '\t', status, '\t', base64encode(message))
end

function reachable_contains()
    fields = split(read(stdin, String), '\0'; keepempty = false)
    length(fields) >= 4 ||
        error("reachable input requires root, entrypoint, limit, and needles")
    root = realpath(fields[1])
    entrypoint = fields[2]
    limit = parse(Int, fields[3])
    needles = fields[4:end]
    targets = Dict(needle => Meta.parse(needle) for needle in needles)
    pending = String[normpath(joinpath(root, entrypoint))]
    visited = Set{String}()
    matches = Dict(needle => Set{String}() for needle in needles)
    incomplete = false
    failure = nothing

    while !isempty(pending)
        candidate = pop!(pending)
        if !isfile(candidate)
            failure = "reachable include does not exist: " * relpath(candidate, root)
            break
        end
        path = realpath(candidate)
        if !inside_root(root, path)
            failure = "reachable include escapes repository: " * path
            break
        end
        path in visited && continue
        if length(visited) >= limit
            incomplete = true
            break
        end
        push!(visited, path)
        source = read(path, String)
        tree = try
            Meta.parseall(source; filename=path)
        catch error
            emit_status(
                "UNVERIFIED",
                "could not parse $(relpath(path, root)): $(sprint(showerror, error))",
            )
            foreach(item -> emit_field("FILE", relpath(item, root)), visited)
            return
        end
        for (needle, target) in targets
            ast_contains(tree, target) &&
                push!(matches[needle], relpath(path, root))
        end
        includes = String[]
        dynamic = Ref(false)
        collect_includes!(tree, includes, dynamic)
        incomplete |= dynamic[]
        append!(
            pending,
            (normpath(joinpath(dirname(path), item)) for item in includes),
        )
    end

    missing = [needle for needle in needles if isempty(matches[needle])]
    status, message = if !isnothing(failure)
        "FAIL", failure
    elseif isempty(missing)
        "PASS", "all required Julia expressions are in the reachable test graph"
    elseif incomplete
        "UNVERIFIED", "reachable graph is incomplete; missing: " * join(missing, ", ")
    else
        "FAIL", "missing from reachable Julia test graph: " * join(missing, ", ")
    end
    emit_status(status, message)
    foreach(path -> emit_field("FILE", relpath(path, root)), visited)
    for needle in needles, path in matches[needle]
        emit_field("MATCH", needle, path)
    end
end

function main()
    if length(ARGS) == 1 && ARGS[1] == "--version-id"
        println("julia/", VERSION)
        return
    end
    if length(ARGS) == 1 && ARGS[1] == "--reachable-contains"
        reachable_contains()
        return
    end
    paths = if length(ARGS) == 1 && ARGS[1] == "--stdin0"
        split(read(stdin, String), '\0'; keepempty = false)
    else
        ARGS
    end
    for p in paths
        try
            r = classify(p)
            println(p, '\t', r.code, '\t', r.comment, '\t', r.blank)
        catch e
            msg = replace(sprint(showerror, e), '\t' => ' ', '\n' => ' ')
            println(p, '\t', "ERR", '\t', first(msg, 160))
        end
    end
end

main()
