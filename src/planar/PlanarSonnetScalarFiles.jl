export SonnetScalarSource, SonnetScalarTable, SonnetScalarFiles, sonnet_scalar_files

"""Exact owned bytes and SHA256 identity of a native scalar dependency."""
struct SonnetScalarSource
    path::String
    bytes::Vector{UInt8}
    sha256::String
end

"""A native CSV `table1` or `table2`, with finite increasing axes and values.
`table1` uses `row_keys` and a one-column value matrix. Keys and values retain
the expression's native units; no STF XML table interpretation is implied."""
struct SonnetScalarTable
    kind::Symbol
    row_keys::Vector{Float64}
    column_keys::Vector{Float64}
    values::Matrix{Float64}
    source::SonnetScalarSource
end

"""Owned native project and external scalar CSV snapshots. `source` is the
exact parent SON source; `project` is the separately owned effective project.
`configuration_sha256` identifies effective project, tables and domain policy.
Mutating retained data invalidates the snapshot and rejects at reuse."""
struct SonnetScalarFiles
    project::SonnetProject
    source::SonnetScalarSource
    tables::Dict{String,SonnetScalarTable}
    root::String
    outside::Symbol
    payload_bytes::Int
    configuration_sha256::String
    conversion::String
    conversion_sources::Vector{SonnetScalarSource}
end
SonnetScalarFiles(p,s,t,r,o,b,h)=SonnetScalarFiles(p,s,t,r,o,b,h,"identity",SonnetScalarSource[])

struct SonnetScalarVariables <: AbstractDict{String,Float64}
    overrides::Dict{String,Float64}
    files::SonnetScalarFiles
end
Base.length(v::SonnetScalarVariables)=length(v.overrides)
Base.iterate(v::SonnetScalarVariables,state...)=iterate(v.overrides,state...)
Base.getindex(v::SonnetScalarVariables,key)=getindex(v.overrides,key)
Base.haskey(v::SonnetScalarVariables,key)=haskey(v.overrides,key)
Base.keys(v::SonnetScalarVariables)=keys(v.overrides)
Base.values(v::SonnetScalarVariables)=values(v.overrides)

function _sonnet_scalar_project_payload(p::SonnetProject)
    n=_checked_payload_sum("native scalar project snapshot",1024,ncodeunits(p.source))
    for r in p.records,t in r.tokens
        n=_checked_payload_sum("native scalar project snapshot",n,64,ncodeunits(t))
    end
    for q in p.polygons
        n=_checked_payload_sum("native scalar project snapshot",n,256,
            _checked_array_payload_bytes(Float64,size(q.vertices)...),ncodeunits(q.target),ncodeunits(q.technology))
        for flag in q.flags;n=_checked_payload_sum("native scalar project snapshot",n,64,ncodeunits(flag));end
    end
    for pairs in (p.variables,p.units),(key,value) in pairs
        n=_checked_payload_sum("native scalar project snapshot",n,128,ncodeunits(key),ncodeunits(value))
    end
    for rows in (p.layers,p.metals),row in rows,t in row
        n=_checked_payload_sum("native scalar project snapshot",n,64,ncodeunits(t))
    end
    for rows in (p.box,p.top,p.bottom),t in rows
        n=_checked_payload_sum("native scalar project snapshot",n,64,ncodeunits(t))
    end
    for port in p.ports,t in port.values
        n=_checked_payload_sum("native scalar project snapshot",n,64,ncodeunits(t))
    end
    for component in p.components,r in component,t in r.tokens
        n=_checked_payload_sum("native scalar effective records",n,64,ncodeunits(t))
    end
    for r in p.sweeps,t in r.tokens
        n=_checked_payload_sum("native scalar effective records",n,64,ncodeunits(t))
    end
    for port in p.ports,r in port.records,t in r.tokens
        n=_checked_payload_sum("native scalar effective records",n,64,ncodeunits(t))
    end
    return n
end

function _sonnet_scalar_identity(p,source,tables,root,outside,
        conversion="identity",conversion_sources=SonnetScalarSource[])
    io=IOBuffer()
    print(io,repr((source.path,source.sha256,root,outside,p.source,p.length_scale,p.frequency_scale,conversion)))
    for s in conversion_sources;print(io,repr((s.path,s.sha256)));end
    for pairs in (p.units,p.variables)
        for key in sort!(collect(keys(pairs)))
            print(io,repr((key,pairs[key])))
        end
    end
    print(io,repr((p.box,p.layers,p.metals,p.top,p.bottom)))
    for q in p.polygons
        print(io,repr((q.kind,q.level,q.material,q.id,q.vertices,q.target,q.technology,q.flags)))
    end
    for port in p.ports
        print(io,repr((port.kind,port.polygon,port.edge,port.number,port.values)))
        for r in port.records;print(io,repr((r.line,r.tokens)));end
    end
    for component in p.components
        print(io,repr([(r.line,r.tokens) for r in component]))
    end
    for r in p.sweeps;print(io,repr((r.line,r.tokens)));end
    for r in p.records
        print(io,repr((r.line,r.tokens)))
    end
    for key in sort!(collect(keys(tables)))
        table=tables[key]
        print(io,repr((key,table.kind,table.source.path,table.source.sha256,
            table.row_keys,table.column_keys,table.values)))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _sonnet_scalar_dependency_sources(files::SonnetScalarFiles)
    sources=SonnetScalarSource[];seen=Set{String}()
    for table in values(files.tables)
        table.source.path in seen && continue
        push!(seen,table.source.path);push!(sources,table.source)
    end
    return sources
end

# File staging deduplicates dependencies by resolved path. Snapshot checks
# must instead visit every retained source: aliases may own distinct byte
# arrays even when they claim the same path and expected digest.
_sonnet_scalar_snapshot_sources(files::SonnetScalarFiles)=Iterators.flatten(
    ((files.source,),files.conversion_sources,(table.source for table in values(files.tables))))

function _sonnet_scalar_files_payload(files::SonnetScalarFiles)
    n=_checked_payload_sum("native scalar snapshot",_sonnet_scalar_project_payload(files.project),
        256,ncodeunits(files.conversion))
    seen=IdDict{Any,Nothing}()
    for source in _sonnet_scalar_snapshot_sources(files)
        haskey(seen,source.bytes) && continue
        seen[source.bytes]=nothing
        n=_checked_payload_sum("native scalar snapshot",n,256,ncodeunits(source.path),length(source.bytes))
    end
    for table in values(files.tables)
        haskey(seen,table.values) || (n=_checked_payload_sum("native scalar snapshot",n,512))
        for array in (table.row_keys,table.column_keys,table.values)
            haskey(seen,array) && continue
            seen[array]=nothing
            n=_checked_payload_sum("native scalar snapshot",n,
                _checked_array_payload_bytes(Float64,size(array)...))
        end
    end
    return n
end

function _sonnet_scalar_numeric_payload(variables)
    variables isa AbstractDict || throw(ArgumentError("native scalar overrides must be a dictionary"))
    n=128
    for (key,value) in variables
        key isa AbstractString || throw(ArgumentError("native scalar override names must be strings"))
        value isa Real || throw(ArgumentError("native scalar overrides must be real"))
        _circuit_stored_real(value,"native scalar override $key")
        n=_checked_payload_sum("native scalar overrides",n,128,ncodeunits(key))
    end
    return n
end
_sonnet_scalar_files(v)=v isa SonnetScalarVariables ? v.files : nothing
_sonnet_scalar_project(p,v)=v isa SonnetScalarVariables ? v.files.project : p
_sonnet_scalar_payload(v)=_checked_payload_sum("native scalar context",
    _sonnet_scalar_numeric_payload(v),v isa SonnetScalarVariables ? _sonnet_scalar_files_payload(v.files) : 0)

function _sonnet_check_scalar_files(files::SonnetScalarFiles)
    files.outside in (:reject,:hold) || throw(ArgumentError("invalid native scalar table domain policy"))
    hashes=IdDict{Vector{UInt8},String}()
    for source in _sonnet_scalar_snapshot_sources(files)
        digest=haskey(hashes,source.bytes) ? hashes[source.bytes] :
            (hashes[source.bytes]=bytes2hex(SHA.sha256(source.bytes)))
        digest==source.sha256 ||
            throw(ArgumentError("native scalar source snapshot was mutated"))
    end
    _sonnet_scalar_identity(files.project,files.source,files.tables,files.root,files.outside,
        files.conversion,files.conversion_sources)==files.configuration_sha256 ||
        throw(ArgumentError("native scalar effective configuration was mutated"))
    return files
end

# Preserve the parsed diagnostic baseline without constructing a second
# project. Whitespace/inline comments may differ; records and their line
# positions must agree. Detached effective edits have a separate identity.
function _sonnet_scalar_parent_records(source::SonnetScalarSource,records)
    isvalid(String,source.bytes) || throw(ArgumentError("native scalar parent must be valid UTF-8"))
    next=1
    for (line,text) in enumerate(eachline(IOBuffer(source.bytes;read=true,write=false)))
        tokens=_sonnet_tokens(text);isempty(tokens) && continue
        next<=length(records) && records[next].line==line && records[next].tokens==tokens ||
            throw(ArgumentError("native scalar parsed parent records do not match the captured SON source"))
        next+=1
    end
    next==length(records)+1 ||
        throw(ArgumentError("native scalar parsed parent records do not match the captured SON source"))
    return source
end

function _sonnet_scalar_requests(p::SonnetProject,extra=String[])
    requests=Dict{String,Symbol}()
    function visit(ex)
        ex isa Expr || return
        if ex.head==:call && ex.args[1] in (:table1,:table2)
            kind=ex.args[1]
            length(ex.args)==(kind==:table1 ? 3 : 4) || throw(ArgumentError("native $kind has invalid argument count"))
            ex.args[2] isa String || throw(ArgumentError("native table filename must be a literal quoted string"))
            name=ex.args[2]
            isempty(name) && throw(ArgumentError("native table filename must not be empty"))
            haskey(requests,name) && requests[name]!=kind && throw(ArgumentError("one native CSV cannot be both table1 and table2"))
            requests[name]=kind
        end
        foreach(visit,ex.args)
    end
    expressions=String[values(p.variables)...]
    append!(expressions,extra)
    append!(expressions,p.box);append!(expressions,p.top);append!(expressions,p.bottom)
    for rows in (p.layers,p.metals),row in rows;append!(expressions,row);end
    for port in p.ports;append!(expressions,port.values);end
    for component in p.components,r in component
        length(r.tokens)>=4 && r.tokens[1]=="TYPE" && r.tokens[2]=="IDEAL" && push!(expressions,r.tokens[4])
    end
    for text in expressions
        occursin(r"\btable[12]\s*\(",text) || continue
        visit(_sonnet_parse_scalar(text))
    end
    return requests
end

function _sonnet_scalar_expression_payload(p,extra)
    bytes=0
    function reserve(text)
        occursin(r"\btable[12]\s*\(",text) || return
        bytes=_checked_payload_sum("native scalar expression scratch",bytes,ncodeunits(text))
    end
    foreach(reserve,values(p.variables));foreach(reserve,extra)
    for row in (p.box,p.top,p.bottom);foreach(reserve,row);end
    for rows in (p.layers,p.metals),row in rows;foreach(reserve,row);end
    for port in p.ports;foreach(reserve,port.values);end
    for component in p.components,r in component
        length(r.tokens)>=4 && r.tokens[1]=="TYPE" && r.tokens[2]=="IDEAL" && reserve(r.tokens[4])
    end
    # Bounded expression ASTs, literal/token strings and the request map use
    # working storage in addition to the separately retained effective data.
    return _checked_array_payload_bytes(UInt8,32,bytes)
end

function _sonnet_has_scalar_tables(p::SonnetProject)
    has(text)=occursin(r"\btable[12]\s*\(",text)
    any(has,values(p.variables)) && return true
    for row in (p.box,p.top,p.bottom)
        any(has,row) && return true
    end
    for rows in (p.layers,p.metals),row in rows
        any(has,row) && return true
    end
    for port in p.ports
        any(has,port.values) && return true
    end
    for component in p.components,r in component
        length(r.tokens)>=4 && r.tokens[1]=="TYPE" && r.tokens[2]=="IDEAL" && has(r.tokens[4]) && return true
    end
    return false
end

function _sonnet_scalar_path(root,name)
    any(c->c in ('\0','\n','\r'),name) && throw(ArgumentError("invalid native scalar dependency path"))
    native=replace(name,'\\'=> '/')
    candidate=isabspath(native) ? native : joinpath(root,native)
    ispath(candidate) || throw(ArgumentError("missing native scalar dependency"))
    path=realpath(candidate)
    isfile(path) || throw(ArgumentError("native scalar dependency must be a regular file"))
    normalized=Sys.iswindows() ? lowercase(path) : path
    boundary=Sys.iswindows() ? lowercase(root) : root
    (normalized==boundary || startswith(normalized,boundary*(Sys.iswindows() ? "\\" : "/"))) ||
        throw(ArgumentError("native scalar dependency escapes the allowed root"))
    return path
end

function _sonnet_scalar_read(path,expected)
    open(path,"r") do io
        filesize(io)==expected || throw(ArgumentError("native scalar source size changed before read"))
        bytes=read(io,expected)
        length(bytes)==expected && eof(io) && filesize(io)==expected ||
            throw(ArgumentError("native scalar source changed during bounded read"))
        return SonnetScalarSource(path,bytes,bytes2hex(SHA.sha256(bytes)))
    end
end

function _sonnet_scalar_number(text)
    occursin(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$",text) ||
        throw(ArgumentError("native scalar table requires finite decimal numbers"))
    value=tryparse(Float64,text)
    value!==nothing && isfinite(value) || throw(ArgumentError("native scalar table number is unrepresentable"))
    mantissa=first(split(lowercase(text),'e';limit=2))
    iszero(value) && any(c->'1'<=c<='9',mantissa) &&
        throw(ArgumentError("native scalar literal underflows stored precision"))
    return value
end

# Native ^ chains associate left, whereas Julia's parser associates right.
# Keep explicit parentheses and the native low-precedence unary sign: a signed
# exponent consumes its following power expression (2^-3^2 = 2^(-(3^2))).
mutable struct _SonnetScalarParser
    chars::Vector{Char}
    position::Int
end

function _sonnet_scalar_space!(p)
    while p.position<=length(p.chars) && isspace(p.chars[p.position]);p.position+=1;end
    return nothing
end

function _sonnet_scalar_peek(p)
    _sonnet_scalar_space!(p)
    return p.position>length(p.chars) ? '\0' : p.chars[p.position]
end

function _sonnet_scalar_atom!(p,depth)
    depth<=128 || throw(ArgumentError("native scalar expression nesting budget exceeded"))
    c=_sonnet_scalar_peek(p);start=p.position
    if c=='('
        p.position+=1
        ex=_sonnet_scalar_sum!(p,depth+1)
        _sonnet_scalar_peek(p)==')' || throw(ArgumentError("native scalar requires a closing parenthesis"))
        p.position+=1
        return ex
    elseif c=='"'
        p.position+=1;out=IOBuffer()
        while p.position<=length(p.chars)
            c=p.chars[p.position];p.position+=1
            c=='"' && return String(take!(out))
            if c=='\\' && p.position<=length(p.chars) && p.chars[p.position] in ('"','\\')
                c=p.chars[p.position];p.position+=1
            end
            c in ('\0','\n','\r') && throw(ArgumentError("invalid native scalar string"))
            print(out,c)
        end
        throw(ArgumentError("unterminated native scalar string"))
    elseif isdigit(c) || c=='.'
        while p.position<=length(p.chars) && (isdigit(p.chars[p.position]) || p.chars[p.position]=='.');p.position+=1;end
        if p.position<=length(p.chars) && p.chars[p.position] in ('e','E')
            p.position+=1
            p.position<=length(p.chars) && p.chars[p.position] in ('+','-') && (p.position+=1)
            while p.position<=length(p.chars) && isdigit(p.chars[p.position]);p.position+=1;end
        end
        return _sonnet_scalar_number(String(p.chars[start:p.position-1]))
    elseif isletter(c) || c=='_'
        p.position+=1
        while p.position<=length(p.chars) && (isletter(p.chars[p.position]) || isdigit(p.chars[p.position]) || p.chars[p.position]=='_');p.position+=1;end
        name=Symbol(String(p.chars[start:p.position-1]))
        _sonnet_scalar_peek(p)=='(' || return name
        p.position+=1;ex=Expr(:call,name)
        if _sonnet_scalar_peek(p)!=')'
            while true
                push!(ex.args,_sonnet_scalar_sum!(p,depth+1))
                _sonnet_scalar_peek(p)==',' || break
                p.position+=1
            end
        end
        _sonnet_scalar_peek(p)==')' || throw(ArgumentError("native scalar requires a closing function parenthesis"))
        p.position+=1
        return ex
    end
    throw(ArgumentError("unsupported Sonnet expression token"))
end

function _sonnet_scalar_power!(p,depth)
    value=_sonnet_scalar_atom!(p,depth)
    while _sonnet_scalar_peek(p)=='^'
        p.position+=1
        _sonnet_scalar_peek(p)=='+' && throw(ArgumentError("native positive exponent sign is unsupported"))
        rhs=_sonnet_scalar_peek(p) in ('+','-') ? _sonnet_scalar_unary!(p,depth+1) : _sonnet_scalar_atom!(p,depth)
        value=Expr(:call,:^,value,rhs)
    end
    return value
end

function _sonnet_scalar_unary!(p,depth)
    depth<=128 || throw(ArgumentError("native scalar expression depth budget exceeded"))
    c=_sonnet_scalar_peek(p)
    if c in ('+','-')
        p.position+=1
        return Expr(:call,Symbol(c),_sonnet_scalar_unary!(p,depth+1))
    end
    return _sonnet_scalar_power!(p,depth)
end

function _sonnet_scalar_fold(op,a,b)
    if op in (:+,:*) && a isa Expr && a.head==:call && first(a.args)==op
        push!(a.args,b)
        return a
    end
    return Expr(:call,op,a,b)
end

function _sonnet_scalar_product!(p,depth)
    value=_sonnet_scalar_unary!(p,depth)
    while _sonnet_scalar_peek(p) in ('*','/')
        op=Symbol(p.chars[p.position]);p.position+=1
        value=_sonnet_scalar_fold(op,value,_sonnet_scalar_unary!(p,depth))
    end
    return value
end

function _sonnet_scalar_sum!(p,depth)
    value=_sonnet_scalar_product!(p,depth)
    while _sonnet_scalar_peek(p) in ('+','-')
        op=Symbol(p.chars[p.position]);p.position+=1
        value=_sonnet_scalar_fold(op,value,_sonnet_scalar_product!(p,depth))
    end
    return value
end

# Check decimal literals before parsing can silently round nonzero input to
# zero. Quoted CSV filenames are skipped; no host-language code is parsed.
function _sonnet_parse_scalar(text::AbstractString)
    ncodeunits(text)<=16384 || throw(ArgumentError("native scalar expression byte budget exceeded"))
    '\0' in text && throw(ArgumentError("invalid native scalar NUL character"))
    chars=collect(text);i=1;quoted=false;nesting=0
    while i<=length(chars)
        c=chars[i]
        if quoted
            c=='\\' && i<length(chars) && (i+=2;continue)
            c=='"' && (quoted=false)
            i+=1;continue
        elseif c=='"'
            quoted=true;i+=1;continue
        end
        if c in ('(','[','{')
            nesting+=1
            nesting<=128 || throw(ArgumentError("native scalar expression nesting budget exceeded"))
        elseif c in (')',']','}')
            nesting-=1
        end
        start=(isdigit(c) || (c=='.' && i<length(chars) && isdigit(chars[i+1]))) &&
            (i==1 || !(isletter(chars[i-1]) || isdigit(chars[i-1]) || chars[i-1]=='_'))
        if start
            j=i
            while j<=length(chars) && (isdigit(chars[j]) || chars[j]=='.');j+=1;end
            if j<=length(chars) && chars[j] in ('e','E')
                j+=1
                j<=length(chars) && chars[j] in ('+','-') && (j+=1)
                while j<=length(chars) && isdigit(chars[j]);j+=1;end
            end
            token=String(chars[i:j-1])
            occursin(r"^[0-9.]+(?:[eE][+-]?[0-9]+)?$",token) && _sonnet_scalar_number(token)
            i=j
        else
            i+=1
        end
    end
    parser=_SonnetScalarParser(chars,1)
    ex=_sonnet_scalar_sum!(parser,0)
    _sonnet_scalar_peek(parser)=='\0' || throw(ArgumentError("unsupported Sonnet expression suffix"))
    pending=Tuple{Any,Int}[(ex,0)]
    while !isempty(pending)
        node,depth=pop!(pending)
        depth<=128 || throw(ArgumentError("native scalar expression depth budget exceeded"))
        node isa Expr || continue
        for child in node.args;push!(pending,(child,depth+1));end
    end
    return ex
end

function _sonnet_scalar_csv(source,kind;max_nodes,max_line_bytes)
    rows=Vector{Vector{Float64}}();header=Float64[];nodes=0
    for raw in eachline(IOBuffer(source.bytes))
        ncodeunits(raw)<=max_line_bytes || throw(ArgumentError("native scalar CSV line byte budget exceeded"))
        line=strip(first(split(raw,'!';limit=2)))
        isempty(line) && continue
        fields=strip.(split(line,',';keepempty=true))
        if kind==:table2 && isempty(header)
            isempty(first(fields)) && popfirst!(fields)
            isempty(fields) && throw(ArgumentError("native table2 requires column keys"))
            header=_sonnet_scalar_number.(fields)
            all(diff(header).>0) || throw(ArgumentError("native table2 columns must be strictly increasing"))
            continue
        end
        expected=kind==:table1 ? 2 : length(header)+1
        length(fields)==expected || throw(ArgumentError("native $kind has inconsistent CSV row width"))
        nodes=_checked_payload_sum("native scalar nodes",nodes,kind==:table1 ? 1 : length(header))
        nodes<=max_nodes || throw(ArgumentError("native scalar table node budget exceeded"))
        push!(rows,_sonnet_scalar_number.(fields))
    end
    isempty(rows) && throw(ArgumentError("native scalar table contains no data rows"))
    keys=[row[1] for row in rows]
    all(diff(keys).>0) || throw(ArgumentError("native scalar row keys must be strictly increasing"))
    values=Matrix{Float64}(undef,length(rows),kind==:table1 ? 1 : length(header))
    for i in eachindex(rows);values[i,:]=rows[i][2:end];end
    return SonnetScalarTable(kind,keys,header,values,source),nodes
end

"""Own the effective native project, exact parent SON and all referenced
`table1`/`table2` CSV dependencies before evaluation. Filenames are literal
quoted strings and resolve inside `root` (default parent SON directory),
including after symlink resolution. Source bytes, files, numeric nodes, line
bytes and estimated retained storage are bounded before use. Axes must be
strictly increasing finite decimal values; duplicate/unsorted axes reject.

`table1` is linear and `table2` bilinear, as verified with independent actual
native material controls. CSV `!` comments and the optional empty leading
table2 header slot are supported. `outside=:reject` is the default documented
domain contract; explicit `:hold` uses nearest endpoints and retains this
choice in the configuration identity. The installed native engine warns and
holds endpoints. This API does not implement STF XML table interpolation.
The original SON bytes and separately effective project identity both remain
available, including when the caller intentionally edits a parsed project."""
function sonnet_scalar_files(p::SonnetProject;root=dirname(p.source),outside::Symbol=:reject,expressions=String[],
        max_files::Integer=64,max_bytes::Integer=8*1024^2,max_nodes::Integer=100000,
        max_line_bytes::Integer=16384,max_storage::Integer=64*1024^2,
        _parent_source=nothing,_conversion::String="identity",_conversion_sources=SonnetScalarSource[])
    all(x->!(x isa Bool) && 0<x<=typemax(Int),(max_files,max_bytes,max_nodes,max_line_bytes,max_storage)) ||
        throw(ArgumentError("native scalar resource limits must be positive representable integers"))
    outside in (:reject,:hold) || throw(ArgumentError("native scalar outside policy must be :reject or :hold"))
    expressions isa AbstractVector{<:AbstractString} || throw(ArgumentError("native scalar extra expressions must be a string vector"))
    projectbytes=_sonnet_scalar_project_payload(p)
    scratch=_sonnet_scalar_expression_payload(p,expressions)
    _enforce_payload_limit(_checked_payload_sum("native scalar expression intake",projectbytes,scratch),max_storage,
        "native scalar project snapshot","max_storage")
    requests=_sonnet_scalar_requests(p,expressions)
    isdir(root) || throw(ArgumentError("native scalar root must be a directory"))
    allowed=realpath(root)
    _conversion_sources isa Vector{SonnetScalarSource} || throw(ArgumentError("invalid native scalar conversion sources"))
    (_parent_source===nothing && _conversion=="identity" && isempty(_conversion_sources)) ||
        (_parent_source isa SonnetScalarSource && _conversion=="static-stf" && length(_conversion_sources)==1) ||
        throw(ArgumentError("invalid native scalar owned-parent conversion contract"))
    if _parent_source===nothing
        isfile(p.source) || throw(ArgumentError("native scalar parent must be a regular SON file"))
        parent=realpath(p.source);parentsize=filesize(parent)
    else
        parent=abspath(_parent_source.path)
        abspath(p.source)==parent || throw(ArgumentError("native scalar owned parent path disagrees with project"))
        parentsize=length(_parent_source.bytes)
    end
    paths=Dict{String,String}();kinds=Dict{String,Symbol}();sizes=Dict{String,Int}()
    total=_checked_payload_sum("native scalar aggregate source bytes",parentsize,
        (length(s.bytes) for s in _conversion_sources)...);sizes[parent]=parentsize
    total<=max_bytes || throw(ArgumentError("native scalar aggregate source byte budget exceeded"))
    parentbase=_checked_payload_sum("native scalar parent intake",projectbytes,scratch,
        _checked_array_payload_bytes(UInt8,8,total))
    _enforce_payload_limit(parentbase,max_storage,"native scalar parent intake","max_storage")
    source=_parent_source===nothing ? _sonnet_scalar_read(parent,parentsize) : deepcopy(_parent_source)
    bytes2hex(SHA.sha256(source.bytes))==source.sha256 || throw(ArgumentError("native scalar owned parent source was mutated"))
    _sonnet_scalar_parent_records(source,p.records)
    conversions=deepcopy(_conversion_sources)
    for s in conversions
        bytes2hex(SHA.sha256(s.bytes))==s.sha256 || throw(ArgumentError("native scalar conversion source was mutated"))
    end
    for (name,kind) in requests
        path=_sonnet_scalar_path(allowed,name)
        haskey(kinds,path) && kinds[path]!=kind && throw(ArgumentError("one CSV dependency has conflicting table dimensions"))
        kinds[path]=kind;paths[name]=path
        haskey(sizes,path) && continue
        size=filesize(path)
        total=_checked_payload_sum("native scalar aggregate source bytes",total,size)
        total<=max_bytes || throw(ArgumentError("native scalar aggregate source byte budget exceeded"))
        sizes[path]=size
    end
    length(kinds)<=max_files || throw(ArgumentError("native scalar dependency count budget exceeded"))
    # Eight byte planes cover owned bytes, UTF-8 text, token characters and
    # bounded parsing/identity scratch (including captured STF provenance).
    # scratch. Node scratch/retained vectors are reserved as they are counted.
    base=_checked_payload_sum("native scalar intake",projectbytes,scratch,
        _checked_array_payload_bytes(UInt8,8,total),1024*length(kinds))
    _enforce_payload_limit(base,max_storage,"native scalar intake","max_storage")
    tables=Dict{String,SonnetScalarTable}();unique=Dict{String,SonnetScalarTable}();nodes=0
    for (path,kind) in kinds
        raw=_sonnet_scalar_read(path,sizes[path])
        # Reserve worst-case numeric scratch from source bytes before splitting
        # fields; max_nodes is an upper bound, not a ceiling-sized allocation.
        possible=min(max_nodes,cld(length(raw.bytes),2))
        scratch=_checked_array_payload_bytes(UInt8,128,possible)
        _enforce_payload_limit(_checked_payload_sum("native scalar parse",base,scratch,
            _checked_array_payload_bytes(UInt8,32,nodes)),
            max_storage,"native scalar parse","max_storage")
        table,count=_sonnet_scalar_csv(raw,kind;max_nodes=max_nodes-nodes,max_line_bytes)
        nodes=_checked_payload_sum("native scalar aggregate nodes",nodes,count)
        unique[path]=table
    end
    for (name,path) in paths;tables[name]=unique[path];end
    owned=deepcopy(p)
    identity=_sonnet_scalar_identity(owned,source,tables,allowed,outside,_conversion,conversions)
    draft=SonnetScalarFiles(owned,source,tables,allowed,outside,0,identity,_conversion,conversions)
    payload=_sonnet_scalar_files_payload(draft)
    _enforce_payload_limit(payload,max_storage,"native scalar retained snapshot","max_storage")
    return SonnetScalarFiles(owned,source,tables,allowed,outside,payload,identity,_conversion,conversions)
end
function sonnet_scalar_files(path::AbstractString;kw...)
    p=read_sonnet_project(path)
    p isa SonnetProject || throw(ArgumentError("native scalar CSV snapshots require a geometry project"))
    return sonnet_scalar_files(p;kw...)
end

function _sonnet_scalar_variables(p,variables;scalar_files=nothing,
        max_bytes::Integer=64*1024^2,max_bytes_source::Integer=8*1024^2,expressions=String[],kw...)
    numeric=_sonnet_scalar_numeric_payload(variables)
    _enforce_payload_limit(numeric,max_bytes,"native scalar overrides","max_bytes")
    existing=_sonnet_scalar_files(variables)
    scalar_files!==nothing && existing!==nothing && scalar_files!==existing &&
        throw(ArgumentError("conflicting native scalar snapshots"))
    files=scalar_files===nothing ? existing : scalar_files
    if files===nothing && (_sonnet_has_scalar_tables(p) || !isempty(expressions))
        files=sonnet_scalar_files(p;max_storage=max_bytes-numeric,max_bytes=max_bytes_source,expressions,kw...)
        # Freshly staged files are already independently owned.
    elseif files!==nothing
        files isa SonnetScalarFiles || throw(ArgumentError("invalid native scalar snapshot"))
        _enforce_payload_limit(_checked_payload_sum("native scalar context",numeric,
            _sonnet_scalar_files_payload(files)),max_bytes,"native scalar context","max_bytes")
        _sonnet_check_scalar_files(files)
        files=deepcopy(files)
    end
    overrides=Dict{String,Float64}(String(k)=>_circuit_stored_real(v,"native scalar override $k") for (k,v) in variables)
    return files===nothing ? overrides : SonnetScalarVariables(overrides,files)
end

function _sonnet_scalar_context_copy(v::SonnetScalarVariables;
        max_bytes::Integer=64*1024^2)
    return _sonnet_scalar_variables(v.files.project,v;max_bytes)
end
Base.copy(v::SonnetScalarVariables)=_sonnet_scalar_context_copy(v)

function _sonnet_scalar_axis(axis,key,outside)
    isfinite(key) || throw(ArgumentError("native scalar table keys must be finite"))
    if key<first(axis) || key>last(axis)
        outside==:hold || throw(ArgumentError("native scalar table key is outside its source domain"))
        key=clamp(key,first(axis),last(axis))
    end
    length(axis)==1 && return (1,1,0.)
    i=clamp(searchsortedlast(axis,key),1,length(axis)-1)
    key==axis[i] && return (i,i+1,0.)
    key==axis[i+1] && return (i,i+1,1.)
    # Subtract stored coordinates before division: normalizing large, close
    # keys independently can round away a material fraction of their gap.
    # Only an interval crossing opposite extreme signs can overflow; halving
    # those coordinates preserves its finite ratio without normalizing away
    # the close-key differences that are exact by Sterbenz's lemma.
    width=axis[i+1]-axis[i]
    fraction=isfinite(width) ? (key-axis[i])/width :
        (key/2-axis[i]/2)/(axis[i+1]/2-axis[i]/2)
    return (i,i+1,fraction)
end
function _sonnet_scalar_lerp(x,y,a)
    a==0 && return x
    a==1 && return y
    x==y && return x
    # Same-sign differences are finite and retain close value gaps. Across
    # opposite signs the convex weighted sum avoids an overflowing y-x.
    value=signbit(x)==signbit(y) ? x+a*(y-x) : (1-a)*x+a*y
    return clamp(value,min(x,y),max(x,y))
end
function _sonnet_scalar_table_value(variables,kind,name,keys)
    variables isa SonnetScalarVariables || throw(ArgumentError("native scalar CSV dependencies must be staged before evaluation"))
    files=variables.files
    table=get(files.tables,name,nothing)
    table!==nothing && table.kind==kind || throw(ArgumentError("native scalar CSV was not staged with the requested dimensions"))
    i,j,a=_sonnet_scalar_axis(table.row_keys,keys[1],files.outside)
    if kind==:table1
        return _sonnet_scalar_lerp(table.values[i,1],table.values[j,1],a)
    end
    k,l,b=_sonnet_scalar_axis(table.column_keys,keys[2],files.outside)
    return _sonnet_scalar_lerp(_sonnet_scalar_lerp(table.values[i,k],table.values[i,l],b),
        _sonnet_scalar_lerp(table.values[j,k],table.values[j,l],b),a)
end

function sonnet_variable_value(files::SonnetScalarFiles,text::AbstractString;
        variables=Dict{String,Float64}(),freq::Real=1e9,max_bytes::Integer=64*1024^2)
    isfinite(freq) && freq>=0 || throw(ArgumentError("native scalar frequency must be finite and nonnegative"))
    freq=_circuit_stored_real(freq,"native scalar frequency")
    context=_sonnet_scalar_variables(files.project,variables;scalar_files=files,max_bytes)
    return sonnet_variable_value(context.files.project,text;variables=context,freq)
end
