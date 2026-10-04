# Bounded linear SPICE library ingestion and direct MNA primitives.
# Source grammar/equations: ngspice47 manual sections2.1/2.6/2.8/2.11/4.2.
# No arbitrary code evaluation, nonlinear/behavioural analysis or load dropping.
export PlanarSpiceLibrary, PlanarSpiceModel, planar_read_spice, planar_spice_model
export circuit_add_spice!, planar_spice_sparams

struct _SpiceCard
    path::String
    line::Int
    text::String
    tokens::Vector{String}
end
struct _SpiceDefinition
    name::String
    pins::Vector{String}
    defaults::Vector{Pair{String,String}}
    cards::Vector{_SpiceCard}
end
"""Owned bounded linear library with source records, definitions,
include edges and SHA256 provenance. Read with [`planar_read_spice`](@ref)
or the explicit native-export subset [`planar_read_spectre`](@ref).
`dialect=:spice` uses case-insensitive names and SPICE suffixes;
`:spectre` preserves names and case-sensitive SI factors.
Unsupported nonlinear, behavioural and analysis cards reject explicitly."""
struct PlanarSpiceLibrary
    source::String
    root::String
    definitions::Dict{String,_SpiceDefinition}
    parameters::Vector{Pair{String,String}}
    provenance::Dict{String,String}
    payload::Int
    records::Vector{_SpiceCard}
    includes::Vector{Tuple{String,Int,String,String}}
    dialect::Symbol
end
PlanarSpiceLibrary(s,r,d,p,h,b,c,i)=PlanarSpiceLibrary(s,r,d,p,h,b,c,i,:spice)
_spice_library_identifier(lib,s)=lib.dialect===:spectre ?
    (occursin(r"^[a-zA-Z0-9_.+\-]+$",s) ? String(s) : throw(ArgumentError("unsupported Spectre identifier $s"))) : _spice_identifier(s)
_spice_library_node(lib,s)=lib.dialect===:spectre ? _spice_library_identifier(lib,s) : _spice_node_name(s)
mutable struct _SpiceBudget
    used::Int
    limit::Int
    max_records::Int
    records::Int
    max_depth::Int
end
function _spice_reserve!(b,n)
    n>=0 && n<=b.limit-b.used || throw(ArgumentError("SPICE payload exceeds max_bytes"))
    b.used+=n
end
_spice_failure(c,msg)=throw(ArgumentError("$(c.path):$(c.line): $msg"))
function _spice_identifier(s)
    occursin(r"^[a-zA-Z0-9_.+\-]+$",s) || throw(ArgumentError("unsupported SPICE identifier $s"))
    lowercase(String(s))
end
_spice_node_name(s)=begin
    key=_spice_identifier(s)
    key=="gnd" ? "0" : key
end
function _spice_limit(name,value)
    value isa Integer && 0<value<=typemax(Int) || throw(ArgumentError("$name must be a positive Int-sized integer"))
    Int(value)
end
function _spice_literal_nonzero(text)
    for character in text
        character in ('e','E') && break
        '1'<=character<='9' && return true
    end
    false
end
function _spice_number(text)
    m=match(r"^([+\-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+\-]?[0-9]+)?)([a-zA-Z]*)$",text)
    m===nothing && throw(ArgumentError("invalid SPICE number $text"))
    suffix=lowercase(m[2])
    scales=("meg"=>1e6,"mil"=>25.4e-6,"t"=>1e12,"g"=>1e9,"k"=>1e3,
        "m"=>1e-3,"u"=>1e-6,"n"=>1e-9,"p"=>1e-12,"f"=>1e-15,"a"=>1e-18)
    scale=1.
    for (name,value) in scales
        startswith(suffix,name) && (scale=value;break)
    end
    out=parse(Float64,m[1])*scale
    isfinite(out) || throw(ArgumentError("SPICE number must be finite"))
    iszero(out) && _spice_literal_nonzero(m[1]) && throw(ArgumentError("nonzero SPICE number underflows Float64"))
    out
end

# Lexical scanner keeps braces and quotes together, splits assignment '=',
# and accepts full-line * and inline ;/$ comments outside expressions.
function _spice_tokens(text)
    result=String[];buf=IOBuffer();depth=0;quotechar='\0'
    function emit()
        position(buf)>0 && push!(result,String(take!(buf)))
    end
    previous=' '
    for c in text
        if quotechar!='\0'
            write(buf,c);c==quotechar && (quotechar='\0')
        elseif depth>0
            write(buf,c)
            c=='{' && (depth+=1);c=='}' && (depth-=1)
        elseif c in ('\'', '"')
            write(buf,c);quotechar=c
        elseif c=='{'
            write(buf,c);depth=1
        elseif c==';' || c=='$' && isspace(previous)
            break
        elseif c=='='
            emit();push!(result,"=")
        elseif isspace(c)
            emit()
        else
            write(buf,c)
        end
        previous=c
    end
    depth==0 && quotechar=='\0' || throw(ArgumentError("unclosed SPICE brace/quote"))
    emit();length(result)<=128 || throw(ArgumentError("too many SPICE fields"))
    result
end
function _spice_assignments(fields)
    result=Pair{String,String}[]
    length(fields)%3==0 || throw(ArgumentError("SPICE parameters require name=value assignments"))
    for k in 1:3:length(fields)
        fields[k+1]=="=" || throw(ArgumentError("missing SPICE parameter equals"))
        key=_spice_identifier(fields[k]);occursin(r"^[a-z][a-z0-9_]*$",key) || throw(ArgumentError("unsupported parameter identifier"))
        key in ("time","temper","hertz","freq") && throw(ArgumentError("reserved/dynamic SPICE parameter"))
        push!(result,key=>fields[k+2])
    end
    result
end

const _SPICE_CALLS=Dict{Symbol,Function}(Symbol("+")=>(+),Symbol("-")=>(-),Symbol("*")=>(*),Symbol("/")=>(/),Symbol("^")=>(^),
    :sqrt=>sqrt,:abs=>abs,:min=>min,:max=>max,:exp=>exp,:log=>log,:sin=>sin,:cos=>cos,:tan=>tan)
function _spice_expression(text,values,budget=nothing)
    t=strip(text)
    ncodeunits(t)<=4096 || throw(ArgumentError("SPICE expression exceeds byte limit"))
    if length(t)>=2 && ((first(t)=='{' && last(t)=='}') || (first(t)=='\'' && last(t)=='\''))
        t=t[2:end-1]
    end
    # Bound nested parser input before creating the AST.
    depth=0
    for c in t
        c=='(' && (depth+=1)
        c==')' && (depth-=1)
        0<=depth<=32 || throw(ArgumentError("SPICE expression nesting exceeds bounds"))
    end
    depth==0 || throw(ArgumentError("unclosed SPICE expression parentheses"))
    if budget!==nothing
        # Parser/AST/evaluator temporaries are bounded by the input bytes.
        # Reserve peak workspace without pretending that it remains owned.
        workspace=1024+256ncodeunits(t)
        _spice_reserve!(budget,workspace);budget.used-=workspace
    end
    # Rewrite only numeric tokens; r2 and other parameter names stay intact.
    t=replace(lowercase(t),r"(?<![a-z0-9_])(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+\-]?[0-9]+)?[a-z]*"=>s->string(_spice_number(s)))
    ast=Meta.parse(t;raise=true);count=Ref(0)
    function evaluate(a,depth)
        count[]+=1;count[]<=512 && depth<=32 || throw(ArgumentError("SPICE expression exceeds bounds"))
        result=if a isa Real
            Float64(a)
        elseif a isa Symbol
            haskey(values,String(a)) || throw(ArgumentError("undefined SPICE parameter $a"))
            values[String(a)]
        elseif a isa Expr && a.head==:call && a.args[1] isa Symbol && haskey(_SPICE_CALLS,a.args[1])
            _SPICE_CALLS[a.args[1]]((evaluate(v,depth+1) for v in a.args[2:end])...)
        else
            throw(ArgumentError("unsupported SPICE expression syntax"))
        end
        result isa Real && isfinite(result) || throw(ArgumentError("SPICE expression must return finite real value"))
        Float64(result)
    end
    evaluate(ast,1)
end
function _spice_setparams!(values,fields,budget=nothing)
    for (k,v) in fields
        # Self-reference is not permitted by ngspice .param semantics.
        previous=pop!(values,k,nothing)
        try
            values[k]=_spice_expression(v,values,budget)
        catch
            previous===nothing || (values[k]=previous)
            rethrow()
        end
    end
end

"""Read a bounded linear SPICE library. Relative `.include`/`.inc` paths
resolve inside `root`, including realpath checks; cycles reject. Retains
original operative records, include edges and each file's SHA256.

Supported definitions use `.subckt`, `.ends`, numeric `.param`, nested X
calls, scalar R/L/C, named K coupled inductors, linear E/G/F/H, independent DC V/I,
lossless T and O using scoped `.model LTRA`. T uses TD or F/NL (default NL=.25).
LTRA represents the proved RLC/RC/LC/RG subset and retains recognized
transient controls as metadata. Analysis uses incremental frequency-domain
equations. Nonlinear/behavioural devices, other `.model` classes,
`.lib`, `.global`, conditional or analysis directives reject. Library mode
does not silently skip an arbitrary first-line title. Finite nonzero numeric
values that overflow or underflow Float64 reject. `gnd` aliases global
zero, following the documented default ngspice convention."""
function planar_read_spice(path::AbstractString;root=dirname(abspath(path)),max_bytes=16*1024^2,
        max_records=10000,max_depth=32,max_line_bytes=65536)
    max_bytes=_spice_limit("max_bytes",max_bytes);max_records=_spice_limit("max_records",max_records)
    max_depth=_spice_limit("max_depth",max_depth);max_line_bytes=_spice_limit("max_line_bytes",max_line_bytes)
    isdir(root) || throw(ArgumentError("SPICE root must be a directory"))
    directory=realpath(root);source=realpath(path)
    provenance=Dict{String,String}();b=_SpiceBudget(0,max_bytes,max_records,0,max_depth)
    active=Set{String}();cards=_SpiceCard[];includes=Tuple{String,Int,String,String}[]
    _spice_reserve!(b,2048)
    function visit(file,depth)
        depth<=max_depth || throw(ArgumentError("SPICE include depth exceeds limit"))
        candidate=abspath(normpath(file));relative=relpath(candidate,directory)
        first(splitpath(relative))!=".." || throw(ArgumentError("SPICE include escapes root"))
        isfile(candidate) || throw(ArgumentError("missing SPICE include"))
        full=realpath(candidate);rel=relpath(full,directory)
        first(splitpath(rel))!=".." || throw(ArgumentError("SPICE include resolves outside root"))
        key=Sys.iswindows() ? lowercase(full) : full
        key in active && throw(ArgumentError("cyclic SPICE include"))
        _spice_reserve!(b,4filesize(full)+256+2ncodeunits(full))
        provenance[full]=open(io->bytes2hex(SHA.sha256(io)),full)
        push!(active,key)
        pending="";firstline=0
        function logical(text,line)
            isempty(text) && return
            fields=_spice_tokens(text);isempty(fields) && return
            if lowercase(first(fields)) in (".include",".inc")
                length(fields)==2 || throw(ArgumentError("invalid SPICE include"))
                name=fields[2]
                startswith(name,'"') && endswith(name,'"') && (name=name[2:end-1])
                startswith(name,'\'') && endswith(name,'\'') && (name=name[2:end-1])
                isabspath(name) && throw(ArgumentError("SPICE include requires a relative path"))
                _spice_reserve!(b,512+2ncodeunits(text)+2ncodeunits(full)+2ncodeunits(name))
                push!(includes,(full,line,String(text),String(name)))
                visit(joinpath(dirname(full),name),depth+1)
            else
                b.records+=1;b.records<=max_records || throw(ArgumentError("SPICE record count exceeds limit"))
                _spice_reserve!(b,256+128length(fields)+2ncodeunits(text))
                push!(cards,_SpiceCard(full,line,text,fields))
            end
        end
        try
            open(full) do io
                for (line,raw) in enumerate(eachline(io))
                    ncodeunits(raw)<=max_line_bytes || throw(ArgumentError("SPICE line exceeds limit"))
                    s=strip(raw);(isempty(s)||startswith(s,'*')) && continue
                    if startswith(s,'+')
                        isempty(pending) && throw(ArgumentError("orphan SPICE continuation"))
                        ncodeunits(pending)+ncodeunits(s)<=max_line_bytes || throw(ArgumentError("SPICE continued line exceeds limit"))
                        pending*=" "*s[2:end]
                    else
                        logical(pending,firstline);pending=String(s);firstline=line
                    end
                end
                logical(pending,firstline)
            end
        finally
            delete!(active,key)
        end
    end
    visit(source,1)
    definitions=Dict{String,_SpiceDefinition}();parameters=Pair{String,String}[];current=nothing
    for c in cards
        fields=c.tokens;kind=lowercase(fields[1])
        if kind==".subckt"
            current===nothing || _spice_failure(c,"nested definitions are unsupported; nested X instances are supported")
            length(fields)>=3 || _spice_failure(c,"invalid subckt header")
            eq=findfirst(==("="),fields);paramstart=eq===nothing ? length(fields)+1 : eq-1
            p=findfirst(x->lowercase(x)=="params:",fields)
            p!==nothing && (paramstart=p)
            name=_spice_identifier(fields[2]);pins=_spice_node_name.(fields[3:paramstart-1])
            !isempty(pins) && !("0" in pins) && length(unique(pins))==length(pins) || _spice_failure(c,"invalid subckt pins")
            haskey(definitions,name) && _spice_failure(c,"duplicate case-insensitive subckt name")
            defaults=_spice_assignments(fields[paramstart+(p===nothing ? 0 : 1):end])
            length(unique(first.(defaults)))==length(defaults) || _spice_failure(c,"duplicate formal parameter")
            current=_SpiceDefinition(name,pins,defaults,_SpiceCard[]);definitions[name]=current
        elseif kind==".ends"
            current!==nothing && length(fields) in (1,2) || _spice_failure(c,"unexpected ends")
            length(fields)==1 || _spice_identifier(fields[2])==current.name || _spice_failure(c,"mismatched ends name")
            current=nothing
        elseif kind==".param"
            pairs=_spice_assignments(fields[2:end])
            current===nothing ? append!(parameters,pairs) : push!(current.cards,c)
        elseif kind==".model"
            length(fields)>=3 && lowercase(fields[3])=="ltra" || _spice_failure(c,"only linear .model LTRA is supported")
            current===nothing || push!(current.cards,c)
        elseif current!==nothing && uppercase(first(kind)) in ('R','L','C','E','G','F','H','V','I','X','K','T','O')
            push!(current.cards,c)
        else
            _spice_failure(c,"unsupported SPICE card $kind; nonlinear, behavioural, model, analysis and global directives are not ingested")
        end
    end
    current===nothing || throw(ArgumentError("unfinished SPICE subckt"))
    !isempty(definitions) || throw(ArgumentError("SPICE library contains no subckt"))
    PlanarSpiceLibrary(source,directory,definitions,parameters,provenance,b.used,cards,includes)
end

struct _SpiceElement
    name::String
    kind::Char
    nodes::Vector{Int}
    control::String
    value::Float64
    card::_SpiceCard
end
struct _SpiceCoupling
    name::String
    inductors::Vector{Int}
    coefficient::Float64
    card::_SpiceCard
end
include("PlanarSpiceLines.jl")
"""Compiled linear SPICE subcircuit with ordered pin identities, unique
instance/internal-node namespaces and physical global zero preserved.
Source/card/instance provenance remains available in `library`, `elements`
and `couplings`; exact ideal constraints do not require finite admittance.
K records are metadata referencing physical L branches. They add no
unknowns or galvanic connections between isolated winding returns.
`lines` retains ordered T/O physical coefficients, scoped model origins
and transient controls; each line adds two physical current unknowns."""
struct PlanarSpiceModel
    elements::Vector{_SpiceElement}
    nodes::Dict{String,Int}
    pins::Dict{String,Int}
    provenance::Dict{String,String}
    payload::Int
    library::PlanarSpiceLibrary
    subckt::String
    parameters::Dict{String,Float64}
    couplings::Vector{_SpiceCoupling}
    inductive_rows::Dict{Int,Vector{Tuple{Int,Float64}}}
    lines::Dict{Int,_SpiceLineSpec}
end
PlanarSpiceModel(e,n,p,h,b,l,s,v,c,r)=PlanarSpiceModel(e,n,p,h,b,l,s,v,c,r,Dict{Int,_SpiceLineSpec}())
PlanarSpiceModel(e,n,p,h,b,l,s,v)=PlanarSpiceModel(e,n,p,h,b,l,s,v,
    _SpiceCoupling[],Dict{Int,Vector{Tuple{Int,Float64}}}())
include("PlanarSpiceCoupling.jl")
"""Compile a named library definition with numeric formal `parameters`.
Nested X calls use lexical defaults/overrides; local `.param` values do
not leak between instances. `pin_nodes` explicitly aliases compiled pins
for standalone use; circuit attachment uses ordered external pin nodes.
Expansion and parameter contexts share `max_bytes`; hierarchy, node and
element/metadata counts are bounded before circuit mutation. K supports
forward/local hierarchical L names, signed finite coefficients in `[-1,1]`
and equal coupling among multiple named coils. First L terminals are dots.
Each active coupling group has an exactly checked finite-dyadic PSD
correlation matrix, including perfect coupling; pairwise bounds alone
are insufficient. `max_coupled_size=256` and `max_coupling_pairs=100000`
bound grouped work; integer workspace is reserved using a Hadamard bound
on minor bit lengths before allocation. Zero L and zero k decouple.
Compiled magnetic coefficients must be representable in Float64.
T/O preserve their two separate return coordinates. LTRA uses exact finite
DC/small-length transfer equations and bounded travelling-wave equations
otherwise; near-DC series resistance is retained without snapping frequency.
Duplicate model definitions, general RLCG, IC/transient instance fields
and other line-card classes reject with source diagnostics."""
function planar_spice_model(lib::PlanarSpiceLibrary,name::AbstractString;
        parameters=Dict{String,Float64}(),pin_nodes=Dict{String,String}(),
        max_bytes=16*1024^2,max_elements=10000,max_nodes=10000,max_depth=32,
        max_coupled_size=256,max_coupling_pairs=100000)
    max_bytes=_spice_limit("max_bytes",max_bytes);max_elements=_spice_limit("max_elements",max_elements)
    max_nodes=_spice_limit("max_nodes",max_nodes);max_depth=_spice_limit("max_depth",max_depth)
    max_coupled_size=_spice_limit("max_coupled_size",max_coupled_size)
    max_coupling_pairs=_spice_limit("max_coupling_pairs",max_coupling_pairs)
    lib.dialect in (:spice,:spectre) || throw(ArgumentError("unsupported linear library dialect"))
    identifier(s)=_spice_library_identifier(lib,s)
    node_name(s)=_spice_library_node(lib,s)
    b=_SpiceBudget(0,max_bytes,max_elements,0,max_depth);_spice_reserve!(b,lib.payload)
    elements=_SpiceElement[];nodes=Dict{String,Int}();names=Set{String}();pins=Dict{String,Int}();active=String[]
    pending=Tuple{String,Vector{String},Float64,_SpiceCard}[]
    lines=Dict{Int,_SpiceLineSpec}()
    function node(key)
        key=="0" && return 0
        if !haskey(nodes,key)
            length(nodes)<max_nodes || throw(ArgumentError("SPICE expanded node count exceeds limit"))
            _spice_reserve!(b,256+2ncodeunits(key));nodes[key]=length(nodes)+1
        end
        nodes[key]
    end
    top=identifier(name);haskey(lib.definitions,top) || throw(ArgumentError("unknown SPICE subckt"))
    _spice_reserve!(b,512+128length(pin_nodes)+2sum(ncodeunits(k)+ncodeunits(v) for (k,v) in pin_nodes;init=0))
    topdef=lib.definitions[top];connections=Dict{String,String}()
    for (k,v) in pin_nodes
        key=node_name(k);haskey(connections,key) && throw(ArgumentError("duplicate library pin binding"))
        connections[key]=node_name(v)
    end
    all(k->k in topdef.pins,keys(connections)) || throw(ArgumentError("unknown external SPICE pin"))
    for pin in topdef.pins;pins[pin]=node(get(connections,pin,"pin."*pin));end
    _spice_reserve!(b,512+128length(lib.parameters)+2sum(ncodeunits(k) for (k,_) in lib.parameters;init=0))
    globals=Dict{String,Float64}();_spice_setparams!(globals,lib.parameters,b)
    globalmodels=_spice_global_models(lib,globals,b)
    function instance(defname,map,parent,overrides,path,depth,inheritedmodels)
        depth<=max_depth || throw(ArgumentError("SPICE hierarchy depth exceeds limit"))
        defname in active && throw(ArgumentError("recursive SPICE subckt instance"))
        haskey(lib.definitions,defname) || throw(ArgumentError("unknown nested SPICE subckt $defname"))
        def=lib.definitions[defname]
        # Retain a conservative reservation for every expanded lexical scope.
        # Empty subcircuits and large parameter trees are therefore bounded too.
        localparameters=sum(length(c.tokens)÷3 for c in def.cards if lowercase(c.tokens[1])==".param";init=0)
        _spice_reserve!(b,512+128*(length(parent)+length(def.defaults)+localparameters)+
            2sum(ncodeunits(k) for k in keys(parent);init=0)+
            2sum(ncodeunits(k) for (k,_) in def.defaults;init=0))
        values=copy(parent)
        all(k->k in first.(def.defaults),keys(overrides)) || throw(ArgumentError("unknown subckt parameter override"))
        for (k,v) in def.defaults
            values[k]=haskey(overrides,k) ? overrides[k] : _spice_expression(v,values,b)
        end
        models=_spice_line_scope(def.cards,values,inheritedmodels,b)
        push!(active,defname)
        localnames=Set{String}()
        localnode(raw)=begin
        key=node_name(raw);key=="0" ? 0 : get(map,key) do;node(path*"/"*key);end
        end
        try
            for c in def.cards
                t=c.tokens;kind=uppercase(first(t[1]))
                if lowercase(t[1])==".param";_spice_setparams!(values,_spice_assignments(t[2:end]),b);continue;end
                lowercase(t[1])==".model" && continue
                localname=identifier(t[1]);localname in localnames && _spice_failure(c,"duplicate library element name")
                push!(localnames,localname)
                if kind=='X'
                    eq=findfirst(==("="),t);paramstart=eq===nothing ? length(t)+1 : eq-1
                    marker=findfirst(x->lowercase(x)=="params:",t);marker!==nothing && (paramstart=marker)
                    sub=identifier(t[paramstart-1]);haskey(lib.definitions,sub) || _spice_failure(c,"undefined subckt $sub")
                    actual=t[2:paramstart-2];length(actual)==length(lib.definitions[sub].pins) || _spice_failure(c,"subckt pin count mismatch")
                    pairs=_spice_assignments(t[paramstart+(marker===nothing ? 0 : 1):end])
                    length(unique(first.(pairs)))==length(pairs) || _spice_failure(c,"duplicate instance parameter")
                    _spice_reserve!(b,512+128*(length(pairs)+length(actual)))
                    over=Dict(k=>_spice_expression(v,values,b) for (k,v) in pairs)
                    instance(sub,Dict(p=>localnode(a) for (p,a) in zip(lib.definitions[sub].pins,actual)),values,over,path*"/"*localname,depth+1,models)
                    continue
                end
                length(elements)+length(pending)<max_elements || _spice_failure(c,"expanded element/metadata limit")
                fullname=path*"/"*localname;push!(names,fullname)
                if kind=='K'
                    length(t)>=4 || _spice_failure(c,"K requires at least two inductors and a scalar coefficient")
                    value=_spice_expression(t[end],values,b)
                    isfinite(value) && -1<=value<=1 || _spice_failure(c,"K coefficient must be finite in -1:1")
                    _spice_reserve!(b,512+256length(t)+2ncodeunits(fullname))
                    push!(pending,(fullname,[path*"/"*identifier(s) for s in t[2:end-1]],value,c))
                    continue
                end
                control="";nds=Int[];value=0.
                if kind in ('T','O')
                    length(t)>=5 || _spice_failure(c,"transmission line requires four terminals")
                    nds=localnode.(t[2:5])
                    spec=if kind=='T'
                        _spice_t_spec(c,values,b)
                    else
                        length(t)==6 || _spice_failure(c,"O requires four terminals and one LTRA model; IC/transient instance fields are not supported")
                        key=identifier(t[6]);haskey(models,key) || _spice_failure(c,"undefined LTRA model $key")
                        models[key]
                    end
                    _spice_reserve!(b,128);lines[length(elements)+1]=spec
                elseif kind in ('R','L','C')
                    length(t)==4 || _spice_failure(c,"only scalar R/L/C cards are supported")
                    nds=localnode.(t[2:3]);value=_spice_expression(t[4],values,b)
                elseif kind in ('E','G')
                    length(t)==6 || kind=='G' && length(t)==9 && lowercase(t[7])=="m" && t[8]=="=" || _spice_failure(c,"only linear scalar E/G cards are supported")
                    nds=localnode.(t[2:5]);value=_spice_expression(t[6],values,b)
                    length(t)==9 && (value*=_spice_expression(t[9],values,b))
                elseif kind in ('F','H')
                    length(t)==5 || kind=='F' && length(t)==8 && lowercase(t[6])=="m" && t[7]=="=" || _spice_failure(c,"only linear scalar F/H cards are supported")
                    nds=localnode.(t[2:3]);control=path*"/"*identifier(t[4]);value=_spice_expression(t[5],values,b)
                    length(t)==8 && (value*=_spice_expression(t[8],values,b))
                elseif kind in ('V','I')
                    # A homogeneous AC network sets independent DC sources to
                    # zero small-signal excitation. Nonzero AC/transient sources
                    # need an affine response and are rejected, not discarded.
                    length(t) in (4,5) || _spice_failure(c,"only DC independent sources are supported")
                    length(t)==5 && lowercase(t[4])!="dc" && _spice_failure(c,"unsupported independent source")
                    value=_spice_expression(t[end],values,b);nds=localnode.(t[2:3])
                else
                    _spice_failure(c,"unsupported element")
                end
                isfinite(value) || _spice_failure(c,"nonfinite compiled element value")
                _spice_reserve!(b,512+2ncodeunits(fullname));push!(elements,_SpiceElement(fullname,kind,nds,control,value,c))
            end
        finally
            pop!(active)
        end
    end
    _spice_reserve!(b,512+128length(parameters)+2sum(ncodeunits(k) for k in keys(parameters);init=0))
    over=Dict{String,Float64}()
    for (k,v) in parameters
        key=_spice_identifier(k);haskey(over,key) && throw(ArgumentError("duplicate case-insensitive parameter override"))
        v isa Real && isfinite(v) || throw(ArgumentError("parameter overrides must be finite real numbers"))
        over[key]=_circuit_stored_real(v,"SPICE parameter override")
    end
    all(isfinite,values(over)) || throw(ArgumentError("nonfinite parameter override"))
    instance(top,pins,globals,over,"xroot",1,globalmodels)
    _spice_reserve!(b,128length(elements))
    elementlookup=Dict(e.name=>e for e in elements)
    for e in elements
        e.kind in ('F','H') || continue
        haskey(elementlookup,e.control) && elementlookup[e.control].kind=='V' ||
            _spice_failure(e.card,"F/H control must name a local independent voltage source")
    end
    _spice_reserve!(b,128length(lib.provenance))
    couplings,rows=_spice_compile_couplings(pending,elements,b,max_coupled_size,max_coupling_pairs)
    PlanarSpiceModel(elements,nodes,pins,copy(lib.provenance),b.used,lib,top,over,couplings,rows,lines)
end

# Scalar primitives contribute one physical branch current; lines contribute
# two inward currents. Voltage sensors add no independent current branch.
struct _CircuitSpicePrimitive <: _PlanarCircuitElement
    terminals::Vector{Tuple{Int,Int}}
    kind::Char
    value::Float64
    control_terminals::Tuple{Int,Int}
    control_element::Int
    origin::_SpiceElement
    owner::NamedTuple{(:instance,:model),Tuple{String,PlanarSpiceModel}}
    inductance_scale::Float64
    inductive_couplings::Vector{Tuple{Int,Float64}}
end
struct _CircuitSpiceLine <: _PlanarCircuitElement
    terminals::Vector{Tuple{Int,Int}}
    spec::_SpiceLineSpec
    origin::_SpiceElement
    owner::NamedTuple{(:instance,:model),Tuple{String,PlanarSpiceModel}}
end
_circuit_owned_model(e::_CircuitSpiceLine)=e.owner.model
_circuit_stamp_extra!(matrix,row,e::_CircuitSpiceLine,f,branch_rows)=_spice_line_stamp!(matrix,row,e.terminals,e.spec,f)
_CircuitSpicePrimitive(t,k,v,c,i,o,w)=_CircuitSpicePrimitive(t,k,v,c,i,o,w,0.,Tuple{Int,Float64}[])
_circuit_gauge_terminals(e::_CircuitSpicePrimitive)=e.kind in ('E','G') ?
    (e.terminals[1],e.control_terminals) : (e.terminals[1],)
_circuit_owned_model(e::_CircuitSpicePrimitive)=e.owner.model
function _circuit_stamp_extra!(M,row,e::_CircuitSpicePrimitive,f,branch_rows)
    omega=2pi*f
    if e.kind in ('R','L')
        if e.kind=='L' && !iszero(e.inductance_scale)
            _circuit_voltage_stamp!(M,row,e.terminals[1],inv(e.inductance_scale))
            M[row,row]=-1im*omega*e.inductance_scale
            for (target,coefficient) in e.inductive_couplings
                M[row,branch_rows[target]]=-1im*omega*coefficient
            end
        else
            _circuit_voltage_stamp!(M,row,e.terminals[1],1.0)
            M[row,row]=e.kind=='R' ? -e.value : -1im*omega*e.value
        end
    elseif e.kind=='C'
        _circuit_voltage_stamp!(M,row,e.terminals[1],1im*omega*e.value)
        M[row,row]=-1.0
    elseif e.kind=='E'
        _circuit_voltage_stamp!(M,row,e.terminals[1],1.0)
        _circuit_voltage_stamp!(M,row,e.control_terminals,-e.value)
    elseif e.kind=='G'
        M[row,row]=1.0
        _circuit_voltage_stamp!(M,row,e.control_terminals,-e.value)
    elseif e.kind=='F'
        M[row,row]=1.0;M[row,branch_rows[e.control_element]]-=e.value
    elseif e.kind=='H'
        _circuit_voltage_stamp!(M,row,e.terminals[1],1.0)
        M[row,branch_rows[e.control_element]]-=e.value
    elseif e.kind=='V'
        _circuit_voltage_stamp!(M,row,e.terminals[1],1.0)
    elseif e.kind=='I'
        M[row,row]=1.0
    else
        throw(ArgumentError("unsupported compiled SPICE primitive"))
    end
    nothing
end

function _spice_circuit_payload(circuit)
    payload=_checked_payload_sum("SPICE circuit attachment",
        2048,_checked_array_payload_bytes(Int,4,circuit.nnodes),
        _checked_array_payload_bytes(ComplexF64,4,length(circuit.ports)),
        _checked_array_payload_bytes(UInt8,1280,length(circuit.elements)))
    for e in circuit.elements
        if e isa _CircuitNetwork && e.response isa AbstractMatrix
            payload=_checked_payload_sum("SPICE circuit attachment",payload,
                _checked_array_payload_bytes(ComplexF64,size(e.response)...))
        end
    end
    payload
end

"""Attach a compiled linear SPICE subcircuit to ordered external node numbers.
The order is its literal `.subckt` pin order. Internal nodes are appended,
while literal `0`/`gnd` retains global ground. `name` is case insensitive and
must be unique among SPICE instances. Source cards, numeric parameters and
file SHA256 remain accessible through each primitive's shared model.

Attachment preflights owned model, mapping and element storage before any
circuit mutation. Node aliases declared during compilation must agree with
attachment wiring. Scalar R/L/C and E/G/F/H use exact linear MNA equations,
including zero gains, ideal shorts and forward voltage-source controllers.
Independent DC V/I values are retained as provenance; their homogeneous
incremental AC excitation is zero. T/O retain physical currents and independent
return gauges, including when F/H or K refer to later scalar current branches.
This is not a DC operating-point solver."""
function circuit_add_spice!(circuit::PlanarCircuit,pin_nodes::AbstractVector,
        model::PlanarSpiceModel;name="spice$(length(circuit.elements)+1)",
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    instance=_spice_identifier(name)
    any(e->e isa Union{_CircuitSpicePrimitive,_CircuitSpiceLine} && e.owner.instance==instance,circuit.elements) &&
        throw(ArgumentError("duplicate case-insensitive SPICE instance name"))
    definition=model.library.definitions[model.subckt]
    length(pin_nodes)==length(definition.pins) || throw(ArgumentError("SPICE attachment pin count mismatch"))
    all(n->n isa Integer && 0<=n<=circuit.nnodes,pin_nodes) ||
        throw(ArgumentError("SPICE external nodes must be in 0:$(circuit.nnodes)"))
    nodecount=length(model.nodes)
    retained=_circuit_owned_model_payload(circuit)
    any(e->_circuit_owned_model(e)===model,circuit.elements) || (retained=_checked_payload_sum("SPICE attachment",retained,model.payload))
    needed=_checked_payload_sum("SPICE attachment",retained,_spice_circuit_payload(circuit),
        _checked_array_payload_bytes(UInt8,1280,length(model.elements)),
        _checked_array_payload_bytes(UInt8,128,sum(length(v) for v in values(model.inductive_rows);init=0)),
        _checked_array_payload_bytes(Int,6,nodecount+length(pin_nodes)+1))
    _enforce_payload_limit(needed,max_bytes,"SPICE attachment","max_bytes")
    mapping=fill(-1,nodecount+1);mapping[1]=0
    for (pin,external) in zip(definition.pins,pin_nodes)
        logical=model.pins[pin];old=mapping[logical+1]
        old in (-1,external) || throw(ArgumentError("SPICE compiled pin aliases conflict with attachment wiring"))
        mapping[logical+1]=Int(external)
    end
    internal=nodecount-count(n->n>=0,view(mapping,2:length(mapping)))
    circuit.nnodes<=typemax(Int)-internal || throw(ArgumentError("SPICE circuit node count overflows Int"))
    nextnode=circuit.nnodes
    for k in 2:length(mapping)
        mapping[k]<0 && (nextnode+=1;mapping[k]=nextnode)
    end
    offset=length(circuit.elements)
    lookup=Dict(e.name=>k+offset for (k,e) in enumerate(model.elements))
    owner=(instance=instance,model=model)
    added=_PlanarCircuitElement[];sizehint!(added,length(model.elements))
    for (index,e) in enumerate(model.elements)
        if e.kind in ('T','O')
            haskey(model.lines,index) || _spice_failure(e.card,"compiled line metadata missing")
            terminals=[(mapping[e.nodes[1]+1],mapping[e.nodes[2]+1]),(mapping[e.nodes[3]+1],mapping[e.nodes[4]+1])]
            push!(added,_CircuitSpiceLine(terminals,model.lines[index],e,owner));continue
        end
        terminals=[(mapping[e.nodes[1]+1],mapping[e.nodes[2]+1])]
        control=e.kind in ('E','G') ? (mapping[e.nodes[3]+1],mapping[e.nodes[4]+1]) : (0,0)
        controller=e.kind in ('F','H') ? lookup[e.control] : 0
        row=get(model.inductive_rows,index,nothing)
        scale=row===nothing ? 0. : sqrt(e.value)
        links=row===nothing ? Tuple{Int,Float64}[] : [(target+offset,value) for (target,value) in row]
        push!(added,_CircuitSpicePrimitive(terminals,e.kind,e.value,control,controller,e,owner,scale,links))
    end
    # Allocate the replacement collection before committing either field.
    combined=vcat(circuit.elements,added)
    circuit.nnodes=nextnode;circuit.elements=combined
    circuit
end

"""Analyze a compiled linear SPICE model using explicit named pin pairs.
`pin_pairs=[("signal","return"), ...]` supplies the represented external
voltage modes; a pin name `"0"` selects literal global ground. Named pins
follow the library's explicit dialect convention. All independent pins
remain circuit nodes, including common modes and disconnected pins.
Returns [`PlanarCircuitResult`](@ref), with retained internal voltages.
Use `floating_gauge=:auto` for physically floating differential networks.
Unsupported nonlinear/behavioural features reject during library reading."""
function planar_spice_sparams(model::PlanarSpiceModel,f::Real;pin_pairs,z0=50.0,
        floating_gauge::Symbol=:auto,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    isfinite(f) && f>=0 || throw(ArgumentError("SPICE frequency must be finite and nonnegative"))
    f=_circuit_stored_real(f,"SPICE frequency")
    _enforce_payload_limit(_checked_payload_sum("SPICE standalone",model.payload,
        _checked_array_payload_bytes(Int,8,length(model.pins)+length(pin_pairs)),2048),
        max_bytes,"SPICE standalone","max_bytes")
    definition=model.library.definitions[model.subckt]
    logical=Dict{Int,Int}(0=>0)
    for pin in definition.pins
        id=model.pins[pin];haskey(logical,id) || (logical[id]=length(logical))
    end
    pinmap=Dict(pin=>logical[model.pins[pin]] for pin in definition.pins)
    pinmap["0"]=0;model.library.dialect===:spice && (pinmap["gnd"]=0)
    terminals=Tuple{Int,Int}[]
    for pair in pin_pairs
        length(pair)==2 || throw(ArgumentError("SPICE ports require two named pins"))
        a,b=_spice_library_node(model.library,pair[1]),_spice_library_node(model.library,pair[2])
        haskey(pinmap,a) && haskey(pinmap,b) || throw(ArgumentError("unknown SPICE port pin"))
        push!(terminals,(pinmap[a],pinmap[b]))
    end
    circuit=PlanarCircuit(length(logical)-1,terminals;z0)
    circuit_add_spice!(circuit,[pinmap[p] for p in definition.pins],model;max_bytes)
    solve_planar_circuit(circuit,f;max_bytes,floating_gauge)
end
