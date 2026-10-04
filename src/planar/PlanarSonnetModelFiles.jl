# Native component files are staged from bounded snapshots, before EM lowering.
export SonnetModelSource, SonnetComponentBinding, SonnetComponentFiles
export sonnet_component_files
export circuit_add_sonnet_files!

"""One exact native dependency snapshot. `bytes` and `sha256` describe the
same input used for parsing; subsequent filesystem edits do not change it."""
struct SonnetModelSource
    source::String
    bytes::Vector{UInt8}
    sha256::String
end

"""One native SPARAM or linear CKT SPROJ response at a single frequency. Geometry labels are in
model-pin order, independently of their numeric values. `response,z0` retain
the evaluated power-wave basis and the physical global box common return.
`file_index=0` identifies a project component; `inherited_sweep` retains its
native flag while its linear circuit is evaluated at the requested frequency."""
struct SonnetComponentBinding
    id::Int
    file_index::Int
    geometry_labels::Vector{Int}
    pin_indices::Vector{Int}
    source::String
    response::Matrix{ComplexF64}
    z0::Vector{ComplexF64}
    inherited_sweep::Union{Nothing,Bool}
end
SonnetComponentBinding(id,index,labels,pins,source,response,z0)=
    SonnetComponentBinding(id,index,labels,pins,source,response,z0,nothing)

"""Owned frequency-specific native SPARAM/SPROJ staging. The effective project,
exact source/model snapshots and parsed datasets are retained separately.
`configuration_sha256` identifies the effective project/grid/variables and
frequency; raw `sources` hashes identify bytes and do not imply those bytes
produced intentional in-memory project edits.
Interpolation occurs in the file's declared basis before optional output
renormalization. No pathname cache or extrapolation is used. `linked` retains
the original SON/STF identity for explicitly materialized static technology.
`scalar_files` retains native CSV dependencies, whose snapshots also appear
in `sources`; their effective identity is included in the configuration hash.
`circuits` retains compiled linear project children, whose data and recursive
project responses use the same owned snapshots and staged frequency.
Arrays belong to this object and should be treated as read-only."""
struct SonnetComponentFiles
    project::SonnetProject
    frequency::Float64
    root::String
    sources::Dict{String,SonnetModelSource}
    networks::Dict{String,PlanarNetworkData}
    bindings::Vector{SonnetComponentBinding}
    payload::Int
    linked::Union{Nothing,SonnetLinkedProject}
    variables::Dict{String,Float64}
    grid::Union{Nothing,Tuple{Int,Int}}
    configuration_sha256::String
    scalar_files::Union{Nothing,SonnetScalarFiles}
    circuits::Dict{String,PlanarCircuit}
end
SonnetComponentFiles(p,f,r,s,n,b,payload,linked,variables,grid,identity,scalar_files)=
    SonnetComponentFiles(p,f,r,s,n,b,payload,linked,variables,grid,identity,scalar_files,
        Dict{String,PlanarCircuit}())
SonnetComponentFiles(p,f,r,s,n,b,payload,linked,variables,grid,identity)=
    SonnetComponentFiles(p,f,r,s,n,b,payload,linked,variables,grid,identity,nothing)

function _sonnet_files_configuration_hash(p,f,grid,variables;scalar_files=nothing)
    context=SHA.SHA2_256_CTX()
    function scalar(value)
        bytes=codeunits(string(value))
        SHA.update!(context,codeunits(string(length(bytes))*":"))
        SHA.update!(context,bytes)
    end
    function sequence(values)
        scalar(length(values));for value in values;scalar(value);end
    end
    function records(values)
        scalar(length(values))
        for record in values;scalar(record.line);sequence(record.tokens);end
    end
    scalar("DiffMoM.native-effective-configuration.v1")
    scalar(p.source);scalar(p.length_scale);scalar(p.frequency_scale)
    for dictionary in (p.units,p.variables,variables)
        scalar(length(dictionary))
        for (key,value) in sort!(collect(dictionary);by=first);scalar(key);scalar(value);end
    end
    for fields in (p.box,p.top,p.bottom);sequence(fields);end
    for rows in (p.layers,p.metals)
        scalar(length(rows));for row in rows;sequence(row);end
    end
    scalar(length(p.polygons))
    for poly in p.polygons
        scalar(poly.kind);scalar(poly.level);scalar(poly.material);scalar(poly.id)
        scalar(size(poly.vertices,1));scalar(size(poly.vertices,2));sequence(poly.vertices)
        scalar(poly.target);scalar(poly.technology);sequence(poly.flags)
    end
    scalar(length(p.ports))
    for port in p.ports
        scalar(port.kind);scalar(port.polygon);scalar(port.edge);scalar(port.number)
        sequence(port.values);records(port.records)
    end
    scalar(length(p.components));for component in p.components;records(component);end
    records(p.sweeps);records(p.records);scalar(f);scalar(grid)
    if scalar_files!==nothing
        scalar("native-scalar-snapshots.v1");scalar(scalar_files.configuration_sha256)
        scalar(scalar_files.source.path);scalar(scalar_files.source.sha256)
    end
    return bytes2hex(SHA.digest!(context))
end

function _sonnet_files_reserve!(budget::_SpiceBudget,bytes)
    _spice_reserve!(budget,bytes)
    nothing
end
function _sonnet_files_limit(name,value)
    value isa Integer && !(value isa Bool) && 0<value<=typemax(Int) ||
        throw(ArgumentError("$name must be a positive machine integer"))
    return Int(value)
end
function _sonnet_files_id(p,component)
    ids=filter(r->first(r.tokens)=="ID",component)
    length(ids)==1 && length(only(ids).tokens)==2 ||
        _sonnet_error(p.source,first(component).line,"component requires one literal ID")
    id=tryparse(Int,only(ids).tokens[2])
    id!==nothing && id>0 || _sonnet_error(p.source,only(ids).line,"component ID must be positive")
    return id
end
function _sonnet_files_kind(p,component)
    isempty(component) && _sonnet_error(p.source,0,"empty component")
    types=filter(r->first(r.tokens)=="TYPE",component)
    length(types)==1 && length(only(types).tokens)>=2 ||
        _sonnet_error(p.source,first(component).line,"component requires exactly one complete TYPE")
    return only(types)
end
function _sonnet_files_contract(p,component)
    kind=_sonnet_files_kind(p,component)
    modelkind=kind.tokens[2]
    modelkind in ("SPARAM","SPROJ","IDEAL","NONE") || _sonnet_error(p.source,kind.line,
        "TYPE $modelkind requires its explicit model adapter")
    id=_sonnet_files_id(p,component)
    if modelkind=="SPARAM"
        length(kind.tokens)==3 || _sonnet_error(p.source,kind.line,"TYPE SPARAM requires one SMDFILES index")
        index=tryparse(Int,kind.tokens[3])
        index!==nothing && index>0 || _sonnet_error(p.source,kind.line,"SPARAM file index must be a positive literal integer")
    elseif modelkind=="SPROJ"
        length(kind.tokens)==3 && endswith(lowercase(kind.tokens[3]),".son") ||
            _sonnet_error(p.source,kind.line,"TYPE SPROJ requires one literal .son path; parameter bindings need their explicit adapter")
        index=0
    elseif modelkind=="IDEAL"
        length(kind.tokens)==4 && kind.tokens[3] in ("RES","CAP","IND") ||
            _sonnet_error(p.source,kind.line,"IDEAL requires one scalar R/L/C model")
    else
        length(kind.tokens)==2 || _sonnet_error(p.source,kind.line,"NONE requires no model fields")
    end
    pins=Tuple{Int,Int}[]
    topground=false;pinblock=false;ground=false;width=false;inherit=nothing
    for record in component
        t=record.tokens;tag=first(t)
        if tag=="GNDREF"
            t in (["GNDREF","AUTO"],["GNDREF","BOX","AUTO"]) ||
                _sonnet_error(p.source,record.line,"automatic SPARAM requires explicit PEC box AUTO ground")
            pinblock ? (ground=true) : (topground=true)
        elseif tag=="INHSWP"
            modelkind=="SPROJ" && inherit===nothing && t in (["INHSWP","N"],["INHSWP","Y"]) ||
                _sonnet_error(p.source,record.line,"SPROJ requires one literal INHSWP N/Y")
            inherit=t[2]=="Y"
        elseif tag=="TERMW"
            t==["TERMW","FEED"] || _sonnet_error(p.source,record.line,
                "TERMW $(join(t[2:end]," ")) needs a physical width adapter")
            pinblock || _sonnet_error(p.source,record.line,"TERMW must describe a model pin")
            width=true
        elseif tag=="SMDP"
            pinblock && _sonnet_error(p.source,record.line,"unterminated model-pin metadata")
            isempty(pins) || ground && width || _sonnet_error(p.source,record.line,
                "each model pin requires explicit AUTO ground and FEED width")
            length(t)==7 || _sonnet_error(p.source,record.line,"SMDP requires level,x,y,orientation,geometry label,model pin")
            t[5] in ("L","R","T","B") || _sonnet_error(p.source,record.line,"unrepresented terminal orientation")
            label=tryparse(Int,t[6]);pin=tryparse(Int,t[7]);level=tryparse(Int,t[2])
            label!==nothing && label>0 && pin!==nothing && pin>0 && level!==nothing && level>=0 ||
                _sonnet_error(p.source,record.line,"model pins require positive labels/indices and a literal nonnegative level")
            push!(pins,(pin,label));ground=false;width=false
        elseif tag=="EXSMDP"
            !isempty(pins) && !pinblock && length(t)==1 || _sonnet_error(p.source,record.line,"unexpected EXSMDP")
            pinblock=true
        elseif tag=="END" && length(t)==2 && t[2]=="EXSMDP"
            pinblock && ground && width || _sonnet_error(p.source,record.line,
                "each model pin requires explicit AUTO ground and FEED width")
            pinblock=false
        elseif tag in ("DRP1","DRP2","REFPLANE")
            _sonnet_error(p.source,record.line,"native component reference planes require their coupled calibration transfer")
        elseif !(tag in ("SMD","ID","SBOX","PBSHW","LPOS","TYPE","END"))
            _sonnet_error(p.source,record.line,"unrepresented component record $tag")
        end
    end
    !pinblock && topground && !isempty(pins) && ground && width ||
        _sonnet_error(p.source,kind.line,"SPARAM requires explicit component/pin ground and FEED metadata")
    sort!(pins;by=first)
    all(k->pins[k][1]==k,eachindex(pins)) || _sonnet_error(p.source,kind.line,"model pin indices must be unique and contiguous from one")
    modelkind=="IDEAL" && length(pins)!=2 && _sonnet_error(p.source,kind.line,"IDEAL requires two model pins")
    modelkind=="SPROJ" && inherit===nothing && _sonnet_error(p.source,kind.line,"SPROJ requires explicit INHSWP N/Y")
    return modelkind in ("SPARAM","SPROJ") ? (;id,index,kind,labels=last.(pins),inherit) : nothing
end

function _sonnet_files_pin_preflight(p,f,grid,variables)
    counts=grid===nothing ? (parse(Int,p.box[4])÷2,parse(Int,p.box[5])÷2) : grid
    a=sonnet_variable_value(p,p.box[2];variables,freq=f)*p.length_scale
    b=sonnet_variable_value(p,p.box[3];variables,freq=f)*p.length_scale
    isfinite(a) && isfinite(b) && a>0 && b>0 || throw(ArgumentError("native component box must be finite and positive"))
    dx,dy=a/counts[1],b/counts[2]
    tol=1e-9*max(a,b)
    for component in p.components
        for record in component
            first(record.tokens)=="SMDP" || continue
            t=record.tokens;level=parse(Int,t[2]);axis=t[5] in ("L","R")
            x=sonnet_variable_value(p,t[3];variables,freq=f)*p.length_scale
            y=sonnet_variable_value(p,t[4];variables,freq=f)*p.length_scale
            all(isfinite,(x,y)) && 0<=x<=a && 0<=y<=b || _sonnet_error(p.source,record.line,"model pin lies outside the finite box")
            axial,transverse=axis ? (x,y) : (y,x);spacing=axis ? dx : dy
            abs(round(axial/spacing)*spacing-axial)<=1e-7*spacing ||
                _sonnet_error(p.source,record.line,"SMD edge is not aligned to this grid; retain or refine the native raster")
            candidates=Tuple{Float64,Float64}[]
            for poly in p.polygons
                poly.kind===:sheet && poly.level==level || continue
                v=poly.vertices;n=size(v,2)
                for k in 1:n
                    next=k==n ? 1 : k+1
                    firstax,lastax=axis ? (v[1,k],v[1,next]) : (v[2,k],v[2,next])
                    firsttr,lasttr=axis ? (v[2,k],v[2,next]) : (v[1,k],v[1,next])
                    abs(firstax-axial)<=tol && abs(lastax-axial)<=tol || continue
                    lo,hi=minmax(firsttr,lasttr)
                    hi-lo>tol && lo-tol<=transverse<=hi+tol || continue
                    push!(candidates,(lo,hi))
                end
            end
            unique!(candidates)
            length(candidates)==1 || _sonnet_error(p.source,record.line,"model pin must identify one straight polygon edge")
        end
    end
    nothing
end

function _sonnet_files_project_payload(p)
    # Conservative owned object/string/container payload before deepcopy.
    total=BigInt(4096)
    for records in (p.records,p.sweeps),record in records
        total+=192+2sum(ncodeunits,record.tokens)+64length(record.tokens)
    end
    for component in p.components,record in component
        total+=192+2sum(ncodeunits,record.tokens)+64length(record.tokens)
    end
    for poly in p.polygons
        total+=256+8length(poly.vertices)+2ncodeunits(poly.target)+2ncodeunits(poly.technology)+
            64length(poly.flags)+2sum(ncodeunits,poly.flags)
    end
    for port in p.ports
        total+=256+64length(port.values)+2sum(ncodeunits,port.values)
        for record in port.records
            total+=192+64length(record.tokens)+2sum(ncodeunits,record.tokens)
        end
    end
    for fields in (p.box,p.top,p.bottom),token in fields;total+=64+2ncodeunits(token);end
    for rows in (p.layers,p.metals),row in rows,token in row;total+=64+2ncodeunits(token);end
    for dictionary in (p.units,p.variables),(key,value) in dictionary;total+=192+2ncodeunits(key)+2ncodeunits(value);end
    return _checked_payload_sum("native component project snapshot",total)
end
function _sonnet_files_snapshot(path,budget)
    bytes=filesize(path)
    _sonnet_files_reserve!(budget,BigInt(2)*(BigInt(bytes)+1)+512+2ncodeunits(path))
    content=open(path) do io
        data=read(io,bytes+1)
        length(data)==bytes && eof(io) || throw(ArgumentError("native dependency changed size during snapshot"))
        data
    end
    return SonnetModelSource(path,content,bytes2hex(SHA.sha256(content)))
end
function _sonnet_files_path(root,p,name,record)
    path=normpath(isabspath(name) ? name : joinpath(dirname(p.source),name))
    isfile(path) || _sonnet_error(p.source,record.line,"missing model dependency $name")
    resolved=realpath(path);relative=relpath(resolved,root)
    !isabspath(relative) && all(x->x!="..",splitpath(relative)) ||
        _sonnet_error(p.source,record.line,"model dependency resolves outside root")
    return resolved
end
function _sonnet_files_data_payload(data)
    _checked_payload_sum("native model retained data",_subdivision_retained_payload(data.frequencies),
        _subdivision_retained_payload(data.s),_subdivision_retained_payload(data.z0),
        _subdivision_retained_payload(data.reference_series),256+128length(data.s)+2sum(ncodeunits,data.port_names))
end
function _sonnet_files_geometry_payload(p,grid)
    nx,ny=if grid===nothing
        hx=tryparse(Int,p.box[4]);hy=tryparse(Int,p.box[5])
        hx!==nothing && hy!==nothing && hx>0 && hy>0 && iseven(hx) && iseven(hy) ||
            throw(ArgumentError("native BOX half-cell counts must be positive even integers"))
        hx÷2,hy÷2
    else
        grid isa Tuple && length(grid)==2 && all(x->x isa Integer && !(x isa Bool) && 0<x<=typemax(Int),grid) ||
            throw(ArgumentError("native component grid must contain two positive machine integers"))
        Int(grid[1]),Int(grid[2])
    end
    # Upper bounds include thick-face expansion, native/synthetic via masks,
    # all rooftop/via basis indices and physical voltage contractions.
    nl=BigInt(length(p.layers))+2length(p.polygons)
    ns=BigInt(2length(p.polygons)+1)
    np=BigInt(length(p.ports))+sum(count(r->first(r.tokens)=="SMDP",c) for c in p.components;init=0)
    cells=BigInt(nx)*ny
    rawports=BigInt(length(p.ports))+np*max(nl,1)
    needed=BigInt(4096)+cells*(BigInt(128)*ns+BigInt(256)*max(nl,1)*max(np,1))+
        BigInt(32)*rawports*max(np,1)+BigInt(512)*rawports
    return _checked_payload_sum("native component geometry preflight",needed)
end

include("PlanarSonnetProjectFiles.jl")

"""Stage native SPARAM/SMDFILES and literal linear CKT SPROJ models from bounded exact snapshots.
Dependencies resolve inside `root`; shared files are parsed once. Model-pin
indices determine order independently of geometry labels. The represented
automatic source contract requires PEC box AUTO return, explicit per-pin
FEED width and no unrepresented pin reference plane. Other widths/grounds
reject before geometry allocation or optional reference callbacks. Frequencies
must be within model coverage. This stages raw physical attachment; native
coupled pin calibration and licensed SMD renderer compatibility remain separate.
Linear SPROJ children support literal R/L/C, Touchstone S<n>P, earlier DEF<n>P
invocations and recursive PRJ children with literal 0/1 inheritance flags.
They evaluate continuously at the requested frequency, retaining explicit
INHSWP N/Y metadata; their saved child sweep is not interpolated. Shared
dependencies parse once; cycles, root escapes, unsupported parameter bindings
and geometry children reject. `max_project_depth` bounds project recursion,
and `max_project_nodes/max_project_elements` bound retained definitions in
each project. Dense nested solves also share the aggregate `max_bytes` bound.
Geometry children still require their explicit calibrated model adapter.
An explicit `requested_reference` only renormalizes the already evaluated
physical response. An extracted archive tree can be supplied as `root`; native
compressed-model formats are not inferred."""
function sonnet_component_files(p::SonnetProject,frequency::Real;
        root=dirname(abspath(p.source)),max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        max_dependencies::Integer=16,max_project_depth::Integer=16,
        max_project_nodes::Integer=10000,max_project_elements::Integer=10000,
        requested_reference=nothing,grid=nothing,
        variables=Dict{String,Float64}(),scalar_files=nothing,
        scalar_outside::Symbol=:reject,scalar_max_files::Integer=64,
        scalar_max_bytes::Integer=8*1024^2,scalar_max_nodes::Integer=100000,
        scalar_max_line_bytes::Integer=16384,_linked=nothing,_preserve_scalar_geometry::Bool=false)
    limit=_sonnet_files_limit("max_bytes",max_bytes);dependencies=_sonnet_files_limit("max_dependencies",max_dependencies)
    depth=_sonnet_files_limit("max_project_depth",max_project_depth)
    nodes=_sonnet_files_limit("max_project_nodes",max_project_nodes)
    elements=_sonnet_files_limit("max_project_elements",max_project_elements)
    f=_circuit_stored_real(frequency,"native component frequency");f>0 || throw(ArgumentError("native component frequency must be representable and positive"))
    budget=_SpiceBudget(0,limit,1,0,1)
    projectbytes=_sonnet_files_project_payload(p)
    _sonnet_files_reserve!(budget,2BigInt(projectbytes)+4096)
    p=deepcopy(p) # Own the effective configuration before reference callbacks.
    parent=(_linked!==nothing && scalar_files===nothing && _sonnet_scalar_files(variables)===nothing &&
        _sonnet_has_scalar_tables(p)) ? _sonnet_scalar_parent(_linked;max_bytes=limit-budget.used) : (;)
    variables=_sonnet_scalar_variables(p,variables;scalar_files,root,
        max_bytes=limit-budget.used,max_bytes_source=scalar_max_bytes,max_files=scalar_max_files,
        outside=scalar_outside,max_nodes=scalar_max_nodes,max_line_bytes=scalar_max_line_bytes,parent...)
    scalar_files=_sonnet_scalar_files(variables)
    _preserve_scalar_geometry || (p=_sonnet_scalar_project(p,variables))
    _sonnet_files_reserve!(budget,_sonnet_scalar_payload(variables))
    owned_variables=variables isa SonnetScalarVariables ? variables.overrides : variables
    descriptions=Any[]
    for component in p.components
        description=_sonnet_files_contract(p,component)
        description===nothing || push!(descriptions,description)
    end
    !isempty(descriptions) || throw(ArgumentError("project has no SPARAM or SPROJ component"))
    length(unique(d.id for d in descriptions))==length(descriptions) || throw(ArgumentError("duplicate native component ID"))
    has_sparam=any(d->d.kind.tokens[2]=="SPARAM",descriptions)
    !has_sparam || count(r->r.tokens==["SMDFILES"],p.records)==1 || throw(ArgumentError("SPARAM requires one SMDFILES section"))
    table=Dict{Int,Tuple{String,SonnetRecord}}()
    for record in (has_sparam ? _sonnet_section(p.records,"SMDFILES",p.source) : SonnetRecord[])
        length(record.tokens)==2 || _sonnet_error(p.source,record.line,"SMDFILES entry requires index and model path")
        index=tryparse(Int,record.tokens[1]);index!==nothing && index>0 || _sonnet_error(p.source,record.line,"invalid SMDFILES index")
        haskey(table,index) && _sonnet_error(p.source,record.line,"duplicate SMDFILES index")
        table[index]=(record.tokens[2],record)
    end
    for cover in (p.top,p.bottom)
        _sonnet_termination(p,cover,f,variables).kind===TERM_PEC ||
            throw(ArgumentError("automatic SPARAM requires two PEC box covers; finite-loss AUTO selection is unresolved"))
    end
    geometrybytes=_sonnet_files_geometry_payload(p,grid)
    _enforce_payload_limit(_checked_payload_sum("native file geometry staging",budget.used,
        geometrybytes),limit,"native file geometry staging","max_bytes")
    _sonnet_files_pin_preflight(p,f,grid,variables)
    directory=realpath(root);parent=realpath(p.source)
    rel=relpath(parent,directory)
    !isabspath(rel) && all(x->x!="..",splitpath(rel)) || throw(ArgumentError("parent project resolves outside root"))
    paths=String[];dimensions=Dict{String,Int}()
    for d in descriptions
        d.kind.tokens[2]=="SPARAM" || continue
        haskey(table,d.index) || _sonnet_error(p.source,d.kind.line,"unknown SMDFILES index $(d.index)")
        name,record=table[d.index];path=_sonnet_files_path(directory,p,name,record)
        suffix=match(r"\.([syz])(\d+)p$"i,path)
        suffix===nothing && lowercase(splitext(path)[2])!=".ts" && _sonnet_error(p.source,record.line,"unrepresented model file format")
        suffix===nothing || tryparse(Int,suffix[2])==length(d.labels) || _sonnet_error(p.source,record.line,"model file port count disagrees with ordered pins")
        haskey(dimensions,path) && dimensions[path]!=length(d.labels) && _sonnet_error(p.source,record.line,"shared model dependency has inconsistent pin dimensions")
        if !haskey(dimensions,path);push!(paths,path);dimensions[path]=length(d.labels);end
    end
    scalar_sources=scalar_files===nothing ? SonnetScalarSource[] : _sonnet_scalar_dependency_sources(scalar_files)
    1+length(paths)+length(scalar_sources)+(_linked===nothing ? 0 : 1)<=dependencies ||
        throw(ArgumentError("native dependency count exceeds max_dependencies"))
    sources=Dict{String,SonnetModelSource}(parent=>_sonnet_files_snapshot(parent,budget))
    for source in scalar_sources
        _sonnet_files_reserve!(budget,256+2BigInt(ncodeunits(source.path)))
        relative=relpath(source.path,directory)
        !isabspath(relative) && all(x->x!="..",splitpath(relative)) ||
            throw(ArgumentError("retained scalar dependency resolves outside model root"))
        sources[source.path]=SonnetModelSource(source.path,source.bytes,source.sha256)
    end
    if _linked!==nothing
        t=_linked.technology
        relative=relpath(t.source,directory)
        !isabspath(relative) && all(x->x!="..",splitpath(relative)) ||
            throw(ArgumentError("retained STF dependency resolves outside model root"))
        _sonnet_files_reserve!(budget,BigInt(16)*(ncodeunits(_linked.raw)+ncodeunits(t.raw))+
            t.limits.estimated_storage_bytes+4096)
        # Preserve the original retained STF snapshot, never silently reread it.
        sources[t.source]=SonnetModelSource(t.source,Vector{UInt8}(codeunits(t.raw)),t.sha256)
        sources[parent].sha256==_linked.sha256 || throw(ArgumentError("linked SON source changed since its retained snapshot"))
    end
    networks=Dict{String,PlanarNetworkData}()
    for path in paths
        snapshot=_sonnet_files_snapshot(path,budget);sources[path]=snapshot
        np=dimensions[path]
        model=mktemp() do file,io
            write(io,snapshot.bytes);close(io)
            planar_read_touchstone(file;nports=np,max_bytes=limit-budget.used)
        end
        _sonnet_files_reserve!(budget,_sonnet_files_data_payload(model));networks[path]=model
    end
    projects=_SonnetLinearProjects(f,directory,budget,dependencies,depth,nodes,elements,
        sources,networks,Dict{String,PlanarCircuit}(),
        Dict{String,Tuple{Matrix{ComplexF64},Vector{ComplexF64}}}(),Set([_sonnet_project_path_key(parent)]))
    project_paths=Dict{Int,String}()
    for d in descriptions
        d.kind.tokens[2]=="SPROJ" || continue
        path=_sonnet_files_path(directory,p,d.kind.tokens[3],d.kind)
        _sonnet_files_linear_project!(projects,path,length(d.labels))
        project_paths[d.id]=path
    end
    # Validate coverage and reserve every binding before any reference provider.
    for d in descriptions
        np=length(d.labels)
        if d.kind.tokens[2]=="SPARAM"
            path=_sonnet_files_path(directory,p,table[d.index]...);model=networks[path]
            first(model.frequencies)<=f<=last(model.frequencies) || _sonnet_error(p.source,d.kind.line,"frequency outside model coverage; extrapolation is forbidden")
        end
        _sonnet_files_reserve!(budget,512+_checked_array_payload_bytes(ComplexF64,12,np,np)+_checked_array_payload_bytes(ComplexF64,12,np))
    end
    _enforce_payload_limit(_checked_payload_sum("native model provider preflight",budget.used,
        geometrybytes),limit,"native model provider preflight","max_bytes")
    bindings=SonnetComponentBinding[]
    for d in descriptions
        np=length(d.labels)
        path=if d.kind.tokens[2]=="SPARAM"
            _sonnet_files_path(directory,p,table[d.index]...)
        else
            project_paths[d.id]
        end
        response,original=if d.kind.tokens[2]=="SPARAM"
            model=networks[path]
            planar_network_response(model,f),copy(_network_reference_at(model,f))
        else
            value,refs=projects.responses[path]
            copy(value),copy(refs)
        end
        refs=requested_reference===nothing ? original : _planar_reference_values(requested_reference,np;freq=f)
        refs==original || (response=planar_renormalize_s(response,original,refs))
        push!(bindings,SonnetComponentBinding(d.id,d.index,copy(d.labels),collect(1:np),path,response,ComplexF64.(refs),d.inherit))
    end
    retained_grid=grid===nothing ? nothing : (Int(grid[1]),Int(grid[2]))
    configuration=_sonnet_files_configuration_hash(p,f,retained_grid,owned_variables;scalar_files)
    return SonnetComponentFiles(p,f,directory,sources,networks,bindings,budget.used,_linked,
        owned_variables,retained_grid,configuration,scalar_files,projects.circuits)
end

function sonnet_component_files(path::AbstractString,frequency::Real;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kwargs...)
    limit=_sonnet_files_limit("max_bytes",max_bytes)
    f=Float64(frequency);isfinite(f) && f>0 || throw(ArgumentError("native component frequency must be representable and positive"))
    source=realpath(path);budget=_SpiceBudget(0,limit,1,0,1)
    snapshot=_sonnet_files_snapshot(source,budget)
    _sonnet_files_reserve!(budget,BigInt(32)*length(snapshot.bytes)+4096)
    records=SonnetRecord[]
    for (line,text) in enumerate(eachsplit(String(copy(snapshot.bytes)),'\n'))
        tokens=_sonnet_tokens(text);isempty(tokens) || push!(records,SonnetRecord(line,tokens))
    end
    p=_sonnet_read_records(source,records)
    p isa SonnetProject || throw(ArgumentError("component files require a native geometry project"))
    result=sonnet_component_files(p,frequency;max_bytes=limit-budget.used,kwargs...)
    result.sources[source].sha256==snapshot.sha256 || throw(ArgumentError("parent project changed during staging"))
    return result
end

function sonnet_component_files(linked::SonnetLinkedProject,frequency::Real;
        technology_variables=Dict{String,Float64}(),max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kwargs...)
    limit=_sonnet_files_limit("max_bytes",max_bytes)
    f=Float64(frequency);isfinite(f) && f>0 || throw(ArgumentError("native component frequency must be representable and positive"))
    preflight=_checked_payload_sum("linked native source materialization",
        BigInt(32)*(ncodeunits(linked.raw)+ncodeunits(linked.technology.raw))+
        linked.technology.limits.estimated_storage_bytes+4096)
    _enforce_payload_limit(preflight,limit,"linked native source materialization","max_bytes")
    owned=deepcopy(linked)
    project=sonnet_materialize_project(owned;technology_variables)
    return sonnet_component_files(project,frequency;max_bytes=limit-preflight,_linked=owned,kwargs...)
end

"""Transactionally attach staged native SPARAM/SPROJ devices to physical circuit
nodes. `labels[node]` is the native geometry label at that node. Ordered model
pins map through these labels; every model port retains box node zero as its
common return. Invalid maps/resources leave the supplied circuit unchanged."""
function circuit_add_sonnet_files!(circuit::PlanarCircuit,files::SonnetComponentFiles,
        labels::AbstractVector{<:Integer};max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    limit=_sonnet_files_limit("max_bytes",max_bytes)
    needed=_checked_payload_sum("native transactional file attachment",files.payload,
        _spice_circuit_payload(circuit),4096,
        (_checked_array_payload_bytes(ComplexF64,2,length(b.geometry_labels),length(b.geometry_labels)) for b in files.bindings)...)
    _enforce_payload_limit(needed,limit,"native transactional file attachment","max_bytes")
    length(labels)==circuit.nnodes && length(unique(labels))==length(labels) ||
        throw(ArgumentError("native geometry labels must identify every circuit node uniquely"))
    maps=Vector{Int}[]
    for binding in files.bindings
        nodes=Int[]
        for label in binding.geometry_labels
            index=findfirst(==(label),labels)
            index===nothing && throw(ArgumentError("model geometry label has no physical circuit node"))
            push!(nodes,index)
        end
        push!(maps,nodes)
    end
    refs=circuit.z0 isa AbstractArray ? copy(circuit.z0) : circuit.z0
    staged=PlanarCircuit(circuit.nnodes,copy(circuit.ports),refs,copy(circuit.elements))
    for (binding,nodes) in zip(files.bindings,maps)
        circuit_add_network!(staged,nodes,binding.response;format=:s,z0=binding.z0)
    end
    circuit.elements=staged.elements
    return circuit
end
