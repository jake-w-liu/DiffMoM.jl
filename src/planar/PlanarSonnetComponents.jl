# Native SMD attachment through physical reference vias and circuit MNA.

export SonnetComponentResult, sonnet_component_model, sonnet_component_current_maps

"""Native SMD solve retaining the raw multi-terminal EM solution,
terminal contraction and composite circuit response. `s,y` are external
responses; component pins remain internal circuit nodes. Raw reference
vias provide physical cover-ground returns and retain their lead effects.
Pin-group calibration must be supplied independently before claiming
calibrated Sonnet equivalence. `model_files` retains exact automatic SPARAM
sources, evaluated references and the staged effective configuration.
`scalar_files,scalar_overrides` retain owned native CSV dependencies and
the numeric overrides used to lower this response."""
struct SonnetComponentResult
    project::SonnetProject
    raw::Union{PlanarResult,PlanarUFFTResult}
    port_numbers::Vector{Int}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    contraction::Matrix{ComplexF64}
    circuit::PlanarCircuitResult
    terminal_nodes::Vector{Int}
    incident_transfer::Matrix{ComplexF64}
    gap_transfer::Matrix{ComplexF64}
    z0::Vector{ComplexF64}
    model_files::Union{Nothing,SonnetComponentFiles}
    scalar_files::Union{Nothing,SonnetScalarFiles}
    scalar_overrides::Dict{String,Float64}
end
SonnetComponentResult(p,r,n,y,s,c,network,nodes,a,g,z0,files)=
    SonnetComponentResult(p,r,n,y,s,c,network,nodes,a,g,z0,files,nothing,Dict{String,Float64}())
SonnetComponentResult(p,r,n,y,s,c,network,nodes,a,g,z0)=
    SonnetComponentResult(p,r,n,y,s,c,network,nodes,a,g,z0,nothing)
SonnetComponentResult(p,r,n,y,s,c,network,nodes,a,g)=
    SonnetComponentResult(p,r,n,y,s,c,network,nodes,a,g,network.z0)

function _sonnet_component_scale(p,kind)
    units,default=kind=="RES" ? (Dict("OH"=>1.,"OHMS"=>1.,"KOH"=>1e3,"MOH"=>1e6),"OH") :
        kind=="CAP" ? (Dict("F"=>1.,"PF"=>1e-12,"FF"=>1e-15,"NF"=>1e-9,"UF"=>1e-6),"PF") :
        kind=="IND" ? (Dict("H"=>1.,"NH"=>1e-9,"PH"=>1e-12,"UH"=>1e-6,"MH"=>1e-3),"NH") :
        throw(ArgumentError("unsupported native IDEAL component $kind"))
    scale=get(units,get(p.units,kind,default),NaN)
    isfinite(scale) || throw(ArgumentError("unsupported native IDEAL $kind units"))
    scale
end
function _sonnet_component_scalar(p,text,scale,variables,freq)
    literal=tryparse(Float64,text)
    literal===nothing && occursin(r"^[+\-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+\-]?[0-9]+)?$",strip(text)) &&
        throw(ArgumentError("native IDEAL scalar is not representable in Float64"))
    literal!==nothing && iszero(literal) && _spice_literal_nonzero(text) &&
        throw(ArgumentError("native IDEAL scalar underflows Float64"))
    value=sonnet_variable_value(p,text;variables,freq)
    stored=value*scale
    isfinite(stored) && stored>=0 && (iszero(value) || !iszero(stored)) ||
        throw(ArgumentError("native IDEAL value must remain finite, nonnegative and preserve nonzero values in SI units"))
    stored
end
function _sonnet_component_variables(variables,max_bytes)
    variables isa AbstractDict || throw(ArgumentError("native component variables must be a dictionary"))
    isempty(variables) && return variables
    all(key->key isa AbstractString,keys(variables)) || throw(ArgumentError("native variable names must be strings"))
    needed=_checked_payload_sum("native component variable storage",512,
        _checked_array_payload_bytes(UInt8,128,length(variables)),2sum(ncodeunits,keys(variables)))
    _enforce_payload_limit(needed,max_bytes,"native component variable storage","max_bytes")
    all(value->value isa Real,values(variables)) || throw(ArgumentError("native variable overrides must be real numbers"))
    Dict(String(key)=>_circuit_stored_real(value,"native component variable override") for (key,value) in variables)
end

function _sonnet_component_ideal_preflight(p,variables,freq)
    for component in p.components,record in component
        t=record.tokens
        length(t)>=2 && t[1:2]==["TYPE","IDEAL"] || continue
        length(t)==4 || _sonnet_error(p.source,record.line,"ideal SMD requires one R/L/C value")
        _sonnet_component_scalar(p,t[4],_sonnet_component_scale(p,t[3]),variables,freq)
    end
end

function _sonnet_component_variable_payload(variables)
    numeric=isempty(variables) ? 0 : _checked_payload_sum("native owned variables",512,
        _checked_array_payload_bytes(UInt8,128,length(variables)),2sum(ncodeunits,keys(variables)))
    files=_sonnet_scalar_files(variables)
    _checked_payload_sum("native owned scalar variables",numeric,
        files===nothing ? 0 : _sonnet_scalar_files_payload(files))
end

function _sonnet_component_base_payload(base)
    seen=Base.IdSet{Any}()
    values=(base.problem.basis,base.problem.sheets,base.problem.vias,base.problem.vols,
        base.problem.stack.layers,base.sheet_zs,base.via_sigma,base.contraction,
        base.floating_common,base.z0,base.labels)
    _checked_payload_sum("native live base geometry",4096,
        _checked_array_payload_bytes(UInt8,256,length(base.problem.ports)),
        _checked_array_payload_bytes(UInt8,128,length(base.problem.sheets)+length(base.problem.vias)+
            length(base.problem.vols)+length(base.problem.stack.layers)),
        (_subdivision_retained_payload(value,seen) for value in values)...)
end

function _sonnet_component_post_payload(base,pins,labels)
    ninput=BigInt(length(base.problem.ports))+sum(length,pins)
    nc=BigInt(size(base.floating_common,2));nn=BigInt(length(labels))
    contacts=sum(BigInt(length(pin.terminal.cells))*length(base.problem.stack.layers)
        for group in pins for pin in group;init=BigInt(0))
    nr=BigInt(length(base.problem.ports))+2contacts
    response=sum(BigInt(length(group))^2 for group in pins;init=BigInt(0))
    _checked_payload_sum("native component post geometry workspace",
        4096+512BigInt(length(pins))+256(nn+ninput),
        _checked_array_payload_bytes(Float64,ninput,nn+nc),
        _checked_array_payload_bytes(Float64,nr,nn+nc),
        _checked_array_payload_bytes(ComplexF64,2response+12nn),
        _checked_array_payload_bytes(Tuple{Int,Int},ninput))
end

function _sonnet_component_retained_payload(model)
    seen=Base.IdSet{Any}()
    values=(model.problem.basis,model.problem.sheets,model.problem.vias,model.problem.vols,
        model.problem.stack.layers,model.sheet_zs,model.via_sigma,model.contraction,
        model.floating_common,model.z0,model.labels,model.external_labels)
    npins=sum(length,model.pins)
    _checked_payload_sum("native retained component model",4096,
        model.model_files===nothing ? 0 : model.model_files.payload,
        model.scalar_files===nothing ? 0 : _sonnet_scalar_files_payload(model.scalar_files),
        _sonnet_component_variable_payload(model.variables),
        _spice_circuit_payload(model.circuit),
        _checked_array_payload_bytes(UInt8,256,length(model.problem.ports)+npins+length(model.reference_paths)),
        _checked_array_payload_bytes(UInt8,128,length(model.problem.sheets)+length(model.problem.vias)+
            length(model.problem.vols)+length(model.problem.stack.layers)),
        (_subdivision_retained_payload(value,seen) for value in values)...,
        (_subdivision_retained_payload(path.cells,seen) for path in model.reference_paths)...)
end

# Resolve each pin's declared terminal width. A TERMW inside a pin's
# EXSMDP block applies to that pin alone; a bare component-level TERMW is
# the shared default. The native modes are FEED (the whole straight
# edge), CELL (the single cell containing the pin coordinate) and
# CUST <width> (the cells covering the declared width centered on the
# pin coordinate); the resolved strip must stay inside the pad edge.
function _sonnet_component_termw(p::SonnetProject,component)
    default="FEED";widths=Dict{Int,String}();inside=false;pin=0
    for record in component
        t=record.tokens
        if t[1]=="SMDP"
            pin=length(t)>=7 ? something(tryparse(Int,t[7]),0) : 0
        elseif t[1]=="EXSMDP"
            inside=true
        elseif t[1]=="END"
            inside=false
        elseif t[1]=="TERMW"
            length(t)>=2 || _sonnet_error(p.source,record.line,"TERMW requires a width mode")
            mode=join(t[2:end]," ")
            if inside
                pin==0 && _sonnet_error(p.source,record.line,"EXSMDP terminal width lacks its pin")
                widths[pin]=mode
            else
                default=mode
            end
        end
    end
    return (default=default,widths=widths)
end

# Decode one GNDREF record. At component level the record selects the
# component's ground-reference mode (`FLOAT`, `AUTO`, or `BOX` = via
# returns to a cover); a `BOX` direction token does not select the cover
# itself. Inside a pin's EXSMDP the record selects that pin's via
# direction (`AUTO`/`BOX AUTO` = nearest conductive cover).
function _sonnet_gndref_value(source,record)
    t=record.tokens
    t==["GNDREF","AUTO"] && return :auto
    (t==["GNDREF","FLOAT"] || t==["GNDREF","F"]) && return :float
    t==["GNDREF","BOX","AUTO"] && return :box_auto
    t==["GNDREF","BOX","BELOW"] && return :below
    t==["GNDREF","BOX","ABOVE"] && return :above
    _sonnet_error(source,record.line,"unrepresented SMD ground reference $(join(t[2:end]," "))")
end

# Resolve a component's declared ground reference. A component-level
# GNDREF selects the mode: `FLOAT` shares one floating local ground
# across the pins, `BOX ...` puts the component in via mode where each
# pin's own record chooses its cover, and `AUTO`/missing lets the model
# kind decide (verified: IDEAL/NONE resolve floating; data-file models
# resolve to a box-cover via). The component-level BOX direction token
# is inert — native resolves each pin's cover separately. `dirs` maps
# native pin index to `:auto`/`:below`/`:above` and only matters in
# `:via` mode; per-pin records elsewhere carry no effect.
function _sonnet_component_gndref(p::SonnetProject,component)
    mode=:auto;dirs=Dict{Int,Symbol}();inside=false;pin=0;pinfloat=nothing
    for record in component
        t=record.tokens
        if t[1]=="SMDP"
            pin=length(t)>=7 ? something(tryparse(Int,t[7]),0) : 0
        elseif t[1]=="EXSMDP"
            inside=true
        elseif t[1]=="END"
            inside=false
        elseif t[1]=="GNDREF"
            value=_sonnet_gndref_value(p.source,record)
            if inside
                pin==0 && _sonnet_error(p.source,record.line,
                    "EXSMDP ground reference lacks its pin")
                value===:float && pinfloat===nothing && (pinfloat=record)
                dirs[pin]=value in (:below,:above) ? value : :auto
            else
                m=value===:float ? :float : value===:auto ? :auto : :via
                mode===:auto || m===:auto || mode===m ||
                    _sonnet_error(p.source,record.line,
                        "mixed component ground-reference modes")
                mode=m===:auto ? mode : m
            end
        end
    end
    # A pin-level FLOAT is only meaningful under a FLOAT component (where
    # it restates the shared local ground, as in the native corpus) and is
    # inert under an unresolved AUTO component; on a via-mode component it
    # is invalid grammar.
    pinfloat!==nothing && mode===:via &&
        _sonnet_error(p.source,pinfloat.line,
            "pin ground reference cannot select FLOAT")
    return (mode=mode,dirs=dirs)
end

_sonnet_component_name(component)=(k=findfirst(r->r.tokens[1]=="SMD",component);
    k===nothing || length(component[k].tokens)<2 ? "SMD" : component[k].tokens[2])

# Whether the component resolves to the floating local-ground adapter:
# explicit FLOAT, or AUTO with a ground-free model (IDEAL R/L/C, NONE).
# Components whose model consumes a real ground node resolve to a
# box-cover via return under AUTO.
function _sonnet_component_floating(p::SonnetProject,component)
    gnd=_sonnet_component_gndref(p,component)
    gnd.mode===:float && return true
    gnd.mode===:via && return false
    kind=findfirst(r->r.tokens[1]=="TYPE",component)
    kind===nothing && return false
    t=component[kind].tokens
    return t==["TYPE","NONE"] || (length(t)==4 && t[2]=="IDEAL")
end

function _sonnet_termw_cells(p,record,termw,lo,hi,transverse,h,variables,freq)
    m=split(termw)
    edge_first=floor(Int,lo/h+1e-8)+1;edge_last=ceil(Int,hi/h-1e-8)
    if m==["FEED"]
        first_cell,last_cell=edge_first,edge_last
    elseif m==["CELL"]
        first_cell=last_cell=floor(Int,transverse/h+1e-8)+1
    elseif length(m)==2 && m[1]=="CUST"
        w=sonnet_variable_value(p,m[2];variables,freq)*p.length_scale
        isfinite(w) && w>0 || _sonnet_error(p.source,record.line,
            "custom SMD terminal width must be positive")
        first_cell=floor(Int,(transverse-w/2)/h+1e-8)+1
        last_cell=ceil(Int,(transverse+w/2)/h-1e-8)
        (first_cell<edge_first || last_cell>edge_last) &&
            _sonnet_error(p.source,record.line,"custom SMD terminal width exceeds its pad edge")
    else
        _sonnet_error(p.source,record.line,"unsupported SMD terminal width $termw")
    end
    first_cell=clamp(first_cell,edge_first,edge_last)
    last_cell=clamp(last_cell,edge_first,edge_last)
    return first_cell,last_cell
end

function _sonnet_component_pin(p,record,problem,variables,freq;termw="FEED")
    t=record.tokens
    length(t)==7 || _sonnet_error(p.source,record.line,"SMDP requires level,x,y,orientation,terminal,pin")
    val(s)=sonnet_variable_value(p,s;variables=variables,freq=freq)
    native_level=parse(Int,t[2]);x,y=val(t[3])*p.length_scale,val(t[4])*p.length_scale
    orient=t[5];orient in ("L","R","T","B") || _sonnet_error(p.source,record.line,
        "SMD terminal orientation must be L/R/T/B")
    number,pin=parse(Int,t[6]),parse(Int,t[7])
    number!=0 && pin>0 || _sonnet_error(p.source,record.line,
        "zero/ground SMD terminals require an explicit return-terminal adapter")
    interface=length(p.layers)-1-native_level
    lv=findfirst(sh->sh.interface==interface,problem.sheets)
    lv===nothing && _sonnet_error(p.source,record.line,"SMD pin lacks its sheet level")
    xaxis=orient in ("L","R");positive=orient in ("R","B")
    axial=xaxis ? x : y;transverse=xaxis ? y : x
    spacing=xaxis ? problem.grid.dx : problem.grid.dy
    edge=round(Int,axial/spacing)
    abs(edge*spacing-axial)<=1e-7*spacing || _sonnet_error(p.source,record.line,
        "SMD edge is not aligned to this grid; retain or refine the native raster")
    tol=1e-9*max(problem.grid.a,problem.grid.b)
    candidates=Tuple{Float64,Float64}[]
    for poly in p.polygons
        poly.kind===:sheet && poly.level==native_level || continue
        v=poly.vertices;n=size(v,2)
        for k in 1:n
            next=k==n ? 1 : k+1
            a1,a2=xaxis ? (v[1,k],v[1,next]) : (v[2,k],v[2,next])
            b1,b2=xaxis ? (v[2,k],v[2,next]) : (v[1,k],v[1,next])
            abs(a1-axial)<=tol && abs(a2-axial)<=tol || continue
            lo,hi=minmax(b1,b2)
            hi-lo>tol && lo-tol<=transverse<=hi+tol || continue
            push!(candidates,(lo,hi))
        end
    end
    unique!(candidates)
    length(candidates)==1 || _sonnet_error(p.source,record.line,
        "SMD pin must identify one straight polygon edge; found $(length(candidates))")
    lo,hi=only(candidates)
    h=xaxis ? problem.grid.dy : problem.grid.dx
    first_cell,last_cell=_sonnet_termw_cells(p,record,termw,lo,hi,transverse,h,variables,freq)
    first_cell<=last_cell || _sonnet_error(p.source,record.line,"SMD pad vanished on raster")
    terminal=PlanarPort(lv,xaxis ? :terminal_x : :terminal_y,edge,first_cell:last_cell,50;
        metal_side=positive ? :positive : :negative,polarity=number<0 ? -1 : 1)
    return (;terminal,number=abs(number),pin,record)
end

"""Lower native IDEAL R/L/C and NONE components to physical cover-ground
reference vias and a circuit template. A component-level `GNDREF BOX
...` selects this via mode and each pin's own `EXSMDP` record resolves
its cover (`AUTO` picks the nearest conductive one; a forced cover must
be conductive, and pins resolving to different covers reject like
native). `GNDREF FLOAT` components, and `AUTO` components whose model
needs no ground node (IDEAL R/L/C, NONE), route through
[`sonnet_floating_model`](@ref) and share one floating local ground.
FEED pad widths are inferred from the complete straight polygon edge;
CELL selects the pin coordinate's single cell and CUST the declared
width centered on it, clamped inside the edge. Unsupported ground
references, unknown terminal-width modes, misaligned/ambiguous pads fail
explicitly.
SPARAM/SMDFILES and literal linear CKT SPROJ models with explicit
ground/width metadata are staged automatically through
[`sonnet_component_files`](@ref); under `AUTO` they keep via returns
since their model consumes a real ground node. Other
vendor/geometry-project/subcircuit models require
`component_response(project,component_records,f_hz)` returning a named
tuple `(response=matrix,format=:s,z0=...)`; no model is silently omitted.
The returned model retains all pin labels and native material arrays;
cover-ground models expose retained numeric storage as `payload`. `max_bytes` also bounds
aggregate geometry, staged models and construction workspace before model
callbacks. Native `table1/table2` expressions stage owned CSV snapshots;
`scalar_root`, `scalar_outside` and the `scalar_max_*` limits control intake.
The model retains `scalar_files` and numeric `variables`, so later source
edits cannot change its evaluated geometry, loads or reference basis."""
function sonnet_component_model(p::SonnetProject,freq::Real;grid=nothing,
        variables=Dict{String,Float64}(),component_response=nothing,
        ground_direction::Symbol=:auto,
        max_bytes::Integer=_default_max_dense_payload_bytes(),
        scalar_files=nothing,scalar_root=dirname(p.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384,
        _model_files::Union{Nothing,SonnetComponentFiles}=nothing,_preserve_scalar_geometry::Bool=false)
    max_bytes=_spice_limit("max_bytes",max_bytes)
    freq=_circuit_stored_real(freq,"native component frequency")
    isfinite(freq) && freq>0 || throw(ArgumentError("native component frequency must be representable and positive"))
    scalar_files===nothing && _model_files!==nothing && (scalar_files=_model_files.scalar_files)
    variables=_sonnet_scalar_variables(p,variables;scalar_files,max_bytes,root=scalar_root,
        outside=scalar_outside,max_files=scalar_max_files,max_bytes_source=scalar_max_bytes,
        max_nodes=scalar_max_nodes,max_line_bytes=scalar_max_line_bytes)
    scalar_files=_sonnet_scalar_files(variables)
    _preserve_scalar_geometry || (p=_sonnet_scalar_project(p,variables))
    owned_project=p
    # Validate the stored physical load before geometry or optional model callbacks.
    _sonnet_component_ideal_preflight(p,variables,freq)
    variable_payload=_sonnet_component_variable_payload(variables)
    staged_payload=_model_files===nothing ? 0 : _model_files.payload
    _enforce_payload_limit(_checked_payload_sum("all native component geometry preflight",
        staged_payload,variable_payload,_sonnet_files_geometry_payload(p,grid)),max_bytes,
        "all native component geometry preflight","max_bytes")
    if _model_files===nothing && component_response===nothing &&
            any(c->_sonnet_files_kind(p,c).tokens[2] in ("SPARAM","SPROJ"),p.components)
        _model_files=sonnet_component_files(p,freq;grid,variables,max_bytes,_preserve_scalar_geometry)
    end
    if _model_files!==nothing
        _model_files.frequency==freq || throw(ArgumentError("staged native component files belong to another frequency"))
        component_response===nothing || throw(ArgumentError("provide staged files or component_response"))
        p=_preserve_scalar_geometry ? _model_files.project : _sonnet_scalar_project(_model_files.project,variables)
        variables==_model_files.variables && grid==_model_files.grid ||
            throw(ArgumentError("staged native files retain their grid and variable configuration"))
        _enforce_payload_limit(_checked_payload_sum("native component staged geometry",
            _model_files.payload,_sonnet_files_geometry_payload(p,grid)),max_bytes,
            "native component staged geometry","max_bytes")
    end
    _sonnet_has_floating(p) && return sonnet_floating_model(p,freq;
        grid,variables,component_response,ground_direction,max_bytes,_model_files)
    !isempty(p.components) || throw(ArgumentError("native project has no components"))
    # Expand physical thick faces before separating geometry and loads,
    # so SMD terminal levels follow the same remapped stack as polygons.
    file_payload=_model_files===nothing ? 0 : _model_files.payload
    p=_sonnet_thick_geometry(p,freq,variables;max_bytes=max_bytes-file_payload-variable_payload)
    # Internal geometric lowering preserves every polygon/material/record;
    # component physics is supplied below, never by a public discard flag.
    geometry=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,
        p.metals,p.top,p.bottom,p.polygons,p.ports,p.variables,Vector{SonnetRecord}[],p.sweeps,p.records)
    base=sonnet_planar_problem(geometry;freq=freq,grid=grid,variables=variables,
        _materials=true,_details=true,max_bytes=max_bytes-file_payload-variable_payload)
    pins=Vector{Any}[]
    labels=copy(base.labels);refs=copy(base.z0)
    for component in p.components
        for record in component
            t=record.tokens
            t[1]=="GNDREF" && _sonnet_gndref_value(p.source,record)
            t[1] in ("DRP1","DRP2","REFPLANE","TWTYPE","TWVALUE") &&
                _sonnet_error(p.source,record.line,
                    "native component $(t[1]) requires its reference-plane/terminal adapter")
            t[1] in ("SMD","ID","GNDREF","TERMW","SMDP","EXSMDP","END","TYPE",
                "SBOX","PBSHW","PBOX","PKG","LPOS","INHSWP","CUPGRP") ||
                tryparse(Float64,t[1])!==nothing || _sonnet_error(p.source,record.line,
                    "unrepresented native component record $(t[1])")
        end
        termw=_sonnet_component_termw(p,component)
        gndref=_sonnet_component_gndref(p,component)
        local_pins=Any[_sonnet_component_pin(p,r,base.problem,variables,freq;
                termw=get(termw.widths,something(tryparse(Int,get(r.tokens,7,"")),0),
                    termw.default))
            for r in component if r.tokens[1]=="SMDP"]
        !isempty(local_pins) || throw(ArgumentError("native component has no terminal pins"))
        sort!(local_pins;by=pin->pin.pin)
        [pin.pin for pin in local_pins]==collect(1:length(local_pins)) ||
            throw(ArgumentError("component pin numbers must be unique and contiguous from one"))
        for pin in local_pins
            if !(pin.number in labels)
                push!(labels,pin.number);push!(refs,50.)
            end
        end
        # Every component reaching this point resolves to the via-return
        # mode (floating-resolution components are routed to the floating
        # adapter upstream). Each pin takes its own EXSMDP direction or
        # the caller's default (native: nearest conductive cover).
        push!(pins,[merge(pin,(;direction=get(gndref.dirs,pin.pin,
            ground_direction))) for pin in local_pins])
    end
    terminals=PlanarPort[pin.terminal for local_pins in pins for pin in local_pins]
    directions=Symbol[pin.direction for local_pins in pins for pin in local_pins]
    live_prefix=_checked_payload_sum("native component live prefix",file_payload,
        variable_payload,_sonnet_component_base_payload(base))
    post_workspace=_sonnet_component_post_payload(base,pins,labels)
    _enforce_payload_limit(_checked_payload_sum("native component provider preflight",
        live_prefix,post_workspace),max_bytes,"native component provider preflight","max_bytes")
    physical=planar_terminal_returns(base.problem,terminals;
        ground_direction=directions,max_bytes=max_bytes-live_prefix-post_workspace)
    # Native rejects a via-mode component whose pins resolve to different
    # covers ("mixed combination of upper/lower ground references").
    pidx=0
    for (c,local_pins) in zip(p.components,pins)
        resolved=unique(path.direction for path in physical.paths[pidx+1:pidx+length(local_pins)])
        pidx+=length(local_pins)
        length(resolved)<=1 || throw(ArgumentError(
            "native component $(_sonnet_component_name(c)) has mixed combination of upper and lower ground references"))
    end
    embedding=zeros(Float64,length(base.problem.ports)+length(terminals),length(labels))
    embedding[1:length(base.problem.ports),1:length(base.labels)].=base.contraction
    raw_index=length(base.problem.ports)
    for local_pins in pins,pin in local_pins
        raw_index+=1;embedding[raw_index,findfirst(==(pin.number),labels)]=1
    end
    contraction=physical.contraction*embedding
    common_embedding=zeros(Float64,size(embedding,1),size(base.floating_common,2))
    common_embedding[1:length(base.problem.ports),:].=base.floating_common
    floating_common=physical.contraction*common_embedding
    external=collect(1:length(base.labels))
    circuit=PlanarCircuit(length(labels),external;z0=base.z0)
    for (component,local_pins) in zip(p.components,pins)
        kind=findfirst(r->r.tokens[1]=="TYPE",component)
        kind===nothing && throw(ArgumentError("native component lacks TYPE"))
        record=component[kind];t=record.tokens
        terminals=[findfirst(==(pin.number),labels) for pin in local_pins]
        if length(t)>=2 && t[2]=="IDEAL"
            length(t)==4 && length(terminals)==2 || _sonnet_error(p.source,record.line,
                "ideal SMD requires two pins and one R/L/C value")
            scale=_sonnet_component_scale(p,t[3])
            value=_sonnet_component_scalar(p,t[4],scale,variables,freq)
            a,b=terminals
            t[3]=="RES" ? circuit_add_rlc!(circuit,a,b;r=value) :
                t[3]=="CAP" ? circuit_add_rlc!(circuit,a,b;c=value) :
                t[3]=="IND" ? circuit_add_rlc!(circuit,a,b;l=value) :
                _sonnet_error(p.source,record.line,"unsupported ideal SMD $(t[3])")
        elseif length(t)==2 && t[2]=="NONE"
            # Explicit no-load model: retain its open EM terminals.
        elseif t[2] in ("SPARAM","SPROJ") && _model_files!==nothing
            id=_sonnet_files_id(p,component)
            matches=filter(b->b.id==id,_model_files.bindings)
            length(matches)==1 || _sonnet_error(p.source,record.line,"missing unique staged component binding")
            binding=only(matches)
            binding.geometry_labels==[pin.number for pin in local_pins] ||
                _sonnet_error(p.source,record.line,"staged model-pin order disagrees with physical geometry labels")
            circuit_add_network!(circuit,terminals,binding.response;format=:s,z0=binding.z0)
        else
            component_response===nothing && _sonnet_error(p.source,record.line,
                "vendor/project SMD requires component_response; its model cannot be omitted")
            supplied=component_response(p,component,freq)
            supplied isa NamedTuple && all(k->haskey(supplied,k),(:response,:format,:z0)) ||
                throw(ArgumentError("component_response must return response,format,z0"))
            circuit_add_network!(circuit,terminals,supplied.response;
                format=supplied.format,z0=supplied.z0)
        end
    end
    model=(;project=owned_project,problem=physical.problem,contraction,floating_common,z0=refs,labels,external_labels=base.labels,
        circuit,sheet_zs=base.sheet_zs,
        via_sigma=vcat(base.via_sigma,fill(Inf,length(physical.problem.vias)-length(base.problem.vias))),
        pins,reference_paths=physical.paths,model_files=_model_files,scalar_files,
        variables=variables isa SonnetScalarVariables ? variables.overrides : variables)
    payload=_sonnet_component_retained_payload(model)
    _enforce_payload_limit(payload,max_bytes,"native retained component model","max_bytes")
    scalar_files===nothing || _sonnet_check_scalar_files(scalar_files)
    return merge(model,(;payload))
end

function _solve_sonnet_components(p::SonnetProject,freq::Real;grid=nothing,
        variables=Dict{String,Float64}(),calibration=nothing,component_response=nothing,
        ground_direction::Symbol=:auto,pin_calibration=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes(),
        scalar_files=nothing,scalar_root=dirname(p.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384,
        _model_files::Union{Nothing,SonnetComponentFiles}=nothing,kw...)
    scalar_options=(;scalar_files,scalar_root,scalar_outside,scalar_max_files,scalar_max_bytes,
        scalar_max_nodes,scalar_max_line_bytes)
    _sonnet_has_floating(p) && return _solve_sonnet_floating(p,freq;
        grid,variables,calibration,component_response,ground_direction,pin_calibration,max_bytes,scalar_options...,_model_files,kw...)
    model=sonnet_component_model(p,freq;grid=grid,variables=variables,
        component_response=component_response,ground_direction=ground_direction,max_bytes=max_bytes,
        scalar_options...,_model_files=_model_files)
    :surface_zs in keys(kw) && throw(ArgumentError("native components retain project material loss"))
    :via_sigma in keys(kw) && throw(ArgumentError("native components retain project axial resistance"))
    # The EM result stays live while the loaded circuit is solved. Reserve
    # contraction/material arrays and network/calibration workspace from
    # the forward budget, then reserve the retained factor/operator from MNA.
    model_payload=model.payload
    response_payload=_checked_payload_sum("native component response workspace",
        _checked_array_payload_bytes(ComplexF64,12,length(model.labels),length(model.labels)),
        _checked_array_payload_bytes(ComplexF64,3,length(model.problem.ports),length(model.labels)))
    _enforce_payload_limit(model_payload+response_payload,max_bytes,"native components","max_bytes")
    raw=solve_planar(model.problem,freq;surface_zs=model.sheet_zs,via_sigma=model.via_sigma,
        max_bytes=max_bytes-model_payload-response_payload,kw...)
    balanced=_sonnet_balanced_y(raw.y,model.contraction,model.floating_common)
    Y=balanced.y
    gap_transfer=Matrix{ComplexF64}(I,length(model.labels),length(model.labels))
    if pin_calibration!==nothing
        pin_labels=unique([pin.number for local_pins in model.pins for pin in local_pins])
        indices=Int[findfirst(==(label),model.labels) for label in pin_labels]
        supplied=pin_calibration isa AbstractMatrix ? pin_calibration : pin_calibration(freq)
        k=length(indices)
        Y=deembed_cocal_group(Y,indices,supplied)
        A=Matrix{ComplexF64}(I,length(model.labels),length(model.labels))
        B=zeros(ComplexF64,length(model.labels),length(model.labels))
        A[indices,indices]=supplied[1:k,1:k];B[indices,indices]=supplied[1:k,k+1:2k]
        gap_transfer=A+B*Y
    end
    circuit_add_network!(model.circuit,collect(eachindex(model.labels)),Y;format=:y)
    circuit=solve_planar_circuit(model.circuit,freq;
        max_bytes=max_bytes-model_payload-response_payload-_subdivision_retained_payload(raw))
    circuit.y===nothing && throw(ArgumentError("native external response has no finite admittance representation"))
    y=calibration===nothing ? circuit.y : deembed_ports(circuit.y,calibration)
    s=planar_y_to_s(y,circuit.z0)
    incident_transfer=Matrix{ComplexF64}(I,size(s,1),size(s,1))
    if calibration!==nothing
        A,B,C,D=_calibration_chain_blocks(calibration,size(s,1))
        incident_transfer=_planar_calibration_incident_transfer(A,B,C,D,y,s,circuit.z0)
    end
    return SonnetComponentResult(model.project,raw,model.external_labels,y,s,balanced.voltage_transfer,circuit,
        collect(eachindex(model.labels)),incident_transfer,gap_transfer,circuit.z0,model.model_files,
        model.scalar_files,model.variables)
end

"""Solve a retained native SPARAM/SPROJ snapshot at its staged frequency. Later
source/model edits do not change this solve. Raw cover-return lead effects
remain until independently supplied coupled pin calibration is applied."""
function solve_sonnet_project(files::SonnetComponentFiles,freq::Real=files.frequency;
        variables=files.variables,grid=files.grid,scalar_files=files.scalar_files,kw...)
    Float64(freq)==files.frequency || throw(ArgumentError("staged native component files belong to another frequency"))
    return solve_sonnet_project(files.project,files.frequency;variables,grid,scalar_files,_model_files=files,kw...)
end

"""Reconstruct native SMD-loaded EM currents for external incident
power waves, including voltages solved at every loaded component pin."""
function sonnet_component_current_maps(result::SonnetComponentResult;
        port::Integer=1,incident_waves=nothing,kw...)
    n=size(result.s,1)
    a=if incident_waves===nothing
        1<=port<=n || throw(ArgumentError("native component port index is invalid"))
        v=zeros(ComplexF64,n);v[port]=1;v
    else
        incident_waves isa AbstractVector && length(incident_waves)==n && all(isfinite,incident_waves) ||
            throw(ArgumentError("native component incident waves must match external ports"))
        _planar_stored_phasor.(incident_waves)
    end
    nodes=result.circuit.voltages*(result.incident_transfer*a)
    return planar_current_maps(result.raw;voltages=result.contraction*(result.gap_transfer*nodes),kw...)
end
