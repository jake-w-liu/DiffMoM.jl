# Native SMD attachment through physical reference vias and circuit MNA.

export SonnetComponentResult, sonnet_component_model, sonnet_component_current_maps

"""Native SMD solve retaining the raw multi-terminal EM solution,
terminal contraction and composite circuit response. `s,y` are external
responses; component pins remain internal circuit nodes. Raw reference
vias provide physical cover-ground returns and retain their lead effects.
Pin-group calibration must be supplied independently before claiming
calibrated Sonnet equivalence. `model_files` retains exact automatic SPARAM
sources, evaluated references and the staged effective configuration."""
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
end
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
    isempty(variables) && return variables
    all(key->key isa AbstractString,keys(variables)) || throw(ArgumentError("native variable names must be strings"))
    needed=_checked_payload_sum("native component variable storage",512,
        _checked_array_payload_bytes(UInt8,128,length(variables)),2sum(ncodeunits,keys(variables)))
    _enforce_payload_limit(needed,max_bytes,"native component variable storage","max_bytes")
    all(value->value isa Real,values(variables)) || throw(ArgumentError("native variable overrides must be real numbers"))
    Dict(String(key)=>_circuit_stored_real(value,"native component variable override") for (key,value) in variables)
end

function _sonnet_component_pin(p,record,problem,variables,freq)
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
    first_cell=floor(Int,lo/h+1e-8)+1;last_cell=ceil(Int,hi/h-1e-8)
    first_cell<=last_cell || _sonnet_error(p.source,record.line,"SMD pad vanished on raster")
    terminal=PlanarPort(lv,xaxis ? :terminal_x : :terminal_y,edge,first_cell:last_cell,50;
        metal_side=positive ? :positive : :negative,polarity=number<0 ? -1 : 1)
    return (;terminal,number=abs(number),pin,record)
end

"""Lower native IDEAL R/L/C and NONE components to physical cover-ground
reference vias and a circuit template. Explicit signed FLOAT pairs route
through [`sonnet_floating_model`](@ref) and retain their finite references.
All pad widths are inferred from
the complete straight polygon edge. Unsupported ground references,
non-feed terminal widths, misaligned/ambiguous pads fail explicitly.
Observed SPARAM/SMDFILES models with explicit PEC AUTO+FEED pins are staged
automatically through [`sonnet_component_files`](@ref). Other
vendor/project/subcircuit models require
`component_response(project,component_records,f_hz)` returning a named
tuple `(response=matrix,format=:s,z0=...)`; no model is silently omitted.
The returned model retains all pin labels and native material arrays."""
function sonnet_component_model(p::SonnetProject,freq::Real;grid=nothing,
        variables=Dict{String,Float64}(),component_response=nothing,
        ground_direction::Symbol=:auto,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        _model_files::Union{Nothing,SonnetComponentFiles}=nothing)
    max_bytes=_spice_limit("max_bytes",max_bytes)
    freq=_circuit_stored_real(freq,"native component frequency")
    isfinite(freq) && freq>0 || throw(ArgumentError("native component frequency must be representable and positive"))
    variables=_sonnet_component_variables(variables,max_bytes)
    # Validate the stored physical load before geometry or optional model callbacks.
    for component in p.components,record in component
        t=record.tokens
        length(t)>=2 && t[1:2]==["TYPE","IDEAL"] || continue
        length(t)==4 || _sonnet_error(p.source,record.line,"ideal SMD requires one R/L/C value")
        _sonnet_component_scalar(p,t[4],_sonnet_component_scale(p,t[3]),variables,freq)
    end
    if _model_files===nothing && component_response===nothing &&
            any(c->_sonnet_files_kind(p,c).tokens[2]=="SPARAM",p.components)
        _model_files=sonnet_component_files(p,freq;grid,variables,max_bytes)
    end
    if _model_files!==nothing
        _model_files.frequency==freq || throw(ArgumentError("staged native component files belong to another frequency"))
        component_response===nothing || throw(ArgumentError("provide staged files or component_response"))
        p=_model_files.project
        variables==_model_files.variables && grid==_model_files.grid ||
            throw(ArgumentError("staged native files retain their grid and variable configuration"))
        _enforce_payload_limit(_checked_payload_sum("native component staged geometry",
            _model_files.payload,_sonnet_files_geometry_payload(p,grid)),max_bytes,
            "native component staged geometry","max_bytes")
    end
    _sonnet_has_floating(p) && return sonnet_floating_model(p,freq;
        grid,variables,component_response,ground_direction,max_bytes)
    !isempty(p.components) || throw(ArgumentError("native project has no components"))
    # Expand physical thick faces before separating geometry and loads,
    # so SMD terminal levels follow the same remapped stack as polygons.
    p=_sonnet_thick_geometry(p,freq,variables)
    # Internal geometric lowering preserves every polygon/material/record;
    # component physics is supplied below, never by a public discard flag.
    geometry=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,
        p.metals,p.top,p.bottom,p.polygons,p.ports,p.variables,Vector{SonnetRecord}[],p.sweeps,p.records)
    base=sonnet_planar_problem(geometry;freq=freq,grid=grid,variables=variables,
        _materials=true,_details=true)
    pins=Vector{Any}[]
    labels=copy(base.labels);refs=copy(base.z0)
    for component in p.components
        for record in component
            t=record.tokens
            t[1]=="GNDREF" && t[end]!="AUTO" && _sonnet_error(p.source,record.line,
                "SMD local ground reference requires its explicit return-terminal adapter")
            t[1]=="TERMW" && t[end]!="FEED" && _sonnet_error(p.source,record.line,
                "custom SMD terminal widths require their geometry adapter")
        end
        local_pins=Any[_sonnet_component_pin(p,r,base.problem,variables,freq)
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
        push!(pins,local_pins)
    end
    terminals=PlanarPort[pin.terminal for local_pins in pins for pin in local_pins]
    file_payload=_model_files===nothing ? 0 : _model_files.payload
    physical=planar_terminal_returns(base.problem,terminals;
        ground_direction=ground_direction,max_bytes=max_bytes-file_payload)
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
        elseif t[2]=="SPARAM" && _model_files!==nothing
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
    return (;problem=physical.problem,contraction,floating_common,z0=refs,labels,external_labels=base.labels,
        circuit,sheet_zs=base.sheet_zs,
        via_sigma=vcat(base.via_sigma,fill(Inf,length(physical.problem.vias)-length(base.problem.vias))),
        pins,reference_paths=physical.paths,model_files=_model_files)
end

function _solve_sonnet_components(p::SonnetProject,freq::Real;grid=nothing,
        variables=Dict{String,Float64}(),calibration=nothing,component_response=nothing,
        ground_direction::Symbol=:auto,pin_calibration=nothing,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        _model_files::Union{Nothing,SonnetComponentFiles}=nothing,kw...)
    _sonnet_has_floating(p) && return _solve_sonnet_floating(p,freq;
        grid,variables,calibration,component_response,ground_direction,pin_calibration,max_bytes,kw...)
    model=sonnet_component_model(p,freq;grid=grid,variables=variables,
        component_response=component_response,ground_direction=ground_direction,max_bytes=max_bytes,
        _model_files=_model_files)
    :surface_zs in keys(kw) && throw(ArgumentError("native components retain project material loss"))
    :via_sigma in keys(kw) && throw(ArgumentError("native components retain project axial resistance"))
    # The EM result stays live while the loaded circuit is solved. Reserve
    # contraction/material arrays and network/calibration workspace from
    # the forward budget, then reserve the retained factor/operator from MNA.
    model_payload=_checked_payload_sum("native component model",
        model.model_files===nothing ? 0 : model.model_files.payload,
        _spice_circuit_payload(model.circuit),
        _checked_array_payload_bytes(Float64,size(model.contraction)...),
        _checked_array_payload_bytes(Float64,size(model.floating_common)...),
        _subdivision_retained_payload(model.sheet_zs),
        _subdivision_retained_payload(model.via_sigma),
        _subdivision_retained_payload(model.problem.basis),
        _subdivision_retained_payload(model.problem.sheets),
        _subdivision_retained_payload(model.problem.vias),
        _subdivision_retained_payload(model.problem.vols))
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
    return SonnetComponentResult(p,raw,model.external_labels,y,s,balanced.voltage_transfer,circuit,
        collect(eachindex(model.labels)),incident_transfer,gap_transfer,circuit.z0,model.model_files)
end

"""Solve a retained native SPARAM snapshot at its staged frequency. Later
source/model edits do not change this solve. Raw cover-return lead effects
remain until independently supplied coupled pin calibration is applied."""
function solve_sonnet_project(files::SonnetComponentFiles,freq::Real=files.frequency;
        variables=files.variables,grid=files.grid,kw...)
    Float64(freq)==files.frequency || throw(ArgumentError("staged native component files belong to another frequency"))
    return solve_sonnet_project(files.project,files.frequency;variables,grid,_model_files=files,kw...)
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
        ComplexF64.(incident_waves)
    end
    nodes=result.circuit.voltages*(result.incident_transfer*a)
    return planar_current_maps(result.raw;voltages=result.contraction*(result.gap_transfer*nodes),kw...)
end
