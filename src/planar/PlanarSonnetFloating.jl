export SonnetFloatingResult,sonnet_floating_model

_sonnet_float_records(records)=any(r->r.tokens[1]=="GNDREF" &&
    r.tokens[end] in ("FLOAT","F"),records)
_sonnet_has_floating(p::SonnetProject)=any(port->_sonnet_float_records(port.records),p.ports) ||
    any(_sonnet_float_records,p.components)

"""Native floating-port network with explicit finite source bridges and
its loaded MNA response. `em` retains the physical source solve;
`source_incidence` maps its source pairs to actual circuit nodes.
`bridges` records all added geometry and calibration cuts. Raw bridge
delay, loss and coupling remain; this result does not claim the native
GLG calibration or its unavailable Lite-license comparison.
`scalar_files,scalar_overrides` retain owned native CSV dependencies and
the numeric overrides used for the physical bridges and loads. `model_files`
retains automatic SPARAM dependencies of ordinary cover-referenced components
that coexist with the floating sources."""
struct SonnetFloatingResult{R}
    project::SonnetProject
    em::R
    circuit::PlanarCircuitResult
    port_numbers::Vector{Int}
    z0::Vector{ComplexF64}
    y::Union{Nothing,Matrix{ComplexF64}}
    s::Matrix{ComplexF64}
    source_incidence::Matrix{Float64}
    bridges::Vector{Any}
    incident_transfer::Matrix{ComplexF64}
    scalar_files::Union{Nothing,SonnetScalarFiles}
    scalar_overrides::Dict{String,Float64}
    model_files::Union{Nothing,SonnetComponentFiles}
end
SonnetFloatingResult(p,em,c,n,z,y,s,d,b,t,files,overrides)=
    SonnetFloatingResult(p,em,c,n,z,y,s,d,b,t,files,overrides,nothing)
SonnetFloatingResult(p,em,c,n,z,y,s,d,b,t)=
    SonnetFloatingResult(p,em,c,n,z,y,s,d,b,t,nothing,Dict{String,Float64}())

struct _SonnetBalancedSourceResult{R}
    raw::R
    problem::PlanarProblem
    freq::ComplexF64
    omega::ComplexF64
    z0::Vector{ComplexF64}
    currents::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    voltage_transfer::Matrix{ComplexF64}
end
_planar_coefficient_columns(result::_SonnetBalancedSourceResult)=result.currents

function planar_current_maps(result::_SonnetBalancedSourceResult;port::Integer=1,
        voltages=nothing,incident_waves=nothing,z_fraction::Real=.5,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    n=length(result.z0)
    voltages===nothing || incident_waves===nothing || throw(ArgumentError("provide voltages or incident_waves"))
    reserve=_checked_payload_sum("balanced native current excitation",
        _checked_array_payload_bytes(ComplexF64,size(result.currents,1)),
        _checked_array_payload_bytes(ComplexF64,3,n))
    _enforce_payload_limit(reserve,max_bytes,"balanced native current excitation","max_bytes")
    supplied=voltages===nothing ? incident_waves : voltages
    v=if supplied===nothing
        1<=port<=n || throw(ArgumentError("native source port is invalid"))
        vector=zeros(ComplexF64,n);vector[port]=1;vector
    else
        supplied isa AbstractVector && length(supplied)==n && all(isfinite,supplied) ||
            throw(ArgumentError("native source excitation must be finite and match ports"))
        _planar_stored_phasor.(supplied)
    end
    incident_waves===nothing || (v=_planar_wave_voltage(result.s,v,result.z0))
    return _planar_current_maps_from_coefficients(result.problem,result.currents*v;
        z_fraction,max_bytes=max_bytes-reserve)
end

function _sonnet_float_validate(p,records)
    for record in records
        t=record.tokens;key=first(t)
        if key=="GNDREF"
            length(t)==2 && t[2] in ("FLOAT","F") || _sonnet_error(p.source,record.line,
                "a floating group must use a consistent FLOAT reference")
        elseif key=="TERMW"
            length(t)==2 && t[2]=="FEED" || _sonnet_error(p.source,record.line,
                "native floating source requires FEED width; CELL/custom widths need explicit resolved terminal geometry")
        elseif key in ("DRP1","DRP2","TWVALUE","TWTYPE","PKG")
            _sonnet_error(p.source,record.line,"native floating $key requires its physical calibration/terminal adapter")
        elseif !(key in ("POR1","POLY","SMD","ID","SBOX","PBSHW","LPOS","PBOX",
                "TYPE","SMDP","EXSMDP","END") || tryparse(Float64,key)!==nothing)
            _sonnet_error(p.source,record.line,"unrepresented native floating field $key")
        end
    end
end

function _sonnet_float_cup_pin(p,port,problem,variables,freq)
    _sonnet_float_validate(p,port.records)
    port.kind===:cup || throw(ArgumentError("FLOAT source requires a native CUP or component contract"))
    length(port.values) in (7,9) || throw(ArgumentError("floating CUP has unsupported termination/reference fields"))
    val(s)=sonnet_variable_value(p,s;variables,freq)
    reference=_sonnet_port_reference(p,port,freq,variables)
    if length(port.values)==9
        port.values[8]=="NONE" && iszero(val(port.values[9])) || throw(ArgumentError(
            "native FLOAT ports cannot silently apply reference planes"))
    end
    poly=only(poly for poly in p.polygons if poly.id==port.polygon)
    poly.kind===:sheet || throw(ArgumentError("floating CUP must attach to sheet metal"))
    v=poly.vertices;n=size(v,2);k=port.edge+1
    1<=k<=n || throw(ArgumentError("floating CUP polygon edge is invalid"))
    q=k==n ? 1 : k+1;dx,dy=v[1,q]-v[1,k],v[2,q]-v[2,k]
    iszero(dx)!=iszero(dy) || throw(ArgumentError("floating CUP requires an axial straight edge"))
    area=sum(v[1,j]*v[2,j==n ? 1 : j+1]-v[1,j==n ? 1 : j+1]*v[2,j] for j in 1:n)
    !iszero(area) || throw(ArgumentError("floating CUP polygon has zero area"))
    orient=iszero(dx) ? (-sign(area)*sign(dy)>0 ? "R" : "L") :
        (sign(area)*sign(dx)>0 ? "B" : "T")
    record=SonnetRecord(first(port.records).line,["SMDP",string(poly.level),
        port.values[6],port.values[7],orient,string(port.number),"1"])
    pin=_sonnet_component_pin(p,record,problem,variables,freq)
    return merge(pin,(signed_number=port.number,z0=reference,group=first(port.records).tokens[end]))
end

function _sonnet_float_endpoint(pin)
    p=pin.terminal
    return PlanarPort(p.level,p.wall,p.cells,p.z0;edge=p.edge,polarity=1,refplane=p.refplane)
end

"""Lower native FLOAT CUP and component ports with explicit signed
positive/negative pad pairs to finite impressed-field bridges. Every pair
has one physical differential voltage and equal/opposite source current;
its reference is never clamped to a cover. Facing pads must share sheet,
axis and complete FEED width. Duplicate/unmatched signed labels and
unrepresented CELL/custom terminal widths reject.

NONE components expose their paired ports; ideal R/L/C and explicit
`component_response(project,records,freq)` blocks load those pairs through
MNA. Ordinary cover-referenced ports/components can coexist, preserving
their actual source modes and material losses. `bridge_zs` explicitly sets
the added fixture's sheet impedance (default PEC). All bridge geometry
and coupling remain in the raw solve. Native GLG effects require supplied
joint launch calibration; native floating accuracy remains unverified
because the installed Lite engine rejects these features."""
function sonnet_floating_model(p::SonnetProject,freq::Real;grid=nothing,
        variables=Dict{String,Float64}(),component_response=nothing,
        ground_direction::Symbol=:auto,bridge_zs::Number=0.,
        max_bytes::Integer=_default_max_dense_payload_bytes(),scalar_files=nothing,
        scalar_root=dirname(p.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384)
    max_bytes=_spice_limit("max_bytes",max_bytes)
    freq=_circuit_stored_real(freq,"native floating frequency")
    freq>0 || throw(ArgumentError("native floating frequency must be representable and positive"))
    variables=_sonnet_scalar_variables(p,variables;scalar_files,max_bytes,root=scalar_root,
        outside=scalar_outside,max_files=scalar_max_files,max_bytes_source=scalar_max_bytes,
        max_nodes=scalar_max_nodes,max_line_bytes=scalar_max_line_bytes)
    scalar_files=_sonnet_scalar_files(variables)
    p=_sonnet_scalar_project(p,variables)
    owned_project=p
    _sonnet_component_ideal_preflight(p,variables,freq)
    # The wrapper can retain one normalized override dictionary while this
    # entry owns another; reserve both before geometry or callbacks.
    variable_payload=_checked_payload_sum("native floating owned variables",
        2BigInt(_sonnet_component_variable_payload(variables)))
    _enforce_payload_limit(variable_payload,max_bytes,"native floating owned variables","max_bytes")
    _sonnet_has_floating(p) || throw(ArgumentError("project has no native FLOAT sources"))
    isfinite(bridge_zs) && real(bridge_zs)>=0 || throw(ArgumentError("floating bridge impedance must be finite and passive"))
    stored_bridge=ComplexF64(bridge_zs)
    isfinite(stored_bridge) || throw(ArgumentError("floating bridge impedance must fit finite ComplexF64"))
    p=_sonnet_thick_geometry(p,freq,variables;max_bytes=max_bytes-variable_payload)
    physical=_sonnet_stack_geometry(p,freq,grid,variables;expand_thick=false,
        max_bytes=max_bytes-variable_payload)
    cellcount=BigInt(physical.grid.nx)*physical.grid.ny
    sheets=length(unique(poly.level for poly in p.polygons if poly.kind===:sheet))
    vias=count(poly->poly.kind===:via,p.polygons)*BigInt(length(p.layers))
    sourcecount=length(p.ports)+sum(count(r->r.tokens[1]=="SMDP",c) for c in p.components;init=0)
    _enforce_payload_limit(_checked_payload_sum("native floating geometry preflight",variable_payload,
        _checked_array_payload_bytes(UInt8,228,cellcount,sheets+vias),
        _checked_array_payload_bytes(ComplexF64,2,cellcount,sheets),
        _checked_array_payload_bytes(Float64,2BigInt(length(p.layers))*sourcecount,sourcecount)),
        max_bytes,"native floating geometry preflight","max_bytes")
    floating_ports=[port for port in p.ports if _sonnet_float_records(port.records)]
    ordinary_ports=[port for port in p.ports if !_sonnet_float_records(port.records)]
    floating_components=[c for c in p.components if _sonnet_float_records(c)]
    ordinary_components=[c for c in p.components if !_sonnet_float_records(c)]
    geometry=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,
        p.metals,p.top,p.bottom,p.polygons,ordinary_ports,p.variables,ordinary_components,p.sweeps,p.records)
    if isempty(ordinary_components)
        base=sonnet_planar_problem(geometry;freq,grid,variables,_materials=true,_details=true,
            _allow_portless=true,max_bytes=max_bytes-variable_payload)
        problem=base.problem;C=copy(base.contraction);refs=copy(base.z0);labels=copy(base.labels)
        common=hasproperty(base,:floating_common) ? copy(base.floating_common) : zeros(length(problem.ports),0)
        nodes=length(labels);pairs=[(i,0) for i in 1:nodes]
        # MNA construction occurs after floating sources provide nodes and
        # any exposed ports when the ordinary geometry has no source.
        circuit=nothing;external_pairs=copy(pairs);external_labels=copy(labels);external_z0=copy(refs)
        sheet_zs=[copy(zs) for zs in base.sheet_zs];via_sigma=copy(base.via_sigma);paths=Any[]
    else
        base=sonnet_component_model(geometry,freq;grid,variables,component_response,ground_direction,
            max_bytes=max_bytes-variable_payload,
            _preserve_scalar_geometry=true)
        problem=base.problem;C=copy(base.contraction);refs=copy(base.z0);labels=copy(base.labels)
        common=hasproperty(base,:floating_common) ? copy(base.floating_common) : zeros(length(problem.ports),0)
        circuit=base.circuit;nodes=circuit.nnodes;pairs=[(i,0) for i in 1:nodes]
        external_pairs=copy(circuit.ports);external_labels=copy(base.external_labels);external_z0=copy(circuit.z0)
        sheet_zs=[copy(zs) for zs in base.sheet_zs];via_sigma=copy(base.via_sigma);paths=copy(base.reference_paths)
    end
    model_files=hasproperty(base,:model_files) ? base.model_files : nothing
    file_payload=model_files===nothing ? 0 : model_files.payload
    groups=Any[]
    if !isempty(floating_ports)
        pins=[_sonnet_float_cup_pin(p,port,problem,variables,freq) for port in floating_ports]
        for label in sort(unique(pin.number for pin in pins))
            selected=[pin for pin in pins if pin.number==label]
            length(selected)==2 && sort([pin.signed_number for pin in selected])==[-label,label] ||
                throw(ArgumentError("native floating CUP requires exactly one positive and one negative pad per label"))
            selected[1].group==selected[2].group && selected[1].z0==selected[2].z0 ||
                throw(ArgumentError("paired FLOAT CUPs must share calibration group and reference impedance"))
            push!(groups,(pins=selected,component=nothing,external=true))
        end
    end
    for component in floating_components
        _sonnet_float_validate(p,component)
        pins=Any[]
        for record in component
            record.tokens[1]=="SMDP" || continue
            pin=_sonnet_component_pin(p,record,problem,variables,freq)
            push!(pins,merge(pin,(signed_number=parse(Int,record.tokens[6]),z0=50.)))
        end
        sort!(pins;by=pin->pin.pin)
        [pin.pin for pin in pins]==collect(1:length(pins)) && !isempty(pins) ||
            throw(ArgumentError("native FLOAT component pins must be unique and contiguous"))
        kind=only(r for r in component if first(r.tokens)=="TYPE")
        none=kind.tokens==["TYPE","NONE"]
        push!(groups,(pins=pins,component=component,external=none))
    end
    bridges=Any[];loads=Any[]
    for group in groups
        local_pairs=Tuple{Int,Int}[]
        for label in sort(unique(pin.number for pin in group.pins))
            selected=[pin for pin in group.pins if pin.number==label]
            length(selected)==2 && sort([pin.signed_number for pin in selected])==[-label,label] ||
                throw(ArgumentError("native FLOAT component requires explicit positive/negative pairs for every source label"))
            label in labels && throw(ArgumentError("native FLOAT label collides with an ordinary or previously defined source mode"))
            positive=only(pin for pin in selected if pin.signed_number>0)
            negative=only(pin for pin in selected if pin.signed_number<0)
            payload=_subdivision_retained_payload(Any[problem,C,common,sheet_zs,via_sigma])
            reserve=_checked_payload_sum("native floating source model",payload,variable_payload,file_payload,
                _checked_array_payload_bytes(Float64,length(problem.ports)+1,length(labels)+1),
                _checked_array_payload_bytes(Float64,length(problem.ports)+1,size(common,2)),
                _checked_array_payload_bytes(ComplexF64,length(problem.sheets),problem.grid.nx,problem.grid.ny))
            _enforce_payload_limit(reserve,max_bytes,"native floating source model","max_bytes")
            built=planar_floating_bridge(problem,_sonnet_float_endpoint(positive),_sonnet_float_endpoint(negative);
                z0=positive.z0,max_bytes=max_bytes-reserve)
            all(!iszero,built.old_port_map) || throw(ArgumentError("native floating endpoint overlaps an existing mathematical source"))
            next=zeros(Float64,length(built.problem.ports),length(labels)+1)
            next[built.old_port_map,1:length(labels)].=C;next[built.port_index,end]=1
            nextcommon=zeros(Float64,length(built.problem.ports),size(common,2))
            nextcommon[built.old_port_map,:].=common
            C=next;common=nextcommon;problem=built.problem
            for (i,j) in built.bridge.cells;sheet_zs[built.bridge.sheet][i,j]=stored_bridge;end
            nodes+=2;pair=(nodes-1,nodes);push!(pairs,pair);push!(local_pairs,pair)
            push!(labels,label);push!(refs,positive.z0)
            push!(bridges,merge(built.bridge,(port=length(labels),label=label,kind=:floating_bridge)))
            if group.external
                push!(external_pairs,pair);push!(external_labels,label);push!(external_z0,positive.z0)
            end
        end
        group.component===nothing || push!(loads,(component=group.component,pairs=local_pairs))
    end
    !isempty(external_labels) || throw(ArgumentError("native floating project has no exposed ports; use NONE or explicit CUP source labels"))
    order=sortperm(external_labels)
    if circuit===nothing
        circuit=PlanarCircuit(nodes,external_pairs[order];z0=external_z0[order])
    else
        circuit.nnodes=nodes;circuit.ports=external_pairs[order];circuit.z0=external_z0[order]
    end
    for load in loads
        component=load.component;record=only(r for r in component if first(r.tokens)=="TYPE");t=record.tokens
        t==["TYPE","NONE"] && continue
        if length(t)==4 && t[2]=="IDEAL"
            length(load.pairs)==1 || throw(ArgumentError("ideal FLOAT R/L/C requires one differential terminal pair"))
            a,b=only(load.pairs)
            value=_sonnet_component_scalar(p,t[4],_sonnet_component_scale(p,t[3]),variables,freq)
            t[3]=="RES" ? circuit_add_rlc!(circuit,a,b;r=value) :
                t[3]=="CAP" ? circuit_add_rlc!(circuit,a,b;c=value) :
                t[3]=="IND" ? circuit_add_rlc!(circuit,a,b;l=value) :
                _sonnet_error(p.source,record.line,"unsupported FLOAT ideal component")
        else
            component_response===nothing && throw(ArgumentError("native FLOAT vendor/subcircuit needs component_response"))
            supplied=component_response(p,component,freq)
            supplied isa NamedTuple && all(k->haskey(supplied,k),(:response,:format,:z0)) ||
                throw(ArgumentError("component_response must return response,format,z0"))
            circuit_add_network!(circuit,load.pairs,supplied.response;format=supplied.format,z0=supplied.z0)
        end
    end
    D=planar_port_incidence(nodes,pairs;max_bytes)
    scalar_files===nothing || _sonnet_check_scalar_files(scalar_files)
    return (;project=owned_project,problem,contraction=C,floating_common=common,z0=refs,labels,circuit,sheet_zs,via_sigma,
        source_incidence=D,source_terminals=pairs,bridges,reference_paths=paths,external_labels=external_labels[order],
        scalar_files,model_files,variables=variables isa SonnetScalarVariables ? variables.overrides : variables)
end

function _solve_sonnet_floating(p::SonnetProject,freq::Real;grid=nothing,
        variables=Dict{String,Float64}(),calibration=nothing,component_response=nothing,
        ground_direction::Symbol=:auto,bridge_zs::Number=0.,pin_calibration=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes(),scalar_files=nothing,
        scalar_root=dirname(p.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384,kw...)
    model=sonnet_floating_model(p,freq;grid,variables,component_response,ground_direction,bridge_zs,max_bytes,
        scalar_files,scalar_root,scalar_outside,scalar_max_files,scalar_max_bytes,scalar_max_nodes,scalar_max_line_bytes)
    :surface_zs in keys(kw) && throw(ArgumentError("native FLOAT retains project material losses"))
    :via_sigma in keys(kw) && throw(ArgumentError("native FLOAT retains axial resistance"))
    payload=_checked_payload_sum("native floating retained model",
        _subdivision_retained_payload(Any[model.problem,model.contraction,model.sheet_zs,
            model.via_sigma,model.source_incidence,model.floating_common]),
        model.scalar_files===nothing ? 0 : _sonnet_scalar_files_payload(model.scalar_files),
        model.model_files===nothing ? 0 : model.model_files.payload,
        _sonnet_component_variable_payload(model.variables))
    n=length(model.labels);nc=size(model.floating_common,2)
    workspace=_checked_payload_sum("native floating solve workspace",
        _checked_array_payload_bytes(ComplexF64,36,n+nc+model.circuit.nnodes,n+nc+model.circuit.nnodes),
        _checked_array_payload_bytes(ComplexF64,planar_basis_count(model.problem.basis),n),
        _checked_array_payload_bytes(Float64,length(model.problem.ports),n+nc))
    _enforce_payload_limit(payload+workspace,max_bytes,"native floating solve","max_bytes")
    modes=nc==0 ? model.contraction : hcat(model.contraction,model.floating_common)
    augmented=solve_planar_contracted(model.problem,freq,modes;z0=vcat(model.z0,fill(50.,nc)),
        surface_zs=model.sheet_zs,via_sigma=model.via_sigma,max_bytes=max_bytes-payload-workspace,kw...)
    em=if nc==0
        augmented
    else
        d=Matrix{Float64}(I,n+nc,n+nc)[:,1:n]
        f=Matrix{Float64}(I,n+nc,n+nc)[:,n+1:end]
        balanced=_sonnet_balanced_y(augmented.y,d,f)
        _SonnetBalancedSourceResult(augmented,model.problem,ComplexF64(freq),augmented.omega,model.z0,
            augmented.currents*balanced.voltage_transfer,balanced.y,planar_y_to_s(balanced.y,model.z0),balanced.voltage_transfer)
    end
    if pin_calibration!==nothing
        supplied=pin_calibration isa AbstractMatrix ? pin_calibration : pin_calibration(freq)
        group=[bridge.port for bridge in model.bridges]
        em=planar_cocalibrate(em,[group],[supplied];max_bytes=workspace)
    end
    circuit_add_em!(model.circuit,model.source_terminals,em)
    circuit=solve_planar_circuit(model.circuit,freq;floating_gauge=:auto,
        max_bytes=max_bytes-payload-workspace-_subdivision_retained_payload(em))
    y=calibration===nothing ? circuit.y :
        circuit.y===nothing ? throw(ArgumentError("native external calibration requires finite admittance")) :
        deembed_ports(circuit.y,calibration)
    s=calibration===nothing ? circuit.s : planar_y_to_s(y,circuit.z0)
    transfer=Matrix{ComplexF64}(I,size(s,1),size(s,1))
    if calibration!==nothing
        A,B,C,D=_calibration_chain_blocks(calibration,size(s,1))
        transfer=_planar_calibration_incident_transfer(A,B,C,D,y,s,circuit.z0)
    end
    return SonnetFloatingResult(model.project,em,circuit,model.external_labels,circuit.z0,y,s,
        model.source_incidence,model.bridges,transfer,model.scalar_files,model.variables,model.model_files)
end

"""Reconstruct the actual physical floating/native loaded current maps
for external incident power waves, including coupled calibration transfer
and explicit signal/reference circuit voltages."""
function planar_current_maps(result::SonnetFloatingResult;port::Integer=1,incident_waves=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    n=length(result.z0)
    reserve=_checked_array_payload_bytes(ComplexF64,n+size(result.source_incidence,1)+size(result.source_incidence,2))
    _enforce_payload_limit(reserve,max_bytes,"native floating current excitation","max_bytes")
    a=if incident_waves===nothing
        1<=port<=n || throw(ArgumentError("native floating external port is invalid"))
        vector=zeros(ComplexF64,n);vector[port]=1;vector
    else
        incident_waves isa AbstractVector && length(incident_waves)==n && all(isfinite,incident_waves) ||
            throw(ArgumentError("native floating incident waves must be finite and match ports"))
        _planar_stored_phasor.(incident_waves)
    end
    v=transpose(result.source_incidence)*(result.circuit.voltages*(result.incident_transfer*a))
    return planar_current_maps(result.em;voltages=v,max_bytes=max_bytes-reserve,kw...)
end
sonnet_component_current_maps(result::SonnetFloatingResult;kw...)=planar_current_maps(result;kw...)
