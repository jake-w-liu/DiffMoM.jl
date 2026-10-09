export PlanarProjectResult,solve_planar_project,planar_project_sweep

"""Solved project with physical EM result `em` and optional loaded
`circuit`. `s,y,z0,port_names` describe the declared external ports.
Current maps reconstruct the actual circuit-solved component-pin voltages
through calibration and physical reference returns into the EM geometry."""
struct PlanarProjectResult{R,C}
    project::PlanarProject
    model::PlanarProjectModel
    em::R
    circuit::C
    freq::Float64
    port_names::Vector{String}
    z0::Vector{ComplexF64}
    y::Union{Nothing,Matrix{ComplexF64}}
    s::Matrix{ComplexF64}
end

function _project_circuit(model::PlanarProjectModel,freq,component_response,max_bytes)
    project=model.project;variables=model.variables
    q(x,d=:dimensionless)=planar_project_value(project,x;dimension=d,freq,variables)
    function node(port)
        port in (0,"gnd") && return 0
        idx=port isa AbstractString ? findfirst(==(port),model.port_names) :
            findfirst(==(_project_int(port,"component port number")),model.port_numbers)
        idx===nothing && throw(ArgumentError("component references undefined port $port"))
        return idx
    end
    function namednode(value)
        key=_project_node_key(value);isempty(key) && return 0
        id=findfirst(==(key),model.node_names)
        id===nothing && throw(ArgumentError("undefined circuit node $value"))
        return id
    end
    function pair(value)
        value isa AbstractVector && length(value)==2 || throw(ArgumentError("network nodes require [positive,negative] pairs"))
        return Tuple(namednode.(value))
    end
    circuit=PlanarCircuit(length(model.node_names),model.port_terminals[model.external];z0=model.z0[model.external])
    names=String[];libraries=Dict{String,PlanarSpiceLibrary}()
    for (k,c) in enumerate(model.components)
        _project_keys(c,["name","type","value","ports","nodes","r","l","c","topology",
            "zc","gamma","length","ratio","path","subckt","parameters","dialect","z0","frequencies","s_real","s_imag"],"component")
        name=String(get(c,"name","component$k"));push!(names,name)
        kind=c["type"];explicit=haskey(c,"nodes")
        raw=explicit ? c["nodes"] : c["ports"]
        ports=explicit ? nothing : node.(raw)
        if kind in ("resistor","capacitor","inductor","rlc")
            length(raw)==2 || throw(ArgumentError("RLC component needs two terminal nodes"))
            terminals=if explicit
                namednode.(raw)
            elseif iszero(ports[1]) || iszero(ports[2])
                anchor=ports[iszero(ports[1]) ? 2 : 1]
                anchor>0 || throw(ArgumentError("RLC load requires a non-ground anchor"))
                # A legacy anchor/0 load spans that physical port's pair.
                collect(model.port_terminals[anchor])
            else
                [model.port_terminals[i][1] for i in ports]
            end
            r=kind=="resistor" ? q(c["value"],:resistance) : haskey(c,"r") ? q(c["r"],:resistance) : nothing
            l=kind=="inductor" ? q(c["value"],:inductance) : haskey(c,"l") ? q(c["l"],:inductance) : nothing
            cap=kind=="capacitor" ? q(c["value"],:capacitance) : haskey(c,"c") ? q(c["c"],:capacitance) : nothing
            circuit_add_rlc!(circuit,terminals...;r,l,c=cap,topology=Symbol(get(c,"topology","series")))
        elseif kind=="transmission_line"
            length(raw)==2 && (explicit || all(>(0),ports)) || throw(ArgumentError("line needs two physical port pairs"))
            terminals=explicit ? pair.(raw) : model.port_terminals[ports]
            circuit_add_line!(circuit,terminals,q(c["zc"],:resistance),
                q(c["gamma"],:inverse_length)*planar_project_value(project,c["length"];dimension=:length,variables))
        elseif kind=="transformer"
            length(raw) in (2,4) || throw(ArgumentError("transformer needs two port pairs or four terminal nodes"))
            pairs=if length(raw)==2
                explicit ? pair.(raw) : model.port_terminals[ports]
            else
                ns=explicit ? namednode.(raw) : [iszero(i) ? 0 : model.port_terminals[i][1] for i in ports]
                [(ns[1],ns[2]),(ns[3],ns[4])]
            end
            circuit_add_transformer!(circuit,pairs,q(c["ratio"]))
        elseif kind=="sparam_file"
            explicit || all(>(0),ports) || throw(ArgumentError("network blocks require physical signal port anchors"))
            ports=explicit ? pair.(raw) : model.port_terminals[ports]
            path=String(c["path"])
            if !isabspath(path)
                isempty(project.source) && throw(ArgumentError("relative component paths require a project source path"))
                path=joinpath(dirname(project.source),path)
            end
            data=planar_read_touchstone(path;max_bytes)
            length(data.z0)==length(ports) || throw(ArgumentError("file component port count differs from anchors"))
            refs=haskey(c,"z0") ? _circuit_z0(c["z0"] isa AbstractVector ?
                [_project_reference_value(project,z,freq,variables) for z in c["z0"]] :
                _project_reference_value(project,c["z0"],freq,variables),length(ports)) : data.z0
            circuit_add_network!(circuit,ports,planar_network_response(data,freq;z0=refs);z0=refs)
        elseif kind=="inline_s"
            explicit || all(>(0),ports) || throw(ArgumentError("network blocks require physical signal port anchors"))
            ports=explicit ? pair.(raw) : model.port_terminals[ports]
            real_parts=c["s_real"]
            _enforce_payload_limit(_checked_payload_sum("inline network input",
                _checked_array_payload_bytes(ComplexF64,2,length(real_parts),length(ports),length(ports)),
                _checked_array_payload_bytes(ComplexF64,4,length(real_parts),length(ports))),
                max_bytes,"inline network input","max_bytes")
            imag_parts=get(c,"s_imag",[zeros(length(ports),length(ports)) for _ in real_parts])
            function matrix(parts)
                parts isa AbstractMatrix && return Matrix{Float64}(parts)
                return reduce(vcat,(permutedims(Float64.(row)) for row in parts))
            end
            length(real_parts)==length(imag_parts) || throw(DimensionMismatch("inline complex response arrays differ"))
            matrices=[complex.(matrix(r),matrix(i)) for (r,i) in zip(real_parts,imag_parts)]
            fs=Float64[q(f,:frequency) for f in c["frequencies"]]
            declared=get(c,"z0",50.)
            at(f)=declared isa AbstractVector ?
                [_project_reference_value(project,z,f,variables) for z in declared] :
                _project_reference_value(project,declared,f,variables)
            reference_series=[_circuit_z0(at(f),length(ports)) for f in fs]
            data=PlanarNetworkData(fs,matrices;reference_series,max_bytes)
            length(data.z0)==length(ports) || throw(ArgumentError("inline network port count differs from anchors"))
            refs=_circuit_z0(at(freq),length(ports))
            circuit_add_network!(circuit,ports,planar_network_response(data,freq;z0=refs);z0=refs)
        elseif kind=="subckt" && haskey(c,"subckt")
            explicit || throw(ArgumentError("linear SPICE components require ordered explicit nodes"))
            path=String(c["path"])
            if !isabspath(path)
                isempty(project.source) && throw(ArgumentError("relative SPICE paths require a project source path"))
                path=joinpath(dirname(project.source),path)
            end
            path=abspath(path);dialect=get(c,"dialect","spice")
            dialect in ("spice","spectre") || throw(ArgumentError("subckt dialect must be spice or spectre"))
            key=dialect*":"*(Sys.iswindows() ? lowercase(path) : path)
            retained=_checked_payload_sum("project SPICE models",_circuit_owned_model_payload(circuit),_spice_circuit_payload(circuit))
            _enforce_payload_limit(retained,max_bytes,"project SPICE models","max_bytes")
            remaining=max_bytes-retained
            library=get!(libraries,key) do
                dialect=="spectre" ? planar_read_spectre(path;root=dirname(path),max_bytes=remaining) :
                    planar_read_spice(path;root=dirname(path),max_bytes=remaining)
            end
            parameters=Dict{String,Float64}()
            for (parameter,value) in get(c,"parameters",Dict())
                parameters[String(parameter)]=_project_real(q(value,:any),"SPICE parameter $parameter")
            end
            compiled=planar_spice_model(library,String(c["subckt"]);parameters,max_bytes=remaining)
            circuit_add_spice!(circuit,namednode.(raw),compiled;name,max_bytes)
        elseif kind in ("subckt","vendor","network")
            explicit || all(>(0),ports) || throw(ArgumentError("network blocks require physical signal port anchors"))
            ports=explicit ? pair.(raw) : model.port_terminals[ports]
            component_response===nothing && throw(ArgumentError("$kind component requires component_response(project,component,freq)"))
            supplied=component_response(project,c,freq)
            supplied isa NamedTuple && all(k->haskey(supplied,k),(:response,:format,:z0)) ||
                throw(ArgumentError("component_response must return response,format,z0"))
            circuit_add_network!(circuit,ports,supplied.response;format=supplied.format,z0=supplied.z0)
        else
            throw(ArgumentError("unknown project component type $kind"))
        end
    end
    length(unique(names))==length(names) && all(!isempty,names) || throw(ArgumentError("component names must be unique and nonempty"))
    return circuit
end

function _project_model_payload(model)
    layout=model.layout;seen=Base.IdSet{Any}()
    owned=Any[layout.problem.basis,layout.source_problem.basis,layout.problem.sheets,
        layout.source_problem.sheets,layout.problem.vias,layout.source_problem.vias,
        layout.problem.vols,layout.source_problem.vols,layout.sheet_materials,layout.contraction,model.volume_sigma,
        model.source_incidence,
        [p.vertices for shape in layout.shapes for p in shape.polygons],
        [v.vertices for shape in layout.shapes for v in shape.vias]]
    return _subdivision_retained_payload(owned,seen)
end

"""Solve a complete declarative planar project at `freq` [Hz]. Dielectric
and conductor expressions are evaluated at this point, all geometry and
component ports are lowered, physical return sources are contracted and
reference planes are applied before loading. RLC, lines, transformers,
Touchstone, inline S and bounded linear SPICE subcircuits use the native
MNA solver. Named SPICE definitions retain literal pin order, hierarchy,
numeric parameters and file provenance. Vendor and unsupported native
models require an explicit response adapter; no load is dropped.
The owned geometry, retained EM solve and loaded circuit share `max_bytes`."""
function solve_planar_project(project::PlanarProject,freq::Real;grid=nothing,
        variables::AbstractDict=Dict{String,Any}(),component_response=nothing,
        terminal_ground::Symbol=:auto,
        max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    freq=_project_stored_frequency(freq)
    model=planar_project_layout(project;freq,grid,variables,terminal_ground,max_bytes)
    owned=_project_model_payload(model);n=max(length(model.port_names),length(model.node_names))
    workspace=_checked_array_payload_bytes(ComplexF64,24,n,n)
    _enforce_payload_limit(owned+workspace,max_bytes,"planar project solve","max_bytes")
    # Validate/evaluate every load before starting an expensive EM solve.
    circuit=isempty(model.components) ? nothing :
        _project_circuit(model,freq,component_response,max_bytes-owned-workspace)
    circuit_owned=circuit===nothing ? 0 : _checked_payload_sum("project circuit storage",
        _circuit_owned_model_payload(circuit),_spice_circuit_payload(circuit))
    _enforce_payload_limit(_checked_payload_sum("project EM solve",owned,workspace,circuit_owned),
        max_bytes,"project EM solve","max_bytes")
    em=solve_planar(model.layout,freq;max_bytes=max_bytes-owned-workspace-circuit_owned,
        volume_sigma=model.volume_sigma,kw...)
    if any(p->p.refplane!==nothing,model.layout.problem.ports)
        em=planar_reference_planes(em;max_bytes=workspace)
    end
    external=model.external;names=model.port_names[external];refs=model.z0[external]
    if circuit===nothing
        length(external)==length(model.port_names) || throw(ArgumentError("unloaded hidden ports require an explicit circuit termination"))
        return PlanarProjectResult(project,model,em,nothing,Float64(freq),names,refs,em.y,em.s)
    end
    circuit_add_em!(circuit,model.port_terminals,em;z0=model.z0)
    loaded=solve_planar_circuit(circuit,freq;
        max_bytes=max_bytes-owned-workspace-_subdivision_retained_payload(em)-_spice_circuit_payload(circuit),floating_gauge=:auto)
    return PlanarProjectResult(project,model,em,loaded,Float64(freq),names,refs,loaded.y,loaded.s)
end

function solve_planar_project(path::AbstractString,freq::Real;
        ascent::Bool=false,max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    freq=_project_stored_frequency(freq)
    return solve_planar_project(load_planar_project(path;ascent,max_bytes),freq;max_bytes,kw...)
end
solve_planar(project::PlanarProject,freq::Real;kw...)=solve_planar_project(project,freq;kw...)
planar_connectivity(project::PlanarProject;kw...)=planar_connectivity(planar_project_layout(project;kw...).layout)

"""Reconstruct a project's loaded EM current maps. External incident
power waves drive the actual solved internal component node voltages."""
function planar_current_maps(result::PlanarProjectResult;port::Integer=1,incident_waves=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    result.circuit===nothing && return planar_current_maps(result.em;port,incident_waves,max_bytes,kw...)
    n=length(result.z0)
    reserve=_checked_array_payload_bytes(ComplexF64,n+size(result.circuit.voltages,1)+length(result.model.port_names))
    _enforce_payload_limit(reserve,max_bytes,"project current excitation","max_bytes")
    waves=if incident_waves===nothing
        1<=port<=n || throw(ArgumentError("project port index is invalid"))
        a=zeros(ComplexF64,n);a[port]=1;a
    else
        incident_waves isa AbstractVector && length(incident_waves)==n && all(isfinite,incident_waves) ||
            throw(ArgumentError("project incident waves must be finite and match external ports"))
        _planar_stored_phasor.(incident_waves)
    end
    node_voltages=result.circuit.voltages*waves
    v=transpose(result.model.source_incidence)*view(node_voltages,1:length(result.model.node_names))
    return planar_current_maps(result.em;voltages=v,max_bytes=max_bytes-reserve,kw...)
end

function _project_radiation_excitation(result::PlanarProjectResult;
        port::Integer=1,voltages=nothing,incident_waves=nothing,max_bytes)
    n=length(result.z0);m=length(result.model.port_names)
    source=_planar_radiation_problem(result.em);nb=planar_basis_count(source.basis)
    reserve=_checked_payload_sum("project radiation excitation",
        _checked_array_payload_bytes(ComplexF64,nb,m+1),
        _checked_array_payload_bytes(ComplexF64,5,n),
        _checked_array_payload_bytes(ComplexF64,4,n,n),
        _checked_array_payload_bytes(ComplexF64,m+(result.circuit===nothing ?
            length(result.model.node_names) : size(result.circuit.voltages,1))))
    # Retained Y needs one current vector. A loaded circuit also needs
    # one external terminal-voltage vector, distinct from its node values.
    power_vectors=result.circuit===nothing ? (result.y===nothing ? 0 : 1) : 2
    reserve=_checked_payload_sum("project physical accepted power",reserve,
        _checked_array_payload_bytes(ComplexF64,power_vectors,n))
    if result.y!==nothing && (voltages!==nothing || result.circuit===nothing)
        # A voltage request owns voltage/current vectors and one root array.
        # The S route owns reference/incident/reflected/voltage vectors; a
        # consistency check owns current, retained recovery owns voltage.
        # Each incident voltage route owns roots.
        helper_vectors=voltages===nothing ? 4+1+1 : 2
        helper_roots=voltages===nothing ? 1+1 : 1
        reserve=_checked_payload_sum("project retained admittance excitation",reserve,
            _checked_array_payload_bytes(ComplexF64,helper_vectors,n),
            _checked_array_payload_bytes(Float64,helper_roots,n))
        if result.circuit===nothing && voltages===nothing
            reserve=_checked_payload_sum("project admittance voltage factor",reserve,
                _checked_array_payload_bytes(ComplexF64,n,n),
                _checked_array_payload_bytes(Int,n))
        end
    end
    raw_columns=result.em isa PlanarCalibratedResult ? result.em.raw.currents : result.em.currents
    if eltype(raw_columns)!==ComplexF64
        bits=_planar_current_precision(raw_columns)
        reserve=_checked_payload_sum("wide project radiation excitation",reserve,
            _planar_owned_current_product_payload(bits,nb),
            result.em isa PlanarCalibratedResult ?
                _planar_owned_current_product_payload(bits,nb,m) : 0)
    end
    _enforce_payload_limit(reserve,max_bytes,"project radiation excitation","max_bytes")
    voltages===nothing || incident_waves===nothing || throw(ArgumentError("provide voltages or incident_waves"))
    supplied=voltages===nothing ? incident_waves : voltages
    input=if supplied===nothing
        1<=port<=n || throw(ArgumentError("project radiation port is invalid"))
        values=zeros(ComplexF64,n);values[port]=1;values
    else
        supplied isa AbstractVector && length(supplied)==n && all(isfinite,supplied) ||
            throw(ArgumentError("project radiation excitation must be finite and match external ports"))
        _planar_stored_phasor.(supplied)
    end
    # Circuit voltages are solved per external incident power wave. An
    # external voltage request must therefore be transformed at the loaded
    # network's reference planes, before its internal node transfer.
    waves=voltages===nothing ? input : result.y===nothing ?
        _planar_incident_from_voltage(result.s,input,result.z0) :
        _planar_incident_admittance(result.y,input,result.z0;max_bytes,retained_bytes=reserve)
    nodes=if result.circuit===nothing
        voltages!==nothing ? input : result.y===nothing ?
            _planar_wave_voltage(result.s,waves,result.z0) :
            _planar_wave_voltage_retained(result.y,result.s,waves,result.z0)
    else
        node_voltages=result.circuit.voltages*waves
        transpose(result.model.source_incidence)*view(node_voltages,1:length(result.model.node_names))
    end
    accepted=if result.circuit!==nothing
        # Circuit currents are external currents into the network. Form
        # physical differential port voltages from the same retained MNA
        # solution, including floating/reference-return terminal pairs.
        terminal_voltages=Vector{ComplexF64}(undef,n)
        for p in 1:n
            positive,negative=result.model.port_terminals[result.model.external[p]]
            terminal_voltages[p]=(iszero(positive) ? 0.0im : node_voltages[positive])-
                (iszero(negative) ? 0.0im : node_voltages[negative])
        end
        current=result.circuit.currents*waves
        _planar_accepted_power(terminal_voltages,current;max_bytes,retained_bytes=reserve)
    elseif result.y!==nothing
        # Independently retained conductance can survive after S rounds to
        # a lossless boundary. Keep the physical peak-phasor V†YV relation.
        current=_planar_terminal_current(result.y,nodes;max_bytes,retained_bytes=reserve)
        _planar_accepted_power(nodes,current;max_bytes,retained_bytes=reserve)
    else
        reflected=result.s*waves
        .5real(dot(waves,waves)-dot(reflected,reflected))
    end
    tolerance=100eps(Float64)*norm(waves)^2
    isfinite(accepted) && accepted>=-tolerance || throw(ArgumentError("project radiation requires passive accepted power"))
    coeff=_planar_current_product(_planar_coefficient_columns(result.em),nodes)
    all(isfinite,coeff) || throw(ArgumentError("project radiation coefficients are nonfinite"))
    return (problem=source,coefficients=coeff,accepted=accepted>0 ? accepted : nothing,
        budget=Int(BigInt(_validated_resource_limit("max_bytes",max_bytes))-reserve))
end

"""Far field of a complete project, including loaded component-node
voltages, physical source contraction and reference-plane transfer.
The default drives one external incident power wave. `voltages` instead
specifies external reference-plane voltages with the solved loads retained.
Supply an explicit `radiation_stack` to change the propagation boundaries;
the current solution's box boundaries are preserved by default."""
function planar_farfield(result::PlanarProjectResult;port::Integer=1,voltages=nothing,
        incident_waves=nothing,max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    excitation=_project_radiation_excitation(result;port,voltages,incident_waves,max_bytes)
    return planar_farfield(excitation.problem,excitation.coefficients,result.freq;
        accepted_power=excitation.accepted,max_bytes=excitation.budget,kw...)
end

"""Radiated power for a project's actual loaded EM currents. Excitation
and reference-plane conventions match [`planar_farfield`](@ref)."""
function planar_radiated_power(result::PlanarProjectResult;port::Integer=1,voltages=nothing,
        incident_waves=nothing,max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    excitation=_project_radiation_excitation(result;port,voltages,incident_waves,max_bytes)
    return planar_radiated_power(excitation.problem,excitation.coefficients,result.freq;
        max_bytes=excitation.budget,kw...)
end

"""Run a project's configured sweep and return a response databank, or
an adaptive rational sweep when `sweep.adaptive=true`. Every sampled
frequency rebuilds its dispersive stack and loads. The retained output
payload is reserved from the solve budget; raw frequency factors are
discarded unless independently requested by a direct solve."""
function planar_project_sweep(project::PlanarProject;variables::AbstractDict=Dict{String,Any}(),
        max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    frequencies=planar_project_frequencies(project;variables,max_bytes)
    sweep=get(project.data,"sweep",Dict());adaptive=get(sweep,"adaptive",false)
    ne=mp=0;relative_tolerance=0.
    if adaptive
        length(frequencies)>=2 || throw(ArgumentError("adaptive sweep needs an interval"))
        ne=_project_int(planar_project_value(project,get(sweep,"n_eval",257);variables),"adaptive n_eval";positive=true)
        mp=_project_int(planar_project_value(project,get(sweep,"max_points",32);variables),"adaptive max_points";positive=true)
        relative_tolerance=planar_project_value(project,get(sweep,"rel_tol",1e-2);variables)
        lo,hi,ne,mp=_planar_abs_parameters(first(frequencies),last(frequencies),relative_tolerance,ne,mp)
        # Port count is known from the schema before evaluating technology
        # or material expressions. Hidden ports make this bound conservative.
        declared=length(project.data["ports"])
        early_storage=_checked_payload_sum("project adaptive output",
            _planar_sweep_storage_bytes(declared,ne,mp),
            _checked_array_payload_bytes(ComplexF64,8,declared,declared),
            _checked_array_payload_bytes(ComplexF64,4,declared))
        _enforce_payload_limit(early_storage,max_bytes,"project adaptive sweep","max_bytes")
        _planar_abs_frequency_grid(lo,hi,ne)
    end
    sample=planar_project_layout(project;freq=first(frequencies),variables,max_bytes)
    n=length(sample.external)
    names=sample.port_names[sample.external];refs=sample.z0[sample.external]
    sample=nothing
    if adaptive
        storage=_checked_payload_sum("project adaptive output",
            _planar_sweep_storage_bytes(n,ne,mp),
            _checked_array_payload_bytes(ComplexF64,8,n,n),
            _checked_array_payload_bytes(ComplexF64,4,n))
        _enforce_payload_limit(storage,max_bytes,"project adaptive sweep","max_bytes")
        options=merge((retain_matrix=false,), (;kw...))
        response(f)=begin
            result=solve_planar_project(project,f;variables,max_bytes=max_bytes-storage,options...)
            planar_renormalize_s(result.s,result.z0,refs)
        end
        return _planar_sweep_abs(response,n,first(frequencies),last(frequencies);
            rel_tol=relative_tolerance,n_eval=ne,max_points=mp,max_bytes,z0=refs)
    end
    storage=_checked_payload_sum("project sweep output",
        _checked_array_payload_bytes(ComplexF64,length(frequencies),n,n),
        _checked_array_payload_bytes(Float64,length(frequencies)),
        _checked_array_payload_bytes(ComplexF64,(4length(frequencies)+3)*n))
    _enforce_payload_limit(storage,max_bytes,"project sweep output","max_bytes")
    options=merge((retain_matrix=false,), (;kw...))
    matrices=Matrix{ComplexF64}[];reference_series=Vector{ComplexF64}[]
    for f in frequencies
        result=solve_planar_project(project,f;variables,max_bytes=max_bytes-storage,options...)
        push!(matrices,result.s);push!(reference_series,result.z0)
    end
    return PlanarNetworkData(frequencies,matrices,first(reference_series),names,Val(:owned);reference_series)
end

function planar_project_sweep(path::AbstractString;ascent::Bool=false,
        max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    return planar_project_sweep(load_planar_project(path;ascent,max_bytes);max_bytes,kw...)
end
