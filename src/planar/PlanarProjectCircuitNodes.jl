export planar_project_source_modes

function _project_node_key(value)
    value in (0,"0","gnd") && return ""
    value isa AbstractString && !isempty(value) && return String(value)
    value isa Integer && value>0 && return "node_$value"
    throw(ArgumentError("circuit node must be a nonempty name, positive integer or ground"))
end

function _project_flat_nodes(value)
    value isa AbstractVector || throw(ArgumentError("component nodes must be an array"))
    out=Any[]
    for node in value
        node isa AbstractVector ? append!(out,node) : push!(out,node)
    end
    return out
end

function _project_node_contracts(project,port_names,components;max_bytes)
    names=String[];pairs=Tuple{Int,Int}[]
    function index(value)
        key=_project_node_key(value);isempty(key) && return 0
        id=findfirst(==(key),names)
        id===nothing && (push!(names,key);id=length(names))
        return id
    end
    for (k,p) in enumerate(project.data["ports"])
        nodes=get(p,"nodes",[port_names[k],get(p,"type","box_wall")=="floating" ?
            "ref:"*p["ref_polygon"] : "gnd"])
        nodes isa AbstractVector && length(nodes)==2 || throw(ArgumentError("port nodes must be [positive,negative]"))
        positive,negative=index.(nodes)
        positive!=negative || throw(ArgumentError("EM port circuit nodes must be distinct"))
        push!(pairs,(positive,negative))
    end
    for c in components
        haskey(c,"nodes") || continue
        foreach(index,_project_flat_nodes(c["nodes"]))
    end
    _enforce_payload_limit(_checked_array_payload_bytes(Float64,length(names),length(pairs)),
        max_bytes,"project source incidence","max_bytes")
    D=planar_port_incidence(length(names),pairs;max_bytes)
    return names,pairs,D
end

"""Describe the actual voltage/current modes represented by a project's
EM sources. The signed `incidence` matrix D maps node voltages to physical
port voltages as `transpose(D)*v`, and port currents to nodal injections
as `D*I`. An EM admittance stamps as `D*Y*transpose(D)`.

`unexcited_voltage_modes` span the null space of `transpose(D)`. A lone
floating signal/reference bridge supplies one differential mode; it does
not supply an independent reference-to-box common mode. Adding an actual
reference source preserves that additional mode. These missing source
modes are distinct from the arbitrary MNA gauges reported by the circuit
solution; no common-mode response is synthesized from differential data."""
function planar_project_source_modes(model::PlanarProjectModel;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    D=model.source_incidence;n,p=size(D)
    _enforce_payload_limit(_checked_array_payload_bytes(Float64,8,n+p,n+p),
        max_bytes,"project source-mode report","max_bytes")
    return (node_names=copy(model.node_names),port_names=copy(model.port_names),
        incidence=copy(D),port_terminals=copy(model.port_terminals),
        represented_modes=LinearAlgebra.rank(D),
        unexcited_voltage_modes=LinearAlgebra.nullspace(transpose(D)))
end
