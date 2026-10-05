# Circuit subdivision of planar EM blocks. All connections use terminal
# voltages/currents, preserving unequal wave references and exact opens.

export PlanarSubdivision, PlanarSubdivisionResult, planar_subdivide
export solve_planar_subdivision, planar_subdivision_currents

"""A circuit template plus planar EM parts. `terminals[k]` maps every
port of `parts[k]` to circuit terminal pairs (or ground-referenced nodes).
Add lumped/data/line elements to `circuit` normally. Optional `origins`
translate reconstructed part maps into common layout coordinates.
Each part is solved independently; coupling between separate EM boxes
is represented only by their circuit connections. Choose cuts on uniform
connecting lines and calibrate their launch discontinuities."""
struct PlanarSubdivision
    circuit::PlanarCircuit
    parts::Vector{PlanarProblem}
    terminals::Vector{Vector{Tuple{Int,Int}}}
    origins::Vector{Tuple{Float64,Float64}}
end

function PlanarSubdivision(circuit::PlanarCircuit,parts::AbstractVector{PlanarProblem},
        terminals;origins=fill((0.0,0.0),length(parts)))
    !isempty(parts) && length(terminals)==length(parts)==length(origins) ||
        throw(ArgumentError("subdivision parts, terminal maps and origins must match"))
    maps = [_circuit_terminals(circuit.nnodes,local_map) for local_map in terminals]
    for (part,map) in zip(parts,maps)
        length(part.ports)==length(map) || throw(ArgumentError(
            "subdivision terminal map must match every local EM port"))
    end
    shifts = Tuple{Float64,Float64}[(Float64(x),Float64(y)) for (x,y) in origins]
    all(p -> all(isfinite,p),shifts) || throw(ArgumentError("part origins must be finite"))
    template = PlanarCircuit(circuit.nnodes,copy(circuit.ports),
        _planar_store_references(circuit.z0,length(circuit.ports)),copy(circuit.elements))
    return PlanarSubdivision(template,collect(parts),maps,shifts)
end

"""Composite circuit response with optional retained EM results. `s,y`
describe the external ports; `circuit` retains node voltages for all unit
incident-wave excitations. Set `retain_results=true` during solving to
enable [`planar_subdivision_currents`](@ref)."""
struct PlanarSubdivisionResult
    plan::PlanarSubdivision
    freq::Float64
    s::Matrix{ComplexF64}
    y::Union{Nothing,Matrix{ComplexF64}}
    circuit::PlanarCircuitResult
    results::Union{Nothing,Vector{Any}}
    z0::Vector{ComplexF64}
end
PlanarSubdivisionResult(p,f,s,y,c,r)=PlanarSubdivisionResult(p,f,s,y,c,r,copy(c.z0))

# Count unique retained Julia array payloads, excluding caller-owned
# problems and opaque FFT plans. Sparse storage and shared mode/family
# arrays are visited explicitly, so aliases do not double-count storage.
function _subdivision_retained_payload(value,seen=Base.IdSet{Any}())
    value===nothing && return 0
    value isa PlanarProblem && return 0
    if value isa SparseMatrixCSC
        return _checked_payload_sum("retained sparse subdivision loss",
            _subdivision_retained_payload(value.colptr,seen),
            _subdivision_retained_payload(value.rowval,seen),
            _subdivision_retained_payload(value.nzval,seen))
    elseif value isa Union{PlanarResult,PlanarUFFTResult,PlanarCalibratedResult,
            PlanarContractedResult,PlanarSourceResult,
            PlanarUFFTOperator,PlanarModeGrid,_PlanarUFFTFamily,_PlanarUFFTFoldedWorkspace,LinearAlgebra.LU,
            PlanarBasisSet,SheetLevel,ViaLevel,VolLevel}
        return _checked_payload_sum("retained subdivision result",
            (_subdivision_retained_payload(getfield(value,k),seen)
                for k in 1:fieldcount(typeof(value)))...)
    elseif value isa AbstractArray
        value in seen && return 0
        push!(seen,value)
        if isbitstype(eltype(value))
            return value isa BitArray ? 8cld(length(value),64) :
                _checked_array_payload_bytes(eltype(value),length(value))
        end
        return _checked_payload_sum("retained subdivision arrays",
            (_subdivision_retained_payload(v,seen) for v in value)...)
    end
    return 0
end

"""Solve each EM part once and recombine through the circuit template.
`chains[k]` optionally supplies one ABCD launch per local port, or a
frequency callback producing those chains. Otherwise each part's port
reference-plane contracts are used. `part_keywords[k]` can override
common solve options (for example modal truncation for differently sized
boxes). Raw dense matrices are omitted by default. Retaining results
also retains their factors/operators for later current reconstruction."""
function solve_planar_subdivision(plan::PlanarSubdivision,f::Real;
        retain_results::Bool=false,chains=nothing,part_keywords=nothing,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    isfinite(f) && f>0 || throw(ArgumentError("EM subdivision frequency must be positive"))
    n = length(plan.parts)
    chains===nothing || length(chains)==n || throw(ArgumentError("chains must match EM parts"))
    part_keywords===nothing || length(part_keywords)==n ||
        throw(ArgumentError("part_keywords must match EM parts"))
    network_payload = _checked_payload_sum("subdivision network responses",
        _checked_array_payload_bytes(ComplexF64,2,length(plan.circuit.ports)),
        (_checked_array_payload_bytes(ComplexF64,length(part.ports)^2+length(part.ports))
            for part in plan.parts)...)
    _enforce_payload_limit(network_payload,max_bytes,"subdivision responses","max_bytes")
    circuit = PlanarCircuit(plan.circuit.nnodes,copy(plan.circuit.ports),
        _planar_store_references(plan.circuit.z0,length(plan.circuit.ports)),
        copy(plan.circuit.elements))
    retained = retain_results ? Any[] : nothing
    retained_payload = 0
    for k in 1:n
        available = max_bytes-network_payload-retained_payload
        _enforce_payload_limit(0,available,"subdivision retained results","max_bytes")
        options = merge((retain_matrix=false,),(;kw...),
            part_keywords===nothing ? NamedTuple() : part_keywords[k])
        options = merge(options,(max_bytes=min(get(options,:max_bytes,available),available),))
        raw = solve_planar(plan.parts[k],f;options...)
        local_chains = chains===nothing ? nothing :
            (chains[k] isa AbstractVector ? chains[k] : chains[k](f))
        calibrated = if local_chains!==nothing || any(p -> p.refplane!==nothing,plan.parts[k].ports)
            planar_reference_planes(raw;chains=local_chains,
                max_bytes=available-_subdivision_retained_payload(raw))
        else
            raw
        end
        circuit_add_network!(circuit,plan.terminals[k],calibrated.s;z0=calibrated.z0)
        retain_results && push!(retained,calibrated)
        retain_results && (retained_payload = _subdivision_retained_payload(retained))
        _enforce_payload_limit(network_payload+retained_payload,max_bytes,
            "subdivision retained results","max_bytes")
        raw = nothing;calibrated = nothing
    end
    combined = solve_planar_circuit(circuit,f;max_bytes=max_bytes-network_payload-retained_payload)
    return PlanarSubdivisionResult(plan,Float64(f),combined.s,combined.y,combined,retained)
end

"""Reconstruct all retained part current maps for external incident
power waves. The default is one unit incident wave on `port`, with the
other external ports matched. Circuit node voltages determine each local
port voltage; calibrated launches map these back to raw gaps. Part maps
are translated by their layout origins. Results are nested by part."""
function planar_subdivision_currents(result::PlanarSubdivisionResult;
        incident_waves=nothing,port::Integer=1,kw...)
    result.results===nothing && throw(ArgumentError(
        "subdivision currents require retain_results=true during solve"))
    n = length(result.plan.circuit.ports)
    waves = if incident_waves===nothing
        1<=port<=n || throw(ArgumentError("external port index is invalid"))
        a=zeros(ComplexF64,n);a[port]=1;a
    else
        incident_waves isa AbstractVector && length(incident_waves)==n && all(isfinite,incident_waves) ||
            throw(ArgumentError("incident waves must be finite and match external ports"))
        _planar_stored_phasor.(incident_waves)
    end
    nodes = result.circuit.voltages*waves
    maps = Vector{Vector{PlanarCurrentMap}}()
    for (k,part) in enumerate(result.results)
        volts = ComplexF64[(p==0 ? 0im : nodes[p])-(m==0 ? 0im : nodes[m])
            for (p,m) in result.plan.terminals[k]]
        local_maps = planar_current_maps(part;voltages=volts,kw...)
        x0,y0 = result.plan.origins[k]
        translated = [PlanarCurrentMap(map.kind,map.level,map.x.+x0,map.y.+y0,
            map.z,map.zmin,map.zmax,map.mask,map.jx,map.jy,map.jz) for map in local_maps]
        push!(maps,translated)
    end
    return maps
end

function _subdivision_spans(bits)
    spans = UnitRange{Int}[]
    start = 0
    for i in 1:length(bits)+1
        on = i<=length(bits) && bits[i]
        if on && start==0
            start=i
        elseif !on && start!=0
            push!(spans,start:i-1);start=0
        end
    end
    return spans
end

function _subdivision_port(p::PlanarPort,grid::CellGrid,axis::Symbol,edge::Int)
    if p.wall===:via
        throw(ArgumentError("automatic cuts require sheet ports; provide explicit maps for via ports"))
    end
    along_walls = axis===:x ? (:west,:east) : (:south,:north)
    if p.wall in along_walls
        return (p.wall===along_walls[1] ? 1 : 2),p
    end
    along_internal = axis===:x ? :internal_x : :internal_y
    if p.wall===along_internal || (axis===:x ? _is_planar_x_terminal(p.wall) :
            _is_planar_terminal(p.wall) && !_is_planar_x_terminal(p.wall))
        p.edge==edge && throw(ArgumentError("an existing internal port lies on the requested cut"))
        side = p.edge<edge ? 1 : 2
        return side,PlanarPort(p.level,p.wall,p.cells,p.z0;
            edge=p.edge-(side==2 ? edge : 0),polarity=p.polarity,refplane=p.refplane)
    end
    first(p.cells)<=edge<last(p.cells) && throw(ArgumentError(
        "requested cut intersects a port span; move the cut or supply explicit parts"))
    side = last(p.cells)<=edge ? 1 : 2
    cells = side==1 ? p.cells : (first(p.cells)-edge):(last(p.cells)-edge)
    return side,PlanarPort(p.level,p.wall,cells,p.z0;
        edge=p.edge,polarity=p.polarity,refplane=p.refplane)
end

"""Automatically cut a raster problem between cells `edge` and
`edge+1` along `axis=:x/:y`. Every contiguous sheet crossing creates a
pair of connected circuit ports; original ports remain external. Vias
and volumes are sliced, but cuts through volume conductors and existing
via ports require explicit subdivision parts. Port spans may not cross
the cut. Grid spacing, stackup, remaining conductor geometry and external
reference-plane contracts are preserved. New cut ports use `z0`.

The new boxes impose their original wall kind at the cut. Their launch
parasitics must be calibrated, and radiation/direct coupling between
parts is absent; this is a circuit approximation, not an exact partition
of a globally coupled electromagnetic operator."""
function planar_subdivide(prob::PlanarProblem,axis::Symbol,edge::Integer;z0=50.0)
    axis in (:x,:y) || throw(ArgumentError("cut axis must be :x or :y"))
    total = axis===:x ? prob.grid.nx : prob.grid.ny
    1<=edge<total || throw(ArgumentError("cut must be an interior grid edge"))
    _planar_store_reference(z0)
    k = Int(edge)
    crossing(mask) = axis===:x ? mask[k,:].&mask[k+1,:] : mask[:,k].&mask[:,k+1]
    any(v -> any(crossing(v.mask)),prob.vols) && throw(ArgumentError(
        "cut crosses a volume conductor; explicit calibrated volume parts are required"))
    spans = [(lv,span) for (lv,sheet) in enumerate(prob.sheets)
        for span in _subdivision_spans(crossing(sheet.mask))]
    isempty(spans) && throw(ArgumentError("cut has no crossing sheet conductor"))
    ports = [PlanarPort[],PlanarPort[]]
    maps = [Int[],Int[]]
    for (i,p) in enumerate(prob.ports)
        side,local_port = _subdivision_port(p,prob.grid,axis,k)
        push!(ports[side],local_port);push!(maps[side],i)
    end
    low,high = axis===:x ? (:west,:east) : (:south,:north)
    for (j,(lv,span)) in enumerate(spans)
        for side in 1:2
            push!(ports[side],PlanarPort(lv,side==1 ? high : low,span,z0))
            push!(maps[side],length(prob.ports)+j)
        end
    end
    parts = PlanarProblem[]
    for side in 1:2
        inds = side==1 ? (1:k) : (k+1:total)
        nx,ny = axis===:x ? (length(inds),prob.grid.ny) : (prob.grid.nx,length(inds))
        a,b = nx*prob.grid.dx,ny*prob.grid.dy
        grid = CellGrid(a,b,nx,ny;walls=prob.grid.walls)
        stack = PlanarStackup(prob.stack.layers,prob.stack.bottom,prob.stack.top,a,b)
        slice(mask) = BitMatrix(axis===:x ? mask[inds,:] : mask[:,inds])
        function sliced_level(level,volume=false)
            new = volume ? vol_level(level.layer,nx,ny) : sheet_level(level.interface,nx,ny)
            new.mask .= slice(level.mask)
            if axis===:x
                new.connect_south .= level.connect_south[inds]
                new.connect_north .= level.connect_north[inds]
                side==1 ? (new.connect_west .= level.connect_west) :
                    (new.connect_east .= level.connect_east)
            else
                new.connect_west .= level.connect_west[inds]
                new.connect_east .= level.connect_east[inds]
                side==1 ? (new.connect_south .= level.connect_south) :
                    (new.connect_north .= level.connect_north)
            end
            return new
        end
        sheets = SheetLevel[sliced_level(sheet) for sheet in prob.sheets]
        for (lv,span) in spans
            connection = axis===:x ? (side==1 ? sheets[lv].connect_east : sheets[lv].connect_west) :
                (side==1 ? sheets[lv].connect_north : sheets[lv].connect_south)
            connection[span].=true
        end
        vias = [ViaLevel(v.layer,slice(v.uni),slice(v.tap)) for v in prob.vias]
        vols = VolLevel[sliced_level(v,true) for v in prob.vols]
        push!(parts,build_planar_problem(stack,grid,sheets,ports[side];vias=vias,vols=vols))
    end
    circuit = PlanarCircuit(length(prob.ports)+length(spans),collect(eachindex(prob.ports));
        z0=[p.z0 for p in prob.ports])
    origin = axis===:x ? (k*prob.grid.dx,0.0) : (0.0,k*prob.grid.dy)
    return PlanarSubdivision(circuit,parts,maps;origins=[(0.0,0.0),origin])
end
