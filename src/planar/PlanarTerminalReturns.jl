export planar_terminal_returns

"""Construct physical box-cover return paths for open-edge terminals.
`terminals` contains `PlanarPort(...,:terminal_x/:terminal_y,...)` contracts;
the input problem retains its existing ports and conductor geometry.
Each pad's metal-adjacent edge cells receive uniform+tapered reference
via columns to an unobstructed PEC cover. `ground_direction=:below/:above`
selects a cover; `:auto` selects the nearest available clear PEC cover.

Returns `problem,contraction,z0,paths`. The contraction maps one voltage
per original port/terminal into the physical raw via sources. For a
multilayer return, each axial source voltage is proportional to its layer
thickness, and the transposed contraction extracts the conjugate terminal
current. Added vias include their complete electromagnetic effects;
calibrate these lead effects before comparing calibrated component data.
`max_bytes` preflights owned via masks, basis storage, contact bookkeeping
and the voltage contraction before those arrays are constructed.
No half-rooftop source is inserted without its physical return path."""
function planar_terminal_returns(prob::PlanarProblem,terminals::AbstractVector{PlanarPort};
        ground_direction::Symbol=:auto,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    ground_direction in (:auto,:below,:above) || throw(ArgumentError(
        "ground_direction must be :auto, :below or :above"))
    !isempty(terminals) || throw(ArgumentError("at least one physical terminal is required"))
    grid=prob.grid;L=length(prob.stack.layers)
    np,nt=length(prob.ports),length(terminals)
    initial_nb=planar_basis_count(prob.basis)
    grid_cells=_checked_array_payload_bytes(UInt8,grid.nx,grid.ny)
    function source_payload(contacts,levels)
        nr=_checked_payload_sum("physical return ports",np,contacts)
        nb=_checked_payload_sum("physical return basis",initial_nb,2BigInt(contacts))
        return _checked_payload_sum("physical terminal return construction",
            _checked_array_payload_bytes(UInt64,2,levels,cld(grid_cells,64)),
            _checked_array_payload_bytes(UInt8,nb),
            _checked_array_payload_bytes(Int,4,nb),
            _checked_array_payload_bytes(Float64,3,nb),
            _checked_array_payload_bytes(Float64,nr,np+nt),
            _checked_array_payload_bytes(Tuple{Int,Float64},nr),
            # Contact tuples and the occupied-path hash table (including
            # its minimum capacity and growth spare slots).
            _checked_array_payload_bytes(Int,64+12BigInt(contacts)),
            _checked_array_payload_bytes(Float64,L),
            _checked_array_payload_bytes(ComplexF64,np+nt),
            _checked_array_payload_bytes(Int,2,grid.nx+grid.ny))
    end
    _enforce_payload_limit(source_payload(0,0),max_bytes,"physical terminal returns","max_bytes")
    heights=Float64[real(layer.thickness) for layer in prob.stack.layers]
    ports=copy(prob.ports);vias=copy(prob.vias)
    weights=Tuple{Int,Float64}[(p,1.0) for p in eachindex(ports)]
    paths=NamedTuple[];occupied=Set{Tuple{Int,Int,Int}}()
    contacts=0;levels_added=0
    for (k,terminal) in enumerate(terminals)
        _is_planar_terminal(terminal.wall) || throw(ArgumentError("physical terminal needs an open-edge contract"))
        1<=terminal.level<=length(prob.sheets) || throw(ArgumentError("terminal sheet level is invalid"))
        interface=prob.sheets[terminal.level].interface
        xaxis=_is_planar_x_terminal(terminal.wall)
        positive=_is_planar_positive_terminal(terminal.wall)
        axis_count=xaxis ? grid.nx : grid.ny
        transverse_count=xaxis ? grid.ny : grid.nx
        !isempty(terminal.cells) && terminal.polarity in (-1,1) &&
            (positive ? 0<=terminal.edge<axis_count : 0<terminal.edge<=axis_count) &&
            1<=first(terminal.cells)<=last(terminal.cells)<=transverse_count ||
            throw(ArgumentError("physical terminal edge, span or polarity is invalid"))
        _planar_store_reference(terminal.z0)
        # A valid path crosses at least one layer. Reserve this minimum
        # source/contact payload before even allocating the cell tuples.
        _enforce_payload_limit(source_payload(
            _checked_payload_sum("physical return contacts",contacts,length(terminal.cells)),
            levels_added),max_bytes,"physical terminal returns","max_bytes")
        cell=terminal.edge+(positive ? 1 : 0)
        opposite=terminal.edge+(positive ? 0 : 1)
        mask=prob.sheets[terminal.level].mask
        for t in terminal.cells
            i,j=xaxis ? (cell,t) : (t,cell)
            mask[i,j] || throw(ArgumentError("physical terminal includes an edge without metal"))
            if 1<=opposite<=axis_count
                oi,oj=xaxis ? (opposite,t) : (t,opposite)
                mask[oi,oj] && throw(ArgumentError("physical terminal must lie on an open metal edge"))
            end
        end
        cells=[xaxis ? (cell,t) : (t,cell) for t in terminal.cells]
        candidates=Tuple{Float64,Symbol,UnitRange{Int}}[]
        for direction in (ground_direction===:auto ? (:below,:above) : (ground_direction,))
            cover=direction===:below ? prob.stack.bottom : prob.stack.top
            cover.kind===TERM_PEC || continue
            layers=direction===:below ? (1:interface) : (interface+1:L)
            isempty(layers) && continue
            function clear_path()
                for (i,j) in cells
                    for sheet in prob.sheets
                        sheet.interface==interface && continue
                        intervening=direction===:below ? 0<sheet.interface<interface :
                            interface<sheet.interface<L
                        intervening && sheet.mask[i,j] && return false
                    end
                    any(v->v.layer in layers && (v.uni[i,j] || v.tap[i,j]),prob.vias) && return false
                    any(v->v.layer in layers && v.mask[i,j],prob.vols) && return false
                    any(l->(l,i,j) in occupied,layers) && return false
                end
                return true
            end
            clear_path() || continue
            push!(candidates,(sum(heights[layers]),direction,layers))
        end
        isempty(candidates) && throw(ArgumentError(
            "terminal $k has no unobstructed physical return to its requested PEC cover"))
        distance,direction,layers=first(sort(candidates;by=first))
        contacts=_checked_payload_sum("physical return contacts",contacts,
            BigInt(length(cells))*length(layers))
        levels_added=_checked_payload_sum("physical return levels",levels_added,length(layers))
        _enforce_payload_limit(source_payload(contacts,levels_added),max_bytes,
            "physical terminal returns","max_bytes")
        for layer in layers
            via=via_level(layer,grid.nx,grid.ny)
            push!(vias,via);vi=length(vias)
            for (i,j) in cells
                via.uni[i,j]=true;via.tap[i,j]=true
                push!(occupied,(layer,i,j))
                linear=i+grid.nx*(j-1)
                polarity=terminal.polarity*(direction===:below ? 1 : -1)
                push!(ports,PlanarPort(vi,:via,linear:linear,terminal.z0;polarity=polarity))
                push!(weights,(length(prob.ports)+k,heights[layer]/distance))
            end
        end
        push!(paths,(direction=direction,layers=layers,cells=cells,length=distance,terminal=terminal))
    end
    problem=build_planar_problem(prob.stack,grid,prob.sheets,ports;vias=vias,vols=prob.vols)
    contraction=zeros(Float64,length(ports),length(prob.ports)+length(terminals))
    for (i,(terminal,weight)) in enumerate(weights)
        contraction[i,terminal]=weight
    end
    z0=[p.z0 for p in vcat(prob.ports,collect(terminals))]
    return (;problem,contraction,z0,paths)
end

# Exact count for a port-free geometry, without allocating a duplicate
# basis merely to discover whether its construction fits the budget.
function _terminal_geometry_basis_count(grid,sheets,vias,vols)
    nx,ny=grid.nx,grid.ny
    _checked_payload_sum("terminal geometry basis count",
        2BigInt(nx)*ny*(length(sheets)+length(vias)+length(vols)))
    nb=0
    for levels in (sheets,vols),level in levels
        _validate_sheet_grid(level,grid)
        mask=level.mask
        for j in 1:ny
            nb+=mask[1,j] && level.connect_west[j]
            nb+=mask[nx,j] && level.connect_east[j]
            for i in 1:nx-1
                nb+=mask[i,j] && mask[i+1,j]
            end
        end
        for i in 1:nx
            nb+=mask[i,1] && level.connect_south[i]
            nb+=mask[i,ny] && level.connect_north[i]
            for j in 1:ny-1
                nb+=mask[i,j] && mask[i,j+1]
            end
        end
    end
    for via in vias
        size(via.uni)==(nx,ny) && size(via.tap)==(nx,ny) ||
            throw(DimensionMismatch("terminal via masks must match the grid"))
        nb+=count(via.uni)+count(via.tap)
    end
    return nb
end

"""Build a planar problem driven only by physical cover-ground terminals.
This overload accepts stack/grid/sheets directly, plus existing `vias` and
`vols`, and constructs the same return paths and contraction as the
problem-based overload."""
function planar_terminal_returns(stack::PlanarStackup,grid::CellGrid,
        sheets::Vector{SheetLevel},terminals::AbstractVector{PlanarPort};
        vias::Vector{ViaLevel}=ViaLevel[],vols::Vector{VolLevel}=VolLevel[],
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    planar_validate(stack)
    stack.a==grid.a && stack.b==grid.b || throw(ArgumentError("terminal stack/grid dimensions must match"))
    nb=_terminal_geometry_basis_count(grid,sheets,vias,vols)
    bare_bytes=_checked_payload_sum("physical terminal input basis",
        _checked_array_payload_bytes(UInt8,nb),
        _checked_array_payload_bytes(Int,4,nb),
        _checked_array_payload_bytes(Float64,3,nb),
        _checked_array_payload_bytes(Int,2,grid.nx+grid.ny))
    _enforce_payload_limit(bare_bytes,max_bytes,"physical terminal input basis","max_bytes")
    basis=build_planar_basis(grid,sheets,PlanarPort[];vias=vias,vols=vols)
    bare=PlanarProblem(stack,grid,sheets,PlanarPort[],vias,basis,vols)
    return planar_terminal_returns(bare,terminals;max_bytes=max_bytes-bare_bytes,kw...)
end
