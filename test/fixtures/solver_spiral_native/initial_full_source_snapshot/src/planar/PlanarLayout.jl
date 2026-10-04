export PlanarLayout, build_planar_layout, planar_layout_problem

"""A rasterized physical layout with its named conductor models preserved.
`problem` contains the EM geometry; `sheet_materials` maps cells to
`material_names`. Material values are surface impedances [ohm] or
frequency callbacks. Via types specify `kind` and bulk `sigma` [S/m].
`solve_planar(layout,f)` evaluates those models at each frequency.
Interfaces are numbered bottom to top, from zero to `length(stack.layers)`.
"""
struct PlanarLayout
    problem::PlanarProblem
    shapes::Vector{PlanarShape}
    sheet_materials::Vector{Matrix{Int32}}
    material_names::Vector{String}
    materials::Vector{Any}
    via_models::Vector{Any}
    source_problem::PlanarProblem
    contraction::Union{Nothing,Matrix{Float64}}
    terminal_paths::Vector{NamedTuple}
    volume_models::Vector{Any}
end

PlanarLayout(prob::PlanarProblem,shapes::Vector{PlanarShape},ids::Vector{Matrix{Int32}},
    names::Vector{String},materials::Vector{Any},vias::Vector{Any},source::PlanarProblem,
    contraction::Union{Nothing,Matrix{Float64}},paths::Vector{NamedTuple}) =
    PlanarLayout(prob,shapes,ids,names,materials,vias,source,contraction,paths,
        Any[Inf for _ in source.vols])

PlanarLayout(prob::PlanarProblem,shapes::Vector{PlanarShape},ids::Vector{Matrix{Int32}},
    names::Vector{String},materials::Vector{Any},vias::Vector{Any}) =
    PlanarLayout(prob,shapes,ids,names,materials,vias,prob,nothing,NamedTuple[])

function _layout_pin_port(pin::PlanarPin, interfaces, grid::CellGrid, z0,sheets)
    lv = findfirst(==(pin.level), interfaces)
    lv === nothing && throw(ArgumentError("pin $(pin.name) has no sheet interface"))
    nx, ny = grid.nx, grid.ny
    tol = 1e-8 * min(grid.dx, grid.dy)
    x, y = pin.point
    dx, dy = pin.direction
    abs(dx) > 1-1e-10 && abs(dy) < 1e-10 ||
        abs(dy) > 1-1e-10 && abs(dx) < 1e-10 ||
        throw(ArgumentError("pin $(pin.name) is diagonal; axis-aligned gap required"))
    xflow = abs(dx) > .5
    coord, transverse, step, count = xflow ?
        (x, y, grid.dy, ny) : (y, x, grid.dx, nx)
    cells = findall(k -> transverse-pin.width/2-tol <= (k-.5)*step <
        transverse+pin.width/2-tol, 1:count)
    isempty(cells) && throw(ArgumentError("pin $(pin.name) vanished at this grid resolution"))
    stop = xflow ? grid.a : grid.b
    axisstep = xflow ? grid.dx : grid.dy
    if abs(coord) <= tol
        return PlanarPort(lv, xflow ? :west : :south, first(cells):last(cells), z0)
    elseif abs(coord-stop) <= tol
        return PlanarPort(lv, xflow ? :east : :north, first(cells):last(cells), z0)
    end
    edge = round(Int, coord/axisstep)
    abs(edge*axisstep-coord) <= tol || throw(ArgumentError(
        "pin $(pin.name) gap is not on a cell edge; align or refine the grid"))
    mask=sheets[lv].mask
    negative=all(t -> xflow ? mask[edge,t] : mask[t,edge],cells)
    positive=all(t -> xflow ? mask[edge+1,t] : mask[t,edge+1],cells)
    direction=xflow ? dx : dy
    if negative && positive
        return PlanarPort(lv,xflow ? :x : :y,edge,first(cells):last(cells),z0;
            polarity=direction>0 ? -1 : 1)
    end
    metal_side=direction>0 ? :negative : :positive
    (metal_side==:negative ? negative : positive) || throw(ArgumentError(
        "pin $(pin.name) does not touch its complete conductor span"))
    return PlanarPort(lv,xflow ? :terminal_x : :terminal_y,edge,
        first(cells):last(cells),z0;metal_side)
end

function _layout_wall_contacts!(sheet::SheetLevel, points, grid::CellGrid)
    tol = 1e-10 * min(grid.dx, grid.dy)
    for k in eachindex(points)
        a, b = points[k], points[mod1(k+1, length(points))]
        for (axis, coordinate, flags, step) in
                ((1, 0., sheet.connect_west, grid.dy),
                 (1, grid.a, sheet.connect_east, grid.dy),
                 (2, 0., sheet.connect_south, grid.dx),
                 (2, grid.b, sheet.connect_north, grid.dx))
            abs(a[axis]-coordinate) <= tol && abs(b[axis]-coordinate) <= tol || continue
            other = 3-axis
            low, high = minmax(a[other], b[other])
            for cell in eachindex(flags)
                low-tol <= (cell-.5)*step < high-tol && (flags[cell] = true)
            end
        end
    end
    return sheet
end

function _layout_footprint(points, grid::CellGrid, name)
    planar_normalize_polygon(points; label=name)
    tol = 1e-10 * min(grid.dx, grid.dy)
    all(p -> -tol <= p[1] <= grid.a+tol && -tol <= p[2] <= grid.b+tol, points) ||
        throw(ArgumentError("$name extends outside the analysis box"))
    scratch = sheet_level(0, grid.nx, grid.ny)
    rasterize_poly!(scratch, grid, [p[1] for p in points], [p[2] for p in points])
    any(scratch.mask) || throw(ArgumentError("$name vanished at this grid resolution"))
    return scratch.mask
end

"""
    build_planar_layout(stack,grid,shapes,ports; metals,via_types,max_bytes)

Lower library shapes to sheets and multi-layer via columns. `ports` contains
`PlanarPort`s, library pins, or `(pin,z0)` tuples. Pins on walls become wall
ports; interior pins inside continuous metal become gap ports, while open
ends receive physical PEC-cover reference-via sources. `terminal_ground`
selects `:auto`, `:below` or `:above`; reference lead effects remain until
calibrated. `problem` retains original geometry and port order, and
`source_problem` includes the physical return sources. Sheet indices in
explicit ports follow the sorted set of polygon interface numbers.

`metals` maps each polygon's metal name to a surface impedance [ohm] or
`f_hz -> impedance` callback; the default defines only PEC. `via_types` maps
names to `(kind=VIA_UNIFORM, sigma=...)`, where conductivity may also be a
frequency callback. Different metals on the same interface are preserved
cell by cell. Overlapping different materials and mixed via conductivities
within one layer reject explicitly. Geometry below the grid resolution
rejects instead of disappearing silently.
"""
function build_planar_layout(stack::PlanarStackup, grid::CellGrid,
        shapes::AbstractVector{PlanarShape}, ports::AbstractVector;
        metals::AbstractDict=Dict("pec"=>0., "PEC"=>0.),
        via_types::AbstractDict=Dict("uniform"=>(kind=VIA_UNIFORM, sigma=Inf)),
        terminal_ground::Symbol=:auto,
        _allow_portless::Bool=false,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    planar_validate(stack)
    terminal_ground in (:auto,:below,:above) || throw(ArgumentError("invalid terminal ground direction"))
    stack.a == grid.a && stack.b == grid.b ||
        throw(ArgumentError("stack and grid box sizes differ"))
    isempty(shapes) && throw(ArgumentError("layout needs at least one shape"))
    polygons = [p for s in shapes for p in s.polygons]
    allvias = [v for s in shapes for v in s.vias]
    names = [p.name for p in polygons]
    append!(names, [v.name for v in allvias])
    length(unique(names)) == length(names) || throw(ArgumentError("duplicate layout geometry names"))
    L = length(stack.layers)
    interfaces = sort!(unique([p.level for p in polygons]))
    all(i -> 0 <= i <= L, interfaces) || throw(ArgumentError("polygon interface outside 0:$L"))
    mnames = sort!(unique([p.metal for p in polygons]))
    all(n -> haskey(metals, n), mnames) || throw(ArgumentError("layout references an undefined metal"))
    length(mnames) <= typemax(Int32) || throw(ArgumentError("too many metal types"))
    # Masks, material IDs and one temporary footprint; via storage is bounded
    # by all stack layers, since columns may cross several interfaces.
    payload = _checked_payload_sum("planar layout",
        _checked_array_payload_bytes(Int32, length(interfaces), grid.nx, grid.ny),
        _checked_array_payload_bytes(UInt8, length(interfaces)+2L+1, grid.nx, grid.ny),
        _checked_array_payload_bytes(UInt8, 4length(interfaces), grid.nx+grid.ny))
    _enforce_payload_limit(payload, max_bytes, "planar layout", "max_bytes")
    sheets = [sheet_level(i, grid.nx, grid.ny) for i in interfaces]
    mids = [zeros(Int32, grid.nx, grid.ny) for _ in interfaces]
    for p in polygons
        lv = findfirst(==(p.level), interfaces)
        material = findfirst(==(p.metal), mnames)
        mask = _layout_footprint(p.vertices, grid, p.name)
        for cell in eachindex(mask)
            mask[cell] || continue
            iszero(mids[lv][cell]) || mids[lv][cell] == material ||
                throw(ArgumentError("different metals overlap in cell $cell on interface $(p.level)"))
            mids[lv][cell] = material
        end
        sheets[lv].mask .|= mask
        _layout_wall_contacts!(sheets[lv], p.vertices, grid)
    end
    levels = Dict{Int,ViaLevel}()
    sigmas = Dict{Int,Any}()
    for v in allvias
        target = v.to_level == -1 ? 0 : v.to_level
        low, high = minmax(v.from_level, target)
        0 <= low < high <= L || throw(ArgumentError("via $(v.name) interfaces outside stack or identical"))
        haskey(via_types, v.via_type) || throw(ArgumentError("undefined via type $(v.via_type)"))
        spec = via_types[v.via_type]
        kind = spec isa ViaKind ? spec : spec.kind
        sigma = spec isa ViaKind ? Inf : spec.sigma
        kind in (VIA_UNIFORM, VIA_TAPER) || throw(ArgumentError("invalid via profile"))
        mask = _layout_footprint(v.vertices, grid, v.name)
        for layer in low+1:high
            level = get!(levels, layer) do
                via_level(layer, grid.nx, grid.ny)
            end
            !haskey(sigmas, layer) || isequal(sigmas[layer], sigma) ||
                throw(ArgumentError("different via conductivities in layer $layer"))
            sigmas[layer] = sigma
            (kind == VIA_UNIFORM ? level.uni : level.tap) .|= mask
        end
    end
    layers = sort!(collect(keys(levels)))
    vias = [levels[l] for l in layers]
    ps = PlanarPort[]
    for p in ports
        if p isa PlanarPort
            push!(ps, p)
        elseif p isa PlanarPin
            push!(ps, _layout_pin_port(p, interfaces, grid, 50.,sheets))
        elseif p isa Tuple && length(p) == 2 && p[1] isa PlanarPin
            push!(ps, _layout_pin_port(p[1], interfaces, grid, p[2],sheets))
        else
            throw(ArgumentError("port must be a PlanarPort, PlanarPin, or (pin,z0) tuple"))
        end
    end
    prob = if isempty(ps) && _allow_portless
        PlanarProblem(stack,grid,sheets,ps,vias,
            build_planar_basis(grid,sheets,ps;vias),VolLevel[])
    else
        build_planar_problem(stack,grid,sheets,ps;vias)
    end
    source=prob;contraction=nothing;paths=NamedTuple[]
    models=Any[sigmas[l] for l in layers]
    direct=findall(p -> !_is_planar_terminal(p.wall),ps)
    terminal=findall(p -> _is_planar_terminal(p.wall),ps)
    if !isempty(terminal)
        physical=if isempty(direct)
            planar_terminal_returns(stack,grid,sheets,ps[terminal];vias,
                ground_direction=terminal_ground,max_bytes=max_bytes-payload)
        else
            base=build_planar_problem(stack,grid,sheets,ps[direct];vias)
            planar_terminal_returns(base,ps[terminal];ground_direction=terminal_ground,max_bytes=max_bytes-payload)
        end
        source=physical.problem
        contraction=physical.contraction[:,invperm(vcat(direct,terminal))]
        paths=NamedTuple[merge(path,(port=terminal[k],)) for (k,path) in enumerate(physical.paths)]
        append!(models,fill(Inf,length(source.vias)-length(vias)))
    end
    return PlanarLayout(prob, collect(shapes), mids, mnames,
        Any[metals[n] for n in mnames],models,source,contraction,paths)
end

"""Return a layout's geometry problem. Material models remain on the layout;
use `solve_planar(layout,f)` to include them in the solve."""
planar_layout_problem(layout::PlanarLayout) = layout.problem

function _layout_surface_values(layout::PlanarLayout, freq)
    values = ComplexF64[]
    for m in layout.materials
        z = m isa Number ? m : m(freq)
        z isa Number && isfinite(z) && real(z) >= 0 ||
            throw(ArgumentError("layout metal model must return a finite passive impedance"))
        push!(values, ComplexF64(z))
    end
    return [map(id -> iszero(id) ? 0.0im : values[id], ids)
        for ids in layout.sheet_materials]
end

"""Solve a physical library layout, evaluating named surface and bulk metal
models at `freq` [Hz] before invoking the dense or FFT EM solver."""
function solve_planar(layout::PlanarLayout, freq::Number;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,volume_sigma=nothing,kw...)
    prob = layout.source_problem
    payload = _checked_payload_sum("layout material values",
        _checked_array_payload_bytes(ComplexF64, length(prob.sheets), prob.grid.nx, prob.grid.ny),
        _checked_array_payload_bytes(ComplexF64, length(layout.materials)+length(prob.vias)+length(prob.vols)))
    _enforce_payload_limit(payload,max_bytes,"layout material values","max_bytes")
    z = _layout_surface_values(layout, freq)
    sigma = [m isa Number ? m : m(freq) for m in layout.via_models]
    bulk=volume_sigma===nothing ? [m isa Number ? m : m(freq) for m in layout.volume_models] : volume_sigma
    remaining=Int(BigInt(_validated_resource_limit("max_bytes",max_bytes))-payload)
    if layout.contraction===nothing
        return solve_planar(prob,freq;max_bytes=remaining,surface_zs=z,via_sigma=sigma,volume_sigma=bulk,kw...)
    end
    return solve_planar_contracted(prob,freq,layout.contraction;problem=layout.problem,
        z0=[p.z0 for p in layout.problem.ports],max_bytes=remaining,
        surface_zs=z,via_sigma=sigma,volume_sigma=bulk,kw...)
end

"""Frequency sweep of a physical layout with its material callbacks."""
planar_sparams(layout::PlanarLayout, freqs::AbstractVector; kw...) =
    [solve_planar(layout, f; kw...).s for f in freqs]

function planar_sweep_abs(layout::PlanarLayout, fmin::Real, fmax::Real;
        rel_tol::Real=1e-2, n_eval::Integer=257, max_points::Integer=32,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES, solve_kw...)
    fmin,fmax,n_eval,max_points=_planar_abs_parameters(fmin,fmax,rel_tol,n_eval,max_points)
    n=length(layout.problem.ports)
    storage = _checked_payload_sum("layout sweep storage",
        _planar_sweep_storage_bytes(n,n_eval,max_points),
        _checked_array_payload_bytes(ComplexF64,8,n,n),
        _checked_array_payload_bytes(ComplexF64,3,n))
    _enforce_payload_limit(storage, max_bytes, "planar layout sweep", "max_bytes")
    _planar_abs_frequency_grid(fmin,fmax,n_eval)
    remaining = _validated_resource_limit("max_bytes", max_bytes) - storage
    options = merge((retain_matrix=false, max_bytes=remaining), (; solve_kw...))
    refs=_planar_reference_values([p.z0 for p in layout.problem.ports],length(layout.problem.ports);freq=fmin)
    sample(f)=begin
        result=solve_planar(layout,f;options...)
        result.z0==refs ? result.s : planar_renormalize_s(result.s,result.z0,refs)
    end
    return _planar_sweep_abs(sample,
        length(layout.problem.ports), fmin, fmax; rel_tol, n_eval,
        max_points, max_bytes,z0=refs)
end

function planar_objective_gradient(layout::PlanarLayout, freq::Number, objective;
        params::AbstractVector{PlanarParam}=planar_default_params(layout.problem.stack),
        gY=nothing,h_fd::Real=1e-6,volume_sigma=nothing,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES, kw...)
    prob = layout.source_problem
    payload = _checked_payload_sum("layout gradient material values",
        _checked_array_payload_bytes(ComplexF64, length(prob.sheets), prob.grid.nx, prob.grid.ny),
        _checked_array_payload_bytes(ComplexF64, length(layout.materials)+length(prob.vias)+length(prob.vols)))
    C=layout.contraction
    n=length(layout.problem.ports);nr=length(prob.ports)
    reduction_payload=C===nothing ? 0 : _checked_payload_sum("layout contracted gradient",
        _checked_array_payload_bytes(ComplexF64,8,n,n),
        _checked_array_payload_bytes(ComplexF64,4,nr,nr),
        _checked_array_payload_bytes(ComplexF64,8,nr,n),
        _checked_array_payload_bytes(Float64,nr,n))
    reserved=_checked_payload_sum("layout gradient",payload,reduction_payload)
    _enforce_payload_limit(reserved, max_bytes, "layout gradient material values and port reduction", "max_bytes")
    z = _layout_surface_values(layout, freq)
    sigma = [m isa Number ? m : m(freq) for m in layout.via_models]
    bulk=volume_sigma===nothing ? [m isa Number ? m : m(freq) for m in layout.volume_models] : volume_sigma
    remaining=Int(BigInt(_validated_resource_limit("max_bytes",max_bytes))-reserved)
    if C===nothing
        return planar_objective_gradient(prob,freq,objective;params,gY,h_fd,
            max_bytes=remaining,surface_zs=z,via_sigma=sigma,volume_sigma=bulk,kw...)
    end
    baseline=Ref{Matrix{ComplexF64}}();terminal_gradient=Ref{Matrix{ComplexF64}}()
    contracted(Y)=Matrix{ComplexF64}(transpose(C)*Y*C)
    wrapped_objective(Y)=objective(contracted(Y))
    function wrapped_gradient(Y)
        Yc=contracted(Y)
        G=gY===nothing ? _planar_wirtinger_fd(objective,Yc,Float64(h_fd)) : gY(Yc)
        G isa AbstractMatrix && size(G)==(n,n) && all(isfinite,G) ||
            throw(ArgumentError("layout objective gradient must be finite and match terminal ports"))
        baseline[]=Y;terminal_gradient[]=Matrix{ComplexF64}(G)
        return C*G*transpose(C)
    end
    J,gradient=planar_objective_gradient(prob,freq,wrapped_objective;params,
        gY=wrapped_gradient,h_fd,max_bytes=remaining,surface_zs=z,via_sigma=sigma,volume_sigma=bulk,kw...)
    # Layer-height weights are part of the physical source contract. Their
    # derivative contributes in addition to the EM matrix adjoint.
    for (k,param) in enumerate(params)
        param.field==:thickness && param.part==:re || continue
        D=zeros(Float64,size(C))
        for path in layout.terminal_paths
            hasproperty(path,:layers) || continue
            param.index in path.layers || continue
            H=path.length;column=path.port
            for row in axes(C,1)
                iszero(C[row,column]) && continue
                port=prob.ports[row];layer=prob.vias[port.level].layer
                height=real(prob.stack.layers[layer].thickness)
                D[row,column]=(param.index==layer ? 1/H : 0.)-height/H^2
            end
        end
        dY=transpose(D)*baseline[]*C+transpose(C)*baseline[]*D
        gradient[k]+=real(sum(terminal_gradient[].*dY))
    end
    return J,gradient
end
