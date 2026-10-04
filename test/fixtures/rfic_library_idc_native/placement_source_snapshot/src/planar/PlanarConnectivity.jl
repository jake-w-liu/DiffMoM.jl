export planar_connectivity

@inline function _planar_net_root!(parent, i)
    while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    return i
end

@inline function _planar_net_join!(parent, a, b)
    (iszero(a) || iszero(b)) && return nothing
    if !iszero(a) && !iszero(b)
        ra = _planar_net_root!(parent, a)
        rb = _planar_net_root!(parent, b)
        parent[max(ra, rb)] = min(ra, rb)
    end
    return nothing
end

function _planar_net_lateral!(parent, labels, sheet::Int=0, cuts=nothing)
    nx, ny = size(labels)
    @inbounds for j in 1:ny, i in 1:nx
        node = labels[i,j]
        iszero(node) && continue
        i < nx && (cuts===nothing || !((sheet,:x,i,j) in cuts)) &&
            _planar_net_join!(parent, node, labels[i+1,j])
        j < ny && (cuts===nothing || !((sheet,:y,j,i) in cuts)) &&
            _planar_net_join!(parent, node, labels[i,j+1])
    end
end

"""
    planar_connectivity(problem_or_layout; max_bytes=...) -> NamedTuple

Audit physical conductor continuity on the analysis grid. Returns sheet,
via and volume component maps (zero is empty), component count, component
numbers touched by each port, components grounded at unexcited PEC wall
contacts, and via cells with a missing top or bottom conductor contact.
Diagonal corner touches do not connect conductor cells. Driven gap edges
are excluded from sidewall grounding. Internal gap ports are virtual
excitations on the continuous metal geometry.

For a `PlanarLayout`, declared polygon net names additionally identify
`open_nets` (one declared net occupies disconnected components) and
`shorted_nets` (different declared nets occupy one component). This is a
geometric continuity audit; it does not replace an electromagnetic solve.
"""
function planar_connectivity(prob::PlanarProblem;
        _sheet_cuts::AbstractVector=NamedTuple[],
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    grid = prob.grid
    nx, ny = grid.nx, grid.ny
    ns, nv, nc = length(prob.sheets), length(prob.vias), length(prob.vols)
    nmax = sum(count(s.mask) for s in prob.sheets; init=0) +
        sum(count(v.mask) for v in prob.vols; init=0) +
        sum(count(k -> v.uni[k] || v.tap[k], eachindex(v.uni)) for v in prob.vias; init=0)
    bytes = _checked_payload_sum("planar connectivity",
        _checked_array_payload_bytes(Int, ns+nv+nc, nx, ny),
        _checked_array_payload_bytes(Int, 4, nmax+1),
        _checked_array_payload_bytes(Int, planar_basis_count(prob.basis)),
        _checked_array_payload_bytes(Int,4,sum(BigInt(length(c.span)) for c in _sheet_cuts;init=BigInt(0))))
    _enforce_payload_limit(bytes, max_bytes, "planar connectivity", "max_bytes")
    sl = [zeros(Int, nx, ny) for _ in 1:ns]
    vl = [zeros(Int, nx, ny) for _ in 1:nv]
    cl = [zeros(Int, nx, ny) for _ in 1:nc]
    n = 0
    for (labels, levels, kind) in ((sl, prob.sheets, :sheet),
            (vl, prob.vias, :via), (cl, prob.vols, :volume))
        for (lv, obj) in enumerate(levels), cell in eachindex(labels[lv])
            occupied = kind == :via ? obj.uni[cell] || obj.tap[cell] : obj.mask[cell]
            occupied || continue
            n += 1
            labels[lv][cell] = n
        end
    end
    ground = n+1
    parent = collect(1:ground)
    cuts=isempty(_sheet_cuts) ? nothing : Set((c.sheet,c.axis,c.source_edge,t) for c in _sheet_cuts for t in c.span)
    for (sheet,labels) in enumerate(sl)
        _planar_net_lateral!(parent,labels,sheet,cuts)
    end
    for labels in cl
        _planar_net_lateral!(parent,labels)
    end
    # Adjacent axial cells form one conductor footprint too.
    for labels in vl
        _planar_net_lateral!(parent, labels)
    end
    for (a, sh) in enumerate(prob.sheets), (b, other) in enumerate(prob.sheets)
        a < b && sh.interface == other.interface || continue
        for cell in eachindex(sl[a])
            _planar_net_join!(parent, sl[a][cell], sl[b][cell])
        end
    end
    function contact!(node, interface, cell)
        found = false
        for (s, sheet) in enumerate(prob.sheets)
            sheet.interface == interface || continue
            other = sl[s][cell]
            iszero(other) && continue
            _planar_net_join!(parent, node, other)
            found = true
        end
        for (v, via) in enumerate(prob.vias)
            (via.layer == interface || via.layer == interface+1) || continue
            other = vl[v][cell]
            (iszero(other) || other == node) && continue
            if !iszero(other) && other != node
                _planar_net_join!(parent, node, other)
                found = true
            end
        end
        for (c, volume) in enumerate(prob.vols)
            (volume.layer == interface || volume.layer == interface+1) || continue
            other = cl[c][cell]
            (iszero(other) || other == node) && continue
            if !iszero(other) && other != node
                _planar_net_join!(parent, node, other)
                found = true
            end
        end
        if interface == 0 && prob.stack.bottom.kind == TERM_PEC ||
                interface == length(prob.stack.layers) && prob.stack.top.kind == TERM_PEC
            _planar_net_join!(parent, node, ground)
            found = true
        end
        return found
    end
    dangling = Tuple{Int,Int,Int}[]
    for (v, via) in enumerate(prob.vias), j in 1:ny, i in 1:nx
        node = vl[v][i,j]
        iszero(node) && continue
        cell = i+(j-1)*nx
        bottom = contact!(node, via.layer-1, cell)
        top = contact!(node, via.layer, cell)
        bottom && top || push!(dangling, (via.layer,i,j))
    end
    for (c, volume) in enumerate(prob.vols), cell in eachindex(cl[c])
        node = cl[c][cell]
        iszero(node) && continue
        contact!(node, volume.layer-1, cell)
        contact!(node, volume.layer, cell)
    end
    if grid.walls == WALL_PEC
        driven = Set{Tuple{Symbol,Int,Symbol,Int}}()
        for p in prob.ports
            volume=_is_planar_volume_port(p.wall)
            wall=volume ? _planar_volume_wall(p.wall) : p.wall
            wall in (:west,:east,:south,:north) || continue
            for cell in p.cells
                push!(driven, (volume ? :volume : :sheet,p.level,wall,cell))
            end
        end
        for (kind,levels,labels) in ((:sheet,prob.sheets,sl),(:volume,prob.vols,cl)),(s, sheet) in enumerate(levels)
            for (wall, flags) in ((:west,sheet.connect_west), (:east,sheet.connect_east),
                    (:south,sheet.connect_south), (:north,sheet.connect_north))
                for k in eachindex(flags)
                    flags[k] && !((kind,s,wall,k) in driven) || continue
                    i,j = wall == :west ? (1,k) : wall == :east ? (nx,k) :
                        wall == :south ? (k,1) : (k,ny)
                    _planar_net_join!(parent, labels[s][i,j], ground)
                end
            end
        end
    end
    roots = Dict{Int,Int}()
    for node in 1:n
        root = _planar_net_root!(parent, node)
        get!(roots, root, length(roots)+1)
    end
    ground_component = get(roots, _planar_net_root!(parent, ground), 0)
    for labels in (sl...,vl...,cl...), cell in eachindex(labels)
        iszero(labels[cell]) && continue
        labels[cell] = roots[_planar_net_root!(parent, labels[cell])]
    end
    pcs = [Int[] for _ in prob.ports]
    @inbounds for b in eachindex(prob.basis.port)
        port = prob.basis.port[b]
        iszero(port) && continue
        kind, lv = prob.basis.kind[b], prob.basis.level[b]
        i,j = prob.basis.ei[b], prob.basis.ej[b]
        sheetkind = _sheet_kind(kind)
        # Half rooftops store a boundary edge, not its occupied cell. In
        # particular the west/south edge is zero and cannot index a mask.
        sheetkind == _BASIS_X_LO && (i += 1)
        sheetkind == _BASIS_Y_LO && (j += 1)
        labels = _is_via_kind(kind) ? vl[lv] : _is_vol_kind(kind) ? cl[lv] : sl[lv]
        push!(pcs[port], labels[i,j])
        sheetkind == _BASIS_X_FULL && push!(pcs[port], labels[i+1,j])
        sheetkind == _BASIS_Y_FULL && push!(pcs[port], labels[i,j+1])
    end
    foreach(v -> unique!(filter!(!iszero, v)), pcs)
    return (; component_count=length(roots), sheet_components=sl,
        via_components=vl, volume_components=cl, port_components=pcs,
        grounded_components=iszero(ground_component) ? Int[] : [ground_component],
        dangling_vias=dangling, open_nets=String[], shorted_nets=Vector{String}[],
        net_components=Dict{String,Vector{Int}}())
end

function planar_connectivity(layout::PlanarLayout; kw...)
    cuts=[path for path in layout.terminal_paths if hasproperty(path,:kind) && path.kind===:floating_bridge]
    result = planar_connectivity(layout.problem;_sheet_cuts=cuts,kw...)
    bynet = result.net_components
    for shape in layout.shapes, p in shape.polygons
        isempty(p.net) && continue
        lv = findfirst(s -> s.interface == p.level, layout.problem.sheets)
        mask = _layout_footprint(p.vertices, layout.problem.grid, p.name)
        components = get!(bynet, p.net, Int[])
        for cell in eachindex(mask)
            mask[cell] && push!(components, result.sheet_components[lv][cell])
        end
    end
    foreach(unique!, values(bynet))
    for (net, components) in bynet
        length(components) > 1 && push!(result.open_nets, net)
    end
    for c in 1:result.component_count
        names = sort!([net for (net, cs) in bynet if c in cs])
        length(names) > 1 && push!(result.shorted_nets, names)
    end
    sort!(result.open_nets)
    return result
end
