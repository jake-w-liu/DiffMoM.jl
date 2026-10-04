function _project_floating_layout(project,layout,entries,ordinary;variables,max_bytes)
    floating=findall(e->e isa NamedTuple,entries)
    isempty(floating) && return layout
    total=length(entries);source=layout.source_problem
    initial_payload=_subdivision_retained_payload(Any[source.basis,source.sheets,source.vias,
        source.vols,layout.sheet_materials,layout.contraction])
    reserve=_checked_payload_sum("project floating source metadata",initial_payload,
        _checked_array_payload_bytes(Float64,3,length(source.ports)+length(floating),total),
        _checked_array_payload_bytes(Int,2,length(source.basis.kind)+
            4BigInt(source.grid.nx)*source.grid.ny*length(floating)),
        _checked_array_payload_bytes(Int32,length(source.sheets),source.grid.nx,source.grid.ny))
    _enforce_payload_limit(reserve,max_bytes,"project floating sources","max_bytes")
    C=zeros(Float64,length(source.ports),total)
    old=layout.contraction===nothing ? Matrix{Float64}(I,length(source.ports),length(ordinary)) : layout.contraction
    C[:,ordinary].=old
    metadata=Vector{PlanarPort}(undef,total)
    metadata[ordinary].=layout.problem.ports
    mids=[copy(ids) for ids in layout.sheet_materials]
    names=copy(layout.material_names);materials=copy(layout.materials)
    paths=copy(layout.terminal_paths);shapes=copy(layout.shapes)
    for (j,k) in enumerate(floating)
        spec=entries[k]
        function endpoint(pin)
            polygon=only(p for shape in layout.shapes for p in shape.polygons if p.name==pin.polygon)
            mapped=PlanarPin(pin.name,pin.polygon,pin.edge,polygon.level,pin.point,pin.direction,pin.width)
            return _layout_pin_port(mapped,[s.interface for s in source.sheets],source.grid,spec.z0,source.sheets)
        end
        signal=endpoint(spec.signal);reference=endpoint(spec.reference)
        built=planar_floating_bridge(source,signal,reference;
            source_edge=spec.source_edge,z0=spec.z0,max_bytes=max_bytes-reserve)
        all(!iszero,built.old_port_map) || throw(ArgumentError("project floating endpoint overlaps an existing source"))
        next=zeros(Float64,length(built.problem.ports),total)
        next[built.old_port_map,:].=C;next[built.port_index,k]=1
        C=next;source=built.problem;metadata[k]=source.ports[built.port_index]
        metal=spec.metal
        id=findfirst(==(metal),names)
        if id===nothing
            push!(names,metal)
            push!(materials,lowercase(metal)=="pec" ? 0. :
                f->planar_project_metal_zs(project,metal,f;variables))
            id=length(names)
        end
        for (i,t) in built.bridge.cells
            iszero(mids[built.bridge.sheet][i,t]) || throw(ArgumentError("floating bridge overlaps an existing material cell"))
            mids[built.bridge.sheet][i,t]=Int32(id)
        end
        bridge=built.bridge;g=source.grid
        low,high=minmax(bridge.signal.edge,bridge.reference.edge)
        firstcell,lastcell=first(bridge.span),last(bridge.span)
        x0,x1,y0,y1=bridge.axis===:x ?
            (low*g.dx,high*g.dx,(firstcell-1)*g.dy,lastcell*g.dy) :
            ((firstcell-1)*g.dx,lastcell*g.dx,low*g.dy,high*g.dy)
        polygon=PlanarShapePolygon("__floating_bridge_$k",bridge.interface,metal,"",
            [_P2(x0,y0),_P2(x1,y0),_P2(x1,y1),_P2(x0,y1)])
        push!(shapes,PlanarShape("__floating_source_$k",[polygon],PlanarShapeVia[],PlanarPin[],Dict{String,Float64}()))
        push!(paths,merge(bridge,(kind=:floating_bridge,port=k,metal=metal)))
    end
    basis=source.basis
    metadata_basis=PlanarBasisSet(basis.kind,basis.ei,basis.ej,basis.level,basis.x0,basis.y0,basis.width,
        [iszero(p) ? 0 : something(findfirst(!iszero,view(C,p,:)),0) for p in basis.port])
    original=PlanarProblem(source.stack,source.grid,source.sheets,metadata,source.vias,metadata_basis,source.vols)
    return PlanarLayout(original,shapes,mids,names,materials,layout.via_models,source,C,paths,layout.volume_models)
end
