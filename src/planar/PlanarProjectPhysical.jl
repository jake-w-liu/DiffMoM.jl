# Physical conductor lowering for declarative projects. Interfaces and
# source voltages are remapped together when conductor faces split a layer.

function _project_half_film(project,name,freq,variables)
    table=project.data["metals"][name]
    q(v,d)=planar_project_value(project,v;dimension=d,freq,variables)
    layers=PlanarConductorLayer[PlanarConductorLayer(q(t["conductivity"],:conductivity),
        q(t["thickness"],:length)/2;mur=q(get(t,"mu_r",1.),:dimensionless))
        for t in get(table,"plating",Any[])]
    push!(layers,PlanarConductorLayer(q(table["conductivity"],:conductivity),
        planar_project_value(project,table["thickness"];dimension=:length,variables)/2;
        mur=q(get(table,"mu_r",1.),:dimensionless)))
    return planar_layered_surface_zs(freq,layers;
        roughness=_project_roughness(project,get(table,"roughness",nothing),freq,variables),
        loss_only=get(get(table,"roughness",Dict()),"loss_only",false))
end

function _project_physical_layout(project,stack,grid,shape,entries;
        freq,variables,metals,via_types,terminal_ground,max_bytes,_allow_portless=false)
    tables=get(project.data,"metals",Dict())
    physical=Dict{String,NamedTuple}()
    z=Float64.(real.(planar_interfaces(stack)))
    newz=copy(z)
    q(v,d=:dimensionless)=planar_project_value(project,v;dimension=d,variables)
    for p in shape.polygons
        table=get(tables,p.metal,Dict())
        kind=get(table,"type","lossless")
        kind in ("thick","volume") || continue
        direction=get(table,"direction","up")
        direction in ("up","down") || throw(ArgumentError("physical metal direction must be up or down"))
        thickness=_project_real(q(table["thickness"],:length),"physical metal thickness";positive=true)
        kind=="thick" && (thickness+=sum(_project_real(q(t["thickness"],:length),
            "plating thickness";positive=true) for t in get(table,"plating",Any[]);init=0.))
        host=direction=="up" ? p.level+1 : p.level
        1<=host<=length(stack.layers) || throw(ArgumentError("physical metal $(p.name) extends outside the box"))
        base=z[p.level+1];face=base+(direction=="up" ? thickness : -thickness)
        tol=64eps(maximum(z));nearest=argmin(abs.(z.-face))
        abs(z[nearest]-face)<=tol && (face=z[nearest])
        z[host]<=face<=z[host+1] || throw(ArgumentError("physical metal $(p.name) crosses its host dielectric layer"))
        subdivisions=_project_int(q(get(table,"axial_subdivisions",1)),"metal axial subdivisions";positive=true)
        kind=="thick" && subdivisions!=1 && throw(ArgumentError("two-face metal uses one physical slab; use volume for axial refinement"))
        _enforce_payload_limit(_checked_array_payload_bytes(Float64,subdivisions+1),max_bytes,
            "physical conductor axial coordinates","max_bytes")
        boundaries=collect(range(min(base,face),max(base,face);length=subdivisions+1))
        if kind=="volume"
            isempty(get(table,"plating",Any[])) && !haskey(table,"roughness") || throw(ArgumentError(
                "uniform volume metal needs explicit separate prisms for plating; roughness requires a resolved surface model"))
            q(get(table,"mu_r",1.))==1 || throw(ArgumentError("local magnetic volume metal requires a magnetic-material adapter"))
        end
        physical[p.name]=(kind=kind,base=base,face=face,boundaries=boundaries,metal=p.metal)
        append!(newz,boundaries)
    end
    isempty(physical) && return (build_planar_layout(stack,grid,[shape],entries;
        metals,via_types,terminal_ground,max_bytes,_allow_portless),Float64[])
    sort!(newz);unique!(newz)
    at(zvalue)=something(findfirst(==(zvalue),newz),0)-1
    maplevel(level)=at(z[level+1])
    layers=PlanarLayer[]
    for k in 1:length(newz)-1
        old=stack.layers[searchsortedlast(z,(newz[k]+newz[k+1])/2)]
        push!(layers,PlanarLayer(old.epsr,old.mur,newz[k+1]-newz[k];epsr_z=old.epsr_z,mur_z=old.mur_z))
    end
    expanded=PlanarStackup(layers,stack.bottom,stack.top,stack.a,stack.b)
    polynomials=PlanarShapePolygon[]
    for p in shape.polygons
        spec=get(physical,p.name,nothing)
        spec!==nothing && spec.kind=="volume" && continue
        push!(polynomials,PlanarShapePolygon(p.name,maplevel(p.level),p.metal,p.net,p.vertices))
        if spec!==nothing
            face_name=p.name*"__physical_face"
            face_name in (s.name for s in shape.polygons) && throw(ArgumentError("generated conductor face name collides"))
            push!(polynomials,PlanarShapePolygon(face_name,at(spec.face),p.metal,p.net,p.vertices))
            metals[p.metal]=f->_project_half_film(project,p.metal,f,variables)
        end
    end
    allvias=[PlanarShapeVia(v.name,v.via_type,maplevel(v.from_level),maplevel(v.to_level),v.vertices) for v in shape.vias]
    interfaces=sort!(unique(p.level for p in polynomials))
    # Reserve masks, IDs, source contracts and the basis payload before any
    # new full-grid conductor allocation. The bound counts all layer cells.
    cells=_checked_array_payload_bytes(UInt8,grid.nx,grid.ny)
    nvol=sum(length(s.boundaries)-1 for s in values(physical) if s.kind=="volume";init=0)
    nraw=sum(e isa PlanarPort ? length(newz) : begin
        spec=get(physical,e[1].polygon,nothing)
        spec===nothing ? 1 : spec.kind=="thick" ? 2 : length(spec.boundaries)-1
    end for e in entries;init=0)
    reserve=_checked_payload_sum("project physical conductors",
        _checked_array_payload_bytes(UInt8,length(interfaces)+2length(layers)+nvol+1,grid.nx,grid.ny),
        _checked_array_payload_bytes(Int32,length(interfaces),grid.nx,grid.ny),
        _checked_array_payload_bytes(Float64,nraw,length(entries)),
        _checked_array_payload_bytes(UInt8,57,6BigInt(cells)*(length(interfaces)+length(layers)+nvol)),
        _checked_array_payload_bytes(Int,4,length(interfaces)+nvol,grid.nx+grid.ny))
    _enforce_payload_limit(reserve,max_bytes,"project physical conductors","max_bytes")
    sheets=[sheet_level(i,grid.nx,grid.ny) for i in interfaces]
    mids=[zeros(Int32,grid.nx,grid.ny) for _ in interfaces]
    mnames=sort!(unique(p.metal for p in polynomials))
    for p in polynomials
        lv=findfirst(==(p.level),interfaces);mid=findfirst(==(p.metal),mnames)
        mask=_layout_footprint(p.vertices,grid,p.name)
        for c in eachindex(mask)
            mask[c] || continue
            mids[lv][c] in (0,mid) || throw(ArgumentError("different physical metals overlap on interface $(p.level)"))
            mids[lv][c]=mid
        end
        sheets[lv].mask .|=mask;_layout_wall_contacts!(sheets[lv],p.vertices,grid)
    end
    vols=VolLevel[];bulk=Float64[];volume_for=Dict{String,Vector{Int}}()
    bulk_models=Any[];bulk_for=Dict{String,Any}()
    for p in shape.polygons
        spec=get(physical,p.name,nothing)
        (spec===nothing || spec.kind!="volume") && continue
        mask=_layout_footprint(p.vertices,grid,p.name)
        sigma=_project_real(planar_project_value(project,tables[p.metal]["conductivity"];
            dimension=:conductivity,freq,variables),"volume conductivity";positive=true)
        model=get!(bulk_for,p.metal) do
            conductivity=tables[p.metal]["conductivity"]
            conductivity isa AbstractString && occursin(r"\bfreq\b",conductivity) ?
                f->_project_real(planar_project_value(project,conductivity;
                    dimension=:conductivity,freq=f,variables),"volume conductivity";positive=true) : sigma
        end
        ids=Int[]
        for layer in at(first(spec.boundaries))+1:at(last(spec.boundaries))
            any(v->v.layer==layer && any(v.mask .&mask),vols) && throw(ArgumentError("volume prisms overlap"))
            vol=vol_level(layer,grid.nx,grid.ny);vol.mask .|=mask
            _layout_wall_contacts!(SheetLevel(layer,vol.mask,vol.connect_west,vol.connect_east,
                vol.connect_south,vol.connect_north),p.vertices,grid)
            push!(vols,vol);push!(bulk,sigma);push!(bulk_models,model);push!(ids,length(vols))
        end
        volume_for[p.name]=ids
    end
    vlevels=Dict{Int,ViaLevel}();sigmas=Dict{Int,Any}()
    function addvia!(layer,mask,kind,sigma)
        haskey(sigmas,layer) && !isequal(sigmas[layer],sigma) && throw(ArgumentError(
            "different via conductivities share physical layer $layer; use separate axial material levels"))
        sigmas[layer]=sigma
        level=get!(vlevels,layer) do;via_level(layer,grid.nx,grid.ny);end
        (kind==VIA_UNIFORM ? level.uni : level.tap) .|=mask
    end
    for v in allvias
        spec=via_types[v.via_type];mask=_layout_footprint(v.vertices,grid,v.name)
        for layer in min(v.from_level,v.to_level)+1:max(v.from_level,v.to_level)
            addvia!(layer,mask,spec.kind,spec.sigma)
        end
    end
    # Three-component volume current: the same finite bulk resistivity
    # applies to Jx/Jy rooftops and both axial profiles. No PEC or surface
    # impedance faces are added to this volume representation.
    for (k,vol) in enumerate(vols)
        addvia!(vol.layer,vol.mask,VIA_UNIFORM,bulk_models[k])
        addvia!(vol.layer,vol.mask,VIA_TAPER,bulk_models[k])
    end
    # Perimeter ties enforce equipotential physical faces without a PEC
    # sheet through the slab; interior volume metal has no sheet shunt.
    for p in shape.polygons
        spec=get(physical,p.name,nothing)
        (spec===nothing || spec.kind!="thick") && continue
        mask=_layout_footprint(p.vertices,grid,p.name);ring=falses(size(mask))
        for j in 1:grid.ny,i in 1:grid.nx
            ring[i,j]=mask[i,j] && (i==1 || j==1 || i==grid.nx || j==grid.ny ||
                !mask[i-1,j] || !mask[i+1,j] || !mask[i,j-1] || !mask[i,j+1])
        end
        for layer in min(at(spec.base),at(spec.face))+1:max(at(spec.base),at(spec.face))
            addvia!(layer,ring,VIA_UNIFORM,Inf)
        end
    end
    vlayers=sort!(collect(keys(vlevels)));vias=[vlevels[l] for l in vlayers]
    rawports=PlanarPort[];owner=Int[]
    old_vlayers=sort!(unique([l for v in shape.vias for l in min(v.from_level,v.to_level)+1:max(v.from_level,v.to_level)]))
    for (k,entry) in enumerate(entries)
        if entry isa PlanarPort
            entry.wall===:via || throw(ArgumentError("physical project explicit port requires its named axial geometry"))
            oldlayer=old_vlayers[entry.level]
            for layer in maplevel(oldlayer-1)+1:maplevel(oldlayer)
                push!(rawports,PlanarPort(findfirst(==(layer),vlayers),:via,entry.cells,entry.z0;
                    polarity=entry.polarity));push!(owner,k)
            end
            continue
        end
        pin,z0=entry;spec=get(physical,pin.polygon,nothing)
        if spec!==nothing && spec.kind=="volume"
            for lv in volume_for[pin.polygon]
                vol=vols[lv];fake=SheetLevel(pin.level,vol.mask,vol.connect_west,vol.connect_east,vol.connect_south,vol.connect_north)
                port=_layout_pin_port(pin,[pin.level],grid,z0,[fake])
                port.wall in (:west,:east,:south,:north) || throw(ArgumentError("volume interior ports require explicit conductive terminal geometry"))
                wall=port.wall===:west ? :volume_west : port.wall===:east ? :volume_east :
                    port.wall===:south ? :volume_south : :volume_north
                push!(rawports,PlanarPort(lv,wall,port.cells,z0;polarity=port.polarity));push!(owner,k)
            end
        else
            levels=spec===nothing ? [maplevel(pin.level)] : [maplevel(pin.level),at(spec.face)]
            for lv in levels
                nextpin=PlanarPin(pin.name,pin.polygon,pin.edge,lv,pin.point,pin.direction,pin.width)
                push!(rawports,_layout_pin_port(nextpin,interfaces,grid,z0,sheets));push!(owner,k)
            end
        end
    end
    C=zeros(Float64,length(rawports),length(entries))
    for (r,k) in enumerate(owner)
        C[r,k]=1
        if entries[k] isa PlanarPort
            layer=vias[rawports[r].level].layer;oldlayer=old_vlayers[entries[k].level]
            C[r,k]=real(expanded.layers[layer].thickness/stack.layers[oldlayer].thickness)
        end
    end
    raw=isempty(rawports) && _allow_portless ?
        PlanarProblem(expanded,grid,sheets,rawports,vias,build_planar_basis(grid,sheets,rawports;vias,vols),vols) :
        build_planar_problem(expanded,grid,sheets,rawports;vias,vols)
    paths=NamedTuple[];models=Any[sigmas[l] for l in vlayers]
    terminal=findall(p->_is_planar_terminal(p.wall),rawports)
    if !isempty(terminal)
        direct=findall(p->!_is_planar_terminal(p.wall),rawports)
        returned=isempty(direct) ? planar_terminal_returns(expanded,grid,sheets,rawports[terminal];vias,vols,
            ground_direction=terminal_ground,max_bytes=max_bytes-reserve) :
            planar_terminal_returns(build_planar_problem(expanded,grid,sheets,rawports[direct];vias,vols),
                rawports[terminal];ground_direction=terminal_ground,max_bytes=max_bytes-reserve)
        raw=returned.problem;C=returned.contraction[:,invperm(vcat(direct,terminal))]*C
        paths=returned.paths;append!(models,fill(Inf,length(raw.vias)-length(vias)))
    end
    firstports=[rawports[findfirst(==(k),owner)] for k in eachindex(entries)]
    meta_basis=PlanarBasisSet(raw.basis.kind,raw.basis.ei,raw.basis.ej,raw.basis.level,
        raw.basis.x0,raw.basis.y0,raw.basis.width,
        [iszero(p) ? 0 : something(findfirst(!iszero,view(C,p,:)),0) for p in raw.basis.port])
    meta=PlanarProblem(raw.stack,raw.grid,raw.sheets,firstports,raw.vias,meta_basis,raw.vols)
    expanded_shape=PlanarShape(shape.name,vcat(polynomials,[p for p in shape.polygons if haskey(volume_for,p.name)]),
        allvias,shape.pins,shape.meta)
    return PlanarLayout(meta,[expanded_shape],mids,mnames,Any[metals[n] for n in mnames],
        models,raw,C,paths,bulk_models),bulk
end
