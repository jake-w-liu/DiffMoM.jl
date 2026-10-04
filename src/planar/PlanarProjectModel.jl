export PlanarProjectModel,planar_project_layout,planar_project_frequencies
export planar_project_metal_zs

function _project_stored_frequency(freq::Real)
    stored=_circuit_stored_real(freq,"project frequency")
    stored>0 || throw(ArgumentError("project frequency must be positive"))
    return stored
end

"""Lowered project with its physical layout, ordered port names/numbers,
external port indices, component tables and the resolved variable context.
Materials are evaluated at the supplied frequency when the stack is built;
lower again at each sweep point to preserve dispersive dielectric fields."""
struct PlanarProjectModel
    project::PlanarProject
    layout::PlanarLayout
    port_names::Vector{String}
    port_numbers::Vector{Int}
    external::Vector{Int}
    z0::Vector{ComplexF64}
    components::Vector{Dict{String,Any}}
    variables::Dict{String,Any}
    volume_sigma::Vector{Float64}
    node_names::Vector{String}
    port_terminals::Vector{Tuple{Int,Int}}
    source_incidence::Matrix{Float64}
end

function _project_reference_value(project,spec,freq,variables)
    q(value,dimension)=planar_project_value(project,value;dimension,freq,variables)
    if spec isa AbstractDict
        _project_keys(spec,["r","x","l","c","topology"],"reference impedance")
        return PlanarPortImpedance(r=q(get(spec,"r",50.),:resistance),
            x=q(get(spec,"x",0.),:resistance),l=q(get(spec,"l",0.),:inductance),
            c=q(get(spec,"c",0.),:capacitance),topology=Symbol(get(spec,"topology","series")))(freq)
    end
    return _planar_reference_number(q(spec,:resistance))
end

function _project_port_reference(project,port,freq,variables)
    haskey(port,"impedance") && return _project_reference_value(project,port["impedance"],freq,variables)
    return _project_reference_value(project,Dict("r"=>get(port,"resistance",50.),
        "x"=>get(port,"reactance",0.),"l"=>get(port,"inductance",0.),
        "c"=>get(port,"capacitance",0.)),freq,variables)
end

function _project_real(value,label;positive=false,nonnegative=false)
    value isa Real && isfinite(value) && (!positive || value>0) &&
        (!nonnegative || value>=0) || throw(ArgumentError("invalid $label"))
    return Float64(value)
end
function _project_int(value,label;positive=false)
    value isa Real && isfinite(value) && isinteger(value) &&
        (!positive || value>0) && typemin(Int)<=value<=typemax(Int) ||
        throw(ArgumentError("$label must be an in-range integer"))
    return Int(value)
end

function _project_passive_relative(value,label)
    value isa Number && isfinite(value) && real(value)>0 && imag(value)<=0 ||
        throw(ArgumentError("$label must have positive real part and nonpositive passive imaginary part"))
    stored=ComplexF64(value)
    isfinite(stored) && real(stored)>0 || throw(ArgumentError("$label must remain finite in Float64"))
    return stored
end

function _project_roughness(project,table,freq,variables)
    table===nothing && return nothing
    _project_keys(table,["model","rms","rf","radius","density","loss_only"],"roughness")
    q(v,d)=planar_project_value(project,v;dimension=d,variables)
    model=get(table,"model","")
    model=="hammerstad" && return HammerstadRoughness(q(table["rms"],:length);
        rf=q(get(table,"rf",2.),:dimensionless))
    model=="huray" && return HurayRoughness(q(table["radius"],:length),q(table["density"],:density))
    throw(ArgumentError("unknown project roughness model $model"))
end

"""Evaluate a named project metal surface impedance [ohm]. Supported
models are PEC/lossless, declared `rs+i(xs+ωls)`, DC sheet `1/(σt)`,
one-face finite-film (`thick`) and bulk skin impedance. `plating` is an
outer-to-inner finite conductor stack; Hammerstad/Huray roughness uses
the declared `loss_only` convention. This utility evaluates the full
one-face film impedance; project geometry lowering divides a `thick`
conductor into its two physical half-film faces. `volume` uses bulk
resistivity in its three current components and has no surface impedance."""
function planar_project_metal_zs(project::PlanarProject,name::AbstractString,freq::Real;
        variables::AbstractDict=Dict{String,Any}())
    freq=_project_stored_frequency(freq)
    metals=get(project.data,"metals",Dict("pec"=>Dict("type"=>"lossless")))
    haskey(metals,name) || throw(ArgumentError("undefined project metal $name"))
    table=_project_keys(metals[name],["type","conductivity","thickness","mu_r",
        "rs","xs","ls","roughness","plating","direction","axial_subdivisions"],"metal $name")
    kind=get(table,"type","lossless")
    kind in ("lossless","pec") && return 0.0im
    kind=="volume" && throw(ArgumentError("volume metal uses physical bulk resistivity, not a surface impedance"))
    q(v,d;dynamic=true)=planar_project_value(project,v;dimension=d,
        freq=dynamic ? freq : nothing,variables)
    if kind in ("surface_impedance","resistive")
        isempty(get(table,"plating",Any[])) && get(table,"roughness",nothing)===nothing ||
            throw(ArgumentError("declared surface impedance cannot also declare plating/roughness"))
        r=_project_real(q(get(table,"rs",0.),:resistance),"metal resistance";nonnegative=true)
        x=_project_real(q(get(table,"xs",0.),:resistance),"surface reactance")
        inductance=_project_real(q(get(table,"ls",0.),:inductance),"surface inductance")
        z=complex(r,x+2pi*freq*inductance)
        isfinite(z) || throw(ArgumentError("surface impedance must remain finite"))
        return ComplexF64(z)
    end
    kind in ("sheet","thick","bulk") || throw(ArgumentError("unknown project metal type $kind"))
    sigma=_project_real(q(table["conductivity"],:conductivity),"metal conductivity";positive=true)
    mur=_project_real(q(get(table,"mu_r",1.),:dimensionless),"metal permeability";positive=true)
    thickness=_project_real(q(get(table,"thickness",0.),:length;dynamic=false),"metal thickness";nonnegative=true)
    rough=_project_roughness(project,get(table,"roughness",nothing),freq,variables)
    loss_only=get(get(table,"roughness",Dict()),"loss_only",false)
    loss_only isa Bool || throw(ArgumentError("roughness loss_only must be Boolean"))
    plating=get(table,"plating",Any[])
    plating isa AbstractVector || throw(ArgumentError("metal plating must be an array"))
    if kind=="sheet" && isempty(plating) && rough===nothing
        thickness>0 || throw(ArgumentError("sheet metal needs positive thickness"))
        return ComplexF64(inv(sigma)/thickness)
    end
    layers=PlanarConductorLayer[]
    for layer in plating
        _project_keys(layer,["conductivity","thickness","mu_r"],"plating layer")
        push!(layers,PlanarConductorLayer(q(layer["conductivity"],:conductivity),
            q(layer["thickness"],:length;dynamic=false);
            mur=q(get(layer,"mu_r",1.),:dimensionless)))
    end
    if kind=="bulk" || thickness==0
        return ComplexF64(planar_layered_surface_zs(freq,layers;
            substrate_sigma=sigma,substrate_mur=mur,roughness=rough,loss_only))
    end
    push!(layers,PlanarConductorLayer(sigma,thickness;mur))
    return ComplexF64(planar_layered_surface_zs(freq,layers;roughness=rough,loss_only))
end

function _project_cover(project,value,freq,variables)
    value isa AbstractString && value in ("pec","lossless","ground") && return TERM_GND
    value=="pmc" && return PlanarTerminator(TERM_PMC)
    value isa AbstractString && value in ("open","free_space") && return TERM_SPACE
    table=_project_keys(value,["type","metal","zs","eps_r","mu_r"],"box cover")
    kind=get(table,"type","surface")
    if kind=="surface"
        z=haskey(table,"metal") ? planar_project_metal_zs(project,table["metal"],freq;variables) :
            planar_project_value(project,table["zs"];dimension=:resistance,freq,variables)
        return PlanarTerminator{ComplexF64}(TERM_SURFACE,z,1.,1.)
    elseif kind=="open"
        return PlanarTerminator{ComplexF64}(TERM_OPEN,0.,
            planar_project_value(project,get(table,"eps_r",1.);freq,variables),
            planar_project_value(project,get(table,"mu_r",1.);freq,variables))
    end
    throw(ArgumentError("unsupported project cover $kind"))
end

function _project_point(project,point,variables)
    point isa AbstractVector && length(point)==2 || throw(ArgumentError("geometry point must have two coordinates"))
    return _P2((_project_real(planar_project_value(project,x;dimension=:length,variables),"coordinate") for x in point)...)
end

function _project_geometry_spec(project,entry,kind)
    allowed=kind=="polygon" ? ["name","level","metal","net","vertices","tech_layer"] :
        ["name","from_level","to_level","via_type","vertices","center","diameter","segments","tech_layer"]
    _project_keys(entry,allowed,kind)
    if haskey(entry,"tech_layer")
        name=entry["tech_layer"];tech=get(project.data,"tech_layers",Dict())
        haskey(tech,name) || throw(ArgumentError("undefined technology layer $name"))
        table=_project_keys(tech[name],["kind","level","metal","from_level","to_level","via_type",
            "stream_layer","datatype"],"technology layer $name")
        get(table,"kind","metal")== (kind=="polygon" ? "metal" : "via") ||
            throw(ArgumentError("geometry and technology-layer kinds differ"))
        return merge(Dict(k=>v for (k,v) in table if !(k in ("kind","stream_layer","datatype"))),entry)
    end
    return entry
end

"""Lower named project geometry, dispersive layers and conductor models
to a physical [`PlanarLayout`](@ref), preserving component anchor ports.
`grid=(nx,ny)` overrides `[mesh]`; absent both, 64 by 64 is the declared
default. Automatic margins translate geometry into a nonnegative box.
Open-edge ports use actual PEC-cover returns. Material expressions are
reevaluated at `freq`; geometry and thickness may not reference `freq`."""
function planar_project_layout(project::PlanarProject;freq::Real=1e9,grid=nothing,
        variables::AbstractDict=Dict{String,Any}(),
        terminal_ground::Symbol=:auto,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    freq=_project_stored_frequency(freq)
    _project_schema(project.data)
    resolve=_project_variables(project,variables)
    unit=get(get(project.data,"project",Dict()),"unit","m")
    q(v,d=:dimensionless;dynamic=false)=_project_quantity(v,unit,d,resolve,dynamic ? freq : nothing)
    function point(value)
        value isa AbstractVector && length(value)==2 || throw(ArgumentError("geometry point must have two coordinates"))
        return _P2((_project_real(q(x,:length),"coordinate") for x in value)...)
    end
    layers_raw=project.data["stackup"]["layers"];L=length(layers_raw)
    ascent=get(project.data["stackup"],"order","bottom_to_top")=="ascent"
    level(v)=v=="gnd" ? 0 : v=="top" ? L : begin
        i=_project_int(q(v),"interface"); ascent ? L-1-i : i
    end
    polygons=PlanarShapePolygon[];vias=PlanarShapeVia[]
    vertices_payload=sum(length(get(p,"vertices",Any[])) for p in get(project.data,"polygons",Any[]);init=0)+
        sum(haskey(v,"vertices") ? length(v["vertices"]) :
            _project_int(q(get(v,"segments",32)),"via segments";positive=true)
            for v in get(project.data,"vias",Any[]);init=0)
    _enforce_payload_limit(_checked_array_payload_bytes(Float64,4,vertices_payload),
        max_bytes,"project geometry coordinates","max_bytes")
    for raw in project.data["polygons"]
        p=_project_geometry_spec(project,raw,"polygon")
        name=String(p["name"]);points=[point(v) for v in p["vertices"]]
        push!(polygons,_lib_polygon(name,level(p["level"]),String(get(p,"metal","pec")),String(get(p,"net","")),points))
    end
    for raw in get(project.data,"vias",Any[])
        v=_project_geometry_spec(project,raw,"via")
        points=if haskey(v,"vertices")
            [point(p) for p in v["vertices"]]
        else
            center=point(v["center"])
            radius=_project_real(q(v["diameter"],:length),"via diameter";positive=true)/2
            count=_project_int(q(get(v,"segments",32)),"via segments";positive=true)
            count>=4 || throw(ArgumentError("circular via needs at least four segments"))
            [center+radius*_P2(cospi(2k/count),sinpi(2k/count)) for k in 0:count-1]
        end
        planar_normalize_polygon(points;label=String(v["name"]))
        push!(vias,PlanarShapeVia(String(v["name"]),String(get(v,"via_type","uniform")),
            level(v["from_level"]),level(v["to_level"]),points))
    end
    box=get(project.data,"box",Dict());shift=_P2(0.,0.)
    if haskey(box,"auto_margin")
        haskey(box,"width") || haskey(box,"length") ? throw(ArgumentError("auto_margin cannot also specify box dimensions")) : nothing
        margin=_project_real(q(box["auto_margin"],:length),"box margin";nonnegative=true)
        points=[point for p in polygons for point in p.vertices]
        append!(points,[point for v in vias for point in v.vertices])
        lo=_P2(minimum(p->p[1],points),minimum(p->p[2],points))
        hi=_P2(maximum(p->p[1],points),maximum(p->p[2],points))
        shift=_P2(margin,margin)-lo;a,b=hi-lo+2*_P2(margin,margin)
        polygons=[PlanarShapePolygon(p.name,p.level,p.metal,p.net,[v+shift for v in p.vertices]) for p in polygons]
        vias=[PlanarShapeVia(v.name,v.via_type,v.from_level,v.to_level,[p+shift for p in v.vertices]) for v in vias]
    else
        a=_project_real(q(box["width"],:length),"box width";positive=true)
        b=_project_real(q(box["length"],:length),"box length";positive=true)
    end
    layers=PlanarLayer[]
    for raw in (ascent ? reverse(layers_raw) : layers_raw)
        material=get(raw,"material",nothing)
        defaults=if material===nothing
            Dict{String,Any}()
        elseif haskey(get(project.data,"materials",Dict()),material)
            _project_keys(project.data["materials"][material],["eps_r","eps_r_z","tan_delta","tan_delta_z",
                "mu_r","mu_r_z","conductivity","conductivity_z"],"material $material")
        else
            preset=planar_material_preset(material)
            Dict{String,Any}("eps_r"=>preset.eps_r,"tan_delta"=>preset.tan_delta)
        end
        t=merge(defaults,raw)
        er=_project_passive_relative(q(get(t,"eps_r",1.);dynamic=true),"relative permittivity")
        erz=haskey(t,"eps_r_z") ? _project_passive_relative(q(t["eps_r_z"];dynamic=true),"axial permittivity") : er
        td=_project_real(q(get(t,"tan_delta",0.);dynamic=true),"loss tangent";nonnegative=true)
        tdz=_project_real(q(get(t,"tan_delta_z",td);dynamic=true),"axial loss tangent";nonnegative=true)
        sigma=_project_real(q(get(t,"conductivity",0.),:conductivity;dynamic=true),"dielectric conductivity";nonnegative=true)
        sigmaz=_project_real(q(get(t,"conductivity_z",sigma),:conductivity;dynamic=true),"axial dielectric conductivity";nonnegative=true)
        mu=_project_passive_relative(q(get(t,"mu_r",1.);dynamic=true),"permeability")
        muz=haskey(t,"mu_r_z") ? _project_passive_relative(q(t["mu_r_z"];dynamic=true),"axial permeability") : mu
        push!(layers,PlanarLayer(er-im*real(er)*td-im*sigma/(2pi*freq*_EPS0),mu,
            _project_real(q(t["thickness"],:length),"layer thickness";positive=true);
            epsr_z=erz-im*real(erz)*tdz-im*sigmaz/(2pi*freq*_EPS0),mur_z=muz))
    end
    stack=PlanarStackup(layers,_project_cover(project,get(box,"bottom_cover","pec"),freq,variables),
        _project_cover(project,get(box,"top_cover","pec"),freq,variables),a,b)
    mesh=get(project.data,"mesh",Dict());counts=grid===nothing ? (get(mesh,"nx",64),get(mesh,"ny",64)) : grid
    length(counts)==2 || throw(ArgumentError("project grid needs nx,ny"))
    walls=get(box,"sidewalls","pec");walls in ("pec","pmc") || throw(ArgumentError("invalid project sidewalls"))
    cellgrid=CellGrid(a,b,_project_int(q(counts[1]),"nx";positive=true),
        _project_int(q(counts[2]),"ny";positive=true);walls=walls=="pec" ? WALL_PEC : WALL_PMC)
    ports_raw=project.data["ports"];pins=PlanarPin[];entries=Any[]
    names=String[];numbers=Int[];refs=ComplexF64[];planes=Any[]
    via_layers=sort!(unique([l for v in vias for l in min(v.from_level,v.to_level)+1:max(v.from_level,v.to_level)]))
    for (k,p) in enumerate(ports_raw)
        _project_keys(p,["name","number","type","polygon","edge","via","layer","cells",
            "ref_polygon","ref_edge","source_edge","bridge_metal",
            "nodes",
            "resistance","reactance","inductance","capacitance","impedance","external","refplane","polarity"],"port")
        if get(p,"type","box_wall")=="via"
            vi=findfirst(v->v.name==p["via"],vias)
            vi===nothing && throw(ArgumentError("axial port references undefined via"))
            layer=_project_int(q(p["layer"]),"axial source layer";positive=true)
            via=vias[vi]
            min(via.from_level,via.to_level)<layer<=max(via.from_level,via.to_level) ||
                throw(ArgumentError("axial source layer lies outside its named via"))
            span=p["cells"]
            span isa AbstractVector && length(span)==2 || throw(ArgumentError("via cells must be [first,last] column-major indices"))
            firstcell,lastcell=(_project_int(q(c),"via source cell";positive=true) for c in span)
            1<=firstcell<=lastcell<=_checked_array_payload_bytes(UInt8,cellgrid.nx,cellgrid.ny) ||
                throw(ArgumentError("axial source span lies outside the grid"))
            for c in firstcell:lastcell
                i=mod1(c,cellgrid.nx);j=(c-1)÷cellgrid.nx+1
                _p2_point_in_poly(_P2((i-.5)*cellgrid.dx,(j-.5)*cellgrid.dy),via.vertices,
                    1e-10*min(cellgrid.dx,cellgrid.dy)) ||
                    throw(ArgumentError("axial source span lies outside its named via footprint"))
            end
            z=_project_port_reference(project,p,freq,variables)
            polarity=_project_int(q(get(p,"polarity",1)),"port polarity")
            polarity in (-1,1) || throw(ArgumentError("port polarity must be +1 or -1"))
            push!(entries,PlanarPort(findfirst(==(layer),via_layers),:via,firstcell:lastcell,z;polarity))
            push!(names,String(get(p,"name","p$(get(p,"number",k))")))
            push!(numbers,_project_int(q(get(p,"number",k)),"port number";positive=true))
            push!(refs,z);push!(planes,get(p,"refplane",nothing))
            continue
        end
        poly=findfirst(x->x.name==p["polygon"],polygons)
        poly===nothing && throw(ArgumentError("port references an undefined polygon"))
        polygon=polygons[poly];edge=_project_int(q(p["edge"]),"port edge";positive=true)
        1<=edge<=length(polygon.vertices) || throw(ArgumentError("port edge outside polygon"))
        u=polygon.vertices[edge];v=polygon.vertices[mod1(edge+1,length(polygon.vertices))]
        delta=v-u;orientation=sign(_p2_signed_area(polygon.vertices))
        dir=_lib_unit(orientation*_P2(delta[2],-delta[1]),"port edge")
        name=String(get(p,"name","p$(get(p,"number",k))"))
        push!(pins,_lib_pin(name,polygon,edge,dir));push!(names,name)
        push!(numbers,_project_int(q(get(p,"number",k)),"port number";positive=true))
        push!(refs,_project_port_reference(project,p,freq,variables))
        kind=get(p,"type","box_wall")
        kind in ("box_wall","internal","terminal","floating") || throw(ArgumentError("unsupported project port type $kind"))
        if kind=="floating"
            refidx=findfirst(x->x.name==p["ref_polygon"],polygons)
            refidx===nothing && throw(ArgumentError("floating reference polygon is undefined"))
            refpoly=polygons[refidx];refedge=_project_int(q(p["ref_edge"]),"floating reference edge";positive=true)
            1<=refedge<=length(refpoly.vertices) || throw(ArgumentError("floating reference edge lies outside polygon"))
            refu=refpoly.vertices[refedge];refv=refpoly.vertices[mod1(refedge+1,length(refpoly.vertices))]
            delta=refv-refu;refdir=_lib_unit(sign(_p2_signed_area(refpoly.vertices))*_P2(delta[2],-delta[1]),"floating reference")
            refpin=_lib_pin(name*"_reference",refpoly,refedge,refdir)
            for poly in (polygon,refpoly)
                table=get(get(project.data,"metals",Dict()),poly.metal,Dict())
                get(table,"type","lossless") in ("thick","volume") && throw(ArgumentError(
                    "floating thick/volume contacts require explicit resolved terminal geometry"))
            end
            cut=haskey(p,"source_edge") ? _project_int(q(p["source_edge"]),"floating source edge";positive=true) : nothing
            push!(entries,(signal=pins[end],reference=refpin,source_edge=cut,z0=refs[end],
                metal=String(get(p,"bridge_metal",polygon.metal))))
        else
            push!(entries,(pins[end],refs[end]))
        end
        kind=="box_wall" && !(abs(u[1])<1e-12 && abs(v[1])<1e-12 ||
            abs(u[1]-a)<1e-12 && abs(v[1]-a)<1e-12 || abs(u[2])<1e-12 && abs(v[2])<1e-12 ||
            abs(u[2]-b)<1e-12 && abs(v[2]-b)<1e-12) && throw(ArgumentError("box_wall port is not on a box wall"))
        push!(planes,get(p,"refplane",nothing))
    end
    for k in eachindex(planes)
        plane=planes[k]
        if plane!==nothing
            _project_keys(plane,["length","zc","gamma"],"reference plane")
            plane=let spec=plane
                PlanarReferencePlane(q(spec["length"],:length),
                    f->planar_project_value(project,spec["zc"];dimension=:resistance,freq=f,variables),
                    f->planar_project_value(project,spec["gamma"];dimension=:inverse_length,freq=f,variables))
            end
        end
        planes[k]=plane
    end
    length(unique(names))==length(names) && all(!isempty,names) &&
        length(unique(numbers))==length(numbers) || throw(ArgumentError("project port names/numbers must be unique"))
    metals=Dict{String,Any}(name=>(f->planar_project_metal_zs(project,name,f;variables))
        for name in unique(p.metal for p in polygons))
    via_types=Dict{String,Any}("uniform"=>(kind=VIA_UNIFORM,sigma=Inf))
    for (name,t) in get(project.data,"via_types",Dict())
        _project_keys(t,["kind","conductivity","metal","wall"],"via type $name")
        get(t,"wall","solid")=="solid" || throw(ArgumentError("non-solid via walls require an explicit meshing adapter"))
        kind=get(t,"kind","uniform");kind in ("uniform","taper") || throw(ArgumentError("invalid via profile"))
        spec=if haskey(t,"conductivity")
            t["conductivity"]
        elseif haskey(t,"metal")
            allmetals=get(project.data,"metals",Dict("pec"=>Dict("type"=>"pec")))
            haskey(allmetals,t["metal"]) || throw(ArgumentError("via references undefined metal"))
            metal=allmetals[t["metal"]]
            get(metal,"type","lossless") in ("lossless","pec") ? nothing :
                get(metal,"conductivity",nothing)===nothing ? throw(ArgumentError("via metal needs bulk conductivity")) : metal["conductivity"]
        else
            nothing
        end
        conductivity=spec===nothing ? Inf :
            spec isa AbstractString && occursin(r"\bfreq\b",spec) ? let expression=spec
                f->_project_real(planar_project_value(project,expression;dimension=:conductivity,
                    freq=f,variables),"via conductivity";positive=true)
            end : _project_real(q(spec,:conductivity;dynamic=true),"via conductivity";positive=true)
        via_types[name]=(kind=kind=="uniform" ? VIA_UNIFORM : VIA_TAPER,sigma=conductivity)
    end
    shape=PlanarShape(get(get(project.data,"project",Dict()),"name","planar_project"),polygons,vias,pins,Dict{String,Float64}())
    ordinary=findall(e->!(e isa NamedTuple),entries)
    layout,volume_sigma=_project_physical_layout(project,stack,cellgrid,shape,entries[ordinary];
        freq,variables,metals,via_types,terminal_ground,
        max_bytes=max_bytes-_checked_array_payload_bytes(Float64,4,vertices_payload),
        _allow_portless=length(ordinary)<length(entries))
    layout=_project_floating_layout(project,layout,entries,ordinary;variables,
        max_bytes=max_bytes-_checked_array_payload_bytes(Float64,4,vertices_payload))
    original=layout.problem;updated=PlanarPort[]
    signs=ones(Float64,length(original.ports))
    for (k,p) in enumerate(original.ports)
        sign=_project_int(q(get(ports_raw[k],"polarity",1)),"port polarity")
        sign in (-1,1) || throw(ArgumentError("port polarity must be +1 or -1"))
        p.wall===:via || (signs[k]=sign)
        push!(updated,PlanarPort(p.level,p.wall,p.cells,p.z0;edge=p.edge,
            polarity=p.polarity*Int(signs[k]),refplane=planes[k]))
    end
    metadata=PlanarProblem(original.stack,original.grid,original.sheets,updated,original.vias,original.basis,original.vols)
    layout=PlanarLayout(metadata,layout.shapes,layout.sheet_materials,layout.material_names,
        layout.materials,layout.via_models,layout.contraction===nothing ? metadata : layout.source_problem,
        layout.contraction===nothing ? nothing : layout.contraction*Diagonal(signs),layout.terminal_paths,layout.volume_models)
    components=Dict{String,Any}[_project_owned(c) for c in get(project.data,"components",Any[])]
    node_names,port_terminals,incidence=_project_node_contracts(project,names,components;max_bytes)
    anchored=Set{Int}()
    for c in components,port in get(c,"ports",Any[])
        port in (0,"gnd") && continue
        idx=port isa AbstractString ? findfirst(==(port),names) : findfirst(==(Int(port)),numbers)
        idx===nothing && throw(ArgumentError("component references undefined port $port"))
        push!(anchored,idx)
    end
    for c in components
        haskey(c,"nodes") || continue
        for value in _project_flat_nodes(c["nodes"])
            key=_project_node_key(value);isempty(key) && continue
            id=findfirst(==(key),node_names)
            for (k,pair) in enumerate(port_terminals)
                id in pair && push!(anchored,k)
            end
        end
    end
    external=Int[k for (k,p) in enumerate(ports_raw) if get(p,"external",!(k in anchored))]
    !isempty(external) || throw(ArgumentError("project requires external ports"))
    return PlanarProjectModel(project,layout,names,numbers,external,refs,components,_project_owned(variables),
        volume_sigma,node_names,port_terminals,incidence)
end

"""Resolve `[sweep]` into increasing positive frequencies [Hz]. Explicit
knots or start/stop/points are accepted; `spacing="log"` is logarithmic.
Geometry variables use the same SI expression rules as the project."""
function planar_project_frequencies(project::PlanarProject;variables::AbstractDict=Dict{String,Any}(),
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    sweep=_project_keys(get(project.data,"sweep",Dict()),["frequencies","start","stop","points",
        "spacing","adaptive","rel_tol","max_points","n_eval"],"sweep")
    q(x)=_project_real(planar_project_value(project,x;dimension=:frequency,variables),"sweep frequency";positive=true)
    fs=if haskey(sweep,"frequencies")
        _enforce_payload_limit(_checked_array_payload_bytes(Float64,length(sweep["frequencies"])),
            max_bytes,"project frequency grid","max_bytes")
        Float64[q(f) for f in sweep["frequencies"]]
    else
        lo,hi=q(sweep["start"]),q(sweep["stop"])
        count=_project_int(planar_project_value(project,get(sweep,"points",2);variables),
            "sweep points";positive=true)
        _enforce_payload_limit(_checked_array_payload_bytes(Float64,count),
            max_bytes,"project frequency grid","max_bytes")
        spacing=get(sweep,"spacing","linear");spacing in ("linear","log") || throw(ArgumentError("invalid sweep spacing"))
        spacing=="linear" ? collect(range(lo,hi;length=count)) : exp.(range(log(lo),log(hi);length=count))
    end
    !isempty(fs) && all(k->fs[k]>fs[k-1],2:length(fs)) || throw(ArgumentError("sweep frequencies must be increasing"))
    return fs
end
