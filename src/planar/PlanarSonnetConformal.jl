export sonnet_conformal_layout,solve_sonnet_conformal

# Decompose the generated background conductors once per interface. Every
# region retains its active via IDs; original physical sheets have priority.
function _sonnet_endpoint_overlay(backgrounds,occupied,max_bytes)
    # Unique-level/index/filter arrays are owned staging, even if the
    # later edge budget rejects. Check before constructing any of them.
    reference_bytes=_checked_array_payload_bytes(UInt,8,length(backgrounds))
    _enforce_payload_limit(_checked_payload_sum("native endpoint overlay staging",
        reference_bytes,_planar_conformal_exact_workspace()),max_bytes,
        "native endpoint overlay staging","max_bytes")
    result=Tuple{Int,Tuple,Vector{_P2}}[]
    total_vertices=0;total_ids=0
    for level in unique(q.level for q in backgrounds)
        level_backgrounds=filter(q->q.level==level,backgrounds)
        ne=sum((BigInt(length(q.vertices)) for q in level_backgrounds);init=BigInt(0))+
            sum((BigInt(length(q.vertices)) for q in occupied if q.level==level);init=BigInt(0))
        budget(nx,parts,vertices,ids)=_planar_conformal_arrangement_budget(ne,nx,parts,vertices,0,
            max_bytes-reference_bytes-_checked_array_payload_bytes(Int,ids))
        budget(2ne,length(result),total_vertices,total_ids)
        edges=Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
        for q in Iterators.flatten((level_backgrounds,(q for q in occupied if q.level==level)))
            for k in eachindex(q.vertices)
                push!(edges,(Tuple(q.vertices[k]),Tuple(q.vertices[mod1(k+1,length(q.vertices))])))
            end
        end
        xmin=minimum(v[1] for q in level_backgrounds for v in q.vertices)
        xmax=maximum(v[1] for q in level_backgrounds for v in q.vertices)
        cuts=Float64[xmin,xmax];intersections=Dict{Tuple{Int,Float64},Float64}()
        for (a,b) in edges
            xmin<a[1]<xmax && push!(cuts,a[1])
            xmin<b[1]<xmax && push!(cuts,b[1])
        end
        for i in eachindex(edges),j in i+1:length(edges)
            point=_planar_conformal_cross_point(edges[i]...,edges[j]...)
            point===nothing && continue
            x,y=point;xmin<x<xmax || continue
            budget(length(cuts)+2length(intersections)+5,length(result),total_vertices,total_ids)
            push!(cuts,x);intersections[(i,x)]=y;intersections[(j,x)]=y
        end
        edge_y(e,x)=get(intersections,(e,x)) do
            _planar_conformal_line_y(edges[e]...,x)
        end
        sort!(unique!(cuts))
        for slab in 1:length(cuts)-1
            xl,xr=cuts[slab],cuts[slab+1];mid=xl+(xr-xl)/2
            xl<mid<xr || throw(ArgumentError("native endpoint overlay is below Float64 resolution"))
            active=[e for (e,(a,b)) in enumerate(edges) if min(a[1],b[1])<mid<max(a[1],b[1])]
            sort!(active;by=e->edge_y(e,mid))
            for k in 1:length(active)-1
                lo,hi=active[k],active[k+1];yl,yu=edge_y(lo,mid),edge_y(hi,mid)
                yl<yu || continue
                point=_P2(mid,yl+(yu-yl)/2)
                any(q->q.level==level && _p2_point_in_poly(point,q.vertices,0.),occupied) && continue
                ids=sort!(unique(q.poly.id for q in level_backgrounds if _p2_point_in_poly(point,q.vertices,0.)))
                isempty(ids) && continue
                budget(length(cuts)+2length(intersections),length(result)+1,total_vertices+4,total_ids+length(ids))
                points=_P2[_P2(xl,edge_y(lo,xl)),_P2(xr,edge_y(lo,xr)),
                    _P2(xr,edge_y(hi,xr)),_P2(xl,edge_y(hi,xl))]
                unique!(points);length(points)>=3 || continue
                push!(result,(level,Tuple(ids),points));total_vertices+=length(points);total_ids+=length(ids)
            end
        end
    end
    result
end

function _sonnet_endpoint_combined_zs(p,polygons,ids,stack,freq,variables)
    zs=_sonnet_via_endpoint_zs(p,polygons[first(ids)],stack,freq,variables)
    for id in Iterators.drop(ids,1)
        next=_sonnet_via_endpoint_zs(p,polygons[id],stack,freq,variables)
        zs=_sonnet_via_parallel_zs(zs,next)
    end
    zs
end


# Subtract existing physical sheets from a generated endpoint footprint.
# Vertical slabs use the same exact orientation/intersection primitives as
# the conformal sheet arrangement, retaining genuine polygon coordinates.
function _sonnet_endpoint_difference(vertices,level,occupied,max_bytes)
    edges=Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
    ne=BigInt(length(vertices))+sum((BigInt(length(q.vertices)) for q in occupied if q.level==level);init=BigInt(0))
    _planar_conformal_arrangement_budget(ne,2ne,0,0,0,max_bytes)
    for k in eachindex(vertices)
        push!(edges,(Tuple(vertices[k]),Tuple(vertices[mod1(k+1,length(vertices))])))
    end
    for q in occupied
        q.level==level || continue
        for k in eachindex(q.vertices)
            push!(edges,(Tuple(q.vertices[k]),Tuple(q.vertices[mod1(k+1,length(q.vertices))])))
        end
    end
    xmin,xmax=extrema(v[1] for v in vertices)
    cuts=Float64[xmin,xmax]
    intersections=Dict{Tuple{Int,Float64},Float64}()
    for (a,b) in edges
        xmin<a[1]<xmax && push!(cuts,a[1])
        xmin<b[1]<xmax && push!(cuts,b[1])
    end
    for i in eachindex(edges),j in i+1:length(edges)
        point=_planar_conformal_cross_point(edges[i]...,edges[j]...)
        point===nothing && continue
        x,y=point;xmin<x<xmax || continue
        _planar_conformal_arrangement_budget(ne,length(cuts)+2length(intersections)+5,0,0,0,max_bytes)
        push!(cuts,x);intersections[(i,x)]=y;intersections[(j,x)]=y
    end
    edge_y(e,x)=get(intersections,(e,x)) do
        _planar_conformal_line_y(edges[e]...,x)
    end
    sort!(unique!(cuts));parts=Vector{Vector{_P2}}()
    for slab in 1:length(cuts)-1
        xl,xr=cuts[slab],cuts[slab+1];mid=xl+(xr-xl)/2
        xl<mid<xr || throw(ArgumentError("native endpoint subtraction is below Float64 resolution"))
        active=[e for (e,(a,b)) in enumerate(edges) if min(a[1],b[1])<mid<max(a[1],b[1])]
        sort!(active;by=e->edge_y(e,mid))
        for k in 1:length(active)-1
            lo,hi=active[k],active[k+1];yl,yu=edge_y(lo,mid),edge_y(hi,mid)
            yl<yu || continue
            point=_P2(mid,yl+(yu-yl)/2)
            _p2_point_in_poly(point,vertices,0.) || continue
            any(q->q.level==level && _p2_point_in_poly(point,q.vertices,0.),occupied) && continue
            points=_P2[_P2(xl,edge_y(lo,xl)),_P2(xr,edge_y(lo,xr)),
                _P2(xr,edge_y(hi,xr)),_P2(xl,edge_y(hi,xl))]
            unique!(points);length(points)>=3 || continue
            _planar_conformal_arrangement_budget(ne,length(cuts)+2length(intersections),length(parts)+1,4(length(parts)+1),0,max_bytes)
            push!(parts,points)
        end
    end
    parts
end

function _sonnet_endpoint_rectangle_payload(mask)
    # Each run owns a distinct cell, so this counter cannot exceed the
    # already stored array length. Widen only checked byte arithmetic.
    rectangles=0
    for j in axes(mask,2),i in axes(mask,1)
        mask[i,j] && (i==first(axes(mask,1)) || !mask[i-1,j]) && (rectangles+=1)
    end
    _checked_payload_sum("native endpoint rectangles",
        _checked_array_payload_bytes(_P2,4,rectangles),
        _checked_array_payload_bytes(UInt,rectangles))
end

function _sonnet_endpoint_rectangles(mask,grid,max_bytes=_default_max_dense_payload_bytes())
    # Count the exact row runs and reserve all coordinate/pointer payload
    # before constructing any rectangle, including failed public imports.
    _enforce_payload_limit(_sonnet_endpoint_rectangle_payload(mask),max_bytes,
        "native endpoint rectangles","max_bytes")
    # Coalesce contiguous occupied cells in each row. Physical sheet geometry
    # is subtracted afterward; source polygons are never rasterized here.
    rectangles=Vector{Vector{_P2}}()
    for j in axes(mask,2)
        i=first(axes(mask,1))
        while i<=last(axes(mask,1))
            if !mask[i,j];i+=1;continue;end
            firstcell=i
            while i<last(axes(mask,1)) && mask[i+1,j];i+=1;end
            x1,x2=(firstcell-1)*grid.dx,i*grid.dx
            y1,y2=(j-1)*grid.dy,j*grid.dy
            push!(rectangles,_P2[_P2(x1,y1),_P2(x2,y1),_P2(x2,y2),_P2(x1,y2)])
            i+=1
        end
    end
    rectangles
end


"""Lower a native Sonnet project onto genuine physical sheet triangles.
The native stack/material helper preserves SI units, variable expressions,
cover impedances, dielectric loss and physical two-face thick geometry.
Sheet polygons are unioned exactly, with material seams and shared axial
source cuts. Tiny sheet features do not pass through a raster lowerer.
Vertices in the native source BOX half-cell wall band attach to that wall;
interior coordinates and caller-owned records remain unchanged.
Sheets extending beyond the box are clipped exactly. Source edge identities
remain native; their driven attachment must still lie within the box band.
`edge_size,interior_size` declare independent physical mesh bounds.

Via columns retain the supported native uniform-grid footprints and exact
axial kernels; BAR requires its dedicated adapter. Cell boundaries constrain
the triangular contacts. This is exact sheet geometry, with a declared
grid approximation for arbitrary via footprints. Bare open interior
terminals require explicit physical return/bridge geometry. Components,
bricks and unsupported native port/material semantics reject.
Returns `(layout,contraction,floating_common,z0,labels,via_sigma,geometry_project)`.
Shared native terminal numbers preserve common voltages; corresponding
negative terminals supply floating references with balanced total current."""
function sonnet_conformal_layout(project::SonnetProject;freq::Real=1e9,
        grid=nothing,variables=Dict{String,Float64}(),
        edge_size::Real,interior_size::Real,edge_band::Real=2edge_size,
        max_triangles::Integer=100_000,max_bytes::Integer=_default_max_dense_payload_bytes(),
        scalar_files=nothing,scalar_root=dirname(project.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384)
    isfinite(freq) && freq>0 || throw(ArgumentError("frequency must be positive and finite"))
    freq=_circuit_stored_real(freq,"native conformal frequency")
    if scalar_files!==nothing || _sonnet_has_scalar_tables(project) || variables isa SonnetScalarVariables
        variables=_sonnet_scalar_variables(project,variables;scalar_files,max_bytes,
            root=scalar_root,outside=scalar_outside,max_files=scalar_max_files,
            max_bytes_source=scalar_max_bytes,max_nodes=scalar_max_nodes,max_line_bytes=scalar_max_line_bytes)
        project=_sonnet_scalar_project(project,variables)
    end
    isempty(project.components) || throw(ArgumentError("native conformal components require their physical source/network lowering"))
    native=_sonnet_stack_geometry(project,freq,grid,variables;max_bytes);p=native.geometry_project
    variables=native.variables
    stack,gr=native.stack,native.grid;L=length(stack.layers);val(x)=sonnet_variable_value(p,x;variables,freq)
    any(poly->poly.kind==:brick,p.polygons) && throw(ArgumentError("native dielectric brick needs its physical volume dielectric model"))
    ne=sum((size(poly.vertices,2) for poly in p.polygons);init=0);cells=BigInt(gr.nx)*gr.ny
    nvia=BigInt(0);nvpoly=0
    for poly in p.polygons
        poly.kind===:via || continue
        target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
        nvia+=abs(BigInt(poly.level)-target);nvpoly+=1
    end
    reserve=_checked_payload_sum("native conformal metadata",_checked_array_payload_bytes(Float64,8,ne),
        _checked_array_payload_bytes(Int,8,ne),_checked_array_payload_bytes(UInt64,4+2nvia+nvpoly,cld(cells,64)),
        _checked_array_payload_bytes(ComplexF64,length(p.metals)+1),
        _sonnet_scalar_files(variables)===nothing ? 0 : _sonnet_scalar_payload(variables))
    if _sonnet_has_volume_skin(p) || any(poly->_sonnet_via_endpoints(p,poly) && poly.material!=-1,p.polygons)
        reserve=_checked_payload_sum("native complex via metadata",reserve,256nvia)
        workspace=_sonnet_volume_polygon_workspace(p)
        iszero(workspace) || (reserve=_checked_payload_sum("native volume material workspace",reserve,workspace))
    end
    _enforce_payload_limit(reserve,max_bytes,"native conformal metadata","max_bytes")
    # Reserve geometry storage before copying any near-wall vertices.
    # Wall ownership uses source BOX counts rather than triangle/bulk sizes.
    p=_sonnet_raster_wall_project(p,stack.a,stack.b,
        parse(Int,p.box[4])÷2,parse(Int,p.box[5])÷2)
    polygons=PlanarPolygon[];metals=Dict{String,Any}("pec"=>0.)
    contacts=PlanarConformalPort[];cp=PlanarConformalPort[];bp=PlanarPort[]
    vias=ViaLevel[];sigma=Float64[];via_group=Dict{Tuple{Int,Float64},Int}();via_indices=Dict{Tuple{Int,Int},Int}();masks=Dict{Int,BitMatrix}()
    numbers=Int[];refs=ComplexF64[];weights=Float64[];cnumbers=Int[];crefs=ComplexF64[]
    function add_sheet(name,level,vertices,metal)
        v,b=planar_normalize_polygon([_P2(vertices[1,i],vertices[2,i]) for i in axes(vertices,2)];label=name)
        push!(polygons,PlanarPolygon(name,level,metal,"",v,b))
    end
    for poly in p.polygons
        if poly.kind===:sheet
            metal=poly.material==-1 ? "pec" : "native_$(poly.material)"
            if poly.material!=-1
                0<=poly.material<length(p.metals) || throw(ArgumentError("native sheet metal index is invalid"))
                record=p.metals[poly.material+1];sonnet_metal_zs(p,record,freq;variables)
                metals[metal]=f->sonnet_metal_zs(p,record,f;variables)
            end
            add_sheet("native_$(poly.id)",L-1-poly.level,poly.vertices,metal)
        else
            target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
            lo,hi=minmax(poly.level,target);-1<=lo<hi<=L-1 || throw(ArgumentError("native via target lies outside the stack"))
            footprint=sheet_level(0,gr.nx,gr.ny);rasterize_poly!(footprint,gr,poly.vertices[1,:],poly.vertices[2,:])
            any(footprint.mask) || throw(ArgumentError("native via footprint vanished at its declared bulk grid"))
            mode="SOLID" in poly.flags || "FULL" in poly.flags ? :full : "VERTICES" in poly.flags ? :vertices :
                "CENTER" in poly.flags ? :center : "BAR" in poly.flags ? :bar : :ring
            mask=_planar_via_mesh_mask(footprint.mask,poly.vertices,gr,mode);masks[poly.id]=mask
            rho_sigma=_sonnet_via_sigma(p,poly,gr,stack,mask,freq,variables)
            if rho_sigma isa Complex && sigma isa Vector{Float64}
                sigma=ComplexF64.(sigma)
                via_group=Dict{Tuple{Int,ComplexF64},Int}((k[1],ComplexF64(k[2]))=>v for (k,v) in via_group)
            end

            for layer in (L-hi):(L-lo-1)
                key=(layer,rho_sigma);index=get(via_group,key,0)
                for (other,v) in enumerate(vias)
                    v.layer==layer && any(v.uni .& mask) && (other!=index || isfinite(rho_sigma)) && throw(ArgumentError("native via resistance footprints overlap in one physical layer"))
                end
                if index==0
                    push!(vias,via_level(layer,gr.nx,gr.ny));push!(sigma,rho_sigma);index=length(vias);via_group[key]=index
                end
                vias[index].uni .|=mask;vias[index].tap .|=mask;via_indices[(poly.id,layer)]=index
            end
        end
    end
    # Shared generated films are one physical interface, independently of
    # polygon order. Exact overlay preserves original physical sheets.
    physical_sheets=copy(polygons)
    backgrounds=NamedTuple[];background_bytes=0
    endpoint_polygons=Dict(poly.id=>poly for poly in p.polygons if _sonnet_via_endpoints(p,poly))
    for poly in p.polygons
        _sonnet_via_endpoints(p,poly) || continue
        target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
        endpoints=count(l->!(l in (-1,L-1)),(poly.level,target))
        iszero(endpoints) && continue
        covers="COVERS" in poly.flags
        footprint_bytes=covers ? _checked_payload_sum("native endpoint polygon",
            _checked_array_payload_bytes(_P2,size(poly.vertices,2)),sizeof(UInt)) :
            _sonnet_endpoint_rectangle_payload(masks[poly.id])
        # Source rectangles, canonical copies and background record slots
        # stay live together until the complete interface overlay is built.
        additional=_checked_payload_sum("native endpoint background",
            (endpoints+1)*footprint_bytes,64endpoints*div(footprint_bytes,72)+256)
        _enforce_payload_limit(_checked_payload_sum("native endpoint backgrounds",reserve,background_bytes,additional),
            max_bytes,"native endpoint backgrounds","max_bytes")
        footprints=covers ? [[_P2(poly.vertices[1,k],poly.vertices[2,k]) for k in axes(poly.vertices,2)]] :
            _sonnet_endpoint_rectangles(masks[poly.id],gr,max_bytes-reserve-background_bytes)
        for nativelevel in (poly.level,target)
            nativelevel in (-1,L-1) && continue
            level=L-1-nativelevel
            for vertices in footprints
                generated=covers ? vertices :
                    [_P2(ntuple(dim->_planar_conformal_contact_coordinate(vertex[dim],level,dim,
                        physical_sheets,PlanarPolygon[],Tuple[],0),2)) for vertex in vertices]
                push!(backgrounds,(;poly,level,vertices=generated))
            end
        end
        background_bytes=_checked_payload_sum("native endpoint backgrounds",background_bytes,additional)
    end
    if !isempty(backgrounds)
        parts=_sonnet_endpoint_overlay(backgrounds,physical_sheets,max_bytes-reserve-background_bytes)
        vertices_count=sum((BigInt(length(vertices)) for (_,_,vertices) in parts);init=BigInt(0))
        ids_count=sum((BigInt(length(ids)) for (_,ids,_) in parts);init=BigInt(0))
        # Overlay records and coordinates stay live while normalized
        # polygons/providers/names are constructed, and during meshing.
        endpoint_bytes=_checked_payload_sum("native retained endpoint geometry",
            _checked_array_payload_bytes(_P2,2,vertices_count),
            _checked_array_payload_bytes(UInt,8,length(parts)),
            _checked_array_payload_bytes(Int,ids_count),
            _checked_array_payload_bytes(PlanarPolygon,length(parts)),
            _checked_array_payload_bytes(UInt8,192,length(parts)),
            _checked_array_payload_bytes(UInt8,24,ids_count))
        reserve=_checked_payload_sum("native retained endpoint geometry",reserve,background_bytes,endpoint_bytes)
        _enforce_payload_limit(reserve,max_bytes,"native retained endpoint geometry","max_bytes")
        for (index,(level,ids,vertices)) in enumerate(parts)
            zs=_sonnet_endpoint_combined_zs(p,endpoint_polygons,ids,stack,freq,variables)
            metal=iszero(zs) ? "pec" : "native_endpoint_$(level)_$(join(ids,'_'))"
            if metal!="pec" && !haskey(metals,metal)
                metals[metal]=f->_sonnet_endpoint_combined_zs(p,endpoint_polygons,ids,stack,f,variables)
            end
            v,b=planar_normalize_polygon(vertices;label="native via endpoint")
            push!(polygons,PlanarPolygon("native_endpoint_$(level)_$index",level,metal,"",v,b))
        end
    end
    # PEC box walls ground every actual sheet boundary touching them,
    # including boundaries introduced by clipping. Absent metal adds no basis.
    for level in unique(poly.level for poly in polygons)
        for (wall,extent) in ((:west,stack.b),(:east,stack.b),(:south,stack.a),(:north,stack.a))
            push!(contacts,PlanarConformalPort(level,wall,(0.,extent)))
        end
    end
    polydict=Dict(poly.id=>poly for poly in p.polygons)
    for ps in p.ports
        ps.kind in (:box,:std,:gap,:via) || throw(ArgumentError("native conformal $(ps.kind) port requires its physical return/calibration model"))
        poly=polydict[ps.polygon];q=ps.values;z0=_sonnet_port_reference(p,ps,freq,variables)
        ps.kind===:via && poly.kind!==:via && throw(ArgumentError("native via port requires a via polygon"))
        if poly.kind===:via && ps.kind in (:std,:via)
            ps.number!=0 || throw(ArgumentError("native interior port zero needs an explicit physical ground model"))
            target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
            ps.kind===:std && !(poly.level in (-1,L-1) || target in (-1,L-1)) &&
                throw(ArgumentError("native standard via port must terminate on a box cover"))
            lo,hi=minmax(poly.level,target);layers=(L-hi):(L-lo-1);height=sum(real(stack.layers[l].thickness) for l in layers)
            for l in layers,cell in findall(vec(masks[poly.id]))
                push!(bp,PlanarPort(via_indices[(poly.id,l)],:via,cell:cell,z0;polarity=ps.number<0 ? -1 : 1))
                push!(numbers,abs(ps.number));push!(refs,z0);push!(weights,real(stack.layers[l].thickness)/height)
            end
            continue
        end
        poly.kind===:sheet || throw(ArgumentError("native sheet port must reference a sheet polygon"))
        v=poly.vertices;n=_sonnet_polygon_edge_count(v);i,j=_sonnet_port_edge_indices(v,ps.edge)
        _sonnet_box_port_edge_inside(v,ps.edge,stack.a,stack.b,
            parse(Int,p.box[4])÷2,parse(Int,p.box[5])÷2) ||
            throw(ArgumentError("native box-port edge is partially or entirely outside the box"))
        a=(v[1,i],v[2,i]);b=(v[1,j],v[2,j]);level=L-1-poly.level
        wall=a[1]==b[1]==0. ? :west : a[1]==b[1]==stack.a ? :east :
            a[2]==b[2]==0. ? :south : a[2]==b[2]==stack.b ? :north : nothing
        if wall===nothing
            ps.number!=0 || throw(ArgumentError("native interior port zero requires its explicit reference conductor"))
            (a[1]==b[1] || a[2]==b[2]) ||
                throw(ArgumentError("native internal port is diagonal"))
            a,b=_sonnet_shared_port_segment(p,poly,i,j)
            # Native internal source orientation is independent of which
            # adjacent polygon is referenced; match positive raster axes.
            dx,dy=b[1]-a[1],b[2]-a[2]
            (abs(dy)>abs(dx) ? dy>0 : dx<0) && ((a,b)=(b,a))
            push!(cp,PlanarConformalPort(level,:internal,(a,b);z0,polarity=ps.number<0 ? -1 : 1))
        else
            ps.number==0 && continue
            dim=wall in (:west,:east) ? 2 : 1;normal=3-dim;position=a[normal]
            lo,hi=minmax(a[dim],b[dim])
            # Collinear subdivisions share one physical wall excitation.
            for direction in (-1,1)
                k=direction==-1 ? i : j
                for _ in 1:n-2
                    next=mod1(k+direction,n)
                    v[normal,next]==position || break
                    lo=min(lo,v[dim,next]);hi=max(hi,v[dim,next]);k=next
                end
            end
            push!(cp,PlanarConformalPort(level,wall,(lo,hi);z0,polarity=ps.number<0 ? -1 : 1))
        end
        push!(cnumbers,abs(ps.number));push!(crefs,z0)
    end
    allnumbers=vcat(cnumbers,numbers);allrefs=vcat(crefs,refs);allweights=vcat(ones(length(cp)),weights)
    isempty(allnumbers) && throw(ArgumentError("native conformal solve requires driven ports"))
    layout=if isempty(polygons)
        bulk=build_planar_problem(stack,gr,SheetLevel[],bp;vias)
        PlanarConformalLayout(bulk,polygons,Int[],String[],Any[])
    else
        build_planar_conformal_layout(stack,polygons,cp;metals,wall_contacts=contacts,clip_to_box=true,
            bulk_grid=isempty(vias) ? nothing : gr,vias,bulk_ports=bp,edge_size,interior_size,edge_band,max_triangles,max_bytes=max_bytes-reserve)
    end
    _enforce_payload_limit(_checked_payload_sum("native conformal contraction",reserve,
        _planar_conformal_layout_payload(layout),_checked_array_payload_bytes(Float64,3,length(allnumbers),length(unique(allnumbers)))),
        max_bytes,"native conformal contraction","max_bytes")
    labels=sort!(unique(allnumbers));C=zeros(Float64,length(allnumbers),length(labels));z0=ComplexF64[]
    for (j,label) in enumerate(labels)
        ids=findall(==(label),allnumbers);all(allrefs[i]==allrefs[first(ids)] for i in ids) || throw(ArgumentError("shared native terminal references must agree"))
        C[ids,j].=allweights[ids];push!(z0,allrefs[first(ids)])
    end
    terminal_maps=_sonnet_terminal_maps(C,[port.polarity for port in vcat(cp,bp)])
    return (;layout,terminal_maps...,z0,labels,via_sigma=sigma,geometry_project=p,
        scalar_files=_sonnet_scalar_files(variables))
end

"""Solve a native project with genuine sheet polygons and analytic
triangle/bulk Galerkin reactions. Native calibration requests require
`calibration` or explicit `raw=true`; gap-plane results retain actual
current coefficients. `edge_size,interior_size` control physical meshing;
no sheet grid approximation is introduced."""
function solve_sonnet_conformal(project::SonnetProject,freq::Real;raw::Bool=false,calibration=nothing,
        grid=nothing,variables=Dict{String,Float64}(),edge_size::Real,interior_size::Real,edge_band::Real=2edge_size,
        max_triangles::Integer=100_000,max_bytes::Integer=_default_max_dense_payload_bytes(),
        scalar_files=nothing,scalar_root=dirname(project.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384,kw...)
    requested=any(r->r.tokens[1]=="OPTIONS" && any(occursin("d",t) for t in r.tokens[2:end]),project.records)
    requested && !raw && calibration===nothing && throw(ArgumentError("native conformal project requests calibration; supply it or explicitly request raw=true"))
    model=sonnet_conformal_layout(project;freq,grid,variables,edge_size,interior_size,edge_band,max_triangles,max_bytes,
        scalar_files,scalar_root,scalar_outside,scalar_max_files,scalar_max_bytes,scalar_max_nodes,scalar_max_line_bytes)
    n=size(model.contraction,2);nr=size(model.contraction,1)
    reserve=_checked_payload_sum("native conformal contraction",_checked_array_payload_bytes(ComplexF64,8,nr,n),
        _checked_array_payload_bytes(ComplexF64,12,n,n),_checked_array_payload_bytes(Float64,2,nr,n),
        _planar_conformal_layout_payload(model.layout),
        sum(_checked_array_payload_bytes(Float64,size(p.vertices)...) for p in model.geometry_project.polygons),
        model.scalar_files===nothing ? 0 : _sonnet_scalar_files_payload(model.scalar_files))
    _enforce_payload_limit(reserve,max_bytes,"native conformal contraction","max_bytes")
    result=model.layout.problem isa Union{PlanarHybridProblem,PlanarProblem} ?
        solve_planar(model.layout,freq;via_sigma=model.via_sigma,max_bytes=max_bytes-reserve,kw...) :
        solve_planar(model.layout,freq;max_bytes=max_bytes-reserve,kw...)
    terminal=_sonnet_balanced_y(result.y,model);y=terminal.y;voltage_transfer=terminal.voltage_transfer
    if calibration!==nothing
        y=deembed_ports(y,calibration)
        A,B,_,_=_calibration_chain_blocks(calibration,length(model.labels))
        voltage_transfer=voltage_transfer*(A+B*y)
    end
    retained_project=model.scalar_files===nothing ? project : model.scalar_files.project
    SonnetPlanarResult(retained_project,result,model.labels,y,planar_y_to_s(y,model.z0),voltage_transfer,model.z0,model.scalar_files)
end
solve_sonnet_conformal(path::AbstractString,freq::Real;kw...)=solve_sonnet_conformal(read_sonnet_project(path),freq;kw...)
