export PlanarConformalLayout,build_planar_conformal_layout

# Float64 segment coordinates have bounded binary exponents. At8192bits,
# their degree-two/three numerators remain exact before the final division,
# including extreme exponents and exact-zero/midpoint intersections.
const _PLANAR_CONFORMAL_EXACT_BITS=8192
function _planar_conformal_exact_workspace()
    _checked_payload_sum("conformal exact scalar workspace",
        _checked_array_payload_bytes(BigFloat,64),
        _checked_array_payload_bytes(UInt64,64,cld(_PLANAR_CONFORMAL_EXACT_BITS,64)))
end

@inline function _planar_conformal_line_y(a,b,x)
    x==a[1] && return a[2]
    x==b[1] && return b[2]
    a[2]==b[2] && return a[2]
    setprecision(BigFloat,_PLANAR_CONFORMAL_EXACT_BITS) do
        ax,ay,bx,by,xx=BigFloat(a[1]),BigFloat(a[2]),BigFloat(b[1]),BigFloat(b[2]),BigFloat(x)
        den=bx-ax
        Float64((ay*den+(xx-ax)*(by-ay))/den)
    end
end

function _planar_conformal_cross_point(a,b,c,d)
    o1=_planar_orient2d(a...,b...,c...);o2=_planar_orient2d(a...,b...,d...)
    o3=_planar_orient2d(c...,d...,a...);o4=_planar_orient2d(c...,d...,b...)
    ((o1>0 && o2<0)||(o1<0 && o2>0)) && ((o3>0 && o4<0)||(o3<0 && o4>0)) || return nothing
    a[1]==b[1] && c[2]==d[2] && return (a[1],c[2])
    a[2]==b[2] && c[1]==d[1] && return (c[1],a[2])
    setprecision(BigFloat,_PLANAR_CONFORMAL_EXACT_BITS) do
        ax,ay=BigFloat.(a);bx,by=BigFloat.(b);cx,cy=BigFloat.(c);dx,dy=BigFloat.(d)
        den=(bx-ax)*(dy-cy)-(by-ay)*(dx-cx)
        num=(cx-ax)*(dy-cy)-(cy-ay)*(dx-cx)
        x=a[1]==b[1] ? a[1] : c[1]==d[1] ? c[1] : Float64((ax*den+num*(bx-ax))/den)
        y=a[2]==b[2] ? a[2] : c[2]==d[2] ? c[2] : Float64((ay*den+num*(by-ay))/den)
        (x,y)
    end
end

function _planar_conformal_arrangement_budget(ne,nx,ntrap,nv,nt,max_bytes)
    _enforce_payload_limit(_checked_payload_sum("conformal polygon arrangement",
        _planar_conformal_exact_workspace(),
        _checked_array_payload_bytes(Float64,8,ne),_checked_array_payload_bytes(Float64,4,nx),
        _checked_array_payload_bytes(Float64,12,ntrap),_checked_array_payload_bytes(Int,12,ntrap),
        _checked_array_payload_bytes(Float64,4,nv),_checked_array_payload_bytes(Int,4,nv),
        _checked_array_payload_bytes(Int,24,nt),_checked_array_payload_bytes(Float64,8,nt)),
        max_bytes,"conformal polygon arrangement","max_bytes")
end

function _planar_conformal_contact_coordinate(value,level,dim,polygons,holes,cs,nsource)
    # Keep supplied physical coordinates. Lattice arithmetic and unit
    # conversion can round a coincident contact to adjacent Float64 values.
    # Only generated contact lines move; ambiguous source geometry rejects.
    lower,upper=prevfloat(value),nextfloat(value);chosen=value;found=false
    for p in Iterators.flatten((polygons,holes))
        p.level==level || continue
        for vertex in p.vertices
            candidate=vertex[dim]
            lower<=candidate<=upper || continue
            found && candidate!=chosen && throw(ArgumentError(
                "distinct physical coordinates within generated grid-contact rounding interval"))
            chosen=candidate;found=true
        end
    end
    for q in 1:nsource
        sourcelevel,(a,b)=cs[q];sourcelevel==level || continue
        for vertex in (a,b)
            candidate=vertex[dim]
            lower<=candidate<=upper || continue
            found && candidate!=chosen && throw(ArgumentError(
                "distinct physical coordinates within generated grid-contact rounding interval"))
            chosen=candidate;found=true
        end
    end
    chosen
end

"""Triangulate the exact union of physical sheet polygons, preserving
overlap boundaries and material seams. `holes` subtracts explicitly named
void polygons. `constraints` is a vector of `(interface,((ax,ay),(bx,by)))`
segments, such as diagonal source cuts or galvanic contact boundaries.
Vertical slab decomposition uses actual vertex and segment-intersection
coordinates; its trapezoids become conforming genuine triangles. There is
no rasterization or staircase geometry. Independent edge/interior sizing
then refines the shared mesh. Different materials overlapping on one
interface reject. Constraints ending inside metal are retained as mesh
vertices; source ports still require a complete shared-edge path.
`clip_box=(a,b)` intersects the union with `[0,a]×[0,b]` before material
selection. Concave polygons may become disconnected. `allow_empty=true`
returns an empty mesh when no positive-area metal remains."""
function planar_conformal_mesh(polygons::AbstractVector{PlanarPolygon};
        holes::AbstractVector{PlanarPolygon}=PlanarPolygon[],constraints=Tuple[],
        edge_size::Real,interior_size::Real,edge_band::Real=2edge_size,
        max_triangles::Integer=100_000,max_bytes::Integer=_default_max_dense_payload_bytes(),
        clip_box=nothing,allow_empty::Bool=false)
    isempty(polygons) && throw(ArgumentError("conformal layout requires physical sheet polygons"))
    all(isfinite,(edge_size,interior_size,edge_band)) && 0<edge_size<=interior_size && edge_band>=0 && max_triangles>=1 ||
        throw(ArgumentError("conformal mesh sizes and triangle limit are invalid"))
    levels=sort!(unique(p.level for p in polygons));all(p->p.level in levels,holes) || throw(ArgumentError("void polygon requires a physical sheet interface"))
    clip_box===nothing || (clip_box isa Tuple && length(clip_box)==2 &&
        all(x->x isa Real && isfinite(x) && x>0,clip_box)) ||
        throw(ArgumentError("conformal clip box must have two positive finite extents"))
    box=clip_box===nothing ? nothing : (Float64(clip_box[1]),Float64(clip_box[2]))
    box===nothing || all(x->isfinite(x) && x>0,box) ||
        throw(ArgumentError("conformal clip box must fit positive finite Float64 extents"))
    ne=sum(length(p.vertices) for p in polygons)+sum((length(p.vertices) for p in holes);init=0)+length(constraints)+
        (box===nothing ? 0 : 4length(levels))
    _planar_conformal_arrangement_budget(ne,2ne,0,0,0,max_bytes)
    edges=Tuple{Int,NTuple{2,Float64},NTuple{2,Float64}}[]
    for p in Iterators.flatten((polygons,holes)),i in eachindex(p.vertices)
        a,b=p.vertices[i],p.vertices[mod1(i+1,length(p.vertices))]
        push!(edges,(p.level,Tuple(a),Tuple(b)))
    end
    for (level,(a,b)) in constraints
        level in levels && all(isfinite,(a...,b...)) && a!=b || throw(ArgumentError("conformal constraint must be finite, nonzero and on a sheet interface"))
        push!(edges,(Int(level),(Float64(a[1]),Float64(a[2])),(Float64(b[1]),Float64(b[2]))))
    end
    if box!==nothing
        a,b=box;corners=((0.,0.),(a,0.),(a,b),(0.,b))
        for level in levels,i in 1:4
            push!(edges,(level,corners[i],corners[mod1(i+1,4)]))
        end
    end
    vertices=NTuple{2,Float64}[];faces=NTuple{3,Int}[];interfaces=Int[]
    canonical=Dict{NTuple{2,Float64},Int}()
    for level in levels
        ledges=[(a,b) for (l,a,b) in edges if l==level]
        cuts=sort!(unique([p[1] for edge in ledges for p in edge]));intersections=Dict{Tuple{Int,Float64},Float64}()
        for i in eachindex(ledges),j in i+1:length(ledges)
            point=_planar_conformal_cross_point(ledges[i]...,ledges[j]...);point===nothing && continue
            x,y=point
            _planar_conformal_arrangement_budget(ne,length(cuts)+2length(intersections)+5,0,length(vertices),length(faces),max_bytes)
            push!(cuts,x);intersections[(i,x)]=y;intersections[(j,x)]=y
        end
        edge_y(e,x)=get(intersections,(e,x)) do
            _planar_conformal_line_y(ledges[e]...,x)
        end
        sort!(unique!(cuts));traps=Tuple{Int,NTuple{2,Float64},NTuple{2,Float64},NTuple{2,Float64},NTuple{2,Float64}}[]
        box===nothing || filter!(x->0<=x<=box[1],cuts)
        seam=[Float64[] for _ in cuts]
        for (a,b) in ledges
            if a[1]==b[1]
                box!==nothing && !(0<=a[1]<=box[1]) && continue
                index=searchsortedfirst(cuts,a[1]);append!(seam[index],(a[2],b[2]))
            end
        end
        for slab in 1:length(cuts)-1
            xl,xr=cuts[slab],cuts[slab+1];mid=xl+(xr-xl)/2
            xl<mid<xr || throw(ArgumentError("polygon intersection spacing is below Float64 mesh resolution"))
            active=[e for (e,(a,b)) in enumerate(ledges) if min(a[1],b[1])<mid<max(a[1],b[1])]
            sort!(active;by=e->edge_y(e,mid))
            for i in 1:length(active)-1
                low,high=active[i],active[i+1];yl=edge_y(low,mid);yu=edge_y(high,mid)
                yl<yu || continue
                box!==nothing && !(0<=yl<yu<=box[2]) && continue
                point=_P2(mid,yl+(yu-yl)/2)
                any(p.level==level && _p2_point_in_poly(point,p.vertices,0.) for p in holes) && continue
                filled=[p.metal for p in polygons if p.level==level && _p2_point_in_poly(point,p.vertices,0.)]
                isempty(filled) && continue
                all(==(first(filled)),filled) || throw(ArgumentError("different conformal sheet materials overlap on interface $level"))
                bl=(xl,edge_y(low,xl));br=(xr,edge_y(low,xr))
                tl=(xl,edge_y(high,xl));tr=(xr,edge_y(high,xr))
                _planar_conformal_arrangement_budget(ne,length(cuts)+2length(intersections),length(traps)+1,length(vertices),length(faces),max_bytes)
                push!(traps,(slab,bl,br,tr,tl));append!(seam[slab],(bl[2],tl[2]));append!(seam[slab+1],(br[2],tr[2]))
            end
        end
        foreach(x->sort!(unique!(x)),seam)
        for (slab,bl,br,tr,tl) in traps
            boundary=NTuple{2,Float64}[bl,br]
            append!(boundary,[(br[1],y) for y in seam[slab+1] if br[2]<y<tr[2]])
            push!(boundary,tr,tl)
            append!(boundary,[(bl[1],y) for y in reverse(seam[slab]) if bl[2]<y<tl[2]])
            cleaned=NTuple{2,Float64}[]
            for point in boundary
                (isempty(cleaned)||last(cleaned)!=point) && push!(cleaned,point)
            end
            first(cleaned)==last(cleaned) && pop!(cleaned)
            center=((bl[1]+br[1])/2,(bl[2]+br[2]+tr[2]+tl[2])/4)
            _planar_conformal_arrangement_budget(ne,length(cuts)+2length(intersections),length(traps),length(vertices)+length(cleaned)+1,length(faces)+length(cleaned),max_bytes)
            length(faces)+length(cleaned)<=max_triangles || throw(ArgumentError("polygon arrangement exceeds max_triangles=$max_triangles"))
            ids=Int[]
            for point in (center,cleaned...)
                id=get(canonical,point,0)
                if id==0
                    push!(vertices,point);id=length(vertices);canonical[point]=id
                end
                push!(ids,id)
            end
            for j in 1:length(cleaned)
                push!(faces,(ids[1],ids[j+1],ids[mod1(j+1,length(cleaned))+1]));push!(interfaces,level)
            end
        end
    end
    if isempty(faces)
        allow_empty || throw(ArgumentError("conformal layout has no metal after subtracting voids"))
        return PlanarConformalMesh(zeros(Float64,2,0),zeros(Int,3,0),Int[],Float64[])
    end
    coords=hcat((collect(p) for p in vertices)...);triangles=hcat((collect(t) for t in faces)...)
    mesh=PlanarConformalMesh(coords,triangles;interfaces,max_bytes)
    planar_refine_conformal(mesh;edge_size,interior_size,edge_band,max_triangles,max_bytes)
end

"""Exact polygon layout on genuine triangles, optionally coupled to
physical bulk grid currents. Named material models are retained per
triangle and evaluated at each solved frequency. `problem` is a
`PlanarConformalProblem`, `PlanarHybridProblem`, or a bulk-only
`PlanarProblem` when box clipping removes all sheets; geometry is never
rasterized into sheet cells."""
struct PlanarConformalLayout{P}
    problem::P
    polygons::Vector{PlanarPolygon}
    triangle_materials::Vector{Int}
    material_names::Vector{String}
    materials::Vector{Any}
end

function _planar_conformal_geometry_payload(prob::PlanarConformalProblem)
    _checked_payload_sum("retained conformal geometry",
        sum(sizeof(eltype(a))*length(a) for a in (prob.mesh.vertices,prob.mesh.triangles,prob.mesh.interfaces,prob.mesh.areas,
            prob.basis.edges,prob.basis.triangles,prob.basis.interfaces,prob.basis.width,prob.basis.port,prob.basis.port_sign)),
        _checked_array_payload_bytes(eltype(prob.stack.layers),length(prob.stack.layers)))
end
function _planar_conformal_geometry_payload(prob::PlanarProblem)
    _checked_payload_sum("retained bulk geometry",
        sum(sizeof(eltype(getproperty(prob.basis,f)))*length(getproperty(prob.basis,f)) for f in fieldnames(PlanarBasisSet)),
        _checked_array_payload_bytes(UInt64,cld(BigInt(prob.grid.nx)*prob.grid.ny,64),length(prob.sheets)+2length(prob.vias)+length(prob.vols)),
        _checked_array_payload_bytes(UInt64,2,cld(BigInt(prob.grid.nx),64)+cld(BigInt(prob.grid.ny),64),length(prob.sheets)+length(prob.vols)),
        _checked_array_payload_bytes(eltype(prob.stack.layers),length(prob.stack.layers)))
end
_planar_conformal_geometry_payload(prob::PlanarHybridProblem)=_checked_payload_sum("retained hybrid geometry",
    _planar_conformal_geometry_payload(prob.conformal),_planar_conformal_geometry_payload(prob.bulk))
_planar_conformal_layout_payload(layout)=_checked_payload_sum("retained conformal layout",
    _planar_conformal_geometry_payload(layout.problem),_checked_array_payload_bytes(Int,length(layout.triangle_materials)))

"""Uniformly refine a genuine sheet layout while preserving every
descendant triangle's material, wall/interior source contracts and physical
bulk contacts. The conductor models and original polygon geometry retain
their names and frequency callbacks."""
function planar_refine_conformal_uniform(layout::PlanarConformalLayout,levels::Integer;
        max_triangles::Integer=100_000,max_bytes::Integer=_default_max_dense_payload_bytes())
    p=layout.problem;p isa PlanarProblem && throw(ArgumentError("uniform triangle refinement requires genuine sheet geometry"))
    c=p isa PlanarHybridProblem ? p.conformal : p
    mesh=planar_refine_conformal_uniform(c.mesh,levels;max_triangles,max_bytes)
    reserve=_checked_payload_sum("refined layout material/mesh",
        _checked_array_payload_bytes(Int,length(mesh.interfaces)),
        sum(sizeof(eltype(a))*length(a) for a in (mesh.vertices,mesh.triangles,mesh.interfaces,mesh.areas)))
    _enforce_payload_limit(reserve,max_bytes,"refined layout geometry","max_bytes")
    updated=PlanarConformalProblem(c.stack,mesh,c.ports;sidewalls=c.sidewalls,wall_contacts=c.wall_contacts,max_bytes=max_bytes-reserve)
    problem=p isa PlanarHybridProblem ? PlanarHybridProblem(updated,p.bulk) : updated
    ids=repeat(layout.triangle_materials;inner=Int(BigInt(4)^Int(levels)))
    PlanarConformalLayout(problem,layout.polygons,ids,layout.material_names,layout.materials)
end

"""Lower normalized physical sheet polygons into genuine conformal
triangles and exact material seams. Ports use `PlanarConformalPort` wall
or shared interior/diagonal cuts. Optional `bulk_grid,vias,vols,bulk_ports`
add mixed axial/transverse bulk currents; contact cell boundaries constrain
the sheet mesh. `holes` supplies explicit voids. Physical coordinates,
materials and independent edge/interior sizes are preserved. Overlapping
different metals reject. Solve with `solve_planar(layout,f)`; a requested
FFT solve requires all resulting vertices on its declared lattice.
`clip_to_box=true` intersects sheets with the physical box, preserving source
polygons and material ownership. Fully clipped sheets can leave a bulk-only
layout when there are no driven sheet ports."""
function build_planar_conformal_layout(stack::PlanarStackup,polygons::AbstractVector{PlanarPolygon},
        ports::AbstractVector{PlanarConformalPort};metals::AbstractDict=Dict("pec"=>0.,"PEC"=>0.),
        holes::AbstractVector{PlanarPolygon}=PlanarPolygon[],constraints=Tuple[],
        bulk_grid=nothing,vias::Vector{ViaLevel}=ViaLevel[],vols::Vector{VolLevel}=VolLevel[],bulk_ports::Vector{PlanarPort}=PlanarPort[],
        sidewalls::SidewallKind=WALL_PEC,wall_contacts::AbstractVector{PlanarConformalPort}=PlanarConformalPort[],
        edge_size::Real,interior_size::Real,edge_band::Real=2edge_size,
        max_triangles::Integer=100_000,max_bytes::Integer=_default_max_dense_payload_bytes(),
        clip_to_box::Bool=false)
    planar_validate(stack)
    _enforce_payload_limit(_checked_array_payload_bytes(UInt8,40,length(constraints)+length(ports)+length(wall_contacts)),
        max_bytes,"conformal layout constraints","max_bytes")
    all(p->0<=p.level<=length(stack.layers) &&
        (clip_to_box || all(v->0<=v[1]<=stack.a && 0<=v[2]<=stack.b,p.vertices)),polygons) ||
        throw(ArgumentError("conformal polygons lie outside the physical box/stack"))
    names=unique(p.metal for p in polygons);all(n->haskey(metals,n),names) || throw(ArgumentError("conformal layout material model is missing"))
    cs=Tuple[constraints...]
    for p in Iterators.flatten((ports,wall_contacts))
        if p.wall===:internal
            ax,ay,bx,by=p.endpoints;push!(cs,(p.interface,((ax,ay),(bx,by))))
        else
            lo,hi=p.span
            a,b=p.wall===:west ? ((0.,lo),(0.,hi)) : p.wall===:east ? ((stack.a,lo),(stack.a,hi)) :
                p.wall===:south ? ((lo,0.),(hi,0.)) : ((lo,stack.b),(hi,stack.b))
            push!(cs,(p.interface,(a,b)))
        end
    end
    isempty(vias) && isempty(vols) && isempty(bulk_ports) || bulk_grid isa CellGrid || throw(ArgumentError("mixed conformal layout needs a physical bulk grid"))
    if bulk_grid!==nothing
        nsource=length(cs)
        bulk_grid isa CellGrid && bulk_grid.walls==sidewalls && bulk_grid.a==stack.a && bulk_grid.b==stack.b || throw(ArgumentError("bulk grid must match the physical box and conformal sidewalls"))
        nc=BigInt(length(cs));levels=Set(p.level for p in polygons)
        for (models,via) in ((vias,true),(vols,false)),model in models
            1<=model.layer<=length(stack.layers) || throw(ArgumentError("conformal bulk layer lies outside the stack"))
            if via
                size(model.uni)==size(model.tap)==(bulk_grid.nx,bulk_grid.ny) || throw(DimensionMismatch("conformal via masks must match the bulk grid"))
            else
                _validate_sheet_grid(model,bulk_grid)
            end
            occupied=count(via ? model.uni[i,j]||model.tap[i,j] : model.mask[i,j] for j in 1:bulk_grid.ny for i in 1:bulk_grid.nx)
            nc+=4BigInt(occupied)*count(l->l in levels,(model.layer-1,model.layer))
        end
        _enforce_payload_limit(_checked_array_payload_bytes(UInt8,40,nc),max_bytes,"conformal bulk contact constraints","max_bytes")
        for (models,via) in ((vias,true),(vols,false)),model in models
            for j in 1:bulk_grid.ny,i in 1:bulk_grid.nx
                (via ? model.uni[i,j]||model.tap[i,j] : model.mask[i,j]) || continue
                for level in (model.layer-1,model.layer)
                    level in levels || continue
                    x0=_planar_conformal_contact_coordinate(((i-1)/bulk_grid.nx)*bulk_grid.a,level,1,polygons,holes,cs,nsource)
                    x1=_planar_conformal_contact_coordinate((i/bulk_grid.nx)*bulk_grid.a,level,1,polygons,holes,cs,nsource)
                    y0=_planar_conformal_contact_coordinate(((j-1)/bulk_grid.ny)*bulk_grid.b,level,2,polygons,holes,cs,nsource)
                    y1=_planar_conformal_contact_coordinate((j/bulk_grid.ny)*bulk_grid.b,level,2,polygons,holes,cs,nsource)
                    for (a,b) in (((x0,y0),(x1,y0)),((x1,y0),(x1,y1)),((x1,y1),(x0,y1)),((x0,y1),(x0,y0)))
                        push!(cs,(level,(a,b)))
                    end
                end
            end
        end
    end
    constraintbytes=_checked_array_payload_bytes(UInt8,40,length(cs))
    _enforce_payload_limit(constraintbytes,max_bytes,"conformal layout constraints","max_bytes")
    mesh=planar_conformal_mesh(polygons;holes,constraints=cs,edge_size,interior_size,edge_band,max_triangles,
        max_bytes=max_bytes-constraintbytes,clip_box=clip_to_box ? (stack.a,stack.b) : nothing,
        allow_empty=clip_to_box && bulk_grid!==nothing)
    meshbytes=_checked_payload_sum("conformal layout mesh",
        sum(sizeof(eltype(a))*length(a) for a in (mesh.vertices,mesh.triangles,mesh.interfaces,mesh.areas)),
        _checked_array_payload_bytes(Int,length(mesh.interfaces)),constraintbytes)
    nr=bulk_grid===nothing ? 0 : _terminal_geometry_basis_count(bulk_grid,SheetLevel[],vias,vols)
    bulkbytes=_checked_payload_sum("conformal layout bulk basis",_checked_array_payload_bytes(Int,8,nr),
        _checked_array_payload_bytes(Float64,6,nr),_checked_array_payload_bytes(UInt8,2,nr))
    _enforce_payload_limit(_checked_payload_sum("conformal layout basis",meshbytes,bulkbytes,
        _checked_array_payload_bytes(Int,21,length(mesh.interfaces)),_checked_array_payload_bytes(Float64,6,length(mesh.interfaces))),
        max_bytes,"conformal layout basis","max_bytes")
    if isempty(mesh.interfaces)
        isempty(ports) || throw(ArgumentError("driven conformal sheet ports have no retained metal"))
        bulk=build_planar_problem(stack,bulk_grid,SheetLevel[],bulk_ports;vias,vols)
        return PlanarConformalLayout(bulk,collect(polygons),Int[],String[],Any[])
    end
    c=PlanarConformalProblem(stack,mesh,ports;sidewalls,wall_contacts,max_bytes=max_bytes-meshbytes-bulkbytes)
    problem=bulk_grid===nothing ? c : PlanarHybridProblem(c,bulk_grid,bulk_ports;vias,vols)
    ids=Int[]
    for t in eachindex(mesh.interfaces)
        point=_P2(sum(mesh.vertices[1,mesh.triangles[j,t]] for j in 1:3)/3,sum(mesh.vertices[2,mesh.triangles[j,t]] for j in 1:3)/3)
        p=findfirst(p->p.level==mesh.interfaces[t] && _p2_point_in_poly(point,p.vertices,0.),polygons)
        p===nothing && throw(ArgumentError("refined triangle escaped its physical polygon"))
        push!(ids,findfirst(==(polygons[p].metal),names))
    end
    PlanarConformalLayout(problem,collect(polygons),ids,names,Any[metals[n] for n in names])
end

function solve_planar(layout::PlanarConformalLayout,freq::Number;max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    :surface_zs in keys(kw) && throw(ArgumentError("conformal layout sheet materials are supplied by the layout"))
    materialbytes=_checked_array_payload_bytes(ComplexF64,length(layout.triangle_materials)+length(layout.materials))
    _enforce_payload_limit(materialbytes,max_bytes,"conformal layout materials","max_bytes")
    values=ComplexF64[m isa Number ? m : m(freq) for m in layout.materials]
    all(isfinite,values) || throw(ArgumentError("conformal material provider returned nonfinite impedance"))
    zs=values[layout.triangle_materials]
    if layout.problem isa PlanarHybridProblem
        solve_planar_hybrid(layout.problem,freq;surface_zs=zs,max_bytes=max_bytes-materialbytes,kw...)
    elseif layout.problem isa PlanarConformalProblem
        solve_planar_conformal(layout.problem,freq;surface_zs=zs,max_bytes=max_bytes-materialbytes,kw...)
    else
        solve_planar(layout.problem,freq;max_bytes=max_bytes-materialbytes,kw...)
    end
end
