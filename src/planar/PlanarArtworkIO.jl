# Ordered constructive artwork shared by Gerber and ODB++ importers.
# Membership retains analytic curves, transparent aperture holes and local
# versus image polarity. Rasterization happens only at an explicit solver grid.
export PlanarArtwork, PlanarArtworkObject, artwork_cell_masks, artwork_planar_problem

abstract type _ArtworkShape end
struct _ArtworkEmpty <: _ArtworkShape end
struct _ArtworkCircle <: _ArtworkShape
    center::NTuple{2,Float64}
    radius::Float64
end
struct _ArtworkPolygon <: _ArtworkShape
    vertices::Matrix{Float64}
end
struct _ArtworkRoundLine <: _ArtworkShape
    start::NTuple{2,Float64}
    stop::NTuple{2,Float64}
    radius::Float64
end
struct _ArtworkRoundArc <: _ArtworkShape
    start::NTuple{2,Float64}
    stop::NTuple{2,Float64}
    center::NTuple{2,Float64}
    clockwise::Bool
    radius::Float64
end
struct _ArtworkPolygonArc <: _ArtworkShape
    aperture::_ArtworkPolygon
    path::_ArtworkRoundArc
end
struct _ArtworkPolygonLine <: _ArtworkShape
    aperture::_ArtworkPolygon
    start::NTuple{2,Float64}
    stop::NTuple{2,Float64}
end
struct _ArtworkRegion <: _ArtworkShape
    segments::Vector{Union{_ArtworkRoundLine,_ArtworkRoundArc}}
end
struct _ArtworkComposite{P<:Union{Tuple,AbstractVector}} <: _ArtworkShape
    parts::P
end
# A stable reference keeps union-valued stroke retrieval from copying a
# value shape onto the heap at every raster point. Geometry is immutable.
mutable struct _ArtworkStrokeReference{S<:_ArtworkShape} <: _ArtworkShape
    const geometry::S
end
const _ArtworkStrokeReferenceUnion=Union{_ArtworkStrokeReference{_ArtworkRoundLine},_ArtworkStrokeReference{_ArtworkPolygon}}
_artwork_concrete_part(dark::Bool,shape::S) where {S<:_ArtworkShape}=Tuple{Bool,S}((dark,shape))
function _ArtworkComposite(parts::Vector{Tuple{Bool,_ArtworkShape}})
    isempty(parts) && return _ArtworkComposite{typeof(parts)}(parts)
    if all(p->p[2] isa Union{_ArtworkRoundLine,_ArtworkPolygon,_ArtworkStrokeReferenceUnion},parts)
        # All font glyphs share one representation, including homogeneous
        # I/M glyphs. This prevents a nested union of different glyph types.
        narrow=Vector{Tuple{Bool,_ArtworkStrokeReferenceUnion}}(undef,length(parts))
        for i in eachindex(parts)
            dark,shape=parts[i]
            narrow[i]=(dark,shape isa _ArtworkStrokeReferenceUnion ? shape : _ArtworkStrokeReference(shape))
        end
        return _ArtworkComposite{typeof(narrow)}(narrow)
    end
    S=typeof(parts[1][2])
    if all(p->typeof(p[2])===S,parts)
        narrow=Vector{Tuple{Bool,S}}(undef,length(parts))
        for i in eachindex(parts);narrow[i]=parts[i];end
        return _ArtworkComposite{typeof(narrow)}(narrow)
    end
    if length(parts)<=8
        # Small heterogeneous apertures keep their actual field types.
        # The fixed size cap bounds compilation growth; large glyph and
        # homogeneous collections use the vector representations above.
        narrow=Tuple(_artwork_concrete_part(dark,shape) for (dark,shape) in parts)
        return _ArtworkComposite{typeof(narrow)}(narrow)
    end
    return _ArtworkComposite{typeof(parts)}(parts)
end
struct _ArtworkTransform{S<:_ArtworkShape} <: _ArtworkShape
    shape::S
    matrix::SMatrix{2,2,Float64,4}
    inverse::SMatrix{2,2,Float64,4}
    origin::NTuple{2,Float64}
end

"""One immutable drawing operation with analytic shape, dark/clear
image polarity, layer name and retained attributes. Aperture-local holes
are transparent parts of `shape`; clear image operations erase prior
objects only where their complete shape is opaque."""
struct PlanarArtworkObject
    layer::String
    shape::_ArtworkShape
    dark::Bool
    bounds::NTuple{4,Float64} # xmin,xmax,ymin,ymax
    attributes::Dict{String,Vector{String}}
end

"""Ordered fabrication artwork in SI, retaining geometric objects and
metadata rather than flattening away clear regions. `negative_layers`
specifies image inversion over the explicitly selected analysis window.
Original coordinates are retained until a caller supplies `offset`."""
struct PlanarArtwork
    source::String
    objects::Vector{PlanarArtworkObject}
    attributes::Dict{String,Vector{String}}
    negative_layers::Set{String}
    coordinate_unit_m::Float64
end

_artwork_bounds(::_ArtworkEmpty)=(0.,0.,0.,0.)
_artwork_bounds(s::_ArtworkStrokeReference)=_artwork_bounds(s.geometry)
_artwork_bounds(s::_ArtworkCircle)=_artwork_certified_bounds(s)
_artwork_bounds(s::_ArtworkPolygon)=(minimum(s.vertices[1,:]),maximum(s.vertices[1,:]),minimum(s.vertices[2,:]),maximum(s.vertices[2,:]))
_artwork_bounds(s::_ArtworkRoundLine)=_artwork_certified_bounds(s)
function _artwork_arc_angles(s::_ArtworkRoundArc)
    a=atan(s.start[2]-s.center[2],s.start[1]-s.center[1]);b=atan(s.stop[2]-s.center[2],s.stop[1]-s.center[1])
    span=s.clockwise ? -mod(a-b,2pi) : mod(b-a,2pi)
    s.start==s.stop && (span=s.clockwise ? -2pi : 2pi)
    return a,span
end
function _artwork_bounds(s::_ArtworkPolygonArc)
    a,b=_artwork_bounds(s.path),_artwork_bounds(s.aperture)
    return (a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4])
end
function _artwork_bounds(s::_ArtworkPolygonLine)
    _artwork_certified_bounds(s)
end

function _artwork_convex_hull(points)
    sorted=sort!(unique(points));length(sorted)>=3 || throw(ArgumentError("degenerate convex stroke"))
    cross(a,b,c)=(b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1])
    lower=Tuple{Float64,Float64}[];upper=Tuple{Float64,Float64}[]
    for point in sorted
        while length(lower)>=2 && cross(lower[end-1],lower[end],point)<=0;pop!(lower);end
        push!(lower,point)
    end
    for point in reverse(sorted)
        while length(upper)>=2 && cross(upper[end-1],upper[end],point)<=0;pop!(upper);end
        push!(upper,point)
    end
    return _artwork_polygon(vcat(lower[1:end-1],upper[1:end-1]))
end
@inline function _artwork_arc_fraction(s::_ArtworkRoundArc,angle)
    a,span=_artwork_arc_angles(s)
    return (span<0 ? -mod(a-angle,2pi) : mod(angle-a,2pi))/span
end
function _artwork_bounds(s::_ArtworkRoundArc)
    _artwork_certified_bounds(s)
end
function _artwork_bounds(s::_ArtworkComposite)
    isempty(s.parts) && return _artwork_bounds(_ArtworkEmpty())
    b=map(p->_artwork_bounds(p[2]),s.parts)
    return (minimum(p[1] for p in b),maximum(p[2] for p in b),minimum(p[3] for p in b),maximum(p[4] for p in b))
end
function _artwork_bounds(s::_ArtworkRegion)
    isempty(s.segments) && throw(ArgumentError("empty artwork contour"))
    b=_artwork_bounds.(s.segments)
    return (minimum(p[1] for p in b),maximum(p[2] for p in b),minimum(p[3] for p in b),maximum(p[4] for p in b))
end
function _artwork_bounds(s::_ArtworkTransform)
    _artwork_canonical_leaf(s)&&return _artwork_certified_bounds(s)
    x0,x1,y0,y1=_artwork_bounds(s.shape)
    points=[s.matrix*SVector(x,y)+SVector(s.origin...) for x in (x0,x1),y in (y0,y1)]
    return (minimum(p[1] for p in points),maximum(p[1] for p in points),minimum(p[2] for p in points),maximum(p[2] for p in points))
end
function _artwork_transform(shape::_ArtworkShape;origin=(0.,0.),rotation=0.,scale=1.,mirror=(1.,1.))
    all(isfinite,origin) && isfinite(rotation) && isfinite(scale) && scale>0 && all(x->x in (-1.,1.),mirror) ||
        throw(ArgumentError("invalid artwork transformation"))
    c,s=cosd(rotation),sind(rotation)
    matrix=SMatrix{2,2,Float64}(c*mirror[1]*scale,s*mirror[1]*scale,-s*mirror[2]*scale,c*mirror[2]*scale)
    return _ArtworkTransform(shape,matrix,_artwork_inverse_matrix(matrix),(Float64(origin[1]),Float64(origin[2])))
end

_artwork_contains(::_ArtworkEmpty,x,y)=false
_artwork_contains(s::_ArtworkCircle,x,y)=_artwork_canonical_contains(s,Float64(x),Float64(y))
@inline function _artwork_point_on_segment(ax,ay,bx,by,x,y)
    # Bounding-box rejection precedes the robust general predicate. Axis
    # alignment has exact literal-coordinate collinearity and needs no
    # high-precision determinant, including zero-length segments.
    min(ax,bx)<=x<=max(ax,bx) && min(ay,by)<=y<=max(ay,by) || return false
    return (ax==bx && x==ax)||(ay==by && y==ay)||_planar_orient2d(ax,ay,bx,by,x,y)==0
end
function _artwork_contains(s::_ArtworkPolygon,x,y)
    v=s.vertices;inside=false;n=size(v,2)
    for i in 1:n
        j=mod1(i+1,n);ax,ay,bx,by=v[1,i],v[2,i],v[1,j],v[2,j]
        if _artwork_point_on_segment(ax,ay,bx,by,x,y)
            return true
        end
        (ay>y)!=(by>y) && x<(bx-ax)*(y-ay)/(by-ay)+ax && (inside=!inside)
    end
    return inside
end
@inline function _artwork_segment_distance(x,y,a,b)
    dx,dy=b[1]-a[1],b[2]-a[2];q=dx*dx+dy*dy
    t=iszero(q) ? 0. : clamp(((x-a[1])*dx+(y-a[2])*dy)/q,0.,1.)
    return hypot(x-a[1]-t*dx,y-a[2]-t*dy)
end
_artwork_contains(s::_ArtworkRoundLine,x,y)=_artwork_canonical_contains(s,Float64(x),Float64(y))
@inline function _artwork_segments_intersect(ax,ay,bx,by,cx,cy,dx,dy)
    max(min(ax,bx),min(cx,dx))<=min(max(ax,bx),max(cx,dx)) &&
        max(min(ay,by),min(cy,dy))<=min(max(ay,by),max(cy,dy)) || return false
    a=_planar_orient2d(ax,ay,bx,by,cx,cy)
    b=_planar_orient2d(ax,ay,bx,by,dx,dy)
    ((a<=0 && b>=0)||(a>=0 && b<=0)) || return false
    c=_planar_orient2d(cx,cy,dx,dy,ax,ay)
    d=_planar_orient2d(cx,cy,dx,dy,bx,by)
    return (c<=0 && d>=0)||(c>=0 && d<=0)
end
function _artwork_contains(s::_ArtworkPolygonLine,x,y)
    # A swept query point defines a segment in the aperture coordinates.
    # It occupies copper iff an endpoint is inside or that segment crosses
    # the aperture boundary. This retains concavities and zero-length paths.
    ax,ay=x-s.start[1],y-s.start[2]
    bx,by=x-s.stop[1],y-s.stop[2]
    (_artwork_contains(s.aperture,ax,ay)||_artwork_contains(s.aperture,bx,by)) && return true
    v=s.aperture.vertices
    for i in axes(v,2)
        j=mod1(i+1,size(v,2))
        _artwork_segments_intersect(ax,ay,bx,by,v[1,i],v[2,i],v[1,j],v[2,j]) && return true
    end
    return false
end
function _artwork_contains(s::_ArtworkRoundArc,x,y)
    _artwork_canonical_contains(s,Float64(x),Float64(y))
end

function _artwork_contains(s::_ArtworkPolygonArc,x,y)
    path=s.path;aperture=s.aperture
    (_artwork_contains(aperture,x-path.start[1],y-path.start[2]) ||
     _artwork_contains(aperture,x-path.stop[1],y-path.stop[2])) && return true
    # A point is swept iff the arc intersects the translated, reflected
    # aperture polygon. Circle/edge roots preserve the exact curved path;
    # the aperture keeps its orientation throughout the stroke.
    cx,cy=path.center;radius=hypot(path.start[1]-cx,path.start[2]-cy)
    vertices=aperture.vertices;n=size(vertices,2)
    for i in 1:n
        j=mod1(i+1,n)
        ax=x-vertices[1,i]-cx;ay=y-vertices[2,i]-cy
        dx=vertices[1,i]-vertices[1,j];dy=vertices[2,i]-vertices[2,j]
        a=dx*dx+dy*dy;b=2muladd(ax,dx,ay*dy);c=muladd(ax,ax,ay*ay)-radius*radius
        discriminant=muladd(b,b,-4a*c);discriminant>=0 || continue
        q=-.5*(b+copysign(sqrt(discriminant),b))
        roots=iszero(q) ? (-b/(2a),-b/(2a)) : (q/a,c/q)
        for t in roots
            0<=t<=1 || continue
            _artwork_arc_fraction(path,atan(ay+t*dy,ax+t*dx))<=1 && return true
        end
    end
    return false
end
@inline _artwork_contains(s::_ArtworkStrokeReference,x,y)=_artwork_contains(s.geometry,x,y)
function _artwork_contains(s::_ArtworkComposite,x,y)
    inside=false
    for (dark,shape) in s.parts
        _artwork_contains(shape,x,y) && (inside=dark)
    end
    return inside
end
@inline _artwork_tuple_contains(::Tuple{},x,y,inside)=inside
@inline function _artwork_tuple_contains(parts::Tuple,x,y,inside)
    dark,shape=first(parts)
    return _artwork_tuple_contains(Base.tail(parts),x,y,_artwork_contains(shape,x,y) ? dark : inside)
end
_artwork_contains(s::_ArtworkComposite{P},x,y) where {P<:Tuple}=
    _artwork_tuple_contains(s.parts,x,y,false)
function _artwork_contains(s::_ArtworkComposite{Vector{Tuple{Bool,_ArtworkStrokeReferenceUnion}}},x,y)
    inside=false
    for (dark,shape) in s.parts
        hit=shape isa _ArtworkStrokeReference{_ArtworkRoundLine} ? _artwork_contains(shape,x,y) :
            _artwork_contains(shape::_ArtworkStrokeReference{_ArtworkPolygon},x,y)
        hit && (inside=dark)
    end
    return inside
end
function _artwork_contains(s::_ArtworkRegion,x,y)
    _artwork_canonical_contains(s,Float64(x),Float64(y))
end

function _artwork_contains(s::_ArtworkTransform,x,y)
    if _artwork_canonical_leaf(s)
        return _artwork_canonical_contains(s,Float64(x),Float64(y))
    end
    p=s.inverse*SVector(x-s.origin[1],y-s.origin[2])
    return _artwork_contains(s.shape,p[1],p[2])
end

function _artwork_object!(objects,layer,shape,dark,attributes,max_objects)
    length(objects)<max_objects || throw(ArgumentError("artwork exceeds max_objects"))
    bounds=_artwork_bounds(shape)
    all(isfinite,bounds) || throw(ArgumentError("artwork bounds must fit finite SI coordinates"))
    push!(objects,PlanarArtworkObject(String(layer),shape,Bool(dark),bounds,
        Dict(String(k)=>String.(v) for (k,v) in attributes)))
end

# Resolve one object's shape type once, outside its cell loop. This avoids
# boxing x/y for dynamic dispatch at every raster sample. The exact geometric
# predicate is unchanged and the output mask owns the only grid-sized payload.
function _artwork_mask_object!(mask,shape::S,xs,ys,bounds,dark) where {S<:_ArtworkShape}
    x0,x1,y0,y1=bounds
    ilo=searchsortedfirst(xs,x0);ihi=searchsortedlast(xs,x1)
    jlo=searchsortedfirst(ys,y0);jhi=searchsortedlast(ys,y1)
    for j in jlo:jhi,i in ilo:ihi
        _artwork_contains(shape,xs[i],ys[j]) && (mask[i,j]=dark)
    end
    return mask
end

"""Evaluate ordered fabrication artwork exactly at solver cell centers.
Circles, circular strokes/arcs and transformed aperture holes retain
analytic membership. `offset` is an explicit SI translation. All original
layer names remain keys of the returned BitMatrix dictionary. Negative
images are complemented only over this analysis grid."""
function artwork_cell_masks(doc::PlanarArtwork,grid::CellGrid;offset=(0.,0.),
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    length(offset)==2 && all(isfinite,offset) || throw(ArgumentError("artwork offset must contain two finite SI coordinates"))
    layers=unique([o.layer for o in doc.objects]);append!(layers,setdiff(collect(doc.negative_layers),layers))
    payload=_checked_payload_sum("artwork cell masks",_checked_array_payload_bytes(UInt64,length(layers),cld(BigInt(grid.nx)*grid.ny,64)),
        _checked_array_payload_bytes(Float64,grid.nx+grid.ny))
    _enforce_payload_limit(payload,max_bytes,"artwork cell masks","max_bytes")
    masks=Dict(layer=>falses(grid.nx,grid.ny) for layer in layers)
    xs=[(i-.5)*grid.dx-offset[1] for i in 1:grid.nx]
    ys=[(j-.5)*grid.dy-offset[2] for j in 1:grid.ny]
    for object in doc.objects
        _artwork_mask_object!(masks[object.layer],object.shape,xs,ys,object.bounds,object.dark)
    end
    for layer in doc.negative_layers
        masks[layer].=.!masks[layer]
    end
    return masks
end

"""Lower fabrication artwork masks onto explicit sheet interfaces and
via-layer spans. Mapping values are named tuples `(;kind=:sheet,interface)`
or `(;kind=:via,from_interface,to_interface)`. Wall connection flags follow
the physically occupied boundaries. No automatic centering/scaling occurs.
The explicit `ports` use the sorted sheet-interface or via-layer order."""
function artwork_planar_problem(doc::PlanarArtwork,stack::PlanarStackup,grid::CellGrid,mapping::AbstractDict,
        ports::Vector{PlanarPort};offset=(0.,0.),max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    planar_validate(stack)
    stack.a==grid.a && stack.b==grid.b || throw(ArgumentError("artwork stack/grid dimensions differ"))
    levels=Set{Int}();vlevels=Set{Int}()
    layers=union(Set(o.layer for o in doc.objects),doc.negative_layers)
    for layer in layers
        haskey(mapping,layer) || throw(ArgumentError("artwork layer $layer has no technology mapping"))
        map=mapping[layer]
        if map.kind==:sheet
            0<=map.interface<=length(stack.layers) || throw(ArgumentError("sheet mapping outside stack"))
            push!(levels,Int(map.interface))
        elseif map.kind==:via
            lo,hi=minmax(map.from_interface,map.to_interface)
            0<=lo<hi<=length(stack.layers) || throw(ArgumentError("via mapping outside stack"))
            union!(vlevels,lo+1:hi)
        else
            throw(ArgumentError("artwork technology kind must be sheet or via"))
        end
    end
    ns,nv=length(levels),length(vlevels);cells=BigInt(grid.nx)*grid.ny
    # Reserve masks, geometry and worst-case basis payload before rendering.
    reserve=_checked_payload_sum("artwork lowering",_checked_array_payload_bytes(Int,64,ns+nv,cells),
        _checked_array_payload_bytes(UInt64,4,ns+2nv,cld(cells,64)))
    _enforce_payload_limit(reserve,max_bytes,"artwork lowering","max_bytes")
    masks=artwork_cell_masks(doc,grid;offset,max_bytes=max_bytes-reserve)
    sheets=[sheet_level(i,grid.nx,grid.ny) for i in sort!(collect(levels))]
    vias=[via_level(i,grid.nx,grid.ny) for i in sort!(collect(vlevels))]
    for (name,mask) in masks
        haskey(mapping,name) || throw(ArgumentError("artwork layer $name has no technology mapping"))
        map=mapping[name]
        if map.kind==:sheet
            sh=only(s for s in sheets if s.interface==map.interface);sh.mask.|=mask
        else
            lo,hi=minmax(map.from_interface,map.to_interface)
            for via in vias
                lo<via.layer<=hi || continue
                via.uni.|=mask;via.tap.|=mask
            end
        end
    end
    for sh in sheets
        sh.connect_west.=sh.mask[1,:];sh.connect_east.=sh.mask[end,:]
        sh.connect_south.=sh.mask[:,1];sh.connect_north.=sh.mask[:,end]
    end
    if isempty(ports)
        basis=build_planar_basis(grid,sheets,ports;vias)
        return PlanarProblem(stack,grid,sheets,ports,vias,basis,VolLevel[])
    end
    return build_planar_problem(stack,grid,sheets,ports;vias)
end
