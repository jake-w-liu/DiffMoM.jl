# Exact translation sweep of an ordered analytic aperture.
# A query segment in aperture coordinates is partitioned at primitive
# boundary candidates; original ordered membership decides each interval.
# Holes therefore remain local to every translated aperture. Ordinary
# interval-certified queries allocate no storage; rare exact witnesses use
# a bounded workspace. Construction bounds nesting and primitive work.

struct _ArtworkCompositeLine{S<:_ArtworkShape} <: _ArtworkShape
    aperture::S
    start::NTuple{2,Float64}
    stop::NTuple{2,Float64}
    boundary_count::Int
    exact_bytes::Int
end
# Preserve the earlier private constructor used by geometric adapters.
function _ArtworkCompositeLine(aperture,start,stop,count::Int)
    bytes=max(0,_DEFAULT_MAX_DENSE_PAYLOAD_BYTES-Base.summarysize(aperture)-256)
    _ArtworkCompositeLine(aperture,start,stop,count,bytes)
end
function _artwork_bounds(s::_ArtworkCompositeLine)
    _artwork_certified_bounds(s)
end
@inline function _artwork_stroke_candidate(t,left,right)
    isfinite(t) && left<t<right ? t : right
end
function _artwork_stroke_circle_next(c,r,a,b,left,right)
    ax,ay=a[1]-c[1],a[2]-c[2];dx,dy=b[1]-a[1],b[2]-a[2]
    scale=max(abs(ax),abs(ay),abs(dx),abs(dy),r)
    iszero(scale) && return right
    ax/=scale;ay/=scale;dx/=scale;dy/=scale;r/=scale
    A=muladd(dx,dx,dy*dy);iszero(A) && return right
    B=2muladd(ax,dx,ay*dy);C=muladd(ax,ax,ay*ay)-r*r
    D=muladd(B,B,-4A*C);D<0 && return right
    q=-.5*(B+copysign(sqrt(D),B))
    t1=iszero(q) ? -B/(2A) : q/A;t2=iszero(q) ? t1 : C/q
    _artwork_stroke_candidate(t2,left,_artwork_stroke_candidate(t1,left,right))
end
function _artwork_stroke_edge_next(c,d,a,b,left,right)
    dx,dy=b[1]-a[1],b[2]-a[2];ex,ey=d[1]-c[1],d[2]-c[2]
    hx,hy=c[1]-a[1],c[2]-a[2]
    scale=max(abs(dx),abs(dy),abs(ex),abs(ey),abs(hx),abs(hy))
    iszero(scale) && return right
    dx/=scale;dy/=scale;ex/=scale;ey/=scale;hx/=scale;hy/=scale
    A=dx*ey-dy*ex
    if !iszero(A)
        return _artwork_stroke_candidate((hx*ey-hy*ex)/A,left,right)
    end
    if abs(dx)>=abs(dy) && !iszero(dx)
        return _artwork_stroke_candidate(((d[1]-a[1])/scale)/dx,left,_artwork_stroke_candidate(hx/dx,left,right))
    elseif !iszero(dy)
        return _artwork_stroke_candidate(((d[2]-a[2])/scale)/dy,left,_artwork_stroke_candidate(hy/dy,left,right))
    end
    right
end
_artwork_stroke_next_boundary(::_ArtworkEmpty,a,b,left,right)=right
_artwork_stroke_next_boundary(s::_ArtworkCircle,a,b,left,right)=_artwork_stroke_circle_next(s.center,s.radius,a,b,left,right)
_artwork_stroke_next_boundary(s::_ArtworkStrokeReference,a,b,left,right)=_artwork_stroke_next_boundary(s.geometry,a,b,left,right)
function _artwork_stroke_next_boundary(s::_ArtworkPolygon,a,b,left,right)
    v=s.vertices
    for i in axes(v,2)
        j=mod1(i+1,size(v,2))
        right=_artwork_stroke_edge_next((v[1,i],v[2,i]),(v[1,j],v[2,j]),a,b,left,right)
    end
    right
end
function _artwork_stroke_next_boundary(s::_ArtworkRegion,a,b,left,right)
    for segment in s.segments
        if segment isa _ArtworkRoundLine
            right=_artwork_stroke_edge_next(segment.start,segment.stop,a,b,left,right)
        else
            r=hypot(segment.start[1]-segment.center[1],segment.start[2]-segment.center[2])
            right=_artwork_stroke_circle_next(segment.center,r,a,b,left,right)
        end
    end
    right
end
function _artwork_stroke_next_boundary(s::_ArtworkRoundLine,a,b,left,right)
    right=_artwork_stroke_circle_next(s.start,s.radius,a,b,left,right)
    right=_artwork_stroke_circle_next(s.stop,s.radius,a,b,left,right)
    dx,dy=s.stop[1]-s.start[1],s.stop[2]-s.start[2];length=hypot(dx,dy)
    iszero(length) && return right
    nx,ny=-dy/length*s.radius,dx/length*s.radius
    for sign in (-1.,1.)
        c=(s.start[1]+sign*nx,s.start[2]+sign*ny)
        d=(s.stop[1]+sign*nx,s.stop[2]+sign*ny)
        right=_artwork_stroke_edge_next(c,d,a,b,left,right)
    end
    right
end
function _artwork_stroke_next_boundary(s::_ArtworkRoundArc,a,b,left,right)
    r=hypot(s.start[1]-s.center[1],s.start[2]-s.center[2])
    right=_artwork_stroke_circle_next(s.center,r+s.radius,a,b,left,right)
    right=_artwork_stroke_circle_next(s.center,max(0.,r-s.radius),a,b,left,right)
    right=_artwork_stroke_circle_next(s.start,s.radius,a,b,left,right)
    _artwork_stroke_circle_next(s.stop,s.radius,a,b,left,right)
end
function _artwork_stroke_next_boundary(s::_ArtworkPolygonLine,a,b,left,right)
    v=s.aperture.vertices
    for i in axes(v,2)
        j=mod1(i+1,size(v,2))
        c=(v[1,i]+s.start[1],v[2,i]+s.start[2])
        d=(v[1,j]+s.start[1],v[2,j]+s.start[2])
        e=(v[1,i]+s.stop[1],v[2,i]+s.stop[2])
        f=(v[1,j]+s.stop[1],v[2,j]+s.stop[2])
        right=_artwork_stroke_edge_next(c,d,a,b,left,right)
        right=_artwork_stroke_edge_next(e,f,a,b,left,right)
        right=_artwork_stroke_edge_next(c,e,a,b,left,right)
    end
    right
end
function _artwork_stroke_next_boundary(s::_ArtworkComposite,a,b,left,right)
    for (_,part) in s.parts
        right=_artwork_stroke_next_boundary(part,a,b,left,right)
    end
    right
end
@inline _artwork_stroke_next_tuple(::Tuple{},a,b,left,right)=right
@inline function _artwork_stroke_next_tuple(parts::Tuple,a,b,left,right)
    _artwork_stroke_next_tuple(Base.tail(parts),a,b,left,_artwork_stroke_next_boundary(first(parts)[2],a,b,left,right))
end
@inline _artwork_stroke_next_boundary(s::_ArtworkComposite{P},a,b,left,right) where {P<:Tuple}=
    _artwork_stroke_next_tuple(s.parts,a,b,left,right)
function _artwork_stroke_next_boundary(s::_ArtworkTransform,a,b,left,right)
    aa=s.inverse*SVector(a[1]-s.origin[1],a[2]-s.origin[2])
    bb=s.inverse*SVector(b[1]-s.origin[1],b[2]-s.origin[2])
    _artwork_stroke_next_boundary(s.shape,(aa[1],aa[2]),(bb[1],bb[2]),left,right)
end
function _artwork_contains(s::_ArtworkCompositeLine,x,y)
    _artwork_certified_stroke(s,Float64(x),Float64(y))
end

function _artwork_stroke_count(n,limit)
    n<=limit || throw(ArgumentError("artwork stroke exceeds max_stroke_boundaries"))
    n
end
_artwork_stroke_boundaries(s::_ArtworkShape,limit,depth)=
    throw(ArgumentError("unsupported analytic geometry in composite line stroke: $(typeof(s))"))
_artwork_stroke_boundaries(::_ArtworkEmpty,limit,depth)=_artwork_stroke_count(1,limit)
function _artwork_stroke_boundaries(s::_ArtworkCircle,limit,depth)
    all(isfinite,(s.center...,s.radius)) && s.radius>=0 ||
        throw(ArgumentError("invalid circle aperture geometry"))
    _artwork_stroke_count(2,limit)
end
function _artwork_stroke_boundaries(s::_ArtworkPolygon,limit,depth)
    size(s.vertices,1)==2 && size(s.vertices,2)>=3 && all(isfinite,s.vertices) ||
        throw(ArgumentError("invalid polygon aperture geometry"))
    _artwork_stroke_count(_checked_array_payload_bytes(UInt8,2,size(s.vertices,2)),limit)
end
function _artwork_stroke_boundaries(s::_ArtworkRoundLine,limit,depth)
    all(isfinite,(s.start...,s.stop...,s.radius)) && s.radius>=0 ||
        throw(ArgumentError("invalid capsule aperture geometry"))
    _artwork_stroke_count(8,limit)
end
function _artwork_stroke_boundaries(s::_ArtworkRoundArc,limit,depth)
    all(isfinite,(s.start...,s.stop...,s.center...,s.radius)) && s.radius>=0 ||
        throw(ArgumentError("invalid circular arc aperture geometry"))
    r=hypot(s.start[1]-s.center[1],s.start[2]-s.center[2])
    isfinite(r+s.radius) || throw(ArgumentError("circular arc aperture extent overflows"))
    _artwork_stroke_count(8,limit)
end
function _artwork_stroke_boundaries(s::_ArtworkPolygonLine,limit,depth)
    all(isfinite,(s.start...,s.stop...)) || throw(ArgumentError("nonfinite polygon stroke aperture"))
    _artwork_stroke_boundaries(s.aperture,limit,depth)
    _artwork_stroke_count(_checked_array_payload_bytes(UInt8,6,size(s.aperture.vertices,2)),limit)
end
function _artwork_stroke_boundaries(s::_ArtworkRegion,limit,depth)
    isempty(s.segments) && throw(ArgumentError("empty contour aperture geometry"))
    count=0
    for segment in s.segments
        all(isfinite,(segment.start...,segment.stop...)) ||
            throw(ArgumentError("nonfinite contour aperture geometry"))
        segment isa _ArtworkRoundArc && !all(isfinite,segment.center) &&
            throw(ArgumentError("nonfinite contour aperture center"))
        if segment isa _ArtworkRoundArc
            isfinite(hypot(segment.start[1]-segment.center[1],segment.start[2]-segment.center[2]))||
                throw(ArgumentError("circular contour extent overflows"))
            count=_artwork_stroke_count(_checked_payload_sum("contour boundary count",count,4),limit)
        else
            count=_artwork_stroke_count(_checked_payload_sum("contour boundary count",count,2),limit)
        end
    end
    count
end
function _artwork_stroke_boundaries(s::_ArtworkComposite,limit,depth)
    depth<64 || throw(ArgumentError("artwork aperture nesting exceeds 64 levels"))
    count=_artwork_stroke_count(1,limit)
    for (_,part) in s.parts
        count+=_artwork_stroke_boundaries(part,limit-count,depth+1)
    end
    count
end
function _artwork_stroke_boundaries(s::_ArtworkTransform,limit,depth)
    depth<64 || throw(ArgumentError("artwork aperture nesting exceeds 64 levels"))
    all(isfinite,s.matrix) && all(isfinite,s.inverse) && all(isfinite,s.origin) ||
        throw(ArgumentError("nonfinite transformed aperture geometry"))
    _artwork_stroke_count(1+_artwork_stroke_boundaries(s.shape,limit-1,depth+1),limit)
end
function _artwork_stroke_boundaries(s::_ArtworkStrokeReference,limit,depth)
    depth<64 || throw(ArgumentError("artwork aperture nesting exceeds 64 levels"))
    _artwork_stroke_count(1+_artwork_stroke_boundaries(s.geometry,limit-1,depth+1),limit)
end
function _artwork_geometry_depth_guard(s::_ArtworkShape,depth=0)
    depth<=64 || throw(ArgumentError("artwork geometry nesting exceeds 64 levels"))
    for i in 1:fieldcount(typeof(s))
        part=getfield(s,i)
        part isa _ArtworkShape && _artwork_geometry_depth_guard(part,depth+1)
    end
    nothing
end
function _artwork_geometry_depth_guard(s::_ArtworkComposite,depth=0)
    depth<64 || throw(ArgumentError("artwork geometry nesting exceeds 64 levels"))
    for (_,part) in s.parts
        _artwork_geometry_depth_guard(part,depth+1)
    end
    nothing
end
function _artwork_composite_line(aperture,start,stop;max_boundaries,max_bytes=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        _copy_aperture::Bool=true)
    limit=_validated_resource_limit("max_stroke_boundaries",max_boundaries)
    all(isfinite,(start...,stop...)) || throw(ArgumentError("nonfinite artwork stroke endpoints"))
    count=_artwork_stroke_boundaries(aperture,limit,0)
    bytes=_validated_resource_limit("max_bytes",max_bytes)
    stroke=_ArtworkCompositeLine(aperture,start,stop,count,bytes)
    owned=Base.summarysize(stroke)
    _enforce_payload_limit(owned,bytes,"owned artwork composite stroke","max_bytes")
    stroke=_ArtworkCompositeLine(aperture,start,stop,count,bytes-owned)
    bounds=_artwork_bounds(aperture)
    all(isfinite,bounds) && bounds[1]<=bounds[2] && bounds[3]<=bounds[4] ||
        throw(ArgumentError("invalid artwork aperture bounds"))
    all(isfinite,_artwork_bounds(stroke)) || throw(ArgumentError("artwork stroke bounds overflow"))
    # The reader retains lookup geometry as well. Reserve this separate
    # owned output before copying so caller mutations cannot invalidate
    # cached bounds, the depth guard or the boundary count.
    _copy_aperture ? deepcopy(stroke) : stroke
end
