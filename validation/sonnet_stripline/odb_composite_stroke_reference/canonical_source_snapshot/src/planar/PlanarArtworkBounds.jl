# Cached clipping bounds enclose the same analytic geometry as membership.
# In particular an affine shape is defined by its stored inverse; its rounded
# forward matrix is not a second definition of the aperture boundary.
_artwork_certified_bounds(s::_ArtworkShape,depth=0)=_artwork_bounds(s)
_artwork_certified_bounds(::Union{_ArtworkEmpty,_ArtworkODBNull},depth=0)=(0.,0.,0.,0.)
function _artwork_bounds_depth(depth)
    depth<=64||throw(ArgumentError("artwork bounds nesting exceeds 64 levels"))
end
@inline _artwork_interval_bounds(x,y)=(x.lo,x.hi,y.lo,y.hi)
@inline function _artwork_bounds_union(a,b)
    (min(a[1],b[1]),max(a[2],b[2]),min(a[3],b[3]),max(a[4],b[4]))
end
function _artwork_certified_bounds(s::_ArtworkCircle,depth=0)
    r=_ArtworkInterval(s.radius);x,y=_artwork_interval_point(s.center)
    ((x-r).lo,(x+r).hi,(y-r).lo,(y+r).hi)
end
function _artwork_certified_bounds(s::_ArtworkPolygon,depth=0)
    v=s.vertices;x0=x1=v[1,1];y0=y1=v[2,1]
    for i in axes(v,2)
        x0=min(x0,v[1,i]);x1=max(x1,v[1,i]);y0=min(y0,v[2,i]);y1=max(y1,v[2,i])
    end
    (x0,x1,y0,y1)
end
function _artwork_certified_bounds(s::_ArtworkRoundLine,depth=0)
    r=_ArtworkInterval(s.radius)
    x=_ArtworkInterval(min(s.start[1],s.stop[1]),max(s.start[1],s.stop[1]))
    y=_ArtworkInterval(min(s.start[2],s.stop[2]),max(s.start[2],s.stop[2]))
    ((x-r).lo,(x+r).hi,(y-r).lo,(y+r).hi)
end
function _artwork_certified_bounds(s::_ArtworkCornerRectangle,depth=0)
    x,y=_artwork_interval_half(s.width),_artwork_interval_half(s.height)
    (-x.hi,x.hi,-y.hi,y.hi)
end
function _artwork_certified_bounds(s::_ArtworkRoundArc,depth=0)
    r=hypot(s.start[1]-s.center[1],s.start[2]-s.center[2])
    isfinite(r)||throw(ArgumentError("circular artwork bounds overflow"))
    cx,cy=_artwork_interval_point(s.center);ri=_ArtworkInterval(r);w=_ArtworkInterval(s.radius)
    x=_ArtworkInterval(min(s.start[1],s.stop[1]),max(s.start[1],s.stop[1]))
    y=_ArtworkInterval(min(s.start[2],s.stop[2]),max(s.start[2],s.stop[2]))
    for direction in ((1.,0.),(0.,1.),(-1.,0.),(0.,-1.))
        state=_artwork_certified_arc_direction(s,_artwork_interval_point(direction))
        state==-1&&(state=Int8(_artwork_exact_bounds_arc_direction(s,direction)))
        state==0&&continue
        xx=cx+(direction[1]==0 ? _ArtworkInterval(0.) : direction[1]>0 ? ri : -ri)
        yy=cy+(direction[2]==0 ? _ArtworkInterval(0.) : direction[2]>0 ? ri : -ri)
        x=_ArtworkInterval(min(x.lo,xx.lo),max(x.hi,xx.hi))
        y=_ArtworkInterval(min(y.lo,yy.lo),max(y.hi,yy.hi))
    end
    ((x-w).lo,(x+w).hi,(y-w).lo,(y+w).hi)
end
function _artwork_certified_bounds(s::_ArtworkRegion,depth=0)
    isempty(s.segments)&&throw(ArgumentError("empty artwork contour"))
    bounds=_artwork_certified_bounds(first(s.segments),depth)
    for part in Iterators.drop(s.segments,1)
        bounds=_artwork_bounds_union(bounds,_artwork_certified_bounds(part,depth))
    end
    bounds
end
function _artwork_certified_bounds(s::_ArtworkComposite,depth=0)
    _artwork_bounds_depth(depth)
    isempty(s.parts)&&return (0.,0.,0.,0.)
    bounds=_artwork_certified_bounds(first(s.parts)[2],depth+1)
    for (_,part) in Iterators.drop(s.parts,1)
        bounds=_artwork_bounds_union(bounds,_artwork_certified_bounds(part,depth+1))
    end
    bounds
end
function _artwork_certified_bounds(s::_ArtworkStrokeReference,depth=0)
    _artwork_bounds_depth(depth)
    _artwork_certified_bounds(s.geometry,depth+1)
end
_artwork_certified_bounds(s::_ArtworkODBDrill,depth=0)=_artwork_certified_bounds(s.circle,depth)
_artwork_certified_bounds(s::_ArtworkODBButterfly,depth=0)=_artwork_certified_bounds(s.outside,depth)
function _artwork_certified_translated_bounds(s,depth)
    bounds=_artwork_certified_bounds(s.aperture,depth)
    x=_ArtworkInterval(bounds[1],bounds[2])+_ArtworkInterval(min(s.start[1],s.stop[1]),max(s.start[1],s.stop[1]))
    y=_ArtworkInterval(bounds[3],bounds[4])+_ArtworkInterval(min(s.start[2],s.stop[2]),max(s.start[2],s.stop[2]))
    _artwork_interval_bounds(x,y)
end
_artwork_certified_bounds(s::_ArtworkPolygonLine,depth=0)=_artwork_certified_translated_bounds(s,depth)
_artwork_certified_bounds(s::_ArtworkCompositeLine,depth=0)=_artwork_certified_translated_bounds(s,depth)
@inline function _artwork_interval_scale2(x::Float64,e::Int)
    value=ldexp(x,e)
    if isfinite(value)&&ldexp(value,-e)==x
        return _ArtworkInterval(value)
    end
    _ArtworkInterval(prevfloat(value),nextfloat(value))
end
@inline function _artwork_interval_scale2(x::_ArtworkInterval,e::Int)
    _ArtworkInterval(_artwork_interval_scale2(x.lo,e).lo,_artwork_interval_scale2(x.hi,e).hi)
end
function _artwork_inverse_matrix(matrix)
    all(isfinite,matrix)||throw(ArgumentError("nonfinite artwork transformation"))
    scale=maximum(abs,matrix);scale>0||throw(ArgumentError("singular artwork transformation"))
    if iszero(matrix[1,2])&&iszero(matrix[2,1])
        !iszero(matrix[1,1])&&!iszero(matrix[2,2])||throw(ArgumentError("singular artwork transformation"))
        inverse=inv(matrix)
        if all(isfinite,inverse)&&!iszero(inverse[1,1])&&!iszero(inverse[2,2])
            return inverse
        end
        # Separate diagonal reciprocals also cover anisotropic scales whose
        # exponent difference exceeds the Float64 subnormal range.
        inverse=typeof(matrix)(inv(matrix[1,1]),0.,0.,inv(matrix[2,2]))
        all(isfinite,inverse)&&!iszero(inverse[1,1])&&!iszero(inverse[2,2])||
            throw(ArgumentError("artwork inverse transformation is not representable in Float64"))
        return inverse
    end
    _,e=frexp(scale)
    # Power-of-two normalization preserves ordinary inverse bits while
    # preventing the determinant from overflowing or underflowing.
    normalized=map(value->ldexp(value,-e),matrix)
    inverse=map(value->ldexp(value,-e),inv(normalized))
    all(isfinite,inverse)&&maximum(abs,inverse)>0||
        throw(ArgumentError("artwork inverse transformation is not representable in Float64"))
    inverse
end
function _artwork_certified_bounds(s::_ArtworkTransform,depth=0)
    _artwork_bounds_depth(depth)
    bounds=_artwork_certified_bounds(s.shape,depth+1);m=s.inverse
    all(isfinite,m)||throw(ArgumentError("nonfinite artwork inverse transformation"))
    scale=maximum(abs,m);scale>0||throw(ArgumentError("singular artwork inverse transformation"))
    _,e=frexp(scale)
    a,b,c,d=(_artwork_interval_scale2(m[1,1],-e),_artwork_interval_scale2(m[1,2],-e),
        _artwork_interval_scale2(m[2,1],-e),_artwork_interval_scale2(m[2,2],-e))
    determinant=a*d-b*c
    determinant.lo<=0<=determinant.hi&&return _artwork_exact_inverse_bounds(s,bounds)
    ia,ib,ic,id=(_artwork_interval_scale2(d/determinant,-e),_artwork_interval_scale2(-b/determinant,-e),
        _artwork_interval_scale2(-c/determinant,-e),_artwork_interval_scale2(a/determinant,-e))
    x,y=_ArtworkInterval(bounds[1],bounds[2]),_ArtworkInterval(bounds[3],bounds[4])
    ox,oy=_artwork_interval_point(s.origin)
    _artwork_interval_bounds(ox+ia*x+ib*y,oy+ic*x+id*y)
end
function _artwork_exact_bounds_preflight(s,extra=())
    depth=_artwork_exact_qtrait(s)
    bits=max(_artwork_exact_operand_bits(s),_artwork_exact_operand_bits(extra))
    payload=_checked_payload_sum("exact artwork bounds workspace",Base.summarysize(s),
        _checked_array_payload_bytes(UInt8,128,depth+1,16*(bits+1)+64))
    _enforce_payload_limit(payload,_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,"exact artwork bounds workspace","max_bytes")
end
function _artwork_exact_bounds_arc_direction(s,direction)
    _artwork_exact_bounds_preflight(s,direction)
    _artwork_exact_qarc_angle(s,_artwork_exact_qadd(_artwork_exact_qpoint(s.center),_artwork_exact_qpoint(direction)))
end
function _artwork_exact_bound_float(q,lower)
    value=Float64(q)
    !isfinite(value)&&return lower ? prevfloat(value) : nextfloat(value)
    represented=_artwork_exact_q(value)
    lower ? (represented>q ? prevfloat(value) : value) : (represented<q ? nextfloat(value) : value)
end
function _artwork_exact_inverse_bounds(s,bounds)
    _artwork_exact_bounds_preflight(s,bounds)
    m=s.inverse;q=_artwork_exact_q
    a,b,c,d=q(m[1,1]),q(m[1,2]),q(m[2,1]),q(m[2,2]);determinant=a*d-b*c
    iszero(determinant)&&throw(ArgumentError("singular artwork inverse transformation"))
    ia,ib,ic,id=d/determinant,-b/determinant,-c/determinant,a/determinant
    x0,x1,y0,y1=map(q,bounds);ox,oy=_artwork_exact_qpoint(s.origin)
    xl,xh=minmax(ia*x0,ia*x1);yl,yh=minmax(ib*y0,ib*y1)
    ul,uh=minmax(ic*x0,ic*x1);vl,vh=minmax(id*y0,id*y1)
    (_artwork_exact_bound_float(ox+xl+yl,true),_artwork_exact_bound_float(ox+xh+yh,false),
        _artwork_exact_bound_float(oy+ul+vl,true),_artwork_exact_bound_float(oy+uh+vh,false))
end
