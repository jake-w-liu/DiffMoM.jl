# Allocation-free outward intervals certify ordinary aperture queries. Exact
# algebraic witnesses resolve the remaining closed-boundary cases.
struct _ArtworkInterval
    lo::Float64
    hi::Float64
end
_ArtworkInterval(x::Float64)=_ArtworkInterval(x,x)
@inline _artwork_interval_zero(a)=iszero(a.lo)&&iszero(a.hi)
@inline function _artwork_interval_mid(a)
    a.lo==a.hi&&return a.lo
    mid=a.lo+(a.hi-a.lo)/2
    isfinite(mid)||(mid=a.lo/2+a.hi/2)
    clamp(mid,a.lo,a.hi)
end
@inline function _artwork_interval_sum_bounds(a,b)
    s=a+b
    isfinite(s)||return (prevfloat(s),nextfloat(s))
    z=s-a;err=(a-(s-z))+(b-z)
    err==0 ? (s,s) : err>0 ? (s,nextfloat(s)) : (prevfloat(s),s)
end
@inline function Base.:+(a::_ArtworkInterval,b::_ArtworkInterval)
    _artwork_interval_zero(a)&&return b
    _artwork_interval_zero(b)&&return a
    lo,_=_artwork_interval_sum_bounds(a.lo,b.lo)
    _,hi=_artwork_interval_sum_bounds(a.hi,b.hi)
    _ArtworkInterval(lo,hi)
end
@inline Base.:-(a::_ArtworkInterval)=_ArtworkInterval(-a.hi,-a.lo)
@inline function Base.:-(a::_ArtworkInterval,b::_ArtworkInterval)
    _artwork_interval_zero(b)&&return a
    a.lo==a.hi==b.lo==b.hi&&return _ArtworkInterval(0.)
    a+(-b)
end
@inline function Base.:*(a::_ArtworkInterval,b::_ArtworkInterval)
    (_artwork_interval_zero(a)||_artwork_interval_zero(b))&&return _ArtworkInterval(0.)
    aa,bb,cc,dd=a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi
    any(isnan,(aa,bb,cc,dd))&&return _ArtworkInterval(-Inf,Inf)
    _ArtworkInterval(prevfloat(min(aa,bb,cc,dd)),nextfloat(max(aa,bb,cc,dd)))
end
@inline function Base.:/(a::_ArtworkInterval,b::_ArtworkInterval)
    b.lo<=0<=b.hi&&return _ArtworkInterval(-Inf,Inf)
    a*_ArtworkInterval(prevfloat(inv(b.hi)),nextfloat(inv(b.lo)))
end
for op in (:+,:-,:*,:/)
    @eval @inline Base.$op(a::_ArtworkInterval,b::Float64)=Base.$op(a,_ArtworkInterval(b))
    @eval @inline Base.$op(a::Float64,b::_ArtworkInterval)=Base.$op(_ArtworkInterval(a),b)
end
@inline function _artwork_interval_square(a)
    _artwork_interval_zero(a)&&return _ArtworkInterval(0.)
    lo=a.lo<=0<=a.hi ? 0. : max(0.,prevfloat(min(a.lo*a.lo,a.hi*a.hi)))
    _ArtworkInterval(lo,nextfloat(max(a.lo*a.lo,a.hi*a.hi)))
end
@inline _artwork_interval_abs(a)=a.lo>=0 ? a : a.hi<=0 ? -a : _ArtworkInterval(0.,max(-a.lo,a.hi))
@inline function _artwork_interval_sqrt(a)
    _ArtworkInterval(max(0.,prevfloat(sqrt(max(0.,a.lo)))),nextfloat(sqrt(max(0.,a.hi))))
end
@inline function _artwork_interval_le(a,b)
    all(isfinite,(a.lo,a.hi,b.lo,b.hi))||return Int8(-1)
    a.hi<=b.lo ? Int8(1) : a.lo>b.hi ? Int8(0) : Int8(-1)
end
@inline _artwork_interval_union(a,b)=a==1||b==1 ? Int8(1) : a==0&&b==0 ? Int8(0) : Int8(-1)
@inline _artwork_interval_intersection(a,b)=a==0||b==0 ? Int8(0) : a==1&&b==1 ? Int8(1) : Int8(-1)
@inline _artwork_interval_cross(a,b)=a[1]*b[2]-a[2]*b[1]
@inline _artwork_interval_dot(a,b)=a[1]*b[1]+a[2]*b[2]
@inline function _artwork_interval_sign(a)
    all(isfinite,(a.lo,a.hi))||return Int8(2)
    a.lo>0 ? Int8(1) : a.hi<0 ? Int8(-1) : a.lo==a.hi==0 ? Int8(0) : Int8(2)
end
@inline function _artwork_certified_ray_cross(a,b,p)
    (a[2]>p[2])!=(b[2]>p[2])||return Int8(0)
    # The vertical ray intersection is the stored edge coordinate itself;
    # no products are needed even at subnormal or extreme finite scales.
    a[1]==b[1]&&return Int8(p[1]<a[1])
    aa,bb,pp=_artwork_interval_point(a),_artwork_interval_point(b),_artwork_interval_point(p)
    sign=_artwork_interval_sign(_artwork_interval_cross(_artwork_interval_sub(bb,aa),_artwork_interval_sub(pp,aa)))
    abs(sign)==1||return Int8(-1)
    Int8(b[2]>a[2] ? sign>0 : sign<0)
end
@inline _artwork_interval_point(p)=(_ArtworkInterval(p[1]),_ArtworkInterval(p[2]))
@inline _artwork_interval_sub(a,b)=(a[1]-b[1],a[2]-b[2])
@inline _artwork_interval_xy(a,v,t)=(a[1]+v[1]*t,a[2]+v[2]*t)
@inline function _artwork_interval_box(x,y,x0,x1,y0,y1)
    x.hi<x0||x.lo>x1||y.hi<y0||y.lo>y1 ? Int8(0) :
        x.lo>=x0&&x.hi<=x1&&y.lo>=y0&&y.hi<=y1 ? Int8(1) : Int8(-1)
end
_artwork_certified_contains(::_ArtworkEmpty,x,y)=Int8(0)
_artwork_certified_contains(::_ArtworkODBNull,x,y)=Int8(0)
@inline function _artwork_certified_circle(c,r,x,y)
    dx,dy=x-c[1],y-c[2]
    _artwork_interval_le(_artwork_interval_square(dx)+_artwork_interval_square(dy),_artwork_interval_square(_ArtworkInterval(r)))
end
_artwork_certified_contains(s::_ArtworkCircle,x,y)=_artwork_certified_circle(s.center,s.radius,x,y)
_artwork_certified_contains(s::_ArtworkODBDrill,x,y)=_artwork_certified_contains(s.circle,x,y)
function _artwork_certified_contains(s::_ArtworkRoundLine,x,y)
    a,b=_artwork_interval_point(s.start),_artwork_interval_point(s.stop)
    e=_artwork_interval_sub(b,a);u=(x-a[1],y-a[2]);den=_artwork_interval_dot(e,e)
    s.start==s.stop&&return _artwork_certified_circle(s.start,s.radius,x,y)
    den.lo>0||return Int8(-1)
    t=_artwork_interval_dot(u,e)/den
    t=_ArtworkInterval(clamp(t.lo,0.,1.),clamp(t.hi,0.,1.))
    dx,dy=u[1]-e[1]*t,u[2]-e[2]*t
    _artwork_interval_le(_artwork_interval_square(dx)+_artwork_interval_square(dy),_artwork_interval_square(_ArtworkInterval(s.radius)))
end
@inline function _artwork_interval_edge_uncertain(c,d,x,y)
    x.hi<min(c[1].lo,d[1].lo)||x.lo>max(c[1].hi,d[1].hi)||
        y.hi<min(c[2].lo,d[2].lo)||y.lo>max(c[2].hi,d[2].hi) || begin
        z=_artwork_interval_cross(_artwork_interval_sub(d,c),(x-c[1],y-c[2]))
        return z.lo<=0<=z.hi
    end
    false
end
@inline function _artwork_interval_axis_boundary(c,d,x,y)
    cx,cy,dx,dy=c[1],c[2],d[1],d[2]
    cx.lo==cx.hi==dx.lo==dx.hi==x.lo==x.hi&&
        y.lo>=min(cy.lo,dy.lo)&&y.hi<=max(cy.hi,dy.hi)&&return true
    cy.lo==cy.hi==dy.lo==dy.hi==y.lo==y.hi&&
        x.lo>=min(cx.lo,dx.lo)&&x.hi<=max(cx.hi,dx.hi)
end
function _artwork_certified_contains(s::_ArtworkPolygon,x,y)
    v=s.vertices;x0=x1=v[1,1];y0=y1=v[2,1]
    for i in axes(v,2)
        x0=min(x0,v[1,i]);x1=max(x1,v[1,i]);y0=min(y0,v[2,i]);y1=max(y1,v[2,i])
    end
    _artwork_interval_box(x,y,x0,x1,y0,y1)==0&&return Int8(0)
    for i in axes(v,2)
        j=mod1(i+1,size(v,2))
        c=(_ArtworkInterval(v[1,i]),_ArtworkInterval(v[2,i]));d=(_ArtworkInterval(v[1,j]),_ArtworkInterval(v[2,j]))
        _artwork_interval_axis_boundary(c,d,x,y)&&return Int8(1)
        _artwork_interval_edge_uncertain(c,d,x,y)&&return Int8(-1)
    end
    p=(_artwork_interval_mid(x),_artwork_interval_mid(y));all(isfinite,p)||return Int8(-1)
    inside=false
    for i in axes(v,2)
        j=mod1(i+1,size(v,2))
        crossing=_artwork_certified_ray_cross((v[1,i],v[2,i]),(v[1,j],v[2,j]),p)
        crossing==-1&&return Int8(-1)
        crossing==1&&(inside=!inside)
    end
    Int8(inside)
end
function _artwork_certified_contains(s::_ArtworkPolygonLine,x,y)
    v=s.aperture.vertices;aa,bb=_artwork_interval_point(s.start),_artwork_interval_point(s.stop)
    for i in axes(v,2)
        j=mod1(i+1,size(v,2));c=(_ArtworkInterval(v[1,i]),_ArtworkInterval(v[2,i]));d=(_ArtworkInterval(v[1,j]),_ArtworkInterval(v[2,j]))
        ca=(c[1]+aa[1],c[2]+aa[2]);da=(d[1]+aa[1],d[2]+aa[2])
        cb=(c[1]+bb[1],c[2]+bb[2]);db=(d[1]+bb[1],d[2]+bb[2])
        (_artwork_interval_edge_uncertain(ca,da,x,y)||_artwork_interval_edge_uncertain(cb,db,x,y)||
            _artwork_interval_edge_uncertain(ca,cb,x,y))&&return Int8(-1)
    end
    p=_artwork_interval_point((_artwork_interval_mid(x),_artwork_interval_mid(y)))
    a,b=_artwork_interval_sub(p,aa),_artwork_interval_sub(p,bb)
    first=_artwork_certified_contains(s.aperture,a...);last=_artwork_certified_contains(s.aperture,b...)
    (first==1||last==1)&&return Int8(1)
    uncertain=first==-1||last==-1
    for i in axes(v,2)
        j=mod1(i+1,size(v,2));c=(_ArtworkInterval(v[1,i]),_ArtworkInterval(v[2,i]));d=(_ArtworkInterval(v[1,j]),_ArtworkInterval(v[2,j]))
        state=_artwork_certified_intersect(a,b,c,d)
        state==1&&return state
        uncertain|=state==-1
    end
    uncertain ? Int8(-1) : Int8(0)
end
function _artwork_certified_intersect(a,b,c,d)
    max(min(a[1].lo,b[1].lo),min(c[1].lo,d[1].lo))>min(max(a[1].hi,b[1].hi),max(c[1].hi,d[1].hi))&&return Int8(0)
    max(min(a[2].lo,b[2].lo),min(c[2].lo,d[2].lo))>min(max(a[2].hi,b[2].hi),max(c[2].hi,d[2].hi))&&return Int8(0)
    ab,cd=_artwork_interval_sub(b,a),_artwork_interval_sub(d,c)
    u=_artwork_interval_sign(_artwork_interval_cross(ab,_artwork_interval_sub(c,a)))
    v=_artwork_interval_sign(_artwork_interval_cross(ab,_artwork_interval_sub(d,a)))
    x=_artwork_interval_sign(_artwork_interval_cross(cd,_artwork_interval_sub(a,c)))
    y=_artwork_interval_sign(_artwork_interval_cross(cd,_artwork_interval_sub(b,c)))
    (abs(u)==abs(v)==1&&u==v||abs(x)==abs(y)==1&&x==y)&&return Int8(0)
    all(sign->abs(sign)==1,(u,v,x,y)) ? Int8(1) : Int8(-1)
end
function _artwork_certified_contains(s::_ArtworkComposite,x,y)
    state=Int8(0)
    for (dark,part) in s.parts
        child=_artwork_certified_contains(part,x,y);value=Int8(dark)
        child==1&&(state=value)
        child==-1&&state!=value&&(state=Int8(-1))
    end
    state
end
@inline _artwork_certified_tuple(::Tuple{},x,y,state)=state
@inline function _artwork_certified_tuple(parts::Tuple,x,y,state)
    dark,part=first(parts);child=_artwork_certified_contains(part,x,y);value=Int8(dark)
    child==1&&(state=value)
    child==-1&&state!=value&&(state=Int8(-1))
    _artwork_certified_tuple(Base.tail(parts),x,y,state)
end
@inline _artwork_certified_contains(s::_ArtworkComposite{P},x,y) where {P<:Tuple}=
    _artwork_certified_tuple(s.parts,x,y,Int8(0))
_artwork_certified_contains(s::_ArtworkStrokeReference,x,y)=_artwork_certified_contains(s.geometry,x,y)
function _artwork_interval_transform(s,p)
    x,y=p[1]-s.origin[1],p[2]-s.origin[2];m=s.inverse
    (m[1,1]*x+m[1,2]*y,m[2,1]*x+m[2,2]*y)
end
function _artwork_certified_contains(s::_ArtworkTransform,x,y)
    a=_artwork_interval_transform(s,(x,y));_artwork_certified_contains(s.shape,a[1],a[2])
end
function _artwork_certified_contains(s::_ArtworkODBButterfly,x,y)
    i0=_ArtworkInterval(0.)
    p=_artwork_interval_intersection(_artwork_interval_le(x,i0),_artwork_interval_le(i0,y))
    q=_artwork_interval_intersection(_artwork_interval_le(i0,x),_artwork_interval_le(y,i0))
    _artwork_interval_intersection(_artwork_interval_union(p,q),_artwork_certified_contains(s.outside,x,y))
end
function _artwork_certified_contains(s::_ArtworkCornerRectangle,x,y)
    w,h=_artwork_interval_half(s.width),_artwork_interval_half(s.height)
    r=_ArtworkInterval(s.radius)
    state=_artwork_interval_intersection(_artwork_interval_le(_artwork_interval_abs(x),w),
        _artwork_interval_le(_artwork_interval_abs(y),h))
    (state==0||s.radius==0)&&return state
    for (i,(positive_x,positive_y)) in enumerate(((true,true),(false,true),(false,false),(true,false)))
        s.corners[i]||continue
        u,v=positive_x ? x : -x,positive_y ? y : -y
        corner=if s.chamfer
            _artwork_interval_le(u+v,w+h-r)
        else
            slabs=_artwork_interval_union(_artwork_interval_le(u,w-r),_artwork_interval_le(v,h-r))
            _artwork_interval_union(slabs,_artwork_certified_circle((w-r,h-r),s.radius,u,v))
        end
        state=_artwork_interval_intersection(state,corner)
        state==0&&return state
    end
    state
end
@inline function _artwork_interval_half(x::Float64)
    h=x/2
    # Binary scaling is exact except when its result enters the subnormal
    # range. Keep that rare rounded result enclosed without BigInt work.
    h+h==x ? _ArtworkInterval(h) : _ArtworkInterval(prevfloat(h),nextfloat(h))
end
function _artwork_certified_arc_angle(s,x,y)
    s.start==s.stop&&return Int8(1)
    c=_artwork_interval_point(s.center)
    _artwork_certified_arc_direction(s,(x-c[1],y-c[2]))
end
function _artwork_certified_arc_direction(s,z)
    s.start==s.stop&&return Int8(1)
    c=_artwork_interval_point(s.center)
    u=_artwork_interval_sub(_artwork_interval_point(s.start),c)
    v=_artwork_interval_sub(_artwork_interval_point(s.stop),c)
    s.clockwise&&((u,v)=(v,u))
    uv,uz,zv=_artwork_interval_cross(u,v),_artwork_interval_cross(u,z),_artwork_interval_cross(z,v)
    left=_artwork_interval_le(_ArtworkInterval(0.),uz);right=_artwork_interval_le(_ArtworkInterval(0.),zv)
    uv.lo>0&&return _artwork_interval_intersection(left,right)
    uv.hi<0&&return _artwork_interval_union(left,right)
    # Diametric and equal-ray cases are infrequent and require an exact sign.
    Int8(-1)
end
function _artwork_certified_contains(s::_ArtworkRoundArc,x,y)
    a=_artwork_certified_circle(s.start,s.radius,x,y)
    b=_artwork_certified_circle(s.stop,s.radius,x,y)
    caps=_artwork_interval_union(a,b);caps==1&&return caps
    r=hypot(s.start[1]-s.center[1],s.start[2]-s.center[2]);c=_artwork_interval_point(s.center)
    dist=_artwork_interval_square(x-c[1])+_artwork_interval_square(y-c[2])
    outer=_artwork_interval_square(_ArtworkInterval(r)+_ArtworkInterval(s.radius))
    diff=_ArtworkInterval(r)-_ArtworkInterval(s.radius)
    inner=_artwork_interval_square(_ArtworkInterval(max(0.,diff.lo),max(0.,diff.hi)))
    body=_artwork_interval_intersection(_artwork_interval_le(dist,outer),_artwork_interval_le(inner,dist))
    body=_artwork_interval_intersection(body,_artwork_certified_arc_angle(s,x,y))
    _artwork_interval_union(caps,body)
end
function _artwork_certified_contains(s::_ArtworkRegion,x,y)
    for part in s.segments
        (x.lo==x.hi==part.start[1]&&y.lo==y.hi==part.start[2]||
            x.lo==x.hi==part.stop[1]&&y.lo==y.hi==part.stop[2])&&return Int8(1)
        a,b=_artwork_interval_point(part.start),_artwork_interval_point(part.stop)
        # A real axis-aligned contour segment is a closed boundary. An arc's
        # straight chord is only a winding construction and cannot use this
        # certificate.
        part isa _ArtworkRoundLine&&_artwork_interval_axis_boundary(a,b,x,y)&&return Int8(1)
        part.start!=part.stop&&_artwork_interval_edge_uncertain(a,b,x,y)&&return Int8(-1)
        if part isa _ArtworkRoundArc
            r=hypot(part.start[1]-part.center[1],part.start[2]-part.center[2])
            _artwork_certified_circle(part.center,r,x,y)==-1&&return Int8(-1)
        end
    end
    p=(_artwork_interval_mid(x),_artwork_interval_mid(y));all(isfinite,p)||return Int8(-1)
    inside=false
    for part in s.segments
        crossing=_artwork_certified_ray_cross(part.start,part.stop,p)
        crossing==-1&&return Int8(-1)
        crossing==1&&(inside=!inside)
        part isa _ArtworkRoundArc||continue
        r=hypot(part.start[1]-part.center[1],part.start[2]-part.center[2])
        disk=_artwork_certified_circle(part.center,r,_ArtworkInterval(p[1]),_ArtworkInterval(p[2]))
        disk==-1&&return Int8(-1)
        disk==0&&continue
        if part.start==part.stop
            inside=!inside
        else
            a,b=_artwork_interval_point(part.start),_artwork_interval_point(part.stop)
            side=_artwork_interval_sign(_artwork_interval_cross(_artwork_interval_sub(b,a),_artwork_interval_sub(_artwork_interval_point(p),a)))
            abs(side)==1||return Int8(-1)
            (part.clockwise ? side>0 : side<0)&&(inside=!inside)
        end
    end
    Int8(inside)
end
function _artwork_canonical_contains(s,x::Float64,y::Float64)
    isfinite(x)&&isfinite(y)||return false
    state=_artwork_certified_contains(s,_ArtworkInterval(x),_ArtworkInterval(y))
    state==-1 ? _artwork_exact_point(s,x,y) : state==1
end

struct _ArtworkRootInterval
    lo::Float64
    hi::Float64
    found::Bool
end
@inline function _artwork_root_candidate(i,cursor,current)
    any(isnan,(i.lo,i.hi))&&(i=_ArtworkInterval(0.,1.))
    lo,hi=max(0.,i.lo),min(1.,i.hi)
    (hi<=cursor||lo>hi)&&return current
    lo=max(cursor,lo)
    !current.found||lo<current.lo ? _ArtworkRootInterval(lo,hi,true) :
        lo==current.lo ? _ArtworkRootInterval(lo,max(hi,current.hi),true) : current
end
function _artwork_interval_roots(A,B,C,cursor,current)
    all(isfinite,(A.lo,A.hi,B.lo,B.hi,C.lo,C.hi))||
        return _artwork_root_candidate(_ArtworkInterval(0.,1.),cursor,current)
    A.lo>0||return _artwork_root_candidate(_ArtworkInterval(0.,1.),cursor,current)
    D=_artwork_interval_square(B)-4.0*A*C
    all(isfinite,(D.lo,D.hi))||return _artwork_root_candidate(_ArtworkInterval(0.,1.),cursor,current)
    D.hi<0&&return current
    root=_artwork_interval_sqrt(D);den=2.0*A
    first=(-B-root)/den;second=(-B+root)/den
    _artwork_root_candidate(second,cursor,_artwork_root_candidate(first,cursor,current))
end
function _artwork_interval_circle_roots(c,r,a,v,cursor,current)
    u=_artwork_interval_sub(a,c)
    _artwork_interval_roots(_artwork_interval_dot(v,v),2.0*_artwork_interval_dot(u,v),
        _artwork_interval_dot(u,u)-_artwork_interval_square(r),cursor,current)
end
function _artwork_interval_edge_roots(c,d,a,v,cursor,current)
    e=_artwork_interval_sub(d,c);h=_artwork_interval_sub(c,a);den=_artwork_interval_cross(v,e)
    if den.lo<=0<=den.hi
        # A broad enclosure covers collinear and nearly parallel cases. The
        # ordered aperture may still certify this whole interval as clear.
        return _artwork_root_candidate(_ArtworkInterval(0.,1.),cursor,current)
    end
    _artwork_root_candidate(_artwork_interval_cross(h,e)/den,cursor,current)
end
_artwork_interval_next(::_ArtworkEmpty,a,v,cursor,current)=current
_artwork_interval_next(::_ArtworkODBNull,a,v,cursor,current)=current
_artwork_interval_next(s::_ArtworkCircle,a,v,cursor,current)=
    _artwork_interval_circle_roots(_artwork_interval_point(s.center),_ArtworkInterval(s.radius),a,v,cursor,current)
_artwork_interval_next(s::_ArtworkODBDrill,a,v,cursor,current)=_artwork_interval_next(s.circle,a,v,cursor,current)
_artwork_interval_next(s::_ArtworkStrokeReference,a,v,cursor,current)=_artwork_interval_next(s.geometry,a,v,cursor,current)
function _artwork_interval_next(s::_ArtworkPolygon,a,v,cursor,current)
    p=s.vertices
    for i in axes(p,2)
        j=mod1(i+1,size(p,2))
        current=_artwork_interval_edge_roots((_ArtworkInterval(p[1,i]),_ArtworkInterval(p[2,i])),
            (_ArtworkInterval(p[1,j]),_ArtworkInterval(p[2,j])),a,v,cursor,current)
    end
    current
end
function _artwork_interval_next(s::_ArtworkComposite,a,v,cursor,current)
    for (_,part) in s.parts;current=_artwork_interval_next(part,a,v,cursor,current);end
    current
end
_artwork_canonical_leaf(::_ArtworkShape)=false
_artwork_canonical_leaf(::Union{_ArtworkEmpty,_ArtworkCircle,_ArtworkPolygon,_ArtworkPolygonLine,_ArtworkRoundLine,
    _ArtworkRoundArc,_ArtworkRegion,_ArtworkCornerRectangle,_ArtworkODBNull,_ArtworkODBDrill})=true
_artwork_canonical_leaf(s::_ArtworkODBButterfly)=_artwork_canonical_leaf(s.outside)
_artwork_canonical_leaf(s::_ArtworkStrokeReference)=_artwork_canonical_leaf(s.geometry)
_artwork_canonical_leaf(s::_ArtworkTransform)=_artwork_canonical_leaf(s.shape)
function _artwork_canonical_leaf(s::_ArtworkComposite)
    all(part->_artwork_canonical_leaf(part[2]),s.parts)
end
_artwork_contains(s::_ArtworkPolygon,x::Float64,y::Float64)=_artwork_canonical_contains(s,x,y)
_artwork_contains(s::_ArtworkPolygonLine,x::Float64,y::Float64)=_artwork_canonical_contains(s,x,y)
_artwork_contains(s::_ArtworkCornerRectangle,x::Float64,y::Float64)=_artwork_canonical_contains(s,x,y)
@inline _artwork_interval_next_tuple(::Tuple{},a,v,cursor,current)=current
@inline function _artwork_interval_next_tuple(parts::Tuple,a,v,cursor,current)
    _artwork_interval_next_tuple(Base.tail(parts),a,v,cursor,
        _artwork_interval_next(first(parts)[2],a,v,cursor,current))
end
@inline _artwork_interval_next(s::_ArtworkComposite{P},a,v,cursor,current) where {P<:Tuple}=
    _artwork_interval_next_tuple(s.parts,a,v,cursor,current)
function _artwork_interval_next(s::_ArtworkTransform,a,v,cursor,current)
    aa=_artwork_interval_transform(s,a);m=s.inverse
    vv=(m[1,1]*v[1]+m[1,2]*v[2],m[2,1]*v[1]+m[2,2]*v[2])
    _artwork_interval_next(s.shape,aa,vv,cursor,current)
end
function _artwork_interval_next(s::_ArtworkRoundLine,a,v,cursor,current)
    aa,bb=_artwork_interval_point(s.start),_artwork_interval_point(s.stop);r=_ArtworkInterval(s.radius)
    current=_artwork_interval_circle_roots(aa,r,a,v,cursor,current)
    current=_artwork_interval_circle_roots(bb,r,a,v,cursor,current)
    e=_artwork_interval_sub(bb,aa);u=_artwork_interval_sub(a,aa)
    x,y=_artwork_interval_cross(v,e),_artwork_interval_cross(u,e)
    _artwork_interval_roots(_artwork_interval_square(x),2.0*x*y,
        _artwork_interval_square(y)-_artwork_interval_square(r)*_artwork_interval_dot(e,e),cursor,current)
end
function _artwork_interval_next(s::_ArtworkRoundArc,a,v,cursor,current)
    r=_ArtworkInterval(hypot(s.start[1]-s.center[1],s.start[2]-s.center[2]));w=_ArtworkInterval(s.radius);c=_artwork_interval_point(s.center)
    current=_artwork_interval_circle_roots(c,r+w,a,v,cursor,current)
    inner=r-w;inner=_ArtworkInterval(max(0.,inner.lo),max(0.,inner.hi))
    current=_artwork_interval_circle_roots(c,inner,a,v,cursor,current)
    current=_artwork_interval_circle_roots(_artwork_interval_point(s.start),w,a,v,cursor,current)
    _artwork_interval_circle_roots(_artwork_interval_point(s.stop),w,a,v,cursor,current)
end
function _artwork_interval_next(s::_ArtworkRegion,a,v,cursor,current)
    for part in s.segments
        current=_artwork_interval_edge_roots(_artwork_interval_point(part.start),_artwork_interval_point(part.stop),a,v,cursor,current)
        if part isa _ArtworkRoundArc
            r=hypot(part.start[1]-part.center[1],part.start[2]-part.center[2])
            current=_artwork_interval_circle_roots(_artwork_interval_point(part.center),_ArtworkInterval(r),a,v,cursor,current)
        end
    end
    current
end
function _artwork_interval_next(s::_ArtworkPolygonLine,a,v,cursor,current)
    p=s.aperture.vertices;aa,bb=_artwork_interval_point(s.start),_artwork_interval_point(s.stop)
    for i in axes(p,2)
        j=mod1(i+1,size(p,2));c=(_ArtworkInterval(p[1,i]),_ArtworkInterval(p[2,i]));d=(_ArtworkInterval(p[1,j]),_ArtworkInterval(p[2,j]))
        ca=(c[1]+aa[1],c[2]+aa[2]);da=(d[1]+aa[1],d[2]+aa[2])
        cb=(c[1]+bb[1],c[2]+bb[2]);db=(d[1]+bb[1],d[2]+bb[2])
        current=_artwork_interval_edge_roots(ca,da,a,v,cursor,current)
        current=_artwork_interval_edge_roots(cb,db,a,v,cursor,current)
        current=_artwork_interval_edge_roots(ca,cb,a,v,cursor,current)
    end
    current
end
function _artwork_interval_next(s::_ArtworkODBButterfly,a,v,cursor,current)
    current=_artwork_interval_next(s.outside,a,v,cursor,current)
    for d in 1:2
        v[d].lo<=0<=v[d].hi&&continue
        current=_artwork_root_candidate(-a[d]/v[d],cursor,current)
    end
    current
end
function _artwork_interval_next(s::_ArtworkCornerRectangle,a,v,cursor,current)
    w,h,r=_ArtworkInterval(s.width)/2.,_ArtworkInterval(s.height)/2.,_ArtworkInterval(s.radius)
    for (c,d) in (((-w,-h),(w,-h)),((w,-h),(w,h)),((w,h),(-w,h)),((-w,h),(-w,-h)))
        current=_artwork_interval_edge_roots(c,d,a,v,cursor,current)
    end
    s.radius>0||return current
    for (i,(sx,sy)) in enumerate(((1.,1.),(-1.,1.),(-1.,-1.),(1.,-1.)))
        s.corners[i]||continue
        current=s.chamfer ? _artwork_interval_edge_roots((sx*(w-r),sy*h),(sx*w,sy*(h-r)),a,v,cursor,current) :
            _artwork_interval_circle_roots((sx*(w-r),sy*(h-r)),r,a,v,cursor,current)
    end
    current
end
function _artwork_certified_stroke(s,x::Float64,y::Float64)
    isfinite(x)&&isfinite(y)||return false
    a=(_ArtworkInterval(x)-s.start[1],_ArtworkInterval(y)-s.start[2])
    v=(_ArtworkInterval(s.start[1])-_ArtworkInterval(s.stop[1]),_ArtworkInterval(s.start[2])-_ArtworkInterval(s.stop[2]))
    uncertain=false;cursor=0.;iterations=0
    # Every certified hit is an actual parameter witness. Boundary enclosures
    # partition every potentially changing interval before absence is accepted.
    for t in (0.,1.)
        p=_artwork_interval_xy(a,v,_ArtworkInterval(t));state=_artwork_certified_contains(s.aperture,p...)
        state==1&&return true
        uncertain|=state==-1
    end
    while cursor<1
        iterations+=1
        iterations<=s.boundary_count+1||throw(ArgumentError("certified artwork stroke work bound exceeded"))
        event=_artwork_interval_next(s.aperture,a,v,cursor,_ArtworkRootInterval(1.,1.,false))
        right=event.found ? event.lo : 1.
        if cursor<right
            t=cursor+(right-cursor)/2
            p=_artwork_interval_xy(a,v,_ArtworkInterval(t));state=_artwork_certified_contains(s.aperture,p...)
            state==1&&return true
            uncertain|=state==-1
        end
        event.found||break
        p=_artwork_interval_xy(a,v,_ArtworkInterval(event.lo,event.hi));state=_artwork_certified_contains(s.aperture,p...)
        state==1&&return true
        uncertain|=state==-1
        event.hi>cursor||throw(ArgumentError("nonprogressing artwork boundary enclosure"))
        cursor=event.hi
    end
    if uncertain
        linear=_artwork_linear_ordered_probe(s,x,y,a,v)
        linear>=0&&return linear==1
        return _artwork_exact_stroke(s,x,y)
    end
    false
end
