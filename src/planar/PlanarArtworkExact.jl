# Exact dyadic/quadratic boundary witnesses for analytic translated apertures.
# Arc centerline radii follow the existing stored-Float64 hypot convention.
# Open intervals use bounded rational separators, never geometric tolerances.
const _ArtworkDyadic=Rational{BigInt}
_artwork_exact_q(x::Real)=_ArtworkDyadic(x)

# Exact real quadratic field for one boundary witness. Coordinates and all
# primitive polynomial predicates share the same D; cross-field operations
# occur only in event ordering and use a certified sign reduction.
struct _ArtworkQuadratic <: Real
    a::_ArtworkDyadic
    b::_ArtworkDyadic
    d::_ArtworkDyadic
end
_ArtworkQuadratic(x::Union{Integer,Rational})=_ArtworkQuadratic(_ArtworkDyadic(x),_artwork_exact_q(0),_artwork_exact_q(0))
function _artwork_exact_asign(a::_ArtworkDyadic,b::_ArtworkDyadic,d::_ArtworkDyadic)
    iszero(b) && return sign(a)
    iszero(a) && return sign(b)
    sa,sb=sign(a),sign(b)
    sa==sb && return sa
    z=a*a-b*b*d
    iszero(z) ? 0 : z>0 ? sa : sb
end
Base.sign(x::_ArtworkQuadratic)=_artwork_exact_asign(x.a,x.b,x.d)
Base.iszero(x::_ArtworkQuadratic)=iszero(sign(x))
function _artwork_exact_adiffsign(x::_ArtworkQuadratic,y::_ArtworkQuadratic)
    a=x.a-y.a
    x.d==y.d && return _artwork_exact_asign(a,x.b-y.b,x.d)
    iszero(y.b) && return _artwork_exact_asign(a,x.b,x.d)
    iszero(x.b) && return _artwork_exact_asign(a,-y.b,y.d)
    su=_artwork_exact_asign(a,x.b,x.d);sv=sign(-y.b)
    iszero(su) && return sv
    su==sv && return su
    # U=a+x.b*sqrt(x.d), V=-y.b*sqrt(y.d), with opposite signs.
    sq=_artwork_exact_asign(a*a+x.b*x.b*x.d-y.b*y.b*y.d,2a*x.b,x.d)
    iszero(sq) ? 0 : sq>0 ? su : sv
end
Base.:(==)(x::_ArtworkQuadratic,y::_ArtworkQuadratic)=_artwork_exact_adiffsign(x,y)==0
Base.isless(x::_ArtworkQuadratic,y::_ArtworkQuadratic)=_artwork_exact_adiffsign(x,y)<0
Base.:<(x::_ArtworkQuadratic,y::_ArtworkQuadratic)=_artwork_exact_adiffsign(x,y)<0
Base.:<=(x::_ArtworkQuadratic,y::_ArtworkQuadratic)=_artwork_exact_adiffsign(x,y)<=0
function _artwork_exact_afield(x::_ArtworkQuadratic,y::_ArtworkQuadratic)
    iszero(x.b) && return y.d
    iszero(y.b) && return x.d
    x.d==y.d || throw(ArgumentError("arithmetic across distinct quadratic fields"))
    x.d
end
Base.:+(x::_ArtworkQuadratic,y::_ArtworkQuadratic)=_ArtworkQuadratic(x.a+y.a,x.b+y.b,_artwork_exact_afield(x,y))
Base.:-(x::_ArtworkQuadratic)=_ArtworkQuadratic(-x.a,-x.b,x.d)
Base.:-(x::_ArtworkQuadratic,y::_ArtworkQuadratic)=x+(-y)
function Base.:*(x::_ArtworkQuadratic,y::_ArtworkQuadratic)
    d=_artwork_exact_afield(x,y);_ArtworkQuadratic(x.a*y.a+x.b*y.b*d,x.a*y.b+x.b*y.a,d)
end
function Base.:/(x::_ArtworkQuadratic,y::_ArtworkQuadratic)
    d=_artwork_exact_afield(x,y);den=y.a*y.a-y.b*y.b*d
    iszero(den) && throw(DivideError())
    _ArtworkQuadratic((x.a*y.a-x.b*y.b*d)/den,(x.b*y.a-x.a*y.b)/den,d)
end
Base.abs(x::_ArtworkQuadratic)=sign(x)<0 ? -x : x
function Base.:^(x::_ArtworkQuadratic,n::Integer)
    n<0 && return _ArtworkQuadratic(1)/(x^(-n))
    result=_ArtworkQuadratic(1)
    for _ in 1:n;result=result*x;end
    result
end
for op in (:+,:-,:*,:/,:<,:<=,:(==))
    @eval Base.$op(x::_ArtworkQuadratic,y::Union{Integer,Rational})=Base.$op(x,_ArtworkQuadratic(y))
    @eval Base.$op(x::Union{Integer,Rational},y::_ArtworkQuadratic)=Base.$op(_ArtworkQuadratic(x),y)
end
function _artwork_exact_rational_separator(left::_ArtworkQuadratic,right::_ArtworkQuadratic,bound)
    left<right || throw(ArgumentError("strict event ordering required"))
    lo,hi=_artwork_exact_q(0),_artwork_exact_q(1)
    for _ in 1:bound
        mid=(lo+hi)/2
        left<mid<right && return mid
        mid<=left ? (lo=mid) : (hi=mid)
    end
    throw(ArgumentError("exact event rational separator work bound exceeded"))
end

_artwork_exact_qpoint(p)=(_artwork_exact_q(p[1]),_artwork_exact_q(p[2]))
_artwork_exact_qsub(a,b)=(a[1]-b[1],a[2]-b[2])
_artwork_exact_qadd(a,b)=(a[1]+b[1],a[2]+b[2])
_artwork_exact_qscale(a,t)=(a[1]*t,a[2]*t)
_artwork_exact_qcross(a,b)=a[1]*b[2]-a[2]*b[1]
_artwork_exact_qdot(a,b)=a[1]*b[1]+a[2]*b[2]
function _artwork_exact_qsegment(a,b,p)
    min(a[1],b[1])<=p[1]<=max(a[1],b[1]) &&
    min(a[2],b[2])<=p[2]<=max(a[2],b[2]) && iszero(_artwork_exact_qcross(_artwork_exact_qsub(b,a),_artwork_exact_qsub(p,a)))
end
function _artwork_exact_qintersect(a,b,c,d)
    max(min(a[1],b[1]),min(c[1],d[1]))<=min(max(a[1],b[1]),max(c[1],d[1])) &&
    max(min(a[2],b[2]),min(c[2],d[2]))<=min(max(a[2],b[2]),max(c[2],d[2])) || return false
    ab=_artwork_exact_qsub(b,a);cd=_artwork_exact_qsub(d,c)
    x,y=_artwork_exact_qcross(ab,_artwork_exact_qsub(c,a)),_artwork_exact_qcross(ab,_artwork_exact_qsub(d,a))
    u,v=_artwork_exact_qcross(cd,_artwork_exact_qsub(a,c)),_artwork_exact_qcross(cd,_artwork_exact_qsub(b,c))
    x*y<=0 && u*v<=0
end
_artwork_exact_qcontains(::_ArtworkEmpty,p)=false
function _artwork_exact_qcontains(s::_ArtworkCircle,p)
    c=_artwork_exact_qpoint(s.center);r=_artwork_exact_q(s.radius)
    (p[1]<c[1]-r||p[1]>c[1]+r||p[2]<c[2]-r||p[2]>c[2]+r)&&return false
    z=_artwork_exact_qsub(p,c)
    _artwork_exact_qdot(z,z)<=r*r
end
function _artwork_exact_qcontains(s::_ArtworkPolygon,p)
    inside=false;v=s.vertices
    for i in axes(v,2)
        j=mod1(i+1,size(v,2));a=_artwork_exact_qpoint(view(v,:,i));b=_artwork_exact_qpoint(view(v,:,j))
        _artwork_exact_qsegment(a,b,p) && return true
        (a[2]>p[2])!=(b[2]>p[2]) && p[1]<(b[1]-a[1])*(p[2]-a[2])/(b[2]-a[2])+a[1] && (inside=!inside)
    end
    inside
end
function _artwork_exact_qcontains(s::_ArtworkRoundLine,p)
    a=_artwork_exact_qpoint(s.start);b=_artwork_exact_qpoint(s.stop);e=_artwork_exact_qsub(b,a);u=_artwork_exact_qsub(p,a);n=_artwork_exact_qdot(e,e);r2=_artwork_exact_q(s.radius)^2
    iszero(n) && return _artwork_exact_qdot(u,u)<=r2
    t=_artwork_exact_qdot(u,e)
    t<=0 && return _artwork_exact_qdot(u,u)<=r2
    t>=n && return _artwork_exact_qdot(_artwork_exact_qsub(p,b),_artwork_exact_qsub(p,b))<=r2
    _artwork_exact_qcross(u,e)^2<=r2*n
end
function _artwork_exact_qcontains(s::_ArtworkPolygonLine,p)
    a=_artwork_exact_qsub(p,_artwork_exact_qpoint(s.start));b=_artwork_exact_qsub(p,_artwork_exact_qpoint(s.stop))
    (_artwork_exact_qcontains(s.aperture,a)||_artwork_exact_qcontains(s.aperture,b)) && return true
    v=s.aperture.vertices
    for i in axes(v,2)
        j=mod1(i+1,size(v,2))
        _artwork_exact_qintersect(a,b,_artwork_exact_qpoint(view(v,:,i)),_artwork_exact_qpoint(view(v,:,j))) && return true
    end
    false
end
function _artwork_exact_qcontains(s::_ArtworkComposite,p)
    inside=false
    for (dark,part) in s.parts
        _artwork_exact_qcontains(part,p) && (inside=dark)
    end
    inside
end
_artwork_exact_qcontains(s::_ArtworkStrokeReference,p)=_artwork_exact_qcontains(s.geometry,p)
function _artwork_exact_qtransform(s,p)
    z=_artwork_exact_qsub(p,_artwork_exact_qpoint(s.origin));M=s.inverse
    (_artwork_exact_q(M[1,1])*z[1]+_artwork_exact_q(M[1,2])*z[2],_artwork_exact_q(M[2,1])*z[1]+_artwork_exact_q(M[2,2])*z[2])
end
_artwork_exact_qcontains(s::_ArtworkTransform,p)=_artwork_exact_qcontains(s.shape,_artwork_exact_qtransform(s,p))


function _artwork_exact_qarc_radius(s)
    # Explicit canonical convention of current Float64 primitive helpers.
    _artwork_exact_q(hypot(s.start[1]-s.center[1],s.start[2]-s.center[2]))
end
function _artwork_exact_qarc_angle(s,p)
    s.start==s.stop && return true
    c=_artwork_exact_qpoint(s.center);u=_artwork_exact_qsub(_artwork_exact_qpoint(s.start),c);v=_artwork_exact_qsub(_artwork_exact_qpoint(s.stop),c);z=_artwork_exact_qsub(p,c)
    s.clockwise && ((u,v)=(v,u))
    uv=_artwork_exact_qcross(u,v);uz=_artwork_exact_qcross(u,z);zv=_artwork_exact_qcross(z,v)
    if uv>0 || (iszero(uv)&&_artwork_exact_qdot(u,v)<0)
        uz>=0 && zv>=0
    elseif uv<0
        uz>=0 || zv>=0
    else
        # Equal endpoint ray with distinct radii has zero angular sweep.
        iszero(_artwork_exact_qcross(u,z)) && _artwork_exact_qdot(u,z)>=0
    end
end
function _artwork_exact_qcontains(s::_ArtworkRoundArc,p)
    c=_artwork_exact_qpoint(s.center);r=_artwork_exact_qarc_radius(s);w=_artwork_exact_q(s.radius);z=_artwork_exact_qsub(p,c);dist=_artwork_exact_qdot(z,z)
    for endpoint in (s.start,s.stop)
        u=_artwork_exact_qsub(p,_artwork_exact_qpoint(endpoint));_artwork_exact_qdot(u,u)<=w*w && return true
    end
    iszero(dist) && return w>=r
    dist<=(r+w)^2 && dist>=max(_artwork_exact_q(0),r-w)^2 && _artwork_exact_qarc_angle(s,p)
end
function _artwork_exact_qlexsign(values)
    for v in values
        !iszero(v) && return sign(v)
    end
    0
end
function _artwork_exact_qdisk_perturbed(p,c,r)
    z=_artwork_exact_qsub(p,c);u=r*r-_artwork_exact_qdot(z,z)
    # Symbolic point p+(epsilon,epsilon^2) selects a consistent side of
    # artificial chord/circle ties without changing physical boundaries.
    _artwork_exact_qlexsign((u,-2z[1],-_artwork_exact_q(1)-2z[2],-_artwork_exact_q(1)))>0
end
function _artwork_exact_qside_perturbed(a,b,p)
    e=_artwork_exact_qsub(b,a)
    _artwork_exact_qlexsign((_artwork_exact_qcross(e,_artwork_exact_qsub(p,a)),-e[2],e[1]))
end
function _artwork_exact_qchord_crosses(a,b,p)
    (a[2]>p[2])!=(b[2]>p[2]) && p[1]<(b[1]-a[1])*(p[2]-a[2])/(b[2]-a[2])+a[1]
end
function _artwork_exact_qcontains(s::_ArtworkRegion,p)
    # A region is its chord polygon XOR the circular segment between each
    # arc and its chord. True physical boundaries are accepted first.
    inside=false
    for segment in s.segments
        a,b=_artwork_exact_qpoint(segment.start),_artwork_exact_qpoint(segment.stop)
        if segment isa _ArtworkRoundLine
            _artwork_exact_qsegment(a,b,p) && return true
        else
            p==a || p==b || begin
                c=_artwork_exact_qpoint(segment.center);z=_artwork_exact_qsub(p,c);r=_artwork_exact_qarc_radius(segment)
                _artwork_exact_qdot(z,z)==r*r && _artwork_exact_qarc_angle(segment,p) && return true
            end
            (p==a || p==b) && return true
        end
        _artwork_exact_qchord_crosses(a,b,p) && (inside=!inside)
        if segment isa _ArtworkRoundArc
            c=_artwork_exact_qpoint(segment.center);r=_artwork_exact_qarc_radius(segment)
            if segment.start==segment.stop
                _artwork_exact_qdisk_perturbed(p,c,r) && (inside=!inside)
            else
                side=_artwork_exact_qside_perturbed(a,b,p)
                desired=segment.clockwise ? side>0 : side<0
                _artwork_exact_qdisk_perturbed(p,c,r) && desired && (inside=!inside)
            end
        end
    end
    inside
end

mutable struct _ArtworkExactEventWork
    events::Vector{_ArtworkQuadratic}
    limit::Int
    irrational::Int
end
function _artwork_exact_qpush!(w,t)
    0<=t<=1 || return
    length(w.events)<w.limit || throw(ArgumentError("exact artwork event bound exceeded"))
    t isa _ArtworkQuadratic&&!iszero(t.b)&&(w.irrational+=1)
    push!(w.events,t isa _ArtworkQuadratic ? t : _ArtworkQuadratic(t))
end
function _artwork_exact_qroots!(w,A,B,C)
    if iszero(A)
        !iszero(B) && _artwork_exact_qpush!(w,-C/B)
        return
    end
    D=B*B-4A*C
    D<0 && return
    n,d=numerator(D),denominator(D);sn,sd=isqrt(n),isqrt(d)
    if sn*sn==n && sd*sd==d
        root=sn//sd
        _artwork_exact_qpush!(w,(-B-root)/(2A))
        !iszero(root) && _artwork_exact_qpush!(w,(-B+root)/(2A))
    else
        _artwork_exact_qpush!(w,_ArtworkQuadratic(-B/(2A),-_artwork_exact_q(1)/(2A),D))
        _artwork_exact_qpush!(w,_ArtworkQuadratic(-B/(2A),_artwork_exact_q(1)/(2A),D))
    end
end
function _artwork_exact_qedge_events!(w,c,d,a,b)
    v=_artwork_exact_qsub(b,a);e=_artwork_exact_qsub(d,c);h=_artwork_exact_qsub(c,a);A=_artwork_exact_qcross(v,e)
    if !iszero(A)
        t,s=_artwork_exact_qcross(h,e)/A,_artwork_exact_qcross(h,v)/A
        0<=s<=1 && _artwork_exact_qpush!(w,t)
    elseif iszero(_artwork_exact_qcross(h,v))
        k=abs(v[1])>=abs(v[2]) ? 1 : 2
        iszero(v[k]) && return
        _artwork_exact_qpush!(w,(c[k]-a[k])/v[k]);_artwork_exact_qpush!(w,(d[k]-a[k])/v[k])
    end
end
_artwork_exact_qevents!(w,::_ArtworkEmpty,a,b)=nothing
function _artwork_exact_qevents!(w,s::_ArtworkCircle,a,b)
    u=_artwork_exact_qsub(a,_artwork_exact_qpoint(s.center));v=_artwork_exact_qsub(b,a)
    _artwork_exact_qroots!(w,_artwork_exact_qdot(v,v),2*_artwork_exact_qdot(u,v),_artwork_exact_qdot(u,u)-_artwork_exact_q(s.radius)^2)
end
function _artwork_exact_qevents!(w,s::_ArtworkPolygon,a,b)
    v=s.vertices
    for i in axes(v,2)
        j=mod1(i+1,size(v,2));_artwork_exact_qedge_events!(w,_artwork_exact_qpoint(view(v,:,i)),_artwork_exact_qpoint(view(v,:,j)),a,b)
    end
end
function _artwork_exact_qevents!(w,s::_ArtworkRoundLine,a,b)
    for c in (s.start,s.stop)
        u=_artwork_exact_qsub(a,_artwork_exact_qpoint(c));v=_artwork_exact_qsub(b,a)
        _artwork_exact_qroots!(w,_artwork_exact_qdot(v,v),2*_artwork_exact_qdot(u,v),_artwork_exact_qdot(u,u)-_artwork_exact_q(s.radius)^2)
    end
    c=_artwork_exact_qpoint(s.start);e=_artwork_exact_qsub(_artwork_exact_qpoint(s.stop),c);v=_artwork_exact_qsub(b,a);u=_artwork_exact_qsub(a,c)
    A,B=_artwork_exact_qcross(v,e),_artwork_exact_qcross(u,e)
    _artwork_exact_qroots!(w,A*A,2A*B,B*B-_artwork_exact_q(s.radius)^2*_artwork_exact_qdot(e,e))
end
function _artwork_exact_qevents!(w,s::_ArtworkPolygonLine,a,b)
    v=s.aperture.vertices
    for i in axes(v,2)
        j=mod1(i+1,size(v,2));c=_artwork_exact_qpoint(view(v,:,i));d=_artwork_exact_qpoint(view(v,:,j))
        _artwork_exact_qedge_events!(w,_artwork_exact_qadd(c,_artwork_exact_qpoint(s.start)),_artwork_exact_qadd(d,_artwork_exact_qpoint(s.start)),a,b)
        _artwork_exact_qedge_events!(w,_artwork_exact_qadd(c,_artwork_exact_qpoint(s.stop)),_artwork_exact_qadd(d,_artwork_exact_qpoint(s.stop)),a,b)
        _artwork_exact_qedge_events!(w,_artwork_exact_qadd(c,_artwork_exact_qpoint(s.start)),_artwork_exact_qadd(c,_artwork_exact_qpoint(s.stop)),a,b)
    end
end
function _artwork_exact_qevents!(w,s::_ArtworkComposite,a,b)
    for (_,part) in s.parts;_artwork_exact_qevents!(w,part,a,b);end
end
_artwork_exact_qevents!(w,s::_ArtworkStrokeReference,a,b)=_artwork_exact_qevents!(w,s.geometry,a,b)
_artwork_exact_qevents!(w,s::_ArtworkTransform,a,b)=_artwork_exact_qevents!(w,s.shape,_artwork_exact_qtransform(s,a),_artwork_exact_qtransform(s,b))

function _artwork_exact_qcircle_events!(w,c,r,a,b)
    u=_artwork_exact_qsub(a,c);v=_artwork_exact_qsub(b,a)
    _artwork_exact_qroots!(w,_artwork_exact_qdot(v,v),2*_artwork_exact_qdot(u,v),_artwork_exact_qdot(u,u)-r*r)
end
function _artwork_exact_qevents!(w,s::_ArtworkRoundArc,a,b)
    c=_artwork_exact_qpoint(s.center);r=_artwork_exact_qarc_radius(s);width=_artwork_exact_q(s.radius)
    _artwork_exact_qcircle_events!(w,c,r+width,a,b)
    _artwork_exact_qcircle_events!(w,c,max(_artwork_exact_q(0),r-width),a,b)
    _artwork_exact_qcircle_events!(w,_artwork_exact_qpoint(s.start),width,a,b)
    _artwork_exact_qcircle_events!(w,_artwork_exact_qpoint(s.stop),width,a,b)
end
function _artwork_exact_qevents!(w,s::_ArtworkRegion,a,b)
    for segment in s.segments
        # Chord events also include literal endpoints. The stored rounded
        # radius can differ from the exact norm of either endpoint.
        _artwork_exact_qedge_events!(w,_artwork_exact_qpoint(segment.start),_artwork_exact_qpoint(segment.stop),a,b)
        if segment isa _ArtworkRoundArc
            _artwork_exact_qcircle_events!(w,_artwork_exact_qpoint(segment.center),_artwork_exact_qarc_radius(segment),a,b)
        end
    end
end

function _artwork_exact_qtrait(s,depth=0)
    depth<=64 || throw(ArgumentError("exact fallback depth bound exceeded"))
    s isa Union{_ArtworkEmpty,_ArtworkCircle,_ArtworkPolygon,_ArtworkRoundLine,_ArtworkPolygonLine,_ArtworkRoundArc,_ArtworkRegion} && return depth
    s isa _ArtworkTransform && return _artwork_exact_qtrait(s.shape,depth+1)
    s isa _ArtworkStrokeReference && return _artwork_exact_qtrait(s.geometry,depth+1)
    if s isa _ArtworkComposite
        result=depth
        for (_,part) in s.parts;result=max(result,_artwork_exact_qtrait(part,depth+1));end
        return result
    end
    throw(ArgumentError("unsupported exact artwork geometry: $(typeof(s))"))
end
function _artwork_exact_algebraic_event_stroke(aperture,start,stop,x,y;max_events=4096,max_bytes=16_000_000)
    depth=_artwork_exact_qtrait(aperture)
    count=_artwork_stroke_boundaries(aperture,max_events,0)+2
    count<=max_events || throw(ArgumentError("event preflight exceeds limit"))
    # Conservative exact dyadic conversion/product depth allowance before
    # materializing any event. This is a fallback reserve, not Float geometry.
    inputbits=max(_artwork_exact_operand_bits(aperture),_artwork_exact_operand_bits((start,stop,x,y)))
    bits=_checked_array_payload_bytes(UInt8,128,inputbits+1,depth+1)
    payload=_checked_payload_sum("exact artwork query workspace",
        _checked_array_payload_bytes(UInt8,6count+128*(depth+1),cld(bits,8)+64),
        _checked_array_payload_bytes(UInt8,256,count))
    _enforce_payload_limit(payload,max_bytes,"exact rational stroke events","max_bytes")
    a=_artwork_exact_qsub((_artwork_exact_q(x),_artwork_exact_q(y)),_artwork_exact_qpoint(start));b=_artwork_exact_qsub((_artwork_exact_q(x),_artwork_exact_q(y)),_artwork_exact_qpoint(stop))
    w=_ArtworkExactEventWork(_ArtworkQuadratic[],count,0);_artwork_exact_qpush!(w,_artwork_exact_q(0));_artwork_exact_qpush!(w,_artwork_exact_q(1));_artwork_exact_qevents!(w,aperture,a,b)
    v=_artwork_exact_qsub(b,a)
    for t in w.events
        _artwork_exact_qcontains(aperture,_artwork_exact_qadd(a,_artwork_exact_qscale(v,t))) && return (;hit=true,certified=true,events=length(w.events),irrational=w.irrational,payload)
    end
    # Boundary witnesses alone do not cover the open intervals between them.
    sort!(w.events);unique!(w.events)
    for i in 1:length(w.events)-1
        t=_artwork_exact_rational_separator(w.events[i],w.events[i+1],bits)
        _artwork_exact_qcontains(aperture,_artwork_exact_qadd(a,_artwork_exact_qscale(v,t))) && return (;hit=true,certified=true,events=length(w.events),irrational=w.irrational,payload)
    end
    (;hit=false,certified=true,events=length(w.events),irrational=w.irrational,payload)
end

# ODB leaves are defined before this file is included. The exact representation
# follows their original aperture equations and ordered wrapper semantics.
_artwork_exact_qcontains(::_ArtworkODBNull,p)=false
_artwork_exact_qcontains(s::_ArtworkODBDrill,p)=_artwork_exact_qcontains(s.circle,p)
_artwork_exact_qcontains(s::_ArtworkODBButterfly,p)=
    ((p[1]<=0&&p[2]>=0)||(p[1]>=0&&p[2]<=0))&&_artwork_exact_qcontains(s.outside,p)
function _artwork_exact_qcontains(s::_ArtworkCornerRectangle,p)
    x,y=abs(p[1]),abs(p[2]);w,h,r=_artwork_exact_q(s.width)/2,_artwork_exact_q(s.height)/2,_artwork_exact_q(s.radius)
    x<=w&&y<=h || return false
    corner=p[1]>=0 ? (p[2]>=0 ? 1 : 4) : (p[2]>=0 ? 2 : 3)
    s.corners[corner]&&x>w-r&&y>h-r || return true
    s.chamfer ? w-x+h-y>=r : (x-(w-r))^2+(y-(h-r))^2<=r*r
end
_artwork_exact_qevents!(w,::_ArtworkODBNull,a,b)=nothing
_artwork_exact_qevents!(w,s::_ArtworkODBDrill,a,b)=_artwork_exact_qevents!(w,s.circle,a,b)
function _artwork_exact_qevents!(w,s::_ArtworkODBButterfly,a,b)
    _artwork_exact_qevents!(w,s.outside,a,b)
    v=_artwork_exact_qsub(b,a)
    !iszero(v[1])&&_artwork_exact_qpush!(w,-a[1]/v[1])
    !iszero(v[2])&&_artwork_exact_qpush!(w,-a[2]/v[2])
end
function _artwork_exact_qevents!(w,s::_ArtworkCornerRectangle,a,b)
    q=_artwork_exact_q;ww,hh,r=q(s.width)/2,q(s.height)/2,q(s.radius)
    for (c,d) in (((-ww,-hh),(ww,-hh)),((ww,-hh),(ww,hh)),((ww,hh),(-ww,hh)),((-ww,hh),(-ww,-hh)))
        _artwork_exact_qedge_events!(w,c,d,a,b)
    end
    r>0 || return
    for (i,(sx,sy)) in enumerate(((1,1),(-1,1),(-1,-1),(1,-1)))
        s.corners[i] || continue
        if s.chamfer
            _artwork_exact_qedge_events!(w,(sx*(ww-r),sy*hh),(sx*ww,sy*(hh-r)),a,b)
        else
            _artwork_exact_qcircle_events!(w,(sx*(ww-r),sy*(hh-r)),r,a,b)
        end
    end
end
_artwork_exact_qtrait(s::Union{_ArtworkODBNull,_ArtworkODBDrill,_ArtworkCornerRectangle},depth=0)=depth
function _artwork_exact_qtrait(s::_ArtworkODBButterfly,depth=0)
    depth<64||throw(ArgumentError("exact artwork nesting exceeds 64 levels"))
    _artwork_exact_qtrait(s.outside,depth+1)
end
_artwork_exact_operand_bits(::Any)=0
function _artwork_exact_operand_bits(x::Float64)
    isfinite(x)||throw(ArgumentError("nonfinite exact artwork coordinate"))
    iszero(x)&&return 1
    _,e=frexp(abs(x))
    max(54,54-e,e+1)
end
function _artwork_exact_operand_bits(s::_ArtworkShape)
    result=1
    for i in 1:fieldcount(typeof(s));result=max(result,_artwork_exact_operand_bits(getfield(s,i)));end
    result
end
function _artwork_exact_operand_bits(parts::Union{Tuple,AbstractArray})
    result=1
    for part in parts;result=max(result,_artwork_exact_operand_bits(part));end
    result
end
function _artwork_exact_stroke(s::_ArtworkCompositeLine,x::Float64,y::Float64)
    isfinite(x)&&isfinite(y)||return false
    _artwork_exact_algebraic_event_stroke(s.aperture,s.start,s.stop,x,y;
        max_events=s.boundary_count+2,max_bytes=s.exact_bytes).hit
end
function _artwork_exact_point(s,x::Float64,y::Float64)
    _artwork_region_point_leaf(s)&&return _artwork_exact_region_point(s,x,y)
    depth=_artwork_exact_qtrait(s)
    inputbits=max(_artwork_exact_operand_bits(s),_artwork_exact_operand_bits((x,y)))
    bits=_checked_array_payload_bytes(UInt8,128,inputbits+1,depth+1)
    payload=_checked_payload_sum("exact aperture point workspace",Base.summarysize(s),
        _checked_array_payload_bytes(UInt8,128*(depth+1),cld(bits,8)+64))
    _enforce_payload_limit(payload,_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,"exact aperture point workspace","max_bytes")
    _artwork_exact_qcontains(s,(_artwork_exact_q(x),_artwork_exact_q(y)))
end
