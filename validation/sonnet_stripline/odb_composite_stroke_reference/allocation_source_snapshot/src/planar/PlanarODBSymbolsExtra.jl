# Additional analytic ODB++ pad symbols. Siemens ODB++ 8.1 Update 3,
# printed pp202-213; v7 printed p22 distinguishes thermal gap angles
# from the clockwise pad-orientation convention.

# Primary p211 depicts hn pads along the horizontal side and vn along
# the vertical side. Each axis obeys n*pad+(n-1)*gap=side. Retain the
# periodic grid instead of allocating a shape per pad.
struct _ArtworkODBDPack <: _ArtworkShape
    width::Float64
    height::Float64
    pad_width::Float64
    pad_height::Float64
    horizontal_gap::Float64
    vertical_gap::Float64
    horizontal_count::Int
    vertical_count::Int
    radius::Float64
end
_artwork_bounds(s::_ArtworkODBDPack)=(-s.width/2,s.width/2,-s.height/2,s.height/2)
function _artwork_contains(s::_ArtworkODBDPack,x,y)
    abs(x)<=s.width/2 && abs(y)<=s.height/2 || return false
    iszero(s.radius) && iszero(s.horizontal_gap) && iszero(s.vertical_gap) && return true
    px=s.pad_width+s.horizontal_gap;py=s.pad_height+s.vertical_gap
    cx=-s.width/2+s.pad_width/2;cy=-s.height/2+s.pad_height/2
    ix=clamp(round((x-cx)/px),0.,Float64(s.horizontal_count-1))
    iy=clamp(round((y-cy)/py),0.,Float64(s.vertical_count-1))
    ax=abs(x-(cx+ix*px));ay=abs(y-(cy+iy*py))
    ax<=s.pad_width/2 && ay<=s.pad_height/2 || return false
    return ax<=s.pad_width/2-s.radius || ay<=s.pad_height/2-s.radius ||
        hypot(ax-(s.pad_width/2-s.radius),ay-(s.pad_height/2-s.radius))<=s.radius
end
function _odb_dpack(w,h,hg,vg,hn,vn,radius)
    # Larger integer counts cannot retain adjacent Float64 pad identities.
    # Reject that representation boundary instead of rounding a count.
    all(isfinite,(w,h,hg,vg,radius)) && w>0 && h>0 && hg>=0 && vg>=0 &&
        hn!==nothing && vn!==nothing && 1<=hn<=2^53 && 1<=vn<=2^53 ||
        throw(ArgumentError("invalid ODB D-Pack dimensions/counts"))
    pw=(w-hg*(hn-1))/hn;ph=(h-vg*(vn-1))/vn
    isfinite(pw) && isfinite(ph) && pw>0 && ph>0 && 0<=radius<=min(pw,ph)/2 ||
        throw(ArgumentError("invalid ODB D-Pack pad dimensions/radius"))
    return _ArtworkODBDPack(w,h,pw,ph,hg,vg,hn,vn,radius)
end

struct _ArtworkODBHalfOval <: _ArtworkShape
    width::Float64
    height::Float64
end
_artwork_bounds(s::_ArtworkODBHalfOval)=(-s.width/2,s.width/2,-s.height/2,s.height/2)
function _artwork_contains(s::_ArtworkODBHalfOval,x,y)
    -s.width/2<=x<=s.width/2 && abs(y)<=s.height/2 || return false
    c=(s.width-s.height)/2
    return x<=c || hypot(x-c,y)<=s.height/2
end

# A radial slot has parallel sides separated by gap, and extends only
# along its positive ray. A periodic nearest-ray calculation avoids an
# allocation or a loop proportional to the number of thermal gaps.
struct _ArtworkODBSquaredThermal{O<:_ArtworkShape,I<:_ArtworkShape} <: _ArtworkShape
    outside::O
    inside::I
    angle::Float64
    count::Int
    gap::Float64
end
_artwork_bounds(s::_ArtworkODBSquaredThermal)=_artwork_bounds(s.outside)
function _artwork_contains(s::_ArtworkODBSquaredThermal,x,y)
    _artwork_contains(s.outside,x,y) && !_artwork_contains(s.inside,x,y) || return false
    iszero(s.gap) && return true
    # Thermal gap angles are counter-clockwise (primary v7 p22).
    # Pad/rotation-suffix orientations have the separate clockwise rule.
    step=2Float64(pi)/s.count
    delta=mod(atan(y,x)-deg2rad(s.angle)+step/2,step)-step/2
    radius=hypot(x,y)
    return radius*cos(delta)<0 || abs(radius*sin(delta))>=s.gap/2
end

# Rounded thermals are swept circular arcs with circular end caps. The
# declared gap is the minimum distance between adjacent caps, not their
# centerline separation. Store only periodic geometry: count never allocates
# a proportional list of spokes or arcs.
struct _ArtworkODBRoundedThermal <: _ArtworkShape
    center_radius::Float64
    cap_radius::Float64
    angle::Float64
    count::Int
    half_gap_angle::Float64
    outer_radius::Float64
end
_artwork_bounds(s::_ArtworkODBRoundedThermal)=(-s.outer_radius,s.outer_radius,-s.outer_radius,s.outer_radius)
function _artwork_contains(s::_ArtworkODBRoundedThermal,x,y)
    radius=hypot(x,y)
    abs(radius-s.center_radius)<=s.cap_radius || return false
    theta=atan(y,x);step=2Float64(pi)/s.count
    delta=mod(theta-s.angle+step/2,step)-step/2
    abs(delta)>=s.half_gap_angle && return true
    endpoint=theta-delta+copysign(s.half_gap_angle,delta)
    return hypot(x-s.center_radius*cos(endpoint),y-s.center_radius*sin(endpoint))<=s.cap_radius
end

# Primary Update 3 p204 fixes the end-cap diameter to (outside-inside)/2.
# The four line thermals keep axis-aligned centerlines; adjacent cap centers
# are separated by sqrt(2)*(center-endpoint).
struct _ArtworkODBLineThermal <: _ArtworkShape
    center::Float64
    endpoint::Float64
    radius::Float64
    outer_extent::Float64
end
_artwork_bounds(s::_ArtworkODBLineThermal)=(-s.outer_extent,s.outer_extent,-s.outer_extent,s.outer_extent)
function _artwork_contains(s::_ArtworkODBLineThermal,x,y)
    ax,ay=abs(x),abs(y)
    return hypot(max(ax-s.endpoint,0.),ay-s.center)<=s.radius ||
        hypot(ax-s.center,max(ay-s.endpoint,0.))<=s.radius
end

function _odb_stencil_region(points,radii)
    n=length(points);n>=3 && length(radii)==n || throw(ArgumentError("invalid stencil corners"))
    all(p->all(isfinite,p),points) && all(r->isfinite(r)&&r>=0,radii) ||
        throw(ArgumentError("nonfinite stencil geometry or invalid radius"))
    area=sum(points[i][1]*points[mod1(i+1,n)][2]-points[mod1(i+1,n)][1]*points[i][2] for i in 1:n)
    area>0 || throw(ArgumentError("stencil boundary must have positive orientation"))
    starts=Vector{NTuple{2,Float64}}(undef,n);stops=similar(starts);centers=similar(starts)
    turns=Vector{Float64}(undef,n);distances=similar(turns);lengths=similar(turns)
    for i in 1:n
        p,v,q=points[mod1(i-1,n)],points[i],points[mod1(i+1,n)]
        li=hypot(v[1]-p[1],v[2]-p[2]);lo=hypot(q[1]-v[1],q[2]-v[2])
        li>0 && lo>0 || throw(ArgumentError("duplicate stencil vertices"))
        ux,uy=(v[1]-p[1])/li,(v[2]-p[2])/li
        wx,wy=(q[1]-v[1])/lo,(q[2]-v[2])/lo
        cross,dot=ux*wy-uy*wx,ux*wx+uy*wy
        turn=atan(cross,dot)
        abs(turn)<Float64(pi) || throw(ArgumentError("degenerate stencil corner"))
        # tan(turn/2)=|cross|/(1+dot). This preserves exactly right
        # angle tangent lengths; tan(pi/4) rounds down and would wrongly
        # admit two meeting radius1 corners on a length2 edge.
        d=radii[i]*abs(cross)/(1+dot)
        starts[i]=(v[1]-d*ux,v[2]-d*uy);stops[i]=(v[1]+d*wx,v[2]+d*wy)
        centers[i]=(starts[i][1]-sign(turn)*radii[i]*uy,
            starts[i][2]+sign(turn)*radii[i]*ux)
        turns[i]=turn;distances[i]=d;lengths[i]=lo
    end
    # Primary p213 specifies strict nonmeeting adjacent tangent portions.
    for i in 1:n
        distances[i]+distances[mod1(i+1,n)]<lengths[i] ||
            throw(ArgumentError("adjacent stencil corner fillets meet"))
    end
    segments=Union{_ArtworkRoundLine,_ArtworkRoundArc}[]
    for i in 1:n
        if radii[i]>0 && !iszero(turns[i])
            push!(segments,_ArtworkRoundArc(starts[i],stops[i],centers[i],turns[i]<0,0.))
        end
        push!(segments,_ArtworkRoundLine(stops[i],starts[mod1(i+1,n)],0.))
    end
    return _ArtworkRegion(segments)
end

function _odb_extra_standard_symbol(name::AbstractString,unit)
    number="([+\\-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+))"
    m=match(Regex("^dpack"*number*"x"*number*"x"*number*"x"*number*
        "x([0-9]+)x([0-9]+)(?:x(?:ra)?"*number*")?"*raw"$"),name)
    if m!==nothing
        w,h,hg,vg=parse.(Float64,m.captures[1:4]).*unit
        hn,vn=tryparse(Int,m[5]),tryparse(Int,m[6])
        radius=m[7]===nothing ? 0. : parse(Float64,m[7])*unit
        return _odb_dpack(w,h,hg,vg,hn,vn,radius)
    end
    m=match(Regex("^(thr|s_thr)"*number*"x"*number*"x"*number*"x([0-9]+)x"*number*raw"$"),name)
    if m!==nothing
        outside,inside,angle,gap=parse.(Float64,(m[2],m[3],m[4],m[6]))
        outside*=unit;inside*=unit;gap*=unit;count=tryparse(Int,m[5])
        count!==nothing && count>0 && all(isfinite,(outside,inside,angle,gap)) &&
            outside>inside>=0 && gap>=0 && 0<=angle<=360 || throw(ArgumentError("invalid ODB rounded thermal parameters"))
        center=outside/4+inside/4;radius=(outside-inside)/4
        radius>0 || throw(ArgumentError("ODB rounded thermal width underflows"))
        if m[1]=="s_thr"
            angle==45 && count==4 || throw(ArgumentError("ODB line thermal requires angle 45 and four spokes"))
            endpoint=center-(gap+2radius)/sqrt(2.)
            isfinite(endpoint) && endpoint>=0 || throw(ArgumentError("ODB line thermal gap leaves negative line length"))
            return _ArtworkODBLineThermal(center,endpoint,radius,outside/2)
        end
        ratio=(gap+2radius)/(2center)
        isfinite(ratio) && 0<=ratio<=1 || throw(ArgumentError("ODB rounded thermal gap exceeds its diameter"))
        delta=asin(ratio)
        delta<=Float64(pi)/count || throw(ArgumentError("ODB rounded thermal gap leaves negative arc length"))
        return _ArtworkODBRoundedThermal(center,radius,deg2rad(angle),count,delta,outside/2)
    end
    m=match(Regex("^oval_h"*number*"x"*number*raw"$"),name)
    if m!==nothing
        w,h=parse.(Float64,m.captures).*unit
        all(isfinite,(w,h)) && h>0 && w>=h/2 ||
            throw(ArgumentError("ODB half oval requires positive axes and width at least half height"))
        return _ArtworkODBHalfOval(w,h)
    end
    m=match(Regex("^(ths|s_ths|sr_ths)"*number*"x"*number*"x"*number*"x([0-9]+)x"*number*raw"$"),name)
    if m!==nothing
        outside,inside,angle,gap=parse.(Float64,(m[2],m[3],m[4],m[6]))
        outside*=unit;inside*=unit;gap*=unit
        count=tryparse(Int,m[5])
        count!==nothing && count>0 && all(isfinite,(outside,inside,angle,gap)) &&
            outside>inside>=0 && gap>=0 && 0<=angle<=360 || throw(ArgumentError("invalid ODB thermal parameters"))
        out=m[1]=="ths" ? _ArtworkCircle((0.,0.),outside/2) : _artwork_rectangle(outside,outside)
        inn=inside==0 ? _ArtworkEmpty() : m[1]=="s_ths" ? _artwork_rectangle(inside,inside) :
            _ArtworkCircle((0.,0.),inside/2)
        return _ArtworkODBSquaredThermal(out,inn,angle,count,gap)
    end
    m=match(Regex("^(rc_ths|o_ths)"*number*"x"*number*"x"*number*"x([0-9]+)x"*number*"x"*number*raw"$"),name)
    if m!==nothing
        w,h,angle,gap,t=parse.(Float64,(m[2],m[3],m[4],m[6],m[7]))
        w*=unit;h*=unit;gap*=unit;t*=unit;count=tryparse(Int,m[5])
        count!==nothing && count>0 && all(isfinite,(w,h,angle,gap,t)) &&
            w>0 && h>0 && 0<t<min(w,h)/2 && gap>=0 && 0<=angle<=360 || throw(ArgumentError("invalid ODB rectangular/oval thermal"))
        m[1]=="rc_ths" && !iszero(rem(angle,45)) && throw(ArgumentError("ODB rectangular thermal angle must be a multiple of 45 degrees"))
        if m[1]=="rc_ths"
            out=_artwork_rectangle(w,h);inn=_artwork_rectangle(w-2t,h-2t)
        else
            oval(w,h)=w>=h ? _ArtworkRoundLine((-(w-h)/2,0.),((w-h)/2,0.),h/2) :
                _ArtworkRoundLine((0.,-(h-w)/2),(0.,(h-w)/2),w/2)
            out=oval(w,h);inn=oval(w-2t,h-2t)
        end
        return _ArtworkODBSquaredThermal(out,inn,angle,count,gap)
    end
    # The primary tables use ordered numeric optional radii; the example
    # figures spell those same parameters ra/ro. Accept both spellings.
    m=match(Regex("^(hplate|rhplate)"*number*"x"*number*"x"*number*"(?:x(?:ra)?"*number*")?(?:x(?:ro)?"*number*")?"*raw"$"),name)
    if m!==nothing
        w,h,c=parse.(Float64,m.captures[2:4]).*unit
        all(isfinite,(w,h,c)) && w>0 && h>0 && 0<=c &&
            (m[1]=="hplate" ? c<=w : c<w) || throw(ArgumentError("invalid ODB home plate dimensions"))
        ra=m[5]===nothing ? 0. : parse(Float64,m[5])*unit
        ro=m[6]===nothing ? 0. : parse(Float64,m[6])*unit
        if c==0
            radii=m[1]=="hplate" ? [ra,ro,ro,ra] : fill(ra,4)
            return _odb_stencil_region([(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)],radii)
        elseif c==w
            return _odb_stencil_region([(-w/2,-h/2),(w/2,0.),(-w/2,h/2)],fill(ra,3))
        end
        if m[1]=="hplate"
            points=[(-w/2,-h/2),(w/2-c,-h/2),(w/2,0.),(w/2-c,h/2),(-w/2,h/2)]
            radii=[ra,ro,ra,ro,ra]
        else
            points=[(-w/2,-h/2),(w/2,-h/2),(w/2-c,0.),(w/2,h/2),(-w/2,h/2)]
            radii=[ra,ra,ro,ra,ra]
        end
        return _odb_stencil_region(points,radii)
    end
    m=match(Regex("^fhplate"*number*"x"*number*"x"*number*"x"*number*"(?:x(?:ra)?"*number*")?(?:x(?:ro)?"*number*")?"*raw"$"),name)
    if m!==nothing
        w,h,vc,hc=parse.(Float64,m.captures[1:4]).*unit
        all(isfinite,(w,h,vc,hc)) && w>0 && h>0 && 0<=vc<h/2 && 0<=hc<w ||
            throw(ArgumentError("invalid ODB flat home plate dimensions"))
        ra=m[5]===nothing ? 0. : parse(Float64,m[5])*unit
        ro=m[6]===nothing ? 0. : parse(Float64,m[6])*unit
        if hc==0 || vc==0
            return _odb_stencil_region([(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)],[ra,ro,ro,ra])
        end
        points=[(-w/2,-h/2),(w/2-hc,-h/2),(w/2,-h/2+vc),(w/2,h/2-vc),(w/2-hc,h/2),(-w/2,h/2)]
        return _odb_stencil_region(points,[ra,ro,ro,ro,ro,ra])
    end
    return nothing
end
