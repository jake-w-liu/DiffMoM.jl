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

# Rounded annuli use parallel inner arcs; their radius is outer radius
# minus the wall thickness, clipped at zero. Primary Update 3 p200 depicts
# donut_s10x8xr2 and donut_rc10x7x1xr2 with these concentric corner arcs.
function _odb_rounded_annulus(w,h,iw,ih,wall,radius,corners)
    all(isfinite,(w,h,iw,ih,wall,radius)) && w>0 && h>0 &&
        0<=iw<w && 0<=ih<h && wall>0 && 0<=radius<=min(w,h)/2 ||
        throw(ArgumentError("invalid ODB rounded annulus dimensions"))
    outside=_odb_corner_rectangle(w,h,radius;corners)
    inside=iw==0 || ih==0 ? _ArtworkEmpty() :
        _odb_corner_rectangle(iw,ih,max(radius-wall,0.);corners)
    return _ArtworkComposite(((true,outside),(false,inside)))
end

function _odb_extra_standard_symbol(name::AbstractString,unit)
    number="([+\\-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+))"
    m=match(Regex("^donut_s"*number*"x"*number*"xr"*number*"(?:x([1-4]+))?"*raw"$"),name)
    if m!==nothing
        outside,inside,radius=parse.(Float64,m.captures[1:3]).*unit
        return _odb_rounded_annulus(outside,outside,inside,inside,
            outside/2-inside/2,radius,something(m[4],"1234"))
    end
    m=match(Regex("^donut_rc"*number*"x"*number*"x"*number*"xr"*number*"(?:x([1-4]+))?"*raw"$"),name)
    if m!==nothing
        w,h,wall,radius=parse.(Float64,m.captures[1:4]).*unit
        return _odb_rounded_annulus(w,h,w-2wall,h-2wall,wall,radius,something(m[5],"1234"))
    end
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
    # Update 3 p211: nr concentric rings of width rw separated by gap rg,
    # innermost ring a solid disc; crossed by a two-line crosshair of width
    # lw and length ll rotated la degrees clockwise (ODB angle sense).
    m=match(Regex("^moire"*number*"x"*number*"x([0-9]+)x"*number*"x"*number*"x"*number*raw"$"),name)
    if m!==nothing
        rw,rg,lw,ll=parse.(Float64,m.captures[[1,2,4,5]]).*unit
        la=parse(Float64,m[6]);nr=tryparse(Int,m[3])
        nr!==nothing && 1<=nr<=10_000 && all(isfinite,(rw,rg,lw,ll,la)) &&
            rw>0 && rg>=0 && lw>=0 && ll>=0 || throw(ArgumentError("invalid ODB moire parameters"))
        parts=Tuple{Bool,_ArtworkShape}[]
        for k in 1:nr
            outer=rw+(k-1)*(rw+rg)
            inner=outer-rw
            # Each ring is a self-contained annulus: a flat (false,inner)
            # part would also erase the previously drawn rings beneath it.
            push!(parts,(true,inner>0 ? _ArtworkComposite(Tuple{Bool,_ArtworkShape}[
                (true,_ArtworkCircle((0.,0.),outer)),
                (false,_ArtworkCircle((0.,0.),inner))]) :
                _ArtworkCircle((0.,0.),outer)))
        end
        if lw>0 && ll>0
            bar=_artwork_rectangle(ll,lw)
            cross=_ArtworkComposite(Tuple{Bool,_ArtworkShape}[(true,bar),
                (true,_artwork_transform(bar;rotation=90.))])
            push!(parts,(true,_artwork_transform(cross;rotation=-la)))
        end
        return _ArtworkComposite(parts)
    end
    # Update 3 p210-211: cross and dogbone are unions of orthogonal
    # constant-width strokes clipped to the declared w x h extent. The
    # cross point (hc[,vc]) shifts the vertical bar skeleton (and the
    # horizontal bar for cross) relative to the symbol origin. Style r
    # gives round stroke caps, s square ends; ra rounds the inside
    # (re-entrant) corners where the bars meet.
    function stroke_bar(a,b,t,round_ends)
        round_ends && return _ArtworkRoundLine(a,b,t/2)
        ux,uy=b[1]-a[1],b[2]-a[2];l=hypot(ux,uy)
        l>0 || throw(ArgumentError("degenerate ODB stencil bar"))
        nx,ny=(-uy/l*t/2,ux/l*t/2)
        _ArtworkRegion(Union{_ArtworkRoundLine,_ArtworkRoundArc}[
            _ArtworkRoundLine((a[1]+nx,a[2]+ny),(b[1]+nx,b[2]+ny),0.),
            _ArtworkRoundLine((b[1]+nx,b[2]+ny),(b[1]-nx,b[2]-ny),0.),
            _ArtworkRoundLine((b[1]-nx,b[2]-ny),(a[1]-nx,a[2]-ny),0.),
            _ArtworkRoundLine((a[1]-nx,a[2]-ny),(a[1]+nx,a[2]+ny),0.)])
    end
    function inside_fillet(cx,cy,sx,sy,ra)
        t1=(cx+sx*ra,cy);t2=(cx,cy+sy*ra);c=(cx+sx*ra,cy+sy*ra)
        _ArtworkRegion(Union{_ArtworkRoundLine,_ArtworkRoundArc}[
            _ArtworkRoundLine(c,t1,0.),
            _ArtworkRoundArc(t1,t2,c,sx*sy>0,0.),
            _ArtworkRoundLine(t2,c,0.)])
    end
    m=match(Regex("^cross"*number*"x"*number*"x"*number*"x"*number*"x"*number*"x"*number*
        "x([rs])(?:x"*number*")?"*raw"$"),name)
    if m!==nothing
        w,h,hs,vs,hc,vc=parse.(Float64,m.captures[1:6]).*unit
        ra=m[8]===nothing ? 0. : parse(Float64,m[8])*unit
        all(isfinite,(w,h,hs,vs,hc,vc,ra)) && w>0 && h>0 && 0<hs<h && 0<vs<w &&
            ra>=0 || throw(ArgumentError("invalid ODB cross parameters"))
        bars=Tuple{Bool,_ArtworkShape}[]
        span_h,span_v=(m[7]=="r" ? (w-hs,h-vs) : (w,h))
        push!(bars,(true,stroke_bar((-span_h/2,vc),(span_h/2,vc),hs,m[7]=="r")))
        push!(bars,(true,stroke_bar((hc,-span_v/2),(hc,span_v/2),vs,m[7]=="r")))
        if ra>0
            for sy in (-1.,1.),sx in (-1.,1.)
                abs(hc+sx*vs/2)<w/2 && abs(vc+sy*hs/2)<h/2 &&
                    push!(bars,(true,inside_fillet(hc+sx*vs/2,vc+sy*hs/2,sx,sy,ra)))
            end
        end
        return _ArtworkComposite(bars)
    end
    m=match(Regex("^dogbone"*number*"x"*number*"x"*number*"x"*number*"x"*number*
        "x([rs])(?:x"*number*")?"*raw"$"),name)
    if m!==nothing
        w,h,hs,vs,hc=parse.(Float64,m.captures[1:5]).*unit
        ra=m[7]===nothing ? 0. : parse(Float64,m[7])*unit
        all(isfinite,(w,h,hs,vs,hc,ra)) && w>0 && h>0 && 0<hs<h/2 && 0<vs<w &&
            ra>=0 || throw(ArgumentError("invalid ODB dogbone parameters"))
        bars=Tuple{Bool,_ArtworkShape}[]
        yoff=(h-hs)/2;span_h=m[6]=="r" ? w-hs : w
        push!(bars,(true,stroke_bar((-span_h/2,yoff),(span_h/2,yoff),hs,m[6]=="r")))
        push!(bars,(true,stroke_bar((-span_h/2,-yoff),(span_h/2,-yoff),hs,m[6]=="r")))
        push!(bars,(true,stroke_bar((hc,-yoff),(hc,yoff),vs,m[6]=="r")))
        if ra>0
            for sy in (-1.,1.),sx in (-1.,1.)
                # the notch opens toward the mid-plane: the fillet bulges
                # inward (-sy) from the h-bar's inner edge
                abs(hc+sx*vs/2)<w/2 &&
                    push!(bars,(true,inside_fillet(hc+sx*vs/2,sy*(yoff-hs/2),sx,-sy,ra)))
            end
        end
        return _ArtworkComposite(bars)
    end
    return nothing
end

# Ordered aperture sweep support. Candidates only partition the query
# segment; interval membership remains the authority, so supporting lines
# rather than finite pieces keep the implementation small and safe.
function _artwork_stroke_next_boundary(s::_ArtworkODBHalfOval,a,b,left,right)
    w,h=s.width/2,s.height/2;c=(s.width-s.height)/2
    for (p,q) in (((-w,-h),(w,-h)),((-w,h),(w,h)),((-w,-h),(-w,h)))
        right=_artwork_stroke_edge_next(p,q,a,b,left,right)
    end
    _artwork_stroke_circle_next((c,0.),h,a,b,left,right)
end
function _artwork_stroke_boundaries(s::_ArtworkODBHalfOval,limit,depth)
    all(isfinite,(s.width,s.height)) && s.height>0 && s.width>=s.height/2 ||
        throw(ArgumentError("invalid ODB half oval aperture"))
    _artwork_stroke_count(8,limit)
end

@inline function _odb_dpack_pads(s::_ArtworkODBDPack)
    (s.pad_width+s.horizontal_gap,s.pad_height+s.vertical_gap,
        -s.width/2+s.pad_width/2,-s.height/2+s.pad_height/2)
end
# Pads form a periodic grid, so a point or segment engine only needs the index
# window covering the query box — never hn*vn enumeration. A one-pad margin
# absorbs boundary rounding; non-finite boxes fall back to the full range.
function _odb_dpack_axis_window(x0,x1,center,pitch,half,count)
    lo=(x0-half-center)/pitch
    hi=(x1+half-center)/pitch
    (isfinite(lo)&&isfinite(hi))||return 0:count-1
    i0=clamp(floor(lo)-1,0.,Float64(count-1))
    i1=clamp(ceil(hi)+1,0.,Float64(count-1))
    i0>i1&&return 0:-1
    Int(i0):Int(i1)
end
function _odb_dpack_window(s::_ArtworkODBDPack,x0,x1,y0,y1)
    px,py,cx,cy=_odb_dpack_pads(s)
    (_odb_dpack_axis_window(x0,x1,cx,px,s.pad_width/2,s.horizontal_count),
        _odb_dpack_axis_window(y0,y1,cy,py,s.pad_height/2,s.vertical_count))
end
function _artwork_stroke_next_boundary(s::_ArtworkODBDPack,a,b,left,right)
    pad=_ArtworkCornerRectangle(s.pad_width,s.pad_height,s.radius,(true,true,true,true),false)
    px,py,cx,cy=_odb_dpack_pads(s)
    ir,jr=_odb_dpack_window(s,min(a[1],b[1]),max(a[1],b[1]),min(a[2],b[2]),max(a[2],b[2]))
    for j in jr,i in ir
        ox,oy=cx+i*px,cy+j*py
        right=_artwork_stroke_next_boundary(pad,(a[1]-ox,a[2]-oy),(b[1]-ox,b[2]-oy),left,right)
    end
    right
end
function _artwork_stroke_boundaries(s::_ArtworkODBDPack,limit,depth)
    all(isfinite,(s.width,s.height,s.pad_width,s.pad_height,s.horizontal_gap,
            s.vertical_gap,s.radius)) && s.horizontal_count>=1 && s.vertical_count>=1 ||
        throw(ArgumentError("invalid ODB D-Pack aperture"))
    pad=_artwork_stroke_boundaries(_ArtworkCornerRectangle(s.pad_width,s.pad_height,
        s.radius,(true,true,true,true),false),limit,depth)
    _artwork_stroke_count(_checked_array_payload_bytes(UInt8,pad,
        s.horizontal_count,s.vertical_count),limit)
end

# A thermal slot is a parallel-sided strip along a positive spoke ray. Its
# candidates are the two offset side lines and the perpendicular origin cap.
@inline _odb_thermal_spoke(angle,k,step)=(cos(angle+k*step),sin(angle+k*step))

# Spoke directions are periodic, so an engine only needs the window covering a
# query box's angular span about the origin — never a per-spoke enumeration.
# The window is (start,length) for indices mod(start+i,count), i=0:length-1.
function _odb_thermal_box_angles(x0,x1,y0,y1)
    (x0<=0<=x1&&y0<=0<=y1)&&return (0.,2Float64(pi))
    lo=min(atan(y0,x0),atan(y0,x1),atan(y1,x0),atan(y1,x1))
    hi=max(atan(y0,x0),atan(y0,x1),atan(y1,x0),atan(y1,x1))
    # A box touching the negative x axis wraps its span across ±pi.
    x1<0&&y0<=0<=y1&&return (hi,lo+2Float64(pi))
    return (lo,hi)
end
function _odb_thermal_spoke_window(base,step,count,slack,x0,x1,y0,y1)
    θ0,θ1=_odb_thermal_box_angles(x0,x1,y0,y1)
    θ1-θ0+2slack>=2Float64(pi)&&return (0,count)
    lo=(θ0-slack-base)/step;hi=(θ1+slack-base)/step
    (isfinite(lo)&&isfinite(hi))||return (0,count)
    f0=floor(lo)-1;f1=ceil(hi)+1
    f1-f0+1>=count&&return (0,count)
    return (floor(Int,mod(f0,count)),floor(Int,f1-f0+1))
end
# A squared-thermal slot reaches a query point only within asin(g/r) of the
# spoke direction for r>g, or within the dot>0 half-plane nearer the origin.
function _odb_thermal_squared_slack(g,step,x0,x1,y0,y1)
    rx=x0<=0<=x1 ? 0. : min(abs(x0),abs(x1))
    ry=y0<=0<=y1 ? 0. : min(abs(y0),abs(y1))
    rmin=hypot(rx,ry)
    return (rmin<=g ? Float64(pi)/2 : asin(g/rmin))+step+1e-12
end
# Rounded thermal per-spoke features live within half_gap_angle of the spoke
# plus the cap circle subtense; a cap enclosing the origin covers all angles.
function _odb_thermal_rounded_slack(s,step)
    s.cap_radius>=s.center_radius&&return Inf
    return s.half_gap_angle+asin(min(1.,s.cap_radius/s.center_radius))+step+1e-12
end
function _artwork_stroke_next_boundary(s::_ArtworkODBSquaredThermal,a,b,left,right)
    right=_artwork_stroke_next_boundary(s.outside,a,b,left,right)
    right=_artwork_stroke_next_boundary(s.inside,a,b,left,right)
    iszero(s.gap) && return right
    step=2Float64(pi)/s.count;base=deg2rad(s.angle);g=s.gap/2
    x0,x1=min(a[1],b[1]),max(a[1],b[1]);y0,y1=min(a[2],b[2]),max(a[2],b[2])
    slack=_odb_thermal_squared_slack(g,step,x0,x1,y0,y1)
    k0,kn=_odb_thermal_spoke_window(base,step,s.count,slack,x0,x1,y0,y1)
    for t in 0:kn-1
        ux,uy=_odb_thermal_spoke(base,mod(k0+t,s.count),step);nx,ny=-uy,ux
        for sgn in (-1.,1.)
            c=(sgn*g*nx,sgn*g*ny)
            right=_artwork_stroke_edge_next(c,(c[1]+ux,c[2]+uy),a,b,left,right)
        end
        right=_artwork_stroke_edge_next((0.,0.),(nx,ny),a,b,left,right)
    end
    right
end
function _artwork_stroke_boundaries(s::_ArtworkODBSquaredThermal,limit,depth)
    depth<64 || throw(ArgumentError("artwork aperture nesting exceeds 64 levels"))
    count=_checked_payload_sum("squared thermal boundary count",
        _artwork_stroke_boundaries(s.outside,limit,depth+1),
        _artwork_stroke_boundaries(s.inside,limit,depth+1))
    iszero(s.gap) || (count=_checked_payload_sum("squared thermal boundary count",count,6s.count))
    _artwork_stroke_count(count,limit)
end

function _artwork_stroke_next_boundary(s::_ArtworkODBRoundedThermal,a,b,left,right)
    cr,cap=s.center_radius,s.cap_radius
    right=_artwork_stroke_circle_next((0.,0.),cr+cap,a,b,left,right)
    right=_artwork_stroke_circle_next((0.,0.),max(0.,cr-cap),a,b,left,right)
    step=2Float64(pi)/s.count
    x0,x1=min(a[1],b[1]),max(a[1],b[1]);y0,y1=min(a[2],b[2]),max(a[2],b[2])
    slack=_odb_thermal_rounded_slack(s,step)
    k0,kn=_odb_thermal_spoke_window(s.angle,step,s.count,slack,x0,x1,y0,y1)
    for t in 0:kn-1
        thc=s.angle+mod(k0+t,s.count)*step
        for sgn in (-1.,1.)
            ux,uy=cos(thc+sgn*s.half_gap_angle),sin(thc+sgn*s.half_gap_angle)
            right=_artwork_stroke_edge_next((0.,0.),(ux,uy),a,b,left,right)
            right=_artwork_stroke_circle_next((cr*ux,cr*uy),cap,a,b,left,right)
        end
    end
    right
end
function _artwork_stroke_boundaries(s::_ArtworkODBRoundedThermal,limit,depth)
    all(isfinite,(s.center_radius,s.cap_radius,s.angle,s.half_gap_angle,s.outer_radius)) &&
        s.count>=1 && s.cap_radius>0 && s.center_radius>s.cap_radius ||
        throw(ArgumentError("invalid ODB rounded thermal aperture"))
    _artwork_stroke_count(_checked_payload_sum("rounded thermal boundary count",4,8s.count),limit)
end

# The four line-thermal arms are full capsules crossing the aperture center.
@inline _odb_thermal_line_arms(s::_ArtworkODBLineThermal)=(
    ((-s.endpoint,s.center),(s.endpoint,s.center)),
    ((-s.endpoint,-s.center),(s.endpoint,-s.center)),
    ((s.center,-s.endpoint),(s.center,s.endpoint)),
    ((-s.center,-s.endpoint),(-s.center,s.endpoint)))
function _artwork_stroke_next_boundary(s::_ArtworkODBLineThermal,a,b,left,right)
    for (c,d) in _odb_thermal_line_arms(s)
        right=_artwork_stroke_next_boundary(_ArtworkRoundLine(c,d,s.radius),a,b,left,right)
    end
    right
end
function _artwork_stroke_boundaries(s::_ArtworkODBLineThermal,limit,depth)
    all(isfinite,(s.center,s.endpoint,s.radius,s.outer_extent)) &&
        s.center>=0 && s.endpoint>=0 && s.radius>0 ||
        throw(ArgumentError("invalid ODB line thermal aperture"))
    _artwork_stroke_count(32,limit)
end
