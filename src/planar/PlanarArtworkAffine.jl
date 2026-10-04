# Legacy image-coordinate transforms leave aperture dimensions unchanged.
# Region geometry can use an inverse affine map; swept apertures require
# transforming the path separately from the aperture.
function _artwork_affine(shape::_ArtworkShape,sx,sy,origin)
    matrix=SMatrix{2,2,Float64}(sx,0.,0.,sy)
    all(isfinite,(sx,sy,origin...)) && !iszero(sx) && !iszero(sy) ||
        throw(ArgumentError("invalid coordinate transformation"))
    inverse=_artwork_inverse_matrix(matrix)
    return _ArtworkTransform(shape,matrix,inverse,(Float64(origin[1]),Float64(origin[2])))
end

struct _ArtworkAffineArc
    path::_ArtworkRoundArc
    sx::Float64
    sy::Float64
    origin::NTuple{2,Float64}
end
struct _ArtworkCircleAffineArc <: _ArtworkShape
    path::_ArtworkAffineArc
    radius::Float64
end
struct _ArtworkPolygonAffineArc <: _ArtworkShape
    path::_ArtworkAffineArc
    aperture::_ArtworkPolygon
end

@inline _artwork_affine_point(p,sx,sy,origin)=(sx*p[1]+origin[1],sy*p[2]+origin[2])
function _artwork_bounds(s::_ArtworkAffineArc)
    x0,x1,y0,y1=_artwork_bounds(s.path)
    ax,bx=minmax(s.sx*x0+s.origin[1],s.sx*x1+s.origin[1])
    ay,by=minmax(s.sy*y0+s.origin[2],s.sy*y1+s.origin[2])
    return (ax,bx,ay,by)
end
function _artwork_bounds(s::_ArtworkCircleAffineArc)
    x0,x1,y0,y1=_artwork_bounds(s.path);r=s.radius
    return (x0-r,x1+r,y0-r,y1+r)
end
function _artwork_bounds(s::_ArtworkPolygonAffineArc)
    a,b=_artwork_bounds(s.path),_artwork_bounds(s.aperture)
    return (a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4])
end
@inline function _artwork_affine_arc_point(s::_ArtworkAffineArc,angle)
    p=s.path;r=hypot(p.start[1]-p.center[1],p.start[2]-p.center[2])
    return _artwork_affine_point((p.center[1]+r*cos(angle),p.center[2]+r*sin(angle)),s.sx,s.sy,s.origin)
end

@inline function _artwork_poly_value(c,x)
    result=last(c)
    for i in length(c)-1:-1:1;result=muladd(result,x,c[i]);end
    return result
end

# Return roots together with derivative stationary points. Extra candidates
# are harmless: membership always evaluates a point on the actual curve.
# Including derivative candidates retains even-multiplicity roots without
# a near-zero threshold. Degree <=4 gives at most 1+2+3+4=10 candidates.
function _artwork_poly_candidates(c::NTuple{N,Float64})::Tuple{NTuple{16,Float64},Int} where N
    values::NTuple{16,Float64}=ntuple(_->0.,Val(16))
    N==1 && return values,0
    iszero(last(c)) && return _artwork_poly_candidates(Base.front(c))
    if N==2
        x=-c[1]/c[2]
        return isfinite(x) && -1<=x<=1 ? (Base.setindex(values,x,1),1) : (values,0)
    end
    derivative=ntuple(i->i*c[i+1],Val(N-1))
    cuts,ncuts=_artwork_poly_candidates(derivative);count=0;left=-1.
    for k in 1:ncuts+1
        right=k<=ncuts ? cuts[k] : 1.
        a,b=_artwork_poly_value(c,left),_artwork_poly_value(c,right)
        if left<right && !iszero(a) && !iszero(b) && signbit(a)!=signbit(b)
            lo,hi=left,right
            for iteration in 1:60
                middle=lo+(hi-lo)/2
                (middle==lo || middle==hi) && break
                v=_artwork_poly_value(c,middle)
                if iszero(v);lo=hi=middle;break
                elseif signbit(v)==signbit(a);lo=middle
                else;hi=middle
                end
            end
            count+=1;count<=16 || error("polynomial candidate bound exceeded")
            values=Base.setindex(values,lo+(hi-lo)/2,count)
        end
        if k<=ncuts && (count==0 || values[count]!=right)
            count+=1;count<=16 || error("polynomial candidate bound exceeded")
            values=Base.setindex(values,right,count)
        end
        left=right
    end
    return values,count
end

function _artwork_contains(s::_ArtworkCircleAffineArc,x,y)
    p=s.path;source=p.path
    for endpoint in (source.start,source.stop)
        a,b=_artwork_affine_point(endpoint,p.sx,p.sy,p.origin)
        hypot(x-a,y-b)<=s.radius && return true
    end
    hit(angle)=begin
        _artwork_arc_fraction(source,angle)<=1 || return false
        qx,qy=_artwork_affine_arc_point(p,angle)
        return hypot(x-qx,y-qy)<=s.radius
    end
    for angle in (0.,pi/2,Float64(pi),3pi/2);hit(angle) && return true;end
    r=hypot(source.start[1]-source.center[1],source.start[2]-source.center[2])
    center=_artwork_affine_point(source.center,p.sx,p.sy,p.origin)
    a,b=abs(p.sx)*r,abs(p.sy)*r
    xp,yp=sign(p.sx)*(x-center[1]),sign(p.sy)*(y-center[2])
    scale=max(a,b,abs(xp),abs(yp));a/=scale;b/=scale;xp/=scale;yp/=scale
    delta=b*b-a*a;ax=a*xp;by=b*yp
    c=(-by,2ax+2delta,0.,2ax-2delta,by)
    # Half-angle and reciprocal charts keep every polynomial solve in
    # [-1,1], including extrema close to theta=pi, without large root bounds.
    roots,n=_artwork_poly_candidates(c)
    for k in 1:n;hit(2atan(roots[k])) && return true;end
    roots,n=_artwork_poly_candidates(reverse(c))
    for k in 1:n;hit(2atan(1.,roots[k])) && return true;end
    return false
end

function _artwork_contains(s::_ArtworkPolygonAffineArc,x,y)
    p=s.path;source=p.path
    for endpoint in (source.start,source.stop)
        a,b=_artwork_affine_point(endpoint,p.sx,p.sy,p.origin)
        _artwork_contains(s.aperture,x-a,y-b) && return true
    end
    vertices=s.aperture.vertices
    r=hypot(source.start[1]-source.center[1],source.start[2]-source.center[2])
    for i in axes(vertices,2)
        j=mod1(i+1,size(vertices,2))
        ax=(x-vertices[1,i]-p.origin[1])/p.sx-source.center[1]
        ay=(y-vertices[2,i]-p.origin[2])/p.sy-source.center[2]
        bx=(x-vertices[1,j]-p.origin[1])/p.sx-source.center[1]
        by=(y-vertices[2,j]-p.origin[2])/p.sy-source.center[2]
        dx,dy=bx-ax,by-ay
        A=dx*dx+dy*dy;B=2*(ax*dx+ay*dy);C=ax*ax+ay*ay-r*r
        iszero(A) && continue
        discriminant=muladd(B,B,-4A*C)
        discriminant>=0 || continue
        root=sqrt(discriminant)
        q=-.5*(B+copysign(root,B))
        t1=iszero(q) ? -B/(2A) : q/A;t2=iszero(q) ? t1 : C/q
        for t in (t1,t2)
            0<=t<=1 || continue
            angle=atan(ay+t*dy,ax+t*dx)
            _artwork_arc_fraction(source,angle)<=1 && return true
        end
    end
    return false
end

function _gerber_coordinate_path(segment::_ArtworkRoundArc,sx,sy,origin)
    if abs(sx)==abs(sy)
        return _ArtworkRoundArc(_artwork_affine_point(segment.start,sx,sy,origin),
            _artwork_affine_point(segment.stop,sx,sy,origin),
            _artwork_affine_point(segment.center,sx,sy,origin),xor(segment.clockwise,sx*sy<0),0.)
    end
    return _ArtworkAffineArc(segment,Float64(sx),Float64(sy),(Float64(origin[1]),Float64(origin[2])))
end
