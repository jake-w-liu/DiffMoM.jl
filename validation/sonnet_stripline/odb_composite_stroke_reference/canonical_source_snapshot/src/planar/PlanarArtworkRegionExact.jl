# A Region point query needs only signs of dyadic polynomials. Its finite
# binary precision is derived from input exponents and affine depth, so it
# evaluates the same exact equations as the rational point fallback while
# avoiding repeated rational normalization at artificial arc chords.
_artwork_region_point_leaf(::_ArtworkShape)=false
_artwork_region_point_leaf(::_ArtworkRegion)=true
_artwork_region_point_leaf(s::_ArtworkTransform)=_artwork_region_point_leaf(s.shape)
_artwork_region_point_leaf(s::_ArtworkStrokeReference)=_artwork_region_point_leaf(s.geometry)
_artwork_region_point(p)=(BigFloat(p[1]),BigFloat(p[2]))
_artwork_region_sub(a,b)=(a[1]-b[1],a[2]-b[2])
_artwork_region_dot(a,b)=a[1]*b[1]+a[2]*b[2]
_artwork_region_cross(a,b)=a[1]*b[2]-a[2]*b[1]
function _artwork_region_angle(s,p)
    s.start==s.stop&&return true
    c=_artwork_region_point(s.center)
    u,v,z=_artwork_region_sub(_artwork_region_point(s.start),c),
        _artwork_region_sub(_artwork_region_point(s.stop),c),_artwork_region_sub(p,c)
    s.clockwise&&((u,v)=(v,u))
    uv,uz,zv=_artwork_region_cross(u,v),_artwork_region_cross(u,z),_artwork_region_cross(z,v)
    if uv>0||(iszero(uv)&&_artwork_region_dot(u,v)<0)
        uz>=0&&zv>=0
    elseif uv<0
        uz>=0||zv>=0
    else
        iszero(uz)&&_artwork_region_dot(u,z)>=0
    end
end
function _artwork_region_chord_crosses(a,b,p)
    (a[2]>p[2])!=(b[2]>p[2])||return false
    cross=_artwork_linear_cross(a,b,p)
    b[2]>a[2] ? cross>0 : cross<0
end
function _artwork_region_chord_state(segment,point)
    a,b=segment.start,segment.stop;y=point[2]
    above_a,below_a=a[2]>y.hi,a[2]<=y.lo
    above_b,below_b=b[2]>y.hi,b[2]<=y.lo
    (above_a&&above_b||below_a&&below_b)&&return Int8(0)
    (above_a&&below_b||below_a&&above_b)||return Int8(-1)
    aa,bb=_artwork_interval_point(a),_artwork_interval_point(b)
    sign=_artwork_interval_sign(_artwork_interval_cross(_artwork_interval_sub(bb,aa),_artwork_interval_sub(point,aa)))
    abs(sign)==1||return Int8(-1)
    Int8(b[2]>a[2] ? sign>0 : sign<0)
end
@inline function _artwork_region_endpoint_possible(endpoint,point)
    point[1].lo<=endpoint[1]<=point[1].hi&&point[2].lo<=endpoint[2]<=point[2].hi
end
function _artwork_region_exact_contains(s::_ArtworkRegion,p)
    x,y=Float64(p[1]),Float64(p[2])
    point=(_ArtworkInterval(prevfloat(x),nextfloat(x)),_ArtworkInterval(prevfloat(y),nextfloat(y)))
    inside=false
    for segment in s.segments
        if segment isa _ArtworkRoundLine
            if _artwork_interval_edge_uncertain(_artwork_interval_point(segment.start),_artwork_interval_point(segment.stop),point...)
                _artwork_linear_on_segment(_artwork_region_point(segment.start),_artwork_region_point(segment.stop),p)&&return true
            end
        else
            for endpoint in (segment.start,segment.stop)
                _artwork_region_endpoint_possible(endpoint,point)&&p==_artwork_region_point(endpoint)&&return true
            end
            radius=hypot(segment.start[1]-segment.center[1],segment.start[2]-segment.center[2])
            disk_state=_artwork_certified_circle(segment.center,radius,point...)
            if disk_state==-1
                z=_artwork_region_sub(p,_artwork_region_point(segment.center));r=BigFloat(radius)
                dist=_artwork_region_dot(z,z)
                dist==r*r&&_artwork_region_angle(segment,p)&&return true
            end
        end
        chord=_artwork_region_chord_state(segment,point)
        if chord==-1
            _artwork_region_chord_crosses(_artwork_region_point(segment.start),_artwork_region_point(segment.stop),p)&&(inside=!inside)
        else
            chord==1&&(inside=!inside)
        end
        if segment isa _ArtworkRoundArc
            # Formal p+(epsilon,epsilon^2) resolves artificial chord ties.
            disk=disk_state==1||disk_state==-1&&_artwork_exact_qlexsign((r*r-dist,-2z[1],-BigFloat(1)-2z[2],-BigFloat(1)))>0
            disk||continue
            if segment.start==segment.stop
                inside=!inside
            else
                aa,bb=_artwork_interval_point(segment.start),_artwork_interval_point(segment.stop)
                side=_artwork_interval_sign(_artwork_interval_cross(_artwork_interval_sub(bb,aa),_artwork_interval_sub(point,aa)))
                if abs(side)!=1
                    a,b=_artwork_region_point(segment.start),_artwork_region_point(segment.stop);e=_artwork_region_sub(b,a)
                    side=_artwork_exact_qlexsign((_artwork_linear_cross(a,b,p),-e[2],e[1]))
                end
                desired=segment.clockwise ? side>0 : side<0
                desired&&(inside=!inside)
            end
        end
    end
    inside
end
_artwork_region_exact_contains(s::_ArtworkStrokeReference,p)=_artwork_region_exact_contains(s.geometry,p)
_artwork_region_exact_contains(s::_ArtworkTransform,p)=_artwork_region_exact_contains(s.shape,_artwork_linear_transform(s,p))
function _artwork_exact_region_point(s,x,y)
    depth=_artwork_exact_qtrait(s)
    bits=max(_artwork_exact_operand_bits(s),_artwork_exact_operand_bits((x,y)))
    precision=_checked_payload_sum("exact region point precision",64,
        _checked_array_payload_bytes(UInt8,8,bits+1,depth+1))
    payload=_checked_payload_sum("exact region point workspace",Base.summarysize(s),
        _checked_array_payload_bytes(UInt8,128,depth+1,cld(precision,8)+64))
    _enforce_payload_limit(payload,_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,"exact region point workspace","max_bytes")
    setprecision(BigFloat,precision) do
        _artwork_region_exact_contains(s,(BigFloat(x),BigFloat(y)))
    end
end
