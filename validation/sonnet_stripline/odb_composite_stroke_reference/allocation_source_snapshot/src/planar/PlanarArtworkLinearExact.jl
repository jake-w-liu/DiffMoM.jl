# Dyadic polygon intersection needs only polynomial signs. A finite binary
# precision derived from operand exponents and transform depth represents
# every degree-two intermediate exactly; it avoids general quadratic fields.
_artwork_linear_leaf(::_ArtworkShape)=false
_artwork_linear_leaf(::_ArtworkPolygon)=true
_artwork_linear_leaf(s::_ArtworkStrokeReference)=_artwork_linear_leaf(s.geometry)
_artwork_linear_leaf(s::_ArtworkTransform)=_artwork_linear_leaf(s.shape)
_artwork_linear_depth(::_ArtworkPolygon)=0
_artwork_linear_depth(s::_ArtworkStrokeReference)=_artwork_linear_depth(s.geometry)
_artwork_linear_depth(s::_ArtworkTransform)=1+_artwork_linear_depth(s.shape)
@inline _artwork_linear_cross(a,b,c)=(b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1])
@inline function _artwork_linear_on_segment(a,b,p)
    min(a[1],b[1])<=p[1]<=max(a[1],b[1])&&min(a[2],b[2])<=p[2]<=max(a[2],b[2])&&iszero(_artwork_linear_cross(a,b,p))
end
function _artwork_linear_point(s::_ArtworkPolygon,p)
    inside=false;v=s.vertices;x0=x1=v[1,1];y0=y1=v[2,1]
    for i in axes(v,2)
        x0=min(x0,v[1,i]);x1=max(x1,v[1,i]);y0=min(y0,v[2,i]);y1=max(y1,v[2,i])
    end
    (p[1]<x0||p[1]>x1||p[2]<y0||p[2]>y1)&&return false
    for i in axes(v,2)
        j=mod1(i+1,size(v,2));a=(BigFloat(v[1,i]),BigFloat(v[2,i]));b=(BigFloat(v[1,j]),BigFloat(v[2,j]))
        cross=_artwork_linear_cross(a,b,p)
        iszero(cross)&&_artwork_linear_on_segment(a,b,p)&&return true
        (a[2]>p[2])!=(b[2]>p[2])&&(b[2]>a[2] ? cross>0 : cross<0)&&(inside=!inside)
    end
    inside
end
function _artwork_linear_intersection(a,b,c,d)
    max(min(a[1],b[1]),min(c[1],d[1]))<=min(max(a[1],b[1]),max(c[1],d[1]))&&
        max(min(a[2],b[2]),min(c[2],d[2]))<=min(max(a[2],b[2]),max(c[2],d[2]))||return false
    u,v=_artwork_linear_cross(a,b,c),_artwork_linear_cross(a,b,d)
    (u<=0<=v||v<=0<=u)||return false
    u,v=_artwork_linear_cross(c,d,a),_artwork_linear_cross(c,d,b)
    u<=0<=v||v<=0<=u
end
function _artwork_linear_hit(s::_ArtworkPolygon,a,b)
    (_artwork_linear_point(s,a)||_artwork_linear_point(s,b))&&return true
    v=s.vertices
    for i in axes(v,2)
        j=mod1(i+1,size(v,2))
        _artwork_linear_intersection(a,b,(BigFloat(v[1,i]),BigFloat(v[2,i])),(BigFloat(v[1,j]),BigFloat(v[2,j])))&&return true
    end
    false
end
_artwork_linear_hit(s::_ArtworkStrokeReference,a,b)=_artwork_linear_hit(s.geometry,a,b)
function _artwork_linear_transform(s,p)
    x,y=p[1]-BigFloat(s.origin[1]),p[2]-BigFloat(s.origin[2]);m=s.inverse
    (BigFloat(m[1,1])*x+BigFloat(m[1,2])*y,BigFloat(m[2,1])*x+BigFloat(m[2,2])*y)
end
_artwork_linear_hit(s::_ArtworkTransform,a,b)=_artwork_linear_hit(s.shape,_artwork_linear_transform(s,a),_artwork_linear_transform(s,b))
function _artwork_exact_linear_stroke(shape,s,x,y)
    depth=_artwork_linear_depth(shape)
    bits=max(_artwork_exact_operand_bits(shape),_artwork_exact_operand_bits((s.start,s.stop,x,y)))
    precision=_checked_payload_sum("exact linear artwork precision",64,
        _checked_array_payload_bytes(UInt8,8,bits+1,depth+1))
    payload=_checked_array_payload_bytes(UInt8,128,depth+1,cld(precision,8)+64)
    _enforce_payload_limit(payload,s.exact_bytes,"exact linear artwork query workspace","max_bytes")
    setprecision(BigFloat,precision) do
        a=(BigFloat(x)-BigFloat(s.start[1]),BigFloat(y)-BigFloat(s.start[2]))
        b=(BigFloat(x)-BigFloat(s.stop[1]),BigFloat(y)-BigFloat(s.stop[2]))
        _artwork_linear_hit(shape,a,b)
    end
end
function _artwork_linear_ordered_probe(s::_ArtworkCompositeLine,x,y,a,v)
    _artwork_linear_leaf(s.aperture)&&return Int8(_artwork_exact_linear_stroke(s.aperture,s,x,y))
    s.aperture isa _ArtworkComposite||return Int8(-1)
    parts=s.aperture.parts;whole=_artwork_interval_xy(a,v,_ArtworkInterval(0.,1.));unknown=false
    for i in eachindex(parts)
        dark,part=parts[i];dark||continue
        _artwork_certified_contains(part,whole...)==0&&continue
        if !_artwork_linear_leaf(part)
            unknown=true;continue
        end
        _artwork_exact_linear_stroke(part,s,x,y)||continue
        clear=false
        for j in i+1:length(parts)
            parts[j][1]&&continue
            _artwork_certified_contains(parts[j][2],whole...)==0||(clear=true;break)
        end
        clear||return Int8(1)
        unknown=true
    end
    unknown ? Int8(-1) : Int8(0)
end
