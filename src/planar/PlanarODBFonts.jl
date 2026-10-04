# Vector fonts and quoted T records from the ODB++ primary specification.
# Font coordinates use nominal metrics; T supplies an absolute stroke width.
struct _ODBFontLine
    start::NTuple{2,Float64}
    stop::NTuple{2,Float64}
    positive::Bool
    rounded::Bool
    width::Float64
end
struct _ODBFont
    xsize::Float64
    ysize::Float64
    offset::Float64
    glyphs::Dict{Char,Vector{_ODBFontLine}}
    reference_bounds::NTuple{4,Float64}
end
function _odb_font_number(s)
    occursin(r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)$",s) || throw(ArgumentError("ODB font/text numbers require fixed decimal notation"))
    value=parse(Float64,s);isfinite(value) || throw(ArgumentError("nonfinite ODB font/text number"));return value
end
function _odb_read_font(path;max_bytes,max_strokes)
    limit=_validated_resource_limit("max_bytes",max_bytes)
    isfile(path) || throw(ArgumentError("missing ODB font file"))
    payload=_checked_array_payload_bytes(UInt8,3,filesize(path))
    _enforce_payload_limit(payload,limit,"ODB font input","max_bytes")
    header=Dict{String,Float64}();glyphs=Dict{Char,Vector{_ODBFontLine}}();glyph=nothing;count=0
    open(path,"r") do io
        for raw in eachline(io)
            line=lstrip(raw);isempty(strip(line)) && continue;startswith(line,'#') && continue
            ncodeunits(line)<=1_000_000 || throw(ArgumentError("ODB font record too long"))
            if startswith(line,"UNITS=")
                strip(line[7:end]) in ("MM","INCH") || throw(ArgumentError("invalid ODB font units"))
                continue # Font coordinates/widths are normalized, irrespective of units.
            elseif startswith(line,"CHAR")
                glyph===nothing || throw(ArgumentError("nested ODB font character"))
                all(haskey(header,k) for k in ("XSIZE","YSIZE","OFFSET")) || throw(ArgumentError("missing ODB font metrics"))
                field=line[5:end];length(field)>1 && startswith(field,' ') && (field=field[2:end])
                length(field)==1 || throw(ArgumentError("ODB font CHAR must name one character"))
                glyph=only(field);haskey(glyphs,glyph) && throw(ArgumentError("duplicate ODB font character"))
                payload=_checked_payload_sum("ODB font",payload,192)
                _enforce_payload_limit(payload,limit,"ODB font characters","max_bytes")
                glyphs[glyph]=_ODBFontLine[];continue
            end
            tokens=split(line);key=first(tokens)
            if key in ("XSIZE","YSIZE","OFFSET")
                isempty(glyphs) && glyph===nothing && length(tokens)==2 || throw(ArgumentError("late/invalid ODB font metric"))
                haskey(header,key) && throw(ArgumentError("duplicate ODB font metric"))
                header[key]=_odb_font_number(tokens[2])
            elseif key=="LINE"
                glyph!==nothing && length(tokens)==8 || throw(ArgumentError("invalid ODB glyph line"))
                values=_odb_font_number.(tokens[[2,3,4,5,8]])
                values[5]>=0 && tokens[6] in ("P","N") && tokens[7] in ("R","S") || throw(ArgumentError("invalid ODB glyph line style/width"))
                count+=1;count<=max_strokes || throw(ArgumentError("ODB font exceeds max_objects"))
                payload=_checked_payload_sum("ODB font",payload,96)
                _enforce_payload_limit(payload,limit,"ODB font strokes","max_bytes")
                push!(glyphs[glyph],_ODBFontLine((values[1],values[2]),(values[3],values[4]),tokens[6]=="P",tokens[7]=="R",values[5]))
            elseif key=="ECHAR"
                glyph!==nothing && length(tokens)==1 || throw(ArgumentError("unexpected ODB ECHAR"));glyph=nothing
            else
                throw(ArgumentError("unknown ODB font record $key"))
            end
        end
    end
    glyph===nothing || throw(ArgumentError("unfinished ODB font character"))
    all(haskey(header,k) for k in ("XSIZE","YSIZE","OFFSET")) || throw(ArgumentError("missing ODB font metrics"))
    xs,ys,offset=header["XSIZE"],header["YSIZE"],header["OFFSET"]
    xs>0 && ys>0 && isfinite(xs+offset) && xs+offset>0 || throw(ArgumentError("nonpositive ODB font metrics"))
    reference=nothing;widest=-Inf
    # Prefer M/W for equal widths so descenders/punctuation do not alter the
    # widest capital character's insertion baseline.
    order=sort!(collect(keys(glyphs));by=c->(c=='M' ? 0 : c=='W' ? 1 : 2,Int(c)))
    for char in order
        lines=glyphs[char];isempty(lines) && continue
        x0=minimum(min(l.start[1],l.stop[1]) for l in lines);x1=maximum(max(l.start[1],l.stop[1]) for l in lines)
        y0=minimum(min(l.start[2],l.stop[2]) for l in lines);y1=maximum(max(l.start[2],l.stop[2]) for l in lines)
        if x1-x0>widest;reference=(x0,x1,y0,y1);widest=x1-x0;end
    end
    reference===nothing && (reference=(0.,0.,0.,0.))
    return _ODBFont(xs,ys,offset,glyphs,reference)
end
function _odb_text_record(line)
    m=match(r"^T\s+(.*?)\s+'(.*)'\s+([01])(;.*)?$",line)
    m===nothing && throw(ArgumentError("invalid quoted ODB text record"))
    h=split(m[1]);length(h) in (8,9) || throw(ArgumentError("invalid ODB text fields"))
    orient=parse(Int,h[5]);0<=orient<=9 || throw(ArgumentError("invalid ODB text orientation"))
    dynamic=orient>=8;length(h)==(dynamic ? 9 : 8) || throw(ArgumentError("invalid ODB text angle field"))
    sizes=_odb_font_number.(h[(dynamic ? 7 : 6):end])
    sizes[1]>0 && sizes[2]>0 && sizes[3]>=0 || throw(ArgumentError("invalid ODB text dimensions"))
    h[4] in ("P","N") || throw(ArgumentError("invalid ODB text polarity"))
    return (;x=_odb_font_number(h[1]),y=_odb_font_number(h[2]),font=String(h[3]),
        positive=h[4]=="P",orientation=orient,angle=dynamic ? _odb_font_number(h[6]) : 0.,
        xsize=sizes[1],ysize=sizes[2],width_factor=sizes[3],text=String(m[2]),
        version=parse(Int,m[3]),suffix=something(m[4],""))
end
function _odb_dynamic_text(text,context,max_bytes)
    occursin(raw"$$",text) || return text
    context isa AbstractDict || throw(ArgumentError("ODB dynamic text requires explicit context"))
    pattern=r"\$\$([A-Z][A-Z0-9_-]*)";bytes=BigInt(ncodeunits(text))
    for m in eachmatch(pattern,text)
        key=String(m[1]);haskey(context,key) || throw(ArgumentError("unresolved ODB text variable $key"))
        value=context[key];value isa AbstractString || throw(ArgumentError("ODB text context values must be strings"))
        bytes+=ncodeunits(value)-ncodeunits(m.match)
    end
    _enforce_payload_limit(_checked_array_payload_bytes(UInt8,3,bytes),max_bytes,"ODB dynamic text expansion","max_bytes")
    result=replace(text,pattern=>(s->String(context[String(s[3:end])])) )
    occursin(raw"$$",result) && throw(ArgumentError("unresolved/recursive ODB dynamic text"))
    return result
end
function _odb_text_context(context,layer,x,y;max_bytes)
    context===nothing || context isa AbstractDict || throw(ArgumentError("ODB text_context must be a dictionary"))
    required=_checked_array_payload_bytes(UInt8,256,context===nothing ? 6 : length(context)+6)
    _enforce_payload_limit(required,max_bytes,"ODB text context","max_bytes")
    if context!==nothing
        for (key,value) in context
            key isa AbstractString && value isa AbstractString || throw(ArgumentError("ODB text context keys/values must be strings"))
            required=_checked_payload_sum("ODB text context",required,2ncodeunits(key),2ncodeunits(value))
            _enforce_payload_limit(required,max_bytes,"ODB text context","max_bytes")
        end
    end
    result=context===nothing ? Dict{String,String}() : Dict{String,String}(String(k)=>String(v) for (k,v) in context)
    result["LAYER"]=String(layer);result["X"]=string(x/.0254);result["Y"]=string(y/.0254)
    result["X_MM"]=string(x/1e-3);result["Y_MM"]=string(y/1e-3)
    return result
end
function _odb_text_shape(font::_ODBFont,record,unit,context;max_strokes,max_bytes)
    text=_odb_dynamic_text(record.text,context,max_bytes)
    count=BigInt(0)
    for char in text
        char==' ' && !haskey(font.glyphs,char) && continue
        haskey(font.glyphs,char) || throw(ArgumentError("missing ODB font glyph $(repr(char))"))
        count+=length(font.glyphs[char])
    end
    count<=max_strokes || throw(ArgumentError("ODB text expansion exceeds max_objects"))
    required=_checked_payload_sum("ODB text geometry",_checked_array_payload_bytes(UInt8,512,count),
        _checked_array_payload_bytes(UInt8,256,length(text)),_checked_array_payload_bytes(UInt8,3,ncodeunits(text)))
    _enforce_payload_limit(required,max_bytes,"ODB expanded text geometry","max_bytes")
    advance=record.xsize*unit;height=record.ysize*unit
    scaleX=advance/(font.xsize+font.offset);scaleY=height/font.ysize
    radius=record.width_factor*(12*.0254/1000)/2
    all(isfinite,(advance,height,scaleX,scaleY,radius)) && advance>0 && height>0 && scaleX>0 && scaleY>0 ||
        throw(ArgumentError("ODB text dimensions do not fit finite SI coordinates"))
    record.width_factor>0 && iszero(radius) && throw(ArgumentError("ODB text stroke width underflows"))
    parts=Tuple{Bool,_ArtworkShape}[]
    for (i,char) in enumerate(text)
        char==' ' && !haskey(font.glyphs,char) && continue
        strokes=Tuple{Bool,_ArtworkShape}[]
        for line in font.glyphs[char]
            a=(line.start[1]*scaleX+(i-1)*advance,line.start[2]*scaleY)
            b=(line.stop[1]*scaleX+(i-1)*advance,line.stop[2]*scaleY)
            all(isfinite,(a...,b...)) || throw(ArgumentError("ODB glyph coordinates overflow"))
            stroke=line.rounded || iszero(radius) ? _ArtworkRoundLine(a,b,radius) :
                _artwork_convex_hull([(q[1]+dx,q[2]+dy) for q in (a,b),dx in (-radius,radius),dy in (-radius,radius)])
            push!(strokes,(line.positive,stroke))
        end
        isempty(strokes) || push!(parts,(true,_ArtworkComposite(strokes)))
    end
    isempty(parts) && return _ArtworkEmpty()
    shape=_ArtworkComposite(parts);x0,_,y0,_=_artwork_bounds(shape)
    if record.version==1
        x0=font.reference_bounds[1]*scaleX-radius;y0=font.reference_bounds[3]*scaleY-radius
    end
    shifted=_artwork_transform(shape;origin=(-x0,-y0))
    return _odb_orientation(shifted,record.orientation,record.angle,(record.x*unit,record.y*unit))
end

# Private symbol templates defer context-sensitive text until the symbol is
# placed. They never reach returned rasterizable artwork unresolved.
struct _ODBDeferredText <: _ArtworkShape
    font::_ODBFont
    record::NamedTuple
    unit::Float64
end
_artwork_bounds(::_ODBDeferredText)=(0.,0.,0.,0.)
_artwork_contains(::_ODBDeferredText,x,y)=throw(ArgumentError("unresolved ODB symbol text template"))
_odb_has_deferred_text(::_ArtworkShape)=false
_odb_has_deferred_text(::_ODBDeferredText)=true
_odb_has_deferred_text(s::_ArtworkTransform)=_odb_has_deferred_text(s.shape)
_odb_has_deferred_text(s::_ArtworkComposite)=any(p->_odb_has_deferred_text(p[2]),s.parts)
function _odb_resolve_symbol_text(shape,context,layer,matrix,origin;max_strokes,max_bytes)
    _odb_has_deferred_text(shape) || return shape
    if shape isa _ODBDeferredText
        q=matrix*SVector(shape.record.x*shape.unit,shape.record.y*shape.unit)+origin
        ctx=_odb_text_context(context,layer,q[1],q[2];max_bytes)
        remaining=max_bytes-Base.summarysize(ctx)
        remaining>=0 || throw(ArgumentError("ODB symbol text context exceeds max_bytes"))
        return _odb_text_shape(shape.font,shape.record,shape.unit,ctx;max_strokes,max_bytes=remaining)
    elseif shape isa _ArtworkTransform
        child=_odb_resolve_symbol_text(shape.shape,context,layer,matrix*shape.matrix,matrix*SVector(shape.origin...)+origin;max_strokes,max_bytes)
        return _ArtworkTransform(child,shape.matrix,shape.inverse,shape.origin)
    else
        parts=Tuple{Bool,_ArtworkShape}[]
        remaining=max_bytes-_checked_array_payload_bytes(UInt8,64,length(shape.parts))
        remaining>=0 || throw(ArgumentError("ODB symbol text containers exceed max_bytes"))
        for (dark,child) in shape.parts
            resolved=_odb_resolve_symbol_text(child,context,layer,matrix,origin;max_strokes,max_bytes=remaining)
            _odb_has_deferred_text(child) && (remaining-=Base.summarysize(resolved))
            remaining>=0 || throw(ArgumentError("ODB symbol text exceeds max_bytes"))
            push!(parts,(dark,resolved))
        end
        return _ArtworkComposite(parts)
    end
end
