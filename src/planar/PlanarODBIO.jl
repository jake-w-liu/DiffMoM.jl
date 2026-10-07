# Siemens ODB++Design Format Specification 8.1 Update 3.
# Strict geometric import; unsupported features cannot disappear silently.
export read_odb_features, read_odb
import JSON: JSON

struct _ArtworkCornerRectangle <: _ArtworkShape
    width::Float64
    height::Float64
    radius::Float64
    corners::NTuple{4,Bool}
    chamfer::Bool
end

struct _ArtworkODBButterfly{S<:_ArtworkShape} <: _ArtworkShape
    outside::S
end
_artwork_bounds(s::_ArtworkODBButterfly)=_artwork_bounds(s.outside)
_artwork_contains(s::_ArtworkODBButterfly,x,y)=
    ((x<=0 && y>=0) || (x>=0 && y<=0)) && _artwork_contains(s.outside,x,y)

struct _ArtworkODBDrill <: _ArtworkShape
    circle::_ArtworkCircle
    plating::Symbol
    positive_tolerance_m::Float64
    negative_tolerance_m::Float64
end
struct _ArtworkODBNull <: _ArtworkShape
    extension::Int
end
_artwork_bounds(::_ArtworkODBNull)=(0.,0.,0.,0.)
_artwork_contains(::_ArtworkODBNull,x,y)=false
_artwork_bounds(s::_ArtworkODBDrill)=_artwork_bounds(s.circle)
_artwork_contains(s::_ArtworkODBDrill,x,y)=_artwork_contains(s.circle,x,y)
function _odb_symbol_properties(s::_ArtworkShape)
    s isa _ArtworkTransform && return _odb_symbol_properties(s.shape)
    s isa _ArtworkODBNull && return Dict("ODB.null.extension"=>[string(s.extension)])
    s isa _ArtworkODBDrill || return Dict{String,Vector{String}}()
    return Dict("ODB.drill.plating"=>[String(s.plating)],
        "ODB.drill.diameter_m"=>[string(2s.circle.radius)],
        "ODB.drill.positive_tolerance_m"=>[string(s.positive_tolerance_m)],
        "ODB.drill.negative_tolerance_m"=>[string(s.negative_tolerance_m)])
end
_artwork_bounds(s::_ArtworkCornerRectangle)=(-s.width/2,s.width/2,-s.height/2,s.height/2)
function _artwork_contains(s::_ArtworkCornerRectangle,x,y)
    ax,ay=abs(x),abs(y);ax<=s.width/2 && ay<=s.height/2 || return false
    corner=x>=0 ? (y>=0 ? 1 : 4) : (y>=0 ? 2 : 3)
    s.corners[corner] && ax>s.width/2-s.radius && ay>s.height/2-s.radius || return true
    return s.chamfer ? s.width/2-ax+s.height/2-ay>=s.radius :
        hypot(ax-(s.width/2-s.radius),ay-(s.height/2-s.radius))<=s.radius
end
function _odb_corner_rectangle(w,h,r;corners="1234",chamfer=false)
    w>0 && h>0 && 0<=r<=min(w,h)/2 || throw(ArgumentError("invalid ODB rectangle dimensions"))
    all(c->c in "1234",corners) && length(unique(corners))==length(corners) || throw(ArgumentError("invalid ODB corner list"))
    return _ArtworkCornerRectangle(w,h,r,ntuple(i->Char(Int('0')+i) in corners,4),chamfer)
end
function _artwork_stroke_next_boundary(s::_ArtworkCornerRectangle,a,b,left,right)
    w,h,r=s.width/2,s.height/2,s.radius
    for (c,d) in (((-w,-h),(w,-h)),((w,-h),(w,h)),((w,h),(-w,h)),((-w,h),(-w,-h)))
        right=_artwork_stroke_edge_next(c,d,a,b,left,right)
    end
    if r>0
        for (i,(sx,sy)) in enumerate(((1.,1.),(-1.,1.),(-1.,-1.),(1.,-1.)))
            s.corners[i] || continue
            if s.chamfer
                right=_artwork_stroke_edge_next((sx*(w-r),sy*h),(sx*w,sy*(h-r)),a,b,left,right)
            else
                right=_artwork_stroke_circle_next((sx*(w-r),sy*(h-r)),r,a,b,left,right)
            end
        end
    end
    right
end
function _artwork_stroke_boundaries(s::_ArtworkCornerRectangle,limit,depth)
    all(isfinite,(s.width,s.height,s.radius)) && s.width>0 && s.height>0 &&
        0<=s.radius<=min(s.width,s.height)/2 || throw(ArgumentError("invalid ODB rectangle aperture"))
    _artwork_stroke_count(8+(s.radius>0 ? 2count(identity,s.corners) : 0),limit)
end
_artwork_stroke_next_boundary(::_ArtworkODBNull,a,b,left,right)=right
_artwork_stroke_boundaries(::_ArtworkODBNull,limit,depth)=_artwork_stroke_count(1,limit)
_artwork_stroke_next_boundary(s::_ArtworkODBDrill,a,b,left,right)=_artwork_stroke_next_boundary(s.circle,a,b,left,right)
_artwork_stroke_boundaries(s::_ArtworkODBDrill,limit,depth)=_artwork_stroke_boundaries(s.circle,limit,depth)
function _artwork_stroke_next_boundary(s::_ArtworkODBButterfly,a,b,left,right)
    right=_artwork_stroke_next_boundary(s.outside,a,b,left,right)
    right=_artwork_stroke_edge_next((0.,0.),(1.,0.),a,b,left,right)
    _artwork_stroke_edge_next((0.,0.),(0.,1.),a,b,left,right)
end
function _artwork_stroke_boundaries(s::_ArtworkODBButterfly,limit,depth)
    depth<64 || throw(ArgumentError("artwork aperture nesting exceeds 64 levels"))
    _artwork_stroke_count(4+_artwork_stroke_boundaries(s.outside,limit,depth+1),limit)
end
function _odb_standard_symbol(name::AbstractString,unit)
    # Rotation suffix is clockwise, unlike pad/step geometric orientation.
    rotation=match(r"^(.*)_([+\-]?\d+(?:\.\d+)?)$",name)
    if rotation!==nothing
        shape=_odb_standard_symbol(rotation[1],unit)
        shape===nothing || return _artwork_transform(shape;rotation=-parse(Float64,rotation[2]))
    end
    # The older primary v7.0 symbol table p215 specifies null<ext>; the
    # Update 3 table preserves the placeholder but omits its name grammar.
    null=match(r"^null([+\-]?\d+)$",name)
    if null!==nothing
        extension=tryparse(Int,null[1]);extension===nothing && throw(ArgumentError("ODB null extension overflows Int"))
        return _ArtworkODBNull(extension)
    end
    number="([+\\-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+))"
    m=match(Regex("^(r|s)"*number*"\$"),name)
    if m!==nothing
        d=parse(Float64,m[2])*unit;d>=0 || throw(ArgumentError("negative ODB symbol size"))
        return m[1]=="r" ? _ArtworkCircle((0.,0.),d/2) : d==0 ? _ArtworkEmpty() : _artwork_rectangle(d,d)
    end
    m=match(Regex("^(bfr|bfs)"*number*"\$"),name)
    if m!==nothing
        d=parse(Float64,m[2])*unit
        isfinite(d) && d>=0 || throw(ArgumentError("invalid ODB butterfly size"))
        outside=m[1]=="bfr" ? _ArtworkCircle((0.,0.),d/2) :
            d==0 ? _ArtworkEmpty() : _artwork_rectangle(d,d)
        return _ArtworkODBButterfly(outside)
    end
    m=match(Regex("^el"*number*"x"*number*"\$"),name)
    if m!==nothing
        w,h=parse.(Float64,m.captures).*unit
        all(isfinite,(w,h)) && w>0 && h>0 || throw(ArgumentError("invalid ODB ellipse axes"))
        return _artwork_affine(_ArtworkCircle((0.,0.),.5),w,h,(0.,0.))
    end
    m=match(Regex("^hole"*number*"x([pnv])x"*number*"x"*number*"\$"),name)
    if m!==nothing
        d,tp,tm=parse.(Float64,(m[1],m[3],m[4])).*unit
        all(isfinite,(d,tp,tm)) && d>=0 && tp>=0 && tm>=0 ||
            throw(ArgumentError("invalid ODB drill dimensions/tolerances"))
        plating=m[2]=="p" ? :plated : m[2]=="n" ? :nonplated : :via
        return _ArtworkODBDrill(_ArtworkCircle((0.,0.),d/2),plating,tp,tm)
    end
    m=match(Regex("^rect"*number*"x"*number*"(?:x([rc])"*number*"(?:x([1-4]+))?)?\$"),name)
    if m!==nothing
        w,h=parse(Float64,m[1])*unit,parse(Float64,m[2])*unit
        radius=m[4]===nothing ? 0. : parse(Float64,m[4])*unit
        return _odb_corner_rectangle(w,h,radius;corners=something(m[5],"1234"),chamfer=m[3]=="c")
    end
    m=match(Regex("^(oval|di|tri)"*number*"x"*number*"\$"),name)
    if m!==nothing
        w,h=parse(Float64,m[2])*unit,parse(Float64,m[3])*unit;w>0 && h>0 || throw(ArgumentError("invalid ODB symbol dimensions"))
        return m[1]=="oval" ? (w>=h ? _ArtworkRoundLine((-(w-h)/2,0.),((w-h)/2,0.),h/2) :
            _ArtworkRoundLine((0.,-(h-w)/2),(0.,(h-w)/2),w/2)) :
            m[1]=="di" ? _artwork_polygon([(0.,h/2),(w/2,0.),(0.,-h/2),(-w/2,0.)]) :
                _artwork_polygon([(-w/2,-h/2),(w/2,-h/2),(0.,h/2)])
    end
    m=match(Regex("^(oct|hex_l|hex_s)"*number*"x"*number*"x"*number*"\$"),name)
    if m!==nothing
        w,h,r=parse.(Float64,m.captures[2:4]).*unit
        w>0 && h>0 && 0<=r<=min(w,h)/2 || throw(ArgumentError("invalid ODB polygon dimensions"))
        vertices=m[1]=="oct" ? [(w/2-r,h/2),(-w/2+r,h/2),(-w/2,h/2-r),(-w/2,-h/2+r),
            (-w/2+r,-h/2),(w/2-r,-h/2),(w/2,-h/2+r),(w/2,h/2-r)] :
            m[1]=="hex_l" ? [(-w/2,0.),(-w/2+r,-h/2),(w/2-r,-h/2),(w/2,0.),(w/2-r,h/2),(-w/2+r,h/2)] :
                [(0.,-h/2),(w/2,-h/2+r),(w/2,h/2-r),(0.,h/2),(-w/2,h/2-r),(-w/2,-h/2+r)]
        return _artwork_polygon(vertices)
    end
    m=match(Regex("^donut_(r|s|sr)"*number*"x"*number*"\$"),name)
    if m!==nothing
        outer,inner=parse(Float64,m[2])*unit,parse(Float64,m[3])*unit
        outer>inner>=0 || throw(ArgumentError("invalid ODB donut dimensions"))
        outside=m[1]=="r" ? _ArtworkCircle((0.,0.),outer/2) : _artwork_rectangle(outer,outer)
        inside=m[1]=="s" ? _artwork_rectangle(inner,inner) : _ArtworkCircle((0.,0.),inner/2)
        return _ArtworkComposite(Tuple{Bool,_ArtworkShape}[(true,outside),(false,inside)])
    end
    m=match(Regex("^donut_(rc|o)"*number*"x"*number*"x"*number*"\$"),name)
    if m!==nothing
        w,h,t=parse.(Float64,m.captures[2:4]).*unit
        w>0 && h>0 && 0<t<min(w,h)/2 || throw(ArgumentError("invalid ODB rectangular donut"))
        if m[1]=="rc"
            outside=_artwork_rectangle(w,h);inside=_artwork_rectangle(w-2t,h-2t)
        else
            outside=_odb_standard_symbol("oval$(w/unit)x$(h/unit)",unit)
            inside=_odb_standard_symbol("oval$((w-2t)/unit)x$((h-2t)/unit)",unit)
        end
        return _ArtworkComposite(Tuple{Bool,_ArtworkShape}[(true,outside),(false,inside)])
    end
    return _odb_extra_standard_symbol(name,unit)
end

function _odb_orientation(shape,orientation,angle,origin)
    0<=orientation<=9 || throw(ArgumentError("invalid ODB orientation"))
    rotation=orientation<8 ? 90mod(orientation,4) : angle
    isfinite(rotation) || throw(ArgumentError("nonfinite ODB orientation angle"))
    # ODB positive angles are clockwise. Its named x-axis mirror negates
    # x coordinates (section "Rotation and Mirroring"). Rotation precedes
    # mirroring; artwork mirrors first, so reverse the rotation on reflection.
    mirror=orientation in (4,5,6,7,9)
    return _artwork_transform(shape;origin,rotation=mirror ? rotation : -rotation,
        mirror=mirror ? (-1.,1.) : (1.,1.))
end
function _odb_attributes(parts,names,texts)
    result=Dict{String,Vector{String}}()
    for (i,part) in enumerate(parts)
        isempty(part) && continue
        if startswith(part,"ID=")
            result["ID"]=[part[4:end]];continue
        end
        i==1 || throw(ArgumentError("unexpected ODB feature suffix"))
        for field in split(part,',')
            p=split(field,'=';limit=2);index=parse(Int,p[1])
            haskey(names,index) || throw(ArgumentError("undefined ODB feature attribute"))
            value=length(p)==1 ? "true" : String(p[2])
            # Retain exact encoded value and text lookup independently:
            # numeric OPTION/INTEGER attributes must not become text.
            result[names[index]]=[value]
            text=tryparse(Int,value)
            text!==nothing && haskey(texts,text) && (result[names[index]*".text_lookup"]=[texts[text]])
        end
    end
    return result
end

function _odb_feature_source(path,max_bytes)
    open(path,"r") do io
        size=filesize(io)
        payload=_checked_array_payload_bytes(UInt8,3,size)
        _enforce_payload_limit(payload,max_bytes,"ODB feature input","max_bytes")
        bytes=read(io,size)
        length(bytes)==size && eof(io) || throw(ArgumentError("ODB feature input changed while snapshotting"))
        isvalid(String,bytes) || throw(ArgumentError("ODB feature input must be valid UTF-8"))
        return bytes,payload
    end
end

# Count a contour's slots before allocating its fixed storage. The owned
# source has already passed UTF-8 validation. Decode only leading whitespace
# so Unicode whitespace follows the same strip/split contract as the parser.
@inline function _odb_source_character(source,i)
    b=source[i]
    b<0x80 && return Char(b),i+1
    if b<0xe0
        value=(UInt32(b&0x1f)<<6)|UInt32(source[i+1]&0x3f)
        return Char(value),i+2
    elseif b<0xf0
        value=(UInt32(b&0x0f)<<12)|(UInt32(source[i+1]&0x3f)<<6)|UInt32(source[i+2]&0x3f)
        return Char(value),i+3
    end
    value=(UInt32(b&0x07)<<18)|(UInt32(source[i+1]&0x3f)<<12)|
        (UInt32(source[i+2]&0x3f)<<6)|UInt32(source[i+3]&0x3f)
    return Char(value),i+4
end

function _odb_contour_segment_count(source,start,max_objects)
    count=0;i=start+1;n=length(source)
    while i<=n
        ending=findnext(==(0x0a),source,i)
        stop=ending===nothing ? n+1 : ending
        j=i
        while j<stop
            c,next=_odb_source_character(source,j)
            isspace(c) || break
            j=next
        end
        if j+1<stop && source[j]==0x4f
            c=source[j+1]
            k=j+2
            token_end=k==stop || source[k]==0x3b || isspace(first(_odb_source_character(source,k)))
            if token_end
                c==0x45 && return count
                if c in (0x53,0x43)
                    count+=1
                    count<=max_objects || throw(ArgumentError("ODB contour exceeds max_objects"))
                end
            end
        end
        i=stop+1
    end
    throw(ArgumentError("unfinished ODB contour"))
end

include("PlanarODBFonts.jl")

"""Read an ODB++ `features` file into ordered SI artwork. Circular and
square strokes, pad orientation, standard/user symbols through
`symbol_resolver(name)`, analytic arc contours, island/hole surfaces,
feature attributes and per-symbol units are preserved. `UNITS=INCH`/`UNITS=MM`
and legacy `U INCH`/`U MM` declarations share the same ordering guards.
Vector-font T records
load from `font_directory`; enclosing products resolve their own fonts.
`text_context` supplies explicit dynamic variables; layer and SI placement
coordinates populate LAYER/X/Y/X_MM/Y_MM. `default_unit_m` implements the
enclosing product's unit fallback. Unknown symbols, barcodes and unsupported
resize records reject explicitly. Input is snapshotted before symbol/font
callbacks; later changes to the source file do not alter that read.
Composite symbol lines preserve local ordered holes; `max_stroke_boundaries`
bounds their primitive work. Unsupported analytic leaves reject explicitly.
All resources and file handles are bounded/closed on failure."""
function read_odb_features(path::AbstractString;layer::AbstractString=basename(dirname(path)),
        default_unit_m::Real=.0254,symbol_resolver=nothing,font_directory=nothing,
        text_context=nothing,max_objects::Integer=1_000_000,max_stroke_boundaries::Integer=256,
        max_bytes::Integer=_default_max_dense_payload_bytes(),
        _font_resolver=nothing,_defer_text::Bool=false,_payload_observer=nothing,_remaining_budget=nothing)
    limit=_validated_resource_limit("max_bytes",max_bytes)
    entity=_odb_entity(path)
    if endswith(entity,".gz") || endswith(entity,".Z")
        return _with_odb_entity(path,limit) do expanded
            doc=read_odb_features(expanded;layer,default_unit_m,symbol_resolver,font_directory,text_context,
                max_objects,max_stroke_boundaries,max_bytes,_font_resolver,_defer_text,_payload_observer,_remaining_budget)
            PlanarArtwork(abspath(path),doc.objects,doc.attributes,doc.negative_layers,doc.coordinate_unit_m)
        end
    end
    unit=Float64(default_unit_m);unit in (.0254,1e-3) || throw(ArgumentError("ODB default units must be inch or mm"))
    max_objects>0 || throw(ArgumentError("max_objects must be positive"))
    _validated_resource_limit("max_stroke_boundaries",max_stroke_boundaries)
    available()=_remaining_budget===nothing ? limit : min(limit,_remaining_budget())
    source,payload=_odb_feature_source(entity,available())
    report()=_payload_observer===nothing ? nothing : _payload_observer(payload)
    report();_enforce_payload_limit(payload,available(),"ODB feature input and active readers","max_bytes")
    objects=PlanarArtworkObject[];symbols=Dict{Int,_ArtworkShape}();names=Dict{Int,String}();texts=Dict{Int,String}()
    fonts=Dict{String,_ODBFont}()
    metadata=Dict{String,Vector{String}}();surface=false;contour=false;surface_dark=true;island=true
    surfaceparts=Tuple{Bool,_ArtworkShape}[];segments=Union{_ArtworkRoundLine,_ArtworkRoundArc}[];segment_used=0
    attrs=Dict{String,Vector{String}}();start=(0.,0.);position=(0.,0.);records=0;declared=nothing;units_seen=false
    function reserve(bytes,label)
        required=_checked_payload_sum(label,payload,bytes)
        _enforce_payload_limit(required,available(),label,"max_bytes")
        payload=required;report()
    end
    function emit(shape,dark,attributes)
        reserve(_checked_payload_sum("ODB feature payload",256,Base.summarysize(shape),
            Base.summarysize(attributes)),"ODB feature geometry and attributes")
        _artwork_object!(objects,layer,shape,dark,attributes,max_objects)
    end
    function get_font(name)
        name=_odb_name(name);haskey(fonts,name) && return fonts[name]
        font=if _font_resolver!==nothing
            _enforce_payload_limit(_checked_payload_sum("ODB font reference",payload,256),available(),"ODB font reference","max_bytes")
            _font_resolver(name,payload)
        else
            font_directory===nothing && throw(ArgumentError("ODB T records require font_directory or an enclosing product font"))
            isdir(font_directory) || throw(ArgumentError("missing ODB font directory"))
            directory=realpath(font_directory);file=_odb_path(directory,name)
            # Primary font entities are uncompressed; archives may transport
            # that ordinary file without changing its internal format.
            isfile(file) || throw(ArgumentError("missing uncompressed ODB font $name"))
            loaded=_odb_read_font(file;max_bytes=available()-payload,max_strokes=max_objects)
            payload=_checked_payload_sum("ODB fonts",payload,Base.summarysize(loaded),192)
            _enforce_payload_limit(payload,available(),"ODB retained font","max_bytes");report()
            loaded
        end
        font isa _ODBFont || throw(ArgumentError("invalid ODB font resolver result"))
        fonts[name]=font;return font
    end
    scalar(s)=begin
        occursin(r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)$",s) || throw(ArgumentError("ODB numeric fields require fixed decimal notation"))
        x=parse(Float64,s);isfinite(x) || throw(ArgumentError("nonfinite ODB scalar"));x
    end
    dcode(s)=begin
        value=tryparse(Int,s);value!==nothing && value>=0 || throw(ArgumentError("invalid ODB dcode"));String(s)
    end
    point(x,y)=(scalar(x)*unit,scalar(y)*unit)
    positive(s)=s in ("P","N") ? s=="P" : throw(ArgumentError("invalid ODB feature polarity"))
    get_symbol(i)=haskey(symbols,i) ? symbols[i] : throw(ArgumentError("undefined ODB feature symbol"))
    let io=IOBuffer(source;read=true,write=false)
        for (line_number,raw) in enumerate(eachline(io))
            line=strip(raw);isempty(line) || startswith(line,'#') || begin
                length(line)<=1_000_000 || throw(ArgumentError("ODB record too long"))
                if startswith(line,"UNITS=") || line=="U" || startswith(line,"U ") || startswith(line,"U\t")
                    !units_seen && records==0 && isempty(symbols) || throw(ArgumentError("late/repeated ODB units declaration"))
                    legacy=!startswith(line,"UNITS=")
                    value=if legacy
                        fields=split(line)
                        length(fields)==2 || throw(ArgumentError("invalid legacy ODB U units record"))
                        fields[2]
                    else
                        line[7:end]
                    end
                    value in ("MM","INCH") || throw(ArgumentError("unknown ODB units"))
                    # Retained syntax/value vectors and dictionary entries
                    # belong to the same aggregate reader budget as geometry.
                    payload=_checked_payload_sum("ODB units metadata",payload,1024)
                    _enforce_payload_limit(payload,available(),"ODB units metadata","max_bytes");report()
                    metadata["ODB.units.syntax"]=[legacy ? "U" : "UNITS="]
                    metadata["ODB.units.value"]=[String(value)]
                    unit=value=="MM" ? 1e-3 : .0254;units_seen=true;continue
                elseif startswith(line,"ID=")
                    reserve(_checked_payload_sum("ODB ID metadata",192,2ncodeunits(line)),"ODB ID metadata")
                    metadata["ID"]=[line[4:end]];continue
                elseif line[1] in ('\$','@','&')
                    p=split(line;limit=2);length(p)==2 || throw(ArgumentError("invalid ODB lookup record"))
                    index=parse(Int,p[1][2:end]);index>=0 || throw(ArgumentError("negative ODB lookup index"))
                    if line[1]=='\$'
                        # Unused definitions remain live in the lookup table.
                        # Reserve its entry before a possible user callback,
                        # then its returned geometry before retaining it.
                        reserve(256,"ODB symbol lookup")
                        definition=split(p[2]);length(definition) in (1,2) || throw(ArgumentError("invalid ODB symbol record"))
                        symbolunit=length(definition)==1 ? unit/1000 : definition[2]=="M" ? 1e-6 : definition[2]=="I" ? .0254/1000 :
                            throw(ArgumentError("invalid ODB per-symbol unit"))
                        shape=_odb_standard_symbol(definition[1],symbolunit)
                        caller_geometry=shape===nothing
                        if shape===nothing
                            symbol_resolver===nothing && throw(ArgumentError("unsupported ODB symbol $(definition[1])"))
                            shape=symbol_resolver(String(definition[1]));shape isa _ArtworkShape || throw(ArgumentError("ODB symbol resolver must return artwork geometry"))
                        end
                        reserve(Base.summarysize(shape),"ODB symbol lookup geometry")
                        if caller_geometry
                            _artwork_geometry_depth_guard(shape)
                            shape=deepcopy(shape)
                        end
                        symbols[index]=shape
                    elseif line[1]=='@'
                        reserve(_checked_payload_sum("ODB attribute lookup",192,2ncodeunits(p[2])),"ODB attribute lookup")
                        names[index]=String(p[2])
                    else
                        reserve(_checked_payload_sum("ODB text lookup",192,2ncodeunits(p[2])),"ODB text lookup")
                        texts[index]=String(p[2])
                    end
                    continue
                end
                record=startswith(line,"T ") || startswith(line,"T\t") ? _odb_text_record(line) : nothing
                parts=record===nothing ? split(line,';') : split(record.suffix,';')
                p=record===nothing ? split(parts[1]) : ["T"];kind=p[1]
                attributes=_odb_attributes(parts[2:end],names,texts)
                if kind=="F"
                    length(p)==2 || throw(ArgumentError("invalid ODB feature count"));declared=parse(Int,p[2])
                    declared>=0 || throw(ArgumentError("negative ODB feature count"));continue
                elseif kind=="S"
                    !surface && length(p)==3 || throw(ArgumentError("invalid/nested ODB surface"))
                    # Contours remain live before SE emits the surface. Their
                    # scratch and retained storage must already fit the same
                    # aggregate budget as completed objects and lookup tables.
                    reserve(256,"ODB surface metadata and contour lookup")
                    surface=true;surface_dark=positive(p[2]);attrs=attributes;attrs["dcode"]=[dcode(p[3])];empty!(surfaceparts);records+=1
                elseif kind=="OB"
                    surface && !contour && length(p)==4 || throw(ArgumentError("invalid ODB contour start"))
                    p[4] in ("I","H") || throw(ArgumentError("invalid ODB contour type"));island=p[4]=="I"
                    reserve(256,"ODB retained contour and surface entry")
                    count=_odb_contour_segment_count(source,Base.position(io),max_objects)
                    # Union arrays retain one tag byte per slot. Allocate the
                    # exact segment count once and transfer it into the region
                    # at OE, without a push! capacity buffer or geometry copy.
                    reserve(_checked_payload_sum("ODB contour segment storage",
                        _checked_array_payload_bytes(eltype(segments),1,count),count),"ODB contour segment storage")
                    segments=Vector{eltype(segments)}(undef,count);segment_used=0
                    contour=true;start=point(p[2],p[3]);position=start
                elseif kind in ("OS","OC")
                    contour && length(p)==(kind=="OS" ? 3 : 6) || throw(ArgumentError("invalid ODB contour segment"))
                    segment_used<length(segments) || throw(ArgumentError("ODB contour segment count mismatch"))
                    stop=point(p[2],p[3]);segment=if kind=="OS"
                        _ArtworkRoundLine(position,stop,0.)
                    else
                        center=point(p[4],p[5]);p[6] in ("Y","N") || throw(ArgumentError("invalid ODB arc direction"))
                        _gerber_arc(position,stop,center[1]-position[1],center[2]-position[2],p[6]=="Y",false,1e-9*unit)
                    end
                    segment_used+=1;segments[segment_used]=segment;position=stop
                elseif kind=="OE"
                    length(p)==1 && contour && segment_used==length(segments) && !isempty(segments) && position==start || throw(ArgumentError("unclosed ODB contour"))
                    push!(surfaceparts,(island,_ArtworkRegion(segments)));contour=false
                elseif kind=="SE"
                    length(p)==1 && surface && !contour && any(first,surfaceparts) || throw(ArgumentError("invalid ODB surface end"))
                    # ODB's natural containment order alternates islands and
                    # holes: a hole must precede an island contained inside it.
                    # Keep that order so nested conductive islands survive;
                    # aperture-local holes remain transparent to older artwork.
                    emit(_ArtworkComposite(copy(surfaceparts)),surface_dark,attrs);surface=false
                elseif kind=="P"
                    !surface && length(p)>=7 || throw(ArgumentError("invalid ODB pad"))
                    p[4]=="-1" && throw(ArgumentError("ODB dimensional pad resize is not implemented"))
                    shape=get_symbol(parse(Int,p[4]));orientation=parse(Int,p[7]);expected=orientation>=8 ? 8 : 7
                    length(p)==expected || throw(ArgumentError("invalid ODB pad orientation fields"))
                    angle=expected==8 ? scalar(p[8]) : 0.;attributes["dcode"]=[dcode(p[6])]
                    merge!(attributes,_odb_symbol_properties(shape))
                    if !_defer_text && _odb_has_deferred_text(shape)
                        transform=_odb_orientation(_ArtworkEmpty(),orientation,angle,point(p[2],p[3]))
                        shape=_odb_resolve_symbol_text(shape,text_context,layer,transform.matrix,SVector(transform.origin...);
                            max_strokes=max_objects,max_bytes=available()-payload)
                    end
                    emit(_odb_orientation(shape,orientation,angle,point(p[2],p[3])),positive(p[5]),attributes);records+=1
                elseif kind=="T"
                    !surface || throw(ArgumentError("ODB text cannot be inside a surface record"))
                    font=get_font(record.font)
                    shape=if _defer_text && occursin(raw"$$",record.text)
                        _ODBDeferredText(font,record,unit)
                    else
                        context=_odb_text_context(text_context,layer,record.x*unit,record.y*unit;max_bytes=available()-payload)
                        _odb_text_shape(font,record,unit,context;max_strokes=max_objects,
                            max_bytes=available()-payload-Base.summarysize(context))
                    end
                    attributes["ODB.text.font"]=[record.font];attributes["ODB.text.source"]=[record.text]
                    attributes["ODB.text.version"]=[string(record.version)]
                    attributes["ODB.text.stroke_width_m"]=[string(record.width_factor*(12*.0254/1000))]
                    emit(shape,record.positive,attributes);records+=1
                elseif kind in ("L","A")
                    !surface && length(p)==(kind=="L" ? 8 : 11) || throw(ArgumentError("invalid ODB stroke"))
                    a,b=point(p[2],p[3]),point(p[4],p[5]);index=kind=="L" ? 6 : 8
                    shape=get_symbol(parse(Int,p[index]));attributes["dcode"]=[dcode(p[index+2])]
                    stroke=if shape isa _ArtworkCircle
                        all(isfinite,(shape.center...,shape.radius)) && shape.radius>=0 ||
                            throw(ArgumentError("invalid ODB circle aperture geometry"))
                        shifted_a=(a[1]+shape.center[1],a[2]+shape.center[2])
                        shifted_b=(b[1]+shape.center[1],b[2]+shape.center[2])
                        all(isfinite,(shifted_a...,shifted_b...)) ||
                            throw(ArgumentError("nonfinite ODB circle stroke geometry"))
                        if kind=="L";_ArtworkRoundLine(shifted_a,shifted_b,shape.radius)
                        else
                            center=point(p[6],p[7]);p[11] in ("Y","N") || throw(ArgumentError("invalid ODB arc direction"))
                            arc=_gerber_arc(a,b,center[1]-a[1],center[2]-a[2],p[11]=="Y",false,1e-9*unit)
                            shifted_center=(arc.center[1]+shape.center[1],arc.center[2]+shape.center[2])
                            all(isfinite,shifted_center) || throw(ArgumentError("nonfinite ODB arc stroke center"))
                            _ArtworkRoundArc(shifted_a,shifted_b,shifted_center,arc.clockwise,shape.radius)
                        end
                    elseif kind=="L" && shape isa _ArtworkPolygon
                        _ArtworkPolygonLine(shape,a,b)
                    elseif kind=="L"
                        _artwork_composite_line(shape,a,b;max_boundaries=max_stroke_boundaries,
                            max_bytes=available()-payload-256-Base.summarysize(attributes),_copy_aperture=false)
                    else
                        throw(ArgumentError("ODB arcs require a round symbol"))
                    end
                    emit(stroke,positive(p[index+1]),attributes);records+=1
                else
                    throw(ArgumentError("unsupported ODB feature $kind at line $line_number"))
                end
            end
        end
    end
    !surface && !contour || throw(ArgumentError("unfinished ODB surface"))
    declared===nothing || declared==records || throw(ArgumentError("ODB declared feature count does not match data"))
    return PlanarArtwork(abspath(path),objects,metadata,Set{String}(),unit)
end

function _odb_structured(path;max_bytes=_default_max_dense_payload_bytes())
    limit=_validated_resource_limit("max_bytes",max_bytes)
    entity=_odb_entity(path)
    if endswith(entity,".gz") || endswith(entity,".Z")
        return _with_odb_entity(path,limit) do expanded
            _odb_structured(expanded;max_bytes)
        end
    end
    payload=_checked_array_payload_bytes(UInt8,3,filesize(entity))
    _enforce_payload_limit(payload,limit,"ODB structured input","max_bytes")
    parameters=Dict{String,String}();blocks=Dict{String,Vector{Dict{String,String}}}();current=nothing
    open(path,"r") do io
        for raw in eachline(io)
            line=strip(raw);isempty(line) || startswith(line,'#') || begin
                if endswith(line,'{')
                    payload=_checked_payload_sum("ODB structured records",payload,256,2ncodeunits(line))
                    _enforce_payload_limit(payload,limit,"ODB structured records","max_bytes")
                    current===nothing || throw(ArgumentError("nested ODB structured record"))
                    key=strip(line[1:end-1]);isempty(key) && throw(ArgumentError("empty ODB structured record name"))
                    current=Dict{String,String}();push!(get!(blocks,key,Dict{String,String}[]),current)
                elseif line=="}"
                    current!==nothing || throw(ArgumentError("unmatched ODB structured terminator"));current=nothing
                else
                    p=split(line,'=';limit=2);length(p)==2 || throw(ArgumentError("invalid ODB structured field"))
                    key,value=String(strip(p[1])),String(strip(p[2]));isempty(key) && throw(ArgumentError("empty ODB field name"))
                    payload=_checked_payload_sum("ODB structured fields",payload,160,2ncodeunits(line))
                    _enforce_payload_limit(payload,limit,"ODB structured fields","max_bytes")
                    target=current===nothing ? parameters : current
                    haskey(target,key) && throw(ArgumentError("duplicate ODB field $key"));target[key]=value
                end
            end
        end
    end
    current===nothing || throw(ArgumentError("unfinished ODB structured record"))
    return (;parameters,blocks)
end
function _odb_name(value)
    1<=ncodeunits(value)<=64 && !endswith(value,'.') && occursin(r"^[a-z0-9_][a-z0-9_.+\-]*$",value) ||
        throw(ArgumentError("invalid ODB entity name"))
    return String(value)
end
function _odb_path(root,parts...)
    path=joinpath(root,parts...)
    for candidate in (path,path*".gz",path*".Z")
        ispath(candidate) || continue
        relative=relpath(realpath(candidate),root)
        !isabspath(relative) && (isempty(splitpath(relative)) || first(splitpath(relative))!="..") ||
            throw(ArgumentError("ODB entity resolves outside product directory"))
    end
    return path
end
include("PlanarODBLayers.jl")
struct _ArtworkIntersection <: _ArtworkShape
    left::_ArtworkShape
    right::_ArtworkShape
end
function _artwork_bounds(s::_ArtworkIntersection)
    a,b=_artwork_bounds(s.left),_artwork_bounds(s.right)
    return (max(a[1],b[1]),min(a[2],b[2]),max(a[3],b[3]),min(a[4],b[4]))
end
_artwork_contains(s::_ArtworkIntersection,x,y)=_artwork_contains(s.left,x,y) && _artwork_contains(s.right,x,y)

"""Read an ODB++ product directory or tar archive, retaining matrix/layer metadata,
user-symbol hierarchy, vector font text and step repetitions as analytic fabrication artwork.
Specify `step` when several steps are present and `layers` to select explicit
technology layers. Negative layers are inverted inside each step profile,
then transformed with that step, rather than over the whole parent box.
`text_context` supplies explicit dynamic text values, including dates/times.
Product/step/layer names and text placement coordinates supply JOB/STEP/LAYER
and X/Y/X_MM/Y_MM when known; dates never depend on ambient clock state.
FLIP reverses a complete symmetric BOARD buildup and mirrors coordinates.
Drill/rout spans map to their reflected layer; `flip_layer_map` resolves
ambiguous equal-span or auxiliary layers with a validated involution.
Layer selection applies to destination layers after each nested flip.
Matrix, repetition and selected layer/step names canonicalize uppercase
logical references locally; raw matrix records remain in result metadata.
Duplicate canonical names reject, and entity paths retain strict validation.
All entity paths stay inside the product directory; recursive references
and unsupported feature/symbol records reject explicitly.
Plain, gzip and UNIX-compress (`.Z`) entities and tar/tar.gz/tgz/tar.Z archives
are supported. Decompressed entities and extracted archives obey `max_bytes`;
`max_entities` bounds archive entries. Links, special files and duplicate or
escaping archive paths reject. Temporary decompression/extraction is scoped
to the call. Barcodes, dimensional resize and remaining unsupported symbol
families reject. Vendor font rendering compatibility requires separate validation."""
function read_odb(path::AbstractString;step=nothing,layers=nothing,
        max_objects::Integer=1_000_000,max_bytes::Integer=_default_max_dense_payload_bytes(),
        max_entities::Integer=100_000,max_stroke_boundaries::Integer=256,text_context=nothing,flip_layer_map=nothing)
    limit=_validated_resource_limit("max_bytes",max_bytes)
    max_objects>0 || throw(ArgumentError("max_objects must be positive"))
    max_entities>0 || throw(ArgumentError("max_entities must be positive"))
    _validated_resource_limit("max_stroke_boundaries",max_stroke_boundaries)
    flip_layer_map===nothing || flip_layer_map isa AbstractDict || throw(ArgumentError("flip_layer_map must be a dictionary"))
    if !isdir(path)
        return _with_odb_archive(path,limit,max_entities) do root
            doc=read_odb(root;step,layers,max_objects,max_bytes,max_entities,max_stroke_boundaries,text_context,flip_layer_map)
            PlanarArtwork(abspath(path),doc.objects,doc.attributes,doc.negative_layers,doc.coordinate_unit_m)
        end
    end
    root=realpath(path);limit=_validated_resource_limit("max_bytes",max_bytes);allocated=0
    prefixes=Ref{Int}[]
    active_payload()=sum((p[] for p in prefixes);init=0)
    remaining_payload()=limit-allocated-active_payload()
    function reserve_file(p)
        entity=_odb_entity(p)
        required=_checked_payload_sum("ODB product",allocated,active_payload(),_checked_array_payload_bytes(UInt8,3,filesize(entity)))
        _enforce_payload_limit(required,limit,"ODB product input and retained geometry","max_bytes")
    end
    function retain(value,label)
        allocated=_checked_payload_sum("ODB product retained payload",allocated,Base.summarysize(value),128)
        _enforce_payload_limit(_checked_payload_sum("ODB product active payload",allocated,active_payload()),limit,label,"max_bytes")
        return value
    end
    matrixpath=_odb_path(root,"matrix","matrix");reserve_file(matrixpath)
    matrix=retain(_odb_structured(matrixpath;max_bytes=remaining_payload()),"ODB retained matrix")
    info_path=_odb_path(root,"misc","info")
    info=_odb_has_entity(info_path) ? (reserve_file(info_path);retain(_odb_structured(info_path;max_bytes=remaining_payload()).parameters,
        "ODB retained product info")) : Dict{String,String}()
    units=get(info,"UNITS","INCH");units in ("MM","INCH") || throw(ArgumentError("invalid ODB product units"))
    default_unit=units=="MM" ? 1e-3 : .0254
    layer_records=retain(_odb_canonical_reference_records(
        _odb_ordered_matrix_records(get(matrix.blocks,"LAYER",Dict{String,String}[]),"ROW";max_bytes=remaining_payload());
        layers=true,max_bytes=remaining_payload()),"ODB canonical layer references")
    byname=Dict(_odb_name(get(record,"NAME",""))=>record for record in layer_records)
    length(byname)==length(layer_records) || throw(ArgumentError("duplicate ODB layer names"))
    selected=layers===nothing ? [record["NAME"] for record in layer_records] : _odb_reference_name.(String.(collect(layers)))
    all(n->haskey(byname,n),selected) || throw(ArgumentError("unknown selected ODB layer"))
    step_records=retain(_odb_canonical_reference_records(
        _odb_ordered_matrix_records(get(matrix.blocks,"STEP",Dict{String,String}[]),"COL";max_bytes=remaining_payload());
        max_bytes=remaining_payload()),"ODB canonical step references")
    available=String[_odb_name(get(r,"NAME","")) for r in step_records]
    length(unique(available))==length(available) || throw(ArgumentError("duplicate ODB step names"))
    allocated=_checked_payload_sum("ODB matrix lookup payload",allocated,
        _checked_array_payload_bytes(UInt8,256,length(layer_records)+length(step_records)))
    _enforce_payload_limit(allocated,limit,"ODB matrix lookups","max_bytes")
    if step===nothing
        length(available)==1 || throw(ArgumentError("select one ODB step explicitly"));step=only(available)
    end
    chosen=_odb_reference_name(step);chosen in available || throw(ArgumentError("unknown ODB step $chosen"))
    symbolcache=Dict{String,_ArtworkShape}();fontcache=Dict{String,_ODBFont}()
    active_symbols=Set{String}();active_steps=Set{String}()
    flipmap=nothing
    function complete_flip_map()
        if flipmap===nothing
            flipmap=_odb_flip_layer_map(layer_records;override=flip_layer_map,max_bytes=remaining_payload())
            retain(flipmap,"ODB retained complete FLIP map")
        end
        return flipmap
    end
    function product_font(name,payload)
        haskey(fontcache,name) && return fontcache[name]
        file=_odb_path(root,"fonts",name)
        isfile(file) || throw(ArgumentError("missing uncompressed ODB font $name"))
        reserve_file(file)
        font=_odb_read_font(file;max_bytes=remaining_payload(),max_strokes=max_objects)
        allocated=_checked_payload_sum("ODB retained fonts",allocated,Base.summarysize(font),192)
        _enforce_payload_limit(_checked_payload_sum("ODB font readers",allocated,active_payload()),limit,
            "ODB fonts and active feature readers","max_bytes")
        fontcache[name]=font;return font
    end
    function features(file;kwargs...)
        prefix=Ref(0);push!(prefixes,prefix)
        try
            return read_odb_features(file;default_unit_m=default_unit,max_objects,max_stroke_boundaries,
                max_bytes=remaining_payload(),_font_resolver=product_font,
                _payload_observer=(v->(prefix[]=v)),
                _remaining_budget=(() -> limit-allocated-active_payload()+prefix[]),kwargs...)
        finally
            pop!(prefixes)
        end
    end
    function custom_symbol(name)
        name=_odb_name(name);haskey(symbolcache,name) && return symbolcache[name]
        name in active_symbols && throw(ArgumentError("cyclic ODB user symbol reference"))
        length(active_symbols)<64 || throw(ArgumentError("ODB symbol hierarchy too deep"))
        push!(active_symbols,name)
        try
            file=_odb_path(root,"symbols",name,"features");reserve_file(file)
            doc=features(file;layer=name,symbol_resolver=custom_symbol,_defer_text=true)
            allocated=_checked_payload_sum("ODB product geometry",allocated,Base.summarysize(doc))
            _enforce_payload_limit(allocated,limit,"ODB retained user symbols","max_bytes")
            shape=_ArtworkComposite(Tuple{Bool,_ArtworkShape}[(o.dark,o.shape) for o in doc.objects]);symbolcache[name]=shape
            return shape
        finally
            delete!(active_symbols,name)
        end
    end
    function step_artwork(name,requested_layers=selected)
        name in active_steps && throw(ArgumentError("cyclic ODB step reference"))
        length(active_steps)<64 || throw(ArgumentError("ODB step hierarchy too deep"));push!(active_steps,name)
        try
            headerpath=_odb_path(root,"steps",name,"stephdr");reserve_file(headerpath)
            header=retain(_odb_structured(headerpath;max_bytes=remaining_payload()),"ODB retained recursive step headers")
            units=get(header.parameters,"UNITS",get(info,"UNITS","INCH"))
            units in ("MM","INCH") || throw(ArgumentError("invalid ODB step units"));unit=units=="MM" ? 1e-3 : .0254
            datum_values=[parse(Float64,get(header.parameters,k,"0"))*unit for k in ("X_DATUM","Y_DATUM")]
            all(isfinite,datum_values) || throw(ArgumentError("ODB step datum must be finite"))
            context=_odb_text_context(text_context,"",0.,0.;max_bytes=remaining_payload())
            allocated=_checked_payload_sum("ODB step text context",allocated,Base.summarysize(context))
            _enforce_payload_limit(_checked_payload_sum("ODB text context readers",allocated,active_payload()),limit,
                "ODB step text context and retained geometry","max_bytes")
            get!(context,"STEP",name)
            job=get(info,"PRODUCT_MODEL_NAME",get(info,"JOB_NAME",nothing))
            job===nothing || get!(context,"JOB",job)
            delete!(context,"LAYER") # Each layer/placed symbol supplies its own layer.
            objects=PlanarArtworkObject[];profile=nothing
            for layer in requested_layers
                featurepath=_odb_path(root,"steps",name,"layers",_odb_name(layer),"features")
                if !_odb_has_entity(featurepath)
                    isdir(dirname(featurepath)) && throw(ArgumentError("missing ODB features for layer $layer"))
                    continue
                end
                reserve_file(featurepath)
                doc=features(featurepath;layer,symbol_resolver=custom_symbol,text_context=context)
                allocated=_checked_payload_sum("ODB product geometry",allocated,Base.summarysize(doc))
                _enforce_payload_limit(allocated,limit,"ODB retained layer geometry","max_bytes")
                polarity=get(byname[layer],"POLARITY","POSITIVE")
                polarity in ("POSITIVE","NEGATIVE") || throw(ArgumentError("unknown ODB layer polarity"))
                if polarity=="POSITIVE"
                    length(objects)+length(doc.objects)<=max_objects || throw(ArgumentError("ODB product exceeds max_objects"))
                    _enforce_payload_limit(_checked_array_payload_bytes(UInt8,32,length(objects)+length(doc.objects)),
                        remaining_payload(),"ODB layer object references","max_bytes")
                    append!(objects,doc.objects)
                else
                    if profile===nothing
                        profilepath=_odb_path(root,"steps",name,"profile");reserve_file(profilepath)
                        profiledoc=features(profilepath;layer="profile",text_context=context)
                        length(profiledoc.objects)==1 || throw(ArgumentError("ODB step profile must contain one surface"))
                        retain(profiledoc,"ODB retained negative-layer profile")
                        profile=only(profiledoc.objects).shape
                    end
                    _enforce_payload_limit(_checked_array_payload_bytes(UInt8,64,length(doc.objects)+1),remaining_payload(),
                        "ODB negative-layer composition","max_bytes")
                    parts=Tuple{Bool,_ArtworkShape}[(true,profile)]
                    append!(parts,Tuple{Bool,_ArtworkShape}[(!o.dark,o.shape) for o in doc.objects])
                    shape=_ArtworkIntersection(_ArtworkComposite(parts),profile)
                    _artwork_object!(objects,layer,shape,true,Dict("ODB.POLARITY"=>["NEGATIVE"]),max_objects)
                end
            end
            for repeat in get(header.blocks,"STEP-REPEAT",Dict{String,String}[])
                childname=_odb_reference_name(get(repeat,"NAME",""));childname in available || throw(ArgumentError("unknown ODB repeated step"))
                get(repeat,"FLIP","NO") in ("YES","NO") || throw(ArgumentError("invalid ODB step FLIP"))
                get(repeat,"MIRROR","NO") in ("YES","NO") || throw(ArgumentError("invalid ODB step mirror"))
                flip=get(repeat,"FLIP","NO")=="YES"
                nx,ny=parse(Int,get(repeat,"NX","1")),parse(Int,get(repeat,"NY","1"))
                nx>0 && ny>0 || throw(ArgumentError("nonpositive ODB repeat count"))
                x,y=parse(Float64,get(repeat,"X","0"))*unit,parse(Float64,get(repeat,"Y","0"))*unit
                dx,dy=parse(Float64,get(repeat,"DX","0"))*unit,parse(Float64,get(repeat,"DY","0"))*unit
                angle=parse(Float64,get(repeat,"ANGLE","0"));mirror=xor(get(repeat,"MIRROR","NO")=="YES",flip)
                all(isfinite,(x,y,dx,dy,angle)) || throw(ArgumentError("ODB repeat coordinates and angle must be finite"))
                mapping=flip ? complete_flip_map() : nothing
                source_layers=flip ? [mapping[layer] for layer in requested_layers] : requested_layers
                child=step_artwork(childname,source_layers)
                count=BigInt(nx)*ny*length(child.objects)
                count+length(objects)<=max_objects || throw(ArgumentError("ODB repeated step exceeds max_objects"))
                clone_payload=_checked_array_payload_bytes(UInt8,1,
                    sum((BigInt(256)+Base.summarysize(o.attributes) for o in child.objects);init=BigInt(0))*nx*ny)
                _enforce_payload_limit(clone_payload,remaining_payload(),"ODB repeated geometry and attribute copies","max_bytes")
                allocated=_checked_payload_sum("ODB retained repeated geometry",allocated,clone_payload)
                # Repetition distances rotate/mirror with the child grid.
                transform=_odb_orientation(_ArtworkEmpty(),mirror ? 9 : 8,angle,(x,y))
                datum=parse.(Float64,child.attributes["ODB.datum"])
                for j in 0:ny-1,i in 0:nx-1,object in child.objects
                    displacement=transform.matrix*SVector(i*dx-datum[1],j*dy-datum[2])+SVector(x,y)
                    shape=_odb_orientation(object.shape,mirror ? 9 : 8,angle,Tuple(displacement))
                    _artwork_object!(objects,flip ? mapping[object.layer] : object.layer,shape,object.dark,object.attributes,max_objects)
                end
            end
            allocated=_checked_payload_sum("ODB step transforms",allocated,_checked_array_payload_bytes(UInt8,192,length(objects)))
            _enforce_payload_limit(allocated,limit,"ODB retained step transforms","max_bytes")
            datum=string.(datum_values)
            return PlanarArtwork(root,objects,Dict("ODB.step"=>[name],"ODB.datum"=>datum),Set{String}(),default_unit)
        finally
            delete!(active_steps,name)
        end
    end
    doc=step_artwork(chosen)
    matrixjson=_odb_bounded_json(Dict("parameters"=>matrix.parameters,"records"=>matrix.blocks),remaining_payload())
    retain(matrixjson,"ODB retained matrix JSON");doc.attributes["ODB.matrix"]=[matrixjson]
    infojson=_odb_bounded_json(info,remaining_payload())
    retain(infojson,"ODB retained info JSON");doc.attributes["ODB.info"]=[infojson]
    if flipmap!==nothing
        mapjson=_odb_bounded_json(flipmap,remaining_payload())
        retain(mapjson,"ODB retained FLIP metadata");doc.attributes["ODB.flip_layer_map"]=[mapjson]
    end
    return doc
end
