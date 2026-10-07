# Ucamco Gerber Layer Format Specification, revision 2026.05.
# All geometry remains analytic. Unknown commands are errors, not omissions.
export read_gerber
struct _ArtworkBlock <: _ArtworkShape
    objects::Vector{PlanarArtworkObject}
end

function _artwork_polygon(points)
    vertices=hcat((Float64[p...] for p in points)...)
    size(vertices,2)>=3 && all(isfinite,vertices) || throw(ArgumentError("invalid artwork polygon"))
    return _ArtworkPolygon(vertices)
end
_artwork_rectangle(w,h;center=(0.,0.))=_artwork_polygon([(center[1]+x,center[2]+y) for
    (x,y) in ((-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2))])
function _artwork_butt_line(a,b,width)
    width>=0 || throw(ArgumentError("negative primitive width"))
    d=hypot(b[1]-a[1],b[2]-a[2]);d>0 || throw(ArgumentError("zero length vector primitive"))
    nx,ny=width*(a[2]-b[2])/(2d),width*(b[1]-a[1])/(2d)
    return _artwork_polygon([(a[1]+nx,a[2]+ny),(b[1]+nx,b[2]+ny),
        (b[1]-nx,b[2]-ny),(a[1]-nx,a[2]-ny)])
end
function _gerber_expression(source,variables)
    length(source)<=4096 || throw(ArgumentError("Gerber macro expression too long"))
    source=replace(String(source),r"\$(\d+)"=>s->"v"*s[2:end], 'x'=>'*','X'=>'*')
    tree=try Meta.parse(source;raise=true) catch;throw(ArgumentError("invalid Gerber macro expression")) end
    function value(node,depth)
        depth<64 || throw(ArgumentError("Gerber macro expression nesting too deep"))
        result=if node isa Real && !(node isa Bool)
            Float64(node)
        elseif node isa Symbol && occursin(r"^v[1-9]\d*$",String(node))
            get(variables,parse(Int,String(node)[2:end]),0.)
        elseif node isa Expr && node.head==:call && node.args[1] in (:+,:-,:*,:/)
            op=node.args[1];args=node.args[2:end]
            !isempty(args) && (op in (:+,:*) || length(args)<=2) || throw(ArgumentError("invalid Gerber macro arithmetic"))
            a=value(args[1],depth+1)
            if length(args)==1
                op in (:+,:-) || throw(ArgumentError("invalid Gerber macro unary operator"))
                op==:- ? -a : a
            else
                result=a
                for arg in args[2:end]
                    b=value(arg,depth+1)
                    result=op==:+ ? result+b : op==:- ? result-b : op==:* ? result*b : result/b
                end
                result
            end
        else
            throw(ArgumentError("Gerber macro permits only numbers, variables and arithmetic"))
        end
        isfinite(result) || throw(ArgumentError("Gerber macro expression is nonfinite"))
        return result
    end
    return value(tree,0)
end

function _gerber_macro(lines,parameters,unit)
    variables=Dict(i=>v for (i,v) in enumerate(parameters));parts=Tuple{Bool,_ArtworkShape}[]
    for line in lines
        line=strip(line);isempty(line) && continue
        occursin(r"^0(?:\s|,|$)",line) && continue
        assignment=match(r"^\$([1-9]\d*)=(.*)$",line)
        if assignment!==nothing
            variables[parse(Int,assignment[1])]=_gerber_expression(assignment[2],variables);continue
        end
        fields=split(line,',');code=parse(Int,fields[1]);p=[_gerber_expression(f,variables) for f in fields[2:end]]
        rotation=0.;dark=true;shape::_ArtworkShape=_ArtworkEmpty()
        if code==1
            length(p) in (4,5) || throw(ArgumentError("invalid Gerber circle primitive"))
            p[2]>=0 || throw(ArgumentError("negative macro circle diameter"))
            dark=p[1]==1;shape=_ArtworkCircle((p[3]*unit,p[4]*unit),p[2]*unit/2)
            length(p)==5 && (rotation=p[5])
        elseif code in (2,20)
            length(p)==7 || throw(ArgumentError("invalid Gerber vector primitive"))
            dark=p[1]==1;shape=_artwork_butt_line((p[3]*unit,p[4]*unit),(p[5]*unit,p[6]*unit),p[2]*unit);rotation=p[7]
        elseif code in (21,22)
            length(p)==6 || throw(ArgumentError("invalid Gerber rectangle primitive"))
            p[2]>=0 && p[3]>=0 || throw(ArgumentError("negative rectangle dimensions"))
            dark=p[1]==1;center=code==21 ? (p[4]*unit,p[5]*unit) : ((p[4]+p[2]/2)*unit,(p[5]+p[3]/2)*unit)
            shape=_artwork_rectangle(p[2]*unit,p[3]*unit;center);rotation=p[6]
        elseif code==4
            length(p)>=11 || throw(ArgumentError("invalid Gerber outline primitive"))
            n=Int(p[2]);3<=n<=5000 && length(p)==2n+5 || throw(ArgumentError("invalid outline vertex count"))
            points=[(p[2k+1]*unit,p[2k+2]*unit) for k in 1:n+1]
            points[1]==points[end] || throw(ArgumentError("macro outline is not closed"))
            dark=p[1]==1;shape=_artwork_polygon(points[1:end-1]);rotation=p[end]
        elseif code==5
            length(p)==6 || throw(ArgumentError("invalid Gerber polygon primitive"))
            n=Int(p[2]);3<=n<=12 && p[5]>=0 || throw(ArgumentError("invalid regular polygon dimensions"))
            dark=p[1]==1;shape=_artwork_polygon([(unit*(p[3]+p[5]*cospi(2k/n)/2),
                unit*(p[4]+p[5]*sinpi(2k/n)/2)) for k in 0:n-1]);rotation=p[6]
        elseif code==7
            length(p)==6 && p[3]>p[4]>=0 && 0<=p[5]<p[3]/sqrt(2) || throw(ArgumentError("invalid thermal primitive"))
            center=(p[1]*unit,p[2]*unit);outer=p[3]*unit;inner=p[4]*unit;gap=p[5]*unit
            shape=_ArtworkComposite(Tuple{Bool,_ArtworkShape}[(true,_ArtworkCircle(center,outer/2)),
                (false,_ArtworkCircle(center,inner/2)),(false,_artwork_rectangle(outer,gap;center)),
                (false,_artwork_rectangle(gap,outer;center))]);rotation=p[6]
        elseif code==6
            length(p)==9 || throw(ArgumentError("invalid legacy moire primitive"))
            center=(p[1]*unit,p[2]*unit);outer=p[3]*unit;thick=p[4]*unit;gap=p[5]*unit
            n=Int(p[6]);outer>0 && thick>0 && gap>=0 && 0<=n<=5000 && p[7]>=0 && p[8]>=0 ||
                throw(ArgumentError("invalid moire dimensions"))
            rings=Tuple{Bool,_ArtworkShape}[]
            for k in 0:n-1
                radius=outer/2-k*(thick+gap);radius>0 || break
                push!(rings,(true,_ArtworkCircle(center,radius)))
                radius>thick && push!(rings,(false,_ArtworkCircle(center,radius-thick)))
            end
            if p[7]>0 && p[8]>0
                push!(rings,(true,_artwork_rectangle(p[8]*unit,p[7]*unit;center)),
                    (true,_artwork_rectangle(p[7]*unit,p[8]*unit;center)))
            end
            shape=_ArtworkComposite(rings);rotation=p[9]
        else
            throw(ArgumentError("unknown Gerber macro primitive $code"))
        end
        code in (1,2,4,5,20,21,22) && !(p[1] in (0,1)) && throw(ArgumentError("macro exposure must be zero or one"))
        push!(parts,(dark,_artwork_transform(shape;rotation)))
    end
    return _ArtworkComposite(parts)
end

function _gerber_aperture(template,parameters,macros,unit)
    if haskey(macros,template)
        return _gerber_macro(macros[template],parameters,unit),:macro
    end
    p=parameters;shape::_ArtworkShape=_ArtworkEmpty();holes=0
    if template=="C"
        1<=length(p)<=3 && p[1]>0 || throw(ArgumentError("invalid circle aperture"))
        shape=_ArtworkCircle((0.,0.),p[1]*unit/2);holes=1
    elseif template in ("R","O")
        2<=length(p)<=4 && all(>(0),p[1:2]) || throw(ArgumentError("invalid rectangle/obround aperture"))
        w,h=p[1]*unit,p[2]*unit;holes=2
        shape=template=="R" ? _artwork_rectangle(w,h) :
            w>=h ? _ArtworkRoundLine((-(w-h)/2,0.),((w-h)/2,0.),h/2) :
                _ArtworkRoundLine((0.,-(h-w)/2),(0.,(h-w)/2),w/2)
    elseif template=="P"
        2<=length(p)<=5 && p[1]>0 && p[2]==round(p[2]) && 3<=p[2]<=12 || throw(ArgumentError("invalid polygon aperture"))
        n=Int(p[2]);rotation=length(p)>=3 ? p[3] : 0.;holes=3
        shape=_artwork_polygon([(unit*p[1]*cosd(rotation+360k/n)/2,unit*p[1]*sind(rotation+360k/n)/2) for k in 0:n-1])
    else
        throw(ArgumentError("undefined Gerber aperture template $template"))
    end
    if length(p)>holes
        h=p[holes+1:end];all(>=(0),h) || throw(ArgumentError("negative aperture hole"))
        hole=length(h)==1 ? _ArtworkCircle((0.,0.),h[1]*unit/2) : _artwork_rectangle(h[1]*unit,h[2]*unit)
        shape=_ArtworkComposite(Tuple{Bool,_ArtworkShape}[(true,shape),(false,hole)])
    end
    return shape,Symbol(template)
end

function _gerber_coordinate(value,digits,zero,unit)
    occursin(r"^[+-]?\d+$",value) || throw(ArgumentError("invalid Gerber coordinate"))
    sign=startswith(value,'-') ? -1. : 1.;v=lstrip(value,['+','-']);n=digits[1]+digits[2]
    length(v)<=n || throw(ArgumentError("Gerber coordinate exceeds declared format"))
    zero==:trailing && (v=rpad(v,n,'0'))
    result=sign*parse(Float64,v)*10.0^(-digits[2])*unit
    isfinite(result) || throw(ArgumentError("nonfinite Gerber coordinate"))
    return result
end
function _gerber_arc(a,b,i,j,clockwise,single,resolution)
    # Legacy G74 specifies a zero-length arc when endpoints coincide;
    # G75 specifies a full circle. Its zero-length swept image is the
    # aperture at the shared endpoint, represented by a point segment.
    single && a==b && return _ArtworkRoundLine(a,b,0.)
    candidates=single ? [(a[1]+sx*abs(i),a[2]+sy*abs(j)) for sx in (-1.,1.),sy in (-1.,1.)] : [(a[1]+i,a[2]+j)]
    valid=Tuple{Float64,_ArtworkRoundArc}[]
    for c in unique(vec(candidates))
        r0=hypot(a[1]-c[1],a[2]-c[2]);r1=hypot(b[1]-c[1],b[2]-c[2]);r0>0 || continue
        mismatch=abs(r0-r1);mismatch<=2sqrt(2)*resolution+32eps(max(r0,r1)) || continue
        dx,dy=b[1]-a[1],b[2]-a[2];d2=dx*dx+dy*dy
        if d2>0
            correction=(r1*r1-r0*r0)/(2d2);c=(c[1]+correction*dx,c[2]+correction*dy)
        end
        arc=_ArtworkRoundArc(a,b,c,clockwise,0.)
        single && abs(_artwork_arc_angles(arc)[2])>pi/2+8resolution/r0 && continue
        push!(valid,(mismatch,arc))
    end
    isempty(valid) && throw(ArgumentError("Gerber arc has inconsistent center/radius or quadrant"))
    sort!(valid;by=first);return valid[1][2]
end

"""Read extended Gerber fabrication artwork in SI without polygonizing curves.
Supports standard/macro/block apertures, linear/circular drawing, regions,
polarity, aperture transforms, step/repeat, attributes and legacy coordinate
formats and MI/SF/OF/IR image transforms. Nonuniform SF scales coordinate
paths into analytic ellipses while preserving aperture dimensions and
step/repeat spacing; AS is retained as device metadata. Local aperture
holes remain transparent. `layer` selects the output
technology name. Image inversion is explicit over the later analysis grid.
Unknown records reject the file; no conductive objects are silently omitted.
`max_bytes` bounds input and a conservative owned geometry payload estimate."""
function read_gerber(path::AbstractString;layer::AbstractString=splitext(basename(path))[1],
        max_objects::Integer=1_000_000,max_bytes::Integer=_default_max_dense_payload_bytes())
    max_objects>0 || throw(ArgumentError("max_objects must be positive"))
    limit=_validated_resource_limit("max_bytes",max_bytes);input=filesize(path)
    input_payload=_checked_array_payload_bytes(UInt8,3,input)
    _enforce_payload_limit(input_payload,limit,"Gerber input","max_bytes")
    data=open(path,"r") do io;read(io,String);end
    objects=PlanarArtworkObject[];contexts=Any[]
    macros=Dict{String,Vector{String}}();apertures=Dict{Int,Tuple{_ArtworkShape,Symbol}}()
    apertureattrs=Dict{Int,Dict{String,Vector{String}}}()
    fileattrs=Dict{String,Vector{String}}();apattrs=Dict{String,Vector{String}}();objattrs=Dict{String,Vector{String}}()
    unit=0.;xdigits=(0,0);ydigits=(0,0);zero=:leading;incremental=false;position=(0.,0.);position_known=false
    selected=0;operation=0;mode=1;single=false;dark=true;rotation=0.;scale=1.;mirror=(1.,1.)
    image_mirror=(1.,1.);image_scale=(1.,1.);image_offset=(0.,0.);image_rotation=0.
    legacy_seen=Set{String}();legacy_started=false
    region=false;segments=Union{_ArtworkRoundLine,_ArtworkRoundArc}[];regionstart=(0.,0.);regionattrs=Dict{String,Vector{String}}()
    negative=false;finished=false;cursor=firstindex(data);allocated=input_payload
    frame()=any(c->c.kind==:ab,contexts) ? (1.,1.,(0.,0.)) :
        (image_mirror[1]*image_scale[1],image_mirror[2]*image_scale[2],
            (image_offset[1]*unit,image_offset[2]*unit))
    image_point(p)=_artwork_affine_point(p,frame()...)
    function emit(shape,polarity=dark,attributes=objattrs)
        isempty(contexts) && !iszero(image_rotation) && (shape=_artwork_transform(shape;rotation=image_rotation))
        allocated=_checked_payload_sum("Gerber geometry",allocated,256,Base.summarysize(shape))
        _enforce_payload_limit(allocated,limit,"Gerber geometry","max_bytes")
        destination=isempty(contexts) ? objects : contexts[end].objects
        _artwork_object!(destination,layer,shape,polarity,attributes,max_objects)
    end
    function finish_contour()
        isempty(segments) && return
        segments[end].stop==regionstart || throw(ArgumentError("Gerber region contour is not closed"))
        sx,sy,origin=frame();shape=_ArtworkRegion(copy(segments))
        transformed=(sx==1 && sy==1 && origin==(0.,0.)) ? shape : _artwork_affine(shape,sx,sy,origin)
        emit(transformed,dark,regionattrs);empty!(segments)
    end
    function close_context(kind)
        !isempty(contexts) && contexts[end].kind==kind || throw(ArgumentError("unmatched Gerber block terminator"))
        context=pop!(contexts)
        if kind==:ab
            haskey(apertures,context.id) && throw(ArgumentError("Gerber aperture redefinition"))
            # Blocks append ordered image operations. Their clear objects
            # erase prior image copper; macro/standard aperture holes remain
            # transparent and therefore cannot represent block polarity.
            shape=_ArtworkBlock(context.objects)
            apertures[context.id]=(shape,:block)
            apertureattrs[context.id]=context.attributes
            position_known=false
        else
            count=BigInt(context.nx)*context.ny*length(context.objects)
            count<=max_objects || throw(ArgumentError("Gerber repeat exceeds max_objects"))
            for j in 0:context.ny-1,i in 0:context.nx-1,object in context.objects
                emit(_artwork_transform(object.shape;origin=(i*context.dx,j*context.dy)),object.dark,object.attributes)
            end
        end
    end
    while cursor<=lastindex(data)
        while cursor<=lastindex(data) && isspace(data[cursor]);cursor=nextind(data,cursor);end
        cursor>lastindex(data) && break
        finished && throw(ArgumentError("Gerber data follows end-of-file command"))
        extended=data[cursor]=='%';start=nextind(data,cursor)
        stop=findnext(extended ? '%' : '*',data,start)
        stop===nothing && throw(ArgumentError("unterminated Gerber command"))
        content=strip(data[(extended ? start : cursor):prevind(data,stop)])
        cursor=nextind(data,stop)
        if extended && startswith(content,"AM")
            words=split(content,'*');name=strip(words[1][3:end]);isempty(name) && throw(ArgumentError("empty Gerber macro name"))
            haskey(macros,name) && throw(ArgumentError("Gerber macro redefinition"))
            macros[name]=String.(filter(!isempty,strip.(words[2:end])));continue
        end
        commands=extended ? filter(!isempty,strip.(split(content,'*'))) : [content]
        for command in commands
            isempty(command) && continue
            finished && throw(ArgumentError("Gerber data follows end-of-file command"))
            if startswith(command,"G04")
                continue
            elseif startswith(command,"FS")
                m=match(r"^FS([LT])([AI])X([1-6])([0-9])Y([1-6])([0-9])$",command)
                m!==nothing || throw(ArgumentError("invalid Gerber format statement"))
                xdigits=(parse(Int,m[3]),parse(Int,m[4]));ydigits=(parse(Int,m[5]),parse(Int,m[6]))
                zero=m[1]=="L" ? :leading : :trailing;incremental=m[2]=="I"
            elseif command in ("MOMM","MOIN","G70","G71")
                newunit=command in ("MOMM","G71") ? 1e-3 : .0254
                unit in (0.,newunit) || throw(ArgumentError("Gerber units change after definition"));unit=newunit
            elseif startswith(command,"AD")
                unit>0 || throw(ArgumentError("Gerber aperture before unit declaration"))
                m=match(r"^ADD?(\d+)([^,]+)(?:,(.*))?$",command)
                m!==nothing || throw(ArgumentError("invalid Gerber aperture definition"))
                id=parse(Int,m[1]);id>=10 && !haskey(apertures,id) || throw(ArgumentError("invalid/redefined Gerber aperture number"))
                p=m[3]===nothing ? Float64[] : parse.(Float64,split(m[3],r"[xX]"))
                all(isfinite,p) || throw(ArgumentError("nonfinite Gerber aperture parameter"))
                apertures[id]=_gerber_aperture(String(m[2]),p,macros,unit)
                apertureattrs[id]=deepcopy(apattrs)
            elseif startswith(command,"MI") || startswith(command,"SF") || startswith(command,"OF") ||
                    startswith(command,"IR") || startswith(command,"AS")
                key=command[1:2]
                !legacy_started && !region && isempty(contexts) || throw(ArgumentError("legacy image transform must precede coordinate data"))
                key in legacy_seen && throw(ArgumentError("duplicate legacy Gerber $key command"))
                push!(legacy_seen,key)
                if key=="AS"
                    command in ("ASAXBY","ASAYBX") || throw(ArgumentError("invalid Gerber axis selection"))
                    # AS positions the image on a plotting device; the
                    # primary specification gives it no exchange-image effect.
                    fileattrs["AS"]=[command[3:end]]
                elseif key=="IR"
                    command in ("IR0","IR90","IR180","IR270") || throw(ArgumentError("invalid Gerber image rotation"))
                    image_rotation=parse(Float64,command[3:end])
                elseif key=="MI"
                    m=match(r"^MI(?:A([01]))?(?:B([01]))?$",command)
                    m===nothing && throw(ArgumentError("invalid Gerber image mirror"))
                    image_mirror=(m[1]=="1" ? -1. : 1.,m[2]=="1" ? -1. : 1.)
                else
                    decimal="(?:[0-9]+(?:\\.[0-9]{0,5})?|\\.[0-9]{1,5})"
                    value=key=="OF" ? "[+\\-]?"*decimal : decimal
                    m=match(Regex("^"*key*"(?:A("*value*"))?(?:B("*value*"))?"*raw"$"),command)
                    m===nothing && throw(ArgumentError("invalid Gerber $key transform"))
                    default=key=="SF" ? 1. : 0.
                    values=ntuple(i->m[i]===nothing ? default : parse(Float64,m[i]),2)
                    if key=="SF"
                        all(v->isfinite(v) && .0001<=v<=999.99999,values) || throw(ArgumentError("Gerber image scale outside allowed range"))
                        image_scale=values
                    else
                        all(v->isfinite(v) && abs(v)<=99999.99999,values) || throw(ArgumentError("Gerber image offset outside allowed range"))
                        image_offset=values
                    end
                end
            elseif startswith(command,"AB")
                region && throw(ArgumentError("Gerber aperture block inside region"))
                if command=="AB"
                    close_context(:ab)
                else
                    m=match(r"^ABD(\d+)$",command);m!==nothing || throw(ArgumentError("invalid aperture block"))
                    id=parse(Int,m[1]);id>=10 || throw(ArgumentError("invalid aperture block number"))
                    push!(contexts,(kind=:ab,id=id,objects=PlanarArtworkObject[],attributes=deepcopy(apattrs)))
                end
            elseif startswith(command,"SR")
                region && throw(ArgumentError("Gerber repeat inside region"))
                if command=="SR"
                    close_context(:sr)
                else
                    any(c->c.kind in (:sr,:ab),contexts) && throw(ArgumentError("Gerber repeat cannot nest or appear inside an aperture block"))
                    m=match(r"^SRX(\d+)Y(\d+)I([+\-\d.eE]+)J([+\-\d.eE]+)$",command)
                    m!==nothing && unit>0 || throw(ArgumentError("invalid Gerber repeat"))
                    nx,ny=parse(Int,m[1]),parse(Int,m[2]);nx>0 && ny>0 || throw(ArgumentError("nonpositive repeat count"))
                    dx,dy=parse(Float64,m[3])*unit,parse(Float64,m[4])*unit
                    all(isfinite,(dx,dy)) || throw(ArgumentError("nonfinite Gerber repeat step"))
                    push!(contexts,(kind=:sr,nx=nx,ny=ny,dx=dx,dy=dy,objects=PlanarArtworkObject[]))
                end
            elseif command in ("LPD","LPC")
                region && throw(ArgumentError("Gerber polarity change inside region"));dark=command=="LPD"
            elseif startswith(command,"LM")
                value=command[3:end];value in ("N","X","Y","XY") || throw(ArgumentError("invalid Gerber mirror"))
                mirror=(occursin('X',value) ? -1. : 1.,occursin('Y',value) ? -1. : 1.)
            elseif startswith(command,"LR") || startswith(command,"LS")
                value=parse(Float64,command[3:end]);isfinite(value) || throw(ArgumentError("nonfinite Gerber aperture transform"))
                if startswith(command,"LR");rotation=value;else;value>0 || throw(ArgumentError("nonpositive aperture scale"));scale=value;end
            elseif startswith(command,"TF") || startswith(command,"TA") || startswith(command,"TO")
                values=split(command[3:end],',');isempty(values[1]) && throw(ArgumentError("empty Gerber attribute name"))
                attributes=startswith(command,"TF") ? fileattrs : startswith(command,"TA") ? apattrs : objattrs
                attributes[String(values[1])]=String.(values[2:end])
            elseif startswith(command,"TD")
                name=command[3:end]
                for attributes in (apattrs,objattrs)
                    isempty(name) ? empty!(attributes) : delete!(attributes,name)
                end
            elseif command in ("IPPOS","IPNEG")
                isempty(objects) && isempty(contexts) || throw(ArgumentError("late Gerber image polarity"));negative=command=="IPNEG"
            elseif startswith(command,"IN") || startswith(command,"LN")
                fileattrs[command[1:2]]=[command[3:end]]
            elseif command in ("G90","G91")
                incremental=command=="G91"
            elseif command in ("G36","G37")
                if command=="G36"
                    !region || throw(ArgumentError("nested Gerber region"));region=true;regionstart=position
                    regionattrs=merge(deepcopy(apattrs),deepcopy(objattrs))
                else
                    region || throw(ArgumentError("unmatched Gerber region end"));finish_contour();region=false
                end
            elseif command in ("G74","G75")
                single=command=="G74"
            elseif command in ("M02","M00")
                !region && isempty(contexts) || throw(ArgumentError("unclosed Gerber region/block"));finished=true
            elseif command=="M01"
                continue
            else
                fields=collect(eachmatch(r"([GXYIJD])([+\-]?\d+)",command))
                join(m.match for m in fields)==command || throw(ArgumentError("unsupported Gerber command $command"))
                coordinates=Dict{Char,String}();action=0
                for m in fields
                    tag=only(m[1]);value=m[2]
                    if tag=='G'
                        g=parse(Int,value)
                        g in (1,2,3,54,55) || throw(ArgumentError("unsupported Gerber G code $g"))
                        g in (1,2,3) && (mode=g)
                    elseif tag=='D'
                        d=parse(Int,value)
                        if d>=10
                            haskey(apertures,d) || throw(ArgumentError("undefined Gerber aperture D$d"));selected=d
                        else
                            d in (1,2,3) || throw(ArgumentError("invalid Gerber operation"));action=d;operation=d
                        end
                    else
                        haskey(coordinates,tag) && throw(ArgumentError("duplicate Gerber coordinate"));coordinates[tag]=String(value)
                    end
                end
                isempty(coordinates) && action==0 && continue
                unit>0 && xdigits[1]>0 && ydigits[1]>0 || throw(ArgumentError("Gerber coordinates before units/format"))
                action==0 && (action=operation);action in (1,2,3) || throw(ArgumentError("Gerber operation not selected"))
                position_known || (action in (2,3) && haskey(coordinates,'X') && haskey(coordinates,'Y')) ||
                    throw(ArgumentError("Gerber current point is undefined; move/flash with both coordinates first"))
                a=position
                any(c->c.kind==:ab,contexts) || (legacy_started=true)
                x=haskey(coordinates,'X') ? _gerber_coordinate(coordinates['X'],xdigits,zero,unit) : incremental ? 0. : a[1]
                y=haskey(coordinates,'Y') ? _gerber_coordinate(coordinates['Y'],ydigits,zero,unit) : incremental ? 0. : a[2]
                b=incremental ? (a[1]+x,a[2]+y) : (x,y)
                if action==2
                    region && (finish_contour();regionstart=b)
                elseif action==3
                    !region && selected>0 || throw(ArgumentError("invalid Gerber flash"))
                    shape,kind=apertures[selected]
                    image_b=image_point(b)
                    if kind===:block
                        for object in (shape::_ArtworkBlock).objects
                            emit(_artwork_transform(object.shape;origin=image_b,rotation,scale,mirror),
                                dark ? object.dark : !object.dark,
                                merge(object.attributes,apertureattrs[selected],objattrs))
                        end
                    else
                        emit(_artwork_transform(shape;origin=image_b,rotation,scale,mirror),
                            dark,merge(apertureattrs[selected],objattrs))
                    end
                else
                    segment=if mode==1
                        _ArtworkRoundLine(a,b,0.)
                    else
                        i=haskey(coordinates,'I') ? _gerber_coordinate(coordinates['I'],xdigits,zero,unit) : 0.
                        j=haskey(coordinates,'J') ? _gerber_coordinate(coordinates['J'],ydigits,zero,unit) : 0.
                        _gerber_arc(a,b,i,j,mode==2,single,unit*10.0^(-min(xdigits[2],ydigits[2])))
                    end
                    if region
                        push!(segments,segment)
                        length(segments)<=max_objects || throw(ArgumentError("Gerber region exceeds segment budget"))
                    else
                        selected>0 || throw(ArgumentError("Gerber draw without aperture"))
                        image_a,image_b=image_point(a),image_point(b)
                        sx,sy,origin=frame()
                        curve=segment isa _ArtworkRoundArc ? _gerber_coordinate_path(segment,sx,sy,origin) : nothing
                        aperture,kind=apertures[selected]
                        stroke=if kind==:C
                            outer=aperture isa _ArtworkComposite ? aperture.parts[1][2] : aperture
                            radius=(outer::_ArtworkCircle).radius*scale
                            segment isa _ArtworkRoundLine ? _ArtworkRoundLine(image_a,image_b,radius) :
                                curve isa _ArtworkRoundArc ? _ArtworkRoundArc(curve.start,curve.stop,curve.center,curve.clockwise,radius) :
                                    _ArtworkCircleAffineArc(curve,radius)
                        elseif kind==:R && aperture isa _ArtworkPolygon
                            transformed=_artwork_transform(aperture;rotation,scale,mirror)
                            vertices=transformed.matrix*aperture.vertices
                            polygon=_ArtworkPolygon(vertices)
                            if segment isa _ArtworkRoundLine
                                _artwork_convex_hull([(p[1]+vertices[1,i],p[2]+vertices[2,i])
                                    for p in (image_a,image_b),i in axes(vertices,2)])
                            else
                                curve isa _ArtworkRoundArc ? _ArtworkPolygonArc(polygon,curve) : _ArtworkPolygonAffineArc(curve,polygon)
                            end
                        else
                            throw(ArgumentError("Gerber drawing requires a circle or solid standard rectangle aperture"))
                        end
                        emit(stroke,dark,merge(apertureattrs[selected],objattrs))
                    end
                end
                position=b;position_known=true
            end
        end
    end
    finished || throw(ArgumentError("Gerber file lacks end-of-file command"))
    return PlanarArtwork(abspath(path),objects,fileattrs,negative ? Set([String(layer)]) : Set{String}(),unit)
end
