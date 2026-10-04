# Planar layout import. GDSII stream records follow the Cadence stream
# specification reprinted by SPIE (https://www.klayout.de/forum/uploads/editor/v2/k4lfi69nrv9k.pdf).
# DXF codes follow Autodesk's DXF reference. SI coordinates are used throughout.
import LinearAlgebra: det
export LayoutPolygon, PlanarLayoutDocument, read_gdsii, read_dxf
export layout_planar_layout, layout_planar_problem

"""Filled layout polygon with its original layer/datatype key and SI vertices."""
struct LayoutPolygon
    layer::Union{String,Tuple{Int,Int}}
    vertices::Matrix{Float64}
end
"""Flattened planar layout with SI polygons and retained text/point annotations."""
struct PlanarLayoutDocument
    source::String
    polygons::Vector{LayoutPolygon}
    annotations::Vector{Any}
    database_unit_m::Float64
end

_layout_u16(b,i)=UInt16(b[i])<<8|UInt16(b[i+1])
_layout_i16(b,i)=reinterpret(Int16,_layout_u16(b,i))
_layout_u32(b,i)=UInt32(b[i])<<24|UInt32(b[i+1])<<16|UInt32(b[i+2])<<8|UInt32(b[i+3])
_layout_i32(b,i)=reinterpret(Int32,_layout_u32(b,i))
function _gds_real(b,i)
    mantissa=UInt64(0)
    for j in i+1:i+7; mantissa=(mantissa<<8)|UInt64(b[j]); end
    return (b[i]&0x80==0 ? 1. : -1.)*ldexp(Float64(mantissa),-56)*16.0^(Int(b[i]&0x7f)-64)
end
_gds_string(b)=String(collect(Iterators.takewhile(!iszero,b)))

function _layout_polygon!(out,key,vertices,max_elements)
    size(vertices,1)==2 && size(vertices,2)>=3 && all(isfinite,vertices) ||
        throw(ArgumentError("layout polygon requires finite 2D vertices"))
    length(out)<max_elements || throw(ArgumentError("layout exceeds max_elements"))
    push!(out,LayoutPolygon(key,Matrix{Float64}(vertices)))
end

function _layout_circle(center,radius,tolerance;segments=nothing)
    radius>0 || throw(ArgumentError("layout circle radius must be positive"))
    tolerance>0 || throw(ArgumentError("curve tolerance must be positive"))
    step=2acos(clamp(1-tolerance/radius,-1.,1.))
    n=segments===nothing ? max(12,ceil(Int,2pi/max(step,eps(Float64)))) : Int(segments)
    n>=3 || throw(ArgumentError("circle_segments must be at least three"))
    n<=1_000_000 || throw(ArgumentError("curve tolerance requires too many vertices"))
    return hcat([center+radius*[cos(2pi*j/n),sin(2pi*j/n)] for j in 0:n-1]...)
end

function _layout_stroke!(out,key,points,width,pathtype,tolerance,max_elements;
        begin_extension=0.,end_extension=0.,closed=false)
    width>0 || throw(ArgumentError("zero-width paths cannot form planar metal"))
    size(points,2)>=2 || throw(ArgumentError("path needs two points"))
    pathtype in (0,1,2,4) || throw(ArgumentError("unsupported GDSII path type $pathtype"))
    h=width/2; n=size(points,2)
    for i in 1:(closed ? n : n-1)
        next=i==n ? 1 : i+1
        a=copy(points[:,i]); b=copy(points[:,next]); d=b-a; length=norm(d)
        length>0 || throw(ArgumentError("path has coincident successive points"))
        tangent=d/length; normal=[-tangent[2],tangent[1]]
        !closed && i==1 && (a-=(pathtype==2 ? h : pathtype==4 ? begin_extension : 0.)*tangent)
        !closed && i==n-1 && (b+=(pathtype==2 ? h : pathtype==4 ? end_extension : 0.)*tangent)
        _layout_polygon!(out,key,hcat(a+h*normal,b+h*normal,b-h*normal,a-h*normal),max_elements)
    end
    # Miter joins close the outer corner without filling the path's interior.
    for i in (closed ? (1:n) : (2:n-1))
        before_index=i==1 ? n : i-1; after_index=i==n ? 1 : i+1
        v=points[:,i]; before=v-points[:,before_index]; after=points[:,after_index]-v
        u=before/norm(before); w=after/norm(after)
        turn=u[1]*w[2]-u[2]*w[1]
        abs(turn)<1e-12 && continue
        s=-sign(turn); n1=s*[-u[2],u[1]]; n2=s*[-w[2],w[1]]
        denom=1+dot(u,w)
        denom>1e-12 || throw(ArgumentError("path doubles back at a join"))
        miter=v+h*(n1+n2)/denom
        _layout_polygon!(out,key,hcat(v,v+h*n1,miter,v+h*n2),max_elements)
    end
    if pathtype==1 && !closed
        _layout_polygon!(out,key,_layout_circle(points[:,1],h,tolerance),max_elements)
        _layout_polygon!(out,key,_layout_circle(points[:,end],h,tolerance),max_elements)
    end
end

"""Read GDSII boundaries, boxes, paths, SREF/AREF hierarchy and annotations.
`topcell` is required for libraries with multiple unreferenced roots. Reflection,
rotation and magnification are propagated through hierarchy. Absolute transform
flags reject explicitly. Curved caps use a specified SI sagitta tolerance."""
function read_gdsii(path::AbstractString;topcell=nothing,curve_tolerance::Real=1e-8,
        max_elements::Integer=1_000_000,max_bytes::Integer=128_000_000)
    max_elements>0 && max_bytes>0 || throw(ArgumentError("layout limits must be positive"))
    filesize(path)<=max_bytes || throw(ArgumentError("GDSII file exceeds max_bytes"))
    bytes=read(path); i=1; unit=NaN; cell=""; current=nothing
    cells=Dict{String,Vector{Dict{Symbol,Any}}}(); ended=false; header=false
    while i<=length(bytes)
        i+3<=length(bytes) || throw(ArgumentError("truncated GDSII record header"))
        n=Int(_layout_u16(bytes,i)); kind=Int(bytes[i+2]); dtype=Int(bytes[i+3])
        n>=4 && iseven(n) && i+n-1<=length(bytes) || throw(ArgumentError("invalid GDSII record length at byte $i"))
        b=@view bytes[i+4:i+n-1]
        if kind==0
            dtype==2 && length(b)==2 || throw(ArgumentError("invalid GDSII HEADER")); header=true
        elseif kind==3
            dtype==5 && length(b)==16 || throw(ArgumentError("invalid GDSII UNITS"))
            unit=_gds_real(b,9); isfinite(unit) && unit>0 || throw(ArgumentError("invalid GDSII database unit"))
        elseif kind==5
            isempty(cell) || throw(ArgumentError("nested GDSII structure")); cell="<pending>"
        elseif kind==6
            cell=="<pending>" || throw(ArgumentError("unexpected GDSII STRNAME"))
            cell=_gds_string(b); haskey(cells,cell) && throw(ArgumentError("duplicate GDSII cell $cell"))
            cells[cell]=Dict{Symbol,Any}[]
        elseif kind==7
            current===nothing || throw(ArgumentError("unclosed GDSII element")); cell=""
        elseif kind in (8,9,10,11,12,21,45)
            !isempty(cell) && cell!="<pending>" && current===nothing || throw(ArgumentError("invalid GDSII element placement"))
            current=Dict{Symbol,Any}(:kind=>kind,:layer=>0,:datatype=>0,:strans=>0,
                :mag=>1.,:angle=>0.,:width=>0.,:pathtype=>0,:begin_extension=>0.,:end_extension=>0.)
        elseif kind==17
            current===nothing && throw(ArgumentError("GDSII ENDEL without element"))
            haskey(current,:xy) || throw(ArgumentError("GDSII element lacks XY"))
            push!(cells[cell],current); current=nothing
        elseif kind==4
            isempty(cell) && current===nothing || throw(ArgumentError("unclosed GDSII structure"))
            ended=true; i+=n; break
        elseif kind in (13,14,22,33,46)
            current===nothing && throw(ArgumentError("GDSII field outside element"))
            dtype==2 && length(b)==2 || throw(ArgumentError("invalid GDSII integer field"))
            field=kind==13 ? :layer : kind in (14,22,46) ? :datatype : :pathtype
            current[field]=Int(_layout_i16(b,1))
        elseif kind in (15,48,49)
            current===nothing && throw(ArgumentError("GDSII field outside element"))
            dtype==3 && length(b)==4 || throw(ArgumentError("invalid GDSII length field"))
            field=kind==15 ? :width : kind==48 ? :begin_extension : :end_extension
            current[field]=Float64(_layout_i32(b,1))
        elseif kind==16
            current===nothing && throw(ArgumentError("GDSII XY outside element"))
            dtype==3 && length(b)%8==0 || throw(ArgumentError("invalid GDSII XY"))
            current[:xy]=reshape([Float64(_layout_i32(b,j)) for j in 1:4:length(b)],2,:)
        elseif kind in (18,25)
            current===nothing && throw(ArgumentError("GDSII string outside element"))
            current[kind==18 ? :name : :text]=_gds_string(b)
        elseif kind==19
            current===nothing && throw(ArgumentError("GDSII COLROW outside element"))
            length(b)==4 || throw(ArgumentError("invalid GDSII COLROW"))
            current[:columns]=Int(_layout_i16(b,1)); current[:rows]=Int(_layout_i16(b,3))
        elseif kind==26
            current===nothing && throw(ArgumentError("GDSII STRANS outside element"))
            length(b)==2 || throw(ArgumentError("invalid GDSII STRANS"))
            current[:strans]=Int(_layout_u16(b,1))
        elseif kind in (27,28)
            current===nothing && throw(ArgumentError("GDSII transform outside element"))
            length(b)==8 || throw(ArgumentError("invalid GDSII transform real"))
            current[kind==27 ? :mag : :angle]=_gds_real(b,1)
        elseif !(kind in (1,2,23,31,32,34,35,36,38,42,43,44,47,50,51,52,53,54,55,56,57,58,59))
            throw(ArgumentError("unsupported GDSII record 0x$(string(kind;base=16))"))
        end
        i+=n
    end
    header && ended && isfinite(unit) && i>length(bytes) || throw(ArgumentError("incomplete/trailing GDSII stream"))
    referenced=Set(String(e[:name]) for es in values(cells) for e in es if e[:kind] in (10,11))
    roots=setdiff(Set(keys(cells)),referenced)
    root=topcell===nothing ? (length(roots)==1 ? only(roots) :
        throw(ArgumentError("GDSII library needs an explicit topcell"))) : String(topcell)
    haskey(cells,root) || throw(ArgumentError("unknown GDSII topcell $root"))
    polygons=LayoutPolygon[]; annotations=Any[]; active=Set{String}()
    function flatten(name,A,offset)
        name in active && throw(ArgumentError("cyclic GDSII hierarchy at $name"))
        haskey(cells,name) || throw(ArgumentError("missing referenced GDSII cell $name"))
        push!(active,name)
        for e in cells[name]
            kind=e[:kind]; xy=e[:xy]; key=(e[:layer],e[:datatype])
            if kind in (10,11)
                flags=e[:strans]
                flags&0x0006==0 || throw(ArgumentError("absolute GDSII transforms require an absolute-coordinate adapter"))
                isfinite(e[:mag]) && e[:mag]>0 && isfinite(e[:angle]) || throw(ArgumentError("invalid GDSII transform"))
                c,s=cosd(e[:angle]),sind(e[:angle]); reflect=flags&0x8000==0 ? 1. : -1.
                B=e[:mag]*[c -s*reflect;s c*reflect]
                if kind==10
                    size(xy,2)==1 || throw(ArgumentError("SREF needs one origin"))
                    flatten(e[:name],A*B,offset+A*xy[:,1])
                else
                    size(xy,2)==3 && haskey(e,:columns) && haskey(e,:rows) || throw(ArgumentError("invalid AREF lattice"))
                    nc,nr=e[:columns],e[:rows]
                    nc>0 && nr>0 && BigInt(nc)*nr<=max_elements || throw(ArgumentError("invalid/oversized AREF array"))
                    dc=(xy[:,2]-xy[:,1])/nc; dr=(xy[:,3]-xy[:,1])/nr
                    for row in 0:nr-1,col in 0:nc-1
                        flatten(e[:name],A*B,offset+A*(xy[:,1]+col*dc+row*dr))
                    end
                end
            elseif kind in (8,45)
                size(xy,2)>=4 && xy[:,1]==xy[:,end] || throw(ArgumentError("GDSII boundary is not closed"))
                _layout_polygon!(polygons,key,(A*xy.+offset).*unit,max_elements)
            elseif kind==9
                points=(A*xy.+offset).*unit
                width=abs(e[:width])*unit*(e[:width]<0 ? 1. : sqrt(abs(det(A))))
                _layout_stroke!(polygons,key,points,width,e[:pathtype],curve_tolerance,max_elements;
                    begin_extension=e[:begin_extension]*unit*sqrt(abs(det(A))),
                    end_extension=e[:end_extension]*unit*sqrt(abs(det(A))))
            else
                length(annotations)<max_elements || throw(ArgumentError("too many layout annotations"))
                push!(annotations,(;kind=kind==12 ? :text : :node,layer=key,
                    positions=(A*xy.+offset).*unit,text=get(e,:text,"")))
            end
        end
        delete!(active,name)
    end
    flatten(root,Matrix{Float64}(I,2,2),zeros(2))
    return PlanarLayoutDocument(abspath(path),polygons,annotations,unit)
end

function _dxf_pairs(path,max_bytes)
    filesize(path)<=max_bytes || throw(ArgumentError("DXF file exceeds max_bytes"))
    lines=readlines(path); iseven(length(lines)) || throw(ArgumentError("truncated ASCII DXF code/value pair"))
    return [(parse(Int,strip(lines[i])),strip(lines[i+1])) for i in 1:2:length(lines)]
end
function _dxf_entities(pairs)
    rows=Vector{Tuple{Int,String}}[]; row=Tuple{Int,String}[]
    for pair in pairs
        if first(pair)==0 && !isempty(row)
            push!(rows,row); row=Tuple{Int,String}[]
        end
        push!(row,pair)
    end
    isempty(row) || push!(rows,row)
    return rows
end
_dxf_field(row,code,default=nothing)=begin
    i=findfirst(p->p[1]==code,row); i===nothing ? default : row[i][2]
end
_dxf_number(row,code,default=0.)=parse(Float64,_dxf_field(row,code,string(default)))

function _dxf_arc_points(a,b,bulge,tolerance)
    iszero(bulge) && return hcat(a,b)
    chord=b-a; distance=norm(chord); distance>0 || throw(ArgumentError("bulged DXF edge has zero length"))
    normal=[-chord[2],chord[1]]/distance
    center=(a+b)/2+distance*(1-bulge^2)/(4bulge)*normal
    radius=norm(a-center); theta=4atan(bulge); start=atan(a[2]-center[2],a[1]-center[1])
    step=2acos(clamp(1-tolerance/radius,-1.,1.)); n=max(1,ceil(Int,abs(theta)/max(step,eps(Float64))))
    n<=1_000_000 || throw(ArgumentError("bulge needs too many vertices"))
    return hcat([center+radius*[cos(start+theta*j/n),sin(start+theta*j/n)] for j in 0:n]...)
end

"""Read planar ASCII DXF closed polylines (including bulges), circles and
filled solids. Unitless files require explicit `unit_m`. Nonplanar extrusion,
open zero-width paths, variable-width curves, and unsupported entities reject.
Text/dimension entities remain available as annotations."""
function read_dxf(path::AbstractString;unit_m=nothing,curve_tolerance::Real=1e-8,circle_segments=nothing,
        max_elements::Integer=1_000_000,max_bytes::Integer=128_000_000)
    max_elements>0 && max_bytes>0 && curve_tolerance>0 || throw(ArgumentError("invalid layout limits"))
    pairs=_dxf_pairs(path,max_bytes)
    ui=findfirst(p->p==(9,"\$INSUNITS"),pairs)
    declared=ui===nothing ? 0 : parse(Int,pairs[ui+1][2])
    unit=unit_m===nothing ? get(Dict(1=>.0254,2=>.3048,4=>1e-3,5=>1e-2,6=>1.,
        8=>25.4e-6,13=>1e-6,14=>1e-1),declared,NaN) : Float64(unit_m)
    isfinite(unit) && unit>0 || throw(ArgumentError("DXF requires a positive explicit unit_m for unitless/unsupported units"))
    begin_entities=findfirst(p->p==(2,"ENTITIES"),pairs)
    begin_entities===nothing && throw(ArgumentError("DXF lacks ENTITIES section"))
    ending=findnext(p->p==(0,"ENDSEC"),pairs,begin_entities+1)
    ending===nothing && throw(ArgumentError("unterminated DXF ENTITIES"))
    entities=_dxf_entities(pairs[begin_entities+1:ending-1])
    polygons=LayoutPolygon[]; annotations=Any[]; i=1
    while i<=length(entities)
        row=entities[i]; kind=row[1][2]; layer=String(_dxf_field(row,8,"0"))
        _dxf_number(row,210,0.)==0 && _dxf_number(row,220,0.)==0 && _dxf_number(row,230,1.)==1 ||
            throw(ArgumentError("nonplanar DXF extrusion requires a 3D adapter"))
        _dxf_number(row,38,0.)==0 && _dxf_number(row,30,0.)==0 || throw(ArgumentError("nonzero DXF elevation"))
        if kind=="LWPOLYLINE" || kind=="POLYLINE"
            points=Vector{Vector{Float64}}(); bulges=Float64[]
            closed=parse(Int,_dxf_field(row,70,"0"))&1!=0
            constant=_dxf_number(row,43,0.)*unit
            if kind=="LWPOLYLINE"
                for (code,value) in row
                    if code==10
                        push!(points,[parse(Float64,value),NaN]); push!(bulges,0.)
                    elseif code==20
                        isempty(points) && throw(ArgumentError("DXF y coordinate precedes x")); points[end][2]=parse(Float64,value)
                    elseif code==42
                        isempty(points) && throw(ArgumentError("DXF bulge precedes vertex")); bulges[end]=parse(Float64,value)
                    elseif code in (40,41) && parse(Float64,value)!=0
                        throw(ArgumentError("variable-width DXF polyline requires a varying-width path adapter"))
                    end
                end
                parse(Int,_dxf_field(row,90,string(length(points))))==length(points) || throw(ArgumentError("DXF vertex-count mismatch"))
            else
                flags=parse(Int,_dxf_field(row,70,"0")); flags&0x58==0 || throw(ArgumentError("3D/polyface DXF polyline requires a 3D adapter"))
                # Legacy POLYLINE defaults are group codes 40/41; VERTEX
                # widths inherit them when absent (Autodesk DXF reference).
                start_width=_dxf_number(row,40,0.)*unit
                end_width=_dxf_number(row,41,0.)*unit
                constant=start_width
                start_width==end_width || throw(ArgumentError("variable-width DXF polyline requires a varying-width path adapter"))
                i+=1
                while i<=length(entities) && entities[i][1][2]=="VERTEX"
                    v=entities[i]; _dxf_number(v,30)==0 || throw(ArgumentError("nonplanar DXF vertex"))
                    _dxf_number(v,40,start_width/unit)*unit==constant &&
                        _dxf_number(v,41,end_width/unit)*unit==constant ||
                        throw(ArgumentError("variable-width DXF polyline requires a varying-width path adapter"))
                    push!(points,[_dxf_number(v,10),_dxf_number(v,20)]); push!(bulges,_dxf_number(v,42))
                    i+=1
                end
                i<=length(entities) && entities[i][1][2]=="SEQEND" || throw(ArgumentError("DXF POLYLINE lacks SEQEND"))
            end
            isfinite(constant) && constant>=0 || throw(ArgumentError("DXF polyline width must be finite and nonnegative"))
            length(points)>=2 && all(p->all(isfinite,p),points) || throw(ArgumentError("invalid DXF polyline"))
            points=[p*unit for p in points]; vertices=Vector{Float64}[]
            n=length(points); count=closed ? n : n-1
            for j in 1:count
                k=j==n ? 1 : j+1; arc=_dxf_arc_points(points[j],points[k],bulges[j],curve_tolerance)
                append!(vertices,[arc[:,v] for v in 1:size(arc,2)-1])
            end
            closed || push!(vertices,points[end])
            if constant>0
                _layout_stroke!(polygons,layer,hcat(vertices...),constant,0,curve_tolerance,max_elements;closed)
            else
                closed || throw(ArgumentError("open zero-width DXF polyline is not filled metal"))
                _layout_polygon!(polygons,layer,hcat(vertices...),max_elements)
            end
        elseif kind=="CIRCLE"
            center=unit*[_dxf_number(row,10),_dxf_number(row,20)]
            _layout_polygon!(polygons,layer,_layout_circle(center,_dxf_number(row,40)*unit,curve_tolerance;segments=circle_segments),max_elements)
        elseif kind in ("SOLID","TRACE","3DFACE")
            all(j->_dxf_number(row,30+j)==0,0:3) || throw(ArgumentError("nonplanar DXF filled face"))
            vertices=hcat([unit*[_dxf_number(row,10+j),_dxf_number(row,20+j)] for j in 0:3]...)
            if kind in ("SOLID","TRACE"); vertices=vertices[:,[1,2,4,3]]; end
            _layout_polygon!(polygons,layer,vertices,max_elements)
        elseif kind in ("TEXT","MTEXT","DIMENSION","POINT")
            push!(annotations,(;kind=Symbol(lowercase(kind)),layer,
                position=unit*[_dxf_number(row,10),_dxf_number(row,20)],text=_dxf_field(row,1,"")))
        else
            throw(ArgumentError("unsupported planar DXF entity $kind"))
        end
        i+=1
    end
    return PlanarLayoutDocument(abspath(path),polygons,annotations,unit)
end

"""Map drawing layers to a physical solve-ready [`PlanarLayout`](@ref).
An integer mapping creates a PEC sheet on that interface. A named tuple
`(interface=1,metal="Cu")` preserves a conductor model from `metals`.
`(:via,first_layer,last_layer)` creates uniform and tapered axial via bases; the named tuple
`(from=0,to=1,via_type="CuVia")` preserves a model from `via_types`.
The optional fourth tuple entry selects `:ring`, `:vertices`, or `:center`
meshing instead of the full footprint.
`nothing` explicitly skips a drawing layer; unknown mappings and geometry
lost at the selected grid fail. `origin` is subtracted after the dimensionless
`linear_transform` acts on SI coordinates, so drawing-axis reflection is explicit.
Use `solve_planar(layout,f)` to evaluate the preserved material models."""
function layout_planar_layout(layout::PlanarLayoutDocument,stack::PlanarStackup,grid::CellGrid,
        layer_map::AbstractDict,ports::AbstractVector;origin=(0.,0.),
        linear_transform=Matrix{Float64}(I,2,2),
        metals::AbstractDict=Dict("pec"=>0.),
        via_types::AbstractDict=Dict("uniform"=>(kind=VIA_UNIFORM,sigma=Inf),
            "taper"=>(kind=VIA_TAPER,sigma=Inf)),
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    length(origin)==2 && all(isfinite,origin) || throw(ArgumentError("layout origin must be two finite SI coordinates"))
    size(linear_transform)==(2,2) && all(isfinite,linear_transform) && !iszero(det(linear_transform)) ||
        throw(ArgumentError("layout transform must be a finite nonsingular 2x2 matrix"))
    polygons=PlanarShapePolygon[]; vias=PlanarShapeVia[]
    for (index,polygon) in enumerate(layout.polygons)
        haskey(layer_map,polygon.layer) || throw(ArgumentError("unmapped drawing layer $(polygon.layer)"))
        mapping=layer_map[polygon.layer]; mapping===nothing && continue
        transformed=linear_transform*polygon.vertices.-collect(origin)
        v=[_P2(transformed[:,i]) for i in axes(transformed,2)]
        name="imported_$index"
        if mapping isa Integer
            push!(polygons,PlanarShapePolygon(name,Int(mapping),"pec","",v))
        elseif mapping isa NamedTuple && haskey(mapping,:interface) && haskey(mapping,:metal)
            push!(polygons,PlanarShapePolygon(name,Int(mapping.interface),String(mapping.metal),"",v))
        elseif mapping isa Tuple && length(mapping) in (3,4) && mapping[1]==:via
            lo,hi=Int(mapping[2]),Int(mapping[3]); 1<=lo<=hi<=length(stack.layers) || throw(ArgumentError("via mapping outside stack"))
            if length(mapping)==3 || mapping[4]==:full
                push!(vias,PlanarShapeVia(name*"_uniform","uniform",lo-1,hi,v))
                push!(vias,PlanarShapeVia(name*"_taper","taper",lo-1,hi,v))
            else
                scratch=sheet_level(0,grid.nx,grid.ny)
                rasterize_poly!(scratch,grid,transformed[1,:],transformed[2,:])
                mask=_planar_via_mesh_mask(scratch.mask,transformed,grid,mapping[4])
                any(mask) || throw(ArgumentError("via drawing polygon vanished on the selected grid"))
                for cell in findall(mask)
                    i,j=Tuple(cell);x0,x1=(i-1)*grid.dx,i*grid.dx;y0,y1=(j-1)*grid.dy,j*grid.dy
                    cellv=_P2[_P2(x0,y0),_P2(x1,y0),_P2(x1,y1),_P2(x0,y1)]
                    cellname=name*"_$(i)_$(j)"
                    push!(vias,PlanarShapeVia(cellname*"_uniform","uniform",lo-1,hi,cellv))
                    push!(vias,PlanarShapeVia(cellname*"_taper","taper",lo-1,hi,cellv))
                end
            end
        elseif mapping isa NamedTuple && haskey(mapping,:from) && haskey(mapping,:to) && haskey(mapping,:via_type)
            push!(vias,PlanarShapeVia(name,String(mapping.via_type),Int(mapping.from),Int(mapping.to),v))
        else
            throw(ArgumentError("invalid drawing-layer mapping"))
        end
    end
    shape=PlanarShape("imported",polygons,vias,PlanarPin[],Dict{String,Float64}())
    return build_planar_layout(stack,grid,[shape],ports;metals=metals,via_types=via_types,max_bytes=max_bytes)
end

"""Geometry view of [`layout_planar_layout`](@ref). For material loss, retain
the returned physical layout and call `solve_planar` on it."""
layout_planar_problem(layout::PlanarLayoutDocument,stack::PlanarStackup,grid::CellGrid,
    layer_map::AbstractDict,ports::AbstractVector;kw...)=
    layout_planar_layout(layout,stack,grid,layer_map,ports;kw...).problem
