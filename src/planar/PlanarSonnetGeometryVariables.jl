# Native anchored/symmetric and radial dimensions with movable points.
# Original SON records remain the source identity of the effective geometry.
struct _SonnetGeometryParameter
    name::String
    kind::String
    axis::Int
    direction::Int
    scaled::Bool
    both_axes::Bool
    nominal::Float64
    references::NTuple{2,Tuple{Int,Int}}
    points::NTuple{2,Vector{Tuple{Int,Int}}}
    explicit_moving_references::Int
    line::Int
end

# Zero saved offsets have a proved translation adapter for ANC/NSCD only.
# A positive source dimension that underflows in SI remains invalid.
@inline _sonnet_geovar_nominal_value_ok(value,nominal,kind,scaled)=
    isfinite(value) && (value>0 ||
        (iszero(nominal) && iszero(value) && kind=="ANC" && !scaled))

function _sonnet_geovar_limit(value::Integer,label)
    0<value<=typemax(Int) || throw(ArgumentError("$label must be a positive stored integer"))
    return Int(value)
end

function _sonnet_geovar_row(rows,index,p)
    index<=length(rows) || _sonnet_error(p.source,0,"truncated GEOVAR block")
    return rows[index]
end

function _sonnet_geovar_position(rows,index,p)
    row=_sonnet_geovar_row(rows,index,p)
    length(row.tokens)==3 && row.tokens[1]=="POS" &&
        all(i->(v=tryparse(Float64,row.tokens[i]);v!==nothing && isfinite(v)),2:3) ||
        _sonnet_error(p.source,row.line,"invalid GEOVAR display position")
    return nothing
end

function _sonnet_geovar_groups(row,p)
    length(row.tokens)==2 || _sonnet_error(p.source,row.line,"invalid GEOVAR point-set group count")
    groups=tryparse(Int,row.tokens[2])
    groups!==nothing && groups>=0 || _sonnet_error(p.source,row.line,"invalid GEOVAR point-set group count")
    return groups
end

function _sonnet_geovar_point(p,polygons,row,index)
    t=row.tokens
    length(t)==3 && t[1]=="POLY" || _sonnet_error(p.source,row.line,"GEOVAR requires a literal POLY point identity")
    id=tryparse(Int,t[2]);id!==nothing && haskey(polygons,id) ||
        _sonnet_error(p.source,row.line,"GEOVAR references an unknown polygon")
    point=tryparse(Int,index)
    vertices=polygons[id].vertices
    closed=vertices[1,1]==vertices[1,end] && vertices[2,1]==vertices[2,end]
    count=size(vertices,2)-Int(closed)
    point!==nothing && 0<=point<count || _sonnet_error(p.source,row.line,"GEOVAR point index is outside its polygon")
    return (id,point+1)
end

function _sonnet_geovar_reference(rows,index,p,polygons,tag)
    row=_sonnet_geovar_row(rows,index,p);t=row.tokens
    length(t)==4 && t[1]==tag && t[2]=="POLY" && t[4]=="1" ||
        _sonnet_error(p.source,row.line,"GEOVAR $tag requires exactly one literal polygon point")
    pointrow=_sonnet_geovar_row(rows,index+1,p)
    length(pointrow.tokens)==1 || _sonnet_error(p.source,pointrow.line,"invalid GEOVAR reference point")
    point=_sonnet_geovar_point(p,polygons,SonnetRecord(row.line,t[2:4]),only(pointrow.tokens))
    return point,index+2
end

function _sonnet_geovar_set(rows,index,p,polygons,tag,max_points,used,max_expanded,expanded)
    row=_sonnet_geovar_row(rows,index,p);t=row.tokens
    length(t)==2 && t[1]==tag || _sonnet_error(p.source,row.line,"GEOVAR requires $tag")
    groups=_sonnet_geovar_groups(row,p)
    groups<=length(rows)-index || _sonnet_error(p.source,row.line,"invalid GEOVAR point-set group count")
    points=Tuple{Int,Int}[];index+=1
    for _ in 1:groups
        row=_sonnet_geovar_row(rows,index,p);t=row.tokens
        length(t)==3 && t[1]=="POLY" || _sonnet_error(p.source,row.line,"unsupported GEOVAR point-set identity")
        count=tryparse(Int,t[3])
        count!==nothing && 0<=count<=length(rows)-index || _sonnet_error(p.source,row.line,"invalid GEOVAR polygon point count")
        if iszero(count)
            # POLY id 0 selects every vertex except the closing duplicate.
            id=tryparse(Int,t[2]);id!==nothing && haskey(polygons,id) ||
                _sonnet_error(p.source,row.line,"GEOVAR references an unknown polygon")
            vertices=polygons[id].vertices
            closed=vertices[1,1]==vertices[1,end] && vertices[2,1]==vertices[2,end]
            count=size(vertices,2)-Int(closed)
            used<=max_points-count || _sonnet_error(p.source,row.line,"GEOVAR point budget exceeded")
            expanded<=max_expanded-count || _sonnet_error(p.source,row.line,
                "GEOVAR whole-polygon workspace exceeds max_bytes")
            used+=count;expanded+=count
            sizehint!(points,length(points)+count)
            for point in 1:count;push!(points,(id,point));end
            index+=1;continue
        end
        used<=max_points-count || _sonnet_error(p.source,row.line,"GEOVAR point budget exceeded")
        used+=count;index+=1
        for _ in 1:count
            pointrow=_sonnet_geovar_row(rows,index,p)
            length(pointrow.tokens)==1 || _sonnet_error(p.source,pointrow.line,"invalid GEOVAR point-set entry")
            push!(points,_sonnet_geovar_point(p,polygons,row,only(pointrow.tokens)));index+=1
        end
    end
    _sonnet_geovar_row(rows,index,p).tokens==["END"] ||
        _sonnet_error(p.source,rows[index].line,"unterminated GEOVAR point set")
    # Native entries are movements: a repeated identity moves repeatedly.
    return points,index+1,used,expanded
end

function _sonnet_geovar_parameters(p,max_parameters,max_points,max_expanded)
    rows=_sonnet_section(p.records,"GEO",p.source)
    polygons=Dict(q.id=>q for q in p.polygons)
    quantities=Dict(r.tokens[2]=>r.tokens[3] for r in rows if length(r.tokens)>=4 && r.tokens[1]=="VALVAR")
    length(polygons)==length(p.polygons) || throw(ArgumentError("duplicate GEOVAR polygon identities"))
    parameters=_SonnetGeometryParameter[];index=1;used=0;expanded=0
    while index<=length(rows)
        row=rows[index];t=row.tokens
        if t[1]!="GEOVAR";index+=1;continue;end
        length(parameters)<max_parameters || _sonnet_error(p.source,row.line,"GEOVAR parameter budget exceeded")
        length(t)==6 && t[3] in ("ANC","SYM","RAD") && t[4] in ("XDIR","YDIR") &&
            t[5] in ("1","-1") && t[6] in ("NSCD","SCUNI","SCXY") ||
            _sonnet_error(p.source,row.line,"GEOVAR requires an ANC/SYM/RAD XDIR/YDIR point-set adapter")
        name=t[2];haskey(p.variables,name) || _sonnet_error(p.source,row.line,"GEOVAR lacks its declared quantity variable")
        get(quantities,name,"")=="LNG" ||
            _sonnet_error(p.source,row.line,"GEOVAR variable must have one LNG declaration")
        _sonnet_geovar_position(rows,index+1,p)
        nominalrow=_sonnet_geovar_row(rows,index+2,p)
        length(nominalrow.tokens)==2 && nominalrow.tokens[1]=="NOM" ||
            _sonnet_error(p.source,nominalrow.line,"GEOVAR requires a literal nominal dimension")
        nominal=tryparse(Float64,nominalrow.tokens[2])
        nominal!==nothing && _sonnet_geovar_nominal_value_ok(nominal,nominal,t[3],t[6]!="NSCD") ||
            _sonnet_error(p.source,nominalrow.line,"GEOVAR nominal dimension needs a positive value or an unscaled anchored zero-offset adapter")
        first,index=_sonnet_geovar_reference(rows,index+3,p,polygons,"REF1")
        second,index=_sonnet_geovar_reference(rows,index,p,polygons,"REF2")
        axis=t[4]=="XDIR" ? 1 : 2
        iszero(nominal) && polygons[first[1]].vertices[axis,first[2]]!=
            polygons[second[1]].vertices[axis,second[2]] &&
            _sonnet_error(p.source,row.line,"zero saved GEOVAR offsets require coincident reference coordinates on their axis")
        used<=max_points-2 || _sonnet_error(p.source,row.line,"GEOVAR reference budget exceeded")
        used+=2
        equation=_sonnet_geovar_row(rows,index,p)
        if equation.tokens[1]=="EQN"
            length(equation.tokens)==2 && isequal(_sonnet_parse_scalar(equation.tokens[2]),
                _sonnet_parse_scalar(p.variables[name])) ||
                _sonnet_error(p.source,equation.line,"GEOVAR EQN disagrees with its quantity definition")
            index+=1
        end
        a,index,used,expanded=_sonnet_geovar_set(rows,index,p,polygons,"PS1",max_points,used,max_expanded,expanded)
        b,index,used,expanded=_sonnet_geovar_set(rows,index,p,polygons,"PS2",max_points,used,max_expanded,expanded)
        explicit_references=0
        for point in b
            explicit_references+=Int(point==second)
        end
        # The implicit moving reference is an additional movement for native
        # unscaled anchored dimensions and radial dimensions under any header.
        (first in a || (second in b && !(t[3]=="RAD" || (t[3]=="ANC" && t[6]=="NSCD")))) && _sonnet_error(p.source,row.line,
            "explicit GEOVAR reference repetitions require their native movement adapter")
        _sonnet_geovar_row(rows,index,p).tokens==["END"] ||
            _sonnet_error(p.source,rows[index].line,"unterminated GEOVAR block")
        index+=1
        t[3] in ("ANC","RAD") && !isempty(a) && _sonnet_error(p.source,row.line,"anchored/radial GEOVAR has an unsupported moving anchor set")
        t[3]=="SYM" && !(first in a) && push!(a,first)
        push!(b,second)
        bset=Set(b)
        (first==second || any(point->point in bset,a) || first in bset) &&
            _sonnet_error(p.source,row.line,"GEOVAR adjustable sets conflict with their references")
        direction=parse(Int,t[5])
        if t[3]=="ANC" && t[6]=="NSCD"
            fixed=polygons[first[1]].vertices[axis,first[2]]
            moving=polygons[second[1]].vertices[axis,second[2]]
            # Native anchored translation follows the reference-coordinate
            # order even when its display direction field disagrees. A zero
            # offset uses that field to choose its initial direction.
            fixed==moving || (direction=moving>fixed ? 1 : -1)
        end
        push!(parameters,_SonnetGeometryParameter(name,t[3],axis,
            direction,t[6]!="NSCD",t[6]=="SCXY",nominal,(first,second),(a,b),explicit_references,row.line))
    end
    return parameters
end

# Native files also retain scaled/dependent dimensions whose saved
# coordinates are usable unchanged. Only exact equality with every saved NOM
# permits this path; an effective change still requires the proved adapter.
function _sonnet_geovar_nominal_geometry(p,frequency,variables)
    rows=_sonnet_section(p.records,"GEO",p.source)
    polygons=Dict(q.id=>q for q in p.polygons)
    quantities=Dict(r.tokens[2]=>r.tokens[3] for r in rows if length(r.tokens)>=4 && r.tokens[1]=="VALVAR")
    unchanged=true
    for (index,row) in enumerate(rows)
        t=row.tokens
        t[1] in ("PS1","PS2") && _sonnet_geovar_groups(row,p)
        if t[1] in ("REF1","REF2") && length(t)==4 && t[2]=="POLY" && t[4]=="1"
            id=tryparse(Int,t[3])
            if id!==nothing && haskey(polygons,id)
                pointrow=_sonnet_geovar_row(rows,index+1,p)
                point=length(pointrow.tokens)==1 ? tryparse(Int,only(pointrow.tokens)) : nothing
                vertices=polygons[id].vertices
                closed=vertices[1,1]==vertices[1,end] && vertices[2,1]==vertices[2,end]
                point!==nothing && 0<=point<size(vertices,2)-Int(closed) ||
                    _sonnet_error(p.source,pointrow.line,"GEOVAR saved out-of-range reference needs its native point adapter")
            end
        end
        t[1]=="GEOVAR" || continue
        length(t)==6 && t[3] in ("ANC","SYM","RAD") && t[4] in ("XDIR","YDIR") &&
            t[5] in ("1","-1") && t[6] in ("NSCD","SCUNI","SCXY") ||
            _sonnet_error(p.source,row.line,"invalid GEOVAR dimension header")
        _sonnet_geovar_position(rows,index+1,p)
        length(t)>=2 && haskey(p.variables,t[2]) && get(quantities,t[2],"")=="LNG" ||
            _sonnet_error(p.source,row.line,"GEOVAR lacks its declared LNG quantity variable")
        nominalrow=_sonnet_geovar_row(rows,index+2,p)
        length(nominalrow.tokens)==2 && nominalrow.tokens[1]=="NOM" ||
            _sonnet_error(p.source,nominalrow.line,"GEOVAR requires a literal nominal dimension")
        nominal=tryparse(Float64,nominalrow.tokens[2])
        nominal!==nothing && isfinite(nominal) && nominal>=0 ||
            _sonnet_error(p.source,nominalrow.line,"GEOVAR saved nominal dimension must be finite and nonnegative")
        target=sonnet_variable_value(p,t[2];variables,freq=frequency)
        requested=target*p.length_scale;saved=nominal*p.length_scale
        ((iszero(nominal) && iszero(target)) ||
            (isfinite(requested) && requested>0 &&
                _sonnet_geovar_nominal_value_ok(saved,nominal,t[3],t[6]!="NSCD"))) ||
            _sonnet_error(p.source,row.line,"GEOVAR dimensions must preserve positive finite SI values")
        unchanged &= target==nominal
    end
    return unchanged
end

@inline function _sonnet_geovar_sum_error(first,second,total)
    tail=total-first
    return (first-(total-tail))+(second-tail)
end

# Native repeated moving references magnify every entry's displacement by
# r(r-1)+1; the implicit reference remains one additional entry. Subtract in
# native units before magnifying, so SI conversion cannot erase the change.
# Ordinary counts need no allocation; rare range cases use bounded precision.
@inline function _sonnet_geovar_repeated_delta(target,nominal,scale,repetitions::Int)
    if repetitions<=94906266
        factor=fma(Float64(repetitions),Float64(repetitions-1),1.)
        change=target-nominal
        delta=(change*factor)*scale
        isfinite(delta) && (!iszero(delta) || target==nominal) && return delta
    end
    return setprecision(BigFloat,4352) do
        setrounding(BigFloat,RoundNearest) do
            r=BigFloat(repetitions)
            Float64((BigFloat(target)-BigFloat(nominal))*(r*(r-1)+1)*BigFloat(scale))
        end
    end
end

# Constant ANC translations can combine an identity's repeated movements in
# native units. Converting once avoids SI addition roundoff changing ownership
# of a cell whose centre lies exactly on an axis-aligned edge. The saved SI
# coordinate remains authoritative when its native-unit roundtrip loses data.
@inline function _sonnet_geovar_repeated_coordinate(value,target,nominal,scale,references,movements,direction)
    native=value/scale
    if movements<=9007199254740992 && isfinite(native) && native*scale==value
        change=direction*_sonnet_geovar_repeated_delta(target,nominal,1.,references)
        total=fma(Float64(movements),change,native);after=total*scale
        safe=abs(total)>sqrt(eps(Float64))*max(abs(native),abs(movements*change))
        isfinite(after) && after!=value && safe && return after
    end
    return setprecision(BigFloat,4352) do
        setrounding(BigFloat,RoundNearest) do
            r=BigFloat(references)
            Float64(BigFloat(value)+direction*(BigFloat(target)-BigFloat(nominal))*
                (r*(r-1)+1)*BigFloat(movements)*BigFloat(scale))
        end
    end
end

# Retain midpoint and subtraction roundoff before magnifying an offset. Near
# one use the relative quantity change; strong contractions use the ratio.
# Rare range/cancellation cases use bounded, task-scoped precision.
@inline function _sonnet_geovar_scaled_coordinate(value,first,second,nominal,target,symmetric)
    left=first/2;right=second/2
    anchor=symmetric ? left+right : first
    anchor_error=symmetric ? _sonnet_geovar_sum_error(left,right,anchor) : 0.
    half_lost=symmetric && (2left!=first || 2right!=second)
    change=(target-nominal)/nominal;ratio=target/nominal;offset=value-anchor
    if isfinite(change) && isfinite(ratio) && ratio>0 && isfinite(offset) && !half_lost
        offset_error=_sonnet_geovar_sum_error(value,-anchor,offset)
        after=if abs(change)<=.5
            fma(offset_error-anchor_error,change,fma(offset,change,value))
        else
            fma(anchor_error,1-ratio,fma(offset_error,ratio,fma(offset,ratio,anchor)))
        end
        fixed=iszero(offset) && iszero(anchor_error)
        rounded_unchanged=iszero(offset) && after==value
        cancellation=abs(after)<=sqrt(eps(Float64))*max(abs(value),abs(anchor))
        isfinite(after) && (fixed || rounded_unchanged || target==nominal ||
            (!iszero(after) && after!=value && !cancellation)) && return after
    end
    return setprecision(BigFloat,4352) do
        setrounding(BigFloat,RoundNearest) do
            a=symmetric ? (BigFloat(first)+BigFloat(second))/2 : BigFloat(first)
            v=BigFloat(value);n=BigFloat(nominal);t=BigFloat(target)
            Float64(v+(v-a)*(t-n)/n)
        end
    end
end

@inline function _sonnet_geovar_fixed_coordinate(value,first,second,symmetric)
    symmetric || return value==first
    first==second && return value==first
    left=first/2;right=second/2;anchor=left+right
    if 2left==first && 2right==second
        return value==anchor && iszero(_sonnet_geovar_sum_error(left,right,anchor))
    end
    return setprecision(BigFloat,4352) do
        setrounding(BigFloat,RoundNearest) do
            BigFloat(first)+BigFloat(second)==2BigFloat(value)
        end
    end
end

# Native RAD moves every selected point by the same signed radial distance;
# its XDIR/YDIR, direction and scaling fields do not change that movement.
@inline function _sonnet_geovar_radial_point(x,y,anchorx,anchory,delta,preserve_movement::Bool=true)
    dx=x-anchorx;dy=y-anchory;radius=hypot(dx,dy)
    if isfinite(radius) && radius>0
        ux=dx/radius;uy=dy/radius
        nx=fma(ux,delta,x);ny=fma(uy,delta,y)
        safe(before,anchor,offset,unit,after)=isfinite(after) &&
            (iszero(offset) || (!iszero(unit) && (after!=before || !preserve_movement) &&
                abs(after)>sqrt(eps(Float64))*max(abs(before),abs(anchor),abs(delta))))
        safe(x,anchorx,dx,ux,nx) && safe(y,anchory,dy,uy,ny) && return (nx,ny)
    end
    return setprecision(BigFloat,4352) do
        setrounding(BigFloat,RoundNearest) do
            bx=BigFloat(x);by=BigFloat(y);bdx=bx-BigFloat(anchorx);bdy=by-BigFloat(anchory)
            bradius=hypot(bdx,bdy)
            iszero(bradius) && throw(ArgumentError("radial GEOVAR point coincides with its anchor"))
            change=BigFloat(delta)/bradius
            (Float64(bx+bdx*change),Float64(by+bdy*change))
        end
    end
end

# The solver applies a native radial parameter three times, recomputing the
# reference distance between adjustments. Away from anchor crossings the sum
# reduces to r(r-1)+1 times the initial change. Crossing an anchor requires the
# actual sequence; the distance is unsigned and each point's ray can reverse.
@inline function _sonnet_geovar_radial_delta(target,x,y,anchorx,anchory)
    radius=hypot(x-anchorx,y-anchory)
    delta=target-radius
    isfinite(radius) && isfinite(delta) && return delta
    return setprecision(BigFloat,4352) do
        setrounding(BigFloat,RoundNearest) do
            Float64(BigFloat(target)-hypot(BigFloat(x)-BigFloat(anchorx),
                BigFloat(y)-BigFloat(anchory)))
        end
    end
end

# Reduce the three adjustments algebraically only when every ray stays away
# from the anchor throughout all three passes. This prevents reference-radius
# rounding from being amplified by large multiplicities on a smooth path.
function _sonnet_geovar_radial_linear_path(polygons,points,owners,referencex,referencey,
        anchorx,anchory,target,nominal,references)
    hypot(referencex-anchorx,referencey-anchory)==nominal || return false
    change=target-nominal
    factor=fma(Float64(references),Float64(references-1),1.)
    minimum_step=min(0.,change,(1-references)*change,factor*change)
    isfinite(minimum_step) || return false
    for (id,index) in points
        radius=hypot(polygons[id][1,index]-anchorx,polygons[id][2,index]-anchory)
        shift=-owners[(id,index)][2]*minimum_step
        isfinite(radius) && isfinite(shift) &&
            radius+shift>sqrt(eps(Float64))*max(radius,abs(shift)) || return false
    end
    return true
end

function _sonnet_geovar_apply_radial!(p,parameter,target,original,polygons,owners)
    firstid,firstpoint=parameter.references[1]
    secondid,secondpoint=parameter.references[2]
    firstvertices=original[firstid].vertices
    # Accumulate in native units and convert once. Use SI only for
    # extreme saved coordinates whose native-unit quotient overflows.
    native_units=all(isfinite,(firstvertices[1,firstpoint]/p.length_scale,
        firstvertices[2,firstpoint]/p.length_scale))
    for (id,index) in parameter.points[2]
        if !all(isfinite,(polygons[id][1,index]/p.length_scale,
                polygons[id][2,index]/p.length_scale))
            native_units=false;break
        end
    end
    unit=native_units ? p.length_scale : 1.
    anchorx=firstvertices[1,firstpoint]/unit;anchory=firstvertices[2,firstpoint]/unit
    for (id,index) in parameter.points[2]
        owner,movements=owners[(id,index)]
        movements>0 || continue
        polygons[id][1,index]/=unit;polygons[id][2,index]/=unit
        owners[(id,index)]=(owner,-movements)
    end
    requested=target*(p.length_scale/unit)
    reference=polygons[secondid]
    linear=_sonnet_geovar_radial_linear_path(polygons,parameter.points[2],owners,
        reference[1,secondpoint],reference[2,secondpoint],anchorx,anchory,
        requested,parameter.nominal*(p.length_scale/unit),parameter.explicit_moving_references)
    step=_sonnet_geovar_repeated_delta(target,parameter.nominal,p.length_scale/unit,
        linear ? parameter.explicit_moving_references : 1)
    for pass in 1:(linear ? 1 : 3)
        if pass>1
            reference=polygons[secondid]
            step=_sonnet_geovar_radial_delta(requested,reference[1,secondpoint],
                reference[2,secondpoint],anchorx,anchory)
        end
        isfinite(step) || _sonnet_error(p.source,parameter.line,"radial GEOVAR displacement is unrepresentable")
        iszero(step) && continue
        for (id,index) in parameter.points[2]
            beforex=polygons[id][1,index];beforey=polygons[id][2,index]
            afterx,aftery=_sonnet_geovar_radial_point(beforex,beforey,anchorx,anchory,step,pass==1)
            # A recomputed radius can differ by one rounding unit.
            # Native correction passes may then leave a coordinate
            # unchanged; the requested initial movement must survive.
            all(isfinite,(afterx,aftery)) && (pass>1 ||
                ((afterx!=beforex || beforex==anchorx) &&
                 (aftery!=beforey || beforey==anchory))) ||
                _sonnet_error(p.source,parameter.line,"radial GEOVAR displacement is lost in its stored coordinate")
            polygons[id][1,index]=afterx;polygons[id][2,index]=aftery
        end
    end
    for (id,index) in parameter.points[2]
        owner,movements=owners[(id,index)]
        movements<0 || continue
        polygons[id][1,index]*=unit;polygons[id][2,index]*=unit
        all(isfinite,(polygons[id][1,index],polygons[id][2,index])) ||
            _sonnet_error(p.source,parameter.line,"radial GEOVAR coordinate is unrepresentable in SI")
        owners[(id,index)]=(owner,-movements)
    end
    return nothing
end

"""Resolve native independent ANC/SYM/RAD dimensions into effective SI geometry.
Original source/records and scalar snapshot identity remain attached. Reference
points belong implicitly to their adjustable set. Ordinary repeated entries
retain their movement multiplicity. RAD follows the native three-adjustment
sequence, recomputing its reference radius. NSCD ANC moving-reference repetitions
apply their native common-displacement factor; ANC follows its reference
coordinate order, using the direction field for zero offsets. Other references,
dependent/overlapping active dimensions and moved component/interior-port semantics
require separate adapters. Whole-polygon selectors expand logical vertices
within point/storage budgets before geometry copy;
zero saved ANC/NSCD offsets require coincident reference coordinates on their axis;
effective coordinates are validated before emission. The supplied project is
never modified."""
function _sonnet_geometry_project(p::SonnetProject,freq::Real,variables=Dict{String,Float64}();
        max_parameters::Integer=1024,max_points::Integer=100000,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    count=Base.count(r->r.tokens[1]=="GEOVAR",p.records);iszero(count) && return p
    nparameters=_sonnet_geovar_limit(max_parameters,"max_parameters")
    npoints=_sonnet_geovar_limit(max_points,"max_points")
    limit=_sonnet_geovar_limit(max_bytes,"max_bytes")
    count<=nparameters || throw(ArgumentError("GEOVAR parameter budget exceeded"))
    reserve=_checked_payload_sum("native geometry variable workspace",_sonnet_scalar_project_payload(p),
        _checked_array_payload_bytes(UInt8,256,length(p.records)),
        _checked_array_payload_bytes(UInt8,1024,count))
    _enforce_payload_limit(reserve,limit,"native geometry variable workspace","max_bytes")
    frequency=_circuit_stored_real(freq,"native geometry variable frequency")
    frequency>0 || throw(ArgumentError("native geometry variable frequency must be positive"))
    _sonnet_geovar_nominal_geometry(p,frequency,variables) && return p
    # Whole-polygon selectors expand beyond their source record count. Reserve
    # their point-list and ownership workspace before materializing those lists.
    parameters=_sonnet_geovar_parameters(p,nparameters,npoints,(limit-reserve)÷256)
    deltas=Float64[];targets=Float64[]
    for parameter in parameters
        target=sonnet_variable_value(p,parameter.name;variables,freq=frequency)::Float64
        nominal=parameter.nominal*p.length_scale;requested=target*p.length_scale
        _sonnet_geovar_nominal_value_ok(nominal,parameter.nominal,parameter.kind,parameter.scaled) &&
            isfinite(requested) && (requested>0 ||
                (iszero(target) && iszero(parameter.nominal) && iszero(requested))) ||
            _sonnet_error(p.source,parameter.line,"GEOVAR dimensions must preserve positive finite SI values")
        displacement=parameter.kind=="RAD" ?
            _sonnet_geovar_repeated_delta(target,parameter.nominal,p.length_scale,1) :
            parameter.explicit_moving_references>1 ?
            _sonnet_geovar_repeated_delta(target,parameter.nominal,p.length_scale,
                parameter.explicit_moving_references) : requested-nominal
        delta=(parameter.kind=="RAD" ? 1 : parameter.direction)*displacement
        isfinite(delta) || _sonnet_error(p.source,parameter.line,"GEOVAR displacement is unrepresentable")
        target==parameter.nominal || !iszero(delta) ||
            _sonnet_error(p.source,parameter.line,"GEOVAR displacement is lost after SI conversion")
        parameter.kind=="SYM" && !iszero(delta) && iszero(delta/2) &&
            _sonnet_error(p.source,parameter.line,"symmetric GEOVAR displacement underflows")
        push!(deltas,delta)
        push!(targets,target)
    end
    all(iszero,deltas) && return p
    isempty(p.components) || throw(ArgumentError("moved GEOVAR component pins require their physical placement adapter"))
    original=Dict(q.id=>q for q in p.polygons)
    all(port->port.kind in (:box,:std) && haskey(original,port.polygon) &&
        original[port.polygon].kind==:sheet,p.ports) ||
        throw(ArgumentError("moved GEOVAR interior/via ports require their attachment adapter"))
    for port in p.ports
        vertices=original[port.polygon].vertices
        closed=vertices[1,1]==vertices[1,end] && vertices[2,1]==vertices[2,end]
        0<=port.edge<size(vertices,2)-Int(closed) ||
            throw(ArgumentError("GEOVAR port edge is outside its polygon's logical edges"))
        length(port.values)>=7 || throw(ArgumentError("GEOVAR port lacks its attached coordinate"))
    end
    owners=Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for (i,parameter) in enumerate(parameters)
        iszero(deltas[i]) && continue
        for point in Iterators.flatten(parameter.points)
            haskey(owners,point) && owners[point][1]!=i && _sonnet_error(p.source,parameter.line,
                "overlapping active GEOVAR dimensions require their ordered geometry adapter")
            movements=get(owners,point,(i,0))[2]
            owners[point]=(i,movements+1)
        end
    end
    for (i,parameter) in enumerate(parameters),point in parameter.references
        haskey(owners,point) && owners[point][1]!=i && _sonnet_error(p.source,parameter.line,
            "dependent GEOVAR dimensions require their ordered geometry adapter")
    end
    polygons=Dict(q.id=>copy(q.vertices) for q in p.polygons)
    for (parameter,delta,target) in zip(parameters,deltas,targets)
        iszero(delta) && continue
        firstid,firstpoint=parameter.references[1]
        secondid,secondpoint=parameter.references[2]
        firstvertices=original[firstid].vertices;secondvertices=original[secondid].vertices
        if parameter.kind=="ANC" && !parameter.scaled && parameter.explicit_moving_references>1
            for point in parameter.points[2]
                owner,movements=owners[point]
                movements>0 || continue
                id,index=point;before=polygons[id][parameter.axis,index]
                after=_sonnet_geovar_repeated_coordinate(before,target,parameter.nominal,p.length_scale,
                    parameter.explicit_moving_references,movements,parameter.direction)
                isfinite(after) && after!=before ||
                    _sonnet_error(p.source,parameter.line,"GEOVAR displacement is lost in its stored coordinate")
                polygons[id][parameter.axis,index]=after
                owners[point]=(owner,-movements)
            end
            continue
        end
        if parameter.kind=="RAD"
            _sonnet_geovar_apply_radial!(p,parameter,target,original,polygons,owners)
            continue
        end
        symmetric=parameter.kind=="SYM"
        for axis in 1:2
            parameter.both_axes || axis==parameter.axis || continue
            first=firstvertices[axis,firstpoint];second=secondvertices[axis,secondpoint]
            anchor=symmetric ? first/2+second/2 : first
            for side in 1:2
                shift=symmetric ? (side==1 ? -delta/2 : delta/2) : delta
                for (id,index) in parameter.points[side]
                    before=polygons[id][axis,index]
                    after=parameter.scaled ? _sonnet_geovar_scaled_coordinate(before,first,second,
                        parameter.nominal,target,symmetric) : before+shift
                    isfinite(after) && (after!=before || (parameter.scaled ?
                        (before==anchor || _sonnet_geovar_fixed_coordinate(before,first,second,symmetric)) : iszero(shift))) ||
                        _sonnet_error(p.source,parameter.line,"GEOVAR displacement is lost in its stored coordinate")
                    polygons[id][axis,index]=after
                end
            end
        end
    end
    emitted=SonnetPolygon[]
    for polygon in p.polygons
        vertices=polygons[polygon.id]
        polygon.vertices[1,1]==polygon.vertices[1,end] && polygon.vertices[2,1]==polygon.vertices[2,end] &&
            (vertices[:,end].=vertices[:,1])
        planar_normalize_polygon([_P2(vertices[1,i],vertices[2,i]) for i in axes(vertices,2)];
            label="GEOVAR polygon $(polygon.id)")
        push!(emitted,SonnetPolygon(polygon.kind,polygon.level,polygon.material,polygon.id,
            vertices,polygon.target,polygon.technology,polygon.flags))
    end
    ports=SonnetPortSpec[]
    parseda=tryparse(Float64,p.box[2]);parsedb=tryparse(Float64,p.box[3])
    boxa=(parseda===nothing ? sonnet_variable_value(p,p.box[2];variables,freq=frequency) : parseda)::Float64
    boxb=(parsedb===nothing ? sonnet_variable_value(p,p.box[3];variables,freq=frequency) : parsedb)::Float64
    boxa*=p.length_scale;boxb*=p.length_scale
    all(isfinite,(boxa,boxb)) && boxa>0 && boxb>0 ||
        throw(ArgumentError("GEOVAR box dimensions must preserve positive finite SI values"))
    halfcounts=(parse(Int,p.box[4]),parse(Int,p.box[5]))
    all(n->n>0 && iseven(n),halfcounts) ||
        throw(ArgumentError("native BOX half-cell counts must be positive even integers"))
    native_nx=halfcounts[1]÷2;native_ny=halfcounts[2]÷2
    for port in p.ports
        new=polygons[port.polygon]
        first,second=_sonnet_port_edge_indices(new,port.edge)
        _sonnet_box_port_edge_inside(new,port.edge,boxa,boxb,native_nx,native_ny) ||
            throw(ArgumentError("GEOVAR box-port edge is partially or entirely outside the box"))
        (new[1,first]!=new[1,second] || new[2,first]!=new[2,second]) ||
            throw(ArgumentError("GEOVAR port has a degenerate edge"))
        values=copy(port.values)
        for component in 1:2
            # Native ReadWrite resets annotations to the referenced edge midpoint.
            coordinate=(new[component,first]/2+new[component,second]/2)/p.length_scale
            isfinite(coordinate) || throw(ArgumentError("GEOVAR port coordinate is unrepresentable"))
            values[5+component]=string(coordinate)
        end
        push!(ports,SonnetPortSpec(port.kind,port.polygon,port.edge,port.number,values,port.records))
    end
    return SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,p.metals,
        p.top,p.bottom,emitted,ports,p.variables,p.components,p.sweeps,p.records)
end
