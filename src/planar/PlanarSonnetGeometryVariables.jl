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
    position::NTuple{2,Float64}
    crossing::Bool
end

# Native zero-offset dimensions are legal for every kind and header
# (18.53-verified: unscaled ANC/SYM/RAD and scaled SYM intake and solve
# with NOM 0); the geometric span - not NOM - is the delta baseline.
# A negative saved dimension remains invalid.
@inline _sonnet_geovar_nominal_value_ok(value)=isfinite(value) && value>=0

# Native anchored dimensions resolve sequentially in dependency order,
# proved against the installed engine on active/inactive, consistent/
# inconsistent and dependent chains alike: every unscaled ANC dimension
# re-anchors its moving reference at REF1_current + direction*target,
# then shifts every PS2 listing by that reference's displacement. A
# dimension whose references are moved by another dimension resolves
# after it, regardless of file order. Scaled and repeated
# moving-reference dimensions keep their own adapters and still reject
# cross-dimension interactions.
@inline _sonnet_geovar_anc_sequential(parameter)=
    parameter.kind=="ANC" && !parameter.scaled &&
        parameter.explicit_moving_references<=1

# Topological order of sequential anchored dimensions: a dimension depends
# on every other sequential dimension whose moved point set contains one
# of its references. Circular dependencies match the native circular
# definition error. Independent dimensions keep record order.
function _sonnet_geovar_sequential_order(parameters,source)
    ids=[i for (i,parameter) in enumerate(parameters) if
        _sonnet_geovar_anc_sequential(parameter)]
    deps=Dict(i=>Set{Int}() for i in ids)
    for i in ids, point in parameters[i].references, j in ids
        i!=j && point in parameters[j].points[2] &&
            (parameters[i].axis==parameters[j].axis ||
                parameters[j].kind=="RAD" || parameters[j].both_axes ||
                parameters[i].kind=="RAD" || parameters[i].both_axes) &&
            push!(deps[i],j)
    end
    order=Int[];pending=copy(ids)
    while !isempty(pending)
        advanced=false
        for i in copy(pending)
            all(j->j in order,deps[i]) || continue
            push!(order,i);deleteat!(pending,findfirst(==(i),pending));advanced=true
        end
        advanced || _sonnet_error(source,parameters[first(pending)].line,
            "circular dependent GEOVAR dimensions")
    end
    return order
end

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
    length(row.tokens)==3 && _sonnet_ieq(row.tokens[1],"POS") ||
        _sonnet_error(p.source,row.line,"invalid GEOVAR display position")
    position=ntuple(i->tryparse(Float64,row.tokens[i+1]),2)
    all(v->v!==nothing && isfinite(v),position) ||
        _sonnet_error(p.source,row.line,"invalid GEOVAR display position")
    return something.(position)
end

function _sonnet_geovar_groups(row,p)
    length(row.tokens)==2 || _sonnet_error(p.source,row.line,"invalid GEOVAR point-set group count")
    groups=tryparse(Int,row.tokens[2])
    groups!==nothing && groups>=0 || _sonnet_error(p.source,row.line,"invalid GEOVAR point-set group count")
    return groups
end

function _sonnet_geovar_point(p,polygons,row,index)
    t=row.tokens
    length(t)==3 && _sonnet_ieq(t[1],"POLY") || _sonnet_error(p.source,row.line,"GEOVAR requires a literal POLY point identity")
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
    length(t)==4 && _sonnet_ieq(t[1],tag) && _sonnet_ieq(t[2],"POLY") && t[4]=="1" ||
        _sonnet_error(p.source,row.line,"GEOVAR $tag requires exactly one literal polygon point")
    pointrow=_sonnet_geovar_row(rows,index+1,p)
    length(pointrow.tokens)==1 || _sonnet_error(p.source,pointrow.line,"invalid GEOVAR reference point")
    point=_sonnet_geovar_point(p,polygons,SonnetRecord(row.line,t[2:4]),only(pointrow.tokens))
    return point,index+2
end

function _sonnet_geovar_set(rows,index,p,polygons,tag,max_points,used,max_expanded,expanded)
    row=_sonnet_geovar_row(rows,index,p);t=row.tokens
    length(t)==2 && _sonnet_ieq(t[1],tag) || _sonnet_error(p.source,row.line,"GEOVAR requires $tag")
    groups=_sonnet_geovar_groups(row,p)
    groups<=length(rows)-index || _sonnet_error(p.source,row.line,"invalid GEOVAR point-set group count")
    points=Tuple{Int,Int}[];index+=1;wildcard=false
    for _ in 1:groups
        row=_sonnet_geovar_row(rows,index,p);t=row.tokens
        length(t)==3 && _sonnet_ieq(t[1],"POLY") || _sonnet_error(p.source,row.line,"unsupported GEOVAR point-set identity")
        count=tryparse(Int,t[3])
        count!==nothing && 0<=count<=length(rows)-index || _sonnet_error(p.source,row.line,"invalid GEOVAR polygon point count")
        if iszero(count)
            wildcard=true
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
    (r->length(r.tokens)==1 && _sonnet_ieq(r.tokens[1],"END"))(_sonnet_geovar_row(rows,index,p)) ||
        _sonnet_error(p.source,rows[index].line,"unterminated GEOVAR point set")
    # Native entries are movements: a repeated identity moves repeatedly.
    return points,index+1,used,expanded,wildcard
end

# Native header grammar (Lite 18.53, bit-exact verified): kind and axis are
# positional and case-insensitive; the remaining tokens are scanned in
# order, where the first NSCD/SCUNI/SCXY keyword sets scaling and the first
# numeric token sets the display direction. Unknown tokens reject while
# scaling is unset and are ignored once it appears; missing fields default
# to NSCD and +1.
function _sonnet_geovar_header(t,source,line)
    kind=length(t)>=3 ? uppercase(t[3]) : ""
    axiskw=length(t)>=4 ? uppercase(t[4]) : ""
    kind in ("ANC","SYM","RAD") && axiskw in ("XDIR","YDIR") ||
        _sonnet_error(source,line,"GEOVAR requires an ANC/SYM/RAD XDIR/YDIR point-set adapter")
    scaling=nothing;direction=1;dirseen=false
    for token in Iterators.drop(t,4)
        keyword=uppercase(token)
        if scaling===nothing && keyword in ("NSCD","SCUNI","SCXY")
            scaling=keyword;continue
        end
        value=tryparse(Float64,token)
        if !dirseen && value!==nothing && isfinite(value)
            dirseen=true;direction=value>=0 ? 1 : -1;continue
        end
        scaling===nothing || continue
        _sonnet_error(source,line,
            "GEOVAR requires an ANC/SYM/RAD XDIR/YDIR point-set adapter")
    end
    return kind,axiskw,direction,scaling===nothing ? "NSCD" : scaling
end

function _sonnet_geovar_parameters(p,max_parameters,max_points,max_expanded)
    rows=_sonnet_section(p.records,"GEO",p.source)
    polygons=Dict(q.id=>q for q in p.polygons)
    quantities=Dict(r.tokens[2]=>r.tokens[3] for r in rows if length(r.tokens)>=4 && _sonnet_ieq(r.tokens[1],"VALVAR"))
    length(polygons)==length(p.polygons) || throw(ArgumentError("duplicate GEOVAR polygon identities"))
    parameters=_SonnetGeometryParameter[];index=1;used=0;expanded=0
    while index<=length(rows)
        row=rows[index];t=row.tokens
        if !_sonnet_ieq(t[1],"GEOVAR");index+=1;continue;end
        length(parameters)<max_parameters || _sonnet_error(p.source,row.line,"GEOVAR parameter budget exceeded")
        kind,axiskw,direction,scale=_sonnet_geovar_header(t,p.source,row.line)
        name=t[2];haskey(p.variables,name) || _sonnet_error(p.source,row.line,"GEOVAR lacks its declared quantity variable")
        _sonnet_ieq(get(quantities,name,""),"LNG") ||
            _sonnet_error(p.source,row.line,"GEOVAR variable must have one LNG declaration")
        position=_sonnet_geovar_position(rows,index+1,p)
        nominalrow=_sonnet_geovar_row(rows,index+2,p)
        length(nominalrow.tokens)==2 && _sonnet_ieq(nominalrow.tokens[1],"NOM") ||
            _sonnet_error(p.source,nominalrow.line,"GEOVAR requires a literal nominal dimension")
        nominal=tryparse(Float64,nominalrow.tokens[2])
        nominal!==nothing && _sonnet_geovar_nominal_value_ok(nominal) ||
            _sonnet_error(p.source,nominalrow.line,"GEOVAR nominal dimension must be finite and nonnegative")
        # The optional EQN record is scanned like the native engine: it can
        # sit directly after NOM or after REF2 (both positions verified).
        equation=_sonnet_geovar_row(rows,index+3,p)
        ref1index=index+3
        if _sonnet_ieq(equation.tokens[1],"EQN")
            length(equation.tokens)==2 && isequal(_sonnet_parse_scalar(equation.tokens[2]),
                _sonnet_parse_scalar(p.variables[name])) ||
                _sonnet_error(p.source,equation.line,"GEOVAR EQN disagrees with its quantity definition")
            ref1index+=1
        end
        first,index=_sonnet_geovar_reference(rows,ref1index,p,polygons,"REF1")
        second,index=_sonnet_geovar_reference(rows,index,p,polygons,"REF2")
        axis=axiskw=="XDIR" ? 1 : 2
        used<=max_points-2 || _sonnet_error(p.source,row.line,"GEOVAR reference budget exceeded")
        used+=2
        equation=_sonnet_geovar_row(rows,index,p)
        if _sonnet_ieq(equation.tokens[1],"EQN")
            length(equation.tokens)==2 && isequal(_sonnet_parse_scalar(equation.tokens[2]),
                _sonnet_parse_scalar(p.variables[name])) ||
                _sonnet_error(p.source,equation.line,"GEOVAR EQN disagrees with its quantity definition")
            index+=1
        end
        a,index,used,expanded,awild=_sonnet_geovar_set(rows,index,p,polygons,"PS1",max_points,used,max_expanded,expanded)
        b,index,used,expanded,bwild=_sonnet_geovar_set(rows,index,p,polygons,"PS2",max_points,used,max_expanded,expanded)
        explicit_references=0
        for point in b
            explicit_references+=Int(point==second)
        end
        # The implicit moving reference is an additional movement for native
        # unscaled anchored dimensions and radial dimensions under any header.
        # Scaled anchored dimensions remain rejected: the retained native
        # pairs falsify composed scaled movements, additive deltas and
        # deduplicated hypotheses alike, and fresh sweeps show scaled
        # moving-reference projects rasterize through a different mesh path
        # entirely (native SUBS counts match no literal candidate, leaving
        # 1.6e-4–2.6e-4 residual floors after sweeping every vertex and port
        # degree of freedom). The exact movement law is still undecoded and
        # accepting an approximation would silently rasterize the wrong
        # cells for other target widths. Unscaled symmetric
        # repetitions support literal lists with one or two explicit
        # moving-reference entries plus whole-polygon PS2 selectors; the
        # latter are native-verified only when the expanded set crosses
        # the symmetry plane through REF1's own vertex with a single
        # REF2 entry (whole-polygon sets always contain that vertex).
        # Wildcard PS1 selectors, crossing lists lacking REF1, crossing
        # repetitions r>=2 and all r>=3 repetitions remain rejected: their
        # native geometry is not literal-expressible (flat residual basins
        # persist after sweeping every vertex, port and rasterization
        # degree of freedom).
        posaxis=position[axis]*p.length_scale
        ref2side=polygons[second[1]].vertices[axis,second[2]]-posaxis
        crossing=!iszero(ref2side) && any(point->
            (polygons[point[1]].vertices[axis,point[2]]-posaxis)*ref2side<0,b)
        sym_moving=kind=="SYM" && scale=="NSCD" && !awild &&
            (!crossing && !bwild && explicit_references in (1,2) ||
            crossing && explicit_references==1 && first in b)
        (first in a || (explicit_references>0 && !(kind=="RAD" ||
            (kind=="ANC" && scale=="NSCD") || sym_moving))) &&
            _sonnet_error(p.source,row.line,
            "explicit GEOVAR reference repetitions require their native movement adapter")
        terminator=_sonnet_geovar_row(rows,index,p)
        (length(terminator.tokens)==1 && _sonnet_ieq(terminator.tokens[1],"END")) ||
            _sonnet_error(p.source,terminator.line,"unterminated GEOVAR block")
        index+=1
        kind in ("ANC","RAD") && !isempty(a) && _sonnet_error(p.source,row.line,"anchored/radial GEOVAR has an unsupported moving anchor set")
        kind=="SYM" && !(first in a) && push!(a,first)
        push!(b,second)
        bset=Set(b)
        (first==second || !sym_moving &&
            (any(point->point in bset,a) || first in bset)) &&
            _sonnet_error(p.source,row.line,"GEOVAR adjustable sets conflict with their references")
        fixed=polygons[first[1]].vertices[axis,first[2]]
        moving=polygons[second[1]].vertices[axis,second[2]]
        # Every dimension resolves along its saved reference span: the
        # display direction field is ignored whenever that span is nonzero
        # (bit-exact verified on ANC, SYM and both-axes dims for direction
        # fields in -2..3). Only a zero span falls back to the field's
        # sign, where nonnegative values behave as +1.
        iszero(moving-fixed) || (direction=moving>fixed ? 1 : -1)
        push!(parameters,_SonnetGeometryParameter(name,kind,axis,
            direction,scale!="NSCD",scale=="SCXY",nominal,(first,second),(a,b),
            explicit_references,row.line,position,crossing))
    end
    return parameters
end

# Native files also retain scaled/dependent dimensions whose saved
# coordinates are usable unchanged. Only exact equality with every saved NOM
# permits this path; an effective change still requires the proved adapter.
function _sonnet_geovar_nominal_geometry(p,frequency,variables)
    rows=_sonnet_section(p.records,"GEO",p.source)
    polygons=Dict(q.id=>q for q in p.polygons)
    quantities=Dict(r.tokens[2]=>r.tokens[3] for r in rows if length(r.tokens)>=4 && _sonnet_ieq(r.tokens[1],"VALVAR"))
    unchanged=true
    for (index,row) in enumerate(rows)
        t=row.tokens
        (_sonnet_ieq(t[1],"PS1") || _sonnet_ieq(t[1],"PS2")) && _sonnet_geovar_groups(row,p)
        if (_sonnet_ieq(t[1],"REF1") || _sonnet_ieq(t[1],"REF2")) && length(t)==4 &&
                _sonnet_ieq(t[2],"POLY") && t[4]=="1"
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
        _sonnet_ieq(t[1],"GEOVAR") || continue
        _sonnet_geovar_header(t,p.source,row.line)
        _sonnet_geovar_position(rows,index+1,p)
        length(t)>=2 && haskey(p.variables,t[2]) && _sonnet_ieq(get(quantities,t[2],""),"LNG") ||
            _sonnet_error(p.source,row.line,"GEOVAR lacks its declared LNG quantity variable")
        nominalrow=_sonnet_geovar_row(rows,index+2,p)
        length(nominalrow.tokens)==2 && _sonnet_ieq(nominalrow.tokens[1],"NOM") ||
            _sonnet_error(p.source,nominalrow.line,"GEOVAR requires a literal nominal dimension")
        nominal=tryparse(Float64,nominalrow.tokens[2])
        nominal!==nothing && isfinite(nominal) && nominal>=0 ||
            _sonnet_error(p.source,nominalrow.line,"GEOVAR saved nominal dimension must be finite and nonnegative")
        target=sonnet_variable_value(p,t[2];variables,freq=frequency)
        requested=target*p.length_scale;saved=nominal*p.length_scale
        isfinite(requested) && _sonnet_geovar_nominal_value_ok(saved) ||
            _sonnet_error(p.source,row.line,"GEOVAR dimensions must preserve finite nonnegative SI values")
        # NOM equality with the target marks a dimension as nominal, which
        # only cheapens the all-nominal parse shortcut; the native engine
        # still resolves any saved-geometry delta on inactive dimensions
        # (18.53-verified collapse-to-target), so activity alone cannot
        # decide whether the saved project is returned unchanged.
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

function _sonnet_geovar_apply_radial!(p,parameter,target,gnominal,original,polygons,owners)
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
        requested,gnominal*(p.length_scale/unit),parameter.explicit_moving_references)
    step=_sonnet_geovar_repeated_delta(target,gnominal,p.length_scale/unit,
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
coordinate order, using the direction field for zero offsets. NSCD SYM
moving references apply the native cell-quantized movement law proved against
the installed engine on both axes. Unscaled anchored dimensions resolve
sequentially in their native dependency order, re-anchoring each moving
reference at REF1+direction*target, so dependent anchored chains and
inactive constraint dimensions apply identically; circular definitions are
rejected like the native engine. Via ports re-anchor to their resolved edge
midpoint like sheet ports. Dimensions share references, cross-axis
moved points and opposite PS sides freely like the native engine, while
same-axis same-side overlaps between non-sequential dimensions and
same-axis non-sequential dependencies reject as circular. Other
references and moved component/gap-port semantics require separate
adapters. Whole-polygon selectors expand logical vertices
within point/storage budgets before geometry copy; zero saved offsets are
legal for every kind and resolve against the geometric span like the
native engine; effective coordinates are validated before emission. The
supplied project is never modified."""
function _sonnet_geometry_project(p::SonnetProject,freq::Real,variables=Dict{String,Float64}();
        max_parameters::Integer=1024,max_points::Integer=100000,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    count=Base.count(r->_sonnet_ieq(r.tokens[1],"GEOVAR"),p.records);iszero(count) && return p
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
    # Whole-polygon selectors expand beyond their source record count. Reserve
    # their point-list and ownership workspace before materializing those lists.
    unchanged=_sonnet_geovar_nominal_geometry(p,frequency,variables)
    parameters=if unchanged
        try
            _sonnet_geovar_parameters(p,nparameters,npoints,(limit-reserve)÷256)
        catch e
            # A fully inactive project keeps its saved geometry even when
            # its dimensions need an unsupported point-set adapter.
            e isa ArgumentError || rethrow()
            return p
        end
    else
        _sonnet_geovar_parameters(p,nparameters,npoints,(limit-reserve)÷256)
    end
    # Lite's structural conflict rules are intake-level (18.53-verified):
    # they fire even when every dimension is inactive. Dimensions share
    # reference points freely, moved points across different axes commute
    # and resolve independently, and moved points on opposite PS sides of
    # the same axis expand in opposite directions. A moved point shared
    # on the same axis and the same PS side is a circular definition
    # between two non-sequential dimensions; anchored pairs keep their
    # proved dependency order and share freely.
    seq(i)=_sonnet_geovar_anc_sequential(parameters[i])
    sameaxis(i,j)=parameters[i].axis==parameters[j].axis ||
        parameters[i].kind=="RAD" || parameters[j].kind=="RAD" ||
        parameters[i].both_axes || parameters[j].both_axes
    # Every point that can move joins the moved set with the PS side that
    # moves it: symmetric dimensions shift every listed point (REF2
    # always, REF1 when gated); anchored and radial dimensions keep REF1
    # as the fixed anchor and only their PS2 side - including the
    # re-anchoring REF2 - moves.
    movers=Dict{Tuple{Int,Int},Vector{Tuple{Int,Symbol}}}()
    for (i,parameter) in enumerate(parameters)
        for (side,points) in zip((:a,:b),
                parameter.kind=="SYM" ? parameter.points :
                    (parameters[i].points[1][1:0],parameter.points[2]))
            for point in points
                push!(get!(movers,point,Tuple{Int,Symbol}[]),(i,side))
            end
        end
    end
    for (point,entries) in movers,(i,aside) in entries,(j,bside) in entries
        i<j && aside==bside && !seq(i) && !seq(j) && sameaxis(i,j) &&
            _sonnet_error(p.source,parameters[j].line,
                "overlapping GEOVAR dimensions require their ordered geometry adapter")
    end
    # A non-sequential dimension whose reference lies on a moved point
    # along its own axis is an unorderable dependency - the native
    # circular-definition error fires on one-way and mutual crossings
    # alike. Cross-axis references never change the measured span and are
    # inert; sequential anchored dimensions may depend on anything
    # because they resolve last in dependency order.
    for (i,parameter) in enumerate(parameters)
        seq(i) && continue
        any(point->haskey(movers,point) &&
            any(entry->entry[1]!=i && sameaxis(i,entry[1]),movers[point]),
            parameter.references) &&
            _sonnet_error(p.source,parameter.line,
                "dependent GEOVAR dimensions require their ordered geometry adapter")
    end
    original=Dict(q.id=>q for q in p.polygons)
    deltas=Float64[];targets=Float64[];gnominals=Float64[];moves=falses(length(parameters))
    for (i,parameter) in enumerate(parameters)
        target=sonnet_variable_value(p,parameter.name;variables,freq=frequency)::Float64
        nominal=parameter.nominal*p.length_scale;requested=target*p.length_scale
        _sonnet_geovar_nominal_value_ok(nominal) && isfinite(requested) ||
            _sonnet_error(p.source,parameter.line,"GEOVAR dimensions must preserve finite nonnegative SI values")
        # Once active, the native engine resolves a dimension against the
        # reference points' geometric span, not the file's NOM field
        # (verified by running identical projects with inconsistent NOM
        # values). ANC measures the signed span along its movement direction;
        # SYM uses the absolute span regardless of reference order; RAD and
        # both-axis dims use the reference distance. NOM only documents the
        # original span.
        firstid,firstpoint=parameter.references[1]
        secondid,secondpoint=parameter.references[2]
        fv=original[firstid].vertices;sv=original[secondid].vertices
        gnominal=(parameter.kind=="RAD" || parameter.both_axes ?
            hypot(sv[1,secondpoint]-fv[1,firstpoint],
                sv[2,secondpoint]-fv[2,firstpoint]) :
            parameter.kind=="SYM" ?
            abs(sv[parameter.axis,secondpoint]-fv[parameter.axis,firstpoint]) :
            parameter.direction*(sv[parameter.axis,secondpoint]-
                fv[parameter.axis,firstpoint]))/p.length_scale
        push!(gnominals,gnominal)
        displacement=parameter.kind=="RAD" ?
            _sonnet_geovar_repeated_delta(target,gnominal,p.length_scale,1) :
            parameter.kind=="ANC" && parameter.explicit_moving_references>1 ?
            _sonnet_geovar_repeated_delta(target,gnominal,p.length_scale,
                parameter.explicit_moving_references) : requested-gnominal*p.length_scale
        delta=(parameter.kind=="RAD" ? 1 : parameter.direction)*displacement
        isfinite(delta) || _sonnet_error(p.source,parameter.line,"GEOVAR displacement is unrepresentable")
        target==gnominal || !iszero(delta) ||
            _sonnet_error(p.source,parameter.line,"GEOVAR displacement is lost after SI conversion")
        parameter.kind=="SYM" && !iszero(delta) && iszero(delta/2) &&
            _sonnet_error(p.source,parameter.line,"symmetric GEOVAR displacement underflows")
        # The native engine resolves every dimension's real displacement
        # regardless of NOM activity - an inconsistent inactive project
        # still collapses/expands to its targets (18.53-verified), so the
        # saved project is returned only when every law is a no-op.
        moves[i]=_sonnet_geovar_anc_sequential(parameter) ?
            abs(fv[parameter.axis,firstpoint]+parameter.direction*requested-
                sv[parameter.axis,secondpoint])>
                sqrt(eps(Float64))*max(abs(requested),abs(gnominal*p.length_scale)) :
            abs(delta)>sqrt(eps(Float64))*max(abs(requested),abs(gnominal*p.length_scale))
        push!(deltas,delta)
        push!(targets,target)
    end
    !any(moves) && return p
    # A moved point shared on the same axis between an anchored and a
    # non-anchored dimension resolves natively, but the ordering of that
    # joint application is still undecoded; it stays rejected while both
    # dimensions actually displace geometry.
    for (point,entries) in movers,i in first.(entries),j in first.(entries)
        i<j && seq(i)!=seq(j) && sameaxis(i,j) && moves[i] && moves[j] &&
            _sonnet_error(p.source,parameters[j].line,
                "overlapping active GEOVAR dimensions require their ordered geometry adapter")
    end
    # Sheet and via ports re-anchor to their referenced edge midpoint
    # after resolution; the POR1 kind token is descriptive (Lite types by
    # placement), so every non-CUP token qualifies by polygon kind.
    # Lite 18.53 accepts STD and GAP tokens on a moved interior edge
    # identically; only the CUP calibration-group port remains a residual.
    all(port->haskey(original,port.polygon) &&
        original[port.polygon].kind in (:sheet,:via),p.ports) ||
        throw(ArgumentError("moved GEOVAR interior/via ports require their attachment adapter"))
    any(port->port.kind===:cup && haskey(original,port.polygon),p.ports) &&
        throw(ArgumentError("native CUP co-calibrated port requires its calibration-group adapter"))
    for port in p.ports
        vertices=original[port.polygon].vertices
        closed=vertices[1,1]==vertices[1,end] && vertices[2,1]==vertices[2,end]
        0<=port.edge<size(vertices,2)-Int(closed) ||
            throw(ArgumentError("GEOVAR port edge is outside its polygon's logical edges"))
        length(port.values)>=7 || throw(ArgumentError("GEOVAR port lacks its attached coordinate"))
    end
    owners=Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for (i,parameter) in enumerate(parameters)
        for point in Iterators.flatten(parameter.points)
            movements=get(owners,point,(i,0))[2]
            owners[point]=(i,movements+1)
        end
    end
    polygons=Dict(q.id=>copy(q.vertices) for q in p.polygons)
    # Non-anchored dimensions only share moved geometry across different
    # axes or opposite PS sides, where record order is safe; anchored
    # dimensions resolve last in their native dependency order.
    applyorder=[i for (i,parameter) in enumerate(parameters) if
        !_sonnet_geovar_anc_sequential(parameter)]
    append!(applyorder,_sonnet_geovar_sequential_order(parameters,p.source))
    for i in applyorder
        parameter,delta,target=parameters[i],deltas[i],targets[i]
        firstid,firstpoint=parameter.references[1]
        secondid,secondpoint=parameter.references[2]
        firstvertices=original[firstid].vertices;secondvertices=original[secondid].vertices
        if _sonnet_geovar_anc_sequential(parameter)
            # Sequential anchored law (native-verified): REF2 lands at
            # REF1's current coordinate plus direction*target; every PS2
            # listing then shifts by that reference displacement. Inactive
            # and dependent dimensions obey the same rule, so the shift is
            # computed from the already-resolved coordinates.
            anchor=polygons[firstid][parameter.axis,firstpoint]
            reference=polygons[secondid][parameter.axis,secondpoint]
            shift=anchor+parameter.direction*target*p.length_scale-reference
            # A consistent anchored dimension can still carry a sub-cell
            # phantom shift from SI round-trips and file-write
            # quantization (~1e-9 relative to the span); only a real
            # displacement applies or errors as lost.
            abs(shift)<=sqrt(eps(Float64))*max(abs(targets[i]),abs(gnominals[i]))*p.length_scale && continue
            isfinite(shift) ||
                _sonnet_error(p.source,parameter.line,"GEOVAR displacement is unrepresentable")
            for (id,index) in parameter.points[2]
                before=polygons[id][parameter.axis,index]
                after=before+shift
                isfinite(after) && after!=before ||
                    _sonnet_error(p.source,parameter.line,"GEOVAR displacement is lost in its stored coordinate")
                polygons[id][parameter.axis,index]=after
            end
            continue
        end
        # Saved geometry can carry a sub-cell delta from SI round-trips
        # and file-write quantization (~1e-9 relative); only a real
        # displacement applies (a sub-quantum active delta is still the
        # legitimate "displacement lost" error inside each law).
        abs(delta)<=sqrt(eps(Float64))*max(abs(targets[i]),abs(gnominals[i]))*p.length_scale && continue
        if parameter.kind=="ANC" && !parameter.scaled && parameter.explicit_moving_references>1
            for point in parameter.points[2]
                owner,movements=owners[point]
                movements>0 || continue
                id,index=point;before=polygons[id][parameter.axis,index]
                after=_sonnet_geovar_repeated_coordinate(before,target,gnominals[i],p.length_scale,
                    parameter.explicit_moving_references,movements,parameter.direction)
                isfinite(after) && after!=before ||
                    _sonnet_error(p.source,parameter.line,"GEOVAR displacement is lost in its stored coordinate")
                polygons[id][parameter.axis,index]=after
                owners[point]=(owner,-movements)
            end
            continue
        end
        if parameter.kind=="RAD"
            _sonnet_geovar_apply_radial!(p,parameter,target,gnominals[i],original,polygons,owners)
            continue
        end
        if parameter.kind=="SYM" && !parameter.scaled &&
                parameter.explicit_moving_references>0
            # Native unscaled symmetric dimensions (Sonnet-verified law, ~130
            # engine runs): REF2 repeats by 3*r*|delta|/4, floored to
            # quantum=min(cell,box/32) in mixed lists once |delta|>=quantum.
            # Each distinct ordinary PS2 point moves once by
            # min(ceil(c*r/2)*d2, c*r*d2*ramp) with d2=|delta|/2 and
            # ramp=clamp(|delta|/quantum-0.5,0.5,1), capped at 3*c*r*|delta|/8
            # on coarse rasters (cell>box/32). The anchor side is gated by
            # 3*r*|delta|/4>cell: REF1 then takes -|delta|/2 while ordinary PS1
            # points quantize to box/64 steps of floor(|delta|/(box/32));
            # ungated PS1 points still drop one box/64 step, sign-independent.
            # When the expanded PS2 set contains a vertex strictly on REF1's
            # side of the POS plane (crossing sets, always including REF1's
            # own vertex under the parser admission rule), REF2 instead takes
            # the full |delta| while ordinary shares are unchanged.
            parsed=tryparse(Float64,p.box[parameter.axis+1])
            boxdir=(parsed===nothing ? sonnet_variable_value(p,
                p.box[parameter.axis+1];variables,freq=frequency) : parsed)*p.length_scale
            cell=boxdir/(parse(Int,p.box[parameter.axis+3])÷2)
            amount=abs(delta);sense=sign(delta)*parameter.direction
            r=parameter.explicit_moving_references
            quantum=min(cell,boxdir/32);step=boxdir/64
            ramp=clamp(amount/quantum-0.5,0.5,1.0)
            gate=3r*amount/4>cell
            isfinite(boxdir) && boxdir>0 && isfinite(cell) && cell>0 &&
                isfinite(amount) ||
                _sonnet_error(p.source,parameter.line,"symmetric GEOVAR raster cell is unrepresentable")
            for point in unique(parameter.points[1])
                id,index=point;before=polygons[id][parameter.axis,index]
                if point==parameter.references[1]
                    gate || continue
                    after=before-sense*amount/2
                else
                    after=gate ? before-sense*step*max(1,floor(amount/(boxdir/32))) :
                        before-step
                end
                isfinite(after) && after!=before ||
                    _sonnet_error(p.source,parameter.line,"GEOVAR displacement is lost in its stored coordinate")
                polygons[id][parameter.axis,index]=after
            end
            explicit_ps2=parameter.points[2][1:end-1]
            mixed=any(point->point!=parameter.references[2],explicit_ps2)
            multiplicities=Dict{Tuple{Int,Int},Int}()
            for point in explicit_ps2
                multiplicities[point]=get(multiplicities,point,0)+1
            end
            for point in unique(parameter.points[2])
                id,index=point;before=polygons[id][parameter.axis,index]
                if point==parameter.references[2]
                    # A PS2 set that crosses the symmetry plane through REF1's
                    # own vertex moves REF2 by the full |delta| instead of the
                    # 3*r*|delta|/4 share (native-verified on literal and
                    # whole-polygon sets, both directions and deltas).
                    share=parameter.crossing ? amount : 3r*amount/4
                    after=before+sense*(mixed && amount>=quantum ?
                        max(share,quantum) : share)
                else
                    movements=multiplicities[point]
                    share=min(cld(movements*r,2)*amount/2,
                        movements*r*amount/2*ramp)
                    cell>quantum && (share=min(share,3movements*r*amount/8))
                    after=before+sense*share
                end
                isfinite(after) && after!=before ||
                    _sonnet_error(p.source,parameter.line,"GEOVAR displacement is lost in its stored coordinate")
                polygons[id][parameter.axis,index]=after
            end
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
                        gnominals[i],target,symmetric) : before+shift
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
            vertices,polygon.target,polygon.technology,polygon.technology_mode,polygon.flags))
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
    isempty(p.components) ||
        _sonnet_geovar_validate_components(p,polygons,original;variables,freq=frequency)
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

"""Native component pins (SMDP records) are file-static coordinates: they are
never translated by a GEOVAR dimension, so a resolved project is only valid
when every pin still lies on exactly one straight edge of a same-level sheet
polygon and no moved metal edge crosses a component's SBOX interior (the
native interference check). Throws when the resolved project would be
rejected by the native engine."""
function _sonnet_geovar_validate_components(p,polygons,original;variables,freq)
    parseda=tryparse(Float64,p.box[2]);parsedb=tryparse(Float64,p.box[3])
    boxa=(parseda===nothing ? sonnet_variable_value(p,p.box[2];variables,freq=freq) : parseda)*p.length_scale
    boxb=(parsedb===nothing ? sonnet_variable_value(p,p.box[3];variables,freq=freq) : parsedb)*p.length_scale
    tol=1e-9*max(boxa,boxb)
    moved=Set{Tuple{Int,Int}}()
    for polygon in p.polygons
        ov=original[polygon.id].vertices;nv=polygons[polygon.id]
        for j in axes(nv,2)
            (nv[1,j]!=ov[1,j] || nv[2,j]!=ov[2,j]) && push!(moved,(polygon.id,j))
        end
    end
    for component in p.components
        sbox=nothing
        for record in component
            t=record.tokens
            if _sonnet_ieq(t[1],"SMDP") && length(t)>=4
                level=parse(Int,t[2])
                x=sonnet_variable_value(p,t[3];variables,freq=freq)*p.length_scale
                y=sonnet_variable_value(p,t[4];variables,freq=freq)*p.length_scale
                orient=length(t)>=5 ? t[5] : ""
                xaxis=orient in ("L","R")
                axial=xaxis ? x : y;transverse=xaxis ? y : x
                found=0
                for polygon in p.polygons
                    polygon.kind===:sheet && polygon.level==level || continue
                    v=polygons[polygon.id];n=size(v,2)
                    for k in 1:n
                        next=k==n ? 1 : k+1
                        a1,a2=xaxis ? (v[1,k],v[1,next]) : (v[2,k],v[2,next])
                        b1,b2=xaxis ? (v[2,k],v[2,next]) : (v[1,k],v[1,next])
                        abs(a1-axial)<=tol && abs(a2-axial)<=tol || continue
                        lo,hi=minmax(b1,b2)
                        hi-lo>tol && lo-tol<=transverse<=hi+tol || continue
                        found+=1
                    end
                end
                found==1 || throw(ArgumentError(
                    "moved GEOVAR detached a component pin from its polygon edge"))
            elseif _sonnet_ieq(t[1],"SBOX") && length(t)>=5
                vals=(sonnet_variable_value(p,t[k];variables,freq=freq) for k in 2:5)
                x1,x2,y1,y2=(v*p.length_scale for v in vals)
                sbox=(min(x1,x2),max(x1,x2),min(y1,y2),max(y1,y2))
            end
        end
        sbox===nothing && continue
        xlo,xhi,ylo,yhi=sbox
        for polygon in p.polygons
            v=polygons[polygon.id];n=size(v,2)
            for k in 1:n
                next=k==n ? 1 : k+1
                (polygon.id,k) in moved || (polygon.id,next) in moved || continue
                x1,x2=v[1,k],v[1,next];y1,y2=v[2,k],v[2,next]
                (x1<=xlo && x2<=xlo || x1>=xhi && x2>=xhi ||
                 y1<=ylo && y2<=ylo || y1>=yhi && y2>=yhi) && continue
                # Separating axes exclude the edge only when it stays entirely
                # outside; anything else reaches the open SBOX interior.
                throw(ArgumentError(
                    "moved GEOVAR metal interferes with a component's SBOX region"))
            end
        end
    end
    return nothing
end
