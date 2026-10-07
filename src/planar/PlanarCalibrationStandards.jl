export planar_group_line_standards
export planar_floating_line_standards

function _floating_standard_strip(cross,seed)
    1<=seed<=length(cross) && cross[seed] || throw(ArgumentError(
        "floating standard terminal does not reach a uniform conductor strip"))
    low=high=seed
    while low>1 && cross[low-1];low-=1;end
    while high<length(cross) && cross[high+1];high+=1;end
    return low:high
end

"""Generate physical `line,double_line,reflect` sister standards for
explicit floating local-source bridges. `bridges` contains the metadata
returned by [`planar_floating_bridge`](@ref), augmented with its `port`
index; project layouts already store this in `terminal_paths`.
`left,right` identify paired complete launch groups. Their bridges must
cross the transverse direction, with identical mirror-symmetric end
fixtures and a uniform longitudinal sheet region between them.

The double line inserts one copy of that uniform region without changing
the cell pitch, launch cuts, transverse cross-section, finite reference
conductors or wall contacts. Reflect removes the signal strips from the
uniform region and retains reference strips and other conductors. Its
coupled EM response must be solved, rather than assumed diagonal. The
returned `length` is the uniform region's physical length and `ordering`
is `[left;right]`, suitable for
[`planar_group_double_delay_calibrate`](@ref) when its checked cascade
assumptions hold. Long-range fixture coupling can violate those assumptions.
This constructs physical standards; it does not implement or claim the
proprietary Sonnet GLG wave algorithm or native floating calibration.
`max_bytes` preflights all retained generated geometry and bases."""
function planar_floating_line_standards(prob::PlanarProblem,bridges::AbstractVector;
        left::AbstractVector{<:Integer},right::AbstractVector{<:Integer},
        uniform_cells::Union{Nothing,UnitRange{Int}}=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    n=length(left);np=length(prob.ports)
    n>0 && length(right)==n && 2BigInt(n)==np &&
        sort(vcat(left,right))==collect(1:np) || throw(ArgumentError(
            "floating sister standards require two complete distinct launch groups"))
    isempty(prob.vias) && isempty(prob.vols) || throw(ArgumentError(
        "floating sister standards currently require sheet launch geometry without axial or volume fixtures"))
    length(bridges)==np && all(b->hasproperty(b,:port),bridges) ||
        throw(ArgumentError("every floating launch requires explicit bridge metadata and its port index"))
    sorted=sort(collect(bridges);by=b->b.port)
    [b.port for b in sorted]==collect(1:np) || throw(ArgumentError(
        "floating bridge metadata must cover each source exactly once"))
    axis=first(sorted).axis
    axis in (:x,:y) && all(b->b.axis===axis,sorted) || throw(ArgumentError(
        "floating launch bridges must share a transverse direction"))
    xaxis=axis===:y
    longitudinal=xaxis ? prob.grid.nx : prob.grid.ny
    transverse=xaxis ? prob.grid.ny : prob.grid.nx
    for b in sorted
        p=prob.ports[b.port]
        expected=xaxis ? :internal_y : :internal_x
        p.wall===expected && p.edge==b.source_edge && p.cells==b.span &&
            p.level==b.sheet && p.polarity==b.polarity || throw(ArgumentError(
                "floating launch metadata does not match its physical source"))
        all(cell->prob.sheets[b.sheet].mask[cell...],b.cells) || throw(ArgumentError(
            "floating bridge cells are missing from the source geometry"))
    end
    lspan=sorted[first(left)].span;rspan=sorted[first(right)].span
    last(lspan)<first(rspan) && first(lspan)+last(rspan)==longitudinal+1 &&
        last(lspan)+first(rspan)==longitudinal+1 || throw(ArgumentError(
            "floating launch spans must occupy opposite mirror-symmetric end fixtures"))
    for (l,r) in zip(left,right)
        bl,br=sorted[l],sorted[r];p,q=prob.ports[l],prob.ports[r]
        bl.span==lspan && br.span==rspan && bl.sheet==br.sheet &&
            p.edge==q.edge && p.polarity==q.polarity && p.z0==q.z0 &&
            bl.signal.wall==br.signal.wall && bl.signal.edge==br.signal.edge &&
            bl.reference.wall==br.reference.wall && bl.reference.edge==br.reference.edge ||
            throw(ArgumentError("paired floating launches must have identical transverse source and reference contracts"))
    end
    middle=uniform_cells===nothing ? ((last(lspan)+1):(first(rspan)-1)) : uniform_cells
    !isempty(middle) && last(lspan)<first(middle)<=last(middle)<first(rspan) &&
        first(middle)+last(middle)==longitudinal+1 || throw(ArgumentError(
            "uniform floating standard region must lie symmetrically between the launch bridges"))
    extra=length(middle);newcount=BigInt(longitudinal)+extra
    newcount<=typemax(Int) || throw(ArgumentError("floating double-line cell count overflows Int"))
    cells=(3BigInt(longitudinal)+extra)*transverse*length(prob.sheets)
    payload=_checked_payload_sum("floating sister standards",
        _checked_array_payload_bytes(UInt8,2,cells),
        _checked_array_payload_bytes(UInt8,57,2cells),
        _checked_array_payload_bytes(Int,12,np),
        _checked_array_payload_bytes(Int,6,sum(BigInt(length(b.cells)) for b in sorted)),
        _checked_array_payload_bytes(UInt8,4,transverse,length(prob.sheets)))
    _enforce_payload_limit(payload,max_bytes,"floating sister standards","max_bytes")
    for sheet in prob.sheets
        for along in 1:longitudinal,t in 1:transverse
            i,j=xaxis ? (along,t) : (t,along)
            ri,rj=xaxis ? (longitudinal+1-along,t) : (t,longitudinal+1-along)
            sheet.mask[i,j]==sheet.mask[ri,rj] || throw(ArgumentError(
                "floating double-delay launches require mirror-symmetric sheet geometry"))
        end
        cross=xaxis ? view(sheet.mask,first(middle),:) : view(sheet.mask,:,first(middle))
        for along in middle,t in 1:transverse
            (xaxis ? sheet.mask[along,t] : sheet.mask[t,along])==cross[t] || throw(ArgumentError(
                "floating sister standards require a uniform interior cross-section"))
        end
        side_a,side_b=xaxis ? (sheet.connect_south,sheet.connect_north) :
            (sheet.connect_west,sheet.connect_east)
        for side in (side_a,side_b)
            all(i->side[i]==side[length(side)+1-i],eachindex(side)) &&
                all(i->side[i]==side[first(middle)],middle) ||
                throw(ArgumentError("floating standard wall contacts must preserve mirror symmetry and the uniform interior"))
        end
        startwall,endwall=xaxis ? (sheet.connect_west,sheet.connect_east) :
            (sheet.connect_south,sheet.connect_north)
        startwall==endwall || throw(ArgumentError("floating end wall contacts must match"))
    end
    # Only signal strips are opened: local grounds retain their actual
    # continuity, delay, resistance and coupling to the global enclosure.
    signal_strips=[falses(transverse) for _ in prob.sheets]
    reference_strips=[falses(transverse) for _ in prob.sheets]
    for b in sorted
        cross=xaxis ? view(prob.sheets[b.sheet].mask,first(middle),:) :
            view(prob.sheets[b.sheet].mask,:,first(middle))
        for (pin,strips) in ((b.signal,signal_strips),(b.reference,reference_strips))
            seed=_is_planar_positive_terminal(pin.wall) ? pin.edge+1 : pin.edge
            strips[b.sheet][_floating_standard_strip(cross,seed)].=true
        end
    end
    any(any(s .& r) for (s,r) in zip(signal_strips,reference_strips)) && throw(ArgumentError(
        "floating signal and reference must remain distinct in the uniform line"))
    cut=longitudinal÷2
    function shift_bridge(b,added)
        delta=first(b.span)>cut ? added : 0
        shifted(pin)=PlanarPort(pin.level,pin.wall,(first(pin.cells)+delta):(last(pin.cells)+delta),
            pin.z0;edge=pin.edge,polarity=pin.polarity,refplane=pin.refplane)
        shiftedcells=[xaxis ? (i+delta,j) : (i,j+delta) for (i,j) in b.cells]
        return merge(b,(cells=shiftedcells,signal=shifted(b.signal),reference=shifted(b.reference),
            span=(first(b.span)+delta):(last(b.span)+delta)))
    end
    function standard(added,reflect)
        count=longitudinal+added
        nx,ny=xaxis ? (count,prob.grid.ny) : (prob.grid.nx,count)
        a,b=xaxis ? (count*prob.grid.dx,prob.grid.b) : (prob.grid.a,count*prob.grid.dy)
        stack=PlanarStackup(prob.stack.layers,prob.stack.bottom,prob.stack.top,a,b)
        grid=CellGrid(a,b,nx,ny;walls=prob.grid.walls)
        source_index(i)=i<=cut ? i : i<=cut+added ? first(middle) : i-added
        sheets=SheetLevel[]
        for (k,old) in enumerate(prob.sheets)
            sheet=sheet_level(old.interface,nx,ny)
            for along in 1:count,t in 1:transverse
                oldalong=source_index(along)
                metal=xaxis ? old.mask[oldalong,t] : old.mask[t,oldalong]
                reflect && oldalong in middle && signal_strips[k][t] && (metal=false)
                i,j=xaxis ? (along,t) : (t,along);sheet.mask[i,j]=metal
            end
            if xaxis
                sheet.connect_west.=old.connect_west;sheet.connect_east.=old.connect_east
                sheet.connect_south.=[old.connect_south[source_index(i)] for i in 1:count]
                sheet.connect_north.=[old.connect_north[source_index(i)] for i in 1:count]
            else
                sheet.connect_south.=old.connect_south;sheet.connect_north.=old.connect_north
                sheet.connect_west.=[old.connect_west[source_index(i)] for i in 1:count]
                sheet.connect_east.=[old.connect_east[source_index(i)] for i in 1:count]
            end
            push!(sheets,sheet)
        end
        ports=[PlanarPort(p.level,p.wall,
            first(p.cells)>cut ? ((first(p.cells)+added):(last(p.cells)+added)) : p.cells,
            p.z0;edge=p.edge,polarity=p.polarity,refplane=p.refplane) for p in prob.ports]
        return build_planar_problem(stack,grid,sheets,ports)
    end
    return (line=standard(0,false),double_line=standard(extra,false),reflect=standard(0,true),
        length=extra*(xaxis ? prob.grid.dx : prob.grid.dy),
        ordering=Int.(vcat(left,right)),left=Int.(left),right=Int.(right),
        line_bridges=copy(sorted),double_line_bridges=[shift_bridge(b,extra) for b in sorted],
        reflect_bridges=copy(sorted),uniform_cells=middle,
        double_uniform_cells=first(middle):(last(middle)+extra))
end

"""Generate coupled multiconductor `line,double_line,reflect` sister
geometries with identical transverse sheets, port spans, materials and
wall boundary conditions. `left,right` list the paired physical wall
ports in conductor order. All ports must belong to these two groups and
the sheets must have a uniform longitudinal cross-section. Unported
reference strips are retained; their sidewall contacts remain explicit.

The returned `length` is the physical first-line length and `ordering`
orders solved responses as `[left;right]` for
[`planar_group_double_delay_calibrate`](@ref). Reflect opens every sheet
between equal end stubs; it remains a coupled EM network and is not an
assumed ideal diagonal reflect standard. This generator does not infer
floating local-ground source bridges or their proprietary calibration."""
function planar_group_line_standards(prob::PlanarProblem;
        left::AbstractVector{<:Integer},right::AbstractVector{<:Integer},
        reflect_cells::Union{Nothing,Integer}=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    n=length(left);np=length(prob.ports)
    n>0 && length(right)==n && 2n==np &&
        sort(vcat(left,right))==collect(1:np) || throw(ArgumentError(
            "coupled line standards require two complete distinct port groups"))
    isempty(prob.vias) && isempty(prob.vols) || throw(ArgumentError(
        "coupled sister standards require a uniform sheet cross-section; axial/volume fixtures need their physical material adapter"))
    a=prob.ports[first(left)];b=prob.ports[first(right)]
    xaxis=a.wall===:west && b.wall===:east
    yaxis=a.wall===:south && b.wall===:north
    xaxis || yaxis || throw(ArgumentError("line groups must be ordered west/east or south/north"))
    for (l,r) in zip(left,right)
        p,q=prob.ports[l],prob.ports[r]
        p.wall===a.wall && q.wall===b.wall && p.level==q.level &&
            p.cells==q.cells && p.polarity==q.polarity && p.z0==q.z0 ||
            throw(ArgumentError("paired line launches must match level, transverse span, polarity and reference"))
    end
    original=xaxis ? prob.grid.nx : prob.grid.ny
    ncells=BigInt(original)*2
    ncells<=typemax(Int) || throw(ArgumentError("double-line cell count overflows Int"))
    reflect_cells=reflect_cells===nothing ? max(1,original÷4) : reflect_cells
    1<=reflect_cells && 2BigInt(reflect_cells)<original || throw(ArgumentError(
        "reflect stubs must leave a positive isolation gap"))
    for sheet in prob.sheets
        cross=xaxis ? view(sheet.mask,1,:) : view(sheet.mask,:,1)
        for along in 1:original,transverse in eachindex(cross)
            (xaxis ? sheet.mask[along,transverse] : sheet.mask[transverse,along])==cross[transverse] ||
                throw(ArgumentError("sister geometry requires longitudinally uniform sheet masks"))
        end
        side_a,side_b=xaxis ? (sheet.connect_south,sheet.connect_north) :
            (sheet.connect_west,sheet.connect_east)
        all(==(first(side_a)),side_a) && all(==(first(side_b)),side_b) ||
            throw(ArgumentError("transverse wall contacts must be uniform along the line"))
    end
    # All three masks/bases remain retained by the returned standards.
    ls=length(prob.sheets);transverse=xaxis ? prob.grid.ny : prob.grid.nx
    payload=_checked_payload_sum("coupled sister standards",
        _checked_array_payload_bytes(UInt8,ls,4BigInt(original),transverse),
        _checked_array_payload_bytes(Int,4,ls,4BigInt(original)+3transverse),
        _checked_array_payload_bytes(UInt8,57,8BigInt(original)*transverse*ls),
        _checked_array_payload_bytes(Int,8,np))
    _enforce_payload_limit(payload,max_bytes,"coupled sister standards","max_bytes")
    function standard(count,reflect)
        nx,ny=xaxis ? (count,prob.grid.ny) : (prob.grid.nx,count)
        width=xaxis ? count*prob.grid.dx : prob.grid.a
        height=xaxis ? prob.grid.b : count*prob.grid.dy
        stack=PlanarStackup(prob.stack.layers,prob.stack.bottom,prob.stack.top,width,height)
        grid=CellGrid(width,height,nx,ny;walls=prob.grid.walls)
        sheets=SheetLevel[]
        for old in prob.sheets
            sheet=sheet_level(old.interface,nx,ny)
            cross=xaxis ? view(old.mask,1,:) : view(old.mask,:,1)
            for along in 1:count,transverse in eachindex(cross)
                cross[transverse] && (!reflect || along<=reflect_cells || along>count-reflect_cells) || continue
                i,j=xaxis ? (along,transverse) : (transverse,along)
                sheet.mask[i,j]=true
            end
            if xaxis
                sheet.connect_west.=old.connect_west;sheet.connect_east.=old.connect_east
                fill!(sheet.connect_south,first(old.connect_south));fill!(sheet.connect_north,first(old.connect_north))
            else
                sheet.connect_south.=old.connect_south;sheet.connect_north.=old.connect_north
                fill!(sheet.connect_west,first(old.connect_west));fill!(sheet.connect_east,first(old.connect_east))
            end
            push!(sheets,sheet)
        end
        ports=[PlanarPort(p.level,p.wall,p.cells,p.z0;polarity=p.polarity,refplane=p.refplane) for p in prob.ports]
        return build_planar_problem(stack,grid,sheets,ports)
    end
    return (line=standard(original,false),double_line=standard(Int(ncells),false),
        reflect=standard(original,true),length=original*(xaxis ? prob.grid.dx : prob.grid.dy),
        ordering=Int.(vcat(left,right)),left=Int.(left),right=Int.(right))
end
