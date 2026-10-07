export planar_floating_bridge

@inline _planar_same_open_endpoint(a,b)=a.level==b.level && a.wall==b.wall && a.edge==b.edge && a.cells==b.cells

function _planar_bridge_connected(mask,start,stop,xaxis,cut,span,ports,level)
    nx,ny=size(mask);seen=falses(nx,ny);queue=Vector{Int}(undef,nx*ny)
    seen[start...]=true;queue[1]=start[1]+nx*(start[2]-1);tail=1;head=1
    while head<=tail
        index=queue[head];head+=1;i=rem(index-1,nx)+1;j=(index-1)÷nx+1
        (i,j)==stop && return true
        for (di,dj) in ((-1,0),(1,0),(0,-1),(0,1))
            ni,nj=i+di,j+dj
            1<=ni<=nx && 1<=nj<=ny && mask[ni,nj] && !seen[ni,nj] || continue
            blocked=xaxis ? di!=0 && min(i,ni)==cut && j in span : dj!=0 && min(j,nj)==cut && i in span
            blocked || (blocked=any(p->p.level==level &&
                (p.wall===:internal_x && di!=0 && min(i,ni)==p.edge && j in p.cells ||
                 p.wall===:internal_y && dj!=0 && min(j,nj)==p.edge && i in p.cells),ports))
            blocked && continue
            seen[ni,nj]=true;tail+=1;queue[tail]=ni+nx*(nj-1)
        end
    end
    false
end

"""Insert an explicit floating two-conductor source bridge between facing
open sheet endpoints `signal` and `reference`. Endpoints use
`PlanarPort(...,:terminal_x/:terminal_y,edge,cells,z0;metal_side=...)`, must
lie on the same sheet, face each other, and have identical transverse
spans. The intervening empty strip is filled with physical sheet metal;
`source_edge` selects a complete shared-rooftop gap cut in that strip.
Positive voltage is `Vsignal-Vreference` and positive current flows from
the reference pad into the signal pad. The reference conductor remains
floating. The added strip's delay, loss and coupling are physical and
must be removed with coupled sister standards for a calibrated local port.

Returns `(problem,port_index,old_port_map,bridge)`. Existing endpoint half
sources are removed, with zero entries in `old_port_map`; other ports
retain their order. `bridge` records filled cells, endpoint contracts,
axis, source cut, width and length for material lowering and calibration.
Original geometry is preserved. A pre-existing or side-attached sheet
path bypassing the full source cut is rejected. `max_bytes` preflights
owned masks, connectivity work, new basis and metadata before allocation.
Bare port-free input geometry is supported."""
function planar_floating_bridge(prob::PlanarProblem,signal::PlanarPort,reference::PlanarPort;
        source_edge::Union{Nothing,Integer}=nothing,z0=signal.z0,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    _is_planar_terminal(signal.wall) && _is_planar_terminal(reference.wall) || throw(ArgumentError("floating bridge endpoints require open sheet terminal contracts"))
    signal.level==reference.level && signal.cells==reference.cells &&
        _is_planar_x_terminal(signal.wall)==_is_planar_x_terminal(reference.wall) || throw(ArgumentError("floating bridge endpoints must share sheet, axis and complete transverse span"))
    _is_planar_positive_terminal(signal.wall)!=_is_planar_positive_terminal(reference.wall) || throw(ArgumentError("floating bridge endpoints must face each other"))
    1<=signal.level<=length(prob.sheets) || throw(ArgumentError("floating bridge sheet is invalid"))
    signal.polarity==reference.polarity==1 || throw(ArgumentError("floating bridge endpoint contracts use physical positive voltage; reverse the source by exchanging signal and reference"))
    grid=prob.grid;xaxis=_is_planar_x_terminal(signal.wall);axis_count=xaxis ? grid.nx : grid.ny;transverse=xaxis ? grid.ny : grid.nx
    left=_is_planar_positive_terminal(signal.wall) ? reference : signal
    right=_is_planar_positive_terminal(signal.wall) ? signal : reference
    lo,hi=left.edge,right.edge;span=signal.cells
    0<lo<hi<axis_count && !isempty(span) && 1<=first(span)<=last(span)<=transverse || throw(ArgumentError("floating bridge requires facing interior pad edges with a nonempty gap"))
    cut=source_edge===nothing ? lo+(hi-lo)÷2 : source_edge
    lo<=cut<=hi && 1<=cut<axis_count || throw(ArgumentError("floating source edge must cross the entire bridge between its pad endpoints"))
    source=PlanarPort(signal.level,xaxis ? :x : :y,cut,span,z0;
        polarity=_is_planar_positive_terminal(signal.wall) ? 1 : -1)
    cells=BigInt(hi-lo)*length(span);boxcells=BigInt(grid.nx)*grid.ny
    nb=max(planar_basis_count(prob.basis),_terminal_geometry_basis_count(grid,prob.sheets,prob.vias,prob.vols))+
        4cells+sum(BigInt(length(p.cells)) for p in prob.ports if _is_planar_terminal(p.wall);init=BigInt(0))
    payload=_checked_payload_sum("floating source bridge",
        _checked_array_payload_bytes(UInt64,3,cld(boxcells,64)),
        _checked_array_payload_bytes(UInt64,2,cld(BigInt(grid.nx),64)+cld(BigInt(grid.ny),64)),
        _checked_array_payload_bytes(Int,2,boxcells),_checked_array_payload_bytes(Int,2,cells),
        _checked_array_payload_bytes(Int,length(prob.ports)),_checked_array_payload_bytes(UInt8,2nb),
        _checked_array_payload_bytes(Int,8,nb),_checked_array_payload_bytes(Float64,6,nb),
        _checked_array_payload_bytes(Int,2,grid.nx+grid.ny))
    _enforce_payload_limit(payload,max_bytes,"floating source bridge","max_bytes")
    original=prob.sheets[signal.level];mask=original.mask
    for t in span
        li,lj=xaxis ? (lo,t) : (t,lo);ri,rj=xaxis ? (hi+1,t) : (t,hi+1)
        mask[li,lj] && mask[ri,rj] || throw(ArgumentError("floating bridge endpoints must span occupied pad edge cells"))
        for e in lo+1:hi
            i,j=xaxis ? (e,t) : (t,e)
            !mask[i,j] || throw(ArgumentError("floating bridge gap must be empty; metal overlaps its source path"))
        end
    end
    leftcell=xaxis ? (lo,first(span)) : (first(span),lo)
    rightcell=xaxis ? (hi+1,first(span)) : (first(span),hi+1)
    _planar_bridge_connected(mask,leftcell,rightcell,xaxis,-1,span,prob.ports,signal.level) && throw(ArgumentError("floating source pads already share a sheet path bypassing the source"))
    sheet=SheetLevel(original.interface,copy(mask),copy(original.connect_west),copy(original.connect_east),copy(original.connect_south),copy(original.connect_north))
    filled=Tuple{Int,Int}[];sizehint!(filled,Int(cells))
    for t in span,e in lo+1:hi
        i,j=xaxis ? (e,t) : (t,e);sheet.mask[i,j]=true;push!(filled,(i,j))
    end
    _planar_bridge_connected(sheet.mask,leftcell,rightcell,xaxis,Int(cut),span,prob.ports,signal.level) && throw(ArgumentError("side-attached metal bypasses the full floating source cut"))
    sheets=copy(prob.sheets);sheets[signal.level]=sheet;ports=PlanarPort[];old_port_map=zeros(Int,length(prob.ports))
    for (i,p) in enumerate(prob.ports)
        (_planar_same_open_endpoint(p,signal)||_planar_same_open_endpoint(p,reference)) && continue
        push!(ports,p);old_port_map[i]=length(ports)
    end
    push!(ports,source);problem=build_planar_problem(prob.stack,grid,sheets,ports;vias=prob.vias,vols=prob.vols)
    bridge=(cells=filled,signal=signal,reference=reference,axis=xaxis ? :x : :y,
        sheet=signal.level,interface=original.interface,source_edge=Int(cut),span=span,
        width=length(span)*(xaxis ? grid.dy : grid.dx),length=(hi-lo)*(xaxis ? grid.dx : grid.dy),
        polarity=source.polarity)
    return (;problem,port_index=length(ports),old_port_map,bridge)
end
