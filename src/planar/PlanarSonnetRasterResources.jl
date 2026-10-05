# Raster lowering owns masks, material maps, source metadata and basis arrays.
# These estimates use the actual physical sheet levels and via spans, rather
# than the component adapter's polygon-by-grid worst-case estimate.
function _sonnet_raster_grid(p,grid)
    hx=tryparse(Int,p.box[4]);hy=tryparse(Int,p.box[5])
    hx!==nothing && hy!==nothing && hx>0 && hy>0 && iseven(hx) && iseven(hy) ||
        throw(ArgumentError("native BOX half-cell counts must be positive even integers"))
    grid===nothing && return (hx÷2,hy÷2)
    grid isa Tuple && length(grid)==2 &&
        all(n->n isa Integer && !(n isa Bool) && 0<n<=typemax(Int),grid) ||
        throw(ArgumentError("native raster grid must contain two positive machine integers"))
    return (Int(grid[1]),Int(grid[2]))
end

function _sonnet_raster_source_workspace(p)
    # Coordinate movement and wall normalization can copy vertices. TMM can
    # add two polygons per sheet and split host-layer records. Reserve their
    # bounded source workspace before calling those transformations.
    thick=count(q->q.kind===:sheet && 0<=q.material<length(p.metals) &&
        length(p.metals[q.material+1])>=3 && p.metals[q.material+1][3]=="TMM",p.polygons)
    rowbytes=maximum(row->sum(t->64+ncodeunits(t),row;init=0),p.layers;init=0)
    return _checked_payload_sum("native raster source workspace",
        3BigInt(_sonnet_scalar_project_payload(p)),2BigInt(thick)*rowbytes)
end

function _sonnet_raster_mask_workspace(p,grid)
    nx,ny=grid.nx,grid.ny;layers=length(p.layers)
    levels=Set{Int}();spans=BigInt(0)
    for poly in p.polygons
        if poly.kind===:sheet
            0<=poly.level<layers || throw(ArgumentError("sheet level $(poly.level) outside native stack"))
            push!(levels,poly.level)
        elseif poly.kind===:via
            target=poly.target=="GND" ? layers-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
            lo,hi=minmax(poly.level,target)
            -1<=lo<hi<=layers-1 || throw(ArgumentError("via $(poly.id) target outside stack"))
            spans+=hi-lo
            if "COVERS" in poly.flags
                for level in (poly.level,target)
                    level in (-1,layers-1) || push!(levels,level)
                end
            end
        else
            throw(ArgumentError("native dielectric bricks require a volume dielectric adapter"))
        end
    end
    cells=BigInt(nx)*ny;mask=8cld(cells,64)
    connections=16(BigInt(cld(nx,64))+cld(ny,64))
    # Per-polygon masks remain live for port attachment. Temporary sheet,
    # via-mesh, overlap and index buffers are bounded before rasterization.
    return _checked_payload_sum("native raster mask workspace",
        BigInt(length(levels))*(mask+connections+16cells),
        2spans*mask,BigInt(length(p.polygons))*mask,
        4mask+connections+(spans>0 ? 8cells : 0),
        _checked_array_payload_bytes(Float64,2,
            maximum(q->size(q.vertices,2),p.polygons;init=0)))
end

function _sonnet_raster_port_workspace(p,grid,masks,polygons)
    layers=length(p.layers);raw=BigInt(0)
    labels=Set{Int}()
    for port in p.ports
        port.number==0 && continue
        push!(labels,abs(port.number))
        poly=get(polygons,port.polygon,nothing)
        poly===nothing && throw(ArgumentError("native port references a missing polygon"))
        if port.kind===:via || (port.kind===:std && poly.kind===:via)
            target=poly.target=="GND" ? layers-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
            raw+=BigInt(abs(poly.level-target))*count(masks[poly.id])
        else
            raw+=1
        end
    end
    nlabels=BigInt(length(labels))
    return _checked_payload_sum("native raster port workspace",
        512raw,_checked_array_payload_bytes(Float64,5,raw,nlabels),
        _checked_array_payload_bytes(Int,4,raw),
        _checked_array_payload_bytes(Int,max(grid.nx,grid.ny)))
end

function _sonnet_raster_basis_workspace(grid,sheets,vias)
    count=_terminal_geometry_basis_count(grid,sheets,vias,VolLevel[])
    # push! growth and per-sheet wall lookup coexist with the final basis.
    return _checked_payload_sum("native raster basis workspace",
        _checked_array_payload_bytes(UInt8,3,count),
        _checked_array_payload_bytes(Int,12,count),
        _checked_array_payload_bytes(Float64,9,count),
        _checked_array_payload_bytes(Int,2,BigInt(grid.nx)+grid.ny))
end

function _sonnet_raster_retained_payload(model,source)
    # The coordinate/material source clone is live while the EM solve runs.
    # Borrowed untransformed project records need no additional reservation.
    geometry=0
    if model.geometry_project!==source
        seen=Base.IdSet{Any}()
        _sonnet_raster_project_payload(source,seen,false)
        geometry=_sonnet_raster_project_payload(model.geometry_project,seen,true)
    end
    return _checked_payload_sum("native retained raster model",
        _sonnet_component_base_payload(model),geometry)
end

function _sonnet_raster_project_payload(value,seen,owned)
    value isa Union{SonnetProject,SonnetPolygon,SonnetPortSpec,SonnetRecord,
        AbstractArray,AbstractDict,AbstractString} || return 0
    value in seen && return 0
    push!(seen,value)
    if value isa AbstractString
        return owned ? _checked_payload_sum("native owned geometry string",64,ncodeunits(value)) : 0
    elseif value isa AbstractArray
        if isbitstype(eltype(value))
            return owned ? _subdivision_retained_payload(value) : 0
        end
        bytes=owned ? _checked_array_payload_bytes(Ptr{Cvoid},length(value)) : 0
        for item in value
            bytes=_checked_payload_sum("native owned geometry array",bytes,
                _sonnet_raster_project_payload(item,seen,owned))
        end
        return bytes
    elseif value isa AbstractDict
        bytes=owned ? _checked_array_payload_bytes(UInt8,128,length(value)) : 0
        for (key,item) in value
            bytes=_checked_payload_sum("native owned geometry dictionary",bytes,
                _sonnet_raster_project_payload(key,seen,owned),
                _sonnet_raster_project_payload(item,seen,owned))
        end
        return bytes
    end
    bytes=owned ? (value isa SonnetProject ? 1024 : 256) : 0
    for field in 1:fieldcount(typeof(value))
        bytes=_checked_payload_sum("native owned geometry records",bytes,
            _sonnet_raster_project_payload(getfield(value,field),seen,owned))
    end
    return bytes
end

function _sonnet_raster_response_workspace(model)
    nr=BigInt(length(model.problem.ports));np=BigInt(length(model.labels))
    nc=BigInt(size(model.floating_common,2))
    # Balanced/common-mode elimination, optional launch de-embedding and
    # power-wave conversion retain their output and factorization workspace.
    return _checked_array_payload_bytes(ComplexF64,
        24np^2+8nr*np+8nc^2+8nc*np+8(nr+np+nc))
end
