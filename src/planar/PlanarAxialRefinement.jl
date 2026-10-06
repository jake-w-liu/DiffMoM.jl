# Physical axial refinement of dielectric and conductor layers.
export PlanarAxialRefinement,planar_refine_axial,solve_planar_axial

"""Axial refinement retaining the original port-voltage meaning.
`problem` contains the subdivided geometry, `contraction` maps original
port voltages to refined port voltages, and `via_parent`/`volume_parent`
map refined conductor indices to their original material indices."""
struct PlanarAxialRefinement
    original::PlanarProblem
    problem::PlanarProblem
    contraction::Matrix{Float64}
    layer_parent::Vector{Int}
    via_parent::Vector{Int}
    volume_parent::Vector{Int}
end

"""`planar_refine_axial(problem, subdivisions; max_bytes=...)`
subdivides each stackup layer into equal physical slices. `subdivisions`
is a positive integer or one positive integer per layer. Sheet interfaces,
volume masks, via footprints, wall connections and port polarities are
preserved. Volume currents gain a resolved axial profile. Uniform via
profiles become piecewise uniform; tapered profiles gain the uniform
parts needed to retain the original global linear ramp.
Volume wall ports apply their full voltage on every corresponding slice;
their terminal current is the sum through the complete physical thickness.
Via-port voltage is distributed across slices in proportion to thickness,
so total voltage and its power-conjugate terminal current are preserved.
No new metal sheets are inserted between slices."""
function planar_refine_axial(prob::PlanarProblem,subdivisions;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    L=length(prob.stack.layers)
    subdivisions isa Integer || subdivisions isa AbstractVector{<:Integer} ||
        throw(ArgumentError("subdivisions must be an integer or integer vector"))
    subdivisions isa AbstractVector && length(subdivisions)!=L &&
        throw(DimensionMismatch("subdivision count must match stackup layers"))
    counts=[subdivisions isa Integer ? subdivisions : subdivisions[l] for l in 1:L]
    all(>=(1),counts) || throw(ArgumentError("subdivisions must be positive"))
    nlayer=sum(BigInt(n) for n in counts)
    nvia=sum((BigInt(counts[v.layer]) for v in prob.vias);init=BigInt(0))
    nvolume=sum((BigInt(counts[v.layer]) for v in prob.vols);init=BigInt(0))
    nport=sum(BigInt(p.wall===:via ? counts[prob.vias[p.level].layer] :
        _is_planar_volume_port(p.wall) ? counts[prob.vols[p.level].layer] : 1) for p in prob.ports)
    # Taper-only refinement gains a uniform component above the first
    # slice. Account for those new unknowns before allocating geometry.
    nbnew=BigInt(0)
    for p in eachindex(prob.basis.kind)
        k=prob.basis.kind[p]
        if _is_vol_kind(k)
            nbnew+=counts[prob.vols[prob.basis.level[p]].layer]
        elseif _is_via_kind(k)
            via=prob.vias[prob.basis.level[p]];n=counts[via.layer]
            nbnew+=n
            k==_BASIS_VIA_T && !via.uni[prob.basis.ei[p],prob.basis.ej[p]] && (nbnew+=n-1)
        else
            nbnew+=1
        end
    end
    words=cld(BigInt(prob.grid.nx)*prob.grid.ny,64)
    metadata=_checked_payload_sum("axial refinement",
        _checked_array_payload_bytes(UInt64,words,2nvia+nvolume+length(prob.sheets)),
        _checked_array_payload_bytes(UInt64,cld(BigInt(prob.grid.nx),64)+cld(BigInt(prob.grid.ny),64),
            2*(nvolume+length(prob.sheets))),
        _checked_array_payload_bytes(eltype(prob.stack.layers),nlayer),
        _checked_array_payload_bytes(Float64,nport,length(prob.ports)),
        _checked_array_payload_bytes(Int,nlayer+nvia+nvolume+2L),
        _checked_array_payload_bytes(Int,4,nbnew),
        _checked_array_payload_bytes(Float64,3,nbnew),
        _checked_array_payload_bytes(UInt8,nbnew))
    _enforce_payload_limit(metadata,max_bytes,"axial refinement","max_bytes")
    counts=Int.(counts)
    ends=cumsum(counts)
    ranges=[(l==1 ? 1 : ends[l-1]+1):ends[l] for l in 1:L]
    layers=eltype(prob.stack.layers)[];layer_parent=Int[]
    for l in 1:L,j in ranges[l]
        old=prob.stack.layers[l]
        push!(layers,PlanarLayer(old.epsr,old.mur,old.thickness/counts[l];
            epsr_z=old.epsr_z,mur_z=old.mur_z))
        push!(layer_parent,l)
    end
    stack=PlanarStackup(layers,prob.stack.bottom,prob.stack.top,prob.stack.a,prob.stack.b)
    sheets=[SheetLevel(sh.interface==0 ? 0 : ends[sh.interface],copy(sh.mask),
        copy(sh.connect_west),copy(sh.connect_east),copy(sh.connect_south),copy(sh.connect_north))
        for sh in prob.sheets]
    vias=ViaLevel[];via_parent=Int[];via_indices=Vector{Vector{Int}}(undef,length(prob.vias))
    for (v,old) in enumerate(prob.vias)
        ids=Int[]
        for (piece,j) in enumerate(ranges[old.layer])
            uniform=copy(old.uni)
            piece>1 && (uniform .|= old.tap)
            push!(vias,ViaLevel(j,uniform,copy(old.tap)));push!(via_parent,v);push!(ids,length(vias))
        end
        via_indices[v]=ids
    end
    vols=VolLevel[];volume_parent=Int[]
    for (v,old) in enumerate(prob.vols),j in ranges[old.layer]
        push!(vols,VolLevel(j,copy(old.mask),copy(old.connect_west),copy(old.connect_east),
            copy(old.connect_south),copy(old.connect_north)))
        push!(volume_parent,v)
    end
    ports=PlanarPort[];C=zeros(Float64,Int(nport),length(prob.ports));row=0
    for (col,p) in enumerate(prob.ports)
        if p.wall===:via
            ids=via_indices[p.level]
            for id in ids
                push!(ports,PlanarPort(id,p.wall,p.cells,p.z0;edge=p.edge,polarity=p.polarity))
                row+=1;C[row,col]=inv(length(ids))
            end
        elseif _is_planar_volume_port(p.wall)
            first_volume=searchsortedfirst(volume_parent,p.level)
            count=counts[prob.vols[p.level].layer]
            for id in first_volume:first_volume+count-1
                # Every depth slice has the same physical wall voltage;
                # power-conjugate currents add across the whole thickness.
                push!(ports,id==p.level ? p : PlanarPort(id,p.wall,p.cells,p.z0,p.edge,p.polarity,p.refplane))
                row+=1;C[row,col]=1
            end
        else
            push!(ports,p);row+=1;C[row,col]=1
        end
    end
    refined=build_planar_problem(stack,prob.grid,sheets,ports;vias,vols)
    return PlanarAxialRefinement(prob,refined,C,layer_parent,via_parent,volume_parent)
end

function _planar_refined_material(values,parent,count,label)
    if values isa AbstractVector
        length(values)==count || throw(DimensionMismatch("$label count must match original conductor levels"))
        return view(values,parent)
    end
    return values
end

"""Solve an axial refinement using original conductor material arrays.
The returned contracted response retains refined raw currents and maps
original port excitations to their physical distributed sources."""
function solve_planar_axial(refinement::PlanarAxialRefinement,freq::Number;
        via_sigma=Inf,volume_sigma=Inf,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    material_bytes=_checked_array_payload_bytes(ComplexF64,
        length(refinement.via_parent)+length(refinement.volume_parent))
    _enforce_payload_limit(material_bytes,max_bytes,"axial materials","max_bytes")
    via=_planar_refined_material(via_sigma,refinement.via_parent,
        length(refinement.original.vias),"via_sigma")
    volume=_planar_refined_material(volume_sigma,refinement.volume_parent,
        length(refinement.original.vols),"volume_sigma")
    remaining=Int(BigInt(max_bytes)-material_bytes)
    return solve_planar_contracted(refinement.problem,freq,refinement.contraction;
        problem=refinement.original,z0=[p.z0 for p in refinement.original.ports],
        via_sigma=via,volume_sigma=volume,max_bytes=remaining,kw...)
end
