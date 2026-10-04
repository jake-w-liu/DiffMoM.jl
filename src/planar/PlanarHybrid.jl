export PlanarHybridProblem, PlanarHybridResult
export assemble_planar_hybrid_z, solve_planar_hybrid, planar_hybrid_current_maps

"""Genuine triangular sheet currents coupled to rectangular via/volume
currents in the same physical stack and sidewalls. `conformal` supplies all
sheet geometry; `bulk` is a `PlanarProblem` containing only via/volume
levels. Ports are ordered conformal first, then bulk. Contact footprints
should be constraints of the sheet mesh so current divergence at galvanic
contacts is resolved. Sheet and bulk material models must represent distinct
conductors, or one consistent volume model without duplicate face IBCs."""
struct PlanarHybridProblem
    conformal::PlanarConformalProblem
    bulk::PlanarProblem
    stack::PlanarStackup
    ports::Vector{Union{PlanarConformalPort,PlanarPort}}
end

function PlanarHybridProblem(conformal::PlanarConformalProblem,bulk::PlanarProblem)
    isempty(bulk.sheets) || throw(ArgumentError("hybrid sheet geometry must use the genuine triangle mesh; bulk contains via/volume levels only"))
    c,b=conformal.stack,bulk.stack
    c.layers==b.layers && c.bottom==b.bottom && c.top==b.top && c.a==b.a && c.b==b.b &&
        conformal.sidewalls==bulk.grid.walls || throw(ArgumentError("hybrid source stacks and sidewalls must agree"))
    all(k->_is_via_kind(k)||_is_vol_kind(k),bulk.basis.kind) || throw(ArgumentError("hybrid bulk basis must contain only via or volume currents"))
    PlanarHybridProblem(conformal,bulk,c,Union{PlanarConformalPort,PlanarPort}[conformal.ports...;bulk.ports...])
end

function PlanarHybridProblem(conformal::PlanarConformalProblem,grid::CellGrid,
        ports::Vector{PlanarPort}=PlanarPort[];vias::Vector{ViaLevel}=ViaLevel[],vols::Vector{VolLevel}=VolLevel[])
    stack=conformal.stack;L=length(stack.layers)
    planar_validate(stack)
    stack.a==grid.a && stack.b==grid.b || throw(ArgumentError("hybrid grid must match the physical box"))
    all(v->1<=v.layer<=L,vias) && all(v->1<=v.layer<=L,vols) || throw(ArgumentError("hybrid bulk levels must lie inside the stack"))
    basis=build_planar_basis(grid,SheetLevel[],ports;vias,vols)
    all(p->any(==(p),basis.port),eachindex(ports)) || throw(ArgumentError("each hybrid bulk port must claim a current trace"))
    PlanarHybridProblem(conformal,PlanarProblem(stack,grid,SheetLevel[],ports,vias,basis,vols))
end

function _planar_hybrid_traces(prob)
    nc=length(prob.conformal.basis.width);nr=planar_basis_count(prob.bulk.basis)
    weights=Vector{Float64}(undef,nc+nr);signs=similar(weights);ports=Vector{Int}(undef,nc+nr)
    weights[1:nc].=prob.conformal.basis.width;signs[1:nc].=prob.conformal.basis.port_sign;ports[1:nc].=prob.conformal.basis.port
    np=length(prob.conformal.ports)
    for b in 1:nr
        p=prob.bulk.basis.port[b];ports[nc+b]=p==0 ? 0 : np+p
        weights[nc+b]=_planar_port_weight(prob.bulk.basis,b)
        signs[nc+b]=p==0 ? 1. : _planar_port_sign(prob.bulk.ports[p])
    end
    return weights,signs,ports
end

function _planar_hybrid_cross_metadata(prob,mx,my)
    rect=prob.bulk;nc=length(prob.conformal.basis.width);nr=planar_basis_count(rect.basis)
    mg=planar_mode_grid(rect.grid,mx,my);fxb=Matrix{Float64}(undef,mg.mx,nr);fyb=Matrix{Float64}(undef,mg.my,nr)
    for b in 1:nr
        _basis_fx!(view(fxb,:,b),rect.basis,b,mg,rect.grid)
        _basis_fy!(view(fyb,:,b),rect.basis,b,mg,rect.grid)
    end
    elems=[_basis_elem(rect.basis,b,rect.sheets,rect.vias,rect.vols) for b in 1:nr]
    clevels=unique(prob.conformal.basis.interfaces);rlevels=unique(elems)
    pairs=[(f,s) for f in clevels for s in rlevels]
    pairindex=Dict(pair=>i for (i,pair) in enumerate(pairs))
    vlay=sort!(unique(_via_elem_layer(e) for e in elems if _is_via_elem(e)))
    volay=sort!(unique(_vol_elem_layer(e) for e in elems if _is_vol_elem(e)))
    workspace=_planar_mode_workspace(length(prob.stack.layers),!isempty(vlay),!isempty(volay))
    return (;mg,fxb,fyb,elems,clevels,rlevels,pairs,pairindex,vlay,volay,workspace,nc,nr)
end

"""Exact analytic modal Galerkin matrix including all sheet/via/volume
cross reactions. The two self blocks retain their established analytic
assemblers; cross kernels reuse the same TE/TM cascade and axial moments.
`surface_zs` applies per triangle; `via_sigma`/`volume_sigma` apply to bulk
levels. No surface, volume or Green quadrature is used."""
function assemble_planar_hybrid_z(prob::PlanarHybridProblem,freq::Number;
        mx::Integer=2prob.bulk.grid.nx,my::Integer=2prob.bulk.grid.ny,
        surface_zs=0.,via_sigma=Inf,volume_sigma=Inf,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    mx>=1 && my>=1 || throw(ArgumentError("hybrid mode counts must be positive"))
    omega=2pi*ComplexF64(freq);isfinite(omega) && real(omega)>0 || throw(ArgumentError("hybrid frequency must be positive and finite"))
    nc=length(prob.conformal.basis.width);nr=planar_basis_count(prob.bulk.basis);n=nc+nr;L=length(prob.stack.layers)
    crossbytes=_checked_payload_sum("hybrid cross workspace",
        _checked_array_payload_bytes(Float64,mx+my,nr),_checked_array_payload_bytes(Float64,7,mx+my),
        _checked_array_payload_bytes(Float64,2,nc+nr),_checked_array_payload_bytes(ComplexF64,nc+nr),
        _checked_array_payload_bytes(ComplexF64,2,nc,nr), # upper bound on pair metadata/kernels
        _checked_array_payload_bytes(Int,4,nc+nr),_checked_array_payload_bytes(ComplexF64,50,L+1))
    matrixbytes=_checked_array_payload_bytes(ComplexF64,n,n)
    reserved=_checked_payload_sum("hybrid assembly",matrixbytes,crossbytes)
    _enforce_payload_limit(reserved,max_bytes,"hybrid assembly","max_bytes")
    Z=zeros(ComplexF64,n,n)
    nc==0 || (Z[1:nc,1:nc].=assemble_planar_conformal_z(prob.conformal,freq;mx,my,surface_zs,max_bytes=max_bytes-reserved))
    rect=prob.bulk
    nr==0 || (Z[nc+1:n,nc+1:n].=assemble_planar_z(rect.stack,rect.grid,rect.sheets,rect.basis,omega;
        mx,my,vias=rect.vias,vols=rect.vols,via_sigma,volume_sigma,max_bytes=max_bytes-reserved))
    (nc==0 || nr==0) && return Z
    meta=_planar_hybrid_cross_metadata(prob,Int(mx),Int(my));mg=meta.mg
    wt=Matrix{Float64}(undef,1,nr);wm=similar(wt);te=Vector{Float64}(undef,nc);tm=similar(te)
    ct=Vector{ComplexF64}(undef,nc);cr=Vector{ComplexF64}(undef,nr)
    vt=Matrix{ComplexF64}(undef,1,length(meta.pairs));vm=similar(vt);ml=[1];nl=[1]
    cte,ctm,scratch,vsts,volsts=meta.workspace
    groups=[(level,findfirst(==(level),prob.conformal.basis.interfaces):findlast(==(level),prob.conformal.basis.interfaces)) for level in meta.clevels]
    for nn in 1:mg.my,mm in 1:mg.mx
        ml[1]=mm;nl[1]=nn
        _planar_weight_block!(wt,wm,ml,nl,mg,meta.fxb,meta.fyb,rect.basis)
        _planar_conformal_weights!(te,tm,prob.conformal,mg.kx[mm],mg.ky[nn])
        _planar_mode_voltages!(vt,vm,cte,ctm,scratch,prob.stack,omega,mg,ml,nl,
            meta.pairs,vsts,meta.vlay,volsts,meta.volay)
        for (Wc,Wr,V) in ((te,wt,vt),(tm,wm,vm)),(level,rows) in groups
            ct.=Wc
            for b in 1:nr
                cr[b]=Wr[1,b]*V[1,meta.pairindex[(level,meta.elems[b])]]
            end
            LinearAlgebra.BLAS.geru!(-1.0+0im,view(ct,rows),cr,view(Z,rows,nc+1:n))
        end
    end
    for q in 1:nr,p in 1:nc
        Z[nc+q,p]=Z[p,nc+q]
    end
    return Z
end

"""Mixed conformal/bulk solve. Rows of `currents` contain triangular
coefficients first, then physical rectangular via/volume coefficients.
`lu_fact` factors the trace-equilibrated matrix with `basis_scale`."""
struct PlanarHybridResult{F}
    problem::PlanarHybridProblem
    freq::ComplexF64
    omega::ComplexF64
    z_mom::Union{Nothing,Matrix{ComplexF64}}
    lu_fact::F
    basis_scale::Vector{Float64}
    currents::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    relative_residuals::Vector{Float64}
    z0::Vector{ComplexF64}
end
PlanarHybridResult(prob::PlanarHybridProblem,freq,omega,Z,F,scale,X,Y,S,residuals)=
    PlanarHybridResult(prob,freq,omega,Z,F,scale,X,Y,S,residuals,
        _planar_reference_values([p.z0 for p in prob.ports],length(prob.ports);freq=real(freq)))

"""Solve coupled genuine sheet triangles and bulk via/volume currents.
All port traces are power conjugate, including via area and volume width.
Retained coefficients preserve both physical source geometries."""
function solve_planar_hybrid(prob::PlanarHybridProblem,freq::Number;
        method::Symbol=:dense,retain_matrix::Bool=true,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    method===:ufft && return solve_planar_hybrid_ufft(prob,freq;max_bytes,kw...)
    method===:dense || throw(ArgumentError("hybrid method must be :dense or :ufft"))
    omega=2pi*ComplexF64(freq)
    isfinite(omega) && real(omega)>0 || throw(ArgumentError("hybrid solve frequency must be finite with positive real part"))
    n=length(prob.conformal.basis.width)+planar_basis_count(prob.bulk.basis);np=length(prob.ports)
    n>0 && np>0 || throw(ArgumentError("hybrid solve needs currents and ports"))
    work=_checked_payload_sum("hybrid solve",_checked_array_payload_bytes(ComplexF64,n,np),
        _checked_array_payload_bytes(ComplexF64,2,n),_checked_array_payload_bytes(Float64,3,n),
        _checked_array_payload_bytes(Int,n),_checked_array_payload_bytes(ComplexF64,5,np,np),
        _checked_array_payload_bytes(Float64,np),_checked_array_payload_bytes(ComplexF64,2,np))
    _enforce_payload_limit(_checked_dense_lu_work_bytes(ComplexF64,n,retain_matrix ? 2 : 1,work;label="hybrid solve"),max_bytes,"hybrid solve","max_bytes")
    refs=_planar_reference_values([p.z0 for p in prob.ports],np;freq=real(freq))
    reserve=_checked_payload_sum("hybrid assembly reservation",work,_checked_array_payload_bytes(ComplexF64,n,n,retain_matrix ? 1 : 0))
    Z=assemble_planar_hybrid_z(prob,freq;max_bytes=max_bytes-reserve,kw...)
    all(isfinite,Z) || throw(ArgumentError("nonfinite hybrid matrix; check lossless box resonances"))
    weights,signs,ports=_planar_hybrid_traces(prob);scale=inv.(weights)
    scaled=retain_matrix ? copy(Z) : Z
    for q in 1:n,p in 1:n
        scaled[p,q]*=scale[p]*scale[q]
    end
    F=lu!(scaled);X=zeros(ComplexF64,n,np)
    for b in 1:n
        p=ports[b];p==0 || (X[b,p]=-signs[b])
    end
    ldiv!(F,X);X .*= scale;Y=zeros(ComplexF64,np,np)
    for b in 1:n
        p=ports[b];p==0 && continue
        for q in 1:np
            Y[p,q]+=signs[b]*weights[b]*X[b,q]
        end
    end
    residuals=retain_matrix ? zeros(Float64,np) : Float64[]
    if retain_matrix
        rhs=zeros(ComplexF64,n);residual=similar(rhs)
        for p in 1:np
            for b in 1:n
                rhs[b]=ports[b]==p ? -signs[b]*weights[b] : 0.0im
            end
            mul!(residual,Z,view(X,:,p))
            for b in 1:n
                residual[b]=(residual[b]-rhs[b])*scale[b];rhs[b]*=scale[b]
            end
            residuals[p]=norm(residual)/norm(rhs)
        end
    end
    PlanarHybridResult(prob,ComplexF64(freq),2pi*ComplexF64(freq),retain_matrix ? Z : nothing,F,scale,X,Y,
        planar_y_to_s(Y,refs),residuals,refs)
end

"""Reconstruct triangular and bulk physical currents for a mixed solve.
Returns `(triangles=..., bulk=...)`; both use the same terminal voltages or
incident power waves and preserve their physical interface/layer geometry."""
function _planar_hybrid_current_maps(result;port::Integer=1,voltages=nothing,incident_waves=nothing,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    prob=result.problem;np=length(prob.ports);nc=length(prob.conformal.basis.width);n=size(result.currents,1)
    reserve=_checked_payload_sum("hybrid current excitation",_checked_array_payload_bytes(ComplexF64,n),
        _checked_array_payload_bytes(ComplexF64,10,np))
    trianglebytes=_planar_conformal_map_payload(prob.conformal)
    _enforce_payload_limit(_checked_payload_sum("hybrid currents",reserve,trianglebytes),max_bytes,"hybrid currents","max_bytes")
    voltages!==nothing && incident_waves!==nothing && throw(ArgumentError("provide voltages or incident_waves"))
    supplied=voltages===nothing ? incident_waves : voltages
    v=if supplied===nothing
        1<=port<=np || throw(ArgumentError("invalid hybrid port"));x=zeros(ComplexF64,np);x[port]=1;x
    else
        supplied isa AbstractVector && length(supplied)==np && all(isfinite,supplied) || throw(ArgumentError("hybrid excitation must match ports"))
        ComplexF64.(supplied)
    end
    incident_waves===nothing || (v=_planar_wave_voltage(result.s,v,result.z0))
    coeff=result.currents*v
    triangles=_planar_conformal_maps_from_coefficients(prob.conformal,view(coeff,1:nc);max_bytes=trianglebytes)
    bulk=_planar_current_maps_from_coefficients(prob.bulk,view(coeff,nc+1:n);max_bytes=max_bytes-reserve-trianglebytes,kw...)
    return (;triangles,bulk)
end
"""Reconstruct mixed triangular and bulk physical currents for terminal
voltages or incident waves. Returns `(triangles,bulk)` with exact affine
triangle integrals and the established via/volume profile maps."""
planar_hybrid_current_maps(result::PlanarHybridResult;kw...)=_planar_hybrid_current_maps(result;kw...)
planar_current_maps(result::PlanarHybridResult;kw...)=_planar_hybrid_current_maps(result;kw...)
