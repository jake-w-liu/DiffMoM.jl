export PlanarHybridUFFTOperator,PlanarHybridUFFTResult
export planar_hybrid_ufft_operator,solve_planar_hybrid_ufft

"""Exact FFT reaction operator coupling genuine triangular sheets to
rectangular axial vias and transverse volume currents. Each geometry uses
its own analytic Fourier families on the same physical lattice; cross
kernels contain the exact multilayer axial moments. No dense cross block
or quadrature is retained. Mutable workspaces require separate operators
for concurrent calls."""
struct PlanarHybridUFFTOperator{C,B} <: AbstractMatrix{ComplexF64}
    n::Int
    nc::Int
    problem::PlanarHybridProblem
    conformal::C
    bulk::B
    k_te::Matrix{ComplexF64}
    k_tm::Matrix{ComplexF64}
    output::Vector{ComplexF64}
end
Base.size(A::PlanarHybridUFFTOperator)=(A.n,A.n)
Base.size(A::PlanarHybridUFFTOperator,d::Integer)=d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? A.n : 1
Base.eltype(::Type{<:PlanarHybridUFFTOperator})=ComplexF64

function _planar_hybrid_bulk_fft_payload(prob,mx,my,block)
    nr=planar_basis_count(prob.basis);nr==0 && return 0
    elems=[_basis_elem(prob.basis,b,prob.sheets,prob.vias,prob.vols) for b in 1:nr]
    ne=length(unique(elems));nf=length(unique((elems[b],prob.basis.kind[b]) for b in 1:nr))
    nm=_checked_array_payload_bytes(UInt8,mx,my;label="hybrid FFT modes")
    _checked_payload_sum("hybrid bulk FFT",
        _checked_array_payload_bytes(ComplexF64,2,nm,ne,ne),_checked_array_payload_bytes(ComplexF64,4,nm,ne),
        _checked_array_payload_bytes(ComplexF64,2BigInt(prob.grid.nx),2BigInt(prob.grid.ny)),
        _checked_array_payload_bytes(ComplexF64,2,nf,mx+my),_checked_array_payload_bytes(Float64,7,mx+my),
        _checked_array_payload_bytes(Int,8,nr),_checked_array_payload_bytes(ComplexF64,9,nr),
        _checked_array_payload_bytes(Int,16,nr),_checked_array_payload_bytes(ComplexF64,50,length(prob.stack.layers)+1),
        _checked_array_payload_bytes(Int,2,min(BigInt(nm),BigInt(block))))
end

function _planar_hybrid_conformal_owned_payload(A)
    A===nothing && return 0
    _checked_payload_sum("retained hybrid triangles",
        sum(sizeof(eltype(x))*length(x) for x in (A.c_te,A.c_tm,A.k_te,A.k_tm,A.source_te,A.source_tm,
            A.field_te,A.field_tm,A.lattice,A.output,A.local_loss.nzval,A.local_loss.colptr,A.local_loss.rowval)),
        sum(sizeof(Int)*(length(f.indices)+length(f.lattice))+sizeof(UInt8)*length(f.halves)+sizeof(Float64)*length(f.amplitudes) for f in A.families),
        _checked_array_payload_bytes(Float64,7,A.modes.mx+A.modes.my))
end

"""Build an exact mixed triangle/via/volume FFT operator. Triangular
vertices must lie on the bulk grid's declared physical lattice. High
`mx,my` modes retain their analytic transforms before alias folding.
Owned numerical array payloads and construction work are reserved across
both geometry operators and the cross kernels before allocation."""
function planar_hybrid_ufft_operator(prob::PlanarHybridProblem,freq::Number;
        mx::Integer=2prob.bulk.grid.nx,my::Integer=2prob.bulk.grid.ny,
        surface_zs=0.,via_sigma=Inf,volume_sigma=Inf,block::Integer=512,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    mx>=1 && my>=1 && block>=1 || throw(ArgumentError("hybrid FFT mode counts and block must be positive"))
    omega=2pi*ComplexF64(freq);isfinite(omega) && real(omega)>0 || throw(ArgumentError("hybrid FFT frequency must be positive and finite"))
    nc=length(prob.conformal.basis.width);nr=planar_basis_count(prob.bulk.basis);nc+nr>0 || throw(ArgumentError("hybrid FFT requires physical currents"))
    # This small topology pass has its own bound before building element lists.
    topology=_checked_array_payload_bytes(Int,8,nc+nr)
    _enforce_payload_limit(topology,max_bytes,"hybrid FFT topology","max_bytes")
    levels=unique(prob.conformal.basis.interfaces)
    elems=unique([_basis_elem(prob.bulk.basis,b,prob.bulk.sheets,prob.bulk.vias,prob.bulk.vols) for b in 1:nr])
    nm=_checked_array_payload_bytes(UInt8,mx,my;label="hybrid FFT mode count")
    cross=_checked_payload_sum("hybrid FFT cross kernels",topology,
        _checked_array_payload_bytes(ComplexF64,2,nm,length(levels),length(elems)),
        _checked_array_payload_bytes(ComplexF64,nc+nr),
        _checked_array_payload_bytes(ComplexF64,50,length(prob.stack.layers)+1),
        _checked_array_payload_bytes(Int,2,min(BigInt(nm),BigInt(block))))
    bulkbytes=_planar_hybrid_bulk_fft_payload(prob.bulk,mx,my,block)
    reserved=_checked_payload_sum("hybrid FFT reservation",cross,bulkbytes)
    _enforce_payload_limit(reserved,max_bytes,"hybrid FFT","max_bytes")
    C=nc==0 ? nothing : planar_conformal_ufft_operator(prob.conformal,freq;
        nx=prob.bulk.grid.nx,ny=prob.bulk.grid.ny,mx,my,surface_zs,max_bytes=max_bytes-reserved)
    cb=_planar_hybrid_conformal_owned_payload(C)
    # Cross reactions consume the individual modal TE/TM source fields.
    B=nr==0 ? nothing : planar_ufft_operator(prob.bulk,freq;mx,my,via_sigma,volume_sigma,block,
        max_bytes=max_bytes-cross-cb,_fold_iterative=false)
    kt=zeros(ComplexF64,nm,length(levels)*length(elems));km=similar(kt)
    if !isempty(levels) && !isempty(elems)
        mg=B.modes;pairs=[(f,s) for f in levels for s in elems]
        vl=sort!(unique(_via_elem_layer(e) for e in elems if _is_via_elem(e)))
        vol=sort!(unique(_vol_elem_layer(e) for e in elems if _is_vol_elem(e)))
        ct,cm,scratch,vs,vols=_planar_mode_workspace(length(prob.stack.layers),!isempty(vl),!isempty(vol))
        mb=Vector{Int}(undef,Int(min(BigInt(nm),BigInt(block))));nb=similar(mb)
        first=1
        while first<=nm
            count=min(length(mb),nm-first+1)
            for q in 1:count
                t=first+q-2;mb[q]=rem(t,Int(mx))+1;nb[q]=t÷Int(mx)+1
            end
            _planar_mode_voltages!(view(kt,first:first+count-1,:),view(km,first:first+count-1,:),
                ct,cm,scratch,prob.stack,omega,mg,view(mb,1:count),view(nb,1:count),pairs,vs,vl,vols,vol)
            first+=count
        end
    end
    all(isfinite,kt) && all(isfinite,km) || throw(ArgumentError("nonfinite hybrid FFT cross kernel; check box resonances"))
    PlanarHybridUFFTOperator(nc+nr,nc,prob,C,B,kt,km,zeros(ComplexF64,nc+nr))
end

function _planar_hybrid_fft_source!(A::PlanarConformalUFFTOperator,x,offset)
    st,sm=A.source_te,A.source_tm;fill!(st,0);fill!(sm,0)
    F=A.lattice;px,py=size(F);mg=A.modes
    for (f,family) in enumerate(A.families)
        fill!(F,0)
        for q in eachindex(family.indices)
            F[family.lattice[q]]+=family.amplitudes[q]*x[offset+family.indices[q]]
        end
        A.forward*F;e=family.element
        for n in 0:mg.my-1,m in 0:mg.mx-1
            t=m+1+mg.mx*n
            for (s,(sx,sy)) in enumerate(_PLANAR_CONFORMAL_SIGNS)
                value=F[mod(-sx*m,px)+1,mod(-sy*n,py)+1]
                st[t,e]+=_planar_conformal_coefficient(A.c_te,t,s,f)*value
                sm[t,e]+=_planar_conformal_coefficient(A.c_tm,t,s,f)*value
            end
        end
    end
end
function _planar_hybrid_fft_source!(A::PlanarUFFTOperator,x,offset)
    st,sm=A.source_te,A.source_tm;fill!(st,0);fill!(sm,0)
    F=A.lattice;px,py=size(F);mg=A.modes
    for family in A.families
        fill!(F,0)
        for q in eachindex(family.indices)
            F[family.lattice[q]]+=x[offset+family.indices[q]]
        end
        A.forward*F;via=_is_via_kind(family.kind);xd=_is_xdir(family.kind);e=family.element
        for n in 0:mg.my-1,m in 0:mg.mx-1
            t=m+1+mg.mx*n;value=_ufft_family_sum(F,m,n,family,px,py)
            st[t,e]+=via ? 0.0im : (xd ? mg.ky[n+1] : -mg.kx[m+1])*value
            sm[t,e]+=(via ? 1. : xd ? mg.kx[m+1] : mg.ky[n+1])*value
        end
    end
end
function _planar_hybrid_fft_self_fields!(A)
    st,sm,ft,fm=A.source_te,A.source_tm,A.field_te,A.field_tm
    fill!(ft,0);fill!(fm,0);ne=size(st,2)
    for f in 1:ne,s in 1:ne,t in axes(st,1)
        pair=(f-1)*ne+s
        ft[t,f]-=A.k_te[t,pair]*st[t,s];fm[t,f]-=A.k_tm[t,pair]*sm[t,s]
    end
end
function _planar_hybrid_fft_gather!(out,A::PlanarConformalUFFTOperator,x,offset)
    F=A.lattice;px,py=size(F);mg=A.modes;ft,fm=A.field_te,A.field_tm
    for (f,family) in enumerate(A.families)
        fill!(F,0);e=family.element
        for n in 0:mg.my-1,m in 0:mg.mx-1
            t=m+1+mg.mx*n
            for (s,(sx,sy)) in enumerate(_PLANAR_CONFORMAL_SIGNS)
                F[mod(sx*m,px)+1,mod(sy*n,py)+1]+=
                    _planar_conformal_coefficient(A.c_te,t,s,f)*ft[t,e]+
                    _planar_conformal_coefficient(A.c_tm,t,s,f)*fm[t,e]
            end
        end
        A.backward*F
        for q in eachindex(family.indices)
            out[offset+family.indices[q]]+=family.amplitudes[q]*F[family.lattice[q]]
        end
    end
    mul!(view(out,offset+1:offset+A.n),A.local_loss,view(x,offset+1:offset+A.n),1.,1.)
end
function _planar_hybrid_fft_gather!(out,A::PlanarUFFTOperator,x,offset)
    F=A.lattice;px,py=size(F);mg=A.modes;ft,fm=A.field_te,A.field_tm
    for family in A.families
        fill!(F,0);via=_is_via_kind(family.kind);xd=_is_xdir(family.kind);e=family.element
        for n in 0:mg.my-1,m in 0:mg.mx-1
            t=m+1+mg.mx*n;wt=via ? 0. : xd ? mg.ky[n+1] : -mg.kx[m+1];wm=via ? 1. : xd ? mg.kx[m+1] : mg.ky[n+1]
            _ufft_family_fold!(F,wt*ft[t,e]+wm*fm[t,e],m,n,family,px,py)
        end
        A.backward*F
        for q in eachindex(family.indices)
            out[offset+family.indices[q]]+=F[family.lattice[q]]
        end
    end
    mul!(view(out,offset+1:offset+A.n),A.local_loss,view(x,offset+1:offset+A.n),1.,1.)
end
function _planar_hybrid_fft_workspace_alias(A,x)
    A===nothing && return false
    Base.mightalias(x,A.output)||Base.mightalias(x,A.lattice)||Base.mightalias(x,A.source_te)||Base.mightalias(x,A.source_tm)||
        Base.mightalias(x,A.field_te)||Base.mightalias(x,A.field_tm)
end
function LinearAlgebra.mul!(y::AbstractVector,A::PlanarHybridUFFTOperator,x::AbstractVector,alpha::Number,beta::Number)
    length(y)==length(x)==A.n || throw(DimensionMismatch("hybrid FFT vector size mismatch"))
    (Base.mightalias(x,A.output)||Base.mightalias(y,A.output)||_planar_hybrid_fft_workspace_alias(A.conformal,x)||
        _planar_hybrid_fft_workspace_alias(A.bulk,x)||_planar_hybrid_fft_workspace_alias(A.conformal,y)||
        _planar_hybrid_fft_workspace_alias(A.bulk,y)) && throw(ArgumentError("hybrid FFT vectors must not alias operator workspaces"))
    C,B=A.conformal,A.bulk
    if C!==nothing
        _planar_hybrid_fft_source!(C,x,0);_planar_hybrid_fft_self_fields!(C)
    end
    if B!==nothing
        _planar_hybrid_fft_source!(B,x,A.nc);_planar_hybrid_fft_self_fields!(B)
    end
    if C!==nothing && B!==nothing
        ne=size(B.source_te,2)
        for f in axes(C.source_te,2),s in 1:ne,t in axes(C.source_te,1)
            pair=(f-1)*ne+s;kt,km=A.k_te[t,pair],A.k_tm[t,pair]
            C.field_te[t,f]-=kt*B.source_te[t,s];C.field_tm[t,f]-=km*B.source_tm[t,s]
            B.field_te[t,s]-=kt*C.source_te[t,f];B.field_tm[t,s]-=km*C.source_tm[t,f]
        end
    end
    fill!(A.output,0)
    C===nothing || _planar_hybrid_fft_gather!(A.output,C,x,0)
    B===nothing || _planar_hybrid_fft_gather!(A.output,B,x,A.nc)
    for p in eachindex(y)
        y[p]=iszero(beta) ? alpha*A.output[p] : alpha*A.output[p]+beta*y[p]
    end
    y
end
LinearAlgebra.mul!(y::AbstractVector,A::PlanarHybridUFFTOperator,x::AbstractVector)=mul!(y,A,x,1.,0.)
Base.:*(A::PlanarHybridUFFTOperator,x::AbstractVector)=mul!(zeros(ComplexF64,A.n),A,x)

function _planar_hybrid_ufft_diagonal(A;max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    reserve=_checked_array_payload_bytes(ComplexF64,A.n)
    _enforce_payload_limit(reserve,max_bytes,"hybrid FFT diagonal","max_bytes")
    d=Vector{ComplexF64}(undef,A.n)
    A.conformal===nothing || (d[1:A.nc].=_planar_conformal_ufft_diagonal(A.conformal;max_bytes=max_bytes-reserve))
    A.bulk===nothing || (d[A.nc+1:end].=_planar_ufft_diagonal(A.bulk))
    d
end

"""Mixed exact FFT solution with physical currents, ports, complete
voltage-unit residuals and bounded Krylov iteration counts."""
struct PlanarHybridUFFTResult
    problem::PlanarHybridProblem
    freq::ComplexF64
    omega::ComplexF64
    operator::PlanarHybridUFFTOperator
    currents::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    iterations::Vector{Int}
    relative_residuals::Vector{Float64}
    z0::Vector{ComplexF64}
end
PlanarHybridUFFTResult(prob::PlanarHybridProblem,freq,omega,A,X,Y,S,iterations,residuals)=
    PlanarHybridUFFTResult(prob,freq,omega,A,X,Y,S,iterations,residuals,
        _planar_reference_values([p.z0 for p in prob.ports],length(prob.ports);freq=real(freq)))

"""Solve mixed triangle/via/volume currents with exact FFT reactions.
Power-conjugate trace scaling and the exact self diagonal precondition
restarted GMRES. `rtol` gates a freshly recomputed full voltage residual;
failure raises an error and retains no unconverged network response."""
function solve_planar_hybrid_ufft(prob::PlanarHybridProblem,freq::Number;
        rtol::Real=1e-9,maxiter::Integer=0,memory::Integer=50,restart::Bool=true,
        precondition::Bool=true,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    omega=2pi*ComplexF64(freq)
    isfinite(omega) && real(omega)>0 || throw(ArgumentError("hybrid FFT solve frequency must be finite with positive real part"))
    n=length(prob.conformal.basis.width)+planar_basis_count(prob.bulk.basis);np=length(prob.ports)
    isfinite(rtol) && rtol>0 && 0<=maxiter<=typemax(Int) && 1<=memory<=typemax(Int) && n>0 && np>0 ||
        throw(ArgumentError("hybrid FFT solve needs ports and valid Krylov limits"))
    mem=restart ? min(BigInt(memory),BigInt(n)) : max(BigInt(memory),maxiter==0 ? 2BigInt(n) : BigInt(maxiter))
    reserve=_checked_payload_sum("hybrid FFT solve",
        _checked_array_payload_bytes(ComplexF64,n,mem+15),_checked_array_payload_bytes(ComplexF64,mem+1,mem),
        _checked_array_payload_bytes(ComplexF64,n,np),_checked_array_payload_bytes(ComplexF64,5,np,np),
        _checked_array_payload_bytes(Float64,5,n),_checked_array_payload_bytes(Int,17,n),
        _checked_array_payload_bytes(Float64,2,np),_checked_array_payload_bytes(Int,np),
        _checked_array_payload_bytes(ComplexF64,2,np))
    _enforce_payload_limit(reserve,max_bytes,"hybrid FFT solve","max_bytes")
    refs=_planar_reference_values([p.z0 for p in prob.ports],np;freq=real(freq))
    A=planar_hybrid_ufft_operator(prob,freq;max_bytes=max_bytes-reserve,kw...)
    weights,signs,ports=_planar_hybrid_traces(prob);D=Diagonal(inv.(weights))
    d=precondition ? _planar_hybrid_ufft_diagonal(A;max_bytes=reserve) : ComplexF64[]
    precondition && any(z->!isfinite(z)||iszero(z),d) && throw(ArgumentError("invalid hybrid FFT diagonal"))
    N=precondition ? Diagonal(weights./d) : I
    X=Matrix{ComplexF64}(undef,n,np);Y=zeros(ComplexF64,np,np);iterations=zeros(Int,np);residuals=zeros(Float64,np)
    rhs=zeros(ComplexF64,n);residual=similar(rhs)
    for p in 1:np
        for b in 1:n
            rhs[b]=ports[b]==p ? -signs[b]*weights[b] : 0.0im
        end
        # A galvanic bridge can have a large charge/current cancellation.
        # Arnoldi may reach its attainable recurrence residual while its
        # freshly applied voltage residual still needs iterative refinement.
        # Limit corrections and count every iteration against the user cap.
        total_limit=maxiter==0 ? _checked_payload_sum("hybrid iteration limit",8BigInt(n)) : Int(maxiter)
        per_limit=min(total_limit,_checked_payload_sum("hybrid iteration limit",2BigInt(n)))
        x,stats=Krylov.gmres(A,rhs;rtol=Float64(rtol)/10,atol=0.,itmax=per_limit,memory=min(Int(memory),n),restart,
            reorthogonalization=true,M=D,N)
        count=stats.niter;rhsnorm=sqrt(sum(abs2(rhs[b]/weights[b]) for b in 1:n));relative=Inf
        for refinement in 0:3
            mul!(residual,A,x);residual.-=rhs
            relative=sqrt(sum(abs2(residual[b]/weights[b]) for b in 1:n))/rhsnorm
            relative<=rtol && break
            (refinement==3 || count>=total_limit || !isfinite(relative)) && break
            residual.*=-1
            correction,correctstats=Krylov.gmres(A,residual;rtol=Float64(rtol)/10,atol=0.,
                itmax=min(per_limit,total_limit-count),memory=min(Int(memory),n),restart,
                reorthogonalization=true,M=D,N)
            x.+=correction;count+=correctstats.niter
        end
        isfinite(relative) && relative<=rtol || throw(ErrorException("hybrid FFT port $p failed full voltage residual $relative > $rtol after $count iterations"))
        X[:,p].=x;iterations[p]=count;residuals[p]=relative
        for b in 1:n
            q=ports[b];q==0 || (Y[q,p]+=signs[b]*weights[b]*x[b])
        end
    end
    PlanarHybridUFFTResult(prob,ComplexF64(freq),2pi*ComplexF64(freq),A,X,Y,planar_y_to_s(Y,refs),iterations,residuals,refs)
end
planar_hybrid_current_maps(result::PlanarHybridUFFTResult;kw...)=_planar_hybrid_current_maps(result;kw...)
planar_current_maps(result::PlanarHybridUFFTResult;kw...)=_planar_hybrid_current_maps(result;kw...)
