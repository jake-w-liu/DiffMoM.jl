export PlanarConformalUFFTOperator, PlanarConformalUFFTResult
export planar_conformal_ufft_operator, solve_planar_conformal_ufft

const _PLANAR_CONFORMAL_SIGNS=((1,1),(-1,1),(1,-1),(-1,-1))
const _PLANAR_CONFORMAL_STORED_SIGNS=((1,1),(1,-1))

# The geometry is real, so its affine transform satisfies F(-k)=conj(F(k)).
# Reversing both signs also negates the TE/TM orientation factors and cancels
# conjugation of 1/(4im). This holds for PEC and PMC sidewalls. Only the
# positive-x sign planes are retained; lossy modal kernels remain complex.
@inline function _planar_conformal_coefficient(coefficients,t,s,f)
    s==1 && return coefficients[t,1,f]
    s==3 && return coefficients[t,2,f]
    s==2 && return conj(coefficients[t,2,f])
    return conj(coefficients[t,1,f])
end
struct _PlanarConformalUFFTFamily
    element::Int
    shape::NTuple{6,Int}
    free::Int
    indices::Vector{Int}
    halves::Vector{UInt8}
    lattice::Vector{Int}
    amplitudes::Vector{Float64}
end

"""Exact modal FFT operator for genuine lattice-aligned triangles.
Every distinct translated triangle/free-vertex shape is one Fourier family;
congruent lattice meshes have a fixed number of families. Analytic affine
triangle transforms are multiplied by the full multilayer kernels before
alias folding, so high modes and near interactions are retained exactly.
Coordinates that do not lie on the declared `nx×ny` lattice are rejected.
Real geometry makes opposite Fourier-sign coefficients conjugate pairs;
only two sign planes are stored, with the other two reconstructed exactly.
Mutable FFT workspace is operator-owned; simultaneous calls need separate
operators. No dense basis-by-basis matrix or Green quadrature is stored."""
struct PlanarConformalUFFTOperator{PF,PB} <: AbstractMatrix{ComplexF64}
    n::Int
    problem::PlanarConformalProblem
    grid::CellGrid
    modes::PlanarModeGrid
    families::Vector{_PlanarConformalUFFTFamily}
    c_te::Array{ComplexF64,3}
    c_tm::Array{ComplexF64,3}
    k_te::Matrix{ComplexF64}
    k_tm::Matrix{ComplexF64}
    source_te::Matrix{ComplexF64}
    source_tm::Matrix{ComplexF64}
    field_te::Matrix{ComplexF64}
    field_tm::Matrix{ComplexF64}
    lattice::Matrix{ComplexF64}
    forward::PF
    backward::PB
    local_loss::SparseMatrixCSC{ComplexF64,Int}
    output::Vector{ComplexF64}
end
Base.size(A::PlanarConformalUFFTOperator)=(A.n,A.n)
Base.size(A::PlanarConformalUFFTOperator,d::Integer)=d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? A.n : 1
Base.eltype(::Type{<:PlanarConformalUFFTOperator})=ComplexF64

"""Construct the exact finite-mode FFT Galerkin operator for genuine
triangle currents. Required `nx,ny` declare the physical vertex lattice;
`mx,my` select independent modal truncation, including modes beyond lattice
Nyquist. `surface_zs` is scalar or one value per triangle. `max_bytes`
preflights owned array payloads before Fourier/FFT/kernel allocation."""
function planar_conformal_ufft_operator(prob::PlanarConformalProblem,freq::Number;
        nx::Integer,ny::Integer,mx::Integer=2nx,my::Integer=2ny,surface_zs=0.,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    omega=2pi*ComplexF64(freq);nb=length(prob.basis.width);nv=size(prob.mesh.vertices,2);nt=length(prob.mesh.interfaces)
    isfinite(omega) && real(omega)>0 && nx>=1 && ny>=1 && mx>=1 && my>=1 && nb>0 ||
        throw(ArgumentError("conformal FFT needs positive lattice/mode counts, frequency and basis functions"))
    _checked_array_payload_bytes(ComplexF64,2BigInt(nx),2BigInt(ny);label="conformal FFT lattice")
    nmode=_checked_array_payload_bytes(UInt8,mx,my;label="conformal FFT mode count")
    metadata=_checked_payload_sum("conformal FFT metadata",_checked_array_payload_bytes(Int,52,nb),
        _checked_array_payload_bytes(Int,2,nv),_checked_array_payload_bytes(Float64,2,nb))
    _enforce_payload_limit(metadata,max_bytes,"conformal FFT metadata","max_bytes")
    grid=CellGrid(prob.stack.a,prob.stack.b,nx,ny;walls=prob.sidewalls)
    ix=Vector{Int}(undef,nv);iy=similar(ix);tol=64eps(Float64)*max(grid.a,grid.b)
    for i in 1:nv
        x,y=prob.mesh.vertices[1,i],prob.mesh.vertices[2,i]
        ix[i]=round(Int,x/grid.dx);iy[i]=round(Int,y/grid.dy)
        abs(x-ix[i]*grid.dx)<=tol && abs(y-iy[i]*grid.dy)<=tol ||
            throw(ArgumentError("conformal vertex $i is not on the declared lattice; provide its physical lattice or use dense conformal assembly"))
    end
    levels=unique(prob.basis.interfaces);element=Dict(level=>i for (i,level) in enumerate(levels));ne=length(levels)
    records=Dict{Tuple{Int,NTuple{6,Int},Int},Vector{Tuple{Int,UInt8,Int,Float64}}}()
    px=2Int(nx)
    for b in 1:nb,half in 1:2
        t=prob.basis.triangles[half,b];t==0 && continue
        ids=sort(collect(prob.mesh.triangles[:,t]);by=i->(ix[i],iy[i]))
        a,c=prob.basis.edges[:,b];free=findfirst(i->i!=a && i!=c,ids)
        ox=minimum(ix[i] for i in ids);oy=minimum(iy[i] for i in ids)
        shape=ntuple(j->isodd(j) ? ix[ids[(j+1)÷2]]-ox : iy[ids[j÷2]]-oy,Val(6))
        sign=half==2 || prob.basis.triangles[2,b]==0 ? -1. : 1.
        amplitude=sign*prob.basis.width[b]/(2prob.mesh.areas[t])
        key=(element[prob.basis.interfaces[b]],shape,free)
        push!(get!(records,key,Tuple{Int,UInt8,Int,Float64}[]),(b,UInt8(half),ox+1+px*oy,amplitude))
    end
    keys_sorted=sort!(collect(keys(records)));nf=length(keys_sorted);L=length(prob.stack.layers)
    est=_checked_payload_sum("conformal FFT",metadata,
        _checked_array_payload_bytes(ComplexF64,4,nmode,nf),
        _checked_array_payload_bytes(ComplexF64,2,nmode,ne,ne),
        _checked_array_payload_bytes(ComplexF64,4,nmode,ne),
        _checked_array_payload_bytes(ComplexF64,2BigInt(nx),2BigInt(ny)),
        _checked_array_payload_bytes(Float64,7,mx+my),
        _checked_array_payload_bytes(ComplexF64,nb),
        _checked_array_payload_bytes(ComplexF64,50,L+1),
        _checked_array_payload_bytes(Float64,24,nt),
        _checked_array_payload_bytes(ComplexF64,18,nt),
        _checked_array_payload_bytes(Int,36,nt))
    _enforce_payload_limit(est,max_bytes,"conformal FFT","max_bytes")
    families=[_PlanarConformalUFFTFamily(key[1],key[2],key[3],
        [r[1] for r in records[key]],[r[2] for r in records[key]],
        [r[3] for r in records[key]],[r[4] for r in records[key]]) for key in keys_sorted]
    mg=planar_mode_grid(grid,mx,my);ct=Array{ComplexF64}(undef,nmode,2,nf);cm=similar(ct)
    pec=prob.sidewalls===WALL_PEC
    for (f,family) in enumerate(families)
        v=[family.shape[1]*grid.dx family.shape[3]*grid.dx family.shape[5]*grid.dx;
           family.shape[2]*grid.dy family.shape[4]*grid.dy family.shape[6]*grid.dy]
        x=ntuple(j->v[1,j]-v[1,family.free],Val(3));y=ntuple(j->v[2,j]-v[2,family.free],Val(3))
        for n in 0:mg.my-1,m in 0:mg.mx-1
            t=m+1+mg.mx*n;kx,ky=mg.kx[m+1],mg.ky[n+1]
            for (s,(sx,sy)) in enumerate(_PLANAR_CONFORMAL_STORED_SIGNS)
                fx=_planar_triangle_affine_fourier(v,x,sx*kx,sy*ky)
                fy=_planar_triangle_affine_fourier(v,y,sx*kx,sy*ky)
                xt,yt=pec ? (sy,sx) : (sx,sy)
                ct[t,s,f]=(ky*xt*fx-kx*yt*fy)/(4im)
                cm[t,s,f]=(kx*xt*fx+ky*yt*fy)/(4im)
            end
        end
    end
    kt=Matrix{ComplexF64}(undef,nmode,ne*ne);km=similar(kt)
    cte,ctm,scratch,_,_=_planar_mode_workspace(L,false,false)
    for n in 1:mg.my,m in 1:mg.mx
        t=m+mg.mx*(n-1);kx,ky=mg.kx[m],mg.ky[n];kc2=kx*kx+ky*ky
        nte=pec ? ky^2*mg.ic[m]*mg.js[n]+kx^2*mg.is[m]*mg.jc[n] : ky^2*mg.is[m]*mg.jc[n]+kx^2*mg.ic[m]*mg.js[n]
        ntm=pec ? kx^2*mg.ic[m]*mg.js[n]+ky^2*mg.is[m]*mg.jc[n] : kx^2*mg.is[m]*mg.jc[n]+ky^2*mg.ic[m]*mg.js[n]
        nte!=0 && planar_mode_cascade!(cte,prob.stack,omega,kc2,TE_POL,scratch)
        ntm!=0 && planar_mode_cascade!(ctm,prob.stack,omega,kc2,TM_POL,scratch)
        for f in 1:ne,s in 1:ne
            pair=(f-1)*ne+s
            kt[t,pair]=nte==0 ? 0.0im : planar_modal_voltage(cte,levels[f],levels[s])/nte
            km[t,pair]=ntm==0 ? 0.0im : planar_modal_voltage(ctm,levels[f],levels[s])/ntm
        end
    end
    rows=Int[];cols=Int[];values=ComplexF64[]
    _planar_conformal_loss_entries(prob,surface_zs) do p,q,value
        push!(rows,p);push!(cols,q);push!(values,value)
    end
    loss=sparse(rows,cols,values,nb,nb);lattice=zeros(ComplexF64,2Int(nx),2Int(ny))
    forward=FFTW.plan_fft!(lattice);backward=FFTW.plan_bfft!(lattice)
    st=zeros(ComplexF64,nmode,ne);sm=similar(st);ft=similar(st);fm=similar(st)
    return PlanarConformalUFFTOperator(nb,prob,grid,mg,families,ct,cm,kt,km,st,sm,ft,fm,lattice,forward,backward,loss,zeros(ComplexF64,nb))
end

function LinearAlgebra.mul!(y::AbstractVector,A::PlanarConformalUFFTOperator,x::AbstractVector,alpha::Number,beta::Number)
    length(y)==length(x)==A.n || throw(DimensionMismatch("conformal FFT vector size mismatch"))
    (Base.mightalias(x,A.output)||Base.mightalias(x,A.lattice)||Base.mightalias(x,A.source_te)||
        Base.mightalias(x,A.source_tm)||Base.mightalias(x,A.field_te)||Base.mightalias(x,A.field_tm)) &&
        throw(ArgumentError("conformal FFT input must not alias operator workspace"))
    st,sm,ft,fm=A.source_te,A.source_tm,A.field_te,A.field_tm
    fill!(st,0);fill!(sm,0);F=A.lattice;px,py=size(F);mg=A.modes
    for (f,family) in enumerate(A.families)
        fill!(F,0)
        for q in eachindex(family.indices)
            F[family.lattice[q]]+=family.amplitudes[q]*x[family.indices[q]]
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
    fill!(ft,0);fill!(fm,0);ne=size(st,2)
    for f in 1:ne,s in 1:ne,t in axes(st,1)
        pair=(f-1)*ne+s
        ft[t,f]-=A.k_te[t,pair]*st[t,s];fm[t,f]-=A.k_tm[t,pair]*sm[t,s]
    end
    fill!(A.output,0)
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
            A.output[family.indices[q]]+=family.amplitudes[q]*F[family.lattice[q]]
        end
    end
    mul!(A.output,A.local_loss,x,1.,1.)
    for p in eachindex(y)
        y[p]=iszero(beta) ? alpha*A.output[p] : alpha*A.output[p]+beta*y[p]
    end
    y
end
LinearAlgebra.mul!(y::AbstractVector,A::PlanarConformalUFFTOperator,x::AbstractVector)=mul!(y,A,x,1.,0.)
Base.:*(A::PlanarConformalUFFTOperator,x::AbstractVector)=mul!(zeros(ComplexF64,A.n),A,x)

# Exact diagonal: half/self terms and the cross term between a basis's two
# triangles are each nine Fourier frequencies, evaluated by folded FFTs.
function _planar_conformal_ufft_diagonal(A;max_bytes::Integer=_default_max_dense_payload_bytes())
    nb=A.n;mg=A.modes;F=A.lattice;px,py=size(F);ne=size(A.source_te,2)
    _enforce_payload_limit(_checked_payload_sum("conformal FFT diagonal",
        _checked_array_payload_bytes(ComplexF64,nb),_checked_array_payload_bytes(Int,16,nb),
        _checked_array_payload_bytes(Float64,2,nb)),max_bytes,"conformal FFT diagonal","max_bytes")
    d=zeros(ComplexF64,nb);familyid=zeros(Int,2,nb);locations=similar(familyid);amp=zeros(Float64,2,nb)
    for (f,family) in enumerate(A.families)
        fill!(F,0);pair=(family.element-1)*ne+family.element
        for n in 0:mg.my-1,m in 0:mg.mx-1
            t=m+1+mg.mx*n
            for (s1,(sx1,sy1)) in enumerate(_PLANAR_CONFORMAL_SIGNS),(s2,(sx2,sy2)) in enumerate(_PLANAR_CONFORMAL_SIGNS)
                value=-(A.k_te[t,pair]*_planar_conformal_coefficient(A.c_te,t,s1,f)*_planar_conformal_coefficient(A.c_te,t,s2,f)+
                    A.k_tm[t,pair]*_planar_conformal_coefficient(A.c_tm,t,s1,f)*_planar_conformal_coefficient(A.c_tm,t,s2,f))
                F[mod((sx1+sx2)*m,px)+1,mod((sy1+sy2)*n,py)+1]+=value
            end
        end
        A.backward*F
        for q in eachindex(family.indices)
            p=family.indices[q];half=family.halves[q];a=family.amplitudes[q];loc=family.lattice[q]
            d[p]+=a*a*F[loc];familyid[half,p]=f;locations[half,p]=loc;amp[half,p]=a
        end
    end
    groups=Dict{NTuple{4,Int},Vector{Int}}()
    for p in 1:nb
        familyid[2,p]==0 && continue
        a,b=locations[1,p]-1,locations[2,p]-1
        key=(familyid[1,p],familyid[2,p],rem(b,px)-rem(a,px),b÷px-a÷px)
        push!(get!(groups,key,Int[]),p)
    end
    for ((f1,f2,dx,dy),indices) in groups
        fill!(F,0);e=A.families[f1].element;pair=(e-1)*ne+e
        for n in 0:mg.my-1,m in 0:mg.mx-1
            t=m+1+mg.mx*n
            for (s1,(sx1,sy1)) in enumerate(_PLANAR_CONFORMAL_SIGNS),(s2,(sx2,sy2)) in enumerate(_PLANAR_CONFORMAL_SIGNS)
                phase=cispi(sx2*m*dx/A.grid.nx+sy2*n*dy/A.grid.ny)
                value=-(A.k_te[t,pair]*_planar_conformal_coefficient(A.c_te,t,s1,f1)*_planar_conformal_coefficient(A.c_te,t,s2,f2)+
                    A.k_tm[t,pair]*_planar_conformal_coefficient(A.c_tm,t,s1,f1)*_planar_conformal_coefficient(A.c_tm,t,s2,f2))*phase
                F[mod((sx1+sx2)*m,px)+1,mod((sy1+sy2)*n,py)+1]+=value
            end
        end
        A.backward*F
        for p in indices
            d[p]+=2amp[1,p]*amp[2,p]*F[locations[1,p]]
        end
    end
    for p in 1:nb
        d[p]+=A.local_loss[p,p]
    end
    return d
end

"""Conformal FFT solution retaining the exact operator, physical coefficient
columns, Y/S, iteration counts and full voltage-unit relative residuals."""
struct PlanarConformalUFFTResult
    problem::PlanarConformalProblem
    freq::ComplexF64
    omega::ComplexF64
    operator::PlanarConformalUFFTOperator
    currents::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    iterations::Vector{Int}
    relative_residuals::Vector{Float64}
    z0::Vector{ComplexF64}
end
PlanarConformalUFFTResult(prob::PlanarConformalProblem,freq,omega,A,X,Y,S,iterations,residuals)=
    PlanarConformalUFFTResult(prob,freq,omega,A,X,Y,S,iterations,residuals,
        _planar_reference_values([p.z0 for p in prob.ports],length(prob.ports);freq=real(freq)))

"""Solve genuine lattice triangles by bounded restarted GMRES. The exact
FFT diagonal supplies a trace-normalized Jacobi preconditioner. `rtol` gates
an independently recomputed complete operator residual in voltage units.
`nx,ny` declare the physical vertex lattice; no geometry is rasterized."""
function solve_planar_conformal_ufft(prob::PlanarConformalProblem,freq::Number;
        rtol::Real=1e-9,maxiter::Integer=0,memory::Integer=50,restart::Bool=true,
        precondition::Bool=true,max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    omega=2pi*ComplexF64(freq)
    isfinite(omega) && real(omega)>0 || throw(ArgumentError("conformal FFT solve frequency must be finite with positive real part"))
    nb=length(prob.basis.width);np=length(prob.ports)
    isfinite(rtol) && rtol>0 && 0<=maxiter<=typemax(Int) && 1<=memory<=typemax(Int) && nb>0 && np>0 ||
        throw(ArgumentError("conformal FFT solve needs ports and valid Krylov limits"))
    mem=restart ? min(BigInt(memory),BigInt(nb)) : max(BigInt(memory),maxiter==0 ? 2BigInt(nb) : BigInt(maxiter))
    reserve=_checked_payload_sum("conformal FFT solve",
        _checked_array_payload_bytes(ComplexF64,nb,mem+12),_checked_array_payload_bytes(ComplexF64,mem+1,mem),
        _checked_array_payload_bytes(ComplexF64,nb,np),_checked_array_payload_bytes(ComplexF64,5,np,np),
        _checked_array_payload_bytes(Float64,4,nb),_checked_array_payload_bytes(Int,16,nb),
        _checked_array_payload_bytes(Float64,2,np),_checked_array_payload_bytes(Int,np),
        _checked_array_payload_bytes(ComplexF64,2,np))
    _enforce_payload_limit(reserve,max_bytes,"conformal FFT solve","max_bytes")
    refs=_planar_reference_values([p.z0 for p in prob.ports],np;freq=real(freq))
    A=planar_conformal_ufft_operator(prob,freq;max_bytes=max_bytes-reserve,kw...)
    weights=prob.basis.width;D=Diagonal(inv.(weights));d=precondition ? _planar_conformal_ufft_diagonal(A;max_bytes=reserve) : ComplexF64[]
    precondition && any(x->!isfinite(x)||iszero(x),d) && throw(ArgumentError("invalid conformal FFT diagonal"))
    N=precondition ? Diagonal(weights./d) : I
    X=Matrix{ComplexF64}(undef,nb,np);Y=zeros(ComplexF64,np,np);iterations=zeros(Int,np);residuals=zeros(Float64,np)
    rhs=zeros(ComplexF64,nb);residual=similar(rhs)
    for p in 1:np
        for b in 1:nb
            rhs[b]=prob.basis.port[b]==p ? -prob.basis.port_sign[b]*weights[b] : 0.0im
        end
        x,stats=Krylov.gmres(A,rhs;rtol=Float64(rtol)/10,atol=0.,itmax=Int(maxiter),
            memory=min(Int(memory),nb),restart,reorthogonalization=true,M=D,N)
        mul!(residual,A,x);residual.-=rhs
        relative=norm(residual./weights)/norm(rhs./weights)
        isfinite(relative) && relative<=rtol || throw(ErrorException("conformal FFT port $p failed full voltage residual $relative > $rtol after $(stats.niter) iterations"))
        X[:,p].=x;iterations[p]=stats.niter;residuals[p]=relative
        for b in 1:nb
            q=prob.basis.port[b];q==0 || (Y[q,p]+=prob.basis.port_sign[b]*weights[b]*x[b])
        end
    end
    PlanarConformalUFFTResult(prob,ComplexF64(freq),2pi*ComplexF64(freq),A,X,Y,
        planar_y_to_s(Y,refs),iterations,residuals,refs)
end

planar_conformal_current_maps(result::PlanarConformalUFFTResult;kw...)=_planar_conformal_current_maps(result;kw...)
planar_current_maps(result::PlanarConformalUFFTResult;kw...)=_planar_conformal_current_maps(result;kw...)
