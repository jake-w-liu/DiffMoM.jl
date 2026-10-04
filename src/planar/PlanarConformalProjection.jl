# Bounded nonuniform RWG projection used only as a preconditioner.
# Original scalar sources are triangle pulses, with rho = -div(J). Applying
# the local pulse correction before D^T K D preserves charge-free loops.
# The accepted forward solution is checked by the independent original
# analytic modal action in PlanarConformalDefect.jl.

function _planar_projection_sheet_scope(prob::PlanarConformalProblem)
    !isempty(prob.basis.width) && !isempty(prob.ports) ||
        throw(ArgumentError("conformal defect solve requires basis functions and wall ports"))
    prob.sidewalls===WALL_PEC || throw(ArgumentError("conformal defect solve requires PEC sidewalls"))
    all(level->0<level<length(prob.stack.layers),prob.mesh.interfaces) ||
        throw(ArgumentError("conformal defect solve requires interior sheets"))
    prob.stack.bottom.kind===TERM_PEC && prob.stack.top.kind===TERM_PEC ||
        throw(ArgumentError("conformal defect solve requires PEC box covers"))
    all(p->p.wall in (:west,:east,:south,:north),prob.ports) ||
        throw(ArgumentError("conformal defect solve supports wall sources"))
    all(l->l.epsr==l.epsr_z && l.mur==l.mur_z,prob.stack.layers) ||
        throw(ArgumentError("conformal defect solve supports isotropic layers"))
    for level in prob.mesh.interfaces
        lower,upper=prob.stack.layers[level],prob.stack.layers[level+1]
        !iszero(lower.epsr+upper.epsr) && !iszero(inv(lower.mur)+inv(upper.mur)) ||
            throw(ArgumentError("conformal projection static interface denominator is zero"))
    end
    return nothing
end

function _planar_projection_scope(prob::PlanarConformalProblem)
    _planar_projection_sheet_scope(prob)
    level=first(prob.mesh.interfaces)
    all(==(level),prob.mesh.interfaces) ||
        throw(ArgumentError("single-interface projection requires one sheet interface"))
    return level
end

# Count without allocating a triangle-sized unique buffer before resource checks.
function _planar_projection_interface_count(prob)
    count=0
    for level in 1:length(prob.stack.layers)-1
        any(==(level),prob.mesh.interfaces) && (count+=1)
    end
    count
end

function _planar_projection_frequency(freq)
    freq isa Real && isfinite(freq) && freq>0 ||
        throw(ArgumentError("conformal defect solve requires a finite positive real frequency"))
    f=try Float64(freq) catch; throw(ArgumentError("conformal defect frequency must fit Float64")); end
    isfinite(f) && f>0 && isfinite(2pi*f) ||
        throw(ArgumentError("conformal defect frequency and angular frequency must fit positive Float64"))
    return f
end

function _planar_lagrange_polynomials(nodes)
    order=length(nodes);coefficients=zeros(order,order)
    for i in 1:order
        polynomial=[1.];denominator=1.
        for j in 1:order
            j==i && continue
            next=zeros(length(polynomial)+1)
            next[1:end-1].-=nodes[j]*polynomial
            next[2:end].+=polynomial
            polynomial=next;denominator*=nodes[i]-nodes[j]
        end
        coefficients[:,i].=polynomial/denominator
    end
    coefficients
end

function _planar_affine_polynomial_moments_fast(vertices,values,center,spacing,order)
    u=ntuple(i->(vertices[1,i]-center[1])/spacing[1],3)
    v=ntuple(i->(vertices[2,i]-center[2])/spacing[2],3)
    # H(s,t)=∏(1-u_i*s-v_i*t)^(-1). Its bivariate coefficients
    # and one duplicated factor produce the exact Dirichlet(1,1,1)
    # affine simplex moments, including all repeated/zero coordinate cases.
    ex,ey=sum(u),sum(v)
    exx=u[1]*u[2]+u[1]*u[3]+u[2]*u[3]
    eyy=v[1]*v[2]+v[1]*v[3]+v[2]*v[3]
    exy=sum(u[i]*v[j]+u[j]*v[i] for i in 1:3 for j in i+1:3)
    exxx=prod(u);eyyy=prod(v)
    exxy=u[1]*u[2]*v[3]+u[1]*u[3]*v[2]+u[2]*u[3]*v[1]
    exyy=u[1]*v[2]*v[3]+u[2]*v[1]*v[3]+u[3]*v[1]*v[2]
    h=zeros(order,order);h[1,1]=1.
    at(a,p,q)=p<0 || q<0 ? 0. : a[p+1,q+1]
    for p in 0:order-1,q in 0:order-1
        p==q==0 && continue
        h[p+1,q+1]=ex*at(h,p-1,q)+ey*at(h,p,q-1)-exx*at(h,p-2,q)-
            exy*at(h,p-1,q-1)-eyy*at(h,p,q-2)+exxx*at(h,p-3,q)+
            exxy*at(h,p-2,q-1)+exyy*at(h,p-1,q-2)+eyyy*at(h,p,q-3)
    end
    total=zeros(order,order);g=similar(h)
    for i in 1:3
        for p in 0:order-1,q in 0:order-1
            g[p+1,q+1]=h[p+1,q+1]+u[i]*at(g,p-1,q)+v[i]*at(g,p,q-1)
            total[p+1,q+1]+=values[i]*g[p+1,q+1]
        end
    end
    area=abs(_planar_orient2d(vertices[1,1],vertices[2,1],vertices[1,2],vertices[2,2],
        vertices[1,3],vertices[2,3]))/2
    factorials=Float64[factorial(big(i)) for i in 0:2order+1]
    for p in 0:order-1,q in 0:order-1
        total[p+1,q+1]*=2area*factorials[p+1]*factorials[q+1]/factorials[p+q+4]
    end
    total
end

function _planar_physical_charge_incidence(prob)
    rows=Int[];cols=Int[];values=Float64[]
    for b in eachindex(prob.basis.width),half in 1:2
        t=prob.basis.triangles[half,b];t==0 && continue
        density=(half==2 || prob.basis.triangles[2,b]==0 ? 1. : -1.)*prob.basis.width[b]/prob.mesh.areas[t]
        push!(rows,t);push!(cols,b);push!(values,density)
    end
    sparse(rows,cols,values,length(prob.mesh.interfaces),length(prob.basis.width))
end

function _planar_pulse_and_first_moments(v,kx,ky)
    factor,weights=_planar_triangle_fourier_weights(v,kx,ky)
    constant=0.0im;x=0.0im;y=0.0im
    for j in 1:3
        moment=factor*weights[j]
        constant+=moment;x+=(v[1,j]-v[1,1])*moment;y+=(v[2,j]-v[2,1])*moment
    end
    constant,x,y
end

struct _PlanarConformalProjection{PF,PB} <: AbstractMatrix{ComplexF64}
    x::SparseMatrixCSC{Float64,Int}
    y::SparseMatrixCSC{Float64,Int}
    pulse::SparseMatrixCSC{Float64,Int}
    incidence::SparseMatrixCSC{Float64,Int}
    vector_correction::SparseMatrixCSC{ComplexF64,Int}
    charge_correction::SparseMatrixCSC{ComplexF64,Int}
    kernels::NTuple{3,Matrix{ComplexF64}}
    lattice::Matrix{ComplexF64}
    modes::Matrix{ComplexF64}
    forward::PF
    backward::PB
    charge::Vector{ComplexF64}
    charge_field::Vector{ComplexF64}
    output::Vector{ComplexF64}
end
Base.size(A::_PlanarConformalProjection)=(size(A.x,2),size(A.x,2))
Base.size(A::_PlanarConformalProjection,d::Integer)=d<1 ? throw(ArgumentError("positive dimension required")) : d<=2 ? size(A.x,2) : 1
Base.eltype(::Type{<:_PlanarConformalProjection})=ComplexF64

@inline function _planar_projection_trig_coefficient(component,sx,sy)
    component==1 && return sy/(4im) # cos(x) sin(y)
    component==2 && return sx/(4im) # sin(x) cos(y)
    return -sx*sy/4 # sin(x) sin(y)
end

function _planar_folded_projection_kernels(prob,nx,ny,frequency,modes;workspace,sigma=.5prob.stack.a/nx)
    level=_planar_projection_scope(prob);lower,upper=prob.stack.layers[level],prob.stack.layers[level+1]
    omega=2pi*frequency;magnetic=im*omega*_MU0/(inv(lower.mur)+inv(upper.mur))
    electric=inv(im*omega*_EPS0*(lower.epsr+upper.epsr))
    workspace.prob===prob && workspace.frequency==frequency && workspace.modes==modes ||
        throw(ArgumentError("projection kernels require the matching original modal workspace"))
    # Wave numbers and normalization depend only on the box and mode counts.
    # Reuse the original action's mode grid/cascades rather than allocating a
    # second complete modal workspace beside the retained one.
    mg=workspace.grid;cte,ctm,scratch=workspace.cte,workspace.ctm,workspace.scratch
    kx=zeros(ComplexF64,2nx,2ny);ky=similar(kx);kc=similar(kx);fill!(ky,0);fill!(kc,0)
    for n in 1:modes,m in 1:modes
        k2=mg.kx[m]^2+mg.ky[n]^2;iszero(k2) && continue
        normx,normy=mg.ic[m]*mg.js[n],mg.is[m]*mg.jc[n];normc=mg.is[m]*mg.js[n]
        normte=mg.ky[n]^2*normx+mg.kx[m]^2*normy
        normte!=0 && planar_mode_cascade!(cte,prob.stack,omega,k2,TE_POL,scratch)
        normc!=0 && planar_mode_cascade!(ctm,prob.stack,omega,k2,TM_POL,scratch)
        vte=normte==0 ? 0.0im : planar_modal_voltage(cte,level,level)
        potential=normc==0 ? 0.0im : (planar_modal_voltage(ctm,level,level)-vte)/k2
        window=erfc(sqrt(k2)*sigma)
        vte+=(window-1)*magnetic/sqrt(k2);potential+=(window-1)*electric/sqrt(k2)
        isfinite(vte) && isfinite(potential) || throw(ArgumentError("nonfinite conformal projection kernel; check box or interface resonances"))
        i,j=mod(m-1,2nx)+1,mod(n-1,2ny)+1
        normx==0 || (kx[i,j]-=vte/normx)
        normy==0 || (ky[i,j]-=vte/normy)
        normc==0 || (kc[i,j]-=potential/normc)
    end
    kx,ky,kc
end

function _planar_charge_projection_fft(prob,projection,pulses,frequency,modes,vector_correction,charge_correction;
        workspace,sigma=.5prob.stack.a/projection.nx,max_bytes=512_000_000)
    nx,ny=projection.nx,projection.ny;nt=length(prob.mesh.interfaces);nb=length(prob.basis.width)
    # Check actual sparse arrays before allocating the folded grid/workspaces.
    sparsebytes=sum(sizeof(eltype(a))*length(a) for P in (projection.x,projection.y,pulses.pulse,vector_correction,charge_correction) for a in (P.nzval,P.colptr,P.rowval))
    reserve=_checked_payload_sum("projection FFT retained arrays",sparsebytes,
        _checked_array_payload_bytes(ComplexF64,5,2BigInt(nx),2BigInt(ny)),
        _checked_array_payload_bytes(ComplexF64,2nt+nb),
        _checked_array_payload_bytes(Float64,6,nb),
        _checked_array_payload_bytes(Int,6,nb))
    _enforce_payload_limit(reserve,max_bytes,"projection FFT retained arrays","max_bytes")
    D=_planar_physical_charge_incidence(prob);kernels=_planar_folded_projection_kernels(prob,nx,ny,frequency,modes;workspace,sigma)
    F=zeros(ComplexF64,2nx,2ny);M=similar(F)
    _PlanarConformalProjection(projection.x,projection.y,pulses.pulse,D,vector_correction,charge_correction,kernels,
        F,M,FFTW.plan_fft!(F),FFTW.plan_bfft!(F),zeros(ComplexF64,nt),zeros(ComplexF64,nt),zeros(ComplexF64,nb))
end

function _planar_projection_fft_component!(A,points,input,output,component)
    F,Q=A.lattice,A.modes;px,py=size(F);fill!(F,0)
    for b in axes(points,2),p in nzrange(points,b)
        F[points.rowval[p]]+=points.nzval[p]*input[b]
    end
    A.forward*F
    for n in 0:py-1,m in 0:px-1
        value=0.0im
        for (sx,sy) in ((1,1),(1,-1),(-1,1),(-1,-1))
            value+=_planar_projection_trig_coefficient(component,sx,sy)*F[mod(-sx*m,px)+1,mod(-sy*n,py)+1]
        end
        Q[m+1,n+1]=A.kernels[component][m+1,n+1]*value
    end
    fill!(F,0)
    for n in 0:py-1,m in 0:px-1,(sx,sy) in ((1,1),(1,-1),(-1,1),(-1,-1))
        F[mod(sx*m,px)+1,mod(sy*n,py)+1]+=_planar_projection_trig_coefficient(component,sx,sy)*Q[m+1,n+1]
    end
    A.backward*F
    for b in axes(points,2),p in nzrange(points,b)
        output[b]+=points.nzval[p]*F[points.rowval[p]]
    end
    output
end

function LinearAlgebra.mul!(out::AbstractVector{ComplexF64},A::_PlanarConformalProjection,x::AbstractVector{ComplexF64})
    length(out)==length(x)==size(A,1) || throw(DimensionMismatch())
    any(a->Base.mightalias(x,a),(A.output,A.charge,A.charge_field,A.lattice,A.modes)) &&
        throw(ArgumentError("conformal projection input must not alias its workspace"))
    fill!(A.output,0);fill!(A.charge_field,0)
    _planar_projection_fft_component!(A,A.x,x,A.output,1)
    _planar_projection_fft_component!(A,A.y,x,A.output,2)
    mul!(A.charge,A.incidence,x)
    _planar_projection_fft_component!(A,A.pulse,A.charge,A.charge_field,3)
    mul!(A.charge_field,A.charge_correction,A.charge,1.,1.)
    mul!(A.output,transpose(A.incidence),A.charge_field,1.,1.)
    mul!(A.output,A.vector_correction,x,1.,1.)
    copyto!(out,A.output)
end
Base.:*(A::_PlanarConformalProjection,x::AbstractVector{ComplexF64})=mul!(similar(x),A,x)

struct _PlanarProjectionLocalPairs
    row::Vector{Int}
    column::Vector{Int}
    offsets::Vector{Int}
    visits::Int
    references::Int
end

function _planar_projection_support_boxes(prob; pulses=false)
    count=pulses ? length(prob.mesh.interfaces) : length(prob.basis.width)
    lo=fill(Inf,2,count);hi=fill(-Inf,2,count)
    for b in 1:count,half in 1:(pulses ? 1 : 2)
        t=pulses ? b : prob.basis.triangles[half,b];t==0 && continue
        for vertex in prob.mesh.triangles[:,t],d in 1:2
            x=prob.mesh.vertices[d,vertex]
            lo[d,b]=min(lo[d,b],x);hi[d,b]=max(hi[d,b],x)
        end
    end
    lo,hi
end

@inline _planar_projection_box_distance(lo,hi,a,b)=hypot(
    max(0.,lo[1,a]-hi[1,b],lo[1,b]-hi[1,a]),
    max(0.,lo[2,a]-hi[2,b],lo[2,b]-hi[2,a]))

# Binning never visits all pairs merely to find the local set. Its actual
# reference and visit counts have independent fixed limits: a large support
# covering many bins, or a dense cluster, cannot silently become an unbounded
# constructor. The two passes reserve the exact pair arrays before filling.
function _planar_projection_local_pairs(lo,hi,a,b,radius;
        max_pairs=2_000_000,max_references=8_000_000,max_visits=80_000_000,
        max_bytes=512_000_000)
    size(lo)==size(hi) && size(lo,1)==2 || throw(DimensionMismatch("support boxes"))
    n=size(lo,2)
    a isa Real && b isa Real && isfinite(a) && isfinite(b) && a>0 && b>0 ||
        throw(ArgumentError("finite positive physical box dimensions required"))
    isfinite(radius) && radius>0 || throw(ArgumentError("finite positive local radius required"))
    all(isfinite,lo) && all(isfinite,hi) && all(lo.<=hi) || throw(ArgumentError("finite ordered boxes required"))
    all(0 .<=lo[1,:].<=hi[1,:].<=a) && all(0 .<=lo[2,:].<=hi[2,:].<=b) ||
        throw(ArgumentError("boxes must be inside the physical PEC box"))
    all(x->x isa Integer && x>=0,(max_pairs,max_references,max_visits,max_bytes)) ||
        throw(ArgumentError("nonnegative integer construction limits required"))
    bx=max(1,_checked_payload_sum("projection x bin count",ceil(BigInt,BigFloat(a)/BigFloat(radius))))
    by=max(1,_checked_payload_sum("projection y bin count",ceil(BigInt,BigFloat(b)/BigFloat(radius))))
    bins=_checked_payload_sum("projection bin count",BigInt(bx)*by)
    base=_checked_payload_sum("projection local bin arrays",
        _checked_array_payload_bytes(Int,3,bins+1),
        _checked_array_payload_bytes(Int,3,n+1))
    _enforce_payload_limit(base,max_bytes,"projection local bins","max_bytes")
    counts=zeros(Int,bins);offsets=zeros(Int,bins+1);cursor=zeros(Int,bins)
    seen=zeros(Int,n);pair_offsets=zeros(Int,n+1)
    xbin(x)=clamp(floor(Int,x/a*bx)+1,1,bx)
    ybin(y)=clamp(floor(Int,y/b*by)+1,1,by)
    refs=0
    for q in 1:n
        xl,xh=xbin(lo[1,q]),xbin(hi[1,q]);yl,yh=ybin(lo[2,q]),ybin(hi[2,q])
        added=BigInt(xh-xl+1)*(yh-yl+1)
        added<=max_references-refs || throw(ArgumentError("projection bin references exceed max_references"))
        refs+=Int(added)
        for j in yl:yh,i in xl:xh;counts[i+bx*(j-1)]+=1;end
    end
    binbytes=_checked_payload_sum("projection local bin references",base,
        _checked_array_payload_bytes(Int,refs))
    _enforce_payload_limit(binbytes,max_bytes,"projection local bins","max_bytes")
    offsets[1]=1
    for k in 1:bins;offsets[k+1]=offsets[k]+counts[k];cursor[k]=offsets[k];end
    ids=Vector{Int}(undef,refs)
    for q in 1:n,j in ybin(lo[2,q]):ybin(hi[2,q]),i in xbin(lo[1,q]):xbin(hi[1,q])
        k=i+bx*(j-1);ids[cursor[k]]=q;cursor[k]+=1
    end
    paircount=0;visits=0
    row=Int[];column=Int[]
    for pass in 1:2
        fill!(seen,0);position=1;passvisits=0
        for q in 1:n
            pass==1 && (pair_offsets[q]=position)
            for j in ybin(max(0.,lo[2,q]-radius)):ybin(min(b,hi[2,q]+radius)),
                    i in xbin(max(0.,lo[1,q]-radius)):xbin(min(a,hi[1,q]+radius))
                k=i+bx*(j-1)
                count=offsets[k+1]-offsets[k]
                count<=max_visits÷2-passvisits || throw(ArgumentError("projection neighbor visits exceed max_visits"))
                passvisits+=count
                for p in offsets[k]:offsets[k+1]-1
                    r=ids[p]
                    (r<q || seen[r]==q) && continue
                    seen[r]=q
                    _planar_projection_box_distance(lo,hi,q,r)<=radius || continue
                    if pass==1
                        position<=max_pairs || throw(ArgumentError("projection local pairs exceed max_pairs"))
                    else
                        row[position]=r;column[position]=q
                    end
                    position+=1
                end
            end
        end
        if pass==1
            paircount=position-1;visits=passvisits;pair_offsets[n+1]=position
            bytes=_checked_payload_sum("projection local pairs and bins",binbytes,
                _checked_array_payload_bytes(Int,2,paircount))
            _enforce_payload_limit(bytes,max_bytes,"projection local pairs and bins","max_bytes")
            row=Vector{Int}(undef,paircount);column=similar(row)
        else
            position==paircount+1 && passvisits==visits || error("local pair passes differ")
        end
    end
    for q in 1:n
        sort!(view(row,pair_offsets[q]:pair_offsets[q+1]-1))
    end
    _PlanarProjectionLocalPairs(row,column,pair_offsets,2visits,refs)
end

function _planar_projection_exact_workspace_payload(prob,modes,block)
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces);chunk=min(block,BigInt(modes)^2)
    _checked_payload_sum("local analytic modal workspace",
        _checked_array_payload_bytes(Float64,2nb+nt,chunk),
        _checked_array_payload_bytes(ComplexF64,3,chunk),
        _checked_array_payload_bytes(Float64,6nt+6nb),
        _checked_array_payload_bytes(ComplexF64,6nt),
        _checked_array_payload_bytes(Float64,7,2BigInt(modes)),
        _checked_array_payload_bytes(ComplexF64,50,length(prob.stack.layers)+1))
end

function _planar_projection_exact_workspace(prob,frequency,modes,block;max_bytes=512_000_000)
    modes isa Integer && modes>=1 && block isa Integer && block>=1 ||
        throw(ArgumentError("positive modal and block counts required"))
    frequency=_planar_projection_frequency(frequency)
    level=_planar_projection_scope(prob)
    payload=_planar_projection_exact_workspace_payload(prob,modes,block)
    _enforce_payload_limit(payload,max_bytes,"local analytic modal workspace","max_bytes")
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces)
    nmode=_checked_payload_sum("local modal count",BigInt(modes)^2);chunk=min(block,nmode)
    triangles=zeros(2,3,nt);frees=zeros(2,2,nb);amplitudes=zeros(2,nb)
    for t in 1:nt,j in 1:3,d in 1:2;triangles[d,j,t]=prob.mesh.vertices[d,prob.mesh.triangles[j,t]];end
    for b in 1:nb,half in 1:2
        t=prob.basis.triangles[half,b];t==0 && continue
        edgea,edgeb=prob.basis.edges[:,b]
        free=only(i for i in prob.mesh.triangles[:,t] if i!=edgea && i!=edgeb)
        for d in 1:2;frees[d,half,b]=prob.mesh.vertices[d,free]-triangles[d,1,t];end
        amplitudes[half,b]=(half==2 || prob.basis.triangles[2,b]==0 ? -1. : 1.)*prob.basis.width[b]/(2prob.mesh.areas[t])
    end
    cte,ctm,scratch,_,_=_planar_mode_workspace(length(prob.stack.layers),false,false)
    grid=planar_mode_grid(CellGrid(prob.stack.a,prob.stack.b,1,1),modes,modes)
    (;prob,frequency,modes,nmode,chunk,triangles,frees,amplitudes,cte,ctm,scratch,grid,level,payload,
        wx=zeros(nb,chunk),wy=zeros(nb,chunk),wc=zeros(nt,chunk),
        kernels=zeros(ComplexF64,3,chunk),plus=zeros(ComplexF64,3,nt),minus=zeros(ComplexF64,3,nt))
end

function _planar_projection_exact_weights_block!(w,start,count)
    prob=w.prob;nb=length(prob.basis.width);nt=length(prob.mesh.interfaces)
    for j in 1:count
        index=start+j-2;m,n=rem(index,w.modes)+1,index÷w.modes+1
        kx,ky=w.grid.kx[m],w.grid.ky[n];k2=kx^2+ky^2
        for t in 1:nt
            plus=_planar_pulse_and_first_moments(view(w.triangles,:,:,t),kx,ky)
            minus=_planar_pulse_and_first_moments(view(w.triangles,:,:,t),kx,-ky)
            for d in 1:3;w.plus[d,t]=plus[d];w.minus[d,t]=minus[d];end
            w.wc[t,j]=(real(minus[1])-real(plus[1]))/2
        end
        for b in 1:nb
            x,y=0.,0.
            for half in 1:2
                t=prob.basis.triangles[half,b];t==0 && continue
                xp=w.amplitudes[half,b]*(w.plus[2,t]-w.frees[1,half,b]*w.plus[1,t])
                xm=w.amplitudes[half,b]*(w.minus[2,t]-w.frees[1,half,b]*w.minus[1,t])
                yp=w.amplitudes[half,b]*(w.plus[3,t]-w.frees[2,half,b]*w.plus[1,t])
                ym=w.amplitudes[half,b]*(w.minus[3,t]-w.frees[2,half,b]*w.minus[1,t])
                x+=(imag(xp)-imag(xm))/2;y+=(imag(yp)+imag(ym))/2
            end
            w.wx[b,j]=x;w.wy[b,j]=y
        end
        normx,normy=w.grid.ic[m]*w.grid.js[n],w.grid.is[m]*w.grid.jc[n]
        normc=w.grid.is[m]*w.grid.js[n];normte=ky^2*normx+kx^2*normy
        if iszero(k2)
            w.kernels[:,j].=0;continue
        end
        normte!=0 && planar_mode_cascade!(w.cte,prob.stack,2pi*w.frequency,k2,TE_POL,w.scratch)
        normc!=0 && planar_mode_cascade!(w.ctm,prob.stack,2pi*w.frequency,k2,TM_POL,w.scratch)
        vte=normte==0 ? 0.0im : planar_modal_voltage(w.cte,w.level,w.level)
        potential=normc==0 ? 0.0im : (planar_modal_voltage(w.ctm,w.level,w.level)-vte)/k2
        w.kernels[1,j]=normx==0 ? 0.0im : -vte/normx
        w.kernels[2,j]=normy==0 ? 0.0im : -vte/normy
        w.kernels[3,j]=normc==0 ? 0.0im : -potential/normc
        all(isfinite,view(w.kernels,:,j)) || throw(ArgumentError("nonfinite original conformal modal kernel; check box resonances"))
    end
    w
end

@inline function _planar_projection_kahan_add(sum,error,value)
    corrected=value-error;updated=sum+corrected
    updated,(updated-sum)-corrected
end

function _planar_projection_exact_local_entries!(vector,scalar,w,vpairs,cpairs)
    fill!(vector,0);fill!(scalar,0)
    ev=zeros(ComplexF64,length(vector));ec=zeros(ComplexF64,length(scalar))
    for start in 1:w.chunk:w.nmode
        count=min(w.chunk,w.nmode-start+1);_planar_projection_exact_weights_block!(w,start,count)
        for p in eachindex(vector)
            a,b=vpairs.row[p],vpairs.column[p];sum,error=vector[p],ev[p]
            for j in 1:count
                value=w.kernels[1,j]*w.wx[a,j]*w.wx[b,j]+w.kernels[2,j]*w.wy[a,j]*w.wy[b,j]
                sum,error=_planar_projection_kahan_add(sum,error,value)
            end
            vector[p],ev[p]=sum,error
        end
        for p in eachindex(scalar)
            a,b=cpairs.row[p],cpairs.column[p];sum,error=scalar[p],ec[p]
            for j in 1:count
                sum,error=_planar_projection_kahan_add(sum,error,w.kernels[3,j]*w.wc[a,j]*w.wc[b,j])
            end
            scalar[p],ec[p]=sum,error
        end
    end
    vector,scalar
end

function _planar_projection_subtract_fft_local!(values,A,points,pairs,components)
    n=size(points,2);input=zeros(ComplexF64,n);output=similar(input)
    for b in 1:n
        fill!(input,0);fill!(output,0);input[b]=1
        for component in components
            _planar_projection_fft_component!(A,component==1 ? A.x : component==2 ? A.y : A.pulse,input,output,component)
        end
        for p in pairs.offsets[b]:pairs.offsets[b+1]-1;values[p]-=output[pairs.row[p]];end
    end
    values
end

function _planar_projection_symmetric_correction(values,pairs,n)
    entries=2length(values)-count(p->pairs.row[p]==pairs.column[p],eachindex(values))
    row=Vector{Int}(undef,entries);column=similar(row);data=Vector{ComplexF64}(undef,entries);q=1
    for p in eachindex(values)
        a,b=pairs.row[p],pairs.column[p]
        row[q]=a;column[q]=b;data[q]=values[p];q+=1
        a==b && continue
        row[q]=b;column[q]=a;data[q]=values[p];q+=1
    end
    sparse(row,column,data,n,n)
end

function _planar_projection_local_value(values,pairs,a,b)
    a<b && ((a,b)=(b,a))
    lower,upper=pairs.offsets[b],pairs.offsets[b+1]-1
    while lower<=upper
        middle=lower+(upper-lower)÷2;row=pairs.row[middle]
        row==a && return values[middle]
        row<a ? (lower=middle+1) : (upper=middle-1)
    end
    error("required self/shared-edge local entry absent")
end

function _planar_projection_exact_local_diagonal(prob,vector,scalar,vpairs,cpairs)
    nb=length(prob.basis.width);diagonal=zeros(ComplexF64,nb)
    for b in 1:nb
        diagonal[b]=_planar_projection_local_value(vector,vpairs,b,b)
        for h in 1:2,k in 1:2
            t,u=prob.basis.triangles[h,b],prob.basis.triangles[k,b]
            (t==0 || u==0) && continue
            dt=(h==2 || prob.basis.triangles[2,b]==0 ? 1. : -1.)*prob.basis.width[b]/prob.mesh.areas[t]
            du=(k==2 || prob.basis.triangles[2,b]==0 ? 1. : -1.)*prob.basis.width[b]/prob.mesh.areas[u]
            diagonal[b]+=dt*du*_planar_projection_local_value(scalar,cpairs,t,u)
        end
    end
    diagonal
end

_planar_projection_sparse_bytes(P)=sum(sizeof(eltype(a))*length(a) for a in (P.nzval,P.rowval,P.colptr))
_planar_projection_pair_bytes(p)=sizeof(p.row)+sizeof(p.column)+sizeof(p.offsets)
function _planar_projection_fft_owned_bytes(A)
    sum(_planar_projection_sparse_bytes,(A.x,A.y,A.pulse,A.incidence,A.vector_correction,A.charge_correction))+
        sum(sizeof,(A.kernels...,A.lattice,A.modes,A.charge,A.charge_field,A.output))
end

# Visit integration patches with a bounded depth-first stack. Neither a
# complete patch list nor a dense mesh-to-grid array is constructed first.
function _planar_projection_visit_patches(visitor,vertices,x,y,spacing;
        max_patches=65536,max_bytes=512_000_000)
    _enforce_payload_limit(256,max_bytes,"projection initial depth-first patch","max_bytes")
    pending=[(Matrix(vertices),x,y)];visited=0
    while !isempty(pending)
        v,u,w=pop!(pending)
        extentx=max(v[1,1],v[1,2],v[1,3])-min(v[1,1],v[1,2],v[1,3])
        extenty=max(v[2,1],v[2,2],v[2,3])-min(v[2,1],v[2,2],v[2,3])
        if max(extentx/spacing[1],extenty/spacing[2])<=2
            visited<max_patches || throw(ArgumentError("projection patch visits exceed max_patches"))
            visited+=1;visitor(v,u,w);continue
        end
        visited+length(pending)+4<=max_patches || throw(ArgumentError("projection patch leaves exceed max_patches"))
        _enforce_payload_limit(_checked_array_payload_bytes(UInt8,256,length(pending)+5),
            max_bytes,"projection depth-first patches","max_bytes")
        points=hcat(v,(v[:,1]+v[:,2])/2,(v[:,2]+v[:,3])/2,(v[:,3]+v[:,1])/2)
        ux=(u...,(u[1]+u[2])/2,(u[2]+u[3])/2,(u[3]+u[1])/2)
        wy=(w...,(w[1]+w[2])/2,(w[2]+w[3])/2,(w[3]+w[1])/2)
        for ids in ((1,4,6),(4,2,5),(6,5,3),(4,5,6))
            push!(pending,(points[:,collect(ids)],ntuple(i->ux[ids[i]],3),ntuple(i->wy[ids[i]],3)))
        end
    end
    visited
end

function _planar_projection_bounded_points(prob,nx,ny,order;pulses=false,
        max_patch_visits=8_000_000,max_stencil_entries=5_000_000,max_bytes=512_000_000)
    dx,dy=prob.stack.a/nx,prob.stack.b/ny;px,py=2nx,2ny
    n=pulses ? length(prob.mesh.interfaces) : length(prob.basis.width)
    nodes=collect(0:order-1).-((order-1)÷2);polynomial=_planar_lagrange_polynomials(nodes)
    rows=Int[];cols=Int[];wx=Float64[];wy=Float64[];wc=Float64[];patchcount=0
    extra=_checked_array_payload_bytes(Float64,16,order,order)+
        _checked_array_payload_bytes(Int,6,n+1)
    bytes_for(entries,active)=_checked_payload_sum("bounded projection stencils",extra,
        _checked_array_payload_bytes(UInt8,pulses ? 128 : 256,entries),
        _checked_array_payload_bytes(UInt8,256,active+order^2+64))
    _enforce_payload_limit(bytes_for(0,0),max_bytes,"bounded projection stencils","max_bytes")
    for b in 1:n
        accumulated=Dict{Int,NTuple{3,Float64}}()
        for half in 1:(pulses ? 1 : 2)
            t=pulses ? b : prob.basis.triangles[half,b];t==0 && continue
            _,x,y=pulses ? (t,(1.,1.,1.),(0.,0.,0.)) : _planar_conformal_half_values(prob,b,half)
            charge=pulses ? 0. : (half==2 || prob.basis.triangles[2,b]==0 ? 1. : -1.)*prob.basis.width[b]/prob.mesh.areas[t]
            vertices=prob.mesh.vertices[:,prob.mesh.triangles[:,t]]
            current=bytes_for(length(rows),length(accumulated))
            visit=_planar_projection_visit_patches(vertices,x,y,(dx,dy);
                    max_patches=min(65536,max_patch_visits-patchcount),max_bytes=max_bytes-current) do v,u,w
                _enforce_payload_limit(bytes_for(length(rows),length(accumulated)),max_bytes,"bounded projection stencils","max_bytes")
                ox=round(Int,sum(v[1,:])/(3dx));oy=round(Int,sum(v[2,:])/(3dy));center=(ox*dx,oy*dy)
                mx=_planar_affine_polynomial_moments_fast(v,u,center,(dx,dy),order)
                vx=transpose(polynomial)*mx*polynomial
                vy=pulses ? nothing : transpose(polynomial)*_planar_affine_polynomial_moments_fast(v,w,center,(dx,dy),order)*polynomial
                vc=pulses ? nothing : transpose(polynomial)*_planar_affine_polynomial_moments_fast(v,(charge,charge,charge),center,(dx,dy),order)*polynomial
                for j in 1:order,i in 1:order
                    row=mod(ox+nodes[i],px)+1+px*mod(oy+nodes[j],py)
                    haskey(accumulated,row) || length(accumulated)<max_stencil_entries-length(rows) ||
                        throw(ArgumentError("projection stencils exceed max_stencil_entries"))
                    old=get(accumulated,row,(0.,0.,0.))
                    accumulated[row]=(old[1]+vx[i,j],old[2]+(pulses ? 0. : vy[i,j]),old[3]+(pulses ? 0. : vc[i,j]))
                end
            end
            patchcount+=visit
        end
        length(accumulated)<=max_stencil_entries-length(rows) || throw(ArgumentError("projection stencils exceed max_stencil_entries"))
        _enforce_payload_limit(bytes_for(length(rows)+length(accumulated),0),max_bytes,"bounded projection triplets and CSC","max_bytes")
        for (row,values) in accumulated
            push!(rows,row);push!(cols,b);push!(wx,values[1])
            pulses || (push!(wy,values[2]);push!(wc,values[3]))
        end
    end
    if pulses
        return (;pulse=sparse(rows,cols,wx,px*py,n),nx,ny,order,patchcount)
    end
    (;x=sparse(rows,cols,wx,px*py,n),y=sparse(rows,cols,wy,px*py,n),charge=sparse(rows,cols,wc,px*py,n),nx,ny,order,patchcount)
end

function _planar_bounded_charge_projection_fft(prob,frequency;
        modes=128,nx=64,ny=nx,order=7,sigma=.5prob.stack.a/nx,radius=6sigma,block=512,
        max_pairs=2_000_000,max_references=8_000_000,max_visits=80_000_000,
        max_patch_visits=8_000_000,max_stencil_entries=5_000_000,
        max_modal_terms=4_000_000,max_pair_mode_products=5_000_000_000,max_bytes=512_000_000)
    frequency=_planar_projection_frequency(frequency)
    _planar_projection_scope(prob)
    all(x->x isa Integer && 1<=x<=typemax(Int),(modes,nx,ny,block)) || throw(ArgumentError("positive representable mode/grid/block counts required"))
    order isa Integer && 3<=order<=9 || throw(ArgumentError("projection order must be3:9"))
    all(x->x isa Real && isfinite(x) && x>0 && isfinite(Float64(x)) && Float64(x)>0,(sigma,radius)) ||
        throw(ArgumentError("finite positive Float64 sigma/radius required"))
    all(x->x isa Integer && x>=0,(max_pairs,max_references,max_visits,max_patch_visits,max_stencil_entries,
            max_modal_terms,max_pair_mode_products,max_bytes)) ||
        throw(ArgumentError("nonnegative integer constructor limits required"))
    BigInt(modes)^2<=max_modal_terms || throw(ArgumentError("modal terms exceed max_modal_terms"))
    modes,nx,ny,block=Int(modes),Int(nx),Int(ny),Int(block)
    sigma,radius=Float64(sigma),Float64(radius)
    # Validate scope and reserve analytic blocks and FFT grids before projection allocation.
    modal=_planar_projection_exact_workspace_payload(prob,modes,block)
    initial=_checked_payload_sum("initial local projection reservation",modal,
        _checked_array_payload_bytes(ComplexF64,5,2BigInt(nx),2BigInt(ny)))
    _enforce_payload_limit(initial,max_bytes,"initial local projection reservation","max_bytes")
    w=_planar_projection_exact_workspace(prob,frequency,modes,block;max_bytes)
    projection=_planar_projection_bounded_points(prob,nx,ny,order;max_patch_visits,max_stencil_entries,max_bytes=max_bytes-modal)
    projectedbytes=sum(_planar_projection_sparse_bytes,(projection.x,projection.y,projection.charge))
    pulses=_planar_projection_bounded_points(prob,nx,ny,order;pulses=true,
        max_patch_visits=max_patch_visits-projection.patchcount,
        max_stencil_entries=max_stencil_entries-nnz(projection.x),max_bytes=max_bytes-modal-projectedbytes)
    projectedbytes+=_planar_projection_sparse_bytes(pulses.pulse)
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces)
    boxes=_checked_array_payload_bytes(Float64,4,BigInt(nb)+nt)
    _enforce_payload_limit(_checked_payload_sum("projection support boxes",modal,projectedbytes,boxes),
        max_bytes,"projection support boxes","max_bytes")
    # Grid folding includes all signed box images. The direct physical-box
    # query includes every reflected/2a,2b-periodic image within the radius;
    # proof/falsifiers below also exercise wall-touching supports.
    vlo,vhi=_planar_projection_support_boxes(prob);clo,chi=_planar_projection_support_boxes(prob;pulses=true)
    available=max_bytes-modal-projectedbytes-boxes
    vpairs=_planar_projection_local_pairs(vlo,vhi,prob.stack.a,prob.stack.b,radius;
        max_pairs,max_references,max_visits,max_bytes=available)
    cpairs=_planar_projection_local_pairs(clo,chi,prob.stack.a,prob.stack.b,radius;
        max_pairs=max_pairs-length(vpairs.row),max_references=max_references-vpairs.references,
        max_visits=max_visits-vpairs.visits,max_bytes=available-_planar_projection_pair_bytes(vpairs))
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces);paircount=length(vpairs.row)+length(cpairs.row)
    BigInt(paircount)*w.nmode<=max_pair_mode_products || throw(ArgumentError("local pair modal work exceeds max_pair_mode_products"))
    # Values+compensations, symmetric sparse conversion triplets and its CSC,
    # unit-column FFT buffers, and both pair lists coexist during construction.
    localpayload=_checked_payload_sum("local correction constructor",
        modal,projectedbytes,boxes,_planar_projection_pair_bytes(vpairs),_planar_projection_pair_bytes(cpairs),
        _checked_array_payload_bytes(UInt8,256,paircount),
        _checked_array_payload_bytes(ComplexF64,2max(nb,nt)+nb))
    _enforce_payload_limit(localpayload,max_bytes,"local correction constructor","max_bytes")
    base=_planar_charge_projection_fft(prob,projection,pulses,frequency,modes,SparseArrays.spzeros(ComplexF64,nb,nb),SparseArrays.spzeros(ComplexF64,nt,nt);
        workspace=w,sigma,max_bytes=max_bytes-localpayload)
    vector=zeros(ComplexF64,length(vpairs.row));scalar=zeros(ComplexF64,length(cpairs.row))
    _planar_projection_exact_local_entries!(vector,scalar,w,vpairs,cpairs)
    diagonal=_planar_projection_exact_local_diagonal(prob,vector,scalar,vpairs,cpairs)
    # The public solve validates the combined electromagnetic + sheet Gram
    # diagonal; a sheet impedance may remove an otherwise zero entry.
    all(isfinite,diagonal) ||
        throw(ArgumentError("conformal defect preconditioner has a nonfinite original diagonal"))
    _planar_projection_subtract_fft_local!(vector,base,base.x,vpairs,(1,2))
    _planar_projection_subtract_fft_local!(scalar,base,base.pulse,cpairs,(3,))
    vc=_planar_projection_symmetric_correction(vector,vpairs,nb);cc=_planar_projection_symmetric_correction(scalar,cpairs,nt)
    A=_PlanarConformalProjection(base.x,base.y,base.pulse,base.incidence,vc,cc,base.kernels,
        base.lattice,base.modes,base.forward,base.backward,base.charge,base.charge_field,base.output)
    owned=_planar_projection_fft_owned_bytes(A)
    constructionbound=owned+localpayload
    _enforce_payload_limit(constructionbound,max_bytes,"local correction aggregate construction","max_bytes")
    (;operator=A,workspace=w,vpairs,cpairs,owned,constructionbound,projection,pulses,radius,sigma,diagonal)
end
