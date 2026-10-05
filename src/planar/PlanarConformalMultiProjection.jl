# Bounded multilevel sheet preconditioner. Original modal action is separate.

struct _PlanarConformalMultiProjection{F,B} <: AbstractMatrix{ComplexF64}
    x::SparseMatrixCSC{Float64,Int}
    y::SparseMatrixCSC{Float64,Int}
    pulse::SparseMatrixCSC{Float64,Int}
    incidence::SparseMatrixCSC{Float64,Int}
    vector_correction::SparseMatrixCSC{ComplexF64,Int}
    charge_correction::SparseMatrixCSC{ComplexF64,Int}
    kernels::Matrix{NTuple{3,Matrix{ComplexF64}}}
    blevel::Vector{Int}
    tlevel::Vector{Int}
    lattice::Matrix{ComplexF64}
    modes::Matrix{ComplexF64}
    forward::F
    backward::B
    source::Vector{ComplexF64}
    target::Vector{ComplexF64}
    charge::Vector{ComplexF64}
    field::Vector{ComplexF64}
    vector_second::Vector{ComplexF64}
    output::Vector{ComplexF64}
end
Base.size(A::_PlanarConformalMultiProjection)=(size(A.x,2),size(A.x,2))
Base.size(A::_PlanarConformalMultiProjection,d::Integer)=d<1 ? throw(ArgumentError("positive dimension required")) : d<=2 ? size(A.x,2) : 1
Base.:*(A::_PlanarConformalMultiProjection,x::AbstractVector{ComplexF64})=mul!(similar(x),A,x)

function _planar_multi_projection_component!(out,A,points,input,component,levels)
    n=length(input);source=view(A.source,1:n);target=view(A.target,1:n)
    fill!(out,0)
    for s in axes(A.kernels,2)
        for b in 1:n;source[b]=levels[b]==s ? input[b] : 0.0im;end
        for r in axes(A.kernels,1)
            fill!(target,0)
            state=(;lattice=A.lattice,modes=A.modes,forward=A.forward,backward=A.backward,kernels=A.kernels[r,s])
            _planar_projection_fft_component!(state,points,source,target,component)
            for b in 1:n;levels[b]==r && (out[b]+=target[b]);end
        end
    end
    out
end

function _planar_multi_projection_vector!(out,A,x)
    _planar_multi_projection_component!(out,A,A.x,x,1,A.blevel)
    second=A.vector_second
    _planar_multi_projection_component!(second,A,A.y,x,2,A.blevel)
    out.+=second
end
@inline function _planar_multi_coefficient_alias(out,x,coefficients)
    (Base.mightalias(x,coefficients) || Base.mightalias(out,coefficients)) &&
        throw(ArgumentError("multilevel conformal input/output aliases retained coefficients"))
    nothing
end
@inline function _planar_multi_sparse_alias(out,x,P::SparseMatrixCSC)
    _planar_multi_coefficient_alias(out,x,P.nzval)
    _planar_multi_coefficient_alias(out,x,P.rowval)
    _planar_multi_coefficient_alias(out,x,P.colptr)
end
function LinearAlgebra.mul!(out::AbstractVector{ComplexF64},A::_PlanarConformalMultiProjection,x::AbstractVector{ComplexF64})
    length(out)==length(x)==size(A,1) || throw(DimensionMismatch())
    # Separate typed calls avoid boxing mixed real/complex CSC field tuples.
    _planar_multi_sparse_alias(out,x,A.x);_planar_multi_sparse_alias(out,x,A.y)
    _planar_multi_sparse_alias(out,x,A.pulse);_planar_multi_sparse_alias(out,x,A.incidence)
    _planar_multi_sparse_alias(out,x,A.vector_correction);_planar_multi_sparse_alias(out,x,A.charge_correction)
    _planar_multi_coefficient_alias(out,x,A.blevel);_planar_multi_coefficient_alias(out,x,A.tlevel)
    for pair in A.kernels,coefficients in pair
        _planar_multi_coefficient_alias(out,x,coefficients)
    end
    any(a->Base.mightalias(x,a),(A.source,A.target,A.charge,A.field,A.vector_second,A.output,A.lattice,A.modes)) &&
        throw(ArgumentError("multilevel conformal input aliases workspace"))
    _planar_multi_projection_vector!(A.output,A,x)
    _planar_projection_charge_mul!(A.charge,A.incidence,x,A.field)
    _planar_multi_projection_component!(A.field,A,A.pulse,A.charge,3,A.tlevel)
    mul!(A.field,A.charge_correction,A.charge,1.,1.)
    mul!(A.output,transpose(A.incidence),A.field,1.,1.)
    mul!(A.output,A.vector_correction,x,1.,1.)
    # Output views of scratch are safe: every reaction is finished, and each
    # scratch region is overwritten before its next read. Inputs may not alias
    # scratch; retained coefficient aliases reject in both directions.
    copyto!(out,A.output)
end

function _planar_projection_fft_owned_bytes(A::_PlanarConformalMultiProjection)
    sum(_planar_projection_sparse_bytes,(A.x,A.y,A.pulse,A.incidence,A.vector_correction,A.charge_correction))+
        sum(sizeof(kernel) for pair in A.kernels for kernel in pair)+
        sum(sizeof,(A.blevel,A.tlevel,A.lattice,A.modes,A.source,A.target,A.charge,A.field,A.vector_second,A.output))
end

function _planar_multi_exact_workspace_payload(prob,modes,block)
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces)
    nl=_planar_projection_interface_count(prob);chunk=min(block,BigInt(modes)^2)
    _checked_payload_sum("multilevel original modal workspace",
        _planar_projection_exact_workspace_payload(prob,modes,block),
        _checked_array_payload_bytes(ComplexF64,3(BigInt(nl)^2-1),chunk),
        _checked_array_payload_bytes(Int,BigInt(nb)+nt+nl))
end

function _planar_multi_exact_workspace(prob,frequency,modes,block;max_bytes)
    _planar_projection_sheet_scope(prob)
    payload=_planar_multi_exact_workspace_payload(prob,modes,block)
    _enforce_payload_limit(payload,max_bytes,"multilevel original modal workspace","max_bytes")
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces)
    nmode=_checked_payload_sum("multilevel modal count",BigInt(modes)^2);chunk=min(block,nmode)
    levels=Int[]
    for level in 1:length(prob.stack.layers)-1
        any(==(level),prob.mesh.interfaces) && push!(levels,level)
    end
    blevel=Int[findfirst(==(l),levels) for l in prob.basis.interfaces]
    tlevel=Int[findfirst(==(l),levels) for l in prob.mesh.interfaces]
    triangles=zeros(2,3,nt);frees=zeros(2,2,nb);amplitudes=zeros(2,nb)
    for t in 1:nt,j in 1:3,d in 1:2
        triangles[d,j,t]=prob.mesh.vertices[d,prob.mesh.triangles[j,t]]
    end
    for b in 1:nb,half in 1:2
        t=prob.basis.triangles[half,b];t==0 && continue
        edgea,edgeb=prob.basis.edges[:,b]
        free=only(i for i in prob.mesh.triangles[:,t] if i!=edgea && i!=edgeb)
        for d in 1:2;frees[d,half,b]=prob.mesh.vertices[d,free]-triangles[d,1,t];end
        amplitudes[half,b]=(half==2 || prob.basis.triangles[2,b]==0 ? -1. : 1.)*
            prob.basis.width[b]/(2prob.mesh.areas[t])
    end
    cte,ctm,scratch,_,_=_planar_mode_workspace(length(prob.stack.layers),false,false)
    grid=planar_mode_grid(CellGrid(prob.stack.a,prob.stack.b,1,1),modes,modes)
    (;prob,frequency,modes,nmode,chunk,triangles,frees,amplitudes,cte,ctm,scratch,grid,levels,blevel,tlevel,payload,
        wx=zeros(nb,chunk),wy=zeros(nb,chunk),wc=zeros(nt,chunk),
        kernels=zeros(ComplexF64,3,length(levels),length(levels),chunk),
        plus=zeros(ComplexF64,3,nt),minus=zeros(ComplexF64,3,nt))
end

function _planar_multi_exact_weights_block!(w,start,count)
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
        if iszero(k2);fill!(view(w.kernels,:,:,:,j),0);continue;end
        normte!=0 && planar_mode_cascade!(w.cte,prob.stack,2pi*w.frequency,k2,TE_POL,w.scratch)
        normc!=0 && planar_mode_cascade!(w.ctm,prob.stack,2pi*w.frequency,k2,TM_POL,w.scratch)
        for r in eachindex(w.levels),s in eachindex(w.levels)
            vte=normte==0 ? 0.0im : planar_modal_voltage(w.cte,w.levels[r],w.levels[s])
            potential=normc==0 ? 0.0im : (planar_modal_voltage(w.ctm,w.levels[r],w.levels[s])-vte)/k2
            w.kernels[1,r,s,j]=normx==0 ? 0.0im : -vte/normx
            w.kernels[2,r,s,j]=normy==0 ? 0.0im : -vte/normy
            w.kernels[3,r,s,j]=normc==0 ? 0.0im : -potential/normc
        end
        all(isfinite,view(w.kernels,:,:,:,j)) ||
            throw(ArgumentError("nonfinite original multilevel kernel; check box resonances"))
    end
    w
end

function _planar_multi_folded_kernels(w,nx,ny,sigma)
    nl=length(w.levels);prob=w.prob;mg=w.grid
    kernels=[ntuple(_->zeros(ComplexF64,2nx,2ny),3) for r in 1:nl,s in 1:nl]
    for n in 1:w.modes,m in 1:w.modes
        k2=mg.kx[m]^2+mg.ky[n]^2;iszero(k2) && continue
        normx,normy=mg.ic[m]*mg.js[n],mg.is[m]*mg.jc[n];normc=mg.is[m]*mg.js[n]
        normte=mg.ky[n]^2*normx+mg.kx[m]^2*normy
        normte!=0 && planar_mode_cascade!(w.cte,prob.stack,2pi*w.frequency,k2,TE_POL,w.scratch)
        normc!=0 && planar_mode_cascade!(w.ctm,prob.stack,2pi*w.frequency,k2,TM_POL,w.scratch)
        for r in 1:nl,s in 1:nl
            vte=normte==0 ? 0.0im : planar_modal_voltage(w.cte,w.levels[r],w.levels[s])
            scalar=normc==0 ? 0.0im : (planar_modal_voltage(w.ctm,w.levels[r],w.levels[s])-vte)/k2
            if r==s
                lower,upper=prob.stack.layers[w.levels[r]],prob.stack.layers[w.levels[r]+1]
                magnetic=im*2pi*w.frequency*_MU0/(inv(lower.mur)+inv(upper.mur))
                electric=inv(im*2pi*w.frequency*_EPS0*(lower.epsr+upper.epsr))
                window=erfc(sqrt(k2)*sigma)
                vte+=(window-1)*magnetic/sqrt(k2);scalar+=(window-1)*electric/sqrt(k2)
            end
            isfinite(vte) && isfinite(scalar) || throw(ArgumentError("nonfinite multilevel projection kernel"))
            i,j=mod(m-1,2nx)+1,mod(n-1,2ny)+1
            normx==0 || (kernels[r,s][1][i,j]-=vte/normx)
            normy==0 || (kernels[r,s][2][i,j]-=vte/normy)
            normc==0 || (kernels[r,s][3][i,j]-=scalar/normc)
        end
    end
    kernels
end

function _planar_multi_exact_local_entries!(vector,scalar,w,vpairs,cpairs)
    fill!(vector,0);fill!(scalar,0)
    ev=zeros(ComplexF64,length(vector));ec=zeros(ComplexF64,length(scalar))
    for start in 1:w.chunk:w.nmode
        count=min(w.chunk,w.nmode-start+1);_planar_multi_exact_weights_block!(w,start,count)
        for p in eachindex(vector)
            a,b=vpairs.row[p],vpairs.column[p];r,s=w.blevel[a],w.blevel[b]
            sum,error=vector[p],ev[p]
            for j in 1:count
                value=w.wx[a,j]*w.kernels[1,r,s,j]*w.wx[b,j]+w.wy[a,j]*w.kernels[2,r,s,j]*w.wy[b,j]
                sum,error=_planar_projection_kahan_add(sum,error,value)
            end
            vector[p],ev[p]=sum,error
        end
        for p in eachindex(scalar)
            a,b=cpairs.row[p],cpairs.column[p];r,s=w.tlevel[a],w.tlevel[b]
            sum,error=scalar[p],ec[p]
            for j in 1:count
                sum,error=_planar_projection_kahan_add(sum,error,w.wc[a,j]*w.kernels[3,r,s,j]*w.wc[b,j])
            end
            scalar[p],ec[p]=sum,error
        end
    end
end

function _planar_multi_original_mul!(output,w,D,X,work)
    nb,np=size(X);nt=size(D,1);nl=length(w.levels)
    size(output)==size(X) && size(D,2)==nb && size(w.wx,1)==nb &&
        size(work.charge)==(nt,np) && size(work.ev)==(nb,np) &&
        size(work.contraction)==(3,nl,np) || throw(DimensionMismatch("multilevel original action dimensions"))
    for a in (output,work.charge,work.field,work.ev,work.ec,work.contraction,work.contraction_error)
        Base.mightalias(a,X) && throw(ArgumentError("original multilevel input aliases output/workspace"))
    end
    for a in (work.charge,work.field,work.ev,work.ec,work.contraction,work.contraction_error)
        Base.mightalias(a,output) && throw(ArgumentError("original multilevel output aliases workspace"))
    end
    for a in (w.wx,w.wy,w.wc,w.kernels,w.plus,w.minus,w.triangles,w.frees,w.amplitudes,
            w.blevel,w.tlevel,w.levels,D.nzval,D.rowval,D.colptr)
        (Base.mightalias(a,X) || Base.mightalias(a,output)) &&
            throw(ArgumentError("original multilevel input/output aliases retained coefficients or modal workspace"))
    end
    charge,field,ev,ec=work.charge,work.field,work.ev,work.ec
    contraction,errors=work.contraction,work.contraction_error
    _planar_projection_charge_mul!(charge,D,X,ec)
    fill!(field,0);fill!(ev,0);fill!(ec,0);fill!(output,0)
    for start in 1:w.chunk:w.nmode
        count=min(w.chunk,w.nmode-start+1);_planar_multi_exact_weights_block!(w,start,count)
        for j in 1:count
            fill!(contraction,0);fill!(errors,0)
            for p in 1:np,b in 1:nb
                l=w.blevel[b]
                contraction[1,l,p],errors[1,l,p]=_planar_projection_kahan_add(contraction[1,l,p],errors[1,l,p],w.wx[b,j]*X[b,p])
                contraction[2,l,p],errors[2,l,p]=_planar_projection_kahan_add(contraction[2,l,p],errors[2,l,p],w.wy[b,j]*X[b,p])
            end
            for p in 1:np,t in 1:nt
                l=w.tlevel[t]
                contraction[3,l,p],errors[3,l,p]=_planar_projection_kahan_add(contraction[3,l,p],errors[3,l,p],w.wc[t,j]*charge[t,p])
            end
            for p in 1:np,b in 1:nb,s in 1:nl
                r=w.blevel[b]
                value=w.wx[b,j]*w.kernels[1,r,s,j]*contraction[1,s,p]+w.wy[b,j]*w.kernels[2,r,s,j]*contraction[2,s,p]
                output[b,p],ev[b,p]=_planar_projection_kahan_add(output[b,p],ev[b,p],value)
            end
            for p in 1:np,t in 1:nt,s in 1:nl
                r=w.tlevel[t]
                value=w.wc[t,j]*w.kernels[3,r,s,j]*contraction[3,s,p]
                field[t,p],ec[t,p]=_planar_projection_kahan_add(field[t,p],ec[t,p],value)
            end
        end
    end
    mul!(output,transpose(D),field,1.,1.)
    output
end

function _planar_multi_charge_projection_fft(prob,projection,pulses,frequency,modes,
        vector_correction,charge_correction;workspace,sigma,max_bytes)
    nx,ny=projection.nx,projection.ny;nt=length(prob.mesh.interfaces);nb=length(prob.basis.width)
    nl=length(workspace.levels)
    sparsebytes=sum(_planar_projection_sparse_bytes,
        (projection.x,projection.y,pulses.pulse,vector_correction,charge_correction))
    reserve=_checked_payload_sum("multilevel FFT retained arrays",sparsebytes,
        _checked_array_payload_bytes(ComplexF64,3BigInt(nl)^2+2,2BigInt(nx),2BigInt(ny)),
        _checked_array_payload_bytes(ComplexF64,2BigInt(max(nb,nt))+2BigInt(nt)+2BigInt(nb)),
        _checked_array_payload_bytes(UInt8,512,BigInt(nb)+1))
    _enforce_payload_limit(reserve,max_bytes,"multilevel FFT retained arrays","max_bytes")
    workspace.prob===prob && workspace.frequency==frequency && workspace.modes==modes ||
        throw(ArgumentError("multilevel projection requires matching original modal workspace"))
    D=_planar_physical_charge_incidence(prob)
    kernels=_planar_multi_folded_kernels(workspace,nx,ny,sigma)
    F=zeros(ComplexF64,2nx,2ny);Q=similar(F)
    _PlanarConformalMultiProjection(projection.x,projection.y,pulses.pulse,D,
        vector_correction,charge_correction,kernels,workspace.blevel,workspace.tlevel,F,Q,
        FFTW.plan_fft!(F),FFTW.plan_bfft!(F),zeros(ComplexF64,max(nb,nt)),zeros(ComplexF64,max(nb,nt)),
        zeros(ComplexF64,nt),zeros(ComplexF64,nt),zeros(ComplexF64,nb),zeros(ComplexF64,nb))
end

function _planar_multi_subtract_fft_local!(values,A,pairs;pulses=false)
    n=pulses ? length(A.tlevel) : length(A.blevel)
    input=zeros(ComplexF64,n);output=similar(input)
    for b in 1:n
        fill!(input,0);input[b]=1
        if pulses
            _planar_multi_projection_component!(output,A,A.pulse,input,3,A.tlevel)
        else
            _planar_multi_projection_vector!(output,A,input)
        end
        for p in pairs.offsets[b]:pairs.offsets[b+1]-1
            values[p]-=output[pairs.row[p]]
        end
    end
    values
end


function _planar_bounded_multi_charge_projection_fft(prob,frequency;
        modes=128,nx=64,ny=nx,order=7,sigma=.5prob.stack.a/nx,radius=6sigma,block=512,
        max_pairs=2_000_000,max_references=8_000_000,max_visits=80_000_000,
        max_patch_visits=8_000_000,max_stencil_entries=5_000_000,
        max_modal_terms=4_000_000,max_pair_mode_products=5_000_000_000,max_bytes=512_000_000)
    frequency=_planar_projection_frequency(frequency)
    _planar_projection_sheet_scope(prob)
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
    modal=_planar_multi_exact_workspace_payload(prob,modes,block)
    initial=_checked_payload_sum("initial local projection reservation",modal,
        _checked_array_payload_bytes(ComplexF64,3BigInt(_planar_projection_interface_count(prob))^2+2,2BigInt(nx),2BigInt(ny)))
    _enforce_payload_limit(initial,max_bytes,"initial local projection reservation","max_bytes")
    w=_planar_multi_exact_workspace(prob,frequency,modes,block;max_bytes)
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
    base=_planar_multi_charge_projection_fft(prob,projection,pulses,frequency,modes,SparseArrays.spzeros(ComplexF64,nb,nb),SparseArrays.spzeros(ComplexF64,nt,nt);
        workspace=w,sigma,max_bytes=max_bytes-localpayload)
    vector=zeros(ComplexF64,length(vpairs.row));scalar=zeros(ComplexF64,length(cpairs.row))
    _planar_multi_exact_local_entries!(vector,scalar,w,vpairs,cpairs)
    diagonal=_planar_projection_exact_local_diagonal(prob,vector,scalar,vpairs,cpairs)
    # The public solve validates the combined electromagnetic + sheet Gram
    # diagonal; a sheet impedance may remove an otherwise zero entry.
    all(isfinite,diagonal) ||
        throw(ArgumentError("conformal defect preconditioner has a nonfinite original diagonal"))
    _planar_multi_subtract_fft_local!(vector,base,vpairs)
    _planar_multi_subtract_fft_local!(scalar,base,cpairs;pulses=true)
    vc=_planar_projection_symmetric_correction(vector,vpairs,nb);cc=_planar_projection_symmetric_correction(scalar,cpairs,nt)
    A=_PlanarConformalMultiProjection(base.x,base.y,base.pulse,base.incidence,vc,cc,base.kernels,
        base.blevel,base.tlevel,base.lattice,base.modes,base.forward,base.backward,
        base.source,base.target,base.charge,base.field,base.vector_second,base.output)
    owned=_planar_projection_fft_owned_bytes(A)
    constructionbound=owned+localpayload
    _enforce_payload_limit(constructionbound,max_bytes,"local correction aggregate construction","max_bytes")
    (;operator=A,workspace=w,vpairs,cpairs,owned,constructionbound,projection,pulses,radius,sigma,diagonal)
end
