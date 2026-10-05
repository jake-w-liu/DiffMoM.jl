export PlanarConformalDefectResult, solve_planar_conformal_defect

"""A nonuniform conformal solution accepted against the original analytic
modal equation. `preconditioner` is an approximate projected operator;
`relative_residuals` measure the original voltage equation, while
`approximate_relative_residuals` measure the final projected equation.
Physical RWG coefficients, Y/S and evaluated references use the same
conventions as `PlanarConformalResult`. `diagnostics` records bounded
correction work and numeric array payload reservations."""
struct PlanarConformalDefectResult{P,D}
    problem::PlanarConformalProblem
    freq::ComplexF64
    omega::ComplexF64
    preconditioner::P
    currents::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    z0::Vector{ComplexF64}
    iterations::Vector{Int}
    relative_residuals::Vector{Float64}
    approximate_relative_residuals::Vector{Float64}
    diagnostics::D
end

# The sheet impedance is local to the physical triangles, independent of
# projection points and filters. Keep its exact sparse Gram in both actions.
struct _PlanarConformalMaterialProjection{P} <: AbstractMatrix{ComplexF64}
    projection::P
    loss::SparseMatrixCSC{ComplexF64,Int}
end
Base.size(A::_PlanarConformalMaterialProjection)=size(A.projection)
Base.size(A::_PlanarConformalMaterialProjection,d::Integer)=size(A.projection,d)
function LinearAlgebra.mul!(out::AbstractVector{ComplexF64},A::_PlanarConformalMaterialProjection,
        x::AbstractVector{ComplexF64})
    length(out)==length(x)==size(A,1) || throw(DimensionMismatch())
    (Base.mightalias(x,A.loss.nzval) || Base.mightalias(out,A.loss.nzval)) &&
        throw(ArgumentError("conformal material input/output must not alias its retained Gram coefficients"))
    # Complete both reactions before writing the caller's output: out may be x.
    # The projection rejects inputs aliasing any of its owned workspaces.
    mul!(A.projection.output,A.projection,x)
    mul!(A.projection.output,A.loss,x,1.,1.)
    copyto!(out,A.projection.output)
end
Base.:*(A::_PlanarConformalMaterialProjection,x::AbstractVector{ComplexF64})=mul!(similar(x),A,x)
_planar_projection_fft_owned_bytes(A::_PlanarConformalMaterialProjection)=
    _planar_projection_fft_owned_bytes(A.projection)+_planar_projection_sparse_bytes(A.loss)

function _planar_projection_impedance_number(z)
    z isa Number || throw(ArgumentError("conformal surface impedances must be numbers"))
    value=try ComplexF64(z) catch
        throw(ArgumentError("conformal surface impedances must fit finite ComplexF64"))
    end
    isfinite(value) || throw(ArgumentError("conformal surface impedances must fit finite ComplexF64"))
    return value
end

# Static validation allocates no triangle-sized arrays; an unknown provider
# reserves the full local construction bound before it is called.
function _planar_projection_material_preflight(surface_zs,nt,f)
    surface_zs isa Number && return !iszero(_planar_projection_impedance_number(surface_zs))
    if surface_zs isa AbstractVector
        length(surface_zs)==nt || throw(ArgumentError("conformal surface_zs requires one value per triangle"))
        nonzero=false
        for z in surface_zs
            nonzero|=!iszero(_planar_projection_impedance_number(z))
        end
        return nonzero
    end
    applicable(surface_zs,f) || throw(ArgumentError("conformal surface_zs must be scalar, per-triangle vector or frequency provider"))
    return true
end

function _planar_projection_material_values(surface_zs,nt,f)
    values=surface_zs isa Union{Number,AbstractVector} ? surface_zs : surface_zs(f)
    values isa Number && return fill(_planar_projection_impedance_number(values),nt)
    values isa AbstractVector && length(values)==nt ||
        throw(ArgumentError("conformal surface_zs provider must return scalar or one value per triangle"))
    return ComplexF64[_planar_projection_impedance_number(z) for z in values]
end

function _planar_projection_material_payload(nb,nt)
    # At most three RWG restrictions occur on a physical triangle, hence
    # at most nine ordered Gram emissions. Include triplets, sparse conversion
    # work/CSC, triangle restrictions and column pointers simultaneously.
    _checked_payload_sum("conformal exact sheet Gram construction",
        _checked_array_payload_bytes(UInt8,128,9BigInt(nt)),
        _checked_array_payload_bytes(UInt8,512,nt),
        _checked_array_payload_bytes(Int,6,BigInt(nb)+1))
end

function _planar_projection_material_gram(prob,zs;max_bytes)
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces)
    bound=_planar_projection_material_payload(nb,nt)
    _enforce_payload_limit(bound,max_bytes,"conformal exact sheet Gram construction","max_bytes")
    maximum_entries=_checked_payload_sum("conformal sheet Gram entry count",9BigInt(nt))
    rows=Int[];columns=Int[];values=ComplexF64[]
    _planar_conformal_loss_entries(prob,zs) do p,q,value
        length(values)<maximum_entries || throw(ArgumentError("more than three RWG restrictions on a physical triangle"))
        isfinite(value) || throw(ArgumentError("conformal sheet Gram reaction is nonfinite"))
        push!(rows,p);push!(columns,q);push!(values,value)
    end
    loss=sparse(rows,columns,values,nb,nb)
    all(isfinite,loss.nzval) || throw(ArgumentError("conformal sheet Gram sum is nonfinite"))
    return loss
end

function _planar_projection_original_work(nb,nt,np;max_bytes,nlevels=1)
    nlevels isa Integer && nlevels>=1 || throw(ArgumentError("positive original interface count required"))
    payload=_checked_payload_sum("original conformal action workspace",
        _checked_array_payload_bytes(ComplexF64,BigInt(nb)+3BigInt(nt),np),
        nlevels==1 ? 0 : _checked_array_payload_bytes(ComplexF64,6,nlevels,np))
    _enforce_payload_limit(payload,max_bytes,"original conformal action workspace","max_bytes")
    basic=(;charge=zeros(ComplexF64,nt,np),field=zeros(ComplexF64,nt,np),
        ev=zeros(ComplexF64,nb,np),ec=zeros(ComplexF64,nt,np),payload)
    nlevels==1 && return basic
    (;basic...,contraction=zeros(ComplexF64,3,nlevels,np),contraction_error=zeros(ComplexF64,3,nlevels,np))
end

# Batched ORIGINAL action, with no projection/filter/local-pair decisions.
# Triangle constant/first moments are reused across physical source columns.
function _planar_projection_original_mul!(output,w,D,X,work)
    hasproperty(w,:levels) && return _planar_multi_original_mul!(output,w,D,X,work)
    nb,np=size(X);nt=size(D,1)
    size(output)==size(X) && size(D,2)==nb && size(w.wx,1)==nb &&
        size(work.charge)==(nt,np) && size(work.ev)==(nb,np) ||
        throw(DimensionMismatch("original conformal action dimensions"))
    any(a->Base.mightalias(a,X),(output,work.charge,work.field,work.ev,work.ec)) &&
        throw(ArgumentError("original conformal action input must not alias output or workspace"))
    any(a->Base.mightalias(a,output),(work.charge,work.field,work.ev,work.ec)) &&
        throw(ArgumentError("original conformal output must not alias its workspace"))
    charge,field,ev,ec=work.charge,work.field,work.ev,work.ec
    _planar_projection_charge_mul!(charge,D,X,ec)
    fill!(field,0);fill!(ev,0);fill!(ec,0);fill!(output,0)
    for start in 1:w.chunk:w.nmode
        count=min(w.chunk,w.nmode-start+1);_planar_projection_exact_weights_block!(w,start,count)
        for j in 1:count,p in 1:np
            vx,vy,vc=0.0im,0.0im,0.0im;ex,ey,eq=0.0im,0.0im,0.0im
            for b in 1:nb
                vx,ex=_planar_projection_kahan_add(vx,ex,w.wx[b,j]*X[b,p])
                vy,ey=_planar_projection_kahan_add(vy,ey,w.wy[b,j]*X[b,p])
            end
            for t in 1:nt
                vc,eq=_planar_projection_kahan_add(vc,eq,w.wc[t,j]*charge[t,p])
            end
            vx*=w.kernels[1,j];vy*=w.kernels[2,j];vc*=w.kernels[3,j]
            for b in 1:nb
                output[b,p],ev[b,p]=_planar_projection_kahan_add(output[b,p],ev[b,p],w.wx[b,j]*vx+w.wy[b,j]*vy)
            end
            for t in 1:nt
                field[t,p],ec[t,p]=_planar_projection_kahan_add(field[t,p],ec[t,p],w.wc[t,j]*vc)
            end
        end
    end
    mul!(output,transpose(D),field,1.,1.)
    return output
end

function _planar_projection_voltage_residual(x,weights,source_norm)
    squared=0.0
    for b in eachindex(weights)
        squared+=abs2(x[b]/weights[b])
    end
    return sqrt(squared)/source_norm
end

"""
    solve_planar_conformal_defect(prob, freq; modes=128, nx=64, ny=nx, ...)

Solve a genuine nonuniform triangular sheet using a bounded projected
FFT preconditioner and outer defect corrections. Every returned column must
satisfy the independently streamed **original analytic modal** voltage
equation at `rtol`; a small projected residual alone never accepts a result.
`modes` sets the original square modal truncation. `nx,ny` set the projection
lattice and do not rasterize or change the triangles or RWG unknowns.

The current scope is interior sheets, isotropic layers, PEC box
covers/sidewalls, real positive frequency and box-wall ports. Projection
`order` is 3:9, `sigma` defaults to half an x cell, and `radius=6sigma`.
Construction bounds pairs, bin references/visits, projection patches/stencils
and pair×mode work using their `max_*` keywords. `max_bytes` bounds the
aggregate numeric array payload of construction, Krylov work, physical
sources/results and original action; opaque FFT plans are outside that
array-payload convention. No dense Galerkin matrix or complete modal
coefficient planes are retained. Original residual actions still cost
O((triangles + unknowns) × modal terms × source columns × sheet interfaces); this is a
bounded storage solver, without a general fast-exact-PFFT claim.

`memory` bounds restarted GMRES, `maxiter` bounds each inner solve, and
`max_outer` (0:8) bounds corrections. Failure of the original equation
throws. `diagnostics` keeps initial original/projected residuals, correction
iterations, action times and array payload reservations separately.

`surface_zs` is a finite complex sheet impedance (ohms/square), one value per
physical triangle, or a frequency provider returning either. Its exact
weighted RWG Gram is included in both the projected and original equations;
no material quadrature or material projection is used. Providers are called
once, after constructor/resource rejection, at the real physical frequency.
Evaluated triangle impedances are retained in `diagnostics.surface_impedances`.
Nonnegative real parts describe passive sheets; purely reactive impedances
are supported. Multiple sheet interfaces retain their complete analytic
interlevel coupling. Via/volume/contact geometry remains outside this
solver's current scope. The optimized single-interface path is preserved;
the multilevel preconditioner retains three FFT kernels per ordered interface
pair and streams all original interface-pair modal kernels in bounded blocks.
"""
function solve_planar_conformal_defect(prob::PlanarConformalProblem,freq::Number;
        modes::Integer=128,nx::Integer=64,ny::Integer=nx,order::Integer=7,
        sigma=.5prob.stack.a/nx,radius=6sigma,block::Integer=512,
        memory::Integer=200,maxiter::Integer=10000,max_outer::Integer=4,
        rtol::Real=1e-10,surface_zs=0.,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        max_pairs::Integer=2_000_000,max_references::Integer=8_000_000,
        max_visits::Integer=80_000_000,max_patch_visits::Integer=8_000_000,
        max_stencil_entries::Integer=5_000_000,max_modal_terms::Integer=4_000_000,
        max_pair_mode_products::Integer=5_000_000_000)
    f=_planar_projection_frequency(freq);_planar_projection_sheet_scope(prob)
    1<=memory<=typemax(Int) && 1<=maxiter<=typemax(Int) && 0<=max_outer<=8 ||
        throw(ArgumentError("conformal defect solve requires positive representable Krylov limits and max_outer in 0:8"))
    tol=try Float64(rtol) catch; throw(ArgumentError("conformal defect tolerance must fit Float64")); end
    isfinite(tol) && 0<tol<1 && tol/10>0 || throw(ArgumentError("conformal defect tolerance must be finite, positive and below one"))
    budget=_validated_resource_limit("max_bytes",max_bytes)
    nb=length(prob.basis.width);nt=length(prob.mesh.interfaces);np=length(prob.ports)
    single=all(==(first(prob.mesh.interfaces)),prob.mesh.interfaces)
    nlevels=single ? 1 : _planar_projection_interface_count(prob)
    has_material=_planar_projection_material_preflight(surface_zs,nt,f)
    material_reserve=has_material ? _planar_projection_material_payload(nb,nt) : 0
    mem=min(Int(memory),nb)
    reserve=_checked_payload_sum("conformal defect aggregate solve reservation",
        _checked_array_payload_bytes(ComplexF64,nb,BigInt(mem)+40+4BigInt(np)),
        _checked_array_payload_bytes(ComplexF64,BigInt(mem)+1,mem),
        _checked_array_payload_bytes(ComplexF64,3BigInt(nt),np),
        _checked_array_payload_bytes(ComplexF64,4,np,np),
        _checked_array_payload_bytes(Float64,8,nb),
        _checked_array_payload_bytes(Float64,16,np),
        _checked_array_payload_bytes(Int,16,np),
        _checked_array_payload_bytes(ComplexF64,nt),
        single ? 0 : _checked_array_payload_bytes(ComplexF64,6,nlevels,np))
    _enforce_payload_limit(_checked_payload_sum("conformal defect solve and sheet reservation",reserve,material_reserve),
        budget,"conformal defect solve and sheet reservation","max_bytes")
    constructor=single ? _planar_bounded_charge_projection_fft : _planar_bounded_multi_charge_projection_fft
    constructor_seconds=@elapsed built=constructor(prob,f;
        modes,nx,ny,order,sigma,radius,block,max_pairs,max_references,max_visits,
        max_patch_visits,max_stencil_entries,max_modal_terms,max_pair_mode_products,
        max_bytes=budget-reserve-material_reserve)
    # All geometric constructor and aggregate rejection precedes providers.
    impedances=_planar_projection_material_values(surface_zs,nt,f)
    loss=all(iszero,impedances) ? nothing :
        _planar_projection_material_gram(prob,impedances;max_bytes=material_reserve)
    A=loss===nothing ? built.operator : _PlanarConformalMaterialProjection(built.operator,loss)
    diagonal=loss===nothing ? built.diagonal : copy(built.diagonal)
    if loss!==nothing
        for b in eachindex(diagonal)
            diagonal[b]+=loss[b,b]
        end
    end
    all(x->isfinite(x) && !iszero(x),diagonal) ||
        throw(ArgumentError("conformal defect preconditioner has a zero or nonfinite combined diagonal"))
    weights=prob.basis.width
    work=_planar_projection_original_work(nb,nt,np;max_bytes=reserve,nlevels)
    M=Diagonal(inv.(weights));N=Diagonal(weights./diagonal)
    all(isfinite,M.diag) && all(isfinite,N.diag) || throw(ArgumentError("nonfinite conformal defect trace normalization"))
    rhs=zeros(ComplexF64,nb,np);R=similar(rhs);X=similar(rhs)
    defect_rhs=zeros(ComplexF64,nb);inner=similar(defect_rhs)
    source_norm=zeros(np);residuals=zeros(np);approximate=zeros(np)
    initial_original=zeros(np);initial_projected=zeros(np)
    iterations=zeros(Int,np);corrections=[Int[] for _ in 1:np];action_seconds=Float64[]
    # All constructor/aggregate resource rejection precedes user providers.
    refs=_planar_reference_values([p.z0 for p in prob.ports],np;freq=f)
    started=time()
    for p in 1:np
        for b in 1:nb
            rhs[b,p]=prob.basis.port[b]==p ? -prob.basis.port_sign[b]*weights[b] : 0.0im
        end
        source_norm[p]=sqrt(count(==(p),prob.basis.port))
        source_norm[p]>0 || throw(ArgumentError("conformal source has no claimed boundary edges"))
        x,stats=Krylov.gmres(A,view(rhs,:,p);M,N,rtol=tol/10,atol=0.,itmax=Int(maxiter),memory=mem,
            restart=true,reorthogonalization=true)
        iterations[p]=stats.niter;X[:,p].=x
        mul!(inner,A,x);inner.-=view(rhs,:,p)
        initial_projected[p]=_planar_projection_voltage_residual(inner,weights,source_norm[p])
        # Arnoldi's recurrence can underestimate the recomputed voltage error
        # after low-frequency cancellation. Normalize the correction source
        # so Krylov's absolute machine-precision stop does not truncate a tiny
        # residual solve. Rounding can need several corrections; bound retries
        # by eight and retain the original total iteration limit.
        for retry in 1:8
            isfinite(initial_projected[p]) && initial_projected[p]>tol || break
            remaining=Int(maxiter)-iterations[p];remaining>0 || break
            scale=initial_projected[p];defect_rhs.=-inner./scale
            delta,stats=Krylov.gmres(A,defect_rhs;M,N,rtol=tol/10,atol=0.,
                itmax=remaining,memory=mem,restart=true,reorthogonalization=true)
            iterations[p]+=stats.niter;X[:,p].+=scale.*delta
            mul!(inner,A,view(X,:,p));inner.-=view(rhs,:,p)
            initial_projected[p]=_planar_projection_voltage_residual(inner,weights,source_norm[p])
        end
        isfinite(initial_projected[p]) && initial_projected[p]<=tol ||
            error("conformal defect port $p failed initial projected voltage residual $(initial_projected[p]) > $tol after $(iterations[p]) iterations")
    end
    for outer in 0:Int(max_outer)
        push!(action_seconds,@elapsed begin
            _planar_projection_original_mul!(R,built.workspace,built.operator.incidence,X,work)
            loss===nothing || mul!(R,loss,X,1.,1.)
        end)
        R.-=rhs
        for p in 1:np
            residuals[p]=_planar_projection_voltage_residual(view(R,:,p),weights,source_norm[p])
            outer==0 && (initial_original[p]=residuals[p])
        end
        all(isfinite,residuals) && maximum(residuals)<=tol && break
        outer==max_outer && error("conformal defect original modal voltage residual $(maximum(residuals)) exceeds $tol after $max_outer bounded corrections")
        for p in 1:np
            residuals[p]<=tol && continue
            defect_rhs.=-view(R,:,p)
            delta,stats=Krylov.gmres(A,defect_rhs;M,N,rtol=tol/10,atol=0.,itmax=Int(maxiter),memory=mem,
                restart=true,reorthogonalization=true)
            push!(corrections[p],stats.niter)
            mul!(inner,A,delta);inner.-=defect_rhs
            relative=_planar_projection_voltage_residual(inner,weights,source_norm[p])
            isfinite(relative) && relative<=tol ||
                error("conformal defect port $p correction voltage error $relative exceeds original source tolerance $tol")
            X[:,p].+=delta
        end
    end
    Y=zeros(ComplexF64,np,np)
    for b in 1:nb
        q=prob.basis.port[b];q==0 && continue
        for p in 1:np
            Y[q,p]+=prob.basis.port_sign[b]*weights[b]*X[b,p]
        end
    end
    for p in 1:np
        mul!(inner,A,view(X,:,p));inner.-=view(rhs,:,p)
        approximate[p]=_planar_projection_voltage_residual(inner,weights,source_norm[p])
    end
    all(isfinite,X) && all(isfinite,Y) && all(isfinite,approximate) || error("nonfinite conformal defect solution")
    diagnostics=(interface_count=nlevels,initial_original_relative_residuals=initial_original,
        initial_projected_relative_residuals=initial_projected,correction_iterations=corrections,
        original_action_seconds=action_seconds,constructor_seconds,solve_seconds=time()-started,
        surface_impedances=impedances,material_constructor_payload_bound=material_reserve,
        owned_preconditioner_payload=_planar_projection_fft_owned_bytes(A),
        constructor_payload_bound=_checked_payload_sum("conformal defect constructor",built.constructionbound,material_reserve),
        solve_payload_bound=reserve,total_payload_bound=_checked_payload_sum("conformal defect total",built.constructionbound,material_reserve,reserve),
        vector_pairs=length(built.vpairs.row),pulse_pairs=length(built.cpairs.row),
        bin_references=built.vpairs.references+built.cpairs.references,neighbor_visits=built.vpairs.visits+built.cpairs.visits)
    return PlanarConformalDefectResult(prob,ComplexF64(f),ComplexF64(2pi*f),A,X,Y,
        planar_y_to_s(Y,refs),refs,iterations,residuals,approximate,diagnostics)
end

planar_conformal_current_maps(result::PlanarConformalDefectResult;kw...)=_planar_conformal_current_maps(result;kw...)
planar_current_maps(result::PlanarConformalDefectResult;kw...)=_planar_conformal_current_maps(result;kw...)
