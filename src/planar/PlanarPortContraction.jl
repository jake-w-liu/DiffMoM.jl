export PlanarContractedResult, PlanarSourceResult, planar_contract_ports, solve_planar_contracted

"""A raw planar result with real voltage-contract matrix `contraction`.
Raw port voltages equal `contraction*voltages`; conjugate port currents
are `transpose(contraction)*raw_currents`. `currents,y,s` use the contracted
ports, and `raw` retains the complete source geometry and factor/operator.
`problem` may preserve an original problem's port metadata after refinement."""
struct PlanarContractedResult{R,T<:Complex}
    raw::R
    problem::PlanarProblem
    freq::ComplexF64
    omega::ComplexF64
    contraction::Matrix{Float64}
    z0::Vector{ComplexF64}
    currents::Matrix{T}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
end

"""Apply original physical ports' per-frequency reference-plane
contracts to a contracted solution, preserving raw geometry and currents."""
function planar_reference_planes(raw::PlanarContractedResult;kw...)
    length(raw.problem.ports)==size(raw.y,1) || throw(ArgumentError(
        "contracted reference planes require original physical port metadata"))
    return _planar_reference_planes(raw;kw...)
end

"""Physical EM source solution driven by voltage contracts directly.
`currents,y,s` have one column/port per physical contract, while `problem`
retains the complete raw source geometry. A dense solve retains the
two-sided equilibrated factorization and `basis_scale`; an FFT solve
retains `operator`. `relative_residuals` are independently recomputed
voltage residuals after dividing Galerkin rows by their trace measure;
`galerkin_relative_residuals` also reports the unscaled residual. Dense
residuals are available for retained and compact dense solutions. Rare
low-frequency recovery retains owned wider coefficients and may retain a
factorization of the original matrix with unit `basis_scale`."""
struct PlanarSourceResult{F,O,T<:Complex}
    problem::PlanarProblem
    freq::ComplexF64
    omega::ComplexF64
    contraction::Matrix{Float64}
    z0::Vector{ComplexF64}
    z_mom::Union{Nothing,Matrix{ComplexF64}}
    lu_fact::F
    operator::O
    basis_scale::Vector{Float64}
    currents::Matrix{T}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    iterations::Vector{Int}
    relative_residuals::Vector{Float64}
    galerkin_relative_residuals::Vector{Float64}
end

# Preserve the original explicit two-parameter constructor's ComplexF64
# current storage. Solvers infer their owned current type through the outer
# constructor; callers opting into a wider explicit type use all parameters.
PlanarSourceResult{F,O}(args...) where {F,O} = PlanarSourceResult{F,O,ComplexF64}(args...)

_planar_coefficient_columns(raw::Union{PlanarResult,PlanarUFFTResult,PlanarContractedResult,PlanarSourceResult})=raw.currents
_planar_coefficient_columns(raw::PlanarCalibratedResult)=_planar_current_product(raw.raw.currents,raw.gap_voltage_transfer)
_planar_raw_problem(raw::PlanarCalibratedResult)=raw.raw.problem
_planar_raw_problem(raw)=raw.problem
_planar_raw_frequency(raw::PlanarCalibratedResult)=raw.raw.freq
_planar_raw_frequency(raw)=raw.freq
_planar_raw_omega(raw::PlanarCalibratedResult)=raw.raw.omega
_planar_raw_omega(raw)=raw.omega

function _planar_contract_payload(nb,nraw,n)
    return _checked_payload_sum("planar port contraction",
        _checked_array_payload_bytes(Float64,nraw,n),
        _checked_array_payload_bytes(ComplexF64,nb,n),
        _checked_array_payload_bytes(ComplexF64,nraw,n),
        _checked_array_payload_bytes(ComplexF64,5,n,n),
        _checked_array_payload_bytes(Float64,2,n),
        _checked_array_payload_bytes(ComplexF64,n),
        _checked_array_payload_bytes(Int,n))
end

"""Contract physical raw port sources into original terminal contracts.
`C` must be finite, real, and have independent columns. `z0` supplies the
positive reference impedances of the contracted ports. Optional `problem`
retains original port geometry/metadata, as for axial refinement. The
returned coefficient columns are `raw.currents*C`, admittance is
`transpose(C)*raw.y*C`, and raw data are unchanged. `max_bytes` bounds the
new wrapper's array workspace; callers retaining raw solves should reserve
their payload separately."""
function planar_contract_ports(raw::Union{PlanarResult,PlanarUFFTResult,
        PlanarCalibratedResult,PlanarContractedResult,PlanarSourceResult},C::AbstractMatrix{<:Real};
        z0=50.0,problem::Union{Nothing,PlanarProblem}=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    nraw,n=size(C)
    nraw==size(raw.y,1) && 1<=n<=nraw && all(isfinite,C) ||
        throw(ArgumentError("port contraction must be finite and match raw ports"))
    nb=size(raw isa PlanarCalibratedResult ? raw.raw.currents : raw.currents,1)
    source=raw isa PlanarCalibratedResult ? raw.raw.currents : raw.currents
    payload=_planar_contract_payload(nb,nraw,n)
    if eltype(source)!==ComplexF64
        bits=_planar_current_precision(source)
        payload=_checked_payload_sum("wide planar contraction",payload,
            _checked_array_payload_bytes(UInt8,_planar_wide_complex_payload(bits),nb,n),
            _checked_array_payload_bytes(UInt8,_planar_wide_scalar_payload(bits),4))
    end
    _enforce_payload_limit(payload,max_bytes,"planar port contraction","max_bytes")
    stored=Matrix{Float64}(C)
    all(isfinite,stored) || throw(ArgumentError("port contraction must fit finite Float64 values"))
    LinearAlgebra.rank(stored)==n || throw(ArgumentError("port voltage contracts must have independent columns"))
    refs=_circuit_z0(z0,n;freq=real(_planar_raw_frequency(raw)))
    problem===nothing || length(problem.ports)==n || throw(ArgumentError(
        "original problem port metadata must match contraction columns"))
    coefficients=if raw isa PlanarCalibratedResult
        _planar_current_product(raw.raw.currents,raw.gap_voltage_transfer*stored)
    else
        _planar_current_product(raw.currents,stored)
    end
    Y=Matrix{ComplexF64}(transpose(stored)*raw.y*stored)
    S=planar_y_to_s(Y,refs)
    return PlanarContractedResult(raw,problem===nothing ? _planar_raw_problem(raw) : problem,
        ComplexF64(_planar_raw_frequency(raw)),ComplexF64(_planar_raw_omega(raw)),stored,refs,
        coefficients,Y,S)
end

function _planar_contracted_rhs!(rhs,prob,C,q)
    fill!(rhs,0)
    for b in eachindex(prob.basis.kind)
        p=prob.basis.port[b]
        p==0 && continue
        rhs[b]=-_planar_port_sign(prob.ports[p])*_planar_port_weight(prob.basis,b)*C[p,q]
    end
    return rhs
end

# Account for the source's owned wide currents and, after direct recovery,
# its independently retained original-matrix factorization and pivot vector.
function _planar_source_wide_payload(X,F)
    bits=_planar_current_precision(X)
    currents=_checked_array_payload_bytes(UInt8,_planar_wide_complex_payload(bits),size(X)...)
    factor=eltype(F.factors)===ComplexF64 ? 0 :
        _checked_payload_sum("owned wide source factor",
            _checked_array_payload_bytes(UInt8,_planar_wide_complex_payload(
                _planar_current_precision(F.factors)),size(F.factors)...),
            _checked_array_payload_bytes(Int,length(F.ipiv)))
    _checked_payload_sum("owned wide source result",currents,factor)
end

function _planar_source_y(prob,C,X)
    n=size(C,2);Y=zeros(ComplexF64,n,n)
    if eltype(X)!==ComplexF64
        bits=_planar_current_precision(X)
        # Each terminal sum owns two accumulators and reuses one coefficient
        # and one product. Round only the complete terminal reaction.
        re,im,coefficient,product=_planar_wide_dot_scratch(bits)
        for q in 1:n,r in 1:n
            _planar_wide_set!(re,0.);_planar_wide_set!(im,0.)
            for b in eachindex(prob.basis.kind)
                p=prob.basis.port[b];p==0 && continue
                weight=_planar_port_sign(prob.ports[p])*_planar_port_weight(prob.basis,b)
                _planar_wide_set!(coefficient,C[p,r]*weight)
                _planar_wide_mul!(product,coefficient,real(X[b,q]));_planar_wide_add!(re,re,product)
                _planar_wide_mul!(product,coefficient,imag(X[b,q]));_planar_wide_add!(im,im,product)
            end
            Y[r,q]=_planar_stored_phasor(complex(re,im))
        end
        return Y
    end
    for b in eachindex(prob.basis.kind)
        p=prob.basis.port[b]
        p==0 && continue
        weight=_planar_port_sign(prob.ports[p])*_planar_port_weight(prob.basis,b)
        for q in 1:n,r in 1:n
            Y[r,q]+=C[p,r]*weight*X[b,q]
        end
    end
    return Y
end

@inline function _planar_source_compensated_add(high,low,a,b)
    product=a*b;total=high+product;recovered=total-high
    error=fma(a,b,-product)+(high-(total-recovered))+(product-recovered)
    return total,low+error
end

function _planar_source_compensated_residual!(buffer,rhs,Z,x,imaginary)
    if imaginary && all(z->iszero(real(z)),x) && all(z->iszero(imag(z)),rhs)
        fill!(buffer,0)
        for j in axes(Z,2),i in axes(Z,1)
            high,low=_planar_source_compensated_add(real(buffer[i]),imag(buffer[i]),-imag(Z[i,j]),imag(x[j]))
            buffer[i]=complex(high,low)
        end
        for i in eachindex(buffer,rhs)
            high,low=_planar_source_compensated_add(real(buffer[i]),imag(buffer[i]),-real(rhs[i]),1.)
            buffer[i]=complex(high+low,0.)
        end
    else
        for i in axes(Z,1)
            real_high=0.;real_low=0.;imag_high=0.;imag_low=0.
            for j in axes(Z,2)
                a=real(Z[i,j]);b=imag(Z[i,j]);c=real(x[j]);d=imag(x[j])
                real_high,real_low=_planar_source_compensated_add(real_high,real_low,a,c)
                real_high,real_low=_planar_source_compensated_add(real_high,real_low,-b,d)
                imag_high,imag_low=_planar_source_compensated_add(imag_high,imag_low,a,d)
                imag_high,imag_low=_planar_source_compensated_add(imag_high,imag_low,b,c)
            end
            real_high,real_low=_planar_source_compensated_add(real_high,real_low,-real(rhs[i]),1.)
            imag_high,imag_low=_planar_source_compensated_add(imag_high,imag_low,-imag(rhs[i]),1.)
            buffer[i]=complex(real_high+real_low,imag_high+imag_low)
        end
    end
    return buffer
end

# Two reusable vectors fit the existing four-vector dense-source reservation.
# Preserve the returned column until both residual measures improve. Acceptance
# continues to use the original ordinary full-matrix voltage residual.
function _planar_dense_source_refine!(X,rhs,residual,weights,Z,F,scale,C,prob,rtol,voltage,galerkin)
    any(v->isfinite(v) && v>rtol,voltage) || return nothing
    imaginary=all(z->iszero(real(z)),Z)
    trial=similar(rhs);carry=similar(rhs)
    weighted_norm(values)=norm((values[k]/weights[k] for k in eachindex(values,weights)))
    for port in axes(X,2)
        isfinite(voltage[port]) && voltage[port]>rtol || continue
        _planar_contracted_rhs!(rhs,prob,C,port)
        source_norm=weighted_norm(rhs)
        column=view(X,:,port);copyto!(trial,column);fill!(carry,0)
        _planar_source_compensated_residual!(residual,rhs,Z,trial,imaginary)
        previous=weighted_norm(residual)/source_norm;best=previous
        # Each correction is quadratic; this bounds refinement by the
        # cubic direct factorization cost and still stops on nonprogress.
        for iteration in 1:size(Z,1)
            residual .*= scale;ldiv!(F,residual);residual .*= scale
            _planar_projection_compensated_update!(trial,residual,-1.,carry)
            vr,gr=_planar_source_residuals!(residual,rhs,weights,Z,trial)
            _planar_source_compensated_residual!(residual,rhs,Z,trial,imaginary)
            accurate=weighted_norm(residual)/source_norm
            isfinite(accurate) && accurate<previous || break
            if isfinite(vr) && vr<voltage[port] && accurate<best
                copyto!(column,trial);voltage[port]=vr;galerkin[port]=gr;best=accurate
            end
            voltage[port]<=rtol && break
            previous=accurate
        end
    end
    return nothing
end

@inline function _planar_source_residuals!(residual,rhs,weights,A,x)
    residual.=rhs./weights
    source_voltage_norm=norm(residual)
    mul!(residual,A,x);residual.-=rhs
    galerkin=norm(residual)/norm(rhs)
    residual./=weights
    return norm(residual)/source_voltage_norm,galerkin
end

"""Solve only the physical port combinations `Vraw=C*V`.
The real, finite, independent columns of `C` define voltage contracts;
terminal currents are the power-conjugate `transpose(C)*Iraw`.
This avoids allocating or solving independently driven raw layer sources.
Returns [`PlanarContractedResult`](@ref), whose `raw` source solution has
the physical coefficient columns and complete EM geometry/factor.

Both solver paths equilibrate differing sheet-width and via-area measures.
Dense uniform trace measures use unit scaling: global rescaling cannot
improve conditioning and introduces avoidable rounding. Iterative FFT `rtol` gates
the independently recomputed voltage-normalized
full-operator residual; the unscaled Galerkin residual is also retained.
The latter mixes different sheet and via row units. Dense `retain_matrix`
controls retention of the original unscaled matrix. Both dense modes
check the original physical equations against `rtol`. Wider owned currents
and an original-matrix factorization recover cancellation-sensitive cases;
failure to meet the requested target raises an error. Compact dense assembly
reuses its real and imaginary reaction accumulators for physical checks. Optional `problem`
preserves original physical port metadata. `max_bytes` preflights owned
numeric payloads across assembly, factor/operator, Krylov work and results."""
function solve_planar_contracted(prob::PlanarProblem,freq::Number,C::AbstractMatrix{<:Real};
        z0=50.,problem::Union{Nothing,PlanarProblem}=nothing,method::Symbol=:dense,
        retain_matrix::Bool=true,rtol::Real=1e-9,maxiter::Integer=0,
        memory::Integer=50,restart::Bool=true,precondition::Bool=true,
        max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    method in (:dense,:dense_fft,:ufft) || throw(ArgumentError("method must be :dense, :dense_fft or :ufft"))
    omega=2pi*ComplexF64(freq)
    isfinite(omega) && real(omega)>0 || throw(ArgumentError("frequency must be finite with Re>0"))
    nr,n=size(C);nb=planar_basis_count(prob.basis)
    nr==length(prob.ports) && 1<=n<=nr && all(isfinite,C) ||
        throw(ArgumentError("port contraction must be finite and match raw ports"))
    problem===nothing || length(problem.ports)==n || throw(ArgumentError(
        "original problem port metadata must match contraction columns"))
    isfinite(rtol) && rtol>0 && 0<=maxiter<=typemax(Int) && 1<=memory<=typemax(Int) ||
        throw(ArgumentError("invalid Krylov tolerance or iteration limits"))
    mem=restart ? min(BigInt(memory),BigInt(nb)) :
        max(BigInt(memory),iszero(maxiter) ? 2BigInt(nb) : BigInt(maxiter))
    common=_checked_payload_sum("contracted source solve",
        _checked_array_payload_bytes(Float64,4,nr,n), # contract validation/workspace
        _checked_array_payload_bytes(Float64,2,nb),
        _checked_array_payload_bytes(ComplexF64,nb,n),
        _checked_array_payload_bytes(ComplexF64,4,nb),
        _checked_array_payload_bytes(ComplexF64,5,n,n),
        _checked_array_payload_bytes(Float64,4,n),
        _checked_array_payload_bytes(ComplexF64,n),
        _checked_array_payload_bytes(Int,n))
    work=method!==:ufft ? _checked_dense_lu_work_bytes(ComplexF64,nb,
        2,common;label="contracted dense source solve") :
        _checked_payload_sum("contracted FFT source solve",common,
            _checked_array_payload_bytes(ComplexF64,nb,mem+10),
            _checked_array_payload_bytes(ComplexF64,mem+1,mem))
    _enforce_payload_limit(work,max_bytes,"contracted source solve","max_bytes")
    stored=Matrix{Float64}(C)
    all(isfinite,stored) && LinearAlgebra.rank(stored)==n ||
        throw(ArgumentError("port voltage contracts must have finite independent Float64 columns"))
    refs=_circuit_z0(z0,n;freq=real(freq))
    weights=Float64[_planar_port_weight(prob.basis,b) for b in 1:nb]
    all(w->isfinite(w) && w>0,weights) || throw(ArgumentError("source basis has invalid trace measure"))
    uniform=!isempty(weights) && all(==(weights[1]),weights)
    scale=precondition && (method===:ufft || !uniform) ? inv.(weights) : ones(Float64,nb)
    X=Matrix{ComplexF64}(undef,nb,n)
    rhs=zeros(ComplexF64,nb);residual=similar(rhs)
    iterations=zeros(Int,n);voltage_residuals=Float64[];galerkin_residuals=Float64[]
    Z=nothing;F=nothing;A=nothing
    if method!==:ufft
        compact_components=!retain_matrix && method===:dense
        assembly_reserve=_checked_payload_sum("contracted assembly reservation",common,
            _checked_array_payload_bytes(ComplexF64,nb,nb,compact_components ? 0 : 1))
        original=method===:dense_fft ?
            assemble_planar_z_ufft(prob,freq;max_bytes=max_bytes-assembly_reserve,kw...) :
            assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,omega;
                vias=prob.vias,vols=prob.vols,_retain_components=compact_components,
                max_bytes=max_bytes-assembly_reserve,kw...)
        all(isfinite,original) || throw(ArgumentError("nonfinite planar source matrix"))
        scaled=compact_components ? original.matrix : copy(original)
        if precondition && !uniform
            for q in 1:nb,p in 1:nb
                scaled[p,q]*=scale[p]*scale[q]
            end
        end
        F=lu!(scaled)
        for q in 1:n
            _planar_contracted_rhs!(rhs,prob,stored,q)
            norm(rhs)>0 || throw(ArgumentError("physical port $q has no source excitation"))
            X[:,q].=scale.*rhs
        end
        ldiv!(F,X)
        X .*= scale
        initial_factor=F
        X,voltage_residuals,F=_planar_dense_checked_currents(X,original,F,prob,work,max_bytes;
            contracts=stored,basis_scale=scale,rtol)
        # A wide recovery factors the original physical matrix, without the
        # two-sided scaling used by the initial Float64 factorization.
        F===initial_factor || fill!(scale,1.)
        Z=retain_matrix ? original : nothing
        if eltype(X)===ComplexF64
            for q in 1:n
                _planar_contracted_rhs!(rhs,prob,stored,q)
                _planar_source_compensated_residual!(residual,rhs,original,view(X,:,q),false)
                push!(galerkin_residuals,norm(residual)/norm(rhs))
            end
        else
            bits=_planar_current_precision(X)
            _enforce_payload_limit(_checked_payload_sum("wide source residual reporting",work,
                _planar_source_wide_payload(X,F),
                # Read-only component references share the solved scalars.
                _checked_array_payload_bytes(UInt8,2sizeof(BigFloat),nb,n),
                _checked_array_payload_bytes(ComplexF64,2,nb,n),
                _checked_array_payload_bytes(UInt8,_planar_wide_scalar_payload(bits),4)),
                max_bytes,"wide source residual reporting","max_bytes")
            parts=(re=map(real,X),im=map(imag,X));scratch=_planar_wide_dot_scratch(bits)
            sources=Matrix{ComplexF64}(undef,nb,n);errors=similar(sources)
            for q in 1:n
                _planar_contracted_rhs!(view(sources,:,q),prob,stored,q)
            end
            for q in 1:n
                voltage_residuals[q]=_planar_wide_physical_residual!(view(errors,:,q:q),original,
                    (re=view(parts.re,:,q:q),im=view(parts.im,:,q:q)),
                    view(sources,:,q:q),weights,scratch)
                push!(galerkin_residuals,norm(view(errors,:,q))/norm(view(sources,:,q)))
            end
        end
    else
        A=planar_ufft_operator(prob,freq;max_bytes=max_bytes-common-
            _checked_payload_sum("contracted Krylov reservation",
                _checked_array_payload_bytes(ComplexF64,nb,mem+10),
                _checked_array_payload_bytes(ComplexF64,mem+1,mem)),kw...)
        D=precondition ? Diagonal(scale) : I
        for q in 1:n
            _planar_contracted_rhs!(rhs,prob,stored,q)
            source_scale=maximum(abs,rhs)
            isfinite(source_scale) && source_scale>0 || throw(ArgumentError(
                "physical port $q must have finite nonzero source excitation"))
            # Krylov also stops at an absolute machine-precision floor.
            # Normalize in the existing residual buffer so small voltage
            # contracts retain the same relative solve accuracy.
            residual.=rhs./source_scale
            x,stats=Krylov.gmres(A,residual;rtol=Float64(rtol)/10,atol=0.,
                itmax=Int(maxiter),memory=min(Int(memory),nb),restart,
                reorthogonalization=true,M=D,N=D)
            x .*= source_scale
            vr,gr=_planar_source_residuals!(residual,rhs,weights,A,x)
            isfinite(vr) && vr<=rtol || throw(ErrorException(
                "contracted FFT port $q failed voltage residual gate: $vr > $rtol after $(stats.niter) iterations"))
            X[:,q].=x;iterations[q]=stats.niter
            push!(voltage_residuals,vr);push!(galerkin_residuals,gr)
        end
    end
    if eltype(X)!==ComplexF64
        bits=_planar_current_precision(X)
        _enforce_payload_limit(_checked_payload_sum("wide source terminal extraction",work,
            _planar_source_wide_payload(X,F),
            _checked_array_payload_bytes(UInt8,_planar_wide_scalar_payload(bits),4)),
            max_bytes,"wide source terminal extraction","max_bytes")
    end
    Y=_planar_source_y(prob,stored,X);S=planar_y_to_s(Y,refs)
    raw=PlanarSourceResult(prob,ComplexF64(freq),omega,stored,refs,Z,F,A,scale,
        X,Y,S,iterations,voltage_residuals,galerkin_residuals)
    return PlanarContractedResult(raw,problem===nothing ? prob : problem,
        ComplexF64(freq),omega,stored,refs,X,Y,S)
end

"""Reconstruct currents for contracted terminal voltages or incident
power waves. The default drives one contracted port at one volt. Sources
map through the retained voltage contraction into the raw physical solve."""
function planar_current_maps(result::PlanarContractedResult;port::Integer=1,
        voltages=nothing,incident_waves=nothing,kw...)
    n=length(result.z0)
    voltages!==nothing && incident_waves!==nothing && throw(ArgumentError(
        "provide voltages or incident_waves"))
    supplied=voltages===nothing ? incident_waves : voltages
    v=if supplied===nothing
        1<=port<=n || throw(ArgumentError("contracted port index is invalid"))
        a=zeros(ComplexF64,n);a[port]=1;a
    else
        supplied isa AbstractVector && length(supplied)==n && all(isfinite,supplied) ||
            throw(ArgumentError("contracted excitation must be finite and match ports"))
        _planar_stored_phasor.(supplied)
    end
    incident_waves===nothing || (v=_planar_wave_voltage(result.s,v,result.z0))
    if result.raw isa PlanarSourceResult && result.currents===result.raw.currents
        return planar_current_maps(result.raw;voltages=v,kw...)
    end
    return planar_current_maps(result.raw;voltages=result.contraction*v,kw...)
end

"""Reconstruct a directly driven source solution in its physical port
basis. `incident_waves` uses the stored physical reference impedances;
the raw source geometry is retained for every conductor map."""
function planar_current_maps(source::PlanarSourceResult;port::Integer=1,
        voltages=nothing,incident_waves=nothing,
        z_fraction::Real=.5,max_bytes::Integer=_default_max_dense_payload_bytes())
    n=length(source.z0)
    voltages!==nothing && incident_waves!==nothing && throw(ArgumentError(
        "provide voltages or incident_waves"))
    nb=size(source.currents,1)
    product_bytes=eltype(source.currents)===ComplexF64 ?
        _checked_array_payload_bytes(ComplexF64,nb) :
        _planar_owned_current_product_payload(_planar_current_precision(source.currents),nb)
    reserve=_checked_payload_sum("physical source current excitation",product_bytes,
        _checked_array_payload_bytes(ComplexF64,3,n))
    _enforce_payload_limit(reserve,max_bytes,"physical source current excitation","max_bytes")
    supplied=voltages===nothing ? incident_waves : voltages
    v=if supplied===nothing
        1<=port<=n || throw(ArgumentError("physical port index is invalid"))
        a=zeros(ComplexF64,n);a[port]=1;a
    else
        supplied isa AbstractVector && length(supplied)==n && all(isfinite,supplied) ||
            throw(ArgumentError("physical excitation must be finite and match ports"))
        _planar_stored_phasor.(supplied)
    end
    all(isfinite,v) || throw(ArgumentError("physical excitation must fit finite ComplexF64 values"))
    incident_waves===nothing || (v=_planar_wave_voltage(source.s,v,source.z0))
    return _planar_current_maps_from_coefficients(source.problem,_planar_current_product(source.currents,v);
        z_fraction,max_bytes=max_bytes-reserve)
end
