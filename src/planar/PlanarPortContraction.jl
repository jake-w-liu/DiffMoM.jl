export PlanarContractedResult, PlanarSourceResult, planar_contract_ports, solve_planar_contracted

"""A raw planar result with real voltage-contract matrix `contraction`.
Raw port voltages equal `contraction*voltages`; conjugate port currents
are `transpose(contraction)*raw_currents`. `currents,y,s` use the contracted
ports, and `raw` retains the complete source geometry and factor/operator.
`problem` may preserve an original problem's port metadata after refinement."""
struct PlanarContractedResult{R}
    raw::R
    problem::PlanarProblem
    freq::ComplexF64
    omega::ComplexF64
    contraction::Matrix{Float64}
    z0::Vector{ComplexF64}
    currents::Matrix{ComplexF64}
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
residuals are available when `z_mom` is retained."""
struct PlanarSourceResult{F,O}
    problem::PlanarProblem
    freq::ComplexF64
    omega::ComplexF64
    contraction::Matrix{Float64}
    z0::Vector{ComplexF64}
    z_mom::Union{Nothing,Matrix{ComplexF64}}
    lu_fact::F
    operator::O
    basis_scale::Vector{Float64}
    currents::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    iterations::Vector{Int}
    relative_residuals::Vector{Float64}
    galerkin_relative_residuals::Vector{Float64}
end

_planar_coefficient_columns(raw::Union{PlanarResult,PlanarUFFTResult,PlanarContractedResult,PlanarSourceResult})=raw.currents
_planar_coefficient_columns(raw::PlanarCalibratedResult)=raw.raw.currents*raw.gap_voltage_transfer
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
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    nraw,n=size(C)
    nraw==size(raw.y,1) && 1<=n<=nraw && all(isfinite,C) ||
        throw(ArgumentError("port contraction must be finite and match raw ports"))
    nb=size(raw isa PlanarCalibratedResult ? raw.raw.currents : raw.currents,1)
    _enforce_payload_limit(_planar_contract_payload(nb,nraw,n),max_bytes,
        "planar port contraction","max_bytes")
    stored=Matrix{Float64}(C)
    all(isfinite,stored) || throw(ArgumentError("port contraction must fit finite Float64 values"))
    LinearAlgebra.rank(stored)==n || throw(ArgumentError("port voltage contracts must have independent columns"))
    refs=_circuit_z0(z0,n;freq=real(_planar_raw_frequency(raw)))
    problem===nothing || length(problem.ports)==n || throw(ArgumentError(
        "original problem port metadata must match contraction columns"))
    coefficients=if raw isa PlanarCalibratedResult
        raw.raw.currents*(raw.gap_voltage_transfer*stored)
    else
        raw.currents*stored
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

function _planar_source_y(prob,C,X)
    n=size(C,2);Y=zeros(ComplexF64,n,n)
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

"""Solve only the physical port combinations `Vraw=C*V`.
The real, finite, independent columns of `C` define voltage contracts;
terminal currents are the power-conjugate `transpose(C)*Iraw`.
This avoids allocating or solving independently driven raw layer sources.
Returns [`PlanarContractedResult`](@ref), whose `raw` source solution has
the physical coefficient columns and complete EM geometry/factor.

Both solver paths equilibrate differing sheet-width and via-area measures.
Dense uniform trace measures use unit scaling: global rescaling cannot
improve conditioning and introduces avoidable rounding. FFT `rtol` gates
the independently recomputed voltage-normalized
full-operator residual; the unscaled Galerkin residual is also retained.
The latter mixes different sheet and via row units. Dense `retain_matrix`
controls retention of the original unscaled matrix. Optional `problem`
preserves original physical port metadata. `max_bytes` preflights owned
numeric payloads across assembly, factor/operator, Krylov work and results."""
function solve_planar_contracted(prob::PlanarProblem,freq::Number,C::AbstractMatrix{<:Real};
        z0=50.,problem::Union{Nothing,PlanarProblem}=nothing,method::Symbol=:dense,
        retain_matrix::Bool=true,rtol::Real=1e-9,maxiter::Integer=0,
        memory::Integer=50,restart::Bool=true,precondition::Bool=true,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
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
        retain_matrix ? 2 : 1,common;label="contracted dense source solve") :
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
        assembly_reserve=_checked_payload_sum("contracted assembly reservation",common,
            _checked_array_payload_bytes(ComplexF64,nb,nb,retain_matrix ? 1 : 0))
        original=method===:dense_fft ?
            assemble_planar_z_ufft(prob,freq;max_bytes=max_bytes-assembly_reserve,kw...) :
            assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,omega;
                vias=prob.vias,vols=prob.vols,max_bytes=max_bytes-assembly_reserve,kw...)
        all(isfinite,original) || throw(ArgumentError("nonfinite planar source matrix"))
        scaled=retain_matrix ? copy(original) : original
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
        if retain_matrix
            Z=original
            for q in 1:n
                _planar_contracted_rhs!(rhs,prob,stored,q)
                mul!(residual,Z,view(X,:,q));residual.-=rhs
                push!(galerkin_residuals,norm(residual)/norm(rhs))
                push!(voltage_residuals,norm(residual./weights)/norm(rhs./weights))
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
            norm(rhs)>0 || throw(ArgumentError("physical port $q has no source excitation"))
            x,stats=Krylov.gmres(A,rhs;rtol=Float64(rtol)/10,atol=0.,
                itmax=Int(maxiter),memory=min(Int(memory),nb),restart,
                reorthogonalization=true,M=D,N=D)
            mul!(residual,A,x);residual.-=rhs
            vr=norm(residual./weights)/norm(rhs./weights)
            gr=norm(residual)/norm(rhs)
            isfinite(vr) && vr<=rtol || throw(ErrorException(
                "contracted FFT port $q failed voltage residual gate: $vr > $rtol after $(stats.niter) iterations"))
            X[:,q].=x;iterations[q]=stats.niter
            push!(voltage_residuals,vr);push!(galerkin_residuals,gr)
        end
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
        z_fraction::Real=.5,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    n=length(source.z0)
    voltages!==nothing && incident_waves!==nothing && throw(ArgumentError(
        "provide voltages or incident_waves"))
    reserve=_checked_payload_sum("physical source current excitation",
        _checked_array_payload_bytes(ComplexF64,size(source.currents,1)),
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
    return _planar_current_maps_from_coefficients(source.problem,source.currents*v;
        z_fraction,max_bytes=max_bytes-reserve)
end
