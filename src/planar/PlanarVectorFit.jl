# Real shared-pole vector fitting. Private numerator coefficients are
# eliminated with one QR factorization before fitting the shared poles;
# workspace grows with nports^2, rather than nports^4.

export PlanarRationalModel, planar_fit_rational, planar_rational_eval
export planar_write_spice, planar_rational_passivity
export planar_rational_certificate

"""Stable real admittance model `Y(s)=d+s*e+sum(residues[k]/(s-poles[k]))`.
Complex poles/residues occur in conjugate pairs. `sampled_passive` and
`passivity_margin` describe the checked frequency grid. `globally_passive`
records the positive-real certificate and `passivity_method` identifies
it. `rms_error` and `relative_rms_error` include conductance loading and
the recorded `capacitance_adjustment` used in passivity repair."""
struct PlanarRationalModel
    poles::Vector{ComplexF64}
    residues::Vector{Matrix{ComplexF64}}
    d::Matrix{Float64}
    e::Matrix{Float64}
    frequencies::Vector{Float64}
    rms_error::Float64
    relative_rms_error::Float64
    sampled_passive::Bool
    globally_passive::Bool
    passivity_method::Symbol
    passivity_margin::Float64
    passivity_shift::Float64
    capacitance_adjustment::Float64
end

"""Evaluate a real, stable N-port admittance model at frequency [Hz]."""
function planar_rational_eval(model::PlanarRationalModel,f::Real)
    isfinite(f) || throw(ArgumentError("model frequency must be finite"))
    s = 2pi*1im*f
    Y = model.d .+ s .* model.e
    for k in eachindex(model.poles)
        Y .+= model.residues[k] ./ (s-model.poles[k])
    end
    return Y
end

function _vf_pair_indices(poles)
    ci = zeros(Int,length(poles))
    k = 1
    while k <= length(poles)
        if iszero(imag(poles[k]))
            k += 1
        else
            k < length(poles) && isapprox(poles[k+1],conj(poles[k]);rtol=1e-7) ||
                throw(ArgumentError("vector-fit poles must contain adjacent conjugate pairs"))
            ci[k],ci[k+1] = 1,2
            k += 2
        end
    end
    return ci
end

_vf_sort(poles) = sort(ComplexF64.(poles);
    by=p -> (round(real(p);sigdigits=8),abs(imag(p)),-imag(p)))

function _vf_basis(s,poles)
    ci = _vf_pair_indices(poles)
    A = Matrix{ComplexF64}(undef,length(s),length(poles)+2)
    for r in eachindex(s)
        k = 1
        while k <= length(poles)
            if ci[k] == 0
                A[r,k] = inv(s[r]-poles[k])
                k += 1
            else
                p = poles[k]
                A[r,k] = inv(s[r]-p)+inv(s[r]-conj(p))
                A[r,k+1] = 1im*(inv(s[r]-p)-inv(s[r]-conj(p)))
                k += 2
            end
        end
        A[r,end-1],A[r,end] = 1.0,s[r]
    end
    return A
end

function _vf_shared_relocate(s,Ys,poles)
    order,m,nel = length(poles),length(s),length(Ys[1])
    basis = _vf_basis(s,poles)
    Ar = vcat(real.(basis),imag.(basis))
    factor = LinearAlgebra.qr(Ar)
    omitted = (order+3):(2m)
    C = zeros(Float64,length(omitted)*nel,order)
    b = zeros(Float64,size(C,1))
    for e in 1:nel
        fe = ComplexF64[Y[e] for Y in Ys]
        sigma = -fe .* view(basis,:,1:order)
        reduced = transpose(factor.Q)*hcat(vcat(real.(sigma),imag.(sigma)),
            vcat(real.(fe),imag.(fe)))
        rows = ((e-1)*length(omitted)+1):(e*length(omitted))
        C[rows,:] .= view(reduced,omitted,1:order)
        b[rows] .= view(reduced,omitted,order+1)
    end
    maximum(abs,b;init=0.0) <= 1e-12*max(maximum(abs,Ys[1]),1e-12) && return poles
    # Rank-revealing SVD prevents a constant/affine response or inactive
    # matrix entry from creating spurious poles through a singular fit.
    F = LinearAlgebra.svd(C;full=false)
    cutoff = maximum(F.S;init=0.0)*1e-10
    x = F.V * [F.S[k] > cutoff ? dot(view(F.U,:,k),b)/F.S[k] : 0.0
        for k in eachindex(F.S)]
    H = zeros(Float64,order,order)
    drive = zeros(Float64,order)
    ci = _vf_pair_indices(poles)
    k = 1
    while k <= order
        if ci[k] == 0
            H[k,k],drive[k] = real(poles[k]),1.0
            k += 1
        else
            a,beta = real(poles[k]),imag(poles[k])
            H[k,k]=H[k+1,k+1]=a
            H[k,k+1],H[k+1,k] = beta,-beta
            drive[k] = 2.0
            k += 2
        end
    end
    H .-= drive*transpose(x)
    return _vf_sort([complex(-max(abs(real(p)),1e-9),imag(p)) for p in eigvals(H)])
end

"""Check the Hermitian part of the admittance on an explicit nonempty
frequency grid. Returns `(passive,margin,frequencies)`; this is a sampled
check, and stable poles establish causality of the fitted rational model."""
function planar_rational_passivity(model::PlanarRationalModel,frequencies;
        tol::Real=1e-9)
    isfinite(tol) && tol >= 0 || throw(ArgumentError("passivity tolerance must be finite and nonnegative"))
    fs = Float64.(collect(frequencies))
    !isempty(fs) && all(f -> isfinite(f) && f >= 0,fs) ||
        throw(ArgumentError("passivity grid must be nonempty, finite, and nonnegative"))
    margin = Inf
    for f in fs
        Y = planar_rational_eval(model,f)
        margin = min(margin,minimum(eigvals(Hermitian((Y+adjoint(Y))/2))))
    end
    return (passive=margin >= -tol,margin=margin,frequencies=fs)
end

function _vf_psd(A,tol)
    # A negative energy direction of any size is not a positive-real
    # certificate. Approximate symmetry is also insufficient: an affine
    # skew part creates an indefinite Hermitian response at large |omega|.
    A == transpose(A) || return false
    all(isfinite,A) && all(p -> A[p,p]>=0,axes(A,1)) || return false
    return minimum(eigvals(LinearAlgebra.Symmetric(A))) >= 0
end

"""Numerical positive-real certificate over all frequencies. A symmetric
positive-semidefinite affine term is required. Sufficient positive-residue
and uniform norm bounds also cover semidefinite feedthrough. Otherwise a
balanced real Hamiltonian locates every potential imaginary-axis zero of
the Hermitian admittance, and strict positive feedthrough plus absence of
such zeros certifies the proper response. `certified=false` is not proof
of nonpassivity; it can indicate a zero/tolerance boundary. The certificate
reports its method and potential crossover frequencies, not a grid claim.

The Hamiltonian criterion follows the positive-real state-space test;
see Semlyen & Gustavsen, IEEE TPWRD 24(1), 2009, DOI 10.1109/TPWRD.2008.923406."""
function planar_rational_certificate(model::PlanarRationalModel;
        tol::Real=1e-8,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    isfinite(tol) && tol>0 || throw(ArgumentError("certificate tolerance must be finite and positive"))
    no = (certified=false,method=:none,crossings_hz=Float64[])
    n = size(model.d,1)
    n>0 && size(model.d)==size(model.e)==(n,n) &&
        length(model.residues)==length(model.poles) &&
        all(isfinite,model.d) && all(isfinite,model.e) &&
        all(R -> size(R)==(n,n) && all(isfinite,R),model.residues) || return no
    all(p -> isfinite(p) && real(p)<0,model.poles) || return no
    ci = _vf_pair_indices(model.poles)
    for k in eachindex(ci)
        if ci[k]==0
            all(iszero,imag.(model.residues[k])) || return no
            iszero(imag(model.poles[k])) || return no
        elseif ci[k]==1
            model.poles[k+1]==conj(model.poles[k]) &&
                model.residues[k+1]==conj.(model.residues[k]) || return no
        end
    end
    _vf_psd(model.e,tol) || return no
    symmetric_d = (model.d+transpose(model.d))/2
    minimum_d = minimum(eigvals(LinearAlgebra.Symmetric(symmetric_d)))
    # Every real negative pole with PSD residue is a positive-real term.
    if _vf_psd(model.d,tol) && all(k -> iszero(imag(model.poles[k])) &&
            all(iszero,imag.(model.residues[k])) &&
            _vf_psd(real.(model.residues[k]),tol),eachindex(model.poles))
        return (certified=true,method=:positive_residues,crossings_hz=Float64[])
    end
    bound = sum(opnorm(model.residues[k],2)/(-real(model.poles[k]))
        for k in eachindex(model.poles);init=0.0)
    if minimum_d >= bound
        return (certified=true,method=:uniform_bound,crossings_hz=Float64[])
    end
    minimum_d > tol*max(opnorm(symmetric_d,2),1e-30) || return no
    states = length(model.poles)*n
    _enforce_payload_limit(_checked_array_payload_bytes(Float64,16,states,states;
        label="positive-real certificate workspace"),max_bytes,"positive-real certificate","max_bytes")
    A,B,C = zeros(Float64,states,states),zeros(Float64,states,n),zeros(Float64,n,states)
    k = 1
    while k<=length(model.poles)
        left = ((k-1)*n+1):(k*n)
        p = model.poles[k]
        if ci[k]==0
            A[left,left] .= real(p)*Matrix{Float64}(I,n,n)
            B[left,:] .= Matrix{Float64}(I,n,n)
            C[:,left] .= real.(model.residues[k])
            k+=1
        else
            right = (k*n+1):((k+1)*n)
            A[left,left] .= real(p)*Matrix{Float64}(I,n,n)
            A[right,right] .= real(p)*Matrix{Float64}(I,n,n)
            A[left,right] .= imag(p)*Matrix{Float64}(I,n,n)
            A[right,left] .= -imag(p)*Matrix{Float64}(I,n,n)
            B[left,:] .= Matrix{Float64}(I,n,n)
            C[:,left] .= 2real.(model.residues[k])
            C[:,right] .= 2imag.(model.residues[k])
            k+=2
        end
    end
    R = model.d+transpose(model.d)
    Abar = A-B*(R\C)
    G,Q = B*(R\transpose(B)),transpose(C)*(R\C)
    balancing = sqrt(max(opnorm(Q,Inf),1e-30)/max(opnorm(G,Inf),1e-30))
    H = [Abar -G*balancing; Q/balancing -transpose(Abar)]
    values = eigvals(H)
    crossings = sort!(unique([abs(imag(value))/(2pi) for value in values
        if abs(real(value)) <= tol*max(abs(imag(value)),1.0)]))
    return (certified=isempty(crossings),method=:hamiltonian,crossings_hz=crossings)
end

"""Fit a real stable shared-pole N-port model from finite Y matrices
(`format=:y`, default) or S matrices (`format=:s`, reference `z0`). The
affine term models capacitance. `enforce_passivity=true` projects the
affine term to PSD, adds conductance when needed to obtain a global
positive-real certificate, records both adjustments, and recomputes fit
error. The global norm-bound fallback can reduce fit accuracy; the model
reports that cost. Native MNA S blocks are preferable for ideal
shorts/thrus whose Y conversion is singular. `max_bytes` bounds fit
workspace before converting input data or allocating fit matrices."""
function planar_fit_rational(series::AbstractVector{<:AbstractMatrix},
        frequencies::AbstractVector{<:Real};format::Symbol=:y,z0=50.0,
        order::Integer=4,iterations::Integer=8,enforce_passivity::Bool=true,
        passivity_tol::Real=1e-9,n_passivity::Integer=max(257,8length(frequencies)),
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    m = length(series)
    m == length(frequencies) && m >= 3 || throw(ArgumentError("fit requires at least three matching samples"))
    order >= 1 && iterations >= 0 || throw(ArgumentError("order must be positive and iterations nonnegative"))
    order+2 < 2m || throw(ArgumentError("fit order exceeds the sample capacity"))
    isfinite(passivity_tol) && passivity_tol >= 0 && n_passivity >= 2 ||
        throw(ArgumentError("invalid passivity tolerance/grid"))
    format in (:y,:s) || throw(ArgumentError("fit format must be :y or :s"))
    n = size(series[1],1)
    n > 0 && all(Y -> size(Y)==(n,n) && all(isfinite,Y),series) ||
        throw(ArgumentError("fit matrices must be finite, square, and have equal sizes"))
    all(f -> isfinite(f) && f > 0,frequencies) || throw(ArgumentError("fit frequencies must be finite and positive"))
    perm = sortperm(frequencies)
    fs = Float64.(frequencies[perm])
    all(diff(fs) .> 0) || throw(ArgumentError("fit frequencies must be distinct"))
    # QR elimination avoids the giant sparse-in-structure shared solve.
    _enforce_payload_limit(_checked_payload_sum("vector fit",
        _checked_array_payload_bytes(ComplexF64,m,n,n),
        _checked_array_payload_bytes(Float64,8,m,n,n,order+2),
        _checked_array_payload_bytes(Float64,8,m,order+2),
        _checked_array_payload_bytes(ComplexF64,2,order+2,n,n)),
        max_bytes,"vector fit","max_bytes")
    Ys = Matrix{ComplexF64}[]
    for index in perm
        H = Matrix{ComplexF64}(series[index])
        if format === :s
            references=_circuit_z0(z0,n;freq=frequencies[index])
            H = planar_s_to_y(H,references;max_bytes)
        end
        push!(Ys,H)
    end
    scale = 2pi*last(fs)
    s = ComplexF64[2pi*1im*f/scale for f in fs]
    poles = ComplexF64[]
    betas = order÷2 <= 1 ? [0.5*(first(fs)/last(fs)+1)] :
        collect(range(first(fs)/last(fs),1.0;length=order÷2))
    for beta in betas
        length(poles)+2 <= order || break
        push!(poles,complex(-beta/100,beta),complex(-beta/100,-beta))
    end
    isodd(order) && push!(poles,complex(-0.5,0.0))
    poles = _vf_sort(poles)
    for _ in 1:iterations
        poles = _vf_shared_relocate(s,Ys,poles)
    end
    basis = _vf_basis(s,poles)
    Ar = vcat(real.(basis),imag.(basis))
    rhs = Matrix{Float64}(undef,2m,n*n)
    for e in 1:n*n,r in 1:m
        rhs[r,e],rhs[m+r,e] = real(Ys[r][e]),imag(Ys[r][e])
    end
    fitted = Ar \ rhs
    residues = [zeros(ComplexF64,n,n) for _ in poles]
    ci = _vf_pair_indices(poles)
    k = 1
    while k <= order
        if ci[k] == 0
            residues[k] .= reshape(view(fitted,k,:),n,n)*scale
            k += 1
        else
            residues[k] .= reshape(view(fitted,k,:),n,n)*scale +
                1im*reshape(view(fitted,k+1,:),n,n)*scale
            residues[k+1] .= conj.(residues[k])
            k += 2
        end
    end
    d = Matrix(reshape(fitted[end-1,:],n,n))
    e = Matrix(reshape(fitted[end,:],n,n))/scale
    poles .*= scale
    capacitance_adjustment = 0.0
    if enforce_passivity
        decomposition = eigen(LinearAlgebra.Symmetric((e+transpose(e))/2))
        floor = 32eps(Float64)*max(maximum(abs,decomposition.values),0)
        corrected = decomposition.vectors*Diagonal(max.(decomposition.values,floor))*transpose(decomposition.vectors)
        corrected = (corrected+transpose(corrected))/2
        capacitance_adjustment = norm(corrected-e)
        e .= corrected
    end
    provisional = PlanarRationalModel(poles,residues,d,e,fs,NaN,NaN,false,false,:none,NaN,0,capacitance_adjustment)
    grid = collect(range(0.0,last(fs);length=n_passivity))
    check = planar_rational_passivity(provisional,grid;tol=passivity_tol)
    shift = enforce_passivity && check.margin < -passivity_tol ?
        -check.margin+passivity_tol : 0.0
    for p in 1:n
        d[p,p] += shift
    end
    certificate = planar_rational_certificate(provisional;max_bytes=max_bytes)
    if enforce_passivity && !certificate.certified
        # A global norm bound is conservative but exact: each Hermitian
        # residue contribution is bounded below by -||R||/|Re(p)|.
        bound = sum(opnorm(residues[k],2)/(-real(poles[k])) for k in eachindex(poles);init=0.0)
        smallest_d = minimum(eigvals(LinearAlgebra.Symmetric((d+transpose(d))/2)))
        extra = max(bound-smallest_d,0)+max(passivity_tol,1e-12*max(bound,1.0))
        for p in 1:n
            d[p,p] += extra
        end
        shift += extra
        certificate = planar_rational_certificate(provisional;max_bytes=max_bytes)
    end
    num = sum(sum(abs2,planar_rational_eval(provisional,fs[k])-Ys[k]) for k in 1:m)
    den = sum(sum(abs2,Y) for Y in Ys)
    margin = check.margin+shift
    return PlanarRationalModel(poles,residues,d,e,fs,sqrt(num/(m*n*n)),
        den>0 ? sqrt(num/den) : sqrt(num/(m*n*n)),margin>=-passivity_tol,
        certificate.certified,certificate.method,margin,shift,capacitance_adjustment)
end

function _vf_spice_damping(io,node,alpha)
    resistance=inv(alpha)
    if isfinite(resistance)
        println(io,"R",node," ",node," 0 ",resistance)
    else
        # A finite decay rate can have an unrepresentable reciprocal.
        # The self-controlled current source is the same conductance,
        # retaining that stable pole with a finite SPICE coefficient.
        println(io,"Gdecay",node," ",node," 0 ",node," 0 ",alpha)
    end
    return nothing
end

"""Write an N-port SPICE `.subckt` realizing the fitted admittance
exactly with real R/C elements and controlled sources. Pole equations
and state voltages are scaled by their decay rate where representable;
the affine term uses copied input voltages and sensed
capacitor currents. The final `dm_ref` terminal is a floating reference.
Auxiliary state voltages use node `0` to avoid subtracting tiny state
voltages from a large reference voltage; all external current and voltage
couplings use the declared reference terminal.
Export preserves the
model's sampled passivity status and reported fit error in comments."""
function planar_write_spice(model::PlanarRationalModel,path::AbstractString;
        subckt_name::AbstractString="diffmom_nport",port_names=nothing)
    n = size(model.d,1)
    size(model.d)==size(model.e)==(n,n) && length(model.residues)==length(model.poles) ||
        throw(ArgumentError("invalid rational model dimensions"))
    ports = port_names === nothing ? ["p$p" for p in 1:n] : String.(port_names)
    length(ports)==n && length(unique(lowercase.(ports)))==n &&
        all(p -> occursin(r"^[A-Za-z][A-Za-z0-9_]*$",p) &&
            lowercase(p)!="gnd" && !startswith(lowercase(p),"dm_"),ports) ||
        throw(ArgumentError("SPICE port names must be distinct identifiers outside the dm_ namespace"))
    occursin(r"^[A-Za-z][A-Za-z0-9_]*$",subckt_name) ||
        throw(ArgumentError("invalid SPICE subcircuit name"))
    all(p -> isfinite(p) && real(p)<0,model.poles) ||
        throw(ArgumentError("SPICE export requires stable finite poles"))
    ci = _vf_pair_indices(model.poles)
    all(isfinite,model.d) && all(isfinite,model.e) &&
        all(R -> size(R)==(n,n) && all(isfinite,R),model.residues) ||
        throw(ArgumentError("rational model coefficients are invalid"))
    # Real realization requires a real residue at every real pole and
    # conjugate residues at each complex pair.
    for k in eachindex(ci)
        if ci[k]==0
            iszero(imag(model.poles[k])) && all(iszero,imag.(model.residues[k])) ||
                throw(ArgumentError("real pole has a complex residue"))
        elseif ci[k]==1
            model.poles[k+1]==conj(model.poles[k]) &&
                model.residues[k+1]==conj.(model.residues[k]) ||
                throw(ArgumentError("pole residues are not conjugate pairs"))
        end
    end
    open(path,"w") do io
        println(io,"* DiffMoM real stable N-port admittance model")
        println(io,"* relative_rms_error: ",model.relative_rms_error)
        println(io,"* sampled_passive: ",model.sampled_passive,
            "  checked_margin: ",model.passivity_margin,"  conductance_shift: ",model.passivity_shift)
        println(io,"* globally_passive: ",model.globally_passive,
            "  certificate: ",model.passivity_method,"  capacitance_adjustment: ",model.capacitance_adjustment)
        println(io,".subckt ",subckt_name," ",join(ports," ")," dm_ref")
        for i in 1:n,j in 1:n
            iszero(model.d[i,j]) || println(io,"Gd",i,"_",j," ",ports[i],
                " dm_ref ",ports[j]," dm_ref ",model.d[i,j])
        end
        for j in 1:n
            any(!iszero,view(model.e,:,j)) || continue
            println(io,"Edm_copy",j," dm_v",j," 0 ",ports[j]," dm_ref 1")
            println(io,"Vdm_sense",j," dm_v",j," dm_c",j," 0")
            println(io,"Cdm_affine",j," dm_c",j," 0 1")
            for i in 1:n
                iszero(model.e[i,j]) || println(io,"Fdm_affine",i,"_",j,
                    " ",ports[i]," dm_ref Vdm_sense",j," ",model.e[i,j])
            end
        end
        state = 0
        k = 1
        while k <= length(model.poles)
            alpha = -real(model.poles[k])
            scale=max(1.0,alpha)
            for j in 1:n
                # Keep DC state voltages and output gains balanced together.
                # When a residue/alpha ratio cannot be stored, retain the
                # finite original residue scale for that input column.
                state_scale=all(r->isfinite(real(r)/alpha) && isfinite(imag(r)/alpha),
                    view(model.residues[k],:,j)) ? alpha : scale
                state += 1
                u = "dm_u$state"
                println(io,"C",u," ",u," 0 ",inv(scale))
                _vf_spice_damping(io,u,alpha/scale)
                # Paired states use twice the input drive and undoubled
                # residues. This retains finite coefficients even when
                # doubling a valid real/imaginary residue would overflow.
                println(io,"Gin",u," ",u," 0 ",ports[j]," dm_ref ",
                    (ci[k]==0 ? -1 : -2)*(state_scale/scale))
                if ci[k]==0
                    for i in 1:n
                        gain = real(model.residues[k][i,j])/state_scale
                        iszero(gain) || println(io,"Gs",u,"_",i," ",ports[i],
                            " dm_ref ",u," 0 ",gain)
                    end
                else
                    state += 1
                    v = "dm_u$state"
                    beta = imag(model.poles[k])/scale
                    println(io,"C",v," ",v," 0 ",inv(scale))
                    _vf_spice_damping(io,v,alpha/scale)
                    println(io,"Gx",u,"_",v," ",u," 0 ",v," 0 ",beta)
                    println(io,"Gx",v,"_",u," ",v," 0 ",u," 0 ",-beta)
                    for i in 1:n
                        residue = model.residues[k][i,j]
                        for (node,gain) in ((u,real(residue)/state_scale),(v,-imag(residue)/state_scale))
                            iszero(gain) || println(io,"Gs",node,"_",i," ",ports[i],
                                " dm_ref ",node," 0 ",gain)
                        end
                    end
                end
            end
            k += ci[k]==0 ? 1 : 2
        end
        println(io,".ends ",subckt_name)
    end
    return path
end
