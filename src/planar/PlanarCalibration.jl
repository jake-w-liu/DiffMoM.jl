# Network-level SOC/TRL launch calibration. Reciprocal launch determinants
# and a known/equal reflect standard resolve eigenvector scaling; raw
# eigenvectors alone do not define calibrated power-wave amplitudes.

export PlanarCalibration, planar_s_to_t, planar_t_to_s
export planar_line_calibrate, planar_calibration_apply, planar_embed_2port
export embed_ports, deembed_cocal_group, planar_mixed_mode, planar_line_standards
export PlanarCalibratedResult, planar_reference_planes
export planar_cocalibrate
export PlanarDoubleDelayCalibration, planar_double_delay_calibrate
export planar_group_double_delay_calibrate, planar_y_to_abcd, planar_abcd_to_y

"""Voltage/current transfer matrix of a 2N-port admittance network,
ordered `[left conductors; right conductors]`. The state convention is
`[Vleft;Ileft]=T*[Vright;-Iright]`. Requires an invertible through block."""
function planar_y_to_abcd(Y::AbstractMatrix)
    n2=size(Y,1)
    n2>0 && iseven(n2) || throw(ArgumentError("transfer needs an even positive number of ports"))
    _check_square(Y,n2,"transfer conversion");_check_finite(Y,"transfer conversion")
    n=n2÷2;left=1:n;right=n+1:n2
    F=lu(Matrix{ComplexF64}(Y[right,left]);check=false)
    issuccess(F) || throw(ArgumentError("transfer through-admittance block is singular"))
    A=-(F\Y[right,right]);B=-(F\Matrix{ComplexF64}(I,n,n))
    return [A B;Y[left,right]+Y[left,left]*A Y[left,left]*B]
end

"""Inverse of [`planar_y_to_abcd`](@ref), requiring an invertible B
block. Ideal zero-length lines have no finite admittance representation."""
function planar_abcd_to_y(T::AbstractMatrix)
    n2=size(T,1)
    n2>0 && iseven(n2) || throw(ArgumentError("transfer needs an even positive dimension"))
    _check_square(T,n2,"transfer inverse");_check_finite(T,"transfer inverse")
    n=n2÷2;l=1:n;r=n+1:n2
    A,B,C,D=T[l,l],T[l,r],T[r,l],T[r,r]
    F=lu(Matrix{ComplexF64}(B);check=false)
    issuccess(F) || throw(ArgumentError("transfer B block is singular"))
    inverseB=F\Matrix{ComplexF64}(I,n,n)
    return [D*inverseB C-D*(F\A);-inverseB F\A]
end

"""Co-calibrated multiconductor double-delay launch extraction from two
2N-port line standards ordered `[left;right]`, of lengths `length` and
`2length`. Identical mirror-symmetric reciprocal launch groups may include
mutual series/shunt coupling. The near-identity matrix square root fixes
their branch. Reciprocal symplectic, end-reversal and uniform-line checks
reject ambiguous/inconsistent standards. Use the recovered `launch` with
[`deembed_cocal_group`](@ref) for each N-port end group."""
function planar_group_double_delay_calibrate(Y1::AbstractMatrix,Y2::AbstractMatrix;
        length::Real,tol::Real=1e-6,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    isfinite(length) && length>0 || throw(ArgumentError("standard length must be positive"))
    length=_circuit_stored_real(length,"calibration standard length")
    isfinite(tol) && tol>0 || throw(ArgumentError("calibration tolerance must be positive"))
    n2=size(Y1,1);n=n2÷2
    n2>0 && iseven(n2) || throw(ArgumentError("group calibration needs an even port count"))
    _check_square(Y2,n2,"group line standard")
    _enforce_payload_limit(_checked_array_payload_bytes(ComplexF64,32,n2,n2),
        max_bytes,"group calibration","max_bytes")
    # Normalize voltage/current units before logarithm and exponential.
    refs=vcat(fill(sqrt(50.0),n),fill(inv(sqrt(50.0)),n))
    normalize(T)=Diagonal(inv.(refs))*T*Diagonal(refs)
    T1,T2=normalize(planar_y_to_abcd(Y1)),normalize(planar_y_to_abcd(Y2))
    F2=lu(T2;check=false)
    issuccess(F2) || throw(ArgumentError("group line standard is singular"))
    E=exp(log(T1*(F2\T1))/2)
    FE=lu(E;check=false)
    issuccess(FE) || throw(ArgumentError("group launch is singular"))
    L=(FE\T1)/FE
    J=[zeros(n,n) Matrix{Float64}(I,n,n);-Matrix{Float64}(I,n,n) zeros(n,n)]
    P=Diagonal(vcat(ones(n),-ones(n)))
    resid=max(maximum(abs,E*L*E-T1),maximum(abs,E*L*L*E-T2))/
        max(maximum(abs,T1),maximum(abs,T2),1.0)
    for T in (E,L)
        resid=max(resid,maximum(abs,transpose(T)*J*T-J),
            maximum(abs,T*P*T-P))
    end
    H=log(L)
    resid=max(resid,maximum(abs,H[1:n,1:n])/max(maximum(abs,H),1.0),
        maximum(abs,H[n+1:n2,n+1:n2])/max(maximum(abs,H),1.0))
    isfinite(resid) && resid<=tol || throw(ArgumentError(
        "group standards do not identify mirror-symmetric reciprocal launches (residual $resid)"))
    denormalize(T)=Matrix{ComplexF64}(Diagonal(refs)*T*Diagonal(inv.(refs)))
    return PlanarDoubleDelayCalibration(denormalize(E),denormalize(L),Float64(length),Float64(resid))
end

"""General symmetric reciprocal launch recovered from two double-delay
standards. `launch` is a full ABCD error box, allowing series and shunt
terms; `line` is the calibrated length-`length` line. Unlike general TRL,
this method requires identical symmetric launches at both standard ends."""
struct PlanarDoubleDelayCalibration
    launch::Matrix{ComplexF64}
    line::Matrix{ComplexF64}
    length::Float64
    residual::Float64
end

"""Recover identical symmetric reciprocal launch boxes from admittance
standards of length `length` and `2length`. For `T1=E*L*E` and
`T2=E*L^2*E`, `E^2=T1*(T2\\T1)`. The principal matrix logarithm selects
the launch continuously connected to identity; strongly resonant launches
need independent reflect standards via [`planar_line_calibrate`](@ref).
Reconstruction, symmetry, reciprocity and uniform-line checks reject
inconsistent standards. The ABCD box may be passed to reference-plane or
subdivision APIs, and removes the full launch rather than just a shunt."""
function planar_double_delay_calibrate(Y1::AbstractMatrix,Y2::AbstractMatrix;
        length::Real,tol::Real=1e-6)
    isfinite(length) && length>0 || throw(ArgumentError("standard length must be finite and positive"))
    length=_circuit_stored_real(length,"calibration standard length")
    isfinite(tol) && tol>0 || throw(ArgumentError("calibration tolerance must be positive"))
    T1,T2 = _abcd_of_y(Y1),_abcd_of_y(Y2)
    factor = lu(T2;check=false)
    issuccess(factor) || throw(ArgumentError("double-delay standard is singular"))
    E = Matrix{ComplexF64}(exp(log(T1*(factor\T1))/2))
    F = lu(E;check=false)
    issuccess(F) || throw(ArgumentError("double-delay launch is singular"))
    L = Matrix{ComplexF64}((F\T1)/F)
    scale = max(maximum(abs,T1),maximum(abs,T2),1.0)
    residual = max(maximum(abs,E*L*E-T1),maximum(abs,E*(L*L)*E-T2))/scale
    for T in (E,L)
        residual = max(residual,abs(T[1,1]-T[2,2])/max(maximum(abs,T),1.0),
            abs(LinearAlgebra.det(T)-1))
    end
    isfinite(residual) && residual<=tol || throw(ArgumentError(
        "double-delay standards do not identify symmetric reciprocal launches (residual $residual)"))
    _line_params_of_abcd(L) # A=D, det=1 and finite modal reconstruction
    return PlanarDoubleDelayCalibration(E,L,Float64(length),Float64(residual))
end

"""Scattering-coordinate transform `[b1;a1]=T*[a2;b2]`. Multiplying
these transforms represents physical cascades when connected travelling
waves agree, as with equal real references. Complex Kurokawa references
require physical constitutive connections; use [`planar_embed_2port`](@ref)."""
function planar_s_to_t(S::AbstractMatrix)
    _check_square(S,2,"scattering chain")
    _check_finite(S,"scattering chain")
    iszero(S[2,1]) && throw(ArgumentError("S21 vanishes; no cascade through path"))
    return ComplexF64[(S[1,2]*S[2,1]-S[1,1]*S[2,2])/S[2,1] S[1,1]/S[2,1];
        -S[2,2]/S[2,1] inv(S[2,1])]
end

"""Raw planar solution together with de-embedded port data and the
voltage transfer back to its raw gap planes. `y,s` live at the requested
reference planes; `raw` retains the original matrix/operator and currents.
Current reconstruction maps calibrated excitations back to raw voltages.
`groups` associates each stored launch in `chains` with source-port indices;
individual reference planes use singleton groups. `z0,freq` retain the
calibrated network's wave references and solved frequency."""
struct PlanarCalibratedResult{R}
    raw::R
    chains::Vector{Matrix{ComplexF64}}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    gap_voltage_transfer::Matrix{ComplexF64}
    groups::Vector{Vector{Int}}
    z0::Vector{ComplexF64}
    freq::ComplexF64
end

function PlanarCalibratedResult(raw,chains,y,s,transfer)
    n=size(y,1)
    refs=hasproperty(raw,:z0) ? raw.z0 : [p.z0 for p in raw.problem.ports]
    return PlanarCalibratedResult(raw,chains,y,s,transfer,[[i] for i in 1:n],
        _circuit_z0(refs,n;freq=real(raw.freq)),ComplexF64(raw.freq))
end

function _calibration_group_blocks(chains,groups,n)
    length(chains)==length(groups)>0 || throw(ArgumentError(
        "coupled calibration needs one launch chain per nonempty group"))
    used=Int[]
    A,D=Matrix{ComplexF64}(I,n,n),Matrix{ComplexF64}(I,n,n)
    B,C=zeros(ComplexF64,n,n),zeros(ComplexF64,n,n)
    for (ports,chain) in zip(groups,chains)
        k=length(ports)
        k>0 && all(p->p isa Integer && 1<=p<=n,ports) &&
            length(unique(ports))==k && !any(p->p in used,ports) || throw(ArgumentError(
                "calibrated groups must be nonempty, disjoint and contain valid port indices"))
        _check_square(chain,2k,"coupled launch chain");_check_finite(chain,"coupled launch chain")
        append!(used,ports)
        A[ports,ports],B[ports,ports],C[ports,ports],D[ports,ports]=
            chain[1:k,1:k],chain[1:k,k+1:2k],chain[k+1:2k,1:k],chain[k+1:2k,k+1:2k]
    end
    return A,B,C,D
end

"""Remove full coupled launch error boxes from a solved physical network.
`groups` lists disjoint source-port groups and `launches` contains their
`2k×2k` ABCD matrices, using `[Vraw;Iraw]=[A B;C D]*[Vcal;Ical]`.
Unlisted ports pass through. Mutual launch terms and coupling to every
other group are retained. The returned [`PlanarCalibratedResult`](@ref)
keeps the raw solve and the full `Vraw=(A+B*Ycal)*Vcal` transfer, so
current maps and radiation reconstruct the actual physical fields.
Existing calibrated launches compose into one global chain without
changing their retained raw solve. `max_bytes` limits the new wrapper's
workspace; reserve the raw result separately when budgeting a workflow."""
function planar_cocalibrate(raw,groups::AbstractVector,launches::AbstractVector;
        max_bytes::Integer=_default_max_dense_payload_bytes())
    hasproperty(raw,:y) && raw.y isa AbstractMatrix && hasproperty(raw,:s) &&
        hasproperty(raw,:freq) && (hasproperty(raw,:currents) || raw isa PlanarCalibratedResult) ||
        throw(ArgumentError("coupled calibration requires a solved physical admittance source network"))
    n=size(raw.y,1);_check_square(raw.y,n,"coupled calibration");_check_finite(raw.y,"coupled calibration")
    _enforce_payload_limit(_checked_payload_sum("coupled calibrated result",
        _checked_array_payload_bytes(ComplexF64,36,n,n),
        _checked_array_payload_bytes(Int,4,n),_checked_array_payload_bytes(ComplexF64,n)),
        max_bytes,"coupled calibrated result","max_bytes")
    A,B,C,D=_calibration_group_blocks(launches,groups,n)
    factor=lu(raw.y*B-D;check=false)
    issuccess(factor) || throw(ArgumentError("coupled calibrated group removal is singular"))
    y=Matrix{ComplexF64}(factor\(C-raw.y*A));_check_finite(y,"coupled calibrated result")
    transfer=Matrix{ComplexF64}(A+B*y);_check_finite(transfer,"coupled gap voltage transfer")
    refs=hasproperty(raw,:z0) ? raw.z0 : [p.z0 for p in raw.problem.ports]
    z0=_circuit_z0(refs,n;freq=real(raw.freq));s=planar_y_to_s(y,z0)
    chains=Matrix{ComplexF64}.(launches);storedgroups=[Int.(group) for group in groups]
    base=raw
    if raw isa PlanarCalibratedResult
        a,b,c,d=_calibration_group_blocks(raw.chains,raw.groups,n)
        chains=[[a b;c d]*[A B;C D]];storedgroups=[collect(1:n)]
        transfer=raw.gap_voltage_transfer*transfer;base=raw.raw
    end
    return PlanarCalibratedResult(base,chains,y,s,transfer,storedgroups,z0,ComplexF64(raw.freq))
end

"""Evaluate each port's physical reference-plane contract at the solved
frequency and return a calibrated wrapper. Explicit `chains` can instead
supply per-port ABCD launch boxes. The raw result and its currents retain
their original gap-plane meaning."""
function planar_reference_planes(raw::Union{PlanarResult,PlanarUFFTResult};kw...)
    return _planar_reference_planes(raw;kw...)
end

function _planar_reference_planes(raw;chains=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    n = length(raw.problem.ports)
    _enforce_payload_limit(_checked_payload_sum("calibrated reference planes",
        _checked_array_payload_bytes(ComplexF64,18,n,n),
        _checked_array_payload_bytes(ComplexF64,4,n),
        _checked_array_payload_bytes(Float64,2,n),
        _checked_array_payload_bytes(Int,n)),max_bytes,"reference-plane calibration","max_bytes")
    supplied = if chains===nothing
        [port.refplane===nothing ? Matrix{ComplexF64}(I,2,2) :
            planar_line_abcd(_circuit_value(port.refplane.zc,real(raw.freq)),
                _circuit_value(port.refplane.gamma,real(raw.freq))*port.refplane.length)
            for port in raw.problem.ports]
    else
        Matrix{ComplexF64}.(chains)
    end
    Y = deembed_ports(raw.y,supplied)
    A,B,C,D = _calibration_chain_blocks(supplied,n)
    transfer = A+B*Y
    refs=hasproperty(raw,:z0) ? raw.z0 :
        _planar_reference_values([port.z0 for port in raw.problem.ports],n;freq=real(raw.freq))
    S = planar_y_to_s(Y,refs)
    return PlanarCalibratedResult(raw,supplied,Y,S,transfer)
end

"""Reconstruct currents for excitations specified at calibrated planes.
`voltages` denotes calibrated terminal voltages; `incident_waves` denotes
power waves at those planes. The resulting gap voltage is transferred
through the calibrated launch before using the retained raw solution."""
function planar_current_maps(result::PlanarCalibratedResult;port::Integer=1,
        voltages=nothing,incident_waves=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    n = size(result.y,1)
    reserve=_checked_array_payload_bytes(ComplexF64,4,n)
    _enforce_payload_limit(reserve,max_bytes,"calibrated current excitation","max_bytes")
    voltages!==nothing && incident_waves!==nothing &&
        throw(ArgumentError("provide voltages or incident_waves"))
    voltage = if voltages!==nothing
        length(voltages)==n && all(isfinite,voltages) ||
            throw(ArgumentError("calibrated voltages must be finite and match ports"))
        _planar_stored_phasor.(voltages)
    elseif incident_waves!==nothing
        length(incident_waves)==n && all(isfinite,incident_waves) ||
            throw(ArgumentError("calibrated incident waves must be finite and match ports"))
        _planar_wave_voltage(result.s,incident_waves,result.z0)
    else
        1<=port<=n || throw(ArgumentError("calibrated port index is invalid"))
        v = zeros(ComplexF64,n);v[port]=1;v
    end
    return planar_current_maps(result.raw;
        voltages=result.gap_voltage_transfer*voltage,max_bytes=max_bytes-reserve,kw...)
end

"""Inverse of [`planar_s_to_t`](@ref)."""
function planar_t_to_s(T::AbstractMatrix)
    _check_square(T,2,"scattering chain inverse")
    _check_finite(T,"scattering chain inverse")
    iszero(T[2,2]) && throw(ArgumentError("T22 vanishes; S representation is singular"))
    return ComplexF64[T[1,2]/T[2,2] LinearAlgebra.det(T)/T[2,2];inv(T[2,2]) -T[2,1]/T[2,2]]
end

"""Embed a two-port DUT between left/right launch S matrices through
physical voltage/current connections. `z0,left_z0,right_z0` identify each
network's Kurokawa references and may differ or be complex. The output
references are the external left/right launch references. Exact thru,
short and open blocks retain their native constitutive equations."""
function planar_embed_2port(S,left,right;z0=50.,left_z0=z0,right_z0=z0,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    for (matrix,label) in ((S,"DUT"),(left,"left launch"),(right,"right launch"))
        _check_square(matrix,2,label);_check_finite(matrix,label)
    end
    _enforce_payload_limit(_checked_array_payload_bytes(ComplexF64,400),max_bytes,
        "two-port embedding","max_bytes")
    lrefs=_planar_reference_values(left_z0,2);rrefs=_planar_reference_values(right_z0,2)
    circuit=PlanarCircuit(4,[1,4];z0=[lrefs[1],rrefs[2]])
    circuit_add_network!(circuit,[1,2],left;z0=lrefs)
    circuit_add_network!(circuit,[2,3],S;z0)
    circuit_add_network!(circuit,[3,4],right;z0=rrefs)
    return solve_planar_circuit(circuit,0.;max_bytes).s
end

"""General reciprocal two-port launch calibration. `left` and `right`
are scattering error boxes in cascade order, using real working waves at
the measured planes and intrinsic line travelling waves internally.
These internal coordinates are not Kurokawa power waves for a complex
line impedance; use `line_impedance` during extraction for physical
calibrated output. `gamma` [1/m] and
`delta_length` [m] describe the line standard. `reflection` is the known
or recovered equal-reflect load. The reference basis is the line's
characteristic impedance; `residual` checks all measured standards."""
struct PlanarCalibration
    left::Matrix{ComplexF64}
    right::Matrix{ComplexF64}
    gamma::ComplexF64
    delta_length::Float64
    reflection::ComplexF64
    residual::Float64
    reference_basis::Symbol
    measurement_z0::Vector{ComplexF64}
    line_impedance::Union{Nothing,ComplexF64}
end
PlanarCalibration(l,r,g,d,re,e,b)=PlanarCalibration(l,r,g,d,re,e,b,fill(50.0+0im,2),nothing)

function _calibration_reflection(S,load)
    return S[1,1]+S[1,2]*S[2,1]*load/(1-S[2,2]*load)
end

"""`planar_line_calibrate(thru,line;delta_length,reflect_standard,
reflection=nothing,reflect_sign=1,phase_hint=nothing,tol=1e-6,z0=50,
line_impedance=nothing)` extracts
general reciprocal launches from thru/line and isolated reflect standards.
This permits both series and shunt discontinuities. Supply a known common
load `reflection`, or infer the common load from both measured reflections
and select the open/short sign using `reflect_sign=+1/-1`. Eigenvector
scaling uses the reciprocal determinant and the reflect equations.
Measured `z0` may be complex; input standards are transformed to real
working references before travelling-wave extraction. A known `reflection`
denotes the intrinsic line travelling-wave load, not an arbitrary measured
power-wave reflection. Independently known `line_impedance` enables
physical Kurokawa output through [`planar_calibration_apply`](@ref).

The conventional TRL output basis is the line's characteristic impedance.
Without independent line-impedance information this basis is intrinsically
ambiguous. Use a line shorter than half a wavelength in every calibration
step, or provide its electrical-length `phase_hint` [rad] for continuation."""
function planar_line_calibrate(thru::AbstractMatrix,line::AbstractMatrix;
        delta_length::Real,reflect_standard::AbstractMatrix,
        reflection::Union{Nothing,Number}=nothing,reflect_sign::Real=1,
        phase_hint::Union{Nothing,Real}=nothing,tol::Real=1e-6,z0=50.,
        line_impedance::Union{Nothing,Number}=nothing)
    isfinite(delta_length) && delta_length>0 || throw(ArgumentError("delta_length must be finite and positive"))
    delta_length=_circuit_stored_real(delta_length,"calibration delta_length")
    isfinite(tol) && tol>0 || throw(ArgumentError("calibration tolerance must be finite and positive"))
    reflect_sign in (-1,1) || throw(ArgumentError("reflect_sign must be +1 or -1"))
    reflection === nothing || (isfinite(reflection) && !iszero(reflection)) ||
        throw(ArgumentError("known reflection must be finite and nonzero"))
    for (S,label) in ((thru,"thru"),(line,"line"),(reflect_standard,"reflect"))
        _check_square(S,2,label);_check_finite(S,label)
    end
    refs=_planar_reference_values(z0,2)
    known_line=line_impedance===nothing ? nothing : _planar_reference_number(line_impedance)
    work=fill(50.,2)
    thru=planar_renormalize_s(thru,refs,work)
    line=planar_renormalize_s(line,refs,work)
    reflect_standard=planar_renormalize_s(reflect_standard,refs,work)
    max(abs(reflect_standard[1,2]),abs(reflect_standard[2,1])) <= tol ||
        throw(ArgumentError("reflect standard must have isolated equal loads"))
    for (S,label) in ((thru,"thru"),(line,"line"))
        abs(S[1,2]-S[2,1]) <= tol*max(abs(S[1,2]),abs(S[2,1]),1e-30) ||
            throw(ArgumentError("$label standard is not reciprocal"))
    end
    Tt,Tl = planar_s_to_t(thru),planar_s_to_t(line)
    eigenvalues,W = eigen(Tl/Tt)
    # Forward attenuation has |lambda|<1; lossless ties choose beta>0.
    forward = sortperm(1:2;by=k -> (round(abs(eigenvalues[k]);digits=10),
        imag(log(eigenvalues[k]))))[1]
    order = [forward,3-forward]
    W = W[:,order]
    Q = W\Tt
    gleft,gright = reflect_standard[1,1],reflect_standard[2,2]
    denominator = gleft*W[2,1]-W[1,1]
    iszero(denominator) && throw(ArgumentError("degenerate reflect calibration"))
    u = (W[1,2]-gleft*W[2,2])/denominator # d1/d2 * Gamma_r
    reflect_load = if reflection === nothing
        denominator2 = Q[1,1]+gright*Q[1,2]
        iszero(denominator2) && throw(ArgumentError("reflect load cannot be identified"))
        value = sqrt(ComplexF64(u*(Q[2,1]+gright*Q[2,2])/denominator2))
        real(value)*reflect_sign < 0 ? -value : value
    else
        ComplexF64(reflection)
    end
    isfinite(reflect_load) && !iszero(reflect_load) ||
        throw(ArgumentError("extracted equal-reflect load is degenerate"))
    ratio = u/reflect_load
    d2 = sqrt(inv(LinearAlgebra.det(W)*ratio))
    X = W*Diagonal(ComplexF64[ratio*d2,d2])
    Y = X\Tt
    left,right = planar_t_to_s(X),planar_t_to_s(Y)
    gam = -log(eigenvalues[forward])
    if phase_hint !== nothing
        isfinite(phase_hint) || throw(ArgumentError("phase_hint must be finite"))
        gam += 2pi*1im*round((phase_hint-imag(gam))/(2pi))
    end
    L = Diagonal(ComplexF64[exp(-gam),exp(gam)])
    reflected_right = right[[2,1],[2,1]]
    residual = max(maximum(abs,planar_t_to_s(X*Y)-thru),
        maximum(abs,planar_t_to_s(X*L*Y)-line),
        abs(_calibration_reflection(left,reflect_load)-gleft),
        abs(_calibration_reflection(reflected_right,reflect_load)-gright))
    isfinite(residual) && residual <= tol || throw(ArgumentError(
        "calibration standards are inconsistent (residual $residual > $tol)"))
    gamma=ComplexF64(gam/delta_length)
    isfinite(gamma) || throw(ArgumentError("calibrated propagation constant must remain finite in ComplexF64"))
    return PlanarCalibration(left,right,gamma,delta_length,
        reflect_load,Float64(residual),:line_impedance,refs,known_line)
end

"""Remove calibrated two-port launches. Input `z0` defaults to the
calibration's measured references. Without `line_impedance` at extraction,
the output retains intrinsic line travelling-wave coordinates and cannot
be renormalized to a physical reference. With known line impedance,
output is Kurokawa power-wave S at that impedance, or `output_z0`."""
function planar_calibration_apply(S::AbstractMatrix,cal::PlanarCalibration;
        z0=cal.measurement_z0,output_z0=nothing)
    X,Y = planar_s_to_t(cal.left),planar_s_to_t(cal.right)
    measured=planar_renormalize_s(S,z0,fill(50.,2))
    result=planar_t_to_s((X\planar_s_to_t(measured))/Y)
    if cal.line_impedance===nothing
        output_z0===nothing || throw(ArgumentError("physical output references require independently known line_impedance"))
        return result
    end
    z=cal.line_impedance
    power=Matrix{ComplexF64}(im*imag(z)/z*I+real(z)/z*result)
    return output_z0===nothing ? power : planar_renormalize_s(power,fill(z,2),output_z0)
end

function _calibration_chain_blocks(chains,n)
    length(chains)==n || throw(DimensionMismatch("one chain per port is required"))
    A,B,C,D = zeros(ComplexF64,n,n),zeros(ComplexF64,n,n),zeros(ComplexF64,n,n),zeros(ComplexF64,n,n)
    for p in 1:n
        _check_square(chains[p],2,"port chain");_check_finite(chains[p],"port chain")
        A[p,p],B[p,p],C[p,p],D[p,p] = chains[p][1,1],chains[p][1,2],chains[p][2,1],chains[p][2,2]
    end
    return A,B,C,D
end

function _planar_calibration_incident_transfer(A,B,C,D,Y,S,refs)
    z=_planar_reference_values(refs,size(Y,1));roots=_planar_reference_roots(z)
    result=Diagonal(inv.(2roots))*(A+B*Y+Diagonal(z)*(C+D*Y))*
        _planar_wave_voltage_matrix(S,z)
    _check_finite(result,"calibrated incident transfer")
    return result
end

"""Embed per-port ABCD launch chains into a port-admittance matrix."""
function embed_ports(Y::AbstractMatrix,chains::AbstractVector{<:AbstractMatrix})
    n = size(Y,1)
    _check_square(Y,n,"port embedding");_check_finite(Y,"port embedding")
    A,B,C,D = _calibration_chain_blocks(chains,n)
    result = (C+D*Y)/(A+B*Y)
    _check_finite(result,"port embedding result")
    return Matrix{ComplexF64}(result)
end

"""Remove a coupled calibration group. `chain` is a 2k×2k voltage/current
transfer matrix `[A B;C D]` for the selected `ports`; the remaining ports
pass through. Full blocks preserve coupling between group members, using
`Y_B=inv(Y_A*B-D)*(C-Y_A*A)` with full group error boxes."""
function deembed_cocal_group(Y::AbstractMatrix,ports::AbstractVector{<:Integer},chain::AbstractMatrix)
    n,k = size(Y,1),length(ports)
    _check_square(Y,n,"co-calibration");_check_finite(Y,"co-calibration")
    k>0 && length(unique(ports))==k && all(p -> 1<=p<=n,ports) ||
        throw(ArgumentError("calibration group must contain distinct valid port indices"))
    _check_square(chain,2k,"group chain");_check_finite(chain,"group chain")
    A,D = Matrix{ComplexF64}(I,n,n),Matrix{ComplexF64}(I,n,n)
    B,C = zeros(ComplexF64,n,n),zeros(ComplexF64,n,n)
    A[ports,ports],B[ports,ports],C[ports,ports],D[ports,ports] =
        chain[1:k,1:k],chain[1:k,k+1:2k],chain[k+1:2k,1:k],chain[k+1:2k,k+1:2k]
    factor = lu(Y*B-D;check=false)
    issuccess(factor) || throw(ArgumentError("co-calibrated group removal is singular"))
    result = factor\(C-Y*A)
    _check_finite(result,"co-calibrated result")
    return Matrix{ComplexF64}(result)
end

"""Convert an N-port S matrix to differential/common power waves for
disjoint `(positive,negative)` port pairs. Reference impedances must match
within each pair. Returns `s,z0,transform`: differential references are
2Z0, common references Z0/2, ordered all differential, all common, then
unpaired ports. The transform is orthogonal and preserves power/passivity."""
function planar_mixed_mode(S::AbstractMatrix,pairs;z0=50.0)
    n = size(S,1)
    _check_square(S,n,"mixed-mode conversion");_check_finite(S,"mixed-mode conversion")
    refs = _circuit_z0(z0,n)
    paired = Int[]
    for (p,m) in pairs
        1<=p<=n && 1<=m<=n && p!=m || throw(ArgumentError("invalid mixed-mode port pair"))
        refs[p]==refs[m] || throw(ArgumentError("paired reference impedances must agree"))
        append!(paired,(p,m))
    end
    length(unique(paired))==length(paired) || throw(ArgumentError("mixed-mode pairs overlap"))
    k = length(pairs)
    U = zeros(Float64,n,n)
    outrefs = ComplexF64[]
    for (j,(p,m)) in enumerate(pairs)
        U[j,p],U[j,m] = inv(sqrt(2)),-inv(sqrt(2))
        U[k+j,p],U[k+j,m] = inv(sqrt(2)),inv(sqrt(2))
        push!(outrefs,2refs[p])
    end
    append!(outrefs,[refs[p]/2 for (p,m) in pairs])
    remaining = setdiff(1:n,paired)
    for (j,p) in enumerate(remaining)
        U[2k+j,p]=1
        push!(outrefs,refs[p])
    end
    return (s=Matrix{ComplexF64}(U*S*transpose(U)),z0=outrefs,transform=U)
end

"""Generate `thru,line,reflect` sister `PlanarProblem`s with the same
stackup, transverse cells and launch widths. The input must have two
aligned opposite-wall ports on one sheet, with matching polarity and no
via/volume launch. The line extends the propagation box by `extra_cells`
at the same grid spacing. Reflect uses isolated geometric open stubs of
`reflect_cells` on each launch; their common reflection is inferred by
[`planar_line_calibrate`](@ref). Returns physical `delta_length` too.
These are ordinary solver inputs and reuse the caller's solve settings."""
function planar_line_standards(prob::PlanarProblem;
        extra_cells::Union{Nothing,Integer}=nothing,
        reflect_cells::Union{Nothing,Integer}=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    length(prob.ports)==2 || throw(ArgumentError("line standards require two ports"))
    p,q = prob.ports
    xaxis = Set((p.wall,q.wall))==Set((:west,:east))
    yaxis = Set((p.wall,q.wall))==Set((:south,:north))
    xaxis || yaxis || throw(ArgumentError("line standards require opposite box-wall ports"))
    p.level==q.level && p.cells==q.cells && p.polarity==q.polarity ||
        throw(ArgumentError("standard launch level, span and polarity must agree"))
    isempty(prob.vias) && isempty(prob.vols) || throw(ArgumentError(
        "automatic line standards require sheet launches; explicit standards are needed for via/volume launches"))
    original = xaxis ? prob.grid.nx : prob.grid.ny
    extra_cells = extra_cells===nothing ? original : extra_cells
    reflect_cells = reflect_cells===nothing ? max(1,original÷4) : reflect_cells
    extra_cells>=1 && reflect_cells>=1 || throw(ArgumentError("standard cell lengths must be positive"))
    2BigInt(reflect_cells)<original || throw(ArgumentError("reflect stubs must leave a positive isolation gap"))
    extended=BigInt(original)+extra_cells
    extended<=typemax(Int) || throw(ArgumentError("line standard cell count overflows Int"))
    transverse=xaxis ? prob.grid.ny : prob.grid.nx
    payload=_checked_payload_sum("line sister standards",
        _checked_array_payload_bytes(UInt8,3BigInt(original)+extra_cells,transverse),
        _checked_array_payload_bytes(UInt8,57,2*(3BigInt(original)+extra_cells)*transverse),
        _checked_array_payload_bytes(Int,4,3BigInt(original)+extra_cells+3transverse))
    _enforce_payload_limit(payload,max_bytes,"line sister standards","max_bytes")
    function standard(ncells,reflect)
        nx,ny = xaxis ? (ncells,prob.grid.ny) : (prob.grid.nx,ncells)
        a = xaxis ? ncells*prob.grid.dx : prob.grid.a
        b = xaxis ? prob.grid.b : ncells*prob.grid.dy
        stack = PlanarStackup(prob.stack.layers,prob.stack.bottom,prob.stack.top,a,b)
        grid = CellGrid(a,b,nx,ny;walls=prob.grid.walls)
        sheet = sheet_level(prob.sheets[p.level].interface,nx,ny)
        for transverse in p.cells,along in 1:ncells
            !reflect || along<=reflect_cells || along>ncells-reflect_cells || continue
            i,j = xaxis ? (along,transverse) : (transverse,along)
            sheet.mask[i,j]=true
        end
        if xaxis
            sheet.connect_west[p.cells].=true;sheet.connect_east[p.cells].=true
        else
            sheet.connect_south[p.cells].=true;sheet.connect_north[p.cells].=true
        end
        ports = [PlanarPort(1,port.wall,port.cells,port.z0;polarity=port.polarity)
            for port in prob.ports]
        return build_planar_problem(stack,grid,[sheet],ports)
    end
    return (thru=standard(original,false),line=standard(Int(extended),false),
        reflect=standard(original,true),delta_length=Float64(extra_cells)*
            (xaxis ? prob.grid.dx : prob.grid.dy))
end
