export PlanarPortImpedance, planar_power_waves, planar_wave_voltages, planar_s_to_y

# Excitations are stored before voltage/current transfers can amplify them.
# Reject conversion loss rather than silently changing a nonzero drive to zero.
@inline function _planar_stored_phasor(value::Number)
    stored=ComplexF64(value)
    isfinite(stored) && (iszero(real(value)) || !iszero(real(stored))) &&
        (iszero(imag(value)) || !iszero(imag(stored))) || throw(ArgumentError(
            "phasors must fit finite ComplexF64 values and preserve nonzero components"))
    return stored
end

function _planar_stored_phasors(values::AbstractVector)
    stored=Vector{ComplexF64}(undef,length(values))
    for (i,value) in enumerate(values)
        stored[i]=_planar_stored_phasor(value)
    end
    return stored
end

function _planar_reference_number(value)
    value isa Number && isfinite(value) && real(value)>0 || throw(ArgumentError(
        "power-wave reference impedances must be finite with positive real part"))
    stored=ComplexF64(value)
    isfinite(stored) && real(stored)>0 || throw(ArgumentError(
        "power-wave reference impedances must remain finite with positive real part in ComplexF64"))
    return stored
end

function _planar_reference_values(z0,n::Integer;freq=nothing)
    n>0 || throw(ArgumentError("power waves require at least one port"))
    if z0 isa Number
        return fill(_planar_reference_number(z0),n)
    elseif z0 isa AbstractVector || z0 isa Tuple
        length(z0)==n || throw(DimensionMismatch("reference impedances must match the port count"))
        return ComplexF64[_planar_reference_number(z isa Number ? z :
            _planar_reference_provider_value(z,freq)) for z in z0]
    else
        evaluated=_planar_reference_provider_value(z0,freq)
        evaluated isa Number || evaluated isa AbstractVector || evaluated isa Tuple ||
            throw(ArgumentError("reference provider must return a number or per-port vector"))
        return _planar_reference_values(evaluated,n;freq)
    end
end

function _planar_reference_provider_value(provider,freq)
    freq===nothing && throw(ArgumentError("frequency-dependent references require freq"))
    freq isa Real && isfinite(freq) && freq>=0 || throw(ArgumentError(
        "reference provider frequency must be finite and nonnegative"))
    applicable(provider,freq) || throw(ArgumentError("reference provider must accept frequency in Hz"))
    return provider(freq)
end

function _planar_store_references(z0,n::Integer)
    if z0 isa Number || ((z0 isa AbstractVector || z0 isa Tuple) && all(z->z isa Number,z0))
        return _planar_reference_values(z0,n)
    elseif z0 isa AbstractVector || z0 isa Tuple
        length(z0)==n || throw(DimensionMismatch("reference impedances must match the port count"))
        for z in z0
            z isa Number ? _planar_reference_number(z) : applicable(z,1.) ||
                throw(ArgumentError("reference providers must accept frequency in Hz"))
        end
        return collect(z0)
    end
    applicable(z0,1.) || throw(ArgumentError("reference providers must accept frequency in Hz"))
    return z0
end

function _planar_store_reference(z0)
    z0 isa Number && return _planar_reference_number(z0)
    applicable(z0,1.) || throw(ArgumentError("reference provider must accept frequency in Hz"))
    return z0
end

"""Reference impedance with SI units. `topology=:series` gives
`R+jX+jωL+1/(jωC)`. `topology=:parallel` places the capacitor across
`R+jX+jωL`, matching the native Sonnet port circuit.
Each keyword may be a real number or a frequency [Hz] provider. `r` is
positive; `x,l,c` may have either sign because they define a reference,
rather than a physical passive load. `c=0`
omits the capacitor. A nonzero series capacitor at DC has an infinite
reference for the series topology and rejects explicitly. Call the object with frequency to
evaluate its impedance; use it directly as a port's `z0` provider."""
struct PlanarPortImpedance{R,X,L,C}
    r::R
    x::X
    l::L
    c::C
    topology::Symbol
end

function _planar_reference_parameter(value,label;positive=false,nonnegative=false)
    value isa Real && isfinite(value) && (!positive || value>0) &&
        (!nonnegative || value>=0) || throw(ArgumentError("invalid reference $label"))
    stored=Float64(value)
    isfinite(stored) && (!positive || stored>0) && (iszero(value) || !iszero(stored)) || throw(ArgumentError(
        "reference $label must remain finite in Float64"))
    return stored
end

function PlanarPortImpedance(;r=50.,x=0.,l=0.,c=0.,topology::Symbol=:series)
    topology in (:series,:parallel) || throw(ArgumentError("reference topology must be :series or :parallel"))
    for (value,label,positive,nonnegative) in ((r,"resistance",true,false),
            (x,"reactance",false,false),(l,"inductance",false,false),(c,"capacitance",false,false))
        value isa Number ? _planar_reference_parameter(value,label;positive,nonnegative) :
            applicable(value,1.) || throw(ArgumentError("reference $label provider must accept frequency in Hz"))
    end
    return PlanarPortImpedance(r,x,l,c,topology)
end
PlanarPortImpedance(r,x,l,c)=PlanarPortImpedance(;r,x,l,c)

function _planar_reference_omega_product(f,value)
    (iszero(f) || iszero(value)) && return 0.
    fm,fe=frexp(f);vm,ve=frexp(value)
    return ldexp((2pi*fm)*vm,fe+ve)
end

function (z::PlanarPortImpedance)(freq::Real)
    isfinite(freq) && freq>=0 || throw(ArgumentError("reference frequency must be finite and nonnegative"))
    freq=_planar_reference_parameter(freq,"frequency")
    evaluate(value)=value isa Number ? value : value(freq)
    r=_planar_reference_parameter(evaluate(z.r),"resistance";positive=true)
    x=_planar_reference_parameter(evaluate(z.x),"reactance")
    l=_planar_reference_parameter(evaluate(z.l),"inductance")
    c=_planar_reference_parameter(evaluate(z.c),"capacitance")
    z.topology in (:series,:parallel) || throw(ArgumentError("invalid reference topology"))
    z.topology===:series && iszero(freq) && !iszero(c) && throw(ArgumentError(
        "nonzero series capacitance has an infinite power-wave reference at DC"))
    base=complex(r,x+_planar_reference_omega_product(freq,l))
    return _planar_reference_number(iszero(c) || (z.topology===:parallel && iszero(freq)) ? base : z.topology===:series ?
        base+inv(im*_planar_reference_omega_product(freq,c)) :
        inv(inv(base)+im*_planar_reference_omega_product(freq,c)))
end

function _planar_wave_voltage(S::AbstractMatrix,a::AbstractVector,refs)
    n=length(a)
    size(S)==(n,n) && all(isfinite,S) && all(isfinite,a) || throw(ArgumentError(
        "power-wave voltage needs finite matching S and incident waves"))
    z=_planar_reference_values(refs,n);r=_planar_reference_roots(z)
    matrix=eltype(S)===ComplexF64 ? S : Matrix{ComplexF64}(S)
    incident=_planar_stored_phasors(a)
    all(isfinite,matrix) && all(isfinite,incident) || throw(ArgumentError("power-wave data must fit finite ComplexF64 values"))
    reflected=matrix*incident;value=Vector{ComplexF64}(undef,n)
    for p in 1:n
        value[p]=iszero(imag(z[p])) ? r[p]*(incident[p]+reflected[p]) :
            (conj(z[p])*incident[p]+z[p]*reflected[p])/r[p]
    end
    all(isfinite,value) || throw(ArgumentError("power-wave voltages are nonfinite"))
    return value
end

# Physical terminal excitation from retained admittance. S can round to an
# exact short while the admittance still carries a finite voltage/current.
function _planar_wave_voltage_admittance(Y::AbstractMatrix,a::AbstractVector,refs)
    n=length(a)
    n>0 && size(Y)==(n,n) && all(isfinite,Y) && all(isfinite,a) ||
        throw(ArgumentError("admittance excitation needs finite matching ports and incident waves"))
    z=if refs isa AbstractVector{ComplexF64}
        length(refs)==n || throw(DimensionMismatch("reference impedances must match the port count"))
        all(ref->isfinite(ref) && real(ref)>0,refs) ||
            throw(ArgumentError("power-wave references must be finite with positive real part"))
        refs
    else
        _planar_reference_values(refs,n)
    end
    r=_planar_reference_roots(z)
    # Stored ComplexF64 incident waves are read-only here; the solution
    # has its own buffer. Wider inputs still use checked conversion.
    incident=a isa AbstractVector{ComplexF64} ? a : _planar_stored_phasors(a)
    K=Matrix{ComplexF64}(undef,n,n);value=Vector{ComplexF64}(undef,n)
    for p in 1:n
        scale=0.0
        for q in 1:n
            entry=z[p]*Y[p,q]+(p==q)
            K[p,q]=entry
            isfinite(K[p,q]) || throw(ArgumentError("admittance voltage coefficients exceed finite ComplexF64 values"))
            scale=max(scale,abs(real(K[p,q])),abs(imag(K[p,q])))
        end
        iszero(scale) && throw(ArgumentError("admittance voltage system is singular"))
        for q in 1:n;K[p,q]/=scale;end
        # Scale the physical RHS with binary exponents before rounding.
        # Dividing first could erase a small RHS that a large wave amplifies.
        rm,re=frexp(r[p]);sm,se=frexp(scale)
        am,ae=frexp(real(incident[p]));bm,be=frexp(imag(incident[p]))
        value[p]=complex(ldexp(2rm*am/sm,re+ae-se),ldexp(2rm*bm/sm,re+be-se))
    end
    all(isfinite,value) || throw(ArgumentError("admittance incident RHS is nonfinite"))
    factor=lu!(K;check=false)
    issuccess(factor) || throw(ArgumentError("admittance voltage system is singular"))
    ldiv!(factor,value)
    all(isfinite,value) || throw(ArgumentError("admittance terminal voltages are nonfinite"))
    return value
end

# Positive magnitude arithmetic rounds outward by one adjacent stored
# value, so a computed magnitude cannot underestimate its exact inputs.
@inline _planar_wave_upper_add(a,b)=iszero(a) && iszero(b) ? 0.0 : nextfloat(a+b)
@inline _planar_wave_upper_mul(a,b)=iszero(a) || iszero(b) ? 0.0 : nextfloat(a*b)
@inline _planar_wave_magnitude(z)=_planar_wave_upper_add(abs(real(z)),abs(imag(z)))

# Existing Kurokawa incident-wave accuracy requirement, independently
# enforced in test_planar_power_waves.jl. This is an integration acceptance
# target, not a universal forward-error bound for conditioned voltage solves.
const _PLANAR_POWER_WAVE_RTOL=3e-14

function _planar_wave_boundary_consistent(Y,v,a,refs)
    n=length(a)
    current=Y*v
    # A complex dot coordinate has a gamma_(2n) bound. The subsequent
    # complex product contributes two rounding steps, voltage addition one,
    # square root one, RHS product one, and residual subtraction one.
    # gamma_k=k*u/(1-k*u), with IEEE nearest-rounding unit u=eps/2.
    operations=2n+2+1+1+1+1
    unit=eps(Float64)/2
    factor=operations*unit
    factor<1 || return false
    gamma=nextfloat(factor/(1-factor))
    tiny=nextfloat(0.0)
    for p in 1:n
        lhs=v[p]+refs[p]*current[p]
        rhs=2sqrt(real(refs[p]))*a[p]
        all(isfinite,(lhs,rhs)) || return false
        terms=0.0
        for q in 1:n
            terms=_planar_wave_upper_add(terms,
                _planar_wave_upper_mul(_planar_wave_magnitude(Y[p,q]),_planar_wave_magnitude(v[q])))
        end
        zsize=_planar_wave_magnitude(refs[p])
        magnitude=_planar_wave_upper_add(_planar_wave_upper_add(_planar_wave_magnitude(v[p]),
            _planar_wave_upper_mul(zsize,terms)),_planar_wave_magnitude(rhs))
        # Each dot-coordinate underflow has at most one least-subnormal
        # storage unit; its error is amplified by the following reference
        # product. Account for the remaining scalar operation units too.
        floor=_planar_wave_upper_add(_planar_wave_upper_mul(zsize,
            _planar_wave_upper_mul(2n,tiny)),_planar_wave_upper_mul(2+1+1+1+1,tiny))
        bound=_planar_wave_upper_add(_planar_wave_upper_mul(gamma,magnitude),floor)
        # A stored scattering response already satisfies this physical
        # incident-wave accuracy requirement. A roundoff-only screen can
        # otherwise replace ordinary S arithmetic by a few last bits and
        # amplify coefficient cancellation in its radiation contraction.
        # Preserve the existing operation/underflow bound for zero drives.
        acceptance=_planar_wave_upper_mul(_PLANAR_POWER_WAVE_RTOL,abs(rhs))
        bound=max(bound,acceptance)
        isfinite(bound) && abs(lhs-rhs)<=bound || return false
    end
    return true
end

# Preserve ordinary declared-S arithmetic when its terminal equation is
# consistent at the operation-derived rounding bound. Test consistency
# before forming a retained-Y system: normalized S can be representable
# even when an unnormalized coefficient would overflow. Zero and nonzero
# S quantization both recover through Y when this equation is inconsistent.
function _planar_wave_voltage_retained(Y::AbstractMatrix,S::AbstractMatrix,
        a::AbstractVector,refs)
    declared=_planar_wave_voltage(S,a,refs)
    _planar_wave_boundary_consistent(Y,declared,a,refs) && return declared
    return _planar_wave_voltage_admittance(Y,a,refs)
end

function _planar_incident_admittance(Y::AbstractMatrix,v::AbstractVector,refs)
    n=length(v)
    n>0 && size(Y)==(n,n) && all(isfinite,Y) && all(isfinite,v) ||
        throw(ArgumentError("admittance excitation needs finite matching ports and terminal voltages"))
    z=if refs isa AbstractVector{ComplexF64}
        length(refs)==n || throw(DimensionMismatch("reference impedances must match the port count"))
        all(ref->isfinite(ref) && real(ref)>0,refs) ||
            throw(ArgumentError("power-wave references must be finite with positive real part"))
        refs
    else
        _planar_reference_values(refs,n)
    end
    r=_planar_reference_roots(z)
    voltage=_planar_stored_phasors(v);current=Y*voltage
    all(isfinite,current) || throw(ArgumentError("admittance terminal currents are nonfinite"))
    for p in 1:n
        voltage[p]=(voltage[p]+z[p]*current[p])/(2r[p])
    end
    all(isfinite,voltage) || throw(ArgumentError("admittance incident waves are nonfinite"))
    return voltage
end

function _planar_wave_voltage_matrix(S::AbstractMatrix,refs)
    n=size(S,1)
    n>0 && size(S,2)==n && all(isfinite,S) || throw(ArgumentError(
        "power-wave voltage transfer needs a finite nonempty square S matrix"))
    z=_planar_reference_values(refs,n);r=_planar_reference_roots(z)
    result=Matrix{ComplexF64}(undef,n,n)
    for q in 1:n,p in 1:n
        result[p,q]=iszero(imag(z[p])) ? r[p]*(S[p,q]+(p==q)) :
            (z[p]*S[p,q]+(p==q ? conj(z[p]) : 0))/r[p]
    end
    all(isfinite,result) || throw(ArgumentError("power-wave voltage transfer is nonfinite"))
    return result
end

function _planar_incident_from_voltage(S::AbstractMatrix,v::AbstractVector,refs)
    length(v)==size(S,1) && all(isfinite,v) || throw(ArgumentError(
        "terminal voltages must be finite and match S ports"))
    factor=lu!(_planar_wave_voltage_matrix(S,refs);check=false)
    issuccess(factor) || throw(ArgumentError("terminal voltages do not determine independent incident waves"))
    result=factor\_planar_stored_phasor.(v)
    all(isfinite,result) || throw(ArgumentError("incident waves are nonfinite"))
    return result
end

"""Kurokawa incident/reflected power waves for physical peak-phasor
terminal `voltages,currents` directed into the network. Returns `a,b`
and `accepted_power=.5real(dot(voltages,currents))` [W]. References
may be complex with positive real part. `a=(V+Z I)/(2sqrt(real(Z)))`
and `b=(V-conj(Z) I)/(2sqrt(real(Z)))`; consequently accepted power
equals `.5(sum(abs2,a)-sum(abs2,b))`."""
function planar_power_waves(voltages::AbstractVector,currents::AbstractVector;z0=50.,
        freq=nothing,max_bytes::Integer=_default_max_dense_payload_bytes())
    n=length(voltages)
    length(currents)==n && n>0 && all(isfinite,voltages) && all(isfinite,currents) ||
        throw(ArgumentError("power-wave voltages and currents must be finite matching vectors"))
    _enforce_payload_limit(_checked_array_payload_bytes(ComplexF64,6,n),max_bytes,
        "power-wave conversion","max_bytes")
    z=_planar_reference_values(z0,n;freq);r=_planar_reference_roots(z)
    v=_planar_stored_phasors(voltages);i=_planar_stored_phasors(currents)
    all(isfinite,v) && all(isfinite,i) || throw(ArgumentError("power-wave phasors must fit finite ComplexF64 values"))
    a=Vector{ComplexF64}(undef,n);b=similar(a)
    for p in 1:n
        a[p]=(v[p]+z[p]*i[p])/(2r[p]);b[p]=(v[p]-conj(z[p])*i[p])/(2r[p])
    end
    all(isfinite,a) && all(isfinite,b) || throw(ArgumentError("power waves must fit finite ComplexF64 values"))
    power=.5real(dot(v,i));isfinite(power) || throw(ArgumentError("accepted power is nonfinite"))
    return (a,b,accepted_power=power)
end

"""Physical terminal voltages for Kurokawa incident waves `a` and
network scattering matrix `S`: `V=(conj(Z)*a+Z*S*a)/sqrt(real(Z))`.
The references are finite with positive real parts; real references
reduce to `sqrt(Z)*(a+S*a)`."""
function planar_wave_voltages(S::AbstractMatrix,a::AbstractVector;z0=50.,
        freq=nothing,max_bytes::Integer=_default_max_dense_payload_bytes())
    _enforce_payload_limit(_checked_payload_sum("power-wave voltage workspace",
        _checked_array_payload_bytes(ComplexF64,6,length(a)),
        eltype(S)===ComplexF64 ? 0 : _checked_array_payload_bytes(ComplexF64,size(S)...)),max_bytes,
        "power-wave voltage conversion","max_bytes")
    return _planar_wave_voltage(S,a,_planar_reference_values(z0,length(a);freq))
end

"""Convert Kurokawa scattering parameters to physical port admittance.
`z0` supplies finite per-port references with positive real parts, which
may be complex. Exact shorts/thrus can have singular terminal-voltage
maps and reject; use native S constitutive equations for those networks."""
function planar_s_to_y(S::AbstractMatrix,z0;
        freq=nothing,max_bytes::Integer=_default_max_dense_payload_bytes())
    n=size(S,1)
    n>0 && size(S,2)==n && all(isfinite,S) || throw(ArgumentError(
        "admittance conversion needs a finite nonempty square S matrix"))
    _enforce_payload_limit(_checked_payload_sum("power-wave admittance conversion",
        _checked_array_payload_bytes(ComplexF64,4,n,n),
        _checked_array_payload_bytes(ComplexF64,3,n)),max_bytes,
        "power-wave admittance conversion","max_bytes")
    refs=_planar_reference_values(z0,n;freq);roots=_planar_reference_roots(refs)
    voltage=_planar_wave_voltage_matrix(S,refs)
    factor=lu!(voltage;check=false)
    issuccess(factor) || throw(ArgumentError("S has no finite admittance representation"))
    current=Matrix{ComplexF64}(S)
    all(isfinite,current) || throw(ArgumentError("S must fit finite ComplexF64 values"))
    for q in 1:n,p in 1:n;current[p,q]=((p==q)-current[p,q])/roots[p];end
    result=Matrix{ComplexF64}(current/factor)
    all(isfinite,result) || throw(ArgumentError("S admittance conversion is nonfinite"))
    return result
end
