# Network databanks use Kurokawa power waves. Touchstone specification:
# https://ibis.org/touchstone_ver2.1/touchstone_ver2_1.pdf
export PlanarNetworkData, planar_read_touchstone, planar_write_touchstone
export planar_network_response, planar_renormalize_s

"""Network databank with increasing `frequencies` [Hz], Kurokawa power-wave
`s` matrices, references `z0` [Ω] with positive real parts, and unique names.
Optional `reference_series` stores each sample's references. Numeric or
frequency-dependent `z0` providers are evaluated at the sample frequencies.
Real fixed references retain their Float64 representation.
The constructor copies its inputs. Touchstone mixed-mode data is converted
to physical single-ended ports by [`planar_read_touchstone`](@ref)."""
struct PlanarNetworkData{T<:Number}
    frequencies::Vector{Float64}
    s::Vector{Matrix{ComplexF64}}
    z0::Vector{T}
    port_names::Vector{String}
    reference_series::Union{Nothing,Vector{Vector{T}}}
    function PlanarNetworkData(freqs::Vector{Float64},series::Vector{Matrix{ComplexF64}},
            refs::AbstractVector{<:Number},names::Vector{String},::Val{:owned};reference_series=nothing)
        n=_network_series_validate(freqs,series)
        base=_planar_reference_values(refs,n)
        evaluated=reference_series===nothing ? nothing :
            [_planar_reference_values(z,n;freq=f) for (f,z) in zip(freqs,reference_series)]
        reference_series===nothing || length(reference_series)==length(freqs) ||
            throw(DimensionMismatch("reference series must match frequency samples"))
        evaluated===nothing || first(evaluated)==base || throw(ArgumentError("base references must match the first sample"))
        T=all(isreal,base) && (evaluated===nothing || all(z->all(isreal,z),evaluated)) ? Float64 : ComplexF64
        stored=evaluated===nothing ? nothing : T===Float64 ? [real.(z) for z in evaluated] : evaluated
        return new{T}(freqs,series,T===Float64 ? real.(base) : base,names,stored)
    end
    function PlanarNetworkData(freqs::AbstractVector{<:Real},
            series::AbstractVector{<:AbstractMatrix}; z0=nothing, reference_series=nothing,port_names=nothing,
            max_bytes::Integer=_default_max_dense_payload_bytes())
        n = _network_series_validate(freqs, series)
        previous=-Inf
        for frequency in freqs
            stored=_circuit_stored_real(frequency,"network frequency")
            stored>previous || throw(ArgumentError("network frequencies collapse in Float64"))
            previous=stored
        end
        names = port_names === nothing ? ["p$i" for i in 1:n] : String.(port_names)
        length(names)==n && length(unique(names))==n && all(!isempty,names) ||
            throw(ArgumentError("network port names must be nonempty, unique and match ports"))
        bytes = _checked_payload_sum("network databank",
            _checked_array_payload_bytes(ComplexF64,length(freqs),n,n),
            _checked_array_payload_bytes(Float64,length(freqs)),
            _checked_array_payload_bytes(ComplexF64,3length(freqs)+3,n))
        _enforce_payload_limit(bytes,max_bytes,"network databank","max_bytes")
        stored_freqs=Float64.(freqs);stored_series=[Matrix{ComplexF64}(s) for s in series]
        _network_series_validate(stored_freqs,stored_series)
        reference_series===nothing || length(reference_series)==length(freqs) ||
            throw(DimensionMismatch("reference series must match frequency samples"))
        dynamic=z0!==nothing && !(z0 isa Number ||
            ((z0 isa AbstractVector || z0 isa Tuple) && all(z->z isa Number,z0)))
        evaluated=reference_series===nothing ? (dynamic ?
            [_planar_reference_values(z0,n;freq=f) for f in stored_freqs] : nothing) :
            [_planar_reference_values(z,n;freq=f) for (f,z) in zip(stored_freqs,reference_series)]
        refs=z0===nothing ? (evaluated===nothing ? fill(50.0+0im,n) : first(evaluated)) :
            dynamic && reference_series===nothing ? first(evaluated) :
            _planar_reference_values(z0,n;freq=first(stored_freqs))
        evaluated===nothing || first(evaluated)==refs || throw(ArgumentError("base references must match the first sample"))
        return PlanarNetworkData(stored_freqs,stored_series,refs,names,Val(:owned);reference_series=evaluated)
    end
end

function _network_series_validate(freqs,series)
    !isempty(freqs) && length(freqs)==length(series) || throw(DimensionMismatch(
        "network requires matching nonempty frequency and matrix arrays"))
    all(f -> isfinite(f) && f>=0,freqs) &&
        all(k -> freqs[k]>freqs[k-1],2:length(freqs)) || throw(ArgumentError(
            "network frequencies must be finite, nonnegative and strictly increasing"))
    n = size(first(series),1)
    n>0 || throw(ArgumentError("network needs at least one port"))
    for (k,s) in enumerate(series)
        size(s)==(n,n) || throw(DimensionMismatch("network matrix $k must be $n by $n"))
        all(isfinite,s) || throw(ArgumentError("network matrix $k must be finite"))
    end
    return n
end

"""Change Kurokawa power-wave references without converting through
Y or Z. Exact opens, shorts and thrus retain finite S matrices."""
function planar_renormalize_s(S::AbstractMatrix,old_z0,new_z0)
    n = size(S,1)
    n>0 && size(S,2)==n && all(isfinite,S) || throw(ArgumentError(
        "renormalization requires a finite nonempty square S matrix"))
    old,new = _planar_reference_values(old_z0,n),_planar_reference_values(new_z0,n)
    matrix=eltype(S)===ComplexF64 ? S : Matrix{ComplexF64}(S)
    all(isfinite,matrix) || throw(ArgumentError("S must remain finite in ComplexF64"))
    old==new && return copy(matrix)
    a=Vector{ComplexF64}(undef,n);b=similar(a);c=similar(a);d=similar(a)
    for p in 1:n
        left,right=sqrt(real(old[p])),sqrt(real(new[p]))
        u=(.5left)/right+(.5right)/left
        w=(.5left)/right-(.5right)/left
        # The unnormalized impedance product, sum and reactance difference
        # can overflow even when every wave-transform coefficient is finite.
        # Root ratios retain the real terms; binary exponents retain the
        # difference divided by 2*left*right without that product.
        difference=imag(new[p])-imag(old[p])
        if isfinite(difference)
            mantissa,exponent=frexp(difference)
        else
            scale=max(abs(imag(old[p])),abs(imag(new[p])))
            mantissa,exponent=frexp(scale)
            mantissa*=imag(new[p])/scale-imag(old[p])/scale
        end
        lm,le=frexp(left);rm,re=frexp(right)
        v=ldexp(mantissa/(2lm*rm),exponent-le-re)
        a[p]=complex(u,v);b[p]=complex(w,-v)
        c[p]=complex(w,v);d[p]=complex(u,-v)
        all(isfinite,(a[p],b[p],c[p],d[p])) || throw(ArgumentError(
            "wave-reference coefficients exceed finite Float64 range"))
    end
    result=(Diagonal(c)+Diagonal(d)*matrix)/(Diagonal(a)+Diagonal(b)*matrix)
    all(isfinite,result) || throw(ArgumentError("singular wave-reference conversion"))
    return Matrix{ComplexF64}(result)
end

function _network_reference_at(data::PlanarNetworkData,f::Real)
    data.reference_series===nothing && return data.z0
    isfinite(f) && first(data.frequencies)<=f<=last(data.frequencies) ||
        throw(ArgumentError("frequency lies outside the sampled reference series"))
    k=searchsortedlast(data.frequencies,f)
    (k==length(data.frequencies) || data.frequencies[k]==f) && return data.reference_series[k]
    t=(f-data.frequencies[k])/(data.frequencies[k+1]-data.frequencies[k])
    return (1-t)*data.reference_series[k]+t*data.reference_series[k+1]
end

# Re-express a terminated input reflection on a real impedance Smith chart.
# The other physical terminations remain as in the input databank.
function _network_impedance_reflection(s,z,r::Real)
    z==r && return s
    numerator=conj(z)+z*s;denominator=1-s
    value=(numerator-r*denominator)/(numerator+r*denominator)
    isfinite(value) || throw(ArgumentError("terminated input has a singular real-reference reflection"))
    return value
end
"""Evaluate a databank at `f` [Hz]. Exact knots return a copy; other points
use linear interpolation of complex S in a common reference basis.
Sampled references are linearly interpolated by default; each neighbouring
matrix is renormalized to that basis before interpolation. Extrapolation
rejects. Optional `z0` supplies the output references (including providers).
Interpolation is a data model, not an EM solve."""
function planar_network_response(data::PlanarNetworkData,f::Real;z0=nothing)
    f=_circuit_stored_real(f,"network response frequency")
    isfinite(f) && first(data.frequencies)<=f<=last(data.frequencies) ||
        throw(ArgumentError("frequency lies outside the network databank"))
    k = searchsortedlast(data.frequencies,f)
    refs=z0===nothing ? _network_reference_at(data,f) : _planar_reference_values(z0,length(data.z0);freq=f)
    sample_ref(i)=data.reference_series===nothing ? data.z0 : data.reference_series[i]
    sample(i)=sample_ref(i)==refs ? copy(data.s[i]) : planar_renormalize_s(data.s[i],sample_ref(i),refs)
    if data.frequencies[k]==f || k==length(data.frequencies)
        S = sample(k)
    else
        t = (f-data.frequencies[k])/(data.frequencies[k+1]-data.frequencies[k])
        S = (1-t)*sample(k)+t*sample(k+1)
    end
    return S
end

_network_float(text) = parse(Float64,replace(text,'D'=>'E','d'=>'e'))

function _touchstone_numeric_bytes(owned,count)
    _checked_array_payload_bytes(Float64,_checked_payload_sum("Touchstone numeric input",owned,count))
end

function _touchstone_append_numbers!(target,text,other_owned,max_bytes;termination=false)
    count=0;continued=false
    for token in eachsplit(text)
        continued && throw(ArgumentError("Sonnet termination continuation marker must end its line"))
        if termination && token=="&"
            continued=true
        else
            count<typemax(Int) || throw(ArgumentError("Touchstone numeric field count overflows Int"))
            count+=1
        end
    end
    termination && count==0 && throw(ArgumentError("empty Sonnet termination record"))
    owned=_checked_payload_sum("Touchstone numeric input",other_owned,length(target))
    _enforce_payload_limit(_touchstone_numeric_bytes(owned,count),max_bytes,
        "Touchstone numeric input","max_bytes")
    start=length(target);resize!(target,start+count)
    for (k,token) in enumerate(eachsplit(text))
        termination && token=="&" && break
        value=_network_float(token)
        isfinite(value) || throw(ArgumentError("nonfinite Touchstone numeric data"))
        target[start+k]=value
    end
    return continued
end

function _touchstone_termination_values(text;max_bytes=_default_max_dense_payload_bytes(),other_owned=0)
    values=Float64[]
    continued=_touchstone_append_numbers!(values,text,other_owned,max_bytes;termination=true)
    return values,continued
end

function _touchstone_minimum_workspace(n,max_bytes)
    n>0 || return
    # Every accepted sample produces a dense matrix even for triangular
    # input, and the reader's conversion workspace reserves eight more.
    _enforce_payload_limit(_checked_array_payload_bytes(ComplexF64,9,n,n),
        max_bytes,"Touchstone minimum workspace","max_bytes")
end

_touchstone_tryfloat(text)=tryparse(Float64,replace(text,'D'=>'E','d'=>'e'))
function _touchstone_options(text;max_bytes=_default_max_dense_payload_bytes(),other_owned=0)
    # No array of tokens or uppercase copy of an arbitrarily long line.
    count=0;references=false
    for token in eachsplit(text)
        if token in ("R","r")
            references=true
        elseif references
            # Known option keywords end the numeric reference group. Count
            # first; do not parse or copy each numeric token before preflight.
            keyword=isletter(first(token)) ? uppercase(token) : ""
            if keyword in ("HZ","KHZ","MHZ","GHZ","S","Y","Z","H","G","RI","MA","DB")
                references=false
            else
                count<typemax(Int)-1 || throw(ArgumentError("Touchstone option reference count overflows Int"))
                count+=1
            end
        end
    end
    _enforce_payload_limit(_touchstone_numeric_bytes(other_owned,count+1),max_bytes,
        "Touchstone option references","max_bytes")
    scale,kind,format,refs = 1e9,"S","MA",[50.]
    tokens=eachsplit(text);state=iterate(tokens)
    while state!==nothing
        token,nextstate=state;t=uppercase(token)
        if t in ("HZ","KHZ","MHZ","GHZ")
            scale=t=="HZ" ? 1. : t=="KHZ" ? 1e3 : t=="MHZ" ? 1e6 : 1e9
        elseif t in ("S","Y","Z","H","G")
            kind=t
        elseif t in ("RI","MA","DB")
            format=t
        elseif t=="R"
            refs=Float64[];state=iterate(tokens,nextstate)
            while state!==nothing
                value=_touchstone_tryfloat(state[1]);value===nothing && break
                push!(refs,value);state=iterate(tokens,state[2])
            end
            isempty(refs) && throw(ArgumentError("Touchstone R requires a positive reference"))
            continue
        else
            throw(ArgumentError("unknown Touchstone option $t"))
        end
        state=iterate(tokens,nextstate)
    end
    all(z -> isfinite(z) && z>0,refs) || throw(ArgumentError("invalid Touchstone reference"))
    return scale,kind,format,refs
end

function _network_constitutive_s(E,F,refs)
    D=Diagonal(sqrt.(refs)); Di=Diagonal(inv.(sqrt.(refs)))
    S = -(E*D-F*Di) \ (E*D+F*Di)
    all(isfinite,S) || throw(ArgumentError("network constitutive equations have a singular wave solve"))
    return Matrix{ComplexF64}(S)
end

function _touchstone_to_s(M,kind,refs,version)
    n=size(M,1)
    kind=="S" && return M
    if version<2
        # Version 1 Y/Z/H/G are normalized; version 2 uses SI values.
        if !all(==(refs[1]),refs)
            # Legacy coordinates are V/sqrt(R), I*sqrt(R). Their wave
            # equations have unit references even when physical R differs
            # per port. Solving in these coordinates avoids denormalizing
            # and immediately renormalizing large or small coefficients.
            return _touchstone_to_s(M,kind,ones(n),2.)
        end
        r=refs[1]
        if kind=="Y"; M ./= r
        elseif kind=="Z"; M .*= r
        elseif kind=="H"; M[1,1]*=r; M[2,2]/=r
        else; M[1,1]/=r; M[2,2]*=r
        end
    end
    kind=="Y" && return planar_y_to_s(M,refs)
    kind=="Z" && return _network_constitutive_s(Matrix{ComplexF64}(I,n,n),-M,refs)
    n==2 || throw(ArgumentError("Touchstone H/G parameters require two ports"))
    if kind=="H"
        E=ComplexF64[1 -M[1,2]; 0 -M[2,2]]
        F=ComplexF64[-M[1,1] 0; -M[2,1] 1]
    else
        E=ComplexF64[-M[1,1] 0; -M[2,1] 1]
        F=ComplexF64[1 -M[1,2]; 0 -M[2,2]]
    end
    return _network_constitutive_s(E,F,refs)
end

function _touchstone_mixed_transforms(order,n,refs)
    length(order)==n || throw(ArgumentError("mixed-mode order must contain one entry per port"))
    U=zeros(Float64,n,n); P=zeros(Float64,n,n); Q=zeros(Float64,n,n)
    for (r,item) in enumerate(order)
        m=match(r"^([SDC])(\d+)(?:,(\d+))?$",uppercase(item))
        m===nothing && throw(ArgumentError("invalid Touchstone mixed-mode entry $item"))
        kind=m.captures[1]; i=parse(Int,m.captures[2])
        1<=i<=n || throw(ArgumentError("mixed-mode port outside network"))
        if kind=="S"
            m.captures[3]===nothing || throw(ArgumentError("single-ended mode cannot have a pair"))
            U[r,i]=P[r,i]=Q[r,i]=1.
        else
            m.captures[3]===nothing && throw(ArgumentError("differential/common mode needs a pair"))
            j=parse(Int,m.captures[3])
            1<=j<=n && j!=i && refs[i]==refs[j] || throw(ArgumentError(
                "mixed-mode pair requires distinct in-range ports with equal references"))
            sign=kind=="D" ? -1. : 1.
            U[r,i]=1/sqrt(2.); U[r,j]=sign/sqrt(2.)
            voltage=kind=="D" ? 1. : .5
            current=kind=="D" ? .5 : 1.
            P[r,i]=voltage; P[r,j]=sign*voltage
            Q[r,i]=current; Q[r,j]=sign*current
        end
    end
    norm(U*transpose(U)-I,Inf)<1e-12 || throw(ArgumentError(
        "mixed-mode entries must form a complete independent port basis"))
    return U,P,Q
end

"""Read Touchstone 1.0/1.1/2.0/2.1 network data: S/Y/Z and two-port H/G,
RI/MA/DB, per-port references, wrapped rows, full/upper/lower matrices,
and standard mixed-mode order. Version 1 non-S values are denormalized.
Version 2 noise blocks and information blocks are parsed separately and
excluded from the network response. Unknown keywords and incomplete
records reject. `nports` may specify the count when the filename has no
`.sNp` suffix. Sonnet `! TERM` (R,X) and `! FTERM` (R,X,L,C) records
are retained as complex references; their values use SI units and FTERM
uses Sonnet's parallel capacitance. Wrapped `&` continuations are consumed
before network data. These extensions require single-ended S data.
`max_bytes` bounds owned numeric storage and conversion workspace. Numeric
records are counted before parsing/appending; version 2 matrix records
retain their permitted arbitrary physical line length."""
function planar_read_touchstone(path::AbstractString;nports::Union{Nothing,Integer}=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    isfile(path) || throw(ArgumentError("network file not found: $path"))
    suffix=match(r"\.s(\d+)p$"i,path)
    n=nports===nothing ? (suffix===nothing ? 0 : parse(Int,suffix.captures[1])) : Int(nports)
    n>=0 || throw(ArgumentError("nports must be positive"))
    _touchstone_minimum_workspace(n,max_bytes)
    version=1.; scale,kind,format,options=_touchstone_options("";max_bytes)
    refs=Float64[]; payload=Float64[]; noise=Float64[]; mixed=String[]
    matrix_format="FULL"; order="21_12"; nf=nothing; nnf=nothing
    state=:network; seen=Set{String}(); option_seen=false; ended=false
    termination_kind=nothing;termination_values=Float64[];termination_continued=false
    open(path,"r") do io
    for (line_no,line) in enumerate(eachline(io))
        text=strip(first(split(line,'!';limit=2)))
        state==:information && uppercase(text)!="[END INFORMATION]" && continue
        extension=match(r"^\s*!\s*(F?TERM)\b(.*)$"i,line)
        if extension!==nothing
            ended && throw(ArgumentError("termination record after Touchstone [End]"))
            termination_kind===nothing || throw(ArgumentError("duplicate Sonnet termination record"))
            isempty(payload) || throw(ArgumentError("Sonnet terminations must precede network data"))
            termination_kind=uppercase(extension.captures[1])
            termination_continued=_touchstone_append_numbers!(termination_values,extension.captures[2],
                length(payload)+length(refs)+length(noise)+length(options),max_bytes;termination=true)
            continue
        elseif termination_continued
            isempty(text) && continue
            termination_continued=_touchstone_append_numbers!(termination_values,text,
                length(payload)+length(refs)+length(noise)+length(options),max_bytes;termination=true)
            continue
        end
        isempty(text) && continue
        ended && throw(ArgumentError("data after Touchstone [End] at line $line_no"))
        if startswith(text,"#")
            # IBIS 2.1 section Option Line: only the first line applies.
            option_seen && continue
            scale,kind,format,options=_touchstone_options(text[2:end];max_bytes,
                other_owned=length(payload)+length(refs)+length(noise)+length(termination_values)); option_seen=true
            continue
        end
        if startswith(text,"[")
            m=match(r"^\[([^\]]+)\]\s*(.*)$",text)
            m===nothing && throw(ArgumentError("invalid Touchstone keyword at line $line_no"))
            key=uppercase(strip(m.captures[1])); value=strip(m.captures[2])
            key in seen && throw(ArgumentError("duplicate Touchstone [$key]"))
            push!(seen,key)
            if key=="VERSION"
                version=_network_float(value)
                version in (2.,2.1) || throw(ArgumentError("unsupported Touchstone version"))
                state=:header
            elseif key=="NUMBER OF PORTS"
                count=parse(Int,value); count>0 || throw(ArgumentError("invalid port count"))
                n in (0,count) || throw(ArgumentError("Touchstone port count disagrees with filename/nports"))
                n=count;_touchstone_minimum_workspace(n,max_bytes);state=:header
            elseif key=="NUMBER OF FREQUENCIES"
                nf=parse(Int,value); nf>0 || throw(ArgumentError("invalid frequency count")); state=:header
            elseif key=="NUMBER OF NOISE FREQUENCIES"
                nnf=parse(Int,value); nnf>=0 || throw(ArgumentError("invalid noise count")); state=:header
            elseif key=="REFERENCE"
                _touchstone_append_numbers!(refs,value,length(payload)+length(noise)+length(termination_values)+length(options),max_bytes)
                state=:reference
            elseif key=="MATRIX FORMAT"
                matrix_format=uppercase(value)
                matrix_format in ("FULL","LOWER","UPPER") || throw(ArgumentError("invalid matrix format"))
                state=:header
            elseif key=="TWO-PORT DATA ORDER"
                order=uppercase(value); order in ("21_12","12_21") || throw(ArgumentError("invalid two-port order")); state=:header
            elseif key=="MIXED-MODE ORDER"
                append!(mixed,split(value)); state=:mixed
            elseif key=="NETWORK DATA"
                state=:network
            elseif key=="NOISE DATA"
                state=:noise
            elseif key=="BEGIN INFORMATION"
                state=:information
            elseif key=="END INFORMATION"
                state==:information || throw(ArgumentError("unmatched information block")); state=:header
            elseif key=="END"
                ended=true
            else
                throw(ArgumentError("unsupported Touchstone keyword [$key]"))
            end
            continue
        end
        state==:information && continue
        if state==:mixed
            append!(mixed,split(text)); continue
        end
        target=state==:reference ? refs : state==:network ? payload : state==:noise ? noise : nothing
        target===nothing && throw(ArgumentError("unexpected numeric data before [Network Data]"))
        owned=_checked_payload_sum("Touchstone numeric input",length(payload),length(refs),length(noise),length(termination_values),length(options))
        _touchstone_append_numbers!(target,text,owned-length(target),max_bytes)
    end
    end
    termination_continued && throw(ArgumentError("unfinished Sonnet termination continuation"))
    option_seen && n>0 || throw(ArgumentError("Touchstone needs an option line and a known port count"))
    if version>=2
        all(k -> k in seen,("NUMBER OF PORTS","NUMBER OF FREQUENCIES","NETWORK DATA","END")) ||
            throw(ArgumentError("incomplete Touchstone version 2 header"))
        n==2 && !("TWO-PORT DATA ORDER" in seen) && throw(ArgumentError("two-port version 2 needs its data order"))
    elseif !isempty(seen)
        throw(ArgumentError("Touchstone keywords require [Version] 2.0 or 2.1"))
    end
    kind in ("H","G") && n!=2 && throw(ArgumentError("H/G data needs two ports"))
    "REFERENCE" in seen && isempty(refs) && throw(ArgumentError(
        "Touchstone [Reference] must contain one resistance per port"))
    refs=isempty(refs) ? (length(options)==1 ? fill(only(options),n) : options) : refs
    refs=_planar_reference_values(refs,n)
    all(isreal,refs) || throw(ArgumentError("standard Touchstone references must be real"))
    refs=real.(refs)
    if termination_kind!==nothing
        kind=="S" && isempty(mixed) || throw(ArgumentError("Sonnet termination extensions require single-ended S data"))
        length(termination_values)==(termination_kind=="TERM" ? 2n : 4n) ||
            throw(ArgumentError("Sonnet termination record must contain all port references"))
    end
    count=matrix_format=="FULL" ? _checked_array_payload_bytes(UInt8,n,n) :
        Int(BigInt(n)*(BigInt(n)+1)÷2)
    stride=_checked_payload_sum("Touchstone record",1,2BigInt(count))
    !isempty(payload) && length(payload)%stride==0 || throw(ArgumentError("incomplete Touchstone network record"))
    samples=length(payload)÷stride
    nf===nothing || samples==nf || throw(ArgumentError("Touchstone frequency count mismatch"))
    if !isempty(noise) || nnf!==nothing
        n==2 && kind=="S" && nnf!==nothing && length(noise)==5nnf || throw(ArgumentError("invalid Touchstone noise block"))
        all(k -> noise[5k-4]>=0 && (k==1 || noise[5k-4]>noise[5k-9]),1:nnf) ||
            throw(ArgumentError("invalid noise frequencies"))
    end
    bytes=_checked_payload_sum("Touchstone workspace",
        _checked_array_payload_bytes(Float64,length(payload)+length(noise)+samples),
        _checked_array_payload_bytes(ComplexF64,samples+8,n,n),
        _checked_array_payload_bytes(ComplexF64,3samples+3,n),
        _checked_array_payload_bytes(Float64,length(termination_values)))
    _enforce_payload_limit(bytes,max_bytes,"Touchstone workspace","max_bytes")
    transforms=isempty(mixed) ? nothing : _touchstone_mixed_transforms(mixed,n,refs)
    !isempty(mixed) && !(kind in ("S","Y","Z")) && throw(ArgumentError("mixed-mode data supports only S/Y/Z"))
    frequencies=Vector{Float64}(undef,samples); matrices=Vector{Matrix{ComplexF64}}(undef,samples)
    reference_series=termination_kind=="FTERM" ? Vector{Vector{ComplexF64}}(undef,samples) : nothing
    term_refs=termination_kind=="TERM" ? _planar_reference_values(
        [complex(termination_values[2p-1],termination_values[2p]) for p in 1:n],n) : nothing
    term_models=termination_kind=="FTERM" ? [PlanarPortImpedance(
        r=termination_values[4p-3],x=termination_values[4p-2],l=termination_values[4p-1],
        c=termination_values[4p],topology=:parallel) for p in 1:n] : nothing
    for k in 1:samples
        offset=(k-1)*stride; frequencies[k]=scale*payload[offset+1]
        reference_series===nothing || (reference_series[k]=_planar_reference_values(term_models,n;freq=frequencies[k]))
        M=zeros(ComplexF64,n,n); index=offset+2
        positions=matrix_format=="FULL" && n==2 && order=="21_12" ?
            ((p,q) for q in 1:n for p in 1:n) :
            ((p,q) for p in 1:n for q in 1:n if matrix_format=="FULL" ||
                (matrix_format=="LOWER" ? q<=p : q>=p))
        for (p,q) in positions
            a,b=payload[index],payload[index+1]; index+=2
            value=format=="RI" ? complex(a,b) : (format=="DB" ? 10.0^(a/20) : a)*cispi(b/180)
            isfinite(value) || throw(ArgumentError("nonfinite converted Touchstone value"))
            M[p,q]=value
            matrix_format=="FULL" || (M[q,p]=value)
        end
        if transforms!==nothing
            U,P,Q=transforms
            M=kind=="S" ? transpose(U)*M*U : kind=="Y" ? Q\(M*P) : P\(M*Q)
        end
        matrices[k]=_touchstone_to_s(M,kind,refs,version)
    end
    # Arrays are already owned: avoid a second complete databank copy.
    stored_refs=term_refs===nothing ? (reference_series===nothing ? refs : first(reference_series)) : term_refs
    return _owned_network_data(frequencies,matrices,stored_refs;reference_series)
end

function _owned_network_data(freqs,series,refs;reference_series=nothing)
    return PlanarNetworkData(freqs,series,refs,["p$i" for i in eachindex(refs)],Val(:owned);reference_series)
end

function _network_fixed_real_export(data::PlanarNetworkData,z0,max_bytes)
    n=_network_series_validate(data.frequencies,data.s)
    refs=_planar_reference_values(z0===nothing ? real.(data.z0) : z0,n)
    all(isreal,refs) || throw(ArgumentError("file exports require fixed real positive output references"))
    _enforce_payload_limit(_network_export_storage_bytes(n),max_bytes,
        "network file conversion","max_bytes")
    # Validate conversions before opening a destination. Do not retain a
    # second complete databank: export recomputes only changed wave bases.
    for k in eachindex(data.frequencies);_network_export_sample(data,k,refs);end
    return real.(refs)
end

_network_export_storage_bytes(n)=_checked_payload_sum("network file conversion",
    _checked_array_payload_bytes(ComplexF64,8,n,n),_checked_array_payload_bytes(ComplexF64,6,n))

function _network_data_for_export(freqs,series;max_bytes,z0=nothing,kw...)
    n=_network_series_validate(freqs,series);reserve=_network_export_storage_bytes(n)
    _enforce_payload_limit(reserve,max_bytes,"network file conversion","max_bytes")
    remaining=Int(BigInt(_validated_resource_limit("max_bytes",max_bytes))-reserve)
    return PlanarNetworkData(freqs,series;z0,max_bytes=remaining,kw...),reserve
end
function _network_export_sample(data,k,refs)
    source=data.reference_series===nothing ? data.z0 : data.reference_series[k]
    return source==refs ? data.s[k] : planar_renormalize_s(data.s[k],source,refs)
end

_network_touchstone_version(version)=version in ("1.0","1.1","2.0","2.1") ? version :
    throw(ArgumentError("Touchstone output version must be 1.0, 1.1, 2.0 or 2.1"))

"""Write a complete Touchstone RI databank. `version=\"2.1\"` is the
default; `\"1.0\"`, `\"1.1\"` and `\"2.0\"` select their standard syntax.
Version 1.0 requires the same real reference at every port; use `z0` to
request a common output basis. Version 1.1 and version 2 retain per-port
real references. Complex or frequency-dependent input references are converted
to one fixed real basis, defaulting to `real.(data.z0)`; `z0` overrides it.
All data is validated before opening `path`. Numeric values
are written with round-trip precision. Noise data is not synthesized."""
function planar_write_touchstone(path::AbstractString,data::PlanarNetworkData;
        z0=nothing,version::AbstractString="2.1",max_bytes::Integer=_default_max_dense_payload_bytes())
    _network_touchstone_version(version)
    n=_network_series_validate(data.frequencies,data.s)
    refs=_network_fixed_real_export(data,z0,max_bytes)
    version=="1.0" && !all(==(refs[1]),refs) && throw(ArgumentError(
        "Touchstone 1.0 output requires a common reference; supply z0 or use version 1.1/2"))
    legacy=startswith(version,"1.")
    open(path,"w") do io
        println(io,"! DiffMoM network databank")
        if legacy
            println(io,"# Hz S RI R ",version=="1.0" ? string(refs[1]) : join(refs," "))
        else
            println(io,"[Version] ",version,"\n# Hz S RI R 50")
            println(io,"[Number of Ports] ",n)
            n==2 && println(io,"[Two-Port Data Order] 12_21")
            println(io,"[Number of Frequencies] ",length(data.frequencies))
            println(io,"[Reference] ",join(refs," "))
            println(io,"[Matrix Format] Full\n[Network Data]")
        end
        for (k,f) in enumerate(data.frequencies)
            S=_network_export_sample(data,k,refs)
            print(io,f)
            if legacy && n==2
                for q in 1:n,p in 1:n
                    print(io," ",real(S[p,q])," ",imag(S[p,q]))
                end
                println(io)
            elseif legacy && n>=3
                # Legacy files start each matrix row on a new line and
                # allow at most four complex pairs on any physical line.
                for p in 1:n
                    for q in 1:n
                        q>1 && (q-1)%4==0 && println(io)
                        print(io," ",real(S[p,q])," ",imag(S[p,q]))
                    end
                    println(io)
                end
            else
                for p in 1:n,q in 1:n
                    print(io," ",real(S[p,q])," ",imag(S[p,q]))
                end
                println(io)
            end
        end
        legacy || println(io,"[End]")
    end
    return path
end

function planar_write_touchstone(path::AbstractString,freqs::AbstractVector,series::AbstractVector;
        z0=nothing,output_z0=nothing,version::AbstractString="2.1",
        max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    _network_touchstone_version(version)
    data,reserve=_network_data_for_export(freqs,series;z0,max_bytes,kw...)
    return planar_write_touchstone(path,data;z0=output_z0,version,max_bytes=reserve)
end
