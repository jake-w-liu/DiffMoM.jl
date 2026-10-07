export planar_write_databank_csv, planar_read_sparam_csv, planar_compare_sweeps
export planar_equation_curves, planar_write_equation_curves
export planar_convergence_certificate, planar_stripline_benchmark, planar_write_report
export planar_dc_sparams

"""Write a lossless numeric S databank as long-form CSV. Frequencies are
Hz and ports use their names. Complex/frequency-dependent references are
converted to fixed real `z0` (default `real.(data.z0)`) before writing.
Per-port output reference impedances are
recorded in comment lines. Validation precedes opening the output file."""
function planar_write_databank_csv(path::AbstractString,data::PlanarNetworkData;
        z0=nothing,max_bytes::Integer=_default_max_dense_payload_bytes())
    n=_network_series_validate(data.frequencies,data.s)
    length(data.port_names)==n && length(unique(data.port_names))==n && all(!isempty,data.port_names) ||
        throw(ArgumentError("CSV port names must be nonempty, unique and match ports"))
    all(name -> !occursin('\n',name) && !occursin('\r',name),data.port_names) ||
        throw(ArgumentError("CSV port names cannot contain line breaks"))
    refs=_network_fixed_real_export(data,z0,max_bytes)
    open(path,"w") do io
        println(io,"# z0_ohm=",join(refs,";"))
        println(io,"frequency_hz,output_port,input_port,s_real,s_imag")
        for (k,f) in enumerate(data.frequencies)
            S=_network_export_sample(data,k,refs)
            for q in 1:n,p in 1:n
                println(io,f,',',_network_csv_escape(data.port_names[p]),',',
                    _network_csv_escape(data.port_names[q]),',',real(S[p,q]),',',imag(S[p,q]))
            end
        end
    end
    return path
end
function planar_write_databank_csv(path::AbstractString,freqs,series;
        z0=nothing,output_z0=nothing,max_bytes::Integer=_default_max_dense_payload_bytes(),kw...)
    data,reserve=_network_data_for_export(freqs,series;z0,max_bytes,kw...)
    return planar_write_databank_csv(path,data;z0=output_z0,max_bytes=reserve)
end

_network_csv_escape(value) = "\""*replace(value,'"'=>"\"\"")*"\""
function _network_csv_fields(text)
    fields=String[];buffer=IOBuffer();quoted=false;closed=false
    chars=collect(text);i=1
    while i<=length(chars)
        c=chars[i]
        if quoted
            if c=='"'
                if i<length(chars) && chars[i+1]=='"';write(buffer,'"');i+=1
                else;quoted=false;closed=true
                end
            else;write(buffer,c)
            end
        elseif c==','
            push!(fields,String(take!(buffer)));closed=false
        elseif c=='"'
            position(buffer)==0 && !closed || throw(ArgumentError("invalid CSV quoting"))
            quoted=true
        else
            closed && !isspace(c) && throw(ArgumentError("text after closing CSV quote"))
            closed || write(buffer,c)
        end
        i+=1
    end
    quoted && throw(ArgumentError("unclosed CSV quote"))
    push!(fields,String(take!(buffer)))
    return fields
end

"""Read the complete long-form `frequency_hz,output_port,input_port,
s_real,s_imag` matrix CSV. Missing or duplicate entries reject. Numeric
indices and named ports are accepted; names retain first-appearance order.
Optional extra columns are ignored. Returns [`PlanarNetworkData`](@ref).
`z0` overrides the file's reference comment when supplied."""
function planar_read_sparam_csv(path::AbstractString;z0=nothing,
        max_bytes::Integer=_default_max_dense_payload_bytes())
    isfile(path) || throw(ArgumentError("S-parameter CSV not found: $path"))
    records=Tuple{Float64,String,String,ComplexF64}[]; labels=String[]
    refs=nothing;columns=nothing
    open(path,"r") do io
    for (line_no,line) in enumerate(eachline(io))
        text=strip(line);isempty(text) && continue
        if startswith(text,"#")
            startswith(text,"# z0_ohm=") && (refs=_network_float.(split(last(split(text,'=';limit=2)),';')))
            continue
        end
        fields=_network_csv_fields(text)
        if columns===nothing
            columns=Dict(v=>i for (i,v) in enumerate(fields))
            all(k -> haskey(columns,k),("frequency_hz","output_port","input_port","s_real","s_imag")) ||
                throw(ArgumentError("CSV missing required S-parameter columns"))
            continue
        end
        length(fields)>=maximum(values(columns)) || throw(ArgumentError("short CSV row at line $line_no"))
        f=_network_float(fields[columns["frequency_hz"]])
        p,q=fields[columns["output_port"]],fields[columns["input_port"]]
        s=complex(_network_float(fields[columns["s_real"]]),_network_float(fields[columns["s_imag"]]))
        isfinite(f) && f>=0 && isfinite(s) && !isempty(p) && !isempty(q) ||
            throw(ArgumentError("invalid S-parameter CSV row at line $line_no"))
        for label in (p,q);label in labels || push!(labels,label);end
        _enforce_payload_limit(_checked_array_payload_bytes(ComplexF64,4,length(records)+1),
            max_bytes,"S-parameter CSV records","max_bytes")
        push!(records,(f,p,q,s))
    end
    end
    isempty(records) && throw(ArgumentError("empty S-parameter CSV"))
    numeric=tryparse.(Int,labels)
    if all(!isnothing,numeric)
        sort!(labels;by=x->parse(Int,x))
        parse.(Int,labels)==collect(1:length(labels)) || throw(ArgumentError("CSV port indices must cover 1:nports"))
    end
    frequencies=sort!(unique(first.(records)));n=length(labels)
    bytes=_checked_payload_sum("CSV network workspace",
        _checked_array_payload_bytes(ComplexF64,4length(records)),
        _checked_array_payload_bytes(ComplexF64,length(frequencies),n,n),
        _checked_array_payload_bytes(UInt8,length(frequencies),n,n))
    _enforce_payload_limit(bytes,max_bytes,"CSV network workspace","max_bytes")
    matrices=[zeros(ComplexF64,n,n) for _ in frequencies]
    seen=[falses(n,n) for _ in frequencies]
    lookup=Dict(label=>i for (i,label) in enumerate(labels))
    for (f,p,q,s) in records
        k=searchsortedfirst(frequencies,f);i,j=lookup[p],lookup[q]
        seen[k][i,j] && throw(ArgumentError("duplicate CSV matrix entry at $f Hz, $p,$q"))
        seen[k][i,j]=true;matrices[k][i,j]=s
    end
    all(all,seen) || throw(ArgumentError("CSV matrix has missing entries"))
    references=_circuit_z0(z0===nothing ? (refs===nothing ? 50. : refs) : z0,n)
    names=all(!isnothing,numeric) ? ["p$i" for i in 1:n] : labels
    return PlanarNetworkData(frequencies,matrices,references,names,Val(:owned))
end

"""Compare complete complex S matrices at the candidate's frequencies.
`interpolate=false` requires exact reference knots; enabling it explicitly
uses the reference databank's interpolation. References must match, or
`renormalize=true` changes the reference wave basis. Returns the maximum
absolute complex error and its frequency/port pair, RMS error and verdict."""
function planar_compare_sweeps(candidate::PlanarNetworkData,reference::PlanarNetworkData;
        atol::Real=1e-3,interpolate::Bool=false,renormalize::Bool=false)
    isfinite(atol) && atol>=0 || throw(ArgumentError("comparison tolerance must be finite and nonnegative"))
    size(first(candidate.s))==size(first(reference.s)) || throw(DimensionMismatch("comparison port counts differ"))
    candidate.port_names==reference.port_names || throw(ArgumentError("comparison port names/order differ"))
    peak=-Inf;worst=(frequency_hz=first(candidate.frequencies),output_port=1,input_port=1)
    errors=Vector{Float64}(undef,length(candidate.frequencies));square=0.;count=0
    for (k,f) in enumerate(candidate.frequencies)
        interpolate || f in reference.frequencies || throw(ArgumentError("reference is missing exact frequency $f"))
        refs=_network_reference_at(candidate,f)
        renormalize || refs==_network_reference_at(reference,f) || throw(ArgumentError("comparison references differ at $f Hz"))
        R=planar_network_response(reference,f;z0=refs)
        E=candidate.s[k]-R;errors[k]=maximum(abs,E)
        square+=sum(abs2,E);count+=length(E)
        if errors[k]>peak
            peak=errors[k];index=argmax(abs.(E));worst=(frequency_hz=f,output_port=index[1],input_port=index[2])
        end
    end
    return (passed=peak<=atol,max_absolute_error=peak,rms_error=sqrt(square/count),
        tolerance=Float64(atol),worst=worst,per_frequency_error=errors,
        interpolated=interpolate,renormalized=renormalize)
end

function _curve_derivative(x,y,k)
    n=length(x)
    n==1 && return NaN
    n==2 && return (y[2]-y[1])/(x[2]-x[1])
    start=clamp(k-1,1,n-2);middle=start+1;last=start+2
    left=x[middle]-x[start];right=x[last]-x[middle]
    before=(y[middle]-y[start])/left;after=(y[last]-y[middle])/right
    # Differentiate the quadratic using offsets and secant slopes. Avoid
    # products of frequency intervals and cancellation of absolute phases;
    # a constant phase then has an exact zero derivative at every point.
    if k==start
        return before-(left/(left+right))*(after-before)
    elseif k==last
        return after+(right/(left+right))*(after-before)
    end
    return (right/(left+right))*before+(left/(left+right))*after
end

"""Compute terminated-input impedance, reflection dB/SWR, effective L/C
and signed resistance quality factor at each port. For two-port sweeps,
`group_delay_s` is `-d arg(S21)/dω`, with phase unwrapping and a quadratic
derivative on nonuniform frequency grids. Phase is undefined where
`abs(S21)<=phase_floor`; those points and their derivative stencils return
NaN. Open input impedance is `Inf+0im`; active-port SWR is NaN.
Reflection dB uses the declared power-wave reference. SWR uses a real
line reference `real(Zref)` at each sample, with the same physical load;
complex power-wave magnitudes do not define a voltage standing-wave ratio.
L/C at DC is undefined, rather than an extrapolated EM result."""
function planar_equation_curves(data::PlanarNetworkData;phase_floor::Real=1e-12)
    isfinite(phase_floor) && phase_floor>=0 || throw(ArgumentError("phase_floor must be nonnegative"))
    n=_network_series_validate(data.frequencies,data.s);nf=length(data.frequencies)
    db=zeros(nf,n);swr=similar(db);zin=zeros(ComplexF64,nf,n)
    l=fill(NaN,nf,n);c=copy(l);q=copy(l);omega=2pi.*data.frequencies
    for k in 1:nf,i in 1:n
        s=data.s[k][i,i];m=abs(s)
        db[k,i]=20log10(m)
        ref=_network_reference_at(data,data.frequencies[k])[i]
        standing=abs(_network_impedance_reflection(s,ref,real(ref)))
        swr[k,i]=standing<1 ? (1+standing)/(1-standing) : standing==1 ? Inf : NaN
        zin[k,i]=s==1 ? complex(Inf,0.) : (conj(ref)+ref*s)/(1-s)
        r,x=real(zin[k,i]),imag(zin[k,i])
        if omega[k]>0 && isfinite(x)
            l[k,i]=x>0 ? x/omega[k] : 0.
            c[k,i]=x<0 ? -inv(omega[k]*x) : 0.
        end
        q[k,i]=r==0 ? (x==0 ? NaN : Inf) : abs(x)/r
    end
    delay=fill(NaN,nf)
    if n==2
        phase=fill(NaN,nf)
        for k in 1:nf
            abs(data.s[k][2,1])>phase_floor || continue
            ph=angle(data.s[k][2,1])
            phase[k]=k>1 && isfinite(phase[k-1]) ? ph+2pi*round((phase[k-1]-ph)/(2pi)) : ph
        end
        for k in 1:nf
            # Multiplying a valid, closely spaced Hz grid by 2pi can merge
            # adjacent Float64 knots. Keep its original coordinates until
            # converting dphase/df to dphase/domega.
            delay[k]=-_curve_derivative(data.frequencies,phase,k)/(2pi)
        end
    end
    return (frequency_hz=copy(data.frequencies),reflection_db=db,swr=swr,zin_ohm=zin,
        l_eff_h=l,c_eff_f=c,q=q,group_delay_s=delay)
end

"""Write the values from [`planar_equation_curves`](@ref) as CSV.
Undefined quantities retain NaN/Inf in the exported data."""
function planar_write_equation_curves(path::AbstractString,data::PlanarNetworkData;kw...)
    curves=planar_equation_curves(data;kw...);n=length(data.z0)
    header=["frequency_hz"]
    for i in 1:n
        append!(header,["s$(i)$(i)_db","swr_$i","zin_$(i)_real_ohm","zin_$(i)_imag_ohm",
            "l_eff_$(i)_h","c_eff_$(i)_f","q_$i"])
    end
    n==2 && push!(header,"s21_group_delay_s")
    open(path,"w") do io
        println(io,join(header,','))
        for k in eachindex(data.frequencies)
            row=Float64[data.frequencies[k]]
            for i in 1:n
                append!(row,[curves.reflection_db[k,i],curves.swr[k,i],
                    real(curves.zin_ohm[k,i]),imag(curves.zin_ohm[k,i]),
                    curves.l_eff_h[k,i],curves.c_eff_f[k,i],curves.q[k,i]])
            end
            n==2 && push!(row,curves.group_delay_s[k]);println(io,join(row,','))
        end
    end
    return path
end

"""Compare successive refined responses on the same frequency/port/reference
grid. A certificate needs at least three resolutions, nonincreasing maximum
complex S differences, and a final difference `<=atol`. This is a successive
refinement certificate; it does not prove absolute accuracy or bound the
remaining continuum error. A single agreeing pair is reported unverified."""
function planar_convergence_certificate(refinements::AbstractVector{<:PlanarNetworkData};atol::Real=1e-3)
    length(refinements)>=2 || throw(ArgumentError("convergence requires at least two responses"))
    deltas=Float64[]
    for k in 2:length(refinements)
        refinements[k].frequencies==refinements[k-1].frequencies || throw(ArgumentError("refinement frequency grids differ"))
        push!(deltas,planar_compare_sweeps(refinements[k],refinements[k-1];atol).max_absolute_error)
    end
    monotone=all(k -> deltas[k]<=deltas[k-1],2:length(deltas))
    verified=length(refinements)>=3 && monotone && last(deltas)<=atol
    return (verified=verified,deltas=deltas,monotone=monotone,tolerance=Float64(atol),
        scope="successive complex S refinement, not an absolute error bound")
end

"""Absolute homogeneous air-stripline benchmark from deembedded uniform-line
Y. Computes `eT=|Zc-Ztarget|/Ztarget+|v-c|/c`; `target_error` is a fraction
(0.001 means 0.1%). The physical geometry must match the supplied analytic
impedance, and the electrical-length branch must be identified."""
function planar_stripline_benchmark(Y::AbstractMatrix,len::Real,f::Real;
        target_impedance::Real=50.,target_error::Real=.001,phase_hint=nothing)
    isfinite(len) && len>0 && isfinite(f) && f>0 || throw(ArgumentError("benchmark length/frequency must be positive"))
    isfinite(target_impedance) && target_impedance>0 && isfinite(target_error) && target_error>=0 ||
        throw(ArgumentError("benchmark target impedance/error is invalid"))
    line=planar_line_params(Y;phase_hint);gamma=line.gl/len
    imag(gamma)>0 || throw(ArgumentError("benchmark needs a forward propagating identified line branch"))
    velocity=2pi*f/imag(gamma);c0=299792458.
    error=abs(line.zc-target_impedance)/target_impedance+abs(velocity-c0)/c0
    return (passed=error<=target_error,absolute_error=Float64(error),target_error=Float64(target_error),
        characteristic_impedance_ohm=line.zc,velocity_m_per_s=velocity,gamma_per_m=gamma)
end

"""Evaluate the DC limit of a stable admittance model as power-wave S.
With `require_passive=true` (default), a global positive-real certificate
is required. This is a model extrapolation from its training band; direct
EM assembly at zero frequency is not implied."""
function planar_dc_sparams(model::PlanarRationalModel;z0=50.,require_passive::Bool=true)
    require_passive && !planar_rational_certificate(model).certified && throw(ArgumentError(
        "DC extrapolation requires a global passive-model certificate"))
    Y=planar_rational_eval(model,0.);refs=_circuit_z0(z0,size(Y,1))
    return planar_y_to_s(Y,refs)
end

"""Write a deterministic text report for a response databank. Reports
sampled reciprocity and passive-wave bounds explicitly. Optional accuracy
and convergence results are printed with their scope; no certification is
inferred from a finite response grid."""
function planar_write_report(path::AbstractString,data::PlanarNetworkData;
        name::AbstractString="planar network",convergence=nothing,accuracy=nothing)
    _network_series_validate(data.frequencies,data.s)
    reciprocal=maximum(maximum(abs,S-transpose(S)) for S in data.s)
    power=maximum(maximum(svdvals(S)) for S in data.s)
    open(path,"w") do io
        println(io,"DiffMoM planar response report\nName: ",name)
        println(io,"Ports: ",length(data.z0),"; frequency points: ",length(data.frequencies))
        println(io,"Frequency range [Hz]: ",first(data.frequencies)," .. ",last(data.frequencies))
        if data.reference_series===nothing
            println(io,"Reference impedances [ohm]: ",join(data.z0,", "))
        else
            println(io,"Reference impedances [ohm]: frequency-dependent Kurokawa power waves")
            for (f,refs) in zip(data.frequencies,data.reference_series)
                println(io,"  ",f," Hz: ",join(refs,", "))
            end
        end
        println(io,"Sampled max |S-S^T|: ",reciprocal)
        println(io,"Sampled max singular value of S: ",power)
        println(io,"Scope: finite response samples; not a broadband passivity certificate")
        convergence===nothing || println(io,"Convergence: ",convergence)
        accuracy===nothing || println(io,"Absolute accuracy: ",accuracy)
    end
    return path
end
