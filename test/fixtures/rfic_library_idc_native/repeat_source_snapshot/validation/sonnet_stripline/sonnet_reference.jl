module SonnetReference

using Dates
using SHA
using TOML
using DiffMoM: planar_read_touchstone, read_sonnet_project, SonnetProject,
    SonnetNetlistProject, _sonnet_port_reference, _planar_reference_values

export find_em, reference_run, parse_sonnet_response, twoport_matrix,
       wrapped_phase_deg, evidence_directory, twoport_quantization_bound, nport_quantization_bound,
       nport_matrix, write_native_fixture, checked_native_touchstone

"""Find an installed Sonnet engine; an explicit invalid override is an error."""
function find_em()
    if haskey(ENV, "SONNET_EM")
        isfile(ENV["SONNET_EM"]) || error("SONNET_EM does not name a file")
        return abspath(ENV["SONNET_EM"])
    end
    roots = unique(filter(!isempty, [get(ENV, "ProgramFiles", ""),
        get(ENV, "ProgramFiles(x86)", ""), raw"C:\Program Files",
        raw"C:\Program Files (x86)"]))
    candidates = Tuple{VersionNumber,String}[]
    for root in roots
        dir = joinpath(root, "Sonnet Software")
        isdir(dir) || continue
        for ver in readdir(dir)
            version = tryparse(VersionNumber, ver)
            version === nothing && continue
            em = joinpath(dir, ver, "bin", "em.exe")
            isfile(em) && push!(candidates, (version, em))
        end
    end
    isempty(candidates) && return nothing
    return last(sort(candidates; by=first))[2]
end

"""Fresh directory retaining native engine output and reproducibility metadata.
Default output is under the repository's ignored `data/` tree."""
function evidence_directory(label::AbstractString)
    root = abspath(get(ENV, "SONNET_VALIDATION_DIR",
        joinpath(@__DIR__, "..", "..", "data", "sonnet_validation")))
    mkpath(root)
    return mktempdir(root; prefix=string(label, "_"), cleanup=false)
end

"""Parse native magnitude/angle response sections, retaining every full matrix.
Section selection is explicit: raw ports and calibrated ports cannot be mixed.
The frequency unit in `log_response.log` is the project's DIM FREQ unit;
these fixtures all declare GHz."""
function parse_sonnet_response(logf::AbstractString; deembedded::Bool=true,nports::Integer=2)
    nports>0 || throw(ArgumentError("native response port count must be positive"))
    rows = Vector{Vector{Float64}}()
    row_tokens = Vector{Vector{String}}()
    z0 = Dict{Tuple{Int,Float64},ComplexF64}()
    selected = false
    pending=String[]
    for line in eachline(logf)
        if occursin("S-Parameters", line)
            isempty(pending) || error("truncated native matrix before section change in $logf")
            selected = occursin("De-embedded S-Parameters", line) == deembedded
            continue
        end
        selected || continue
        m = match(r"P(\d+)\s+F=([-+0-9.eE]+).*Z0=\(([-+0-9.eE]+)\s*\+\s*j([-+0-9.eE]+)\)", line)
        if m !== nothing
            z0[(parse(Int, m[1]), parse(Float64, m[2]))] =
                complex(parse(Float64, m[3]), parse(Float64, m[4]))
        end
        tok = split(line)
        expected=isempty(pending) ? (nports<=2 ? 1+2nports^2 : 1+2nports) : 2nports
        length(tok)==expected || continue
        vals = tryparse.(Float64, tok)
        all(v -> v !== nothing && isfinite(v), vals) || continue
        append!(pending,String.(tok))
        if length(pending)==1+2nports^2
            push!(rows,parse.(Float64,pending));push!(row_tokens,copy(pending));empty!(pending)
        end
    end
    isempty(pending) || error("truncated native matrix in $logf")
    isempty(rows) && error("no $(deembedded ? "de-embedded" : "raw") $nports-port S rows in $logf")
    length(unique(first.(rows))) == length(rows) ||
        error("multiple analysis rows at one frequency in $logf")
    return (; rows, row_tokens, z0)
end

"""Native magnitude/angle row -> full complex matrix. Native two-port output
uses S11,S21,S12,S22 order; larger matrices are printed one physical row per line."""
function nport_matrix(row::AbstractVector{<:Real})
    n=isqrt((length(row)-1)÷2)
    n>0 && length(row)==1+2n^2 || throw(DimensionMismatch("native full matrix row has invalid size"))
    matrix=reshape(ComplexF64[row[i]*cispi(row[i+1]/180) for i in 2:2:length(row)-1],n,n)
    return n<=2 ? matrix : permutedims(matrix)
end

"""S11,S21,S12,S22 native Touchstone order -> complex 2×2 matrix."""
function twoport_matrix(row::AbstractVector{<:Real})
    length(row) == 9 || throw(DimensionMismatch("two-port response requires 9 values"))
    return reshape(ComplexF64[row[i] * cispi(row[i + 1] / 180)
        for i in 2:2:8], 2, 2)
end

wrapped_phase_deg(a::Number, b::Number) = rad2deg(angle(a * conj(b)))

"""Write a deliberately limited geometry benchmark, never a general project
round trip. Coordinates and frequency retain the source DIM units. Controls,
calibration records and components are not accepted by this fixture writer."""
function write_native_fixture(path,p;frequency_native,high_precision::Bool=false)
    isempty(p.variables) && isempty(p.components) ||
        throw(ArgumentError("fixture writer requires literal geometry without components"))
    all(q->q.kind in (:sheet,:via),p.polygons) || throw(ArgumentError("fixture writer supports sheet/via polygons only"))
    all(q->q.kind in (:box,:std,:via),p.ports) || throw(ArgumentError("fixture writer supports wall/axial sources only"))
    open(path,"w") do io
        println(io,"FTYP SONPROJ 19\nDIM")
        for (name,unit) in sort(collect(p.units);by=first);println(io,name," ",unit);end
        println(io,"END DIM\nCONTROL\nOPTIONS\nSPEED 0\nSUBSPLAM N 100\nEND CONTROL\nGEO")
        for (name,row) in (("TMET",p.top),("BMET",p.bottom))
            println(io,name," \"",row[1],"\" ",join(row[2:end]," "))
        end
        for row in p.metals;println(io,"MET \"",row[1],"\" ",join(row[2:end]," "));end
        println(io,"BOX ",join(p.box," "))
        for row in p.layers;println(io,join(row[1:7]," ")," \"",row[end],"\"");end
        for port in p.ports
            kind=port.kind==:via ? "VIA" : port.kind==:std ? "STD" : "BOX"
            println(io,"POR1 ",kind,"\nPOLY ",port.polygon," 1\n",port.edge,"\n",join(port.values," "))
        end
        println(io,"NUM ",length(p.polygons))
        for q in p.polygons
            q.kind==:via && println(io,"VIA POLYGON")
            println(io,q.level," ",size(q.vertices,2)," ",q.material," N ",q.id," 1 1 100 100 0 0 0 Y")
            if q.kind==:via
                modes=filter(x->x in ("RING","SOLID","FULL","CENTER","VERTICES","BAR"),q.flags)
                length(modes)<=1 || throw(ArgumentError("ambiguous native benchmark via mesh"))
                mode=isempty(modes) ? "RING" : only(modes)
                println(io,"TOLEVEL ",q.target," ",mode," NOCOVERS")
            end
            for k in axes(q.vertices,2)
                println(io,q.vertices[1,k]/p.length_scale," ",q.vertices[2,k]/p.length_scale)
            end
            println(io,"END")
        end
        println(io,"END GEO\nVARSWP\nENABLED Y\nFREQ Y AN SWEEP ",frequency_native,"\nEND\nEND VARSWP")
        high_precision && println(io,"FILEOUT\nTOUCH ND Y native_raw.s2p IC 15 S RI R 50\nFOLDER .\nEND FILEOUT")
    end
    return path
end

function _printed_resolution(token::AbstractString)
    parts=split(lowercase(token),'e')
    exponent=length(parts)==2 ? parse(Int,parts[2]) : 0
    decimal=findfirst(==('.'),parts[1])
    digits=decimal===nothing ? 0 : length(parts[1])-decimal
    return 10.0^(exponent-digits)
end

"""Per-entry uncertainty caused solely by native magnitude/angle log rounding.
An angle printed to 0.01 degrees cannot certify complex error below its own
rounding interval; this bound follows that interval, not observed error."""
function twoport_quantization_bound(tokens::AbstractVector{<:AbstractString})
    length(tokens)==9 || throw(DimensionMismatch("two-port row requires 9 tokens"))
    return nport_quantization_bound(tokens)
end

"""Full native matrix rounding intervals, in the same physical order as
[`nport_matrix`](@ref). These bounds select data consistently; they do not
replace a benchmark's independent physics or solver-residual gates."""
function nport_quantization_bound(tokens::AbstractVector{<:AbstractString})
    length(tokens)>=3 || throw(DimensionMismatch("invalid native matrix tokens"))
    n=isqrt((length(tokens)-1)÷2)
    n>0 && length(tokens)==1+2n^2 || throw(DimensionMismatch("invalid native matrix tokens"))
    bounds=Float64[]
    for i in 2:2:length(tokens)-1
        magnitude=parse(Float64,tokens[i])
        dm=_printed_resolution(tokens[i])/2
        da=deg2rad(_printed_resolution(tokens[i+1])/2)
        push!(bounds,dm+(abs(magnitude)+dm)*2sin(da/2))
    end
    matrix=reshape(bounds,n,n)
    return n<=2 ? matrix : permutedims(matrix)
end

function _native_reference_contract(source,n)
    project=read_sonnet_project(source)
    if project isa SonnetNetlistProject
        definition=last(filter(r->occursin(r"^DEF\d+P$",r.tokens[1]),project.circuit))
        parse(Int,match(r"^DEF(\d+)P$",definition.tokens[1])[1])==n ||
            error("final native network definition and selected log port counts differ")
        # DEF reference resistance is in ohms, independently of DIM RES.
        return (f->fill(parse(Float64,definition.tokens[end]),n)),"native final DEF reference"
    elseif project isa SonnetProject
        labels=sort!(unique(abs(port.number) for port in project.ports if port.number!=0))
        length(labels)==n || error("native port reference labels do not identify every selected log port")
        groups=[[port for port in project.ports if abs(port.number)==label] for label in labels]
        evaluate=function(f)
            values=ComplexF64[]
            for group in groups
                refs=[_sonnet_port_reference(project,port,f) for port in group]
                all(z->abs(z-first(refs))<=64eps(max(1.,abs(z),abs(first(refs)))),refs) ||
                    error("native terminals sharing a port label have inconsistent references")
                push!(values,first(refs))
            end
            return values
        end
        return evaluate,"native geometry port R/X/L/C references"
    end
    error("native source reference contract is unavailable")
end

"""Read an actual external Touchstone file only after checking every matrix
entry against the explicitly intended native raw/calibrated response log.
The comparison uses independent magnitude/angle printing intervals and
requires the same response frequency count/order and wave-reference basis.
`deembedded` is mandatory. References are derived from the final CKT DEF or
geometry port R/X/L/C fields. `expected_z0` supplies a scalar, per-port vector,
or frequency provider for sources whose references cannot be derived. When
both contracts exist they must agree. Every output sample's header/TERM/FTERM
references must match; data is never silently renormalized. Metadata retains
OPTIONS, the intended selector, reference contracts, file hash, and outcome.

The installed18.53 engine can write calibrated numbers to a TOUCH ND file
when OPTIONS -d is enabled; a filename or header comment alone is insufficient.
GUI Graph postprocessing and deliberate output-selection falsifiers use a
separate oracle and should not call this engine-response guard."""
function checked_native_touchstone(reference,path::AbstractString;
        deembedded::Bool,frequency_scale=nothing,expected_z0=nothing)
    metadatafile=joinpath(reference.output_dir,"metadata.toml")
    metadata=TOML.parsefile(metadatafile)
    source=joinpath(reference.output_dir,basename(metadata["source"]))
    options=String[];nativeunit=nothing;indim=false
    for line in eachline(source)
        tokens=split(strip(line));isempty(tokens) && continue
        if tokens[1]=="OPTIONS"
            options=String.(tokens[2:end])
        elseif tokens[1]=="DIM"
            indim=true
        elseif tokens[1]=="END" && length(tokens)>1 && tokens[2]=="DIM"
            indim=false
        elseif indim && tokens[1]=="FREQ" && length(tokens)==2
            nativeunit=uppercase(tokens[2])
        end
    end
    entry=Dict{String,Any}("intended_deembedded"=>deembedded,
        "intended_log_section"=>deembedded ? "calibrated" : "raw",
        "source_options"=>options,"output"=>abspath(path),
        "selected_reference_deembedded"=>get(metadata,"deembedded",false))
    outcomes=get!(metadata,"touchstone_selected_log_checks",Dict{String,Any}())
    outcomes[basename(path)]=entry
    try
        get(metadata,"deembedded",nothing)===deembedded ||
            error("intended Touchstone state disagrees with selected native log")
        scale=frequency_scale===nothing ? get(Dict("HZ"=>1.,"KHZ"=>1e3,
            "MHZ"=>1e6,"GHZ"=>1e9,"THZ"=>1e12),nativeunit,NaN) : Float64(frequency_scale)
        isfinite(scale) && scale>0 || error("native response frequency scale is required")
        entry["frequency_scale_hz"]=scale
        entry["source_frequency_unit"]=nativeunit===nothing ? "explicit" : nativeunit
        data=planar_read_touchstone(path;nports=size(nport_matrix(first(reference.rows)),1))
        length(data.s)==length(reference.rows)==length(reference.row_tokens) ||
            error("external Touchstone and selected log frequency counts differ")
        n=length(data.z0)
        contract,contract_kind=try
            _native_reference_contract(source,n)
        catch err
            entry["source_reference_error"]=sprint(showerror,err)
            expected_z0===nothing && error("native source references cannot be derived; explicit expected_z0 is required")
            nothing,"explicit expected_z0"
        end
        entry["reference_contract"]=contract_kind
        entry["expected_reference_real"]=Vector{Vector{Float64}}()
        entry["expected_reference_imag"]=Vector{Vector{Float64}}()
        entry["output_reference_real"]=Vector{Vector{Float64}}()
        entry["output_reference_imag"]=Vector{Vector{Float64}}()
        differences=Float64[];ratios=Float64[]
        for (k,(f,s,row,tokens)) in enumerate(zip(data.frequencies,data.s,reference.rows,reference.row_tokens))
            expected=nport_matrix(row)
            size(s)==size(expected) || error("external Touchstone and selected log port counts differ")
            df=abs(f-first(row)*scale)
            df<=_printed_resolution(first(tokens))/2*scale+64eps(max(1.,abs(f))) ||
                error("external Touchstone frequency disagrees with selected log")
            source_refs=contract===nothing ? nothing : _planar_reference_values(contract(f),n;freq=f)
            explicit_refs=expected_z0===nothing ? nothing : _planar_reference_values(expected_z0,n;freq=f)
            if source_refs!==nothing && explicit_refs!==nothing
                all(abs.(source_refs.-explicit_refs).<=64eps.(max.(1.,abs.(source_refs),abs.(explicit_refs)))) ||
                    error("explicit expected_z0 disagrees with native source references")
            end
            refs=source_refs===nothing ? explicit_refs : source_refs
            actual_refs=data.reference_series===nothing ? data.z0 : data.reference_series[k]
            push!(entry["expected_reference_real"],real.(refs));push!(entry["expected_reference_imag"],imag.(refs))
            push!(entry["output_reference_real"],real.(actual_refs));push!(entry["output_reference_imag"],imag.(actual_refs))
            all(abs.(actual_refs.-refs).<=64eps.(max.(1.,abs.(actual_refs),abs.(refs)))) ||
                error("external Touchstone wave references disagree with intended native log basis")
            delta=abs.(s-expected);bounds=nport_quantization_bound(tokens)
            roundoff=64eps(max(1.,maximum(abs,s),maximum(abs,expected)))
            push!(differences,maximum(delta));push!(ratios,maximum(delta./(bounds.+roundoff)))
            entry["max_difference"]=maximum(differences)
            entry["max_printed_bound_ratio"]=maximum(ratios)
            all(delta.<=bounds.+roundoff) ||
                error("external Touchstone numbers disagree with intended $(deembedded ? "calibrated" : "raw") native log beyond printed rounding intervals")
        end
        entry["status"]="PASS";entry["output_sha256"]=bytes2hex(sha256(read(path)))
        entry["max_difference"]=maximum(differences)
        entry["max_printed_bound_ratio"]=maximum(ratios)
        return data
    catch err
        entry["status"]="FAIL";entry["error"]=sprint(showerror,err)
        isfile(path) && (entry["output_sha256"]=bytes2hex(sha256(read(path))))
        rethrow()
    finally
        # A later rejected check of the same basename must not erase the
        # earlier accepted outcome, or any earlier rejection evidence.
        push!(get!(metadata,"touchstone_selected_log_check_history",Any[]),deepcopy(entry))
        open(metadatafile,"w") do io;TOML.print(io,metadata);end
    end
end

"""Execute the real engine in a fresh retained directory, with no cached data.
Reject a failed process or missing native response even if em exits silently."""
function reference_run(em::AbstractString, source::AbstractString;
        output_dir::AbstractString=evidence_directory("reference"),
        deembedded::Bool=true,dependencies::AbstractVector{<:AbstractString}=String[],nports::Integer=2)
    nports>0 || throw(ArgumentError("native response port count must be positive"))
    # The engine is launched in output_dir. Make paths absolute before
    # constructing its command, otherwise a relative destination is
    # interpreted again beneath that working directory.
    source=abspath(source)
    output_dir=abspath(output_dir)
    isfile(source) || error("missing Sonnet fixture: $source")
    mkpath(output_dir)
    dst = joinpath(output_dir, basename(source))
    ispath(dst) && error("reference directory already contains $(basename(source))")
    cp(source, dst)
    dependency_hashes=Dict{String,String}()
    for dependency in dependencies
        isfile(dependency) || error("missing Sonnet dependency: $dependency")
        target=joinpath(output_dir,basename(dependency))
        ispath(target) && error("duplicate Sonnet dependency name: $(basename(dependency))")
        cp(dependency,target)
        dependency_hashes[basename(dependency)]=bytes2hex(sha256(read(dependency)))
    end
    cmd = Cmd(`$em -v $dst`; dir=output_dir)
    started = Dates.now(Dates.UTC)
    process = open(joinpath(output_dir, "engine_stdout.log"), "w") do out
        open(joinpath(output_dir, "engine_stderr.log"), "w") do err
            run(pipeline(ignorestatus(cmd), stdout=out, stderr=err))
        end
    end
    logf = joinpath(output_dir, "sondata", splitext(basename(source))[1],
        "log_response.log")
    metadata = Dict("source" => abspath(source), "source_sha256" =>
        bytes2hex(sha256(read(source))), "em" => abspath(em),
        "command" => string(cmd), "started_utc" => string(started),
        "ended_utc" => string(Dates.now(Dates.UTC)),
        "process_success" => success(process), "julia_version" => string(VERSION),
        "deembedded" => deembedded,"dependency_sha256"=>dependency_hashes,"response_nports"=>nports)
    stdout=read(joinpath(output_dir,"engine_stdout.log"),String)
    version=match(r"Em version ([^\r\n]+)",stdout)
    version===nothing || (metadata["engine_version"]=strip(version[1]))
    counts=[parse(Int,m[1]) for m in eachmatch(r"requires (\d+) subsections",stdout)]
    isempty(counts) || (metadata["native_subsections"]=counts)
    box=match(r"(?m)^BOX \d+ \S+ \S+ (\d+) (\d+)",read(source,String))
    if box!==nothing
        halfcounts=parse.(Int,collect(box.captures))
        metadata["box_halfcell_counts"]=halfcounts
        metadata["actual_cell_counts"]=halfcounts.÷2
    end
    open(joinpath(output_dir, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
    success(process) || error("Sonnet analysis failed; see $output_dir")
    isfile(logf) || error("Sonnet produced no native response; see $output_dir")
    response = parse_sonnet_response(logf; deembedded=deembedded,nports=nports)
    return (; response..., output_dir=abspath(output_dir), logf)
end

end
