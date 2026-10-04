# Validation-only followup to the preserved phase-branch and power candidates.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_normal_resistivity_mz8D3l")
    evidence=evidence_directory("native_scalar_function_phase")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    primary="C:/Program Files/Sonnet Software/18.53/doc/Sonnet_Suites/FunctionsandOperators.html"
    cp(primary,joinpath(evidence,"FunctionsandOperators.html"))
    helper=joinpath(@__DIR__,"sonnet_reference.jl");helper_sha=bytes2hex(sha256(read(helper)))
    em=find_em();em===nothing && error("actual native Sonnet required")
    original=read(joinpath(origin,"srvy_2","srvy_2.son"),String)
    original=join(filter(line->!startswith(line,"VAR Rho "),split(original,'\n')),'\n')
    original=replace(original,"LORGN "=>"VALVAR neg SRES \"-1\" \"Native phase variable\"\r\nLORGN ")
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    cases=(("phase_cos","2+deg(cos(3.141592653589793))/180",1.),
        ("phase_sin","2+deg(sin(-3.141592653589793/2))/180",1.),
        ("phase_conj","2+deg(conj(-1))/180",3.),
        ("phase_real","2+deg(real(-1))/180",3.),
        ("phase_imag","2+deg(imag(cmplx(0,-1)))/180",3.),
        ("phase_subtract","2+deg(2-3)/180",3.),
        ("phase_variable","2+deg(neg)/180",1.))
    rows=Dict{String,Any}[];literals=Dict{Float64,Vector{Matrix{ComplexF64}}}()
    report=Dict{String,Any}("artifact"=>evidence,"scope"=>"explicit power-parenthesis controls and phase quadrant/negative-axis hypotheses; failures retained",
        "primary_sha256"=>bytes2hex(sha256(read(primary))),"reference_helper_sha256"=>helper_sha,
        "evaluator_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarSonnetIO.jl")))),
        "full_s_gate"=>.005,"literal_native_gate"=>2e-13,"runs"=>rows)
    for (tag,expression,value) in (("literal_quarter","",.25),("literal_one","",1.),("literal_two","",2.),("literal_three","",3.),cases...)
        text=replace(original,material=>"MET \"Sheet\" 0 NOR "*(isempty(expression) ? string(value) : "\"Loss\"")*" .5 .001 SRVY\r")
        isempty(expression) || (text=replace(text,"LORGN "=>"VALVAR Loss SRES \"$expression\" \"Native phase and power control\"\r\nLORGN "))
        source=joinpath(evidence,tag*".son");write(source,text)
        row=Dict{String,Any}("tag"=>tag,"expression"=>expression,"hypothesized_literal_ohm_square"=>value,
            "source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            p=read_sonnet_project(source)
            isempty(expression) || try row["current_reader_value"]=sonnet_variable_value(p,"Loss";freq=1e9)
                catch err;row["current_reader_error"]=sprint(showerror,err);end
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false)
            data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
            data.frequencies==[1e9,1e10] || error("unexpected native sweep")
            errors=read(joinpath(native.output_dir,"engine_stderr.log"),String)
            row["native_stderr"]=errors
            row["native_expression_status"]=occursin("Bad Equation",errors) || occursin("Invalid entry",errors) ? "REJECTED_WITH_FALLBACK" : "NO_EQUATION_WARNING"
            if isempty(expression);literals[value]=data.s
            else
                row["native_literal_delta_s"]=[maximum(abs,a-b) for (a,b) in zip(data.s,literals[value])]
                row["native_literal_gate_pass"]=row["native_expression_status"]=="NO_EQUATION_WARNING" && all(<(2e-13),row["native_literal_delta_s"])
                if haskey(row,"current_reader_value")
                    row["current_reader_native_delta_s"]=[maximum(abs,solve_sonnet_project(p,f;raw=true,mx=160,my=160,method=:dense_fft).s-s) for (f,s) in zip(data.frequencies,data.s)]
                    row["current_reader_native_gate_pass"]=all(<(.005),row["current_reader_native_delta_s"])
                end
            end
            row["native_variable_lines"]=filter(line->occursin("Loss",line),readlines(joinpath(native.output_dir,"engine_stdout.log")))
            row["s_real"]=[vec(real.(s)) for s in data.s];row["s_imag"]=[vec(imag.(s)) for s in data.s]
            row["status"]="SIMULATED"
        catch err
            row["error"]=sprint(showerror,err)
        end
        push!(rows,row)
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println(tag," ",get(row,"native_variable_lines",get(row,"error",""))," ",get(row,"native_literal_delta_s",[])," ",get(row,"current_reader_native_delta_s",[]))
    end
    bytes2hex(sha256(read(helper)))==helper_sha || error("registered reference helper changed")
    println(evidence)
end
main()
