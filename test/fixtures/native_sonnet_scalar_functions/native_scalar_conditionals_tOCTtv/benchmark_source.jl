# Validation-only hypotheses: comparison/conditional syntax is absent from
# the installed FunctionsandOperators page. Native warnings are not support.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_normal_resistivity_mz8D3l")
    evidence=evidence_directory("native_scalar_conditionals")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    primary="C:/Program Files/Sonnet Software/18.53/doc/Sonnet_Suites/FunctionsandOperators.html"
    cp(primary,joinpath(evidence,"FunctionsandOperators.html"))
    helper=joinpath(@__DIR__,"sonnet_reference.jl");helper_sha=bytes2hex(sha256(read(helper)))
    em=find_em();em===nothing && error("actual native Sonnet required")
    original=read(joinpath(origin,"srvy_2","srvy_2.son"),String)
    original=join(filter(line->!startswith(line,"VAR Rho "),split(original,'\n')),'\n')
    original=replace(original,"FREQ Y AN SWEEP 1.0 10.0 9.0"=>"FREQ Y AN SWEEP 1.0 3.0 2.0")
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    cases=(("greater","1+(FREQ>2000000000)",[1.,2.]),("less","1+(FREQ<2000000000)",[2.,1.]),
        ("greater_equal","1+(FREQ>=2000000000)",[1.,2.]),("less_equal","1+(FREQ<=2000000000)",[2.,1.]),
        ("equal","1+(FREQ==1000000000)",[2.,1.]),("not_equal","1+(FREQ!=1000000000)",[1.,2.]),
        ("ternary","(FREQ>2000000000) ? 2 : 1",[1.,2.]),
        ("if_function","if(FREQ>2000000000,2,1)",[1.,2.]),
        ("ifelse_function","ifelse(FREQ>2000000000,2,1)",[1.,2.]))
    rows=Dict{String,Any}[];standards=Dict{Float64,Vector{Matrix{ComplexF64}}}()
    report=Dict{String,Any}("artifact"=>evidence,"scope"=>"undocumented comparison/conditional hypotheses versus independent literals; rejected/warned fallback remains failure evidence",
        "primary_sha256"=>bytes2hex(sha256(read(primary))),"reference_helper_sha256"=>helper_sha,
        "full_s_gate"=>.005,"literal_native_gate"=>2e-13,"runs"=>rows)
    for (tag,expression,values) in (("literal1","",[1.,1.]),("literal2","",[2.,2.]),cases...)
        text=replace(original,material=>"MET \"Sheet\" 0 NOR "*(isempty(expression) ? string(first(values)) : "\"Loss\"")*" .5 .001 SRVY\r")
        isempty(expression) || (text=replace(text,"LORGN "=>"VALVAR Loss SRES \"$expression\" \"Undocumented conditional hypothesis\"\r\nLORGN "))
        source=joinpath(evidence,tag*".son");write(source,text)
        row=Dict{String,Any}("tag"=>tag,"expression"=>expression,"independent_expected_ohm_square"=>values,
            "source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            p=read_sonnet_project(source)
            if !isempty(expression)
                for f in (1e9,3e9)
                    key="current_reader_$(Int(f/1e9))GHz"
                    try row[key]=sonnet_variable_value(p,"Loss";freq=f)
                    catch err;row[key*"_error"]=sprint(showerror,err);end
                end
            end
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false)
            data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
            data.frequencies==[1e9,3e9] || error("unexpected native sweep")
            errors=read(joinpath(native.output_dir,"engine_stderr.log"),String)
            row["native_stderr"]=errors
            row["native_expression_status"]=occursin("Bad Equation",errors) || occursin("Invalid entry",errors) ? "REJECTED_WITH_FALLBACK" : "NO_EQUATION_WARNING"
            if isempty(expression);standards[first(values)]=data.s
            else
                expected=[standards[value][i] for (i,value) in enumerate(values)]
                row["native_literal_delta_s"]=[maximum(abs,a-b) for (a,b) in zip(data.s,expected)]
                row["native_literal_gate_pass"]=row["native_expression_status"]=="NO_EQUATION_WARNING" && all(<(2e-13),row["native_literal_delta_s"])
            end
            row["native_variable_lines"]=filter(line->occursin("Loss",line),readlines(joinpath(native.output_dir,"engine_stdout.log")))
            row["s_real"]=[vec(real.(s)) for s in data.s];row["s_imag"]=[vec(imag.(s)) for s in data.s]
            row["status"]="SIMULATED"
        catch err
            row["error"]=sprint(showerror,err)
        end
        push!(rows,row)
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println(tag," ",get(row,"native_expression_status",get(row,"error",""))," ",get(row,"native_literal_delta_s",[]))
    end
    bytes2hex(sha256(read(helper)))==helper_sha || error("registered reference helper changed")
    println(evidence)
end
main()
