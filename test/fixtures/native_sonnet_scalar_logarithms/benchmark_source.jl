# Validation-only literal controls for documented native scalar functions.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_normal_resistivity_mz8D3l")
    evidence=evidence_directory("native_scalar_logarithms")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    primary="C:/Program Files/Sonnet Software/18.53/doc/Sonnet_Suites/FunctionsandOperators.html"
    cp(primary,joinpath(evidence,"FunctionsandOperators.html"))
    helper=joinpath(@__DIR__,"sonnet_reference.jl");helper_sha=bytes2hex(sha256(read(helper)))
    em=find_em();em===nothing && error("actual installed Sonnet required")
    original=read(joinpath(origin,"srvy_2","srvy_2.son"),String)
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    expected=planar_read_touchstone(joinpath(origin,"sigma_5e5","native.s2p"))
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("artifact"=>evidence,"scope"=>"documented native logarithms versus independently declared Rs2 material; raw R50; validation only",
        "primary_sha256"=>bytes2hex(sha256(read(primary))),"reference_helper_sha256"=>helper_sha,
        "literal_reference_sha256"=>bytes2hex(sha256(read(joinpath(origin,"sigma_5e5","native.s2p")))),
        "full_s_gate"=>.005,"runs"=>rows)
    for (tag,expression) in (("ln_positive","ln(exp(2))"),("ln_negative","ln(-exp(2))"),
            ("log10_positive","log10(100)"),("log10_negative","log10(-100)"),("log_alias","log(exp(2))"))
        source=joinpath(evidence,tag*".son")
        text=replace(original,material=>"MET \"Sheet\" 0 NOR \"Loss\" .5 .001 SRVY\r",
            "LORGN "=>"VALVAR Loss SRES \"$expression\" \"Manufactured scalar-function control\"\r\nLORGN ")
        write(source,text)
        row=Dict{String,Any}("tag"=>tag,"expression"=>expression,
            "source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            p=read_sonnet_project(source)
            try row["current_reader_value"]=sonnet_variable_value(p,"Loss";freq=1e9)
            catch err;row["current_reader_error"]=sprint(showerror,err);end
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false)
            data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
            data.frequencies==expected.frequencies || error("unexpected native frequency set")
            row["native_literal_delta_s"]=[maximum(abs,a-b) for (a,b) in zip(data.s,expected.s)]
            row["native_literal_gate_pass"]=all(<(.005),row["native_literal_delta_s"])
            row["native_variable_lines"]=filter(line->occursin("Loss",line),readlines(joinpath(native.output_dir,"engine_stdout.log")))
            row["s_real"]=[vec(real.(s)) for s in data.s];row["s_imag"]=[vec(imag.(s)) for s in data.s]
            row["status"]="SIMULATED"
        catch err
            row["error"]=sprint(showerror,err)
        end
        push!(rows,row)
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println(tag," ",row)
    end
    bytes2hex(sha256(read(helper)))==helper_sha || error("registered reference helper changed during validation")
    println(evidence)
end
main()
