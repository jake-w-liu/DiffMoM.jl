# Literal native controls for documented real and complex scalar functions.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference
function main()
    original=read(joinpath(@__DIR__,"../../data/sonnet_validation/native_normal_resistivity_mz8D3l/srvy_2/srvy_2.son"),String)
    original=join(filter(line->!startswith(line,"VAR Rho "),split(original,'\n')),'\n')
    original=replace(original,"FREQ Y AN SWEEP 1.0 10.0 9.0"=>"FREQ Y AN SWEEP 1.0")
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    evidence=evidence_directory("native_scalar_math")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    primary="C:/Program Files/Sonnet Software/18.53/doc/Sonnet_Suites/FunctionsandOperators.html"
    cp(primary,joinpath(evidence,"FunctionsandOperators.html"))
    em=find_em();em===nothing && error("actual native Sonnet required")
    cases=(("fmod_positive","fmod(5.5,2)",1.5),("fmod_negative","abs(fmod(-5.5,2))",1.5),
        ("int_positive","int(2.8)",2.),("int_negative","abs(int(-2.8))",2.),
        ("max","max(2,3)",3.),("min","min(2,3)",2.),("hypot","hypot(3,4)",5.),
        ("real","real(cmplx(2,3))",2.),("imag","imag(cmplx(2,3))",3.),
        ("mag","mag(cmplx(3,4))",5.),("abs","abs(cmplx(3,4))",5.),
        ("conj","abs(imag(conj(cmplx(2,3))))",3.),
        ("deg","deg(cmplx(1,1))",45.),("rad","rad(cmplx(1,1))",pi/4),
        ("ln_complex","ln(cmplx(3,4))",log(5.)),
        ("sqrt_complex","imag(sqrt(cmplx(-1,0)))",1.),
        ("sqrt_negative","imag(sqrt(-1))",1.),
        ("complex_power","real(cmplx(0,1)^2)+3",2.),
        ("exp_complex","real(exp(cmplx(1,1)))",exp(1.)*cos(1.)))
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("artifact"=>evidence,"scope"=>"documented native scalar real and complex functions versus independently declared literal DC sheet materials",
        "primary_sha256"=>bytes2hex(sha256(read(primary))),"literal_native_gate"=>2e-13,"runs"=>rows)
    for (tag,expression,value) in cases
        literal_s=nothing
        for mode in ("literal","expression")
            variant=tag*"_"*mode
            text=replace(original,material=>"MET \"Sheet\" 0 NOR "*(mode=="literal" ? string(value) : "\"Loss\"")*" .5 .001 SRVY\r")
            mode=="literal" || (text=replace(text,"LORGN "=>"VALVAR Loss SRES \"$expression\" \"Native mathematical function control\"\r\nLORGN "))
            source=joinpath(evidence,variant*".son");write(source,text)
            row=Dict{String,Any}("tag"=>variant,"expression"=>expression,"independent_literal_ohm_square"=>value,"source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
            try
                p=read_sonnet_project(source)
                if mode!="literal"
                    try row["current_reader_value"]=sonnet_variable_value(p,"Loss")
                    catch err;row["current_reader_error"]=sprint(showerror,err);end
                end
                native=reference_run(em,source;output_dir=joinpath(evidence,variant),deembedded=false)
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
                data.frequencies==[1e9] || error("unexpected native sweep")
                s=only(data.s)
                if mode=="literal";literal_s=s
                else
                    literal_s===nothing && error("independent native literal unavailable")
                    row["native_literal_delta_s"]=maximum(abs,s-literal_s)
                    row["native_literal_gate_pass"]=row["native_literal_delta_s"]<=2e-13
                end
                row["native_stderr"]=read(joinpath(native.output_dir,"engine_stderr.log"),String)
                row["native_variable_lines"]=filter(line->occursin("Loss",line),readlines(joinpath(native.output_dir,"engine_stdout.log")))
                row["s_real"]=vec(real.(s));row["s_imag"]=vec(imag.(s));row["status"]="SIMULATED"
            catch err
                row["error"]=sprint(showerror,err)
            end
            push!(rows,row)
            open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
            println(variant," ",get(row,"native_literal_delta_s",get(row,"error","literal retained"))," ",get(row,"current_reader_error",""))
        end
    end
    println(evidence)
end
main()
