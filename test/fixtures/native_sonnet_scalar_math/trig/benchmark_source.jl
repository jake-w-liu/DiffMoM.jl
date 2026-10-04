# Validation-only installed primary trig/inverse-trig unit/physical oracle.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_normal_resistivity_mz8D3l")
    evidence=evidence_directory("native_scalar_trig_units")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    primary="C:/Program Files/Sonnet Software/18.53/doc/Sonnet_Suites/FunctionsandOperators.html"
    cp(primary,joinpath(evidence,"FunctionsandOperators.html"))
    helper=joinpath(@__DIR__,"sonnet_reference.jl");helper_sha=bytes2hex(sha256(read(helper)))
    em=find_em();em===nothing && error("actual native Sonnet required")
    original=read(joinpath(origin,"srvy_2","srvy_2.son"),String)
    original=join(filter(line->!startswith(line,"VAR Rho "),split(original,'\n')),'\n')
    original=replace(original,"FREQ Y AN SWEEP 1.0 10.0 9.0"=>"FREQ Y AN SWEEP 1.0")
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    cases=(("sin_one","sin(1)",sin(1.)),("cos_one","cos(1)",cos(1.)),("tan_one","tan(1)",tan(1.)),
        ("asin_half","asin(.5)",pi/6),("acos_half","acos(.5)",pi/3),("atan_one","atan(1)",pi/4),
        ("atan2_quadrant2","atan2(1,-1)",3pi/4),("atan2_quadrant3","abs(atan2(-1,-1))",3pi/4),
        ("sinh_one","sinh(1)",sinh(1.)),("cosh_one","cosh(1)",cosh(1.)),("tanh_one","tanh(1)",tanh(1.)),
        ("asinh_one","asinh(1)",log(1+sqrt(2.))),("acosh_two","acosh(2)",log(2+sqrt(3.))),
        ("atanh_half","atanh(.5)",log(3.)/2))
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("artifact"=>evidence,"scope"=>"documented native trig/inverse/hyperbolic real functions versus independent literal materials under declared DIM ANG DEG/RAD",
        "primary_sha256"=>bytes2hex(sha256(read(primary))),"reference_helper_sha256"=>helper_sha,
        "evaluator_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarSonnetIO.jl")))),
        "full_s_gate"=>.005,"literal_native_gate"=>2e-13,"runs"=>rows)
    for (tag,expression,value) in cases
        literal_s=nothing
        for mode in ("literal","DEG","RAD")
            variant=tag*"_"*lowercase(mode)
            text=replace(original,material=>"MET \"Sheet\" 0 NOR "*(mode=="literal" ? string(value) : "\"Loss\"")*" .5 .001 SRVY\r")
            if mode!="literal"
                text=replace(text,"ANG DEG"=>"ANG $mode","LORGN "=>"VALVAR Loss SRES \"$expression\" \"Trig unit coupon\"\r\nLORGN ")
            end
            source=joinpath(evidence,variant*".son");write(source,text)
            row=Dict{String,Any}("tag"=>variant,"expression"=>expression,"declared_angle_unit"=>mode,
                "independent_literal_ohm_square"=>value,"source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
            try
                p=read_sonnet_project(source)
                if mode!="literal"
                    try row["current_reader_value"]=sonnet_variable_value(p,"Loss";freq=1e9)
                    catch err;row["current_reader_error"]=sprint(showerror,err);end
                end
                native=reference_run(em,source;output_dir=joinpath(evidence,variant),deembedded=false)
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
                data.frequencies==[1e9] || error("unexpected native sweep")
                s=only(data.s)
                if mode=="literal";literal_s=s
                else
                    literal_s===nothing && error("independent literal unavailable")
                    row["native_literal_delta_s"]=maximum(abs,s-literal_s)
                    row["native_literal_gate_pass"]=row["native_literal_delta_s"]<=2e-13
                    if haskey(row,"current_reader_value")
                        solved=solve_sonnet_project(p,1e9;raw=true,mx=160,my=160,method=:dense_fft)
                        row["current_reader_native_delta_s"]=maximum(abs,solved.s-s)
                        row["current_reader_native_gate_pass"]=row["current_reader_native_delta_s"]<.005
                    end
                end
                row["native_variable_lines"]=filter(line->occursin("Loss",line),readlines(joinpath(native.output_dir,"engine_stdout.log")))
                row["native_stderr"]=read(joinpath(native.output_dir,"engine_stderr.log"),String)
                row["s_real"]=vec(real.(s));row["s_imag"]=vec(imag.(s));row["status"]="SIMULATED"
            catch err
                row["error"]=sprint(showerror,err)
            end
            push!(rows,row)
            open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
            println(variant," ",get(row,"native_literal_delta_s",get(row,"error","literal retained")),
                " ",get(row,"current_reader_error",get(row,"current_reader_native_delta_s","")))
        end
    end
    bytes2hex(sha256(read(helper)))==helper_sha || error("registered native helper changed during validation")
    println(evidence)
end
main()
