# Validation only: native documented general/complex functions and arithmetic.
# Every supported hypothesis has an independently declared 2 ohm/square oracle.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_normal_resistivity_mz8D3l")
    evidence=evidence_directory("native_scalar_math_branches")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    primary="C:/Program Files/Sonnet Software/18.53/doc/Sonnet_Suites/FunctionsandOperators.html"
    cp(primary,joinpath(evidence,"FunctionsandOperators.html"))
    helper=joinpath(@__DIR__,"sonnet_reference.jl");helper_sha=bytes2hex(sha256(read(helper)))
    em=find_em();em===nothing && error("actual native Sonnet required")
    original=read(joinpath(origin,"srvy_2","srvy_2.son"),String)
    original=join(filter(line->!startswith(line,"VAR Rho "),split(original,'\n')),'\n')
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    cases=(("int_signed","4+int(-2.9)","DEG"),
        ("fmod_signed","4+fmod(-5,3)","DEG"),
        ("fmod_divisor_negative","fmod(5,-3)","DEG"),
        ("fmod_both_negative","4+fmod(-5,-3)","DEG"),
        ("sqrt_negative_signed","imag(sqrt(-4))+4","DEG"),
        ("asin_branch","imag(asin(2))/ln(2+sqrt(3))+1","DEG"),
        ("acos_branch","imag(acos(2))/ln(2+sqrt(3))+3","DEG"),
        ("acosh_branch","3+imag(acosh(-2))/3.141592653589793","DEG"),
        ("atanh_branch","3+imag(atanh(2))/(3.141592653589793/2)","DEG"),
        ("complex_power_poszero","imag(cmplx(-1,0)^.5)*2","DEG"),
        ("complex_power_negzero","4+imag(cmplx(-1,-0)^.5)*2","DEG"),
        ("sin_complex","imag(sin(cmplx(0,1)))/sinh(1)*2","DEG"),
        ("cos_complex","real(cos(cmplx(0,1)))/cosh(1)*2","DEG"),
        ("tan_complex","imag(tan(cmplx(0,1)))/tanh(1)*2","DEG"),
        ("sinh_complex","imag(sinh(cmplx(0,1)))/sin(1)*2","DEG"),
        ("cosh_complex","real(cosh(cmplx(0,1)))/cos(1)*2","DEG"),
        ("tanh_complex","imag(tanh(cmplx(0,1)))/tan(1)*2","DEG"),
        ("asinh_complex","imag(asinh(cmplx(0,.5)))/(3.141592653589793/6)*2","DEG"),
        ("ln_complex","ln(cmplx(0,exp(2)))","DEG"),
        ("log10_complex","log10(cmplx(0,100))","DEG"),
        ("db10_complex","db10(cmplx(0,100))/10","DEG"),
        ("db20_complex","db20(cmplx(0,10))/10","DEG"),
        ("exp_complex","imag(exp(cmplx(0,3.141592653589793/2)))+1","DEG"),
        ("signed_power_chain","2^-3^2*128","DEG"),
        ("signed_power_left","(2^-3)^2*128","DEG"),
        ("signed_power_right","2^(-3^2)*1024","DEG"))
    rows=Dict{String,Any}[];literal_s=nothing
    report=Dict{String,Any}("artifact"=>evidence,
        "scope"=>"documented complex/signed mathematical branch hypotheses versus independently declared literal material; failures retained",
        "primary_sha256"=>bytes2hex(sha256(read(primary))),"reference_helper_sha256"=>helper_sha,
        "evaluator_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarSonnetIO.jl")))),
        "full_s_gate"=>.005,"literal_native_gate"=>2e-13,"runs"=>rows)
    for (tag,expression,angle) in (("literal2","","DEG"),cases...)
        text=replace(original,material=>"MET \"Sheet\" 0 NOR "*(isempty(expression) ? "2" : "\"Loss\"")*" .5 .001 SRVY\r","ANG DEG"=>"ANG $angle")
        isempty(expression) || (text=replace(text,"LORGN "=>"VALVAR Loss SRES \"$expression\" \"Primary documented math coupon\"\r\nLORGN "))
        source=joinpath(evidence,tag*".son");write(source,text)
        row=Dict{String,Any}("tag"=>tag,"expression"=>expression,"declared_angle_unit"=>angle,
            "independent_literal_ohm_square"=>2.,"source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            p=read_sonnet_project(source)
            if !isempty(expression)
                try row["current_reader_value"]=sonnet_variable_value(p,"Loss";freq=1e9)
                catch err;row["current_reader_error"]=sprint(showerror,err);end
            end
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false)
            data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
            data.frequencies==[1e9,1e10] || error("unexpected native sweep")
            errors=read(joinpath(native.output_dir,"engine_stderr.log"),String)
            row["native_stderr"]=errors
            row["native_expression_status"]=occursin("Bad Equation",errors) || occursin("Invalid entry",errors) ? "REJECTED_WITH_FALLBACK" : "NO_EQUATION_WARNING"
            if isempty(expression);literal_s=data.s
            else
                literal_s===nothing && error("independent literal unavailable")
                row["native_literal_delta_s"]=[maximum(abs,a-b) for (a,b) in zip(data.s,literal_s)]
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
        println(tag," ",get(row,"native_expression_status",get(row,"error",""))," ",get(row,"native_literal_delta_s",[])," ",get(row,"current_reader_error",get(row,"current_reader_value","")))
    end
    bytes2hex(sha256(read(helper)))==helper_sha || error("registered reference helper changed")
    println(evidence)
end
main()
