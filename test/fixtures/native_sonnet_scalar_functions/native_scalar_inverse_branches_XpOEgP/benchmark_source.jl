# Validation only: native documented general/complex functions and arithmetic.
# Every supported hypothesis has an independently declared 2 ohm/square oracle.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_normal_resistivity_mz8D3l")
    evidence=evidence_directory("native_scalar_inverse_branches")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    primary="C:/Program Files/Sonnet Software/18.53/doc/Sonnet_Suites/FunctionsandOperators.html"
    cp(primary,joinpath(evidence,"FunctionsandOperators.html"))
    helper=joinpath(@__DIR__,"sonnet_reference.jl");helper_sha=bytes2hex(sha256(read(helper)))
    em=find_em();em===nothing && error("actual native Sonnet required")
    original=read(joinpath(origin,"srvy_2","srvy_2.son"),String)
    original=join(filter(line->!startswith(line,"VAR Rho "),split(original,'\n')),'\n')
    original=replace(original,"FREQ Y AN SWEEP 1.0 10.0 9.0"=>"FREQ Y AN SWEEP 1.0")
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    cases=(
        ("asin_real_upper","real(asin(cmplx(0.2,0.3)))/(0.19245144716045312)*2","DEG"),
        ("asin_imag_upper","imag(asin(cmplx(0.2,0.3)))/(0.30107353944664245)*2","DEG"),
        ("asin_real_lower","real(asin(cmplx(0.2,-0.3)))/(0.19245144716045312)*2","DEG"),
        ("asin_imag_lower","imag(asin(cmplx(0.2,-0.3)))/(-0.30107353944664245)*2","DEG"),
        ("acos_real_upper","real(acos(cmplx(0.2,0.3)))/(1.3783448796344435)*2","DEG"),
        ("acos_imag_upper","imag(acos(cmplx(0.2,0.3)))/(-0.30107353944664245)*2","DEG"),
        ("acos_real_lower","real(acos(cmplx(0.2,-0.3)))/(1.3783448796344435)*2","DEG"),
        ("acos_imag_lower","imag(acos(cmplx(0.2,-0.3)))/(0.30107353944664245)*2","DEG"),
        ("atan_real_upper","real(atan(cmplx(0.2,0.3)))/(0.21547449370018829)*2","DEG"),
        ("atan_imag_upper","imag(atan(cmplx(0.2,0.3)))/(0.29574992023641433)*2","DEG"),
        ("atan_real_lower","real(atan(cmplx(0.2,-0.3)))/(0.21547449370018829)*2","DEG"),
        ("atan_imag_lower","imag(atan(cmplx(0.2,-0.3)))/(-0.29574992023641433)*2","DEG"),
        ("asinh_real_upper","real(asinh(cmplx(0.2,0.3)))/(0.2077263762481231)*2","DEG"),
        ("asinh_imag_upper","imag(asinh(cmplx(0.2,0.3)))/(0.29803439984315466)*2","DEG"),
        ("asinh_real_lower","real(asinh(cmplx(0.2,-0.3)))/(0.2077263762481231)*2","DEG"),
        ("asinh_imag_lower","imag(asinh(cmplx(0.2,-0.3)))/(-0.29803439984315466)*2","DEG"),
        ("acosh_real_upper","real(acosh(cmplx(0.2,0.3)))/(0.30107353944664245)*2","DEG"),
        ("acosh_imag_upper","imag(acosh(cmplx(0.2,0.3)))/(1.3783448796344435)*2","DEG"),
        ("acosh_real_lower","real(acosh(cmplx(0.2,-0.3)))/(0.30107353944664245)*2","DEG"),
        ("acosh_imag_lower","imag(acosh(cmplx(0.2,-0.3)))/(-1.3783448796344435)*2","DEG"),
        ("atanh_real_upper","real(atanh(cmplx(0.2,0.3)))/(0.18499462006101108)*2","DEG"),
        ("atanh_imag_upper","imag(atanh(cmplx(0.2,0.3)))/(0.30187466669871815)*2","DEG"),
        ("atanh_real_lower","real(atanh(cmplx(0.2,-0.3)))/(0.18499462006101108)*2","DEG"),
        ("atanh_imag_lower","imag(atanh(cmplx(0.2,-0.3)))/(-0.30187466669871815)*2","DEG"),
        ("asin_cut_real_upper","real(asin(cmplx(2.0,0.1)))/(1.5131571227434844)*2","DEG"),
        ("asin_cut_imag_upper","imag(asin(cmplx(2.0,0.1)))/(1.3188765472109005)*2","DEG"),
        ("asin_cut_real_lower","real(asin(cmplx(2.0,-0.1)))/(1.5131571227434844)*2","DEG"),
        ("asin_cut_imag_lower","imag(asin(cmplx(2.0,-0.1)))/(-1.3188765472109005)*2","DEG"),
        ("acos_cut_real_upper","real(acos(cmplx(2.0,0.1)))/(0.057639204051412353)*2","DEG"),
        ("acos_cut_imag_upper","imag(acos(cmplx(2.0,0.1)))/(-1.3188765472109005)*2","DEG"),
        ("acos_cut_real_lower","real(acos(cmplx(2.0,-0.1)))/(0.057639204051412353)*2","DEG"),
        ("acos_cut_imag_lower","imag(acos(cmplx(2.0,-0.1)))/(1.3188765472109005)*2","DEG"),
        ("acosh_cut_real_upper","real(acosh(cmplx(2.0,0.1)))/(1.3188765472109005)*2","DEG"),
        ("acosh_cut_imag_upper","imag(acosh(cmplx(2.0,0.1)))/(0.057639204051412353)*2","DEG"),
        ("acosh_cut_real_lower","real(acosh(cmplx(2.0,-0.1)))/(1.3188765472109005)*2","DEG"),
        ("acosh_cut_imag_lower","imag(acosh(cmplx(2.0,-0.1)))/(-0.057639204051412353)*2","DEG"),
        ("atanh_cut_real_upper","real(atanh(cmplx(2.0,0.1)))/(0.54709618519176961)*2","DEG"),
        ("atanh_cut_imag_upper","imag(atanh(cmplx(2.0,0.1)))/(1.5376224984884392)*2","DEG"),
        ("atanh_cut_real_lower","real(atanh(cmplx(2.0,-0.1)))/(0.54709618519176961)*2","DEG"),
        ("atanh_cut_imag_lower","imag(atanh(cmplx(2.0,-0.1)))/(-1.5376224984884392)*2","DEG"),
        ("asin_axis_positive","imag(asin(2.0))/(1.3169578969248166)*2","DEG"),
        ("asin_axis_negative","imag(asin(-2.0))/(1.3169578969248166)*2","DEG"),
        ("acos_axis_positive","imag(acos(2.0))/(-1.3169578969248166)*2","DEG"),
        ("acos_axis_negative","imag(acos(-2.0))/(-1.3169578969248166)*2","DEG"),
        ("acosh_axis_negative","imag(acosh(-2.0))/(3.1415926535897931)*2","DEG"),
        ("atanh_axis_positive","imag(atanh(2.0))/(1.5707963267948966)*2","DEG"),
        ("atanh_axis_negative","imag(atanh(-2.0))/(1.5707963267948966)*2","DEG"),
        ("sqrt_axis","imag(sqrt(-4))","DEG"),
        ("power_plus_chain","2^+3^2/256","DEG"),
        ("power_minus_axis","imag((-1)^.5)*2","DEG"))
    rows=Dict{String,Any}[];literal_s=nothing
    report=Dict{String,Any}("artifact"=>evidence,
        "scope"=>"complex inverse branch hypotheses versus separately declared literal; numerical normalizers from independent Python cmath, failures retained",
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
            data.frequencies==[1e9] || error("unexpected native sweep")
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
