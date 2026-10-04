# Validation only: field grammar and real projection, never assume inline math.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference
function main()
    origin=joinpath(@__DIR__,"../../data/sonnet_validation/native_normal_resistivity_mz8D3l/srvy_2/srvy_2.son")
    text=read(origin,String)
    text=join(filter(line->!startswith(line,"VAR Rho "),split(text,'\n')),'\n')
    text=replace(text,"FREQ Y AN SWEEP 1.0 10.0 9.0"=>"FREQ Y AN SWEEP 1.0")
    metal=only(filter(x->startswith(x,"MET \"Sheet\""),split(text,'\n')))
    evidence=evidence_directory("native_remaining_complex_boundaries");em=find_em()
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    module_paths=[joinpath(@__DIR__,"../../src/planar/PlanarSonnetIO.jl"),joinpath(@__DIR__,"../../src/planar/PlanarSonnetScalarFiles.jl"),joinpath(@__DIR__,"sonnet_reference.jl")]
    module_before=Dict(basename(p)=>bytes2hex(sha256(read(p))) for p in module_paths)
    dependencies=[joinpath(evidence,"keys1.csv"),joinpath(evidence,"keys2.csv")]
    write(dependencies[1],"-10,1\n0,11\n10,21\n")
    write(dependencies[2],",-10,0,10\n-10,1,21,41\n0,11,31,51\n10,21,41,61\n")
    rows=Dict{String,Any}[];standards=Dict{Float64,Matrix{ComplexF64}}()
    cases=(
        ("literal0","0","",0),
        ("literal1","0.42920367320510344","",0.42920367320510344),
        ("literal2","1","",1),
        ("literal3","1.5707963267948966","",1.5707963267948966),
        ("literal4","2","",2),
        ("literal5","3","",3),
        ("literal6","3.1415926535897931","",3.1415926535897931),
        ("literal7","8","",8),
        ("literal8","11","",11),
        ("literal9","14","",14),
        ("literal10","26","",26),
        ("literal11","36","",36),
        ("literal12","37","",37),
        ("h2p_complex_real","\"Loss\"","real(h2p(cmplx(3000000000,4000000000)))",3),
        ("p2h_complex_real","\"Loss\"","real(p2h(cmplx(3,4)))/1000000000",3),
        ("m2p_complex_real_correct_unit","\"Loss\"","real(m2p(cmplx(.003,.004)))",3),
        ("p2m_complex_real","\"Loss\"","real(p2m(cmplx(3,4)))/.001",3),
        ("m2p_complex_negative","\"Loss\"","4+real(m2p(cmplx(-.003,.004)))",1),
        ("h2p_pure_imaginary","\"Loss\"","real(h2p(cmplx(0,4000000000)))",0),
        ("atan2_zero_x_positive_real","\"Loss\"","real(atan2(cmplx(2,3),0))",1.5707963267948966),
        ("atan2_zero_x_positive_imag","\"Loss\"","imag(atan2(cmplx(2,3),0))",3),
        ("atan2_zero_x_negative_real","\"Loss\"","2+real(atan2(cmplx(-2,3),0))",0.42920367320510344),
        ("atan2_zero_x_negative_imag","\"Loss\"","imag(atan2(cmplx(-2,3),0))",3),
        ("atan2_zero_x_pure_imaginary_real","\"Loss\"","real(atan2(cmplx(0,3),0))",0),
        ("atan2_zero_x_pure_imaginary_imag","\"Loss\"","imag(atan2(cmplx(0,3),0))",3),
        ("atan2_zero_x_negative_pure_imaginary_real","\"Loss\"","real(atan2(cmplx(0,-3),0))",0),
        ("atan2_zero_x_negative_pure_imaginary_imag","\"Loss\"","4+imag(atan2(cmplx(0,-3),0))",1),
        ("atan2_zero_both_real","\"Loss\"","real(atan2(0,0))",0),
        ("atan2_zero_both_imag","\"Loss\"","imag(atan2(0,0))",0),
        ("atan2_negative_zero_x","\"Loss\"","imag(atan2(cmplx(1,2),-0))",2),
        ("atan2_pure_y_negative_x_real","\"Loss\"","real(atan2(cmplx(0,1),-2))",3.1415926535897931),
        ("atan2_negative_pure_y_negative_x_real","\"Loss\"","real(atan2(cmplx(0,-1),-2))",3.1415926535897931),
        ("table1_complex","\"Loss\"","table1(\"keys1.csv\",cmplx(3,4))",14),
        ("table1_negative_complex","\"Loss\"","table1(\"keys1.csv\",cmplx(-3,4))",8),
        ("table1_pure_imaginary","\"Loss\"","table1(\"keys1.csv\",cmplx(0,5))",11),
        ("table2_complex","\"Loss\"","table2(\"keys2.csv\",cmplx(3,4),cmplx(1,1))",36),
        ("table2_pure_imaginary","\"Loss\"","table2(\"keys2.csv\",cmplx(0,5),cmplx(3,4))",37),
        ("table2_negative_complex","\"Loss\"","table2(\"keys2.csv\",cmplx(-3,4),cmplx(-1,2))",26))
    for (tag,field,expression,expected) in cases
        source=joinpath(evidence,tag*".son")
        source_text=replace(text,metal=>"MET \"Sheet\" 0 NOR $field .5 .001 SRVY\r")
        encoded=replace(expression,"\""=>"\\\"")
        isempty(expression) || (source_text=replace(source_text,"LORGN "=>"VALVAR Loss SRES \"$encoded\" \"Complex quantity boundary control\"\r\nLORGN "))
        write(source,source_text)
        row=Dict{String,Any}("tag"=>tag,"field"=>field,"expression"=>expression,"hypothesized_real_ohm_square"=>expected,
            "source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            p=read_sonnet_project(source)
            try row["current_value"]=sonnet_variable_value(p,isempty(expression) ? replace(field,"\""=>"") : "Loss")
            catch err;row["current_error"]=sprint(showerror,err);end
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false,dependencies)
            native_s=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
            @assert native_s.frequencies==[1e9]
            s=only(native_s.s)
            errors=read(joinpath(native.output_dir,"engine_stderr.log"),String)
            row["native_stderr"]=errors
            row["native_variable_lines"]=filter(x->occursin("Loss",x),readlines(joinpath(native.output_dir,"engine_stdout.log")))
            row["s_real"]=vec(real.(s));row["s_imag"]=vec(imag.(s));row["status"]="SIMULATED"
            if startswith(tag,"literal");standards[expected]=s
            else
                row["native_literal_delta_s"]=maximum(abs,s-standards[expected])
                row["native_literal_gate_pass"]=isempty(errors) && row["native_literal_delta_s"]<=2e-13
            end
        catch err
            row["error"]=sprint(showerror,err)
            native_errors=joinpath(evidence,tag,"engine_stderr.log")
            isfile(native_errors) && (row["native_stderr"]=read(native_errors,String))
        end
        push!(rows,row)
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,Dict("artifact"=>evidence,"runs"=>rows,"module_before"=>module_before,"module_after"=>Dict(basename(p)=>bytes2hex(sha256(read(p))) for p in module_paths)));end
        println(tag," ",get(row,"native_variable_lines",get(row,"error",""))," ",get(row,"native_literal_delta_s","")," ",get(row,"native_stderr",""))
    end
    println(evidence)
end
main()
