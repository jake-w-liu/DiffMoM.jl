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
    evidence=evidence_directory("native_complex_function_corrected");em=find_em()
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    module_paths=[joinpath(@__DIR__,"../../src/planar/PlanarSonnetIO.jl"),joinpath(@__DIR__,"../../src/planar/PlanarSonnetScalarFiles.jl"),joinpath(@__DIR__,"sonnet_reference.jl")]
    module_before=Dict(basename(p)=>bytes2hex(sha256(read(p))) for p in module_paths)
    dependencies=[joinpath(evidence,"keys1.csv"),joinpath(evidence,"keys2.csv")]
    write(dependencies[1],"-10,1\n0,11\n10,21\n")
    write(dependencies[2],",-10,0,10\n-10,1,21,41\n0,11,31,51\n10,21,41,61\n")
    rows=Dict{String,Any}[];standards=Dict{Float64,Matrix{ComplexF64}}()
    cases=(
        ("literal0","0.0030000000000000001","",0.0030000000000000001),
        ("literal1","1.5707963267948966","",1.5707963267948966),
        ("literal2","2","",2),
        ("literal3","3.5707963267948966","",3.5707963267948966),
        ("literal4","24","",24),
        ("literal5","38","",38),
        ("literal6","42","",42),
        ("m2p_original_unit_corrected","\"Loss\"","real(m2p(cmplx(.003,.004)))/1000",0.0030000000000000001),
        ("atan2_negative_real_zero_shifted","\"Loss\"","2+real(atan2(-2,0))",3.5707963267948966),
        ("atan2_negative_complex_zero_shifted","\"Loss\"","2+real(atan2(cmplx(-2,3),0))",3.5707963267948966),
        ("atan2_negative_real_zero_direct","\"Loss\"","real(atan2(-2,0))",1.5707963267948966),
        ("atan2_negative_complex_zero_direct","\"Loss\"","real(atan2(cmplx(-2,3),0))",1.5707963267948966),
        ("atan2_real_zero_imaginary","\"Loss\"","imag(atan2(2,0))",2),
        ("table2_original_complex_corrected","\"Loss\"","table2(\"keys2.csv\",cmplx(3,4),cmplx(1,1))",38),
        ("table2_original_pure_imaginary_corrected","\"Loss\"","table2(\"keys2.csv\",cmplx(0,5),cmplx(3,4))",42),
        ("table2_original_negative_complex_corrected","\"Loss\"","table2(\"keys2.csv\",cmplx(-3,4),cmplx(-1,2))",24))
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
