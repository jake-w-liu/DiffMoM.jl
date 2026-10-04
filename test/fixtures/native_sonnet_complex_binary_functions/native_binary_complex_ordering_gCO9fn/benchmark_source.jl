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
    evidence=evidence_directory("native_binary_complex_ordering");em=find_em()
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    rows=Dict{String,Any}[];standards=Dict{Float64,Matrix{ComplexF64}}()
    cases=(
        ("literal0","1","",1),
        ("literal1","2","",2),
        ("literal2","3","",3),
        ("literal3","5","",5),
        ("max_pure_imaginary","\"Loss\"","imag(max(cmplx(0,5),3))",5),
        ("min_pure_imaginary","\"Loss\"","min(cmplx(0,5),3)",3),
        ("max_negative_pure_imaginary","\"Loss\"","6+imag(max(cmplx(0,-5),3))",1),
        ("min_negative_pure_imaginary","\"Loss\"","min(cmplx(0,-5),3)",3),
        ("max_negative_complex_negative_real","\"Loss\"","6+real(max(cmplx(-3,4),-4))",2),
        ("min_negative_complex_negative_real","\"Loss\"","6+real(min(cmplx(-3,4),-4))",3),
        ("max_same_signed_magnitude","\"Loss\"","5+imag(max(cmplx(-3,4),cmplx(-3,-4)))",1),
        ("min_same_signed_magnitude","\"Loss\"","5+imag(min(cmplx(-3,4),cmplx(-3,-4)))",1))
    for (tag,field,expression,expected) in cases
        source=joinpath(evidence,tag*".son")
        source_text=replace(text,metal=>"MET \"Sheet\" 0 NOR $field .5 .001 SRVY\r")
        isempty(expression) || (source_text=replace(source_text,"LORGN "=>"VALVAR Loss SRES \"$expression\" \"Complex quantity boundary control\"\r\nLORGN "))
        write(source,source_text)
        row=Dict{String,Any}("tag"=>tag,"field"=>field,"expression"=>expression,"hypothesized_real_ohm_square"=>expected,
            "source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            p=read_sonnet_project(source)
            try row["current_value"]=sonnet_variable_value(p,isempty(expression) ? replace(field,"\""=>"") : "Loss")
            catch err;row["current_error"]=sprint(showerror,err);end
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false)
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
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,Dict("artifact"=>evidence,"runs"=>rows));end
        println(tag," ",get(row,"native_variable_lines",get(row,"error",""))," ",get(row,"native_literal_delta_s","")," ",get(row,"native_stderr",""))
    end
    println(evidence)
end
main()
