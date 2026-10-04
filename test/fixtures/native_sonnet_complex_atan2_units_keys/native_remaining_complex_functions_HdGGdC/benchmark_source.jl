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
    evidence=evidence_directory("native_remaining_complex_functions");em=find_em()
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    module_paths=[joinpath(@__DIR__,"../../src/planar/PlanarSonnetIO.jl"),joinpath(@__DIR__,"../../src/planar/PlanarSonnetScalarFiles.jl"),joinpath(@__DIR__,"sonnet_reference.jl")]
    module_before=Dict(basename(p)=>bytes2hex(sha256(read(p))) for p in module_paths)
    rows=Dict{String,Any}[];standards=Dict{Float64,Matrix{ComplexF64}}()
    cases=(
        ("literal0","0","",0),
        ("literal1","0.067065996648669857","",0.067065996648669857),
        ("literal2","0.40235947810852507","",0.40235947810852507),
        ("literal3","0.54930614433405489","",0.54930614433405489),
        ("literal4","0.5535743588970452","",0.5535743588970452),
        ("literal5","1.0172219678978514","",1.0172219678978514),
        ("literal6","1.4119817053072521","",1.4119817053072521),
        ("literal7","1.5707963267948966","",1.5707963267948966),
        ("literal8","2.1243706856919418","",2.1243706856919418),
        ("literal9","2.184682519981993","",2.184682519981993),
        ("literal10","2.5880182946927479","",2.5880182946927479),
        ("literal11","3","",3),
        ("literal12","3.4464256411029548","",3.4464256411029548),
        ("literal13","3.450693855665945","",3.450693855665945),
        ("literal14","3.5976405218914751","",3.5976405218914751),
        ("literal15","4","",4),
        ("h2p_complex_imag","\"Loss\"","imag(h2p(cmplx(3000000000,4000000000)))",4),
        ("p2h_complex_imag","\"Loss\"","imag(p2h(cmplx(3,4)))/1000000000",4),
        ("m2p_complex_imag","\"Loss\"","imag(m2p(cmplx(.003,.004)))/1000",4),
        ("p2m_complex_imag","\"Loss\"","imag(p2m(cmplx(3,4)))/.000001",4),
        ("m2p_complex_real","\"Loss\"","real(m2p(cmplx(.003,.004)))/1000",3),
        ("atan2_ypositive_real","\"Loss\"","0+real(atan2(cmplx(1,1),cmplx(2,0)))",0.5535743588970452),
        ("atan2_ypositive_imag","\"Loss\"","0+imag(atan2(cmplx(1,1),cmplx(2,0)))",0.40235947810852507),
        ("atan2_xpositive_real","\"Loss\"","0+real(atan2(cmplx(2,0),cmplx(1,1)))",1.0172219678978514),
        ("atan2_xpositive_imag","\"Loss\"","4+imag(atan2(cmplx(2,0),cmplx(1,1)))",3.5976405218914751),
        ("atan2_xnegative_real","\"Loss\"","0+real(atan2(cmplx(1,1),cmplx(-2,0)))",2.5880182946927479),
        ("atan2_xnegative_imag","\"Loss\"","4+imag(atan2(cmplx(1,1),cmplx(-2,0)))",3.5976405218914751),
        ("atan2_ynegative_real","\"Loss\"","4+real(atan2(cmplx(-1,1),cmplx(2,0)))",3.4464256411029548),
        ("atan2_ynegative_imag","\"Loss\"","0+imag(atan2(cmplx(-1,1),cmplx(2,0)))",0.40235947810852507),
        ("atan2_bothnegative_real","\"Loss\"","4+real(atan2(cmplx(-1,1),cmplx(-2,0)))",1.4119817053072521),
        ("atan2_bothnegative_imag","\"Loss\"","4+imag(atan2(cmplx(-1,1),cmplx(-2,0)))",3.5976405218914751),
        ("atan2_bothcomplex_xnegative_real","\"Loss\"","0+real(atan2(cmplx(3,1),cmplx(-2,1)))",2.1243706856919418),
        ("atan2_bothcomplex_xnegative_imag","\"Loss\"","4+imag(atan2(cmplx(3,1),cmplx(-2,1)))",3.5976405218914751),
        ("atan2_xnegative_lower_real","\"Loss\"","0+real(atan2(cmplx(3,1),cmplx(-2,-1)))",2.184682519981993),
        ("atan2_xnegative_lower_imag","\"Loss\"","0+imag(atan2(cmplx(3,1),cmplx(-2,-1)))",0.067065996648669857),
        ("atan2_pure_y_real","\"Loss\"","0+real(atan2(cmplx(0,1),cmplx(2,0)))",0),
        ("atan2_pure_y_imag","\"Loss\"","0+imag(atan2(cmplx(0,1),cmplx(2,0)))",0.54930614433405489),
        ("atan2_pure_x_real","\"Loss\"","0+real(atan2(cmplx(2,0),cmplx(0,1)))",1.5707963267948966),
        ("atan2_pure_x_imag","\"Loss\"","4+imag(atan2(cmplx(2,0),cmplx(0,1)))",3.450693855665945),
        ("atan2_complex_y_zero_x","\"Loss\"","real(atan2(cmplx(1,1),0))",1.5707963267948966),
        ("atan2_complex_y_zero_x_imag","\"Loss\"","imag(atan2(cmplx(1,1),0))",0))
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
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,Dict("artifact"=>evidence,"runs"=>rows,"module_before"=>module_before,"module_after"=>Dict(basename(p)=>bytes2hex(sha256(read(p))) for p in module_paths)));end
        println(tag," ",get(row,"native_variable_lines",get(row,"error",""))," ",get(row,"native_literal_delta_s","")," ",get(row,"native_stderr",""))
    end
    println(evidence)
end
main()
