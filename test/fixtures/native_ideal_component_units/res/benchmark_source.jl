using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference
function main()
    original=joinpath(@__DIR__,"..","..","test","fixtures","native_sparam_model_files","native","ideal.son")
    text=read(original,String);evidence=evidence_directory("native_ideal_component_units")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    em=find_em();em===nothing && error("actualinstalledSonnetrequired")
    rows=Dict{String,Any}[];matrices=Dict{String,Matrix{ComplexF64}}()
    for (tag,unit,value) in (("ohm","OH","16.77"),("kohm_same_literal","KOH","16.77"),("kohm_same_physical","KOH",".01677"))
        source=joinpath(evidence,tag*".son")
        control=replace(text,"RES OH"=>"RES $unit","TYPE IDEAL RES 16.77"=>"TYPE IDEAL RES $value", "FREQ Y AN SWEEP 200 400 100"=>"FREQ Y AN SWEEP 200")
        write(source,control);row=Dict{String,Any}("tag"=>tag,"dim_res"=>unit,"literal_component_value"=>value,"source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false)
            data=checked_native_touchstone(native,joinpath(native.output_dir,"native_device.s2p");deembedded=false,expected_z0=50.)
            matrices[tag]=only(data.s);row["s_real"]=vec(real.(only(data.s)));row["s_imag"]=vec(imag.(only(data.s)));row["status"]="SIMULATED"
            println(tag," S21=",only(data.s)[2,1])
        catch err
            row["error"]=sprint(showerror,err);println(tag," UNVERIFIED ",row["error"])
        end
        push!(rows,row)
    end
    report=Dict{String,Any}("artifact"=>evidence,"scope"=>"manufacturedcomponent literalDIMRES nativecrosscontrols; no nativecalibrationfit",
        "original_source_sha256"=>bytes2hex(sha256(read(original))),"runs"=>rows)
    if length(matrices)==3
        report["ohm_vs_kohm_same_literal_delta_s"]=maximum(abs,matrices["ohm"]-matrices["kohm_same_literal"])
        report["ohm_vs_kohm_same_physical_delta_s"]=maximum(abs,matrices["ohm"]-matrices["kohm_same_physical"])
    end
    open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
    println(evidence)
end
main()
