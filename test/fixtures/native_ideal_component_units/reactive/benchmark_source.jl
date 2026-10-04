using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    original=joinpath(@__DIR__,"..","..","test","fixtures","native_sparam_model_files","native","ideal.son")
    text=read(original,String);evidence=evidence_directory("native_ideal_reactive_units")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    em=find_em();em===nothing && error("actual installed Sonnet required")
    rows=Dict{String,Any}[];matrices=Dict{String,Matrix{ComplexF64}}()
    for (kind,base_unit,larger_unit) in (("CAP","PF","NF"),("IND","NH","UH"))
        for (variant,unit,value) in (("base",base_unit,"1"),("same_literal",larger_unit,"1"),("same_physical",larger_unit,".001"))
            tag=lowercase(kind)*"_"*variant;source=joinpath(evidence,tag*".son")
            control=replace(text,"$kind $base_unit"=>"$kind $unit",
                "TYPE IDEAL RES 16.77"=>"TYPE IDEAL $kind $value",
                "FREQ Y AN SWEEP 200 400 100"=>"FREQ Y AN SWEEP 200")
            write(source,control)
            row=Dict{String,Any}("tag"=>tag,"kind"=>kind,"dim_unit"=>unit,
                "literal_component_value"=>value,"source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
            try
                native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false)
                data=checked_native_touchstone(native,joinpath(native.output_dir,"native_device.s2p");deembedded=false,expected_z0=50.)
                matrices[tag]=only(data.s)
                row["s_real"]=vec(real.(only(data.s)));row["s_imag"]=vec(imag.(only(data.s)))
                row["status"]="SIMULATED";println(tag," S21=",only(data.s)[2,1])
            catch err
                row["error"]=sprint(showerror,err);println(tag," UNVERIFIED ",row["error"])
            end
            push!(rows,row)
        end
    end
    report=Dict{String,Any}("artifact"=>evidence,
        "scope"=>"manufactured component DIM CAP/IND native unit controls; exact raw R50 selection; no calibration fit",
        "original_source_sha256"=>bytes2hex(sha256(read(original))),"runs"=>rows)
    for kind in ("cap","ind")
        if all(haskey(matrices,kind*"_"*v) for v in ("base","same_literal","same_physical"))
            report[kind*"_same_literal_delta_s"]=maximum(abs,matrices[kind*"_base"]-matrices[kind*"_same_literal"])
            report[kind*"_same_physical_delta_s"]=maximum(abs,matrices[kind*"_base"]-matrices[kind*"_same_physical"])
        end
    end
    open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
    println(evidence)
end
main()
