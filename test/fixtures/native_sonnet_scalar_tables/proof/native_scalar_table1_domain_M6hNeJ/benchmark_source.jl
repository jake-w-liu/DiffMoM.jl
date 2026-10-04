# Independent native off-node / out-of-domain CSV controls, unregistered.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_scalar_table1_aOrl1r")
    evidence=evidence_directory("native_scalar_table1_domain")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    original=read(joinpath(origin,"escaped.son"),String)
    csv=joinpath(origin,"loss.csv")
    em=find_em();em===nothing && error("installed native Sonnet required")
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("artifact"=>evidence,
        "scope"=>"actual native table1 interpolation/domain; independently declared Rs11 at midpoint; no STF interpolation inference",
        "parent_source_sha256"=>bytes2hex(sha256(read(joinpath(origin,"escaped.son")))),
        "dependency_sha256"=>bytes2hex(sha256(read(csv))),"runs"=>rows)
    responses=Dict{String,Matrix{ComplexF64}}()
    for (tag,frequency,loss) in (("literal_midpoint",5.5,"11"),
            ("table_midpoint",5.5,"Loss"),("table_below",.5,"Loss"),("table_above",11.,"Loss"))
        text=replace(original,"FREQ Y AN SWEEP 1.0 10.0 9.0"=>"FREQ Y AN SWEEP $frequency")
        if loss!="Loss"
            text=replace(text,"NOR \"Loss\" .5 .001 SRVY"=>"NOR $loss .5 .001 SRVY")
            text=join(filter(line->!startswith(line,"VALVAR Loss "),split(text,'\n')),'\n')
        end
        source=joinpath(evidence,tag*".son");write(source,text)
        row=Dict{String,Any}("tag"=>tag,"frequency_Hz"=>frequency*1e9,
            "source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false,dependencies=loss=="Loss" ? [csv] : String[])
            data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
            data.frequencies==[frequency*1e9] || error("unexpected native frequency")
            responses[tag]=only(data.s)
            row["s_real"]=vec(real.(only(data.s)));row["s_imag"]=vec(imag.(only(data.s)))
            if tag=="table_midpoint"
                row["native_literal_delta_s"]=maximum(abs,only(data.s)-responses["literal_midpoint"])
            end
            row["status"]="SIMULATED"
        catch err
            row["error"]=sprint(showerror,err)
        end
        push!(rows,row)
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println(tag," ",row)
    end
    println(evidence)
end
main()
