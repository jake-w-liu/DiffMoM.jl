# Validation-only native CSV table expression probe. It does not extend the
# production expression interpreter or infer STF table interpolation.
using DiffMoM,SHA,TOML
include("sonnet_reference.jl")
using .SonnetReference

function main()
    origin=joinpath(@__DIR__,"..","..","data","sonnet_validation","native_normal_resistivity_mz8D3l")
    evidence=evidence_directory("native_scalar_table1")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    csv=joinpath(evidence,"loss.csv")
    write(csv,"! key in Hz; independent literal Rs nodes in ohms/square\n1000000000,2\n10000000000,20 ! final node\n")
    primary="C:/Program Files/Sonnet Software/18.53/doc/Sonnet_Suites/FunctionsandOperators.html"
    cp(primary,joinpath(evidence,"FunctionsandOperators.html"))
    em=find_em();em===nothing && error("actual installed Sonnet required")
    original=read(joinpath(origin,"srvy_2","srvy_2.son"),String)
    material=only(filter(line->startswith(line,"MET \"Sheet\""),split(original,'\n')))
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("artifact"=>evidence,
        "scope"=>"native table1 CSV scalar expression; separate from unproved STF XML interpolation; no production changes",
        "csv_sha256"=>bytes2hex(sha256(read(csv))),
        "primary_sha256"=>bytes2hex(sha256(read(primary))),"runs"=>rows)
    variants=(("escaped","table1(\\\"loss.csv\\\",FREQ)"),
        ("nested","table1(\"loss.csv\",FREQ)"))
    for (tag,expression) in variants
        text=replace(original,material=>"MET \"Sheet\" 0 NOR \"Loss\" .5 .001 SRVY\r",
            "LORGN "=>"VALVAR Loss SRES \"$expression\" \"CSV scalar probe\"\r\nLORGN ")
        source=joinpath(evidence,tag*".son");write(source,text)
        row=Dict{String,Any}("tag"=>tag,"expression_in_source"=>expression,
            "source_sha256"=>bytes2hex(sha256(read(source))),"status"=>"UNVERIFIED")
        try
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false,dependencies=[csv])
            data=checked_native_touchstone(native,joinpath(native.output_dir,"native.s2p");deembedded=false,expected_z0=50.)
            data.frequencies==[1e9,1e10] || error("unexpected frequencies")
            expected=[planar_read_touchstone(joinpath(origin,target,"native.s2p")).s[k]
                for (k,target) in enumerate(("sigma_5e5","sigma_5e4"))]
            row["native_literal_delta_s"]=[maximum(abs,a-b) for (a,b) in zip(data.s,expected)]
            row["s_real"]=[vec(real.(s)) for s in data.s]
            row["s_imag"]=[vec(imag.(s)) for s in data.s]
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
