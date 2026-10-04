using DiffMoM,SHA,TOML,Dates
producer=joinpath(@__DIR__,"radial_geovar_native_probe_20261004.jl")
include_string(Main,replace(read(producer,String),r"\nmain\(\)\s*$"=>"\n"),producer)
function errors()
    directory=mktempdir(joinpath(repo,"data/sonnet_validation");prefix="radial_geovar_edges_",cleanup=false)
    report=Dict{String,Any}("started_utc"=>string(now(UTC)),"producer_sha256"=>bytes2hex(sha256(read(producer))),"cases"=>Any[])
    sources=Dict("cross_anchor"=>radial_source(.03125),"at_anchor"=>radial_source(.0625),
        "offwall"=>replace(read(joinpath(repo,"test/fixtures/native_sonnet_geometry_variables/accepted/single_ydir_anc_positive_parameter.son"),String)," ANC "=>" RAD "))
    for (tag,text) in sources
        source=joinpath(directory,tag*".son");write(source,text)
        row=Dict{String,Any}("tag"=>tag,"status"=>"UNVERIFIED");push!(report["cases"],row)
        try
            native=reference_run(find_em(),source;output_dir=joinpath(directory,tag),deembedded=false)
            checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
            row["status"]="ACCEPT"
        catch err
            row["status"]="REJECT";row["error"]=sprint(showerror,err)
        end
        println(row);flush(stdout)
    end
    open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
    println("Retained edge evidence: ",directory)
end
errors()
