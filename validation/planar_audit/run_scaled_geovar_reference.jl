using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_scaled_geovar")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root)
    output=mktempdir(root;prefix="scaled_geovar_reference_",cleanup=false)
    em=find_em();em===nothing && error("installed native Sonnet engine unavailable")
    paths=[joinpath(dir,file) for folder in ("src","test/fixtures/native_scaled_geovar")
        for (dir,_,files) in walkdir(joinpath(repo,folder)) for file in files]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native replay of exact captured SCUNI parameter/literal inputs; full matrix identity gate 1e-12",
        "source_before"=>hashes(),"cases"=>Any[])
    try
        for name in ("native","native_x","native_contraction")
            directory=joinpath(fixture,name)
            for row in TOML.parsefile(joinpath(directory,"comparison.toml"))["cases"]
                tag=get(row,"tag",lowercase(row["kind"])*(row["direction"]==1 ? "_positive" : "_negative"))
                matrices=Matrix{ComplexF64}[]
                for mode in ("parameter","proportional_literal")
                    source=joinpath(directory,tag*"_"*mode*".son")
                    native=reference_run(em,source;output_dir=joinpath(output,name,tag*"_"*mode),deembedded=false)
                    data=checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false)
                    push!(matrices,only(data.s))
                end
                full_s_error=maximum(abs,matrices[1]-matrices[2])
                result=Dict("capture"=>name,"case"=>tag,"full_s_error"=>full_s_error,
                    "bit_identical_s"=>matrices[1]==matrices[2],"status"=>full_s_error<=1e-12 ? "PASS" : "FAIL")
                push!(report["cases"],result);println(result);flush(stdout)
            end
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained replay: ",output)
    end
    report["source_unchanged"] || error("source changed during native replay")
    all(row->row["status"]=="PASS",report["cases"]) || error("native scaled dimension comparison failed")
end
main()
