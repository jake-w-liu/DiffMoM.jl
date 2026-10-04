using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_zero_nominal_geovar")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="zero_nominal_geovar_reference_",cleanup=false)
    paths=[joinpath(dir,file) for folder in (joinpath(repo,"src"),fixture)
        for (dir,_,files) in walkdir(folder) for file in files]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"Exact zero-NOM ANC/NSCD native parameter/literal replay; geometry identity gate1e-12. Six physical PASS/two FAIL baseline remains separate.","source_before"=>hashes(),"cases"=>Any[])
    em=find_em();em===nothing && error("installed native Sonnet engine unavailable")
    try
        for row in TOML.parsefile(joinpath(fixture,"comparison.toml"))["cases"]
            tag="$(row["axis"])_dir_$(row["direction"])_target_$(row["target"])"
            matrices=map(("parameter","literal")) do kind
                name=tag*"_"*kind
                native=reference_run(em,joinpath(fixture,"native",name*".son");output_dir=joinpath(output,name),deembedded=false)
                only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            end
            error=maximum(abs,matrices[1]-matrices[2])
            result=Dict("case"=>tag,"full_s_error"=>error,"bit_identical_s"=>matrices[1]==matrices[2],"status"=>error<=1e-12 ? "PASS" : "FAIL")
            push!(report["cases"],result);println(result);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained native replay: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==8
    @assert all(row["status"]=="PASS" for row in report["cases"])
end
main()
