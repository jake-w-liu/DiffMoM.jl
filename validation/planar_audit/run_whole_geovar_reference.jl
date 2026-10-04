using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_geovar_whole_polygon")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="whole_geovar_reference_",cleanup=false)
    em=find_em();em===nothing && error("installed native Sonnet engine unavailable")
    paths=[joinpath(dir,file) for folder in (joinpath(repo,"src"),fixture)
        for (dir,_,files) in walkdir(folder) for file in files]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native whole-polygon zero-count, explicit vertex-list and literal triples; axes/directions/contraction/expansion; full matrix identity gate 1e-12", "source_before"=>hashes(),"cases"=>Any[])
    try
        for row in TOML.parsefile(joinpath(fixture,"variants/comparison.toml"))["cases"]
            tag=row["tag"];matrices=Matrix{ComplexF64}[]
            for suffix in ("parameter","explicit","literal")
                native=reference_run(em,joinpath(fixture,"variants",tag*"_"*suffix*".son");output_dir=joinpath(output,tag*"_"*suffix),deembedded=false)
                push!(matrices,only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s))
            end
            full_s_error=maximum(maximum(abs,matrices[1]-other) for other in matrices[2:3])
            result=Dict("case"=>tag,"full_s_error"=>full_s_error,"all_bit_identical"=>matrices[1]==matrices[2]==matrices[3],"status"=>full_s_error<=1e-12 ? "PASS" : "FAIL")
            push!(report["cases"],result);println(result);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained replay: ",output)
    end
    report["source_unchanged"] && length(report["cases"])==16 && all(row->row["status"]=="PASS",report["cases"]) || error("native whole-polygon replay failed")
end
main()
