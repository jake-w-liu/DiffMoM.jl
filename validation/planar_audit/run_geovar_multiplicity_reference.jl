using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_geovar_point_multiplicity")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="geovar_multiplicity_reference_",cleanup=false)
    em=find_em();em===nothing && error("installed native Sonnet engine unavailable")
    paths=[joinpath(dir,file) for folder in (joinpath(repo,"src"),fixture)
        for (dir,_,files) in walkdir(folder) for file in files]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"exact native repeated ordinary points and NSCD ANC/RAD moving references; parameter/literal full matrix identity gate 1e-12", "source_before"=>hashes(),"cases"=>Any[])
    cases=TOML.parsefile(joinpath(fixture,"source_before/geovar_ordinary_multiplicity_before_20261004.toml"))["cases"]
    append!(cases,[Dict("kind"=>"anc","tag"=>"nscd_reference","literal_suffix"=>"multiplicity"),
        Dict("kind"=>"rad","tag"=>"rad_reference","literal_suffix"=>"multiplicity")])
    try
        for row in cases
            family=row["kind"]=="sym" ? "symmetric" : "anchored_radial"
            directory=joinpath(fixture,family);tag=row["tag"];matrices=Matrix{ComplexF64}[]
            for suffix in ("parameter",row["literal_suffix"])
                native=reference_run(em,joinpath(directory,tag*"_"*suffix*".son");output_dir=joinpath(output,family,tag*"_"*suffix),deembedded=false)
                push!(matrices,only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s))
            end
            error=maximum(abs,matrices[1]-matrices[2])
            result=Dict("family"=>family,"case"=>tag,"full_s_error"=>error,"bit_identical_s"=>matrices[1]==matrices[2],"status"=>error<=1e-12 ? "PASS" : "FAIL")
            push!(report["cases"],result);println(result);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained replay: ",output)
    end
    report["source_unchanged"] && all(r->r["status"]=="PASS",report["cases"]) || error("native multiplicity replay failed or sources changed")
end
main()
