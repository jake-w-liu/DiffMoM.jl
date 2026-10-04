using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_radial_geovar")
    directory=joinpath(fixture,"native")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="radial_geovar_reference_",cleanup=false)
    em=find_em();em===nothing && error("installed native Sonnet engine unavailable")
    paths=[joinpath(dir,file) for folder in (joinpath(repo,"src"),fixture)
        for (dir,_,files) in walkdir(folder) for file in files]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native replay of exact captured RAD parameter/literal inputs; full matrix identity gate 1e-12", "source_before"=>hashes(),"cases"=>Any[])
    try
        literals=Dict{String,Matrix{ComplexF64}}()
        for prefix in ("expand","contract")
            tag=prefix*"_radial_literal"
            native=reference_run(em,joinpath(directory,tag*".son");output_dir=joinpath(output,tag),deembedded=false)
            literals[prefix]=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
        end
        for row in TOML.parsefile(joinpath(directory,"comparison.toml"))["cases"]
            tag=row["tag"];prefix=row["target"]>.3125 ? "expand" : "contract"
            native=reference_run(em,joinpath(directory,tag*".son");output_dir=joinpath(output,tag),deembedded=false)
            matrix=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
            full_s_error=maximum(abs,matrix-literals[prefix])
            result=Dict("case"=>tag,"full_s_error"=>full_s_error,"bit_identical_s"=>matrix==literals[prefix],"status"=>full_s_error<=1e-12 ? "PASS" : "FAIL")
            push!(report["cases"],result);println(result);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained replay: ",output)
    end
    report["source_unchanged"] || error("source changed during native replay")
    all(row->row["status"]=="PASS",report["cases"]) || error("native radial dimension comparison failed")
end
main()
