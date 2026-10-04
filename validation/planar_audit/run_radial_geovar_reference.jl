using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference

function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_radial_geovar")
    reference_fixture=joinpath(repo,"test/fixtures/native_rad_reference_headers")
    directory=joinpath(fixture,"native")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="radial_geovar_reference_",cleanup=false)
    em=find_em();em===nothing && error("installed native Sonnet engine unavailable")
    paths=[joinpath(dir,file) for folder in (joinpath(repo,"src"),fixture,reference_fixture)
        for (dir,_,files) in walkdir(folder) for file in files]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native replay of exact captured RAD parameter/literal inputs including explicit moving references under all headers; full matrix identity gate 1e-12", "source_before"=>hashes(),"cases"=>Any[])
    function native_matrix(source,tag)
        native=reference_run(em,source;output_dir=joinpath(output,tag),deembedded=false)
        return only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
    end
    function compare(tag,matrix,literal)
        full_s_error=maximum(abs,matrix-literal)
        result=Dict("case"=>tag,"full_s_error"=>full_s_error,"bit_identical_s"=>matrix==literal,"status"=>full_s_error<=1e-12 ? "PASS" : "FAIL")
        push!(report["cases"],result);println(result);flush(stdout)
    end
    try
        literals=Dict{String,Matrix{ComplexF64}}()
        for prefix in ("expand","contract")
            tag=prefix*"_radial_literal"
            literals[prefix]=native_matrix(joinpath(directory,tag*".son"),tag)
        end
        for row in TOML.parsefile(joinpath(directory,"comparison.toml"))["cases"]
            tag=row["tag"];prefix=row["target"]>.3125 ? "expand" : "contract"
            compare(tag,native_matrix(joinpath(directory,tag*".son"),tag),literals[prefix])
        end
        for row in TOML.parsefile(joinpath(reference_fixture,"comparison.toml"))["cases"]
            tag=row["tag"];name=row["name"]
            if !haskey(literals,tag)
                literals[tag]=native_matrix(joinpath(reference_fixture,"inputs",tag*"_literal.son"),tag*"_literal")
            end
            compare(name,native_matrix(joinpath(reference_fixture,name*".son"),name),literals[tag])
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
