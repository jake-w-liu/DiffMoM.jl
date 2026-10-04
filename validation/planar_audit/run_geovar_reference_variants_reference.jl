using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_geovar_reference_variants")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="geovar_reference_variants_",cleanup=false)
    paths=[joinpath(dir,file) for folder in (joinpath(repo,"src"),fixture)
        for (dir,_,files) in walkdir(folder) for file in files]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(path,repo)=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"native explicit-reference count boundary: four single-occurrence literal matches, three multiple-occurrence mismatches and one native rejection; unchanged archived matrix/rejection gates", "source_before"=>hashes(),"cases"=>Any[])
    try
        for row in TOML.parsefile(joinpath(fixture,"native/comparison.toml"))["cases"]
            tag=row["tag"];matrices=Dict{String,Matrix{ComplexF64}}();rejected=String[]
            for suffix in ("parameter","literal")
                source=joinpath(fixture,"native",tag*"_"*suffix*".son")
                native=try
                    reference_run(find_em(),source;output_dir=joinpath(output,tag*"_"*suffix),deembedded=false)
                catch err
                    tag=="nscd_contract_2" && suffix=="parameter" || rethrow()
                    occursin("has no subsections",read(joinpath(output,tag*"_"*suffix,"engine_stderr.log"),String)) || rethrow()
                    push!(rejected,suffix);continue
                end
                matrices[suffix]=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                archived=only(planar_read_touchstone(joinpath(fixture,"native",tag*"_"*suffix,"native_raw.s2p")).s)
                maximum(abs,matrices[suffix]-archived)<=1e-12 || error("native replay differs from retained matrix: $tag $suffix")
            end
            full_s_error=length(matrices)==2 ? maximum(abs,matrices["parameter"]-matrices["literal"]) : Inf
            expected=endswith(tag,"_1") ? full_s_error<=1e-12 : (tag=="nscd_contract_2" ? rejected==["parameter"] : full_s_error>.01)
            result=Dict("case"=>tag,"full_s_error"=>full_s_error,"native_rejected"=>rejected,"status"=>expected ? "PASS" : "FAIL")
            push!(report["cases"],result);println(result);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained replay: ",output)
    end
    report["source_unchanged"] && length(report["cases"])==8 && all(r->r["status"]=="PASS",report["cases"]) || error("reference-count replay failed")
end
main()
