using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_radial_adjustment_sequence")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="radial_adjustment_sequence_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=[joinpath(d,f) for folder in (joinpath(repo,"src"),fixture)
        for (d,_,fs) in walkdir(folder) for f in fs]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Fresh native replay of36 radial reference-radius recomputation and anchor-crossing parameter/literal " *
        "pairs, original1e-12 full complex S gate and immutable captured matrices; no historical output " *
        "changes.","source_before"=>hashes(),"cases"=>Any[])
    try
        for row in TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
            name=row["name"];matrices=Matrix{ComplexF64}[]
            for suffix in ("parameter","literal")
                source=joinpath(fixture,"cases",name,suffix*".son")
                native=reference_run(find_em(),source;output_dir=joinpath(output,name,suffix),deembedded=false)
                actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                archived=only(planar_read_touchstone(joinpath(fixture,"cases",name,suffix,"native_raw.s2p")).s)
                maximum(abs,actual-archived)<=1e-12 || error("native matrix drift: $name $suffix")
                push!(matrices,actual)
            end
            error=maximum(abs,matrices[1]-matrices[2])
            result=Dict("case"=>name,"native_full_s_error"=>error,"status"=>error<=1e-12 ? "PASS" : "FAIL")
            push!(report["cases"],result);println(result);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained radial-adjustment replay: ",output)
    end
    report["source_unchanged"] && length(report["cases"])==36 && all(r->r["status"]=="PASS",report["cases"]) || error("radial-adjustment replay failed")
end
main()
