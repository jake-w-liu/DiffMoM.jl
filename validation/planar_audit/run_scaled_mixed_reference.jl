using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_geovar_scaled_mixed_reference")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="geovar_scaled_mixed_reference_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=[joinpath(d,f) for folder in (joinpath(repo,"src"),fixture)
        for (d,_,fs) in walkdir(folder) for f in fs]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Fresh native replay of scaled ANC mixed moving-reference parameter/literal pairs, original 1e-12 full complex S gate and immutable captured matrices; no historical output changes.","source_before"=>hashes(),"cases"=>Any[])
    try
        for row in TOML.parsefile(joinpath(fixture,"comparison.toml"))["cases"]
            name=row["case"];matrices=Matrix{ComplexF64}[]
            for suffix in ("parameter","literal")
                source=joinpath(fixture,"native",name*"_"*suffix*".son")
                isfile(source) || continue
                native=reference_run(find_em(),source;output_dir=joinpath(output,name,suffix),deembedded=false)
                actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                archived=only(planar_read_touchstone(joinpath(fixture,"native",name*"_"*suffix,"native_raw.s2p")).s)
                maximum(abs,actual-archived)<=1e-12 || error("native matrix drift: $name $suffix")
                push!(matrices,actual)
            end
            error=length(matrices)==2 ? maximum(abs,matrices[1]-matrices[2]) : NaN
            result=Dict("case"=>name,"native_full_s_error"=>error,
                "status"=>error<=1e-12 ? "PASS" : (isnan(error) ? "PARAMETER-ONLY" : "FAIL"))
            push!(report["cases"],result);println(result);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained scaled mixed-reference replay: ",output)
    end
    report["source_unchanged"] && all(r->r["status"]!="FAIL",report["cases"]) || error("scaled mixed-reference replay failed")
end
main()
