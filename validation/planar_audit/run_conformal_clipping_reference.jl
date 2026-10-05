using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_conformal_box_clipping")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="conformal_clipping_replay_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=[joinpath(d,f) for folder in (joinpath(repo,"src"),fixture)
        for (d,_,fs) in walkdir(folder) for f in fs]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Fresh native replay of ten exact box clipping controls; unchanged complete-matrix identity gate 1e-12 and native rejection reason. Conformal constructor outcomes checked independently.",
        "source_before"=>hashes(),"status"=>"RUNNING","cases"=>Any[])
    try
        for row in TOML.parsefile(joinpath(fixture,"original_before_comparison.toml"))["cases"]
            name=row["case"];folder=joinpath(fixture,"cases",name);source=joinpath(folder,"project.son")
            valid=row["native_status"]=="ACCEPT";native=nothing;failure=nothing
            try
                native=reference_run(find_em(),source;output_dir=joinpath(output,name),deembedded=false,nports=1)
            catch err
                failure=err
            end
            (failure===nothing)==valid || error("unexpected native outcome: $name")
            result=Dict{String,Any}("case"=>name,"native_status"=>row["native_status"])
            if valid
                # These one-port source projects explicitly request this filename.
                actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                retained=only(planar_read_touchstone(joinpath(folder,"native/native_raw.s2p");nports=1).s)
                drift=maximum(abs,actual-retained);drift<=1e-12 || error("native matrix drift: $name")
                result["native_full_s_error"]=drift
            else
                occursin("partially or entirely outside",read(joinpath(output,name,"engine_stderr.log"),String)) ||
                    error("unexpected native rejection reason: $name")
            end
            conformal_failure=nothing
            try
                sonnet_conformal_layout(read_sonnet_project(source);freq=1e9,
                    edge_size=.125e-3,interior_size=.25e-3,edge_band=.05e-3)
            catch err
                conformal_failure=err
            end
            (conformal_failure===nothing)==valid || error("unexpected conformal outcome: $name")
            valid || conformal_failure isa ArgumentError || error("unexpected conformal failure: $name")
            result["conformal_status"]=valid ? "ACCEPT" : "REJECT"
            push!(report["cases"],result);println(result);flush(stdout)
        end
        report["status"]="PASS"
    catch
        report["status"]="FAIL"
        rethrow()
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained conformal clipping replay: ",output)
    end
    report["source_unchanged"] && report["status"]=="PASS" && length(report["cases"])==10 || error("conformal clipping replay failed")
end
main()
