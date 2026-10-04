using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixture=joinpath(repo,"test/fixtures/native_box_port_attachment")
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="port_attachment_replay_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=[joinpath(d,f) for folder in (joinpath(repo,"src"),fixture)
        for (d,_,fs) in walkdir(folder) for f in fs]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Fresh native replay of20 edge attachment, annotation, collinear and undriven wall controls; complete retained matrices use the unchanged1e-12 identity gate.",
        "source_before"=>hashes(),"status"=>"RUNNING","cases"=>Any[])
    try
        for row in TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
            name=row["name"];folder=joinpath(fixture,"cases",name);source=joinpath(folder,"project.son")
            nports=row["nports"];filename="native_raw.s$(nports)p";valid=row["native_status"]=="ACCEPT"
            native=nothing;failure=nothing
            try
                native=reference_run(find_em(),source;output_dir=joinpath(output,name),deembedded=false,nports)
            catch err
                failure=err
            end
            (failure===nothing)==valid || error("unexpected native outcome: $name")
            result=Dict{String,Any}("case"=>name,"native_status"=>row["native_status"])
            if valid
                actual=only(checked_native_touchstone(native,joinpath(native.output_dir,filename);deembedded=false).s)
                retained=only(planar_read_touchstone(joinpath(folder,"native",filename)).s)
                drift=maximum(abs,actual-retained);drift<=1e-12 || error("native matrix drift: $name")
                sonnet_planar_problem(read_sonnet_project(source);freq=1e9)
                result["native_full_s_error"]=drift;result["raster_status"]="ACCEPT"
            else
                stderr=read(joinpath(output,name,"engine_stderr.log"),String)
                retained=read(joinpath(folder,"native/engine_stderr.log"),String)
                message=occursin("requested index",retained) ? "requested index" : "no ground reference"
                occursin(message,stderr) || error("unexpected rejection reason: $name")
                rejection=nothing
                try
                    sonnet_planar_problem(read_sonnet_project(source);freq=1e9)
                catch err
                    rejection=err
                end
                rejection isa ArgumentError || error("missing importer rejection: $name")
                result["raster_status"]="REJECT"
            end
            push!(report["cases"],result);println(result);flush(stdout)
        end
        report["status"]="PASS"
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained port attachment replay: ",output)
    end
    report["source_unchanged"] && report["status"]=="PASS" && length(report["cases"])==20 || error("port attachment replay failed")
end
main()
