using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
function main()
    original=joinpath(repo,"data/planar_audit/plain_box_port_extent_69slJp")
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="plain_box_port_extent_conformal_",cleanup=false);cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=vcat([@__FILE__],[joinpath(d,f) for folder in (joinpath(repo,"src"),original) for (d,_,fs) in walkdir(folder) for f in fs])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Conformal follow-up of immutable6 native box-port extent controls with required mesh-size keywords supplied. Original native runs are reused byte for byte; prior UndefKeywordError is a harness limitation, not a geometry verdict.","source_before"=>hashes(),"cases"=>Any[])
    try
        for row in TOML.parsefile(joinpath(original,"comparison.toml"))["cases"]
            name=row["case"];p=read_sonnet_project(joinpath(original,name*".son"));outcome=Dict{String,Any}("case"=>name,"native_status"=>row["native_status"],"raster_status"=>row["raster_status"])
            try
                sonnet_conformal_layout(p;freq=1e9,edge_size=.0625e-3,interior_size=.0625e-3)
                outcome["conformal_status"]="ACCEPT"
            catch err
                outcome["conformal_status"]="REJECT";outcome["conformal_error"]=sprint(showerror,err)
            end
            push!(report["cases"],outcome);println(outcome);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained conformal follow-up: ",output)
    end
    @assert report["source_unchanged"] && length(report["cases"])==6
end
main()
