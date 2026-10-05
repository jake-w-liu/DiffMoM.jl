using DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"../sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
function main()
    repo=normpath(joinpath(@__DIR__,"../.."))
    fixtures=[joinpath(repo,"test/fixtures",name) for name in
        ("native_plain_box_port_extent","native_box_port_boundary_allowance","native_moved_box_port_boundary_allowance")]
    cases=NamedTuple[]
    for fixture in fixtures,row in TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
        folder=joinpath(fixture,"cases",row["name"])
        for suffix in (endswith(fixture,"native_moved_box_port_boundary_allowance") ? ("parameter","literal") : ("project",))
            reference=joinpath(folder,suffix=="project" ? "native" : suffix,"native_raw.s2p")
            push!(cases,(name=basename(fixture)*"__"*row["name"]*"__"*suffix,
                source=joinpath(folder,suffix*".son"),reference,status=row["native_status"]))
        end
    end
    root=isempty(ARGS) ? joinpath(repo,"data/planar_audit") : abspath(ARGS[1])
    mkpath(root);output=mktempdir(root;prefix="box_port_extent_replay_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    paths=[joinpath(d,f) for folder in vcat([joinpath(repo,"src")],fixtures)
        for (d,_,fs) in walkdir(folder) for f in fs]
    append!(paths,[@__FILE__,joinpath(repo,"validation/sonnet_stripline/sonnet_reference.jl")])
    hashes()=Dict(relpath(p,repo)=>bytes2hex(sha256(read(p))) for p in paths)
    report=Dict{String,Any}("scope"=>"Fresh replay of80 native inputs: literal edge extent, native boundary allowance and active ANC " *
        "parameter/literal controls. Explicit rejection messages and complete retained matrices use the " *
        "unchanged1e-12 identity gate; public raster outcomes must match.",
        "source_before"=>hashes(),"status"=>"RUNNING","cases"=>Any[])
    try
        for row in cases
            name=row.name;source=row.source
            native=nothing;failure=nothing
            try
                native=reference_run(find_em(),source;output_dir=joinpath(output,name),deembedded=false)
            catch err
                failure=err
            end
            valid=row.status=="ACCEPT"
            (failure===nothing)==valid || error("unexpected native outcome: $name")
            if valid
                actual=only(checked_native_touchstone(native,joinpath(native.output_dir,"native_raw.s2p");deembedded=false).s)
                archived=only(planar_read_touchstone(row.reference).s)
                matrix_error=maximum(abs,actual-archived)
                matrix_error<=1e-12 || throw(ArgumentError("native matrix drift: $name"))
                sonnet_planar_problem(read_sonnet_project(source);freq=1e9)
                result=Dict("case"=>name,"native_status"=>"ACCEPT","native_full_s_error"=>matrix_error,"raster_status"=>"ACCEPT")
            else
                stderr=read(joinpath(output,name,"engine_stderr.log"),String)
                occursin("partially or entirely outside of the box",stderr) || error("unexpected native rejection: $name")
                rejection=nothing
                try
                    sonnet_planar_problem(read_sonnet_project(source);freq=1e9)
                catch err
                    rejection=err
                end
                rejection isa ArgumentError && occursin("box-port edge",sprint(showerror,rejection)) ||
                    error("missing raster extent rejection: $name")
                result=Dict("case"=>name,"native_status"=>"REJECT","raster_status"=>"REJECT")
            end
            push!(report["cases"],result);println(result);flush(stdout)
        end
        report["status"]="PASS"
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained box-port extent replay: ",output)
    end
    report["source_unchanged"] && report["status"]=="PASS" && length(report["cases"])==80 || error("box-port extent replay failed")
end
main()
