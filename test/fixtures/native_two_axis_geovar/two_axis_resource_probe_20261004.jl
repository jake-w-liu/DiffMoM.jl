using DiffMoM,SHA,TOML,Statistics
function main()
    repo=normpath(joinpath(@__DIR__,".."))
    source=joinpath(repo,"test/fixtures/native_two_axis_geovar/native/ydir_anc_positive_expand_parameter.son")
    implementation=joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl")
    before=bytes2hex(sha256(read(implementation)))
    report=Dict{String,Any}("scope"=>"warmed cumulative Julia GC allocation, not process peak RSS; identical metadata and point counts for NSCD/SCUNI/SCXY controls",
        "implementation_before"=>before,"parameter_storage_bytes"=>sizeof(DiffMoM._SonnetGeometryParameter),"cases"=>Any[])
    for mode in ("NSCD","SCUNI","SCXY")
        input=joinpath(@__DIR__,"two_axis_allocation_"*lowercase(mode)*".son")
        write(input,replace(read(source,String),"SCXY"=>mode))
        p=read_sonnet_project(input)
        for _ in 1:20;DiffMoM._sonnet_geometry_project(p,1e9);end
        allocation=@allocated DiffMoM._sonnet_geometry_project(p,1e9)
        seconds=median([@elapsed DiffMoM._sonnet_geometry_project(p,1e9) for _ in 1:101])
        row=Dict("mode"=>mode,"gc_allocation_bytes"=>allocation,"median_seconds"=>seconds)
        push!(report["cases"],row);println(row)
    end
    report["implementation_after"]=bytes2hex(sha256(read(implementation)))
    open(joinpath(@__DIR__,"two_axis_resource_probe_20261004.toml"),"w") do io;TOML.print(io,report);end
    @assert report["implementation_before"]==report["implementation_after"]
end
main()
