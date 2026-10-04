using DiffMoM,Random,TOML,SHA,Statistics

function coordinate_loop(values)
    total=0.
    for value in values
        total+=DiffMoM._sonnet_geovar_scaled_coordinate(value,.375,.625,.25,.375,true)
    end
    total
end

function main()
    repo=normpath(joinpath(@__DIR__,".."))
    root=joinpath(repo,"test/fixtures/native_scaled_geovar")
    source=joinpath(root,"native/anc_positive_parameter.son")
    control=joinpath(@__DIR__,"scaled_geovar_nscd_allocation_control.son")
    write(control,replace(read(source,String),"SCUNI"=>"NSCD"))
    p=read_sonnet_project(source);q=read_sonnet_project(control)
    implementation=joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl")
    before=bytes2hex(sha256(read(implementation)))
    values=randn(MersenneTwister(732),100000)
    coordinate_loop(values)
    normal_allocation=@allocated coordinate_loop(values)
    normal_seconds=median([@elapsed coordinate_loop(values) for _ in 1:15])
    tiny=nextfloat(0.);large=floatmax(Float64)
    helper=DiffMoM._sonnet_geovar_scaled_coordinate
    helper(tiny,0.,0.,tiny,large,false)
    exceptional_allocation=@allocated helper(tiny,0.,0.,tiny,large,false)
    resolver=Dict{String,Any}()
    for (name,project) in (("SCUNI",p),("NSCD",q))
        for _ in 1:20;DiffMoM._sonnet_geometry_project(project,1e9);end
        allocation=@allocated DiffMoM._sonnet_geometry_project(project,1e9)
        seconds=median([@elapsed DiffMoM._sonnet_geometry_project(project,1e9) for _ in 1:101])
        resolver[name]=Dict("gc_allocation_bytes_per_resolution"=>allocation,"median_seconds"=>seconds)
    end
    report=Dict{String,Any}("scope"=>"warmed cumulative Julia GC allocation, not process peak RSS; identical metadata and point counts for NSCD/SCUNI controls",
        "julia_version"=>string(VERSION),"normal_point_count"=>length(values),
        "normal_coordinate_gc_allocation_bytes"=>normal_allocation,"normal_coordinate_median_seconds"=>normal_seconds,
        "exceptional_coordinate_gc_allocation_bytes"=>exceptional_allocation,"resolver"=>resolver,
        "implementation_sha256_before"=>before,"implementation_sha256_after"=>bytes2hex(sha256(read(implementation))))
    open(joinpath(@__DIR__,"scaled_geovar_resource_probe_20261004.toml"),"w") do io;TOML.print(io,report);end
    println(report)
    normal_allocation==0 || error("normal coordinate loop unexpectedly allocates")
    report["implementation_sha256_before"]==report["implementation_sha256_after"] || error("implementation changed during resource measurement")
end
main()
