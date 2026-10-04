using DiffMoM,SHA,TOML,Statistics
function main()
    repo=normpath(joinpath(@__DIR__,".."));implementation=joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl")
    before=bytes2hex(sha256(read(implementation)))
    point=DiffMoM._sonnet_geovar_radial_point
    cases=[("ordinary",(.75,.625,1.,.4375,.0625)),
        ("range",(1e308,1e308,-1e308,-1e308,-1e308)),
        ("cancellation",(1.,1.,-1.,-1.,-sqrt(2.)))]
    report=Dict{String,Any}("scope"=>"warmed cumulative Julia GC allocations, not peak process memory; native ordinary radial geometry and bounded fallback points", "implementation_before"=>before,"parameter_storage_bytes"=>sizeof(DiffMoM._SonnetGeometryParameter),"points"=>Any[])
    for (tag,args) in cases
        for _ in 1:100;point(args...);end
        bytes=@allocated point(args...)
        seconds=median([@elapsed point(args...) for _ in 1:101])
        row=Dict("case"=>tag,"gc_allocation_bytes"=>bytes,"median_seconds"=>seconds)
        push!(report["points"],row);println(row)
    end
    p=read_sonnet_project(joinpath(repo,"test/fixtures/native_radial_geovar/native/expand_nscd_xdir_negative.son"))
    for _ in 1:100;DiffMoM._sonnet_geometry_project(p,1e9);end
    report["geometry_gc_bytes"]=@allocated DiffMoM._sonnet_geometry_project(p,1e9)
    report["geometry_median_seconds"]=median([@elapsed DiffMoM._sonnet_geometry_project(p,1e9) for _ in 1:101])
    report["implementation_after"]=bytes2hex(sha256(read(implementation)))
    open(joinpath(@__DIR__,"radial_resource_probe_20261004.toml"),"w") do io;TOML.print(io,report);end
    @assert report["implementation_before"]==report["implementation_after"] && report["points"][1]["gc_allocation_bytes"]==0
    println("Geometry cumulative GC bytes: ",report["geometry_gc_bytes"])
end
main()
