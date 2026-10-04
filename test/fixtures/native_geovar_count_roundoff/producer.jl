using DiffMoM,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
function main()
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="geovar_reference_count_mask_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    geometry=joinpath(repo,"src/planar/PlanarSonnetGeometryVariables.jl")
    cp(geometry,joinpath(output,"geometry_before.jl"))
    before=bytes2hex(sha256(read(geometry)));rows=Any[]
    fixture=joinpath(repo,"test/fixtures/native_geovar_reference_count_law")
    for row in TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
        folder=joinpath(fixture,"cases",row["name"])
        p=read_sonnet_project(joinpath(folder,"parameter.son"));q=read_sonnet_project(joinpath(folder,"literal.son"))
        a=sonnet_planar_problem(p;freq=1e9);b=sonnet_planar_problem(q;freq=1e9)
        mismatch=sum(count(x.mask .!= y.mask) for (x,y) in zip(a.sheets,b.sheets))
        iszero(mismatch) && continue
        effective=DiffMoM._sonnet_geometry_project(p,1e9)
        coordinates=[Dict("actual"=>vec(x.vertices),"literal"=>vec(y.vertices),"error"=>maximum(abs,x.vertices-y.vertices)) for (x,y) in zip(effective.polygons,q.polygons)]
        record=Dict("case"=>row["name"],"cell_mismatch"=>mismatch,"coordinates"=>coordinates)
        push!(rows,record);println(record)
    end
    report=Dict("source_before"=>before,"source_after"=>bytes2hex(sha256(read(geometry))),"cases"=>rows)
    open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
    @assert report["source_before"]==report["source_after"]
    println("Retained masks: ",output)
end
main()
