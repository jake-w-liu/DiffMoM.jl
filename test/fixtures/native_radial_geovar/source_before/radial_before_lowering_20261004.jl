using DiffMoM,SHA,TOML
function main()
    fixture=joinpath(@__DIR__,"../test/fixtures/native_radial_geovar")
    implementation=joinpath(pwd(),"src/planar/PlanarSonnetGeometryVariables.jl")
    before=bytes2hex(sha256(read(implementation)))
    report=Dict{String,Any}("scope"=>"preimplementation public RAD lowering on stable detached 8c121a7c", "implementation_before"=>before,"cases"=>Any[])
    for row in TOML.parsefile(joinpath(fixture,"native/comparison.toml"))["cases"]
        p=read_sonnet_project(joinpath(fixture,"native",row["tag"]*".son"));saved=deepcopy(p.polygons)
        result=Dict{String,Any}("case"=>row["tag"],"status"=>"UNVERIFIED")
        try
            sonnet_planar_problem(p;freq=1e9);result["status"]="ACCEPT"
        catch err
            result["status"]="REJECT";result["error"]=sprint(showerror,err)
        end
        result["input_unchanged"]=all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        push!(report["cases"],result)
    end
    report["implementation_after"]=bytes2hex(sha256(read(implementation)))
    open(joinpath(@__DIR__,"radial_before_lowering_20261004.toml"),"w") do io;TOML.print(io,report);end
    @assert before==report["implementation_after"] && all(r->r["status"]=="REJECT" && r["input_unchanged"],report["cases"])
    println("Confirmed public unsupported RAD cases: ",length(report["cases"]))
end
main()
