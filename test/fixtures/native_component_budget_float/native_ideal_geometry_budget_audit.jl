using DiffMoM,SHA,TOML
fixture=joinpath(@__DIR__,"../../test/fixtures/native_ideal_component_units/res/ohm/ohm.son")
p=read_sonnet_project(fixture)
function probe(p,grid,limit)
    try
        sonnet_component_model(p,200e6;grid,max_bytes=limit)
        return "accepted"
    catch e
        e isa ArgumentError || rethrow()
        return sprint(showerror,e)
    end
end
output=Dict{String,Any}("source_sha256"=>bytes2hex(sha256(read(fixture))),
    "production_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarSonnetComponents.jl")))))
for grid in ((32,40),(128,160))
    limit=20000
    probe(p,grid,limit)
    message=Ref("")
    allocation=@allocated message[]=probe(p,grid,limit)
    # The base lowerer allocates ComplexF64 sheet_zs before entering the
    # budgeted physical-return helper; this is a live numeric lower bound.
    minimum_base_sheet_zs_bytes=sizeof(ComplexF64)*prod(grid)
    println(grid," max_bytes=",limit," allocation=",allocation,
        " live_sheet_zs_lower_bound=",minimum_base_sheet_zs_bytes," ",message[])
    output[string(grid)]=Dict("max_bytes"=>limit,"warmed_allocated_bytes"=>allocation,
        "live_sheet_zs_lower_bound"=>minimum_base_sheet_zs_bytes,"diagnostic"=>message[])
end
open(joinpath(@__DIR__,"native_ideal_geometry_budget_audit.toml"),"w") do io
    TOML.print(io,output)
end
