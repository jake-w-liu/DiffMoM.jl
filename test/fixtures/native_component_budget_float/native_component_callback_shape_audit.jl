using DiffMoM,SHA,TOML
p=read_sonnet_project(joinpath(@__DIR__,"../../test/fixtures/native_ideal_component_units/res/ohm/ohm.son"))
t=only(filter(r->first(r.tokens)=="TYPE",only(p.components))).tokens
empty!(t);append!(t,["TYPE","SPROJ","explicit.son"])
response=zeros(ComplexF64,1024,1024)
callback=(args...)->(response=response,format=:s,z0=50.)
function probe(p,callback)
    try
        sonnet_component_model(p,200e6;component_response=callback,max_bytes=10000000)
        return "accepted"
    catch e
        return sprint(showerror,e)
    end
end
probe(p,callback)
message=Ref("")
allocation=@allocated message[]=probe(p,callback)
println("max_bytes=10000000 preexisting_bad_shape=1024x1024 allocated=",allocation," ",message[])
open(joinpath(@__DIR__,"native_component_callback_shape_audit.toml"),"w") do io
    TOML.print(io,Dict("production_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/planar/PlanarSonnetComponents.jl")))),
        "max_bytes"=>10000000,"warmed_allocation"=>allocation,"diagnostic"=>message[]))
end
