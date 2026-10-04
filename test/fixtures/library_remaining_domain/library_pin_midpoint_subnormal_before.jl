using DiffMoM,Test,SHA,TOML
const source_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"..","..","src","planar","PlanarLibrary.jl"))))
const rows=Dict{String,Any}[]
@testset "Subnormal placement regression in new half-plus-half midpoint" begin
    shape=planar_line(length=.001,width=.0001,level=1,metal="pec")
    for offset in (nextfloat(0.),-nextfloat(0.))
        placed=planar_transform(shape;offset=(offset,0.))
        pin=planar_pin(placed,"p1");vertices=only(placed.polygons).vertices
        a=vertices[pin.edge];b=vertices[mod1(pin.edge+1,length(vertices))]
        expected=Float64((Rational{BigInt}(a[1])+Rational{BigInt}(b[1]))/2)
        @test expected==offset
        @test a[1]==b[1]==offset
        @test pin.point[1]==0
        push!(rows,Dict("offset"=>offset,"edge_x"=>a[1],"stored_midpoint_x"=>pin.point[1],
            "exact_midpoint_x"=>expected))
    end
end
open(joinpath(@__DIR__,"library_pin_midpoint_subnormal_before.toml"),"w") do io
    TOML.print(io,Dict("source_sha256"=>source_hash,"rows"=>rows))
end
