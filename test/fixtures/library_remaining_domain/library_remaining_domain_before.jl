using DiffMoM, Test, LinearAlgebra, SHA, TOML

const output=joinpath(@__DIR__,"library_remaining_domain_before.toml")
const rows=Dict{String,Any}[]
@testset "Confirmed accepted library state contradictions" begin
    for value in (big"1e-1000",-big"1e-1000")
        shape=planar_broadside_coupled_lines(length=.001,width=.0001,
            upper_level=2,lower_level=1,offset=value,metal="pec")
        @test value!=0
        @test shape.meta["offset"]==0.
        @test shape.polygons[1].vertices==shape.polygons[2].vertices
        push!(rows,Dict("case"=>"nonzero_broadside_offset_stored_zero",
            "input"=>string(value),"stored_offset"=>shape.meta["offset"],
            "status"=>"accepted"))
    end
    original=planar_line(length=.001,width=.0001,level=1,metal="pec")
    for offset in (1e9,1e10,1e11)
        shape=planar_transform(original;offset=(0.,offset))
        polygon=only(shape.polygons)
        for pin in shape.pins
            a=polygon.vertices[pin.edge]
            b=polygon.vertices[mod1(pin.edge+1,length(polygon.vertices))]
            emitted_width=hypot((b-a)...)
            @test isfinite(emitted_width) && emitted_width>0
            @test pin.width!=emitted_width
            @test abs(pin.width-emitted_width)/pin.width>=.001
            push!(rows,Dict("case"=>"placed_pin_width_differs_from_its_edge",
                "offset_m"=>offset,"pin"=>pin.name,"stored_width_m"=>pin.width,
                "emitted_width_m"=>emitted_width,
                "relative_difference"=>abs(pin.width-emitted_width)/pin.width,
                "status"=>"accepted"))
        end
    end
end
open(output,"w") do io
    TOML.print(io,Dict("source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,
        "..","..","src","planar","PlanarLibrary.jl")))),"rows"=>rows))
end
