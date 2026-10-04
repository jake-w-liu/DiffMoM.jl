using DiffMoM,Test,SHA,TOML
const rows=Dict{String,Any}[]
@testset "Accepted air bridge contains collapsed via polygons" begin
    for span in (.001,1.),margin in (prevfloat(.00005),)
        shape=planar_air_bridge(;span,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=margin)
        @test 2margin<.0001
        for via in shape.vias
            @test length(unique(via.vertices))==2
            @test_throws ArgumentError planar_normalize_polygon(via.vertices)
            push!(rows,Dict("case"=>"accepted_collapsed_air_bridge_via",
                "span_m"=>span,"margin_m"=>margin,"via"=>via.name,
                "vertices"=>[collect(v) for v in via.vertices]))
        end
    end
end
@testset "Reference capacitance loses a representable result in an intermediate ratio" begin
    for (height,epsr,area) in ((1e-300,1e100,1e-300),
            (1e300,1e-100,1e300),(1e-300,1e100,1e-200))
        stack=PlanarStackup([PlanarLayer(epsr,1.,height)],TERM_GND,TERM_GND,.001,.001)
        @test planar_validate(stack)===nothing
        expected=setprecision(BigFloat,256) do
            Float64(BigFloat(DiffMoM._EPS0)*BigFloat(area)*BigFloat(epsr)/BigFloat(height))
        end
        @test isfinite(expected) && expected>0
        @test_throws ArgumentError planar_parallel_plate_capacitance(stack,1,0,area)
        push!(rows,Dict("case"=>"rejected_representable_capacitance",
            "thickness_m"=>height,"epsr_z"=>epsr,"area_m2"=>area,
            "independent_expected_F"=>expected,"float_denominator"=>string(height/epsr)))
    end
end
open(joinpath(@__DIR__,"library_air_bridge_capacitance_before.toml"),"w") do io
    TOML.print(io,Dict("source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,
        "..","..","src","planar","PlanarLibrary.jl")))),"rows"=>rows))
end
