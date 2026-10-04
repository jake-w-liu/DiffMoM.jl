module ArchivedLibrary
using DiffMoM
import DiffMoM: _circuit_stored_real,_P2,_p2_seg_len,_p2_signed_area,
    _planar_regular_ngon,_EPS0,_MU0
include(joinpath(@__DIR__,"..","..","test","fixtures","library_stored_domain","source_after.jl"))
end
using DiffMoM,Test,SHA,TOML
const rows=Dict{String,Any}[]
@testset "Archived air bridge accepts zero-width vias" begin
    for span in (.001,1.)
        shape=ArchivedLibrary.planar_air_bridge(;span,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=prevfloat(.00005))
        for via in shape.vias
            @test length(unique(via.vertices))==2
            @test_throws ArgumentError planar_normalize_polygon(via.vertices)
        end
    end
end
@testset "Archived capacitance rejects representable range-safe answer" begin
    for (height,epsr,area) in ((1e-300,1e100,1e-300),
            (1e300,1e-100,1e300),(1e-300,1e100,1e-200))
        stack=PlanarStackup([PlanarLayer(epsr,1.,height)],TERM_GND,TERM_GND,.001,.001)
        expected=setprecision(BigFloat,4096) do
            Float64(BigFloat(DiffMoM._EPS0)*BigFloat(area)*BigFloat(epsr)/BigFloat(height))
        end
        @test isfinite(expected)&&expected>0
        @test_throws ArgumentError ArchivedLibrary.planar_parallel_plate_capacitance(stack,1,0,area)
        push!(rows,Dict("height"=>height,"epsr"=>epsr,"area"=>area,"independent_expected"=>expected))
    end
end
open(joinpath(@__DIR__,"library_air_bridge_capacitance_archived_before.toml"),"w") do io
    TOML.print(io,Dict("source_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,
        "..","..","test","fixtures","library_stored_domain","source_after.jl")))),"rows"=>rows))
end
