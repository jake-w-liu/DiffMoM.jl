module ScaledMixedReferenceInputTests
using Test
include(joinpath(@__DIR__,"..","validation","planar_audit","run_scaled_mixed_reference.jl"))

@testset "Native replay requires its declared captured inputs" begin
    mktempdir() do dir
        mkpath(joinpath(dir,"native"))
        row=Dict("case"=>"control","status"=>"S_EQUIVALENT")
        @test_throws ArgumentError _scaled_mixed_reference_sources(dir,row)
        parameter=joinpath(dir,"native","control_parameter.son");write(parameter,"parameter")
        @test_throws ArgumentError _scaled_mixed_reference_sources(dir,row)
        # The retained unresolved count2 class deliberately has no literal.
        @test _scaled_mixed_reference_sources(dir,Dict("case"=>"control","status"=>"REJECTED"))==(("parameter",parameter),)
        literal=joinpath(dir,"native","control_literal.son");write(literal,"literal")
        @test _scaled_mixed_reference_sources(dir,row)==(("parameter",parameter),("literal",literal))
    end
end
end
