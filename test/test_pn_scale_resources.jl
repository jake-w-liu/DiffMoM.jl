using Test
using LinearAlgebra

if !isdefined(Main, :_study_field_difference_scales)
    include(joinpath(@__DIR__, "..", "validation", "pn3d", "study_analysis.jl"))
end

function _pn_scale_resource_measure(first_field, second_field, floor_value)
    return @allocated _study_field_difference_scales(
        first_field, second_field, floor_value)
end

@testset "PN3D difference scales reuse a linear-size workspace" begin
    for looks in (12, 36)
        first_field = reshape(ComplexF64.(1:(2looks)), 2, looks)
        second_field = 0.99 .* first_field
        reference = [max(sqrt(sum(abs2,
            first_field[:, look] - second_field[:, look])), 1e-10)
            for look in 1:looks]
        @test _study_field_difference_scales(
            first_field, second_field, 1e-10) ≈ reference rtol=4eps(Float64)
        @test _study_field_difference_scales(
            first_field, first_field, 1e-10) == fill(1e-10, looks)
        _pn_scale_resource_measure(first_field, second_field, 1e-10)
        @test _pn_scale_resource_measure(first_field, second_field, 1e-10) <=
              64looks + 1024
    end
    @test_throws DimensionMismatch _study_field_difference_scales(
        zeros(2, 12), zeros(2, 36), 1e-10)
end
