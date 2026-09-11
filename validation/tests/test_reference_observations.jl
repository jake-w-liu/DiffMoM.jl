using Test
using DataFrames
include(joinpath(@__DIR__, "..", "pn3d", "reference_population.jl"))

length(ARGS) == 1 || error("usage: test_reference_observations.jl POPULATION_DIR")
population = abspath(ARGS[1])
@testset "Exact registered reference observations" begin
    registered = CSV.read(joinpath(population, "looks.csv"), DataFrame)
    grid, indices = reference_observation_grid(population)
    @test size(grid.rhat) == (3, 36)
    @test length(unique(indices)) == 12
    @test grid.theta[indices] == registered.theta_rad
    @test grid.phi[indices] == registered.phi_rad
    @test grid.rhat[:, indices] == permutedims(Matrix(
        registered[:, [:direction_x, :direction_y, :direction_z]]))
    extra = setdiff(axes(grid.rhat, 2), indices)
    nominal = make_sph_grid(3, 12)
    @test grid.rhat[:, extra] == nominal.rhat[:, extra]
    @test grid.theta[extra] == nominal.theta[extra]
    @test grid.phi[extra] == nominal.phi[extra]
    mktempdir() do temporary
        for defect in (:missing, :reordered, :angle, :direction, :duplicate_id)
            broken = copy(registered)
            if defect == :missing
                broken = broken[1:end-1, :]
            elseif defect == :reordered
                broken = broken[end:-1:1, :]
            elseif defect == :angle
                broken.phi_rad[1] = nextfloat(broken.phi_rad[1])
            elseif defect == :direction
                broken.direction_x[1] = nextfloat(broken.direction_x[1])
            else
                broken.look_id[end] = broken.look_id[1]
            end
            CSV.write(joinpath(temporary, "looks.csv"), broken)
            @test_throws ErrorException reference_observation_grid(temporary)
        end
    end
end
