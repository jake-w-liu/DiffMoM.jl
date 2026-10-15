using DiffMoM, LinearAlgebra, Test

@testset "Observation consistency survives finite extreme scales" begin
    mass = Matrix{ComplexF64}(I, 2, 2)
    information = ComplexF64[1 0; 1 0]
    amplitude = 1.5e308
    caller_precision = precision(BigFloat)
    for phase in (1.0 + 0im, 0.0 + 1im)
        contradictory = phase .* ComplexF64[amplitude, -amplitude]
        # An independent finite-scale oracle: no coefficient can satisfy
        # equal left-hand sides with opposite nonzero right-hand sides.
        @test norm(contradictory ./ amplitude) > 1
        @test_throws ArgumentError condition_error_observations(
            mass, information, contradictory)

        observations = fill(phase * amplitude, 2)
        conditioned = condition_error_observations(mass, information, observations)
        @test conditioned.rank == 1
        @test all(isfinite, conditioned.mean_coefficients)
        @test norm(information * (conditioned.mean_coefficients ./ amplitude) -
                   observations ./ amplitude) < 1e-10
        @test isfinite(conditioned.diagnostics.support_residual)
        @test isfinite(conditioned.diagnostics.observation_residual)
        @test precision(BigFloat) == caller_precision

        # The ordinary factorization fits, but a wide residual recovery
        # must reserve its additional working storage before allocating it.
        ordinary_budget = Base.summarysize((mass, information, observations)) +
                          DiffMoM._conditioning_work_bytes(2, 2)
        @test_throws ArgumentError condition_error_observations(
            mass, information, observations; max_work_bytes=ordinary_budget)
    end
    @test_throws ArgumentError condition_error_observations(
        mass, information, ComplexF64[1e308, -1e308])
end
