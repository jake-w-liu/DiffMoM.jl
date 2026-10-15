using Test
using SparseArrays
using DiffMoM

@testset "Density derivatives retain mass-scaled coefficient range" begin
    for (density, impedance, mass_value) in (
            (1e-200, 1.0 + 0im, 1e200 + 0im),
            (1.0, 1e308 + 0im, 1e-308 + 0im),
            (1e-160, 1e300 + 0im, 1.0 + 0im),
            (1e-200, 1.0 + 0.5im, 1e200 - 0.3e200im))
        config = DensityConfig(3.0, impedance, 0.5)
        reference = setprecision(BigFloat, 4352) do
            ComplexF64(-BigFloat(config.p) *
                BigFloat(density)^(BigFloat(config.p) - 1) *
                Complex{BigFloat}(impedance) * Complex{BigFloat}(mass_value))
        end
        @test isfinite(reference) && !iszero(reference)
        for mass in (fill(ComplexF64(mass_value), 1, 1),
                     sparse([1], [1], ComplexF64[mass_value], 1, 1),
                     LocalMassMatrix(1, [1], [1], ComplexF64[mass_value]))
            derivative = assemble_dZ_drhobar([mass], [density], config, 1)
            @test eltype(derivative) == ComplexF64
            @test derivative[1, 1] == reference
        end
    end

    # Final overflow remains an error; sparse zero entries stay zero even
    # when the scalar coefficient itself overflows Float64.
    config = DensityConfig(3.0, 1e308, 0.5)
    @test_throws OverflowError assemble_dZ_drhobar(
        [ones(ComplexF64, 1, 1)], [1.0], config, 1)
    @test iszero(assemble_dZ_drhobar(
        [spzeros(ComplexF64, 1, 1)], [1.0], config, 1)[1, 1])
end
