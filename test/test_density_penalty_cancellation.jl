using Test
using SparseArrays
using DiffMoM

@testset "Density penalties preserve small void fractions and coefficient range" begin
    for (density, power, impedance, mass_value) in (
            (prevfloat(1.0), 1.5, 1.0 + 0im, 1.0 + 0im),
            (prevfloat(1.0), 2.7, 2.0 + 0.3im, 1.0 - 0.4im),
            (1.0 - 1e-10, 1.5, 1.0 + 0im, 1.0 + 0im),
            (prevfloat(1.0), 1.0, 1e-310 + 0im, 1e308 + 0im),
            (prevfloat(1.0), 1.5, 1e-310 + 0.2e-310im, 1e308 - 0.3e308im))
        config = DensityConfig(power, impedance, 0.5)
        reference = setprecision(BigFloat, 4352) do
            ComplexF64((1 - BigFloat(density)^BigFloat(power)) *
                Complex{BigFloat}(impedance) * Complex{BigFloat}(mass_value))
        end
        @test isfinite(reference) && !iszero(reference)
        for mass in (fill(ComplexF64(mass_value), 1, 1),
                     sparse([1], [1], ComplexF64[mass_value], 1, 1),
                     LocalMassMatrix(1, [1], [1], ComplexF64[mass_value]))
            penalty = assemble_Z_penalty([mass], [density], config)
            @test penalty[1, 1] ≈ reference rtol=8eps(Float64)
            @test assemble_Z_penalty([mass], [1.0], config)[1, 1] == 0
        end
    end
end
