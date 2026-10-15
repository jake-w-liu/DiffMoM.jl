using DiffMoM, Test

@testset "Huray screening preserves a finite corrected numerator" begin
    frequency = 1 / (pi * DiffMoM._MU0)
    roughness = HurayRoughness(1.0, 1.4e308 / (4pi))
    for conductivity in (1.0, 0.25)
        expected = setprecision(BigFloat, 512) do
            depth = inv(sqrt(BigFloat(pi) * BigFloat(DiffMoM._MU0) *
                             BigFloat(frequency) * BigFloat(conductivity)))
            radius = BigFloat(roughness.radius)
            ratio = depth / radius
            Float64(1 + BigFloat(1.5) * 4 * BigFloat(pi) * radius^2 *
                    BigFloat(roughness.density) / (1 + ratio + ratio^2 / 2))
        end
        actual = roughness_factor(roughness, frequency, conductivity)
        @test isfinite(actual)
        @test actual ≈ expected rtol=8eps(Float64)
    end
end
