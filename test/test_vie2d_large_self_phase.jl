using DiffMoM, LinearAlgebra, Test

@testset "VIE self interaction keeps the supplied large radial phase" begin
    for (k, span) in ((1e200, 1.0), (1e200, 1e100), (1e160, 1e150))
        mesh = Mesh2D((0.0, span), (0.0, span), 1, 1)
        radius = equivalent_radius(mesh)
        # The independently returned integral is representable in these
        # cases. Multiply in a wider exponent range for a second API route.
        integral = self_cell_integral_2d(k, radius)
        interaction = setprecision(BigFloat, 4352) do
            ComplexF64(BigFloat(k)^2 * Complex{BigFloat}(integral))
        end
        bessel_reference = setprecision(BigFloat, 4352) do
            argument = BigFloat(k) * BigFloat(radius)
            hankel = complex(DiffMoM.besselj(1, argument), -DiffMoM.bessely(1, argument))
            ComplexF64((-im * BigFloat(pi) / 2) *
                       (argument * hankel - 2im / BigFloat(pi)))
        end
        @test interaction ≈ bessel_reference rtol=8eps(Float64)
        for contrast in (0.0, 1.0, -0.25)
            z, d = assemble_vie_2d(mesh, k, [contrast])
            @test d[1] == integral
            @test z[1] ≈ 1 - contrast * interaction rtol=8eps(Float64)
            @test all(isfinite, z)
        end
        result = solve_vie_2d(mesh, k, [1.0], ComplexF64[1.0])
        @test result.E_total[1] ≈ inv(1 - interaction) rtol=16eps(Float64)
    end
end
