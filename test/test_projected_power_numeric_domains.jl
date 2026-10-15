module ProjectedPowerNumericDomainTests
using DiffMoM, Test

function bound_allocations(values)
    DiffMoM._projected_power_component_bounds(values)
    return @allocated DiffMoM._projected_power_component_bounds(values)
end

@testset "Projected power preserves finite rational and wide-exponent inputs" begin
    grid = make_sph_grid(1, 1)
    tiny = big(1) // (big(1) << 2000)
    field = reshape([tiny, zero(tiny), zero(tiny)], 3, 1)
    polarization = reshape(ComplexF64[1, 0, 0], 3, 1)
    @test projected_power(field, grid, polarization) == 0.0
    large_polarization = setprecision(BigFloat, 256) do
        reshape(Complex{BigFloat}[ldexp(BigFloat(1), 2000), 0, 0], 3, 1)
    end
    @test projected_power(field, grid, large_polarization) == grid.w[1]

    # Storage and precision depend on the significand, even when a field
    # and its polarization have large, mutually cancelling exponents.
    for shift in (100, 1_000_000)
        wide_field, wide_pol = setprecision(BigFloat, 256) do
            (reshape(BigFloat[ldexp(BigFloat(1), -shift), 0, 0], 3, 1),
             reshape(Complex{BigFloat}[ldexp(BigFloat(1), shift), 0, 0], 3, 1))
        end
        @test projected_power(wide_field, grid, wide_pol) == grid.w[1]
        @test bound_allocations(wide_field) < 4096
    end
end
end
