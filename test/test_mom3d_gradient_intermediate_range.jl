using Test
using LinearAlgebra
using DiffMoM

# Independent axis-aligned form of the documented electric-dipole Green
# tensor. Keep every factor in BigFloat until the final real contraction.
function _axis_dda_gradient_reference(res, lambda)
    return setprecision(BigFloat, 4352) do
        map(1:res.grid.nvoxels) do source
            derivative = 9 * BigFloat(res.grid.volumes[source]) /
                         (Complex{BigFloat}(res.eps_r[source]) + 2)^2
            moment = derivative .* Complex{BigFloat}.(res.E_total[source])
            total = zero(BigFloat)
            for observation in 1:res.grid.nvoxels
                observation == source && continue
                displacement = BigFloat(res.grid.centers[observation][1]) -
                               BigFloat(res.grid.centers[source][1])
                distance = abs(displacement)
                @test res.grid.centers[observation][2:3] ==
                      res.grid.centers[source][2:3]
                k = BigFloat(res.k0)
                phase = exp(Complex{BigFloat}(0, -k * distance)) /
                        (4 * BigFloat(pi))
                transverse = [zero(Complex{BigFloat}), moment[2], moment[3]]
                near = [2moment[1], -moment[2], -moment[3]]
                field = phase .* ((k^2 / distance) .* transverse +
                    (inv(distance^3) + im * k / distance^2) .* near)
                first = 3observation - 2
                total += real(dot(Complex{BigFloat}.(lambda[first:(first + 2)]), field))
            end
            Float64(2 * total)
        end
    end
end

@testset "DDA gradients retain representable contractions across intermediate ranges" begin
    grid = VoxelGrid3D((0.0, 2.0), (0.0, 1.0), (0.0, 1.0), 2, 1, 1)
    for (permittivity, amplitude) in ((1e170 + 0im, 1.0),
                                     (1e170 - 1e170im, 1.0),
                                     (1e150 + 0im, 1e-30))
        incident = fill(CVec3(amplitude, 0.3amplitude, -0.2amplitude), 2)
        result = solve_dda_3d(grid, 1.0, permittivity, incident)
        lambda = fill(ComplexF64(1e300, 0.2e300), 6)
        reference = _axis_dda_gradient_reference(result, lambda)
        @test all(isfinite, reference)
        @test all(!iszero, reference)
        @test gradient_epsr_dda_3d(result, lambda) ≈ reference rtol=32eps(Float64)
        @test gradient_epsr_dda_3d(result, -lambda) ≈ -reference rtol=32eps(Float64)
        @test gradient_epsr_dda_3d(result, zeros(ComplexF64, 6)) == zeros(2)
    end

    # The derivative can also overflow before multiplication by a tiny Green
    # tensor restores a finite gradient. This is a solved physical system.
    spacing = 1e102
    large_grid = VoxelGrid3D((0.0, 2spacing), (0.0, spacing),
                            (0.0, spacing), 2, 1, 1)
    large_result = solve_dda_3d(large_grid, 1e-103, -1.9,
        fill(CVec3(1.0, 0.3, -0.2), 2))
    lambda = ones(ComplexF64, 6)
    reference = _axis_dda_gradient_reference(large_result, lambda)
    @test all(isfinite, reference)
    @test all(!iszero, reference)
    @test gradient_epsr_dda_3d(large_result, lambda) ≈ reference rtol=32eps(Float64)
end
