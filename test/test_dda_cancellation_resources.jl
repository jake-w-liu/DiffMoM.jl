using Test
using LinearAlgebra
using DiffMoM

function _dda_full_precision_reference!(output, operator, input)
    for voxel in 1:operator.grid.nvoxels
        value = DiffMoM._dda_operator_field_bigfloat_3d(operator, input, voxel)
        for component in 1:3
            output[3voxel - 3 + component] = value[component]
        end
    end
    return output
end

@testset "DDA recomputes only affected cancellation components" begin
    grid = VoxelGrid3D((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), 2, 2, 2)
    k = 2pi
    incident = planewave_dda_3d(grid, Vec3(0, 0, k), 1.0 + 0.2im,
                               Vec3(1, 0.3, 0))
    result = solve_dda_3d(grid, k, 2.5 - 0.05im, incident)
    operator = dda_operator_3d(grid, k, 2.5 - 0.05im)
    input = reduce(vcat, result.E_total)
    output = similar(input)
    reference = similar(input)
    _dda_full_precision_reference!(reference, operator, input)
    mul!(output, operator, input)
    @test output ≈ reduce(vcat, incident) rtol=1e-12 atol=1e-14
    # Each z component cancels in this ordinary plane-wave solve. It must
    # still receive the same 2304-bit recomputation as the full-vector path.
    @test output[3:3:end] == reference[3:3:end]
    for voxel in 1:grid.nvoxels, component in 1:3
        @test DiffMoM._dda_operator_component_bigfloat_3d(
            operator, input, voxel, component) == reference[3voxel - 3 + component]
    end

    # Compare the allocation of the actual public matvec with the previous
    # full-vector fallback; ratios accommodate platform object-size variance.
    full_bytes = @allocated _dda_full_precision_reference!(reference, operator, input)
    product_bytes = @allocated mul!(output, operator, input)
    @test product_bytes < 0.8full_bytes

    # The selected component path must keep the public aliasing and scaled
    # accumulation contract when the input and previous output share storage.
    alpha, beta = 0.7 + 0.2im, -0.1 + 0.3im
    aliased = copy(input)
    mul!(aliased, operator, aliased, alpha, beta)
    @test aliased ≈ alpha .* output .+ beta .* input rtol=32eps(Float64)
end
