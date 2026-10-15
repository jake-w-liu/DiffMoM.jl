module DDAWideScratchResourceTests
using DiffMoM, Test, LinearAlgebra, StaticArrays, Random

function immutable_reference(A, x)
    output = similar(x)
    for voxel in 1:A.grid.nvoxels
        value = DiffMoM._dda_operator_field_bigfloat_3d(A, x, voxel)
        output[3voxel-2:3voxel] = value
    end
    output
end

@testset "DDA reusable 2304-bit scratch matches immutable physical kernels" begin
    grid = VoxelGrid3D((-.2, .2), (-.2, .2), (-.2, .2), 2, 2, 2)
    rng = MersenneTwister(29184)
    normal = randn(rng, ComplexF64, 3grid.nvoxels)
    tensor = SMatrix{3,3,ComplexF64}(
        [1e-4-2e-5im 3e-6im 1e-6;
         3e-6im 1.2e-4-1e-5im -2e-6im;
         1e-6 -2e-6im 9e-5-1e-5im])
    materials = (ComplexF64[isodd(j) ? 1e-4-2e-5im : 0 for j in 1:grid.nvoxels],
        [isodd(j) ? tensor : zero(tensor) for j in 1:grid.nvoxels])
    work = DiffMoM._dda_wide_scratch()
    @test all(x -> precision(x) == 2304, work)
    for alpha in materials
        A = DDAOperator3D(grid, 2pi, fill(2.5-.05im, grid.nvoxels), alpha, false)
        for scale in (1., 0x1p-1000, 0x1p900),
                voxel in 1:grid.nvoxels
            x = scale .* normal
            expected = DiffMoM._dda_operator_field_bigfloat_3d(A, x, voxel)
            @test DiffMoM._dda_operator_wide_3d!(work, A, x, voxel, 0) == expected
            for component in 1:3
                scalar = DiffMoM._dda_operator_component_bigfloat_3d(A, x, voxel, component)
                @test DiffMoM._dda_operator_wide_3d!(work, A, x, voxel, component)[component] == scalar
            end
        end
        tiny = fill(ComplexF64(nextfloat(0.), -nextfloat(0.)), 3grid.nvoxels)
        for voxel in 1:grid.nvoxels
            @test DiffMoM._dda_operator_wide_3d!(work, A, tiny, voxel, 0) ==
                DiffMoM._dda_operator_field_bigfloat_3d(A, tiny, voxel)
        end
    end

    incident = planewave_dda_3d(grid, Vec3(0, 0, 2pi), 1+.2im, Vec3(1, .3, 0))
    solved = solve_dda_3d(grid, 2pi, 2.5-.05im, incident)
    A = dda_operator_3d(grid, 2pi, 2.5-.05im)
    x = reduce(vcat, solved.E_total)
    expected = immutable_reference(A, x)
    output = A*x
    @test output ≈ reduce(vcat, incident) rtol=1e-12 atol=1e-14
    @test output[3:3:end] == expected[3:3:end]
    saved = copy(x)
    for bits in (32, 8192)
        before = (precision(BigFloat), rounding(BigFloat))
        scoped = setprecision(BigFloat, bits) do
            A*x
        end
        @test scoped == output
        @test (precision(BigFloat), rounding(BigFloat)) == before
    end
    for mode in (RoundUp, RoundDown)
        setrounding(BigFloat, mode) do
            reference = immutable_reference(A, x)
            product = A*x
            @test product[3:3:end] == reference[3:3:end]
            @test rounding(BigFloat) == mode
        end
    end
    alpha, beta = .7+.2im, -.1+.3im
    aliased = copy(x)
    mul!(aliased, A, aliased, alpha, beta)
    @test aliased ≈ alpha .* output .+ beta .* x rtol=32eps(Float64)
    @test x == saved
    tasks = [Threads.@spawn(A*x) for _ in 1:4]
    @test all(task -> fetch(task) == output, tasks)
    @test x == saved
    @test A.alpha == dda_operator_3d(grid, 2pi, 2.5-.05im).alpha

    # Ordinary non-cancelling matvecs must never allocate a wide workspace.
    random = randn(rng, ComplexF64, length(x))
    mul!(output, A, random)
    @test (@allocated mul!(output, A, random)) < 1024
end

@testset "DDA solved-field fallback allocation stays bounded at 64 voxels" begin
    grid = VoxelGrid3D((-.2, .2), (-.2, .2), (-.2, .2), 4, 4, 4)
    incident = planewave_dda_3d(grid, Vec3(0, 0, 2pi), 1+.2im, Vec3(1, .3, 0))
    solved = solve_dda_3d(grid, 2pi, 2.5-.05im, incident)
    A = dda_operator_3d(grid, 2pi, 2.5-.05im)
    x = reduce(vcat, solved.E_total)
    output = similar(x)
    mul!(output, A, x)
    # The immutable single-component path allocated 185 MB on this normal
    # solved-field workload. The fixed scratch keeps Julia objects constant;
    # MPFR still owns bounded native transcendental temporaries per pair.
    @test (@allocated mul!(output, A, x)) < 32_000_000
    @test output ≈ reduce(vcat, incident) rtol=1e-12 atol=1e-14
end
end
