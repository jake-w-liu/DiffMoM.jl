using DiffMoM, Test, LinearAlgebra

function _check_fft_plan_reuse(operator, direct)
    x = ComplexF64[sin(0.19i) + im*cos(0.07i) for i in 1:size(operator, 2)]
    reference = direct*x
    y = similar(x)
    mul!(y, operator, x)
    @test y ≈ reference rtol=1e-12
    mul!(y, operator, x)
    @test (@allocated mul!(y, operator, x)) == 0
    fill!(y, 0.2-0.3im)
    mul!(y, operator, x, 0.3+0.2im, -0.4+0.1im)
    @test y ≈ (0.3+0.2im)*reference .+ (-0.4+0.1im)*(0.2-0.3im) rtol=1e-12
    aliased = copy(x)
    mul!(aliased, operator, aliased)
    @test aliased ≈ reference rtol=1e-12
    # Plans and their buffers share the existing operator lock.
    tasks = [Threads.@spawn(operator * (scale*x)) for scale in (1, 2im)]
    @test fetch.(tasks) ≈ [reference, 2im*reference] rtol=1e-12
end

@testset "FFT operators reuse batched plans without matvec allocations" begin
    for dims in ((2,1,1), (2,2,2))
        grid = VoxelGrid3D((-.2,.2), (-.2,.2), (-.2,.2), dims...)
        _check_fft_plan_reuse(fft_dda_operator_3d(grid, 2pi, 2.5-.05im),
                             dda_operator_3d(grid, 2pi, 2.5-.05im))
        _check_fft_plan_reuse(fft_em_dda_operator_3d(grid, 2pi, 2.5-.05im, 1.),
                             em_dda_operator_3d(grid, 2pi, 2.5-.05im, 1.))
    end
end
