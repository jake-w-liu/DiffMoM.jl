# Focused tests for FFT-accelerated 3D DDA matvecs

using Test
using LinearAlgebra

if isdefined(Main, :DiffMoM)
    using .DiffMoM
else
    using DiffMoM
end

println("\n── Test 47: FFT-accelerated 3D DDA/EM-DDA matvec ──")

function _test_shared_fft_operator_concurrency(A, inputs)
    references = [A * input for input in inputs]
    Threads.nthreads() > 1 || return

    for _ in 1:4
        gate = Base.Event()
        tasks = map(eachindex(inputs)) do i
            Threads.@spawn begin
                wait(gate)
                A * inputs[i]
            end
        end
        yield()
        notify(gate)
        results = fetch.(tasks)
        @test results ≈ references rtol=1e-12
    end
end

@testset "FFT-accelerated 3D DDA/EM-DDA matvec" begin
    k0 = 2π

    single = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 1, 1, 1)
    @test_throws ArgumentError fft_dda_kernel_3d(single, Inf)
    @test_throws ArgumentError fft_dda_operator_3d(single, Inf, 3.0)
    @test_throws ArgumentError fft_em_dda_kernel_3d(single, Inf)
    @test_throws ArgumentError fft_em_dda_operator_3d(
        single, Inf, 3.0, 1.2)
    A_single = fft_dda_operator_3d(single, k0, 3.0 + 0.1im)
    @test size(A_single, 3) == 1
    @test_throws BoundsError size(A_single, 0)
    x_single = ComplexF64[1.0 + 2.0im, -0.5 + 0.25im, 0.75 - 0.1im]
    y_single = zeros(ComplexF64, 3)
    mul!(y_single, A_single, x_single)
    @test y_single ≈ x_single
    @test A_single.kernel.interaction_scale == single.volumes[1]

    tiny_spacing = 1.0e-103
    tiny_grid = VoxelGrid3D(
        (0.0, 2tiny_spacing),
        (0.0, tiny_spacing),
        (0.0, tiny_spacing),
        2, 1, 1,
    )
    tiny_direct = dda_operator_3d(tiny_grid, 1.0, 2.0)
    tiny_fft = fft_dda_operator_3d(tiny_grid, 1.0, 2.0)
    tiny_x = ComplexF64[
        1.0 + 0.2im, -0.4 + 0.3im, 0.7 - 0.1im,
        -0.2 + 0.5im, 0.8 - 0.4im, 0.1 + 0.6im,
    ]
    @test tiny_fft.kernel.interaction_scale == tiny_grid.volumes[1]
    @test all(isfinite, tiny_fft.kernel.kernel_hat)
    @test all(isfinite, tiny_fft * tiny_x)
    @test tiny_fft * tiny_x ≈ tiny_direct * tiny_x rtol=8eps(Float64)

    tiny_em_direct = em_dda_operator_3d(tiny_grid, 1.0, 2.0, 1.5)
    tiny_em_fft = fft_em_dda_operator_3d(tiny_grid, 1.0, 2.0, 1.5)
    tiny_em_x = ComplexF64[
        sin(index) + 1im * cos(0.2index) for index in 1:12
    ]
    @test tiny_em_fft.kernel.interaction_scale == tiny_grid.volumes[1]
    @test all(isfinite, tiny_em_fft.kernel.kernel_hat)
    @test all(isfinite, tiny_em_fft * tiny_em_x)
    @test tiny_em_fft * tiny_em_x ≈ tiny_em_direct * tiny_em_x rtol=8eps(Float64)

    cancellation_alpha = zeros(ComplexF64, 6, 6)
    cancellation_alpha[1, 1] = 1.0e308
    cancellation_alpha[1, 2] = -1.0e308
    cancellation_direct = em_dda_operator_3d(tiny_grid, 1.0, cancellation_alpha)
    cancellation_fft = fft_em_dda_operator_3d(tiny_grid, 1.0, cancellation_alpha)
    cancellation_x = ComplexF64[
        2, 2, 0, 0, 0, 0,
        2, 2, 0, 0, 0, 0,
    ]
    @test cancellation_direct * cancellation_x == cancellation_x
    @test cancellation_fft * cancellation_x == cancellation_x

    range_grid = VoxelGrid3D(
        (0.0, 2.0), (0.0, 1.0), (0.0, 1.0), 2, 1, 1)
    range_k = 1.0
    axial_green = DiffMoM._electric_dipole_apply_3d(
        range_grid.centers[1],
        range_grid.centers[2],
        range_k,
        CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im),
    )[1]
    target_alpha = inv(axial_green)
    range_epsr = (3 + 2target_alpha) / (3 - target_alpha)
    range_direct = dda_operator_3d(range_grid, range_k, range_epsr)
    range_fft = fft_dda_operator_3d(range_grid, range_k, range_epsr)
    range_amplitude = 0.5floatmax(Float64)
    range_x = ComplexF64[
        range_amplitude, 0.0, 0.0,
        range_amplitude, 0.0, 0.0,
    ]
    @test_throws OverflowError DiffMoM._scaled_alpha_apply_3d(
        range_fft.alpha[1],
        CVec3(range_amplitude + 0im, 0.0 + 0im, 0.0 + 0im),
        range_fft.kernel.interaction_scale,
        "FFT range regression",
        1,
    )
    range_direct_y = zeros(ComplexF64, length(range_x))
    range_fft_y = similar(range_direct_y)
    mul!(range_direct_y, range_direct, range_x)
    mul!(range_fft_y, range_fft, range_x)
    @test all(isfinite, range_direct_y)
    @test range_fft_y == range_direct_y

    range_initial_y = ComplexF64[
        0.01index - 0.02im * index for index in eachindex(range_x)]
    range_direct_scaled = copy(range_initial_y)
    range_fft_scaled = copy(range_initial_y)
    mul!(range_direct_scaled, range_direct, range_x,
         0.3 - 0.2im, -0.4 + 0.1im)
    mul!(range_fft_scaled, range_fft, range_x,
         0.3 - 0.2im, -0.4 + 0.1im)
    @test range_fft_scaled == range_direct_scaled

    range_em_direct = em_dda_operator_3d(
        range_grid, range_k, range_epsr, 1.0)
    range_em_fft = fft_em_dda_operator_3d(
        range_grid, range_k, range_epsr, 1.0)
    range_em_x = ComplexF64[
        range_amplitude, 0.0, 0.0, 0.0, 0.0, 0.0,
        range_amplitude, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    @test_throws OverflowError DiffMoM._scaled_alpha_apply_3d(
        range_em_fft.alpha[1],
        DiffMoM._CVec6DDA(
            range_amplitude, 0.0, 0.0, 0.0, 0.0, 0.0),
        range_em_fft.kernel.interaction_scale,
        "FFT EM-DDA range regression",
        1,
    )
    range_em_direct_y = zeros(ComplexF64, length(range_em_x))
    range_em_fft_y = similar(range_em_direct_y)
    mul!(range_em_direct_y, range_em_direct, range_em_x)
    mul!(range_em_fft_y, range_em_fft, range_em_x)
    @test all(isfinite, range_em_direct_y)
    @test range_em_fft_y == range_em_direct_y

    transform_overflow_scale = 4.0
    transform_overflow_kernel = FFTDDAKernel3D(
        range_fft.kernel.pad_dims,
        transform_overflow_scale *
            range_fft.kernel.interaction_scale,
        transform_overflow_scale .* range_fft.kernel.kernel_hat,
    )
    transform_overflow_fft = FFTDDAOperator3D(
        range_grid,
        range_k,
        range_fft.eps_r,
        range_fft.alpha,
        false,
        transform_overflow_kernel,
        zeros(ComplexF64,
              transform_overflow_kernel.pad_dims..., 3),
        zeros(ComplexF64, transform_overflow_kernel.pad_dims...),
    )
    transform_overflow_y = similar(range_direct_y)
    mul!(transform_overflow_y, transform_overflow_fft, range_x)
    @test !all(isfinite, transform_overflow_fft.qhat)
    @test transform_overflow_y == range_direct_y

    convolution_overflow_scale = 8.0
    convolution_overflow_kernel = FFTDDAKernel3D(
        range_fft.kernel.pad_dims,
        convolution_overflow_scale *
            range_fft.kernel.interaction_scale,
        convolution_overflow_scale .* range_fft.kernel.kernel_hat,
    )
    convolution_overflow_fft = FFTDDAOperator3D(
        range_grid,
        range_k,
        range_fft.eps_r,
        range_fft.alpha,
        false,
        convolution_overflow_kernel,
        zeros(ComplexF64,
              convolution_overflow_kernel.pad_dims..., 3),
        zeros(ComplexF64, convolution_overflow_kernel.pad_dims...),
    )
    convolution_initial_y = copy(range_initial_y)
    convolution_direct_y = copy(convolution_initial_y)
    convolution_fft_y = copy(convolution_initial_y)
    mul!(convolution_direct_y, range_direct, range_x,
         0.3 - 0.2im, -0.4 + 0.1im)
    mul!(convolution_fft_y, convolution_overflow_fft, range_x,
         0.3 - 0.2im, -0.4 + 0.1im)
    @test all(isfinite, convolution_overflow_fft.qhat)
    @test !DiffMoM._fft_convolution_range_safe_3d(
        DiffMoM._fft_array_component_scale_3d(
            convolution_overflow_kernel.kernel_hat),
        DiffMoM._fft_array_component_scale_3d(
            convolution_overflow_fft.qhat),
        3,
        length(convolution_overflow_fft.conv),
    )
    @test convolution_fft_y == convolution_direct_y

    em_convolution_kernel = FFTEMDDAKernel3D(
        range_em_fft.kernel.pad_dims,
        convolution_overflow_scale *
            range_em_fft.kernel.interaction_scale,
        convolution_overflow_scale .* range_em_fft.kernel.kernel_hat,
    )
    em_convolution_fft = FFTEMDDAOperator3D(
        range_grid,
        range_k,
        range_em_fft.alpha,
        false,
        em_convolution_kernel,
        zeros(ComplexF64, em_convolution_kernel.pad_dims..., 6),
        zeros(ComplexF64, em_convolution_kernel.pad_dims...),
    )
    em_convolution_y = similar(range_em_direct_y)
    mul!(em_convolution_y, em_convolution_fft, range_em_x)
    @test all(isfinite, em_convolution_fft.qhat)
    @test em_convolution_y == range_em_direct_y

    grid = VoxelGrid3D((-0.15, 0.15), (-0.1, 0.1), (-0.05, 0.05), 3, 2, 2)
    epsv = ComplexF64[2.2 + 0.03im + 0.01 * sin(j) for j in 1:grid.nvoxels]

    A_direct = dda_operator_3d(grid, k0, epsv)
    A_fft = fft_dda_operator_3d(grid, k0, epsv)

    @test size(A_fft) == size(A_direct)
    @test A_fft.alpha == A_direct.alpha
    @test A_fft.eps_r == A_direct.eps_r
    @test A_fft.kernel.pad_dims == (2grid.nx - 1, 2grid.ny - 1, 2grid.nz - 1)
    @test A_fft.kernel.interaction_scale == grid.volumes[1]
    @test A_fft.work_lock isa ReentrantLock

    x = ComplexF64[sin(0.19 * i) + 1im * cos(0.07 * i) for i in 1:size(A_fft, 2)]
    y_direct = zeros(ComplexF64, size(A_direct, 1))
    y_fft = zeros(ComplexF64, size(A_fft, 1))

    mul!(y_direct, A_direct, x)
    mul!(y_fft, A_fft, x)

    @test norm(y_fft - y_direct) / norm(y_direct) < 1e-12
    fill!(y_fft, ComplexF64(NaN, NaN))
    mul!(y_fft, A_fft, x, 1.0 + 0im, 0.0 + 0im)
    @test y_fft ≈ y_direct rtol=1e-12

    mul!(y_fft, A_fft, x)  # warm-up allocation probe
    @test (@allocated mul!(y_fft, A_fft, x)) < 8192

    y_scaled_direct = ComplexF64[0.01 * i - 0.02im * i for i in 1:size(A_direct, 1)]
    y_scaled_fft = copy(y_scaled_direct)
    mul!(y_scaled_direct, A_direct, x, 0.3 - 0.2im, -0.4 + 0.1im)
    mul!(y_scaled_fft, A_fft, x, 0.3 - 0.2im, -0.4 + 0.1im)

    @test norm(y_scaled_fft - y_scaled_direct) / norm(y_scaled_direct) < 1e-12
    overlap_storage = vcat(x, 0.0 + 0im)
    overlap_x = view(overlap_storage, 1:length(x))
    overlap_y = view(overlap_storage, 2:(length(x) + 1))
    overlap_expected = A_direct * copy(overlap_x)
    mul!(overlap_y, A_fft, overlap_x)
    @test overlap_y ≈ overlap_expected rtol=1e-12
    _test_shared_fft_operator_concurrency(
        A_fft,
        [x, (0.2 - 0.3im) .* x, reverse(x), conj.(x)],
    )

    eps_tensor = [ComplexF64[
        2.4  0.03 0.0
        0.01 1.7  0.0
        0.0  0.0  1.2
    ] for _ in 1:grid.nvoxels]
    A_tensor_direct = dda_operator_3d(grid, k0, eps_tensor)
    A_tensor_fft = fft_dda_operator_3d(grid, k0, eps_tensor)
    y_tensor_direct = zeros(ComplexF64, size(A_tensor_direct, 1))
    y_tensor_fft = zeros(ComplexF64, size(A_tensor_fft, 1))
    mul!(y_tensor_direct, A_tensor_direct, x)
    mul!(y_tensor_fft, A_tensor_fft, x)

    @test norm(y_tensor_fft - y_tensor_direct) / norm(y_tensor_direct) < 1e-12

    A_em_direct = em_dda_operator_3d(grid, k0, epsv, 1.3 + 0.02im)
    A_em_fft = fft_em_dda_operator_3d(grid, k0, epsv, 1.3 + 0.02im)
    @test size(A_em_fft) == size(A_em_direct)
    @test size(A_em_fft, 3) == 1
    @test_throws BoundsError size(A_em_fft, -1)
    @test A_em_fft.alpha == A_em_direct.alpha
    @test A_em_fft.kernel.pad_dims == A_fft.kernel.pad_dims
    @test A_em_fft.kernel.interaction_scale == grid.volumes[1]
    @test A_em_fft.work_lock isa ReentrantLock

    x_em = ComplexF64[sin(0.09 * i) + 1im * cos(0.05 * i) for i in 1:size(A_em_fft, 2)]
    y_em_direct = zeros(ComplexF64, size(A_em_direct, 1))
    y_em_fft = zeros(ComplexF64, size(A_em_fft, 1))
    mul!(y_em_direct, A_em_direct, x_em)
    mul!(y_em_fft, A_em_fft, x_em)
    @test norm(y_em_fft - y_em_direct) / norm(y_em_direct) < 1e-12
    fill!(y_em_fft, ComplexF64(NaN, NaN))
    mul!(y_em_fft, A_em_fft, x_em, 1.0 + 0im, 0.0 + 0im)
    @test y_em_fft ≈ y_em_direct rtol=1e-12

    overlap_em_storage = vcat(x_em, 0.0 + 0im)
    overlap_em_x = view(overlap_em_storage, 1:length(x_em))
    overlap_em_y = view(overlap_em_storage, 2:(length(x_em) + 1))
    overlap_em_expected = A_em_direct * copy(overlap_em_x)
    mul!(overlap_em_y, A_em_fft, overlap_em_x)
    @test overlap_em_y ≈ overlap_em_expected rtol=1e-12

    mul!(y_em_fft, A_em_fft, x_em)
    @test (@allocated mul!(y_em_fft, A_em_fft, x_em)) < 32768
    _test_shared_fft_operator_concurrency(
        A_em_fft,
        [x_em, (0.2 - 0.3im) .* x_em, reverse(x_em), conj.(x_em)],
    )

    @testset "Scaled-output exponent range" begin
        scale_grid = VoxelGrid3D(
            (0.0, 0.2), (0.0, 0.1), (0.0, 0.1), 2, 1, 1)
        operators_and_inputs = (
            (
                fft_dda_operator_3d(
                    scale_grid, k0, 2.5 + 0.1im),
                ComplexF64[
                    10 * (sin(0.17 * i) + 1im * cos(0.11 * i))
                    for i in 1:6
                ],
            ),
            (
                fft_em_dda_operator_3d(
                    scale_grid, k0, 2.5 + 0.1im, 1.3 + 0.02im),
                ComplexF64[
                    10 * (sin(0.17 * i) + 1im * cos(0.11 * i))
                    for i in 1:12
                ],
            ),
        )
        scale = 1.0e308 + 0im

        for (operator, input) in operators_and_inputs
            product = operator * input
            previous = -product
            @test all(isfinite, product)
            @test any(!isfinite, scale .* product)
            reference = setprecision(BigFloat, 4608) do
                ComplexF64[
                    Complex{BigFloat}(scale) *
                        Complex{BigFloat}(product[i]) +
                    Complex{BigFloat}(scale) *
                        Complex{BigFloat}(previous[i])
                    for i in eachindex(product)
                ]
            end
            @test all(isfinite, reference)
            result = copy(previous)
            mul!(result, operator, input, scale, scale)
            @test result == reference
        end
    end

    E_inc, H_inc = planewave_em_dda_3d(
        single, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0),
    )
    res_fft = solve_em_dda_3d(single, k0, 2.3 + 0.0im, 1.4 + 0.0im,
                              E_inc, H_inc; solver=:fft_gmres, tol=1e-13)
    @test res_fft.A isa FFTEMDDAOperator3D
    @test res_fft.A_LU === nothing
    @test res_fft.solver == :fft_gmres
    @test res_fft.E_total[1] ≈ E_inc[1] atol=1e-13
    @test res_fft.H_total[1] ≈ H_inc[1] atol=1e-13

    @test_throws ErrorException solve_em_dda_3d(
        grid, k0, epsv, 1.3 + 0.02im,
        planewave_em_dda_3d(
            grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0),
        )...;
        solver=:fft_gmres, tol=1e-14, maxiter=1, memory=1,
    )
end

println("  PASS")
