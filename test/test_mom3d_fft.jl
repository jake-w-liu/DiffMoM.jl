# Focused tests for FFT-accelerated 3D DDA matvecs

using Test
using LinearAlgebra

if isdefined(Main, :DiffMoM)
    using .DiffMoM
else
    using DiffMoM
end

println("\n── Test 47: FFT-accelerated 3D DDA/EM-DDA matvec ──")

@testset "FFT-accelerated 3D DDA/EM-DDA matvec" begin
    k0 = 2π

    single = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 1, 1, 1)
    A_single = fft_dda_operator_3d(single, k0, 3.0 + 0.1im)
    x_single = ComplexF64[1.0 + 2.0im, -0.5 + 0.25im, 0.75 - 0.1im]
    y_single = zeros(ComplexF64, 3)
    mul!(y_single, A_single, x_single)
    @test y_single ≈ x_single

    grid = VoxelGrid3D((-0.15, 0.15), (-0.1, 0.1), (-0.05, 0.05), 3, 2, 2)
    epsv = ComplexF64[2.2 + 0.03im + 0.01 * sin(j) for j in 1:grid.nvoxels]

    A_direct = dda_operator_3d(grid, k0, epsv)
    A_fft = fft_dda_operator_3d(grid, k0, epsv)

    @test size(A_fft) == size(A_direct)
    @test A_fft.alpha == A_direct.alpha
    @test A_fft.eps_r == A_direct.eps_r
    @test A_fft.kernel.pad_dims == (2grid.nx - 1, 2grid.ny - 1, 2grid.nz - 1)

    x = ComplexF64[sin(0.19 * i) + 1im * cos(0.07 * i) for i in 1:size(A_fft, 2)]
    y_direct = zeros(ComplexF64, size(A_direct, 1))
    y_fft = zeros(ComplexF64, size(A_fft, 1))

    mul!(y_direct, A_direct, x)
    mul!(y_fft, A_fft, x)

    @test norm(y_fft - y_direct) / norm(y_direct) < 1e-12

    mul!(y_fft, A_fft, x)  # warm-up allocation probe
    @test (@allocated mul!(y_fft, A_fft, x)) < 8192

    y_scaled_direct = ComplexF64[0.01 * i - 0.02im * i for i in 1:size(A_direct, 1)]
    y_scaled_fft = copy(y_scaled_direct)
    mul!(y_scaled_direct, A_direct, x, 0.3 - 0.2im, -0.4 + 0.1im)
    mul!(y_scaled_fft, A_fft, x, 0.3 - 0.2im, -0.4 + 0.1im)

    @test norm(y_scaled_fft - y_scaled_direct) / norm(y_scaled_direct) < 1e-12

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
    @test A_em_fft.alpha == A_em_direct.alpha
    @test A_em_fft.kernel.pad_dims == A_fft.kernel.pad_dims

    x_em = ComplexF64[sin(0.09 * i) + 1im * cos(0.05 * i) for i in 1:size(A_em_fft, 2)]
    y_em_direct = zeros(ComplexF64, size(A_em_direct, 1))
    y_em_fft = zeros(ComplexF64, size(A_em_fft, 1))
    mul!(y_em_direct, A_em_direct, x_em)
    mul!(y_em_fft, A_em_fft, x_em)
    @test norm(y_em_fft - y_em_direct) / norm(y_em_direct) < 1e-12

    mul!(y_em_fft, A_em_fft, x_em)
    @test (@allocated mul!(y_em_fft, A_em_fft, x_em)) < 32768

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
end

println("  PASS")
