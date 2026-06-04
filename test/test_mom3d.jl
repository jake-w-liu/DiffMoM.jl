# Test 46: 3D vector material DDA solver

using Test
using LinearAlgebra

if !isdefined(Main, :DiffMoM)
    using DiffMoM
end

println("\n── Test 46: 3D vector material DDA solver ──")

@testset "3D vector material DDA solver" begin
    k0 = 2π

    @testset "Free-space limit" begin
        grid = VoxelGrid3D((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 2, 1, 1)
        E_inc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
        res = solve_dda_3d(grid, k0, 1.0 + 0im, E_inc)

        @test norm(reduce(vcat, res.E_total) - reduce(vcat, E_inc)) < 1e-13
        @test norm(scattered_field_dda_3d(res, [Vec3(1.0, 0.0, 0.0)])[1]) < 1e-13
        @test all(iszero, res.alpha)
    end

    @testset "Reciprocal dyadic block symmetry" begin
        grid = VoxelGrid3D((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 2, 1, 1)
        A, alpha, epsv = assemble_dda_3d(grid, k0, 2.5 + 0im)
        @test all(epsv .== 2.5 + 0im)
        @test alpha[1] == alpha[2]

        block12 = A[1:3, 4:6]
        block21 = A[4:6, 1:3]
        @test norm(block12 - transpose(block21)) < 1e-13
    end

    @testset "Single-voxel Rayleigh dipole far field" begin
        grid = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 1, 1, 1)
        epsr = 2.5 + 0im
        E_inc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
        res = solve_dda_3d(grid, k0, epsr, E_inc)

        q = induced_dipoles_dda_3d(res)[1]
        n = Vec3(0.0, 1.0, 0.0)
        I3 = Matrix{Float64}(I, 3, 3)
        expected = (k0^2 / (4π)) * ((I3 - n * transpose(n)) * q) *
                   exp(1im * k0 * dot(n, grid.centers[1]))

        @test norm(farfield_dda_3d(res, n) - expected) / norm(expected) < 1e-13
        @test abs(res.alpha[1] - clausius_mossotti_polarizability(epsr, grid.volumes[1])) < 1e-16
    end

    @testset "Anisotropic tensor polarizability" begin
        grid = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 1, 1, 1)
        eps_tensor = ComplexF64[
            2.5  0.12 0.0
            0.04 1.8  0.0
            0.0  0.0  1.0
        ]
        E_inc = [CVec3(1.0 + 0im, 0.25 + 0im, 0.0 + 0im)]
        res = solve_dda_3d(grid, k0, eps_tensor, E_inc)
        alpha_expected = clausius_mossotti_polarizability(eps_tensor, grid.volumes[1])

        @test res.alpha[1] ≈ alpha_expected atol=1e-16
        @test res.E_total[1] ≈ E_inc[1] atol=1e-14
        @test induced_dipoles_dda_3d(res)[1] ≈ alpha_expected * E_inc[1] atol=1e-16
    end

    @testset "Matrix-free operator equivalence and storage" begin
        grid = VoxelGrid3D((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 3, 3, 3)
        epsv = fill(2.5 + 0.1im, grid.nvoxels)
        A_dense, _, _ = assemble_dda_3d(grid, k0, epsv)
        A_op = dda_operator_3d(grid, k0, epsv)

        x = ComplexF64[sin(0.17 * i) + 1im * cos(0.11 * i) for i in 1:size(A_op, 2)]
        y = zeros(ComplexF64, size(A_op, 1))
        mul!(y, A_op, x)
        @test norm(y - A_dense * x) / norm(A_dense * x) < 1e-13

        A_adj = adjoint(A_op)
        y_adj = zeros(ComplexF64, size(A_adj, 1))
        mul!(y_adj, A_adj, x)
        @test norm(y_adj - adjoint(A_dense) * x) / norm(adjoint(A_dense) * x) < 1e-13

        # The matrix-free operator stores O(N) material/geometric data instead
        # of the O(N^2) dense interaction matrix.
        @test Base.summarysize(A_op) < Base.summarysize(A_dense) / 20

        mul!(y, A_op, x)  # warm-up before allocation probe
        @test (@allocated mul!(y, A_op, x)) < 1024

        eps_tensor = [ComplexF64[
            2.4 + 0.02im 0.03          0.0
            0.01          1.7 + 0.01im 0.0
            0.0           0.0          1.2
        ] for _ in 1:grid.nvoxels]
        A_tensor_dense, _, _ = assemble_dda_3d(grid, k0, eps_tensor)
        A_tensor_op = dda_operator_3d(grid, k0, eps_tensor)
        y_tensor = zeros(ComplexF64, size(A_tensor_op, 1))
        mul!(y_tensor, A_tensor_op, x)
        @test norm(y_tensor - A_tensor_dense * x) / norm(A_tensor_dense * x) < 1e-13
    end

    @testset "Matrix-free GMRES solve agrees with dense direct" begin
        grid = VoxelGrid3D((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 2, 2, 2)
        epsv = fill(2.5 + 0.05im, grid.nvoxels)
        E_inc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))

        res_direct = solve_dda_3d(grid, k0, epsv, E_inc)
        res_gmres = solve_dda_3d(grid, k0, epsv, E_inc;
                                 solver=:gmres, tol=1e-11, maxiter=100)

        E_direct = reduce(vcat, res_direct.E_total)
        E_gmres = reduce(vcat, res_gmres.E_total)
        @test norm(E_gmres - E_direct) / norm(E_direct) < 1e-10
        @test res_gmres.A isa DDAOperator3D
        @test res_gmres.A_LU === nothing
        @test res_gmres.solver == :gmres
    end

    @testset "Voxelized small dielectric sphere polarizability" begin
        a = 0.05
        lambda = 10.0
        k_small = 2π / lambda
        eps_sphere = 2.5 + 0im
        grid = VoxelGrid3D((-a, a), (-a, a), (-a, a), 7, 7, 7)
        epsv = ones(ComplexF64, grid.nvoxels)
        inside = 0
        for j in 1:grid.nvoxels
            if norm(grid.centers[j]) <= a
                epsv[j] = eps_sphere
                inside += 1
            end
        end
        @test inside > 0

        E_inc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k_small), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
        res = solve_dda_3d(grid, k_small, epsv, E_inc)
        q_total = sum(induced_dipoles_dda_3d(res))

        alpha_rayleigh = 4π * a^3 * (eps_sphere - 1) / (eps_sphere + 2)
        rel_err = abs(q_total[1] - alpha_rayleigh) / abs(alpha_rayleigh)

        @test abs(q_total[2]) / abs(q_total[1]) < 1e-10
        @test abs(q_total[3]) / abs(q_total[1]) < 1e-10
        @test rel_err < 0.02

        rhat = Vec3(0.0, 1.0, 0.0)
        F_dda = farfield_dda_3d(res, rhat)
        sigma_dda = 4π * real(dot(F_dda, F_dda))
        sigma_mie = mie_bistatic_rcs_dielectric(k_small, a,
                                                Vec3(0.0, 0.0, 1.0),
                                                Vec3(1.0, 0.0, 0.0),
                                                rhat, eps_sphere)
        sigma_rayleigh = 4π * k_small^4 * a^6 *
                         abs2((eps_sphere - 1) / (eps_sphere + 2))

        @test abs(sigma_mie - sigma_rayleigh) / sigma_rayleigh < 1e-3
        @test abs(sigma_dda - sigma_mie) / sigma_mie < 0.06
    end
end

println("  PASS ✓")
