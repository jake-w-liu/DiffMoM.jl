# Test 48: Coupled electric-magnetic 3D DDA solver

using Test
using LinearAlgebra

if isdefined(Main, :DiffMoM)
    using .DiffMoM
else
    using DiffMoM
end

println("\n── Test 48: Coupled electric-magnetic 3D DDA solver ──")

@testset "Coupled electric-magnetic 3D DDA solver" begin
    k0 = 2π

    @testset "Free-space magnetodielectric limit" begin
        grid = VoxelGrid3D((-0.1, 0.1), (-0.05, 0.05), (-0.05, 0.05), 2, 1, 1)
        E_inc, H_inc = planewave_em_dda_3d(
            grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0),
        )
        res = solve_em_dda_3d(grid, k0, 1.0 + 0im, 1.0 + 0im, E_inc, H_inc)

        @test norm(reduce(vcat, res.E_total) - reduce(vcat, E_inc)) < 1e-13
        @test norm(reduce(vcat, res.H_total) - reduce(vcat, H_inc)) < 1e-13
        q, m = induced_dipoles_em_dda_3d(res)
        @test all(iszero, q)
        @test all(iszero, m)

        Es, Hs = scattered_fields_em_dda_3d(res, [Vec3(1.0, 0.0, 0.0)])
        @test norm(Es[1]) < 1e-13
        @test norm(Hs[1]) < 1e-13
    end

    @testset "Electric-only reduction matches DDA" begin
        grid = VoxelGrid3D((-0.12, 0.12), (-0.05, 0.05), (-0.05, 0.05), 2, 1, 1)
        epsr = fill(2.4 + 0.04im, grid.nvoxels)
        E_inc, H_inc = planewave_em_dda_3d(
            grid, Vec3(0.0, 0.0, k0), 1.0 + 0.2im, Vec3(1.0, 0.0, 0.0),
        )

        res_e = solve_dda_3d(grid, k0, epsr, E_inc)
        res_em = solve_em_dda_3d(grid, k0, epsr, 1.0 + 0im, E_inc, H_inc)

        @test norm(reduce(vcat, res_em.E_total) - reduce(vcat, res_e.E_total)) /
              norm(reduce(vcat, res_e.E_total)) < 1e-13

        q_em, m_em = induced_dipoles_em_dda_3d(res_em)
        q_e = induced_dipoles_dda_3d(res_e)
        @test norm(reduce(vcat, q_em) - reduce(vcat, q_e)) / norm(reduce(vcat, q_e)) < 1e-13
        @test norm(reduce(vcat, m_em)) < 1e-13
    end

    @testset "Single-voxel magnetic response" begin
        grid = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 1, 1, 1)
        mur = 2.5 + 0im
        E_inc, H_inc = planewave_em_dda_3d(
            grid, Vec3(0.0, 0.0, k0), 2.0 + 0.1im, Vec3(1.0, 0.0, 0.0),
        )
        res = solve_em_dda_3d(grid, k0, 1.0 + 0im, mur, E_inc, H_inc)

        alpha_m = magnetic_clausius_mossotti_polarizability(mur, grid.volumes[1])
        q, m = induced_dipoles_em_dda_3d(res)

        @test res.E_total[1] ≈ E_inc[1] atol=1e-14
        @test res.H_total[1] ≈ H_inc[1] atol=1e-14
        @test norm(q[1]) < 1e-15
        @test m[1] ≈ alpha_m * H_inc[1] atol=1e-16
    end

    @testset "Explicit bianisotropic polarizability" begin
        grid = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 1, 1, 1)
        alpha6 = zeros(ComplexF64, 6, 6)
        alpha6[1, 1] = 1.0e-4
        alpha6[1, 5] = 2.0e-4 - 1.0e-5im
        alpha6[5, 1] = -3.0e-7 + 2.0e-8im
        alpha6[5, 5] = 4.0e-7
        alpha = BianisotropicPolarizability3D(alpha6)

        E_inc = [CVec3(1.0 + 0.1im, 0.2 - 0.3im, 0.0 + 0im)]
        H_inc = [CVec3(0.0 + 0im, 0.004 + 0.001im, 0.0 + 0im)]
        res = solve_em_dda_3d(grid, k0, alpha, E_inc, H_inc)

        x = ComplexF64[E_inc[1][1], E_inc[1][2], E_inc[1][3],
                       H_inc[1][1], H_inc[1][2], H_inc[1][3]]
        expected = alpha6 * x
        q, m = induced_dipoles_em_dda_3d(res)

        @test res.E_total[1] ≈ E_inc[1] atol=1e-14
        @test res.H_total[1] ≈ H_inc[1] atol=1e-14
        @test q[1] ≈ CVec3(expected[1], expected[2], expected[3]) atol=1e-16
        @test m[1] ≈ CVec3(expected[4], expected[5], expected[6]) atol=1e-16
    end

    @testset "Bianisotropic constitutive closure" begin
        grid = VoxelGrid3D((-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), 1, 1, 1)
        epsr = 2.4 + 0im
        mur = 1.7 + 0im
        C6 = Matrix{ComplexF64}(I, 6, 6)
        C6[1, 1] = epsr
        C6[2, 2] = epsr
        C6[3, 3] = epsr
        C6[4, 4] = mur
        C6[5, 5] = mur
        C6[6, 6] = mur
        material = BianisotropicMaterial3D(C6)
        alpha_mat = em_dda_polarizabilities(grid, k0, material)[1]
        alpha_em = em_dda_polarizabilities(grid, k0, epsr, mur)[1]
        @test alpha_mat ≈ alpha_em atol=1e-16

        C6[1, 5] = 0.02
        C6[5, 1] = 0.02
        coupled = BianisotropicMaterial3D(C6)
        alpha_coupled = bianisotropic_clausius_mossotti_polarizability(coupled, grid.volumes[1])
        @test abs(alpha_coupled[1, 5]) > 0
        @test abs(alpha_coupled[5, 1]) > 0
    end

    @testset "Matrix-free operator equivalence and storage" begin
        grid = VoxelGrid3D((-0.15, 0.15), (-0.1, 0.1), (-0.05, 0.05), 3, 3, 2)
        epsv = fill(2.3 + 0.03im, grid.nvoxels)
        muv = fill(1.4 + 0.02im, grid.nvoxels)

        A_dense, alpha = assemble_em_dda_3d(grid, k0, epsv, muv)
        A_op = em_dda_operator_3d(grid, k0, epsv, muv)
        @test A_op.alpha == alpha

        x = ComplexF64[sin(0.13 * i) + 1im * cos(0.17 * i) for i in 1:size(A_op, 2)]
        y = zeros(ComplexF64, size(A_op, 1))
        mul!(y, A_op, x)
        @test norm(y - A_dense * x) / norm(A_dense * x) < 1e-13
        @test Base.summarysize(A_op) < Base.summarysize(A_dense) / 4

        mul!(y, A_op, x)
        @test (@allocated mul!(y, A_op, x)) < 4096

        # The optimized dense builder fills each 6x6 voxel-pair block once; it
        # must be bit-identical to the generic per-entry `getindex` conversion.
        A_generic = Array{ComplexF64}(undef, size(A_op))
        for col in 1:size(A_op, 2), r in 1:size(A_op, 1)
            A_generic[r, col] = A_op[r, col]
        end
        @test A_dense == A_generic

        # Same property must hold for fully coupled (non block-diagonal) 6x6
        # polarizabilities so the block builder is exercised off the diagonal.
        a6 = zeros(ComplexF64, 6, 6)
        for d in 1:3
            a6[d, d] = 0.4 + 0.05im
            a6[d + 3, d + 3] = 0.2 + 0.01im
        end
        a6[1, 5] = 0.03 + 0.01im
        a6[5, 1] = 0.02 - 0.01im
        a6[3, 4] = 0.015 + 0.0im
        alpha6 = fill(a6, grid.nvoxels)
        A_dense_b, _ = assemble_em_dda_3d(grid, k0, alpha6)
        A_op_b = em_dda_operator_3d(grid, k0, alpha6)
        A_generic_b = Array{ComplexF64}(undef, size(A_op_b))
        for col in 1:size(A_op_b, 2), r in 1:size(A_op_b, 1)
            A_generic_b[r, col] = A_op_b[r, col]
        end
        @test A_dense_b == A_generic_b
        xb = ComplexF64[cos(0.07 * i) + 1im * sin(0.11 * i) for i in 1:size(A_op_b, 2)]
        yb = zeros(ComplexF64, size(A_op_b, 1))
        mul!(yb, A_op_b, xb)
        @test norm(yb - A_dense_b * xb) / norm(A_dense_b * xb) < 1e-13
    end

    @testset "Matrix-free GMRES solve agrees with dense direct" begin
        grid = VoxelGrid3D((-0.1, 0.1), (-0.05, 0.05), (-0.05, 0.05), 2, 1, 1)
        E_inc, H_inc = planewave_em_dda_3d(
            grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0),
        )
        res_direct = solve_em_dda_3d(grid, k0, 2.3 + 0.02im, 1.5 + 0.01im,
                                     E_inc, H_inc)
        res_gmres = solve_em_dda_3d(grid, k0, 2.3 + 0.02im, 1.5 + 0.01im,
                                    E_inc, H_inc;
                                    solver=:gmres, tol=1e-12, maxiter=50)

        @test norm(reduce(vcat, res_gmres.E_total) - reduce(vcat, res_direct.E_total)) /
              norm(reduce(vcat, res_direct.E_total)) < 1e-10
        @test norm(reduce(vcat, res_gmres.H_total) - reduce(vcat, res_direct.H_total)) /
              norm(reduce(vcat, res_direct.H_total)) < 1e-10
        @test res_gmres.A isa EMDDAOperator3D
        @test res_gmres.A_LU === nothing
        @test res_gmres.solver == :gmres
    end

    @testset "Magnetic coupling obeys radiation condition (regression)" begin
        # Guards the magnetic-dipole electric-field cross term. With exp(+iωt),
        # G=e^{-ikR}/(4πR), the field of a magnetic dipole is E = -ikη₀(∇G×m)
        # (a REAL far-field coefficient, dual to the electric dipole). A spurious
        # factor i there breaks the far-field radiation condition FE = -η₀(n̂×FH)
        # and the large-R scattered-field condition E_s ≈ -η₀(n̂×H_s) whenever a
        # scatterer has μ≠1 (m≠0). All other EM-DDA tests use μ=1 or a single
        # voxel, so none of them exercise this inter-voxel coupling.
        eta0 = 376.730313668
        grid = VoxelGrid3D((-0.1, 0.1), (-0.05, 0.05), (-0.05, 0.05), 3, 1, 1)
        E_inc, H_inc = planewave_em_dda_3d(
            grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0),
        )
        res = solve_em_dda_3d(grid, k0, 2.3 + 0.02im, 1.8 + 0.01im, E_inc, H_inc)
        _, m = induced_dipoles_em_dda_3d(res)
        @test norm(reduce(vcat, m)) > 0   # m≠0, so the cross term is active

        for th in range(0.3, π - 0.3, length=4), ph in range(0.0, 2π, length=4)
            n = Vec3(sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
            FE, FH = farfield_em_dda_3d(res, n)
            @test norm(FE + eta0 * cross(n, FH)) / max(norm(FE), eps()) < 1e-10
        end

        Rbig = 4.0e3   # ~640 wavelengths: deep far zone, 1/(kR) ~ 4e-5
        for nraw in (Vec3(0.3, 0.4, 0.866), Vec3(-0.5, 0.2, 0.84))
            n = nraw / norm(nraw)
            Es, Hs = scattered_fields_em_dda_3d(res, [Rbig * n])
            @test norm(Es[1] + eta0 * cross(n, Hs[1])) / max(norm(Es[1]), eps()) < 1e-3
        end
    end
end

println("  PASS")
