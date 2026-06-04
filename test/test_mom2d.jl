using Test
using DiffMoM
using LinearAlgebra

@testset "2D TM MoM" begin

    @testset "Mesh2D construction" begin
        mesh = Mesh2D((-1.0, 1.0), (-0.5, 0.5), 4, 2)
        @test mesh.ncells == 8
        @test mesh.nx == 4
        @test mesh.ny == 2
        @test mesh.dx ≈ 0.5
        @test mesh.dy ≈ 0.5
        @test mesh.cell_area ≈ 0.25

        # Centers should be at cell midpoints
        @test mesh.centers[1] ≈ Vec2(-0.75, -0.25)
        @test mesh.centers[8] ≈ Vec2(0.75, 0.25)

        # Equivalent radius: πa² = cell_area
        @test equivalent_radius(mesh)^2 * π ≈ mesh.cell_area atol=1e-14

        # Invalid inputs
        @test_throws AssertionError Mesh2D((1.0, -1.0), (-0.5, 0.5), 4, 2)
        @test_throws AssertionError Mesh2D((-1.0, 1.0), (-0.5, 0.5), 0, 2)
    end

    @testset "2D Green's function" begin
        k = 2π
        r1 = Vec2(0.0, 0.0)
        r2 = Vec2(1.0, 0.0)

        G = greens_2d(r1, r2, k)

        # G₂D = (-i/4) H₀⁽²⁾(kR), verify against direct computation
        using SpecialFunctions
        R = 1.0
        G_ref = (-im / 4) * besselh(0, 2, k * R)
        @test G ≈ G_ref atol=1e-14

        # Symmetry: G(r1,r2) = G(r2,r1)
        @test greens_2d(r1, r2, k) ≈ greens_2d(r2, r1, k) atol=1e-14

        # Self-term: returns zero (handled by self_cell_integral)
        @test greens_2d(r1, r1, k) == 0.0 + 0.0im

        # Decay with distance
        G_near = abs(greens_2d(r1, Vec2(0.5, 0.0), k))
        G_far = abs(greens_2d(r1, Vec2(5.0, 0.0), k))
        @test G_near > G_far
    end

    @testset "Self-cell integral" begin
        k = 2π
        a_eq = 0.01  # small cell

        D_self = self_cell_integral_2d(k, a_eq)
        @test !isnan(D_self)
        @test !isinf(D_self)

        # For very small ka, the self-cell integral should be dominated by
        # the logarithmic singularity of H₀⁽²⁾: G ~ (-i/4)(1 - 2i/π ln(kR/2) - ...)
        # Numerical check: should have nonzero real and imaginary parts
        @test abs(real(D_self)) > 0
        @test abs(imag(D_self)) > 0

        # Positive equivalent radius required
        @test_throws AssertionError self_cell_integral_2d(k, 0.0)
        @test_throws AssertionError self_cell_integral_2d(0.0, a_eq)
    end

    @testset "VIE assembly and solve" begin
        k0 = 2π
        mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 5, 5)
        chi = fill(1.0, mesh.ncells)  # εᵣ = 2

        Z, D = assemble_vie_2d(mesh, k0, chi)
        @test size(Z) == (25, 25)
        @test size(D) == (25, 25)

        # Z should be invertible
        @test cond(Z) < 1e10

        # With chi = 0 (free space), Z = I
        Z0, _ = assemble_vie_2d(mesh, k0, zeros(mesh.ncells))
        @test Z0 ≈ I(mesh.ncells) atol=1e-14

        # Solve with plane wave
        E_inc = planewave_2d(mesh, k0, 0.0)
        @test all(abs.(E_inc) .≈ 1.0)  # unit amplitude

        vr = solve_vie_2d(mesh, k0, chi, E_inc)
        @test length(vr.E_total) == 25
        @test !any(isnan, vr.E_total)
        @test !any(isinf, vr.E_total)

        # In free space (chi=0), total field = incident field
        vr_free = solve_vie_2d(mesh, k0, zeros(mesh.ncells), E_inc)
        @test vr_free.E_total ≈ E_inc atol=1e-12
    end

    @testset "Plane wave excitation" begin
        mesh = Mesh2D((-1.0, 1.0), (-1.0, 1.0), 4, 4)
        k0 = 2π

        # Different incident angles
        for phi_inc in [0.0, π/4, π/2, π]
            E_inc = planewave_2d(mesh, k0, phi_inc)
            @test all(abs.(E_inc) .≈ 1.0)  # unit amplitude for plane wave
        end

        # Phase consistency: E(r) = exp(-ik₀ k̂·r)
        E_inc = planewave_2d(mesh, k0, 0.0)
        for i in 1:mesh.ncells
            expected = exp(-im * k0 * mesh.centers[i][1])
            @test E_inc[i] ≈ expected atol=1e-14
        end
    end

    @testset "Mie series - PEC cylinder" begin
        k0 = 2π
        a = 0.5

        c, N = mie_coefficients_2d(k0, a, 1.0; pec=true)
        @test length(c) == 2N + 1
        @test !any(isnan, c)

        # PEC: cₙ = -Jₙ(k₀a) / Hₙ⁽²⁾(k₀a)
        using SpecialFunctions
        k0a = k0 * a
        c0_ref = -besselj(0, k0a) / besselh(0, 2, k0a)
        @test c[N + 1] ≈ c0_ref atol=1e-14  # n=0 coefficient

        # Total field on cylinder surface should be near zero for PEC
        # Tolerance limited by Mie series truncation at finite nmax
        r_surf = [Vec2(a * cos(phi), a * sin(phi)) for phi in range(0, 2π, length=37)[1:36]]
        E_total = mie_total_field_2d(k0, a, 1.0, r_surf; pec=true)
        @test maximum(abs.(E_total)) < 1e-6
    end

    @testset "Mie series - dielectric cylinder" begin
        k0 = 2π
        a = 0.3
        eps_r = 4.0

        c, N = mie_coefficients_2d(k0, a, eps_r)
        @test !any(isnan, c)

        # Symmetry: c_{-n} should satisfy specific relations
        # For symmetric incidence (phi_inc=0), c_{-n} = c_n
        for n in 1:min(N, 5)
            @test c[-n + N + 1] ≈ c[n + N + 1] atol=1e-12
        end
    end

    @testset "MoM vs Mie convergence" begin
        # Circular dielectric cylinder
        # Note: rectangular-grid VIE has non-monotonic convergence for curved
        # boundaries due to staircase approximation. We test overall accuracy.
        freq = 1e9; c0 = 3e8; lambda = c0 / freq; k0 = 2π / lambda
        a = 0.1 * lambda; eps_r = 4.0; chi_val = eps_r - 1.0

        r_obs = [Vec2(3a * cos(phi), 3a * sin(phi))
                 for phi in range(0, 2π, length=37)[1:36]]
        E_scat_mie = mie_scattered_field_2d(k0, a, eps_r, r_obs; phi_inc=0.0)

        errors = Float64[]
        for n in [10, 20, 40]
            mesh = Mesh2D((-a, a), (-a, a), n, n)
            chi = zeros(mesh.ncells)
            for i in 1:mesh.ncells
                r = sqrt(mesh.centers[i][1]^2 + mesh.centers[i][2]^2)
                if r <= a; chi[i] = chi_val; end
            end
            vr = solve_vie_2d(mesh, k0, chi, planewave_2d(mesh, k0, 0.0))
            E_scat_mom = scattered_field_2d(vr, r_obs)
            push!(errors, norm(E_scat_mom - E_scat_mie) / norm(E_scat_mie))
        end

        # Coarsest mesh should be reasonable (< 5%)
        @test errors[1] < 0.05

        # Finest mesh should be significantly better than coarsest
        @test errors[3] < errors[1]

        # Finest mesh should be within 1%
        @test errors[3] < 0.01
    end

    @testset "Jacobian accuracy" begin
        freq = 1e9; c0 = 3e8; lambda = c0 / freq; k0 = 2π / lambda
        a = 0.1 * lambda; eps_r = 2.5; chi_val = eps_r - 1.0

        mesh = Mesh2D((-a, a), (-a, a), 8, 8)
        chi = zeros(mesh.ncells)
        for i in 1:mesh.ncells
            r = sqrt(mesh.centers[i][1]^2 + mesh.centers[i][2]^2)
            if r <= a; chi[i] = chi_val; end
        end
        E_inc = planewave_2d(mesh, k0, 0.0)
        vr = solve_vie_2d(mesh, k0, chi, E_inc)

        r_obs = [Vec2(3a * cos(phi), 3a * sin(phi))
                 for phi in range(0, 2π, length=13)[1:12]]
        E_scat_ref = scattered_field_2d(vr, r_obs)
        J, _ = jacobian_scattered_field_2d(vr, r_obs)

        @test size(J) == (12, mesh.ncells)
        @test !any(isnan, J)
        @test !any(isinf, J)

        # Verify 5 random cells against finite differences
        delta = 1e-7
        cells_to_test = [findfirst(x -> x > 0, chi)]  # inside cell
        push!(cells_to_test, findfirst(x -> x == 0, chi))  # outside cell
        for _ in 1:3
            p = rand(1:mesh.ncells)
            p ∉ cells_to_test && push!(cells_to_test, p)
        end

        for p in cells_to_test
            chi_pert = copy(chi); chi_pert[p] += delta
            vr_pert = solve_vie_2d(mesh, k0, chi_pert, E_inc)
            E_scat_pert = scattered_field_2d(vr_pert, r_obs)
            J_fd = (E_scat_pert - E_scat_ref) / delta

            if norm(J[:, p]) > 1e-15
                rel_err = norm(J[:, p] - J_fd) / norm(J[:, p])
                @test rel_err < 1e-4  # FD accuracy limited by step size
            end
        end
    end

    @testset "Reciprocity check" begin
        # For a reciprocal medium: G(r1,r2) = G(r2,r1)
        # This means the D matrix should be symmetric
        k0 = 2π
        mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 6, 6)
        D = DiffMoM.assemble_D_matrix(mesh, k0)
        @test D ≈ transpose(D) atol=1e-13
    end

    @testset "Line source excitation" begin
        k0 = 2π
        mesh = Mesh2D((-0.5, 0.5), (-0.5, 0.5), 6, 6)
        r_src = Vec2(3.0, 0.0)

        E_inc = linesource_2d(mesh, k0, r_src)
        @test length(E_inc) == mesh.ncells
        @test !any(isnan, E_inc)
        @test !any(isinf, E_inc)

        # Amplitude should decrease with distance from source
        # Find nearest and farthest cells
        dists = [norm(mesh.centers[i] - r_src) for i in 1:mesh.ncells]
        @test abs(E_inc[argmin(dists)]) > abs(E_inc[argmax(dists)])
    end

end
