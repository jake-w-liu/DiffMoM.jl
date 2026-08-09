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
        @test_throws ArgumentError VoxelGrid3D(
            (0.0, Inf), (0.0, 1.0), (0.0, 1.0), 1, 1, 1)
        @test_throws ArgumentError VoxelGrid3D(
            Vec3[], Float64[], 0, 0, 1, 1,
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        @test_throws DimensionMismatch VoxelGrid3D(
            [Vec3(0.5, 0.5, 0.5)], [1.0], 2, 1, 1, 1,
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        @test_throws ArgumentError VoxelGrid3D(
            [Vec3(NaN, 0.5, 0.5)], [1.0], 1, 1, 1, 1,
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        @test_throws ArgumentError VoxelGrid3D(
            [Vec3(0.5, 0.5, 0.5)], [2.0], 1, 1, 1, 1,
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        @test_throws ArgumentError VoxelGrid3D(
            [Vec3(0.6, 0.5, 0.5)], [1.0], 1, 1, 1, 1,
            1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        large_origin = 1.0e16
        duplicate_center = Vec3(large_origin + 2.0, 0.5, 0.5)
        @test_throws ArgumentError VoxelGrid3D(
            [duplicate_center, duplicate_center], [4.0, 4.0],
            2, 2, 1, 1, 4.0, 1.0, 1.0, large_origin, 0.0, 0.0)
        @test_throws ArgumentError VoxelGrid3D(
            (1.0, nextfloat(1.0)), (0.0, 1.0), (0.0, 1.0),
            3, 1, 1)
        grid = VoxelGrid3D((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 2, 1, 1)
        @test_throws ArgumentError clausius_mossotti_polarizability(
            2.5, grid.volumes[1];
            k0=Inf,
            radiative_correction=true)
        @test_throws ArgumentError clausius_mossotti_polarizability(
            Inf, grid.volumes[1])
        large_alpha = clausius_mossotti_polarizability(2.0, 1.0e308)
        @test large_alpha == ComplexF64(7.5e307)
        large_tensor_alpha = clausius_mossotti_polarizability(
            Matrix{Float64}(2I, 3, 3), 1.0e308)
        @test all(isfinite, large_tensor_alpha)
        @test large_tensor_alpha == ComplexF64(7.5e307) *
                                      Matrix{ComplexF64}(I, 3, 3)
        @test_throws OverflowError clausius_mossotti_polarizability(10.0, 1.0e308)

        corrected_alpha = clausius_mossotti_polarizability(
            2.5, 1.0;
            k0=6.0e102,
            radiative_correction=true)
        @test isfinite(corrected_alpha)
        @test real(corrected_alpha) == 0.0
        @test imag(corrected_alpha) ≈ -8.726646259971649e-308 rtol=1e-14
        corrected_tensor_alpha = clausius_mossotti_polarizability(
            Matrix{Float64}(2.5I, 3, 3), 1.0;
            k0=6.0e102,
            radiative_correction=true)
        @test all(isfinite, corrected_tensor_alpha)
        @test corrected_tensor_alpha == corrected_alpha *
                                           Matrix{ComplexF64}(I, 3, 3)
        @test_throws OverflowError clausius_mossotti_polarizability(
            Matrix{Float64}(10I, 3, 3), 1.0e308)
        near_resonance_delta = 1.0e-10
        near_resonance_scalar = clausius_mossotti_polarizability(
            -2.0 + near_resonance_delta, 1.0)
        near_resonance_tensor = clausius_mossotti_polarizability(
            Matrix{Float64}(
                (-2.0 + near_resonance_delta) * I, 3, 3),
            1.0)
        @test near_resonance_tensor == near_resonance_scalar *
                                          Matrix{ComplexF64}(I, 3, 3)
        @test_throws ErrorException clausius_mossotti_polarizability(
            Matrix{Float64}(-2I, 3, 3), 1.0)
        @test_throws ArgumentError electric_dipole_dyadic_3d(
            grid.centers[1], grid.centers[2], Inf)
        @test_throws ArgumentError dda_operator_3d(grid, Inf, 2.5)
        @test_throws ArgumentError assemble_dda_3d(grid, Inf, 2.5)
        @test_throws ArgumentError dda_operator_3d(grid, k0, Inf)
        @test_throws ArgumentError dda_operator_3d(
            grid, k0, (2.0, NaN, 2.0))
        @test_throws ArgumentError planewave_dda_3d(
            grid, Vec3(0.0, 0.0, k0), NaN, Vec3(1.0, 0.0, 0.0))
        @test_throws ArgumentError planewave_dda_3d(
            grid, Vec3(0.0, 0.0, Inf), 1.0, Vec3(1.0, 0.0, 0.0))
        centered_grid = VoxelGrid3D(
            (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), 1, 1, 1)
        extreme_direction = Vec3(floatmax(Float64), floatmax(Float64), 0.0)
        transverse_pol = Vec3(inv(sqrt(2.0)), -inv(sqrt(2.0)), 0.0)
        extreme_E = planewave_dda_3d(
            centered_grid, extreme_direction, 1.0, transverse_pol)
        @test extreme_E == [CVec3(transverse_pol)]
        extreme_pol = CVec3(
            floatmax(Float64) + 0im,
            floatmax(Float64) + 0im,
            0.0 + 0im)
        @test planewave_dda_3d(
            centered_grid, Vec3(0.0, 0.0, 1.0), 1.0, extreme_pol) ==
              [extreme_pol]
        @test_throws OverflowError planewave_dda_3d(
            centered_grid, Vec3(0.0, 0.0, 1.0), 1.0e200,
            CVec3(1.0e200 + 0im, 0.0 + 0im, 0.0 + 0im))
        E_inc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k0), 1.0 + 0im, Vec3(1.0, 0.0, 0.0))
        res = solve_dda_3d(grid, k0, 1.0 + 0im, E_inc)

        @test norm(reduce(vcat, res.E_total) - reduce(vcat, E_inc)) < 1e-13
        @test norm(scattered_field_dda_3d(res, [Vec3(1.0, 0.0, 0.0)])[1]) < 1e-13
        @test iszero(farfield_dda_3d(res, extreme_direction))
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

    @testset "Exponent-safe subnormal-volume interactions" begin
        spacing = 1.0e-103
        grid = VoxelGrid3D(
            (0.0, 2spacing), (0.0, spacing), (0.0, spacing), 2, 1, 1)
        A, alpha, _ = assemble_dda_3d(grid, 1.0, 2.0)
        longitudinal_ref, transverse_ref = setprecision(BigFloat, 256) do
            R = abs(BigFloat(grid.centers[2][1]) -
                    BigFloat(grid.centers[1][1]))
            alpha_b = Complex{BigFloat}(alpha[2])
            expfac = exp(Complex{BigFloat}(0, -R)) /
                     (4 * BigFloat(pi))
            near = inv(R^3) + Complex{BigFloat}(0, 1) / R^2
            transverse = inv(R)
            return ComplexF64(-2 * alpha_b * expfac * near),
                   ComplexF64(-alpha_b * expfac * (transverse - near))
        end

        @test all(isfinite, A)
        @test A[1, 4] ≈ longitudinal_ref rtol=8eps(Float64)
        @test A[2, 5] ≈ transverse_ref rtol=8eps(Float64)
        @test A[3, 6] ≈ transverse_ref rtol=8eps(Float64)

        A_op = dda_operator_3d(grid, 1.0, 2.0)
        x = ComplexF64[1.0 + 0.2im, -0.4 + 0.3im, 0.7 - 0.1im,
                       -0.2 + 0.5im, 0.8 - 0.4im, 0.1 + 0.6im]
        @test all(isfinite, A_op * x)
        @test A_op * x ≈ A * x rtol=8eps(Float64)
        @test all(isfinite, adjoint(A_op) * x)
        @test adjoint(A_op) * x ≈ adjoint(A) * x rtol=8eps(Float64)

        E_inc = [CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im),
                 CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im)]
        result = solve_dda_3d(grid, 1.0, 2.0, E_inc)
        @test all(isfinite, reduce(vcat, result.E_total))
        @test all(isfinite, scattered_field_dda_3d(
            result, [Vec3(3spacing, spacing / 2, spacing / 2)])[1])
        @test all(isfinite, gradient_epsr_dda_3d(
            result, ones(ComplexF64, 3grid.nvoxels)))
    end

    @testset "Exponent-safe combined polarizability interactions" begin
        spacing = 4.0e102
        grid = VoxelGrid3D(
            (0.0, 2spacing), (0.0, spacing), (0.0, spacing), 2, 1, 1)
        k = 1.0e-103
        A, _, _ = assemble_dda_3d(grid, k, 2.0 + 0im)
        operator = dda_operator_3d(grid, k, 2.0 + 0im)

        fields = fill(10.0 + 0im, 3grid.nvoxels)
        dense_forward = A * fields
        matrixfree_forward = operator * fields
        @test all(isfinite, matrixfree_forward)
        @test matrixfree_forward ≈ dense_forward rtol=16eps(Float64)

        adjoint_rhs = ComplexF64[1, 2, 3, 4, 5, 6]
        dense_adjoint = adjoint(A) * adjoint_rhs
        matrixfree_adjoint = adjoint(operator) * adjoint_rhs
        @test all(isfinite, matrixfree_adjoint)
        @test matrixfree_adjoint ≈ dense_adjoint rtol=16eps(Float64)

        incident = fill(CVec3(10.0 + 0im, 10.0 + 0im, 10.0 + 0im), 2)
        result = solve_dda_3d(grid, k, 2.0 + 0im, incident)
        observation = Vec3(3spacing, spacing / 2, spacing / 2)
        scattered = scattered_field_dda_3d(result, [observation])[1]
        scattered_reference = setprecision(BigFloat, 256) do
            total = zeros(Complex{BigFloat}, 3)
            for j in 1:grid.nvoxels
                separation = [
                    BigFloat(observation[a]) - BigFloat(grid.centers[j][a])
                    for a in 1:3
                ]
                distance = sqrt(sum(abs2, separation))
                direction = separation / distance
                dipole = Complex{BigFloat}(result.alpha[j]) .*
                          Complex{BigFloat}.(result.E_total[j])
                radial = sum(direction[a] * dipole[a] for a in 1:3)
                transverse = dipole - radial * direction
                near = 3 * radial * direction - dipole
                kb = BigFloat(k)
                phase = exp(Complex{BigFloat}(0, -kb * distance)) /
                        (4 * BigFloat(pi))
                total .+= phase .* (
                    (kb^2 / distance) .* transverse +
                    (inv(distance^3) + Complex{BigFloat}(0, 1) * kb /
                     distance^2) .* near)
            end
            CVec3(ComplexF64(total[1]), ComplexF64(total[2]),
                  ComplexF64(total[3]))
        end
        @test all(isfinite, scattered)
        @test scattered ≈ scattered_reference rtol=16eps(Float64)

        direction = Vec3(1.0, 0.0, 0.0)
        projection = Matrix{Float64}(I, 3, 3) - direction * transpose(direction)
        prefactor = k^2 / (4π)
        farfield_reference = zero(CVec3)
        for j in 1:grid.nvoxels
            phase = exp(1im * k * dot(direction, grid.centers[j]))
            farfield_reference += phase * (
                projection * ((prefactor * result.alpha[j]) * result.E_total[j]))
        end
        farfield = farfield_dda_3d(result, direction)
        @test all(isfinite, farfield)
        @test farfield ≈ farfield_reference rtol=16eps(Float64)
    end

    @testset "Induced dipole exponent range" begin
        spacing = 4.0e102
        grid = VoxelGrid3D(
            (0.0, spacing), (0.0, spacing), (0.0, spacing), 1, 1, 1)
        E_inc = [CVec3(10.0 + 0im, 0.0 + 0im, 0.0 + 0im)]
        result = solve_dda_3d(grid, 1.0e-100, 2.0 + 0im, E_inc)
        @test all(isfinite, result.E_total[1])
        @test_throws OverflowError induced_dipoles_dda_3d(result)
    end

    @testset "Direct solve rejects non-finite output" begin
        grid = VoxelGrid3D(
            (0.0, 2.0), (0.0, 1.0), (0.0, 1.0), 2, 1, 1)
        probe = dda_operator_3d(grid, 1.0, 2.0 + 0im)
        coupling_per_alpha = probe[1, 4] / probe.alpha[1]
        target = 1.0
        near_singular_eps = nothing
        for _ in 1:32
            target = prevfloat(target)
            desired_alpha = target / coupling_per_alpha
            ratio = desired_alpha / (3 * grid.volumes[1])
            candidate_eps = (1 + 2ratio) / (1 - ratio)
            candidate = dda_operator_3d(grid, 1.0, candidate_eps)
            gap = abs(1 - candidate[1, 4])
            if 0 < gap < 1.0e-14
                near_singular_eps = candidate_eps
                break
            end
        end
        @test near_singular_eps !== nothing
        near_singular_eps === nothing && error(
            "failed to construct the near-singular DDA regression system")
        huge_incident = [
            CVec3(1.0e308 + 0im, 0.0 + 0im, 0.0 + 0im),
            CVec3(-1.0e308 + 0im, 0.0 + 0im, 0.0 + 0im),
        ]
        @test_throws ErrorException solve_dda_3d(
            grid, 1.0, near_singular_eps, huge_incident)
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

    @testset "Far-field prefactor exponent range" begin
        spacing = 1.0e-34
        grid = VoxelGrid3D(
            (-spacing / 2, spacing / 2),
            (-spacing / 2, spacing / 2),
            (-spacing / 2, spacing / 2),
            1, 1, 1)
        large_k = 1.0e200
        E_inc = [CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im)]
        res = solve_dda_3d(grid, large_k, 2.5, E_inc)
        field = farfield_dda_3d(res, Vec3(0.0, 1.0, 0.0))
        expected = setprecision(BigFloat, 256) do
            ComplexF64(
                BigFloat(large_k)^2 * Complex{BigFloat}(res.alpha[1]) /
                (4 * BigFloat(pi)))
        end

        @test all(isfinite, field)
        @test field[1] ≈ expected rtol=4eps(Float64)
        @test iszero(field[2])
        @test iszero(field[3])

        large_spacing = 1.0e100
        large_grid = VoxelGrid3D(
            (-large_spacing / 2, large_spacing / 2),
            (-large_spacing / 2, large_spacing / 2),
            (-large_spacing / 2, large_spacing / 2),
            1, 1, 1)
        small_k = 1.0e-200
        large_res = solve_dda_3d(large_grid, small_k, 2.5, E_inc)
        small_field = farfield_dda_3d(large_res, Vec3(0.0, 1.0, 0.0))
        expected_small = setprecision(BigFloat, 256) do
            ComplexF64(
                BigFloat(small_k)^2 * Complex{BigFloat}(large_res.alpha[1]) /
                (4 * BigFloat(pi)))
        end
        @test small_field[1] == expected_small
        @test !iszero(small_field[1])
        @test_throws OverflowError farfield_dda_3d(
            solve_dda_3d(
                VoxelGrid3D((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), 1, 1, 1),
                large_k, 2.5, E_inc),
            Vec3(0.0, 1.0, 0.0))
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
        @test size(A_op, 3) == 1
        @test_throws BoundsError size(A_op, 0)

        x = ComplexF64[sin(0.17 * i) + 1im * cos(0.11 * i) for i in 1:size(A_op, 2)]
        y = zeros(ComplexF64, size(A_op, 1))
        mul!(y, A_op, x)
        @test norm(y - A_dense * x) / norm(A_dense * x) < 1e-13
        fill!(y, ComplexF64(NaN, NaN))
        mul!(y, A_op, x, 1.0 + 0im, 0.0 + 0im)
        @test y ≈ A_dense * x rtol=1e-13

        A_adj = adjoint(A_op)
        @test size(A_adj, 3) == 1
        @test_throws BoundsError size(A_adj, -1)
        y_adj = zeros(ComplexF64, size(A_adj, 1))
        mul!(y_adj, A_adj, x)
        @test norm(y_adj - adjoint(A_dense) * x) / norm(adjoint(A_dense) * x) < 1e-13
        fill!(y_adj, ComplexF64(NaN, NaN))
        mul!(y_adj, A_adj, x, 1.0 + 0im, 0.0 + 0im)
        @test y_adj ≈ adjoint(A_dense) * x rtol=1e-13

        overlap_storage = vcat(x, 0.0 + 0im)
        overlap_x = view(overlap_storage, 1:length(x))
        overlap_y = view(overlap_storage, 2:(length(x) + 1))
        overlap_expected = A_dense * copy(overlap_x)
        mul!(overlap_y, A_op, overlap_x)
        @test overlap_y ≈ overlap_expected rtol=1e-13

        adjoint_overlap_storage = vcat(x, 0.0 + 0im)
        adjoint_overlap_x = view(adjoint_overlap_storage, 1:length(x))
        adjoint_overlap_y = view(adjoint_overlap_storage, 2:(length(x) + 1))
        adjoint_overlap_expected = adjoint(A_dense) * copy(adjoint_overlap_x)
        mul!(adjoint_overlap_y, A_adj, adjoint_overlap_x)
        @test adjoint_overlap_y ≈ adjoint_overlap_expected rtol=1e-13

        # The matrix-free operator stores O(N) material/geometric data instead
        # of the O(N^2) dense interaction matrix.
        @test Base.summarysize(A_op) < Base.summarysize(A_dense) / 20

        allocation_grid = VoxelGrid3D(
            (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), 512, 1, 1)
        allocation_op = dda_operator_3d(allocation_grid, k0, 2.5 + 0.1im)
        constructor_bytes = @allocated dda_operator_3d(
            allocation_grid, k0, 2.5 + 0.1im)
        stored_material_bytes =
            Base.summarysize(allocation_op.eps_r) +
            Base.summarysize(allocation_op.alpha)
        @test constructor_bytes <= stored_material_bytes + 4096

        allocation_epsv = fill(2.5 + 0.1im, allocation_grid.nvoxels)
        dda_operator_3d(allocation_grid, k0, allocation_epsv)
        vector_constructor_bytes = @allocated dda_operator_3d(
            allocation_grid, k0, allocation_epsv)
        @test vector_constructor_bytes <= stored_material_bytes + 4096

        mul!(y, A_op, x)  # warm-up before allocation probe
        @test (@allocated mul!(y, A_op, x)) < 1024
        mul!(y_adj, A_adj, x)
        @test (@allocated mul!(y_adj, A_adj, x)) < 1024

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

        @test_throws ErrorException solve_dda_3d(
            grid, k0, epsv, E_inc;
            solver=:gmres, tol=1e-14, maxiter=1, memory=1,
        )
        res_partial = solve_dda_3d(
            grid, k0, epsv, E_inc;
            solver=:gmres, tol=1e-14, maxiter=1, memory=1,
            check_gmres_convergence=false,
        )
        @test !res_partial.stats.solved
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
