# Test 46: 3D vector material DDA solver

using Test
using LinearAlgebra
using StaticArrays

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
        @test_throws ArgumentError VoxelGrid3D(
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            2, 2, 2; max_voxels=7)
        @test_throws ArgumentError VoxelGrid3D(
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            2, 2, 2; max_raw_bytes=8 * (sizeof(Vec3) + sizeof(Float64)) - 1)
        @test VoxelGrid3D(
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            2, 2, 2;
            max_voxels=8,
            max_raw_bytes=8 * (sizeof(Vec3) + sizeof(Float64))).nvoxels == 8
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
        planewave_dda_3d(
            centered_grid, Vec3(0.0, 0.0, 1.0), 1.0,
            Vec3(1.0, 0.0, 0.0))
        planewave_alloc = @allocated planewave_dda_3d(
            centered_grid, Vec3(0.0, 0.0, 1.0), 1.0,
            Vec3(1.0, 0.0, 0.0))
        Vector{CVec3}(undef, centered_grid.nvoxels)
        planewave_output_alloc = @allocated Vector{CVec3}(
            undef, centered_grid.nvoxels)
        @test planewave_alloc <= planewave_output_alloc + 128

        phase_x = 1e200
        phase_overflow_grid = VoxelGrid3D(
            (phase_x, nextfloat(phase_x)),
            (0.0, 1.0), (0.0, 1.0), 1, 1, 1)
        phase_kvec = Vec3(1e200, 0.0, 0.0)
        phase_pol = Vec3(0.0, 1.0, 0.0)
        phase_reference = setprecision(BigFloat, 256) do
            argument = BigFloat(phase_kvec[1]) *
                       BigFloat(phase_overflow_grid.centers[1][1])
            ComplexF64(exp(
                Complex{BigFloat}(zero(BigFloat), -argument)))
        end
        phase_E = planewave_dda_3d(
            phase_overflow_grid, phase_kvec, 1.0, phase_pol)
        @test phase_E == [CVec3(0.0 + 0im, phase_reference, 0.0 + 0im)]
        phase_E_em, phase_H_em = planewave_em_dda_3d(
            phase_overflow_grid, phase_kvec, 1.0, phase_pol)
        @test phase_E_em == phase_E
        @test phase_H_em == [CVec3(
            0.0 + 0im, 0.0 + 0im,
            phase_reference / 376.730313668)]

        # A finite Float64 dot product can still discard low product bits that
        # determine the oscillatory phase.  The public DDA and EM-DDA plane
        # waves must use the exact supplied binary inputs in that regime.
        phase_cancel_angle = 0.2
        phase_cancel_kvec = Vec3(
            cos(phase_cancel_angle), sin(phase_cancel_angle), 0.0)
        phase_cancel_x = 1.0e100
        phase_cancel_y = -Float64(
            (phase_cancel_kvec[1] / phase_cancel_kvec[2]) *
            phase_cancel_x)
        phase_cancel_center = Vec3(
            phase_cancel_x, phase_cancel_y, 0.0)
        phase_cancel_grid = VoxelGrid3D(
            [phase_cancel_center], [1.0], 1, 1, 1, 1,
            1.0, 1.0, 1.0,
            phase_cancel_x, phase_cancel_y, -0.5)
        phase_cancel_reference = setprecision(BigFloat, 4352) do
            argument = sum(
                BigFloat(phase_cancel_kvec[index]) *
                BigFloat(phase_cancel_center[index]) for index in 1:3)
            ComplexF64(exp(Complex{BigFloat}(0, -argument)))
        end
        phase_cancel_pol = Vec3(0.0, 0.0, 1.0)
        phase_cancel_E = planewave_dda_3d(
            phase_cancel_grid, phase_cancel_kvec, 1.0, phase_cancel_pol)
        @test phase_cancel_E == [CVec3(
            0.0 + 0im, 0.0 + 0im, phase_cancel_reference)]
        phase_cancel_E_em, _ = planewave_em_dda_3d(
            phase_cancel_grid, phase_cancel_kvec, 1.0, phase_cancel_pol)
        @test phase_cancel_E_em == phase_cancel_E

        # The same phase product appears in DDA far-field translation.  Use a
        # one-voxel stored result with a z-directed dipole so the projection is
        # exact and independently check the full public result.
        phase_cancel_result = DDAResult3D(
            CVec3[CVec3(0.0 + 0im, 0.0 + 0im, 1.0 + 0im)],
            CVec3[zero(CVec3)],
            ComplexF64[2.0], ComplexF64[1.0],
            Matrix{ComplexF64}(I, 3, 3), nothing, :direct, nothing,
            phase_cancel_grid, 1.0, false)
        phase_cancel_rhat = DiffMoM._normalized_real_direction_dda_3d(
            phase_cancel_kvec, "phase-cancellation reference direction")
        phase_cancel_farfield_reference = setprecision(BigFloat, 4352) do
            argument = sum(
                BigFloat(phase_cancel_rhat[index]) *
                BigFloat(phase_cancel_center[index]) for index in 1:3)
            value = ComplexF64(
                exp(Complex{BigFloat}(0, argument)) /
                (4 * BigFloat(pi)))
            CVec3(0.0 + 0im, 0.0 + 0im, value)
        end
        @test farfield_dda_3d(
            phase_cancel_result, phase_cancel_kvec) ==
              phase_cancel_farfield_reference
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
        matrix_bytes = sizeof(ComplexF64) * (3grid.nvoxels)^2
        @test_throws ArgumentError assemble_dda_3d(
            grid, k0, 2.5 + 0im;
            max_output_bytes=matrix_bytes - 1)
        A, alpha, epsv = assemble_dda_3d(
            grid, k0, 2.5 + 0im;
            max_output_bytes=matrix_bytes)
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

    @testset "Far-field accumulation exponent range" begin
        # Construct a solved four-voxel system whose first two individually
        # representable far-field terms overflow when added, while all four
        # cancel to a finite result.
        grid = VoxelGrid3D(
            (0.0, 32.0), (0.0, 1.0), (0.0, 1.0), 4, 1, 1)
        k = 1.0
        epsr = 1.0e16 + 0im
        A, alpha, _ = assemble_dda_3d(
            grid, k, epsr; radiative_correction=false)
        direction = Vec3(0.0, 1.0, 0.0)
        source_coefficient = abs(k^2 * alpha[1] / (4π))
        target_amplitude =
            (0.75 * floatmax(Float64)) / source_coefficient
        target = [
            CVec3(sign * target_amplitude + 0im, 0im, 0im)
            for sign in (1.0, 1.0, -1.0, -1.0)
        ]
        rhs = A * reduce(vcat, target)
        @test all(isfinite, rhs)
        incident = [
            CVec3(rhs[3j - 2], rhs[3j - 1], rhs[3j])
            for j in 1:grid.nvoxels
        ]
        result = solve_dda_3d(
            grid, k, epsr, incident; radiative_correction=false)

        terms, reference = setprecision(BigFloat, 4096) do
            n = BigFloat.(direction)
            prefactor = BigFloat(k)^2 / (4 * BigFloat(pi))
            total = zeros(Complex{BigFloat}, 3)
            contributions = CVec3[]
            for j in 1:grid.nvoxels
                dipole = Complex{BigFloat}(result.alpha[j]) .*
                         Complex{BigFloat}.(result.E_total[j])
                projected = dipole - n * sum(
                    n[index] * dipole[index] for index in 1:3)
                phase_argument = BigFloat(k) * sum(
                    n[index] * BigFloat(grid.centers[j][index])
                    for index in 1:3)
                contribution = exp(Complex{BigFloat}(
                    zero(BigFloat), phase_argument)) * prefactor * projected
                push!(contributions, CVec3(
                    ComplexF64(contribution[1]),
                    ComplexF64(contribution[2]),
                    ComplexF64(contribution[3])))
                total .+= contribution
            end
            contributions,
            CVec3(ComplexF64(total[1]), ComplexF64(total[2]),
                  ComplexF64(total[3]))
        end
        @test all(all(isfinite, term) for term in terms)
        @test !isfinite(terms[1][1] + terms[2][1])

        field = farfield_dda_3d(result, direction)
        @test all(isfinite, field)
        @test all(
            isapprox(real(field[index]), real(reference[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(imag(field[index]), imag(reference[index]);
                     rtol=16eps(Float64), atol=0.0)
            for index in 1:3)

        # Exercise cancellation across almost the entire Float64 exponent
        # range. The finite result is around 10^-94 even though each source
        # term is around 10^308, so a short BigFloat mantissa is insufficient.
        dy = ldexp(1.0, -158)
        dx = 32dy
        precision_grid = VoxelGrid3D(
            (0.0, 2dx), (0.0, 2dy), (0.0, 1.0e308), 2, 2, 1)
        precision_k = ldexp(1.0, -509)
        precision_direction = Vec3(inv(sqrt(2.0)), inv(sqrt(2.0)), 0.0)
        base_field = ldexp(1.0, 1023)
        weights = (1.0, 1.0, -17 / 16, -15 / 16)
        precision_fields = [
            CVec3(weights[j] * base_field + 0im, 0im, 0im)
            for j in 1:precision_grid.nvoxels
        ]
        precision_alpha = fill(1.0e308 + 0im, precision_grid.nvoxels)
        precision_result = DDAResult3D(
            precision_fields,
            copy(precision_fields),
            fill(-3.0 + 0im, precision_grid.nvoxels),
            precision_alpha,
            zeros(ComplexF64, 3precision_grid.nvoxels,
                  3precision_grid.nvoxels),
            nothing,
            :direct,
            nothing,
            precision_grid,
            precision_k,
            false,
        )

        precision_terms, precision_reference =
            setprecision(BigFloat, 4096) do
                n = BigFloat.(precision_direction)
                prefactor = BigFloat(precision_k)^2 /
                            (4 * BigFloat(pi))
                total = zeros(Complex{BigFloat}, 3)
                contributions = CVec3[]
                for j in 1:precision_grid.nvoxels
                    dipole = Complex{BigFloat}(precision_alpha[j]) .*
                             Complex{BigFloat}.(precision_fields[j])
                    projected = dipole - n * sum(
                        n[index] * dipole[index] for index in 1:3)
                    phase_argument = BigFloat(precision_k) * sum(
                        n[index] *
                        BigFloat(precision_grid.centers[j][index])
                        for index in 1:3)
                    contribution = exp(Complex{BigFloat}(
                        zero(BigFloat), phase_argument)) *
                        prefactor * projected
                    push!(contributions, CVec3(
                        ComplexF64(contribution[1]),
                        ComplexF64(contribution[2]),
                        ComplexF64(contribution[3])))
                    total .+= contribution
                end
                contributions,
                CVec3(ComplexF64(total[1]), ComplexF64(total[2]),
                      ComplexF64(total[3]))
            end
        @test all(all(isfinite, term) for term in precision_terms)
        @test !isfinite(
            precision_terms[1][1] + precision_terms[2][1])
        @test 1.0e-100 < abs(real(precision_reference[1])) < 1.0e-90

        precision_field =
            farfield_dda_3d(precision_result, precision_direction)
        @test all(
            isapprox(
                real(precision_field[index]),
                real(precision_reference[index]);
                rtol=16eps(Float64), atol=0.0) &&
            isapprox(
                imag(precision_field[index]),
                imag(precision_reference[index]);
                rtol=16eps(Float64), atol=0.0)
            for index in 1:3)
    end

    @testset "Scattered-field accumulation exponent range" begin
        grid = VoxelGrid3D(
            (2.0, 18.0), (0.0, 1.0), (0.0, 1.0), 4, 1, 1)
        observation = Vec3(0.0, 0.5, 0.5)
        k = 1.0
        alpha = fill(1.0e4 + 0im, grid.nvoxels)
        target_magnitude = 0.75 * floatmax(Float64)
        signs = (1.0, 1.0, -1.0, -1.0)
        fields = CVec3[]
        for j in 1:grid.nvoxels
            dyadic = electric_dipole_dyadic_3d(
                observation, grid.centers[j], k)
            push!(fields, CVec3(
                signs[j] * target_magnitude /
                (alpha[j] * dyadic[1, 1]),
                0im,
                0im,
            ))
        end
        @test all(all(isfinite, field) for field in fields)
        result = DDAResult3D(
            fields,
            copy(fields),
            fill(1.0 + 0im, grid.nvoxels),
            alpha,
            zeros(ComplexF64, 3grid.nvoxels, 3grid.nvoxels),
            nothing,
            :direct,
            nothing,
            grid,
            k,
            false,
        )

        terms, reference = setprecision(BigFloat, 4096) do
            total = zeros(Complex{BigFloat}, 3)
            contributions = CVec3[]
            for j in 1:grid.nvoxels
                separation = [
                    BigFloat(observation[index]) -
                    BigFloat(grid.centers[j][index])
                    for index in 1:3
                ]
                distance = sqrt(sum(abs2, separation))
                direction = separation / distance
                dipole = Complex{BigFloat}(alpha[j]) .*
                         Complex{BigFloat}.(fields[j])
                radial = sum(
                    direction[index] * dipole[index]
                    for index in 1:3)
                transverse = dipole - radial * direction
                near = 3 * radial * direction - dipole
                phase = exp(Complex{BigFloat}(
                    zero(BigFloat), -BigFloat(k) * distance)) /
                    (4 * BigFloat(pi))
                contribution = phase .* (
                    (BigFloat(k)^2 / distance) .* transverse +
                    (inv(distance^3) + Complex{BigFloat}(0, 1) *
                     BigFloat(k) / distance^2) .* near)
                push!(contributions, CVec3(
                    ComplexF64(contribution[1]),
                    ComplexF64(contribution[2]),
                    ComplexF64(contribution[3])))
                total .+= contribution
            end
            contributions,
            CVec3(ComplexF64(total[1]), ComplexF64(total[2]),
                  ComplexF64(total[3]))
        end
        @test all(all(isfinite, term) for term in terms)
        @test !isfinite(terms[1][1] + terms[2][1])

        scattered = scattered_field_dda_3d(result, [observation])[1]
        @test all(isfinite, scattered)
        @test all(
            isapprox(real(scattered[index]), real(reference[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(imag(scattered[index]), imag(reference[index]);
                     rtol=16eps(Float64), atol=0.0)
            for index in 1:3)
    end

    @testset "Direct solve right-hand-side exponent range" begin
        grid = VoxelGrid3D(
            (0.0, 2.0), (0.0, 1.0), (0.0, 1.0), 2, 1, 1)
        k = 1.0
        probe = dda_operator_3d(
            grid, k, 2.0 + 0im; radiative_correction=false)
        coupling_per_alpha = probe[1, 4] / probe.alpha[1]
        desired_alpha = (-2.0 + 0im) / coupling_per_alpha
        ratio = desired_alpha / (3 * grid.volumes[1])
        epsr = (1 + 2ratio) / (1 - ratio)
        A, _, _ = assemble_dda_3d(
            grid, k, epsr; radiative_correction=false)
        @test A[1, 4] ≈ -2.0 + 0im rtol=2eps(Float64)

        amplitude = 0.8 * floatmax(Float64)
        incident = [
            CVec3(amplitude + 0im, 0im, 0im),
            CVec3(amplitude + 0im, 0im, 0im),
        ]
        rhs = reduce(vcat, incident)
        reference = setprecision(BigFloat, 4096) do
            ComplexF64.(
                Matrix{Complex{BigFloat}}(A) \
                Complex{BigFloat}.(rhs))
        end
        @test all(isfinite, reference)

        result = solve_dda_3d(
            grid, k, epsr, incident; radiative_correction=false)
        solution = reduce(vcat, result.E_total)
        @test all(isfinite, solution)
        @test all(
            isapprox(real(solution[index]), real(reference[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(imag(solution[index]), imag(reference[index]);
                     rtol=16eps(Float64), atol=0.0)
            for index in eachindex(reference))
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
        @test_throws OverflowError solve_dda_3d(
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
        farfield_dda_3d(res, n)
        @test @allocated(farfield_dda_3d(res, n)) <= 128
        observations = [Vec3(1.0, 0.0, 0.0)]
        scattered_field_dda_3d(res, observations)
        output_allocation =
            @allocated Vector{CVec3}(undef, length(observations))
        scattered_allocation =
            @allocated scattered_field_dda_3d(res, observations)
        @test scattered_allocation <= output_allocation + 128
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

    @testset "Matrix-free scaled-output exponent range" begin
        scale_grid = VoxelGrid3D(
            (0.0, 0.2), (0.0, 0.1), (0.0, 0.1), 2, 1, 1)
        scale_operator = dda_operator_3d(
            scale_grid, k0, 2.5 + 0.1im)
        scale_input = ComplexF64[
            10 * (sin(0.17 * i) + 1im * cos(0.11 * i))
            for i in 1:size(scale_operator, 2)
        ]
        scale = 1.0e308 + 0im

        for operator in (scale_operator, adjoint(scale_operator))
            product = operator * scale_input
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
            mul!(result, operator, scale_input, scale, scale)
            @test result == reference
        end
    end

    @testset "Matrix-free exponent-range cancellation" begin
        component_scale = value ->
            max(abs(real(value)), abs(imag(value)))
        dipole_block = function (grid, observation, source, k)
            block = Matrix{ComplexF64}(undef, 3, 3)
            for component in 1:3
                basis = CVec3(ntuple(index ->
                    index == component ? 1.0 + 0im : 0.0 + 0im, 3))
                block[:, component] = DiffMoM._electric_dipole_apply_3d(
                    grid.centers[observation],
                    grid.centers[source],
                    k,
                    basis,
                )
            end
            block
        end
        adjoint_dipole_block = function (grid, observation, source, k)
            block = Matrix{ComplexF64}(undef, 3, 3)
            for component in 1:3
                basis = CVec3(ntuple(index ->
                    index == component ? 1.0 + 0im : 0.0 + 0im, 3))
                block[:, component] =
                    DiffMoM._electric_dipole_alpha_adjoint_apply_3d(
                        grid.centers[observation],
                        grid.centers[source],
                        k,
                        1.0 + 0im,
                        basis,
                    )
            end
            block
        end

        forward_grid = VoxelGrid3D(
            (0.0, 4.0), (0.0, 3.0), (0.0, 1.0), 4, 3, 1)
        forward_n = forward_grid.nvoxels
        forward_k = 1.0
        forward_target = 5
        forward_sources = (3, 7, 8, 11)
        cancellation_signs = (1.0, 1.0, -1.0, -1.0)
        forward_dipoles = fill(zero(CVec3), forward_n)
        target_value = ComplexF64[0.0, 0.0, 1.0]
        for (source, sign) in zip(
                forward_sources, cancellation_signs)
            forward_dipoles[source] = CVec3(
                dipole_block(
                    forward_grid, forward_target, source, forward_k) \
                (sign * target_value))
        end
        forward_fields_normalized = fill(zero(CVec3), forward_n)
        forward_scale_denominator = 0.0
        for observation in 1:forward_n
            field = zero(CVec3)
            for source in 1:forward_n
                observation == source && continue
                contribution = DiffMoM._electric_dipole_apply_3d(
                    forward_grid.centers[observation],
                    forward_grid.centers[source],
                    forward_k,
                    forward_dipoles[source],
                )
                field += contribution
                forward_scale_denominator = max(
                    forward_scale_denominator,
                    maximum(component_scale, contribution),
                )
            end
            forward_fields_normalized[observation] = field
            forward_scale_denominator = max(
                forward_scale_denominator,
                maximum(component_scale, field),
            )
        end
        forward_scale =
            0.60 * floatmax(Float64) / forward_scale_denominator
        forward_fields = CVec3[
            forward_scale * field
            for field in forward_fields_normalized
        ]
        forward_alpha = Vector{DiffMoM._CMat3DDA}(undef, forward_n)
        for voxel in 1:forward_n
            field_norm = sum(abs2, forward_fields_normalized[voxel])
            forward_alpha[voxel] = iszero(field_norm) ?
                zero(DiffMoM._CMat3DDA) :
                DiffMoM._CMat3DDA(
                    forward_dipoles[voxel] *
                    adjoint(forward_fields_normalized[voxel]) /
                    field_norm)
        end
        forward_operator = DDAOperator3D(
            forward_grid,
            forward_k,
            fill(1.0 + 0im, forward_n),
            forward_alpha,
            false,
        )
        forward_x = reduce(vcat, forward_fields)
        ordinary_forward = similar(forward_x)
        maximum_forward_contribution = 0.0
        for observation in 1:forward_n
            field = forward_fields[observation]
            for source in 1:forward_n
                observation == source && continue
                contribution =
                    DiffMoM._electric_dipole_alpha_apply_3d(
                        forward_grid.centers[observation],
                        forward_grid.centers[source],
                        forward_k,
                        forward_alpha[source],
                        forward_fields[source],
                    )
                @test all(isfinite, contribution)
                maximum_forward_contribution = max(
                    maximum_forward_contribution,
                    maximum(component_scale, contribution),
                )
                field -= contribution
            end
            ordinary_forward[(3observation - 2):(3observation)] = field
        end
        forward_overflow_indices = findall(!isfinite, ordinary_forward)
        @test !isempty(forward_overflow_indices)
        @test maximum(component_scale, forward_x) < floatmax(Float64)
        @test maximum_forward_contribution < floatmax(Float64)

        forward_reference = setprecision(BigFloat, 4096) do
            dipoles_big = Vector{SVector{3,Complex{BigFloat}}}(
                undef, forward_n)
            for source in 1:forward_n
                field_big = SVector{3,Complex{BigFloat}}(ntuple(
                    component -> Complex{BigFloat}(
                        forward_fields[source][component]), 3))
                dipoles_big[source] =
                    SVector{3,Complex{BigFloat}}(ntuple(row -> begin
                        total = zero(Complex{BigFloat})
                        for column in 1:3
                            total += Complex{BigFloat}(
                                forward_alpha[source][row, column]) *
                                     field_big[column]
                        end
                        total
                    end, 3))
            end
            reference = Vector{ComplexF64}(undef, 3forward_n)
            for observation in 1:forward_n
                total = SVector{3,Complex{BigFloat}}(ntuple(
                    component -> Complex{BigFloat}(
                        forward_fields[observation][component]), 3))
                for source in 1:forward_n
                    observation == source && continue
                    total -= DiffMoM._electric_dipole_value_bigfloat_3d(
                        forward_grid.centers[observation],
                        forward_grid.centers[source],
                        forward_k,
                        dipoles_big[source],
                    )
                end
                for component in 1:3
                    reference[3(observation - 1) + component] =
                        ComplexF64(total[component])
                end
            end
            reference
        end
        @test all(isfinite, forward_reference)
        forward_y = similar(forward_x)
        mul!(forward_y, forward_operator, forward_x)
        @test all(isfinite, forward_y)
        @test forward_y[forward_overflow_indices] ==
              forward_reference[forward_overflow_indices]

        adjoint_grid = VoxelGrid3D(
            (0.0, 3.0), (0.0, 2.0), (0.0, 1.0), 3, 2, 1)
        adjoint_n = adjoint_grid.nvoxels
        adjoint_k = 5.0
        adjoint_target = 5
        adjoint_sources = (1, 3, 4, 6)
        adjoint_fields_normalized = fill(zero(CVec3), adjoint_n)
        for (source, sign) in zip(
                adjoint_sources, cancellation_signs)
            adjoint_fields_normalized[source] = CVec3(
                adjoint_dipole_block(
                    adjoint_grid, adjoint_target, source, adjoint_k) \
                (sign * target_value))
        end
        raw_adjoint_interaction = function (observation, source, field)
            DiffMoM._electric_dipole_alpha_adjoint_apply_3d(
                adjoint_grid.centers[observation],
                adjoint_grid.centers[source],
                adjoint_k,
                1.0 + 0im,
                field,
            )
        end
        target_field = zero(CVec3)
        for source in 1:adjoint_n
            source == adjoint_target && continue
            target_field += raw_adjoint_interaction(
                adjoint_target,
                source,
                adjoint_fields_normalized[source],
            )
        end
        adjoint_fields_normalized[adjoint_target] = target_field
        raw_adjoint_sums = fill(zero(CVec3), adjoint_n)
        for observation in 1:adjoint_n
            for source in 1:adjoint_n
                observation == source && continue
                raw_adjoint_sums[observation] += raw_adjoint_interaction(
                    observation,
                    source,
                    adjoint_fields_normalized[source],
                )
            end
        end
        adjoint_alpha = fill(zero(DiffMoM._CMat3DDA), adjoint_n)
        adjoint_alpha[adjoint_target] = DiffMoM._CI3_DDA
        for voxel in adjoint_sources
            interaction_norm = sum(abs2, raw_adjoint_sums[voxel])
            adjoint_alpha[voxel] = DiffMoM._CMat3DDA(
                raw_adjoint_sums[voxel] *
                adjoint(adjoint_fields_normalized[voxel]) /
                interaction_norm)
        end
        adjoint_scale_denominator = 0.0
        for observation in 1:adjoint_n
            adjoint_scale_denominator = max(
                adjoint_scale_denominator,
                maximum(
                    component_scale,
                    adjoint_fields_normalized[observation],
                ),
            )
            for source in 1:adjoint_n
                observation == source && continue
                contribution =
                    DiffMoM._electric_dipole_alpha_adjoint_apply_3d(
                        adjoint_grid.centers[observation],
                        adjoint_grid.centers[source],
                        adjoint_k,
                        adjoint_alpha[observation],
                        adjoint_fields_normalized[source],
                    )
                adjoint_scale_denominator = max(
                    adjoint_scale_denominator,
                    maximum(component_scale, contribution),
                )
            end
        end
        adjoint_scale =
            0.60 * floatmax(Float64) / adjoint_scale_denominator
        adjoint_fields = CVec3[
            adjoint_scale * field
            for field in adjoint_fields_normalized
        ]
        adjoint_operator = DDAOperator3D(
            adjoint_grid,
            adjoint_k,
            fill(1.0 + 0im, adjoint_n),
            adjoint_alpha,
            false,
        )
        adjoint_x = reduce(vcat, adjoint_fields)
        ordinary_adjoint = similar(adjoint_x)
        maximum_adjoint_contribution = 0.0
        for observation in 1:adjoint_n
            field = adjoint_fields[observation]
            accumulated = zero(CVec3)
            for source in 1:adjoint_n
                observation == source && continue
                contribution =
                    DiffMoM._electric_dipole_alpha_adjoint_apply_3d(
                        adjoint_grid.centers[observation],
                        adjoint_grid.centers[source],
                        adjoint_k,
                        adjoint_alpha[observation],
                        adjoint_fields[source],
                    )
                @test all(isfinite, contribution)
                maximum_adjoint_contribution = max(
                    maximum_adjoint_contribution,
                    maximum(component_scale, contribution),
                )
                accumulated += contribution
            end
            ordinary_adjoint[
                (3observation - 2):(3observation)] = field - accumulated
        end
        adjoint_overflow_indices = findall(!isfinite, ordinary_adjoint)
        @test !isempty(adjoint_overflow_indices)
        @test maximum(component_scale, adjoint_x) < floatmax(Float64)
        @test maximum_adjoint_contribution < floatmax(Float64)

        adjoint_reference = setprecision(BigFloat, 4096) do
            fields_big = [SVector{3,Complex{BigFloat}}(ntuple(
                component -> Complex{BigFloat}(
                    adjoint_fields[voxel][component]), 3))
                for voxel in 1:adjoint_n]
            reference = Vector{ComplexF64}(undef, 3adjoint_n)
            for observation in 1:adjoint_n
                total = fields_big[observation]
                for source in 1:adjoint_n
                    observation == source && continue
                    conjugated_field =
                        SVector{3,Complex{BigFloat}}(ntuple(
                            component -> conj(
                                fields_big[source][component]), 3))
                    conjugated_interaction =
                        DiffMoM._electric_dipole_value_bigfloat_3d(
                            adjoint_grid.centers[observation],
                            adjoint_grid.centers[source],
                            adjoint_k,
                            conjugated_field,
                        )
                    adjoint_interaction =
                        SVector{3,Complex{BigFloat}}(ntuple(
                            component -> conj(
                                conjugated_interaction[component]), 3))
                    total -=
                        DiffMoM._alpha_adjoint_apply_bigfloat_vector_3d(
                            adjoint_alpha[observation],
                            adjoint_interaction,
                        )
                end
                for component in 1:3
                    reference[3(observation - 1) + component] =
                        ComplexF64(total[component])
                end
            end
            reference
        end
        @test all(isfinite, adjoint_reference)
        adjoint_y = similar(adjoint_x)
        mul!(adjoint_y, adjoint(adjoint_operator), adjoint_x)
        @test all(isfinite, adjoint_y)
        @test adjoint_y[adjoint_overflow_indices] ==
              adjoint_reference[adjoint_overflow_indices]
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
