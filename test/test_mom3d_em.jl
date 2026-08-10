# Test 48: Coupled electric-magnetic 3D DDA solver

using Test
using LinearAlgebra
using StaticArrays

if isdefined(Main, :DiffMoM)
    using .DiffMoM
else
    using DiffMoM
end

println("\n── Test 48: Coupled electric-magnetic 3D DDA solver ──")

@noinline function _allocate_em_field_outputs_3d(n::Int)
    return Vector{CVec3}(undef, n), Vector{CVec3}(undef, n)
end

@testset "Coupled electric-magnetic 3D DDA solver" begin
    k0 = 2π

    @testset "Free-space magnetodielectric limit" begin
        grid = VoxelGrid3D((-0.1, 0.1), (-0.05, 0.05), (-0.05, 0.05), 2, 1, 1)
        @test_throws ArgumentError em_dda_operator_3d(
            grid, Inf, 2.5, 1.2)
        @test_throws ArgumentError em_dda_operator_3d(
            grid, k0, 2.5, Inf)
        @test_throws ArgumentError em_dda_operator_3d(
            grid, k0, fill(2.5 + 0im, grid.nvoxels),
            ComplexF64[1.2 + 0im, NaN + 0im])
        @test_throws ArgumentError bianisotropic_clausius_mossotti_polarizability(
            Matrix{ComplexF64}(I, 6, 6), grid.volumes[1];
            eta0=Inf)
        large_bianisotropic_alpha =
            bianisotropic_clausius_mossotti_polarizability(
                Matrix{Float64}(2I, 6, 6), 1.0e308)
        @test all(isfinite, large_bianisotropic_alpha)
        @test large_bianisotropic_alpha == ComplexF64(7.5e307) *
                                            Matrix{ComplexF64}(I, 6, 6)
        corrected_bianisotropic_alpha =
            bianisotropic_clausius_mossotti_polarizability(
                Matrix{Float64}(2.5I, 6, 6), 1.0;
                k0=6.0e102,
                radiative_correction=true)
        @test all(isfinite, corrected_bianisotropic_alpha)
        @test corrected_bianisotropic_alpha ==
              ComplexF64(0.0, -8.726646259971649e-308) *
              Matrix{ComplexF64}(I, 6, 6)
        @test_throws OverflowError bianisotropic_clausius_mossotti_polarizability(
            Matrix{Float64}(10I, 6, 6), 1.0e308)
        near_resonance_delta = 1.0e-10
        near_resonance_scalar = clausius_mossotti_polarizability(
            -2.0 + near_resonance_delta, 1.0)
        near_resonance_coupled =
            bianisotropic_clausius_mossotti_polarizability(
                Matrix{Float64}(
                    (-2.0 + near_resonance_delta) * I, 6, 6),
                1.0)
        @test near_resonance_coupled == near_resonance_scalar *
                                           Matrix{ComplexF64}(I, 6, 6)
        @test_throws ErrorException bianisotropic_clausius_mossotti_polarizability(
            Matrix{Float64}(-2I, 6, 6), 1.0)
        @test_throws ArgumentError planewave_em_dda_3d(
            grid, Vec3(0.0, 0.0, k0), 1.0 + 0im,
            Vec3(1.0, 0.0, 0.0); eta0=Inf)
        centered_grid = VoxelGrid3D(
            (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), 1, 1, 1)
        extreme_direction = Vec3(floatmax(Float64), floatmax(Float64), 0.0)
        extreme_E, extreme_H = planewave_em_dda_3d(
            centered_grid, extreme_direction, 1.0, Vec3(0.0, 0.0, 1.0))
        @test extreme_E == [CVec3(0.0 + 0im, 0.0 + 0im, 1.0 + 0im)]
        expected_H = CVec3(inv(sqrt(2.0)), -inv(sqrt(2.0)), 0.0) /
                     376.730313668
        @test extreme_H[1] ≈ expected_H rtol=4eps(Float64)
        @test_throws OverflowError planewave_em_dda_3d(
            centered_grid, Vec3(0.0, 0.0, 1.0), 1.0,
            Vec3(1.0, 0.0, 0.0);
            eta0=nextfloat(0.0))
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
        @test all(iszero, farfield_em_dda_3d(res, extreme_direction))
    end

    @testset "Polarizability application exponent safety" begin
        large_spacing = 4.0e102
        large_grid = VoxelGrid3D(
            (0.0, large_spacing),
            (0.0, large_spacing),
            (0.0, large_spacing),
            1, 1, 1)
        large_E = [CVec3(10.0 + 0im, 0.0 + 0im, 0.0 + 0im)]
        zero_H = [zero(CVec3)]
        large_result = solve_em_dda_3d(
            large_grid, 1.0e-100, 2.0 + 0im, 1.0 + 0im,
            large_E, zero_H)
        @test_throws OverflowError induced_dipoles_em_dda_3d(large_result)

        cancel_alpha = zeros(ComplexF64, 6, 6)
        cancel_alpha[1, 1] = 1.0e308
        cancel_alpha[1, 2] = -1.0e308
        single = VoxelGrid3D(
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), 1, 1, 1)
        cancel_E = [CVec3(2.0 + 0im, 2.0 + 0im, 0.0 + 0im)]
        cancel_result = solve_em_dda_3d(
            single, 1.0, cancel_alpha, cancel_E, zero_H)
        q, m = induced_dipoles_em_dda_3d(cancel_result)
        @test q == [zero(CVec3)]
        @test m == [zero(CVec3)]

        pair = VoxelGrid3D(
            (0.0, 2.0), (0.0, 1.0), (0.0, 1.0), 2, 1, 1)
        operator = em_dda_operator_3d(pair, 1.0, cancel_alpha)
        fields = ComplexF64[
            2, 2, 0, 0, 0, 0,
            2, 2, 0, 0, 0, 0,
        ]
        @test operator * fields == fields

        interaction_grid = VoxelGrid3D(
            (0.0, 2large_spacing),
            (0.0, large_spacing),
            (0.0, large_spacing),
            2, 1, 1)
        large_alpha = zeros(ComplexF64, 6, 6)
        for component in 1:6
            large_alpha[component, component] = 4.8e307
        end
        large_operator = em_dda_operator_3d(
            interaction_grid, 1.0e-103, large_alpha)
        large_fields = fill(10.0 + 0im, 6interaction_grid.nvoxels)
        dense_large = Matrix(large_operator) * large_fields
        matrixfree_large = large_operator * large_fields
        @test all(isfinite, matrixfree_large)
        @test matrixfree_large ≈ dense_large rtol=16eps(Float64)

        complex_alpha = zeros(ComplexF64, 6, 6)
        complex_alpha[1, 1] = 1.3e308 + 1.3e308im
        complex_operator = em_dda_operator_3d(
            interaction_grid, 1.0e-103, complex_alpha)
        complex_dense = Matrix(complex_operator)
        complex_fields = zeros(ComplexF64, 6interaction_grid.nvoxels)
        complex_fields[1] = 1.0 + 0im
        complex_fields[7] = 1.0 + 0im
        @test all(isfinite, complex_dense)
        @test complex_operator * complex_fields ≈
              complex_dense * complex_fields rtol=16eps(Float64)

        post_fields = fill(
            CVec3(10.0 + 0im, 10.0 + 0im, 10.0 + 0im),
            interaction_grid.nvoxels)
        zero_fields = fill(zero(CVec3), interaction_grid.nvoxels)
        observation = [Vec3(
            3large_spacing, large_spacing / 2, large_spacing / 2)]
        direction = Vec3(1.0, 0.0, 0.0)
        dda_result = solve_dda_3d(
            interaction_grid, 1.0e-103, 2.0 + 0im, post_fields)
        dda_scattered = scattered_field_dda_3d(dda_result, observation)[1]
        dda_farfield = farfield_dda_3d(dda_result, direction)

        electric_result = solve_em_dda_3d(
            interaction_grid, 1.0e-103, 2.0 + 0im, 1.0 + 0im,
            post_fields, zero_fields)
        electric_scattered, magnetic_cross =
            scattered_fields_em_dda_3d(electric_result, observation)
        electric_farfield, magnetic_farfield_cross =
            farfield_em_dda_3d(electric_result, direction)
        @test electric_scattered[1] ≈ dda_scattered rtol=16eps(Float64)
        @test electric_farfield ≈ dda_farfield rtol=16eps(Float64)
        @test all(isfinite, magnetic_cross[1])
        @test all(isfinite, magnetic_farfield_cross)

        magnetic_result = solve_em_dda_3d(
            interaction_grid, 1.0e-103, 1.0 + 0im, 2.0 + 0im,
            zero_fields, post_fields)
        electric_cross, magnetic_scattered =
            scattered_fields_em_dda_3d(magnetic_result, observation)
        electric_farfield_cross, magnetic_farfield =
            farfield_em_dda_3d(magnetic_result, direction)
        @test magnetic_scattered[1] ≈ dda_scattered rtol=16eps(Float64)
        @test magnetic_farfield ≈ dda_farfield rtol=16eps(Float64)
        @test all(isfinite, electric_cross[1])
        @test all(isfinite, electric_farfield_cross)

        solve_grid = VoxelGrid3D(
            (0.0, 2.0), (0.0, 1.0), (0.0, 1.0), 2, 1, 1)
        unit_alpha = zeros(ComplexF64, 6, 6)
        unit_alpha[1, 1] = 1.0 + 0im
        unit_operator = em_dda_operator_3d(solve_grid, 1.0, unit_alpha)
        near_singular_alpha = zeros(ComplexF64, 6, 6)
        near_singular_alpha[1, 1] =
            prevfloat(1.0) / unit_operator[1, 7]
        near_singular_operator = em_dda_operator_3d(
            solve_grid, 1.0, near_singular_alpha)
        @test 0 < abs(1 - near_singular_operator[1, 7]) < 1.0e-14
        huge_E = [
            CVec3(1.0e308 + 0im, 0.0 + 0im, 0.0 + 0im),
            CVec3(-1.0e308 + 0im, 0.0 + 0im, 0.0 + 0im),
        ]
        huge_H = fill(zero(CVec3), solve_grid.nvoxels)
        @test_throws ErrorException solve_em_dda_3d(
            solve_grid, 1.0, near_singular_alpha, huge_E, huge_H)
    end

    @testset "Direct solve right-hand-side exponent range" begin
        grid = VoxelGrid3D(
            (0.0, 2.0), (0.0, 1.0), (0.0, 1.0), 2, 1, 1)
        k = 1.0
        probe = dda_operator_3d(
            grid, k, 2.0 + 0im; radiative_correction=false)
        coupling_per_alpha = probe[1, 4] / probe.alpha[1]
        desired_alpha = (-2.0 + 0im) / coupling_per_alpha
        alpha6 = zeros(ComplexF64, 6, 6)
        alpha6[1:3, 1:3] .=
            desired_alpha .* Matrix{ComplexF64}(I, 3, 3)
        operator = em_dda_operator_3d(grid, k, alpha6)
        A = Matrix(operator)
        @test A[1, 7] ≈ -2.0 + 0im rtol=2eps(Float64)

        amplitude = 0.8 * floatmax(Float64)
        incident_E = [
            CVec3(amplitude + 0im, 0im, 0im),
            CVec3(amplitude + 0im, 0im, 0im),
        ]
        incident_H = fill(zero(CVec3), grid.nvoxels)
        rhs = ComplexF64[]
        for j in 1:grid.nvoxels
            append!(rhs, incident_E[j])
            append!(rhs, incident_H[j])
        end
        reference = setprecision(BigFloat, 4096) do
            ComplexF64.(
                Matrix{Complex{BigFloat}}(A) \
                Complex{BigFloat}.(rhs))
        end
        @test all(isfinite, reference)

        result = solve_em_dda_3d(
            grid, k, alpha6, incident_E, incident_H)
        solution = ComplexF64[]
        for j in 1:grid.nvoxels
            append!(solution, result.E_total[j])
            append!(solution, result.H_total[j])
        end
        @test all(isfinite, solution)
        @test all(
            isapprox(real(solution[index]), real(reference[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(imag(solution[index]), imag(reference[index]);
                     rtol=16eps(Float64), atol=0.0)
            for index in eachindex(reference))
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
        farfield_em_dda_3d(res, Vec3(0.0, 1.0, 0.0))
        @test @allocated(
            farfield_em_dda_3d(res, Vec3(0.0, 1.0, 0.0))) <= 128
        observations = [Vec3(1.0, 0.0, 0.0)]
        scattered_fields_em_dda_3d(res, observations)
        _allocate_em_field_outputs_3d(length(observations))
        output_allocation = @allocated _allocate_em_field_outputs_3d(
            length(observations))
        scattered_allocation = @allocated scattered_fields_em_dda_3d(
            res, observations)
        @test scattered_allocation <= output_allocation + 128
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
        H_inc = [CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)]
        res = solve_em_dda_3d(
            grid, large_k, 2.5, 1.0, E_inc, H_inc)
        FE, FH = farfield_em_dda_3d(res, Vec3(0.0, 1.0, 0.0))
        expected_E = setprecision(BigFloat, 256) do
            ComplexF64(
                BigFloat(large_k)^2 * Complex{BigFloat}(res.alpha[1][1, 1]) /
                (4 * BigFloat(pi)))
        end

        @test all(isfinite, FE)
        @test all(isfinite, FH)
        @test FE[1] ≈ expected_E rtol=4eps(Float64)
        @test iszero(FE[2])
        @test iszero(FE[3])
        @test iszero(FH[1])
        @test iszero(FH[2])
        @test FH[3] ≈ -expected_E / 376.730313668 rtol=4eps(Float64)
        overflow_grid = VoxelGrid3D(
            (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), 1, 1, 1)
        overflow_res = solve_em_dda_3d(
            overflow_grid, large_k, 2.5, 1.0, E_inc, H_inc)
        @test_throws OverflowError farfield_em_dda_3d(
            overflow_res, Vec3(0.0, 1.0, 0.0))
    end

    @testset "Far-field accumulation exponent range" begin
        grid = VoxelGrid3D(
            (0.0, 32.0), (0.0, 1.0), (0.0, 1.0), 4, 1, 1)
        k = 1.0
        epsr = 1.0e16 + 0im
        mur = 1.0 + 0im
        operator = em_dda_operator_3d(
            grid, k, epsr, mur; radiative_correction=false)
        direction = Vec3(0.0, 1.0, 0.0)
        source_coefficient =
            abs(k^2 * operator.alpha[1][1, 1] / (4π))
        target_amplitude =
            (0.75 * floatmax(Float64)) / source_coefficient
        target_E = [
            CVec3(sign * target_amplitude + 0im, 0im, 0im)
            for sign in (1.0, 1.0, -1.0, -1.0)
        ]
        target_H = fill(zero(CVec3), grid.nvoxels)
        target = ComplexF64[]
        for j in 1:grid.nvoxels
            append!(target, target_E[j])
            append!(target, target_H[j])
        end
        rhs = operator * target
        @test all(isfinite, rhs)
        E_inc = [
            CVec3(rhs[6j - 5], rhs[6j - 4], rhs[6j - 3])
            for j in 1:grid.nvoxels
        ]
        H_inc = [
            CVec3(rhs[6j - 2], rhs[6j - 1], rhs[6j])
            for j in 1:grid.nvoxels
        ]
        result = solve_em_dda_3d(
            grid, k, epsr, mur, E_inc, H_inc;
            radiative_correction=false)

        electric_terms, magnetic_terms, electric_reference,
        magnetic_reference = setprecision(BigFloat, 4096) do
            n = BigFloat.(direction)
            eta = BigFloat(376.730313668)
            prefactor = BigFloat(k)^2 / (4 * BigFloat(pi))
            total_E = zeros(Complex{BigFloat}, 3)
            total_H = zeros(Complex{BigFloat}, 3)
            terms_E = CVec3[]
            terms_H = CVec3[]
            for j in 1:grid.nvoxels
                field = Complex{BigFloat}[
                    result.E_total[j][1], result.E_total[j][2],
                    result.E_total[j][3], result.H_total[j][1],
                    result.H_total[j][2], result.H_total[j][3],
                ]
                dipoles = Complex{BigFloat}.(result.alpha[j]) * field
                q = dipoles[1:3]
                m = dipoles[4:6]
                projected_q = q - n * sum(
                    n[index] * q[index] for index in 1:3)
                projected_m = m - n * sum(
                    n[index] * m[index] for index in 1:3)
                phase_argument = BigFloat(k) * sum(
                    n[index] * BigFloat(grid.centers[j][index])
                    for index in 1:3)
                phase_prefactor = exp(Complex{BigFloat}(
                    zero(BigFloat), phase_argument)) * prefactor
                contribution_E = phase_prefactor .* (
                    projected_q - eta .* cross(n, m))
                contribution_H = phase_prefactor .* (
                    cross(n, q) ./ eta + projected_m)
                push!(terms_E, CVec3(
                    ComplexF64(contribution_E[1]),
                    ComplexF64(contribution_E[2]),
                    ComplexF64(contribution_E[3])))
                push!(terms_H, CVec3(
                    ComplexF64(contribution_H[1]),
                    ComplexF64(contribution_H[2]),
                    ComplexF64(contribution_H[3])))
                total_E .+= contribution_E
                total_H .+= contribution_H
            end
            terms_E,
            terms_H,
            CVec3(ComplexF64(total_E[1]), ComplexF64(total_E[2]),
                  ComplexF64(total_E[3])),
            CVec3(ComplexF64(total_H[1]), ComplexF64(total_H[2]),
                  ComplexF64(total_H[3]))
        end
        @test all(all(isfinite, term) for term in electric_terms)
        @test all(all(isfinite, term) for term in magnetic_terms)
        @test !isfinite(electric_terms[1][1] + electric_terms[2][1])

        field_E, field_H = farfield_em_dda_3d(result, direction)
        @test all(isfinite, field_E)
        @test all(isfinite, field_H)
        @test all(
            isapprox(real(field_E[index]), real(electric_reference[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(imag(field_E[index]), imag(electric_reference[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(real(field_H[index]), real(magnetic_reference[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(imag(field_H[index]), imag(magnetic_reference[index]);
                     rtol=16eps(Float64), atol=0.0)
            for index in 1:3)
    end

    @testset "Scattered-field accumulation exponent range" begin
        grid = VoxelGrid3D(
            (2.0, 18.0), (0.0, 1.0), (0.0, 1.0), 4, 1, 1)
        observation = Vec3(0.0, 0.5, 0.5)
        k = 1.0
        alpha6 = zeros(ComplexF64, 6, 6)
        alpha6[1:3, 1:3] .=
            1.0e4 .* Matrix{ComplexF64}(I, 3, 3)
        operator = em_dda_operator_3d(grid, k, alpha6)
        target_magnitude = 0.75 * floatmax(Float64)
        signs = (1.0, 1.0, -1.0, -1.0)
        fields_E = CVec3[]
        for j in 1:grid.nvoxels
            dyadic = electric_dipole_dyadic_3d(
                observation, grid.centers[j], k)
            push!(fields_E, CVec3(
                0im,
                signs[j] * target_magnitude /
                (alpha6[2, 2] * dyadic[2, 2]),
                0im,
            ))
        end
        fields_H = fill(zero(CVec3), grid.nvoxels)
        @test all(all(isfinite, field) for field in fields_E)
        result = EMDDAResult3D(
            fields_E,
            fields_H,
            copy(fields_E),
            copy(fields_H),
            operator.alpha,
            operator,
            nothing,
            :direct,
            nothing,
            grid,
            k,
            false,
        )

        terms_E, terms_H, reference_E, reference_H =
            setprecision(BigFloat, 4096) do
                eta = BigFloat(376.730313668)
                total_E = zeros(Complex{BigFloat}, 3)
                total_H = zeros(Complex{BigFloat}, 3)
                contributions_E = CVec3[]
                contributions_H = CVec3[]
                for j in 1:grid.nvoxels
                    field = Complex{BigFloat}[
                        fields_E[j][1], fields_E[j][2], fields_E[j][3],
                        fields_H[j][1], fields_H[j][2], fields_H[j][3],
                    ]
                    dipoles = Complex{BigFloat}.(operator.alpha[j]) * field
                    q = dipoles[1:3]
                    m = dipoles[4:6]
                    separation = [
                        BigFloat(observation[index]) -
                        BigFloat(grid.centers[j][index])
                        for index in 1:3
                    ]
                    distance = sqrt(sum(abs2, separation))
                    direction = separation / distance
                    radial_q = sum(
                        direction[index] * q[index] for index in 1:3)
                    radial_m = sum(
                        direction[index] * m[index] for index in 1:3)
                    transverse_q = q - radial_q * direction
                    transverse_m = m - radial_m * direction
                    near_q = 3 * radial_q * direction - q
                    near_m = 3 * radial_m * direction - m
                    kb = BigFloat(k)
                    scalar_green = exp(Complex{BigFloat}(
                        zero(BigFloat), -kb * distance)) /
                        (4 * BigFloat(pi) * distance)
                    radial_derivative =
                        (Complex{BigFloat}(0, -kb) - inv(distance)) *
                        scalar_green
                    electric_q = scalar_green .* (
                        kb^2 .* transverse_q +
                        (inv(distance^2) +
                         Complex{BigFloat}(0, 1) * kb / distance) .* near_q)
                    electric_m = scalar_green .* (
                        kb^2 .* transverse_m +
                        (inv(distance^2) +
                         Complex{BigFloat}(0, 1) * kb / distance) .* near_m)
                    contribution_E = electric_q +
                        Complex{BigFloat}(0, -eta * kb) *
                        radial_derivative .* cross(direction, m)
                    contribution_H =
                        Complex{BigFloat}(0, kb / eta) *
                        radial_derivative .* cross(direction, q) +
                        electric_m
                    push!(contributions_E, CVec3(
                        ComplexF64(contribution_E[1]),
                        ComplexF64(contribution_E[2]),
                        ComplexF64(contribution_E[3])))
                    push!(contributions_H, CVec3(
                        ComplexF64(contribution_H[1]),
                        ComplexF64(contribution_H[2]),
                        ComplexF64(contribution_H[3])))
                    total_E .+= contribution_E
                    total_H .+= contribution_H
                end
                contributions_E,
                contributions_H,
                CVec3(ComplexF64(total_E[1]), ComplexF64(total_E[2]),
                      ComplexF64(total_E[3])),
                CVec3(ComplexF64(total_H[1]), ComplexF64(total_H[2]),
                      ComplexF64(total_H[3]))
            end
        @test all(all(isfinite, term) for term in terms_E)
        @test all(all(isfinite, term) for term in terms_H)
        @test !isfinite(terms_E[1][2] + terms_E[2][2])
        @test !iszero(reference_H[3])

        scattered_E, scattered_H =
            scattered_fields_em_dda_3d(result, [observation])
        @test all(isfinite, scattered_E[1])
        @test all(isfinite, scattered_H[1])
        @test all(
            isapprox(real(scattered_E[1][index]), real(reference_E[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(imag(scattered_E[1][index]), imag(reference_E[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(real(scattered_H[1][index]), real(reference_H[index]);
                     rtol=16eps(Float64), atol=0.0) &&
            isapprox(imag(scattered_H[1][index]), imag(reference_H[index]);
                     rtol=16eps(Float64), atol=0.0)
            for index in 1:3)
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

        allocation_grid = VoxelGrid3D(
            (0.0, 256.0), (0.0, 1.0), (0.0, 1.0), 256, 1, 1)
        material_vector = fill(coupled, allocation_grid.nvoxels)
        alpha_vector = em_dda_polarizabilities(
            allocation_grid, k0, material_vector)
        allocated = @allocated em_dda_polarizabilities(
            allocation_grid, k0, material_vector)
        @test all(==(alpha_vector[1]), alpha_vector)
        @test allocated <= Base.summarysize(alpha_vector) + 4096
    end

    @testset "Matrix-free operator equivalence and storage" begin
        grid = VoxelGrid3D((-0.15, 0.15), (-0.1, 0.1), (-0.05, 0.05), 3, 3, 2)
        epsv = fill(2.3 + 0.03im, grid.nvoxels)
        muv = fill(1.4 + 0.02im, grid.nvoxels)

        A_dense, alpha = assemble_em_dda_3d(grid, k0, epsv, muv)
        A_op = em_dda_operator_3d(grid, k0, epsv, muv)
        @test A_op.alpha == alpha
        @test size(A_op, 3) == 1
        @test_throws BoundsError size(A_op, 0)

        x = ComplexF64[sin(0.13 * i) + 1im * cos(0.17 * i) for i in 1:size(A_op, 2)]
        y = zeros(ComplexF64, size(A_op, 1))
        mul!(y, A_op, x)
        @test norm(y - A_dense * x) / norm(A_dense * x) < 1e-13
        fill!(y, ComplexF64(NaN, NaN))
        mul!(y, A_op, x, 1.0 + 0im, 0.0 + 0im)
        @test y ≈ A_dense * x rtol=1e-13
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

        overlap_storage = vcat(xb, 0.0 + 0im)
        overlap_x = view(overlap_storage, 1:length(xb))
        overlap_y = view(overlap_storage, 2:(length(xb) + 1))
        overlap_expected = A_dense_b * copy(overlap_x)
        mul!(overlap_y, A_op_b, overlap_x)
        @test overlap_y ≈ overlap_expected rtol=1e-13
    end

    @testset "Matrix-free scaled-output exponent range" begin
        scale_grid = VoxelGrid3D(
            (0.0, 0.2), (0.0, 0.1), (0.0, 0.1), 2, 1, 1)
        operator = em_dda_operator_3d(
            scale_grid, k0, 2.5 + 0.1im, 1.3 + 0.02im)
        input = ComplexF64[
            10 * (sin(0.17 * i) + 1im * cos(0.11 * i))
            for i in 1:size(operator, 2)
        ]
        product = operator * input
        previous = -product
        scale = 1.0e308 + 0im
        @test all(isfinite, product)
        @test any(!isfinite, scale .* product)
        reference = setprecision(BigFloat, 4608) do
            ComplexF64[
                Complex{BigFloat}(scale) * Complex{BigFloat}(product[i]) +
                Complex{BigFloat}(scale) * Complex{BigFloat}(previous[i])
                for i in eachindex(product)
            ]
        end
        @test all(isfinite, reference)
        result = copy(previous)
        mul!(result, operator, input, scale, scale)
        @test result == reference
    end

    @testset "Matrix-free exponent-range cancellation" begin
        component_scale = value ->
            max(abs(real(value)), abs(imag(value)))
        interaction_block = function (grid, observation, source, k)
            block = Matrix{ComplexF64}(undef, 6, 6)
            for component in 1:6
                basis_dipoles = DiffMoM._CVec6DDA(ntuple(index ->
                    index == component ? 1.0 + 0im : 0.0 + 0im, 6))
                q, m = DiffMoM._split_em_field(basis_dipoles)
                E, H = DiffMoM._em_interaction_apply_3d(
                    grid.centers[observation],
                    grid.centers[source],
                    k,
                    q,
                    m,
                )
                block[:, component] = vcat(E, H)
            end
            block
        end

        grid = VoxelGrid3D(
            (0.0, 3.0), (0.0, 3.0), (0.0, 1.0), 3, 3, 1)
        n = grid.nvoxels
        k = 1.0
        target = 5
        sources = (2, 4, 6, 8)
        signs = (1.0, 1.0, -1.0, -1.0)
        target_field = ComplexF64[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        source_dipoles = fill(zero(DiffMoM._CVec6DDA), n)
        for (source, sign) in zip(sources, signs)
            source_dipoles[source] = DiffMoM._CVec6DDA(
                interaction_block(grid, target, source, k) \
                (sign * target_field))
        end

        normalized_fields = fill(zero(DiffMoM._CVec6DDA), n)
        scale_denominator = 0.0
        for observation in 1:n
            field = zero(DiffMoM._CVec6DDA)
            for source in 1:n
                observation == source && continue
                q, m = DiffMoM._split_em_field(source_dipoles[source])
                E, H = DiffMoM._em_interaction_apply_3d(
                    grid.centers[observation],
                    grid.centers[source],
                    k,
                    q,
                    m,
                )
                contribution = DiffMoM._join_em_field(E, H)
                field += contribution
                scale_denominator = max(
                    scale_denominator,
                    maximum(component_scale, contribution),
                )
            end
            normalized_fields[observation] = field
            scale_denominator = max(
                scale_denominator,
                maximum(component_scale, field),
            )
        end
        scale = 0.60 * floatmax(Float64) / scale_denominator
        fields = DiffMoM._CVec6DDA[
            scale * field for field in normalized_fields]
        alpha = Vector{DiffMoM._CMat6DDA}(undef, n)
        for voxel in 1:n
            field_norm = sum(abs2, normalized_fields[voxel])
            alpha[voxel] = iszero(field_norm) ?
                zero(DiffMoM._CMat6DDA) :
                DiffMoM._CMat6DDA(
                    source_dipoles[voxel] *
                    adjoint(normalized_fields[voxel]) /
                    field_norm)
        end
        operator = em_dda_operator_3d(grid, k, alpha)
        x = reduce(vcat, fields)

        ordinary = similar(x)
        maximum_contribution = 0.0
        for observation in 1:n
            field = fields[observation]
            for source in 1:n
                observation == source && continue
                E, H = DiffMoM._em_alpha_interaction_apply_3d(
                    grid.centers[observation],
                    grid.centers[source],
                    k,
                    operator.alpha[source],
                    fields[source],
                )
                contribution = DiffMoM._join_em_field(E, H)
                @test all(isfinite, contribution)
                maximum_contribution = max(
                    maximum_contribution,
                    maximum(component_scale, contribution),
                )
                field -= contribution
            end
            ordinary[(6observation - 5):(6observation)] = field
        end
        overflow_indices = findall(!isfinite, ordinary)
        @test !isempty(overflow_indices)
        @test maximum(component_scale, x) < floatmax(Float64)
        @test maximum_contribution < floatmax(Float64)

        reference = setprecision(BigFloat, 4096) do
            dipoles_big = [DiffMoM._alpha_apply_bigfloat_vector_3d(
                operator.alpha[source], fields[source])
                for source in 1:n]
            result = Vector{ComplexF64}(undef, 6n)
            for observation in 1:n
                total = SVector{6,Complex{BigFloat}}(ntuple(
                    component -> Complex{BigFloat}(
                        fields[observation][component]), 6))
                for source in 1:n
                    observation == source && continue
                    source_dipoles = dipoles_big[source]
                    q = SVector{3,Complex{BigFloat}}(
                        source_dipoles[1],
                        source_dipoles[2],
                        source_dipoles[3],
                    )
                    m = SVector{3,Complex{BigFloat}}(
                        source_dipoles[4],
                        source_dipoles[5],
                        source_dipoles[6],
                    )
                    E, H = DiffMoM._em_interaction_value_bigfloat_3d(
                        grid.centers[observation],
                        grid.centers[source],
                        k,
                        q,
                        m,
                    )
                    total -= SVector{6,Complex{BigFloat}}(
                        E[1], E[2], E[3], H[1], H[2], H[3])
                end
                for component in 1:6
                    result[6(observation - 1) + component] =
                        ComplexF64(total[component])
                end
            end
            result
        end
        @test all(isfinite, reference)

        y = similar(x)
        mul!(y, operator, x)
        @test all(isfinite, y)
        @test y[overflow_indices] == reference[overflow_indices]
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

        @test_throws ErrorException solve_em_dda_3d(
            grid, k0, 2.3 + 0.02im, 1.5 + 0.01im, E_inc, H_inc;
            solver=:gmres, tol=1e-14, maxiter=1, memory=1,
        )
        res_partial = solve_em_dda_3d(
            grid, k0, 2.3 + 0.02im, 1.5 + 0.01im, E_inc, H_inc;
            solver=:gmres, tol=1e-14, maxiter=1, memory=1,
            check_gmres_convergence=false,
        )
        @test !res_partial.stats.solved
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
