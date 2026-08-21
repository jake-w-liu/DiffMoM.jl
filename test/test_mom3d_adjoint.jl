# Test: 3D DDA material adjoint sensitivities

using Test
using LinearAlgebra

if !isdefined(Main, :DiffMoM)
    using DiffMoM
end

if !isdefined(DiffMoM, :solve_dda_adjoint_3d)
    Base.include(DiffMoM, joinpath(@__DIR__, "..", "src", "mom3d", "Adjoint3D.jl"))
end

println("\n-- Test: 3D DDA material adjoint sensitivities --")

@testset "3D DDA material adjoint sensitivities" begin
    k0 = 2π
    grid = VoxelGrid3D((-0.11, 0.11), (-0.06, 0.06), (-0.05, 0.05), 2, 1, 1)
    epsr = ComplexF64[2.2 + 0im, 2.7 + 0im]
    E_inc = planewave_dda_3d(grid, Vec3(0.0, 0.0, k0), 1.0 + 0.2im,
                             Vec3(1.0, 0.3, 0.0))

    weights = ComplexF64[0.7, 1.1, 0.9, 0.4, 1.3, 0.8]

    function objective(epsv)
        res = solve_dda_3d(grid, k0, epsv, E_inc)
        E = reduce(vcat, res.E_total)
        return real(dot(E, weights .* E))
    end

    res = solve_dda_3d(grid, k0, epsr, E_inc)
    E = reduce(vcat, res.E_total)
    @test_throws ArgumentError solve_dda_adjoint_3d(
        res, ComplexF64[]; solver=:unsupported)
    malformed_result = DDAResult3D(
        CVec3[], res.E_inc, res.eps_r, res.alpha,
        res.A, res.A_LU, res.solver, res.stats,
        res.grid, res.k0, res.radiative_correction)
    @test_throws DimensionMismatch solve_dda_adjoint_3d(
        malformed_result, weights .* E)
    @test_throws DimensionMismatch gradient_epsr_dda_3d(
        malformed_result, weights .* E)
    @test_throws ArgumentError solve_dda_adjoint_3d(
        res, fill(ComplexF64(NaN, 0.0), length(E)))
    nonfinite_system = copy(res.A)
    nonfinite_system[1, 1] = ComplexF64(NaN, 0.0)
    nonfinite_system_result = DDAResult3D(
        res.E_total, res.E_inc, res.eps_r, res.alpha,
        nonfinite_system, res.A_LU, res.solver, res.stats,
        res.grid, res.k0, res.radiative_correction)
    nonfinite_system_error = try
        solve_dda_adjoint_3d(nonfinite_system_result, weights .* E)
        nothing
    catch err
        err
    end
    @test nonfinite_system_error isa ArgumentError
    @test occursin(
        "DDAResult3D A must contain only finite values",
        sprint(showerror, nonfinite_system_error))
    tensor_grid = VoxelGrid3D(
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), 1, 1, 1)
    tensor_epsr = [2.0 .* Matrix{ComplexF64}(I, 3, 3)]
    tensor_incident = [CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im)]
    tensor_result = solve_dda_3d(
        tensor_grid, 1.0, tensor_epsr, tensor_incident;
        radiative_correction=false)
    tensor_gradient_error = try
        gradient_epsr_dda_3d(tensor_result, ones(ComplexF64, 3))
        nothing
    catch err
        err
    end
    @test tensor_gradient_error isa ArgumentError
    @test occursin(
        "scalar-permittivity", sprint(showerror, tensor_gradient_error))
    lambda = solve_dda_adjoint_3d(res, weights .* E)
    grad = gradient_epsr_dda_3d(res, lambda)

    tiny_scale = nextfloat(0.0)
    tiny_rhs = floatmin(Float64)
    tiny_grid = VoxelGrid3D(
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), 1, 1, 1)
    tiny_matrix = Matrix(Diagonal(fill(ComplexF64(tiny_scale), 3)))
    tiny_result = DDAResult3D(
        [CVec3(0im, 0im, 0im)],
        [CVec3(0im, 0im, 0im)],
        ComplexF64[1.0],
        ComplexF64[1.0],
        tiny_matrix,
        lu(tiny_matrix),
        :direct,
        nothing,
        tiny_grid,
        1.0,
        false,
    )
    tiny_reference = setprecision(BigFloat, 4096) do
        fill(ComplexF64(BigFloat(tiny_rhs) / BigFloat(tiny_scale)), 3)
    end
    @test tiny_reference == fill(ComplexF64(2.0^52), 3)
    @test solve_dda_adjoint_3d(
        tiny_result, fill(ComplexF64(tiny_rhs), 3)) == tiny_reference

    h = 1e-6
    grad_fd = similar(grad)
    for j in eachindex(epsr)
        eps_plus = copy(epsr)
        eps_minus = copy(epsr)
        eps_plus[j] += h
        eps_minus[j] -= h
        grad_fd[j] = (objective(eps_plus) - objective(eps_minus)) / (2h)
    end

    @test isapprox(grad, grad_fd; rtol=2e-5, atol=2e-8)
    @test all(isreal, grad)

    lambda_gmres = solve_dda_adjoint_3d(res, weights .* E;
                                        solver=:gmres, tol=1e-12, maxiter=50)
    @test norm(lambda_gmres - lambda) / norm(lambda) < 1e-6
    matrixfree_result = solve_dda_3d(
        grid, k0, epsr, E_inc;
        solver=:gmres, tol=1e-12, maxiter=50)
    @test matrixfree_result.A_LU === nothing
    @test !(matrixfree_result.A isa Matrix)
    @test_throws ArgumentError solve_dda_adjoint_3d(
        matrixfree_result, weights .* reduce(vcat, matrixfree_result.E_total))

    @test_throws ErrorException solve_dda_adjoint_3d(
        res, weights .* E;
        solver=:gmres, tol=1e-14, maxiter=1, memory=1,
    )
    lambda_partial = solve_dda_adjoint_3d(
        res, weights .* E;
        solver=:gmres, tol=1e-14, maxiter=1, memory=1,
        check_gmres_convergence=false,
    )
    @test length(lambda_partial) == length(lambda)

    spacing = 4.0e102
    extreme_grid = VoxelGrid3D(
        (0.0, 2spacing), (0.0, spacing), (0.0, spacing), 2, 1, 1)
    extreme_k = 1.0e-103
    extreme_incident = fill(
        CVec3(1.0 + 0im, 0.5 + 0im, -0.25 + 0im), 2)
    extreme_result = solve_dda_3d(
        extreme_grid, extreme_k, 2.0 + 0im, extreme_incident)
    extreme_lambda = ones(ComplexF64, 6)
    extreme_gradient = gradient_epsr_dda_3d(
        extreme_result, extreme_lambda)
    extreme_reference = setprecision(BigFloat, 256) do
        distance = abs(
            BigFloat(extreme_grid.centers[2][1]) -
            BigFloat(extreme_grid.centers[1][1]))
        kb = BigFloat(extreme_k)
        phase = exp(Complex{BigFloat}(0, -kb * distance)) /
                (4 * BigFloat(pi))
        map(1:2) do j
            derivative = 9 * BigFloat(extreme_grid.volumes[j]) /
                         (Complex{BigFloat}(extreme_result.eps_r[j]) + 2)^2
            dipole = [
                derivative * Complex{BigFloat}(extreme_result.E_total[j][a])
                for a in 1:3
            ]
            transverse = Complex{BigFloat}[0, dipole[2], dipole[3]]
            near = Complex{BigFloat}[2dipole[1], -dipole[2], -dipole[3]]
            interaction = phase .* (
                (kb^2 / distance) .* transverse +
                (inv(distance^3) + Complex{BigFloat}(0, 1) * kb /
                 distance^2) .* near)
            Float64(2 * real(sum(interaction)))
        end
    end
    @test all(isfinite, extreme_gradient)
    @test extreme_gradient ≈ extreme_reference rtol=16eps(Float64)

    cancellation_grid = VoxelGrid3D(
        (0.0, 5.0), (0.0, 1.0), (0.0, 1.0), 5, 1, 1)
    cancellation_fields = [
        voxel == 1 ? CVec3(128.0 + 0im, 0im, 0im) :
                     CVec3(0im, 0im, 0im)
        for voxel in 1:cancellation_grid.nvoxels
    ]
    cancellation_matrix = Matrix{ComplexF64}(
        I, 3cancellation_grid.nvoxels, 3cancellation_grid.nvoxels)
    cancellation_result = DDAResult3D(
        cancellation_fields,
        copy(cancellation_fields),
        fill(1.0 + 0im, cancellation_grid.nvoxels),
        fill(1.0 + 0im, cancellation_grid.nvoxels),
        cancellation_matrix,
        lu(cancellation_matrix),
        :direct,
        nothing,
        cancellation_grid,
        1.0,
        false,
    )
    cancellation_derivative = DiffMoM._dalpha_depsr_clausius_mossotti(
        1.0 + 0im, cancellation_grid.volumes[1])
    cancellation_target = 0.75 * floatmax(Float64)
    cancellation_signs = (1.0, 1.0, -1.0, -1.0)
    cancellation_lambda = zeros(
        ComplexF64, 3cancellation_grid.nvoxels)
    cancellation_terms = ComplexF64[]
    for (offset, observation_voxel) in
        enumerate(2:cancellation_grid.nvoxels)
        interaction = DiffMoM._electric_dipole_alpha_apply_3d(
            cancellation_grid.centers[observation_voxel],
            cancellation_grid.centers[1],
            1.0,
            cancellation_derivative,
            cancellation_fields[1],
        )
        lambda_component = conj(
            cancellation_signs[offset] * cancellation_target /
            interaction[1])
        cancellation_lambda[3(observation_voxel - 1) + 1] =
            lambda_component
        push!(cancellation_terms, dot(
            CVec3(lambda_component, 0im, 0im), interaction))
    end
    @test all(isfinite, cancellation_terms)
    @test !isfinite(foldl(+, cancellation_terms))
    cancellation_reference = setprecision(BigFloat, 4096) do
        total = zero(BigFloat)
        for (offset, observation_voxel) in
            enumerate(2:cancellation_grid.nvoxels)
            interaction = DiffMoM._electric_dipole_alpha_apply_3d(
                cancellation_grid.centers[observation_voxel],
                cancellation_grid.centers[1],
                1.0,
                cancellation_derivative,
                cancellation_fields[1],
            )
            lambda_component =
                cancellation_lambda[3(observation_voxel - 1) + 1]
            total +=
                BigFloat(real(lambda_component)) *
                BigFloat(real(interaction[1])) +
                BigFloat(imag(lambda_component)) *
                BigFloat(imag(interaction[1]))
        end
        Float64(2 * total)
    end
    cancellation_gradient = gradient_epsr_dda_3d(
        cancellation_result, cancellation_lambda)
    @test cancellation_reference == 3.995537081919199e292
    @test cancellation_gradient[1] == cancellation_reference
    @test cancellation_gradient[2:end] == zeros(4)

    # A finite ordinary accumulation can still lose the small remainder
    # between much larger terms.  The gradient must use the exact stored
    # Float64 products when that cancellation exceeds its rounding bound.
    finite_cancellation_targets = (1.0e16, 1.0, -1.0e16, 0.0)
    finite_cancellation_lambda = zeros(
        ComplexF64, 3cancellation_grid.nvoxels)
    finite_cancellation_terms = Float64[]
    for (offset, observation_voxel) in
        enumerate(2:cancellation_grid.nvoxels)
        interaction = DiffMoM._electric_dipole_alpha_apply_3d(
            cancellation_grid.centers[observation_voxel],
            cancellation_grid.centers[1],
            1.0,
            cancellation_derivative,
            cancellation_fields[1],
        )
        lambda_component = conj(
            finite_cancellation_targets[offset] / interaction[1])
        finite_cancellation_lambda[3(observation_voxel - 1) + 1] =
            lambda_component
        push!(finite_cancellation_terms,
              real(dot(
                  CVec3(lambda_component, 0im, 0im), interaction)))
    end
    finite_cancellation_reference = setprecision(BigFloat, 4096) do
        total = zero(BigFloat)
        for observation_voxel in 2:cancellation_grid.nvoxels
            interaction = DiffMoM._electric_dipole_alpha_apply_3d(
                cancellation_grid.centers[observation_voxel],
                cancellation_grid.centers[1],
                1.0,
                cancellation_derivative,
                cancellation_fields[1],
            )
            lambda_component =
                finite_cancellation_lambda[3(observation_voxel - 1) + 1]
            total +=
                BigFloat(real(lambda_component)) *
                BigFloat(real(interaction[1])) +
                BigFloat(imag(lambda_component)) *
                BigFloat(imag(interaction[1]))
        end
        Float64(2 * total)
    end
    @test 2sum(finite_cancellation_terms) !=
          finite_cancellation_reference
    finite_cancellation_gradient = gradient_epsr_dda_3d(
        cancellation_result, finite_cancellation_lambda)
    @test finite_cancellation_gradient[1] ==
          finite_cancellation_reference
    @test finite_cancellation_gradient[2:end] == zeros(4)

    solve_grid = VoxelGrid3D(
        (0.0, 2.0), (0.0, 1.0), (0.0, 1.0), 2, 1, 1)
    probe = dda_operator_3d(solve_grid, 1.0, 2.0 + 0im)
    coupling_per_alpha = probe[1, 4] / probe.alpha[1]
    target = 1.0
    near_singular_eps = nothing
    for _ in 1:32
        target = prevfloat(target)
        desired_alpha = target / coupling_per_alpha
        ratio = desired_alpha / (3 * solve_grid.volumes[1])
        candidate_eps = (1 + 2ratio) / (1 - ratio)
        candidate = dda_operator_3d(solve_grid, 1.0, candidate_eps)
        gap = abs(1 - candidate[1, 4])
        if 0 < gap < 1.0e-14
            near_singular_eps = candidate_eps
            break
        end
    end
    @test near_singular_eps !== nothing
    near_singular_eps === nothing && error(
        "failed to construct the near-singular DDA adjoint regression system")
    stable_incident = fill(
        CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im), 2)
    near_singular_result = solve_dda_3d(
        solve_grid, 1.0, near_singular_eps, stable_incident)
    huge_adjoint_rhs = ComplexF64[
        1.0e308, 0, 0,
        -1.0e308, 0, 0,
    ]
    @test_throws OverflowError solve_dda_adjoint_3d(
        near_singular_result, huge_adjoint_rhs)
end

println("  PASS")
