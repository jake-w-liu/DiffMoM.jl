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
    lambda = solve_dda_adjoint_3d(res, weights .* E)
    grad = gradient_epsr_dda_3d(res, lambda)

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
end

println("  PASS")
