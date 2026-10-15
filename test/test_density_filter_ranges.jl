using Test
using DiffMoM
using LinearAlgebra
using SparseArrays

@testset "Density filtering is invariant to weight units" begin
    for weight in (1e-320, 1e-200, 1.0, 1e200, 1e308)
        weights = sparse(reshape([weight], 1, 1))
        for input in ([0.3], [2.0], [1e-100], [1e100], [0.3 + 0.7im])
            @test apply_filter(weights, [weight], input) ≈ input rtol=8eps(Float64)
            @test apply_filter_transpose(weights, [weight], input) ≈ input rtol=8eps(Float64)
        end
    end

    weights = sparse([1.0 2.0; 3.0 1.0])
    sums = [3.0, 4.0]
    density = [0.3, 0.8]
    gradient = [0.7, -0.2]
    for scale in (1e-300, 1e300)
        scaled = scale * weights
        scaled_sums = scale * sums
        forward = apply_filter(scaled, scaled_sums, density)
        reverse = apply_filter_transpose(scaled, scaled_sums, gradient)
        @test forward ≈ apply_filter(weights, sums, density) rtol=8eps(Float64)
        @test reverse ≈ apply_filter_transpose(weights, sums, gradient) rtol=8eps(Float64)
        @test dot(gradient, forward) ≈ dot(reverse, density) rtol=8eps(Float64)
    end

    weights32 = sparse(reshape(Float32[1e-30], 1, 1))
    @test apply_filter(weights32, Float32[1e-30], Float32[1e-30]) ≈ Float32[1e-30]
    @test apply_filter_transpose(weights32, Float32[1e-30], Float32[1e30]) ≈ Float32[1e30]

    @test_throws OverflowError apply_filter(sparse(reshape([1e308], 1, 1)), [1.0], [2.0])
    @test_throws OverflowError apply_filter_transpose(sparse(reshape([1e308], 1, 1)), [1.0], [2.0])
end

@testset "Projection matches its formula outside the density interval" begin
    inputs = [-0.1, 0.0, 0.3, 1.0, 1.1]
    for beta in (1.0, 4.0), eta in (0.0, 0.5, 1.0)
        reference = Float64.(heaviside_project(BigFloat.(inputs), BigFloat(beta), BigFloat(eta)))
        @test heaviside_project(inputs, beta, eta) ≈ reference rtol=1e-13 atol=1e-15
        for value in (-0.1, 1.1)
            numerical = fd_grad(theta -> only(heaviside_project(theta, beta, eta)),
                                [value], 1; h=1e-5)
            @test only(heaviside_derivative([value], beta, eta)) ≈ numerical rtol=1e-7
        end
    end
end
