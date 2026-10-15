module DensityProjectionContinuationTests
using Test, DiffMoM, Random

function reference(rho, beta, eta)
    setprecision(BigFloat, 2048) do
        r, b, e = BigFloat(rho), BigFloat(beta), BigFloat(eta)
        denominator = tanh(b * e) + tanh(b * (1 - e))
        value = (tanh(b * e) + tanh(b * (r - e))) / denominator
        derivative = b / (cosh(b * (r - e))^2 * denominator)
        Float64(value), Float64(derivative)
    end
end

function output_allocation(n)
    zeros(n)
    @allocated zeros(n)
end

@testset "sharp density continuation retains tails with output-sized allocation" begin
    @test only(heaviside_project([.01], 100., .5)) == 2.3767774103081315e-43
    @test only(heaviside_derivative([.01], 100., .5)) == 5.49757001582043e-41
    for beta in (32., 64., 100., 128., 256., 512.),
            eta in (0., .01, .1, .5, .9, .99, 1.)
        densities = collect(range(0., 1.; length=17))
        append!(densities, filter(x -> 0 <= x <= 1,
            [nextfloat(0.), .01, prevfloat(eta), eta, nextfloat(eta),
             prevfloat(1.), eta-372.75/beta, eta-372.5/beta]))
        values = heaviside_project(densities, beta, eta)
        derivatives = heaviside_derivative(densities, beta, eta)
        for (index, rho) in enumerate(densities)
            expected_value, expected_derivative = reference(rho, beta, eta)
            @test abs(values[index] - expected_value) <=
                8max(eps(expected_value), nextfloat(0.))
            @test abs(derivatives[index] - expected_derivative) <=
                8max(eps(expected_derivative), nextfloat(0.))
            expected_value > 0 && @test values[index] > 0
            expected_derivative > 0 && @test derivatives[index] > 0
        end
    end

    densities = collect(range(0., 1.; length=1000))
    permutation = randperm(MersenneTwister(3812), length(densities))
    for beta in (64., 128., 256.), eta in (0., .1, .5, .9, 1.)
        values = heaviside_project(densities, beta, eta)
        @test issorted(values)
        @test heaviside_project(densities[permutation], beta, eta) == values[permutation]
        @test all(x -> 0 <= x <= 1, values)
        heaviside_derivative(densities, beta, eta)
        @test (@allocated heaviside_project(densities, beta, eta)) <=
            output_allocation(length(densities)) + 128
        @test (@allocated heaviside_derivative(densities, beta, eta)) <=
            output_allocation(length(densities)) + 128
    end

    # Keep the existing exceptional range and public type contracts.
    @test heaviside_project([.5], floatmax(Float64), .5) == [.5]
    @test heaviside_derivative([.5], floatmax(Float64), .5) == [floatmax(Float64)/2]
    @test heaviside_project([.5], nextfloat(0.), 0.) == [.5]
    @test heaviside_derivative([.5], nextfloat(0.), 0.) == [1.]
    @test eltype(heaviside_project(Float32[.25], 128f0, .5f0)) == Float32
    @test eltype(heaviside_derivative(Float32[.25], 128f0, .5f0)) == Float32
end
end
