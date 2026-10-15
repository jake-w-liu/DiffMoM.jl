using Test
using LinearAlgebra
using DiffMoM

@testset "Quadratic expectations retain cancellation across all Gaussian terms" begin
    small = 2.0^-60
    correlated = ComplexF64[1 0; 1 1]
    cases = (
        (fill(1e150+0im, 2), 1e-150Matrix{ComplexF64}(I, 2, 2),
         ComplexF64[1 small; small -1]),
        (zeros(ComplexF64, 2), correlated, ComplexF64[2 small; small -1]),
        (ComplexF64[1, 0], correlated, ComplexF64[1 small; small -1]),
    )
    for (mean, factor, weight) in cases
        covariance = factor * adjoint(factor)
        # The shared factor defines the Gaussian, including contributions
        # which round away while constructing its Float64 covariance.
        reference = setprecision(BigFloat, 6656) do
            mu, f, w = Complex{BigFloat}.(mean), Complex{BigFloat}.(factor), Complex{BigFloat}.(weight)
            sigma = f * adjoint(f)
            product = w * sigma
            (mean=Float64(real(dot(mu, w*mu) + tr(product))),
             variance=Float64(real(tr(product*product) + 2dot(w*mu, sigma*(w*mu)))))
        end
        for shared in (factor, repeat(factor/2, 1, 4))
            outputs = ErrorOutputs(mean, covariance, shared, copy(mean), (;))
            actual = quadratic_output_moments(outputs, weight)
            @test reference.mean > 0
            @test actual.mean ≈ reference.mean rtol=32eps(Float64)
            @test actual.variance ≈ reference.variance rtol=32eps(Float64)
        end
    end
end
