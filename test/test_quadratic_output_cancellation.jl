using Test
using LinearAlgebra
using Random
using DiffMoM

@testset "Proper-complex quadratic variance survives trace cancellation" begin
    small = 2.0^-30
    factor = ones(ComplexF64, 2, 1)
    covariance = factor * adjoint(factor)
    weight = ComplexF64[1 small; small -1]
    for mean_scale in (0.0 + 0im, 0.3 + 0.2im)
        mean = vec(factor .* mean_scale)
        outputs = ErrorOutputs(mean, covariance, factor, copy(mean), (;))
        expected = setprecision(BigFloat, 256) do
            mu = Complex{BigFloat}.(mean)
            sigma = Complex{BigFloat}.(covariance)
            w = Complex{BigFloat}.(weight)
            product = w * sigma
            (mean=Float64(real(dot(mu, w * mu) + tr(product))),
             variance=Float64(real(tr(product * product) +
                2dot(w * mu, sigma * (w * mu)))))
        end
        actual = quadratic_output_moments(outputs, weight)
        @test actual.mean ≈ expected.mean rtol=32eps(Float64)
        @test actual.variance ≈ expected.variance rtol=32eps(Float64)
        @test actual.variance > 0
    end

    # The output factor can have many more latent columns than output rows.
    # Its equivalent representation must preserve the same quadratic law.
    wide_factor = repeat(factor ./ 2, 1, 4)
    outputs = ErrorOutputs(zeros(ComplexF64, 2), covariance,
        wide_factor, zeros(ComplexF64, 2), (;))
    moments = quadratic_output_moments(outputs, weight)
    @test moments.variance ≈ 4small^2 rtol=1e-6
    wide_moments = quadratic_output_moments(outputs, BigFloat.(real.(weight)))
    @test wide_moments.variance ≈ 4small^2 rtol=1e-6
    samples = sample_error_outputs(outputs, Xoshiro(23), 20_000)
    values = [real(dot(view(samples, :, column), weight * view(samples, :, column)))
              for column in axes(samples, 2)]
    observed_mean = sum(values) / length(values)
    observed_variance = sum(abs2, values .- observed_mean) / (length(values) - 1)
    @test observed_mean ≈ moments.mean rtol=0.05
    @test observed_variance ≈ moments.variance rtol=0.1

    malformed = ErrorOutputs(zeros(ComplexF64, 2), ones(ComplexF64, 1, 1),
        factor, zeros(ComplexF64, 2), (;))
    @test_throws DimensionMismatch quadratic_output_moments(malformed, weight)
end
