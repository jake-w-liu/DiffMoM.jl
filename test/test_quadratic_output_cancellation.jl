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

function _quadratic_zero_resource_measure(outputs, weight)
    return @allocated quadratic_output_moments(outputs, weight)
end

@testset "Exact-zero quadratic moments avoid latent recomputation" begin
    rng = Xoshiro(73)
    for p in (12, 24)
        factor = randn(rng, ComplexF64, p, 128) / sqrt(128)
        mean = randn(rng, ComplexF64, p)
        outputs = ErrorOutputs(mean, factor * adjoint(factor), factor, copy(mean), (;))
        zero_weight = zeros(ComplexF64, p, p)
        @test quadratic_output_moments(outputs, zero_weight) == (mean=0.0, variance=0.0)
        _quadratic_zero_resource_measure(outputs, zero_weight)
        @test _quadratic_zero_resource_measure(outputs, zero_weight) <= 1024

        deterministic = ErrorOutputs(zeros(ComplexF64, p), zeros(ComplexF64, p, p),
            zeros(ComplexF64, p, 128), zeros(ComplexF64, p), (;))
        identity_weight = Matrix{ComplexF64}(I, p, p)
        @test quadratic_output_moments(deterministic, identity_weight) ==
              (mean=0.0, variance=0.0)
        _quadratic_zero_resource_measure(deterministic, identity_weight)
        @test _quadratic_zero_resource_measure(deterministic, identity_weight) <=
              64p^2 + 4096

        shifted = ErrorOutputs(mean, deterministic.covariance,
            deterministic.square_root, copy(mean), (;))
        moments = quadratic_output_moments(shifted, identity_weight)
        @test moments.mean ≈ sum(abs2, mean) rtol=8eps(Float64)
        @test moments.variance == 0
    end

    # A zero covariance-derived scale alone cannot prove zero variance. The
    # nonzero shared factor retains values rescued by a large finite weight.
    factor = fill(1e-170 + 0im, 1, 1)
    outputs = ErrorOutputs(zeros(ComplexF64, 1), zeros(ComplexF64, 1, 1),
        factor, zeros(ComplexF64, 1), (;))
    weight = fill(1e300, 1, 1)
    reference = setprecision(BigFloat, 4352) do
        latent = BigFloat(1e-170)^2 * BigFloat(1e300)
        (mean=Float64(latent), variance=Float64(latent^2))
    end
    moments = quadratic_output_moments(outputs, weight)
    @test moments.mean ≈ reference.mean rtol=8eps(Float64)
    @test moments.variance ≈ reference.variance rtol=8eps(Float64)
    @test moments.mean > 0 && moments.variance > 0

    invalid = ErrorOutputs(zeros(ComplexF64, 1), ones(ComplexF64, 1, 1),
        fill(ComplexF64(NaN), 1, 1), zeros(ComplexF64, 1), (;))
    @test_throws ArgumentError quadratic_output_moments(invalid, zeros(1, 1))
end
