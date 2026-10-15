using Test, LinearAlgebra, DiffMoM

@testset "planar extraction retains finite admittance and angular-frequency ranges" begin
    # A nonzero finite admittance is a current path even when squaring its
    # magnitude (or its spacing) underflows or overflows.
    for scale in (1e-200, 1e200), phase in (1.0 + 0im, -1im, 1 - 1im)
        admittance = scale * phase
        for Y in (reshape([admittance], 1, 1),
                ComplexF64[admittance -admittance; -admittance admittance])
            extracted = planar_inductor(Y, 1e9)
            reference = inv(admittance)
            @test extracted.z == reference
            @test extracted.r == real(reference)
            @test extracted.l ≈ Float64(BigFloat(imag(reference)) /
                (2big(pi) * BigFloat(1e9))) rtol=4eps(Float64)
        end
    end
    @test_throws ArgumentError planar_inductor(zeros(ComplexF64, 1, 1), 1e9)
    @test_throws ArgumentError planar_inductor(reshape(ComplexF64[Inf], 1, 1), 1e9)

    # Both a huge angular frequency and a huge intermediate value/f are
    # avoidable. Compare each result with arithmetic that has wider range.
    for (reactance, frequency) in ((1e308, 1e308), (1., 1e308),
            (1e308, .1), (1e-308, 1e-308), (-1e308, 1e308))
        expected = Float64(BigFloat(reactance) /
            (2big(pi) * BigFloat(frequency)))
        for extracted in (planar_inductor(reshape(ComplexF64[inv(im * reactance)], 1, 1), frequency),
                DiffMoM._rlgc_of(1., im * reactance, frequency))
            @test extracted.l ≈ expected rtol=4eps(Float64) atol=nextfloat(0.)
            !iszero(expected) && @test !iszero(extracted.l)
        end
    end

    # Exercise the public single- and multiconductor line paths. The
    # measured electrical length is independent of the frequency metadata.
    gl, zc, length_m, frequency = .02 + .2im, 50., .1, 1e308
    a, b = coth(gl) / zc, -csch(gl) / zc
    Y = ComplexF64[a b; b a]
    scalar = planar_rlgc(Y, length_m, frequency)
    expected_l = Float64(imag(BigFloat(zc) * Complex{BigFloat}(gl) /
        BigFloat(length_m)) / (2big(pi) * BigFloat(frequency)))
    expected_c = Float64(imag(Complex{BigFloat}(gl) / BigFloat(zc) /
        BigFloat(length_m)) / (2big(pi) * BigFloat(frequency)))
    @test scalar.l ≈ expected_l rtol=1e-12 atol=nextfloat(0.)
    @test scalar.c ≈ expected_c rtol=1e-12 atol=nextfloat(0.)
    @test scalar.l > 0 && scalar.c > 0
    coupled = zeros(ComplexF64, 4, 4)
    coupled[[1, 3], [1, 3]] = Y
    coupled[[2, 4], [2, 4]] = Y
    multi = planar_rlgc(coupled, length_m, frequency)
    @test diag(multi.l) ≈ fill(expected_l, 2) rtol=1e-12 atol=nextfloat(0.)
    @test diag(multi.c) ≈ fill(expected_c, 2) rtol=1e-12 atol=nextfloat(0.)
    @test all(>(0), diag(multi.l)) && all(>(0), diag(multi.c))
end
