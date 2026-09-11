using Test, Random, LinearAlgebra

function _pn_columns_oracle(A, X)
    setprecision(BigFloat, 4352) do
        # Every Float64 input and product is exact at this precision, including
        # the full exponent range and sums over these finite test matrices.
        ComplexF64.(Complex{BigFloat}.(A) * Complex{BigFloat}.(X))
    end
end

@testset "checked batched dense products" begin
    rng = MersenneTwister(17092026)
    for (m, n, q) in ((1, 1, 1), (3, 2, 5), (7, 13, 3), (31, 19, 7))
        for is_adjoint in (false, true)
            parent = randn(rng, ComplexF64, is_adjoint ? (n, m) : (m, n))
            A = is_adjoint ? adjoint(parent) : parent
            X = randn(rng, ComplexF64, n, q)
            saved_A, saved_X = copy(A), copy(X)
            expected = _pn_columns_oracle(A, X)
            actual = DiffMoM._finite_matrix_columns(A, X, "batched oracle")
            @test size(actual) == (m, q)
            @test isapprox(actual, expected; rtol=64n * eps(Float64), atol=0)
            @test A == saved_A
            @test X == saved_X
        end
    end

    cancellation = ComplexF64[
        1e16 1 -1e16;
        1e16im 1+im -1e16im;
        1e16+1e16im 1-2im -1e16-1e16im;
        2.0^128 2.0^-128 -2.0^128
    ]
    multipliers = ComplexF64[1 1im -1+im; 1 1im -1+im; 1 1im -1+im]
    @test DiffMoM._finite_matrix_columns(cancellation, multipliers, "cancellation") ==
          _pn_columns_oracle(cancellation, multipliers)
    @test DiffMoM._finite_matrix_columns(
        adjoint(copy(adjoint(cancellation))), multipliers, "adjoint cancellation") ==
          _pn_columns_oracle(cancellation, multipliers)

    tiny = nextfloat(0.0)
    extremes = [
        (ComplexF64[tiny tiny; 0 tiny], ComplexF64[0.5 1; 0.5 0]),
        (ComplexF64[1e300 0; 0 1e-300], ComplexF64[1e-300 2e-300; 1e300 2e300]),
        (ComplexF64[1e300 1 -1e300], reshape(ComplexF64[1, 1, 1], :, 1)),
        (ComplexF64[1e-300 1e300; 1e-300 -1e300],
         ComplexF64[1e300 1e300im; 1e-300 1e-300im]),
    ]
    for (A, X) in extremes
        @test DiffMoM._finite_matrix_columns(A, X, "extreme batched oracle") ==
              _pn_columns_oracle(A, X)
    end

    @test DiffMoM._finite_matrix_columns(
        zeros(ComplexF64, 3, 0), zeros(ComplexF64, 0, 2), "empty contraction") ==
          zeros(ComplexF64, 3, 2)
    @test size(DiffMoM._finite_matrix_columns(
        zeros(ComplexF64, 3, 2), zeros(ComplexF64, 2, 0), "empty columns")) == (3, 0)
    @test size(DiffMoM._finite_matrix_columns(
        zeros(ComplexF64, 0, 2), ones(ComplexF64, 2, 3), "empty rows")) == (0, 3)

    Astore = randn(rng, ComplexF64, 10, 12)
    Xstore = randn(rng, ComplexF64, 12, 6)
    A = view(Astore, 1:2:9, 2:2:12)
    X = view(Xstore, 1:2:11, 1:2:5)
    @test isapprox(DiffMoM._finite_matrix_columns(A, X, "strided views"),
        _pn_columns_oracle(A, X); rtol=384eps(Float64), atol=0)
    @test_throws DimensionMismatch DiffMoM._finite_matrix_columns(
        zeros(ComplexF64, 3, 2), zeros(ComplexF64, 3, 1), "wrong inner size")
end
