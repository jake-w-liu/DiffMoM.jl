using Test
using LinearAlgebra
using Random
using DiffMoM
using JSON

@testset "Adaptive exact residual precision" begin
    rng = MersenneTwister(17140)
    for n in (2, 5, 12), trial in 1:5
        matrix = randn(rng, ComplexF64, n, n) + n * I
        current = randn(rng, ComplexF64, n)
        rhs = matrix * current
        for operator in (matrix, adjoint(matrix))
            physical_rhs = operator === matrix ? rhs : operator * current
            expected = setprecision(BigFloat, 4352) do
                exact_a = Complex{BigFloat}.(operator)
                exact_x = Complex{BigFloat}.(current)
                exact_b = Complex{BigFloat}.(physical_rhs)
                Float64(norm(exact_a * exact_x - exact_b) / norm(exact_b))
            end
            actual = DiffMoM._true_residual_ratio(operator, current, physical_rhs, "precision test")
            @test isapprox(actual, expected; rtol=4eps(Float64), atol=0.0)
            bits = DiffMoM._true_residual_dot_precision(
                operator, 1, DiffMoM._true_residual_component_bits(current), physical_rhs[1])
            @test bits == 256
        end
    end
    # Column scaling creates wide IEEE exponent spans while keeping the
    # products and physical RHS representable.
    scales = exp2.([-900.0, -300.0, 300.0, 900.0])
    ordinary = ComplexF64[1 2im -1 1im; 2 1 -1im 2; -1 1im 3 1; 1im -2 1 4]
    matrix = ordinary .* transpose(scales)
    current = ComplexF64[0.1, 0.2im, 0.3, -0.4im] ./ scales
    rhs = matrix * current
    expected = setprecision(BigFloat, 4352) do
        aa, xx, bb = Complex{BigFloat}.(matrix), Complex{BigFloat}.(current), Complex{BigFloat}.(rhs)
        Float64(norm(aa * xx - bb) / norm(bb))
    end
    actual = DiffMoM._true_residual_ratio(matrix, current, rhs, "wide precision test")
    @test isapprox(actual, expected; rtol=4eps(Float64), atol=0.0)
    @test DiffMoM._true_residual_dot_precision(
        matrix, 1, DiffMoM._true_residual_component_bits(current), rhs[1]) > 256
end

@testset "Saturated expansion retains every exact residual term" begin
    # Freeze the failing numerical inputs, not a recomputed LU solution whose
    # rounded coefficients can vary with the BLAS implementation or settings.
    # The fixture contains no expected residual; MPFR supplies that oracle.
    fixture = JSON.parsefile(joinpath(@__DIR__, "fixtures", "residual_expansion_sphere_row.json"))
    current = ComplexF64.(complex.(fixture["current_real"], fixture["current_imag"]))
    n = length(current)
    matrix = zeros(ComplexF64, n, n)
    matrix[1, :] = complex.(fixture["matrix_real"], fixture["matrix_imag"])
    rhs = zeros(ComplexF64, n)
    rhs[1] = complex(fixture["rhs_real"], fixture["rhs_imag"])
    expected_vector, expected_ratio = setprecision(BigFloat, 4352) do
        residual = sum(Complex{BigFloat}(matrix[1, j]) * Complex{BigFloat}(current[j])
                       for j in eachindex(current)) - Complex{BigFloat}(rhs[1])
        expected = zeros(ComplexF64, n)
        expected[1] = ComplexF64(residual)
        expected, Float64(abs(residual) / abs(Complex{BigFloat}(rhs[1])))
    end
    parts = zeros(Float64, DiffMoM._RESIDUAL_EXPANSION_SLOTS)
    imaginary_parts = similar(parts)
    saturated = setprecision(BigFloat, 4352) do
        DiffMoM._true_residual_expansion_row(
            matrix, 1, current, rhs[1], parts, imaginary_parts) === nothing
    end
    @test saturated
    actual_ratio = DiffMoM._true_residual_ratio(matrix, current, rhs, "saturated expansion")
    @test isapprox(actual_ratio, expected_ratio; rtol=4eps(Float64), atol=0.0)
    # Express the residual as a matrix product too, exercising its independent
    # caller of the same bounded expansion rather than only the norm path.
    extended = hcat(matrix, -rhs)
    vector = reshape(vcat(current, 1.0+0im), :, 1)
    for operator in (extended, adjoint(copy(adjoint(extended))))
        result = DiffMoM._finite_matrix_columns(operator, vector, "saturated column product")
        @test result[:, 1] == expected_vector
    end
    @test DiffMoM._residual_expansion_add!(Float64[], 0, 1.0) == -1
    @test DiffMoM._residual_expansion_add!(zeros(1), -1, 1.0) == -1
end
