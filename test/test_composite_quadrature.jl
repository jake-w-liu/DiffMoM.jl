using Test, LinearAlgebra

@testset "composite triangle quadrature" begin
    for order in (28, 112)
        points, weights = tri_quad_rule(order)
        @test length(points) == length(weights) == order
        @test all(>(0), weights)
        @test all(p -> p[1] > 0 && p[2] > 0 && sum(p) < 1, points)
        @test length(unique(points)) == order
        for a in 0:5, b in 0:(5-a)
            # The exact reference-triangle moment is a! b! / (a+b+2)!.
            expected = Float64(factorial(big(a)) * factorial(big(b)) //
                               factorial(big(a+b+2)))
            computed = sum(w * p[1]^a * p[2]^b for (p, w) in zip(points, weights))
            @test isapprox(computed, expected; rtol=2e-13, atol=0)
        end
    end
    for invalid in (0, 2, 8, 27, 29, 113, typemax(Int))
        @test_throws ErrorException tri_quad_rule(invalid)
    end

    mesh = make_rect_plate(0.08, 0.06, 2, 2)
    rwg = build_rwg(mesh)
    k = 2pi * 3e9 / 299792458.0
    electric = CVec3(1+1im, -0.25+0.5im, 0)
    imported = make_imported_excitation(r -> electric; min_quad_order=8)
    @test imported.min_quad_order == 8
    @test DiffMoM._effective_quad_order(3, 8) == 28
    @test DiffMoM._effective_quad_order(29, 3) == 112
    @test_throws ArgumentError make_imported_excitation(r -> electric; min_quad_order=113)

    expected_rhs = ComplexF64[]
    for n in 1:rwg.nedges
        plus = triangle_center(mesh, rwg.tplus[n]) -
               Vec3(mesh.xyz[:, rwg.vplus_opp[n]])
        minus = triangle_center(mesh, rwg.tminus[n]) -
                Vec3(mesh.xyz[:, rwg.vminus_opp[n]])
        value = -(rwg.len[n] / 2) * (
            conj(rwg.coeff_plus[n]) * dot(plus, electric) -
            conj(rwg.coeff_minus[n]) * dot(minus, electric))
        push!(expected_rhs, value)
    end
    for order in (28, 112)
        actual_rhs = assemble_excitation(mesh, rwg, imported; quad_order=order)
        @test isapprox(actual_rhs, expected_rhs; rtol=2e-13, atol=1e-17)
        cache = DiffMoM._build_efie_cache(mesh, rwg, k; quad_order=order)
        @test cache.Nq == order
        @test length(cache.wq_hi) == order
        @test all(p -> length(p) == order, cache.quad_pts_hi)
        @test_throws ArgumentError DiffMoM._build_efie_cache(
            mesh, rwg, k; quad_order=order, max_cache_bytes=1)
    end
    @test isapprox(assemble_excitation(mesh, rwg, imported; quad_order=8),
        expected_rhs; rtol=2e-13, atol=1e-17)
    default_cache = DiffMoM._build_efie_cache(mesh, rwg, k)
    @test default_cache.Nq == 3
    @test length(default_cache.wq_hi) == 7

    matrix = assemble_Z_efie(mesh, rwg, k; quad_order=28)
    operator = matrixfree_efie_operator(mesh, rwg, k; quad_order=28)
    vector = ComplexF64.(1:rwg.nedges) .+ 0.25im
    @test isapprox(operator * vector, matrix * vector; rtol=2e-13, atol=1e-16)
    @test isapprox(adjoint(operator) * vector, adjoint(matrix) * vector;
        rtol=2e-13, atol=1e-16)
end
