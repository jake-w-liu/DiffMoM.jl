module NearfieldPrefactorResourceTests
using DiffMoM, Test, LinearAlgebra, Random

# Independent basis-by-basis direct quadrature, using the differentiated
# scalar Green formula rather than the production triangle-current cache.
function direct_reference(mesh, rwg, currents, observations, k)
    xi, weights = DiffMoM.tri_quad_rule(3)
    result = zeros(ComplexF64, 3, length(observations))
    eta = 376.730313668
    for (column, observation) in enumerate(observations)
        field = zero(CVec3)
        for basis in eachindex(currents), triangle in
                (rwg.tplus[basis], rwg.tminus[basis])
            points = DiffMoM.tri_quad_points(mesh, triangle, xi)
            area = triangle_area(mesh, triangle)
            divergence = div_rwg(rwg, basis, triangle)
            for q in eachindex(weights)
                offset = observation - points[q]
                distance = norm(offset)
                green = exp(-im*k*distance) / (4pi*distance)
                gradient = -(1+im*k*distance) * green / distance^2 * offset
                current = eval_rwg(rwg, basis, points[q], triangle)
                field += currents[basis] * (2area*weights[q]) *
                    ((-im*k*eta) * green * current -
                     (im*eta/k) * divergence * gradient)
            end
        end
        result[:, column] = field
    end
    result
end

function evaluation_allocation(mesh, rwg, currents, points, k)
    compute_nearfield(mesh, rwg, currents, points, k)
    @allocated compute_nearfield(mesh, rwg, currents, points, k)
end

@testset "near-field prefactors keep quadrature allocation independent of interactions" begin
    mesh = make_rect_plate(.2, .15, 4, 4)
    rwg = build_rwg(mesh)
    currents = randn(MersenneTwister(31241), ComplexF64, rwg.nedges)
    observations = vec([Vec3(x, y, .2)
        for x in range(-.15, .15; length=5), y in range(-.1, .1; length=5)])
    k = 2pi*3e9/299792458.
    for wavenumber in (k, k-.2im)
        expected = direct_reference(mesh, rwg, currents, observations, wavenumber)
        actual = compute_nearfield(mesh, rwg, currents, observations, wavenumber)
        @test actual ≈ expected rtol=1e-12
        @test compute_nearfield(mesh, rwg, currents, observations[1], wavenumber) ≈
            actual[:, 1] rtol=1e-13
    end

    wave = make_plane_wave(Vec3(0., 0., -k), .7, Vec3(1., 0., 0.))
    expected = direct_reference(mesh, rwg, currents, observations, k)
    for (column, point) in enumerate(observations)
        expected[:, column] .+= wave.E0 .* wave.pol .* exp(-im*dot(wave.k_vec, point))
    end
    @test compute_total_field(mesh, rwg, currents, wave, observations, k) ≈
        expected rtol=1e-12

    # Before the prefactor snapshot, these 2400 ordinary quadrature terms
    # allocated over 1.6 MB through dynamic scalar/tuple dispatch alone.
    small = evaluation_allocation(mesh, rwg, currents, observations, k)
    repeated = repeat(observations, 4)
    larger = evaluation_allocation(mesh, rwg, currents, repeated, k)
    @test small < 350_000
    @test larger < small + 100_000
    @test_throws ArgumentError compute_nearfield(mesh, rwg, currents, observations,
        k; max_work_bytes=1)
end
end
