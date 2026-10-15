module POPhaseResourceTests
using DiffMoM, LinearAlgebra, Test

function phase_allocations(k, direction, vertices, area)
    DiffMoM._phase_integral_analytical(k, direction, vertices..., area)
    return @allocated DiffMoM._phase_integral_analytical(k, direction, vertices..., area)
end

@testset "PO canonical phase integration avoids per-interaction storage" begin
    vertices = (Vec3(-.03, .01, .02), Vec3(.08, .02, .02), Vec3(.01, .09, .02))
    area = norm(cross(vertices[2]-vertices[1], vertices[3]-vertices[1]))/2
    nodes, weights = DiffMoM.gauss_legendre(24)
    permutations = ((1,2,3), (2,3,1), (3,1,2), (1,3,2), (3,2,1), (2,1,3))
    k = 7.0
    for direction in (Vec3(0,0,1), Vec3(1,0,.4), Vec3(.2,-.7,.3))
        # Independent product quadrature on the unit simplex.
        reference = sum(Iterators.product(eachindex(nodes), eachindex(nodes))) do (i, j)
            u, v = (nodes[i]+1)/2, (nodes[j]+1)/2
            point = vertices[1] + u*(vertices[2]-vertices[1]) +
                    (1-u)*v*(vertices[3]-vertices[1])
            (area/2)*weights[i]*weights[j]*(1-u)*exp(im*k*dot(direction,point))
        end
        for permutation in permutations
            ordered = ntuple(i -> vertices[permutation[i]], 3)
            @test DiffMoM._phase_integral_analytical(k, direction, ordered..., area) ≈
                  reference rtol=1e-12 atol=1e-15
        end
        @test phase_allocations(k, direction, vertices, area) == 0
    end
end
end
