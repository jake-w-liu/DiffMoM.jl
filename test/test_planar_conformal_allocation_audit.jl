using DiffMoM, Test

function _solver_audit_conformal_fixture(walls)
    a, b = .003, .002
    stack = PlanarStackup([PlanarLayer(1., 1., .0005),
        PlanarLayer(1., 1., .0005)], TERM_GND, TERM_GND, a, b)
    mesh = PlanarConformalMesh([0. a a 0.; 0. 0. b b], [1 1; 2 3; 3 4])
    PlanarConformalProblem(stack, mesh, [PlanarConformalPort(1, :west, (0., b)),
        PlanarConformalPort(1, :east, (0., b))]; sidewalls=walls)
end

function _solver_audit_conformal_weight_allocations(prob, te, tm, kx, ky)
    DiffMoM._planar_conformal_weights!(te, tm, prob, kx, ky)
    @allocated DiffMoM._planar_conformal_weights!(te, tm, prob, kx, ky)
end

@testset "conformal modal weights: reuse buffers without edge slices" begin
    for walls in (WALL_PEC, WALL_PMC)
        prob = _solver_audit_conformal_fixture(walls)
        te = zeros(length(prob.basis.width)); tm = similar(te)
        for (m, n) in ((0, 1), (1, 0), (2, 3), (97, 89))
            kx, ky = m*pi/prob.stack.a, n*pi/prob.stack.b
            @test _solver_audit_conformal_weight_allocations(prob, te, tm, kx, ky) == 0
            @test all(isfinite, te) && all(isfinite, tm)
        end
    end
end
