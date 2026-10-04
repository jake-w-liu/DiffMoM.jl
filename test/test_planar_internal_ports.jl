using DiffMoM, Test, LinearAlgebra

function _internal_port_problem(; polarity=1)
    grid = CellGrid(4e-3, 3e-3, 8, 6)
    stack = PlanarStackup([PlanarLayer(4.0 - 0.02im, 1.0, 0.5e-3),
        PlanarLayer(1.0, 1.0, 1e-3)], TERM_GND, TERM_SPACE, grid.a, grid.b)
    sheet = sheet_level(1, 8, 6)
    sheet.mask[2:7, 3:4] .= true
    return build_planar_problem(stack, grid, [sheet],
        [PlanarPort(1, :x, 3, 3:4, 50.0; polarity=polarity),
         PlanarPort(1, :x, 6, 3:4, 75.0)])
end

@testset "planar: internal delta-gap ports" begin
    prob = _internal_port_problem()
    r = solve_planar(prob, 5e9; mx=24, my=18)
    @test r.y ≈ transpose(r.y) rtol=1e-10 atol=1e-10
    @test r.s ≈ transpose(r.s) rtol=1e-10 atol=1e-10
    @test opnorm(r.s) <= 1 + 1e-9
    # Each voltage couples to the integral of J across its claimed cut.
    B = zeros(ComplexF64, planar_basis_count(prob.basis), 2)
    for b in eachindex(prob.basis.port)
        p = prob.basis.port[b]
        p == 0 && continue
        B[b, p] = prob.grid.dy
    end
    @test r.y ≈ -transpose(B) * (r.z_mom \ B) rtol=1e-11
    # Reversing one terminal flips its coupling and leaves diagonal data.
    flipped = solve_planar(_internal_port_problem(polarity=-1), 5e9;
        mx=24, my=18)
    D = Diagonal([-1.0, 1.0])
    @test flipped.y ≈ D * r.y * D rtol=1e-11
    @test flipped.s ≈ D * r.s * D rtol=1e-11
    @test_throws ArgumentError PlanarPort(1, :west, 1:1, 50.0; polarity=0)
    @test_throws ArgumentError PlanarPort(1, :diagonal, 2, 1:1, 50.0)
    @test_throws ArgumentError build_planar_problem(prob.stack, prob.grid,
        prob.sheets, [PlanarPort(1, :x, 0, 3:4, 50.0)])
    @test_throws ArgumentError build_planar_problem(prob.stack, prob.grid,
        prob.sheets, [PlanarPort(1, :x, 3, 1:4, 50.0)])
    @test_throws ArgumentError build_planar_problem(prob.stack, prob.grid,
        prob.sheets, [PlanarPort(1, :x, 3, 3:4, 50.0),
                     PlanarPort(1, :x, 3, 4:4, 50.0)])
    bad_stack = PlanarStackup(prob.stack.layers, prob.stack.bottom,
        prob.stack.top, 2prob.stack.a, prob.stack.b)
    @test_throws ArgumentError build_planar_problem(bad_stack, prob.grid,
        prob.sheets, prob.ports)
end

@testset "planar: axial via gap port" begin
    grid = CellGrid(2e-3, 2e-3, 4, 4)
    stack = PlanarStackup([PlanarLayer(4.0 - 0.05im, 1.0, 0.5e-3)],
        TERM_GND, TERM_SPACE, grid.a, grid.b)
    via = via_level(1, 4, 4)
    via.uni[2, 2] = true
    via.tap[2, 2] = true
    prob = build_planar_problem(stack, grid, SheetLevel[],
        [PlanarPort(1, :via, 6:6, 50.0)]; vias=[via])
    r = solve_planar(prob, 5e9; mx=16, my=16)
    B = ComplexF64[grid.dx * grid.dy, 0.5 * grid.dx * grid.dy]
    @test r.currents[:, 1] ≈ -(r.z_mom \ B) rtol=1e-12
    @test r.y[1, 1] ≈ -transpose(B) * (r.z_mom \ B) rtol=1e-12
    @test all(isfinite, r.s)
    @test_throws ArgumentError build_planar_problem(stack, grid, SheetLevel[],
        [PlanarPort(1, :via, 5:6, 50.0)]; vias=[via])
end
