using Test, DiffMoM

@testset "planar adjoint objective remains finite after storage conversion" begin
    Y = zeros(ComplexF64, 1, 1)
    for value in (big"1e500", -big"1e500", Inf, NaN)
        @test_throws ArgumentError DiffMoM._checked_objective(_ -> value, Y)
    end
    for value in (1, 1.25f0, 5//4, big"1.25")
        @test DiffMoM._checked_objective(_ -> value, Y) == Float64(value)
    end

    grid = CellGrid(.004, .003, 2, 1)
    stack = PlanarStackup([PlanarLayer(2., 1., .0004),
        PlanarLayer(1., 1., .0004)], TERM_GND, TERM_GND, grid.a, grid.b)
    sheet = sheet_level(1, 2, 1)
    sheet.mask .= true
    sheet.connect_west .= true
    sheet.connect_east .= true
    problem = build_planar_problem(stack, grid, [sheet],
        [PlanarPort(1, :west, 1:1, 50.), PlanarPort(1, :east, 1:1, 50.)])
    params = [PlanarParam(1, :epsr, :re)]
    calls, derivatives = Ref(0), Ref(0)
    objective(Y) = (calls[] += 1; big"1e500")
    pullback(Y) = (derivatives[] += 1; zeros(ComplexF64, size(Y)))
    for gY in (nothing, pullback)
        @test_throws ArgumentError planar_objective_gradient(problem, 5e9,
            objective; params, gY, mx=4, my=2)
    end
    @test calls[] == 2
    @test derivatives[] == 0
    value, gradient = planar_objective_gradient(problem, 5e9, _ -> big"1.25";
        params, gY=Y -> zeros(ComplexF64, size(Y)), mx=4, my=2)
    @test value == 1.25
    @test gradient == [0.]
end
