using DiffMoM, Test, LinearAlgebra

function _planar_integration_fixture()
    grid = CellGrid(4e-3, 3e-3, 6, 5)
    stack = PlanarStackup([PlanarLayer(2.0-.02im, 1., .4e-3; epsr_z=2.2),
        PlanarLayer(1.0, 1., .4e-3)], TERM_GND, TERM_GND, grid.a, grid.b)
    sheet = sheet_level(1, 6, 5)
    rasterize_rect!(sheet, grid, 0., grid.a, .6e-3, 2.4e-3)
    rows = findall(sheet.mask[1, :])
    sheet.connect_west[rows] .= true
    sheet.connect_east[rows] .= true
    return build_planar_problem(stack, grid, [sheet],
        [PlanarPort(1, :west, first(rows):last(rows), 50.),
         PlanarPort(1, :east, first(rows):last(rows), 75.)])
end

@testset "planar: solver, currents, sweep and adjoint integration" begin
    prob = _planar_integration_fixture()
    kw = (; mx=20, my=18, block=48, surface_zs=.1+.1im)
    dense = solve_planar(prob, 5e9; kw...)
    fast = solve_planar(prob, 5e9; method=:ufft, rtol=1e-11,
        memory=100, kw...)
    @test fast isa PlanarUFFTResult
    @test fast.s ≈ dense.s rtol=1e-9
    @test_throws ArgumentError solve_planar(prob, 5e9; method=:unknown)
    waves = ComplexF64[.2+.3im, -.1+.1im]
    maps = planar_current_maps(dense; incident_waves=waves)
    fastmaps = planar_current_maps(fast; incident_waves=waves)
    @test fastmaps[1].jx ≈ maps[1].jx rtol=1e-8
    @test fastmaps[1].jy ≈ maps[1].jy rtol=1e-8
    @test planar_sparams(prob, [5e9]; method=:ufft, rtol=1e-11,
        memory=100, kw...)[1] ≈ dense.s rtol=1e-9

    params = [PlanarParam(1, :epsr, :re),
        PlanarParam(1, :thickness, :re), PlanarParam(1, :epsr_z, :re)]
    objective(Y) = sum(abs2, Y)
    pullback(Y) = 2conj.(Y)
    J, grad = planar_objective_gradient(prob, 5e9, objective;
        params, gY=pullback, kw...)
    Jfast, gradfast = planar_objective_gradient(prob, 5e9, objective;
        params, gY=pullback, method=:ufft, rtol=1e-11, memory=100, kw...)
    @test Jfast ≈ J rtol=1e-9
    @test gradfast ≈ grad rtol=1e-8
    values = planar_param_values(prob.stack, params)
    for p in eachindex(params)
        delta = abs(values[p])*1e-5
        upper, lower = copy(values), copy(values)
        upper[p] += delta
        lower[p] -= delta
        function sample(theta)
            stack = planar_with_params(prob.stack, params, theta)
            changed = build_planar_problem(stack, prob.grid, prob.sheets, prob.ports)
            return objective(solve_planar(changed, 5e9; kw...).y)
        end
        fd = (sample(upper)-sample(lower))/(2delta)
        @test grad[p] ≈ fd rtol=3e-5 atol=1e-9
    end
end

@testset "planar: gradient aggregate resource preflight" begin
    prob = _planar_integration_fixture()
    calls = Ref(0)
    objective(Y) = (calls[] += 1; sum(abs2, Y))
    @test_throws ArgumentError planar_objective_gradient(prob, 5e9,
        objective; params=fill(PlanarParam(1, :epsr, :re), 1000),
        mx=4, my=4, block=1, max_bytes=100_000)
    @test calls[] == 0
    @test_throws ArgumentError planar_objective_gradient(prob, 5e9,
        objective; mx=typemax(Int), my=typemax(Int))
    @test_throws ArgumentError planar_objective_gradient(prob, 5e9,
        objective; block=0)
    @test calls[] == 0
end
