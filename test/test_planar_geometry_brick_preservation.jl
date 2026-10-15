using Test, DiffMoM

@testset "sheet geometry transformations retain dielectric bricks" begin
    grid = CellGrid(4e-3, 3e-3, 8, 6)
    stack = PlanarStackup([PlanarLayer(1, 1, .5e-3), PlanarLayer(1, 1, .5e-3)],
                         TERM_GND, TERM_GND, grid.a, grid.b)
    brick = brick_level(1, 8, 6, (8., 9., 10.))
    brick.mask[2, 2] = brick.mask[6, 5] = true
    original = copy(brick.mask)
    sheet = sheet_level(1, 8, 6)
    sheet.mask[2:3, 3:4] .= true
    sheet.mask[6:7, 3:4] .= true
    basis = build_planar_basis(grid, [sheet], PlanarPort[]; bricks=[brick])
    bare = PlanarProblem(stack, grid, [sheet], PlanarPort[], ViaLevel[], basis, VolLevel[], [brick])
    signal = PlanarPort(1, :terminal_x, 5, 3:4, 50.; metal_side=:positive)
    reference = PlanarPort(1, :terminal_x, 3, 3:4, 50.; metal_side=:negative)
    bridge = planar_floating_bridge(bare, signal, reference; source_edge=4)
    @test length(bridge.problem.bricks) == 1
    @test only(bridge.problem.bricks).mask == original
    @test only(bridge.problem.bricks).epsr == brick.epsr
    @test count(DiffMoM._is_brick_kind, bridge.problem.basis.kind) ==
          count(DiffMoM._is_brick_kind, bare.basis.kind)
    @test_throws ArgumentError planar_floating_bridge(bare, signal, reference; max_bytes=1)

    sheet.mask .= true
    sheet.connect_west[2] = sheet.connect_east[2] = true
    ports = [PlanarPort(1, :west, 2:2, 50.), PlanarPort(1, :east, 2:2, 50.)]
    problem = build_planar_problem(stack, grid, [sheet], ports; bricks=[brick])
    for (axis, edge) in ((:x, 4), (:y, 3))
        plan = planar_subdivide(problem, axis, edge)
        @test all(part -> length(part.bricks) == 1, plan.parts)
        masks = [only(part.bricks).mask for part in plan.parts]
        @test (axis == :x ? vcat(masks...) : hcat(masks...)) == original
        @test all(part -> only(part.bricks).epsr == brick.epsr, plan.parts)
        @test all(part -> any(DiffMoM._is_brick_kind, part.basis.kind), plan.parts)
        masks[1][2, 2] = false
        @test brick.mask == original
    end
    circuit=PlanarCircuit(2,[1,2])
    circuit.spice_instances=Dict{String,Any}("Xowner"=>(nodes=[1,2],))
    explicit=PlanarSubdivision(circuit,[problem],[[1,2]])
    @test explicit.circuit.spice_instances == circuit.spice_instances
    @test explicit.circuit.spice_instances !== circuit.spice_instances
    shared=zeros(ComplexF64,2,2);anchors=[1];refs=ComplexF64[50,50]
    solved=PlanarCircuitResult(1.,shared,shared,shared,shared,anchors,refs)
    @test DiffMoM._subdivision_retained_payload(solved) == sizeof(shared)+sizeof(anchors)+sizeof(refs)
    bricks=[brick,brick]
    @test DiffMoM._subdivision_retained_payload(bricks) == sizeof(bricks)+8cld(length(brick.mask),64)
end
