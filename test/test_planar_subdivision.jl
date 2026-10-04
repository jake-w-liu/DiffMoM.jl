using Test,LinearAlgebra,DiffMoM

function _subdivision_test_line(n;transpose_geometry=false)
    a,b,nx,ny = transpose_geometry ? (6e-3,n*0.25e-3,24,n) : (n*0.25e-3,6e-3,n,24)
    grid=CellGrid(a,b,nx,ny)
    stack=PlanarStackup([PlanarLayer(1,1,0.5e-3),PlanarLayer(1,1,0.5e-3)],
        TERM_GND,TERM_GND,a,b)
    sheet=sheet_level(1,nx,ny)
    if transpose_geometry
        sheet.mask[10:15,:].=true
        sheet.connect_south[10:15].=true;sheet.connect_north[10:15].=true
        ports=[PlanarPort(1,:south,10:15,50),PlanarPort(1,:north,10:15,50)]
    else
        sheet.mask[:,10:15].=true
        sheet.connect_west[10:15].=true;sheet.connect_east[10:15].=true
        ports=[PlanarPort(1,:west,10:15,50),PlanarPort(1,:east,10:15,50)]
    end
    return build_planar_problem(stack,grid,[sheet],ports)
end

@testset "general double-delay: full series/shunt launch" begin
    # Symmetric reciprocal launch with a nonzero series term; the legacy
    # pure-shunt extraction cannot represent this box.
    E=planar_line_abcd(43,0.015+0.04im)
    L=planar_line_abcd(67,0.03+0.6im)
    y(T)=DiffMoM._y_of_abcd(T)
    cal=planar_double_delay_calibrate(y(E*L*E),y(E*L*L*E);length=0.002)
    @test cal.launch≈E rtol=1e-11
    @test cal.line≈L rtol=1e-11
    @test cal.residual<1e-11
    @test deembed_ports(y(E*L*E),[cal.launch,cal.launch])≈y(L) rtol=1e-11
    @test_throws ArgumentError planar_double_delay_calibrate(y(E*L*E),
        y(E*L*L*E);length=-1)
end

@testset "planar subdivision: independently solved whole EM line" begin
    whole=_subdivision_test_line(24)
    plan=planar_subdivide(whole,:x,8)
    @test length(plan.parts)==2
    @test [p.grid.nx for p in plan.parts]==[8,16]
    @test plan.parts[1].grid.dx==whole.grid.dx
    @test plan.parts[2].grid.dy==whole.grid.dy
    @test plan.terminals==[[(1,0),(3,0)],[(2,0),(3,0)]]
    identity_chain=Matrix{ComplexF64}(I,2,2)
    for f in (1e9,5e9,10e9)
        # Calibration standards are shorter than the whole (8 and16cells),
        # so the 24-cell oracle is independent of launch extraction.
        raw8=solve_planar(_subdivision_test_line(8),f;mx=32,my=96,retain_matrix=false)
        raw16=solve_planar(_subdivision_test_line(16),f;mx=64,my=96,retain_matrix=false)
        cal=planar_double_delay_calibrate(raw8.y,raw16.y;length=2e-3)
        # Local second-part ordering is [external east, cut west].
        result=solve_planar_subdivision(plan,f;retain_results=true,
            chains=[[identity_chain,cal.launch],[identity_chain,cal.launch]],
            part_keywords=[(mx=32,my=96),(mx=64,my=96)])
        reference=solve_planar(whole,f;mx=96,my=96,retain_matrix=false)
        @test maximum(abs,result.s-reference.s)<5e-5
        @test result.y≈reference.y rtol=1e-4
        @test all(r -> r.raw.z_mom===nothing,result.results)
        maps=planar_subdivision_currents(result;incident_waves=ComplexF64[1,0])
        @test length(maps)==2
        @test minimum(maps[2][1].x)>maximum(maps[1][1].x)
        @test all(isfinite,maps[1][1].jx)
        nodes=result.circuit.voltages[:,1]
        direct=planar_current_maps(result.results[1];voltages=nodes[[1,3]])
        @test maps[1][1].jx≈direct[1].jx
    end
    lightweight=solve_planar_subdivision(plan,1e9;mx=24,my=48)
    @test lightweight.results===nothing
    @test_throws ArgumentError planar_subdivision_currents(lightweight)
    @test_throws ArgumentError solve_planar_subdivision(plan,1e9;max_bytes=128)
    @test_throws ArgumentError planar_subdivide(whole,:x,0)
    @test_throws ArgumentError planar_subdivide(whole,:z,8)
end

@testset "planar subdivision: y cuts and multi-conductor mapping" begin
    prob=_subdivision_test_line(12;transpose_geometry=true)
    plan=planar_subdivide(prob,:y,4)
    @test [p.grid.ny for p in plan.parts]==[4,8]
    @test plan.origins[2]==(0.0,1e-3)
    @test plan.parts[1].ports[end].wall===:north
    @test plan.parts[2].ports[end].wall===:south
    grid=CellGrid(4e-3,6e-3,16,24)
    stack=PlanarStackup([PlanarLayer(1,1,0.5e-3),PlanarLayer(1,1,0.5e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,16,24)
    sheet.mask[:,7:9].=true;sheet.mask[:,16:18].=true
    sheet.connect_west[[7,8,9,16,17,18]].=true
    sheet.connect_east[[7,8,9,16,17,18]].=true
    ports=[PlanarPort(1,:west,7:9,50),PlanarPort(1,:west,16:18,50),
        PlanarPort(1,:east,7:9,50),PlanarPort(1,:east,16:18,50)]
    mult=build_planar_problem(stack,grid,[sheet],ports)
    divided=planar_subdivide(mult,:x,8)
    @test divided.terminals==[[(1,0),(2,0),(5,0),(6,0)],
        [(3,0),(4,0),(5,0),(6,0)]]
    @test all(p -> length(p.ports)==4,divided.parts)
    @test all(isfinite,solve_planar_subdivision(divided,2e9;mx=24,my=48).s)
end
