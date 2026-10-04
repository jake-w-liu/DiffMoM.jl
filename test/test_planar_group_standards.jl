using DiffMoM,Test,LinearAlgebra

function _coupled_standard_problem(;rotated=false)
    a,b=2e-3,2e-3;nx=ny=8
    grid=CellGrid(a,b,nx,ny)
    stack=PlanarStackup([PlanarLayer(1.,1.,.2e-3),PlanarLayer(1.,1.,.2e-3)],TERM_GND,TERM_GND,a,b)
    signal=sheet_level(1,nx,ny)
    for t in (2,5),along in 1:8
        i,j=rotated ? (t,along) : (along,t)
        signal.mask[i,j]=true
    end
    low,high=rotated ? (:south,:north) : (:west,:east)
    firstwall,lastwall=rotated ? (signal.connect_south,signal.connect_north) :
        (signal.connect_west,signal.connect_east)
    firstwall[[2,5]].=true;lastwall[[2,5]].=true
    reference=sheet_level(1,nx,ny)
    for along in 1:8
        i,j=rotated ? (7,along) : (along,7)
        reference.mask[i,j]=true
    end
    ports=[PlanarPort(1,wall,t:t,50.) for t in (2,5) for wall in (low,high)]
    return build_planar_problem(stack,grid,[signal,reference],ports)
end

@testset "coupled sister standards preserve physical references" begin
    prob=_coupled_standard_problem()
    standards=planar_group_line_standards(prob;left=[1,3],right=[2,4])
    @test standards.ordering==[1,3,2,4]
    @test standards.length==prob.grid.a
    @test standards.line.grid.dx==standards.double_line.grid.dx
    @test count(standards.line.sheets[2].mask)==8
    @test count(standards.double_line.sheets[2].mask)==16
    @test count(standards.reflect.sheets[2].mask)==4
    @test !any(standards.double_line.sheets[2].connect_west)
    @test !any(standards.double_line.sheets[2].connect_east)
    @test planar_connectivity(standards.line).port_components==planar_connectivity(prob).port_components
    @test isempty(planar_connectivity(standards.line).grounded_components)
    a=solve_planar(standards.line,1e9;mx=48,my=32,surface_zs=.02)
    b=solve_planar(standards.double_line,1e9;mx=48,my=32,surface_zs=.02)
    @test opnorm(a.s)<=1+1e-10
    @test opnorm(b.s)<=1+1e-10
    @test a.s≈transpose(a.s) rtol=1e-11
    ordering=standards.ordering
    cal=planar_group_double_delay_calibrate(a.y[ordering,ordering],b.y[ordering,ordering];
        length=standards.length,tol=1e-10)
    deembedded=deembed_cocal_group(a.y[ordering,ordering],[1,2],cal.launch)
    deembedded=deembed_cocal_group(deembedded,[3,4],cal.launch)
    @test deembedded≈planar_abcd_to_y(cal.line) rtol=1e-10
    @test cal.residual<1e-10
    rotated=planar_group_line_standards(_coupled_standard_problem(rotated=true);left=[1,3],right=[2,4])
    @test rotated.double_line.grid.dy==standards.double_line.grid.dx
    @test rotated.double_line.sheets[2].mask≈transpose(standards.double_line.sheets[2].mask)
    @test_throws ArgumentError planar_group_line_standards(prob;left=[1,2],right=[3,4])
    @test_throws ArgumentError planar_group_line_standards(prob;left=[1,3],right=[1,4])
    @test_throws ArgumentError planar_group_line_standards(prob;left=[1,3],right=[2,4],max_bytes=1)
    one=build_planar_problem(prob.stack,prob.grid,[prob.sheets[1]],
        [PlanarPort(1,:west,2:2,50.),PlanarPort(1,:east,2:2,50.)])
    @test_throws ArgumentError planar_line_standards(one;max_bytes=1)
    @test_throws ArgumentError planar_line_standards(one;extra_cells=typemax(Int))
    @test_throws ArgumentError planar_line_standards(one;extra_cells=10^9,max_bytes=10000)
    prob.sheets[1].mask[4,2]=false
    @test_throws ArgumentError planar_group_line_standards(prob;left=[1,3],right=[2,4])
end
