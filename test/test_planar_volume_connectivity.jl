using DiffMoM,Test

@testset "volume wall driven versus galvanic connectivity" begin
    for walls in (WALL_PEC,WALL_PMC)
        grid=CellGrid(.004,.004,4,4;walls)
        stack=PlanarStackup([PlanarLayer(1.,1.,.001) for _ in 1:3],TERM_GND,TERM_GND,grid.a,grid.b)
        volume=vol_level(2,4,4);volume.mask[:,2:3].=true
        volume.connect_west[2:3].=true;volume.connect_east[2:3].=true
        both=[PlanarPort(1,:volume_west,2:3,50.),PlanarPort(1,:volume_east,2:3,50.)]
        driven=build_planar_problem(stack,grid,SheetLevel[],both;vols=[volume])
        check=planar_connectivity(driven)
        @test check.component_count==1
        @test isempty(check.grounded_components)
        @test check.port_components==[[1],[1]]
        # The undriven west row3 is a real PEC return, not another port.
        some=[PlanarPort(1,:volume_west,2:2,50.),PlanarPort(1,:volume_east,2:3,50.)]
        grounded=planar_connectivity(build_planar_problem(stack,grid,SheetLevel[],some;vols=[volume]))
        @test grounded.grounded_components==(walls==WALL_PEC ? [1] : Int[])
        @test_throws ArgumentError planar_connectivity(driven;max_bytes=1)
    end
end
