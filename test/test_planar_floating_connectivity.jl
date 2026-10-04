using DiffMoM,Test,StaticArrays

@testset "floating source cut preserves independent declared conductor nets" begin
    G=DiffMoM
    for rotated in (false,true)
        grid=rotated ? CellGrid(.003,.004,6,8) : CellGrid(.004,.003,8,6)
        stack=PlanarStackup([PlanarLayer(1.,1.,.0005),PlanarLayer(1.,1.,.0005)],
            TERM_GND,TERM_GND,grid.a,grid.b)
        sheet=sheet_level(1,grid.nx,grid.ny)
        if rotated
            sheet.mask[3:4,2:3].=true;sheet.mask[3:4,6:7].=true
            reference=PlanarPort(1,:terminal_y,3,3:4,50.;metal_side=:negative)
            signal=PlanarPort(1,:terminal_y,5,3:4,50.;metal_side=:positive)
        else
            sheet.mask[2:3,3:4].=true;sheet.mask[6:7,3:4].=true
            reference=PlanarPort(1,:terminal_x,3,3:4,50.;metal_side=:negative)
            signal=PlanarPort(1,:terminal_x,5,3:4,50.;metal_side=:positive)
        end
        ports=PlanarPort[];basis=build_planar_basis(grid,[sheet],ports)
        bare=PlanarProblem(stack,grid,[sheet],ports,basis)
        model=planar_floating_bridge(bare,signal,reference;source_edge=4)
        prob=model.problem
        vertices(a,b)=SVector{2,Float64}[rotated ? SVector(y,x) : SVector(x,y) for
            (x,y) in ((a,.001),(b,.001),(b,.002),(a,.002))]
        polys=[PlanarShapePolygon("reference",1,"pec","G",vertices(.0005,.0015)),
            PlanarShapePolygon("signal",1,"pec","S",vertices(.0025,.0035))]
        shape=PlanarShape("pair",polys,PlanarShapeVia[],PlanarPin[],Dict{String,Float64}())
        bridge=merge(model.bridge,(;kind=:floating_bridge,port=1,metal="pec"))
        layout=PlanarLayout(prob,[shape],[zeros(Int32,grid.nx,grid.ny)],["pec"],Any[0.],
            Any[],prob,reshape([1.],1,1),NamedTuple[bridge])
        ordinary=planar_connectivity(prob)
        @test ordinary.component_count==1
        audited=planar_connectivity(layout)
        @test audited.component_count==2
        @test length(only(audited.port_components))==2
        @test isempty(audited.grounded_components)
        @test isempty(audited.shorted_nets) && isempty(audited.open_nets)
        @test only(audited.net_components["S"])!=only(audited.net_components["G"])
        @test_throws ArgumentError planar_connectivity(layout;max_bytes=1)
        # A physical conductor path around the source must still be found
        # as a passive short, regardless of the declared source metadata.
        if rotated;prob.sheets[1].mask[2,2:7].=true
        else;prob.sheets[1].mask[2:7,2].=true
        end
        shorted=planar_connectivity(layout)
        @test shorted.component_count==1
        @test shorted.shorted_nets==[["G","S"]]
    end
end
