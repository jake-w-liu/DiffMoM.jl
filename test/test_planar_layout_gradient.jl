using DiffMoM, Test, LinearAlgebra

@testset "layout bulk conductivity participates in gradient forward solve" begin
    grid=CellGrid(.003,.002,6,4);height=20e-6
    stack=PlanarStackup([PlanarLayer(1.,1.,.0005),PlanarLayer(1.,1.,height),
        PlanarLayer(1.,1.,.0005)],TERM_GND,TERM_GND,grid.a,grid.b)
    volume=vol_level(2,6,4);volume.mask[:,2:3].=true
    volume.connect_west[2:3].=true;volume.connect_east[2:3].=true
    prob=build_planar_problem(stack,grid,SheetLevel[],
        [PlanarPort(1,:volume_west,2:3,50.),PlanarPort(1,:volume_east,2:3,50.)];vols=[volume])
    objective(Y)=real(-inv(Y[1,2]))
    function pullback(Y)
        G=zeros(ComplexF64,size(Y));G[1,2]=inv(Y[1,2]^2);G
    end
    calls=Ref(0)
    model(f)=(calls[]+=1;1e4*f/1e6)
    param=[PlanarParam(2,:thickness,:re)]
    for contraction in (nothing,Matrix{Float64}(I,2,2))
        layout=PlanarLayout(prob,PlanarShape[],Matrix{Int32}[],String[],Any[],Any[],
            prob,contraction,NamedTuple[],Any[model])
        for frequency in (1e6,2e6)
            expected=15e6/frequency
            result=solve_planar(layout,frequency;mx=24,my=20)
            J,g=planar_objective_gradient(layout,frequency,objective;
                params=param,gY=pullback,mx=24,my=20)
            @test J≈objective(result.y) rtol=1e-10
            @test J≈expected rtol=1e-6
            @test g[1]≈-expected/height rtol=1e-6
        end
        previous=calls[]
        J,g=planar_objective_gradient(layout,1e6,objective;volume_sigma=2e4,
            params=param,gY=pullback,mx=24,my=20)
        @test calls[]==previous
        @test J≈7.5 rtol=1e-6
        @test g[1]≈-7.5/height rtol=1e-6
        @test_throws ArgumentError planar_objective_gradient(layout,1e6,objective;
            params=param,gY=pullback,max_bytes=1)
        @test calls[]==previous
    end
end
