module AxialVolumeWallTests
using DiffMoM,Test,LinearAlgebra

function conductor(rotated;z0=50.,polarity=1,refplane=nothing)
    a,b=rotated ? (2e-3,3e-3) : (3e-3,2e-3)
    nx,ny=rotated ? (4,6) : (6,4)
    grid=CellGrid(a,b,nx,ny);h=20e-6
    stack=PlanarStackup([PlanarLayer(1.,1.,.4e-3),PlanarLayer(1.,1.,h),
        PlanarLayer(1.,1.,.6e-3)],TERM_GND,TERM_GND,a,b)
    volume=vol_level(2,nx,ny)
    if rotated
        rasterize_rect!(volume,grid,.5e-3,1.5e-3,0,b)
        volume.connect_south[2:3].=true;volume.connect_north[2:3].=true
    else
        rasterize_rect!(volume,grid,0,a,.5e-3,1.5e-3)
        volume.connect_west[2:3].=true;volume.connect_east[2:3].=true
    end
    walls=rotated ? (:volume_south,:volume_north) : (:volume_west,:volume_east)
    ports=[PlanarPort(1,wall,2:3,z0;polarity,refplane) for wall in walls]
    return build_planar_problem(stack,grid,SheetLevel[],ports;vols=[volume])
end

@testset "axial volume wall ports preserve physical conductor resistance" begin
    sigma=1e4;height=20e-6
    # Independent rectangular-conductor DC law; lateral currents from
    # every depth slice add under one common terminal voltage.
    reference=3e-3/(sigma*1e-3*height)
    for rotated in (false,true),N in (1,2,4)
        original=conductor(rotated);volume=only(original.vols)
        fine=planar_refine_axial(original,[1,N,1])
        walls=rotated ? (:volume_south,:volume_north) : (:volume_west,:volume_east)
        manual_stack=PlanarStackup([original.stack.layers[1];
            [PlanarLayer(1.,1.,height/N) for _ in 1:N];original.stack.layers[3]],
            TERM_GND,TERM_GND,original.grid.a,original.grid.b)
        manual_volumes=[VolLevel(i+1,copy(volume.mask),copy(volume.connect_west),
            copy(volume.connect_east),copy(volume.connect_south),copy(volume.connect_north)) for i in 1:N]
        manual_ports=[PlanarPort(i,wall,2:3,50.) for wall in walls for i in 1:N]
        C=zeros(2N,2);C[1:N,1].=1;C[N+1:2N,2].=1
        manual=build_planar_problem(manual_stack,original.grid,SheetLevel[],manual_ports;vols=manual_volumes)
        @test length(fine.problem.ports)==2N
        @test fine.contraction==C
        @test [p.level for p in fine.problem.ports]==[collect(1:N);collect(1:N)]
        mx,my=rotated ? (20,24) : (24,20)
        solved=solve_planar_axial(fine,1e6;mx,my,volume_sigma=[sigma])
        independent=solve_planar_contracted(manual,1e6,C;mx,my,z0=50.,volume_sigma=fill(sigma,N))
        @test real(2/(solved.y[1,1]-solved.y[1,2]))≈reference rtol=5e-6
        @test solved.y≈independent.y rtol=1e-12
        @test solved.currents≈independent.currents rtol=1e-12
        @test opnorm(solved.s)<=1+1e-12
        maps=planar_current_maps(solved;voltages=[1.,-1.])
        @test count(m->m.kind===:volume,maps)==N
        coefficients=solved.currents*[1.,-1.]
        width=rotated ? original.grid.dx : original.grid.dy
        current=sum(width*coefficients[b] for b in eachindex(coefficients)
            if fine.problem.basis.port[b] in 1:N)
        @test current≈(solved.y*[1.,-1.])[1] rtol=1e-12
        if N==4
            dense=solve_planar_axial(fine,1e9;mx,my,volume_sigma=[sigma])
            fast=solve_planar_axial(fine,1e9;method=:ufft,mx,my,volume_sigma=[sigma],memory=200,rtol=1e-10)
            @test fast.y≈dense.y rtol=1e-9
            @test maximum(fast.raw.relative_residuals)<=1e-10
        end
    end
end

@testset "axial volume wall ordinal and metadata contracts" begin
    provider=f->50+1im*f/1e9
    plane=PlanarReferencePlane(1e-6,50.,f->1im*f/1e8)
    for rotated in (false,true)
        original=conductor(rotated;z0=provider,polarity=-1,refplane=plane)
        fine=planar_refine_axial(original,[1,4,1])
        @test all(p.polarity==-1 for p in fine.problem.ports)
        @test all(p.z0===original.ports[1].z0 for p in fine.problem.ports)
        @test all(p.refplane===plane for p in fine.problem.ports)
        @test all(p.cells==2:3 && p.edge==0 for p in fine.problem.ports)
        @test [p.wall for p in fine.problem.ports]==repeat([original.ports[1].wall,original.ports[2].wall];inner=4)
        identity=planar_refine_axial(original,[1,1,1])
        @test identity.problem.ports[1]===original.ports[1]
        @test identity.problem.ports[2]===original.ports[2]
        @test_throws ArgumentError planar_refine_axial(original,[1,4,1];max_bytes=1)
        @test_throws ArgumentError planar_refine_axial(original,[1,typemax(Int),1])
    end
    grid=CellGrid(1e-3,1e-3,2,2)
    stack=PlanarStackup([PlanarLayer(1.,1.,20e-6),PlanarLayer(1.,1.,30e-6)],TERM_GND,TERM_GND,grid.a,grid.b)
    first=vol_level(1,2,2);first.mask.=true
    second=vol_level(2,2,2);second.mask.=true
    second.connect_west.=true;second.connect_east.=true
    ports=[PlanarPort(2,:volume_west,1:2,50.),PlanarPort(2,:volume_east,1:2,50.)]
    original=build_planar_problem(stack,grid,SheetLevel[],ports;vols=[first,second])
    fine=planar_refine_axial(original,[2,3])
    @test fine.volume_parent==[1,1,2,2,2]
    @test [p.level for p in fine.problem.ports]==[3,4,5,3,4,5]
    @test size(fine.contraction)==(6,2)
    @test fine.contraction==[ones(3) zeros(3);zeros(3) ones(3)]
    @test all(fine.problem.basis.level[b] in 3:5 for b in eachindex(fine.problem.basis.port)
        if fine.problem.basis.port[b]>0)
end
end
