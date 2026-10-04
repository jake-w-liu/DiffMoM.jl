using DiffMoM,Test,LinearAlgebra

function floating_bridge_fixture(;endpoints=false,rotated=false)
    grid=rotated ? CellGrid(3e-3,4e-3,6,8) : CellGrid(4e-3,3e-3,8,6)
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,grid.a,grid.b)
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
    ports=endpoints ? [reference,signal] : PlanarPort[]
    basis=build_planar_basis(grid,[sheet],ports)
    prob=PlanarProblem(stack,grid,[sheet],ports,basis)
    return (;prob,signal,reference)
end

@testset "multiple floating sources retain all impressed cuts" begin
    grid=CellGrid(4e-3,4e-3,8,8)
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,8,8);sheet.mask[2:7,2].=true;sheet.mask[2:7,5].=true
    bare=PlanarProblem(stack,grid,[sheet],PlanarPort[],build_planar_basis(grid,[sheet],PlanarPort[]))
    signal(i)=PlanarPort(1,:terminal_y,4,i:i,50.;metal_side=:positive)
    reference(i)=PlanarPort(1,:terminal_y,2,i:i,50.;metal_side=:negative)
    left=planar_floating_bridge(bare,signal(2),reference(2);source_edge=3)
    right=planar_floating_bridge(left.problem,signal(7),reference(7);source_edge=3)
    @test right.old_port_map==[1]
    @test right.port_index==2
    @test length(right.problem.ports)==2
    @test right.problem.ports[1]==left.problem.ports[1]
    @test right.problem.sheets[1].mask[2,3:4]==[true,true]
    @test right.problem.sheets[1].mask[7,3:4]==[true,true]
    r=solve_planar(right.problem,1e6;surface_zs=1.,mx=24,my=24)
    # Closed loop has16fullrooftops:16selfintegrals2/3 plus12shared
    # collinear overlap pairs1/3. Fourturncells carrytwoorthogonal
    # ramps, whose loss differs from a straight centerline16-square rule.
    @test real(2/(r.y[1,1]-r.y[1,2]))≈16*(2/3)+12*(1/3) rtol=2e-6
    @test r.y≈transpose(r.y) rtol=1e-12
    @test opnorm(r.s)<=1+1e-12
    bypass=PlanarProblem(stack,grid,deepcopy(left.problem.sheets),left.problem.ports,left.problem.basis)
    bypass.sheets[1].mask[4,3:4].=true
    @test_throws ArgumentError planar_floating_bridge(bypass,signal(7),reference(7);source_edge=3)
end

@testset "physical floating source bridge geometry and strict contracts" begin
    x=floating_bridge_fixture(;endpoints=true);original=copy(x.prob.sheets[1].mask)
    lowered=planar_floating_bridge(x.prob,x.signal,x.reference;source_edge=4)
    p=lowered.problem
    @test x.prob.sheets[1].mask==original
    @test lowered.old_port_map==[0,0]
    @test lowered.port_index==1
    @test only(p.ports).wall===:internal_x
    @test only(p.ports).edge==4
    @test only(p.ports).polarity==1
    @test Set(lowered.bridge.cells)==Set([(4,3),(4,4),(5,3),(5,4)])
    @test lowered.bridge.width==1e-3
    @test lowered.bridge.length==1e-3
    @test isempty(p.vias) && isempty(p.vols)
    @test all(!any(c) for c in (p.sheets[1].connect_west,p.sheets[1].connect_east,p.sheets[1].connect_south,p.sheets[1].connect_north))
    @test count(==(1),p.basis.port)==2
    @test all(==(DiffMoM._BASIS_X_FULL),p.basis.kind[findall(==(1),p.basis.port)])
    @test_throws ArgumentError planar_floating_bridge(x.prob,x.signal,x.reference;max_bytes=1)
    @test_throws ArgumentError planar_floating_bridge(x.prob,x.signal,x.reference;source_edge=2)
    @test_throws ArgumentError planar_floating_bridge(x.prob,x.signal,PlanarPort(1,:terminal_x,3,3:3,50.;metal_side=:negative))
    @test_throws ArgumentError planar_floating_bridge(x.prob,x.signal,PlanarPort(1,:terminal_x,3,3:4,50.;metal_side=:positive))
    @test_throws ArgumentError planar_floating_bridge(x.prob,x.signal,PlanarPort(1,:terminal_x,3,3:4,50.;metal_side=:negative,polarity=-1))
    # An occupied gap or pre-existing sheet return would passively bypass
    # the impressed differential source and must not be quietly lowered.
    overlap=floating_bridge_fixture();overlap.prob.sheets[1].mask[4,3]=true
    @test_throws ArgumentError planar_floating_bridge(overlap.prob,overlap.signal,overlap.reference)
    bypass=floating_bridge_fixture();bypass.prob.sheets[1].mask[2:7,2].=true
    @test_throws ArgumentError planar_floating_bridge(bypass.prob,bypass.signal,bypass.reference)
    bare=floating_bridge_fixture();b=planar_floating_bridge(bare.prob,bare.signal,bare.reference)
    @test isempty(b.old_port_map) && b.port_index==1
    @test b.problem.basis.kind==p.basis.kind
    swapped=planar_floating_bridge(bare.prob,bare.reference,bare.signal;source_edge=4)
    @test only(swapped.problem.ports).polarity==-1
end

@testset "floating pair charge continuity and physical electromagnetic response" begin
    x=floating_bridge_fixture();lowered=planar_floating_bridge(x.prob,x.signal,x.reference;source_edge=4);p=lowered.problem
    responses=ComplexF64[]
    for f in (1e6,1e7)
        result=solve_planar(p,f;mx=24,my=24)
        @test imag(result.y[1,1])>0
        @test abs(real(result.y[1,1]))<1e-10abs(imag(result.y[1,1]))
        @test abs(result.s[1,1])≈1 rtol=1e-12
        push!(responses,result.y[1,1])
        # Exact finite-volume divergence is the normal flux of each full
        # rooftop, an independent physical current/charge continuity oracle.
        flux=zeros(ComplexF64,p.grid.nx,p.grid.ny)
        for b in eachindex(p.basis.kind)
            k=p.basis.kind[b];i,j=p.basis.ei[b],p.basis.ej[b];current=p.basis.width[b]*result.currents[b,1]
            if k==DiffMoM._BASIS_X_FULL
                flux[i,j]+=current;flux[i+1,j]-=current
            elseif k==DiffMoM._BASIS_Y_FULL
                flux[i,j]+=current;flux[i,j+1]-=current
            else
                error("floating bridge unexpectedly acquired a return-to-ground or open-half current")
            end
        end
        charge=flux/(-2pi*im*f)
        @test abs(sum(charge))<1e-12sum(abs,charge)
        @test sum(charge[5:end,:])≈result.y[1,1]/(2pi*im*f) rtol=1e-12
        @test sum(charge[1:4,:])≈-result.y[1,1]/(2pi*im*f) rtol=1e-12
    end
    @test responses[2]≈10responses[1] rtol=2e-4
    d=solve_planar(p,1e9;mx=24,my=24,surface_zs=2.)
    u=solve_planar(p,1e9;method=:ufft,mx=24,my=24,surface_zs=2.,memory=80,rtol=1e-10)
    @test u.y≈d.y rtol=1e-9
    @test maximum(u.relative_residuals)<=1e-10
    @test abs(d.s[1,1])<1
    swapped=planar_floating_bridge(x.prob,x.reference,x.signal;source_edge=4)
    reverse=solve_planar(swapped.problem,1e9;mx=24,my=24,surface_zs=2.)
    @test reverse.y≈d.y rtol=1e-12
    @test reverse.currents≈-d.currents rtol=1e-12
    rotated=floating_bridge_fixture(;rotated=true);yr=planar_floating_bridge(rotated.prob,rotated.signal,rotated.reference;source_edge=4)
    rotation=solve_planar(yr.problem,1e9;mx=24,my=24,surface_zs=2.)
    @test rotation.y≈d.y rtol=1e-11
end
