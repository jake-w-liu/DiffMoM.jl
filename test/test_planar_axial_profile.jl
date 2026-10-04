using DiffMoM,Test,LinearAlgebra

@testset "planar volume: independently resolved skin current" begin
    G=DiffMoM
    f=10e9;omega=2pi*f;sigma=5.8e7;t=8.6e-6;kc2=(pi/.001)^2
    # Independent decaying-wave Maxwell solution in a conductive slab
    # surrounded by air, driven by a unit sheet current on the lower face.
    q0=kc2-(omega/G._C0)^2;ga=sqrt(complex(q0))
    ya=ga/(1im*omega*G._MU0)
    gamma=sqrt(q0+1im*omega*G._MU0*sigma)
    ym=gamma/(1im*omega*G._MU0);decay=exp(-gamma*t)
    lower=inv(ym+ya+(ya-ym)*(ym-ya)/(ym+ya)*decay^2)
    upper=decay*(ym-ya)/(ym+ya)*lower
    errors=Float64[]
    for N in (16,64,128)
        h=t/N
        layers=[PlanarLayer(1.,1.,1e-5);[PlanarLayer(1.,1.,h) for _ in 1:N];
            PlanarLayer(1.,1.,1e-5)]
        stack=PlanarStackup(layers,TERM_SPACE,TERM_SPACE,.001,.001)
        casc=planar_mode_cascade(stack,omega,kc2,TE_POL)
        states=Dict(j=>G._vol_layer_state(stack,casc,omega,kc2,j,TE_POL) for j in 2:N+1)
        matrix=ComplexF64[G._vol_vol_kern(i+1,j+1,casc,states[i+1],states[j+1],1.0)+
            (i==j ? inv(sigma*h) : 0) for i in 1:N,j in 1:N]
        impressed=ComplexF64[states[i+1].wbar*(planar_modal_voltage(casc,i+1,1)+
            planar_modal_voltage(casc,i,1)) for i in 1:N]
        current=matrix\impressed
        reference=ComplexF64[sigma/gamma*(lower*(exp(-gamma*(i-1)*h)-exp(-gamma*i*h))+
            upper*(exp(-gamma*(t-i*h))-exp(-gamma*(t-(i-1)*h)))) for i in 1:N]
        push!(errors,norm(current-reference)/norm(reference))
        @test matrix≈transpose(matrix) rtol=1e-12
        @test sum(current)≈sum(reference) rtol=1e-4
        @test all(isfinite,current)
    end
    @test errors[2]<errors[1]/10
    @test errors[3]<errors[2]/3.9
    @test errors[end]<1e-3
end

@testset "planar axial: geometry material and terminal contracts" begin
    G=DiffMoM
    grid=CellGrid(2e-3,3e-3,3,3)
    stack=PlanarStackup([PlanarLayer(2-.03im,1.,3e-6;epsr_z=3.1),
        PlanarLayer(4.,1.2,.2e-3;mur_z=1.4)],TERM_GND,TERM_SPACE,grid.a,grid.b)
    sheet=sheet_level(1,3,3);sheet.mask.=true
    sheet.connect_west.=true;sheet.connect_east.=true
    volume=vol_level(1,3,3);volume.mask.=true
    via=via_level(2,3,3);via.tap[2,2]=true
    ports=[PlanarPort(1,:west,1:3,50.),PlanarPort(1,:east,1:3,75.),
        PlanarPort(1,:via,5:5,50.;polarity=-1)]
    prob=build_planar_problem(stack,grid,[sheet],ports;vols=[volume],vias=[via])
    refinement=planar_refine_axial(prob,[4,3])
    fine=refinement.problem
    @test length(fine.stack.layers)==7
    @test fine.sheets[1].interface==4
    @test planar_interfaces(fine.stack)[5]≈planar_interfaces(stack)[2]
    @test last(planar_interfaces(fine.stack))≈last(planar_interfaces(stack))
    @test refinement.layer_parent==[1,1,1,1,2,2,2]
    @test refinement.volume_parent==fill(1,4)
    @test refinement.via_parent==fill(1,3)
    @test all(l.epsr_z==stack.layers[1].epsr_z for l in fine.stack.layers[1:4])
    @test all(l.mur_z==stack.layers[2].mur_z for l in fine.stack.layers[5:7])
    @test all(v.mask==volume.mask for v in fine.vols)
    @test all(v.uni[2,2] for v in fine.vias[2:3])
    @test !fine.vias[1].uni[2,2] && fine.vias[1].tap[2,2]
    @test fine.sheets[1].mask!==sheet.mask
    @test fine.vols[1].mask!==volume.mask
    @test size(refinement.contraction)==(5,3)
    @test refinement.contraction[1:2,1:2]==Matrix{Float64}(I,2,2)
    @test refinement.contraction[3:5,3]==fill(1/3,3)
    @test all(p.polarity==-1 for p in fine.ports[3:5])
    @test all(sum(refinement.contraction;dims=1).≈1.)
    @test_throws ArgumentError planar_refine_axial(prob,0)
    @test_throws DimensionMismatch planar_refine_axial(prob,[1])
    @test_throws ArgumentError planar_refine_axial(prob,[typemax(Int),1])
    @test_throws ArgumentError planar_refine_axial(prob,[4,3];max_bytes=1)
    # Identity subdivision changes no conductor space or port response.
    identity=planar_refine_axial(prob,[1,1])
    raw=solve_planar(prob,3e9;mx=12,my=12,via_sigma=[1e4],volume_sigma=[2e4])
    same=solve_planar_axial(identity,3e9;mx=12,my=12,via_sigma=[1e4],volume_sigma=[2e4])
    @test same.y≈raw.y rtol=1e-12
    @test same.currents≈raw.currents rtol=1e-12
    @test_throws ArgumentError solve_planar_axial(refinement,3e9;max_bytes=1)
    @test_throws DimensionMismatch solve_planar_axial(refinement,3e9;via_sigma=[1.,2.])
    refined=solve_planar_axial(refinement,3e9;mx=12,my=12,via_sigma=[1e4],volume_sigma=[2e4])
    C=refinement.contraction
    independent=solve_planar(refinement.problem,3e9;mx=12,my=12,
        via_sigma=fill(1e4,length(refinement.via_parent)),
        volume_sigma=fill(2e4,length(refinement.volume_parent)))
    @test refined.y≈transpose(C)*independent.y*C rtol=1e-12
    @test refined.currents≈independent.currents*C rtol=1e-12
    @test size(refined.raw.currents,2)==length(prob.ports)
    @test refined.raw isa PlanarSourceResult
    @test opnorm(refined.s)<=1+1e-10
    maps=planar_current_maps(refined;port=3)
    @test count(m->m.kind===:volume,maps)==4
    @test count(m->m.kind===:via,maps)==3
end

@testset "planar axial: independently preserved via resistance" begin
    grid=CellGrid(1e-3,1e-3,2,2);h=20e-6;sigma=1e4
    stack=PlanarStackup([PlanarLayer(1.,1.,h)],TERM_GND,TERM_GND,grid.a,grid.b)
    via=via_level(1,2,2);via.uni[1,1]=true;via.tap[1,1]=true
    prob=build_planar_problem(stack,grid,SheetLevel[],[PlanarPort(1,:via,1:1,50.)];vias=[via])
    refinement=planar_refine_axial(prob,4)
    result=solve_planar_axial(refinement,1e6;via_sigma=[sigma],mx=16,my=16)
    resistance=h/(sigma*grid.dx*grid.dy)
    @test real(inv(result.y[1,1]))≈resistance rtol=1e-5
    # Verify FFT equivalence at an RF frequency where the dynamically
    # induced field is resolved above low-frequency charge cancellation.
    rf=solve_planar_axial(refinement,1e9;via_sigma=[sigma],mx=16,my=16)
    fast=solve_planar_axial(refinement,1e9;method=:ufft,via_sigma=[sigma],
        mx=16,my=16,memory=30,rtol=1e-10)
    @test fast.y≈rf.y rtol=1e-9
    @test opnorm(result.s)<=1+1e-12
end
