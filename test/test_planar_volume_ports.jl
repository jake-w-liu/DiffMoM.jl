using DiffMoM,Test,LinearAlgebra

@testset "bulk volume wall sources independent conductor resistance" begin
    G=DiffMoM;a,b=3e-3,2e-3;h=20e-6;sigma=1e4
    grid=CellGrid(a,b,6,4)
    # Actual bulk conductor floats between air layers; there are no
    # zero-resistance face sheets that could shunt its finite bulk loss.
    stack=PlanarStackup([PlanarLayer(1.,1.,.4e-3),PlanarLayer(1.,1.,h),
        PlanarLayer(1.,1.,.6e-3)],TERM_GND,TERM_GND,a,b)
    volume=vol_level(2,6,4);rasterize_rect!(volume,grid,0,a,.5e-3,1.5e-3)
    volume.connect_west[2:3].=true;volume.connect_east[2:3].=true
    ports=[PlanarPort(1,:volume_west,2:3,50.),PlanarPort(1,:volume_east,2:3,50.)]
    prob=build_planar_problem(stack,grid,SheetLevel[],ports;vols=[volume])
    @test all(G._is_vol_kind,prob.basis.kind)
    @test count(==(1),prob.basis.port)==2
    @test count(==(2),prob.basis.port)==2
    @test all(b->G._planar_port_weight(prob.basis,b)==grid.dy,findall(>(0),prob.basis.port))
    @test G._planar_port_sign(ports[1])==1 && G._planar_port_sign(ports[2])==-1
    expected=a/(sigma*(1e-3)*h)
    values=ComplexF64[]
    for f in (1e6,1e7)
        result=solve_planar(prob,f;mx=24,my=20,volume_sigma=sigma)
        impedance=2/(result.y[1,1]-result.y[1,2])
        @test real(impedance)≈expected rtol=5e-6
        @test result.y≈transpose(result.y) rtol=1e-10
        @test opnorm(result.s)<=1+1e-12
        push!(values,impedance)
        maps=planar_current_maps(result;voltages=[1.,-1.])
        m=only(maps)
        @test m.kind===:volume
        # Power-conjugate port extraction must be independent of the
        # volume's 1/h-normalized axial basis profile.
        coeff=result.currents*[1.,-1.]
        current=sum(grid.dy*coeff[bb] for bb in findall(==(1),prob.basis.port))
        @test current≈(result.y*[1.,-1.])[1] rtol=1e-13
    end
    @test imag(values[2])≈10imag(values[1]) rtol=5e-4
    dense=solve_planar(prob,1e9;mx=24,my=20,volume_sigma=sigma)
    fft=solve_planar(prob,1e9;method=:ufft,mx=24,my=20,volume_sigma=sigma,memory=80,rtol=1e-10)
    @test fft.y≈dense.y rtol=1e-9
    @test maximum(fft.relative_residuals)<=1e-10
    # Rotate the physical conductor and ports; x/y current normalization
    # and sign contracts must produce the same differential impedance.
    rotated_grid=CellGrid(b,a,4,6)
    rotated_stack=PlanarStackup(stack.layers,stack.bottom,stack.top,b,a)
    rotated=vol_level(2,4,6);rasterize_rect!(rotated,rotated_grid,.5e-3,1.5e-3,0,a)
    rotated.connect_south[2:3].=true;rotated.connect_north[2:3].=true
    rp=build_planar_problem(rotated_stack,rotated_grid,SheetLevel[],
        [PlanarPort(1,:volume_south,2:3,50.),PlanarPort(1,:volume_north,2:3,50.)];vols=[rotated])
    rr=solve_planar(rp,1e7;mx=20,my=24,volume_sigma=sigma)
    @test 2/(rr.y[1,1]-rr.y[1,2])≈values[2] rtol=1e-10
end
