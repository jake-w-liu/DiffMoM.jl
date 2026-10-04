using Test,LinearAlgebra,DiffMoM

function _terminal_test_problem(;polarity=1)
    grid=CellGrid(6e-3,4e-3,12,8)
    stack=PlanarStackup([PlanarLayer(2.5-0.02im,1,0.5e-3),
        PlanarLayer(1,1,1e-3)],TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(1,12,8)
    sheet.mask[2:5,3:6].=true;sheet.mask[8:11,3:6].=true
    ports=[PlanarPort(1,:terminal_x,5,3:6,50;metal_side=:negative,polarity=polarity),
        PlanarPort(1,:terminal_x,7,3:6,75;metal_side=:positive),
        PlanarPort(1,:terminal_y,2,2:5,50;metal_side=:positive),
        PlanarPort(1,:terminal_y,6,8:11,50;metal_side=:negative)]
    return build_planar_problem(stack,grid,[sheet],ports)
end

@testset "planar terminals: physical open-edge basis contract" begin
    prob=_terminal_test_problem()
    for (index,p) in enumerate(prob.ports)
        ids=findall(==(index),prob.basis.port)
        @test length(ids)==length(p.cells)
        if p.wall in (:terminal_x_lo,:terminal_x_hi)
            @test all(prob.basis.ei[ids].==p.edge)
            @test all(prob.basis.x0[ids].==p.edge*prob.grid.dx)
        else
            @test all(prob.basis.ej[ids].==p.edge)
            @test all(prob.basis.y0[ids].==p.edge*prob.grid.dy)
        end
    end
    @test DiffMoM._planar_port_sign(prob.ports[1])==-1
    @test DiffMoM._planar_port_sign(prob.ports[2])==1
    @test DiffMoM._planar_port_sign(prob.ports[3])==1
    @test DiffMoM._planar_port_sign(prob.ports[4])==-1
    @test_throws ArgumentError PlanarPort(1,:terminal_x,4,3:6,50;metal_side=:both)
    @test_throws ArgumentError build_planar_problem(prob.stack,prob.grid,prob.sheets,
        [PlanarPort(1,:terminal_x,4,3:6,50;metal_side=:negative)])
    @test_throws ArgumentError build_planar_problem(prob.stack,prob.grid,prob.sheets,
        [PlanarPort(1,:terminal_x,5,2:6,50;metal_side=:negative)])
    @test_throws ArgumentError build_planar_problem(prob.stack,prob.grid,prob.sheets,
        [prob.ports[1],prob.ports[1]])
end

@testset "planar terminals: network/current invariants" begin
    prob=_terminal_test_problem()
    dense=solve_planar(prob,3e9;mx=36,my=24,surface_zs=0.02+0.01im)
    fft=solve_planar(prob,3e9;method=:ufft,mx=36,my=24,
        surface_zs=0.02+0.01im,rtol=1e-10,maxiter=400,memory=100)
    @test dense.y≈transpose(dense.y) rtol=1e-10
    @test dense.s≈transpose(dense.s) rtol=1e-10
    @test opnorm(dense.s)<=1+1e-10
    @test fft.y≈dense.y rtol=1e-8
    @test fft.s≈dense.s rtol=1e-8
    flipped=solve_planar(_terminal_test_problem(polarity=-1),3e9;mx=36,my=24,
        surface_zs=0.02+0.01im)
    signs=Diagonal([-1.,1.,1.,1.])
    @test flipped.y≈signs*dense.y*signs rtol=1e-10
    @test flipped.s≈signs*dense.s*signs rtol=1e-10
    maps=planar_current_maps(dense;incident_waves=ComplexF64[1,0,0,0])
    @test all(isfinite,maps[1].jx)
    @test all(isfinite,maps[1].jy)
    @test all(iszero,maps[1].jx[.!prob.sheets[1].mask])
    @test all(iszero,maps[1].jy[.!prob.sheets[1].mask])
    # Independent matrix reaction with exact terminal trace/sign weights.
    B=zeros(ComplexF64,planar_basis_count(prob.basis),4)
    for b in eachindex(prob.basis.port)
        p=prob.basis.port[b];p==0 && continue
        B[b,p]=DiffMoM._planar_port_sign(prob.ports[p])*prob.basis.width[b]
    end
    @test dense.y≈-transpose(B)*(dense.z_mom\B) rtol=1e-10
end
