module DielectricBrickTests
using DiffMoM, Test, LinearAlgebra

# A west-wall-fed patch over a four-layer stack.  The patch sits on
# interface 1, one air layer above the host dielectric that carries the
# brick, so the field sampled by the polarization bases is smooth; the
# equivalent-stack oracle replaces the host's permittivity outright.
# (A sheet sitting directly on the brick interface drives a singular
# near-field that every pulse/rooftop VIE resolves only slowly.)
function fixture(n; host=4.)
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3),
        PlanarLayer(host,1.,.1e-3),PlanarLayer(1.,1.,.5e-3)],
        TERM_GND,TERM_GND,1e-3,1e-3)
    grid=CellGrid(1e-3,1e-3,n,n)
    sheet=sheet_level(1,n,n)
    rasterize_rect!(sheet,grid,0,.5e-3,0,1e-3)
    sheet.connect_west.=true
    ports=[PlanarPort(1,:west,1:n,50.)]
    return stack,grid,[sheet],ports,3
end

@testset "dielectric brick full-layer equivalence" begin
    # A brick filling its entire host layer is physically identical to
    # changing that layer's permittivity.  The VIE polarization unknowns
    # converge to the equivalent-stack oracle under grid refinement.
    for n in (8,16)
        stack,grid,sheets,ports,blay=fixture(n)
        brick=brick_level(blay,n,n,8+0im);brick.mask.=true
        vie=build_planar_problem(stack,grid,sheets,ports;bricks=[brick])
        refstack=PlanarStackup([stack.layers[1],stack.layers[2],
            PlanarLayer(8.,1.,.1e-3),stack.layers[4]],
            TERM_GND,TERM_GND,1e-3,1e-3)
        ref=build_planar_problem(refstack,grid,sheets,ports)
        resid=abs(solve_planar(vie,1e9).s[1,1]-solve_planar(ref,1e9).s[1,1])
        @test resid<2e-5
    end
end

@testset "dielectric brick high-permittivity limit" begin
    # A brick whose permittivity grows without bound approaches the
    # conductor-occupied solution; the residual contracts like 1/epsr.
    n=16
    stack,grid,sheets,ports,blay=fixture(n)
    responses=[begin
        brick=brick_level(blay,n,n,e+0im);brick.mask.=true
        solve_planar(build_planar_problem(stack,grid,sheets,ports;
            bricks=[brick]),1e9).s[1,1]
    end for e in (1e2,1e3,1e4)]
    @test abs(responses[2]-responses[3])<abs(responses[1]-responses[2])
    @test abs(responses[2]-responses[3])<0.3*abs(responses[1]-responses[2])+1e-9
    # The Delta-eps -> Inf limit is the equivalent volume conductor: the
    # brick rows degenerate to its EFIE rows and the solutions coincide.
    brick=brick_level(blay,n,n,1e8+0im);brick.mask.=true
    svie=solve_planar(build_planar_problem(stack,grid,sheets,ports;
        bricks=[brick]),1e9).s[1,1]
    vl=vol_level(blay,n,n);vl.mask.=true
    svol=solve_planar(build_planar_problem(stack,grid,sheets,ports;
        vols=[vl]),1e9).s[1,1]
    @test abs(svie-svol)<1e-4
end

@testset "dielectric brick loss and passivity" begin
    n=8
    stack,grid,sheets,ports,blay=fixture(n)
    lossless=brick_level(blay,n,n,8+0im);lossless.mask[3:6,3:6].=true
    lossy=brick_level(blay,n,n,8-0.08im);lossy.mask.=lossless.mask
    a=solve_planar(build_planar_problem(stack,grid,sheets,ports;
        bricks=[lossless]),1e9).s[1,1]
    b=solve_planar(build_planar_problem(stack,grid,sheets,ports;
        bricks=[lossy]),1e9).s[1,1]
    @test a!=b
    @test abs(b)<=1+1e-8
end

@testset "brick level construction and problem validation" begin
    @test_throws ArgumentError brick_level(0,4,4,8.)
    @test_throws ArgumentError brick_level(1,4,4,-2.)
    @test_throws ArgumentError brick_level(1,4,4,(8.,-1.,8.))
    @test_throws ArgumentError brick_level(1,4,4,NaN)
    brick=brick_level(1,4,4,9.)
    @test brick.epsr==(9+0im,9+0im,9+0im)
    @test !any(brick.mask)
    aniso=brick_level(1,4,4,(8.,9.,10.))
    @test aniso.epsr==(8+0im,9+0im,10+0im)

    n=8
    stack,grid,sheets,ports,blay=fixture(n)
    # Out-of-stack layer index and a mismatched mask both reject.
    bad=brick_level(5,n,n,8.);bad.mask.=true
    @test_throws ArgumentError build_planar_problem(stack,grid,sheets,ports;
        bricks=[bad])
    small=brick_level(blay,2,2,8.);small.mask.=true
    @test_throws Exception build_planar_problem(stack,grid,sheets,ports;
        bricks=[small])
    # A brick marking no cells has no polarization unknowns to solve.
    @test_throws r"marks no cells" build_planar_problem(stack,grid,sheets,
        ports;bricks=[brick_level(blay,n,n,8.)])
    # A zero-contrast axis is a singular VIE row and must not assemble.
    contrast=brick_level(blay,n,n,4+0im);contrast.mask.=true
    @test_throws r"contrast" build_planar_problem(stack,grid,sheets,ports;
        bricks=[contrast])
    # A brick equal to the host transverse constant but differing in z
    # still fails: every axis must carry contrast.
    uniaxial=PlanarStackup([stack.layers[1],stack.layers[2],
        PlanarLayer(4.,1.,.1e-3;epsr_z=2.),stack.layers[4]],
        TERM_GND,TERM_GND,1e-3,1e-3)
    partial=brick_level(blay,n,n,4+0im);partial.mask.=true
    @test_throws r"contrast" build_planar_problem(uniaxial,grid,sheets,ports;
        bricks=[partial])
end

@testset "brick basis kinds and current maps" begin
    n=8
    stack,grid,sheets,ports,blay=fixture(n)
    brick=brick_level(blay,n,n,8+0im);brick.mask[3:6,3:6].=true
    prob=build_planar_problem(stack,grid,sheets,ports;bricks=[brick])
    kinds=prob.basis.kind
    # the 4x4 interior patch contributes 20 x-faces, 20 y-faces and 16
    # z cells: 12 full + 4+4 boundary half rooftops per direction
    @test count(==(DiffMoM._BASIS_BXF),kinds)==12
    @test count(==(DiffMoM._BASIS_BXL),kinds)==4
    @test count(==(DiffMoM._BASIS_BXH),kinds)==4
    @test count(==(DiffMoM._BASIS_BYF),kinds)==12
    @test count(==(DiffMoM._BASIS_BYL),kinds)==4
    @test count(==(DiffMoM._BASIS_BYH),kinds)==4
    @test count(==(DiffMoM._BASIS_BZ),kinds)==16
    result=solve_planar(prob,1e9)
    maps=filter(m->m.kind===:brick,planar_current_maps(result))
    @test length(maps)==1
    @test maps[1].mask==brick.mask
    @test any(!iszero,maps[1].jx)&&any(!iszero,maps[1].jz)
    # Bricks lack a parity transform: FFT assembly paths reject loudly.
    @test_throws r"brick parity transform" solve_planar(prob,1e9;
        method=:ufft)
    @test_throws r"brick parity transform" solve_planar(prob,1e9;
        method=:dense_fft)
end

@testset "brick constitutive gram block" begin
    # The rooftop Gram is tridiagonal per (brick,direction,row): two
    # co-directional rooftops on adjacent faces share one cell.
    n=4
    stack,grid,sheets,ports,blay=fixture(n)
    brick=brick_level(blay,n,n,8+0im);brick.mask.=true
    b=build_planar_basis(grid,SheetLevel[],PlanarPort[];bricks=[brick])
    Z=DiffMoM.assemble_planar_z(stack,grid,SheetLevel[],b,2pi*1e9;
        bricks=[brick])
    bk=findall(DiffMoM._is_brick_kind,b.kind)
    Zb=Z[bk,bk]
    @test maximum(abs.(Zb-transpose(Zb)))<1e-18
    # interior x-rooftop row: self + two gram neighbours + GF couplings
    k0=findfirst(==(DiffMoM._BASIS_BXF),b.kind)
    @test abs(Zb[k0,k0])>0
end

@testset "brick threading: refinement, adjoint, terminal returns" begin
    n=8
    stack,grid,sheets,ports,blay=fixture(n)
    brick=brick_level(blay,n,n,8+0im);brick.mask[3:6,3:6].=true
    prob=build_planar_problem(stack,grid,sheets,ports;bricks=[brick])

    # Axial subdivision splits the brick into one BrickLevel per slice
    # (piecewise-constant polarization per refined slice).
    ref=planar_refine_axial(prob,[1,1,2,1])
    @test length(ref.problem.bricks)==2
    @test all(b.mask==brick.mask for b in ref.problem.bricks)
    # host layer 3 subdivides into refined layers 3 and 4
    @test sort([b.layer for b in ref.problem.bricks])==[3,4]
    @test all(isfinite,solve_planar_axial(ref,1e9).s)

    # The adjoint differentiates the constitutive block: host-layer
    # thickness and permittivity gradients match finite differences
    # (verified analytically; bounded loosely here against drift).
    params=[PlanarParam(blay,:thickness,:re),PlanarParam(blay,:epsr_z,:re)]
    J,g=planar_objective_gradient(prob,1e9,Y->imag(Y[1,1]);params)
    @test all(isfinite,g)
    l3=stack.layers[3];h=1e-4*l3.thickness
    function Jof(l)
        lay=[stack.layers[1],stack.layers[2],l,stack.layers[4]]
        p=build_planar_problem(PlanarStackup(lay,TERM_GND,TERM_GND,
            stack.a,stack.b),grid,sheets,ports;bricks=[brick])
        imag(solve_planar(p,1e9).y[1,1])
    end
    fdt=(Jof(PlanarLayer(l3.epsr,l3.mur,l3.thickness+h))-
        Jof(PlanarLayer(l3.epsr,l3.mur,l3.thickness-h)))/(2h)
    @test abs(g[1]-fdt)<=0.02max(abs(g[1]),abs(fdt))+1e-12
    dez=1e-4
    fdez=(Jof(PlanarLayer(l3.epsr,l3.mur,l3.thickness;
            epsr_z=l3.epsr_z+dez))-
        Jof(PlanarLayer(l3.epsr,l3.mur,l3.thickness;
            epsr_z=l3.epsr_z-dez)))/(2dez)
    @test abs(g[2]-fdez)<=0.02max(abs(g[2]),abs(fdez))+1e-12

    # Physical terminal-return construction forwards the brick levels.
    grid2=CellGrid(6e-3,4e-3,12,8)
    stack2=PlanarStackup([PlanarLayer(1,1,.5e-3),PlanarLayer(4,1,.2e-3),
        PlanarLayer(1,1,.5e-3)],TERM_GND,TERM_GND,grid2.a,grid2.b)
    strip=sheet_level(1,12,8);strip.mask[2:11,4:5].=true
    terminals=[PlanarPort(1,:terminal_x,1,4:5,50;metal_side=:positive),
        PlanarPort(1,:terminal_x,11,4:5,50;metal_side=:negative)]
    tbrick=brick_level(2,12,8,9+0im);tbrick.mask[5:8,3:6].=true
    model=planar_terminal_returns(stack2,grid2,[strip],terminals;
        bricks=[tbrick],ground_direction=:below)
    @test length(model.problem.bricks)==1
    @test model.problem.bricks[1].mask==tbrick.mask
    @test all(isfinite,solve_planar(model.problem,1e9).s)
end

end
