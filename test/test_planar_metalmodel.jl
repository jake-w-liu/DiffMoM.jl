using DiffMoM,Test,LinearAlgebra,Random

# Independent conductor Maxwell BVP by RK4 propagation of two fundamental
# fields, then imposing H(top)=Jtop and H(bottom)=-Jbottom.
function _metal_bvp_faces(f,sigma,t,mur;n=4000)
    omega=2pi*f
    function propagate(E,H)
        h=t/n
        rhs(E,H)=(-1im*omega*DiffMoM._MU0*mur*H,-sigma*E)
        for _ in 1:n
            a,b=rhs(E,H)
            c,d=rhs(E+h*a/2,H+h*b/2)
            e,g=rhs(E+h*c/2,H+h*d/2)
            k,l=rhs(E+h*e,H+h*g)
            E+=h*(a+2c+2*e+k)/6
            H+=h*(b+2d+2g+l)/6
        end
        return E,H
    end
    from_e=propagate(1.0+0im,0.0im)
    from_h=propagate(0.0im,1.0+0im)
    Z=zeros(ComplexF64,2,2)
    for q in 1:2
        jt,jb=q==1 ? (1.,0.) : (0.,1.)
        top=(-jb-from_h[2]*jt)/from_e[2]
        bottom=from_e[1]*top+from_h[1]*jt
        Z[:,q]=[top,bottom]
    end
    return Z
end

@testset "planar metal: two-face Maxwell conductor" begin
    G=DiffMoM
    sigma=5.8e7
    for f in (1e8,1e10),t in (1e-8,1e-6,2e-6)
        Z=G._planar_conductor_face_zs(f,sigma,t)
        reference=_metal_bvp_faces(f,sigma,t,1.)
        @test Z≈reference rtol=2e-9
        @test Z==transpose(Z)
        @test minimum(eigvals(Hermitian(real.(Z)))) >= -1e-13
    end
    t=1e-6
    low=G._planar_conductor_face_zs(1.,sigma,t)
    @test real(low[1,1])≈1/(sigma*t) rtol=1e-12
    @test real(low[1,2])≈1/(sigma*t) rtol=1e-12
    @test imag(low[1,1])≈2pi*G._MU0*t/3 rtol=1e-12
    effective=(low[1,1]+low[1,2])/2
    @test imag(effective)≈2pi*G._MU0*t/12 rtol=1e-12
    skin=planar_surface_zs(10e9,sigma)
    thick=G._planar_conductor_face_zs(10e9,sigma,1e-3)
    @test thick[1,1]≈skin rtol=1e-12
    @test thick[1,2]==0
    @test all(isfinite,thick)
    rough=HammerstadRoughness(.5e-6)
    smooth=G._planar_conductor_face_zs(10e9,sigma,t)
    corrected=G._planar_conductor_face_zs(10e9,sigma,t;roughness=rough)
    factor=roughness_factor(rough,10e9,sigma)
    @test corrected≈factor*smooth rtol=1e-12
    @test_throws ArgumentError G.planar_two_sheet_zs(0.,sigma,t)
    @test_throws ArgumentError G.PlanarConductorLayer(-sigma,t)
    pair=G.planar_two_sheet_zs(1.,sigma,t)
    @test pair[1,2]==pair[2,1]==0
    @test pair[1,1]==pair[2,2]
    @test real(pair[1,1])/2≈1/(sigma*t) rtol=1e-12
    @test imag(pair[1,1])≈2pi*G._MU0*t/6 rtol=1e-12
end

@testset "planar metal: plating cascade and limits" begin
    G=DiffMoM
    f=10e9
    layers=[G.PlanarConductorLayer(4.1e7,.05e-6),
        G.PlanarConductorLayer(1.4e7,3e-6)]
    sigma_cu=5.8e7
    load=planar_surface_zs(f,sigma_cu)
    M=Matrix{ComplexF64}(I,2,2)
    for layer in layers
        gamma=sqrt(1im*2pi*f*G._MU0*layer.mur*layer.sigma)
        zc=1im*2pi*f*G._MU0*layer.mur/gamma
        x=gamma*layer.thickness
        section=ComplexF64[cosh(x) zc*sinh(x);sinh(x)/zc cosh(x)]
        M=M*section
    end
    reference=(M[1,1]*load+M[1,2])/(M[2,1]*load+M[2,2])
    @test G.planar_layered_surface_zs(f,layers;substrate_sigma=sigma_cu)≈reference rtol=1e-12
    sigma=4.1e7;t=2e-6
    for terminal in (:open,:pec,3.0+2im)
        whole=G.planar_layered_surface_zs(f,[G.PlanarConductorLayer(sigma,t)];load=terminal)
        split=G.planar_layered_surface_zs(f,[G.PlanarConductorLayer(sigma,t/2),
            G.PlanarConductorLayer(sigma,t/2)];load=terminal)
        @test whole≈split rtol=1e-12
    end
    tiny=G.planar_layered_surface_zs(f,[G.PlanarConductorLayer(sigma,1e-30)];substrate_sigma=sigma_cu)
    @test tiny≈load rtol=1e-12
    thin=G.planar_layered_surface_zs(1.,[G.PlanarConductorLayer(sigma,1e-10)])
    @test real(thin)≈1/(sigma*1e-10) rtol=1e-12
    @test G.planar_layered_surface_zs(f,[G.PlanarConductorLayer(sigma,1e-3)])≈
        planar_surface_zs(f,sigma) rtol=1e-12
    @test_throws ArgumentError G.planar_layered_surface_zs(f,G.PlanarConductorLayer[])
    @test_throws ArgumentError G.planar_layered_surface_zs(f,layers;substrate_sigma=sigma_cu,load=:pec)
end

@testset "planar metal: coupled sheets assemble and solve" begin
    G=DiffMoM
    grid=CellGrid(4e-3,3e-3,5,4)
    stack=PlanarStackup([PlanarLayer(1.,1.,.4e-3),
        PlanarLayer(1.,1.,2e-6),PlanarLayer(1.,1.,.4e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheets=[sheet_level(1,5,4),sheet_level(2,5,4)]
    for sh in sheets
        sh.mask.=true
        sh.connect_west.=true;sh.connect_east.=true
    end
    # Both faces have identical tangential bases, so their local metal
    # matrix must equal the Kronecker product of the face Z and one Gram.
    ports=[PlanarPort(1,:west,2:3,50.),PlanarPort(1,:east,2:3,50.)]
    prob=build_planar_problem(stack,grid,sheets,ports)
    b1=build_planar_basis(grid,[sheets[1]],PlanarPort[])
    n=planar_basis_count(b1)
    gram=zeros(ComplexF64,n,n)
    G._add_gram!(gram,b1,grid,1.)
    face=G.planar_two_sheet_zs(8e9,5.8e7,2e-6)
    localZ=zeros(ComplexF64,2n,2n)
    G._add_gram!(localZ,prob.basis,grid,0.,face)
    @test localZ≈kron(face,gram) rtol=1e-12
    dense=solve_planar(prob,8e9;sheet_coupling_zs=face,mx=14,my=13)
    fast=G.solve_planar_ufft(prob,8e9;sheet_coupling_zs=face,mx=14,my=13,
        memory=100,rtol=1e-9)
    @test fast.s≈dense.s rtol=2e-8
    @test opnorm(dense.s)<=1+1e-10
    @test dense.s≈transpose(dense.s) rtol=1e-10
    @test_throws ArgumentError G._add_gram!(copy(localZ),prob.basis,grid,0.,
        ComplexF64[1 2;3 1])
    @test_throws DimensionMismatch G._add_gram!(copy(localZ),prob.basis,grid,0.,zeros(3,2))
end

@testset "planar metal: physical line resistance" begin
    a,b=4e-3,3e-3;nx,ny=16,12;t=2e-6;sigma=1e7;f=1e6
    grid=CellGrid(a,b,nx,ny)
    stack=PlanarStackup([PlanarLayer(1.,1.,.4e-3),PlanarLayer(1.,1.,t),
        PlanarLayer(1.,1.,.4e-3)],TERM_GND,TERM_GND,a,b)
    sheets=[sheet_level(1,nx,ny),sheet_level(2,nx,ny)]
    for sh in sheets
        sh.mask[:,5:8].=true;sh.connect_west[5:8].=true;sh.connect_east[5:8].=true
    end
    ports=[PlanarPort(lv,wall,5:8,50.) for lv in 1:2 for wall in (:west,:east)]
    prob=build_planar_problem(stack,grid,sheets,ports)
    r=solve_planar(prob,f;mx=64,my=64,sheet_coupling_zs=planar_two_sheet_zs(f,sigma,t))
    # Independent equipotential-face terminal contraction and Ohm's law.
    P=ComplexF64[1 0;0 1;1 0;0 1];Y=transpose(P)*r.y*P
    series=-inv(Y[1,2]);want=a/(sigma*t*(4grid.dy))
    @test real(series)≈want rtol=1e-4
    S=planar_y_to_s(Y,[50.,50.])
    @test opnorm(S)<=1+1e-12
    @test S≈transpose(S) rtol=1e-12
    # Symmetric plated film: each physical face carries half of every
    # material's allocated thickness. Both half stacks sum to t.
    half=[PlanarConductorLayer(4.1e7,.025e-6),
        PlanarConductorLayer(1.4e7,.2e-6),PlanarConductorLayer(5.8e7,.775e-6)]
    zp=planar_layered_surface_zs(f,half)
    rs=solve_planar(prob,f;mx=64,my=64,surface_zs=[zp,zp])
    Yp=transpose(P)*rs.y*P
    expected=a/(4grid.dy*2sum(l.sigma*l.thickness for l in half))
    @test real(-inv(Yp[1,2]))≈expected rtol=1e-3
    @test opnorm(planar_y_to_s(Yp,[50.,50.]))<=1+1e-12
end
