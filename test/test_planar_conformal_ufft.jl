using DiffMoM,Test,LinearAlgebra,Random

function conformal_sign_quadrature(v,values,kx,ky;n=48)
    beta=[j/sqrt(4j*j-1) for j in 1:n-1]
    quadrature=eigen(SymTridiagonal(zeros(n),beta))
    nodes=(quadrature.values.+1)./2;weights=quadrature.vectors[1,:].^2
    jacobian=abs(det(hcat(v[:,2]-v[:,1],v[:,3]-v[:,1])))
    value=0.0im
    for (i,u) in enumerate(nodes),(j,w) in enumerate(nodes)
        lambda=((1-u)*(1-w),u,(1-u)*w)
        x=sum(lambda[k]*v[1,k] for k in 1:3);y=sum(lambda[k]*v[2,k] for k in 1:3)
        affine=sum(lambda[k]*values[k] for k in 1:3)
        value+=weights[i]*weights[j]*jacobian*(1-u)*affine*cis(kx*x+ky*y)
    end
    value
end

function conformal_ufft_fixture(;walls=WALL_PEC,multiple=false)
    a,b=4e-3,3e-3;nx,ny=4,6
    xs=collect(range(0.,a;length=nx+1));ys=collect(range(b/3,2b/3;length=3))
    v=hcat(([x,y] for y in ys for x in xs)...);id(i,j)=i+1+(nx+1)*j
    triangles=hcat(([id(i,j),id(i+1,j),id(i,j+1)] for j in 0:1 for i in 0:nx-1)...,
        ([id(i+1,j+1),id(i,j+1),id(i+1,j)] for j in 0:1 for i in 0:nx-1)...)
    if multiple
        triangles=hcat(triangles,triangles);levels=vcat(fill(1,16),fill(2,16))
    else
        levels=ones(Int,16)
    end
    mesh=PlanarConformalMesh(v,triangles;interfaces=levels)
    stack=PlanarStackup([PlanarLayer(2-.003im,1.,.3e-3),PlanarLayer(3-.001im,1.,.7e-3),
        PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    ports=[PlanarConformalPort(1,:west,(b/3,2b/3)),PlanarConformalPort(1,:east,(b/3,2b/3))]
    if multiple
        append!(ports,[PlanarConformalPort(2,:west,(b/3,2b/3)),PlanarConformalPort(2,:east,(b/3,2b/3))])
    end
    return PlanarConformalProblem(stack,mesh,ports;sidewalls=walls),nx,ny
end

@testset "exact conformal FFT all modes half/full cross terms multilayers" begin
    rng=MersenneTwister(517)
    for walls in (WALL_PEC,WALL_PMC),multiple in (false,true)
        prob,nx,ny=conformal_ufft_fixture(;walls,multiple)
        zs=collect(range(.1,.3;length=length(prob.mesh.interfaces)))
        Z=assemble_planar_conformal_z(prob,2e9;mx=19,my=23,surface_zs=zs)
        A=planar_conformal_ufft_operator(prob,2e9;nx,ny,mx=19,my=23,surface_zs=zs)
        @test size(A.c_te)==(19*23,2,length(A.families))
        @test size(A.c_tm)==size(A.c_te)
        @test sizeof(A.c_te)+sizeof(A.c_tm)==sizeof(ComplexF64)*4*19*23*length(A.families)
        # Independent area quadrature verifies every Fourier sign, including
        # signs reconstructed by conjugacy. The Green kernels are lossy and
        # remain complex; only real geometric coefficients are paired.
        for (f,family) in enumerate(A.families)
            v=[family.shape[1]*A.grid.dx family.shape[3]*A.grid.dx family.shape[5]*A.grid.dx;
               family.shape[2]*A.grid.dy family.shape[4]*A.grid.dy family.shape[6]*A.grid.dy]
            xvalues=ntuple(i->v[1,i]-v[1,family.free],3)
            yvalues=ntuple(i->v[2,i]-v[2,family.free],3)
            for (m,n) in ((1,1),(2,3),(19,23)),(s,(sx,sy)) in enumerate(DiffMoM._PLANAR_CONFORMAL_SIGNS)
                t=m+A.modes.mx*(n-1);kx,ky=A.modes.kx[m],A.modes.ky[n]
                fx=conformal_sign_quadrature(v,xvalues,sx*kx,sy*ky)
                fy=conformal_sign_quadrature(v,yvalues,sx*kx,sy*ky)
                xt,yt=walls===WALL_PEC ? (sy,sx) : (sx,sy)
                @test DiffMoM._planar_conformal_coefficient(A.c_te,t,s,f)≈
                    (ky*xt*fx-kx*yt*fy)/(4im) rtol=1e-10 atol=1e-20
                @test DiffMoM._planar_conformal_coefficient(A.c_tm,t,s,f)≈
                    (kx*xt*fx+ky*yt*fy)/(4im) rtol=1e-10 atol=1e-20
            end
        end
        x=randn(rng,ComplexF64,A.n);y=similar(x)
        @test norm(A*x-Z*x)/norm(Z*x)<4e-13
        @test DiffMoM._planar_conformal_ufft_diagonal(A)≈diag(Z) rtol=3e-13
        mul!(y,A,x);@test (@allocated mul!(y,A,x))==0
        original=copy(y);mul!(y,A,x,.7+.2im,-.3)
        @test y≈(.7+.2im)*(Z*x)-.3original rtol=3e-13
        alias=copy(x);mul!(alias,A,alias)
        @test alias≈Z*x rtol=3e-13
        @test size(A,3)==1
        @test_throws ArgumentError size(A,0)
        @test_throws ArgumentError mul!(y,A,A.output)
        @test_throws ArgumentError planar_conformal_ufft_operator(prob,2e9;nx,ny,max_bytes=1)
        @test_throws ArgumentError planar_conformal_ufft_operator(prob,2e9;nx=3,ny)
        @test_throws ArgumentError DiffMoM._planar_conformal_ufft_diagonal(A;max_bytes=1)
        dense=solve_planar_conformal(prob,2e9;mx=19,my=23,surface_zs=zs)
        fast=solve_planar_conformal(prob,2e9;method=:ufft,nx,ny,mx=19,my=23,surface_zs=zs,
            rtol=1e-10,memory=150)
        @test fast.y≈dense.y rtol=2e-9
        @test fast.currents≈dense.currents rtol=2e-9
        @test maximum(fast.relative_residuals)<=1e-10
        @test opnorm(fast.s)<=1+1e-10
        @test fast.y≈transpose(fast.y) rtol=2e-9
        currents=planar_conformal_current_maps(fast;port=2)
        reference=planar_conformal_current_maps(dense;port=2)
        @test all(isapprox(currents[t].current,reference[t].current;rtol=2e-9) for t in eachindex(currents))
    end
    prob,nx,ny=conformal_ufft_fixture()
    @test_throws ArgumentError solve_planar_conformal(prob,1e9;method=:unknown)
    @test_throws ArgumentError solve_planar_conformal(prob,1e9;method=:ufft,nx,ny,rtol=NaN)
    @test_throws ArgumentError solve_planar_conformal(prob,1e9;method=:ufft,nx,ny,memory=0)
    @test_throws ArgumentError solve_planar_conformal(prob,1e9;method=:ufft,nx,ny,max_bytes=1)
    # A budget below the former coefficient payload now fits the complete
    # operator. Both signs still reproduce the full finite-mode reaction.
    mx,my=129,131;nf=6
    oldcoeffbytes=sizeof(ComplexF64)*8*mx*my*nf
    bounded=planar_conformal_ufft_operator(prob,2e9;nx,ny,mx,my,max_bytes=oldcoeffbytes-1)
    @test sizeof(bounded.c_te)+sizeof(bounded.c_tm)==oldcoeffbytes÷2
    @test DiffMoM._planar_hybrid_conformal_owned_payload(bounded)<oldcoeffbytes-1
    @test_throws ArgumentError planar_conformal_ufft_operator(prob,2e9;nx,ny,mx,my,max_bytes=oldcoeffbytes÷2-1)
end
