using DiffMoM,Test,LinearAlgebra,Random
isdefined(DiffMoM,:_planar_triangle_affine_fourier) || Base.include(DiffMoM,joinpath(@__DIR__,"../src/planar/PlanarTriangleTransforms.jl"))
function triangle_gauss_fourier(v,values,kx,ky;n=48)
    beta=[j/sqrt(4j*j-1) for j in 1:n-1]
    quadrature=eigen(SymTridiagonal(zeros(n),beta))
    nodes=(quadrature.values.+1)./2
    weights=quadrature.vectors[1,:].^2
    det=abs((v[1,2]-v[1,1])*(v[2,3]-v[2,1])-(v[2,2]-v[2,1])*(v[1,3]-v[1,1]))
    result=0.0im
    for (i,u) in enumerate(nodes),(j,w) in enumerate(nodes)
        lambda=((1-u)*(1-w),u,(1-u)*w)
        x=sum(lambda[k]*v[1,k] for k in 1:3)
        y=sum(lambda[k]*v[2,k] for k in 1:3)
        scalar=sum(lambda[k]*values[k] for k in 1:3)
        result+=weights[i]*weights[j]*det*(1-u)*scalar*cis(kx*x+ky*y)
    end
    return result
end
@testset "planar triangle: exact affine Fourier moments" begin
    G=DiffMoM
    vertices=[.123 .31 .157;.27 .301 .463]
    for values in ((1.,1.,1.),(1.,0.,0.),(-.3,2.1,-1.2)),(kx,ky) in
            ((0.,0.),(1e-12,-2e-12),(3.,-7.),(100.,-137.),(37.,0.),(0.,-53.))
        actual=G._planar_triangle_affine_fourier(vertices,values,kx,ky)
        reference=triangle_gauss_fourier(vertices,values,kx,ky)
        @test actual≈reference rtol=1e-10 atol=1e-13
    end
    rng=MersenneTwister(2307)
    for _ in 1:20
        v=rand(rng,2,3);values=Tuple(randn(rng,3));kx,ky=randn(rng,2).*30
        @test G._planar_triangle_affine_fourier(v,values,kx,ky)≈
            triangle_gauss_fourier(v,values,kx,ky;n=64) rtol=1e-10 atol=1e-12
    end
    a,b=.17,.29;v=[0. a 0.;0. 0. b]
    for k in (1e-3,1.,100.,10000.)
        actual=G._planar_triangle_affine_fourier(v,(1.,1.,1.),k,0.)
        exact=b*(1im/k+(1-exp(1im*k*a))/(a*k*k))
        @test actual≈exact rtol=(k==1e-3 ? 1e-8 : 1e-12) atol=1e-12
    end
    # Two actual triangles tile a rectangle exactly: no raster or surface
    # quadrature is involved in their combined modal transform.
    v2=[a a 0.;0. b b]
    for (kx,ky) in ((0.,0.),(2.,-3.),(74.,0.),(0.,-81.))
        total=G._planar_triangle_affine_fourier(v,(1.,1.,1.),kx,ky)+
            G._planar_triangle_affine_fourier(v2,(1.,1.,1.),kx,ky)
        x=kx==0 ? a : (exp(1im*kx*a)-1)/(1im*kx)
        y=ky==0 ? b : (exp(1im*ky*b)-1)/(1im*ky)
        @test total≈x*y rtol=1e-12 atol=1e-13
    end
end
