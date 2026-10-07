module DerivedTriangleNumericsTests
using DiffMoM,Test,SHA,TOML
const Q=Rational{BigInt}
function exact_simplex_affine(v,values,kx,ky)
    z=ntuple(j->complex(Q(0),Q(kx)*Q(v[1,j])+Q(ky)*Q(v[2,j])),Val(3))
    radius=maximum(abs,imag.(z));m=0
    # Independent absolute power-series tail, with enough coefficient-space
    # degree to compare separately representable affine moments.
    u=Q(eps(Float64))/2;term=radius
    while term/(1-radius//(m+2))>u
        m+=1;term*=radius//(m+1)
    end
    order=m+2length(values)-1
    integral=complex(Q(0),Q(0))
    for n in 0:order,p in 0:n,q in 0:n-p
        r=n-p-q
        monomial=z[1]^p*z[2]^q*z[3]^r/factorial(big(n+3))
        integral+=monomial*(Q(values[1])*(p+1)+Q(values[2])*(q+1)+Q(values[3])*(r+1))
    end
    area=abs((Q(v[1,2])-Q(v[1,1]))*(Q(v[2,3])-Q(v[2,1]))-(Q(v[1,3])-Q(v[1,1]))*(Q(v[2,2])-Q(v[2,1])))
    integral*=area
    # Conversion operands and working precision derive from exact bit lengths.
    function stored(q)
        n,d=numerator(q),denominator(q)
        bits=ndigits(n;base=2)+ndigits(d;base=2)+precision(Float64)
        setprecision(BigFloat,bits) do
            Float64(BigFloat(n)/BigFloat(d))
        end
    end
    complex(stored(real(integral)),stored(imag(integral)))
end
function allocation(v,x,y,kx,ky)
    DiffMoM._planar_triangle_affine_fourier(v,x,kx,ky)
    DiffMoM._planar_triangle_affine_fourier_pair(v,x,y,kx,ky)
    ((@allocated DiffMoM._planar_triangle_affine_fourier(v,x,kx,ky)),(@allocated DiffMoM._planar_triangle_affine_fourier_pair(v,x,y,kx,ky)))
end

function run_tests()
file=joinpath(dirname(@__DIR__),"src/planar/PlanarTriangleTransforms.jl");before=bytes2hex(sha256(read(file)))
smallest=exponent(nextfloat(0.));radii=unique([0.,nextfloat(0.),prevfloat(1.),1.,1/2,2/3,3/4])
append!(radii,[ldexp(1.,k) for k in smallest:0])
radii=unique(radii);rows=Dict{String,Any}[]
result=@testset "Triangle Taylor order follows an outward exact tail bound" begin
    for radius in radii
        order=DiffMoM._planar_exp_series_order(radius)
        @test order>=0
        @test (@allocated DiffMoM._planar_exp_series_order(radius))==0
        push!(rows,Dict("radius"=>radius,"order"=>order))
    end
    @test DiffMoM._planar_exp_series_order(0.)==0
    for radius in (-nextfloat(0.),nextfloat(1.),Inf,NaN)
        @test_throws ArgumentError DiffMoM._planar_exp_series_order(radius)
    end
    for phase in (0.,.5,1.),sign in (-1.,1.)
        nodes=ntuple(j->ComplexF64(sign*phase*im),Val(3))
        weights=DiffMoM._planar_triangle_exp_weights3(nodes)
        @test weights==ntuple(_->exp(only(unique(nodes)))*(1/6),Val(3))
    end
    @test bytes2hex(sha256(read(file)))==before
end
    v=[0. 1. 0.;0. 0. 1.];rows=Dict{String,Any}[]
    phases=(eps(Float64)^2,eps(Float64),sqrt(eps(Float64)),eps(Float64)^(1/4))
    x,y=(1.,-1.,0.),(1.,-2.,1.)
    result=@testset "Exact simplex cancellation moments and owned scalar pair allocation" begin
        for k in phases,values in (x,y),(kx,ky) in ((k,0.),(k,2k),(-k,-2k))
            expected=exact_simplex_affine(v,values,kx,ky)
            actual=DiffMoM._planar_triangle_affine_fourier(v,values,kx,ky)
            # Retain the original independent triangle-series tolerance and
            # its two-subnormal-step convention, separately per component.
            @test real(actual)≈real(expected) rtol=5e-13 atol=2nextfloat(0.)
            @test imag(actual)≈imag(expected) rtol=5e-13 atol=2nextfloat(0.)
            pair=DiffMoM._planar_triangle_affine_fourier_pair(v,values,values,kx,ky)
            @test pair==(actual,actual)
            @test allocation(v,values,values,kx,ky)==(0,0)
            push!(rows,Dict("kx"=>kx,"ky"=>ky,"values"=>collect(values),"expected_real"=>real(expected),"expected_imag"=>imag(expected),"actual_real"=>real(actual),"actual_imag"=>imag(actual)))
        end
    end
    @testset "Mixed mean components preserve exact scalar pair equivalence" begin
        ordinary=(1.,1.,1.)
        for k in phases,(first,second) in ((x,ordinary),(ordinary,y),(x,y))
            pair=DiffMoM._planar_triangle_affine_fourier_pair(v,first,second,k,2k)
            @test pair==(DiffMoM._planar_triangle_affine_fourier(v,first,k,2k),
                DiffMoM._planar_triangle_affine_fourier(v,second,k,2k))
            @test allocation(v,first,second,k,2k)==(0,0)
        end
    end
    @testset "Exact simplex permutation and coefficient range components" begin
    tiny=eps(Float64)^2
    examples=((1.,tiny,-1.),(1.,-2.,nextfloat(1.)),(1.,-2.,prevfloat(1.)),
        (1.,1.,-2.),(1.,-1.,0.),(floatmax(Float64)/2,floatmax(Float64)/2,-floatmax(Float64)/2),
        (floatmax(Float64),floatmax(Float64),-floatmax(Float64)),
        (floatmax(Float64),floatmax(Float64),floatmax(Float64)),
        (-floatmax(Float64),-floatmax(Float64),floatmax(Float64)),
        (floatmax(Float64),tiny,-floatmax(Float64)),
        (floatmax(Float64),1.,-floatmax(Float64)))
    for values in examples,permutation in ((1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1)),
        k in (0.,eps(Float64)^2,sqrt(eps(Float64)),eps(Float64)^(1/4))
        a=ntuple(j->values[permutation[j]],Val(3))
        expected=exact_simplex_affine(v,a,k,2k)
        actual=DiffMoM._planar_triangle_affine_fourier(v,a,k,2k)
        @test real(actual)≈real(expected) rtol=5e-13 atol=2nextfloat(0.)
        @test imag(actual)≈imag(expected) rtol=5e-13 atol=2nextfloat(0.)
    end
    end
end
run_tests()
end
