module ConformalPulseAreaTests
using DiffMoM, Test
const DM=DiffMoM
include(joinpath(@__DIR__, "helpers", "conformal_pulse_moments.jl"))

function polynomial_moment(v,values,p,q)
    # Expand x^p y^q in barycentric coordinates, then integrate each term.
    x=Q.(v[1,:]);y=Q.(v[2,:]);coefficients=Dict((0,0,0)=>Q(1))
    for row in (fill(x,p)...,fill(y,q)...)
        next=Dict{NTuple{3,Int},Q}()
        for (powers,value) in coefficients,j in 1:3
            key=ntuple(k->powers[k]+(k==j),3)
            next[key]=get(next,key,Q(0))+value*row[j]
        end
        coefficients=next
    end
    integral=Q(0)
    for (powers,value) in coefficients,j in 1:3
        exponents=ntuple(k->powers[k]+(k==j),3)
        integral+=value*Q(values[j])*prod(factorial(big(e)) for e in exponents)/
            factorial(big(sum(exponents)+2))
    end
    Float64(2exact_area(v)*integral)
end

@testset "conformal pulses retain exact nonzero area and first moments" begin
    base=[0. 1. nextfloat(1.);0. nextfloat(1.) nextfloat(nextfloat(1.))]
    @test (base[1,2]*base[2,3]-base[2,2]*base[1,3])==0
    # Keep first moments normal so the relative-error gate tests analytic
    # accuracy, rather than the representable denormal spacing.
    for scale in (2.0^-200,2.0^-10,1.,1e100),order in ((1,2,3),(1,3,2),(2,3,1))
        v=base[:,collect(order)].*scale;area=Float64(exact_area(v))
        @test area>0
        mesh=PlanarConformalMesh(v,reshape([1,2,3],3,1))
        @test only(mesh.areas)==area
        pulse=DM._planar_pulse_and_first_moments(v,0.,0.)
        @test pulse[1]≈area rtol=4eps(1.) atol=0
        for d in 1:2
            expected=Float64(exact_area(v)*sum(Q(v[d,j])-Q(v[d,1]) for j in 1:3)/3)
            # Subnormal values cannot retain a relative tolerance smaller
            # than one representable step; allow two steps of rounding.
            @test pulse[d+1]≈expected rtol=8eps(1.) atol=2nextfloat(0.)
        end
        for (u,w) in ((.7,1.2),(0.,1.),(1.,0.),(-.7,1.2))
            kx,ky=u/scale,w/scale
            expected=series_moments(v,kx,ky)
            actual=DM._planar_pulse_and_first_moments(v,kx,ky)
            for j in 1:3
                @test actual[j]≈expected[j] rtol=5e-13 atol=2nextfloat(0.)
            end
        end
    end
    # Exactly collinear triangles still have zero pulse and first moments.
    v=[0. 1. 2.;0. 1. 2.]
    @test DM._planar_pulse_and_first_moments(v,.7,1.2)==(0im,0im,0im)
end

@testset "projection polynomial moments retain accepted triangle area" begin
    base=[0. 1. nextfloat(1.);0. nextfloat(1.) nextfloat(nextfloat(1.))]
    for scale in (2.0^-10,1.,2.0^10),order in ((1,2,3),(1,3,2),(2,3,1)),values in ((1.,1.,1.),(1.,2.,3.))
        v=base[:,collect(order)].*scale
        moments=DM._planar_affine_polynomial_moments_fast(v,values,(0.,0.),(1.,1.),3)
        for p in 0:2,q in 0:2
            @test moments[p+1,q+1]≈polynomial_moment(v,values,p,q) rtol=3e-14 atol=0
        end
    end
end

function allocated_pulse(v,kx,ky)
    DM._planar_pulse_and_first_moments(v,kx,ky)
    @allocated DM._planar_pulse_and_first_moments(v,kx,ky)
end

@testset "ordinary original pulse action keeps zero-allocation storage" begin
    v=[0. .003 0.;0. 0. .002]
    for (kx,ky) in ((0.,0.),(0.,400.),(300.,0.),(300.,400.),(1e5,2e5))
        @test allocated_pulse(v,kx,ky)==0
    end
end
end
