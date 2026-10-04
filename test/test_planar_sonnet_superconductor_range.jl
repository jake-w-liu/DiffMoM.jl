module NativeSonnetSuperconductorRangeTests
using Test, DiffMoM, LinearAlgebra

const fixture=joinpath(@__DIR__,"fixtures/native_ideal_component_units/res/ohm/ohm.son")
const p=read_sonnet_project(fixture)
row(rdc,rrf,xdc=0.,ls=0.)=["Sheet","1","SUP",string(rdc),string(rrf),string(xdc),string(ls)]

function oracle(rdc,rrf,f,xdc=0.,ls=0.)
    setprecision(BigFloat,8192) do
        d,r,frequency,x,l=BigFloat.((rdc,rrf,f,xdc,ls))
        rf=complex(r*sqrt(frequency),r*sqrt(frequency))
        resistance=iszero(r) ? complex(d) : iszero(d) ? rf : rf/tanh(rf/d)
        ComplexF64(resistance+im*(x+BigFloat(2)*BigFloat(pi)*frequency*l*BigFloat(1e-12)))
    end
end

@testset "Native SUP crossover preserves finite DC and RF limits" begin
    for (rdc,rrf,f) in ((1.,1e-320,1e-20),(1e100,1e-300,1e9),
            (1e308,1e-308,1e9),(1e200,1.,1e9),(1.,1e-100,1e9),
            (1.,0.,1e9),(0.,1.,1e9),(1.,1.,1e9))
        z=sonnet_metal_zs(p,row(rdc,rrf),f)
        expected=oracle(rdc,rrf,f)
        @test isfinite(z)
        @test real(z)≈real(expected) rtol=4e-15
        @test imag(z)≈imag(expected) rtol=4e-15
    end
    # Ratios sweep the crossover and both branch boundaries. The independent
    # high precision formula checks each real/imaginary component separately.
    for f in (1.,1e6,1e9,1e12),ratio in (1e-8,1e-4,.01,.1,prevfloat(.25),.25,nextfloat(.25),
            1.,10.,prevfloat(20.),20.,nextfloat(20.),100.)
        rrf=ratio/sqrt(f)
        z=sonnet_metal_zs(p,row(1.,rrf),f)
        expected=oracle(1.,rrf,f)
        @test real(z)≈real(expected) rtol=4e-15
        @test imag(z)≈imag(expected) rtol=4e-15
    end
    # At an unrepresentable ratio, the finite skin limit remains defined.
    @test sonnet_metal_zs(p,row(nextfloat(0.),1e-6),1e8)≈.01+.01im rtol=1e-15
    for (f,ls) in ((1e308,1e-300),(1e308,nextfloat(0.)),(1e-300,1e308),
            (1e9,.11),(1e308,-1e-300))
        z=sonnet_metal_zs(p,row(1.,0.,.25,ls),f)
        @test z≈oracle(1.,0.,f,.25,ls) rtol=4e-15
    end
    # Individually overflowing signed reactances may cancel to a finite
    # stored impedance; summing after exponent scaling retains that result.
    for (rdc,rrf,f,xdc,ls) in ((1.,0.,1e308,-1e308,4e11),
            (0.,1e154,1e308,-1e308,-1e12/(2pi)))
        z=sonnet_metal_zs(p,row(rdc,rrf,xdc,ls),f)
        expected=oracle(rdc,rrf,f,xdc,ls)
        @test isfinite(z)
        @test real(z)≈real(expected) rtol=4e-15
        @test imag(z)≈imag(expected) rtol=4e-15
    end
    for (metal,f) in ((row(1.,1e308),1e9),(row(0.,nextfloat(0.)),1e-100),
            (row(1.,0.,0.,1e308),1e308))
        @test_throws ArgumentError sonnet_metal_zs(p,metal,f)
    end
    @test_throws ArgumentError sonnet_metal_zs(p,row(1.,0.),BigFloat("1e1000"))
    @test_throws ArgumentError sonnet_metal_zs(p,row(1.,0.),BigFloat("1e-1000"))
end
end
