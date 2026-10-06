module ConductorSurfaceRangeTests
using DiffMoM,LinearAlgebra,SHA,TOML,Test
function surface_transfer_reference(f,layers,load,substrate)
    setprecision(BigFloat,4096) do
        omega=2*BigFloat(pi)*Complex{BigFloat}(f);mu=BigFloat(DiffMoM._MU0)
        M=Matrix{Complex{BigFloat}}(I,2,2)
        for layer in layers
            sigma=Complex{BigFloat}(layer.sigma);h=Complex{BigFloat}(layer.thickness);mur=Complex{BigFloat}(layer.mur)
            gamma=sqrt(1im*omega*mu*mur*sigma);zc=1im*omega*mu*mur/gamma
            A=cosh(gamma*h);B=sinh(gamma*h)
            M=M*[A zc*B;B/zc A]
        end
        terminal=if substrate!==nothing
            gamma=sqrt(1im*omega*mu*Complex{BigFloat}(substrate))
            1im*omega*mu/gamma
        elseif load===:open
            nothing
        elseif load===:pec
            zero(Complex{BigFloat})
        else
            Complex{BigFloat}(load)
        end
        value=terminal===nothing ? M[1,1]/M[2,1] :
            (M[1,1]*terminal+M[1,2])/(M[2,1]*terminal+M[2,2])
        return ComplexF64(value)
    end
end

const SURFACE_CASES=[
    (f=8000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(58000000.0,2e-06;mur=1.0),PlanarConductorLayer(14000000.0,5e-07;mur=1.0)],load=:open,substrate=nothing),
    (f=8000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(58000000.0,2e-06;mur=1.0),PlanarConductorLayer(14000000.0,5e-07;mur=1.0)],load=:pec,substrate=nothing),
    (f=8000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(58000000.0,2e-06;mur=1.0),PlanarConductorLayer(14000000.0,5e-07;mur=1.0)],load=complex(50.0,2.0),substrate=nothing),
    (f=8000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(58000000.0,2e-06;mur=1.0),PlanarConductorLayer(14000000.0,5e-07;mur=1.0)],load=complex(1e-200,1e-200),substrate=nothing),
    (f=8000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(58000000.0,2e-06;mur=1.0),PlanarConductorLayer(14000000.0,5e-07;mur=1.0)],load=complex(1e+200,1e+200),substrate=nothing),
    (f=1e+100,layers=PlanarConductorLayer[PlanarConductorLayer(1e+220,1e-160;mur=1.0),PlanarConductorLayer(2e+220,2e-160;mur=1.0)],load=:open,substrate=nothing),
    (f=1e+100,layers=PlanarConductorLayer[PlanarConductorLayer(1e+220,1e-160;mur=1.0),PlanarConductorLayer(2e+220,2e-160;mur=1.0)],load=:pec,substrate=nothing),
    (f=1e+100,layers=PlanarConductorLayer[PlanarConductorLayer(1e+220,1e-160;mur=1.0),PlanarConductorLayer(2e+220,2e-160;mur=1.0)],load=complex(50.0,2.0),substrate=nothing),
    (f=1e+100,layers=PlanarConductorLayer[PlanarConductorLayer(1e+220,1e-160;mur=1.0),PlanarConductorLayer(2e+220,2e-160;mur=1.0)],load=complex(1e-200,1e-200),substrate=nothing),
    (f=1e+100,layers=PlanarConductorLayer[PlanarConductorLayer(1e+220,1e-160;mur=1.0),PlanarConductorLayer(2e+220,2e-160;mur=1.0)],load=complex(1e+200,1e+200),substrate=nothing),
    (f=1e-200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-200,1e+200;mur=1.0),PlanarConductorLayer(2e-200,2e+200;mur=1.0)],load=:open,substrate=nothing),
    (f=1e-200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-200,1e+200;mur=1.0),PlanarConductorLayer(2e-200,2e+200;mur=1.0)],load=:pec,substrate=nothing),
    (f=1e-200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-200,1e+200;mur=1.0),PlanarConductorLayer(2e-200,2e+200;mur=1.0)],load=complex(50.0,2.0),substrate=nothing),
    (f=1e-200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-200,1e+200;mur=1.0),PlanarConductorLayer(2e-200,2e+200;mur=1.0)],load=complex(1e-200,1e-200),substrate=nothing),
    (f=1e-200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-200,1e+200;mur=1.0),PlanarConductorLayer(2e-200,2e+200;mur=1.0)],load=complex(1e+200,1e+200),substrate=nothing),
    (f=1e+200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-100,1e-180;mur=1.0),PlanarConductorLayer(2e-100,2e-180;mur=1.0)],load=:open,substrate=nothing),
    (f=1e+200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-100,1e-180;mur=1.0),PlanarConductorLayer(2e-100,2e-180;mur=1.0)],load=:pec,substrate=nothing),
    (f=1e+200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-100,1e-180;mur=1.0),PlanarConductorLayer(2e-100,2e-180;mur=1.0)],load=complex(50.0,2.0),substrate=nothing),
    (f=1e+200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-100,1e-180;mur=1.0),PlanarConductorLayer(2e-100,2e-180;mur=1.0)],load=complex(1e-200,1e-200),substrate=nothing),
    (f=1e+200,layers=PlanarConductorLayer[PlanarConductorLayer(1e-100,1e-180;mur=1.0),PlanarConductorLayer(2e-100,2e-180;mur=1.0)],load=complex(1e+200,1e+200),substrate=nothing),
    (f=complex(1e+100,-1e+99),layers=PlanarConductorLayer[PlanarConductorLayer(complex(1e+220,2e+219),complex(1e-160,1.5e-161);mur=complex(1.2,0.05)),PlanarConductorLayer(complex(2e+220,-1e+219),complex(2e-160,-1e-161);mur=1.0)],load=:open,substrate=nothing),
    (f=complex(1e+100,-1e+99),layers=PlanarConductorLayer[PlanarConductorLayer(complex(1e+220,2e+219),complex(1e-160,1.5e-161);mur=complex(1.2,0.05)),PlanarConductorLayer(complex(2e+220,-1e+219),complex(2e-160,-1e-161);mur=1.0)],load=:pec,substrate=nothing),
    (f=complex(1e+100,-1e+99),layers=PlanarConductorLayer[PlanarConductorLayer(complex(1e+220,2e+219),complex(1e-160,1.5e-161);mur=complex(1.2,0.05)),PlanarConductorLayer(complex(2e+220,-1e+219),complex(2e-160,-1e-161);mur=1.0)],load=complex(50.0,2.0),substrate=nothing),
    (f=complex(1e+100,-1e+99),layers=PlanarConductorLayer[PlanarConductorLayer(complex(1e+220,2e+219),complex(1e-160,1.5e-161);mur=complex(1.2,0.05)),PlanarConductorLayer(complex(2e+220,-1e+219),complex(2e-160,-1e-161);mur=1.0)],load=complex(1e-200,1e-200),substrate=nothing),
    (f=complex(1e+100,-1e+99),layers=PlanarConductorLayer[PlanarConductorLayer(complex(1e+220,2e+219),complex(1e-160,1.5e-161);mur=complex(1.2,0.05)),PlanarConductorLayer(complex(2e+220,-1e+219),complex(2e-160,-1e-161);mur=1.0)],load=complex(1e+200,1e+200),substrate=nothing),
    (f=1000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(10000000.0,complex(1e-07,0.01);mur=1.0)],load=:open,substrate=nothing),
    (f=1000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(10000000.0,complex(1e-07,0.01);mur=1.0)],load=:pec,substrate=nothing),
    (f=1000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(10000000.0,complex(1e-07,0.01);mur=1.0)],load=complex(50.0,2.0),substrate=nothing),
    (f=1000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(10000000.0,complex(1e-07,0.01);mur=1.0)],load=complex(1e-200,1e-200),substrate=nothing),
    (f=1000000000.0,layers=PlanarConductorLayer[PlanarConductorLayer(10000000.0,complex(1e-07,0.01);mur=1.0)],load=complex(1e+200,1e+200),substrate=nothing),
    (f=1e+100,layers=PlanarConductorLayer[],load=:open,substrate=1e+220),
    (f=1e-200,layers=PlanarConductorLayer[],load=:open,substrate=1e-200)
]


@testset "conductor surface range preserves full slab and plating equations" begin
    for c in SURFACE_CASES
        expected=surface_transfer_reference(c.f,c.layers,c.load,c.substrate)
        actual=planar_layered_surface_zs(c.f,c.layers;load=c.load,substrate_sigma=c.substrate)
        @test isfinite(actual)
        # Check components separately: a large DC resistance cannot hide
        # the loss of a smaller, representable inductive impedance.
        @test abs(real(actual)-real(expected))<=max(1e-12*abs(real(expected)),2eps(real(expected)))
        @test abs(imag(actual)-imag(expected))<=max(1e-12*abs(imag(expected)),2eps(imag(expected)))
    end
end

@testset "two-sheet conductor range retains physical half-film impedance" begin
    for (f,sigma,h) in ((8e9,5.8e7,2e-6),(1e100,1e220,1e-160),
        (1e-200,1e-200,1e200),(1e200,1e-100,1e-180))
        expected=surface_transfer_reference(f,[PlanarConductorLayer(sigma,h/2)],:open,nothing)
        pair=planar_two_sheet_zs(f,sigma,h)
        @test all(isfinite,pair)
        @test pair[1,2]==pair[2,1]==0
        @test pair[1,1]==pair[2,2]
        @test abs(real(pair[1,1])-real(expected))<=max(1e-12*abs(real(expected)),2eps(real(expected)))
        @test abs(imag(pair[1,1])-imag(expected))<=max(1e-12*abs(imag(expected)),2eps(imag(expected)))
    end
end

@testset "conductor range precision remains scoped across concurrent calls" begin
    initial=precision(BigFloat)
    tasks=[Threads.@spawn setprecision(BigFloat,requested) do
        before=precision(BigFloat);yield()
        value=planar_layered_surface_zs(1e100,[PlanarConductorLayer(1e220,1e-160),PlanarConductorLayer(2e220,2e-160)])
        yield()
        (requested=requested,before=before,after=precision(BigFloat),value=value)
    end for requested in (128,384,512,768)]
    values=fetch.(tasks)
    @test precision(BigFloat)==initial
    for value in values
        @test value.before==value.requested
        @test value.after==value.requested
        @test isfinite(value.value)
    end
    @test_throws ArgumentError planar_layered_surface_zs(1.,[PlanarConductorLayer(1e-300,1e-300)])
    # This substrate is large but representable; use an independent
    # characteristic impedance instead of misclassifying it as overflow.
    reference=surface_transfer_reference(1e308,PlanarConductorLayer[],:open,1e-308)
    value=planar_layered_surface_zs(1e308,PlanarConductorLayer[];substrate_sigma=1e-308)
    @test isfinite(value)
    @test real(value)≈real(reference) rtol=1e-12
    @test imag(value)≈imag(reference) rtol=1e-12
    @test_throws ArgumentError planar_layered_surface_zs(1e308,PlanarConductorLayer[];substrate_sigma=1e-320)
end

@testset "conductor range preserves mixed arithmetic and owned precision" begin
    sigma=BigFloat("1e-300");height=BigFloat("1e-6")
    for f in (1e308,complex(1e308,1e300)),bare in (false,true)
        layers=bare ? PlanarConductorLayer[] : [PlanarConductorLayer(sigma,height)]
        substrate=bare ? sigma : nothing
        expected=surface_transfer_reference(f,layers,:open,substrate)
        actual=planar_layered_surface_zs(f,layers;substrate_sigma=substrate)
        @test isfinite(actual)
        @test Float64(real(actual))≈real(expected) rtol=1e-12
        @test Float64(imag(actual))≈imag(expected) rtol=1e-12
    end
    # BigFloat frequency does not widen a Float64 thickness square.
    for f in (BigFloat("1e200"),Complex{BigFloat}(BigFloat("1e200")))
        layers=[PlanarConductorLayer(1e-100,1e-180)]
        expected=surface_transfer_reference(f,layers,:open,nothing)
        actual=planar_layered_surface_zs(f,layers)
        @test isfinite(actual)
        @test Float64(real(actual))≈real(expected) rtol=1e-12
        @test Float64(imag(actual))≈imag(expected) rtol=1e-12
    end
    setprecision(BigFloat,20000) do
        layers=[PlanarConductorLayer(BigFloat("1e-300"),BigFloat("1e-6"))]
        result=planar_layered_surface_zs(1e308,layers)
        @test precision(BigFloat)==20000
        @test precision(real(result))>=20000
    end
    owned_sigma=setprecision(BigFloat,20000) do
        BigFloat("1e-300")
    end
    setprecision(BigFloat,128) do
        layers=[PlanarConductorLayer(owned_sigma,BigFloat("1e-6"))]
        result=planar_layered_surface_zs(1e308,layers)
        @test precision(BigFloat)==128
        @test precision(real(result))>=precision(owned_sigma)
    end
end

@testset "public open-film impedance retains separate components across range" begin
    for ef in (-250,-100,0,100,250), es in (-250,-100,0,100,250),
            eh in (-200,-50,0,50,200), mur in (.5,2.)
        f=10.0^ef;sigma=10.0^es;h=10.0^eh
        expected=setprecision(BigFloat,4096) do
            omega=2*BigFloat(pi)*BigFloat(f)
            mu=BigFloat(DiffMoM._MU0)*BigFloat(mur)
            gamma=sqrt(complex(0,omega*mu*BigFloat(sigma)))
            ComplexF64(complex(0,omega*mu)/gamma/tanh(gamma*BigFloat(h)))
        end
        layers=[PlanarConductorLayer(sigma,h;mur)]
        if isfinite(expected) && !iszero(expected)
            actual=planar_layered_surface_zs(f,layers)
            @test isfinite(actual)
            @test abs(real(actual)-real(expected))<=max(1e-12*abs(real(expected)),2eps(real(expected)))
            @test abs(imag(actual)-imag(expected))<=max(1e-12*abs(imag(expected)),2eps(imag(expected)))
        else
            @test_throws ArgumentError planar_layered_surface_zs(f,layers)
        end
    end
end

@testset "two-sheet half-film preserves subnormal thickness components" begin
    u=nextfloat(0.)
    for h in (u,3u,complex(u,u),complex(3u,u))
        expected=setprecision(BigFloat,4096) do
            omega=2*BigFloat(pi);mu=BigFloat(DiffMoM._MU0);sigma=BigFloat(1e308)
            gamma=sqrt(complex(0,omega*mu*sigma));zc=complex(0,omega*mu)/gamma
            ComplexF64(zc/tanh(gamma*(Complex{BigFloat}(h)/2)))
        end
        actual=planar_two_sheet_zs(1.,1e308,h)
        @test all(isfinite,actual)
        @test actual[1,2]==actual[2,1]==0
        @test actual[1,1]==actual[2,2]
        @test abs(real(actual[1,1])-real(expected))<=max(1e-12*abs(real(expected)),2eps(real(expected)))
        @test abs(imag(actual[1,1])-imag(expected))<=max(1e-12*abs(imag(expected)),2eps(imag(expected)))
    end
end
end
