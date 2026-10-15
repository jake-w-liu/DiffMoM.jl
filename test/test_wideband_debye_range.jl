using DiffMoM,Test

@testset "Wideband Debye finite products and coherent pole sums" begin
    oracle(f,epsi,delta,tau)=setprecision(BigFloat,8192) do
        ComplexF64(Complex{BigFloat}(epsi)+sum(BigFloat(d)/
            complex(one(BigFloat),2BigFloat(pi)*BigFloat(f)*BigFloat(t)) for (d,t) in zip(delta,tau);init=zero(Complex{BigFloat})))
    end
    for (f,t) in ((1e308,1e-308),(1e-308,1e308),(1e200,1e-200),(0.,1e308))
        expected=oracle(f,1.,[1.],[t])
        @test wideband_debye_epsr_3d(f;eps_inf=1,delta=[1.],tau=[t]) ≈ expected rtol=4eps(Float64)
        model=WidebandDebyePermittivity3D(1,[1.],[t])
        @test material_epsr_3d(model,f) ≈ expected rtol=4eps(Float64)
    end
    for f in (0.,1e9)
        d=[1e16,3.,-1e16];t=fill(1e-9,3)
        @test wideband_debye_epsr_3d(f;eps_inf=0,delta=d,tau=t,passive=false) ≈ oracle(f,0,d,t) rtol=4eps(Float64)
    end
    @test_throws OverflowError wideband_debye_epsr_3d(0.;eps_inf=1e308,delta=[1e308],tau=[1.],passive=false)
    @test_throws ErrorException wideband_debye_epsr_3d(1e308;eps_inf=1,delta=[1.,NaN],tau=[1e-308,1.])
    d=[1.,2.,3.];t=[1e-10,1e-9,1e-8]
    wideband_debye_epsr_3d(1e9;eps_inf=1,delta=d,tau=t)
    @test (@allocated wideband_debye_epsr_3d(1e9;eps_inf=1,delta=d,tau=t)) <= 256
end
