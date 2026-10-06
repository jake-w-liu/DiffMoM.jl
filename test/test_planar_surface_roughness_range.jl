module SurfaceRoughnessRangeTests
using DiffMoM,Test
function smooth_reference(f,sigma,mur)
    setprecision(BigFloat,4096) do
        factor=BigFloat(pi)*BigFloat(DiffMoM._MU0)*BigFloat(f)*BigFloat(mur)
        Float64(inv(sqrt(factor*BigFloat(sigma)))),ComplexF64(sqrt(factor/BigFloat(sigma))*(1+1im))
    end
end
@testset "skin depth and smooth impedance recover independent material equations" begin
    for (f,sigma,mur) in ((8e9,5.8e7,1.),(1e100,1e220,1.),(1e-200,1e-200,1.),(1e308,1e308,1e308))
        depth,expected=smooth_reference(f,sigma,mur)
        actual_depth=planar_skin_depth(f,sigma;mur)
        actual=planar_surface_zs(f,sigma;mur)
        @test isfinite(actual)
        @test abs(actual_depth-depth)<=max(1e-12*abs(depth),2eps(depth))
        @test abs(real(actual)-real(expected))<=max(1e-12*abs(real(expected)),2eps(real(expected)))
        @test abs(imag(actual)-imag(expected))<=max(1e-12*abs(imag(expected)),2eps(imag(expected)))
    end
end
@testset "zero and finite roughness preserve corrected public film equations" begin
    for rms in (0.,1e-180),loss_only in (false,true)
        f=1e100;sigma=1e220;h=1e-160;rough=HammerstadRoughness(rms)
        factor,expected=setprecision(BigFloat,4096) do
            omega=2*BigFloat(pi)*BigFloat(f);mu=BigFloat(DiffMoM._MU0);s=BigFloat(sigma)
            d=inv(sqrt(omega*mu*s/2));k=1+(2/BigFloat(pi))*atan(BigFloat(1.4)*(BigFloat(rms)/d)^2)
            gamma=sqrt(complex(0,omega*mu*s));film=complex(0,omega*mu)/gamma/tanh(gamma*BigFloat(h))
            Float64(k),ComplexF64(loss_only ? complex(k*real(film),imag(film)) : k*film)
        end
        actual=planar_layered_surface_zs(f,[PlanarConductorLayer(sigma,h)];roughness=rough,loss_only)
        @test isfinite(roughness_factor(rough,f,sigma))
        @test roughness_factor(rough,f,sigma)≈factor rtol=1e-12
        @test isfinite(actual)
        @test abs(real(actual)-real(expected))<=max(1e-12*abs(real(expected)),2eps(real(expected)))
        @test abs(imag(actual)-imag(expected))<=max(1e-12*abs(imag(expected)),2eps(imag(expected)))
    end
    for density in (0.,1e-300)
        f=8e9;sigma=5.8e7;radius=1e200;rough=HurayRoughness(radius,density)
        expected=setprecision(BigFloat,4096) do
            depth=inv(sqrt(BigFloat(pi)*BigFloat(DiffMoM._MU0)*BigFloat(f)*BigFloat(sigma)));r=BigFloat(radius)
            Float64(1+BigFloat(1.5)*4*BigFloat(pi)*r^2*BigFloat(density)/(1+depth/r+(depth/r)^2/2))
        end
        actual=roughness_factor(rough,f,sigma)
        @test isfinite(actual)
        @test actual≈expected rtol=1e-12
    end
end
@testset "surface recovery preserves owned precision and concurrent scopes" begin
    initial=precision(BigFloat)
    owned=setprecision(BigFloat,20000) do;BigFloat("1e308");end
    setprecision(BigFloat,128) do
        depth=planar_skin_depth(1e308,owned;mur=1e308)
        actual=planar_surface_zs(1e308,owned;mur=1e308)
        @test precision(BigFloat)==128
        @test precision(depth)>=precision(owned)
        @test precision(real(actual))>=precision(owned)
        @test isfinite(actual)
        _,expected=smooth_reference(1e308,owned,1e308)
        @test Float64(real(actual))≈real(expected) rtol=1e-12
        @test Float64(imag(actual))≈imag(expected) rtol=1e-12
    end
    tasks=[Threads.@spawn setprecision(BigFloat,requested) do
        before=precision(BigFloat);yield()
        result=roughness_factor(HurayRoughness(1e200,1e-300),8e9,5.8e7)
        yield();(before=before,after=precision(BigFloat),requested=requested,result=result)
    end for requested in (128,384,512,768)]
    for row in fetch.(tasks)
        @test row.before==row.requested
        @test row.after==row.requested
        @test isfinite(row.result)
        @test row.result≈1.884955592153876e101 rtol=1e-12
    end
    @test precision(BigFloat)==initial
end

@testset "complete corrected impedance survives amplified component range" begin
    for density in (1e-90,1e-110),loss_only in (false,true)
        f=1e-308;sigma=1e308;h=1e-3;radius=1e200;rough=HurayRoughness(radius,density)
        depth,expected,expected_film=setprecision(BigFloat,4096) do
            F=BigFloat(f);S=BigFloat(sigma);mu=BigFloat(DiffMoM._MU0)
            D=inv(sqrt(BigFloat(pi)*F*mu*S));R=BigFloat(radius)
            K=1+BigFloat(1.5)*4*BigFloat(pi)*R^2*BigFloat(density)/(1+D/R+(D/R)^2/2)
            rs=sqrt(BigFloat(pi)*F*mu/S);gamma=sqrt(complex(0,2*BigFloat(pi)*F*mu*S))
            slab=complex(0,2*BigFloat(pi)*F*mu)/gamma/tanh(gamma*BigFloat(h))
            apply(z)=loss_only ? complex(K*real(z),imag(z)) : K*z
            Float64(D),ComplexF64(apply(rs*(1+1im))),ComplexF64(apply(slab))
        end
        actual_depth=planar_skin_depth(f,sigma)
        actual=planar_surface_zs(f,sigma;roughness=rough,loss_only)
        film=planar_layered_surface_zs(f,[PlanarConductorLayer(sigma,h)];roughness=rough,loss_only)
        @test abs(actual_depth-depth)<=max(1e-12*abs(depth),2eps(depth))
        @test isfinite(actual)
        @test abs(real(actual)-real(expected))<=max(1e-12*abs(real(expected)),2eps(real(expected)))
        @test abs(imag(actual)-imag(expected))<=max(1e-12*abs(imag(expected)),2eps(imag(expected)))
        @test isfinite(film)
        @test abs(real(film)-real(expected_film))<=max(1e-12*abs(real(expected_film)),2eps(real(expected_film)))
        @test abs(imag(film)-imag(expected_film))<=max(1e-12*abs(imag(expected_film)),2eps(imag(expected_film)))
    end
end
end
