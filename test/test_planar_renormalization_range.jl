module PlanarRenormalizationRangeTests
using DiffMoM, Test, LinearAlgebra

# Solve the physical peak-phasor source equations in high precision;
# no production reference conversion or intermediate wave transforms.
function load_reflection(z,reference)
    z=Complex{BigFloat}(z);reference=Complex{BigFloat}(reference)
    return ComplexF64((z-conj(reference))/(z+reference))
end

@testset "Wave renormalization preserves finite extreme reference scales" begin
    for scale in (1e-300,1e-200,1.,1e200,1e300)
        old=complex(scale,.2scale);new=complex(2scale,-.3scale)
        load=complex(1.5scale,.4scale)
        before=fill(load_reflection(load,old),1,1)
        converted=planar_renormalize_s(before,old,new)
        @test only(converted)≈load_reflection(load,new) atol=5e-16
        @test planar_renormalize_s(converted,new,old)≈before atol=5e-16
    end
    for (old,new) in ((1e308+1e308im,1.1e308-1e308im),
            (1e-10+1e308im,1e300+1e308im),
            (nextfloat(0.)+0im,2nextfloat(0.)+0im))
        before=fill(0.2+0im,1,1)
        # Recover the physical load from the old reflection in BigFloat.
        oldbig=Complex{BigFloat}(old)
        load=(conj(oldbig)+oldbig*big"0.2")/(1-big"0.2")
        @test only(planar_renormalize_s(before,old,new))≈load_reflection(load,new) atol=6e-16
    end
    for scale in (1e-200,1e200)
        thru=ComplexF64[0 1;1 0]
        converted=planar_renormalize_s(thru,[scale,scale],[2scale,2scale])
        @test converted≈thru atol=5e-16
        @test converted'*converted≈Matrix{ComplexF64}(I,2,2) atol=1e-15
    end
    @test_throws ArgumentError planar_renormalize_s(fill(big"1e1000",1,1),50.,50.)
    @test_throws ArgumentError planar_renormalize_s(fill(big"1e1000",1,1),50.,75.)
end
end
