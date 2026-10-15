module PlanarLineShortSectionsTests
using DiffMoM, Test, LinearAlgebra

@testset "Line modal extraction retains first-order short-section length" begin
    for magnitude in (1e-300,1e-150,1e-100,1e-10,1e-8),
            direction in (1.,1im,1+.01im),impedance in (1.,50.,1e100,1e-100)
        expected=magnitude*direction
        transfer=planar_line_abcd(impedance,expected)
        # Both modal entries must be present in the stored representation.
        all(isfinite,transfer) && !iszero(transfer[1,2]) && !iszero(transfer[2,1]) || continue
        zc,gl=DiffMoM._line_params_of_abcd(transfer)
        @test zc≈impedance rtol=5e-14
        @test gl≈expected rtol=5e-14
    end
end

@testset "Sinh inversion keeps the principal line branch" begin
    for attenuation in (-.1,0.,.1),phase in (-pi,-2.8,-1.7,-.2,.2,1.7,2.8,pi)
        expected=complex(attenuation,phase)
        transfer=planar_line_abcd(50+3im,expected)
        zc,gl=DiffMoM._line_params_of_abcd(transfer)
        @test zc≈50+3im rtol=2e-13
        @test gl≈expected rtol=2e-13
        @test planar_line_abcd(zc,gl)≈transfer rtol=2e-13
    end
    @test_throws ArgumentError DiffMoM._line_params_of_abcd(ComplexF64[1 0;0 1])
    @test_throws ArgumentError DiffMoM._line_params_of_abcd(ComplexF64[2 1;1 2])
end
end
