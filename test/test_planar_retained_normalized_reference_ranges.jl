module RetainedNormalizedReferenceTests
using DiffMoM,LinearAlgebra,Test
@testset "Retained voltage preserves valid normalized reference ranges" begin
    edge=exponent(floatmax(Float64))-precision(Float64)
    bits=precision(Float64)
    for (top,lower) in ((edge,ldexp(1.0,bits)),(edge+1,prevfloat(ldexp(1.0,bits))),(edge+1,0.0))
        refs=ComplexF64[ldexp(1.0,top),ldexp(1.0,-top)]
        Y=ComplexF64[ldexp(1.0,-top+2bits) ldexp(1.0,bits);lower ldexp(1.0,top)]
        a=ComplexF64[0,1]
        S=planar_y_to_s(Y,refs)
        saved=copy(Y),copy(S),copy(a),copy(refs)
        reference,passive=setprecision(BigFloat,2*(top+bits)) do
            yy,zz,aa=Complex{BigFloat}.(Y),Complex{BigFloat}.(refs),Complex{BigFloat}.(a)
            voltage=(Matrix{Complex{BigFloat}}(I,2,2)+Diagonal(zz)*yy)\(2sqrt.(real.(zz)).*aa)
            # Exact positive Hermitian diagonal and determinant certify these
            # two-port inputs without ill-conditioned Float64 eigenvalues.
            off=(yy[1,2]+conj(yy[2,1]))/2
            positive=real(yy[1,1])>0 && real(yy[2,2])>0 &&
                real(yy[1,1])*real(yy[2,2])-abs2(off)>=0
            ComplexF64.(voltage),positive
        end
        voltage=@inferred DiffMoM._planar_wave_voltage_retained(Y,S,a,refs)
        @test all(isfinite,Y) && all(isfinite,refs) && all(isfinite,a)
        @test all(isfinite,S) && passive
        @test all(isapprox(actual,wanted;rtol=3e-14,atol=0) for (actual,wanted) in zip(planar_wave_voltages(S,a;z0=refs),reference))
        @test all(isfinite,voltage) && all(isapprox(actual,wanted;rtol=3e-14,atol=0) for (actual,wanted) in zip(voltage,reference))
        @test (Y,S,a,refs)==saved
    end
end
end
