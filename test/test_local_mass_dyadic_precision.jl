module LocalMassDyadicPrecisionTests
using DiffMoM,LinearAlgebra,Test
@testset "Local mass products retain actual dyadic input precision" begin
    gap=DiffMoM._LOCAL_MASS_FALLBACK_PRECISION+precision(Float64)+1
    x=setprecision(BigFloat,gap+2) do
        large=ldexp(BigFloat(1),gap)
        [large+1,-large]
    end
    original=copy(x)
    mass=LocalMassMatrix(2,[1,1,2,2],[1,2,1,2],ones(4))
    for mode in (RoundNearest,RoundDown,RoundUp),operator in (mass,adjoint(mass))
        oldprecision=precision(BigFloat);oldrounding=rounding(BigFloat)
        setrounding(BigFloat,mode) do
            setprecision(BigFloat,precision(Float64)) do
                y=zeros(2);mul!(y,operator,x)
                @test y==ones(2)
                @test precision(BigFloat)==precision(Float64)
                @test rounding(BigFloat)==mode
                y=[Inf,NaN];mul!(y,operator,x,1.,0.)
                @test y==ones(2)
                complex_y=zeros(ComplexF64,2);mul!(complex_y,operator,x,1+im,0.)
                @test complex_y==fill(1+im,2)
                @test x==original
            end
        end
        @test precision(BigFloat)==oldprecision
        @test rounding(BigFloat)==oldrounding
    end
    large=big(1)<<gap;integer_x=[large+1,-large]
    y=zeros(2);mul!(y,mass,integer_x)
    @test y==ones(2)
    @test integer_x==[large+1,-large]
    ordinary_x=[2.,-1.];ordinary_y=zeros(2)
    mul!(ordinary_y,mass,ordinary_x)
    @test ordinary_y==ones(2)
    @test mass.vals==ones(4)
end
end
