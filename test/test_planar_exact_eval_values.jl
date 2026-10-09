module ExactEvaluationTests
using DiffMoM,LinearAlgebra,Test,TOML
function planted(poles,residues;d=zeros(size(first(residues))),e=zeros(size(d)))
    PlanarRationalModel(poles,residues,copy(d),copy(e),[0.,1.],0.,0.,false,false,:none,NaN,0.,0.)
end
function exercise(root,output)
    @assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root) && (output===nothing || !isfile(output))
    result=@testset "Exact rational evaluation preserves finite sums and denominator components" begin
        M=floatmax(Float64);tiny=nextfloat(0.)
        for n in (1,2),magnitude in (tiny,floatmin(Float64),1.,M),sign in (-1.,1.)
            # Independent closed forms: equal pole terms combine before
            # division, without reproducing the evaluation loop.
            R=fill(ComplexF64(sign*magnitude),n,n)
            cancellation=planted(fill(ComplexF64(-1),3),[copy(R),copy(R),-R])
            saved=deepcopy(cancellation.residues),copy(cancellation.d),copy(cancellation.e)
            actual=@inferred planar_rational_eval(cancellation,0.)
            @test actual==R
            @test (cancellation.residues,cancellation.d,cancellation.e)==saved
            half=planted(fill(ComplexF64(-2),2),[copy(R),copy(R)])
            @test (@inferred planar_rational_eval(half,0.))==R
        end
        for sign in (-1.,1.)
            p=complex(-1.,M);r=complex(0.,M)
            pair=planted(ComplexF64[p,conj(p)],[fill(r,1,1),fill(conj(r),1,1)])
            f=sign*M/(2pi);s=2pi*1im*f
            X=Complex{Rational{BigInt}}
            # Conjugate-pair closed numerator/denominator, independent
            # of production's list of exact inverse poles.
            z=X(s)-Rational{BigInt}(real(p))
            expected=ComplexF64((-2Rational{BigInt}(imag(r))*Rational{BigInt}(imag(p)))/
                (z*z+Rational{BigInt}(imag(p))^2))
            @test only(planar_rational_eval(pair,f))==expected
            @test real(expected)==-.5 && abs(imag(expected))==M
        end
        finite=planted(ComplexF64[-1.],[fill(ComplexF64(M),1,1)])
        overflow=planted(ComplexF64[-1.,-1.],[fill(ComplexF64(M),1,1),fill(ComplexF64(M),1,1)])
        @test isfinite(only(planar_rational_eval(finite,0.)))
        @test isinf(real(only(planar_rational_eval(overflow,0.))))
        subnormal=planted(ComplexF64[-2.,-2.],[fill(ComplexF64(tiny),1,1),fill(ComplexF64(tiny),1,1)])
        for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,bits) do
                setrounding(BigFloat,mode) do
                    @test only(planar_rational_eval(subnormal,0.))==tiny
                    @test precision(BigFloat)==bits && rounding(BigFloat)===mode
                end
            end
        end
        @test only(fetch(Threads.@spawn planar_rational_eval(subnormal,0.)))==tiny
        @test_throws ArgumentError planar_rational_eval(subnormal,floatmax(BigFloat))
        @test_throws ArgumentError planar_rational_eval(subnormal,Inf)
    end
    counts=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io;TOML.print(io,Dict("version"=>string(VERSION),"passes"=>counts.passes+counts.cumulative_passes,"fails"=>counts.fails+counts.cumulative_fails,"errors"=>counts.errors+counts.cumulative_errors),sorted=true);end
end
length(ARGS)==2 ? exercise(ARGS...) : exercise(normpath(joinpath(@__DIR__,"..")),nothing)
end
