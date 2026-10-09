module SampledPassivityRangeTests
using DiffMoM, LinearAlgebra, Test, TOML

function constant_model(d)
    n=size(d,1)
    PlanarRationalModel(ComplexF64[],Matrix{ComplexF64}[],copy(d),zeros(n,n),
        [0.,1.],0.,0.,false,false,:none,NaN,0.,0.)
end

function independent_margin(d)
    # The real symmetric 2x2 spectrum has a closed-form characteristic
    # equation. Use extra precision only in the independent reference.
    setprecision(BigFloat,2precision(Float64)) do
        setrounding(BigFloat,RoundNearest) do
            if size(d,1)==1
                return only(d)
            end
            a,b,c,e=BigFloat(d[1,1]),BigFloat(d[1,2]),BigFloat(d[2,1]),BigFloat(d[2,2])
            middle=(a+e)/2;off=(b+c)/2
            Float64(middle-sqrt(((a-e)/2)^2+off^2))
        end
    end
end

function exercise(root,output)
    @assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root)
    @assert output===nothing || !isfile(output)
    result=@testset "Sampled passivity retains finite Hermitian means across stored range" begin
        for scale in (1.,floatmax(Float64),nextfloat(0.),ldexp(1.,-exponent(floatmax(Float64))))
            for sign in (-1.,1.)
                d=fill(sign*scale,1,1);model=constant_model(d);saved=copy(model.d),copy(model.e)
                actual=@inferred planar_rational_passivity(model,[0.,1.])
                expected=independent_margin(d)
                @test isfinite(actual.margin) && actual.margin==expected
                @test actual.passive==(expected>=-1e-9)
                @test actual.frequencies==[0.,1.]
                @test (model.d,model.e)==saved
            end
        end
        for sign in (-1.,1.),scale in (1.,floatmax(Float64))
            d=sign.*[(scale/4)*3 scale/8;scale/8 scale/2];model=constant_model(d)
            actual=@inferred planar_rational_passivity(model,[0.,1.])
            @test isfinite(actual.margin)
            @test isapprox(actual.margin,independent_margin(d);rtol=1e-9,atol=0)
            @test actual.passive==(sign>0)
        end
        # A real skew affine term gives a Hermitian imaginary off-diagonal.
        # Its exact two eigenvalues are +/- angular-frequency*coefficient.
        coefficient=((floatmax(Float64)/4)*3)/(2pi)
        model=constant_model(zeros(2,2));model.e[1,2]=coefficient;model.e[2,1]=-coefficient
        actual=@inferred planar_rational_passivity(model,[1.])
        @test isfinite(actual.margin) && isapprox(actual.margin,-(2pi*coefficient);rtol=1e-9,atol=0)
        @test !actual.passive
        @test all(isfinite,planar_rational_eval(model,1.))
        for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,bits) do
                setrounding(BigFloat,mode) do
                    actual=planar_rational_passivity(constant_model(fill(floatmax(Float64),1,1)),[0.])
                    @test actual.margin==floatmax(Float64)
                    @test precision(BigFloat)==bits && rounding(BigFloat)===mode
                end
            end
        end
        @test_throws ArgumentError planar_rational_passivity(constant_model(ones(1,1)),Float64[])
        @test_throws ArgumentError planar_rational_passivity(constant_model(ones(1,1)),[Inf])
    end
    counts=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io
        TOML.print(io,Dict("version"=>string(VERSION),"passes"=>counts.passes+counts.cumulative_passes,
            "fails"=>counts.fails+counts.cumulative_fails,"errors"=>counts.errors+counts.cumulative_errors))
    end
end
length(ARGS)==2 ? exercise(ARGS...) : exercise(normpath(joinpath(@__DIR__,"..")),nothing)
end
