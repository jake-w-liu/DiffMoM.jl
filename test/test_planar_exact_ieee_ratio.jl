module ExactIEEERatioTests
using DiffMoM,Test,TOML
root,output=length(ARGS)==2 ? (realpath(ARGS[1]),abspath(ARGS[2])) : (realpath(joinpath(@__DIR__,"..")),nothing)
@assert realpath(dirname(dirname(pathof(DiffMoM))))==root && (output===nothing || !isfile(output))
function exercise()
    result=@testset "Exact rational IEEE nearest rounding independent of caller MPFR state" begin
        scratch=DiffMoM._vf_ratio_scratch();X=Rational{BigInt}
        values=[0.,nextfloat(0.),floatmin(Float64),1.,2.,floatmax(Float64)]
        for value in values,sign in (-1,1)
            x=sign*X(value)
            @test DiffMoM._vf_float_ratio(numerator(x),denominator(x),scratch)==sign*value
        end
        for low in (0.,nextfloat(0.),prevfloat(floatmin(Float64)),prevfloat(1.),1.,2.,prevfloat(floatmax(Float64)))
            high=nextfloat(low);mid=(X(low)+X(high))/2
            # IEEE ties-to-even is determined from the stored low bit.
            even_low=iseven(reinterpret(UInt64,low));expected=even_low ? low : high
            gap=X(high)-X(low);below=mid-gap/4;above=mid+gap/4
            for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
                setprecision(BigFloat,bits) do
                    setrounding(BigFloat,mode) do
                        @test DiffMoM._vf_float_ratio(numerator(mid),denominator(mid),scratch)==expected
                        @test DiffMoM._vf_float_ratio(numerator(below),denominator(below),scratch)==low
                        @test DiffMoM._vf_float_ratio(numerator(above),denominator(above),scratch)==high
                        @test precision(BigFloat)==bits && rounding(BigFloat)===mode
                    end
                end
            end
        end
        top=X(floatmax(Float64));step=top-X(prevfloat(floatmax(Float64)))
        for value in (top+step/2,top+step)
            @test isinf(DiffMoM._vf_float_ratio(numerator(value),denominator(value),scratch))
        end
    end
    c=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io;TOML.print(io,Dict("version"=>string(VERSION),"passes"=>c.passes+c.cumulative_passes,"fails"=>c.fails+c.cumulative_fails,"errors"=>c.errors+c.cumulative_errors),sorted=true);end
end
exercise()

end
