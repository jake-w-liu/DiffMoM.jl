module ConstantCertificateTests
using DiffMoM,LinearAlgebra,Test,TOML
function model(d,e=zeros(size(d));decay=false)
    poles=decay ? ComplexF64[-1.] : ComplexF64[]
    residues=decay ? [zeros(ComplexF64,size(d))] : Matrix{ComplexF64}[]
    PlanarRationalModel(poles,residues,copy(d),copy(e),[0.,1.],
        0.,0.,false,false,:none,NaN,0.,0.)
end
function exercise(root,output)
    @assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root)
    @assert output===nothing || !isfile(output)
    result=@testset "Rational certificate preserves constant nonreciprocal energy and feedthrough signs" begin
        for conductance in (0.,1.,floatmax(Float64)),capacitive in (false,true)
            d=[conductance 1.;-1. conductance]
            e=capacitive ? Matrix{Float64}(I,2,2) : zeros(2,2)
            value=model(d,e);saved=copy(value.d),copy(value.e)
            actual=@inferred planar_rational_certificate(value)
            @test actual.certified
            @test actual.method===:uniform_bound
            @test isempty(actual.crossings_hz)
            @test (value.d,value.e)==saved
        end
        for decay in (false,true),sign in (-1.,1.)
            # The first coordinate is an exact signed energy direction.
            # Normalization by the second must not erase that sign.
            value=model([sign*nextfloat(0.) 0.;0. floatmax(Float64)];decay)
            saved=copy(value.d)
            actual=@inferred planar_rational_certificate(value)
            @test actual.certified==(sign>0)
            @test value.d==saved
        end
        for sign in (-1.,1.)
            d=[sign*nextfloat(0.) 1.;-1. floatmax(Float64)]
            @test planar_rational_certificate(model(d)).certified==(sign>0)
        end
        for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,bits) do
                setrounding(BigFloat,mode) do
                    @test planar_rational_certificate(model([0. 1.;-1. 0.])).certified
                    @test precision(BigFloat)==bits && rounding(BigFloat)===mode
                end
            end
        end
    end
    counts=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io
        TOML.print(io,Dict("version"=>string(VERSION),"passes"=>counts.passes+counts.cumulative_passes,
            "fails"=>counts.fails+counts.cumulative_fails,"errors"=>counts.errors+counts.cumulative_errors))
    end
end
length(ARGS)==2 ? exercise(ARGS...) : exercise(normpath(joinpath(@__DIR__,"..")),nothing)
end
