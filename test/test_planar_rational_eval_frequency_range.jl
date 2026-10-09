module RationalEvaluationFrequencyRangeTests
using DiffMoM,LinearAlgebra,Test
function range_model(kind)
    poles=kind===:pole ? ComplexF64[-1] : ComplexF64[]
    residues=kind===:pole ? [ones(ComplexF64,1,1)] : Matrix{ComplexF64}[]
    d=kind===:pole ? zeros(1,1) : ones(1,1)
    e=fill(kind===:capacitor ? nextfloat(0.0) : 0.0,1,1)
    PlanarRationalModel(poles,residues,d,e,Float64[0,1],0.,0.,true,true,:positive_residues,0.,0.,0.)
end
function promoted_reference(model,f)
    setprecision(BigFloat,2precision(Float64)) do
        setrounding(BigFloat,RoundNearest) do
            s=Complex{BigFloat}(0,BigFloat(2pi)*BigFloat(f))
            value=model.d .+ s .* model.e
            for k in eachindex(model.poles)
                value .+= model.residues[k] ./ (s-model.poles[k])
            end
            ComplexF64.(value)
        end
    end
end
coordinate_match(actual,expected)=all(isapprox(x,y;rtol=1e-9,atol=0)
    for (a,b) in zip(actual,expected) for (x,y) in ((real(a),real(b)),(imag(a),imag(b))))
@testset "Rational evaluation preserves finite responses beyond stored angular-frequency range" begin
    for kind in (:constant,:capacitor,:pole)
        model=range_model(kind)
        for f in (1.0,floatmax(Float64),-floatmax(Float64))
            saved=copy(model.poles),deepcopy(model.residues),copy(model.d),copy(model.e)
            expected=promoted_reference(model,f)
            actual=@inferred planar_rational_eval(model,f)
            @test planar_rational_certificate(model).certified
            @test all(isfinite,expected) && all(isfinite,actual)
            @test coordinate_match(actual,expected)
            @test (model.poles,model.residues,model.d,model.e)==saved
        end
    end
    model=range_model(:capacitor);expected=promoted_reference(model,floatmax(Float64))
    for bits in (precision(Float64),2precision(Float64)), mode in (RoundNearest,RoundDown,RoundUp)
        setprecision(BigFloat,bits) do
            setrounding(BigFloat,mode) do
                actual=planar_rational_eval(model,floatmax(Float64))
                @test coordinate_match(actual,expected)
                @test precision(BigFloat)==bits
                @test rounding(BigFloat)==mode
            end
        end
    end
    # MPFR's own exponent range is an explicit storage boundary; a
    # nonrecursive fallback rejects it without repeatedly calling itself.
    largest=floatmax(BigFloat)
    @test isfinite(largest) && !isfinite(2pi*1im*largest)
    @test_throws ArgumentError planar_rational_eval(range_model(:constant),largest)
end
end
