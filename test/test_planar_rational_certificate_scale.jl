module CertificateScaleTests
using DiffMoM,LinearAlgebra,Test,TOML
function planted(amplitude,frequency_scale;active=true)
    poles=frequency_scale.*ComplexF64[-.001+10im,-.001-10im]
    sign=active ? -1 : 1
    residues=[fill(complex(sign*.01*amplitude*frequency_scale),1,1) for _ in eachindex(poles)]
    PlanarRationalModel(poles,residues,fill(amplitude,1,1),zeros(1,1),[.1,10.],NaN,NaN,false,false,:none,NaN,0.,0.)
end
function crossings_reference(model)
    # Re(Y(jw))=D+R*a*[1/(a²+(w-w0)²)+1/(a²+(w+w0)²)].
    # Multiplying the denominators gives a quadratic in x=w².
    setprecision(BigFloat,2precision(Float64)) do
        setrounding(BigFloat,RoundNearest) do
            a=-BigFloat(real(model.poles[1]));w0=abs(BigFloat(imag(model.poles[1])))
            d=BigFloat(only(model.d));r=BigFloat(real(only(model.residues[1])))
            h=a^2+w0^2;q=2r*a/d
            b=2(a^2-w0^2)+q;c=h^2+q*h
            discriminant=sqrt(b^2-4c)
            Float64.([sqrt((-b-discriminant)/2)/(2BigFloat(pi)),sqrt((-b+discriminant)/2)/(2BigFloat(pi))])
        end
    end
end
function exercise(root,output)
    @assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root)
    @assert output===nothing || !isfile(output)
    edge=exponent(floatmax(Float64))-precision(Float64)
    result=@testset "Rational certificate preserves finite admittance and frequency scaling" begin
        for (amplitude,frequency_scale) in ((1.,1.),(ldexp(1.,edge),1.),(ldexp(1.,-edge),1.),
                (ldexp(1.,-exponent(floatmax(Float64))),1.),(1.,ldexp(1.,edge)),(1.,ldexp(1.,-edge)))
            active=planted(amplitude,frequency_scale)
            saved=(copy(active.poles),deepcopy(active.residues),copy(active.d),copy(active.e))
            actual=@inferred planar_rational_certificate(active)
            reference=crossings_reference(active)
            @test !actual.certified && actual.method===:hamiltonian
            @test length(actual.crossings_hz)==length(reference)
            @test isapprox(actual.crossings_hz,reference;rtol=1e-8,atol=0)
            @test (active.poles,active.residues,active.d,active.e)==saved
            passive=planted(amplitude,frequency_scale;active=false)
            @test planar_rational_certificate(passive).certified
            @test_throws ArgumentError planar_rational_certificate(active;max_bytes=0)
        end
        d=floatmax(Float64)
        model=PlanarRationalModel(ComplexF64[-1],[zeros(ComplexF64,1,1)],fill(d,1,1),zeros(1,1),[.1,10.],NaN,NaN,false,false,:none,NaN,0.,0.)
        @test (@inferred planar_rational_certificate(model)).certified
        for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,bits) do
                setrounding(BigFloat,mode) do
                    model=planted(ldexp(1.,edge),1.)
                    @test !planar_rational_certificate(model).certified
                    @test precision(BigFloat)==bits && rounding(BigFloat)===mode
                end
            end
        end
    end
    counts=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io
        TOML.print(io,Dict("version"=>string(VERSION),"threads"=>Threads.nthreads(),"passes"=>counts.passes+counts.cumulative_passes,"fails"=>counts.fails+counts.cumulative_fails,"errors"=>counts.errors+counts.cumulative_errors))
    end
end
length(ARGS)==2 ? exercise(ARGS...) : exercise(normpath(joinpath(@__DIR__,"..")),nothing)
end
