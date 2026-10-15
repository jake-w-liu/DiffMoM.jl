module PublicPowerProductRangeTests
using DiffMoM,LinearAlgebra,Test,TOML
function reference_power(v,i)
    total=zero(Rational{BigInt})
    for p in eachindex(v,i)
        total+=Rational{BigInt}(real(v[p]))*Rational{BigInt}(real(i[p]))+
            Rational{BigInt}(imag(v[p]))*Rational{BigInt}(imag(i[p]))
    end
    Float64(total/2)
end
function exercise(root,output)
    @assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root)
    @assert output===nothing || !isfile(output)
    edge=exponent(floatmax(Float64))÷2+precision(Float64)
    amplitude=ldexp(1.0,edge);z=ldexp(1.0,-edge)
    fixtures=(
        (fill(complex(amplitude),2),ComplexF64[im*amplitude,-im*amplitude]),
        (fill(complex(amplitude),2),ComplexF64[amplitude,-amplitude]),
        # Overflow cancellation leaves a finite, nonzero physical power.
        (ComplexF64[amplitude,amplitude,1],ComplexF64[amplitude,-amplitude,1]),
        # The real dot itself overflows but its physical half fits.
        (ComplexF64[ldexp(1.0,cld(exponent(floatmax(Float64)),2))],
            ComplexF64[ldexp(1.0,cld(exponent(floatmax(Float64)),2))]),
        (ComplexF64[1,1],ComplexF64[im,-im]),
    )
    rows=[]
    result=@testset "Public power products preserve finite accepted power" begin
        for (fixture,(v,i)) in enumerate(fixtures)
            saved_v,saved_i=copy(v),copy(i)
            expected=reference_power(v,i)
            wave_reference=setprecision(BigFloat,2precision(Float64)) do
                vv,ii=Complex{BigFloat}.(v),Complex{BigFloat}.(i)
                zz=BigFloat(z)
                (a=ComplexF64.((vv+zz.*ii)./(2sqrt(zz))),b=ComplexF64.((vv-zz.*ii)./(2sqrt(zz))))
            end
            # Public waves must fit; fixtures derive all scales from Float64.
            @test all(isfinite,wave_reference.a) && all(isfinite,wave_reference.b)
            value=@inferred planar_power_waves(v,i;z0=z)
            @test value.accepted_power==expected
            @test isapprox(value.a,wave_reference.a;rtol=3e-14,atol=0)
            @test isapprox(value.b,wave_reference.b;rtol=3e-14,atol=0)
            @test v==saved_v && i==saved_i
            @test planar_power_waves(view(v,:),view(i,:);z0=z).accepted_power==expected
            # The public bound charges six ComplexF64 vectors. Reactive
            # fixtures 1 and 5 have no nonzero real-coordinate products,
            # so their exact zero power needs no MPFR recovery even when
            # the unused imaginary dot overflows and BLAS returns NaN.
            retained=6sizeof(ComplexF64)*length(v)
            @test_throws ArgumentError planar_power_waves(v,i;z0=z,max_bytes=retained-1)
            if fixture in (2,3,4)
                bits=DiffMoM._planar_power_product_precision(length(v))
                # Recovery owns four simultaneously live MPFR scalars.
                required=retained+4Base.summarysize(BigFloat(0.;precision=bits))
                @test_throws ArgumentError planar_power_waves(v,i;z0=z,max_bytes=required-1)
                @test planar_power_waves(v,i;z0=z,max_bytes=required).accepted_power==expected
            else
                @test planar_power_waves(v,i;z0=z,max_bytes=retained).accepted_power==expected
            end
            push!(rows,Dict("ports"=>length(v),"accepted_power"=>value.accepted_power,
                "raw_dot_real_finite"=>isfinite(real(dot(v,i)))))
        end
        # Recovery must reject an actual physical power outside storage range.
        v=ComplexF64[amplitude];i=ComplexF64[amplitude]
        @test_throws ArgumentError planar_power_waves(v,i;z0=z)
        for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,bits) do
                setrounding(BigFloat,mode) do
                    v,i=fixtures[1]
                    @test planar_power_waves(v,i;z0=z).accepted_power==0
                    @test precision(BigFloat)==bits
                    @test rounding(BigFloat)===mode
                end
            end
        end
    end
    counts=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io
        TOML.print(io,Dict("version"=>string(VERSION),"root"=>realpath(root),"passes"=>counts.passes,
            "fails"=>counts.fails,"errors"=>counts.errors,"rows"=>rows))
    end
end
length(ARGS)==2 ? exercise(ARGS...) : exercise(normpath(joinpath(@__DIR__,"..")),nothing)
end
