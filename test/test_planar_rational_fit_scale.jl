module RationalFitScaleTests
using DiffMoM,LinearAlgebra,SparseArrays,Test

@testset "Rational fitting retains amplitude homogeneity and finite RMS statistics" begin
    # Original physical one-pole fixture and accuracy gates.
    fs=collect(range(1e8,1e10;length=51))
    D=[0.02 -0.005;-0.005 0.03]
    R=[1e8 -2e7;-2e7 8e7]
    E=[1e-12 -2e-13;-2e-13 2e-12]
    caller_bits=precision(BigFloat)
    amplitudes=(1.0,sqrt(eps(Float64)),eps(Float64),eps(Float64)^2,
        ldexp(1.0,exponent(floatmin(Float64))÷2-precision(Float64)),
        ldexp(1.0,exponent(floatmax(Float64))÷2+precision(Float64)))
    for amplitude in amplitudes,kind in (:zero,:constant,:affine,:rational)
        series=[amplitude*(kind===:zero ? zeros(ComplexF64,2,2) : kind===:constant ? ComplexF64.(D) :
            kind===:affine ? D+2pi*1im*f*E : D+2pi*1im*f*E+R/(2pi*1im*f+2e9)) for f in fs]
        model=planar_fit_rational(series,fs;order=1,enforce_passivity=false)
        predicted=[planar_rational_eval(model,f) for f in fs]
        rms,relative=setprecision(BigFloat,2precision(Float64)) do
            num=sum(sum(abs2,Complex{BigFloat}.(a)-Complex{BigFloat}.(b)) for (a,b) in zip(predicted,series))
            den=sum(sum(abs2,Complex{BigFloat}.(Y)) for Y in series)
            absolute=sqrt(num/(length(fs)*length(D)))
            Float64(absolute),Float64(iszero(den) ? absolute : sqrt(num/den))
        end
        @test isfinite(model.rms_error) && isapprox(model.rms_error,rms)
        @test isfinite(model.relative_rms_error) && isapprox(model.relative_rms_error,relative)
        @test relative<1e-10
        @test all(Y->all(isfinite,Y),predicted)
        if kind===:rational
            @test isapprox(model.poles[1],-2e9;rtol=1e-8)
        end
    end
    @test precision(BigFloat)==caller_bits
end

@testset "Rational fitting checks dense and wide Y/S storage before model conversion" begin
    fs=collect(range(1e8,1e10;length=51))
    tiny=BigFloat(nextfloat(0.0))/2
    huge=BigFloat(floatmax(Float64))*2
    for format in (:y,:s)
        for value in (complex(tiny,BigFloat(0)),complex(BigFloat(.02),tiny),complex(huge,BigFloat(0)))
            series=[fill(value,1,1) for f in fs]
            calls=Ref(0);reference(f)=(calls[]+=1;50.0)
            @test_throws ArgumentError planar_fit_rational(series,fs;format,z0=reference,order=1,enforce_passivity=false)
            @test calls[]==0
        end
        for matrix in (fill(.02,1,1),fill(Float32(.02),1,1),fill(BigFloat(.02),1,1),
                sparse(fill(.02,1,1)),view(fill(.02,1,1),:,:))
            series=fill(matrix,length(fs))
            model=planar_fit_rational(series,fs;format,order=1,enforce_passivity=false)
            value=Float64(only(matrix))
            expected=format===:y ? value : (1-value)/(50.0*(1+value))
            @test isapprox(only(planar_rational_eval(model,first(fs))),expected)
            @test all(isfinite,model.d) && all(isfinite,model.e) && all(isfinite,model.poles)
        end
        calls=Ref(0);reference(f)=(calls[]+=1;50.0)
        @test_throws ArgumentError planar_fit_rational([fill(.02,1,1) for f in fs],fs;
            format,z0=reference,order=1,enforce_passivity=false,max_bytes=0)
        @test calls[]==0
    end
end

@testset "Rational fitting preserves finite slow decay without a damping floor" begin
    fs=collect(range(1e8,1e10;length=51))
    D=[0.02 -0.005;-0.005 0.03]
    R=[1e8 -2e7;-2e7 8e7]
    E=[1e-12 -2e-13;-2e-13 2e-12]
    for normalized_decay in (cbrt(eps(Float64)),sqrt(eps(Float64)),eps(Float64))
        decay=normalized_decay*2pi*last(fs)
        series=[D+2pi*1im*f*E+R/(2pi*1im*f+decay) for f in fs]
        model=planar_fit_rational(series,fs;order=1,enforce_passivity=false)
        actual=[planar_rational_eval(model,f) for f in fs]
        relative=sqrt(sum(sum(abs2,a-b) for (a,b) in zip(actual,series))/sum(sum(abs2,Y) for Y in series))
        @test relative<1e-10
        @test all(p->isfinite(p) && real(p)<0,model.poles)
        @test isapprox(model.relative_rms_error,relative)
    end
end
end
