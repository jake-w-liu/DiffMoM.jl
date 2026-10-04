module NativeSonnetScalarMathTests
using Test,DiffMoM,LinearAlgebra,SHA,TOML

@testset "Actual native scalar math retains real material values and physical responses" begin
    root=joinpath(@__DIR__,"fixtures/native_sonnet_scalar_math")
    for family in ("trig","math")
        directory=joinpath(root,family)
        report=TOML.parsefile(joinpath(directory,"comparison.toml"))
        @test bytes2hex(sha256(read(joinpath(directory,"FunctionsandOperators.html"))))==report["primary_sha256"]
        literal=Dict{String,Matrix{ComplexF64}}()
        for row in report["runs"]
            @test row["status"]=="SIMULATED"
            tag=row["tag"];projectfile=joinpath(directory,tag,tag*".son")
            @test bytes2hex(sha256(read(projectfile)))==row["source_sha256"]
            metadata=TOML.parsefile(joinpath(directory,tag,"metadata.toml"))
            @test metadata["source_sha256"]==row["source_sha256"]
            @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
            native=planar_read_touchstone(joinpath(directory,tag,"native.s2p"))
            @test native.frequencies==[1e9]
            actual=only(native.s)
            @test actual==reshape(complex.(row["s_real"],row["s_imag"]),2,2)
            base=first(rsplit(tag,'_';limit=2))
            if endswith(tag,"_literal")
                literal[base]=actual
                continue
            end
            @test row["native_literal_gate_pass"]
            @test maximum(abs,actual-literal[base])<=report["literal_native_gate"]
            @test isempty(row["native_stderr"])
            p=read_sonnet_project(projectfile)
            @test sonnet_variable_value(p,"Loss")≈row["independent_literal_ohm_square"] rtol=1e-14
            @test any(line->occursin("Loss =",line),row["native_variable_lines"])
            result=solve_sonnet_project(p,1e9;raw=true,mx=160,my=160,method=:dense_fft)
            @test maximum(abs,result.s-actual)<.005
            rhs=zeros(ComplexF64,size(result.raw.currents))
            basis=result.raw.problem.basis
            for b in eachindex(basis.kind)
                port=basis.port[b];port==0 && continue
                rhs[b,port]=(port==1 ? -1 : 1)*basis.width[b]
            end
            @test norm(result.raw.z_mom*result.raw.currents-rhs)/norm(rhs)<1e-9
            @test opnorm(result.s)<=1+1e-10
        end
    end
end

@testset "Native scalar math validates arity, real physical outputs and domains" begin
    p=read_sonnet_project(joinpath(@__DIR__,"fixtures/native_sonnet_scalar_math/math/real_expression/real_expression.son"))
    for expression in ("asin()","acos(1,2)","atan2(1)","sinh(1,2)","asinh()",
            "min(1)","max(1,2,3)","fmod(1)","int(1,2)","hypot(1)",
            "cmplx(1)","real(1,2)","imag()","mag(1,2)","conj()","deg(1,2)","rad()")
        @test_throws ArgumentError sonnet_variable_value(p,expression)
    end
    @test sonnet_variable_value(p,"sqrt(cmplx(4,0))")==2.
    @test sonnet_variable_value(p,"real(cmplx(2,3)+cmplx(4,5))")==6.
    @test sonnet_variable_value(p,"imag(cmplx(2,3)*cmplx(4,5))")==22.
    @test sonnet_variable_value(p,"real(cmplx(2,3)/cmplx(4,5))")≈23/41
    # Actual inline NOR controls in test_planar_sonnet_functions retain
    # the real projection of finite complex quantities at this boundary.
    @test sonnet_variable_value(p,"cmplx(2,3)")==2.
    @test sonnet_variable_value(p,"sqrt(-1)")==0.
    @test sonnet_variable_value(p,"min(cmplx(2,3),4)")==2.
    # Native min retains its selected complex operand; final material
    # storage then projects real. The actual binary corpus proves this.
    @test sonnet_variable_value(p,"imag(min(cmplx(2,3),4))")==3.
    @test sonnet_variable_value(p,"max(cmplx(2,3),4)")==4.
    @test_throws ArgumentError sonnet_variable_value(p,"log(exp(2))")
    @test_throws ArgumentError sonnet_variable_value(p,"exp(1000)")
    p.variables["ComplexIntermediate"]="cmplx(3,4)"
    @test sonnet_variable_value(p,"abs(ComplexIntermediate)")==3.
    @test sonnet_variable_value(p,"abs(cmplx(3,4))")==5.
end
end
