module NativeSonnetFunctionsTests
using Test,DiffMoM,LinearAlgebra,TOML,SHA,JSON

function crc(bytes)
    value=typemax(UInt32)
    for byte in bytes
        value⊻=UInt32(byte)
        for _ in 1:8;value=isodd(value) ? (value>>1)⊻0xedb88320 : value>>1;end
    end
    return ~value
end

@testset "Actual native mathematical function and branch corpus" begin
    fixture=joinpath(@__DIR__,"fixtures/native_sonnet_scalar_functions")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
        @test string(crc(bytes);base=16,pad=8)==file["crc32"]
    end
    provenance=JSON.parsefile(joinpath(fixture,"provenance.json"))
    @test provenance["jobs"]==210 && provenance["matrices"]==316
    grid=CellGrid(.001,.001,20,20);sheet=sheet_level(1,20,20)
    sheet.mask[:,9:12].=true;sheet.connect_west[9:12].=true;sheet.connect_east[9:12].=true
    stack=PlanarStackup([PlanarLayer(1.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
    handprob=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:west,9:12,50.),PlanarPort(1,:east,9:12,50.)])
    rhs=zeros(ComplexF64,length(handprob.basis.kind),2)
    for b in eachindex(handprob.basis.kind)
        q=handprob.basis.port[b];q==0 && continue
        rhs[b,q]=(q==1 ? -1 : 1)*handprob.basis.width[b]
    end
    for artifact in provenance["artifacts"]
        report=TOML.parsefile(joinpath(fixture,artifact,"comparison.toml"))
        for record in report["runs"]
            tag=record["tag"];dir=joinpath(fixture,artifact,tag)
            p=read_sonnet_project(joinpath(dir,tag*".son"))
            metadata=TOML.parsefile(joinpath(dir,"metadata.toml"))
            @test metadata["source_sha256"]==bytes2hex(sha256(read(p.source)))
            @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
            @test metadata["actual_cell_counts"]==[20,20]
            @test metadata["touchstone_selected_log_checks"]["native.s2p"]["status"]=="PASS"
            haskey(p.variables,"Loss") || continue
            original=deepcopy(p.variables)
            errors=read(joinpath(dir,"engine_stderr.log"),String)
            if occursin("Bad Equation",errors) || occursin("Invalid entry",errors)
                @test_throws ArgumentError sonnet_variable_value(p,"Loss")
                @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true,grid=(typemax(Int),typemax(Int)))
                continue
            end
            printed=match(r"Loss = ([-+0-9.eE]+) Ohms/sq",read(joinpath(dir,"engine_stdout.log"),String))
            @test printed!==nothing
            expected=parse(Float64,printed.captures[1])
            value=sonnet_variable_value(p,"Loss")
            # Native variable logging prints six significant decimal digits;
            # full S below uses the independent high precision native export.
            @test isapprox(value,expected;rtol=6e-6,atol=5e-7)
            if value<0
                @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true,grid=(typemax(Int),typemax(Int)))
                continue
            end
            native=planar_read_touchstone(joinpath(dir,"native.s2p"))
            for (f,s) in zip(native.frequencies,native.s)
                actual=solve_sonnet_project(p,f;raw=true,mx=160,my=160,method=:dense_fft)
                @test maximum(abs,actual.s-s)<.005
                @test norm(actual.raw.z_mom*actual.raw.currents-rhs)/norm(rhs)<1e-9
                @test opnorm(actual.s)<=1+1e-10
                @test only(actual.raw.problem.sheets).mask==sheet.mask
                rho=value*1e-6
                zs=if iszero(rho)
                    0.
                else
                    skin=(1+im)*sqrt(pi*f*(4pi*1e-7)*rho)
                    (5/9)*skin/tanh((5/9)*(skin/rho)*1e-6)
                end
                hand=solve_planar(handprob,f;method=:dense_fft,mx=160,my=160,surface_zs=zs)
                @test maximum(abs,actual.s-hand.s)<2e-10
                @test actual.raw.currents≈hand.currents rtol=1e-8
                a=only(planar_current_maps(actual;voltages=ComplexF64[.6+.2im,-.3+.4im]))
                b=only(planar_current_maps(hand;voltages=ComplexF64[.6+.2im,-.3+.4im]))
                @test hypot(norm(a.jx-b.jx),norm(a.jy-b.jy))<1e-8*hypot(norm(b.jx),norm(b.jy))
            end
            @test p.variables==original
        end
    end
end

@testset "Native scalar grammar, real physical boundary, and provider order" begin
    p=read_sonnet_project(joinpath(@__DIR__,"fixtures/native_sonnet_scalar_logarithms/ln_positive/ln_positive.son"))
    for (text,value) in (("2^3^2",64.),("(2^3)^2",64.),("2^(3^2)",512.),
            ("2^3^2^1",64.),("2^-3^2",1/512), ("-2^2",-4.),("(-2)^2",4.),
            ("2^(-3)^2",1/64),("2^3*2",16.),("4+int(-2.9)",2.),
            ("4+fmod(-5,3)",2.),("fmod(5,-3)",2.),("atan2(-0,-1)",pi),
            ("deg(-1)",-180.),("deg(cmplx(-1,0))",180.),("deg(cmplx(-1,-0))",-180.),
            ("imag(sqrt(-4))",2.),("imag(asin(2))",-acosh(2.)),
            ("imag(acos(2))",acosh(2.)),("imag(acosh(-2))",pi),
            ("imag(atanh(2))",-pi/2),("deg(cos(pi))",-180.),
            ("deg(sin(-pi/2))",180.),("deg(atan(-1))",180.),
            ("deg(asinh(-1))",180.),("deg(sinh(-1))",-180.),("deg(tanh(-1))",-180.),
            ("hypot(cmplx(3,4),4)",sqrt(41.)),("hypot(cmplx(1,1),2)",sqrt(6.)),
            ("int(cmplx(2.9,4))",2.),("int(cmplx(1,1))",1.),
            ("imag(cmplx(cmplx(3,4),5))",9.),
            ("real(cmplx(cmplx(3,4),cmplx(1,2)))",1.),
            ("fmod(cmplx(5,2),cmplx(3,4))",sqrt(29.)-5),
            ("fmod(cmplx(-5,2),cmplx(3,4))",5-sqrt(29.)),
            ("fmod(cmplx(0,-5),3)",2.),("fmod(cmplx(-0,5),3)",2.),
            ("min(cmplx(3,4),4)",4.),("max(cmplx(-3,4),2)",2.),
            ("min(cmplx(-3,4),4)",-3.),
            ("imag(max(cmplx(3,4),cmplx(3,-4)))",-4.),
            ("imag(min(cmplx(3,4),cmplx(3,-4)))",-4.),
            ("deg(int(-.2))",0.),("deg(fmod(-6,3))",0.),("deg(fmod(-0,3))",0.),
            ("real(atan2(cmplx(1,1),2))",real(atan((1+im)/2))),
            ("imag(atan2(cmplx(1,1),2))",imag(atan((1+im)/2))),
            ("real(atan2(-2,0))",pi/2),("imag(atan2(-2,0))",-2.),
            ("real(atan2(cmplx(0,3),0))",pi/2),
            ("imag(atan2(cmplx(0,3),0))",0.),
            ("h2p(cmplx(3e9,4e9))",3.),("imag(p2h(cmplx(3,4)))",0.),
            ("cmplx(1,2)",1.),("sqrt(-4)",0.),("asin(2)",pi/2),
            ("acosh(-2)",acosh(2.)),("atanh(2)",atanh(.5)))
        @test sonnet_variable_value(p,text)≈value rtol=2e-15
    end
    p.variables["Negative"]="-1"
    @test sonnet_variable_value(p,"deg(Negative)")==180.
    @test sonnet_variable_value(p,"Negative")==-1.
    @test sonnet_variable_value(p,join(fill("1",3000),"+"))==3000.
    for text in ("1>0","1<0","1>=0","1<=0","1==0","1!=0","1 ? 2 : 3",
            "if(1,2,3)","ifelse(1,2,3)","2^+3^2","fmod(2,0)","db10(0)","db20(0)",
            "real()","atan(1,2)","atan2(1)","hypot(1)","cmplx(1)","max(1,2,3)",
            "fmod(1,2,3)","int(1,2)","atan2(0,0)","atan2(cmplx(0,1),1)",
            "real(atan2(cmplx(0,1),1))","imag(atan2(cmplx(0,1),1))",
            "h2p(cmplx(1,exp(1000)))","p2h(cmplx(1,exp(1000)))",
            "m2p(cmplx(1,exp(1000)))","p2m(cmplx(1,exp(1000)))",
            "int(cmplx(1,exp(1000)))","max(1,exp(1000))","min(1,exp(1000))",
            "fmod(1,exp(1000))","hypot(1,exp(1000))","cmplx(1,exp(1000))",
            "table1(\"missing.csv\",cmplx(1,2))","log(2)","FORMAT(1)",
            "1e500","cmplx(1e-500,1)",repeat("-",3500)*"1",repeat("1-(",3500)*"1"*repeat(")",3500))
        @test_throws ArgumentError sonnet_variable_value(p,text)
    end
    dynamic=deepcopy(p);dynamic.variables["Loss"]="hypot(sin(FREQ/1e9),cos(FREQ/1e9))+real(cmplx(0,1)^2)+2"
    original=deepcopy(dynamic.variables)
    for f in (1e9,5.5e9,1e10)
        @test sonnet_variable_value(dynamic,"Loss";freq=f)≈2. rtol=2e-15
        literal=deepcopy(p);literal.variables["Loss"]="2"
        actual=solve_sonnet_project(dynamic,f;raw=true,mx=80,my=80,method=:dense_fft)
        hand=solve_sonnet_project(literal,f;raw=true,mx=80,my=80,method=:dense_fft)
        @test maximum(abs,actual.s-hand.s)<1e-10
        @test actual.raw.currents≈hand.raw.currents rtol=1e-8
    end
    @test dynamic.variables==original
    @test_throws ArgumentError sonnet_variable_value(p,"1\0+2")
end

@testset "Actual native real-axis phase normalization controls" begin
    fixture=joinpath(@__DIR__,"fixtures/native_sonnet_hyperbolic_phase")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
        @test string(crc(bytes);base=16,pad=8)==file["crc32"]
    end
    provenance=JSON.parsefile(joinpath(fixture,"provenance.json"))
    @test provenance["jobs"]==12 && provenance["matrices"]==24
    for row in TOML.parsefile(joinpath(fixture,only(provenance["artifacts"]),"comparison.toml"))["runs"]
        tag=row["tag"];dir=joinpath(fixture,only(provenance["artifacts"]),tag)
        p=read_sonnet_project(joinpath(dir,tag*".son"))
        haskey(p.variables,"Loss") || continue
        log=read(joinpath(dir,"engine_stdout.log"),String)
        expected=parse(Float64,match(r"Loss = ([-+0-9.eE]+) Ohms/sq",log).captures[1])
        @test sonnet_variable_value(p,"Loss")≈expected rtol=2e-15
        data=planar_read_touchstone(joinpath(dir,"native.s2p"))
        for (f,s) in zip(data.frequencies,data.s)
            solved=solve_sonnet_project(p,f;raw=true,mx=160,my=160,method=:dense_fft)
            @test maximum(abs,solved.s-s)<.005
            @test opnorm(solved.s)<=1+1e-10
        end
    end
end

@testset "Actual named and inline native finite complex quantity projection" begin
    fixture=joinpath(@__DIR__,"fixtures/native_sonnet_final_complex_material")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
        @test string(crc(bytes);base=16,pad=8)==file["crc32"]
    end
    provenance=JSON.parsefile(joinpath(fixture,"provenance.json"))
    @test provenance["jobs"]==12 && provenance["matrices"]==12
    dir=joinpath(fixture,only(provenance["artifacts"]))
    standards=Dict(value=>only(planar_read_touchstone(joinpath(dir,tag,"native.s2p")).s)
        for (tag,value) in (("literal0",0.),("literal3",3.),("literal4",4.),("literal5",5.)))
    for row in TOML.parsefile(joinpath(dir,"comparison.toml"))["runs"]
        tag=row["tag"];source=joinpath(dir,tag,tag*".son")
        p=read_sonnet_project(source);before=deepcopy(p.variables)
        expected=row["hypothesized_real_ohm_square"]
        expression=only(filter(metal->metal[1]=="Sheet",p.metals))[4]
        @test sonnet_variable_value(p,expression)==expected
        snapshot=sonnet_scalar_files(p)
        @test sonnet_variable_value(snapshot,expression)==expected
        @test isempty(read(joinpath(dir,tag,"engine_stderr.log"),String))
        native=only(planar_read_touchstone(joinpath(dir,tag,"native.s2p")).s)
        @test maximum(abs,native-standards[expected])==0.
        solved=solve_sonnet_project(p,1e9;raw=true,mx=160,my=160,method=:dense_fft)
        retained=solve_sonnet_project(snapshot.project,1e9;raw=true,scalar_files=snapshot,mx=160,my=160,method=:dense_fft)
        @test maximum(abs,solved.s-native)<.005
        @test maximum(abs,solved.s-retained.s)<1e-11
        @test solved.raw.currents≈retained.raw.currents rtol=1e-9
        rhs=zeros(ComplexF64,length(solved.raw.problem.basis.kind),2)
        for b in eachindex(solved.raw.problem.basis.kind)
            q=solved.raw.problem.basis.port[b];q==0 && continue
            rhs[b,q]=(q==1 ? -1 : 1)*solved.raw.problem.basis.width[b]
        end
        @test norm(solved.raw.z_mom*solved.raw.currents-rhs)/norm(rhs)<1e-9
        @test p.variables==before
    end
    p=read_sonnet_project(joinpath(dir,"literal3/literal3.son"))
    for text in ("cmplx(1,exp(1000))","cmplx(exp(1000),1)","cmplx(1e500,2)","cmplx(1e-500,2)")
        @test_throws ArgumentError sonnet_variable_value(p,text)
    end
end

@testset "Actual native complex binary operands and exact-zero phase" begin
    fixture=joinpath(@__DIR__,"fixtures/native_sonnet_complex_binary_functions")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
        @test string(crc(bytes);base=16,pad=8)==file["crc32"]
    end
    provenance=JSON.parsefile(joinpath(fixture,"provenance.json"))
    @test provenance["jobs"]==73 && provenance["matrices"]==72
    # Independently evaluated operands, not the quantized native log or the
    # original hypotheses (many of which were falsified by these controls).
    expected=Dict(
        "hypot_complex"=>sqrt(41.),"int_complex"=>2.,"int_complex_imag"=>0.,
        "fmod_complex"=>sqrt(29.)-5,"max_complex"=>3.,"min_complex"=>4.,
        "cmplx_nested_imag"=>9.,"cmplx_nested_real"=>1.,
        "max_negative"=>5.,"min_negative"=>2.,"max_both_negative"=>3.,"min_both_negative"=>2.,
        "max_tie_real"=>7.,"min_tie_real"=>1.,"max_tie_imag"=>1.,"min_tie_imag"=>1.,
        "fmod_negative_complex"=>5-sqrt(29.),"fmod_negative_real_complex"=>2.,
        "fmod_zero_imaginary"=>1.,"fmod_complex_negative_real"=>sqrt(29.)-3,
        "fmod_negative_reals"=>1.,"hypot_pure_imaginary"=>5.,
        "max_selected_imag"=>4.,"min_selected_imag"=>4.,"int_negative_complex"=>2.,
        "fmod_pure_imaginary"=>5.,"fmod_negative_pure_imaginary"=>5.,
        "fmod_negative_zero_real"=>5.,"fmod_imaginary_divisor"=>2.,"fmod_return_real"=>0.,
        "max_negative_complex"=>7.,"min_negative_complex"=>2.,"max_zero_imaginary_negative"=>4.,
        "max_complex_unequal_real"=>9.,"min_complex_unequal_real"=>2.,
        "max_pure_imaginary"=>5.,"min_pure_imaginary"=>3.,
        "max_negative_pure_imaginary"=>1.,"min_negative_pure_imaginary"=>3.,
        "max_negative_complex_negative_real"=>2.,"min_negative_complex_negative_real"=>3.,
        "max_same_signed_magnitude"=>1.,"min_same_signed_magnitude"=>1.,
        "int_zero_phase"=>180.,"fmod_zero_phase"=>180.,
        "fmod_negative_zero_phase"=>180.,"hypot_zero_phase"=>180.)
    for artifact in provenance["artifacts"]
        report=TOML.parsefile(joinpath(fixture,artifact,"comparison.toml"))
        for row in report["runs"]
            tag=row["tag"];dir=joinpath(fixture,artifact,tag)
            p=read_sonnet_project(joinpath(dir,tag*".son"))
            metadata=TOML.parsefile(joinpath(dir,"metadata.toml"))
            @test row["source_sha256"]==bytes2hex(sha256(read(p.source)))==metadata["source_sha256"]
            @test metadata["actual_cell_counts"]==[20,20]
            if tag=="atan2_complex"
                @test row["status"]=="UNVERIFIED"
                @test metadata["process_success"]==false
                @test occursin("18.53",metadata["em"])
                @test occursin("IMSL Math Function ERROR",read(joinpath(dir,"engine_stderr.log"),String))
                @test !isfile(joinpath(dir,"native.s2p"))
                @test_throws ArgumentError sonnet_variable_value(p,"Loss")
                continue
            end
            @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
            @test row["status"]=="SIMULATED"
            @test metadata["touchstone_selected_log_checks"]["native.s2p"]["status"]=="PASS"
            @test isempty(read(joinpath(dir,"engine_stderr.log"),String))
            expression=only(filter(metal->metal[1]=="Sheet",p.metals))[4]
            value=startswith(tag,"literal") ? row["hypothesized_real_ohm_square"] : expected[tag]
            original=deepcopy(p.variables)
            @test sonnet_variable_value(p,expression)≈value rtol=3e-15
            snapshot=sonnet_scalar_files(p)
            @test sonnet_variable_value(snapshot,expression)≈value rtol=3e-15
            if haskey(p.variables,"Loss")
                printed=match(r"Loss = ([-+0-9.eE]+) Ohms/sq",read(joinpath(dir,"engine_stdout.log"),String))
                @test printed!==nothing
                @test value≈parse(Float64,printed.captures[1]) rtol=6e-6 atol=5e-7
            end
            if value<0
                # Retain native negative-R fallback without accepting an
                # active material as a passive constitutive definition.
                @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true,grid=(typemax(Int),typemax(Int)))
                continue
            end
            data=planar_read_touchstone(joinpath(dir,"native.s2p"))
            @test data.frequencies==[1e9]
            native=only(data.s)
            @test native==reshape(complex.(row["s_real"],row["s_imag"]),2,2)
            solved=solve_sonnet_project(p,1e9;raw=true,mx=160,my=160,method=:dense_fft)
            retained=solve_sonnet_project(snapshot.project,1e9;raw=true,scalar_files=snapshot,mx=160,my=160,method=:dense_fft)
            @test maximum(abs,solved.s-native)<.005
            @test maximum(abs,solved.s-retained.s)<1e-11
            @test solved.raw.currents≈retained.raw.currents rtol=1e-9
            rhs=zeros(ComplexF64,length(solved.raw.problem.basis.kind),2)
            for b in eachindex(solved.raw.problem.basis.kind)
                q=solved.raw.problem.basis.port[b];q==0 && continue
                rhs[b,q]=(q==1 ? -1 : 1)*solved.raw.problem.basis.width[b]
            end
            @test norm(solved.raw.z_mom*solved.raw.currents-rhs)/norm(rhs)<1e-9
            @test opnorm(solved.s)<=1+1e-10
            maps=only(planar_current_maps(solved;voltages=ComplexF64[.6+.2im,-.3+.4im]))
            copy_maps=only(planar_current_maps(retained;voltages=ComplexF64[.6+.2im,-.3+.4im]))
            @test hypot(norm(maps.jx-copy_maps.jx),norm(maps.jy-copy_maps.jy))<1e-8*hypot(norm(maps.jx),norm(maps.jy))
            @test p.variables==original
        end
    end
end
end
