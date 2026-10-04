using Test,DiffMoM,LinearAlgebra,TOML,SHA,JSON

function _native_log_crc(bytes)
    crc=typemax(UInt32)
    for byte in bytes
        crc⊻=UInt32(byte)
        for _ in 1:8;crc=isodd(crc) ? (crc>>1)⊻0xedb88320 : crc>>1;end
    end
    ~crc
end

@testset "Actual native magnitude logarithms and physical current response" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sonnet_scalar_logarithms")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
        @test string(_native_log_crc(bytes);base=16,pad=8)==file["crc32"]
    end
    reference=planar_read_touchstone(joinpath(fixture,"literal_sigma_5e5","native.s2p"))
    grid=CellGrid(.001,.001,20,20);sheet=sheet_level(1,20,20)
    sheet.mask[:,9:12].=true;sheet.connect_west[9:12].=true;sheet.connect_east[9:12].=true
    stack=PlanarStackup([PlanarLayer(1.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
    handprob=build_planar_problem(stack,grid,[sheet],[PlanarPort(1,:west,9:12,50.),PlanarPort(1,:east,9:12,50.)])
    rhs=zeros(ComplexF64,length(handprob.basis.kind),2)
    for b in eachindex(handprob.basis.kind)
        q=handprob.basis.port[b];q==0 && continue
        rhs[b,q]=(q==1 ? -1 : 1)*handprob.basis.width[b]
    end
    for tag in ("ln_positive","ln_negative","log10_positive","log10_negative")
        dir=joinpath(fixture,tag);p=read_sonnet_project(joinpath(dir,tag*".son"))
        original=deepcopy(p.variables);bytes=read(p.source)
        metadata=TOML.parsefile(joinpath(dir,"metadata.toml"))
        @test metadata["source_sha256"]==bytes2hex(sha256(bytes))
        @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
        @test metadata["actual_cell_counts"]==[20,20] && metadata["native_subsections"]==[38]
        @test metadata["touchstone_selected_log_checks"]["native.s2p"]["status"]=="PASS"
        @test occursin("Loss = 2 Ohms/sq",read(joinpath(dir,"engine_stdout.log"),String))
        native=planar_read_touchstone(joinpath(dir,"native.s2p"))
        @test native.frequencies==reference.frequencies==[1e9,1e10]
        for (i,f) in enumerate(native.frequencies)
            @test maximum(abs,native.s[i]-reference.s[i])==0.
            @test sonnet_variable_value(p,"Loss";freq=f)≈2. rtol=2e-15
            skin=(1+im)*sqrt(pi*f*(4pi*1e-7)/500000.)
            zs=(5/9)*skin/tanh((5/9)*500000.0*skin*1e-6)
            hand=solve_planar(handprob,f;method=:dense_fft,mx=160,my=160,surface_zs=zs)
            actual=solve_sonnet_project(p,f;raw=true,method=:dense_fft,mx=160,my=160)
            @test maximum(abs,actual.s-native.s[i])<.005
            @test maximum(abs,actual.s-hand.s)<2e-10
            @test norm(actual.raw.z_mom*actual.raw.currents-rhs)/norm(rhs)<1e-9
            @test actual.raw.currents≈hand.currents rtol=1e-8
            @test only(actual.raw.problem.sheets).mask==sheet.mask
            @test opnorm(actual.s)<=1+1e-10
            v=ComplexF64[.6+.2im,-.3+.4im]
            a=only(planar_current_maps(actual;voltages=v));b=only(planar_current_maps(hand;voltages=v))
            @test hypot(norm(a.jx-b.jx),norm(a.jy-b.jy))<1e-8*hypot(norm(b.jx),norm(b.jy))
        end
        @test p.variables==original && read(p.source)==bytes
    end
    invalid=read_sonnet_project(joinpath(fixture,"log_alias","log_alias.son"))
    @test occursin("Unknown function name starting with 'log'",read(joinpath(fixture,"log_alias","engine_stderr.log"),String))
    @test occursin("Loss = 0 Ohms/sq",read(joinpath(fixture,"log_alias","engine_stdout.log"),String))
    before=TOML.parsefile(joinpath(fixture,"comparison.toml"))
    bad=only(r for r in before["runs"] if r["tag"]=="log_alias")
    @test bad["current_reader_value"]==2.
    @test minimum(bad["native_literal_delta_s"])>.09
    @test_throws ArgumentError sonnet_variable_value(invalid,"Loss")
    err=try solve_sonnet_project(invalid,1e9;raw=true,grid=(typemax(Int),typemax(Int)));nothing catch e;e end
    @test err isa ArgumentError && occursin("unsupported Sonnet expression function log",sprint(showerror,err))
end

@testset "Native logarithm scalar domains, arity and frequency providers" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sonnet_scalar_logarithms","ln_positive")
    p=read_sonnet_project(joinpath(fixture,"ln_positive.son"))
    for (text,expected) in (("ln(1)",0.),("ln(-1)",0.),("ln(exp(2))",2.),("ln(-exp(2))",2.),
            ("log10(1)",0.),("log10(-1)",0.),("log10(-100)",2.),("log10(1e308)",308.),
            ("ln(1e-300)",log(1e-300)))
        @test sonnet_variable_value(p,text)≈expected rtol=2e-15
    end
    for text in ("log(2)","ln()","ln(2,missing)","log10()","log10(2,missing)",
            "ln(0)","log10(0)","ln(1e-500)","log10(1e500)")
        @test_throws ArgumentError sonnet_variable_value(p,text)
    end
    for value in (Inf,NaN,big"1e-500",big"1e500")
        @test_throws ArgumentError sonnet_variable_value(p,"ln(X)";variables=Dict("X"=>value))
    end
    @test sonnet_variable_value(p,"ln(X)";variables=Dict("X"=>-exp(2.)))≈2.
    dynamic=deepcopy(p);dynamic.variables["Loss"]="ln(-exp(1+FREQ/1000000000))"
    for f in (1e9,5.5e9,1e10)
        @test sonnet_variable_value(dynamic,"Loss";freq=f)≈1+f/1e9 rtol=2e-15
    end
    # Independent literal material providers produce identical native wrapper
    # matrices/currents for the same frequency-dependent real load.
    f=5.5e9;literal=deepcopy(p);literal.variables["Loss"]="6.5"
    actual=solve_sonnet_project(dynamic,f;raw=true,mx=80,my=80,method=:dense_fft)
    expected=solve_sonnet_project(literal,f;raw=true,mx=80,my=80,method=:dense_fft)
    @test maximum(abs,actual.s-expected.s)<1e-11
    @test actual.raw.currents≈expected.raw.currents rtol=1e-9
    @test p.variables["Loss"]=="ln(exp(2))"
end
