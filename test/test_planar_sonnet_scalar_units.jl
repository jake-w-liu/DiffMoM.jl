using Test,DiffMoM,LinearAlgebra,TOML,SHA,JSON

function _native_fixture_crc32(bytes)
    crc=typemax(UInt32)
    for byte in bytes
        crc⊻=UInt32(byte)
        for _ in 1:8
            crc=isodd(crc) ? (crc>>1)⊻0xedb88320 : crc>>1
        end
    end
    ~crc
end

@testset "Actual native scalar and NOR material selectors" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sonnet_scalar_units")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
        @test string(_native_fixture_crc32(bytes);base=16,pad=8)==file["crc32"]
    end
    g=CellGrid(.001,.001,20,20);sheet=sheet_level(1,20,20)
    sheet.mask[:,9:12].=true;sheet.connect_west[9:12].=true;sheet.connect_east[9:12].=true
    stack=PlanarStackup([PlanarLayer(1.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
    manual=build_planar_problem(stack,g,[sheet],[PlanarPort(1,:west,9:12,50.),PlanarPort(1,:east,9:12,50.)])
    rhs=zeros(ComplexF64,length(manual.basis.kind),2)
    for b in eachindex(manual.basis.kind)
        q=manual.basis.port[b];q==0 && continue;rhs[b,q]=(q==1 ? -1 : 1)*manual.basis.width[b]
    end
    provenance=JSON.parsefile(joinpath(fixture,"provenance.json"))
    @test length(provenance["cases"])==24
    for case in provenance["cases"]
        directory=joinpath(fixture,case["path"])
        p=read_sonnet_project(joinpath(directory,case["tag"]*".son"))
        source=read(p.source);metals=deepcopy(p.metals);variables=copy(p.variables)
        native=planar_read_touchstone(joinpath(directory,"native.s2p"))
        @test native.frequencies==case["frequencies_Hz"]
        metadata=TOML.parsefile(joinpath(directory,"metadata.toml"))
        @test metadata["source_sha256"]==case["source_sha256"]==bytes2hex(sha256(source))
        @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
        @test metadata["actual_cell_counts"]==[20,20]
        @test all(>(0),metadata["native_subsections"])
        @test metadata["touchstone_selected_log_checks"]["native.s2p"]["status"]=="PASS"
        for (index,f) in enumerate(native.frequencies)
            sigma=case["sigma_S_m"][index]
            zs=if sigma=="PEC"
                0im
            else
                # Independent finite-film equation with explicitly known
                # conductivity/thickness/current ratio from literal nodes.
                skin=(1+im)*sqrt(pi*f*(4pi*1e-7)/sigma)
                k=5/9
                k*skin/tanh(k*sigma*skin*1e-6)
            end
            hand=solve_planar(manual,f;method=:dense_fft,mx=160,my=160,surface_zs=zs)
            result=solve_sonnet_project(p,f;raw=true,method=:dense_fft,mx=160,my=160)
            @test maximum(abs,result.s-native.s[index])<.005
            @test maximum(abs,result.s-hand.s)<2e-10
            @test norm(result.raw.z_mom*result.raw.currents-rhs)/norm(rhs)<1e-9
            @test result.raw.currents≈hand.currents rtol=1e-8
            @test only(result.raw.problem.sheets).mask==sheet.mask
            @test result.raw.problem.grid.nx==20 && result.raw.problem.grid.ny==20
            voltage=ComplexF64[1+.2im,-.3+.4im]
            actual=only(planar_current_maps(result;voltages=voltage))
            expected=only(planar_current_maps(hand;voltages=voltage))
            @test hypot(norm(actual.jx-expected.jx),norm(actual.jy-expected.jy))<
                1e-8*hypot(norm(expected.jx),norm(expected.jy))
            @test opnorm(result.s)<=1+1e-10
        end
        @test p.metals==metals && p.variables==variables && read(p.source)==source
    end
    before=TOML.parsefile(joinpath(fixture,"proof","scalar_falsifier_before.toml"))
    @test !before["all_scalar_gates_pass"] && !before["all_full_s_gates_pass"]
    @test maximum(r["full_s_error"] for r in before["runs"])>.98
    @test all(!r["scalar_gate_pass"] for r in before["runs"])
    after=TOML.parsefile(joinpath(fixture,"proof","scalar_falsifier_after_final.toml"))
    @test after["all_scalar_gates_pass"] && after["all_full_s_gates_pass"]
    for file in ("variable_MOHSQ_tech.stf","literal_MOHSQ_tech.stf","variable_OHSQ_tech.stf")
        t=read_sonnet_technology(joinpath(fixture,"proof",file))
        @test sonnet_technology_material(t,"Sheet";kind=:conductor).resistance==2.
    end
end

@testset "NOR limits, provider order and scalar units" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sonnet_scalar_units")
    p=read_sonnet_project(joinpath(fixture,"selectors","srvy_2","srvy_2.son"))
    for f in (1.,1e6,1e9,1e12,1e18),ratio in (0.,1e-308,1e-100,.5,1.,2.,1e100,1e308)
        metal=["Sheet","0","NOR","500000",string(ratio),".001"]
        z=sonnet_metal_zs(p,metal,f)
        k=setprecision(BigFloat,256) do
            r=BigFloat(ratio)
            Float64((1+r^2)/(1+r)^2)
        end
        skin=(1+im)*sqrt(pi*f*(4pi*1e-7)/5e5)
        expected=k*skin/tanh(k*5e5*skin*1e-6)
        @test isfinite(z)
        @test z≈expected rtol=2e-12
        @test real(z)>0
        if ratio>=1e100 || ratio<=1e-100
            @test z≈sonnet_metal_zs(p,metal,f;cover=true) rtol=2e-12
        end
    end
    for selector in ("RSVY","SRVY")
        @test sonnet_metal_zs(p,["Sheet","0","NOR","0",".5",".001",selector],1e9)==0im
    end
    for (selector,loss) in (("CDVY","500000"),("RSVY",".0002"),("SRVY","2"))
        @test sonnet_metal_zs(p,["Sheet","0","NOR",loss,".5",".001",selector],1e9)≈
            sonnet_metal_zs(p,["Sheet","0","NOR","500000",".5",".001"],1e9)
    end
    for (loss,ratio,thickness,selector) in (("-1",".5",".001","SRVY"),
            ("2","-1",".001","SRVY"),("2",".5","0","SRVY"),
            ("1e-320",".5",".001","SRVY"),("1e308",".5","1e308","SRVY"),
            ("0",".5",".001","CDVY"),("2",".5",".001","UNKNOWN"))
        bad=deepcopy(p);bad.metals[2]=["Sheet","0","NOR",loss,ratio,thickness,selector]
        # Invalid providers reject before attempting an impossible geometry.
        error=try
            sonnet_planar_problem(bad;freq=1e9,grid=(typemax(Int),typemax(Int)),_materials=true)
            nothing
        catch e
            e
        end
        @test error isa ArgumentError
        @test occursin("NOR",sprint(showerror,error))
    end
    @test_throws ArgumentError sonnet_metal_zs(p,["Sheet","0","NOR","2",".5",".001","SRVY","SRVY"],1e9)
    # This thick-film RF limit is finite even though an intermediate
    # angular-frequency product exceeds Float64 range.
    large_reference=setprecision(BigFloat,4096) do
        ratio=BigFloat(.5);k=(1+ratio^2)/(1+ratio)^2
        ComplexF64(k*sqrt(BigFloat(pi)*BigFloat(1e308)*BigFloat(DiffMoM._MU0)/BigFloat(5e5))*(1+1im))
    end
    large=sonnet_metal_zs(p,["Sheet","0","NOR","500000",".5",".001"],1e308)
    @test isfinite(large)
    @test real(large)≈real(large_reference) rtol=2e-12
    @test imag(large)≈imag(large_reference) rtol=2e-12
    # Preserve explicit rejection when the actual material response,
    # rather than an avoidable intermediate, cannot be represented.
    @test_throws ArgumentError sonnet_metal_zs(p,["Sheet","0","NOR","1e-320",".5",".001"],1e308)
    for unit in ("OH","KOH","MOH")
        q=deepcopy(p);q.units["RES"]=unit
        @test sonnet_metal_zs(q,p.metals[2],1e9)==sonnet_metal_zs(p,p.metals[2],1e9)
    end
    @test sonnet_variable_value(p,"FREQ";freq=1e9)==1e9
    @test sonnet_variable_value(p,"h2p(FREQ)";freq=1e9)==1
    @test sonnet_variable_value(p,"p2h(1)")==1e9
    @test sonnet_variable_value(p,"m2p(.001)")==1
    @test sonnet_variable_value(p,"p2m(1)")==.001
    mhz=read_sonnet_project(joinpath(fixture,"variables","srvy_mhz_freq","srvy_mhz_freq.son"))
    @test sonnet_variable_value(mhz,"FREQ";freq=1e9)==1e9
    @test sonnet_variable_value(mhz,"h2p(FREQ)";freq=1e9)==1000
    @test sonnet_variable_value(mhz,"p2h(1000)")==1e9
    @test sonnet_variable_value(p,"FREQ";freq=0)==0
    for freq in (-1.,Inf,NaN,big"1e-500",big"1e500")
        @test_throws ArgumentError sonnet_variable_value(p,"1";freq)
    end
    for override in (Inf,NaN,big"1e-500",big"1e500",1im)
        @test_throws ArgumentError sonnet_variable_value(p,"Loss";variables=Dict("Loss"=>override))
    end
    for expression in ("h2p()","p2h(1,2)","m2p()","p2m(1,2)","unknown(1)")
        @test_throws ArgumentError sonnet_variable_value(p,expression)
    end
end
