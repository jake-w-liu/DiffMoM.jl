using Test,DiffMoM,LinearAlgebra,TOML,SHA,JSON

function _scalar_table_crc(bytes)
    crc=typemax(UInt32)
    for byte in bytes
        crc⊻=UInt32(byte)
        for _ in 1:8;crc=isodd(crc) ? (crc>>1)⊻0xedb88320 : crc>>1;end
    end
    ~crc
end

@testset "Scalar parent correspondence and bounded expression depth" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sonnet_scalar_tables","table1_nodes")
    template=read(joinpath(fixture,"escaped.son"),String)
    mktempdir() do dir
        source=joinpath(dir,"coupon.son");write(source,template)
        write(joinpath(dir,"loss.csv"),"1000000000,2\n10000000000,20\n")
        p=read_sonnet_project(source);original=sonnet_scalar_files(p)
        write(source,replace(template,"VALVAR Loss SRES"=>"VALVAR Other SRES"))
        @test_throws ArgumentError sonnet_scalar_files(p)
        rm(joinpath(dir,"loss.csv"))
        rejected=try sonnet_scalar_files(p);nothing catch err;err end
        @test rejected isa ArgumentError && occursin("parsed parent records",sprint(showerror,rejected))
        write(joinpath(dir,"loss.csv"),"1000000000,2\n10000000000,20\n")
        @test_throws ArgumentError solve_sonnet_project(p,5.5e9;raw=true,grid=(typemax(Int),typemax(Int)))
        write(source,template)
        work=DiffMoM._checked_payload_sum("test expression reserve",DiffMoM._sonnet_scalar_project_payload(p),
            DiffMoM._sonnet_scalar_expression_payload(p,String[]))
        @test_throws ArgumentError sonnet_scalar_files(p;max_storage=work-1)
        @test_throws ArgumentError sonnet_scalar_files(p;expressions=[1])
        edited=deepcopy(p);edited.variables["Loss"]="2*table1(\"loss.csv\",FREQ)"
        files=sonnet_scalar_files(edited)
        @test files.source.sha256==original.source.sha256
        @test files.configuration_sha256!=original.configuration_sha256
        @test sonnet_variable_value(files,"Loss";freq=5.5e9)==22.
        @test p.variables["Loss"]!=edited.variables["Loss"]
        attached=deepcopy(p);attached.records[1].tokens[2]="INVALID"
        @test_throws ArgumentError sonnet_scalar_files(attached)
        write(source,replace(template,"END DIM"=>"END   DIM ! same decoded record"))
        lexical=sonnet_scalar_files(p)
        @test lexical.source.sha256!=original.source.sha256
        @test sonnet_variable_value(lexical,"Loss";freq=5.5e9)==11.
        write(source,"! inserted diagnostic line\n"*template)
        @test_throws ArgumentError sonnet_scalar_files(p)
        write(source,[codeunits(template);0xff])
        @test_throws ArgumentError sonnet_scalar_files(p)
        write(source,template)
        for key in (:max_bytes,:max_storage,:max_files,:max_nodes,:max_line_bytes)
            @test_throws ArgumentError sonnet_scalar_files(p;Dict(key=>true)...)
        end
        safe=repeat("1-(",32)*"table1(\"loss.csv\",FREQ)"*repeat(")",32)
        q=deepcopy(p);q.variables["Depth"]=safe
        @test sonnet_variable_value(sonnet_scalar_files(q),"Depth";freq=1e9)==2.
        for expression in (repeat("1-(",3500)*"table1(\"loss.csv\",FREQ)"*repeat(")",3500),
                repeat("1-",1500)*"table1(\"loss.csv\",FREQ)")
            @test ncodeunits(expression)<16384
            q=deepcopy(p);q.variables["Depth"]=expression
            @test_throws ArgumentError sonnet_scalar_files(q)
            @test_throws ArgumentError sonnet_variable_value(p,expression)
        end
        q=deepcopy(p)
        for i in 1:150;q.variables["Chain$i"]=i==150 ? "1" : "Chain$(i+1)";end
        snapshot=sonnet_scalar_files(q)
        @test_throws ArgumentError sonnet_variable_value(snapshot,"Chain1")
        @test sonnet_variable_value(snapshot,"Chain140")==1.
    end
end

@testset "Captured linked STF scalar parent and effective provider identity" begin
    fixture=joinpath(@__DIR__,"..","validation","sonnet_stripline","native_technology_reference")
    original=read(joinpath(fixture,"linked.son"),String)
    withtable=replace(original,"NUM 1"=>"VALVAR Ref RES \"table1(\\\"gain.csv\\\",FREQ)\" \"CSV reference\"\nNUM 1",
        "1 50 0 0 0 0 .5"=>"1 Ref 0 0 0 0 .5","2 50 0 0 0 1 .5"=>"2 Ref 0 0 0 1 .5")
    mktempdir() do dir
        source=joinpath(dir,"linked.son");xml=joinpath(dir,"mini.stf")
        write(source,withtable);cp(joinpath(fixture,"mini.stf"),xml)
        write(joinpath(dir,"gain.csv"),"1000000000,50\n10000000000,75\n")
        linked=read_sonnet_linked_project(source)
        files=sonnet_scalar_files(linked)
        @test files.conversion=="static-stf" && length(files.conversion_sources)==1
        @test only(files.conversion_sources).sha256==linked.technology.sha256
        @test files.source.sha256==linked.sha256 && files.source.bytes==codeunits(withtable)
        @test sonnet_variable_value(files,"Ref";freq=5.5e9)==62.5
        @test files.payload_bytes==DiffMoM._sonnet_scalar_files_payload(files)
        override=sonnet_scalar_files(linked;technology_variables=Dict("TopH"=>200.))
        @test override.source.sha256==files.source.sha256
        @test override.configuration_sha256!=files.configuration_sha256
        @test parse(Float64,override.project.layers[1][1])≈.2 rtol=2e-15
        inline=read_sonnet_project(joinpath(fixture,"inline.son"))
        for i in eachindex(inline.ports)
            port=deepcopy(inline.ports[i]);port.values[2]="62.5";inline.ports[i]=port
        end
        result=solve_sonnet_project(files.project,5.5e9;raw=true,scalar_files=files,mx=80,my=80,method=:dense_fft)
        hand=solve_sonnet_project(inline,5.5e9;raw=true,mx=80,my=80,method=:dense_fft)
        @test maximum(abs,result.s-hand.s)<1e-11
        @test result.z0==[62.5,62.5]
        @test result.scalar_files.conversion=="static-stf"
        @test result.raw.currents≈hand.raw.currents rtol=1e-9
        @test opnorm(result.s)<=1+1e-10
        rhs=zeros(ComplexF64,size(result.raw.currents))
        for b in eachindex(result.raw.problem.basis.kind)
            q=result.raw.problem.basis.port[b];q==0 && continue
            rhs[b,q]=(q==1 ? -1 : 1)*result.raw.problem.basis.width[b]
        end
        @test norm(result.raw.z_mom*result.raw.currents-rhs)/norm(rhs)<1e-9
        for option in ((;max_storage=1),(;max_bytes=1),(;max_bytes=ncodeunits(withtable)))
            @test_throws ArgumentError sonnet_scalar_files(linked;option...)
        end
        bad=deepcopy(linked);bad.records[1].tokens[2]="INVALID"
        @test_throws ArgumentError sonnet_scalar_files(bad)
        bad=deepcopy(linked);bad.technology.units["lunit"]="MM"
        @test_throws ArgumentError sonnet_scalar_files(bad)
        bad=deepcopy(linked);bad.technology.variables["TopH"].attributes["value"]="999"
        @test_throws ArgumentError sonnet_scalar_files(bad)
        bad=deepcopy(linked)
        bad.technology.variables["TopH"]=SonnetTechnologyNode("var",Dict("name"=>"TopH","value"=>"999"),"",SonnetTechnologyNode[])
        @test_throws ArgumentError sonnet_scalar_files(bad)
        bad=SonnetLinkedProject(linked.source,linked.raw*"!changed\n",linked.sha256,linked.records,
            linked.declared_technology,linked.technology)
        @test_throws ArgumentError sonnet_scalar_files(bad)
        stale=deepcopy(files);only(stale.conversion_sources).bytes[1]⊻=0x01
        @test_throws ArgumentError sonnet_variable_value(stale,"Ref";freq=5.5e9)
        write(source,replace(withtable,"1 Ref"=>"1 99"));write(xml,"<changed/>")
        retained=sonnet_scalar_files(linked)
        @test retained.source.sha256==files.source.sha256
        @test only(retained.conversion_sources).bytes==only(files.conversion_sources).bytes
        @test sonnet_variable_value(retained,"Ref";freq=5.5e9)==62.5
        rm(source);rm(xml)
        @test sonnet_variable_value(files,"Ref";freq=5.5e9)==62.5
    end
end

@testset "Actual native CSV nodes, interpolation and explicit domain policy" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sonnet_scalar_tables")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
        @test string(_scalar_table_crc(bytes);base=16,pad=8)==file["crc32"]
    end
    g=CellGrid(.001,.001,20,20);sh=sheet_level(1,20,20)
    sh.mask[:,9:12].=true;sh.connect_west[9:12].=true;sh.connect_east[9:12].=true
    st=PlanarStackup([PlanarLayer(1.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
    manual=build_planar_problem(st,g,[sh],[PlanarPort(1,:west,9:12,50.),PlanarPort(1,:east,9:12,50.)])
    rhs=zeros(ComplexF64,length(manual.basis.kind),2)
    for b in eachindex(manual.basis.kind)
        q=manual.basis.port[b];q==0 && continue
        rhs[b,q]=(q==1 ? -1 : 1)*manual.basis.width[b]
    end
    for case in JSON.parsefile(joinpath(fixture,"provenance.json"))["cases"]
        dir=joinpath(fixture,case["tag"]);p=read_sonnet_project(joinpath(dir,case["source"]))
        saved=deepcopy(p.variables);bytes=read(p.source)
        files=sonnet_scalar_files(p;outside=Symbol(case["outside"]))
        @test files.project!==p && files.project.variables!==p.variables
        @test files.source.bytes==bytes && files.source.sha256==bytes2hex(sha256(bytes))
        @test files.payload_bytes==DiffMoM._sonnet_scalar_files_payload(files)
        @test DiffMoM._sonnet_check_scalar_files(files)===files
        @test length(DiffMoM._sonnet_scalar_dependency_sources(files))==1
        metadata=TOML.parsefile(joinpath(dir,"metadata.toml"))
        @test metadata["source_sha256"]==files.source.sha256
        @test metadata["actual_cell_counts"]==[20,20]
        @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
        @test metadata["touchstone_selected_log_checks"]["native.s2p"]["status"]=="PASS"
        native=planar_read_touchstone(joinpath(dir,"native.s2p"))
        for (i,f) in enumerate(native.frequencies)
            resistance=case["resistance_ohm_square"][i]
            @test sonnet_variable_value(files,"Loss";freq=f)≈resistance rtol=2e-15
            sigma=inv(resistance*1e-6);skin=(1+im)*sqrt(pi*f*(4pi*1e-7)/sigma)
            zs=(5/9)*skin/tanh((5/9)*sigma*skin*1e-6)
            hand=solve_planar(manual,f;method=:dense_fft,mx=160,my=160,surface_zs=zs)
            result=solve_sonnet_project(p,f;raw=true,scalar_files=files,method=:dense_fft,mx=160,my=160)
            @test maximum(abs,result.s-native.s[i])<.005
            @test maximum(abs,result.s-hand.s)<2e-10
            @test norm(result.raw.z_mom*result.raw.currents-rhs)/norm(rhs)<1e-9
            @test result.raw.currents≈hand.currents rtol=1e-8
            @test result.scalar_files!==files
            @test result.scalar_files.configuration_sha256==files.configuration_sha256
            @test result.project===result.scalar_files.project
            @test result.z0==[50.,50.] && opnorm(result.s)<=1+1e-10
            v=ComplexF64[.8+.1im,-.3+.2im]
            a=only(planar_current_maps(result;voltages=v));b=only(planar_current_maps(hand;voltages=v))
            @test hypot(norm(a.jx-b.jx),norm(a.jy-b.jy))<
                1e-8*hypot(norm(b.jx),norm(b.jy))
            if case["outside"]=="hold"
                @test_throws ArgumentError solve_sonnet_project(p,f;raw=true,grid=(typemax(Int),typemax(Int)))
                @test occursin("Key is ",read(joinpath(dir,"engine_stderr.log"),String))
            end
        end
        @test p.variables==saved && read(p.source)==bytes
    end
    before=read(joinpath(fixture,"proof","native_scalar_table1_parse_falsifier.log"),String)
    @test occursin("ParseError",before)
    @test occursin("loss=0.0",read(joinpath(fixture,"proof","native_scalar_text_precision_falsifier.log"),String))
    quote_fail=TOML.parsefile(joinpath(fixture,"proof","native_scalar_table1_aOrl1r","comparison.toml"))
    @test maximum(last(quote_fail["runs"])["native_literal_delta_s"])>.49
    @test occursin("Loss = 0",read(joinpath(fixture,"proof","nested","engine_stdout.log"),String))
end

@testset "Scalar dependency ownership, numeric domains and budgets" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sonnet_scalar_tables","table1_nodes")
    template=read(joinpath(fixture,"escaped.son"),String)
    mktempdir() do dir
        source=joinpath(dir,"coupon.son");csv=joinpath(dir,"loss.csv")
        write(source,template);write(csv,"! comment\n1000000000,2\n10000000000,20 ! final\n")
        p=read_sonnet_project(source);files=sonnet_scalar_files(p)
        context=DiffMoM._sonnet_scalar_variables(p,Dict("Extra"=>big"2");scalar_files=files)
        @test context["Extra"]==2. && context.files!==files
        @test DiffMoM._sonnet_scalar_files(context)===context.files
        @test DiffMoM._sonnet_scalar_project(p,context)===context.files.project
        @test DiffMoM._sonnet_scalar_payload(context)>context.files.payload_bytes
        copied=copy(context)
        @test copied.files!==context.files && copied.files.tables["loss.csv"].values!==context.files.tables["loss.csv"].values
        @test !(DiffMoM._sonnet_scalar_variables(p,Dict()) isa Dict{String,Float64})
        plain=deepcopy(p);plain.variables["Loss"]="2"
        @test DiffMoM._sonnet_scalar_variables(plain,Dict("R"=>2))==Dict("R"=>2.)
        write(csv,"1000000000,99\n10000000000,99\n");rm(source)
        @test sonnet_variable_value(files,"Loss";freq=5.5e9)==11.
        @test sonnet_variable_value(copied.files,"Loss";freq=5.5e9)==11.
        @test_throws ArgumentError sonnet_variable_value(files,"Loss";freq=big"1e500",max_bytes=1)
        bad=deepcopy(files);bad.source.bytes[1]⊻=0x01
        @test_throws ArgumentError sonnet_variable_value(bad,"Loss";freq=1e9)
        bad=deepcopy(files);bad.tables["loss.csv"].values[1]=3
        @test_throws ArgumentError sonnet_variable_value(bad,"Loss";freq=1e9)
        bad=deepcopy(files);bad.project.variables["Loss"]="4"
        @test_throws ArgumentError sonnet_variable_value(bad,"Loss";freq=1e9)
        bad=deepcopy(files)
        push!(bad.project.components,[SonnetRecord(0,["TYPE","IDEAL","RES","33.54"])])
        @test_throws ArgumentError sonnet_variable_value(bad,"Loss";freq=1e9)
        bad=deepcopy(files);r=bad.project.sweeps[1]
        bad.project.sweeps[1]=SonnetRecord(r.line,[r.tokens[1:end-1];"1"])
        @test_throws ArgumentError sonnet_variable_value(bad,"Loss";freq=1e9)
        bad=deepcopy(files);port=bad.project.ports[1]
        bad.project.ports[1]=SonnetPortSpec(port.kind,port.polygon,port.edge,port.number,port.values,
            [SonnetRecord(0,["REFPLANE","1"])])
        @test_throws ArgumentError sonnet_variable_value(bad,"Loss";freq=1e9)
        write(source,template);write(csv,"1000000000,2\n10000000000,20\n")
        for option in ((;max_files=0),(;max_bytes=1),(;max_nodes=1),(;max_line_bytes=3),
                (;max_storage=128),(;max_storage=typemax(Int),max_bytes=0),(;outside=:unknown))
            @test_throws ArgumentError sonnet_scalar_files(p;option...)
        end
        @test sonnet_scalar_files(p;max_bytes=typemax(Int)).tables["loss.csv"].values[:,1]==[2,20]
        sonnet_scalar_files(p;max_bytes=1024^2);sonnet_scalar_files(p;max_bytes=typemax(Int))
        small=@allocated sonnet_scalar_files(p;max_bytes=1024^2)
        large=@allocated sonnet_scalar_files(p;max_bytes=typemax(Int))
        @test large<=small+32768
        @test large<1024^2
        @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true,max_bytes=128,grid=(typemax(Int),typemax(Int)))
        for content in ("1,2\n1,3\n","2,2\n1,3\n","1,NaN\n2,3\n","1,1e-500\n2,3\n",
                "1,1e500\n2,3\n","1,2,3\n2,3,4\n","! empty\n")
            write(csv,content)
            @test_throws ArgumentError sonnet_scalar_files(p)
        end
        write(csv,"1000000000,2\n10000000000,20\n")
        for expression in ("table1(\"../outside.csv\",FREQ)","table1(unknown,FREQ)",
                "table1(\"loss.csv\")","table2(\"loss.csv\",1,2)")
            q=deepcopy(p);q.variables["Loss"]=expression
            @test_throws ArgumentError sonnet_scalar_files(q)
        end
        q=deepcopy(p);q.variables["Other"]="table1(\"./loss.csv\",FREQ)"
        aliases=sonnet_scalar_files(q;max_files=1)
        @test aliases.tables["loss.csv"]===aliases.tables["./loss.csv"]
        @test length(DiffMoM._sonnet_scalar_dependency_sources(aliases))==1
        @test sonnet_variable_value(plain,"table1(\"loss.csv\",FREQ)";freq=5.5e9)==11.
        for expression in ("1e-500","2*1e-500","-1e-500")
            @test_throws ArgumentError sonnet_variable_value(p,expression)
        end
        @test sonnet_variable_value(p,"0e-500")==0.
        @test sonnet_variable_value(p,"0.0")==0.
        for frequency in (-1.,NaN,Inf,big"1e-500",big"1e500")
            @test_throws ArgumentError solve_sonnet_project(p,frequency;raw=true)
        end
        for value in (Inf,big"1e500",big"1e-500",1im)
            @test_throws ArgumentError DiffMoM._sonnet_scalar_variables(p,Dict("X"=>value))
        end
        @test_throws ArgumentError sonnet_variable_value(p,"table1(\"loss.csv\",1/0)";scalar_outside=:hold)
    end
    mktempdir() do outer
        dir=joinpath(outer,"project");mkpath(dir)
        write(joinpath(outer,"outside.csv"),"1,2\n2,3\n")
        write(joinpath(dir,"coupon.son"),replace(template,"loss.csv"=>"../outside.csv"))
        @test_throws ArgumentError sonnet_scalar_files(read_sonnet_project(joinpath(dir,"coupon.son")))
    end
    for x in (-1e308,-1.,0.,1.,1e308)
        i,j,a=DiffMoM._sonnet_scalar_axis([-1e308,1e308],x,:reject)
        @test (i,j)==(1,2) && isfinite(a) && 0<=a<=1
        expected=setprecision(BigFloat,256) do
            Float64((BigFloat(x)+BigFloat(1e308))/(2BigFloat(1e308)))
        end
        @test a≈expected atol=1e-15
        @test isfinite(DiffMoM._sonnet_scalar_lerp(-1e308,1e308,a))
    end
    @test DiffMoM._sonnet_scalar_lerp(floatmax(Float64),floatmax(Float64),.3)==floatmax(Float64)
end
