module NativeScalarComponentTests
using DiffMoM,Test,LinearAlgebra,SHA
const DM=DiffMoM
function project(directory;floating=false,sparam=false)
    fixture=sparam ? joinpath(@__DIR__,"fixtures/native_sparam_model_files/native/sparameter.son") :
        joinpath(@__DIR__,"fixtures/native_ideal_component_units/res/ohm/ohm.son")
    source=joinpath(directory,"coupon.son");cp(fixture,source;force=true)
    p=read_sonnet_project(source)
    # Synthetic effective geometry is independent of the literal SON records.
    # The latter continue to identify the exact parent bytes at scalar intake.
    for i in eachindex(p.components);p.components[i]=deepcopy(p.components[i]);end
    for i in eachindex(p.ports);p.ports[i]=deepcopy(p.ports[i]);end
    if sparam
        cp(joinpath(dirname(fixture),"device_model.s2p"),joinpath(directory,"device_model.s2p");force=true)
    elseif floating
        for r in only(p.components)
            first(r.tokens)=="GNDREF" && (r.tokens[end]="FLOAT")
        end
        filter(r->r.tokens[1]=="SMDP",only(p.components))[2].tokens[6]="-3"
    end
    p
end
function literal_project(p,value)
    q=deepcopy(p)
    for i in eachindex(q.components);q.components[i]=deepcopy(q.components[i]);end
    only(r for r in only(q.components) if r.tokens[1]=="TYPE").tokens[end]=string(value)
    empty!(q.variables)
    q
end
const waves=ComplexF64[1.,.3+.2im]
maps(result::SonnetComponentResult;kw...)=sonnet_component_current_maps(result;kw...)
maps(result::SonnetFloatingResult;kw...)=planar_current_maps(result;kw...)

@testset "Native scalar tables retain physical IDEAL/FLOAT projections" begin
    mktempdir() do directory
        table=joinpath(directory,"load.csv")
        write(table,"200000000,16.77\n400000000,33.54\n")
        for floating in (false,true),frequency in (200e6,300e6,400e6)
            p=project(directory;floating)
            p.variables["Load"]="Scale*table1(\"load.csv\",FREQ)"
            p.variables["Scale"]="1"
            only(r for r in only(p.components) if r.tokens[1]=="TYPE").tokens[end]="Load"
            overrides=Dict("Scale"=>2.)
            expected=literal_project(p,33.54frequency/200e6)
            actual=solve_sonnet_project(p,frequency;variables=overrides,raw=true,mx=8,my=10,retain_matrix=false)
            reference=solve_sonnet_project(expected,frequency;raw=true,mx=8,my=10,retain_matrix=false)
            @test maximum(abs,actual.s-reference.s)<1e-10
            @test actual.circuit.voltages≈reference.circuit.voltages rtol=1e-10
            @test maps(actual;incident_waves=waves)[1].jx≈
                maps(reference;incident_waves=waves)[1].jx rtol=1e-10
            @test actual.scalar_files isa SonnetScalarFiles
            @test actual.scalar_overrides==overrides
            @test actual.scalar_overrides!==overrides
            @test actual.scalar_files.tables["load.csv"].source.sha256==bytes2hex(sha256(read(table)))
            @test actual.project===actual.scalar_files.project
            @test actual.project!==p
            @test overrides==Dict("Scale"=>2.)
        end
        p=project(directory)
        p.variables["Load"]="table2(\"load2.csv\",1+FREQ/200000000,15)"
        only(r for r in only(p.components) if r.tokens[1]=="TYPE").tokens[end]="Load"
        write(joinpath(directory,"load2.csv"),",10,20\n1,10,20\n3,30,60\n")
        model=sonnet_component_model(p,200e6)
        @test only(model.circuit.elements).r==30.
        @test model.scalar_files.tables["load2.csv"].kind==:table2
        @test model.payload>=model.scalar_files.payload_bytes
    end
end

@testset "Scalar component snapshots, outside policy and callback preflight" begin
    mktempdir() do directory
        table=joinpath(directory,"load.csv")
        write(table,"200000000,16.77\n400000000,33.54\n")
        p=project(directory)
        p.variables["Load"]="table1(\"load.csv\",FREQ)"
        only(r for r in only(p.components) if r.tokens[1]=="TYPE").tokens[end]="Load"
        snapshot=sonnet_scalar_files(p)
        first=sonnet_component_model(p,300e6;scalar_files=snapshot)
        oldhash=snapshot.configuration_sha256
        write(table,"200000000,100\n400000000,200\n")
        old=sonnet_component_model(p,300e6;scalar_files=snapshot)
        fresh=sonnet_component_model(p,300e6)
        @test only(old.circuit.elements).r==only(first.circuit.elements).r
        @test only(fresh.circuit.elements).r==150.
        @test old.scalar_files.configuration_sha256==oldhash
        @test fresh.scalar_files.configuration_sha256!=oldhash
        @test old.scalar_files!==snapshot
        @test old.scalar_files.tables["load.csv"].values!==snapshot.tables["load.csv"].values
        detached=deepcopy(snapshot)
        detached.project.components[1]=deepcopy(only(detached.project.components))
        only(r for r in only(detached.project.components) if r.tokens[1]=="TYPE").tokens[end]="33.54"
        @test_throws ArgumentError sonnet_component_model(p,300e6;scalar_files=detached)
        @test_throws ArgumentError sonnet_component_model(p,500e6)
        @test only(sonnet_component_model(p,500e6;scalar_outside=:hold).circuit.elements).r==200.
        calls=Ref(0)
        kind=only(r for r in only(p.components) if r.tokens[1]=="TYPE").tokens
        empty!(kind);append!(kind,["TYPE","SPROJ","explicit.son"])
        callback=(args...)->(calls[]+=1;(response=ComplexF64[0 1;1 0],format=:s,z0=50.))
        for kw in ((;max_bytes=1),(;scalar_max_bytes=1),(;scalar_max_nodes=1),(;scalar_max_line_bytes=1))
            @test_throws ArgumentError sonnet_component_model(p,300e6;component_response=callback,kw...)
            @test calls[]==0
        end
        # The provider receives the owned effective project. Mutation must
        # not yield geometry with a stale scalar configuration identity.
        mutation=(owned,args...)->begin
            calls[]+=1;owned.variables["Load"]="1"
            (response=ComplexF64[0 1;1 0],format=:s,z0=50.)
        end
        @test_throws ArgumentError sonnet_component_model(p,300e6;component_response=mutation)
        @test calls[]==1
        @test p.variables["Load"]=="table1(\"load.csv\",FREQ)"
    end
end

@testset "Automatic SPARAM retains scalar provenance and cumulative dependencies" begin
    mktempdir() do directory
        p=project(directory;sparam=true)
        table=joinpath(directory,"reference.csv")
        write(table,"200000000,50\n400000000,75\n")
        p.variables["Reference"]="table1(\"reference.csv\",FREQ)"
        p.ports[1].values[2]="Reference"
        files=sonnet_component_files(p,300e6;max_dependencies=3)
        @test files.scalar_files isa SonnetScalarFiles
        @test length(files.sources)==3
        @test haskey(files.sources,realpath(table))
        @test files.sources[realpath(table)].bytes===files.scalar_files.tables["reference.csv"].source.bytes
        @test_throws ArgumentError sonnet_component_files(p,300e6;max_dependencies=2)
        actual=solve_sonnet_project(files;raw=true,mx=8,my=10,retain_matrix=false)
        q=deepcopy(p);empty!(q.variables);q.ports[1].values[2]="62.5"
        reference=solve_sonnet_project(q,300e6;raw=true,mx=8,my=10,retain_matrix=false)
        # Scaled interpolation may round the exact midpoint by one ULP;
        # the independently literal physical response gate stays 1e-10.
        @test actual.z0≈ComplexF64[62.5,50] rtol=eps(Float64)
        @test maximum(abs,actual.s-reference.s)<1e-10
        @test actual.circuit.voltages≈reference.circuit.voltages rtol=1e-10
        @test maps(actual;incident_waves=waves)[1].jx≈
            maps(reference;incident_waves=waves)[1].jx rtol=1e-10
        @test actual.model_files===files
        @test actual.scalar_files!==files.scalar_files
        @test actual.scalar_overrides==files.variables
        before=files.configuration_sha256
        write(table,"200000000,100\n400000000,200\n")
        retained=solve_sonnet_project(files;raw=true,mx=8,my=10,retain_matrix=false)
        updated=sonnet_component_files(p,300e6)
        @test retained.z0==actual.z0
        @test retained.s≈actual.s rtol=1e-10
        @test updated.configuration_sha256!=before
        @test updated.scalar_files.configuration_sha256!=files.scalar_files.configuration_sha256
    end
end

@testset "Mixed cover/FLOAT scalar context preserves derived component geometry" begin
    mktempdir() do directory
        p=project(directory)
        write(joinpath(directory,"load.csv"),"200000000,16.77\n400000000,33.54\n")
        p.variables["Load"]="table1(\"load.csv\",FREQ)"
        only(r for r in only(p.components) if r.tokens[1]=="TYPE").tokens[end]="Load"
        floating=deepcopy(only(p.components))
        for r in floating
            r.tokens[1]=="ID" && (r.tokens[end]="200")
            r.tokens[1]=="GNDREF" && (r.tokens[end]="FLOAT")
        end
        pins=filter(r->r.tokens[1]=="SMDP",floating)
        pins[1].tokens[6]="5";pins[2].tokens[6]="-5"
        push!(p.components,floating)
        literal=deepcopy(p);empty!(literal.variables)
        for component in literal.components
            only(r for r in component if r.tokens[1]=="TYPE").tokens[end]="25.155"
        end
        model=sonnet_floating_model(p,300e6)
        @test length(model.circuit.elements)==2
        @test model.labels==[1,2,3,4,5]
        @test [[(r.line,r.tokens) for r in c] for c in model.scalar_files.project.components]==
            [[(r.line,r.tokens) for r in c] for c in p.components]
        actual=solve_sonnet_project(p,300e6;raw=true,mx=8,my=10,retain_matrix=false)
        expected=solve_sonnet_project(literal,300e6;raw=true,mx=8,my=10,retain_matrix=false)
        @test maximum(abs,actual.s-expected.s)<1e-10
        @test actual.circuit.voltages≈expected.circuit.voltages rtol=1e-10
        @test maps(actual;incident_waves=waves)[1].jx≈maps(expected;incident_waves=waves)[1].jx rtol=1e-10
    end
end

@testset "Mixed FLOAT retains automatic SPARAM sources and physical projection" begin
    mktempdir() do directory
        p=project(directory;sparam=true)
        floating=deepcopy(only(p.components))
        for record in floating
            t=record.tokens
            t[1]=="ID" && (t[end]="200")
            t[1]=="GNDREF" && (t[end]="FLOAT")
            if t[1]=="TYPE";empty!(t);append!(t,["TYPE","NONE"]);end
        end
        pins=filter(r->r.tokens[1]=="SMDP",floating)
        pins[1].tokens[6]="5";pins[2].tokens[6]="-5"
        push!(p.components,floating)
        write(joinpath(directory,"reference.csv"),"200000000,50\n400000000,75\n")
        p.variables["Reference"]="table1(\"reference.csv\",FREQ)"
        p.ports[1].values[2]="Reference"
        data=planar_read_touchstone(joinpath(directory,"device_model.s2p"))
        provider=(owned,records,f)->(response=planar_network_response(data,f),format=:s,z0=data.z0)
        model=sonnet_floating_model(p,300e6)
        @test model.model_files isa SonnetComponentFiles
        @test only(model.model_files.bindings).geometry_labels==[3,4]
        @test length(model.model_files.sources)==3
        actual=solve_sonnet_project(p,300e6;raw=true,mx=8,my=10,retain_matrix=false)
        manual=solve_sonnet_project(p,300e6;component_response=provider,raw=true,mx=8,my=10,retain_matrix=false)
        @test actual.model_files isa SonnetComponentFiles
        @test actual.model_files.scalar_files isa SonnetScalarFiles
        @test manual.model_files===nothing
        @test maximum(abs,actual.s-manual.s)<1e-10
        @test actual.circuit.voltages≈manual.circuit.voltages rtol=1e-10
        a=ComplexF64[1.,.3+.2im,.2im]
        @test maps(actual;incident_waves=a)[1].jx≈maps(manual;incident_waves=a)[1].jx rtol=1e-10
        old=SonnetFloatingResult(actual.project,actual.em,actual.circuit,actual.port_numbers,
            actual.z0,actual.y,actual.s,actual.source_incidence,actual.bridges,actual.incident_transfer,
            actual.scalar_files,actual.scalar_overrides)
        @test old.s===actual.s
        @test old.model_files===nothing
        path=realpath(joinpath(directory,"device_model.s2p"))
        snapshot=actual.model_files.sources[path]
        original=read(path)
        @test snapshot.bytes==original
        @test snapshot.sha256==bytes2hex(sha256(original))
        write(path,"later invalid mutable model contents")
        @test snapshot.bytes==original
        @test snapshot.sha256==bytes2hex(sha256(original))
        @test_throws ArgumentError solve_sonnet_project(p,300e6;raw=true,mx=8,my=10,retain_matrix=false)
    end
end

@testset "Linked CSV model staging retains original SON and STF conversion" begin
    mktempdir() do directory
        p=project(directory;sparam=true)
        source=joinpath(directory,"linked.son")
        text=replace(read(p.source,String),"TMET \"Lossless\""=>"STF mini.stf\nTMET \"Lossless\"",
            "GEO\n"=>"GEO\nVALVAR Reference SRES \"table1(\\\"reference.csv\\\",FREQ)\" \"CSV reference control\"\n",
            "1 50 0 0 0 0 187.5"=>"1 Reference 0 0 0 0 187.5")
        write(source,text)
        xml="""<technology_file version="1700"><units/><public><materials>
        <dielectric name="Air"><params/></dielectric>
        <dielectric name="Alumina"><params erel="9.8"/></dielectric>
        </materials><metal_model_defs><metal_model name="Normal" model_type="Normal"/></metal_model_defs>
        <stackup><TOP material="Lossless" model="Normal"/>
        <diel name="Upper" dielectric="Air" thickness="6350"/>
        <diel name="Lower" dielectric="Alumina" thickness="635"/>
        <BOTTOM material="Lossless" model="Normal"/></stackup></public></technology_file>"""
        write(joinpath(directory,"mini.stf"),xml)
        write(joinpath(directory,"reference.csv"),"200000000,50\n400000000,75\n")
        linked=read_sonnet_linked_project(source)
        effective=sonnet_materialize_project(linked)
        files=sonnet_component_files(linked,300e6;max_dependencies=4)
        @test files.scalar_files.conversion=="static-stf"
        @test files.scalar_files.source.sha256==linked.sha256
        @test only(files.scalar_files.conversion_sources).sha256==linked.technology.sha256
        @test length(files.sources)==4
        @test_throws ArgumentError sonnet_component_files(linked,300e6;max_dependencies=3)
        actual=solve_sonnet_project(files;raw=true,mx=8,my=10,retain_matrix=false)
        literal=deepcopy(effective);empty!(literal.variables);literal.ports[1].values[2]="62.5"
        reference=solve_sonnet_project(literal,300e6;raw=true,mx=8,my=10,retain_matrix=false)
        @test maximum(abs,actual.s-reference.s)<1e-10
        @test actual.circuit.voltages≈reference.circuit.voltages rtol=1e-10
        @test maps(actual;incident_waves=waves)[1].jx≈maps(reference;incident_waves=waves)[1].jx rtol=1e-10
    end
end
end
