module NativeSonnetGeometryVariableTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra

const fixture=joinpath(@__DIR__,"fixtures/native_sonnet_geometry_variables")

with_geometry(p,polygons,ports=p.ports)=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,
    p.box,p.layers,p.metals,p.top,p.bottom,polygons,ports,p.variables,p.components,p.sweeps,p.records)

@testset "Native ANC/SYM unscaled dimensions and exact source identity" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    proof=TOML.parsefile(joinpath(fixture,"accepted/comparison.toml"))
    @test proof["source_unchanged"]
    @test proof["source_before"]==proof["source_after"]
    @test length(proof["cases"])==16
    for row in proof["cases"]
        @test row["status"]=="PASS" && row["bit_identical_s"] && row["full_s_error"]==0
        tag="$(row["split_polygons"] ? "split" : "single")_$(lowercase(row["axis"]))_$(lowercase(row["kind"]))_$(row["direction"]==1 ? "positive" : "negative")"
        source=joinpath(fixture,"accepted",tag*"_parameter.son")
        literal=joinpath(fixture,"accepted",tag*"_literal.son")
        p=read_sonnet_project(source);q=read_sonnet_project(literal)
        original=deepcopy(p.polygons)
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        @test [x.id for x in resolved.polygons]==[x.id for x in q.polygons]
        @test all(x.vertices==y.vertices for (x,y) in zip(p.polygons,original))
        for (emitted,expected) in zip(resolved.polygons,q.polygons)
            @test emitted.vertices≈expected.vertices rtol=8eps(Float64) atol=0
            @test emitted.vertices!==only(filter(v->v.id==emitted.id,p.polygons)).vertices
        end
        # Public lowering consumes the effective geometry once; physical raw
        # wall sources then compare against freshly generated native full S.
        lowered=sonnet_planar_problem(p;freq=1e9)
        expected=sonnet_planar_problem(q;freq=1e9)
        @test only(lowered.sheets).mask==only(expected.sheets).mask
        reference=planar_read_touchstone(joinpath(fixture,"accepted",tag*"_parameter/native_raw.s2p"))
        @test reference.s==planar_read_touchstone(joinpath(fixture,"accepted",tag*"_literal/native_raw.s2p")).s
        metadata=TOML.parsefile(joinpath(fixture,"accepted",tag*"_parameter/metadata.toml"))
        @test metadata["process_success"] && !metadata["deembedded"]
        @test metadata["touchstone_selected_log_checks"]["native_raw.s2p"]["status"]=="PASS"
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        @test maximum(abs,result.s-only(reference.s))<=.005
        rhs=zeros(ComplexF64,size(result.raw.currents))
        for b in eachindex(result.raw.problem.basis.port)
            port=result.raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*result.raw.problem.basis.width[b]
        end
        @test norm(result.raw.z_mom*result.raw.currents-rhs)/norm(rhs)<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10
        @test opnorm(result.s)<=1+1e-9
        result=nothing;GC.gc()
    end
end

@testset "Native geometry variable domains, grammar and transactional resources" begin
    source=joinpath(fixture,"accepted/single_ydir_anc_positive_parameter.son")
    p=read_sonnet_project(source)
    @test DiffMoM._sonnet_geometry_project(p,1e9,Dict("Width"=>.25))===p
    for value in (0.,-1.,NaN,Inf,BigFloat("1e1000"),BigFloat("1e-1000"))
        saved=deepcopy(p.polygons)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9,Dict("Width"=>value))
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
    end
    for frequency in (0.,-1.,Inf,NaN,BigFloat("1e-1000"))
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,frequency)
    end
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_bytes=1)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_points=1)
    # REF1/REF2 consume the budget even when the saved PS1/PS2 omit them.
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_points=2)
    @test DiffMoM._sonnet_geometry_project(p,1e9;max_points=3).polygons[1].vertices==
        DiffMoM._sonnet_geometry_project(p,1e9).polygons[1].vertices
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_parameters=0)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_points=0)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_bytes=0)
    original=read(source,String)
    invalid_fixture=joinpath(@__DIR__,"fixtures/native_invalid_nominal_geovar")
    for (file,digest) in TOML.parsefile(joinpath(invalid_fixture,"sha256.toml"))["sha256"]
        @test bytes2hex(sha256(read(joinpath(invalid_fixture,file))))==digest
    end
    @test occursin("Geo Variable Scaled Keyword not understood",
        read(joinpath(invalid_fixture,"engine_stderr.log"),String))
    invalid_nominal=read_sonnet_project(joinpath(invalid_fixture,"invalid_nominal.son"))
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(invalid_nominal,1e9)
    @test_throws ArgumentError sonnet_planar_problem(invalid_nominal;freq=1e9)
    # Nominal imported geometry remains valid for metadata that has no proved
    # displacement adapter. Every dimension must match NOM exactly.
    for row in TOML.parsefile(joinpath(fixture,"compatibility/index.toml"))["cases"]
        @test row["status"]=="PASS" && row["same_project"]
        path=joinpath(fixture,"compatibility",row["fixture"])
        @test bytes2hex(sha256(read(path)))==row["sha256"]
        nominal=read_sonnet_project(path)
        @test DiffMoM._sonnet_geometry_project(nominal,1e9)===nominal
        radial=[r.tokens[2] for r in nominal.records if length(r.tokens)>=3 &&
            r.tokens[1]=="GEOVAR" && r.tokens[3]=="RAD"]
        if !isempty(radial)
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(nominal,1e9,
                Dict(first(radial)=>2sonnet_variable_value(nominal,first(radial))))
        end
    end
    mktempdir() do directory
        for mode in ("RAD","ANC")
            scaling=mode=="RAD" ? "NSCD" : "SCXY"
            text=replace(original," ANC "=>" $mode "," NSCD"=>" $scaling",
                "VALVAR Width LNG 0.375"=>"VALVAR Width LNG 0.25")
            file=joinpath(directory,"nominal.son");write(file,text)
            nominal=read_sonnet_project(file)
            @test DiffMoM._sonnet_geometry_project(nominal,1e9)===nominal
            @test only(sonnet_planar_problem(nominal;freq=1e9).sheets).mask==
                only(sonnet_planar_problem(p;freq=1e9,variables=Dict("Width"=>.25)).sheets).mask
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(nominal,1e9,Dict("Width"=>.375))
        end
    end
    # Sonnet rejects these headers before evaluating NOM. The unchanged
    # geometry shortcut must not accept malformed native dimension metadata.
    mktempdir() do directory
        nominal=replace(original,"VALVAR Width LNG 0.375"=>"VALVAR Width LNG 0.25")
        for (from,to) in ((" NSCD"," SCD"),(" ANC "," INVALID "),
                ("YDIR 1","ZDIR 1"),("YDIR 1","YDIR 0"),
                ("YDIR 1 NSCD","YDIR 1 NSCD EXTRA"))
            file=joinpath(directory,"invalid_nominal.son")
            write(file,replace(nominal,from=>to))
            candidate=read_sonnet_project(file)
            saved=deepcopy(candidate.polygons)
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(candidate,1e9)
            @test all(a.vertices==b.vertices for (a,b) in zip(candidate.polygons,saved))
            @test_throws ArgumentError sonnet_planar_problem(candidate;freq=1e9)
        end
    end
    polygon=only(p.polygons)
    openpolygon=SonnetPolygon(polygon.kind,polygon.level,polygon.material,polygon.id,
        polygon.vertices[:,1:end-1],polygon.target,polygon.technology,polygon.flags)
    openproject=with_geometry(p,[openpolygon])
    @test DiffMoM._sonnet_geometry_project(openproject,1e9).polygons[1].vertices==
        DiffMoM._sonnet_geometry_project(p,1e9).polygons[1].vertices[:,1:end-1]
    # Edge 3 is the real closing edge for four logical points. A duplicate
    # serialized endpoint cannot introduce a valid additional zero edge.
    west=first(p.ports)
    for edge in (-1,4,999)
        degenerate=SonnetPortSpec(west.kind,west.polygon,edge,west.number,west.values,west.records)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(with_geometry(p,p.polygons,
            [degenerate,last(p.ports)]),1e9)
    end
    incomplete=SonnetPortSpec(west.kind,west.polygon,west.edge,west.number,west.values[1:5],west.records)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(with_geometry(p,p.polygons,
        [incomplete,last(p.ports)]),1e9)
    huge=copy(polygon.vertices);huge[2,:].=[2.0^50,2.0^50,2.0^50+.25,2.0^50+.25,2.0^50]
    shifted=SonnetPolygon(polygon.kind,polygon.level,polygon.material,polygon.id,huge,
        polygon.target,polygon.technology,polygon.flags)
    cancellation=with_geometry(p,[shifted]);saved=copy(huge)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(cancellation,1e9)
    @test huge==saved
    for (from,to) in ((" ANC "," RAD "),(" NSCD"," SCD"),
            ("YDIR 1","ZDIR 1"),("YDIR 1","YDIR 0"),
            ("REF1 POLY 1 1","REF1 POLY 999 1"),
            ("REF1 POLY 1 1","REF1 POLY 1 2"),
            ("POLY 1 1\n2\nEND","POLY 1 1\n99\nEND"),
            ("PS2 1","PS2 999999999"),("NOM 0.25","NOM 0"),
            ("VALVAR Width LNG","VALVAR Width NONE"),
            ("PS1 0\nEND","PS1 1\nPOLY 1 1\n1\nEND"))
        mktempdir() do directory
            file=joinpath(directory,"bad.son");write(file,replace(original,from=>to))
            candidate=read_sonnet_project(file)
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(candidate,1e9)
        end
    end
    # Independent source declarations use the existing safe scalar evaluator.
    mktempdir() do directory
        text=replace(original,"VALVAR Width LNG 0.375"=>
            "VALVAR Change LNG 0.125\nVALVAR Width LNG \"0.25+Change\"",
            "PS1 0"=>"EQN \"0.25+Change\"\nPS1 0")
        file=joinpath(directory,"equation.son");write(file,text)
        project=read_sonnet_project(file)
        @test DiffMoM._sonnet_geometry_project(project,1e9).polygons[1].vertices≈
            DiffMoM._sonnet_geometry_project(p,1e9).polygons[1].vertices
        @test DiffMoM._sonnet_geometry_project(project,1e9,Dict("Change"=>0.))===project
        bad=replace(text,"EQN \"0.25+Change\""=>"EQN \"0.5+Change\"")
        write(file,bad)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(read_sonnet_project(file),1e9)
    end
    # Two dimensions sharing a moving point are explicitly unproven rather
    # than applied in arbitrary record order; rejected calls do not mutate.
    mktempdir() do directory
        start=findfirst("GEOVAR",original);stop=findfirst("POR1",original)
        block=original[first(start):first(stop)-1]
        text=replace(original,"POR1 BOX"=>replace(block,"GEOVAR Width"=>"GEOVAR Other")*"POR1 BOX";count=1)
        text=replace(text,"VALVAR Width LNG 0.375"=>"VALVAR Other LNG 0.375\nVALVAR Width LNG 0.375")
        file=joinpath(directory,"dependent.son");write(file,text)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(read_sonnet_project(file),1e9)
    end
end
end
