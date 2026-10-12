module NativeSonnetGeometryVariableTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra,JSON

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
            # These supported adapters make this particular wall attachment
            # invalid; both failures are retained from actual native Sonnet.
            @test_throws ArgumentError sonnet_planar_problem(nominal;freq=1e9,variables=Dict("Width"=>.375))
        end
    end
    # Sonnet rejects these headers before evaluating NOM. The unchanged
    # geometry shortcut must not accept malformed native dimension metadata.
    mktempdir() do directory
        nominal=replace(original,"VALVAR Width LNG 0.375"=>"VALVAR Width LNG 0.25")
        for (from,to) in ((" NSCD"," SCD"),(" ANC "," INVALID "),
                ("YDIR 1","ZDIR 1"),("YDIR 1","YDIR FOO"),
                ("YDIR 1 NSCD","YDIR JUNK 1 NSCD"))
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
        polygon.vertices[:,1:end-1],polygon.target,polygon.technology,polygon.technology_mode,polygon.flags)
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
        polygon.target,polygon.technology,polygon.technology_mode,polygon.flags)
    cancellation=with_geometry(p,[shifted]);saved=copy(huge)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(cancellation,1e9)
    @test huge==saved
    for (from,to) in ((" NSCD"," SCD"),
            ("YDIR 1","ZDIR 1"),("YDIR 1","YDIR FOO"),
            ("REF1 POLY 1 1","REF1 POLY 999 1"),
            ("REF1 POLY 1 1","REF1 POLY 1 2"),
            ("POLY 1 1\n2\nEND","POLY 1 1\n99\nEND"),
            ("PS2 1","PS2 999999999"),("NOM 0.25","NOM -0.25"),
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
    # Two dimensions sharing each other's moving references are a circular
    # definition under the native sequential law (engine-verified rejection);
    # rejected calls do not mutate.
    mktempdir() do directory
        start=findfirst("GEOVAR",original);stop=findfirst("POR1",original)
        block=original[first(start):first(stop)-1]
        text=replace(original,"POR1 BOX"=>replace(block,"GEOVAR Width"=>"GEOVAR Other")*"POR1 BOX";count=1)
        text=replace(text,"VALVAR Width LNG 0.375"=>"VALVAR Other LNG 0.375\nVALVAR Width LNG 0.375")
        file=joinpath(directory,"dependent.son");write(file,text)
        project=read_sonnet_project(file)
        saved=deepcopy(project.polygons)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(project,1e9)
        @test all(a.vertices==b.vertices for (a,b) in zip(project.polygons,saved))
    end
end

@testset "Native component pins stay file-static under moved dimensions" begin
    # Native never rewrites SMDP pin coordinates: a dimension may move any
    # polygon that does not carry a pin or a port edge (native em run
    # bit-identical to the resolved literal), while a move that takes a pinned
    # edge away from its pin fails the native detached-pin check.
    comp=raw"""FTYP SONPROJ 19
VER "18.53"
DIM
ANG DEG
CAP PF
CON /OH
FREQ MHZ
IND NH
LNG MIL
RES OH
END DIM
CONTROL
VARSWP
OPTIONS
SPEED 0
END CONTROL
GEO
TMET "Lossless" 0 SUP 0 0 0 0
BMET "Lossless" 0 SUP 0 0 0 0
BOX 1 400 500 64 80 20 0
250 1 1 0 0 0 1 "Air"
25 9.8 1 0 0 0 1 "Alumina"
LORGN 0 500 U
POR1 BOX
POLY 104 1
3
1 50 0 0 0 0 187.5
POR1 BOX
POLY 106 1
1
2 50 0 0 0 400 187.5
SMD 0 "ManufacturedSeriesResistor"
ID 100
GNDREF AUTO
SBOX 93.75 106.25 185.625 189.375
PBSHW N
LPOS 101.1060104 180.4625383
TYPE IDEAL CAP 1
SMDP 0 87.5 187.5 L 3 1
EXSMDP
GNDREF AUTO
TERMW FEED
END EXSMDP
SMDP 0 112.5 187.5 R 4 2
EXSMDP
GNDREF AUTO
TERMW FEED
END EXSMDP
END
"""
    tail=raw"""END GEO
VARSWP
ENABLED Y
FREQ Y AN SWEEP 200
END
END VARSWP
FILEOUT
TOUCH ND Y native_device.s2p IC 15 S RI R 50
FOLDER .
END FILEOUT
"""
    strips=raw"""0 5 -1 N 104 1 1 100 100 0 0 0 Y
0 175
87.5 175
87.5 200
0 200
0 175
END
0 5 -1 N 106 1 1 100 100 0 0 0 Y
112.5 175
400 175
400 200
112.5 200
112.5 175
END
"""
    floater=raw"""VALVAR FX LNG 275 "floater edge"
GEOVAR FX ANC XDIR 1 NSCD
POS 0 0
NOM 250
REF1 POLY 107 1
0
REF2 POLY 107 1
1
PS1 0
END
PS2 1
POLY 107 1
2
END
END
NUM 3
"""
    poly107=raw"""0 5 -1 N 107 1 1 100 100 0 0 0 Y
200 250
250 250
250 300
200 300
200 250
END
"""
    pinned=raw"""VALVAR GapX LNG 100 "edge shift"
GEOVAR GapX ANC XDIR 1 NSCD
POS 0 0
NOM 87.5
REF1 POLY 104 1
0
REF2 POLY 104 1
1
PS1 0
END
PS2 1
POLY 104 1
2
END
END
NUM 2
"""
    mktempdir() do directory
        # A dimension on a polygon carrying no pin or port edge resolves like
        # any other polygon: REF2 lands at REF1+target=475 (native-verified,
        # the file's inconsistent NOM=250 is ignored by the engine).
        file=joinpath(directory,"floater.son")
        write(file,comp*floater*strips*poly107*tail)
        project=read_sonnet_project(file)
        resolved=DiffMoM._sonnet_geometry_project(project,1e9)
        moved=only(filter(q->q.id==107,resolved.polygons))
        scale=project.length_scale
        @test moved.vertices./scale≈[200 475 475 200 200;250 250 300 300 250] rtol=8eps()
        @test [q.id for q in resolved.polygons]==[104,106,107]
        @test resolved.components==project.components
        untouched=only(filter(q->q.id==104,resolved.polygons))
        @test untouched.vertices./scale≈[0 87.5 87.5 0 0;175 175 200 200 175] rtol=8eps()
        # A dimension whose resolved edge leaves its SMDP pin behind matches
        # the native detached-pin rejection.
        file=joinpath(directory,"pinned.son")
        write(file,comp*pinned*strips*tail)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(
            read_sonnet_project(file),1e9)
        # Moving the polygon's box-port edge is rejected too, either by the
        # boundary allowance or because the reference itself leaves its set.
        file=joinpath(directory,"ported.son")
        write(file,comp*replace(pinned,"POLY 104 1\n2\nEND"=>
            "POLY 104 1\n0\nEND")*strips*tail)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(
            read_sonnet_project(file),1e9)
    end
end

@testset "Native sequential anchored dimensions and via ports" begin
    # Native anchored dimensions resolve in record order: each dimension
    # lands its moving reference at REF1_current+direction*target, then
    # shifts every PS2 listing by that displacement. Inactive dimensions
    # apply identically once any dimension is active, and inactive
    # dimensions whose saved span disagrees with NOM still re-anchor at
    # REF1+target. All cases below are engine runs whose resolved literal
    # projects match the parameterized run bit-identically.
    head=raw"""FTYP SONPROJ 19
VER "18.53"
DIM
ANG DEG
CAP PF
CON /OH
FREQ MHZ
IND NH
LNG MIL
RES OH
END DIM
CONTROL
VARSWP
OPTIONS
SPEED 0
END CONTROL
GEO
TMET "Lossless" 0 SUP 0 0 0 0
BMET "Lossless" 0 SUP 0 0 0 0
BOX 1 400 500 64 80 20 0
250 1 1 0 0 0 1 "Air"
25 9.8 1 0 0 0 1 "Alumina"
LORGN 0 500 U
POR1 BOX
POLY 104 1
3
1 50 0 0 0 0 187.5
POR1 BOX
POLY 106 1
1
2 50 0 0 0 400 187.5
"""
    tail=raw"""END GEO
VARSWP
ENABLED Y
FREQ Y AN SWEEP 200
END
END VARSWP
FILEOUT
TOUCH ND Y native_device.s2p IC 15 S RI R 50
FOLDER .
END FILEOUT
"""
    strips=raw"""NUM 2
0 5 -1 N 104 1 1 100 100 0 0 0 Y
0 175
87.5 175
87.5 200
0 200
0 175
END
0 5 -1 N 106 1 1 100 100 0 0 0 Y
112.5 175
400 175
400 200
112.5 200
112.5 175
END
"""
    wdim=raw"""VALVAR W LNG 150 "strip width"
GEOVAR W ANC XDIR 1 NSCD
POS 0 0
NOM 87.5
REF1 POLY 104 1
0
REF2 POLY 104 1
1
PS1 0
END
PS2 1
POLY 104 1
2
END
END
"""
    xdim(target)=replace(raw"""VALVAR X LNG XT "gap"
GEOVAR X ANC XDIR 1 NSCD
POS 0 0
NOM 25
REF1 POLY 104 1
2
REF2 POLY 106 1
0
PS1 0
END
PS2 1
POLY 106 1
3
END
END
""","XT"=>string(target))
    mktempdir() do directory
        # Active dependent chain: X's REF2 lands at REF1_resolved+target.
        file=joinpath(directory,"dep.son")
        write(file,head*wdim*xdim(200)*strips*tail)
        project=read_sonnet_project(file)
        resolved=DiffMoM._sonnet_geometry_project(project,1e9)
        scale=project.length_scale
        moved=only(filter(q->q.id==106,resolved.polygons))
        @test moved.vertices./scale≈[350 400 400 350 350;175 175 200 200 175] rtol=8eps()
        # Inactive consistent dimension propagates the same constraint:
        # REF2 lands at REF1_resolved+NOM=175 and carries PS2 with it.
        file=joinpath(directory,"depi.son")
        write(file,head*wdim*xdim(25)*strips*tail)
        resolved=DiffMoM._sonnet_geometry_project(read_sonnet_project(file),1e9)
        moved=only(filter(q->q.id==106,resolved.polygons))
        @test moved.vertices./scale≈[175 400 400 175 175;175 175 200 200 175] rtol=8eps()
        # Inactive inconsistent dimension still re-anchors at REF1+target
        # rather than preserving its saved span.
        file=joinpath(directory,"depj.son")
        write(file,head*wdim*xdim(25)*replace(strips,
            "112.5 175\n400"=>"117.5 175\n400",
            "112.5 200\n112.5 175"=>"117.5 200\n117.5 175")*tail)
        resolved=DiffMoM._sonnet_geometry_project(read_sonnet_project(file),1e9)
        moved=only(filter(q->q.id==106,resolved.polygons))
        @test moved.vertices./scale≈[175 400 400 175 175;175 175 200 200 175] rtol=8eps()
        # A moving reference already placed by a prior dimension leaves the
        # later dimension's ordinary PS2 members untouched (zero extra shift).
        file=joinpath(directory,"depk.son")
        write(file,head*replace(wdim,"PS2 1\nPOLY 104 1\n2\nEND"=>
            "PS2 2\nPOLY 104 1\n2\nPOLY 106 1\n0\nEND")*xdim(25)*strips*tail)
        resolved=DiffMoM._sonnet_geometry_project(read_sonnet_project(file),1e9)
        moved=only(filter(q->q.id==106,resolved.polygons))
        @test moved.vertices./scale≈[175 400 400 112.5 175;175 175 200 200 175] rtol=8eps()
        # An untouched inconsistent dimension re-anchors at REF1+target once
        # the sequential resolver runs (native-verified: 25, not 117.5).
        file=joinpath(directory,"depn.son")
        write(file,head*wdim*replace(xdim(25),"REF1 POLY 104 1\n2"=>
            "REF1 POLY 104 1\n0")*replace(strips,
            "112.5 175\n400"=>"117.5 175\n400",
            "112.5 200\n112.5 175"=>"117.5 200\n117.5 175")*tail)
        resolved=DiffMoM._sonnet_geometry_project(read_sonnet_project(file),1e9)
        moved=only(filter(q->q.id==106,resolved.polygons))
        @test moved.vertices./scale≈[25 400 400 25 25;175 175 200 200 175] rtol=8eps()
        # Record order does not drive resolution: a dependent dimension
        # listed before its mover still resolves last (native-verified:
        # poly106 reaches 175, not 112.5).
        file=joinpath(directory,"depo.son")
        write(file,head*xdim(25)*wdim*strips*tail)
        resolved=DiffMoM._sonnet_geometry_project(read_sonnet_project(file),1e9)
        moved=only(filter(q->q.id==106,resolved.polygons))
        @test moved.vertices./scale≈[175 400 400 175 175;175 175 200 200 175] rtol=8eps()
        # Circular dependencies match the native circular-definition error.
        file=joinpath(directory,"depcirc.son")
        write(file,head*replace(wdim,"REF2 POLY 104 1\n1"=>"REF2 POLY 106 1\n0")*
            xdim(25)*strips*tail)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(
            read_sonnet_project(file),1e9)
        # A symmetric mover's points are still rejected when a non-anchored
        # dimension references them; only anchored dependents compose.
        file=joinpath(directory,"depmixed.son")
        write(file,head*wdim*xdim(200)*replace(strips,
            "0 5 -1 N 106"=>"0 5 -1 N 106")*tail |> s->
            replace(s,"GEOVAR X ANC XDIR 1 NSCD"=>"GEOVAR X SYM XDIR 1 NSCD"))
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(
            read_sonnet_project(file),1e9)
        # A via port re-anchors to its resolved edge midpoint (engine-verified
        # against the parameterized run).
        viahead=replace(head,"""BMET "Lossless" 0 SUP 0 0 0 0
BOX 1 400 500 64 80 20 0
250 1 1 0 0 0 1 "Air"
25 9.8 1 0 0 0 1 "Alumina"
LORGN"""=>"""BMET "Lossless" 0 SUP 0 0 0 0
MET "Copper" 1 NOR 58000000 0.5 1.4
MET "Copper" 1 VOL 58000000 1.4
BOX 1 400 500 64 80 20 0
250 1 1 0 0 0 2 "Air"
25 9.8 1 0.0009 0 0 2 "Alumina"
TECHLAY METAL Copper Copper 0 0
0 0 0 N 0 1 1 100 100 0 0 0 Y
END
END
TECHLAY VIA Via Via 0 0
VIA POLYGON
0 0 1 N 0 1 1 100 100 0 0 0 Y
TOLEVEL GND RING NOCOVERS
END
END
LORGN""")
        viahead=replace(viahead,"""POR1 BOX
POLY 106 1
1
2 50 0 0 0 400 187.5
"""=>"""POR1 VIA
POLY 106 1
0
2 50 0 0 0 118.75 175
""")
        viapoly=raw"""VIA POLYGON
0 5 1 N 106 1 1 100 100 0 0 0 Y
TOLEVEL GND RING NOCOVERS
TLAYNAM Via INH
112.5 175
125 175
125 200
112.5 200
112.5 175
END
"""
        file=joinpath(directory,"vp.son")
        write(file,viahead*wdim*xdim(200)*replace(strips,
            "0 5 -1 N 106 1 1 100 100 0 0 0 Y\n112.5 175\n400 175\n400 200\n112.5 200\n112.5 175\nEND\n"=>"")*
            viapoly*tail)
        resolved=DiffMoM._sonnet_geometry_project(read_sonnet_project(file),1e9)
        moved=only(filter(q->q.id==106,resolved.polygons))
        @test moved.vertices./scale≈[350 125 125 350 350;175 175 200 200 175] rtol=8eps()
        port=only(filter(p->p.kind===:via,resolved.ports))
        @test port.values[6:7]==["237.5","175.0"]
    end
end

@testset "Native moved interior ports re-anchor by geometry, not token" begin
    # Lite 18.53 accepts POR1 STD and POR1 GAP on a GEOVAR-moved interior
    # edge identically: the token is descriptive, the port rides its
    # polygon, and the coordinate annotation re-anchors at the resolved
    # edge midpoint. The archived native matrices are bit-identical
    # between the two tokens (manifest: fixtures/native_geovar_moved_port).
    directory=joinpath(@__DIR__,"fixtures/native_geovar_moved_port")
    reference=planar_read_touchstone(joinpath(directory,"ip_move.s2p"))
    @test reference.s==planar_read_touchstone(joinpath(directory,"ip_gap.s2p")).s
    for (file,kind) in (("ip_move.son",:std),("ip_gap.son",:gap))
        p=read_sonnet_project(joinpath(directory,file))
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        port=only(filter(q->q.number==2,resolved.ports))
        @test port.kind==kind
        # Polygon 2 translates +0.0625 mm; its shared edge midpoint moves
        # from (0.5,0.5) to (0.5,0.5625) in project millimetres.
        @test port.values[6:7]==["0.5","0.5625"]
        moved=only(filter(q->q.id==2,resolved.polygons))
        @test collect(extrema(moved.vertices[2,:]))≈[0.0005,0.000625] rtol=1e-9
        result=solve_sonnet_project(p,1e9;raw=true,mx=128,my=128)
        @test maximum(abs,result.s-only(reference.s))<=.005
    end
    # A moved CUP port still routes to its calibration-group adapter.
    p=read_sonnet_project(joinpath(directory,"ip_gap.son"))
    changed=DiffMoM.SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,
        p.box,p.layers,p.metals,p.top,p.bottom,p.polygons,
        [DiffMoM.SonnetPortSpec(:cup,q.polygon,q.edge,q.number,q.values,q.records) for q in p.ports],
        p.variables,p.components,p.sweeps,p.records)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(changed,1e9)
end

@testset "Native GEOVAR dimension interactions are axis- and side-aware" begin
    # Lite 18.53 intake rules (fixture native_geovar_dim_interactions):
    # the circular-definition verdict fires on moved points shared on the
    # same axis AND same PS side, even when every dimension is inactive;
    # cross-axis and opposite-side sharing resolve. A non-sequential
    # dimension's same-axis reference on a moved point is an unorderable
    # dependency; sequential anchored dimensions may depend on anything.
    directory=joinpath(@__DIR__,"fixtures/native_geovar_dim_interactions")
    @test isfile(joinpath(directory,"manifest.json"))
    base=Dict("L"=>2600.0,"Y"=>1250.0,"X"=>600.0,"W2"=>1300.0,
        "A2"=>600.0,"A3"=>600.0,"A4"=>600.0)
    resolve(file,vars=Dict{String,Float64}())=
        DiffMoM._sonnet_geometry_project(
            read_sonnet_project(joinpath(directory,file)),1e9,
            merge(base,vars))
    # Circular verdicts: same-axis same-side overlap (even both-inactive),
    # subset overlap, and same-axis dependency crossings (mutual or
    # one-way, anchored or symmetric owner).
    for (file,vars,msg) in (
            ("b4_sym_sym.son",Dict("W"=>1300.0,"W2"=>700.0),"overlap"),
            ("b4_ss_bothinact.son",Dict("W"=>1300.0,"W2"=>1300.0),"overlap"),
            ("b4_part_act.son",Dict("W"=>1400.0,"A2"=>700.0,"W2"=>1400.0),"overlap"),
            ("b4_dup_act.son",Dict("W"=>1400.0,"A2"=>700.0,"W2"=>1400.0),"overlap"),
            ("b4_dep_anc_on_sym.son",Dict("W"=>1400.0,"A3"=>700.0),"depend"),
            ("b4_sym_dep_on_anc.son",Dict("W"=>1400.0,"A4"=>700.0),"depend"),
            ("b4_dep_oneway.son",Dict("W"=>1400.0,"A4"=>700.0),"depend"),
            ("b4_dep_symowner.son",Dict("W"=>1400.0,"A2"=>700.0,"W2"=>1400.0),"circular"))
        @test_throws ArgumentError resolve(file,vars)
    end
    # b4_sym_anc keeps its phantom-delta regression coverage: the W
    # dimension is inactive (target==NOM) yet its SI round-trip leaves a
    # ~1e-17 span delta that used to register false ownership and
    # conflicts. Native accepts: the ANC anchor resolves after it.
    resolved=resolve("b4_sym_anc.son",Dict("A2"=>700.0))
    @test resolved!==nothing
    # Pure cross-axis sharing with both dimensions active resolves: v3 of
    # polygon 97 sits in W's XDIR PS1 and L's YDIR PS2, so each dimension
    # moves its own axis component (manifest records the matching native
    # acceptance and an 8.3e-5 literal residual for this composition).
    resolved=resolve("b4_xaxis.son",Dict("W"=>1400.0,"L"=>2700.0,"X"=>100.0))
    moved=only(filter(q->q.id==97,resolved.polygons))
    @test moved.vertices[1,4]<0.23749  # v3.x took W's XDIR PS1 shift
    @test moved.vertices[2,4]>0.28702  # v3.y took L's YDIR PS2 shift
    # The undecoded same-axis active ANC+SYM joint ordering stays a loud
    # residual (native accepts the combination; the composition law is
    # not yet proven).
    @test_throws ArgumentError resolve("b4_symL_act.son",
        Dict("W"=>1400.0,"A2"=>700.0,"L"=>2700.0,"X"=>100.0))
end

@testset "Native GEOVAR header grammar scans and folds case" begin
    # Lite 18.53 (fixture native_geovar_header_grammar): kind and axis are
    # positional and case-insensitive; trailing tokens are scanned - first
    # NSCD/SCUNI/SCXY sets scaling, first finite numeric token sets the
    # sign-normalized display direction; unknown tokens reject while
    # scaling is unset and are ignored after it; missing fields default
    # to NSCD/+1 including a bare four-token header. The direction field
    # itself is ignored for application whenever the reference span is
    # nonzero (all variants below land the via identically). Structural
    # keywords fold case; DIM unit entries and VARSWP contents stay
    # strict. EQN is optional after NOM or after REF2.
    directory=joinpath(@__DIR__,"fixtures/native_geovar_header_grammar")
    manifest=JSON.parsefile(joinpath(directory,"manifest.json"))
    expected=[10550.0,10600.0,10600.0,10550.0]
    for name in ("x1200","xdm1","xd0","xd2","xdm2","xd3","xh5","xh7",
            "xdir_only","xlc","g_swaptok","dirstr","onlysc","dir_sc",
            "j2","j5","j8","e1","e3","e4","e5","k4","geo_lc","all_lc3",
            "eqn_pos1","eqn_pos2","poly_lc2")
        file="b1_"*name*".son"
        @test manifest["cases"]["b1_"*name]["verdict"]=="accept"
        p=read_sonnet_project(joinpath(directory,file))
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        via=only(filter(q->q.kind==:via,resolved.polygons))
        @test isapprox(via.vertices[1,1:end-1]./p.length_scale,expected,atol=1e-9)
    end
    # First scaling keyword wins: SCUNI is the effective adapter here and
    # lands one cell wider (native parameterized s2p is bit-identical to
    # this literal per the manifest).
    p=read_sonnet_project(joinpath(directory,"b1_j7.son"))
    resolved=DiffMoM._sonnet_geometry_project(p,1e9)
    via=only(filter(q->q.kind==:via,resolved.polygons))
    @test isapprox(via.vertices[1,1:end-1]./p.length_scale,
        [10550.0,10650.0,10650.0,10550.0],atol=1e-9)
    # Lowercase kind on an active symmetric dimension resolves the
    # symmetric expansion (impl literal s2p bit-identical to the native
    # parameterized run per the manifest).
    p=read_sonnet_project(joinpath(directory,"b1_casa.son"))
    resolved=DiffMoM._sonnet_geometry_project(p,1e9)
    sheet=only(filter(q->q.id==97,resolved.polygons))
    @test isapprox(sheet.vertices[1,1:end-1]./p.length_scale,
        [9050.0,10950.0,10950.0,9050.0],atol=1e-9)
    # The reject set covers unsupported kinds/axes/scaling keywords,
    # kind/axis position swaps, unknown tokens while scaling is unset,
    # second numeric tokens and truncated headers - all with the native
    # error evidence archived beside them.
    for name in ("zdir","scd","kind_abs","kind_len","nc","scaled","g_foo",
            "g_4tok","extra_mid","sw1","sw4","junk_sc","j1","j3","j4","j6","e2")
        file="b1_"*name*".son"
        @test manifest["cases"]["b1_"*name]["verdict"]=="reject"
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(
            read_sonnet_project(joinpath(directory,file)),1e9)
    end
end
end
