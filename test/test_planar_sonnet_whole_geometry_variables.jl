module NativeSonnetWholeGeometryVariableTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_geovar_whole_polygon")
module PreviousGeometry
using DiffMoM
using DiffMoM: _sonnet_error,_sonnet_section,_sonnet_parse_scalar,
    _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,_checked_payload_sum,_sonnet_scalar_project_payload,
    _checked_array_payload_bytes,_enforce_payload_limit,_circuit_stored_real,_P2
include(joinpath(@__DIR__,"fixtures/native_geovar_point_multiplicity/source_before/PlanarSonnetGeometryVariables.jl"))
end
const observations=NamedTuple[]
@testset "Native whole-polygon logical vertices and full physical output" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    before=TOML.parsefile(joinpath(fixture,"source_before/geovar_whole_polygon_before_20261004.toml"))
    @test before["source_unchanged"] && before["source_before"]==before["source_after"]
    @test length(before["cases"])==2
    for row in before["cases"]
        @test row["parameter_status"]=="REJECT" && row["literal_status"]=="PASS"
        @test row["literal_full_s_error"]<=.005 && row["literal_voltage_residual"]<=1e-9
    end
    directory=joinpath(fixture,"variants");proof=TOML.parsefile(joinpath(directory,"comparison.toml"))
    @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
    @test length(proof["cases"])==16
    for row in proof["cases"]
        tag=row["tag"];p=read_sonnet_project(joinpath(directory,tag*"_parameter.son"))
        q=read_sonnet_project(joinpath(directory,tag*"_literal.son"))
        explicit=read_sonnet_project(joinpath(directory,tag*"_explicit.son"))
        saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
        @test row["status"]=="PASS" && row["all_bit_identical"]
        @test_throws ArgumentError PreviousGeometry._sonnet_geometry_project(p,1e9)
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        for (polygon,literal) in zip(resolved.polygons,q.polygons)
            @test polygon.vertices≈literal.vertices rtol=8eps(Float64)
            @test polygon.vertices!==p.polygons[polygon.id].vertices
            @test polygon.vertices[:,end]==polygon.vertices[:,1]
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(resolved.polygons,DiffMoM._sonnet_geometry_project(explicit,1e9).polygons))
        for (port,literal) in zip(resolved.ports,q.ports)
            @test parse.(Float64,port.values[6:7])≈parse.(Float64,literal.values[6:7]) rtol=8eps(Float64)
        end
        lowered=sonnet_planar_problem(p;freq=1e9);literal=sonnet_planar_problem(q;freq=1e9)
        @test length(lowered.sheets)==length(literal.sheets) && all(a.mask==b.mask for (a,b) in zip(lowered.sheets,literal.sheets))
        native=planar_read_touchstone(joinpath(directory,tag*"_parameter/native_raw.s2p"))
        @test native.s==planar_read_touchstone(joinpath(directory,tag*"_literal/native_raw.s2p")).s==planar_read_touchstone(joinpath(directory,tag*"_explicit/native_raw.s2p")).s
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
        end
        full_s_error=maximum(abs,result.s-only(native.s));residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test full_s_error<=.005 && residual<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10
        @test opnorm(result.s)<=1+1e-9
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(p.ports,savedports))
        for _ in 1:30;DiffMoM._sonnet_geometry_project(p,1e9);DiffMoM._sonnet_geometry_project(explicit,1e9);end
        whole_gc_bytes=@allocated DiffMoM._sonnet_geometry_project(p,1e9)
        explicit_gc_bytes=@allocated DiffMoM._sonnet_geometry_project(explicit,1e9)
        @test whole_gc_bytes<=explicit_gc_bytes
        push!(observations,(case=tag,full_s_error,original_voltage_residual=residual,whole_gc_bytes,explicit_gc_bytes))
    end
end
@testset "Whole-polygon aggregate resources, identity and unsupported reference scaling" begin
    directory=joinpath(fixture,"variants")
    p=read_sonnet_project(joinpath(directory,"nscd_ydir_positive_expand_parameter.son"))
    saved=deepcopy(p.polygons)
    reserve=DiffMoM._checked_payload_sum("native geometry variable workspace",DiffMoM._sonnet_scalar_project_payload(p),
        DiffMoM._checked_array_payload_bytes(UInt8,256,length(p.records)),DiffMoM._checked_array_payload_bytes(UInt8,1024,1))
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_points=5)
    @test DiffMoM._sonnet_geometry_project(p,1e9;max_points=6).polygons[2].vertices==DiffMoM._sonnet_geometry_project(p,1e9).polygons[2].vertices
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_bytes=reserve+4*256-1)
    @test DiffMoM._sonnet_geometry_project(p,1e9;max_bytes=reserve+4*256).polygons[2].vertices==DiffMoM._sonnet_geometry_project(p,1e9).polygons[2].vertices
    @test DiffMoM._sonnet_geometry_project(p,1e9,Dict("Move"=>.125);max_bytes=reserve)===p
    original=read(p.source,String)
    mktempdir() do temporary
        for (suffix,text) in (("unknown",replace(original,"POLY 2 0"=>"POLY 999 0")),
                ("negative",replace(original,"POLY 2 0"=>"POLY 2 -1")),
                ("anchor",replace(original,"POLY 2 0"=>"POLY 3 0")),
                ("repeated",replace(original,"PS2 1\nPOLY 2 0"=>"PS2 2\nPOLY 2 0\nPOLY 2 0")))
            source=joinpath(temporary,suffix*".son");write(source,text);invalid=read_sonnet_project(source)
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(invalid,1e9)
            suffix=="repeated" || continue
            extra=DiffMoM._checked_payload_sum("native geometry variable workspace",DiffMoM._sonnet_scalar_project_payload(invalid),
                DiffMoM._checked_array_payload_bytes(UInt8,256,length(invalid.records)),DiffMoM._checked_array_payload_bytes(UInt8,1024,1))
            err=try DiffMoM._sonnet_geometry_project(invalid,1e9;max_points=10,max_bytes=extra+4*256);nothing catch caught;caught end
            @test err isa ArgumentError && occursin("max_bytes",sprint(showerror,err))
            err=try DiffMoM._sonnet_geometry_project(invalid,1e9;max_points=9);nothing catch caught;caught end
            @test err isa ArgumentError && occursin("point budget",sprint(showerror,err))
        end
    end
    for mode in ("scuni","scxy")
        source=joinpath(fixture,"initial_controls",mode*"_parameter.son")
        parameter=read_sonnet_project(source)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(parameter,1e9)
        @test_throws ArgumentError solve_sonnet_project(parameter,1e9;raw=true)
    end
    @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
end
end
