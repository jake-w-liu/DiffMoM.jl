module NativeSonnetRadialReferenceHeaderTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_rad_reference_headers")
module PreviousGeometry
using DiffMoM
using DiffMoM: _sonnet_error,_sonnet_section,_sonnet_parse_scalar,
    _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,_checked_payload_sum,_sonnet_scalar_project_payload,
    _checked_array_payload_bytes,_enforce_payload_limit,_circuit_stored_real,_P2
include(joinpath(@__DIR__,"fixtures/native_rad_reference_headers/geometry_source_before.jl"))
end
const observations=NamedTuple[]
@testset "Native radial explicit reference ignores scaling headers" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    proof=TOML.parsefile(joinpath(fixture,"comparison.toml"))
    @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
    @test length(proof["cases"])==24
    for row in proof["cases"]
        name=row["name"];tag=row["tag"]
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        q=read_sonnet_project(joinpath(fixture,"inputs",tag*"_literal.son"))
        saved=deepcopy(p.polygons);ports=deepcopy(p.ports)
        @test row["native_bit_identical"] && row["native_full_s_error"]==0
        @test row["literal_public_full_s_error"]<=.005 && row["literal_original_voltage_residual"]<=1e-9
        if row["mode"]!="NSCD"
            @test row["before_public_status"]=="REJECT" && occursin("reference repetitions",row["before_public_error"])
            @test_throws ArgumentError PreviousGeometry._sonnet_geometry_project(p,1e9)
        end
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        @test all(isapprox(a.vertices,b.vertices;rtol=8eps(Float64)) for (a,b) in zip(resolved.polygons,q.polygons))
        @test all(a.vertices!==b.vertices && a.vertices[:,end]==a.vertices[:,1] for (a,b) in zip(resolved.polygons,p.polygons))
        @test all(isapprox(parse.(Float64,a.values[6:7]),parse.(Float64,b.values[6:7]);rtol=8eps(Float64)) for (a,b) in zip(resolved.ports,q.ports))
        lowered=sonnet_planar_problem(p;freq=1e9);literal=sonnet_planar_problem(q;freq=1e9)
        @test length(lowered.sheets)==length(literal.sheets) && all(a.mask==b.mask for (a,b) in zip(lowered.sheets,literal.sheets))
        native=planar_read_touchstone(joinpath(fixture,name,"native_raw.s2p"))
        @test native.s==planar_read_touchstone(joinpath(fixture,tag*"_literal","native_raw.s2p")).s
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
        end
        error=maximum(abs,result.s-only(native.s));residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test error<=.005 && residual<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10 && opnorm(result.s)<=1+1e-9
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved)) && all(a.values==b.values for (a,b) in zip(p.ports,ports))
        reserve=DiffMoM._checked_payload_sum("native geometry variable workspace",DiffMoM._sonnet_scalar_project_payload(p),
            DiffMoM._checked_array_payload_bytes(UInt8,256,length(p.records)),DiffMoM._checked_array_payload_bytes(UInt8,1024,1))
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_points=5)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_bytes=reserve+4*256-1)
        @test DiffMoM._sonnet_geometry_project(p,1e9;max_points=6,max_bytes=reserve+4*256).polygons[2].vertices==resolved.polygons[2].vertices
        @test DiffMoM._sonnet_geometry_project(p,1e9,Dict("Move"=>.125);max_bytes=reserve)===p
        for _ in 1:30;DiffMoM._sonnet_geometry_project(p,1e9);end
        allocated=@allocated DiffMoM._sonnet_geometry_project(p,1e9)
        push!(observations,(case=name,full_s_error=error,original_voltage_residual=residual,cumulative_geometry_gc_bytes=allocated))
    end
end
@testset "Radial reference expansion preserves ownership rejection" begin
    source=joinpath(fixture,"rad_ydir_positive_expand_scxy.son")
    original=read(source,String)
    mktempdir() do directory
        for (name,text) in (("repeated",replace(original,"PS2 1\nPOLY 2 0"=>"PS2 2\nPOLY 2 0\nPOLY 2 0")),
                ("anchor",replace(original,"PS1 0\nEND"=>"PS1 1\nPOLY 3 1\n2\nEND")))
            path=joinpath(directory,name*".son");write(path,text)
            invalid=read_sonnet_project(path)
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(invalid,1e9)
            @test_throws ArgumentError solve_sonnet_project(invalid,1e9;raw=true)
        end
    end
end
end
