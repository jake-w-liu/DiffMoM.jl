module NativeSonnetNominalGeometryGrammarTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra

const fixture=joinpath(@__DIR__,"fixtures/native_nominal_geovar_grammar")

@testset "Native nominal dimension grammar and accepted unused identities" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    proof=TOML.parsefile(joinpath(fixture,"comparison.toml"))
    @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
    @test length(proof["cases"])==5
    baseline=read_sonnet_project(joinpath(fixture,"valid.son"))
    mask=only(sonnet_planar_problem(baseline;freq=1e9).sheets).mask
    native=planar_read_touchstone(joinpath(fixture,"valid/native_raw.s2p"))
    for row in proof["cases"]
        name=row["case"]
        @test row["reader_acceptance"]=="NOMINAL_PASSTHROUGH"
        p=read_sonnet_project(joinpath(fixture,name*".son"))
        saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
        if name in ("display_nonnumeric","negative_group_count")
            @test row["native_acceptance"]=="REJECTED"
            phrase=name=="display_nonnumeric" ? "Geometry Point" : "greater than or equal to 0"
            @test occursin(phrase,row["native_stderr"])
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9)
            @test_throws ArgumentError sonnet_planar_problem(p;freq=1e9)
        elseif name=="unknown_point"
            @test row["native_acceptance"]=="ACCEPTED"
            reference=planar_read_touchstone(joinpath(fixture,name,"native_raw.s2p"))
            @test maximum(abs,only(reference.s)-only(native.s))>1
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9)
            @test_throws ArgumentError sonnet_planar_problem(p;freq=1e9)
        else
            @test row["native_acceptance"]=="ACCEPTED"
            @test DiffMoM._sonnet_geometry_project(p,1e9)===p
            @test only(sonnet_planar_problem(p;freq=1e9).sheets).mask==mask
            reference=planar_read_touchstone(joinpath(fixture,name,"native_raw.s2p"))
            @test reference.frequencies==native.frequencies && reference.s==native.s
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(p.ports,savedports))
    end
    effect=TOML.parsefile(joinpath(fixture,"reference_effect/comparison.toml"))
    @test effect["source_unchanged"] && effect["source_before"]==effect["source_after"]
    @test length(effect["cases"])==4
    for row in effect["cases"]
        @test row["original_voltage_residual"]<=1e-9
        if row["case"]=="unknown_point"
            @test row["current_full_s_error"]>1
        else
            @test row["current_full_s_error"]<=.005
        end
    end
    result=solve_sonnet_project(baseline,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
    @test maximum(abs,result.s-only(native.s))<=.005
    raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
    for b in eachindex(raw.problem.basis.port)
        port=raw.problem.basis.port[b];iszero(port) && continue
        rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
    end
    @test norm(raw.z_mom*raw.currents-rhs)/norm(rhs)<=1e-9
end
end
