module NativeSonnetZeroNominalGeometryVariableTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_zero_nominal_geovar")
module PreviousGeometry
using DiffMoM
using DiffMoM: _sonnet_error,_sonnet_section,_sonnet_parse_scalar,
    _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,_checked_payload_sum,_sonnet_scalar_project_payload,
    _checked_array_payload_bytes,_enforce_payload_limit,_circuit_stored_real,_P2
include(joinpath(@__DIR__,"fixtures/native_zero_nominal_geovar/source_before/PlanarSonnetGeometryVariables.jl"))
end
const observations=NamedTuple[]

@testset "Native zero saved anchored offsets and scoped physical controls" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    proof=TOML.parsefile(joinpath(fixture,"comparison.toml"))
    baseline=TOML.parsefile(joinpath(fixture,"source_before/literal_physics.toml"))
    @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
    @test baseline["source_unchanged"] && baseline["source_before"]==baseline["source_after"]
    @test length(proof["cases"])==length(baseline["cases"])==8
    @test baseline["status"]=="FAIL" && count(row->row["status"]=="PASS",baseline["cases"])==6
    for row in proof["cases"]
        tag="$(row["axis"])_dir_$(row["direction"])_target_$(row["target"])"
        p=read_sonnet_project(joinpath(fixture,"native",tag*"_parameter.son"))
        q=read_sonnet_project(joinpath(fixture,"native",tag*"_literal.son"))
        saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
        @test row["parameter_status"]==row["literal_status"]=="ACCEPT"
        @test row["bit_identical_s"] && row["full_s_error"]==0
        @test row["public_geometry_status"]=="REJECT"
        @test_throws ArgumentError PreviousGeometry._sonnet_geometry_project(p,1e9)
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        @test resolved.polygons[1].vertices≈q.polygons[1].vertices rtol=8eps(Float64)
        @test resolved.polygons[1].vertices!==p.polygons[1].vertices
        @test resolved.polygons[1].vertices[:,end]==resolved.polygons[1].vertices[:,1]
        @test all(parse.(Float64,a.values[6:7])≈parse.(Float64,b.values[6:7])
            for (a,b) in zip(resolved.ports,q.ports))
        lowered=sonnet_planar_problem(p;freq=1e9);literal=sonnet_planar_problem(q;freq=1e9)
        @test all(a.mask==b.mask for (a,b) in zip(lowered.sheets,literal.sheets))
        @test DiffMoM._sonnet_geometry_project(p,1e9,Dict("Offset"=>0.))===p
        native=planar_read_touchstone(joinpath(fixture,"native",tag*"_parameter/native_raw.s2p"))
        @test native.s==planar_read_touchstone(joinpath(fixture,"native",tag*"_literal/native_raw.s2p")).s
        before=only(filter(b->b["axis"]==row["axis"] && b["direction"]==row["direction"] && b["target"]==row["target"],baseline["cases"]))
        # Preserve the literal baseline's two historical FAILs. The current
        # vertical-ray raster repair must pass every original physical gate.
        if before["status"]=="PASS"
            @test before["full_s_error"]<=.005 && before["original_voltage_residual"]<=1e-9
        end
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
        end
        full_s_error=maximum(abs,result.s-only(native.s))
        residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test full_s_error<=.005 && residual<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10 && opnorm(result.s)<=1+1e-9
        push!(observations,(case=tag,full_s_error,original_voltage_residual=residual))
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(p.ports,savedports))
    end
end

@testset "Zero-offset budgets and unsupported or unrepresentable dimensions" begin
    source=joinpath(fixture,"native/XDIR_dir_-1_target_0.125_parameter.son")
    p=read_sonnet_project(source);saved=deepcopy(p.polygons)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_points=2)
    @test DiffMoM._sonnet_geometry_project(p,1e9;max_points=3).polygons[1].vertices==
        DiffMoM._sonnet_geometry_project(p,1e9).polygons[1].vertices
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_bytes=1)
    mktempdir() do directory
        original=read(source,String)
        for (name,text) in (("scaled",replace(original,"NSCD"=>"SCUNI")),
                ("two_axes",replace(original,"NSCD"=>"SCXY")),
                ("symmetric",replace(original," ANC "=>" SYM ")),
                ("radial",replace(original," ANC "=>" RAD ")),
                ("distinct",replace(original,"REF1 POLY 1 1\n2"=>"REF1 POLY 1 1\n3")),
                ("negative",replace(original,"NOM 0"=>"NOM -0.125")))
            file=joinpath(directory,name*".son");write(file,text);candidate=read_sonnet_project(file)
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(candidate,1e9)
        end
    end
    tiny=SonnetProject(p.source,p.units,nextfloat(0.),p.frequency_scale,p.box,p.layers,p.metals,
        p.top,p.bottom,p.polygons,p.ports,p.variables,p.components,p.sweeps,p.records)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(tiny,1e9)
    @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
end
end
