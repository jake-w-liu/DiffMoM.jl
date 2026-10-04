module NativeSonnetPointMultiplicityTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_geovar_point_multiplicity")
module PreviousGeometry
using DiffMoM
using DiffMoM: _sonnet_error,_sonnet_section,_sonnet_parse_scalar,
    _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,_checked_payload_sum,_sonnet_scalar_project_payload,
    _checked_array_payload_bytes,_enforce_payload_limit,_circuit_stored_real,_P2
include(joinpath(@__DIR__,"fixtures/native_geovar_point_multiplicity/source_before/PlanarSonnetGeometryVariables.jl"))
end
const observations=NamedTuple[]

@testset "Actual native repeated ordinary vertices and original failure" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    before=TOML.parsefile(joinpath(fixture,"source_before/geovar_ordinary_multiplicity_before_20261004.toml"))
    @test before["source_unchanged"] && before["source_before"]==before["source_after"]
    @test length(before["cases"])==10
    for row in before["cases"]
        directory=joinpath(fixture,row["kind"]=="sym" ? "symmetric" : "anchored_radial")
        tag=row["tag"];suffix=row["literal_suffix"]
        p=read_sonnet_project(joinpath(directory,tag*"_parameter.son"))
        q=read_sonnet_project(joinpath(directory,tag*"_"*suffix*".son"))
        saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
        legacy=PreviousGeometry._sonnet_geometry_project(p,1e9)
        @test maximum(abs,only(legacy.polygons).vertices-only(q.polygons).vertices)>1e-5
        @test row["literal_full_s_error"]<=.005 && row["literal_voltage_residual"]<=1e-9
        tag=="rad_ordinary" && @test row["parameter_full_s_error"]>1
        startswith(tag,"scxy_ps") && @test row["parameter_full_s_error"]>.005
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        @test only(resolved.polygons).vertices≈only(q.polygons).vertices rtol=8eps(Float64)
        @test only(resolved.polygons).vertices!==only(p.polygons).vertices
        for (port,literal) in zip(resolved.ports,q.ports)
            @test parse.(Float64,port.values[6:7])≈parse.(Float64,literal.values[6:7]) rtol=8eps(Float64)
        end
        @test only(sonnet_planar_problem(p;freq=1e9).sheets).mask==only(sonnet_planar_problem(q;freq=1e9).sheets).mask
        native=planar_read_touchstone(joinpath(directory,tag*"_parameter/native_raw.s2p"))
        @test native.s==planar_read_touchstone(joinpath(directory,tag*"_"*suffix*"/native_raw.s2p")).s
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        full_s_error=maximum(abs,result.s-only(native.s));raw=result.raw
        rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
        end
        residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test full_s_error<=.005 && residual<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10
        @test opnorm(result.s)<=1+1e-9
        for _ in 1:30;DiffMoM._sonnet_geometry_project(p,1e9);PreviousGeometry._sonnet_geometry_project(p,1e9);end
        current_bytes=@allocated DiffMoM._sonnet_geometry_project(p,1e9)
        previous_bytes=@allocated PreviousGeometry._sonnet_geometry_project(p,1e9)
        @test current_bytes<=previous_bytes
        for options in ((max_points=1,),(max_bytes=1,),(max_parameters=0,))
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;options...)
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(p.ports,savedports))
        push!(observations,(case=tag,full_s_error,original_voltage_residual=residual,current_bytes,previous_bytes))
    end
end

@testset "Explicit reference repetitions reject before physical output" begin
    directory=joinpath(fixture,"anchored_radial")
    proof=TOML.parsefile(joinpath(directory,"comparison.toml"))
    @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
    for row in proof["cases"]
        row["kind"]=="reference" || continue
        @test row["parameter_status"]=="ACCEPT"
        @test row["deduplicated_full_s_error"]>1e-3
        p=read_sonnet_project(joinpath(directory,row["tag"]*"_parameter.son"));saved=deepcopy(p.polygons)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9)
        @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true)
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
    end
end
end
