module NativeSonnetTwoAxisGeometryVariableTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_two_axis_geovar")
const native=joinpath(fixture,"native")
const observations=NamedTuple[]

@testset "Actual native two-axis dimensions, attached ports and full S" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    proof=TOML.parsefile(joinpath(native,"comparison.toml"))
    @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
    @test length(proof["cases"])==16
    for row in proof["cases"]
        @test row["status"]=="PASS" && row["bit_identical_s"] && row["full_s_error"]==0
        haskey(row,"axis_only_full_s_error") && @test row["axis_only_full_s_error"]>.014
        tag=row["tag"]
        p=read_sonnet_project(joinpath(native,tag*"_parameter.son"))
        q=read_sonnet_project(joinpath(native,tag*"_proportional_literal.son"))
        saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(p.ports,savedports))
        @test only(resolved.polygons).vertices≈only(q.polygons).vertices rtol=8eps(Float64)
        @test only(resolved.polygons).vertices!==only(p.polygons).vertices
        # The notch moves in both axes while the owning wall stays fixed.
        @test all(only(resolved.polygons).vertices[axis,:]!=only(p.polygons).vertices[axis,:] for axis in 1:2)
        for (port,literal) in zip(resolved.ports,q.ports)
            @test parse.(Float64,port.values[6:7])≈parse.(Float64,literal.values[6:7]) rtol=8eps(Float64)
            @test port.values!==only(filter(v->v.number==port.number,p.ports)).values
        end
        @test only(sonnet_planar_problem(p;freq=1e9).sheets).mask==only(sonnet_planar_problem(q;freq=1e9).sheets).mask
        reference=planar_read_touchstone(joinpath(native,tag*"_parameter/native_raw.s2p"))
        @test reference.s==planar_read_touchstone(joinpath(native,tag*"_proportional_literal/native_raw.s2p")).s
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        full_s_error=maximum(abs,result.s-only(reference.s))
        @test full_s_error<=.005
        raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
        end
        original_voltage_residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test original_voltage_residual<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10
        @test opnorm(result.s)<=1+1e-9
        @test DiffMoM._sonnet_geometry_project(p,1e9,Dict("Width"=>row["nominal"]))===p
        push!(observations,(case=tag,full_s_error,original_voltage_residual))
    end
end

@testset "Two-axis resource, override, ownership and native invalid ports" begin
    for row in TOML.parsefile(joinpath(native,"comparison.toml"))["cases"]
        tag=row["tag"]
        p=read_sonnet_project(joinpath(native,tag*"_parameter.son"))
        saved=deepcopy(p.polygons)
        other=row["target"]>.25 ? replace(tag,"_expand"=>"_contract") : replace(tag,"_contract"=>"_expand")
        target=row["target"]>.25 ? .125 : .375
        q=read_sonnet_project(joinpath(native,other*"_proportional_literal.son"))
        @test only(DiffMoM._sonnet_geometry_project(p,1e9,Dict("Width"=>target)).polygons).vertices≈only(q.polygons).vertices rtol=8eps(Float64)
        for options in ((max_bytes=1,),(max_points=1,),(max_parameters=0,))
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;options...)
        end
        for value in (0.,-1.,NaN,Inf,BigFloat("1e1000"),BigFloat("1e-1000"))
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9,Dict("Width"=>value))
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
    end
    rejected=joinpath(fixture,"rejected_diagonal_port")
    @test occursin("Port 2 is diagonal",read(joinpath(rejected,"native/engine_stderr.log"),String))
    p=read_sonnet_project(joinpath(rejected,"offwall.son"));saved=deepcopy(p.polygons)
    @test_throws ArgumentError sonnet_planar_problem(p;freq=1e9)
    @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
end
end
