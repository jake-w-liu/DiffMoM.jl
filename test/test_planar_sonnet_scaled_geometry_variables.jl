module NativeSonnetScaledGeometryVariableTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra,Random

const fixture=joinpath(@__DIR__,"fixtures/native_scaled_geovar")
const native=joinpath(fixture,"native")
const observations=NamedTuple[]

function coordinate_sum(values)
    total=0.
    for value in values
        total+=DiffMoM._sonnet_geovar_scaled_coordinate(value,.375,.625,.25,.375,true)
    end
    total
end

function check_native_case(directory,tag)
    p=read_sonnet_project(joinpath(directory,tag*"_parameter.son"))
    q=read_sonnet_project(joinpath(directory,tag*"_proportional_literal.son"))
    saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
    resolved=DiffMoM._sonnet_geometry_project(p,1e9)
    @test resolved.source==p.source && resolved.records===p.records
    @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
    @test all(a.values==b.values for (a,b) in zip(p.ports,savedports))
    @test only(resolved.polygons).vertices≈only(q.polygons).vertices rtol=8eps(Float64)
    @test only(resolved.polygons).vertices!==only(p.polygons).vertices
    for (port,expected) in zip(resolved.ports,q.ports)
        @test parse.(Float64,port.values[6:7])≈parse.(Float64,expected.values[6:7]) rtol=8eps(Float64)
    end
    @test only(sonnet_planar_problem(p;freq=1e9).sheets).mask==
        only(sonnet_planar_problem(q;freq=1e9).sheets).mask
    reference=planar_read_touchstone(joinpath(directory,tag*"_parameter/native_raw.s2p"))
    @test reference.s==planar_read_touchstone(joinpath(directory,tag*"_proportional_literal/native_raw.s2p")).s
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
    push!(observations,(directory=basename(directory),case=tag,full_s_error,original_voltage_residual))
end

@testset "Actual native one-axis scaled dimensions and rejected translation" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    for name in ("native","native_x","native_contraction")
        directory=joinpath(fixture,name)
        proof=TOML.parsefile(joinpath(directory,"comparison.toml"))
        @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
        @test length(proof["cases"])==(name=="native_contraction" ? 8 : 4)
        for row in proof["cases"]
            @test row["status"]=="PASS"
            @test get(row,"bit_identical_proportional",get(row,"bit_identical_s",false))
            @test get(row,"proportional_full_s_error",get(row,"full_s_error",NaN))==0
            haskey(row,"translation_full_s_error") && @test row["translation_full_s_error"]>.006
            tag=get(row,"tag",lowercase(row["kind"])*(row["direction"]==1 ? "_positive" : "_negative"))
            check_native_case(directory,tag)
        end
    end
end

@testset "Scaled dimensions retain public domains and resource rejection" begin
    p=read_sonnet_project(joinpath(native,"anc_positive_parameter.son"))
    @test DiffMoM._sonnet_geometry_project(p,1e9,Dict("Width"=>.25))===p
    for directory in (native,joinpath(fixture,"native_x")),kind in ("anc","sym"),sign in ("positive","negative")
        tag=kind*"_"*sign
        parameter=read_sonnet_project(joinpath(directory,tag*"_parameter.son"))
        axis=directory==native ? "ydir" : "xdir"
        literal=read_sonnet_project(joinpath(fixture,"native_contraction",axis*"_"*tag*"_proportional_literal.son"))
        result=DiffMoM._sonnet_geometry_project(parameter,1e9,Dict("Width"=>.125))
        @test only(result.polygons).vertices≈only(literal.polygons).vertices rtol=8eps(Float64)
        @test parameter.variables["Width"]=="0.375"
    end
    saved=deepcopy(p.polygons)
    for options in ((max_bytes=1,),(max_points=1,),(max_points=6,),(max_parameters=0,))
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;options...)
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
    end
    for value in (0.,-1.,NaN,Inf,BigFloat("1e1000"),BigFloat("1e-1000"))
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9,Dict("Width"=>value))
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
    end
    helper=DiffMoM._sonnet_geovar_scaled_coordinate
    # Independent high-precision affine oracle, not the helper's FMA route.
    rng=MersenneTwister(6174)
    values=1 .+ rand(rng,8192)
    coordinate_sum(values)
    @test (@allocated coordinate_sum(values))==0
    for symmetric in (false,true),_ in 1:120
        first=randn(rng);second=first+rand(rng);value=randn(rng)
        nominal=exp2(rand(rng)*30-15);target=nominal*exp2(rand(rng)*12-6)
        expected=setprecision(BigFloat,4608) do
            anchor=symmetric ? (BigFloat(first)+BigFloat(second))/2 : BigFloat(first)
            Float64(anchor+(BigFloat(value)-anchor)*BigFloat(target)/BigFloat(nominal))
        end
        @test abs(helper(value,first,second,nominal,target,symmetric)-expected)<=
            8eps(Float64)*max(abs(value),abs(expected),abs(first),abs(second))
    end
    for symmetric in (false,true),_ in 1:64
        value=ldexp(randn(rng),rand(rng,-1000:1000))
        first=ldexp(randn(rng),rand(rng,-1000:1000))
        second=ldexp(randn(rng),rand(rng,-1000:1000))
        nominal=ldexp(1.,rand(rng,-1000:1000));target=ldexp(1.,rand(rng,-1000:1000))
        expected=setprecision(BigFloat,4608) do
            anchor=symmetric ? (BigFloat(first)+BigFloat(second))/2 : BigFloat(first)
            Float64(anchor+(BigFloat(value)-anchor)*BigFloat(target)/BigFloat(nominal))
        end
        actual=helper(value,first,second,nominal,target,symmetric)
        if isfinite(expected)
            @test abs(actual-expected)<=8eps(Float64)*max(abs(value),abs(expected),abs(first),abs(second))
        else
            @test actual==expected
        end
    end
    tiny=nextfloat(0.);large=floatmax(Float64)
    @test helper(tiny,0.,0.,tiny,large,false)==large
    @test helper(-large,large,large,1.,.5,false)==0.
    @test helper(0.,0.,tiny,tiny,3tiny,true)==-tiny
    @test helper(1e300,0.,0.,prevfloat(1.),1.,false)>1e300
    # Exceptional arithmetic preserves its result under caller precision and
    # rounding settings and restores those settings after returning.
    for bits in (64,160),rounding in (RoundDown,RoundUp)
        setprecision(BigFloat,bits) do
            setrounding(BigFloat,rounding) do
                @test helper(tiny,0.,0.,tiny,large,false)==large
                @test precision(BigFloat)==bits && Base.rounding(BigFloat)==rounding
            end
        end
    end
    tasks=map(1:4) do i
        Threads.@spawn setprecision(BigFloat,64+32i) do
            setrounding(BigFloat,isodd(i) ? RoundDown : RoundUp) do
                result=helper(tiny,0.,0.,tiny,large,false)
                (result,precision(BigFloat),Base.rounding(BigFloat))
            end
        end
    end
    for (i,task) in enumerate(tasks)
        @test fetch(task)==(large,64+32i,isodd(i) ? RoundDown : RoundUp)
    end
end
end
