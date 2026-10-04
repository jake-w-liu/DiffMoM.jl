module NativeSonnetRadialAdjustmentSequenceTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra,Random
const fixture=joinpath(@__DIR__,"fixtures/native_radial_adjustment_sequence")
module PreviousGeometry
using DiffMoM
using DiffMoM: _sonnet_error,_sonnet_section,_sonnet_parse_scalar,
    _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,_checked_payload_sum,_sonnet_scalar_project_payload,
    _checked_array_payload_bytes,_enforce_payload_limit,_circuit_stored_real,_P2
include(joinpath(@__DIR__,"fixtures/native_radial_adjustment_sequence/source_before/PlanarSonnetGeometryVariables.jl"))
end
const observations=NamedTuple[]
function subsection_support(path)
    rows=readlines(path);start=findfirst(row->startswith(row,"SUB "),rows)
    count=parse(Int,split(rows[start])[3])
    Set(Tuple(parse.(Int,split(row)[2:end])) for row in rows[start+1:start+count])
end
@testset "Native RAD reference-radius recomputation and anchor crossings" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),fixture),'\\'=>'/')
        for (d,_,fs) in walkdir(fixture) for f in fs if f!="sha256.toml")
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    cases=TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
    @test length(cases)==36
    previous_mismatches=0
    for row in cases
        folder=joinpath(fixture,"cases",row["name"])
        p=read_sonnet_project(joinpath(folder,"parameter.son"));q=read_sonnet_project(joinpath(folder,"literal.son"))
        saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
        for suffix in ("parameter","literal")
            metadata=TOML.parsefile(joinpath(folder,suffix,"metadata.toml"))
            @test metadata["process_success"] && !metadata["deembedded"]
            @test metadata["source_sha256"]==bytes2hex(sha256(read(joinpath(folder,suffix*".son"))))
            @test metadata["touchstone_selected_log_checks"]["native_raw.s2p"]["status"]=="PASS"
        end
        @test subsection_support(joinpath(folder,"parameter/jxy/current_1.sid"))==
            subsection_support(joinpath(folder,"literal/jxy/current_1.sid"))
        native=planar_read_touchstone(joinpath(folder,"parameter/native_raw.s2p"))
        @test native.s==planar_read_touchstone(joinpath(folder,"literal/native_raw.s2p")).s
        old_failed=try
            old=PreviousGeometry._sonnet_geometry_project(p,1e9)
            maximum(maximum(abs,a.vertices-b.vertices) for (a,b) in zip(old.polygons,q.polygons))>1e-8
        catch err
            err isa ArgumentError || rethrow()
            true
        end
        previous_mismatches+=Int(old_failed)
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        @test all(isapprox(a.vertices,b.vertices;rtol=64eps(Float64),atol=2e-18) for (a,b) in zip(resolved.polygons,q.polygons))
        @test all(a.vertices!==b.vertices && a.vertices[:,end]==a.vertices[:,1] for (a,b) in zip(resolved.polygons,p.polygons))
        @test all(a.mask==b.mask for (a,b) in zip(sonnet_planar_problem(p;freq=1e9).sheets,
            sonnet_planar_problem(q;freq=1e9).sheets))
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
        end
        error=maximum(abs,result.s-only(native.s));residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test error<=.005 && residual<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10 && opnorm(result.s)<=1+1e-9
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved)) && all(a.values==b.values for (a,b) in zip(p.ports,savedports))
        for options in ((max_points=1,),(max_bytes=1,),(max_parameters=0,))
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;options...)
        end
        push!(observations,(case=row["name"],previous_mismatch=old_failed,full_s_error=error,original_voltage_residual=residual))
    end
    @test previous_mismatches>=20
    # The native engine reports a memory allocation error for this case.
    # Its full S remains unverified. The exact radial sequence reaches the
    # anchor after pass two, so the point's next ray is independently undefined.
    folder=joinpath(fixture,"unverified/axial__r5_target0.078125")
    @test occursin("Memory allocation error",read(joinpath(folder,"engine_stderr.log"),String))
    p=read_sonnet_project(joinpath(folder,"parameter.son"));saved=deepcopy(p.polygons)
    @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9)
    @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
end
function delta_oracle(target,x,y,ax,ay)
    setprecision(BigFloat,4608) do
        t,bx,by,bax,bay=BigFloat.((target,x,y,ax,ay))
        Float64(t-sqrt((bx-bax)^2+(by-bay)^2))
    end
end
@testset "Native moved box-port edges retain their finite extent" begin
    folder=joinpath(@__DIR__,"fixtures/native_geovar_box_port_extent")
    for (path,digest) in TOML.parsefile(joinpath(folder,"sha256.toml"))["sha256"]
        @test bytes2hex(sha256(read(joinpath(folder,path))))==digest
    end
    @test occursin("partially or entirely outside of the box",
        read(joinpath(folder,"native/engine_stderr.log"),String))
    p=read_sonnet_project(joinpath(folder,"parameter.son"));saved=deepcopy(p.polygons)
    error=try
        DiffMoM._sonnet_geometry_project(p,1e9)
        nothing
    catch err
        err
    end
    @test error isa ArgumentError && occursin("box-port edge",sprint(showerror,error))
    @test_throws ArgumentError solve_sonnet_project(p,1e9;raw=true)
    @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
end
function delta_batch!(output,input)
    for i in eachindex(input);output[i]=DiffMoM._sonnet_geovar_radial_delta(input[i]...);end
    output
end
@testset "Radial corrective distance range and allocation" begin
    rng=MersenneTwister(10506)
    cases=[(1e308,1e308,1e308,-1e308,-1e308),(nextfloat(0.),nextfloat(0.),0.,0.,0.),
        (1.,1.,1.,0.,0.),(.25,.775,.825,.625,.625)]
    for _ in 1:128
        push!(cases,Tuple(ldexp(rand(rng)*2-1,rand(rng,-1070:1020)) for _ in 1:5))
    end
    bits=precision(BigFloat);mode=rounding(BigFloat)
    for args in cases
        actual=DiffMoM._sonnet_geovar_radial_delta(args...);expected=delta_oracle(args...)
        if isfinite(expected)
            # Native recomputation rounds hypot before subtracting. The error
            # bound includes that rounding even for near-zero corrections.
            magnitude=maximum(abs,args)
            @test isfinite(actual) && abs(actual-expected)<=32eps(Float64)*magnitude+nextfloat(0.)
        else
            @test isequal(actual,expected)
        end
    end
    @test precision(BigFloat)==bits && rounding(BigFloat)==mode
    input=[(.25+i/10000,.775,.825,.625,.625) for i in 1:64];output=zeros(64)
    delta_batch!(output,input)
    @test (@allocated delta_batch!(output,input))==0
    @test output==[args[1]-hypot(args[2]-args[4],args[3]-args[5]) for args in input]
end
end
