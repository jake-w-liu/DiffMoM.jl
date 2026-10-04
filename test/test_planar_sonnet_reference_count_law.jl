module NativeSonnetReferenceCountLawTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra,Random
const fixture=joinpath(@__DIR__,"fixtures/native_geovar_reference_count_law")
module PreviousGeometry
using DiffMoM
using DiffMoM: _sonnet_error,_sonnet_section,_sonnet_parse_scalar,
    _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,_checked_payload_sum,_sonnet_scalar_project_payload,
    _checked_array_payload_bytes,_enforce_payload_limit,_circuit_stored_real,_P2
include(joinpath(@__DIR__,"fixtures/native_geovar_reference_count_law/source_before/PlanarSonnetGeometryVariables.jl"))
end
const observations=NamedTuple[]

@testset "Native anchored direction and repeated-reference geometry" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),fixture),'\\'=>'/')
        for (d,_,fs) in walkdir(fixture) for f in fs if f!="sha256.toml")
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    for name in ("literal_physical","direction_physical")
        proof=TOML.parsefile(joinpath(fixture,"source_before",name*".toml"))
        @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
        @test all(row->row["status"]=="PASS" && row["full_s_error"]<=.005 &&
            row["original_voltage_residual"]<=1e-9,proof["cases"])
        name=="direction_physical" && @test all(row->row["before_parameter_full_s_error"]>.005,proof["cases"])
    end
    # Failed alternatives remain failures: count4 rejects the linear,
    # triangular and exponential rules; the original direction rule failed.
    proof=TOML.parsefile(joinpath(fixture,"provenance/count34/comparison.toml"))
    @test proof["source_unchanged"] && length(proof["cases"])==8
    for row in proof["cases"]
        factor=1+row["explicit_reference_count"]*(row["explicit_reference_count"]-1)
        @test factor in row["matching_factors"]
        row["explicit_reference_count"]==4 && @test row["matching_factors"]==[13]
    end
    rejected=joinpath(fixture,"rejected_hypotheses/nscd_4_xnegative")
    @test maximum(abs,only(planar_read_touchstone(joinpath(rejected,"parameter/native_raw.s2p")).s)-
        only(planar_read_touchstone(joinpath(rejected,"literal/native_raw.s2p")).s))>.01
    cases=TOML.parsefile(joinpath(fixture,"index.toml"))["cases"]
    @test length(cases)==94
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
        native=planar_read_touchstone(joinpath(folder,"parameter/native_raw.s2p"))
        @test native.s==planar_read_touchstone(joinpath(folder,"literal/native_raw.s2p")).s
        if row["explicit_reference_count"]>1
            @test_throws ArgumentError PreviousGeometry._sonnet_geometry_project(p,1e9)
        else
            old=PreviousGeometry._sonnet_geometry_project(p,1e9)
            @test maximum(maximum(abs,a.vertices-b.vertices) for (a,b) in zip(old.polygons,q.polygons))>1e-5
        end
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        # Each native literal accumulates in project units, whereas the adapter
        # accumulates in SI. Bound both sums' roundoff by their entry count.
        tolerance=max(8,4*(row["explicit_reference_count"]+2))*eps(Float64)
        @test all(isapprox(a.vertices,b.vertices;rtol=tolerance,atol=0) for (a,b) in zip(resolved.polygons,q.polygons))
        @test all(a.vertices!==b.vertices && a.vertices[:,end]==a.vertices[:,1] for (a,b) in zip(resolved.polygons,p.polygons))
        @test all(isapprox(parse.(Float64,a.values[6:7]),parse.(Float64,b.values[6:7]);rtol=tolerance,atol=0)
            for (a,b) in zip(resolved.ports,q.ports))
        lowered=sonnet_planar_problem(p;freq=1e9);literal=sonnet_planar_problem(q;freq=1e9)
        @test all(a.mask==b.mask for (a,b) in zip(lowered.sheets,literal.sheets))
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
        push!(observations,(case=row["name"],full_s_error=error,original_voltage_residual=residual))
    end
end

function delta_oracle(target,nominal,scale,r)
    setprecision(BigFloat,4608) do
        setrounding(BigFloat,RoundNearest) do
            count=BigFloat(r)
            Float64((BigFloat(target)-BigFloat(nominal))*(count*(count-1)+1)*BigFloat(scale))
        end
    end
end
function delta_batch!(output,input)
    for i in eachindex(input);output[i]=DiffMoM._sonnet_geovar_repeated_delta(input[i]...);end
    output
end
@testset "Reference-count precision, allocation and transactional point budget" begin
    rng=MersenneTwister(10505)
    inputs=[(nextfloat(.25),.25,.001,99997),
        (nextfloat(0.),0.,.001,99997),(floatmax(Float64),prevfloat(floatmax(Float64)),.001,typemax(Int)),
        (1e308,1e307,1e-308,99997),(1e-308,0.,1e308,99997)]
    for _ in 1:256
        push!(inputs,(ldexp(rand(rng)+.5,rand(rng,-1070:1020)),
            ldexp(rand(rng)+.5,rand(rng,-1070:1020)),ldexp(rand(rng)+.5,rand(rng,-1070:1020)),
            rand(rng,(2,3,4,32,99997,94906266,94906267,typemax(Int)))))
    end
    bits=precision(BigFloat);mode=rounding(BigFloat)
    for input in inputs
        actual=DiffMoM._sonnet_geovar_repeated_delta(input...);expected=delta_oracle(input...)
        @test isfinite(expected) ? isapprox(actual,expected;rtol=16eps(Float64),atol=nextfloat(0.)) : isequal(actual,expected)
    end
    @test precision(BigFloat)==bits && rounding(BigFloat)==mode
    ordinary=[(.28+i/10000,.25,.001,2+i) for i in 1:64];output=zeros(64)
    delta_batch!(output,ordinary)
    @test (@allocated delta_batch!(output,ordinary))==0
    @test output≈[delta_oracle(input...) for input in ordinary] rtol=16eps(Float64)
    # The largest default aggregate point budget includes two reference
    # identities and one ordinary point. The factor must not overflow Int32
    # or discard an adjacent representable native quantity after SI conversion.
    source=read(joinpath(fixture,"cases/factor3__nscd_expand_2/parameter.son"),String)
    source=replace(source,"\r\n"=>"\n");r=99997;entries=vcat(2,fill(3,r))
    start=findfirst("PS2 1",source);stop=findnext("END\nEND\n",source,last(start))
    text=source[1:first(start)-1]*"PS2 1\nPOLY 1 $(length(entries))\n"*join(entries,'\n')*"\n"*source[first(stop):end]
    text=replace(text,"LNG 0.28125"=>"LNG $(nextfloat(.25))")
    mktempdir() do directory
        path=joinpath(directory,"large_count.son");write(path,text);p=read_sonnet_project(path)
        saved=copy(only(p.polygons).vertices)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_points=99999)
        resolved=DiffMoM._sonnet_geometry_project(p,1e9;max_points=100000)
        delta=delta_oracle(nextfloat(.25),.25,.001,r)
        @test only(resolved.polygons).vertices[2,3]≈saved[2,3]+delta rtol=8eps(Float64)
        @test only(resolved.polygons).vertices[2,4]≈saved[2,4]+(r+1)*delta rtol=4*(r+1)*eps(Float64)
        @test only(p.polygons).vertices==saved
        @test DiffMoM._sonnet_geometry_project(p,1e9,Dict("Width"=>.25))===p
        reserve=DiffMoM._checked_payload_sum("native geometry variable workspace",DiffMoM._sonnet_scalar_project_payload(p),
            DiffMoM._checked_array_payload_bytes(UInt8,256,length(p.records)),DiffMoM._checked_array_payload_bytes(UInt8,1024,1))
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;max_bytes=reserve-1)
    end
end
end
