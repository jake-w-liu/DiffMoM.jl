module NativeSonnetRadialGeometryVariableTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra,Random
const fixture=joinpath(@__DIR__,"fixtures/native_radial_geovar")
const native=joinpath(fixture,"native")
const observations=NamedTuple[]

@testset "Actual native radial dimensions, header independence and full S" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    proof=TOML.parsefile(joinpath(native,"comparison.toml"))
    @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
    @test length(proof["cases"])==24
    before=TOML.parsefile(joinpath(fixture,"source_before/radial_before_lowering_20261004.toml"))
    @test before["implementation_before"]==before["implementation_after"]==
        bytes2hex(sha256(read(joinpath(fixture,"source_before/PlanarSonnetGeometryVariables.jl"))))
    @test all(r->r["status"]=="REJECT" && r["input_unchanged"] &&
        occursin("ANC/SYM",r["error"]),before["cases"])
    for row in proof["cases"]
        @test row["status"]=="ACCEPT" && row["radial_bit_identical"] && row["radial_full_s_error"]==0
        @test row["proportional_full_s_error"]>5e-4
        tag=row["tag"];prefix=row["target"]>.3125 ? "expand" : "contract"
        p=read_sonnet_project(joinpath(native,tag*".son"))
        q=read_sonnet_project(joinpath(native,prefix*"_radial_literal.son"))
        saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
        resolved=DiffMoM._sonnet_geometry_project(p,1e9)
        @test resolved.source==p.source && resolved.records===p.records
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(p.ports,savedports))
        @test only(resolved.polygons).vertices≈only(q.polygons).vertices rtol=8eps(Float64)
        @test only(resolved.polygons).vertices!==only(p.polygons).vertices
        for (port,literal) in zip(resolved.ports,q.ports)
            @test parse.(Float64,port.values[6:7])≈parse.(Float64,literal.values[6:7]) rtol=8eps(Float64)
        end
        @test only(sonnet_planar_problem(p;freq=1e9).sheets).mask==only(sonnet_planar_problem(q;freq=1e9).sheets).mask
        reference=planar_read_touchstone(joinpath(native,tag,"native_raw.s2p"))
        @test reference.s==planar_read_touchstone(joinpath(native,prefix*"_radial_literal/native_raw.s2p")).s
        result=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        full_s_error=maximum(abs,result.s-only(reference.s))
        @test full_s_error<=.005
        raw=result.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
        end
        residual=norm(raw.z_mom*raw.currents-rhs)/norm(rhs)
        @test residual<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10
        @test opnorm(result.s)<=1+1e-9
        @test DiffMoM._sonnet_geometry_project(p,1e9,Dict("Radius"=>.3125))===p
        other=read_sonnet_project(joinpath(native,(prefix=="expand" ? "contract" : "expand")*"_radial_literal.son"))
        target=prefix=="expand" ? .25 : .375
        @test only(DiffMoM._sonnet_geometry_project(p,1e9,Dict("Radius"=>target)).polygons).vertices≈only(other.polygons).vertices rtol=8eps(Float64)
        for options in ((max_bytes=1,),(max_points=2,),(max_parameters=0,))
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9;options...)
        end
        for value in (0.,-1.,NaN,Inf,BigFloat("1e1000"),BigFloat("1e-1000"))
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(p,1e9,Dict("Radius"=>value))
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        push!(observations,(case=tag,full_s_error,original_voltage_residual=residual))
    end
    for tag in ("offwall","at_anchor","cross_anchor")
        p=read_sonnet_project(joinpath(fixture,"edges",tag*".son"));saved=deepcopy(p.polygons)
        @test_throws ArgumentError sonnet_planar_problem(p;freq=1e9)
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
    end
    @test occursin("Port 2 is diagonal",read(joinpath(fixture,"edges/offwall/engine_stderr.log"),String))
    @test occursin("outside of the box",read(joinpath(fixture,"edges/at_anchor/engine_stderr.log"),String))
    # Native accepts this crossing-edge circuit. Its polygon repair remains
    # unsupported by the public simple-polygon model; keep that limit explicit.
    @test only(filter(r->r["tag"]=="cross_anchor",TOML.parsefile(joinpath(fixture,"edges/comparison.toml"))["cases"]))["status"]=="ACCEPT"
end

radial_oracle(x,y,ax,ay,delta)=setprecision(BigFloat,4608) do
    bx,by,bax,bay,bd=BigFloat.((x,y,ax,ay,delta))
    dx=bx-bax;dy=by-bay;rho=sqrt(dx^2+dy^2)
    (Float64(bx+bd*dx/rho),Float64(by+bd*dy/rho))
end

function radial_batch!(output,input)
    for i in eachindex(output,input)
        output[i]=DiffMoM._sonnet_geovar_radial_point(input[i]...)
    end
    return output
end

@testset "Radial range, cancellation, ownership and allocation" begin
    rng=MersenneTwister(10404);cases=[(.75,.625,1.,.4375,.0625),
        (1.,.6875,1.,.4375,-.0625),(1.,0.,0.,0.,-prevfloat(1.)),
        (1e308,1e308,-1e308,-1e308,-1e308),
        (nextfloat(0.),nextfloat(0.),0.,0.,nextfloat(0.)),
        (1e-300,1e308,0.,0.,1e308),(1.,1.,-1.,-1.,-sqrt(2.)),
        (1e16,1.,1.,0.,-1e16)]
    for _ in 1:256
        push!(cases,Tuple(ldexp(rand(rng)*2-1,rand(rng,-1070:1020)) for _ in 1:5))
    end
    bits=precision(BigFloat);mode=rounding(BigFloat)
    for case in cases
        actual=DiffMoM._sonnet_geovar_radial_point(case...);expected=radial_oracle(case...)
        for (a,b) in zip(actual,expected)
            @test isfinite(b) ? isfinite(a) && isapprox(a,b;rtol=16eps(Float64),atol=nextfloat(0.)) : isequal(a,b)
        end
    end
    @test precision(BigFloat)==bits && rounding(BigFloat)==mode
    @test_throws ArgumentError DiffMoM._sonnet_geovar_radial_point(1.,1.,1.,1.,.125)
    # Measure a function over runtime inputs with observable results. A loop
    # evaluated at module scope also counts interpreter/iteration allocations.
    input=[(.75+i/100000,.625,1.,.4375,.0625) for i in 1:64]
    output=Vector{Tuple{Float64,Float64}}(undef,length(input))
    radial_batch!(output,input)
    @test (@allocated radial_batch!(output,input))==0
    for (args,result) in zip(input,output)
        @test all(isapprox(a,b;rtol=16eps(Float64),atol=0) for (a,b) in zip(result,radial_oracle(args...)))
    end
end
end
