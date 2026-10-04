module NativeSonnetReferenceCountRoundoffTests
using Test,DiffMoM,SHA,TOML,Random
const fixture=joinpath(@__DIR__,"fixtures/native_geovar_count_roundoff")
module PreviousGeometry
using DiffMoM
using DiffMoM: _sonnet_error,_sonnet_section,_sonnet_parse_scalar,
    _DEFAULT_MAX_DENSE_PAYLOAD_BYTES,_checked_payload_sum,_sonnet_scalar_project_payload,
    _checked_array_payload_bytes,_enforce_payload_limit,_circuit_stored_real,_P2
include(joinpath(@__DIR__,"fixtures/native_geovar_count_roundoff/PlanarSonnetGeometryVariables.jl"))
end
@testset "Repeated translation preserves native centre-cell ownership" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    @test Set(keys(hashes))==Set(f for f in readdir(fixture) if f!="sha256.toml")
    for (file,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,file))))==digest
    end
    proof=TOML.parsefile(joinpath(fixture,"comparison.toml"))
    @test proof["source_before"]==proof["source_after"] && length(proof["cases"])==4
    for row in proof["cases"]
        folder=joinpath(@__DIR__,"fixtures/native_geovar_reference_count_law/cases",row["case"])
        p=read_sonnet_project(joinpath(folder,"parameter.son"));q=read_sonnet_project(joinpath(folder,"literal.son"))
        # The effective project retains original records. Bind its dimension
        # to NOM when lowering the old result so it is not moved a second time.
        before=sonnet_planar_problem(PreviousGeometry._sonnet_geometry_project(p,1e9);
            freq=1e9,variables=Dict("Move"=>.125))
        literal=sonnet_planar_problem(q;freq=1e9);current=sonnet_planar_problem(p;freq=1e9)
        @test sum(count(a.mask .!= b.mask) for (a,b) in zip(before.sheets,literal.sheets))==row["cell_mismatch"]==16
        @test all(a.mask==b.mask for (a,b) in zip(current.sheets,literal.sheets))
    end
end
function coordinate_oracle(value,target,nominal,scale,r,movements,direction)
    setprecision(BigFloat,4608) do
        setrounding(BigFloat,RoundNearest) do
            count=BigFloat(r)
            Float64(BigFloat(value)+direction*(BigFloat(target)-BigFloat(nominal))*
                (count*(count-1)+1)*BigFloat(movements)*BigFloat(scale))
        end
    end
end
function coordinate_batch!(output,input)
    for i in eachindex(input);output[i]=DiffMoM._sonnet_geovar_repeated_coordinate(input[i]...);end
    output
end
@testset "Combined translation precision and allocation" begin
    rng=MersenneTwister(10506)
    inputs=[(.4375*.001,.1328125,.125,.001,2,2,1),
        (.5625*.001,.1328125,.125,.001,2,2,1),
        (1.,1e308,1e307,1e-308,99997,99998,-1),
        (nextfloat(0.),nextfloat(0.),0.,1.,2,3,1),
        (floatmax(Float64),1.,.5,1e-308,typemax(Int),typemax(Int),1)]
    for _ in 1:256
        push!(inputs,(ldexp(rand(rng)+.5,rand(rng,-1070:1020)),
            ldexp(rand(rng)+.5,rand(rng,-1070:1020)),ldexp(rand(rng)+.5,rand(rng,-1070:1020)),
            ldexp(rand(rng)+.5,rand(rng,-1070:1020)),rand(rng,(2,3,32,99997,typemax(Int))),
            rand(rng,(1,2,33,99998,typemax(Int))),rand(rng,(-1,1))))
    end
    bits=precision(BigFloat);mode=rounding(BigFloat)
    for input in inputs
        actual=DiffMoM._sonnet_geovar_repeated_coordinate(input...);expected=coordinate_oracle(input...)
        @test isfinite(expected) ? isapprox(actual,expected;rtol=16eps(Float64),atol=nextfloat(0.)) : isequal(actual,expected)
    end
    @test precision(BigFloat)==bits && rounding(BigFloat)==mode
    ordinary=[((.5+i/10000)*.001,.13,.125,.001,2,2+i,1) for i in 1:64];output=zeros(64)
    coordinate_batch!(output,ordinary)
    @test (@allocated coordinate_batch!(output,ordinary))==0
    @test output≈[coordinate_oracle(input...) for input in ordinary] rtol=16eps(Float64)
end
end
