module NativeSonnetScaledMixedReferenceTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_geovar_scaled_mixed_reference")

@testset "Native scaled anchored mixed moving reference" begin
    # Retained native pairs for scaled ANC moving-reference dimensions: the
    # literals are raster-cell-equivalent hypotheses falsified at other target
    # widths, so the adapter rejects every scaled moving-reference form.
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    files=Set(replace(relpath(joinpath(d,f),fixture),'\\'=>'/')
        for (d,_,fs) in walkdir(fixture) for f in fs
            if f ∉ ("sha256.toml","README.md","comparison.toml"))
    @test Set(keys(hashes))==files
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    proof=TOML.parsefile(joinpath(fixture,"comparison.toml"))
    @test proof["source_unchanged"]
    @test length(proof["cases"])==4
    implemented=Set{String}()
    for row in proof["cases"]
        row["status"]=="PASS" && push!(implemented,row["case"])
        parameter=read_sonnet_project(joinpath(fixture,"native",row["case"]*"_parameter.son"))
        native=planar_read_touchstone(joinpath(fixture,"native",row["case"]*"_parameter/native_raw.s2p"))
        literal=if row["case"]!="count2"
            q=read_sonnet_project(joinpath(fixture,"native",row["case"]*"_literal.son"))
            reference=planar_read_touchstone(joinpath(fixture,"native",row["case"]*"_literal/native_raw.s2p"))
            @test native.s==reference.s
            q
        end
        saved=deepcopy(parameter.polygons);savedports=deepcopy(parameter.ports)
        # Every scaled moving-reference form stays rejected. The forward and
        # reversed literals are raster-cell-equivalent hypotheses that native
        # falsified at other target widths, so the parser admits none of them.
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(parameter,1e9)
        @test_throws ArgumentError solve_sonnet_project(parameter,1e9;raw=true)
        @test all(a.vertices==b.vertices for (a,b) in zip(parameter.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(parameter.ports,savedports))
    end
    @test implemented==Set{String}()
    # The same project text with a scaled SYM header keeps the general
    # explicit-reference rejection as well.
    source=read(joinpath(fixture,"native/forward_parameter.son"),String)
    text=replace(source," ANC "=>" SYM ")
    mktempdir() do directory
        path=joinpath(directory,"sym_mixed.son");write(path,text)
        candidate=read_sonnet_project(path)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(candidate,1e9)
    end
end
end
