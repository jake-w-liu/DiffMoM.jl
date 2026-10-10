module NativeSonnetSymMovingReferenceTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_geovar_sym_moving_reference")

@testset "Native unscaled symmetric moving reference" begin
    # Retained native pairs for unscaled SYM moving-reference dimensions. The
    # repeated-reference movement 3*r*delta/4 and the deduplicated ceil(c*r/2)
    # ordinary law are confirmed bit-identically at the retained widths, but
    # the PS1-side trigger is non-monotonic in delta and remains undecoded —
    # so the adapter rejects every explicit SYM moving-reference form.
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
    @test length(proof["cases"])==17
    implemented=Set{String}()
    for row in proof["cases"]
        row["status"]=="PASS" && push!(implemented,row["case"])
        parameter=read_sonnet_project(joinpath(fixture,"native",row["case"]*".son"))
        native=planar_read_touchstone(joinpath(fixture,"native",row["case"],"native_raw.s2p"))
        for name in split(row["matched"]," and ")
            reference=planar_read_touchstone(joinpath(fixture,"native",name,"native_raw.s2p"))
            if row["bit_identical_s"]
                @test native.s==reference.s
            else
                @test native.s!=reference.s
            end
        end
        saved=deepcopy(parameter.polygons);savedports=deepcopy(parameter.ports)
        # Every unscaled SYM explicit moving-reference form stays rejected:
        # the confirmed law fragments do not cover the PS1-side trigger, so
        # admitting them would silently resolve wrong coordinates in the
        # non-monotone band.
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(parameter,1e9)
        @test_throws ArgumentError solve_sonnet_project(parameter,1e9;raw=true)
        @test all(a.vertices==b.vertices for (a,b) in zip(parameter.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(parameter.ports,savedports))
    end
    @test implemented==Set{String}()
    # The same project text under a scaled header keeps the rejection as well.
    source=read(joinpath(fixture,"native/mixed.son"),String)
    text=replace(source," NSCD"=>" SCUNI")
    mktempdir() do directory
        path=joinpath(directory,"sym_scaled_mixed.son");write(path,text)
        candidate=read_sonnet_project(path)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(candidate,1e9)
    end
end
end
