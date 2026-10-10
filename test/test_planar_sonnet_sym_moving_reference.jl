module NativeSonnetSymMovingReferenceTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/native_geovar_sym_moving_reference")

# Parse the retained literal geometry: NUM vertex block plus the port
# reference-plane coordinates (fields 6-7 hold the edge midpoint).
function _literal_geometry(path)
    text=read(path,String)
    m=match(r"NUM 1\r?\n0 \d+ -1 N 1 1 1 100 100 0 0 0 Y\r?\n((?:[-\d.]+ [-\d.]+\r?\n)+)",text)
    m===nothing && error("fixture literal lacks a NUM block: "*path)
    vertices=hcat([parse.(Float64,split(line)) for line in
        split(strip(m.captures[1]),r"\r?\n")]...).*1e-3
    ports=[split(m.match)[6:7] for m in eachmatch(r"(?m)^\d+ 50 0 0 0 \S+ \S+",text)]
    vertices,ports
end

@testset "Native unscaled symmetric moving reference" begin
    # Retained native pairs for unscaled SYM moving-reference dimensions. The
    # full movement law is decoded and bit-verified: REF2 repeats by
    # 3*r*|delta|/4 floored to min(cell,box/32) in mixed lists, ordinary PS2
    # points move min(ceil(c*r/2), c*r*ramp)*delta/2 with a 3*c*r*delta/8
    # coarse-raster cap, the anchor side is gated by 3*r*|delta|/4>cell, and
    # ungated PS1 points drop one box/64 raster step. Each admitted case ships
    # a resolved_<case> literal whose native S is bit-identical to the case's.
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
    rejected=Set{String}()
    implemented=Set{String}()
    for row in proof["cases"]
        case=row["case"]
        parameter=read_sonnet_project(joinpath(fixture,"native",case*".son"))
        native=planar_read_touchstone(joinpath(fixture,"native",case,"native_raw.s2p"))
        for name in split(row["matched"]," and ")
            reference=planar_read_touchstone(joinpath(fixture,"native",name,"native_raw.s2p"))
            if row["bit_identical_s"]
                @test native.s==reference.s
            else
                @test native.s!=reference.s
            end
        end
        saved=deepcopy(parameter.polygons);savedports=deepcopy(parameter.ports)
        if row["status"]=="IMPLEMENTED"
            push!(implemented,case)
            resolved=DiffMoM._sonnet_geometry_project(parameter,1e9)
            expected_vertices,expected_ports=_literal_geometry(
                joinpath(fixture,"native","resolved_"*case*".son"))
            # The resolved literal geometry must equal the retained file that
            # native Sonnet solved bit-identically.
            @test resolved.polygons[1].vertices≈expected_vertices
            @test [port.values[6:7] for port in resolved.ports]==expected_ports
            resolution=planar_read_touchstone(joinpath(fixture,"native",
                "resolved_"*case,"native_raw.s2p"))
            @test native.s==resolution.s
        else
            # r>=3 repetitions stay rejected: the native basin flattens below
            # the raster resolution and the law is unconfirmed there.
            push!(rejected,case)
            @test_throws ArgumentError DiffMoM._sonnet_geometry_project(parameter,1e9)
            @test_throws ArgumentError solve_sonnet_project(parameter,1e9;raw=true)
        end
        @test all(a.vertices==b.vertices for (a,b) in zip(parameter.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(parameter.ports,savedports))
    end
    @test implemented==Set(row["case"] for row in proof["cases"] if row["status"]=="IMPLEMENTED")
    @test rejected==Set(["o1r3","ref3_only"])
    # The same project text under a scaled header keeps the rejection as well.
    source=read(joinpath(fixture,"native/mixed.son"),String)
    text=replace(source," NSCD"=>" SCUNI")
    mktempdir() do directory
        path=joinpath(directory,"sym_scaled_mixed.son");write(path,text)
        candidate=read_sonnet_project(path)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(candidate,1e9)
    end
    # XDIR moving references and whole-polygon selectors stay rejected.
    xdir=replace(source,"GEOVAR Width SYM YDIR 1 NSCD"=>"GEOVAR Width SYM XDIR 1 NSCD")
    mktempdir() do directory
        path=joinpath(directory,"sym_xdir.son");write(path,xdir)
        candidate=read_sonnet_project(path)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(candidate,1e9)
    end
    # POLY id 0 expands to every vertex; the proved law only covers literal
    # point lists, so whole-polygon selectors stay rejected.
    wild=replace(source,r"POLY 1 2\r?\n2\r?\n3"=>"POLY 1 0")
    mktempdir() do directory
        path=joinpath(directory,"sym_wildcard.son");write(path,wild)
        candidate=read_sonnet_project(path)
        @test_throws ArgumentError DiffMoM._sonnet_geometry_project(candidate,1e9)
    end
end
end
