module NativeSonnetPolygonIntakeTests
using Test,DiffMoM
const DM=DiffMoM
function rejected(source,records)
    try
        DM._sonnet_read_records(source,records)
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
    error("malformed polygon unexpectedly accepted")
end
@testset "Native polygon counts and terminators preflight owned vertices" begin
    fixture=joinpath(@__DIR__,"fixtures/native_sonnet_scalar_logarithms/ln_positive/ln_positive.son")
    p=read_sonnet_project(fixture)
    records=deepcopy(p.records)
    ni=findfirst(r->r.tokens[1]=="NUM",records)
    for count in (3,6,100000,typemax(Int))
        records[ni+1].tokens[2]=string(count)
        @test_throws ArgumentError DM._sonnet_read_records(p.source,records)
        rejected(p.source,records)
        @test (@allocated rejected(p.source,records))<65536
    end
    mktempdir() do directory
        raw=read(fixture,String)
        source=joinpath(directory,"missing_end.son")
        write(source,replace(raw,r"END\r?\nEND GEO"=>"END GEO"))
        @test_throws ArgumentError read_sonnet_project(source)
        rm(source)
        @test !ispath(source)
    end
    malformed=deepcopy(p.records)
    resize!(malformed,ni+1)
    malformed[end]=SonnetRecord(malformed[end].line,["VIA","POLYGON"])
    push!(malformed,SonnetRecord(malformed[end].line+1,["END","GEO"]))
    @test_throws ArgumentError DM._sonnet_read_records(p.source,malformed)
    ordinary=DM._sonnet_read_records(p.source,p.records)
    @test only(ordinary.polygons).vertices==only(p.polygons).vertices
    fields(ports)=[(ps.kind,ps.polygon,ps.edge,ps.number,ps.values) for ps in ports]
    @test fields(ordinary.ports)==fields(p.ports)
    # The native polygon endpoint and its order remain exactly represented.
    vertices=only(ordinary.polygons).vertices
    @test vertices[:,1]==vertices[:,end]
end
end
