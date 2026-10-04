module NetworkFrequencyDomainTests
using DiffMoM,Test,LinearAlgebra,SHA,TOML
@testset "Frequency proof retains original failure and discarded candidate" begin
    directory=joinpath(@__DIR__,"fixtures","network_frequency_domain")
    hashes=TOML.parsefile(joinpath(directory,"sha256.toml"))["sha256"]
    @test length(hashes)==12
    for (file,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(directory,file))))==digest
    end
    before=TOML.parsefile(joinpath(directory,"frequency_before.toml"))
    after=TOML.parsefile(joinpath(directory,"frequency_after.toml"))
    @test before["source_sha256"]==hashes["source_before.jl"]
    @test after["source_sha256"]==hashes["source_after.jl"]
    @test before["rows"][1]["status"]=="accepted"
    @test before["rows"][1]["stored_frequency_hz"]==0
    @test before["rows"][1]["provider_calls_hz"]==[0.]
    @test after["rows"][1]["status"]=="rejected"
    @test isempty(after["rows"][1]["provider_calls_hz"])
    for index in (2,3)
        @test after["rows"][index]["status"]=="accepted"
        @test after["rows"][index]["stored_frequency_hz"]==before["rows"][index]["stored_frequency_hz"]
    end
    discarded=TOML.parsefile(joinpath(directory,"discarded_text_candidate.toml"))
    @test discarded["before_source_sha256"]==hashes["source_before.jl"]
    @test discarded["after_source_sha256"]==hashes["source_after.jl"]
    for row in discarded["rows"][1:3]
        @test row["before"]["status"]=="rejected"
        @test row["after"]["status"]=="rejected"
    end
    @test last(discarded["rows"])["before"]["status"]=="accepted"
    @test last(discarded["rows"])["after"]["status"]=="accepted"
end
@testset "Databank stored-frequency preflight precedes providers" begin
    matrix=reshape(ComplexF64[.25],1,1);calls=Float64[]
    provider=f->(push!(calls,f);50.)
    for frequency in (big"1e-500",big"1e500",BigFloat(Inf),BigFloat(NaN))
        @test_throws ArgumentError PlanarNetworkData([frequency],[matrix];z0=provider)
        @test isempty(calls)
    end
    collapsed=BigFloat[BigFloat(1.),BigFloat(1.)+eps(BigFloat)]
    @test_throws ArgumentError PlanarNetworkData(collapsed,[matrix,matrix];z0=provider)
    @test isempty(calls)
    for frequency in (BigFloat(0),BigFloat(1e9),BigFloat(nextfloat(0.)))
        empty!(calls)
        data=PlanarNetworkData([frequency],[matrix];z0=provider)
        @test data.frequencies==[Float64(frequency)]
        @test calls==[Float64(frequency)]
        @test data.s==[matrix]
    end
    empty!(calls)
    data=PlanarNetworkData([0.,1e9],[matrix,matrix])
    for frequency in (big"1e-500",big"1e500")
        @test_throws ArgumentError planar_network_response(data,frequency;z0=provider)
        @test isempty(calls)
    end
    @test planar_network_response(data,BigFloat(0);z0=provider)==matrix
    @test calls==[0.]
    empty!(calls)
    @test planar_network_response(data,BigFloat(1e9);z0=provider)==matrix
    @test calls==[1e9]
end
@testset "Touchstone nonzero numeric text must remain nonzero" begin
    mktempdir() do directory
        path=joinpath(directory,"domain.s1p")
        for record in ("1e-500 .25 0","1 1e-500 0","1 0 -1D-500")
            write(path,"# HZ S RI R 50\n"*record*"\n")
            @test_throws ArgumentError planar_read_touchstone(path)
        end
        write(path,"# HZ S MA R 50\n1 -1e-500 0\n")
        @test_throws ArgumentError planar_read_touchstone(path)
        for field in ("0e-500","-0D999",".000e500","0.0","-0.0")
            write(path,"# HZ S RI R 50\n"*field*" .25 0\n")
            data=planar_read_touchstone(path)
            @test only(data.frequencies)==0
            @test only(data.s)==reshape(ComplexF64[.25],1,1)
        end
        for field in ("5e-324","1D-300","1e9")
            write(path,"# HZ S RI R 50\n"*field*" .25 0\n")
            @test only(planar_read_touchstone(path).frequencies)>0
        end
    end
end
end
