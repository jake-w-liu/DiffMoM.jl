module NativeSonnetParserLifetimeTests
using Test,DiffMoM

@testset "Native project parser closes files on tokenization failures" begin
    mktempdir() do directory
        enabled=GC.enable(false)
        try
            for text in ("FTYP SONPROJ\nDIM\nLNG \"unfinished\n", "\"unterminated\n", "FTYP SONPROJ\n\"bad\n")
                source=joinpath(directory,"malformed.son")
                write(source,text)
                @test_throws ArgumentError read_sonnet_project(source)
                # Windows rejects deletion with EBUSY while the parser's read
                # descriptor remains open. Disabling GC prevents its finalizer
                # from masking an unscoped eachline(path) implementation.
                rm(source)
                @test !ispath(source)
            end
            source=joinpath(directory,"valid.son")
            cp(joinpath(@__DIR__,"fixtures/native_sonnet_scalar_logarithms/ln_positive/ln_positive.son"),source)
            p=read_sonnet_project(source)
            rm(source)
            @test p isa SonnetProject
            @test !ispath(source)
        finally
            GC.enable(enabled)
            GC.gc()
        end
    end
end
end
