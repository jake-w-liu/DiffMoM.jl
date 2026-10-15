module SonnetReferenceSnapshotTests
using Test,DiffMoM,SHA,TOML
include(joinpath(@__DIR__,"..","validation","sonnet_stripline","sonnet_reference.jl"))

@testset "External runner metadata identifies captured inputs" begin
    python=Sys.which("python3")
    python===nothing && (python=Sys.which("python"))
    if python===nothing
        @test_skip false # External-runner boundary probe requires Python.
    else
        mktempdir() do dir
            source=joinpath(dir,"fake.son");dependency=joinpath(dir,"network.s1p")
            write(dependency,"initial dependency")
            # A stand-in process mutates only original TEMP inputs while the
            # runner executes its copied input. This tests provenance, not EM.
            program="from pathlib import Path\nPath("*repr(source)*").write_text('mutated original')\n"*
                "Path("*repr(dependency)*").write_text('mutated dependency')\n"*
                "p=Path('sondata/fake');p.mkdir(parents=True)\n"*
                "(p/'log_response.log').write_text('S-Parameters\\n1 0 0\\n')\n"
            write(source,program)
            reference=SonnetReference.reference_run(python,source;output_dir=joinpath(dir,"run"),
                dependencies=[dependency],deembedded=false,nports=1)
            metadata=TOML.parsefile(joinpath(reference.output_dir,"metadata.toml"))
            @test read(source,String)=="mutated original"
            @test read(joinpath(reference.output_dir,"fake.son"),String)==program
            @test metadata["source_sha256"]==bytes2hex(sha256(program))
            @test metadata["dependency_sha256"]["network.s1p"]==bytes2hex(sha256("initial dependency"))
        end
    end
end
end
