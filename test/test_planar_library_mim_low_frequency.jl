module LibraryMIMLowFrequencyTests
using DiffMoM,Test,SHA,TOML,LinearAlgebra
include(joinpath(@__DIR__,"../validation/sonnet_stripline/library_mim_fixture.jl"))

@testset "Public MIM low-frequency overlap capacitance" begin
    directory=joinpath(@__DIR__,"fixtures","library_mim_low_frequency")
    hashes=TOML.parsefile(joinpath(directory,"sha256.toml"))["sha256"]
    stored=Set{String}()
    for (folder,_,files) in walkdir(directory),file in files
        path=replace(relpath(joinpath(folder,file),directory),'\\'=>'/')
        path=="sha256.toml" || push!(stored,path)
    end
    @test Set(keys(hashes))==stored
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(directory,path))))==digest
    end
    report=TOML.parsefile(joinpath(directory,"final.toml"))
    @test report["source_unchanged"]
    @test report["source_before"]==report["source_after"]
    @test length(report["source_after"])==8
    for (name,digest) in report["source_after"]
        @test bytes2hex(sha256(read(joinpath(directory,"source_snapshot",name))))==digest
    end
    @test report["relative_capacitance_gate"]==.05
    @test report["original_voltage_residual_gate"]==1e-9
    @test length(report["cases"])==6
    @test all(row->row["status"]=="PASS",report["cases"])
    # Literal plate and lead overlap in SI. This reference is independent
    # of the production capacitance estimator and modal/FFT assembly.
    area=(.25*.25+.0625*.125)*1e-6
    reference=8.854187817e-12*7.5*area/1e-6
    @test report["literal_overlap_area_m2"]==area
    @test report["overlap_capacitance_reference_f"]==reference
    for cells in (16,32,64),frequency in (1e8,5e8)
        layout=mim_library_layout(cells)
        result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
            retain_matrix=true,max_bytes=2_000_000_000)
        capacitance=-imag(result.y[1,2])/(2pi*frequency)
        @test isfinite(capacitance) && capacitance>0
        @test abs(capacitance-reference)/reference<=.05
        @test mim_original_residual(layout.problem,result)<=1e-9
        @test maximum(abs,result.s-transpose(result.s))<=1e-10
        @test eigmax(Hermitian(result.s'*result.s))<=1+1e-9
        row=only(filter(r->r["cells"]==cells && r["frequency_hz"]==frequency,report["cases"]))
        @test row["relative_capacitance_error"]<=.05
        @test row["original_voltage_residual"]<=1e-9
        result=nothing;GC.gc()
    end
end
end
