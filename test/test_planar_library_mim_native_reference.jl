module LibraryMIMNativeReferenceTests
using DiffMoM,Test,SHA,TOML,LinearAlgebra
include(joinpath(@__DIR__,"..","validation","sonnet_stripline","library_mim_fixture.jl"))
@testset "Public RFIC library MIM vs actual native full matrices" begin
    directory=joinpath(@__DIR__,"fixtures","rfic_library_mim_native")
    hashes=TOML.parsefile(joinpath(directory,"sha256.toml"))["sha256"]
    @test length(hashes)==554
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(directory,path))))==digest
    end
    proof=joinpath(directory,"final_repeat")
    report=TOML.parsefile(joinpath(proof,"comparison.toml"))
    @test report["source_unchanged"]
    @test report["source_sha256_before"]==report["source_sha256_after"]
    @test length(report["runs"])==18
    @test report["full_complex_s_gate"]==.06
    @test report["original_voltage_residual_gate"]==1e-9
    @test all(row->row["status"]=="PASS",report["runs"])
    @test length(report["source_sha256_after"])==12
    for (path,digest) in report["source_sha256_after"]
        suffix=match(r"/((?:src|validation)/.+)$",replace(path,'\\'=>'/'))
        @test suffix!==nothing
        @test bytes2hex(sha256(read(joinpath(directory,"source_snapshot",suffix.captures[1]))))==digest
    end
    # Snapshot hashes describe the retained native run. The live solver
    # comparisons below remain valid after source line-ending normalization.
    initial=TOML.parsefile(joinpath(directory,"initial_eighteen","comparison.toml"))
    @test initial["source_unchanged"]
    @test length(initial["runs"])==18
    @test all(row->row["status"]=="PASS",initial["runs"])
    original=TOML.parsefile(joinpath(directory,"initial_mask_oracle_failure","comparison.toml"))
    @test original["source_unchanged"]
    @test length(original["runs"])==2
    @test all(row->row["status"]=="PASS",original["runs"])
    for resistance in (0.,.1),cells in (16,32,64)
        layout=mim_library_layout(cells;sheet_resistance=resistance)
        problem=planar_layout_problem(layout)
        @test [sheet.mask for sheet in problem.sheets]==mim_literal_masks(cells)
        connectivity=planar_connectivity(layout)
        @test connectivity.component_count==2
        @test isempty(connectivity.open_nets) && isempty(connectivity.shorted_nets)
        @test length(unique(only.(connectivity.port_components)))==2
        @test_throws ArgumentError solve_planar(layout,1e9;max_bytes=1)
        for frequency in (1e9,1e10,2e10)
            tag="native_$(resistance==0 ? "pec" : "resistive")_$(cells)_$(Int(frequency))"
            folder=joinpath(proof,tag)
            metadata=TOML.parsefile(joinpath(folder,"metadata.toml"))
            @test startswith(metadata["engine_version"],"18.53-Lite")
            @test metadata["process_success"]
            @test !metadata["deembedded"]
            @test metadata["response_nports"]==2
            @test metadata["source_sha256"]==bytes2hex(sha256(read(joinpath(folder,tag*".son"))))
            guard=metadata["touchstone_selected_log_checks"]["native_raw.s2p"]
            @test guard["status"]=="PASS"
            @test guard["intended_log_section"]=="raw"
            @test guard["source_options"]==String[]
            @test guard["max_printed_bound_ratio"]<=1
            file=joinpath(folder,"native_raw.s2p")
            @test guard["output_sha256"]==bytes2hex(sha256(read(file)))
            native=planar_read_touchstone(file)
            @test native.frequencies==[frequency]
            @test native.z0==fill(50.,2)
            @test native.s==planar_read_touchstone(joinpath(directory,"initial_eighteen",tag,"native_raw.s2p")).s
            result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
                max_bytes=2_000_000_000,retain_matrix=true)
            @test maximum(abs,result.s-only(native.s))<=.06
            @test mim_original_residual(problem,result)<=1e-9
            @test maximum(abs,result.s-transpose(result.s))<=1e-10
            @test eigmax(Hermitian(result.s'*result.s))<=1+1e-9
            row=only(filter(row->row["actual_grid"]==[cells,cells] &&
                row["frequency_hz"]==frequency && row["sheet_resistance_ohm_per_square"]==resistance,report["runs"]))
            @test planar_basis_count(problem.basis)==row["basis_count"]
            result=nothing;GC.gc()
        end
    end
end
end
