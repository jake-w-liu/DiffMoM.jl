module SolverSpiralNativeReferenceTests
using DiffMoM,Test,SHA,TOML,LinearAlgebra
include(joinpath(@__DIR__,"../validation/sonnet_stripline/solver_spiral_fixture.jl"))

@testset "Public rectangular spiral vs actual native raw matrices" begin
    directory=joinpath(@__DIR__,"fixtures/solver_spiral_native")
    hashes=TOML.parsefile(joinpath(directory,"sha256.toml"))["sha256"]
    stored=Set{String}()
    for (folder,_,files) in walkdir(directory),file in files
        relative=replace(relpath(joinpath(folder,file),directory),'\\'=>'/')
        relative=="sha256.toml" || push!(stored,relative)
    end
    @test Set(keys(hashes))==stored
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(directory,path))))==digest
    end
    initial=TOML.parsefile(joinpath(directory,"initial_full/comparison.toml"))
    @test !initial["source_unchanged"]
    @test initial["source_sha256_before"]!=initial["source_sha256_after"]
    @test length(initial["runs"])==12
    @test all(r->r["status"]=="PASS",initial["runs"])
    # Initial native outputs remain valid immutable references. Their
    # concurrent importer drift is preserved and explicitly disqualifies
    # them as a stable-source native repeat certificate.
    proof=joinpath(directory,"initial_full")
    if isdir(joinpath(directory,"final_repeat"))
        proof=joinpath(directory,"final_repeat")
        repeated=TOML.parsefile(joinpath(proof,"comparison.toml"))
        @test repeated["source_unchanged"]
        @test repeated["source_sha256_before"]==repeated["source_sha256_after"]
        @test length(repeated["runs"])==12
        for (path,digest) in repeated["source_sha256_after"]
            relative=match(r"/((?:src|validation)/.+)$",replace(path,'\\'=>'/')).captures[1]
            @test bytes2hex(sha256(read(joinpath(directory,"final_repeat_source_snapshot",relative))))==digest
        end
    end
    report=TOML.parsefile(joinpath(proof,"comparison.toml"))
    @test report["full_complex_s_gate"]==.005
    @test report["original_voltage_residual_gate"]==1e-9
    for resistance in (0.,.1),cells in (32,64)
        layout=solver_spiral_layout(cells;sheet_resistance=resistance)
        @test only(layout.problem.sheets).mask==solver_spiral_masks(cells)
        net=planar_connectivity(layout)
        @test net.component_count==1 && net.port_components==[[1],[1]]
        @test isempty(net.open_nets) && isempty(net.shorted_nets)
        @test all(p->p.direction==:below && p.layers==1:1,layout.terminal_paths)
        @test_throws ArgumentError solve_planar(layout,1e9;max_bytes=1)
        for frequency in (1e9,5e9,1e10)
            tag="native_$(iszero(resistance) ? "pec" : "resistive")_$(cells)_$(Int(frequency))"
            folder=joinpath(proof,tag)
            metadata=TOML.parsefile(joinpath(folder,"metadata.toml"))
            @test startswith(metadata["engine_version"],"18.53-Lite")
            @test metadata["process_success"] && !metadata["deembedded"]
            @test metadata["response_nports"]==2
            guard=metadata["touchstone_selected_log_checks"]["native_raw.s2p"]
            @test guard["status"]=="PASS" && guard["intended_log_section"]=="raw"
            @test guard["source_options"]==String[] && guard["max_printed_bound_ratio"]<=1
            path=joinpath(folder,"native_raw.s2p")
            @test guard["output_sha256"]==bytes2hex(sha256(read(path)))
            native=planar_read_touchstone(path)
            @test native.frequencies==[frequency] && native.z0==fill(50.,2)
            original=planar_read_touchstone(joinpath(directory,"initial_full",tag,"native_raw.s2p"))
            @test only(native.s)==only(original.s)
            imported=sonnet_planar_problem(read_sonnet_project(joinpath(folder,tag*".son"));
                freq=frequency,_details=true,_materials=true)
            @test only(imported.problem.sheets).mask==solver_spiral_masks(cells)
            @test reduce(.|,(v.uni for v in imported.problem.vias))==
                reduce(.|,(v.uni for v in layout.source_problem.vias))
            result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
                max_bytes=2_000_000_000,retain_matrix=true)
            @test maximum(abs,result.s-only(native.s))<=.005
            @test solver_spiral_original_residual(layout,result)<=1e-9
            @test maximum(abs,result.s-transpose(result.s))<=1e-10
            @test opnorm(result.s)<=1+1e-9
            if resistance>0
                powers=solver_spiral_sheet_power(layout,result,resistance)
                @test powers.supplied>0 && powers.relative_error<=1e-8
            end
            row=only(filter(r->r["actual_grid"]==[cells,cells] && r["frequency_hz"]==frequency &&
                r["sheet_resistance_ohm_per_square"]==resistance,report["runs"]))
            @test planar_basis_count(layout.source_problem.basis)==row["basis_count"]
            result=nothing;GC.gc()
        end
    end
end
end
