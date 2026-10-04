module LibraryIDCNativeReferenceTests
using DiffMoM,Test,SHA,TOML,LinearAlgebra
include(joinpath(@__DIR__,"..","validation","sonnet_stripline","library_idc_fixture.jl"))
@testset "Public RFIC library IDC vs actual native full matrices" begin
    directory=joinpath(@__DIR__,"fixtures","rfic_library_idc_native")
    hashes=TOML.parsefile(joinpath(directory,"sha256.toml"))["sha256"]
    @test length(hashes)==941
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(directory,path))))==digest
    end
    proof=joinpath(directory,"pec_resistive")
    report=TOML.parsefile(joinpath(proof,"comparison.toml"))
    @test report["source_unchanged"]
    @test report["source_sha256_before"]==report["source_sha256_after"]
    @test length(report["runs"])==18
    @test report["full_complex_s_gate"]==.005
    @test report["original_voltage_residual_gate"]==1e-9
    @test all(row->row["status"]=="PASS",report["runs"])
    for (path,digest) in report["source_sha256_before"]
        suffix=match(r"/((?:src|validation)/.+)$",replace(path,'\\'=>'/'))
        @test suffix!==nothing
        archived=joinpath(directory,"source_snapshot",suffix.captures[1])
        @test bytes2hex(sha256(read(archived)))==digest
    end
    repeat=joinpath(directory,"library_domain_repeat")
    repeated=TOML.parsefile(joinpath(repeat,"comparison.toml"))
    @test repeated["source_unchanged"]
    @test repeated["source_sha256_before"]==repeated["source_sha256_after"]
    @test length(repeated["runs"])==18
    @test repeated["full_complex_s_gate"]==.005
    @test repeated["original_voltage_residual_gate"]==1e-9
    @test all(row->row["status"]=="PASS",repeated["runs"])
    for (path,digest) in repeated["source_sha256_before"]
        suffix=match(r"/((?:src|validation)/.+)$",replace(path,'\\'=>'/'))
        @test suffix!==nothing
        archived=joinpath(directory,"repeat_source_snapshot",suffix.captures[1])
        @test bytes2hex(sha256(read(archived)))==digest
    end
    domain=TOML.parsefile(joinpath(@__DIR__,"fixtures","library_stored_domain","after.toml"))
    library_key=only(filter(path->endswith(path,"PlanarLibrary.jl"),collect(keys(repeated["source_sha256_before"]))))
    @test repeated["source_sha256_before"][library_key]==domain["source_sha256"]
    placement=joinpath(directory,"placement_ratio_repeat")
    placed=TOML.parsefile(joinpath(placement,"comparison.toml"))
    @test placed["source_unchanged"]
    @test placed["source_sha256_before"]==placed["source_sha256_after"]
    @test length(placed["runs"])==18
    @test placed["full_complex_s_gate"]==.005
    @test placed["original_voltage_residual_gate"]==1e-9
    @test all(row->row["status"]=="PASS",placed["runs"])
    @test length(placed["source_sha256_after"])==12
    for (path,digest) in placed["source_sha256_after"]
        suffix=match(r"/((?:src|validation)/.+)$",replace(path,'\\'=>'/'))
        @test suffix!==nothing
        @test bytes2hex(sha256(read(joinpath(directory,"placement_source_snapshot",suffix.captures[1]))))==digest
    end
    # Recorded source hashes identify the immutable native-run snapshot
    # checked above. Current behavior is checked against native matrices
    # below; Git may normalize current source line endings on checkout.
    initial=TOML.parsefile(joinpath(directory,"initial_pec","comparison.toml"))
    @test initial["source_unchanged"]
    @test length(initial["runs"])==9
    @test all(row->row["status"]=="PASS",initial["runs"])
    original_key=only(filter(path->endswith(path,"validate_sonnet_library_idc.jl"),collect(keys(initial["source_sha256_before"]))))
    @test bytes2hex(sha256(read(joinpath(directory,"initial_validator.jl"))))==initial["source_sha256_before"][original_key]
    for resistance in (0.,.1),cells in (16,32,64)
        layout=idc_library_layout(cells;sheet_resistance=resistance)
        problem=planar_layout_problem(layout)
        expected=BitMatrix([any(x0<=(i-.5)/cells<=x1 && y0<=(j-.5)/cells<=y1
            for (x0,x1,y0,y1) in IDC_NATIVE_RECTANGLES) for i in 1:cells,j in 1:cells])
        @test only(problem.sheets).mask==expected
        connectivity=planar_connectivity(layout)
        @test connectivity.component_count==2
        @test connectivity.port_components==[[1],[2]]
        @test isempty(connectivity.open_nets) && isempty(connectivity.shorted_nets)
        @test_throws ArgumentError solve_planar(layout,1e9;max_bytes=1)
        for frequency in (1e9,1e10,2e10)
            tag="native_$(resistance==0 ? "pec" : "resistive")_$(cells)_$(Int(frequency))"
            folder=joinpath(placement,tag)
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
            path=joinpath(folder,"native_raw.s2p")
            @test guard["output_sha256"]==bytes2hex(sha256(read(path)))
            native=planar_read_touchstone(path)
            original=planar_read_touchstone(joinpath(proof,tag,"native_raw.s2p"))
            @test only(native.s)==only(original.s)
            @test native.frequencies==[frequency]
            @test native.z0==fill(50.,2)
            result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
                max_bytes=2_000_000_000,retain_matrix=true)
            @test maximum(abs,result.s-only(native.s))<=.005
            @test idc_original_residual(problem,result)<=1e-9
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
