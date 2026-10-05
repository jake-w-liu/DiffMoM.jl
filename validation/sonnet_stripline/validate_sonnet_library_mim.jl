# Native finite-grid physical MIM proof; no measured/continuum certificate.
using DiffMoM,Test,TOML,SHA,LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference
include("library_mim_fixture.jl")

function mim_native_proof()
    em=find_em();em===nothing && error("installed native engine required")
    directory=evidence_directory("library_mim_native")
    sourcepaths=[@__FILE__,joinpath(@__DIR__,"sonnet_reference.jl"),
        joinpath(@__DIR__,"library_mim_fixture.jl"),
        [joinpath(@__DIR__,"../../src/planar/"*name) for name in
            ("PlanarLibrary.jl","PlanarLayout.jl","PlanarLayoutIO.jl",
             "PlanarConnectivity.jl","PlanarSolve.jl","PlanarGreens.jl",
             "PlanarFFTAssembly.jl","PlanarSonnetIO.jl","PlanarSonnetScalarFiles.jl")]...]
    hashes()=Dict(abspath(path)=>bytes2hex(sha256(read(path))) for path in sourcepaths)
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("scope"=>"public RFIC library two-level MIM plus literal wall leads versus independent native six-rectangle " *
        "GEO; matched finite grids, 1um isotropic dielectric, PEC/constant film; no interior calibration, " *
        "measured-device, nonlinear, general loss or continuum claim",
        "full_complex_s_gate"=>.06,"original_voltage_residual_gate"=>1e-9,
        "source_sha256_before"=>hashes(),"runs"=>rows)
    try
        @testset "Actual native public RFIC library MIM" begin
            for resistance in (0.,.1),cells in parse.(Int,split(get(ENV,"SONNET_MIM_GRIDS","16,32,64"),','))
                cells>=16 && cells%16==0 || error("MIM grids must be positive multiples of16")
                layout=mim_library_layout(cells;sheet_resistance=resistance);problem=planar_layout_problem(layout)
                expected=mim_literal_masks(cells)
                @test [sheet.mask for sheet in problem.sheets]==expected
                connectivity=planar_connectivity(layout)
                @test connectivity.component_count==2
                @test isempty(connectivity.open_nets) && isempty(connectivity.shorted_nets)
                @test length(unique(only.(connectivity.port_components)))==2
                for frequency in parse.(Float64,split(get(ENV,"SONNET_MIM_FREQUENCIES","1e9,1e10,2e10"),','))
                    tag="$(resistance==0 ? "pec" : "resistive")_$(cells)_$(Int(frequency))";source=joinpath(directory,"native_$(tag).son")
                    write(source,mim_literal_native(cells,frequency;sheet_resistance=resistance))
                    reference=reference_run(em,source;output_dir=joinpath(directory,"native_$(tag)"),deembedded=false)
                    file=joinpath(reference.output_dir,"native_raw.s2p")
                    native=only(checked_native_touchstone(reference,file;deembedded=false).s)
                    imported=sonnet_planar_problem(read_sonnet_project(source);freq=frequency,_materials=true)
                    # Native polygon encounter order need not match the
                    # public library's ascending interface order.
                    @test Dict(sheet.interface=>sheet.mask for sheet in imported.sheets)==
                        Dict(interface=>expected[interface] for interface in (1,2))
                    result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
                        max_bytes=2_000_000_000,retain_matrix=true)
                    residual=mim_original_residual(problem,result);delta=maximum(abs,result.s-native)
                    push!(rows,Dict("actual_grid"=>[cells,cells],"frequency_hz"=>frequency,
                        "sheet_resistance_ohm_per_square"=>resistance,
                        "basis_count"=>planar_basis_count(problem.basis),"full_complex_s_error"=>delta,
                        "original_voltage_residual"=>residual,
                        "native_source_sha256"=>bytes2hex(sha256(read(source))),
                        "native_output_sha256"=>bytes2hex(sha256(read(file))),
                        "status"=>delta<=.06 && residual<=1e-9 ? "PASS" : "FAIL"))
                    println(last(rows))
                    @test delta<=.06
                    @test residual<=1e-9
                    @test maximum(abs,result.s-transpose(result.s))<=1e-10
                    @test eigmax(Hermitian(result.s'*result.s))<=1+1e-9
                    planar_write_touchstone(joinpath(directory,"native_$(tag)","diffmom.s2p"),
                        PlanarNetworkData([frequency],[result.s]))
                    result=nothing;GC.gc()
                end
            end
        end
    finally
        report["source_sha256_after"]=hashes()
        report["source_unchanged"]=report["source_sha256_before"]==report["source_sha256_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("evidence: ",directory)
    end
    @test report["source_unchanged"]
end
mim_native_proof()
