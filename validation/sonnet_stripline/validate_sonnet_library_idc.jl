# Public RFIC library IDC and physical wall leads versus literal native GEO.
# A matched finite-grid comparison does not certify measured-device accuracy.
using DiffMoM,Test,TOML,SHA,LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference
include("library_idc_fixture.jl")

function idc_native_proof()
    em=find_em();em===nothing && error("installed native engine required")
    directory=evidence_directory("library_idc_native")
    sourcepaths=[@__FILE__,joinpath(@__DIR__,"sonnet_reference.jl"),
        joinpath(@__DIR__,"library_idc_fixture.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarLibrary.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarLayout.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarLayoutIO.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarConnectivity.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarSonnetIO.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarSonnetScalarFiles.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarGreens.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarFFTAssembly.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarSolve.jl")]
    hashes()=Dict(abspath(path)=>bytes2hex(sha256(read(path))) for path in sourcepaths)
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("scope"=>"public RFIC library two-finger-per-electrode IDC plus physical wall leads versus independent literal native GEO; PEC and constant resistive film, air/substrate and matched finite grids; no measured-device, nonlinear, general loss or continuum claim",
        "full_complex_s_gate"=>.005,"original_voltage_residual_gate"=>1e-9,
        "source_sha256_before"=>hashes(),"runs"=>rows)
    try
        @testset "Actual native public RFIC library IDC" begin
            for resistance in (0.,.1),cells in parse.(Int,split(get(ENV,"SONNET_IDC_GRIDS","16,32,64"),','))
                cells>=16 && cells%16==0 || error("IDC grids must be positive multiples of16")
                layout=idc_library_layout(cells;sheet_resistance=resistance);problem=planar_layout_problem(layout)
                expected=BitMatrix([any(x0<=(i-.5)/cells<=x1 && y0<=(j-.5)/cells<=y1
                    for (x0,x1,y0,y1) in IDC_NATIVE_RECTANGLES) for i in 1:cells,j in 1:cells])
                @test only(problem.sheets).mask==expected
                connectivity=planar_connectivity(layout)
                @test connectivity.component_count==2
                @test isempty(connectivity.open_nets) && isempty(connectivity.shorted_nets)
                @test connectivity.port_components==[[1],[2]]
                for frequency in parse.(Float64,split(get(ENV,"SONNET_IDC_FREQUENCIES","1e9,1e10,2e10"),','))
                    tag="$(resistance==0 ? "pec" : "resistive")_$(cells)_$(Int(frequency))";source=joinpath(directory,"native_$(tag).son")
                    write(source,idc_literal_native(cells,frequency;sheet_resistance=resistance))
                    reference=reference_run(em,source;output_dir=joinpath(directory,"native_$(tag)"),deembedded=false)
                    file=joinpath(reference.output_dir,"native_raw.s2p")
                    native=only(checked_native_touchstone(reference,file;deembedded=false).s)
                    imported=sonnet_planar_problem(read_sonnet_project(source);freq=frequency,_materials=true)
                    @test only(imported.sheets).mask==expected
                    result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
                        max_bytes=2_000_000_000,retain_matrix=true)
                    residual=idc_original_residual(problem,result)
                    delta=maximum(abs,result.s-native)
                    push!(rows,Dict("actual_grid"=>[cells,cells],"frequency_hz"=>frequency,
                        "sheet_resistance_ohm_per_square"=>resistance,
                        "basis_count"=>planar_basis_count(problem.basis),"full_complex_s_error"=>delta,
                        "original_voltage_residual"=>residual,
                        "native_source_sha256"=>bytes2hex(sha256(read(source))),
                        "native_output_sha256"=>bytes2hex(sha256(read(file))),
                        "status"=>delta<=.005 && residual<=1e-9 ? "PASS" : "FAIL"))
                    println(last(rows))
                    @test delta<=.005
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
idc_native_proof()
