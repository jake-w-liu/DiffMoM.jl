# Native raw interior-terminal proof for a public rectangular spiral.
using DiffMoM, LinearAlgebra, TOML, SHA, Test
include("sonnet_reference.jl")
using .SonnetReference
include("solver_spiral_fixture.jl")
BLAS.set_num_threads(1)

function solver_spiral_native_proof()
    em=find_em(); em===nothing && error("installed native engine required")
    directory=evidence_directory("solver_spiral_native")
    sourcepaths=[@__FILE__,joinpath(@__DIR__,"solver_spiral_fixture.jl"),
        joinpath(@__DIR__,"sonnet_reference.jl"),
        [joinpath(@__DIR__,"../../src/planar/"*name) for name in
            ("PlanarLibrary.jl","PlanarLayout.jl","PlanarTerminalReturns.jl",
             "PlanarPortContraction.jl","PlanarSolve.jl","PlanarGreens.jl",
             "PlanarFFTAssembly.jl","PlanarSonnetIO.jl")]...]
    hashes()=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in sourcepaths)
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("scope"=>"public rectangular two-turn spiral with raw interior pins and physical PEC-cover returns versus independent literal eight-rectangle native GEO with explicit axial-source return strips; matched finite grids, isotropic dielectric, PEC/constant film; no native automatic terminal semantics, calibration, continuum convergence, underpass, measured-device or general native-loss claim",
        "full_complex_s_gate"=>.005,"original_voltage_residual_gate"=>1e-9,
        "relative_power_gate"=>1e-8,
        "source_sha256_before"=>hashes(),"runs"=>rows)
    try
        for resistance in parse.(Float64,split(get(ENV,"SOLVER_SPIRAL_RESISTANCE","0,.1"),',')),
            cells in parse.(Int,split(get(ENV,"SOLVER_SPIRAL_GRIDS","32,64"),','))
            cells>=32 && cells%32==0 || error("grids must be multiples of32")
            layout=solver_spiral_layout(cells;sheet_resistance=resistance)
            @test layout.problem.sheets[1].mask == solver_spiral_masks(cells)
            connectivity=planar_connectivity(layout)
            @test connectivity.component_count==1
            @test connectivity.port_components==[[1],[1]]
            @test isempty(connectivity.open_nets) && isempty(connectivity.shorted_nets)
            @test length(layout.terminal_paths)==2
            @test all(p->p.direction==:below && p.layers==1:1,layout.terminal_paths)
            for frequency in parse.(Float64,split(get(ENV,"SOLVER_SPIRAL_FREQUENCIES","1e9,5e9,1e10"),','))
                tag="$(iszero(resistance) ? "pec" : "resistive")_$(cells)_$(Int(frequency))"
                source=joinpath(directory,"native_$(tag).son")
                write(source,solver_spiral_literal_native(cells,frequency;sheet_resistance=resistance))
                native=reference_run(em,source;output_dir=joinpath(directory,"native_$(tag)"),deembedded=false)
                output=joinpath(native.output_dir,"native_raw.s2p")
                sn=only(checked_native_touchstone(native,output;deembedded=false).s)
                imported=sonnet_planar_problem(read_sonnet_project(source);freq=frequency,_details=true,_materials=true)
                @test imported.problem.sheets[1].mask == solver_spiral_masks(cells)
                @test reduce(.|, (v.uni for v in imported.problem.vias)) ==
                    reduce(.|, (v.uni for v in layout.source_problem.vias))
                @test all(v->v.layer==1 && v.tap==v.uni,imported.problem.vias)
                result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
                    retain_matrix=true,max_bytes=2_000_000_000)
                residual=solver_spiral_original_residual(layout,result)
                delta=maximum(abs,result.s-sn)
                row=Dict{String,Any}("actual_grid"=>[cells,cells],"frequency_hz"=>frequency,
                    "sheet_resistance_ohm_per_square"=>resistance,
                    "basis_count"=>planar_basis_count(layout.source_problem.basis),
                    "full_complex_s_error"=>delta,"original_voltage_residual"=>residual,
                    "native_source_sha256"=>bytes2hex(sha256(read(source))),
                    "native_output_sha256"=>bytes2hex(sha256(read(output))),
                    "max_s_singular_value"=>opnorm(result.s),
                    "reciprocity_error"=>maximum(abs,result.s-transpose(result.s)),
                    "status"=>delta<=.005 && residual<=1e-9 ? "PASS" : "FAIL")
                push!(rows,row); println(row)
                planar_write_touchstone(joinpath(native.output_dir,"diffmom.s2p"),
                    PlanarNetworkData([frequency],[result.s]))
                # Record all candidate evidence before applying the declared
                # native gate; failure remains diagnostic, not a fitted oracle.
                @test residual<=1e-9
                @test maximum(abs,result.s-transpose(result.s))<=1e-10
                @test opnorm(result.s)<=1+1e-9
                if resistance>0
                    powers=solver_spiral_sheet_power(layout,result,resistance)
                    row["supplied_power_per_volt_squared"]=powers.supplied
                    row["integrated_sheet_dissipation_per_volt_squared"]=powers.dissipated
                    row["relative_power_error"]=powers.relative_error
                    @test powers.supplied>0 && powers.relative_error<=1e-8
                end
                if cells==32 && frequency==1e9
                    direct=solve_planar(layout,frequency;method=:dense,mx=4cells,my=4cells,
                        retain_matrix=true,max_bytes=2_000_000_000)
                    @test direct.s≈result.s rtol=1e-9
                    @test solver_spiral_original_residual(layout,direct)<=1e-9
                    direct=nothing
                end
                result=nothing; GC.gc()
            end
        end
    finally
        report["source_sha256_after"]=hashes()
        report["source_unchanged"]=report["source_sha256_before"]==report["source_sha256_after"]
        open(joinpath(directory,"comparison.toml"),"w") do io; TOML.print(io,report); end
        println("evidence: ",directory)
    end
    @test report["source_unchanged"]
    @test all(r->r["status"]=="PASS",rows)
end
solver_spiral_native_proof()
