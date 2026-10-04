# Public RFIC library IDC and physical wall leads versus literal native GEO.
# A matched finite-grid comparison does not certify measured-device accuracy.
using DiffMoM,Test,TOML,SHA,LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference

const IDC_NATIVE_RECTANGLES=[
    (.25,.3125,.3125,.75),(.6875,.75,.3125,.75),
    (.3125,.625,.3125,.375),(.375,.6875,.4375,.5),
    (.3125,.625,.5625,.625),(.375,.6875,.6875,.75),
    (0.,.3125,.4375,.5625),(.6875,1.,.4375,.5625)]

function idc_library_layout(cells)
    grid=CellGrid(.001,.001,cells,cells)
    stack=PlanarStackup([PlanarLayer(4.,1.,.0001),PlanarLayer(1.,1.,.0001)],
        TERM_GND,TERM_GND,.001,.001)
    idc=planar_transform(planar_interdigital_capacitor(fingers=2,
        finger_length=.3125e-3,width=.0625e-3,gap=.0625e-3,
        bus_width=.0625e-3,level=1,metal="pec",net1="left",net2="right");
        offset=(.25e-3,.53125e-3))
    left=planar_transform(planar_line(length=.3125e-3,width=.125e-3,
        level=1,metal="pec",name="left_lead",net="left");offset=(0.,.5e-3))
    right=planar_transform(planar_line(length=.3125e-3,width=.125e-3,
        level=1,metal="pec",name="right_lead",net="right");offset=(.6875e-3,.5e-3))
    return build_planar_layout(stack,grid,[idc,left,right],
        [planar_pin(left,"p1"),planar_pin(right,"p2")])
end

function idc_literal_native(cells,frequency)
    io=IOBuffer()
    println(io,"FTYP SONPROJ 19\nDIM\nANG DEG\nCAP PF\nCON /OH\nFREQ GHZ\nIND NH\nLNG MM\nRES OH\nEND DIM\nCONTROL\nVARSWP\nOPTIONS\nSPEED 0\nSUBSPLAM N 100\nEND CONTROL\nGEO")
    println(io,"TMET \"PEC\" 0 SUP 0 0 0 0\nBMET \"PEC\" 0 SUP 0 0 0 0\nBOX 1 1 1 $(2cells) $(2cells) 100 0\n.1 1 1 0 0 0 2 \"Air\"\n.1 4 1 0 0 0 2 \"Substrate\"")
    println(io,"POR1 BOX\nPOLY 7 1\n3\n1 50 0 0 0 0 .5\nPOR1 BOX\nPOLY 8 1\n1\n2 50 0 0 0 1 .5\nNUM 8")
    for (id,(x0,x1,y0,y1)) in enumerate(IDC_NATIVE_RECTANGLES)
        println(io,"0 5 -1 N $(id) 1 1 100 100 0 0 0 Y\n$(x0) $(y0)\n$(x1) $(y0)\n$(x1) $(y1)\n$(x0) $(y1)\n$(x0) $(y0)\nEND")
    end
    println(io,"END GEO\nVARSWP\nENABLED Y\nFREQ Y AN SWEEP $(frequency/1e9)\nEND\nEND VARSWP\nFILEOUT\nTOUCH ND Y native_raw.s2p IC 15 S RI R 50\nFOLDER .\nEND FILEOUT")
    return String(take!(io))
end

function idc_native_proof()
    em=find_em();em===nothing && error("installed native engine required")
    directory=evidence_directory("library_idc_native")
    sourcepaths=[@__FILE__,joinpath(@__DIR__,"sonnet_reference.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarLibrary.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarLayout.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarLayoutIO.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarConnectivity.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarSolve.jl")]
    hashes()=Dict(abspath(path)=>bytes2hex(sha256(read(path))) for path in sourcepaths)
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("scope"=>"public RFIC library two-finger-per-electrode IDC plus physical wall leads versus independent literal native GEO; PEC, air/substrate and matched finite grids; no measured-device, nonlinear, loss or continuum claim",
        "full_complex_s_gate"=>.06,"original_voltage_residual_gate"=>1e-9,
        "source_sha256_before"=>hashes(),"runs"=>rows)
    try
        @testset "Actual native public RFIC library IDC" begin
            for cells in parse.(Int,split(get(ENV,"SONNET_IDC_GRIDS","16,32,64"),','))
                cells>=16 && cells%16==0 || error("IDC grids must be positive multiples of16")
                layout=idc_library_layout(cells);problem=planar_layout_problem(layout)
                expected=BitMatrix([any(x0<=(i-.5)/cells<=x1 && y0<=(j-.5)/cells<=y1
                    for (x0,x1,y0,y1) in IDC_NATIVE_RECTANGLES) for i in 1:cells,j in 1:cells])
                @test only(problem.sheets).mask==expected
                connectivity=planar_connectivity(layout)
                @test connectivity.component_count==2
                @test isempty(connectivity.open_nets) && isempty(connectivity.shorted_nets)
                @test connectivity.port_components==[[1],[2]]
                for frequency in parse.(Float64,split(get(ENV,"SONNET_IDC_FREQUENCIES","1e9,1e10,2e10"),','))
                    tag="$(cells)_$(Int(frequency))";source=joinpath(directory,"native_$(tag).son")
                    write(source,idc_literal_native(cells,frequency))
                    reference=reference_run(em,source;output_dir=joinpath(directory,"native_$(tag)"),deembedded=false)
                    file=joinpath(reference.output_dir,"native_raw.s2p")
                    native=only(checked_native_touchstone(reference,file;deembedded=false).s)
                    imported=sonnet_planar_problem(read_sonnet_project(source);freq=frequency)
                    @test only(imported.sheets).mask==expected
                    result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
                        max_bytes=2_000_000_000,retain_matrix=true)
                    raw=result isa PlanarContractedResult ? result.raw : result
                    # Independent unit-voltage wall source, retaining the
                    # original assembled equations rather than LU storage.
                    rhs=zeros(ComplexF64,size(raw.currents))
                    for basis in eachindex(problem.basis.port)
                        port=problem.basis.port[basis];port==0 && continue
                        rhs[basis,port]=(port==1 ? -1. : 1.)*problem.basis.width[basis]
                    end
                    errors=raw.z_mom*raw.currents-rhs
                    residual=maximum(norm(view(errors,:,port))/norm(view(rhs,:,port)) for port in 1:2)
                    delta=maximum(abs,result.s-native)
                    push!(rows,Dict("actual_grid"=>[cells,cells],"frequency_hz"=>frequency,
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
