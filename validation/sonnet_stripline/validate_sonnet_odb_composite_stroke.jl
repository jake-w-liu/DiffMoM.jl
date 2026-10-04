# Public enclosing ODB product composite stroke versus independent native GEO.
# Native ODB translation and continuum convergence are separate gates.
using DiffMoM,Test,TOML,SHA,LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference

function native_proof()
    em=find_em();em===nothing && error("installed native engine required")
    directory=evidence_directory("odb_composite_line_public_native")
    product=joinpath(directory,"odb_product")
    mkpath(joinpath(product,"matrix"))
    write(joinpath(product,"matrix","matrix"),"STEP {\nCOL=1\nNAME=board\n}\nLAYER {\nROW=1\nNAME=metal\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}\n")
    step=joinpath(product,"steps","board");mkpath(joinpath(step,"layers","metal"))
    write(joinpath(step,"stephdr"),"UNITS=MM\n")
    write(joinpath(step,"layers","metal","features"),
        "UNITS=MM\n\$0 framed_island\n\$1 rect1000x125\nL .4375 .5 .5625 .5 0 P 0\nP .5 .3125 1 P 0 0\nP .5 .6875 1 P 0 0\n")
    symbol=joinpath(product,"symbols","framed_island");mkpath(symbol)
    write(joinpath(symbol,"features"),"UNITS=MM\n\$0 rect62.5x125\n"*
        "S P 0\nOB -.1875 -.25 I\nOS .1875 -.25\nOS .1875 .25\nOS -.1875 .25\nOS -.1875 -.25\nOE\n"*
        "OB -.125 -.125 H\nOS .125 -.125\nOS .125 .125\nOS -.125 .125\nOS -.125 -.125\nOE\nSE\nP 0 0 0 P 0 0\n")
    artwork=read_odb(product;step="board",layers=["metal"],max_stroke_boundaries=27)
    # Independent decomposition of the physical swept frame plus reinserted
    # island, declared in Cartesian coordinates rather than read from a mask.
    rectangles=[(.25,.4375,.375,.625),(.5625,.75,.375,.625),
        (0.,1.,.25,.375),(0.,1.,.625,.75),(.40625,.59375,.4375,.5625)]
    sourcepaths=[@__FILE__,joinpath(@__DIR__,"sonnet_reference.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarODBIO.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarArtworkIO.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarArtworkSweep.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarArtworkCertified.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarArtworkExact.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarArtworkLinearExact.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarArtworkRegionExact.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarArtworkBounds.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarArtworkAffine.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarODBSymbolsExtra.jl"),
        joinpath(@__DIR__,"../../src/planar/PlanarSolve.jl")]
    hashes()=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in sourcepaths)
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("scope"=>"public enclosing ODB product with original ordered composite-aperture sweep and physical solve versus independent literal native GEO; no native ODB translator or continuum claim",
        "initial_boundary_fixture_artifact"=>"data/sonnet_validation/odb_composite_line_prototype_FhYQAl",
        "initial_boundary_failure_reason"=>"four 16-cell centers exactly coincide with the literal island boundary; exact stored SI subtraction places them outside by 2.71e-20m, independently verified without a geometric tolerance change",
        "repeat_fixture"=>"island height .125mm gives positive geometric clearance on every declared grid; full-S and original-equation gates unchanged",
        "production_import"=>"read_odb enclosing board with local user symbol","full_complex_s_gate"=>.06,
        "original_voltage_residual_gate"=>1e-9,"source_sha256_before"=>hashes(),"runs"=>rows)
    try
        @testset "Actual native public ODB composite-line coupon" begin
            for cells in parse.(Int,split(get(ENV,"SONNET_COMPOSITE_GRIDS","16,32,64"),','))
                grid=CellGrid(.001,.001,cells,cells)
                stack=PlanarStackup([PlanarLayer(1.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
                lower=cells÷4+1:3cells÷8;upper=5cells÷8+1:3cells÷4
                ports=[PlanarPort(1,:west,lower,50.),PlanarPort(1,:east,lower,50.),
                    PlanarPort(1,:west,upper,50.),PlanarPort(1,:east,upper,50.)]
                expected=BitMatrix([any(x0<=(i-.5)/cells<=x1 && y0<=(j-.5)/cells<=y1
                    for (x0,x1,y0,y1) in rectangles) for i in 1:cells,j in 1:cells])
                prob=artwork_planar_problem(artwork,stack,grid,Dict("metal"=>(;kind=:sheet,interface=1)),ports;max_bytes=2_000_000_000)
                @test only(prob.sheets).mask==expected
                for frequency in (1e9,5e9,1e10)
                    tag="$(cells)_$(Int(frequency))";source=joinpath(directory,"native_$(tag).son")
                    open(source,"w") do io
                        println(io,"FTYP SONPROJ 19\nDIM\nANG DEG\nCAP PF\nCON /OH\nFREQ GHZ\nIND NH\nLNG MM\nRES OH\nEND DIM\nCONTROL\nVARSWP\nOPTIONS\nSPEED 0\nSUBSPLAM N 100\nEND CONTROL\nGEO")
                        println(io,"TMET \"PEC\" 0 SUP 0 0 0 0\nBMET \"PEC\" 0 SUP 0 0 0 0\nBOX 1 1 1 $(2cells) $(2cells) 100 0\n.1 1 1 0 0 0 2 \"Air\"\n.1 1 1 0 0 0 2 \"Air\"")
                        for (number,id,edge,y) in ((1,3,3,.3125),(2,3,1,.3125),(3,4,3,.6875),(4,4,1,.6875))
                            println(io,"POR1 BOX\nPOLY $(id) 1\n$(edge)\n$(number) 50 0 0 0 $(iseven(number) ? 1 : 0) $(y)")
                        end
                        println(io,"NUM $(length(rectangles))")
                        for (id,(x0,x1,y0,y1)) in enumerate(rectangles)
                            println(io,"0 5 -1 N $(id) 1 1 100 100 0 0 0 Y\n$(x0) $(y0)\n$(x1) $(y0)\n$(x1) $(y1)\n$(x0) $(y1)\n$(x0) $(y0)\nEND")
                        end
                        println(io,"END GEO\nVARSWP\nENABLED Y\nFREQ Y AN SWEEP $(frequency/1e9)\nEND\nEND VARSWP\nFILEOUT\nTOUCH ND Y native_raw.s4p IC 15 S RI R 50\nFOLDER .\nEND FILEOUT")
                    end
                    ref=reference_run(em,source;output_dir=joinpath(directory,"native_$(tag)"),deembedded=false,nports=4)
                    file=joinpath(ref.output_dir,"native_raw.s4p")
                    native=only(checked_native_touchstone(ref,file;deembedded=false).s)
                    result=solve_planar_contracted(prob,frequency,Matrix{Float64}(I,4,4);z0=fill(50.,4),
                        method=:dense_fft,mx=4cells,my=4cells,max_bytes=2_000_000_000,retain_matrix=true)
                    error=maximum(abs,result.s-native);residual=maximum(result.raw.relative_residuals)
                    push!(rows,Dict("actual_grid"=>[cells,cells],"frequency_hz"=>frequency,
                        "full_complex_s_error"=>error,"original_voltage_residual"=>residual,
                        "native_output_sha256"=>bytes2hex(sha256(read(file))),
                        "native_source_sha256"=>bytes2hex(sha256(read(source))),
                        "status"=>error<=.06 && residual<=1e-9 ? "PASS" : "FAIL"))
                    println(last(rows))
                    @test error<=.06
                    @test residual<=1e-9
                    @test maximum(abs,result.s-transpose(result.s))<1e-10
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
native_proof()
