# Independent native GEO versus a D-Pack and two connecting bus features.
# Native polygons are literal hand-authored rectangles, never importer masks.
using DiffMoM, Test, SHA, TOML, LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference

function main()
    em=find_em();em===nothing && error("installed Sonnet engine required")
    directory=evidence_directory("odb_dpack_native")
    features=joinpath(directory,"dpack.features")
    write(features,"UNITS=MM\nF 3\n\$0 dpack900x500x75x125x3x2\n\$1 rect1000x125\nP .5 .5 0 P 0 0\nP .5 .3125 1 P 0 0\nP .5 .6875 1 P 0 0\n")
    artwork=read_odb_features(features;layer="metal")
    # Independently declared 6 pads and 2 buses in physical millimeters.
    # Native pads retain only the exposed protrusions: the two buses supply
    # the remaining pad area. Thus polygons touch without overlapping and
    # the buses alone meet the wall-port apertures.
    rectangles=[(.05,.3,.375,.4375),(.375,.625,.375,.4375),(.7,.95,.375,.4375),
        (.05,.3,.5625,.625),(.375,.625,.5625,.625),(.7,.95,.5625,.625),
        (0.,1.,.25,.375),(0.,1.,.625,.75)]
    sourcepaths=[@__FILE__,joinpath(@__DIR__,"sonnet_reference.jl"),
        joinpath(@__DIR__,"..","..","src","planar","PlanarODBSymbolsExtra.jl"),
        joinpath(@__DIR__,"..","..","src","planar","PlanarODBIO.jl"),
        joinpath(@__DIR__,"..","..","src","planar","PlanarArtworkIO.jl"),
        joinpath(@__DIR__,"..","..","src","planar","PlanarSolve.jl")]
    hashes()=Dict(abspath(path)=>bytes2hex(sha256(read(path))) for path in sourcepaths)
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("scope"=>"public D-Pack import/physical solve versus independent hand-authored native GEO; no native ODB translator or mesh-convergence certificate",
        "feature_sha256"=>bytes2hex(sha256(read(features))),"source_sha256_before"=>hashes(),
        "full_complex_s_gate"=>.06,"original_voltage_residual_gate"=>1e-9,"runs"=>rows)
    try
        @testset "Actual native D-Pack loaded geometry acceptance" begin
            for cells in parse.(Int,split(get(ENV,"SONNET_DPACK_GRIDS","16,32,64"),','))
                grid=CellGrid(.001,.001,cells,cells)
                stack=PlanarStackup([PlanarLayer(1.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
                lower=cells÷4+1:3cells÷8;upper=5cells÷8+1:3cells÷4
                ports=[PlanarPort(1,:west,lower,50.),PlanarPort(1,:east,lower,50.),
                    PlanarPort(1,:west,upper,50.),PlanarPort(1,:east,upper,50.)]
                expected=BitMatrix([any(x0<=(i-.5)/cells<=x1 && y0<=(j-.5)/cells<=y1
                    for (x0,x1,y0,y1) in rectangles) for i in 1:cells,j in 1:cells])
                prob=artwork_planar_problem(artwork,stack,grid,Dict("metal"=>(;kind=:sheet,interface=1)),ports;max_bytes=2_000_000_000)
                @test only(prob.sheets).mask==expected
                for frequency in parse.(Float64,split(get(ENV,"SONNET_DPACK_FREQUENCIES","1e9,5e9,1e10"),','))
                    tag="$(cells)_$(Int(frequency))";source=joinpath(directory,"native_$(tag).son")
                    open(source,"w") do io
                        println(io,"FTYP SONPROJ 19\nDIM\nANG DEG\nCAP PF\nCON /OH\nFREQ GHZ\nIND NH\nLNG MM\nRES OH\nEND DIM\nCONTROL\nVARSWP\nOPTIONS\nSPEED 0\nSUBSPLAM N 100\nEND CONTROL\nGEO")
                        println(io,"TMET \"PEC\" 0 SUP 0 0 0 0\nBMET \"PEC\" 0 SUP 0 0 0 0\nBOX 1 1 1 $(2cells) $(2cells) 100 0\n.1 1 1 0 0 0 2 \"Air\"\n.1 1 1 0 0 0 2 \"Air\"")
                        for (number,polygon,edge,y) in ((1,7,3,.3125),(2,7,1,.3125),(3,8,3,.6875),(4,8,1,.6875))
                            println(io,"POR1 BOX\nPOLY $(polygon) 1\n$(edge)\n$(number) 50 0 0 0 $(iseven(number) ? 1 : 0) $(y)")
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
                    row=Dict{String,Any}("actual_grid"=>[cells,cells],"frequency_hz"=>frequency,
                        "full_complex_s_error"=>error,"original_voltage_residual"=>residual,
                        "native_output_sha256"=>bytes2hex(sha256(read(file))),"native_source_sha256"=>bytes2hex(sha256(read(source))),
                        "unknowns"=>planar_basis_count(prob.basis),"independent_mask_equal"=>true,
                        "status"=>error<=.06 && residual<=1e-9 ? "PASS" : "FAIL")
                    push!(rows,row)
                    println(row)
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
main()
