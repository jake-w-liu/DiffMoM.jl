# A concave aperture swept along a finite line, compared with independently
# declared native rectangular conductor pieces. No native ODB translator.
using DiffMoM,Test,SHA,TOML,LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference

function main()
    em=find_em();em===nothing && error("installed native Sonnet engine required")
    directory=evidence_directory("odb_concave_stroke_native")
    features=joinpath(directory,"concave_stroke.features")
    write(features,"UNITS=MM\nF 3\n\$0 custom_u\n\$1 rect1000x125\nL .5 .5 .625 .5 0 P 0\nP .5 .3125 1 P 0 0\nP .5 .6875 1 P 0 0\n")
    vertices=[(-.25,-.25),(.25,-.25),(.25,.25),(.125,.25),(.125,-.125),
        (-.125,-.125),(-.125,.25),(-.25,.25)]
    polygon=DiffMoM._artwork_polygon([(x*1e-3,y*1e-3) for (x,y) in vertices])
    open(joinpath(directory,"custom_u.toml"),"w") do io
        TOML.print(io,Dict("units"=>"MM","vertices"=>[collect(v) for v in vertices],
            "scope"=>"explicit public symbol_resolver polygon; not enclosing-product user-symbol stroke acceptance"))
    end
    artwork=read_odb_features(features;layer="metal",symbol_resolver=name->
        name=="custom_u" ? polygon : error("unexpected custom symbol"))
    rectangles=[(.25,.5,.375,.625),(.625,.875,.375,.625),
        (0.,1.,.25,.375),(0.,1.,.625,.75)]
    paths=[@__FILE__,joinpath(@__DIR__,"sonnet_reference.jl"),
        joinpath(@__DIR__,"..","..","src","planar","PlanarArtworkIO.jl"),
        joinpath(@__DIR__,"..","..","src","planar","PlanarODBIO.jl"),
        joinpath(@__DIR__,"..","..","src","planar","PlanarSolve.jl")]
    hashes()=Dict(abspath(p)=>bytes2hex(sha256(read(p))) for p in paths)
    rows=Dict{String,Any}[]
    report=Dict{String,Any}("scope"=>"public concave polygon line stroke/physical EM versus independent literal native GEO; no native ODB translator or continuum-convergence certificate",
        "full_complex_s_gate"=>.06,"original_voltage_residual_gate"=>1e-9,
        "source_sha256_before"=>hashes(),"runs"=>rows)
    try
        @testset "Actual native concave-stroke physical coupon" begin
            for cells in parse.(Int,split(get(ENV,"SONNET_CONCAVE_GRIDS","16,32,64"),','))
                grid=CellGrid(.001,.001,cells,cells)
                stack=PlanarStackup([PlanarLayer(1.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
                lower=cells÷4+1:3cells÷8;upper=5cells÷8+1:3cells÷4
                ports=[PlanarPort(1,:west,lower,50.),PlanarPort(1,:east,lower,50.),
                    PlanarPort(1,:west,upper,50.),PlanarPort(1,:east,upper,50.)]
                expected=BitMatrix([any(x0<=(i-.5)/cells<=x1 && y0<=(j-.5)/cells<=y1
                    for (x0,x1,y0,y1) in rectangles) for i in 1:cells,j in 1:cells])
                prob=artwork_planar_problem(artwork,stack,grid,Dict("metal"=>(;kind=:sheet,interface=1)),ports;max_bytes=2_000_000_000)
                @test only(prob.sheets).mask==expected
                for frequency in parse.(Float64,split(get(ENV,"SONNET_CONCAVE_FREQUENCIES","1e9,5e9,1e10"),','))
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
                        "unknowns"=>planar_basis_count(prob.basis),
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
main()
