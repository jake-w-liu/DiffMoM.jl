# Independent end-to-end artwork coupon: Gerber ordered dark/clear regions,
# legacy MI/SF/OF/IR coordinates, and a complete minimal ODB++ job.
# Native GEO is authored from declared physical rectangle constants below;
# it is never generated from the importer-returned mask or objects.
using DiffMoM, SHA, TOML, LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference
function main()
    em=find_em();em===nothing && return 2
    evidence=evidence_directory("artwork_native")
    plain="""%FSLAX36Y36*%
%MOMM*%
%LPD*%
G36*
X0Y250000D02*
X1000000Y250000D01*
X1000000Y750000D01*
X0Y750000D01*
X0Y250000D01*
G37*
%LPC*%
G36*
X375000Y375000D02*
X625000Y375000D01*
X625000Y625000D01*
X375000Y625000D01*
X375000Y375000D01*
G37*
M02*
"""
    # Fixed legacy execution order yields physical X=.25-.5*y and
    # Y=.5-2*x (all raw coordinates in mm). These literal raw regions
    # independently yield the same physical strip and central clear window.
    legacy="""%FSLAX36Y36*%
%MOMM*%
%MIA1*%
%SFA2B.5*%
%OFA.5B-.25*%
%IR90*%
%LPD*%
G36*
X-125000Y-1500000D02*
X125000Y-1500000D01*
X125000Y500000D01*
X-125000Y500000D01*
X-125000Y-1500000D01*
G37*
%LPC*%
G36*
X-62500Y-750000D02*
X62500Y-750000D01*
X62500Y-250000D01*
X-62500Y-250000D01*
X-62500Y-750000D01*
G37*
M02*
"""
    plainfile=joinpath(evidence,"plain_clear_window.gbr");write(plainfile,plain)
    legacyfile=joinpath(evidence,"legacy_clear_window.gbr");write(legacyfile,legacy)
    job=joinpath(evidence,"odb_clear_window")
    mkpath(joinpath(job,"matrix"));write(joinpath(job,"matrix","matrix"),"STEP {\nCOL=1\nNAME=board\n}\nLAYER {\nROW=1\nNAME=metal\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}\n")
    mkpath(joinpath(job,"steps","board","layers","metal"))
    write(joinpath(job,"steps","board","stephdr"),"UNITS=MM\n")
    features="UNITS=MM\nS P 0\nOB 0 .25 I\nOS 1 .25\nOS 1 .75\nOS 0 .75\nOS 0 .25\nOE\nSE\nS N 0\nOB .375 .375 I\nOS .625 .375\nOS .625 .625\nOS .375 .625\nOS .375 .375\nOE\nSE\n"
    featurefile=joinpath(job,"steps","board","layers","metal","features");write(featurefile,features)
    docs=[("gerber_plain",read_gerber(plainfile;layer="metal"),plainfile),
        ("gerber_legacy",read_gerber(legacyfile;layer="metal"),legacyfile),
        ("odb_job",read_odb(job;step="board",layers=["metal"]),featurefile)]
    rows=Dict{String,Any}[]
    sourcefiles=[plainfile,legacyfile,joinpath(job,"matrix","matrix"),
        joinpath(job,"steps","board","stephdr"),featurefile]
    filehashes=Dict(relpath(path,evidence)=>bytes2hex(sha256(read(path))) for path in sourcefiles)
    modulehashes=Dict(name=>bytes2hex(sha256(read(joinpath(@__DIR__,"..","..","src","planar",name))))
        for name in ("PlanarArtworkIO.jl","PlanarArtworkAffine.jl","PlanarGerberIO.jl",
            "PlanarODBIO.jl","PlanarFFTAssembly.jl","PlanarSolve.jl"))
    priornative=Dict{Float64,Matrix{ComplexF64}}()
    priorcandidate=Dict{Tuple{String,Float64},Matrix{ComplexF64}}()
    report=Dict{String,Any}("scope"=>"independent hand-authored native GEO versus Gerber/ODB import and physical solve of one complete PEC strip with clear window; no native artwork translator claim",
        "physical_box_m"=>[.001,.001],"physical_strip_mm"=>[0.,1.,.25,.75],"physical_clear_window_mm"=>[.375,.625,.375,.625],
        "native_geometry_source"=>"four literal independently declared rectangles sharing true edges; never generated from masks",
        "legacy_coordinate_transform_mm"=>["X=.25-.5*y","Y=.5-2*x"],
        "source_file_sha256"=>filehashes,"module_source_sha256"=>modulehashes,
        "benchmark_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "complex_s_gate"=>.06,"voltage_residual_gate"=>1e-9,"variants"=>rows)
    rects=[(0.,.375,.25,.75),(.625,1.,.25,.75),(.375,.625,.25,.375),(.375,.625,.625,.75)]
    for cells in parse.(Int,split(get(ENV,"SONNET_ARTWORK_GRIDS","16,32,64"),','))
        grid=CellGrid(.001,.001,cells,cells)
        stack=PlanarStackup([PlanarLayer(1.,1.,.1e-3),PlanarLayer(1.,1.,.1e-3)],TERM_GND,TERM_GND,.001,.001)
        rowsport=cells÷4+1:3cells÷4
        ports=[PlanarPort(1,:west,rowsport,50.),PlanarPort(1,:east,rowsport,50.)]
        expected=falses(cells,cells)
        for j in 1:cells,i in 1:cells
            x=(i-.5)/cells;y=(j-.5)/cells
            expected[i,j]=.25<y<.75 && !( .375<x<.625 && .375<y<.625 )
        end
        problems=Tuple{String,PlanarProblem,String}[]
        for (name,doc,path) in docs
            prob=artwork_planar_problem(doc,stack,grid,Dict("metal"=>(;kind=:sheet,interface=1)),ports;max_bytes=2_000_000_000)
            only(prob.sheets).mask==expected || error("$name differs from independent analytic physical membership on $cells cells")
            push!(problems,(name,prob,path))
        end
        for f in parse.(Float64,split(get(ENV,"SONNET_ARTWORK_FREQUENCIES","1e9,5e9,1e10"),','))
            tag="$(cells)_$(Int(f))"; source=joinpath(evidence,"native_"*tag*".son")
            open(source,"w") do io
                println(io,"FTYP SONPROJ 19\nDIM\nANG DEG\nCAP PF\nCON /OH\nFREQ GHZ\nIND NH\nLNG MM\nRES OH\nEND DIM\nCONTROL\nVARSWP\nOPTIONS\nSPEED 0\nSUBSPLAM N 100\nEND CONTROL\nGEO")
                println(io,"TMET \"PEC\" 0 SUP 0 0 0 0\nBMET \"PEC\" 0 SUP 0 0 0 0\nBOX 1 1 1 ",2cells," ",2cells," 100 0\n.1 1 1 0 0 0 2 \"Air\"\n.1 1 1 0 0 0 2 \"Air\"")
                println(io,"POR1 BOX\nPOLY 1 1\n3\n1 50 0 0 0 0 .5\nPOR1 BOX\nPOLY 2 1\n1\n2 50 0 0 0 1 .5\nNUM 4")
                for (id,(x0,x1,y0,y1)) in enumerate(rects)
                    println(io,"0 5 -1 N ",id," 1 1 100 100 0 0 0 Y\n",x0," ",y0,"\n",x1," ",y0,"\n",x1," ",y1,"\n",x0," ",y1,"\n",x0," ",y0,"\nEND")
                end
                println(io,"END GEO\nVARSWP\nENABLED Y\nFREQ Y AN SWEEP ",f/1e9,"\nEND\nEND VARSWP\nFILEOUT\nTOUCH ND Y native_raw.s2p IC 15 S RI R 50\nFOLDER .\nEND FILEOUT")
            end
            native=reference_run(em,source;output_dir=joinpath(evidence,"native_"*tag),deembedded=false)
            nativeoutput=joinpath(native.output_dir,"native_raw.s2p");sr=only(checked_native_touchstone(native,nativeoutput;deembedded=false).s)
            nativecounts=get(TOML.parsefile(joinpath(native.output_dir,"metadata.toml")),"native_subsections",Int[])
            for (name,prob,path) in problems
                row=Dict{String,Any}("format"=>name,"frequency_hz"=>f,"actual_grid"=>[cells,cells],"box_halfcell_counts"=>[2cells,2cells],"modes"=>[4cells,4cells],"unknowns"=>planar_basis_count(prob.basis),"occupied_cells"=>count(expected),"independent_membership_equal"=>true,"fabrication_source_sha256"=>bytes2hex(sha256(read(path))),"native_source_sha256"=>bytes2hex(sha256(read(source))),"native_output_sha256"=>bytes2hex(sha256(read(nativeoutput))))
                started=time()
                row["native_subsections"]=nativecounts
                haskey(priornative,f) && (row["native_grid_refinement_complex_step"]=maximum(abs.(sr-priornative[f])))
                try
                    result=solve_planar_contracted(prob,f,Matrix{Float64}(I,2,2);z0=[50.,50.],method=:dense_fft,mx=4cells,my=4cells,max_bytes=2_000_000_000,retain_matrix=true)
                    ds=maximum(abs.(result.s-sr));residual=maximum(result.raw.relative_residuals)
                    row["max_complex_s_error"]=ds;row["voltage_residual"]=residual
                    row["status"]=residual>report["voltage_residual_gate"] ? "UNVERIFIED" : ds<=report["complex_s_gate"] ? "PASS" : "FAIL"
                    row["s_real"]=collect(vec(real.(result.s)));row["s_imag"]=collect(vec(imag.(result.s)))
                    row["native_s_real"]=collect(vec(real.(sr)));row["native_s_imag"]=collect(vec(imag.(sr)))
                    haskey(priorcandidate,(name,f)) && (row["candidate_grid_refinement_complex_step"]=maximum(abs.(result.s-priorcandidate[(name,f)])))
                    priorcandidate[(name,f)]=copy(result.s)
                    write_touchstone(joinpath(evidence,name*"_"*tag*".s2p"),[f],[result.s])
                    result=nothing
                catch e
                    row["status"]="UNVERIFIED";row["error"]=sprint(showerror,e)
                end
                row["seconds"]=time()-started;push!(rows,row);GC.gc()
                open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
                println(name," grid",cells," ",f," ",row["status"]," ",get(row,"max_complex_s_error",get(row,"error","")))
            end
            priornative[f]=copy(sr)
        end
    end
    println("evidence: ",evidence)
    return all(row->row["status"]=="PASS",rows) ? 0 : 1
end
exit(main())

