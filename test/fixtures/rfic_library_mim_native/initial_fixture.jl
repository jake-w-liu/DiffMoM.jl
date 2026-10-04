# Public SI library geometry and independent literal native rectangles in mm.
using DiffMoM,LinearAlgebra
const MIM_NATIVE_RECTANGLES=[
    (0,.375,.625,.375,.625),(0,.125,.375,.4375,.5625),
    (1,.3125,.6875,.3125,.6875),(1,.6875,.9375,.4375,.5625),
    (0,0.,.125,.4375,.5625),(1,.9375,1.,.4375,.5625)]

function mim_library_layout(cells;sheet_resistance=0.)
    grid=CellGrid(.001,.001,cells,cells)
    stack=PlanarStackup([PlanarLayer(4.,1.,.0001),PlanarLayer(7.5,1.,1e-6),
        PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
    cap=planar_transform(planar_mim_capacitor(width=.25e-3,length=.25e-3,
        upper_level=2,lower_level=1,upper_metal="film",overhang=.0625e-3,
        lead_width=.125e-3,lead_length=.25e-3,net_top="top",net_bottom="bottom");
        offset=(.5e-3,.5e-3))
    left=planar_transform(planar_line(length=.125e-3,width=.125e-3,
        level=2,metal="film",name="left_lead",net="top");offset=(0.,.5e-3))
    right=planar_transform(planar_line(length=.0625e-3,width=.125e-3,
        level=1,metal="film",name="right_lead",net="bottom");offset=(.9375e-3,.5e-3))
    return build_planar_layout(stack,grid,[cap,left,right],
        [planar_pin(left,"p1"),planar_pin(right,"p2")];metals=Dict("film"=>sheet_resistance))
end

function mim_literal_native(cells,frequency;sheet_resistance=0.)
    io=IOBuffer();material=sheet_resistance==0 ? -1 : 0
    println(io,"FTYP SONPROJ 19\nDIM\nANG DEG\nCAP PF\nCON /OH\nFREQ GHZ\nIND NH\nLNG MM\nRES OH\nEND DIM\nCONTROL\nVARSWP\nOPTIONS\nSPEED 0\nSUBSPLAM N 100\nEND CONTROL\nGEO")
    println(io,"TMET \"PEC\" 0 SUP 0 0 0 0\nBMET \"PEC\" 0 SUP 0 0 0 0")
    sheet_resistance==0 || println(io,"MET \"Film\" 1 RES $(sheet_resistance)")
    println(io,"BOX 2 1 1 $(2cells) $(2cells) 100 0\n.1 1 1 0 0 0 2 \"Air\"\n.001 7.5 1 0 0 0 2 \"Insulator\"\n.1 4 1 0 0 0 2 \"Substrate\"")
    println(io,"POR1 BOX\nPOLY 5 1\n3\n1 50 0 0 0 0 .5\nPOR1 BOX\nPOLY 6 1\n1\n2 50 0 0 0 1 .5\nNUM 6")
    for (id,(level,x0,x1,y0,y1)) in enumerate(MIM_NATIVE_RECTANGLES)
        println(io,"$(level) 5 $(material) N $(id) 1 1 100 100 0 0 0 Y\n$(x0) $(y0)\n$(x1) $(y0)\n$(x1) $(y1)\n$(x0) $(y1)\n$(x0) $(y0)\nEND")
    end
    println(io,"END GEO\nVARSWP\nENABLED Y\nFREQ Y AN SWEEP $(frequency/1e9)\nEND\nEND VARSWP\nFILEOUT\nTOUCH ND Y native_raw.s2p IC 15 S RI R 50\nFOLDER .\nEND FILEOUT")
    return String(take!(io))
end

function mim_literal_masks(cells)
    return [BitMatrix([any(level==2-interface && x0<=(i-.5)/cells<=x1 &&
        y0<=(j-.5)/cells<=y1 for (level,x0,x1,y0,y1) in MIM_NATIVE_RECTANGLES)
        for i in 1:cells,j in 1:cells]) for interface in (1,2)]
end

function mim_original_residual(problem,result)
    raw=result isa PlanarContractedResult ? result.raw : result
    rhs=zeros(ComplexF64,size(raw.currents))
    for basis in eachindex(problem.basis.port)
        port=problem.basis.port[basis];port==0 && continue
        rhs[basis,port]=(port==1 ? -1. : 1.)*problem.basis.width[basis]
    end
    errors=raw.z_mom*raw.currents-rhs
    return maximum(norm(view(errors,:,port))/norm(view(rhs,:,port)) for port in 1:2)
end
