# Shared construction for the native proof and its package replay.
# Native rectangles are independently declared in mm; library shapes use SI.
using DiffMoM,LinearAlgebra
const IDC_NATIVE_RECTANGLES=[
    (.25,.3125,.3125,.75),(.6875,.75,.3125,.75),
    (.3125,.625,.3125,.375),(.375,.6875,.4375,.5),
    (.3125,.625,.5625,.625),(.375,.6875,.6875,.75),
    (0.,.3125,.4375,.5625),(.6875,1.,.4375,.5625)]

function idc_library_layout(cells;sheet_resistance=0.)
    grid=CellGrid(.001,.001,cells,cells)
    stack=PlanarStackup([PlanarLayer(4.,1.,.0001),PlanarLayer(1.,1.,.0001)],
        TERM_GND,TERM_GND,.001,.001)
    idc=planar_transform(planar_interdigital_capacitor(fingers=2,
        finger_length=.3125e-3,width=.0625e-3,gap=.0625e-3,
        bus_width=.0625e-3,level=1,metal="film",net1="left",net2="right");
        offset=(.25e-3,.53125e-3))
    left=planar_transform(planar_line(length=.3125e-3,width=.125e-3,
        level=1,metal="film",name="left_lead",net="left");offset=(0.,.5e-3))
    right=planar_transform(planar_line(length=.3125e-3,width=.125e-3,
        level=1,metal="film",name="right_lead",net="right");offset=(.6875e-3,.5e-3))
    return build_planar_layout(stack,grid,[idc,left,right],
        [planar_pin(left,"p1"),planar_pin(right,"p2")];metals=Dict("film"=>sheet_resistance))
end

function idc_literal_native(cells,frequency;sheet_resistance=0.)
    io=IOBuffer();material=sheet_resistance==0 ? -1 : 0
    println(io,"FTYP SONPROJ 19\nDIM\nANG DEG\nCAP PF\nCON /OH\nFREQ GHZ\nIND NH\nLNG MM\nRES OH\nEND DIM\nCONTROL\nVARSWP\nOPTIONS\nSPEED 0\nSUBSPLAM N 100\nEND CONTROL\nGEO")
    println(io,"TMET \"PEC\" 0 SUP 0 0 0 0\nBMET \"PEC\" 0 SUP 0 0 0 0")
    sheet_resistance==0 || println(io,"MET \"Film\" 1 RES $(sheet_resistance)")
    println(io,"BOX 1 1 1 $(2cells) $(2cells) 100 0\n.1 1 1 0 0 0 2 \"Air\"\n.1 4 1 0 0 0 2 \"Substrate\"")
    println(io,"POR1 BOX\nPOLY 7 1\n3\n1 50 0 0 0 0 .5\nPOR1 BOX\nPOLY 8 1\n1\n2 50 0 0 0 1 .5\nNUM 8")
    for (id,(x0,x1,y0,y1)) in enumerate(IDC_NATIVE_RECTANGLES)
        println(io,"0 5 $(material) N $(id) 1 1 100 100 0 0 0 Y\n$(x0) $(y0)\n$(x1) $(y0)\n$(x1) $(y1)\n$(x0) $(y1)\n$(x0) $(y0)\nEND")
    end
    println(io,"END GEO\nVARSWP\nENABLED Y\nFREQ Y AN SWEEP $(frequency/1e9)\nEND\nEND VARSWP\nFILEOUT\nTOUCH ND Y native_raw.s2p IC 15 S RI R 50\nFOLDER .\nEND FILEOUT")
    return String(take!(io))
end

function idc_original_residual(problem,result)
    raw=result isa PlanarContractedResult ? result.raw : result
    # Independent original unit-voltage sources at the two literal walls.
    rhs=zeros(ComplexF64,size(raw.currents))
    for basis in eachindex(problem.basis.port)
        port=problem.basis.port[basis];port==0 && continue
        rhs[basis,port]=(port==1 ? -1. : 1.)*problem.basis.width[basis]
    end
    errors=raw.z_mom*raw.currents-rhs
    return maximum(norm(view(errors,:,port))/norm(view(rhs,:,port)) for port in 1:2)
end
