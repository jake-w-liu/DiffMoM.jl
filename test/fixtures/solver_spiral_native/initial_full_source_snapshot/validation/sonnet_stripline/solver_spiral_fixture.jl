# Public library spiral in SI, independently specified native rectangles in mm.
using DiffMoM, LinearAlgebra

const SOLVER_SPIRAL_RECTANGLES = [
    (.3125, 1.75, .25, .375), (1.625, 1.75, .25, 1.75),
    (.25, 1.75, 1.625, 1.75), (.25, .375, .5, 1.75),
    (.25, 1.5, .5, .625), (1.375, 1.5, .5, 1.5),
    (.5, 1.5, 1.375, 1.5), (.5, .625, .8125, 1.5)]

function solver_spiral_layout(cells; sheet_resistance=0.)
    grid = CellGrid(.002, .002, cells, cells)
    stack = PlanarStackup([PlanarLayer(4., 1., .0001),
        PlanarLayer(1., 1., .0001)], TERM_GND, TERM_GND, .002, .002)
    spiral = planar_transform(planar_spiral(shape=:rectangular, turns=2.,
        width=.125e-3, spacing=.125e-3, d_out=1.5e-3,
        level=1, metal="film", net="coil"); offset=(1e-3, 1e-3))
    return build_planar_layout(stack, grid, [spiral], spiral.pins;
        metals=Dict("film"=>sheet_resistance), terminal_ground=:below)
end

function solver_spiral_masks(cells)
    return BitMatrix([any(x0 < (i-.5)*2/cells < x1 &&
        y0 < (j-.5)*2/cells < y1 for (x0,x1,y0,y1) in SOLVER_SPIRAL_RECTANGLES)
        for i in 1:cells, j in 1:cells])
end

function solver_spiral_literal_native(cells, frequency; sheet_resistance=0.)
    io=IOBuffer(); material=iszero(sheet_resistance) ? -1 : 0
    println(io,"FTYP SONPROJ 19\nDIM\nANG DEG\nCAP PF\nCON /OH\nFREQ GHZ\nIND NH\nLNG MM\nRES OH\nEND DIM\nCONTROL\nVARSWP\nOPTIONS\nSPEED 0\nSUBSPLAM N 100\nEND CONTROL\nGEO")
    println(io,"TMET \"PEC\" 0 SUP 0 0 0 0\nBMET \"PEC\" 0 SUP 0 0 0 0")
    iszero(sheet_resistance) || println(io,"MET \"Film\" 1 RES $(sheet_resistance)")
    println(io,"BOX 1 2 2 $(2cells) $(2cells) 100 0\n.1 1 1 0 0 0 2 \"Air\"\n.1 4 1 0 0 0 2 \"Substrate\"")
    println(io,"POR1 VIA\nPOLY 9 1\n0\n1 50 0 0 0 .3125 .3125\nPOR1 VIA\nPOLY 10 1\n0\n2 50 0 0 0 .5625 .8125\nNUM 10")
    for (id,(x0,x1,y0,y1)) in enumerate(SOLVER_SPIRAL_RECTANGLES)
        println(io,"0 5 $(material) N $(id) 1 1 100 100 0 0 0 Y\n$(x0) $(y0)\n$(x1) $(y0)\n$(x1) $(y1)\n$(x0) $(y1)\n$(x0) $(y0)\nEND")
    end
    # Literal physical return strips at the metal-adjacent terminal cells.
    # These exactly match the public pin workflow's one-cell-deep contacts,
    # but their dimensions are derived independently from the native grid.
    cell=2/cells
    for (id,(x0,x1,y0,y1)) in enumerate([
            (.3125,.3125+cell,.25,.375),(.5,.625,.8125,.8125+cell)])
        println(io,"VIA POLYGON\n0 5 -1 N $(id+8) 1 1 100 100 0 0 0 Y\nTOLEVEL GND RING NOCOVERS\n$(x0) $(y0)\n$(x1) $(y0)\n$(x1) $(y1)\n$(x0) $(y1)\n$(x0) $(y0)\nEND")
    end
    println(io,"END GEO\nVARSWP\nENABLED Y\nFREQ Y AN SWEEP $(frequency/1e9)\nEND\nEND VARSWP\nFILEOUT\nTOUCH ND Y native_raw.s2p IC 15 S RI R 50\nFOLDER .\nEND FILEOUT")
    return String(take!(io))
end

function solver_spiral_original_residual(layout, result)
    raw=result.raw; problem=layout.source_problem
    rhs=zeros(ComplexF64,size(result.currents))
    for b in eachindex(problem.basis.port)
        p=problem.basis.port[b]; iszero(p) && continue
        # Uniform axial voltage source: the layer's distributed E=V/h
        # integrated against h*cell_area (half that for the tapered profile).
        if problem.basis.kind[b] in (DiffMoM._BASIS_VIA_U, DiffMoM._BASIS_VIA_T)
            factor=problem.basis.kind[b] == DiffMoM._BASIS_VIA_T ? .5 : 1.
            rhs[b,:] .= -problem.grid.dx*problem.grid.dy*factor*problem.ports[p].polarity .* view(layout.contraction,p,:)
        end
    end
    error=raw.z_mom*result.currents-rhs
    # Divide every Galerkin row by its independent physical trace measure;
    # the lateral sheet and axial source measures have different SI units.
    measures=Float64[k<=3 ? problem.grid.dy : k<=6 ? problem.grid.dx :
        problem.grid.dx*problem.grid.dy*(k==8 ? .5 : 1.) for k in problem.basis.kind]
    return maximum(norm(view(error,:,p)./measures)/norm(view(rhs,:,p)./measures) for p in 1:2)
end

# Independent exact cell integrals of the retained piecewise-linear sheet J.
function solver_spiral_sheet_power(layout,result,resistance;voltages=ComplexF64[1,0])
    problem=layout.source_problem;grid=problem.grid
    coefficients=result.currents*voltages
    xedges=zeros(ComplexF64,grid.nx+1,grid.ny)
    yedges=zeros(ComplexF64,grid.nx,grid.ny+1)
    for b in eachindex(problem.basis.kind)
        k=problem.basis.kind[b]
        if k<=3
            xedges[problem.basis.ei[b]+1,problem.basis.ej[b]]+=coefficients[b]
        elseif k<=6
            yedges[problem.basis.ei[b],problem.basis.ej[b]+1]+=coefficients[b]
        end
    end
    integral=0.
    for cell in findall(only(problem.sheets).mask)
        i,j=Tuple(cell)
        xl,xr=xedges[i,j],xedges[i+1,j]
        yl,yr=yedges[i,j],yedges[i,j+1]
        integral+=abs2(xl)+abs2(xr)+real(conj(xl)*xr)+
            abs2(yl)+abs2(yr)+real(conj(yl)*yr)
    end
    dissipated=resistance*grid.dx*grid.dy/3*integral
    supplied=real(dot(voltages,result.y*voltages))
    return (;dissipated,supplied,relative_error=abs(dissipated-supplied)/supplied)
end
