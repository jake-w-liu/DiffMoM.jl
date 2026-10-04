using DiffMoM,Test,LinearAlgebra

function legacy_gerber_text(body)
    mktempdir() do dir
        file=joinpath(dir,"legacy.gbr")
        write(file,"%FSLAX36Y36*%\n%MOMM*%\n"*body*"\nM02*\n")
        return read_gerber(file)
    end
end
function legacy_gerber_member(doc,x,y)
    result=false
    for o in doc.objects
        DiffMoM._artwork_contains(o.shape,x*1e-3,y*1e-3) && (result=o.dark)
    end
    return result
end
const LEGACY_OFFSET_AP="%AMoffset*1,1,0.2,0.2,0.1*%\n%ADD10offset*%\n"

@testset "Gerber legacy coordinates preserve apertures and device-only AS" begin
    for (command,center) in (("%MIA1*%",(-.8,2.1)),("%SFA2B3*%",(2.2,6.1)),
            ("%OFA.4B-.2*%",(1.6,1.9)),("%IR90*%",(-2.1,1.2)),
            ("%ASAYBX*%",(1.2,2.1)),("%ASAXBY*%",(1.2,2.1)))
        doc=legacy_gerber_text(command*"\n"*LEGACY_OFFSET_AP*"D10*X1000000Y2000000D03*")
        for dx in (-.15,-.05,0.,.05,.15),dy in (-.15,-.05,0.,.05,.15)
            @test legacy_gerber_member(doc,center[1]+dx,center[2]+dy)==(hypot(dx,dy)<=.1)
        end
    end
    # Required MI,SF,OF,IR execution order is independent of record order.
    commands=["%MIA1*%","%SFA2B3*%","%OFA.4B-.2*%","%IR90*%"]
    for a in 1:4,b in 1:4,c in 1:4,d in 1:4
        length(Set((a,b,c,d)))==4 || continue
        doc=legacy_gerber_text(join(commands[[a,b,c,d]],"\n")*"\n"*LEGACY_OFFSET_AP*"D10*X1000000Y2000000D03*")
        @test legacy_gerber_member(doc,-5.9,-1.4)
        @test !legacy_gerber_member(doc,-5.9,-1.55)
    end
    sr=legacy_gerber_text(join(commands,"\n")*"\n"*LEGACY_OFFSET_AP*
        "%SRX2Y2I.7J.4*%\nD10*X1000000Y2000000D03*\n%SR*%")
    @test length(sr.objects)==4
    for j in 0:1,i in 0:1
        @test legacy_gerber_member(sr,-5.9-j*.4,-1.4+i*.7)
        @test !legacy_gerber_member(sr,-5.9-j*.4-.15,-1.4+i*.7)
    end
    # Aperture block coordinates are aperture geometry, not image coordinates.
    ab=legacy_gerber_text(join(commands,"\n")*"\n%ADD10C,.2*%\n%ABD20*%\n"*
        "D10*X200000Y100000D03*\n%AB*%\nD20*X1000000Y2000000D03*")
    @test legacy_gerber_member(ab,-5.9,-1.4)
    @test !legacy_gerber_member(ab,-5.9,-1.55)
    for invalid in ("%MIAX*%","%MIA2*%","%MI*%\n%MI*%","%IR45*%","%IR90*%\n%IR90*%",
            "%ASBYAX*%","%SFA0*%","%SFA.00001*%","%SFA1000*%","%SFA-1*%",
            "%SFA1.000001*%","%OFA100000*%","%OFA1.000001*%")
        @test_throws ArgumentError legacy_gerber_text(invalid*"\n%ADD10C,.2*%\nD10*X0Y0D03*")
    end
    @test_throws ArgumentError legacy_gerber_text("%ADD10C,.2*%\nD10*X0Y0D02*\n%IR90*%\nX1000000Y0D03*")
end

function legacy_arc_body(aperture;full=true,clockwise=false,rectangle=false)
    a=clockwise && !full ? "X0Y1000000" : "X1000000Y0"
    b=full ? a : clockwise ? "X1000000Y0" : "X0Y1000000"
    ij=clockwise && !full ? "I0J-1000000" : "I-1000000J0"
    code=clockwise ? "G02" : "G03"
    return "%SFA2B.5*%\n%ADD10"*aperture*"*%\nD10*"*a*"D02*\nG75*"*code*b*ij*"D01*"
end

@testset "nonuniform Gerber circle strokes use analytic ellipse extrema" begin
    samples=8193
    for full in (true,false),clockwise in (true,false)
        doc=legacy_gerber_text(legacy_arc_body("C,.2";full,clockwise))
        stop=full ? 2pi : pi/2;angles=range(0,stop;length=samples)
        xs=2cos.(angles);ys=.5sin.(angles)
        # Independent witness/lower-bound oracle: sampled distance is an
        # upper bound; speed*half-step bounds the unsampled distance change.
        bound=2stop/(2(samples-1));checked=0
        for x in range(-2.25,2.25;length=29),y in range(-.75,.75;length=27)
            distance=minimum(hypot(x-xs[k],y-ys[k]) for k in eachindex(xs))
            (distance<=.1 || distance>.1+bound) || continue
            @test legacy_gerber_member(doc,x,y)==(distance<=.1)
            checked+=1
        end
        @test checked>750
        for angle in range(.05,(full ? 2pi : pi/2)-.05;length=17)
            px,py=2cos(angle),.5sin(angle)
            nx,ny=cos(angle)/2,sin(angle)/.5;normn=hypot(nx,ny);nx/=normn;ny/=normn
            @test legacy_gerber_member(doc,px+(.1-1e-8)*nx,py+(.1-1e-8)*ny)
            @test !legacy_gerber_member(doc,px+(.1+1e-8)*nx,py+(.1+1e-8)*ny)
        end
    end
end

@testset "nonuniform rectangle ellipse strokes and affine regions" begin
    for full in (true,false),clockwise in (true,false)
        doc=legacy_gerber_text(legacy_arc_body("R,.2X.4";full,clockwise))
        for x in range(-2.3,2.3;length=43),y in range(-.8,.8;length=37)
            loX,hiX=x-.1,x+.1;loY,hiY=y-.2,y+.2
            if !full
                loX=max(loX,0.);hiX=min(hiX,2.);loY=max(loY,0.);hiY=min(hiY,.5)
            end
            if loX>hiX || loY>hiY
                expected=false
            else
                nearestX=clamp(0.,loX,hiX);nearestY=clamp(0.,loY,hiY)
                minimumq=(nearestX/2)^2+(nearestY/.5)^2
                maximumq=(max(abs(loX),abs(hiX))/2)^2+(max(abs(loY),abs(hiY))/.5)^2
                min(abs(minimumq-1),abs(maximumq-1))<1e-12 && continue
                expected=minimumq<=1<=maximumq
            end
            @test legacy_gerber_member(doc,x,y)==expected
        end
        region=legacy_gerber_text("%MIA1*%\n%SFA2B.5*%\nG36*X1000000Y0D02*\nG75*"*
            (clockwise ? "G02" : "G03")*"X1000000Y0I-1000000J0D01*G37*")
        for x in range(-2.2,2.2;length=33),y in range(-.7,.7;length=29)
            q=(x/2)^2+(y/.5)^2;abs(q-1)<1e-12 && continue
            @test legacy_gerber_member(region,x,y)==(q<=1)
        end
    end
end

function legacy_ellipse_mask_allocation(doc,grid)
    artwork_cell_masks(doc,grid;offset=(.0023,.0008))
    return @allocated artwork_cell_masks(doc,grid;offset=(.0023,.0008))
end
@testset "ellipse raster scratch is independent of grid cells" begin
    doc=legacy_gerber_text(legacy_arc_body("C,.2"))
    bytes=legacy_ellipse_mask_allocation(doc,CellGrid(.0046,.0016,80,80))
    @test bytes<100_000
end
