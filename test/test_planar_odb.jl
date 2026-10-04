using DiffMoM,Test,LinearAlgebra
isdefined(DiffMoM,:read_odb) || Base.include(DiffMoM,joinpath(@__DIR__,"..","src","planar","PlanarODBIO.jl"))
function _test_odb_features(text;kw...)
    mktempdir() do directory
        path=joinpath(directory,"features");write(path,text);read_odb_features(path;layer="metal",kw...)
    end
end
_odb_member(doc,x,y)=begin
    occupied=false
    for object in doc.objects
        DiffMoM._artwork_contains(object.shape,x*1e-3,y*1e-3) && (occupied=object.dark)
    end
    occupied
end
@testset "ODB features exact pad units, orientation, polarity and strokes" begin
    body=raw"""
UNITS=MM
F 5
$0 r1000
$1 s1000
$2 rect1x3 I
@0 .smd
@1 .net_name
&0 signal
P 1 1 0 P 10 0;0,1=0;ID=42
P 1 1 1 N 11 0
L 2 2 4 4 1 P 0
A 7 5 5 7 5 5 0 P 0 N
P 10 0 2 P 0 9 30
"""
    doc=_test_odb_features(body)
    @test length(doc.objects)==5
    @test !_odb_member(doc,1,1)
    @test !_odb_member(doc,2.51,1.49)
    @test _odb_member(doc,2.49,1.51) # square aperture produces square end corners
    @test _odb_member(doc,3.45,3.45)
    @test _odb_member(doc,5+sqrt(2),5+sqrt(2))
    @test !_odb_member(doc,5-sqrt(2),5+sqrt(2))
    @test doc.objects[1].attributes[".smd"]==["true"]
    @test doc.objects[1].attributes[".net_name"]==["0"]
    @test doc.objects[1].attributes[".net_name.text_lookup"]==["signal"]
    @test doc.objects[1].attributes["ID"]==["42"]
    # Explicit I symbol dimensions are mils even in an MM features file;
    # ODB clockwise rotation precedes its x-coordinate mirror.
    @test _odb_member(doc,9.988,.020)
    @test !_odb_member(doc,10.012,.020)
    @test_throws ArgumentError _test_odb_features(body;max_bytes=1)
    @test_throws ArgumentError _test_odb_features(body;max_objects=2)
    @test_throws ArgumentError _test_odb_features(replace(body,"F 5"=>"F 6"))
end

@testset "ODB native curved surfaces and transparent contour holes" begin
    body=raw"""
UNITS=MM
$0 r400
P 0 0 0 P 0 0
S P 0
OB 2 0 I
OC 2 0 0 0 N
OE
OB 1 0 H
OC 1 0 0 0 N
OE
SE
"""
    doc=_test_odb_features(body)
    @test _odb_member(doc,0,0) # surface hole leaves older pad visible
    @test !_odb_member(doc,.5,0)
    @test _odb_member(doc,1.5,0)
    @test !_odb_member(doc,2.1,0)
    @test _odb_member(doc,-1.1,1.1)
    @test_throws ArgumentError _test_odb_features("UNITS=MM\nS P 0\nOB 0 0 I\nOS 1 0\nOE\nSE")
    @test_throws ArgumentError _test_odb_features("UNITS=MM\nT 0 0 standard P 0 1 1 1 'text' 1")
    @test_throws ArgumentError _test_odb_features("UNITS=MM\n\$0 strange\nP 0 0 0 P 0 0")
end

@testset "ODB standard analytic symbol families" begin
    G=DiffMoM
    for (name,inside,outside) in [("r2000",(.5,.5),(1.,1.)),("s2000",(.9,.9),(1.1,0.)),
        ("rect4000x2000",(1.9,.9),(2.1,0.)),("rect4000x2000xr500x1",(-1.9,.9),(1.9,.9)),
        ("rect4000x2000xc500x12",(1.6,.8),(1.9,.9)),("oval4000x2000",(1.5,.5),(1.9,.8)),
        ("di4000x2000",(0.,.9),(1.5,.9)),("tri4000x2000",(0.,.9),(1.5,.9)),
        ("oct4000x2000x500",(1.6,.8),(1.9,.9)),("hex_l4000x2000x500",(1.5,.9),(1.9,.9)),
        ("hex_s4000x2000x500",(1.9,.5),(1.9,.9)),("donut_r4000x2000",(1.5,0.),(.5,0.)),
        ("donut_s4000x2000",(1.5,1.5),(.5,.5)),("donut_sr4000x2000",(.9,.9),(.5,0.)),
        ("donut_rc4000x2000x200",(1.9,.5),(0.,0.)),("donut_o4000x2000x200",(1.5,.7),(0.,0.))]
        shape=G._odb_standard_symbol(name,1e-6)
        @test G._artwork_contains(shape,inside[1]*1e-3,inside[2]*1e-3)
        @test !G._artwork_contains(shape,outside[1]*1e-3,outside[2]*1e-3)
    end
    shape=G._odb_standard_symbol("rect4000x2000_90",1e-6)
    @test G._artwork_contains(shape,.0005,.0019)
    @test !G._artwork_contains(shape,.0019,.0005)
    @test_throws ArgumentError G._odb_standard_symbol("rect1000x1000xr700",1e-6)
    triangle=G._artwork_polygon([(0.,0.),(2.,0.),(0.,1.)])
    for orientation in 0:9
        angle=orientation<8 ? 90mod(orientation,4) : 37.
        c,s=cosd(angle),sind(angle)
        # Independent literal clockwise rotation followed by x inversion.
        rotated=[c s;-s c]*[.6,.2]
        orientation in (4,5,6,7,9) && (rotated[1]=-rotated[1])
        transformed=G._odb_orientation(triangle,orientation,37.,(.3,.4))
        @test G._artwork_contains(transformed,rotated[1]+.3,rotated[2]+.4)
        @test transformed.matrix*[.6,.2]≈rotated rtol=1e-14 atol=1e-14
    end
end

@testset "ODB butterfly, ellipse and drill standard symbols" begin
    G=DiffMoM
    round=G._odb_standard_symbol("bfr2000",1e-6)
    square=G._odb_standard_symbol("bfs2000",1e-6)
    ellipse=G._odb_standard_symbol("el4000x2000",1e-6)
    # Literal quadrant and conic equations are independent of the importer.
    for x in (-1.2,-.8,-.3,0.,.4,.9,1.2),y in (-1.2,-.8,-.3,0.,.4,.9,1.2)
        quarters=(x<=0 && y>=0) || (x>=0 && y<=0)
        @test G._artwork_contains(round,x*1e-3,y*1e-3)==(quarters && hypot(x,y)<=1)
        @test G._artwork_contains(square,x*1e-3,y*1e-3)==(quarters && abs(x)<=1 && abs(y)<=1)
        @test G._artwork_contains(ellipse,x*1e-3,y*1e-3)==((x/2)^2+y^2<=1)
    end
    @test !G._artwork_contains(ellipse,1.6e-3,.65e-3)
    @test G._artwork_contains(G._odb_standard_symbol("oval4000x2000",1e-6),1.6e-3,.65e-3)
    rotated=G._odb_standard_symbol("bfr2000_90",1e-6)
    @test G._artwork_contains(rotated,.0004,.0004)
    @test !G._artwork_contains(rotated,-.0004,.0004)
    @test_throws ArgumentError G._odb_standard_symbol("bfr-1",1e-6)
    @test_throws ArgumentError G._odb_standard_symbol("el0x2000",1e-6)
    @test_throws ArgumentError G._odb_standard_symbol("hole2000xpx-1x0",1e-6)
    body="UNITS=MM\n\$0 hole2000xpx100x200\n\$1 hole2000xnx100x200\n"*
        "\$2 hole2000xvx100x200\nP 3 4 0 P 0 9 31\nP 6 4 1 P 0 0\nP 9 4 2 P 0 0\n"
    drills=_test_odb_features(body)
    for (i,status) in enumerate(("plated","nonplated","via"))
        @test _odb_member(drills,3i,4.)
        @test !_odb_member(drills,3i+1.1,4.)
        attrs=drills.objects[i].attributes
        @test attrs["ODB.drill.plating"]==[status]
        @test parse(Float64,only(attrs["ODB.drill.diameter_m"]))≈.002
        @test parse(Float64,only(attrs["ODB.drill.positive_tolerance_m"]))≈.0001
        @test parse(Float64,only(attrs["ODB.drill.negative_tolerance_m"]))≈.0002
    end
    imperial=_test_odb_features("UNITS=MM\n\$0 hole2xpx.1x.2 I\nP 0 0 0 P 0 0\n")
    @test parse(Float64,only(only(imperial.objects).attributes["ODB.drill.diameter_m"]))≈.0000508
    @test parse(Float64,only(only(imperial.objects).attributes["ODB.drill.positive_tolerance_m"]))≈.00000254
    # Butterfly gaps are aperture-local transparency even for a clear pad.
    negative=_test_odb_features("UNITS=MM\n\$0 r2000\n\$1 bfr2000\n"*
        "P 0 0 0 P 0 0\nP 0 0 1 N 0 0\n")
    @test _odb_member(negative,.4,.4)
    @test !_odb_member(negative,-.4,.4)
    # Null records retain feature IDs/attributes and count without drawing.
    placeholder=_test_odb_features("UNITS=MM\nF 2\n\$0 r2000\n\$1 null37\n"*
        "P 0 0 0 P 0 0\nP 0 0 1 N 0 9 31;;ID=93\n")
    @test length(placeholder.objects)==2
    @test _odb_member(placeholder,0.,0.)
    @test only(placeholder.objects[2].attributes["ODB.null.extension"])=="37"
    @test placeholder.objects[2].attributes["ID"]==["93"]
    @test !G._artwork_contains(placeholder.objects[2].shape,0.,0.)
    @test_throws ArgumentError _test_odb_features("UNITS=MM\n\$0 null"*repeat("9",30)*"\nP 0 0 0 P 0 0\n")
end

function _test_odb_nested_contours(;polarity="P")
    io=IOBuffer()
    println(io,"S ",polarity," 0")
    # Primary ODB 8.1 Update 3 p40 / FAQ p532: each containing contour
    # precedes its child, including an island nested inside a hole.
    for (island,lo,hi) in [(true,0.,10.),(false,2.,8.),(true,3.,7.),
            (false,4.,6.),(true,4.5,5.5),(true,12.,14.),(false,12.5,13.5)]
        println(io,"OB ",lo," ",lo," ",island ? "I" : "H")
        corners=island ? [(lo,hi),(hi,hi),(hi,lo),(lo,lo)] :
            [(hi,lo),(hi,hi),(lo,hi),(lo,lo)]
        for (x,y) in corners;println(io,"OS ",x," ",y);end
        println(io,"OE")
    end
    println(io,"SE")
    return String(take!(io))
end

@testset "ODB natural containment preserves deep nested islands" begin
    body="UNITS=MM\n"*_test_odb_nested_contours()
    doc=_test_odb_features(body)
    @test length(doc.objects)==1
    @test first.(only(doc.objects).shape.parts)==[true,false,true,false,true,true,false]
    # Independent alternating rectangular-ring oracle, including a separate
    # outer island and hole. Sampling avoids contour-boundary conventions.
    oracle(x,y)=begin
        inner=0<x<10 && 0<y<10
        for (lo,hi,filled) in [(2.,8.,false),(3.,7.,true),(4.,6.,false),(4.5,5.5,true)]
            lo<x<hi && lo<y<hi && (inner=filled)
        end
        separate=12<x<14 && 12<y<14 && !(12.5<x<13.5 && 12.5<y<13.5)
        inner || separate
    end
    for x in (-1.,1.,2.5,3.5,4.25,5.,7.5,9.,11.,12.25,13.,13.75,15.),
            y in (1.,2.5,3.5,4.25,5.,7.5,9.,12.25,13.,13.75)
        @test _odb_member(doc,x,y)==oracle(x,y)
    end
    # Transparent holes do not erase a pad that was drawn earlier.
    withpad=_test_odb_features("UNITS=MM\n\$0 r400\nP 2.5 2.5 0 P 0 0\n"*_test_odb_nested_contours())
    @test _odb_member(withpad,2.5,2.5)
    @test !_odb_member(withpad,2.75,2.75)
    @test _odb_member(withpad,5.,5.)
    # A clear surface erases its filled islands; its local holes stay
    # transparent. This also protects a nested clear island from being lost.
    negative=_test_odb_features("UNITS=MM\n\$0 rect16000x16000\nP 7 7 0 P 0 0\n"*
        _test_odb_nested_contours(;polarity="N"))
    for (x,y) in [(1.,1.),(2.5,2.5),(3.5,3.5),(4.25,4.25),(5.,5.),
            (12.25,12.25),(13.,13.),(13.75,13.75)]
        @test _odb_member(negative,x,y)==!oracle(x,y)
    end
end

@testset "ODB nested circular islands and reused user symbols" begin
    curves="UNITS=MM\nS P 0\n"
    for (radius,island) in [(5.,true),(4.,false),(3.,true),(2.,false),(1.,true)]
        curves*="OB $(radius) 0 $(island ? 'I' : 'H')\n"*
            "OC $(radius) 0 0 0 $(island ? 'Y' : 'N')\nOE\n"
    end
    curves*="SE\n"
    doc=_test_odb_features(curves)
    for (radius,expected) in [(.5,true),(1.5,false),(2.5,true),(3.5,false),(4.5,true),(5.5,false)],
            angle in (0.,.37,1.29,2.83)
        @test _odb_member(doc,radius*cos(angle),radius*sin(angle))==expected
    end
    mktempdir() do directory
        function entity(parts,text)
            path=joinpath(directory,parts...);mkpath(dirname(path));write(path,text)
        end
        entity(["matrix","matrix"],"STEP {\nCOL=1\nNAME=board\n}\n"*
            "LAYER {\nROW=1\nNAME=top\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}")
        entity(["misc","info"],"UNITS=MM")
        entity(["steps","board","stephdr"],"UNITS=MM")
        entity(["symbols","nested","features"],"UNITS=MM\n"*_test_odb_nested_contours())
        entity(["steps","board","layers","top","features"],
            "UNITS=MM\n\$0 nested\nP 20 5 0 P 0 9 0\nP 40 5 0 P 0 1\n")
        reused=read_odb(directory;layers=["top"])
        @test length(reused.objects)==2
        for (localx,localy,expected) in [(1.,1.,true),(2.5,2.5,false),(3.5,3.5,true),
                (4.25,4.25,false),(5.,5.,true),(12.25,12.25,true),(13.,13.,false)]
            @test _odb_member(reused,20-localx,5+localy)==expected
            @test _odb_member(reused,40+localy,5-localx)==expected
        end
    end
end

@testset "Analytic circular contour ray crossings are orientation independent" begin
    G=DiffMoM
    for center in ((0.,0.),(2.5,-.7)),startangle in (0.,pi/2,.37),clockwise in (false,true),
            pieces in (1,4)
        angles=[startangle+(clockwise ? -1 : 1)*2pi*k/pieces for k in 0:pieces-1]
        vertices=[(center[1]+cos(a),center[2]+sin(a)) for a in angles]
        # The one-arc path is a genuine full circle; four arcs are separate
        # quarters sharing literal endpoints. Both describe the same disk.
        segments=Union{G._ArtworkRoundLine,G._ArtworkRoundArc}[
            G._ArtworkRoundArc(vertices[k],vertices[mod1(k+1,pieces)],center,clockwise,0.)
            for k in 1:pieces]
        circle=G._ArtworkRegion(segments)
        for dx in (-1.2,-.8,-.3,0.,.4,.9,1.2),dy in (-1.2,-.8,-.3,0.,.4,.9,1.2)
            @test G._artwork_contains(circle,center[1]+dx,center[2]+dy)==(hypot(dx,dy)<1)
        end
        for k in eachindex(vertices)
            @test G._artwork_contains(circle,vertices[k]...)
        end
    end
    # A semicircular cap uses an arc and a closing straight edge. This
    # protects non-full-circle endpoints from trigonometric closing noise.
    for clockwise in (false,true)
        a,b=clockwise ? ((-1.,0.),(1.,0.)) : ((1.,0.),(-1.,0.))
        cap=G._ArtworkRegion(Union{G._ArtworkRoundLine,G._ArtworkRoundArc}[
            G._ArtworkRoundArc(a,b,(0.,0.),clockwise,0.),G._ArtworkRoundLine(b,a,0.)])
        for x in (-1.2,-.7,0.,.7,1.2),y in (-.9,-.2,0.,.2,.9)
            @test G._artwork_contains(cap,x,y)==(y>=0 && hypot(x,y)<=1)
        end
    end
end

@testset "ODB product hierarchy, custom symbols and local negative layers" begin
    mktempdir() do directory
        function entity(parts,text)
            path=joinpath(directory,parts...);mkpath(dirname(path));write(path,text)
        end
        entity(["matrix","matrix"],"STEP {\nCOL=1\nNAME=board\n}\nSTEP {\nCOL=2\nNAME=panel\n}\nLAYER {\nROW=1\nNAME=top\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}\nLAYER {\nROW=2\nNAME=gnd\nTYPE=POWER_GROUND\nPOLARITY=NEGATIVE\n}")
        entity(["misc","info"],"UNITS=MM")
        entity(["steps","board","stephdr"],"UNITS=MM\nX_DATUM=1\nY_DATUM=1")
        entity(["steps","panel","stephdr"],"UNITS=MM\nSTEP-REPEAT {\nNAME=board\nX=5\nY=5\nDX=5\nDY=5\nNX=2\nNY=1\nANGLE=90\nMIRROR=YES\nFLIP=NO\n}")
        entity(["symbols","pad","features"],"UNITS=MM\n\$0 r1000\nP 0 0 0 P 0 0")
        entity(["steps","board","layers","top","features"],"UNITS=MM\n\$0 pad\nP 1 1 0 P 0 0")
        entity(["steps","board","layers","gnd","features"],"UNITS=MM\n\$0 r1000\nP 1 1 0 P 0 0")
        entity(["steps","board","profile"],"UNITS=MM\nS P 0\nOB 0 0 I\nOS 3 0\nOS 3 3\nOS 0 3\nOS 0 0\nOE\nSE")
        @test_throws ArgumentError read_odb(directory)
        doc=read_odb(directory;step="panel",layers=["top"])
        @test length(doc.objects)==2
        @test _odb_member(doc,5,5)
        @test _odb_member(doc,5,0) # rotated/mirrored repetition pitch
        @test !_odb_member(doc,10,5)
        @test haskey(doc.attributes,"ODB.matrix")
        doc=read_odb(directory;step="board",layers=["gnd"])
        @test !_odb_member(doc,1,1)
        @test _odb_member(doc,2,2)
        @test !_odb_member(doc,4,4)
        @test_throws ArgumentError read_odb(directory;step="board",layers=["missing"])
        @test_throws ArgumentError read_odb(directory;step="board",max_bytes=1)
        entity(["symbols","pad","features"],"UNITS=MM\n\$0 pad\nP 0 0 0 P 0 0")
        @test_throws ArgumentError read_odb(directory;step="board",layers=["top"])
        @test_throws ArgumentError DiffMoM._odb_name("../escape")
        for name in ("Uppercase",".hidden","-start","+start","trailing.",repeat("a",65))
            @test_throws ArgumentError DiffMoM._odb_name(name)
        end
        @test DiffMoM._odb_name("layer_1.a+2")=="layer_1.a+2"
        @test_throws ArgumentError _test_odb_features("UNITS=MM\nUNITS=MM\n")
        @test_throws ArgumentError _test_odb_features("UNITS=MM\nS P 0\nUNITS=INCH\n")
    end
end
