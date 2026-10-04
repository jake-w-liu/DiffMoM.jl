module ProductionODBCompositeSweepTests
using DiffMoM,Test,LinearAlgebra,Printf
import Tar: Tar
import CodecZlib: GzipCompressorStream
const DM=DiffMoM
const CM=DiffMoM
function imported_line(aperture,start,stop;max_stroke_boundaries=256,polarity="P")
    mktempdir() do directory
        path=joinpath(directory,"features")
        coordinates=join((string(1000v) for v in (start...,stop...))," ")
        write(path,"UNITS=MM\n\$0 original_aperture\nL "*coordinates*" 0 "*polarity*" 0\n")
        only(read_odb_features(path;symbol_resolver=name->aperture,max_stroke_boundaries).objects).shape
    end
end
const CP=(;Line=imported_line,contains=DM._artwork_contains)
# Independent ordered Cartesian interval oracle, without polygon edges.
function rectangle_oracle(parts,start,stop,x,y)
    intervals=Tuple{Bool,Float64,Float64}[];cuts=[0.,1.]
    for (dark,(x0,x1,y0,y1)) in parts
        lo,hi=0.,1.
        for (q,d,lower,upper) in ((x-start[1],stop[1]-start[1],x0,x1),
                                 (y-start[2],stop[2]-start[2],y0,y1))
            if iszero(d)
                lower<=q<=upper || (lo=1.;hi=0.;break)
            else
                aa,bb=(q-lower)/d,(q-upper)/d
                lo=max(lo,min(aa,bb));hi=min(hi,max(aa,bb))
            end
        end
        if lo<=hi
            push!(intervals,(dark,lo,hi));push!(cuts,lo,hi)
        end
    end
    sort!(unique!(cuts))
    candidates=vcat(cuts,[(cuts[i]+cuts[i+1])/2 for i in 1:length(cuts)-1])
    any(candidates) do t
        state=false
        for (dark,lo,hi) in intervals
            lo<=t<=hi && (state=dark)
        end
        state
    end
end

function main()
    configurations=(
        [(true,(-1.,1.,-1.,1.)),(false,(-.5,.5,-.5,.5))],
        [(true,(-1.,1.,-1.,1.)),(false,(-.5,.5,-.5,.5)),(true,(-.1,.1,-.1,.1))],
        [(true,(-1.,1.,-1.,1.)),(false,(-.5,.5,-.5,.5)),(true,(-.75,-.25,-.75,-.25))])
    checks=0
    @testset "Original ordered composite line sweep versus interval oracle" begin
        for parts in configurations,stop in ((0.,0.),(.1,0.),(0.,.3),(.5,.5),(-.5,.5),(2.,0.),(0.,2.),(3.,-2.))
            shapes=Tuple{Bool,DM._ArtworkShape}[]
            for (dark,(x0,x1,y0,y1)) in parts
                push!(shapes,(dark,DM._artwork_polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])))
            end
            stroke=CP.Line(DM._ArtworkComposite(shapes),(.125,-.375),stop)
            for x in range(-2.013,4.013;length=57),y in range(-3.017,3.017;length=55)
                @test CP.contains(stroke,x,y)==rectangle_oracle(parts,stroke.start,stop,x,y)
                checks+=1
            end
            CP.contains(stroke,.131,.227)
            @test (@allocated CP.contains(stroke,.131,.227))==0
        end
    end
    @testset "Concentric clear-hole circle sweep versus distance extrema" begin
        aperture=DM._ArtworkComposite(Tuple{Bool,DM._ArtworkShape}[
            (true,DM._ArtworkCircle((0.,0.),1.)),(false,DM._ArtworkCircle((0.,0.),.5))])
        for stop in ((0.,0.),(.1,0.),(.5,.2),(2.,0.),(3.,-2.))
            stroke=CP.Line(aperture,(0.,0.),stop)
            for x in range(-2.013,4.013;length=61),y in range(-3.017,3.017;length=59)
            dx,dy=stop;dd=dx*dx+dy*dy
            t=iszero(dd) ? 0. : clamp((x*dx+y*dy)/dd,0.,1.)
            minimum_radius=hypot(x-t*dx,y-t*dy)
            maximum_radius=max(hypot(x,y),hypot(x-dx,y-dy))
            @test CP.contains(stroke,x,y)==(minimum_radius<=1 && maximum_radius>.5)
            checks+=1
            end
        end
    end
end
main()
function segment_distance(x,y,a,b)
    dx,dy=b[1]-a[1],b[2]-a[2];norm2=dx*dx+dy*dy
    t=iszero(norm2) ? 0. : clamp(((x-a[1])*dx+(y-a[2])*dy)/norm2,0.,1.)
    hypot(x-a[1]-t*dx,y-a[2]-t*dy)
end
function capsule_sweep_oracle(x,y,a,b,path,radius)
    # The centerline segment plus the translation segment is a convex
    # parallelogram. Dilation by a disk is distance <= radius from it.
    p=[a,b,(b[1]+path[1],b[2]+path[2]),(a[1]+path[1],a[2]+path[2])]
    cross=[(p[mod1(i+1,4)][1]-p[i][1])*(y-p[i][2])-
        (p[mod1(i+1,4)][2]-p[i][2])*(x-p[i][1]) for i in 1:4]
    area=(b[1]-a[1])*path[2]-(b[2]-a[2])*path[1]
    inside=!iszero(area) && (all(>=(0),cross)||all(<=(0),cross))
    inside || minimum(segment_distance(x,y,p[i],p[mod1(i+1,4)]) for i in 1:4)<=radius
end
function typed_allocations(stroke::S,x,y) where S
    CP.contains(stroke,x,y)
    @allocated CP.contains(stroke,x,y)
end
function harden()
    rows=Dict{String,Any}[]
    @testset "Capsule aperture sweep versus convex distance oracle" begin
        a=(-.4,-.2);b=(.5,.3);radius=.17
        aperture=CM._ArtworkRoundLine(a,b,radius)
        for stop in ((0.,0.),(.1,0.),(0.,.3),(.5,.5),(-.5,.5),(2.,0.),(0.,2.),(3.,-2.))
            shape=CP.Line(aperture,(0.,0.),stop)
            for x in range(-1.017,4.013;length=49),y in range(-3.013,3.017;length=47)
                @test CP.contains(shape,x,y)==capsule_sweep_oracle(x,y,a,b,stop,radius)
            end
            @test typed_allocations(shape,.03,.09)==0
        end
    end
    @testset "Curved contour clear hole follows original region equations" begin
        function disk(r)
            CM._ArtworkRegion(Union{CM._ArtworkRoundLine,CM._ArtworkRoundArc}[
                CM._ArtworkRoundArc((r,0.),(r,0.),(0.,0.),false,0.)])
        end
        aperture=CM._ArtworkComposite(Tuple{Bool,CM._ArtworkShape}[(true,disk(1.)),(false,disk(.5))])
        for stop in ((0.,0.),(.1,0.),(.5,.2),(2.,0.),(3.,-2.))
            shape=CP.Line(aperture,(0.,0.),stop)
            for x in range(-2.013,4.013;length=49),y in range(-3.017,3.017;length=47)
            dx,dy=stop;dd=dx*dx+dy*dy
            t=iszero(dd) ? 0. : clamp((x*dx+y*dy)/dd,0.,1.)
            @test CP.contains(shape,x,y)==(hypot(x-t*dx,y-t*dy)<=1 &&
                max(hypot(x,y),hypot(x-dx,y-dy))>.5)
            end
        end
    end
    @testset "Affine and nested ordered composites preserve the same sweep" begin
        aperture=CM._ArtworkComposite(Tuple{Bool,CM._ArtworkShape}[
            (true,CM._artwork_rectangle(2.,2.)),(false,CM._artwork_rectangle(1.,1.)),
            (true,CM._ArtworkComposite(Tuple{Bool,CM._ArtworkShape}[
                (true,CM._ArtworkCircle((0.,0.),.2)),(false,CM._ArtworkCircle((0.,0.),.1))]))])
        start=(.125,-.375);stop=(.7,.3)
        base=CP.Line(aperture,start,stop)
        for (sx,sy,angle) in ((1.,1.,37.),(-1.,1.,-23.),(1.7,.8,0.))
            rotate=CM._artwork_transform(aperture;rotation=angle,mirror=(sign(sx),sign(sy)))
            m=CM.SMatrix{2,2,Float64}(rotate.matrix*Diagonal([abs(sx),abs(sy)]))
            transformed=CM._ArtworkTransform(aperture,m,inv(m),(0.,0.))
            aa=m*CM.SVector(start...);bb=m*CM.SVector(stop...)
            shape=CP.Line(transformed,(aa[1],aa[2]),(bb[1],bb[2]))
            for x in range(-1.017,2.013;length=43),y in range(-1.013,1.017;length=41)
                q=m*CM.SVector(x,y)
                @test CP.contains(shape,q[1],q[2])==CP.contains(base,x,y)
            end
            @test typed_allocations(shape,.03,.09)==0
        end
        grid=CellGrid(4.,4.,400,400)
        object=PlanarArtworkObject("m",base,true,CM._artwork_bounds(base),Dict{String,Vector{String}}())
        doc=PlanarArtwork("prototype",[object],Dict{String,Vector{String}}(),Set{String}(),1.)
        artwork_cell_masks(doc,grid;offset=(1.5,1.5))
        bytes=@allocated artwork_cell_masks(doc,grid;offset=(1.5,1.5))
        @test bytes<100000
        push!(rows,Dict("scope"=>"nested six-boundary composite 400x400 raster",
            "warmed_cumulative_allocation_bytes"=>bytes))
    end
end
harden()

@testset "Public composite stroke work limits and original local polarity" begin
    annulus=DM._ArtworkComposite(Tuple{Bool,DM._ArtworkShape}[
        (true,DM._ArtworkCircle((0.,0.),1.)),(false,DM._ArtworkCircle((0.,0.),.5))])
    @test_throws ArgumentError imported_line(annulus,(0.,0.),(.1,0.);max_stroke_boundaries=4)
    shape=imported_line(annulus,(0.,0.),(.1,0.);max_stroke_boundaries=5)
    @test shape.boundary_count==5
    @test !DM._artwork_contains(shape,0.,0.)
    @test DM._artwork_contains(shape,.8,0.)
    @test_throws ArgumentError imported_line(annulus,(0.,0.),(.1,0.);max_stroke_boundaries=-1)
    empty=DM._ArtworkComposite(Tuple{Bool,DM._ArtworkShape}[])
    @test_throws ArgumentError imported_line(empty,(0.,0.),(.1,0.);max_stroke_boundaries=0)
    @test !DM._artwork_contains(imported_line(empty,(0.,0.),(.1,0.);max_stroke_boundaries=1),0.,0.)
    # A cyclic user callback geometry must fail through the explicit depth
    # bound before recursive bounds traversal or query evaluation.
    parts=Tuple{Bool,DM._ArtworkShape}[];cycle=DM._ArtworkComposite(parts)
    push!(parts,(true,cycle))
    @test_throws ArgumentError imported_line(cycle,(0.,0.),(.1,0.))
    mktempdir() do directory
        path=joinpath(directory,"features")
        write(path,"UNITS=MM\n\$0 frame\nA 1 0 0 1 0 0 0 P 0 N\n")
        @test_throws ArgumentError read_odb_features(path;symbol_resolver=name->annulus)
    end
end

@testset "Enclosing ODB product user symbol retains swept holes and islands" begin
    mktempdir() do directory
        mkpath(joinpath(directory,"matrix"));write(joinpath(directory,"matrix","matrix"),
            "STEP {\nCOL=1\nNAME=board\n}\nLAYER {\nROW=1\nNAME=metal\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}\n")
        step=joinpath(directory,"steps","board");mkpath(joinpath(step,"layers","metal"))
        write(joinpath(step,"stephdr"),"UNITS=MM\n")
        write(joinpath(step,"layers","metal","features"),
            "UNITS=MM\n\$0 framed_island\nL .4375 .5 .5625 .5 0 P 0\n")
        symbol=joinpath(directory,"symbols","framed_island");mkpath(symbol)
        write(joinpath(symbol,"features"),"UNITS=MM\n\$0 rect62.5x125\n"*
            "S P 0\nOB -.1875 -.25 I\nOS .1875 -.25\nOS .1875 .25\nOS -.1875 .25\nOS -.1875 -.25\nOE\n"*
            "OB -.125 -.125 H\nOS .125 -.125\nOS .125 .125\nOS -.125 .125\nOS -.125 -.125\nOE\nSE\nP 0 0 0 P 0 0\n")
        document=read_odb(directory;step="board",layers=["metal"],max_stroke_boundaries=27)
        shape=only(document.objects).shape
        for x in range(.013,.987;length=53),y in range(.017,.983;length=51)
            expected=(.25<=x<=.75 && .25<=y<=.75 && !(.4375<x<.5625 && .375<y<.625)) ||
                (.40625<=x<=.59375 && .4375<=y<=.5625)
            @test DM._artwork_contains(shape,x*1e-3,y*1e-3)==expected
        end
        @test_throws ArgumentError read_odb(directory;step="board",layers=["metal"],max_stroke_boundaries=26)
        @test_throws ArgumentError read_odb(directory;step="board",layers=["metal"],max_bytes=1024)
        mktempdir() do archives
            tarball=joinpath(archives,"product.tar");Tar.create(directory,tarball)
            archived=read_odb(tarball;step="board",layers=["metal"],max_stroke_boundaries=27)
            @test only(archived.objects).bounds==only(document.objects).bounds
            @test_throws ArgumentError read_odb(tarball;step="board",layers=["metal"],max_stroke_boundaries=26)
        end
    end
end

@testset "Public circle strokes preserve a literal local aperture center" begin
    center=(.2e-3,.1e-3);circle=DM._ArtworkCircle(center,.05e-3)
    shape=imported_line(circle,(0.,0.),(.1e-3,0.))
    @test maximum(abs.(collect(DM._artwork_bounds(shape)).-[.15e-3,.35e-3,.05e-3,.15e-3]))<1e-18
    for x in range(-.013,.513;length=59),y in range(-.017,.313;length=57)
        expected=segment_distance(x*1e-3,y*1e-3,center,(.3e-3,.1e-3))<=.05e-3
        @test DM._artwork_contains(shape,x*1e-3,y*1e-3)==expected
    end
    mktempdir() do directory
        for clockwise in (true,false)
            record=clockwise ? "A 0 .1 .1 0 0 0 0 P 0 Y" : "A .1 0 0 .1 0 0 0 P 0 N"
            path=joinpath(directory,"features");write(path,"UNITS=MM\n\$0 offset\n"*record*"\n")
            shape=only(read_odb_features(path;symbol_resolver=name->circle).objects).shape
            @test maximum(abs.(collect(DM._artwork_bounds(shape)).-[.15e-3,.35e-3,.05e-3,.25e-3]))<1e-18
            for x in range(-.013,.513;length=59),y in range(-.017,.313;length=57)
                dx,dy=x*1e-3-center[1],y*1e-3-center[2]
                distance=dx>=0 && dy>=0 ? abs(hypot(dx,dy)-.1e-3) :
                    min(hypot(dx-.1e-3,dy),hypot(dx,dy-.1e-3))
                @test DM._artwork_contains(shape,x*1e-3,y*1e-3)==(distance<=.05e-3)
            end
        end
    end
    for circle in (DM._ArtworkCircle((NaN,0.),1.),DM._ArtworkCircle((0.,0.),-1.))
        @test_throws ArgumentError imported_line(circle,(0.,0.),(.1,0.))
    end
end
function quarter_arc_segment_distance(a,b,radius)
    # Independently minimize distance between a finite segment and a
    # quarter-circle centerline: endpoints, circle crossings and the two
    # circle normals to the supporting line cover the stationary cases.
    distance=min(segment_distance(radius,0.,a,b),segment_distance(0.,radius,a,b))
    for (x,y) in (a,b)
        d=x>=0 && y>=0 ? abs(hypot(x,y)-radius) : min(hypot(x-radius,y),hypot(x,y-radius))
        distance=min(distance,d)
    end
    dx,dy=b[1]-a[1],b[2]-a[2];length=hypot(dx,dy)
    iszero(length) && return distance
    for sign in (-1.,1.)
        x,y=sign*radius*(-dy/length),sign*radius*(dx/length)
        x>=0 && y>=0 && (distance=min(distance,segment_distance(x,y,a,b)))
    end
    A=dx*dx+dy*dy;B=2(a[1]*dx+a[2]*dy);C=a[1]^2+a[2]^2-radius^2
    discriminant=B^2-4A*C
    if discriminant>=0
        for t in ((-B-sqrt(discriminant))/(2A),(-B+sqrt(discriminant))/(2A))
            0<=t<=1 && a[1]+t*dx>=0 && a[2]+t*dy>=0 && return 0.
        end
    end
    distance
end
@testset "User symbol circular and polygon stroke leaves" begin
    for clockwise in (true,false),stop in ((0.,0.),(.17,0.),(.3,-.2),(-.4,.7))
        arc=DM._ArtworkRoundArc(clockwise ? (0.,1.) : (1.,0.),clockwise ? (1.,0.) : (0.,1.),(0.,0.),clockwise,.13)
        shape=imported_line(arc,(0.,0.),stop)
        for x in range(-.717,1.913;length=47),y in range(-.413,2.017;length=43)
            expected=quarter_arc_segment_distance((x,y),(x-stop[1],y-stop[2]),1.)<=.13
            @test DM._artwork_contains(shape,x,y)==expected
        end
        @test typed_allocations(shape,.3,.7)==0
    end
    for stop in ((0.,0.),(.3,-.1),(-.3,.7))
        aperture=DM._ArtworkPolygonLine(DM._artwork_rectangle(2.,2.),(0.,0.),(.4,.2))
        shape=imported_line(aperture,(0.,0.),stop)
        generators=((1.,0.),(0.,1.),(.2,.1),(stop[1]/2,stop[2]/2))
        center=(.2+stop[1]/2,.1+stop[2]/2)
        for x in range(-1.713,2.017;length=47),y in range(-1.417,2.113;length=43)
            expected=all(generators) do (gx,gy)
                nx,ny=-gy,gx
                abs((x-center[1])*nx+(y-center[2])*ny)<=sum(abs(vx*nx+vy*ny) for (vx,vy) in generators)
            end
            @test DM._artwork_contains(shape,x,y)==expected
        end
        @test typed_allocations(shape,.3,.7)==0
    end
end
@testset "ODB rounded and chamfered rectangle sweep leaves" begin
    rounded=DM._odb_corner_rectangle(2.,1.,.5)
    for stop in ((0.,0.),(.3,-.1),(-.3,.7))
        shape=imported_line(rounded,(0.,0.),stop)
        for x in range(-1.713,2.017;length=47),y in range(-1.417,2.113;length=43)
            @test DM._artwork_contains(shape,x,y)==capsule_sweep_oracle(x,y,(-.5,0.),(.5,0.),stop,.5)
        end
        @test typed_allocations(shape,.3,.7)==0
    end
    diamond=DM._odb_corner_rectangle(2.,2.,1.;chamfer=true)
    for stop in ((0.,0.),(.3,-.1),(-.3,.7))
        shape=imported_line(diamond,(0.,0.),stop)
        for x in range(-1.713,2.017;length=47),y in range(-1.417,2.113;length=43)
            expected=all(((1.,1.),(1.,-1.),(-stop[2],stop[1]))) do (nx,ny)
                support=max(abs(nx),abs(ny))
                lo,hi=minmax(0.,stop[1]*nx+stop[2]*ny)
                lo-support<=x*nx+y*ny<=hi+support
            end
            @test DM._artwork_contains(shape,x,y)==expected
        end
        @test typed_allocations(shape,.3,.7)==0
    end
end
@testset "Composite strokes retain fixed-decimal scale covariance" begin
    parts=[(true,(-1.,1.,-1.,1.)),(false,(-.5,.5,-.5,.5)),(true,(-.1,.1,-.1,.1))]
    start=(.125,-.375);stop=(.7,.3)
    for scale in (1e-200,1e-160,1e-100,1.,1e100,1e160,1e200)
        shapes=Tuple{Bool,DM._ArtworkShape}[]
        for (dark,(x0,x1,y0,y1)) in parts
            push!(shapes,(dark,DM._artwork_polygon([(x0*scale,y0*scale),(x1*scale,y0*scale),
                (x1*scale,y1*scale),(x0*scale,y1*scale)])))
        end
        shape=mktempdir() do directory
            aperture=DM._ArtworkComposite(shapes);path=joinpath(directory,"features")
            coordinates=join((@sprintf("%.240f",1000scale*v) for v in (start...,stop...))," ")
            write(path,"UNITS=MM\n\$0 aperture\nL "*coordinates*" 0 P 0\n")
            only(read_odb_features(path;symbol_resolver=name->aperture).objects).shape
        end
        for x in range(-1.013,2.017;length=43),y in range(-1.017,1.013;length=41)
            @test DM._artwork_contains(shape,x*scale,y*scale)==rectangle_oracle(parts,start,stop,x,y)
        end
        @test typed_allocations(shape,.13scale,.227scale)==0
    end
end
@testset "Compressed composite features and clear image operations" begin
    annulus=DM._ArtworkComposite(Tuple{Bool,DM._ArtworkShape}[
        (true,DM._ArtworkCircle((0.,0.),1.)),(false,DM._ArtworkCircle((0.,0.),.5))])
    mktempdir() do directory
        path=joinpath(directory,"features.gz")
        open(path,"w") do io
            compressor=GzipCompressorStream(io)
            try
                write(compressor,"UNITS=MM\n\$0 r4000000\n\$1 annulus\nP 0 0 0 P 0 0\nL 0 0 100 0 1 N 0\n")
            finally
                close(compressor)
            end
        end
        doc=read_odb_features(path;layer="metal",symbol_resolver=name->annulus,max_stroke_boundaries=5)
        @test length(doc.objects)==2
        @test doc.source==abspath(path)
        @test doc.objects[1].dark
        @test !doc.objects[2].dark
        grid=CellGrid(4.,4.,40,40)
        masks=artwork_cell_masks(doc,grid;offset=(2.,2.))
        for i in 1:40,j in 1:40
            x,y=(i-.5)*.1-2.,(j-.5)*.1-2.
            t=clamp(x/.1,0.,1.)
            clear=hypot(x-.1t,y)<=1 && max(hypot(x,y),hypot(x-.1,y))>.5
            expected=hypot(x,y)<=2 && !clear
            @test masks["metal"][i,j]==expected
        end
        @test_throws ArgumentError read_odb_features(path;symbol_resolver=name->annulus,max_stroke_boundaries=4)
    end
end
@testset "Caller geometry snapshots preserve imported bounds and work guards" begin
    mktempdir() do directory
        vertices=[0. .125 0.;0. 0. .125]
        polygon=DM._ArtworkPolygon(vertices)
        parts=Tuple{Bool,DM._ArtworkShape}[(true,polygon),(false,DM._ArtworkEmpty())]
        aperture=DM._ArtworkComposite(parts)
        path=joinpath(directory,"features")
        for record in ("P 0 0 0 P 0 0","L 0 0 10 0 0 P 0")
            write(path,"UNITS=MM\n\$0 custom\n"*record*"\n")
            doc=read_odb_features(path;symbol_resolver=name->aperture)
            object=only(doc.objects);bounds=object.bounds
            @test !DM._artwork_contains(object.shape,1.,.01)
            vertices[1,:].+=1.
            @test !DM._artwork_contains(object.shape,1.,.01)
            @test object.bounds==bounds
            @test DM._artwork_bounds(object.shape)==bounds
            vertices[1,:].-=1.
            push!(parts,(true,aperture))
            @test !DM._artwork_contains(object.shape,1.,1.)
            pop!(parts)
        end
        cycleparts=Tuple{Bool,DM._ArtworkShape}[(true,isodd(i) ? polygon : DM._ArtworkCircle((0.,0.),.125)) for i in 1:9]
        cycle=DM._ArtworkComposite(cycleparts)
        doc=read_odb_features(path;symbol_resolver=name->cycle)
        push!(cycleparts,(true,cycle))
        @test !DM._artwork_contains(only(doc.objects).shape,1.,1.)
        @test_throws ArgumentError read_odb_features(path;symbol_resolver=name->cycle)
        # Dedicated polygon lines and flashes also own callback arrays.
        for record in ("P 0 0 0 P 0 0","L 0 0 10 0 0 P 0")
            write(path,"UNITS=MM\n\$0 custom\n"*record*"\n")
            doc=read_odb_features(path;symbol_resolver=name->polygon)
            shape=only(doc.objects).shape
            vertices[1,:].+=1.
            @test !DM._artwork_contains(shape,1.,.01)
            vertices[1,:].-=1.
        end
        empty=DM._ArtworkComposite(Tuple{Bool,DM._ArtworkShape}[(true,DM._ArtworkEmpty()) for _ in 1:10000])
        @test_throws ArgumentError read_odb_features(path;symbol_resolver=name->empty,max_stroke_boundaries=1)
        # Reject a large callback payload before making its owned copy.
        big=DM._ArtworkPolygon(repeat(vertices,1,10000))
        reject()=try
            read_odb_features(path;symbol_resolver=name->big,max_bytes=20000)
            false
        catch e
            e isa ArgumentError || rethrow()
            true
        end
        @test reject()
        @test (@allocated reject())<80000
    end
end
end
