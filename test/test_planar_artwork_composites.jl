module ArtworkCompositeAllocationTests
using DiffMoM,Test
const DM=DiffMoM
@testset "Shared artwork literal boundaries retain geometry without allocation" begin
    for scale in (1e-100,1e-3,1.,1e100)
        polygon=DM._artwork_rectangle(2scale,2scale)
        points=[(-scale,-scale),(scale,-scale),(scale,scale),(-scale,scale)]
        region=DM._ArtworkRegion(Union{DM._ArtworkRoundLine,DM._ArtworkRoundArc}[
            DM._ArtworkRoundLine(points[i],points[mod1(i+1,4)],0.) for i in 1:4])
        for shape in (polygon,region),x in (-2.,-1.,-.7,0.,.7,1.,2.),y in (-2.,-1.,-.7,0.,.7,1.,2.)
            @test DM._artwork_contains(shape,x*scale,y*scale)==(-1<=x<=1 && -1<=y<=1)
        end
        for (x,y) in ((scale,.1scale),(scale,2scale),(-scale,.1scale),
                (.1scale,scale),(.1scale,-scale),(scale,scale))
            for shape in (polygon,region)
                member(s::S) where S=DM._artwork_contains(s,x,y)
                member(shape)
                @test (@allocated member(shape))==0
            end
        end
    end
    # General edges retain the original high-precision determinant path.
    for reverse_edge in (false,true),scale in (1e-100,1e-3,1.,1e100)
        a,b=(0.,0.),(2scale,scale)
        reverse_edge && ((a,b)=(b,a))
        for x in (-scale,0.,scale,2scale,3scale),y in (-scale,0.,.5scale,scale,2scale)
            determinant=setprecision(BigFloat,256) do
                (BigFloat(b[1])-BigFloat(a[1]))*(BigFloat(y)-BigFloat(a[2]))-
                    (BigFloat(b[2])-BigFloat(a[2]))*(BigFloat(x)-BigFloat(a[1]))
            end
            expected=iszero(determinant) && min(a[1],b[1])<=x<=max(a[1],b[1]) &&
                min(a[2],b[2])<=y<=max(a[2],b[2])
            @test DM._artwork_point_on_segment(a[1],a[2],b[1],b[2],x,y)==expected
        end
    end
    @test DM._artwork_point_on_segment(1.,2.,1.,2.,1.,2.)
    @test !DM._artwork_point_on_segment(1.,2.,1.,2.,1.,2.1)
end
struct LegacyComposite <: DM._ArtworkShape
    parts::Vector{Tuple{Bool,DM._ArtworkShape}}
end
DM._artwork_bounds(s::LegacyComposite)=DM._artwork_bounds(DM._ArtworkComposite(s.parts))
function DM._artwork_contains(s::LegacyComposite,x,y)
    inside=false
    for (dark,shape) in s.parts
        DM._artwork_contains(shape,x,y) && (inside=dark)
    end
    return inside
end
function legacy(s)
    s isa DM._ArtworkStrokeReference && return s.geometry
    s isa DM._ArtworkTransform && return DM._ArtworkTransform(legacy(s.shape),s.matrix,s.inverse,s.origin)
    s isa DM._ArtworkComposite && return LegacyComposite(Tuple{Bool,DM._ArtworkShape}[(dark,legacy(p)) for (dark,p) in s.parts])
    return s
end
function legacy(doc::PlanarArtwork)
    objects=[PlanarArtworkObject(o.layer,legacy(o.shape),o.dark,o.bounds,o.attributes) for o in doc.objects]
    return PlanarArtwork(doc.source,objects,doc.attributes,doc.negative_layers,doc.coordinate_unit_m)
end
function memberbytes(s::S) where {S<:DM._ArtworkShape}
    DM._artwork_contains(s,.0000762,.001)
    return @allocated DM._artwork_contains(s,.0000762,.001)
end
const font="XSIZE 2\nYSIZE 2\nOFFSET 1\nCHAR I\nLINE 1 0 1 2 P R .2\nECHAR\nCHAR M\nLINE 0 0 0 2 P S .2\nLINE 0 2 1 0 P S .2\nLINE 1 0 2 2 P S .2\nLINE 2 2 2 0 P S .2\nECHAR\n"
@testset "Vector text full-mask equivalence and bounded raster allocations" begin
    mktempdir() do directory
        write(joinpath(directory,"literal"),font);file=joinpath(directory,"features")
        for text in ("I","M","IM","MIMIMIMI")
            write(file,"UNITS=MM\nT 0 0 literal P 0 3 2 .5 '$(text)' 0\n")
            new=read_odb_features(file;font_directory=directory,layer="metal");old=legacy(new)
            grid=CellGrid(.003length(text)+.001,.003,400,400);offset=(.0005,.0005)
            @test artwork_cell_masks(new,grid;offset)==artwork_cell_masks(old,grid;offset)
            @test (@allocated artwork_cell_masks(new,grid;offset))<100_000
            @test memberbytes(only(new.objects).shape)==0
            # The counterfactual clone compacts the original push!-grown
            # vectors. Each stable stroke reference may add one pointer;
            # owned geometry must stay within that per-stroke bound.
            strokes=sum(c=='M' ? 4 : 1 for c in text)
            @test Base.summarysize(only(new.objects).shape)<=Base.summarysize(only(old.objects).shape)+8strokes
        end
    end
end
@testset "Typed composites retain ordered local clear strokes" begin
    box=DM._artwork_rectangle(4.,4.)
    line=DM._ArtworkRoundLine((-1.,0.),(1.,0.),.2)
    for parts in (Tuple{Bool,DM._ArtworkShape}[(true,box),(false,line)],
            Tuple{Bool,DM._ArtworkShape}[(true,line),(false,box)],
            Tuple{Bool,DM._ArtworkShape}[(true,line),(false,line),(true,box)])
        new=DM._ArtworkComposite(parts);old=LegacyComposite(parts)
        for x in range(-2.2,2.2;length=35),y in range(-2.2,2.2;length=33)
            @test DM._artwork_contains(new,x,y)==DM._artwork_contains(old,x,y)
        end
        @test DM._artwork_bounds(new)==DM._artwork_bounds(old)
    end
    empty=DM._ArtworkComposite(Tuple{Bool,DM._ArtworkShape}[])
    @test !DM._artwork_contains(empty,0.,0.)
end
@testset "Small heterogeneous aperture storage and complete-mask equivalence" begin
    circle=DM._ArtworkCircle((0.,0.),.002)
    box=DM._artwork_rectangle(.006,.004)
    line=DM._ArtworkRoundLine((-.002,-.001),(.002,.001),.0002)
    corners=DM._odb_corner_rectangle(.005,.003,.0003;corners="13")
    shape_options=(circle,box,line,corners)
    for n in 2:8
        parts=Tuple{Bool,DM._ArtworkShape}[(isodd(i),shape_options[mod1(i,4)]) for i in 1:n]
        new=DM._ArtworkComposite(parts);old=LegacyComposite(parts)
        @test new.parts isa Tuple
        @test length(new.parts)==n
        @test all(p->isconcretetype(typeof(p)),new.parts)
        for x in range(-.0032,.0032;length=31),y in range(-.0022,.0022;length=29)
            @test DM._artwork_contains(new,x,y)==DM._artwork_contains(old,x,y)
        end
        @test memberbytes(new)==0
        @test DM._artwork_bounds(new)==DM._artwork_bounds(old)
        @test Base.summarysize(new)<=Base.summarysize(old)
    end
    long=DM._ArtworkComposite(Tuple{Bool,DM._ArtworkShape}[(isodd(i),shape_options[mod1(i,4)]) for i in 1:9])
    @test long.parts isa Vector # bounded tuple specialization policy
    mktempdir() do directory
        file=joinpath(directory,"features")
        write(file,"UNITS=MM\nF 1\n\$0 donut_sr6000x4000\nP 0 0 0 P 0 0\n")
        new=read_odb_features(file;layer="metal");old=legacy(new)
        grid=CellGrid(.007,.005,400,400);offset=(.0035,.0025)
        a=artwork_cell_masks(new,grid;offset);b=artwork_cell_masks(old,grid;offset)
        @test a==b
        @test count(only(values(a)))==79336
        @test (@allocated artwork_cell_masks(new,grid;offset))<100_000
        @test memberbytes(only(new.objects).shape)==0
        @test Base.summarysize(only(new.objects).shape)<=Base.summarysize(only(old.objects).shape)
    end
end
end
