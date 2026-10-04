module ODBExtraSymbolTests
using DiffMoM,Test,Random
const DM=DiffMoM
shape(name;unit=1.)=DM._odb_extra_standard_symbol(name,unit)
contains(s,x,y)=DM._artwork_contains(s,x,y)
@testset "Half oval independent rectangle/semicircle" begin
    for (w,h) in ((6.,3.),(3.,4.),(2.,4.),(30.,1.))
        s=shape("oval_h$(w)x$(h)")
        @test DM._artwork_bounds(s)==(-w/2,w/2,-h/2,h/2)
        for x in range(-w*.6,w*.6;length=31),y in range(-h*.6,h*.6;length=29)
            expected=(-w/2<=x<=(w-h)/2 && abs(y)<=h/2) ||
                ((w-h)/2<=x && (x-(w-h)/2)^2+y^2<=(h/2)^2)
            @test contains(s,x,y)==expected
        end
    end
    @test_throws ArgumentError shape("oval_h1x4")
end

function explicit_slots(x,y,angle,n,gap)
    for k in 0:n-1
        # Independent Cartesian dot products; no angular-periodic modulo.
        a=deg2rad(angle)+2pi*k/n
        along=x*cos(a)+y*sin(a);perp=-x*sin(a)+y*cos(a)
        along>=0 && abs(perp)<gap/2 && return true
    end
    return false
end
@testset "Primary stencil ordered numeric radius parameters" begin
    for prefix in ("hplate6x4x1","rhplate6x4x1","fhplate6x4x0.5x1"),
            unit in (1.,1e-6,25.4e-6),suffix in ("x0.2","x0.2x0.1","xra0.2x0.1","x0.2xro0.1")
        numeric=shape(prefix*suffix;unit)
        named=shape(prefix*(suffix=="x0.2" ? "xra0.2" : "xra0.2xro0.1");unit)
        @test numeric!==nothing
        @test DM._artwork_bounds(numeric)==DM._artwork_bounds(named)
        for x in range(-3.3,3.3;length=17),y in range(-2.3,2.3;length=13)
            @test contains(numeric,x*unit,y*unit)==contains(named,x*unit,y*unit)
        end
    end
end
@testset "Squared thermal independent rays and boundaries" begin
    Random.seed!(9283)
    for kind in ("ths","s_ths","sr_ths"),n in (1,2,3,4,5,9),angle in (0.,13.,45.,179.,333.,360.)
        s=shape("$(kind)6x4x$(angle)x$(n)x0.5")
        for _ in 1:71
            x,y=randn(2).*2
            outside=kind=="ths" ? x*x+y*y<=9 : max(abs(x),abs(y))<=3
            inside=kind=="s_ths" ? max(abs(x),abs(y))<=2 : x*x+y*y<=4
            expected=outside && !inside && !explicit_slots(x,y,angle,n,.5)
            @test contains(s,x,y)==expected
        end
    end
    for (w,h) in ((6.,4.),(4.,6.)),kind in ("rc_ths","o_ths"),n in (1,2,3,4,5)
        s=shape("$(kind)$(w)x$(h)x45x$(n)x0.5x0.4")
        capsule(x,y,w,h)=w>=h ? (max(abs(x)-(w-h)/2,0)^2+y^2<=(h/2)^2) :
            (x^2+max(abs(y)-(h-w)/2,0)^2<=(w/2)^2)
        for _ in 1:121
            x,y=randn(2).*2
            outside=kind=="rc_ths" ? abs(x)<=w/2&&abs(y)<=h/2 : capsule(x,y,w,h)
            inside=kind=="rc_ths" ? abs(x)<=w/2-.4&&abs(y)<=h/2-.4 : capsule(x,y,w-.8,h-.8)
            @test contains(s,x,y)==(outside && !inside && !explicit_slots(x,y,45,n,.5))
        end
    end
    @test_throws ArgumentError shape("rc_ths6x4x13x4x0.5x0.4")
    @test_throws ArgumentError shape("ths6x4x0x0x0.5")
    @test_throws ArgumentError shape("ths4x6x0x4x0.5")
    @test_throws ArgumentError shape("ths6x4x0x4x-1")
    @test_throws ArgumentError shape("ths6x4x-27x4x0.5")
    @test_throws ArgumentError shape("ths6x4x361x4x0.5")
end

@testset "Home plate independent inequality boundaries" begin
    for kind in ("hplate","rhplate")
        s=shape("$(kind)6x4x1")
        for x in range(-3.2,3.2;length=73),y in range(-2.2,2.2;length=61)
            expected=abs(y)<=2 && x>=-3 && x<=(kind=="hplate" ? 3-abs(y)/2 : 2+abs(y)/2)
            # Avoid floating equalities at sloping boundaries.
            abs(x-(kind=="hplate" ? 3-abs(y)/2 : 2+abs(y)/2))<1e-13 && continue
            @test contains(s,x,y)==expected
        end
    end
    s=shape("fhplate6x4x0.5x1")
    for x in range(-3.2,3.2;length=73),y in range(-2.2,2.2;length=61)
        expected=abs(y)<=2 && -3<=x<=3 && (x<=2 || abs(y)<=2-.5*(x-2))
        abs(abs(y)-(2-.5*(x-2)))<1e-13 && continue
        @test contains(s,x,y)==expected
    end
    for name in ("hplate6x4x0","rhplate6x4x0","fhplate6x4x0x1","fhplate6x4x0.5x0")
        s=shape(name)
        for (x,y) in ((0.,0.),(2.9,1.9),(-2.9,-1.9),(3.1,0.),(0.,2.1))
            @test contains(s,x,y)==(abs(x)<=3 && abs(y)<=2)
        end
    end
    triangle=shape("hplate6x4x6")
    for x in (-3.,-2.9,0.,2.9,3.1),y in (-1.9,-.5,.1,1.9)
        @test contains(triangle,x,y)==(-3<=x<=3 && abs(y)<=2*(3-x)/6)
    end
end

@testset "Fillet analytic circles and strict edge constraints" begin
    # Equal-radius rectangle oracle is independent of region crossings.
    rounded=DM._odb_stencil_region([(-3.,-2.),(3.,-2.),(3.,2.),(-3.,2.)],fill(.4,4))
    for x in range(-3.2,3.2;length=65),y in range(-2.2,2.2;length=65)
        expected=abs(x)<=3 && abs(y)<=2 && (abs(x)<=2.6 || abs(y)<=1.6 ||
            (abs(x)-2.6)^2+(abs(y)-1.6)^2<=.16)
        abs((abs(x)-2.6)^2+(abs(y)-1.6)^2-.16)<1e-13 && continue
        @test contains(rounded,x,y)==expected
    end
    @test_throws ArgumentError DM._odb_stencil_region([(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)],fill(1.,4))
    @test_throws ArgumentError shape("hplate6x4x1xra3xro3")
    # A reflex notch is filled by a tangent fillet of its own radius.
    sharp=shape("rhplate6x4x1")
    rounded=shape("rhplate6x4x1xro0.2")
    @test !contains(sharp,2.01,0.)
    @test contains(rounded,2.01,0.)
    @test contains(rounded,2.2,0.)==false
    for name in ("hplate6x4x1xra0.2","hplate6x4x1xra0.2xro0.1","rhplate6x4x1xra0.2xro0.1",
            "fhplate6x4x0.5x1xra0.2xro0.1")
        s=shape(name);@test contains(s,0.,0.)
        @test all(isfinite,DM._artwork_bounds(s))
    end
end

@testset "Symbol units, transforms and transparent inner shape" begin
    for name in ("oval_h6x3","ths6x4x13x5x0.5","hplate6x4x1xra0.2")
        a=shape(name);b=shape(name;unit=.001)
        for (x,y) in ((0.,0.),(2.5,0.),(2.,1.8),(-2.8,-1.8),(3.1,0.))
            @test contains(a,x,y)==contains(b,.001x,.001y)
        end
    end
    thermal=shape("ths6x4x45x4x0.5")
    composite=DM._ArtworkComposite(Tuple{Bool,DM._ArtworkShape}[(true,DM._ArtworkCircle((0.,0.),.1)),(true,thermal)])
    @test contains(composite,0.,0.)
    @test !contains(thermal,0.,0.)
    @test contains(thermal,2.5,0.)
    @test !contains(thermal,2.,2.)
    @test shape("s_tho6x4x45x4x1")===nothing # Pending exact open-corner semantics.
    @test shape("thr6x4x45x4x1")!==nothing
end

function read_fixture(body;kw...)
    mktempdir() do directory
        path=joinpath(directory,"features");write(path,body)
        return read_odb_features(path;layer="metal",kw...)
    end
end
function occupied(doc,x,y)
    value=false
    for object in doc.objects
        DM._artwork_contains(object.shape,x,y) && (value=object.dark)
    end
    return value
end
@testset "Public extra-symbol reader, units, orientation and polarity" begin
    names=("oval_h6000x3000","ths6000x4000x13x5x500","s_ths6000x4000x13x5x500",
        "sr_ths6000x4000x13x5x500","rc_ths6000x4000x45x4x500x400",
        "o_ths6000x4000x90x4x500x400","hplate6000x4000x1000xra200xro100",
        "rhplate6000x4000x1000xra200xro100","fhplate6000x4000x500x1000xra200xro100")
    # Boundary behavior is tested directly above. These transformed probes
    # have finite clearance: adding/subtracting the 7/11mm origins rounds
    # exact inner/outer boundary coordinates to opposite sides by one ulp.
    points=((0.,0.),(2.5,0.),(2.03,1.81),(-2.8,-1.8),(3.1,0.),(2.03,2.01),(-1.,1.))
    for name in names,orientation in 0:9
        angle=orientation>=8 ? " 23" : ""
        doc=read_fixture("UNITS=MM\nF 1\n\$0 $(name) M\nP 7 11 0 P 0 $(orientation)$(angle);ID=123\n")
        @test length(doc.objects)==1
        @test doc.objects[1].attributes["ID"]==["123"]
        base=shape(name;unit=1e-6)
        degrees=orientation<8 ? 90mod(orientation,4) : 23
        reflected=orientation in (4,5,6,7,9)
        for (x,y) in points
            u=cosd(degrees)*x+sind(degrees)*y
            v=-sind(degrees)*x+cosd(degrees)*y
            reflected && (u=-u)
            got=occupied(doc,7e-3+1e-3u,11e-3+1e-3v);expected=contains(base,1e-3x,1e-3y)
            @test got==expected
        end
    end
    # Transparent holes leave old copper under both dark and clear pads.
    body="UNITS=MM\nF 3\n\$0 r1000\n\$1 ths6000x4000x45x4x500\nP 0 0 0 P 0 0\nP 0 0 1 P 0 0\nP 0 0 1 N 0 0\n"
    doc=read_fixture(body)
    @test occupied(doc,0.,0.)
    @test !occupied(doc,2.5e-3,0.)
    @test_throws ArgumentError read_fixture(body;max_objects=2)
    @test_throws ArgumentError read_fixture(body;max_bytes=1)
    # Symbol-rotation suffix is clockwise; internal thermal gap is CCW.
    doc=read_fixture("UNITS=MM\nF 1\n\$0 ths6000x4000x13x1x500_17\nP 0 0 0 P 0 0\n")
    @test !occupied(doc,2.5e-3*cosd(-4),2.5e-3*sind(-4))
    @test occupied(doc,2.5e-3*cosd(30),2.5e-3*sind(30))
end
function membership_allocations(s::S) where {S<:DM._ArtworkShape}
    contains(s,.0025,0.)
    return @allocated contains(s,.0025,0.)
end
@testset "Extra analytic symbol rasters do not allocate per cell" begin
    for name in ("ths6000x4000x13x5x500","s_ths6000x4000x13x5x500",
            "hplate6000x4000x1000xra200xro100","oval_h6000x3000")
        doc=read_fixture("UNITS=MM\nF 1\n\$0 $(name)\nP 0 0 0 P 0 0\n")
        grid=CellGrid(.007,.005,400,400);offset=(.0035,.0025)
        artwork_cell_masks(doc,grid;offset)
        @test (@allocated artwork_cell_masks(doc,grid;offset))<100_000
        @test membership_allocations(only(doc.objects).shape)==0
        @test length(only(values(artwork_cell_masks(doc,grid;offset))))==160_000
    end
end
end
