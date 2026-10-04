module ODBDPackTests
using DiffMoM, Test, Random
const DM=DiffMoM

# Independent explicit rectangles with rounded corners. Pad lengths follow
# the bounding side minus all intervening gaps, divided by the pad count.
function oracle(x,y,w,h,hg,vg,hn,vn,r)
    pw=(w-(hn-1)*hg)/hn;ph=(h-(vn-1)*vg)/vn
    for j in 0:vn-1,i in 0:hn-1
        left=-w/2+i*(pw+hg);bottom=-h/2+j*(ph+vg)
        left<=x<=left+pw && bottom<=y<=bottom+ph || continue
        r==0 && return true
        dx=max(left+r-x,0.,x-(left+pw-r))
        dy=max(bottom+r-y,0.,y-(bottom+ph-r))
        dx*dx+dy*dy<=r*r && return true
    end
    return false
end

@testset "ODB D-Pack independent geometry and constant storage" begin
    Random.seed!(572884)
    for (w,h) in ((100.,80.),(17.,63.)),(hn,vn) in ((1,1),(1,4),(3,1),(3,4),(7,5)),
            fraction in (0.,.1,.45),radius_fraction in (0.,.3,.5)
        hg=fraction*w/hn;vg=fraction*h/vn
        pw=(w-hg*(hn-1))/hn;ph=(h-vg*(vn-1))/vn;r=radius_fraction*min(pw,ph)
        name="dpack$(w)x$(h)x$(hg)x$(vg)x$(hn)x$(vn)x$(r)"
        s=DM._odb_standard_symbol(name,1.)
        @test s isa DM._ArtworkODBDPack
        @test DM._artwork_bounds(s)==(-w/2,w/2,-h/2,h/2)
        @test Base.summarysize(s)<=80
        for _ in 1:501
            x=(rand()-.5)*1.2w;y=(rand()-.5)*1.2h
            @test DM._artwork_contains(s,x,y)==oracle(x,y,w,h,hg,vg,hn,vn,r)
        end
        for unit in (1e-6,25.4e-6)
            # Derive the inclusive radius boundary in the stored units;
            # scaling an already rounded boundary may exceed it by one ulp.
            spw=(w*unit-hg*unit*(hn-1))/hn;sph=(h*unit-vg*unit*(vn-1))/vn
            sr=radius_fraction*min(spw,sph)
            scaled=DM._odb_dpack(w*unit,h*unit,hg*unit,vg*unit,hn,vn,sr)
            for _ in 1:31
                x=(rand()-.5)*w;y=(rand()-.5)*h
                @test DM._artwork_contains(s,x,y)==DM._artwork_contains(scaled,x*unit,y*unit)
            end
        end
    end
    for name in ("dpack100x80x15x7x3x4", "dpack100x80x15x7x3x4xra5")
        @test DM._odb_standard_symbol(name,1.) isa DM._ArtworkODBDPack
    end
    sharp=DM._odb_standard_symbol("dpack6x4x2x2x2x2",1.)
    rounded=DM._odb_standard_symbol("dpack6x4x2x2x2x2x0.5",1.)
    @test DM._artwork_contains(sharp,-2.,-1.5)
    @test !DM._artwork_contains(sharp,0.,0.)
    @test DM._artwork_contains(sharp,-3.,-2.)
    @test !DM._artwork_contains(rounded,-3.,-2.)
    @test DM._artwork_contains(rounded,-2.,-1.5)
    for count in (1,10,1_000_000,1_000_000_000)
        s=DM._odb_dpack(1.,1.,0.,0.,count,count,0.)
        @test Base.summarysize(s)<=80
        for point in ((0.,0.),(.031,.039),(.5,.5))
            @test DM._artwork_contains(s,point...)
        end
        DM._artwork_contains(s,.031,.039)
        @test (@allocated DM._artwork_contains(s,.031,.039))==0
    end
    # Nonzero gaps/radii also keep membership allocation free.
    s=DM._odb_standard_symbol("dpack100x80x15x7x3x4x5",1.)
    DM._artwork_contains(s,31.,29.)
    @test (@allocated DM._artwork_contains(s,31.,29.))==0
    for name in ("dpack1x1x1x0x2x1","dpack1x1x0x1x1x2",
            "dpack1x1x0x0x0x1","dpack1x1x0x0x1x0",
            "dpack1x1x-1x0x2x2","dpack1x1x0x0x2x2x0.3",
            "dpack1x1x0x0x9007199254740993x1",
            "dpack1x1x0x0x999999999999999999999999999999x1")
        @test_throws ArgumentError DM._odb_standard_symbol(name,1.)
    end
    @test_throws ArgumentError DM._odb_dpack(1.,Inf,0.,0.,2,2,0.)
    @test_throws ArgumentError DM._odb_dpack(floatmin(Float64),1.,0.,0.,2^53,1,0.)
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

@testset "ODB D-Pack public units, orientation, polarity and bounds" begin
    for unit in ("MM","INCH"),orientation in 0:9,r in (0.,5.)
        symbol_unit=unit=="MM" ? 1e-6 : 25.4e-6
        coordinate_unit=1000symbol_unit
        angle=orientation>=8 ? " 23" : ""
        body="UNITS=$(unit)\nF 1\n\$0 dpack100x80x15x7x3x4x$(r)\nP 7 11 0 P 0 $(orientation)$(angle);ID=39\n"
        doc=read_fixture(body)
        @test length(doc.objects)==1
        @test doc.objects[1].attributes["ID"]==["39"]
        degrees=orientation<8 ? 90mod(orientation,4) : 23
        for _ in 1:101
            x=(rand()-.5)*120;y=(rand()-.5)*96
            u=cosd(degrees)*x+sind(degrees)*y;v=-sind(degrees)*x+cosd(degrees)*y
            orientation in (4,5,6,7,9) && (u=-u)
            @test occupied(doc,7coordinate_unit+symbol_unit*u,11coordinate_unit+symbol_unit*v)==
                oracle(x,y,100.,80.,15.,7.,3,4,r)
        end
    end
    # A clear D-Pack clears its pads but keeps copper in its transparent gaps.
    body="UNITS=MM\nF 2\n\$0 rect100x80\n\$1 dpack100x80x15x7x3x4x5\nP 0 0 0 P 0 0\nP 0 0 1 N 0 0\n"
    doc=read_fixture(body)
    for _ in 1:501
        x=(rand()-.5)*100;y=(rand()-.5)*80
        @test occupied(doc,x*1e-6,y*1e-6)==!oracle(x,y,100.,80.,15.,7.,3,4,5.)
    end
    @test_throws ArgumentError read_fixture(body;max_objects=1)
    @test_throws ArgumentError read_fixture(body;max_bytes=1)
    # Counts never grow retained geometry or transient per-pad allocations.
    huge="UNITS=MM\nF 1\n\$0 dpack100x100x0x0x1000000000x1000000000\nP 0 0 0 P 0 0\n"
    doc=read_fixture(huge;max_bytes=8192)
    @test length(doc.objects)==1
    @test occupied(doc,0.,0.)
    @test Base.summarysize(doc)<8192
end
end
