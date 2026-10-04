module ODBRoundedThermalTests
using DiffMoM, Test, Random
const D=DiffMoM
shape(name;unit=1.)=D._odb_extra_standard_symbol(name,unit)
member(s,x,y)=D._artwork_contains(s,x,y)

# Independent explicit arc/circle union, without periodic nearest-gap lookup.
function round_oracle(od,id,angle,n,gap,x,y)
    R=(od+id)/4;r=(od-id)/4
    alpha=asin((gap+(od-id)/2)/((od+id)/2))
    for k in 0:n-1
        lo=deg2rad(angle)+2pi*k/n+alpha
        hi=deg2rad(angle)+2pi*(k+1)/n-alpha
        for t in (lo,hi)
            hypot(x-R*cos(t),y-R*sin(t))<=r && return true
        end
        t=mod(atan(y,x)-lo,2pi)
        t<=hi-lo && abs(hypot(x,y)-R)<=r && return true
    end
    false
end

# Independent Cartesian closest-point calculation for four capped segments.
function line_oracle(os,is,gap,x,y)
    r=(os-is)/4;c=(os+is)/4;e=c-(gap+2r)/sqrt(2)
    for (a,b) in (((-e,c),(e,c)),((-e,-c),(e,-c)),((c,-e),(c,e)),((-c,-e),(-c,e)))
        dx,dy=b[1]-a[1],b[2]-a[2]
        den=dx*dx+dy*dy
        t=den==0 ? 0. : clamp(((x-a[1])*dx+(y-a[2])*dy)/den,0,1)
        hypot(x-a[1]-t*dx,y-a[2]-t*dy)<=r && return true
    end
    false
end

@testset "Round thermal independent arcs, cap clearance and normal width" begin
    Random.seed!(87031)
    for n in (1,2,3,4,7),angle in (0.,13.,45.,135.,333.,360.),
            (od,id,gap) in ((60.,40.,n<=4 ? 10. : .1),(60.,56.,.5))
        s=shape("thr$(od)x$(id)x$(angle)x$(n)x$(gap)")
        for _ in 1:257
            x,y=randn(2).*od/2
            @test member(s,x,y)==round_oracle(od,id,angle,n,gap,x,y)
        end
        R=(od+id)/4;r=(od-id)/4;delta=asin((gap+2r)/(2R))
        a=deg2rad(angle);p=(R*cos(a+delta),R*sin(a+delta));q=(R*cos(a-delta),R*sin(a-delta))
        # Literal spoke gap is the nearest distance between adjacent caps.
        @test hypot(p[1]-q[1],p[2]-q[2])-2r≈gap rtol=1e-12
        mid=a+pi/n
        for sign in (-1,1)
            @test member(s,(R+sign*r*(1-1e-8))*cos(mid),(R+sign*r*(1-1e-8))*sin(mid))
            @test !member(s,(R+sign*r*(1+1e-8))*cos(mid),(R+sign*r*(1+1e-8))*sin(mid))
        end
        # End-cap normal facing the adjacent cap: inward/outward probes.
        v=((q[1]-p[1])/(gap+2r),(q[2]-p[2])/(gap+2r))
        @test member(s,p[1]+r*(1-1e-8)*v[1],p[2]+r*(1-1e-8)*v[2])
        gap>0 && @test !member(s,p[1]+r*(1+1e-8)*v[1],p[2]+r*(1+1e-8)*v[2])
    end
    # No per-spoke geometry allocation, even for a high valid count.
    thin=shape("thr1000x999.9999x13x1000000x0.0001")
    @test thin!==nothing
    @test Base.summarysize(thin)<256
    for name in ("thr60x40x45x0x10","thr40x60x45x4x10","thr60x40x-1x4x10",
            "thr60x40x361x4x10","thr60x40x45x4x-1","thr60x40x45x4x40",
            "thr60x40x45x20x10","thr60x40x45x99999999999999999999999999x10")
        @test_throws ArgumentError shape(name)
    end
end

@testset "Line thermal literal Update 3 end diameter and cap gap" begin
    Random.seed!(38573)
    for (os,is,gap) in ((60.,40.,10.),(60.,56.,.5),(8.,4.,0.))
        s=shape("s_thr$(os)x$(is)x45x4x$(gap)")
        r=(os-is)/4;c=(os+is)/4;e=c-(gap+2r)/sqrt(2)
        @test 2r==(os-is)/2
        @test sqrt(2)*(c-e)-2r≈gap atol=1e-14
        @test c+r==os/2
        @test c-r==is/2
        @test D._artwork_bounds(s)==(-os/2,os/2,-os/2,os/2)
        for _ in 1:1001
            x,y=randn(2).*os/2
            @test member(s,x,y)==line_oracle(os,is,gap,x,y)
        end
        for sign in (-1,1)
            @test member(s,0.,c+sign*r*(1-1e-8))
            @test !member(s,0.,c+sign*r*(1+1e-8))
        end
        @test member(s,e+r*(1-1e-8)/sqrt(2),c-r*(1-1e-8)/sqrt(2))
        gap>0 && @test !member(s,e+r*(1+1e-8)/sqrt(2),c-r*(1+1e-8)/sqrt(2))
    end
    for name in ("s_thr60x40x0x4x10","s_thr60x40x135x4x10","s_thr60x40x45x3x10",
            "s_thr60x40x45x4x40","s_thr40x60x45x4x10","s_thr60x40x45x4x-1")
        @test_throws ArgumentError shape(name)
    end
end

function reader(text;kw...)
    mktempdir() do dir
        file=joinpath(dir,"features");write(file,text)
        read_odb_features(file;kw...)
    end
end
function occupied(doc,x,y)
    state=false
    for obj in doc.objects
        member(obj.shape,x,y) && (state=obj.dark)
    end
    state
end
function allocation(s::S) where {S<:D._ArtworkShape}
    member(s,.0025,0.)
    @allocated member(s,.0025,0.)
end
@testset "Public rounded thermal units, orientation, polarity and resources" begin
    for name in ("thr6000x4000x13x3x500","s_thr6000x4000x45x4x500"),
            units in (("MM",1e-6),("INCH",25.4e-6)),orientation in 0:9
        unit,m=units;theta=orientation>=8 ? " 23" : ""
        doc=reader("UNITS=$unit\nF 1\n\$0 $name\nP 7 11 0 P 0 $orientation$theta;ID=14\n")
        base=shape(name;unit=m)
        @test only(doc.objects).attributes["ID"]==["14"]
        degrees=orientation<8 ? 90mod(orientation,4) : 23
        reflected=orientation in (4,5,6,7,9)
        coord=unit=="MM" ? 1e-3 : .0254
        for (x,y) in ((0.,0.),(2500.,0.),(1700.,2100.),(-1200.,2500.),(3100.,0.),(-2600.,-400.))
            u=cosd(degrees)*x+sind(degrees)*y
            v=-sind(degrees)*x+cosd(degrees)*y
            reflected && (u=-u)
            @test occupied(doc,7coord+m*u,11coord+m*v)==member(base,m*x,m*y)
        end
    end
    for name in ("thr6000x4000x45x4x500","s_thr6000x4000x45x4x500")
        body="UNITS=MM\nF 3\n\$0 r1000\n\$1 $name\nP 0 0 0 P 0 0\nP 0 0 1 P 0 0\nP 0 0 1 N 0 0\n"
        doc=reader(body)
        @test occupied(doc,0.,0.) # Inner opening is transparent.
        @test !occupied(doc,0.,.0025)
        @test_throws ArgumentError reader(body;max_objects=2)
        @test_throws ArgumentError reader(body;max_bytes=1)
        single=reader("UNITS=MM\nF 1\n\$0 $name\nP 0 0 0 P 0 0\n")
        grid=CellGrid(.007,.007,400,400);offset=(.0035,.0035)
        artwork_cell_masks(single,grid;offset)
        @test (@allocated artwork_cell_masks(single,grid;offset))<100_000
        @test allocation(only(single.objects).shape)==0
    end
    # Standard suffix rotates CW; the circular gap itself is CCW.
    doc=reader("UNITS=MM\nF 1\n\$0 thr6000x4000x13x1x500_17\nP 0 0 0 P 0 0\n")
    @test !occupied(doc,.0025cosd(-4),.0025sind(-4))
    @test occupied(doc,.0025cosd(90),.0025sind(90))
end
end
