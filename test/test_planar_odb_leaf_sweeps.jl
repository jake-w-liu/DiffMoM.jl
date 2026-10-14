module ODBLeafSweepTests
using DiffMoM,Test,Random
const DM=DiffMoM

# ---- independent segment/primitive boundary solvers (test side) ----
# t in [0,1] parameterizes the aperture query segment a->b.
function tev_edge(c,d,a,b)
    vx,vy=b[1]-a[1],b[2]-a[2];ex,ey=d[1]-c[1],d[2]-c[2]
    den=vx*ey-vy*ex
    iszero(den)&&return Float64[]
    hx,hy=c[1]-a[1],c[2]-a[2]
    [(hx*ey-hy*ex)/den]
end
function tev_circle(c,r,a,b)
    vx,vy=b[1]-a[1],b[2]-a[2];fx,fy=a[1]-c[1],a[2]-c[2]
    A=vx*vx+vy*vy;B=2(fx*vx+fy*vy);C=fx*fx+fy*fy-r*r
    iszero(A)&&return Float64[]
    D=B*B-4A*C;D<0&&return Float64[]
    s=sqrt(D)
    [(-B-s)/(2A),(-B+s)/(2A)]
end
# Boundary candidates for canonical leaf primitives used inside composites.
function tev_primitive(s,a,b)
    s isa DM._ArtworkEmpty&&return Float64[]
    s isa DM._ArtworkCircle&&return tev_circle(s.center,s.radius,a,b)
    if s isa DM._ArtworkPolygon
        ev=Float64[];v=s.vertices
        for i in axes(v,2)
            j=mod1(i+1,size(v,2))
            append!(ev,tev_edge((v[1,i],v[2,i]),(v[1,j],v[2,j]),a,b))
        end
        return ev
    end
    if s isa DM._ArtworkRoundLine
        ev=Float64[]
        append!(ev,tev_circle(s.start,s.radius,a,b))
        append!(ev,tev_circle(s.stop,s.radius,a,b))
        dx,dy=s.stop[1]-s.start[1],s.stop[2]-s.start[2];L=hypot(dx,dy)
        iszero(L)&&return ev
        nx,ny=-dy/L*s.radius,dx/L*s.radius
        for sgn in (-1.,1.)
            append!(ev,tev_edge((s.start[1]+sgn*nx,s.start[2]+sgn*ny),
                (s.stop[1]+sgn*nx,s.stop[2]+sgn*ny),a,b))
        end
        return ev
    end
    throw(ArgumentError("test oracle lacks primitive $(typeof(s))"))
end

# ---- independent membership references ----
function ref_hit(s::DM._ArtworkODBHalfOval,x,y)
    w,h,c=s.width/2,s.height/2,(s.width-s.height)/2
    (-w<=x<=w&&abs(y)<=h)||return false
    x<=c||hypot(x-c,y)<=h
end
function ref_hit(s::DM._ArtworkODBDPack,x,y)
    abs(x)<=s.width/2&&abs(y)<=s.height/2||return false
    px=s.pad_width+s.horizontal_gap;py=s.pad_height+s.vertical_gap
    for j in 0:s.vertical_count-1,i in 0:s.horizontal_count-1
        left=-s.width/2+i*px;bottom=-s.height/2+j*py
        left<=x<=left+s.pad_width&&bottom<=y<=bottom+s.pad_height||continue
        ax=abs(x-(left+s.pad_width/2));ay=abs(y-(bottom+s.pad_height/2))
        (ax<=s.pad_width/2-s.radius||ay<=s.pad_height/2-s.radius)&&return true
        hypot(ax-(s.pad_width/2-s.radius),ay-(s.pad_height/2-s.radius))<=s.radius&&return true
    end
    false
end
function ref_slot(qx,qy,angle,count,step,gap)
    # Explicit per-spoke Cartesian test; no angular-periodic modulo.
    for k in 0:count-1
        a=angle+k*step
        along=qx*cos(a)+qy*sin(a);perp=-qx*sin(a)+qy*cos(a)
        along>=0&&abs(perp)<gap/2&&return true
    end
    false
end
function ref_hit(s::DM._ArtworkODBSquaredThermal,x,y)
    DM._artwork_contains(s.outside,x,y)||return false
    DM._artwork_contains(s.inside,x,y)&&return false
    !ref_slot(x,y,deg2rad(s.angle),s.count,2pi/s.count,s.gap)
end
function ref_hit(s::DM._ArtworkODBRoundedThermal,x,y)
    r=hypot(x,y);abs(r-s.center_radius)<=s.cap_radius||return false
    theta=atan(y,x);step=2pi/s.count
    delta=mod(theta-s.angle+step/2,step)-step/2
    abs(delta)>=s.half_gap_angle&&return true
    ep=theta-delta+copysign(s.half_gap_angle,delta)
    hypot(x-s.center_radius*cos(ep),y-s.center_radius*sin(ep))<=s.cap_radius
end
function ref_hit(s::DM._ArtworkODBLineThermal,x,y)
    # Independent four-capsule distance test (no mirrored hypot form).
    for (c,d) in (((-s.endpoint,s.center),(s.endpoint,s.center)),
            ((-s.endpoint,-s.center),(s.endpoint,-s.center)),
            ((s.center,-s.endpoint),(s.center,s.endpoint)),
            ((-s.center,-s.endpoint),(-s.center,s.endpoint)))
        dx,dy=d[1]-c[1],d[2]-c[2];q=dx*dx+dy*dy
        t=iszero(q) ? 0. : clamp(((x-c[1])*dx+(y-c[2])*dy)/q,0.,1.)
        hypot(x-c[1]-t*dx,y-c[2]-t*dy)<=s.radius&&return true
    end
    false
end

# ---- independent boundary events per leaf ----
function leaf_events(s::DM._ArtworkODBHalfOval,a,b)
    w,h,c=s.width/2,s.height/2,(s.width-s.height)/2
    ev=Float64[]
    for (p,q) in (((-w,-h),(w,-h)),((-w,h),(w,h)),((-w,-h),(-w,h)))
        append!(ev,tev_edge(p,q,a,b))
    end
    append!(ev,tev_circle((c,0.),h,a,b));ev
end
function leaf_events(s::DM._ArtworkODBDPack,a,b)
    px,py,cx,cy=DM._odb_dpack_pads(s);ev=Float64[]
    w,h,r=s.pad_width/2,s.pad_height/2,s.radius
    for j in 0:s.vertical_count-1,i in 0:s.horizontal_count-1
        ox,oy=cx+i*px,cy+j*py
        for (p,q) in (((ox-w,oy-h),(ox+w,oy-h)),((ox+w,oy-h),(ox+w,oy+h)),
                ((ox+w,oy+h),(ox-w,oy+h)),((ox-w,oy+h),(ox-w,oy-h)))
            append!(ev,tev_edge(p,q,a,b))
        end
        r>0||continue
        for (sx,sy) in ((1.,1.),(-1.,1.),(-1.,-1.),(1.,-1.))
            append!(ev,tev_circle((ox+sx*(w-r),oy+sy*(h-r)),r,a,b))
        end
    end
    ev
end
function leaf_events(s::DM._ArtworkODBSquaredThermal,a,b)
    ev=tev_primitive(s.outside,a,b);append!(ev,tev_primitive(s.inside,a,b))
    iszero(s.gap)&&return ev
    step=2pi/s.count;base=deg2rad(s.angle);g=s.gap/2
    for k in 0:s.count-1
        ux,uy=DM._odb_thermal_spoke(base,k,step);nx,ny=-uy,ux
        for sgn in (-1.,1.)
            c=(sgn*g*nx,sgn*g*ny)
            append!(ev,tev_edge(c,(c[1]+ux,c[2]+uy),a,b))
        end
        append!(ev,tev_edge((0.,0.),(nx,ny),a,b))
    end
    ev
end
function leaf_events(s::DM._ArtworkODBRoundedThermal,a,b)
    cr,cap=s.center_radius,s.cap_radius
    ev=Float64[]
    append!(ev,tev_circle((0.,0.),cr+cap,a,b))
    append!(ev,tev_circle((0.,0.),max(0.,cr-cap),a,b))
    step=2pi/s.count
    for k in 0:s.count-1
        thc=s.angle+k*step
        for sgn in (-1.,1.)
            ux,uy=cos(thc+sgn*s.half_gap_angle),sin(thc+sgn*s.half_gap_angle)
            append!(ev,tev_edge((0.,0.),(ux,uy),a,b))
            append!(ev,tev_circle((cr*ux,cr*uy),cap,a,b))
        end
    end
    ev
end
function leaf_events(s::DM._ArtworkODBLineThermal,a,b)
    ev=Float64[]
    for (c,d) in (((-s.endpoint,s.center),(s.endpoint,s.center)),
            ((-s.endpoint,-s.center),(s.endpoint,-s.center)),
            ((s.center,-s.endpoint),(s.center,s.endpoint)),
            ((-s.center,-s.endpoint),(-s.center,s.endpoint)))
        append!(ev,tev_primitive(DM._ArtworkRoundLine(c,d,s.radius),a,b))
    end
    ev
end

function swept_oracle(shape,start,stop,x,y)
    a=(x-start[1],y-start[2]);b=(x-stop[1],y-stop[2])
    dx,dy=b[1]-a[1],b[2]-a[2]
    ts=sort!(unique!(filter(t->0<=t<=1,push!(leaf_events(shape,a,b),0.,1.))))
    hit(t)=ref_hit(shape,a[1]+dx*t,a[2]+dy*t)
    any(hit,ts)&&return true
    for i in 1:length(ts)-1
        hit((ts[i]+ts[i+1])/2)&&return true
    end
    false
end

function sweep_checks(shape,start,stop,xs,ys)
    stroke=DM._artwork_composite_line(shape,start,stop;max_boundaries=4096)
    checks=0
    for x in xs,y in ys
        @test DM._artwork_contains(stroke,x,y)==swept_oracle(shape,start,stop,x,y)
        checks+=1
    end
    checks
end

@testset "Specialized ODB leaf sweeps versus independent oracles" begin
    oval=DM._ArtworkODBHalfOval(6.,3.)
    @test sweep_checks(oval,(-7.,-1.),(5.,2.5),range(-9.,8.;length=61),range(-4.,4.;length=43))>0
    # Degenerate stroke reduces to the aperture itself.
    @test sweep_checks(oval,(1.,1.),(1.,1.),range(-4.5,5.5;length=41),range(-2.5,2.5;length=29))>0

    dpack=DM._odb_dpack(6.,4.,1.,.5,3,2,.2)
    @test sweep_checks(dpack,(-8.,0.),(6.,1.),range(-10.,10.;length=71),range(-4.,4.;length=37))>0
    dpack2=DM._odb_dpack(5.,5.,.5,.5,2,2,0.)
    @test sweep_checks(dpack2,(-6.,-3.),(7.,3.),range(-9.,9.;length=67),range(-5.,5.;length=41))>0

    st=DM._ArtworkODBSquaredThermal(DM._ArtworkCircle((0.,0.),3.),
        DM._ArtworkCircle((0.,0.),1.5),13.,4,.5)
    @test sweep_checks(st,(-6.,-.5),(5.,.75),range(-8.,8.;length=73),range(-4.,4.;length=41))>0
    st0=DM._ArtworkODBSquaredThermal(DM._artwork_rectangle(6.,6.),
        DM._ArtworkEmpty(),37.,3,.4)
    @test sweep_checks(st0,(-5.,2.),(6.,-1.),range(-9.,9.;length=67),range(-5.,5.;length=43))>0

    rt=DM._ArtworkODBRoundedThermal(3.,.8,deg2rad(17.),4,asin(.4),3.8)
    @test sweep_checks(rt,(-7.,1.),(6.,-1.5),range(-9.,9.;length=71),range(-5.,5.;length=43))>0
    # One spoke: reflexive-free single gap sector.
    rt1=DM._ArtworkODBRoundedThermal(3.,.8,deg2rad(33.),1,asin(.4),3.8)
    @test sweep_checks(rt1,(-7.,0.),(6.,0.),range(-9.,9.;length=61),range(-5.,5.;length=37))>0

    lt=DM._ArtworkODBLineThermal(1.2,2.2,.4,3.4)
    @test sweep_checks(lt,(-7.,-.75),(6.,.5),range(-9.,9.;length=67),range(-5.,5.;length=41))>0

    # Certified membership stays allocation free away from boundaries.
    stroke=DM._artwork_composite_line(rt,(0.,0.),(4.,0.);max_boundaries=4096)
    DM._artwork_contains(stroke,.2,.3)
    @test (@allocated DM._artwork_contains(stroke,.2,.3))==0
end

@testset "Certified trits agree with point membership" begin
    Random.seed!(447711)
    leaves=Any[
        DM._ArtworkODBHalfOval(6.,3.),
        DM._odb_dpack(6.,4.,1.,.5,3,2,.2),
        DM._ArtworkODBSquaredThermal(DM._ArtworkCircle((0.,0.),3.),DM._ArtworkCircle((0.,0.),1.5),13.,4,.5),
        DM._ArtworkODBSquaredThermal(DM._artwork_rectangle(6.,6.),DM._ArtworkEmpty(),37.,3,.4),
        DM._ArtworkODBRoundedThermal(3.,.8,deg2rad(17.),4,asin(.4),3.8),
        DM._ArtworkODBLineThermal(1.2,2.2,.4,3.4)]
    for s in leaves
        b=DM._artwork_bounds(s);cx=(b[1]+b[2])/2;cy=(b[3]+b[4])/2;R=max(b[2]-b[1],b[4]-b[3])
        for _ in 1:1500
            x,y=cx+randn()*R,cy+randn()*R
            trit=DM._artwork_certified_contains(s,DM._ArtworkInterval(x),DM._ArtworkInterval(y))
            ref=ref_hit(s,x,y)
            trit==1&&(@test ref;continue)
            trit==0&&(@test !ref;continue)
            @test trit==-1&&DM._artwork_exact_point(s,x,y)==ref
        end
    end
end

@testset "Thermal slot perpendiculars and degenerate annuli" begin
    # A two-spoke squared thermal sees a slot perpendicular at x=0 for
    # angle zero; membership must match the modulo-free reference there.
    st=DM._ArtworkODBSquaredThermal(DM._ArtworkCircle((0.,0.),4.),
        DM._ArtworkCircle((0.,0.),1.),0.,2,3.)
    for y in (-4.,-1.2,-.5,.5,1.2,4.)
        @test DM._artwork_contains(st,0.,y)==ref_hit(st,0.,y)
        @test DM._artwork_exact_point(st,0.,y)==ref_hit(st,0.,y)
    end
    @test DM._artwork_contains(st,0.,0.)==ref_hit(st,0.,0.)
    # A one-spoke slot keeps the half-plane behind its ray direction.
    st1=DM._ArtworkODBSquaredThermal(DM._ArtworkCircle((0.,0.),4.),
        DM._ArtworkEmpty(),0.,1,2.)
    for (x,y) in ((-1.,0.),(-1.,.4),(-1.,-.4),(0.,0.),(-3.9,.01))
        @test DM._artwork_contains(st1,x,y)==ref_hit(st1,x,y)
        @test DM._artwork_exact_point(st1,x,y)==ref_hit(st1,x,y)
    end
    # Hand-built annulus with cap wider than its center radius still
    # answers near the origin instead of rejecting small radii.
    wide=DM._ArtworkODBRoundedThermal(.5,2.,0.,4,.3,2.5)
    @test DM._artwork_contains(wide,.1,.1)==ref_hit(wide,.1,.1)
    @test DM._artwork_exact_point(wide,.1,.1)==ref_hit(wide,.1,.1)
end

@testset "Feature records sweep specialized symbols end to end" begin
    function imported(aperture,start,stop;kind="L",extra="")
        mktempdir() do directory
            path=joinpath(directory,"features")
            coords=join((string(1000v) for v in (start...,stop...))," ")
            body=kind=="L" ? "L "*coords*" 0 P 0\n" :
                "A "*coords*" "*extra*" 0 P 0 Y\n"
            write(path,"UNITS=MM\n\$0 $(aperture)\n"*body)
            DM.read_odb_features(path)
        end
    end
    for (name,leaf) in (("oval_h6000x3000",DM._ArtworkODBHalfOval),
            ("dpack6000x4000x1000x500x3x2xra200",DM._ArtworkODBDPack),
            ("thr7600x4000x45x4x800",DM._ArtworkODBRoundedThermal),
            ("s_thr7600x4000x45x4x800",DM._ArtworkODBLineThermal),
            ("ths6000x4000x13x4x500",DM._ArtworkODBSquaredThermal))
        doc=imported(name,(0.,0.),(12.,4.))
        @test length(doc.objects)==1
        stroke=doc.objects[1].shape
        @test stroke isa DM._ArtworkCompositeLine
        @test stroke.aperture isa leaf
        for x in (2.3,6.001,11.7),y in (-1.4,.499,2.9)
            @test DM._artwork_contains(stroke,x,y)==swept_oracle(stroke.aperture,stroke.start,stroke.stop,x,y)
        end
    end
    # A polygon symbol on an ODB arc becomes an analytic arc stroke.
    doc=imported("s6",(0.,0.),(0.,0.);kind="A",extra="6000 0")
    arc=only(doc.objects).shape
    @test arc isa DM._ArtworkPolygonArc
    @test arc.aperture isa DM._ArtworkPolygon
    doc=imported("di6x4",(0.,0.),(0.,0.);kind="A",extra="6000 0")
    @test only(doc.objects).shape isa DM._ArtworkPolygonArc
    # Unsupported arc apertures still fail loudly.
    mktempdir() do directory
        path=joinpath(directory,"features")
        write(path,"UNITS=MM\n\$0 ths6000x4000x13x4x500\nA 0 0 0 0 6000 0 0 P 0 Y\n")
        @test_throws ArgumentError DM.read_odb_features(path)
    end
end
end
