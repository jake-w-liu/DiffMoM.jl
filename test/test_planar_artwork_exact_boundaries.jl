module PlanarArtworkExactBoundaryTests
using DiffMoM, Test, LinearAlgebra
const D=DiffMoM
const Q=Rational{BigInt}

stroke(shape,a,b;max_bytes=64_000_000)=D._artwork_composite_line(shape,a,b;max_boundaries=256,max_bytes)
typedquery(s::S,x,y) where S=D._artwork_contains(s,x,y)
function triangle_interval(r,a,b,x,y)
    u=(Q(x)-Q(a[1]),Q(y)-Q(a[2]));v=(Q(a[1])-Q(b[1]),Q(a[2])-Q(b[2]))
    lo,hi=Q(0),Q(1)
    for (c,d) in ((-u[1],-v[1]),(-u[2],-v[2]),(u[1]+u[2]-Q(r),v[1]+v[2]))
        if iszero(d)
            c<=0||return nothing
        elseif d>0
            hi=min(hi,-c/d)
        else
            lo=max(lo,-c/d)
        end
    end
    lo<=hi ? (lo,hi) : nothing
end
function corner_oracle(s,x,y)
    x,y=Q(x),Q(y);w,h,r=Q(s.width)/2,Q(s.height)/2,Q(s.radius)
    abs(x)<=w&&abs(y)<=h||return false
    for (i,(sx,sy)) in enumerate(((1,1),(-1,1),(-1,-1),(1,-1)))
        s.corners[i]||continue
        u,v=sx*x,sy*y
        if s.chamfer
            u+v<=w+h-r||return false
        elseif u>w-r&&v>h-r
            (u-w+r)^2+(v-h+r)^2<=r*r||return false
        end
    end
    true
end

@testset "Closed tangent events preserve ordered clear and redraw" begin
    tri=D._ArtworkPolygon([0. .125 0.;0. 0. .125]);empty=D._ArtworkEmpty()
    aperture=D._ArtworkComposite(((true,tri),(false,empty)))
    a=(0.,0.);b=(.375,.375);p=(.25,.375)
    s=stroke(aperture,a,b)
    @test triangle_interval(.125,a,b,p...)==(Q(2)/3,Q(2)/3)
    @test typedquery(s,p...)
    @test !typedquery(s,p[1],nextfloat(p[2]))
    @test typedquery(s,p[1],prevfloat(p[2]))
    @test typedquery(stroke(aperture,b,a),p...)
    @test !typedquery(stroke(D._ArtworkComposite(((true,tri),(false,tri))),a,b),p...)
    @test typedquery(stroke(D._ArtworkComposite(((true,tri),(false,tri),(true,tri))),a,b),p...)
    for x in (-.25,0.,.125,.25,.375,.5),y in (-.25,0.,.125,.25,.375,.5)
        @test typedquery(s,x,y)==!isnothing(triangle_interval(.125,a,b,x,y))
        @test typedquery(s,x,y)==typedquery(stroke(aperture,b,a),x,y)
    end
    # Mutation of the source aperture cannot invalidate exact event metadata.
    tri.vertices[1,1]=100.
    @test typedquery(s,p...)
    transformed=D._artwork_transform(s.aperture;scale=2.,origin=(.125,.25))
    @test typedquery(stroke(transformed,a,(.75,.75)),.625,1.)
end

@testset "Canonical flash and zero/reversed sweep share analytic boundaries" begin
    r=.125;circle=D._ArtworkCircle((0.,0.),r)
    region=D._ArtworkRegion(Union{D._ArtworkRoundLine,D._ArtworkRoundArc}[
        D._ArtworkRoundArc((r,0.),(r,0.),(0.,0.),false,0.)])
    line=D._ArtworkRoundLine((-.25,0.),(.25,0.),r)
    arc=D._ArtworkRoundArc((r,0.),(0.,r),(0.,0.),false,.03125)
    # These literal formulas reproduce the formerly rounded/banded answers.
    @test hypot(r,1e-20)==r
    @test abs(hypot(nextfloat(r),0.)-r)<=8eps(r)
    @test !D._artwork_contains(circle,r,1e-20)
    @test !D._artwork_contains(region,nextfloat(r),0.)
    @test !D._artwork_contains(line,.25+r,1e-20)
    for shape in (circle,region,line,arc),p in ((0.,0.),(r,0.),(nextfloat(r),0.),(r,1e-20),(.0625,.0625),(.25,.25))
        flash=D._artwork_contains(shape,p...)
        @test typedquery(stroke(shape,(0.,0.),(0.,0.)),p...)==flash
        @test typedquery(stroke(shape,(.125,-.25),(.125,-.25)),p[1]+.125,p[2]-.25)==
            D._artwork_exact_qcontains(shape,(Q(p[1]+.125)-Q(.125),Q(p[2]-.25)+Q(.25)))
    end
    for shape in (circle,region,line,arc)
        forward=stroke(shape,(-.5,0.),(.5,0.));reverse=stroke(shape,(.5,0.),(-.5,0.))
        for y in (r,prevfloat(r),nextfloat(r),.0625,-r),x in (-.125,0.,.125)
            @test typedquery(forward,x,y)==typedquery(reverse,x,y)
        end
    end
    @test typedquery(stroke(circle,(-.5,0.),(.5,0.)),0.,r)
    @test !typedquery(stroke(circle,(-.5,0.),(.5,0.)),0.,nextfloat(r))
    @test typedquery(stroke(arc,(-.5,0.),(.5,0.)),0.,r+.03125)
    @test !typedquery(stroke(arc,(-.5,0.),(.5,0.)),0.,nextfloat(r+.03125))
end

@testset "Arc/Region exact events keep artificial chords internal" begin
    r=.125
    quarter=D._ArtworkRoundArc((r,0.),(0.,r),(0.,0.),false,0.)
    lens=D._ArtworkRegion(Union{D._ArtworkRoundLine,D._ArtworkRoundArc}[
        quarter,D._ArtworkRoundLine((0.,r),(r,0.),0.)])
    disk=D._ArtworkRegion(Union{D._ArtworkRoundLine,D._ArtworkRoundArc}[
        quarter,D._ArtworkRoundArc((0.,r),(-r,0.),(0.,0.),false,0.),
        D._ArtworkRoundArc((-r,0.),(0.,-r),(0.,0.),false,0.),
        D._ArtworkRoundArc((0.,-r),(r,0.),(0.,0.),false,0.)])
    for p in ((.0625,.0625),(-.0625,.0625),(-.0625,-.0625),(.0625,-.0625),(0.,0.),(r,0.))
        @test D._artwork_contains(disk,p...)
    end
    for x in (-.25,-r,-.07,0.,.0625,r,.25),y in (-.25,-r,-.07,0.,.0625,r,.25)
        @test D._artwork_contains(disk,x,y)==(Q(x)^2+Q(y)^2<=Q(r)^2)
    end
    @test D._artwork_contains(lens,.0625,.0625)
    @test !D._artwork_contains(lens,0.,0.)
    @test !D._artwork_contains(lens,.09,.09)
    @test D._artwork_contains(lens,.0875,.0875)
    for leaf in (quarter,lens,disk)
        @test typedquery(stroke(leaf,(-.5,0.),(.5,0.)),0.,r)
        @test !typedquery(stroke(leaf,(-.5,0.),(.5,0.)),0.,nextfloat(r))
        cleared=D._ArtworkComposite(((true,leaf),(false,leaf)))
        @test !typedquery(stroke(cleared,(-.5,0.),(.5,0.)),0.,r)
    end
    # A literal endpoint is closed even when the computed Float64 radius
    # rounds below its exact norm. This point has a unique t=2/3 witness.
    endpoint=(.25,.375)
    singleton=D._ArtworkRegion(Union{D._ArtworkRoundLine,D._ArtworkRoundArc}[
        D._ArtworkRoundArc(endpoint,endpoint,(0.,0.),false,0.)])
    @test Q(hypot(endpoint...))^2<Q(endpoint[1])^2+Q(endpoint[2])^2
    @test D._artwork_contains(singleton,endpoint...)
    @test typedquery(stroke(singleton,(0.,0.),(.5625,-.375)),.625,.125)
    @test typedquery(stroke(singleton,(.5625,-.375),(0.,0.)),.625,.125)
    @test !typedquery(stroke(D._ArtworkComposite(((true,singleton),(false,singleton))),
        (0.,0.),(.5625,-.375)),.625,.125)
    @test typedquery(stroke(D._ArtworkComposite(((true,singleton),(false,singleton),(true,singleton))),
        (0.,0.),(.5625,-.375)),.625,.125)
end

@testset "Corner and affine flash agree with independent dyadic equations" begin
    for chamfer in (false,true),corners in ("","1","13","1234")
        shape=D._odb_corner_rectangle(.5,.375,.0625;chamfer,corners)
        zero=stroke(shape,(0.,0.),(0.,0.))
        transformed=D._artwork_transform(shape;rotation=37.,scale=2.,origin=(.125,-.25))
        tz=stroke(transformed,(0.,0.),(0.,0.));m=transformed.inverse
        for x in (-.375,-.25,-.1875,-.0625,0.,.0625,.1875,.25,.375),
            y in (-.25,-.1875,-.125,-.0625,0.,.0625,.125,.1875,.25)
            expected=corner_oracle(shape,x,y)
            @test D._artwork_contains(shape,x,y)==expected
            @test typedquery(zero,x,y)==expected
            u,v=Q(x)-Q(transformed.origin[1]),Q(y)-Q(transformed.origin[2])
            affine_expected=corner_oracle(shape,Q(m[1,1])*u+Q(m[1,2])*v,Q(m[2,1])*u+Q(m[2,2])*v)
            @test D._artwork_contains(transformed,x,y)==affine_expected
            @test typedquery(tz,x,y)==affine_expected
        end
    end
    diamond=D._odb_corner_rectangle(2.,2.,1.;chamfer=true)
    shape=stroke(diamond,(0.,0.),(0.,0.))
    typedquery(shape,.3,.7)
    @test corner_oracle(diamond,.3,.7)
    @test (@allocated typedquery(shape,.3,.7))==0
end

@testset "Outward intervals enclose exact dyadic arithmetic" begin
    values=(-1e308,-1.,-.3,-floatmin(Float64),-nextfloat(0.),0.,
        nextfloat(0.),floatmin(Float64),.1,.3,1.,1e308)
    encloses(i,q)=(!isfinite(i.lo)||Q(i.lo)<=q)&&(!isfinite(i.hi)||q<=Q(i.hi))
    for a in values,b in values
        aa,bb=D._ArtworkInterval(a),D._ArtworkInterval(b)
        @test encloses(aa+bb,Q(a)+Q(b))
        @test encloses(aa-bb,Q(a)-Q(b))
        @test encloses(aa*bb,Q(a)*Q(b))
        iszero(b)||@test encloses(aa/bb,Q(a)/Q(b))
    end
    for x in values
        @test encloses(D._artwork_interval_half(x),Q(x)/2)
    end
end

@testset "Extreme-scale parity stays inside the original convex geometry" begin
    function convex_oracle(vertices,x,y)
        px,py=Q(x),Q(y)
        all(axes(vertices,2)) do i
            j=mod1(i+1,size(vertices,2))
            ax,ay,bx,by=Q(vertices[1,i]),Q(vertices[2,i]),Q(vertices[1,j]),Q(vertices[2,j])
            (bx-ax)*(py-ay)-(by-ay)*(px-ax)>=0
        end
    end
    for k in (1e-200,1e-160,1.,1e160,1e200)
        polygon=D._ArtworkPolygon([0. k .5k 0.;0. 0. k k])
        v=polygon.vertices
        region=D._ArtworkRegion(Union{D._ArtworkRoundLine,D._ArtworkRoundArc}[
            D._ArtworkRoundLine((v[1,i],v[2,i]),(v[1,mod1(i+1,4)],v[2,mod1(i+1,4)]),0.) for i in 1:4])
        polygonline=D._ArtworkPolygonLine(polygon,(0.,0.),(0.,0.))
        transformed=D._artwork_transform(polygon;rotation=37.)
        zero=stroke(D._ArtworkComposite(((true,polygon),(false,D._ArtworkEmpty()))),(0.,0.),(0.,0.))
        for (x,y) in ((.25k,.25k),(.75k,.25k),(.25k,.75k),(.75k,.75k),(-.25k,.25k),(1.25k,.25k))
            expected=convex_oracle(v,x,y)
            for leaf in (polygon,region,polygonline,zero)
                @test D._artwork_contains(leaf,x,y)==expected
            end
            m=transformed.inverse
            u,w=Q(m[1,1])*Q(x)+Q(m[1,2])*Q(y),Q(m[2,1])*Q(x)+Q(m[2,2])*Q(y)
            @test D._artwork_contains(transformed,x,y)==convex_oracle(v,u,w)
        end
    end
    pointregion=D._ArtworkRegion(Union{D._ArtworkRoundLine,D._ArtworkRoundArc}[
        D._ArtworkRoundArc((0.,0.),(0.,0.),(0.,0.),false,0.)])
    @test D._artwork_contains(pointregion,0.,0.)
    @test !D._artwork_contains(pointregion,nextfloat(0.),0.)
end

@testset "Cached affine clipping encloses inverse-defined geometry" begin
    function inverse_corners(transform,box)
        m=transform.inverse;a,b,c,d=Q(m[1,1]),Q(m[1,2]),Q(m[2,1]),Q(m[2,2]);det=a*d-b*c
        [(Q(transform.origin[1])+(d*Q(x)-b*Q(y))/det,
          Q(transform.origin[2])+(-c*Q(x)+a*Q(y))/det) for x in (box[1],box[2]),y in (box[3],box[4])]
    end
    function enclosed(bounds,p)
        Q(bounds[1])<=p[1]<=Q(bounds[2])&&Q(bounds[3])<=p[2]<=Q(bounds[4])
    end
    for k in (1e-200,1e-160,1.,1e160,1e200),rotation in (0.,37.,108.0485313789156,180.,270.)
        polygon=D._artwork_rectangle(2k,k)
        circle=D._ArtworkCircle((.25k,-.125k),.5k)
        for shape in (polygon,circle)
            transformed=D._artwork_transform(shape;rotation,scale=2.,origin=(.125,-.25))
            box=shape===polygon ? (-Q(k),Q(k),-Q(k)/2,Q(k)/2) :
                (Q(circle.center[1])-Q(circle.radius),Q(circle.center[1])+Q(circle.radius),
                 Q(circle.center[2])-Q(circle.radius),Q(circle.center[2])+Q(circle.radius))
            bounds=D._artwork_bounds(transformed)
            for p in inverse_corners(transformed,box)
                @test enclosed(bounds,p)
            end
            nested=D._artwork_transform(transformed;rotation=-17.,scale=.5,origin=(.25,.5))
            for p in inverse_corners(nested,D._artwork_bounds(transformed))
                @test enclosed(D._artwork_bounds(nested),p)
            end
        end
    end
    # The first interval determinant straddles zero, requiring the bounded
    # rational inverse fallback rather than a singular/rounded answer.
    rect=D._artwork_rectangle(2.,2.)
    ordinary=D._artwork_transform(rect)
    matrix=typeof(ordinary.inverse)(1.,1.,1.,nextfloat(1.))
    illconditioned=D._ArtworkTransform(rect,ordinary.matrix,matrix,(0.,0.))
    for p in inverse_corners(illconditioned,(-1.,1.,-1.,1.))
        @test enclosed(D._artwork_bounds(illconditioned),p)
    end
    w=710.5300605439081;h=5.0444195593837
    transformed=D._artwork_transform(D._artwork_rectangle(w,h);rotation=108.0485313789156,scale=65.21480961783402)
    x,y=7334.526273372109,-21977.55869920587
    @test D._artwork_contains(transformed,x,y)
    @test enclosed(D._artwork_bounds(transformed),(Q(x),Q(y)))
    grid=CellGrid(2x,-2y,1,1)
    mktempdir() do directory
        path=joinpath(directory,"features")
        write(path,"UNITS=MM\n\$0 custom\nP 0 0 0 P 0 0\n")
        artwork=read_odb_features(path;symbol_resolver=name->transformed,max_bytes=1_000_000)
        @test only(values(artwork_cell_masks(artwork,grid;offset=(0.,-2y))))[1,1]
    end
    # Stored centerline radius and literal endpoints share the same bounds.
    for clockwise in (false,true),start in ((.125,0.),(.25,.375)),stop in ((0.,.125),start)
        arc=D._ArtworkRoundArc(start,stop,(0.,0.),clockwise,0.)
        box=D._artwork_bounds(arc);r=hypot(start...)
        @test enclosed(box,(Q(start[1]),Q(start[2])))
        @test enclosed(box,(Q(stop[1]),Q(stop[2])))
        for p in ((r,0.),(0.,r),(-r,0.),(0.,-r))
            D._artwork_contains(arc,p...)&&@test enclosed(box,(Q(p[1]),Q(p[2])))
        end
    end
    for scale in (1e-200,1e-160,1.,1e160,1e200)
        transformed=D._artwork_transform(D._artwork_rectangle(1.,1.);rotation=37.,scale)
        @test all(isfinite,transformed.inverse)
        @test any(!iszero,transformed.inverse)
        @test D._artwork_contains(transformed,0.,0.)
        @test !D._artwork_contains(transformed,2scale,0.)
        @test all(isfinite,D._artwork_bounds(transformed))
    end
    for rotation in (0.,37.,108.0485313789156,180.,270.),scale in (1e-50,.5,1.,2.,65.21480961783402,1e50)
        transformed=D._artwork_transform(D._artwork_rectangle(1.,1.);rotation,scale)
        @test isequal(transformed.inverse,inv(transformed.matrix))
    end
    @test_throws ArgumentError D._artwork_transform(rect;scale=nextfloat(0.))
    for (sx,sy) in ((1e-200,1e-200),(1e200,1e200),(1e200,1e-200),(1e-200,-1e200))
        transformed=D._artwork_affine(D._ArtworkCircle((0.,0.),.5),sx,sy,(0.,0.))
        @test all(isfinite,transformed.inverse)
        @test !iszero(transformed.inverse[1,1])&&!iszero(transformed.inverse[2,2])
        @test D._artwork_contains(transformed,0.,0.)
        @test !D._artwork_contains(transformed,2sx,0.)
        @test !D._artwork_contains(transformed,0.,2sy)
        @test all(isfinite,D._artwork_bounds(transformed))
        for p in inverse_corners(transformed,(-.5,.5,-.5,.5))
            @test enclosed(D._artwork_bounds(transformed),p)
        end
    end
end

@testset "Dyadic Region point shortcut matches exact rational predicates" begin
    for k in (1e-200,1e-160,1.,1e160,1e200)
        circle=D._ArtworkRegion(Union{D._ArtworkRoundLine,D._ArtworkRoundArc}[
            D._ArtworkRoundArc((.125k,0.),(.125k,0.),(0.,0.),false,0.)])
        lens=D._ArtworkRegion(Union{D._ArtworkRoundLine,D._ArtworkRoundArc}[
            D._ArtworkRoundArc((.125k,0.),(0.,.125k),(0.,0.),false,0.),
            D._ArtworkRoundLine((0.,.125k),(.125k,0.),0.)])
        for leaf in (circle,lens),x in (-.25k,-.125k,-.0625k,0.,.0625k,.125k,.25k),
            y in (-.25k,-.125k,-.0625k,0.,.0625k,.125k,.25k)
            @test D._artwork_exact_region_point(leaf,x,y)==D._artwork_exact_qcontains(leaf,(Q(x),Q(y)))
        end
    end
    region=D._odb_extra_standard_symbol("hplate6000x4000x1000xra200xro100",1e-6)
    for p in ((-.00284375,-.00195625),(-.00293125,-.00186875),
              (-.00293125,.0018687500000000002),(-.00284375,.00195625))
        @test D._artwork_exact_region_point(region,p...)==D._artwork_exact_qcontains(region,(Q(p[1]),Q(p[2])))
    end
end

@testset "ODB leaf coverage, algebraic ordering and bounded fallback" begin
    r=.125;circle=D._ArtworkCircle((0.,0.),r)
    for shape in (D._ArtworkODBDrill(circle,:plated,0.,0.),D._ArtworkODBButterfly(circle),
                  D._odb_corner_rectangle(.25,.25,.03125),D._odb_corner_rectangle(.25,.25,.03125;chamfer=true),
                  D._ArtworkODBNull(3),D._ArtworkEmpty())
        for x in (-.25,-r,0.,r,.25),y in (-r,0.,r)
            result=typedquery(stroke(shape,(-.25,0.),(.25,0.)),x,y)
            @test result==D._artwork_exact_algebraic_event_stroke(shape,(-.25,0.),(.25,0.),x,y).hit
        end
    end
    A=D._ArtworkQuadratic
    @test A(Q(0),Q(1),Q(2))<A(Q(0),Q(1),Q(3))
    @test A(Q(0),Q(1),Q(2))==A(Q(0),Q(1)/2,Q(8))
    @test A(Q(3),-Q(1),Q(2))>A(Q(0),Q(1),Q(2))
    @test A(Q(2),-Q(1),Q(2))<A(Q(0),Q(1),Q(2))
    tri=D._ArtworkPolygon([0. .125 0.;0. 0. .125])
    low=D._ArtworkCompositeLine(tri,(0.,0.),(.375,.375),6,1)
    typedquery(low,1.,1.) # certain absence needs no exact workspace
    @test_throws ArgumentError typedquery(low,.25,.375)
    @test (@allocated try typedquery(low,.25,.375) catch;end)<100000
    @test_throws ArgumentError D._artwork_exact_rational_separator(A(Q(0)),A(Q(1)/BigInt(2)^100),1)
    huge=D._ArtworkCircle((0.,0.),1e308)
    @test D._artwork_contains(huge,1e308,0.)
    @test !D._artwork_contains(huge,1.1e308,0.)
    tiny=D._ArtworkCircle((0.,0.),nextfloat(0.))
    @test D._artwork_contains(tiny,nextfloat(0.),0.)
    @test !D._artwork_contains(tiny,nextfloat(nextfloat(0.)),0.)
    @test !typedquery(stroke(circle,(0.,0.),(.1,0.)),NaN,0.)
    @test !typedquery(stroke(circle,(0.,0.),(.1,0.)),Inf,0.)
    typical=stroke(D._ArtworkComposite(((true,D._artwork_rectangle(2.,2.)),(false,D._artwork_rectangle(1.,1.)))),(.125,-.375),(.7,.3))
    typedquery(typical,.131,.227)
    @test (@allocated typedquery(typical,.131,.227))==0
end
end
