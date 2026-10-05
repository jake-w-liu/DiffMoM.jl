using DiffMoM,Test,LinearAlgebra

function genuine_layout_polygon(name,vertices;level=1,metal="pec")
    v,b=planar_normalize_polygon(vertices)
    PlanarPolygon(name,level,metal,"",v,b)
end

function independent_convex_intersection_area(a,b)
    points=Tuple.(a.vertices)
    cross(p,q,r)=(q[1]-p[1])*(r[2]-p[2])-(q[2]-p[2])*(r[1]-p[1])
    for j in eachindex(b.vertices)
        q,r=Tuple(b.vertices[j]),Tuple(b.vertices[mod1(j+1,length(b.vertices))]);output=Tuple{Float64,Float64}[]
        isempty(points) && return 0.
        previous=last(points);dprevious=cross(q,r,previous)
        for current in points
            dcurrent=cross(q,r,current)
            if (dcurrent>=0)!=(dprevious>=0)
                t=dprevious/(dprevious-dcurrent)
                push!(output,(previous[1]+t*(current[1]-previous[1]),previous[2]+t*(current[2]-previous[2])))
            end
            dcurrent>=0 && push!(output,current)
            previous=current;dprevious=dcurrent
        end
        points=output
    end
    abs(sum(points[i][1]*points[mod1(i+1,length(points))][2]-points[i][2]*points[mod1(i+1,length(points))][1] for i in eachindex(points)))/2
end

@testset "exact polygon union holes constraints and material seams" begin
    rect(name,x0,x1,y0,y1;kw...)=genuine_layout_polygon(name,[(x0,y0),(x1,y0),(x1,y1),(x0,y1)];kw...)
    opts=(edge_size=.6,interior_size=1.,edge_band=.2)
    polygons=[rect("left",0.,2.,0.,2.),rect("right",1.,3.,1.,3.)]
    mesh=planar_conformal_mesh(polygons;opts...)
    @test sum(mesh.areas)≈7. rtol=1e-14
    hole=rect("void",.5,1.5,.5,1.5)
    hollow=planar_conformal_mesh([rect("outer",0.,2.,0.,2.)];holes=[hole],opts...)
    @test sum(hollow.areas)≈3. rtol=1e-14
    @test all(!( .5<sum(hollow.vertices[1,hollow.triangles[:,t]])/3<1.5 && .5<sum(hollow.vertices[2,hollow.triangles[:,t]])/3<1.5) for t in eachindex(hollow.areas))
    # Union of actual sloped quadrilaterals with intersecting boundaries.
    crossed=[genuine_layout_polygon("a",[(0.,0.),(2.,0.),(2.,1.),(0.,2.)]),
        genuine_layout_polygon("b",[(0.,1.),(2.,2.),(2.,3.),(0.,3.)])]
    crossing=planar_conformal_mesh(crossed;opts...)
    @test sum(crossing.areas)≈5.5 rtol=1e-14
    # Reproduces differing Float64 line evaluation at the same oblique
    # intersection. Both incident edges must use one canonical vertex.
    arbitrary=[genuine_layout_polygon("a",[(.13,.11),(1.81,.32),(.35,1.77)]),
        genuine_layout_polygon("b",[(.61,.21),(1.95,1.37),(.16,1.63)])]
    exact=planar_conformal_mesh(arbitrary;opts...)
    area(p)=abs(sum(p.vertices[i][1]*p.vertices[mod1(i+1,length(p.vertices))][2]-p.vertices[i][2]*p.vertices[mod1(i+1,length(p.vertices))][1] for i in eachindex(p.vertices)))/2
    @test sum(exact.areas)≈sum(area,arbitrary)-independent_convex_intersection_area(arbitrary...) rtol=2e-14
    stack=PlanarStackup([PlanarLayer(1.,1.,.5),PlanarLayer(1.,1.,.5)],TERM_GND,TERM_GND,2.,2.)
    cut=PlanarConformalPort(1,:internal,((0.,0.),(2.,2.)))
    layout=build_planar_conformal_layout(stack,[rect("full",0.,2.,0.,2.)],[cut];opts...)
    @test count(==(1),layout.problem.basis.port)>1
    @test sum(layout.problem.basis.width[layout.problem.basis.port.==1])≈sqrt(8.) rtol=1e-14
    material=[rect("copper",0.,1.,0.,2.;metal="a"),rect("nickel",1.,2.,0.,2.;metal="b")]
    ports=[PlanarConformalPort(1,:west,(0.,2.)),PlanarConformalPort(1,:east,(0.,2.))]
    layered=build_planar_conformal_layout(stack,material,ports;metals=Dict("a"=>1.,"b"=>f->2.),opts...)
    @test Set(layered.material_names)==Set(("a","b"))
    @test sum(layered.problem.mesh.areas[layered.triangle_materials.==1])≈2.
    @test sum(layered.problem.mesh.areas[layered.triangle_materials.==2])≈2.
    @test_throws ArgumentError planar_conformal_mesh([rect("a",0.,2.,0.,2.;metal="a"),rect("b",1.,3.,1.,3.;metal="b")];opts...)
    @test_throws ArgumentError planar_conformal_mesh(polygons;opts...,max_bytes=1)
    @test_throws ArgumentError planar_conformal_mesh(polygons;opts...,max_triangles=3)
    @test_throws ArgumentError build_planar_conformal_layout(stack,material,ports;opts...)
end

@testset "genuine material layout DC solve and constrained bulk contact" begin
    a,b=2e-3,2e-3;stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    polys=[genuine_layout_polygon("left",[(0.,.5e-3),(a/2,.5e-3),(a/2,1.5e-3),(0.,1.5e-3)];metal="a"),
        genuine_layout_polygon("right",[(a/2,.5e-3),(a,.5e-3),(a,1.5e-3),(a/2,1.5e-3)];metal="b")]
    ports=[PlanarConformalPort(1,:west,(.5e-3,1.5e-3)),PlanarConformalPort(1,:east,(.5e-3,1.5e-3))]
    kw=(edge_size=.6e-3,interior_size=.8e-3,edge_band=.1e-3)
    layout=build_planar_conformal_layout(stack,polys,ports;metals=Dict("a"=>1.,"b"=>2.),kw...)
    r=solve_planar(layout,1e6;mx=24,my=24)
    @test real(2/(r.y[1,1]-r.y[1,2]))≈3. rtol=5e-6
    @test opnorm(r.s)<=1+1e-12
    @test_throws ArgumentError solve_planar(layout,1e6;surface_zs=0.)
    grid=CellGrid(a,b,4,4);via=via_level(1,4,4);via.uni[2,2:3].=true;via.tap[2,2:3].=true
    hybrid=build_planar_conformal_layout(stack,polys,ports;metals=Dict("a"=>0.,"b"=>0.),bulk_grid=grid,vias=[via],kw...)
    short=solve_planar(hybrid,1e8;mx=24,my=24)
    @test hybrid.problem isa PlanarHybridProblem
    @test abs(short.s[2,1])<.04
    @test opnorm(short.s)<=1+1e-10
end

@testset "exact box intersection preserves disconnected geometry and materials" begin
    rect(name,x0,x1,y0,y1;kw...)=genuine_layout_polygon(name,[(x0,y0),(x1,y0),(x1,y1),(x0,y1)];kw...)
    opts=(edge_size=.4,interior_size=.6,edge_band=.1)
    box=rect("box",0.,2.,0.,2.)
    source=rect("cover",-1.,3.,-1.,3.)
    mesh=planar_conformal_mesh([source];clip_box=(2.,2.),opts...)
    @test sum(mesh.areas)≈4. rtol=1e-14
    @test all(x->0<=x<=2.,mesh.vertices)
    triangle=genuine_layout_polygon("oblique",[(-1.,.25),(3.,.5),(1.,3.)])
    clipped=planar_conformal_mesh([triangle];clip_box=(2.,2.),opts...)
    @test sum(clipped.areas)≈independent_convex_intersection_area(triangle,box) rtol=2e-14
    @test all(x->0<=x<=2.,clipped.vertices)
    split=genuine_layout_polygon("split",[(-1.,.25),(2.5,.25),(2.5,.5),(-.5,.5),
        (-.5,1.5),(2.5,1.5),(2.5,1.75),(-1.,1.75)])
    pieces=planar_conformal_mesh([split];clip_box=(2.,2.),opts...)
    @test sum(pieces.areas)≈1. rtol=2e-14
    @test all(t->begin
        ys=pieces.vertices[2,pieces.triangles[:,t]]
        maximum(ys)<=.5 || minimum(ys)>=1.5
    end,eachindex(pieces.interfaces))
    hole=rect("void",-1.,1.,.5,1.5)
    hollow=planar_conformal_mesh([source];holes=[hole],clip_box=(2.,2.),opts...)
    @test sum(hollow.areas)≈3. rtol=1e-14
    # Overlap outside the box must not create an in-box material conflict.
    outside=rect("outside",-2.,-1.,0.,2.;metal="other")
    ignored=planar_conformal_mesh([rect("inside",-2.,.5,0.,2.),outside];clip_box=(2.,2.),opts...)
    @test sum(ignored.areas)≈1. rtol=1e-14
    empty=planar_conformal_mesh([outside];clip_box=(2.,2.),allow_empty=true,opts...)
    @test size(empty.vertices)==(2,0) && size(empty.triangles)==(3,0)
    @test isempty(empty.areas) && isempty(empty.interfaces)
    @test_throws ArgumentError planar_conformal_mesh([outside];clip_box=(2.,2.),opts...)
    for invalid in ((0.,2.),(-1.,2.),(Inf,2.),(NaN,2.),(big"1e1000",2.))
        @test_throws ArgumentError planar_conformal_mesh([source];clip_box=invalid,opts...)
    end
    @test_throws ArgumentError planar_conformal_mesh([source];clip_box=(2.,2.),max_bytes=1,opts...)
    @test_throws ArgumentError planar_conformal_mesh([source];clip_box=(2.,2.),max_triangles=1,opts...)
    stack=PlanarStackup([PlanarLayer(1.,1.,.5),PlanarLayer(1.,1.,.5)],TERM_GND,TERM_GND,2.,2.)
    ports=[PlanarConformalPort(1,:west,(0.,2.)),PlanarConformalPort(1,:east,(0.,2.))]
    @test_throws ArgumentError build_planar_conformal_layout(stack,[source],ports;opts...)
    material=[rect("left",-1.,1.,0.,2.;metal="a"),rect("right",1.,3.,0.,2.;metal="b")]
    layout=build_planar_conformal_layout(stack,material,ports;clip_to_box=true,metals=Dict("a"=>1.,"b"=>2.),opts...)
    @test layout.polygons==material
    @test sum(layout.problem.mesh.areas[layout.triangle_materials.==1])≈2. rtol=1e-14
    @test sum(layout.problem.mesh.areas[layout.triangle_materials.==2])≈2. rtol=1e-14
    @test_throws ArgumentError build_planar_conformal_layout(stack,[source],ports;clip_to_box=true,max_bytes=1,opts...)
    grid=CellGrid(2.,2.,4,4);via=via_level(1,4,4);via.uni[2,2]=true;via.tap[2,2]=true
    bulk=build_planar_conformal_layout(stack,[outside],PlanarConformalPort[];
        clip_to_box=true,bulk_grid=grid,vias=[via],bulk_ports=[PlanarPort(1,:via,6:6,50.)],
        metals=Dict("other"=>0.),opts...)
    @test bulk.problem isa PlanarProblem && isempty(bulk.triangle_materials)
    @test isempty(bulk.materials) && planar_basis_count(bulk.problem.basis)==2
    @test_throws ArgumentError build_planar_conformal_layout(stack,[outside],ports;
        clip_to_box=true,bulk_grid=grid,vias=[via],bulk_ports=[PlanarPort(1,:via,6:6,50.)],
        metals=Dict("other"=>0.),opts...)
    # Axis intersections keep exact plane coordinates rather than a residual
    # from finite-precision division of the oblique line parameter.
    point=DiffMoM._planar_conformal_cross_point((-.3,-.1),(.6,.2),(-1.,0.),(1.,0.))
    @test point!==nothing && point[2]===0.
end
