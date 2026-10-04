using DiffMoM,Test

function rounding_contact_layout(rep;volume=false)
    g=CellGrid(1e-3,1e-3,20,20)
    x(i)=rep===:normalized ? g.a*(i/g.nx) : i*g.dx
    y(i)=rep===:normalized ? g.b*(i/g.ny) : i*g.dy
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3) for _ in 1:3],TERM_GND,TERM_GND,g.a,g.b)
    function rectangle(name,level,xl,xr)
        v,b=planar_normalize_polygon([(xl,y(6)),(xr,y(6)),(xr,y(14)),(xl,y(14))])
        PlanarPolygon(name,level,"pec","",v,b)
    end
    polygons=[rectangle("left",2,0.,x(14)),rectangle("right",1,x(6),g.a)]
    ports=[PlanarConformalPort(2,:west,(y(6),y(14))),PlanarConformalPort(1,:east,(y(6),y(14)))]
    original=[copy(p.vertices) for p in polygons]
    via=via_level(2,20,20);via.uni[7:14,[7,14]].=true;via.uni[[7,14],7:14].=true
    vol=vol_level(2,20,20);vol.mask.=via.uni
    models=volume ? (;vols=[vol]) : (;vias=[via])
    layout=build_planar_conformal_layout(stack,polygons,ports;bulk_grid=g,models...,
        edge_size=.2e-3,interior_size=.4e-3,edge_band=0.,max_bytes=64_000_000)
    @test all(p.vertices==v for (p,v) in zip(polygons,original))
    layout,polygons,g
end

@testset "genuine bulk contacts preserve source coordinate rounding" begin
    G=DiffMoM
    for rep in (:normalized,:product),volume in (false,true)
        layout,polygons,g=rounding_contact_layout(rep;volume)
        mesh=layout.problem.conformal.mesh
        @test layout.problem isa PlanarHybridProblem
        @test all(>(0),mesh.areas)
        @test all(p->all(vertex->any(i->mesh.vertices[1,i]==vertex[1] && mesh.vertices[2,i]==vertex[2],axes(mesh.vertices,2)),p.vertices),polygons)
        for level in (1,2)
            p=only(filter(p->p.level==level,polygons));xs=extrema(v[1] for v in p.vertices);ys=extrema(v[2] for v in p.vertices)
            ids=findall(==(level),mesh.interfaces)
            @test sum(mesh.areas[ids])≈(xs[2]-xs[1])*(ys[2]-ys[1]) rtol=3e-14
            for (kx,ky) in ((0.,0.),(1234.,-5667.),(3e4,2e4))
                integral=sum(G._planar_pulse_and_first_moments(view(mesh.vertices,:,mesh.triangles[:,t]),kx,ky)[1] for t in ids)
                axis(lo,hi,k)=(hi-lo)*cis(k*(lo+hi)/2)*sinc(k*(hi-lo)/(2pi))
                @test integral≈axis(xs...,kx)*axis(ys...,ky) rtol=5e-13 atol=1e-20
            end
        end
        @test_throws ArgumentError build_planar_conformal_layout(layout.problem.stack,polygons,layout.problem.conformal.ports;
            bulk_grid=g,vias=layout.problem.bulk.vias,vols=layout.problem.bulk.vols,
            edge_size=.2e-3,interior_size=.4e-3,edge_band=0.,max_bytes=1)
    end
    value=.0003;near=nextfloat(value);distant=nextfloat(near)
    # A unique source coordinate wins, even when it uses the other rounding.
    polygon=(;level=1,vertices=[(near,0.),(near,1.)])
    @test G._planar_conformal_contact_coordinate(value,1,1,[polygon],[],Tuple[],0)==near
    @test G._planar_conformal_contact_coordinate(value,2,1,[polygon],[],Tuple[],0)==value
    @test G._planar_conformal_contact_coordinate(value,1,1,[(;level=1,vertices=[(distant,0.)])],[],Tuple[],0)==value
    # Distinct physical coordinates are never fused.
    ambiguous=(;level=1,vertices=[(value,0.),(near,1.)])
    @test_throws ArgumentError G._planar_conformal_contact_coordinate(value,1,1,[ambiguous],[],Tuple[],0)
    cut=[(1,((near,0.),(near,1.)))]
    @test G._planar_conformal_contact_coordinate(value,1,1,[],[],cut,1)==near
    @test_throws ArgumentError G._planar_conformal_contact_coordinate(value,1,1,[(;level=1,vertices=[(value,0.)])],[],cut,1)
end
