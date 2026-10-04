using DiffMoM,Test,LinearAlgebra

function native_conformal_project(dir;ground=false,tiny=false)
    polygons="""0 5 -1 N 9 1 1 100 100 0 0 0 Y
0 .375
1 .25
1 .75
0 .625
0 .375
END
"""
    extra=tiny ? """0 4 -1 N 10 1 1 100 100 0 0 0 Y
.6 .85
.601 .85
.6 .851
.6 .85
END
""" : ground ? """VIA POLYGON
0 5 -1 N 10 1 1 100 100 0 0 0 Y
TOLEVEL GND SOLID NOCOVERS
.375 .375
.625 .375
.625 .625
.375 .625
.375 .375
END
""" : ""
    text="""FTYP SONPROJ 19
VER "18.53"
DIM
LNG MM
FREQ GHZ
END DIM
CONTROL
OPTIONS
END CONTROL
GEO
TMET "PEC" 0 SUP 0 0 0 0
BMET "PEC" 0 SUP 0 0 0 0
BOX 1 1 1 16 16 100 0
.5 1 1 0 0 0 2 "Air"
.5 1 1 0 0 0 2 "Air"
LORGN 0 1 U
POR1 BOX
POLY 9 1
3
1 50 0 0 0 0 .5
POR1 BOX
POLY 9 1
1
2 50 0 0 0 1 .5
NUM $(isempty(extra) ? 1 : 2)
$polygons$extra
END GEO
"""
    path=joinpath(dir,"genuine.son");write(path,text);read_sonnet_project(path)
end

@testset "native genuine sheets preserve sloped and tiny geometry" begin
    mktempdir() do dir
        p=native_conformal_project(dir);sizes=(edge_size=.15e-3,interior_size=.25e-3,edge_band=.05e-3)
        model=sonnet_conformal_layout(p;freq=10e9,sizes...)
        c=model.layout.problem
        @test c isa PlanarConformalProblem
        @test sum(c.mesh.areas)≈.375e-6 rtol=1e-14
        @test model.labels==[1,2]
        @test model.contraction==Matrix{Float64}(I,2,2)
        @test [port.span for port in c.ports]==[(.375e-3,.625e-3),(.25e-3,.75e-3)]
        r=solve_sonnet_conformal(p,10e9;raw=true,sizes...,mx=32,my=32)
        @test r.raw isa PlanarConformalResult
        @test r.y≈r.raw.y rtol=1e-14
        @test r.s≈transpose(r.s) rtol=1e-10
        @test opnorm(r.s)<=1+1e-10
        tiny=native_conformal_project(dir;tiny=true)
        @test_throws ArgumentError sonnet_planar_problem(tiny)
        preserved=sonnet_conformal_layout(tiny;sizes...)
        @test sum(preserved.layout.problem.mesh.areas)≈.375e-6+.5e-12 rtol=1e-14
        @test_throws ArgumentError sonnet_conformal_layout(p;sizes...,max_bytes=1)
        @test_throws ArgumentError solve_sonnet_conformal(p,10e9;sizes...,max_bytes=1)
        # BOX annotations do not choose the physical source edge. The kind
        # and referenced edge retain their native meaning.
        function ports_project(ports;polygons=p.polygons,box=p.box,layers=p.layers)
            SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,box,layers,p.metals,p.top,p.bottom,
                polygons,ports,p.variables,p.components,p.sweeps,p.records)
        end
        ps=first(p.ports)
        @test_throws ArgumentError sonnet_conformal_layout(ports_project([
            SonnetPortSpec(:via,ps.polygon,ps.edge,ps.number,ps.values,ps.records)]);sizes...)
        values=copy(ps.values);values[7]=".9"
        annotated=ports_project([SonnetPortSpec(ps.kind,ps.polygon,ps.edge,ps.number,values,ps.records),last(p.ports)])
        amended=sonnet_conformal_layout(annotated;sizes...)
        @test [port.span for port in amended.layout.problem.ports]==[port.span for port in c.ports]
        @test first(annotated.ports).values[7]==".9"
        # Native negative labels define a floating reference. They do not
        # impose arbitrary equal magnitudes of the two ground voltages.
        east=last(p.ports);negative=SonnetPortSpec(east.kind,east.polygon,east.edge,-1,east.values,east.records)
        balanced=ports_project([ps,negative]);bm=sonnet_conformal_layout(balanced;sizes...)
        @test bm.contraction==fill(.5,2,1)
        @test bm.floating_common==reshape([1.,-1.],2,1)
        br=solve_sonnet_conformal(balanced,1e9;raw=true,sizes...,mx=32,my=32)
        @test size(br.y)==(1,1)
        @test abs(only(transpose(bm.floating_common)*br.raw.y*br.voltage_transfer))<1e-12
        @test br.y≈transpose(bm.contraction)*br.raw.y*br.voltage_transfer rtol=1e-13
        @test opnorm(br.s)<=1+1e-10
        @test br.z0==[50.]
        native_maps=planar_current_maps(br)
        raw_maps=planar_conformal_current_maps(br.raw;voltages=vec(br.voltage_transfer))
        @test length(native_maps)==length(br.raw.problem.mesh.interfaces)
        @test all(native_maps[i].current≈raw_maps[i].current for i in eachindex(native_maps))
        waves=ComplexF64[.5im];wv=sqrt.(br.z0).*(waves+br.s*waves)
        wave_maps=planar_current_maps(br;incident_waves=waves)
        raw_wave_maps=planar_conformal_current_maps(br.raw;voltages=br.voltage_transfer*wv)
        @test all(wave_maps[i].current≈raw_wave_maps[i].current for i in eachindex(native_maps))
        @test_throws ArgumentError sonnet_conformal_layout(ports_project([negative]);sizes...)
        # Native reference capacitance is in parallel with R+jX+jωL.
        # The genuine adapter retains this distinct native topology.
        rlcvalues=copy(ps.values);rlcvalues[2:5]=["40","-10","1.2",".2"]
        rlcport=SonnetPortSpec(ps.kind,ps.polygon,ps.edge,ps.number,rlcvalues,ps.records)
        rlcproject=ports_project([rlcport,east]);rfreq=2e9
        series=40+im*(-10+2pi*rfreq*1.2e-9)
        rlc_expected=inv(inv(series)+im*2pi*rfreq*.2e-12)
        rm=sonnet_conformal_layout(rlcproject;freq=rfreq,sizes...)
        @test rm.z0≈[rlc_expected,50.] rtol=2e-14
        rr=solve_sonnet_conformal(rlcproject,rfreq;raw=true,sizes...,mx=24,my=24)
        @test rr.z0≈rm.z0 rtol=2e-14
        @test rr.s≈planar_y_to_s(rr.y,rm.z0) rtol=2e-13
        # Grounded bulk contact is constrained in the same genuine tapered
        # sheet, rather than rasterizing the triangle into its via cells.
        grounded=native_conformal_project(dir;ground=true)
        g=solve_sonnet_conformal(grounded,1e8;raw=true,sizes...,mx=32,my=32)
        @test g.raw isa PlanarHybridResult
        @test abs(g.s[2,1])<.03
        @test opnorm(g.s)<=1+1e-10
        # Native adjacent triangles form one genuine sheet with an actual
        # diagonal source constraint, not an axis-aligned replacement.
        va=[.25e-3 .75e-3 .75e-3 .25e-3;.25e-3 .25e-3 .75e-3 .25e-3]
        vb=[.25e-3 .75e-3 .25e-3 .25e-3;.25e-3 .75e-3 .75e-3 .25e-3]
        polys=[SonnetPolygon(:sheet,0,-1,9,va,"","",String[]),SonnetPolygon(:sheet,0,-1,10,vb,"","",String[])]
        gap=SonnetPortSpec(:gap,9,2,1,["1","50","0","0","0",".5",".5"],SonnetRecord[])
        diagonal=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,p.metals,p.top,p.bottom,
            polys,[gap],p.variables,p.components,p.sweeps,p.records)
        d=sonnet_conformal_layout(diagonal;sizes...)
        @test only(d.layout.problem.ports).wall===:internal
        @test sum(d.layout.problem.basis.width[d.layout.problem.basis.port.==1])≈sqrt(2)*.5e-3 rtol=1e-14
        dr=solve_sonnet_conformal(diagonal,1e7;raw=true,sizes...,mx=24,my=24)
        @test imag(dr.y[1,1])>0
        @test abs(dr.s[1,1])≈1 rtol=1e-10
        # Sheet-free native axial sources retain their established bulk
        # route; the genuine-sheet adapter adds no artificial face sheet.
        axial=SonnetPortSpec(:via,10,0,1,["1","50","0","0","0",".5",".5"],SonnetRecord[])
        onlyvia=SonnetProject(grounded.source,grounded.units,grounded.length_scale,grounded.frequency_scale,grounded.box,grounded.layers,
            grounded.metals,grounded.top,grounded.bottom,[last(grounded.polygons)],[axial],grounded.variables,grounded.components,grounded.sweeps,grounded.records)
        v=solve_sonnet_conformal(onlyvia,1e9;raw=true,sizes...,mx=24,my=24)
        @test v.raw isa PlanarResult
        @test isempty(v.raw.problem.sheets)
        @test all(isfinite,v.y)
        # A standard axial source is anchored at a physical cover;
        # an interlevel source requires the explicit via-port kind.
        inter=SonnetPolygon(:via,0,-1,10,last(grounded.polygons).vertices,"1","",["SOLID","NOCOVERS"])
        box=copy(p.box);box[1]="2";layers=vcat(p.layers,[copy(first(p.layers))])
        std=SonnetPortSpec(:std,10,0,1,axial.values,axial.records)
        @test_throws ArgumentError sonnet_conformal_layout(ports_project([std];polygons=[inter],box,layers);sizes...)
    end
end

@testset "explicit conformal PEC wall ground contacts" begin
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,1e-3,1e-3)
    mesh=planar_conformal_mesh([0. 1e-3 1e-3 0.;.375e-3 .375e-3 .625e-3 .625e-3];edge_size=.15e-3,interior_size=.3e-3)
    port=PlanarConformalPort(1,:west,(.375e-3,.625e-3));contact=PlanarConformalPort(1,:east,(.375e-3,.625e-3))
    open=PlanarConformalProblem(stack,mesh,[port]);short=PlanarConformalProblem(stack,mesh,[port];wall_contacts=[contact])
    @test count(==(0),short.basis.port)>count(==(0),open.basis.port)
    ro=solve_planar_conformal(open,1e7;mx=24,my=24);rs=solve_planar_conformal(short,1e7;mx=24,my=24)
    @test imag(ro.y[1,1])>0 && imag(rs.y[1,1])<0
    @test abs(rs.y[1,1])>100abs(ro.y[1,1])
    @test_throws ArgumentError PlanarConformalProblem(stack,mesh,[port];sidewalls=WALL_PMC,wall_contacts=[contact])
end
