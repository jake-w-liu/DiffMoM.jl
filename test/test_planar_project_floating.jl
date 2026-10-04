using DiffMoM,Test,LinearAlgebra

function _floating_project(;resistance=.2,reverse=false)
    return planar_project_from_dict(Dict("project"=>Dict("unit"=>"mm"),
        "box"=>Dict("width"=>6.,"length"=>4.),"mesh"=>Dict("nx"=>12,"ny"=>8),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5),Dict("thickness"=>.5)]),
        "metals"=>Dict("film"=>Dict("type"=>"surface_impedance","rs"=>resistance)),
        "polygons"=>[Dict("name"=>"reference","level"=>1,"metal"=>"film","net"=>"local_G",
                "vertices"=>[[.5,1.5],[2.5,1.5],[2.5,2.5],[.5,2.5]]),
            Dict("name"=>"signal","level"=>1,"metal"=>"film","net"=>"S",
                "vertices"=>[[3.5,1.5],[5.5,1.5],[5.5,2.5],[3.5,2.5]])],
        "ports"=>[Dict("type"=>"floating","name"=>"diff","number"=>1,"polygon"=>"signal","edge"=>4,
            "ref_polygon"=>"reference","ref_edge"=>2,"source_edge"=>6,"polarity"=>reverse ? -1 : 1)]))
end

@testset "project floating source and loaded differential coordinate" begin
    p=_floating_project();model=planar_project_layout(p;freq=1e9)
    @test model.layout.problem.ports[1].wall===:internal_x
    @test isempty(model.layout.problem.vias)
    @test isempty(model.layout.source_problem.vias)
    @test model.layout.contraction==ones(1,1)
    @test only(model.layout.terminal_paths).kind===:floating_bridge
    @test only(model.layout.terminal_paths).length≈.001
    @test only(model.layout.terminal_paths).metal=="film"
    @test count(model.layout.problem.sheets[1].mask)==20
    @test all(model.layout.sheet_materials[1][6:7,4:5].>0)
    @test all(iszero,model.layout.problem.sheets[1].connect_west)
    @test all(iszero,model.layout.problem.sheets[1].connect_east)
    r=solve_planar_project(p,1e9;mx=36,my=24)
    @test opnorm(r.s)<=1+1e-10
    # Build the two pads and impressed field independently from project
    # lowering. Apply the same sheet resistance to pads and bridge.
    g=CellGrid(.006,.004,12,8)
    st=PlanarStackup([PlanarLayer(1.,1.,.0005),PlanarLayer(1.,1.,.0005)],TERM_GND,TERM_GND,g.a,g.b)
    sh=sheet_level(1,12,8);rasterize_rect!(sh,g,.0005,.0025,.0015,.0025)
    rasterize_rect!(sh,g,.0035,.0055,.0015,.0025)
    bare=PlanarProblem(st,g,[sh],PlanarPort[],build_planar_basis(g,[sh],PlanarPort[]))
    sig=PlanarPort(1,:terminal_x,7,4:5,50.;metal_side=:positive)
    ref=PlanarPort(1,:terminal_x,5,4:5,50.;metal_side=:negative)
    physical=planar_floating_bridge(bare,sig,ref;source_edge=6)
    direct=solve_planar(physical.problem,1e9;mx=36,my=24,surface_zs=.2)
    @test r.y≈direct.y rtol=1e-11
    @test r.em.currents≈direct.currents rtol=1e-10
    reversed=solve_planar_project(_floating_project(reverse=true),1e9;mx=36,my=24)
    @test reversed.s≈r.s rtol=1e-12
    @test reversed.em.currents≈-r.em.currents rtol=1e-11
    p.data["components"]=[Dict("name"=>"load","type"=>"resistor","value"=>"100 ohm","ports"=>[1,0])]
    p.data["ports"][1]["external"]=true
    loaded=solve_planar_project(p,1e9;mx=36,my=24)
    @test loaded.y≈r.y.+.01 rtol=1e-10
    maps=planar_current_maps(loaded;incident_waves=[1.])
    manual=planar_current_maps(loaded.em;voltages=transpose(loaded.model.source_incidence)*loaded.circuit.voltages[:,1])
    @test maps[1].jx≈manual[1].jx rtol=1e-12
    p.data["ports"][1]["refplane"]=Dict("length"=>".1 mm","zc"=>50.,"gamma"=>"=complex(0,freq/1e8)")
    @test solve_planar_project(p,1e9;mx=36,my=24).em isa PlanarCalibratedResult
    @test_throws ArgumentError planar_project_layout(p;max_bytes=1)
    invalid=_floating_project();invalid.data["ports"][1]["ref_edge"]=4
    @test_throws ArgumentError planar_project_layout(invalid)
    invalid=_floating_project();invalid.data["ports"][1]["source_edge"]=2
    @test_throws ArgumentError planar_project_layout(invalid)
    invalid=_floating_project();invalid.data["ports"][1]["bridge_metal"]="undefined"
    @test_throws ArgumentError solve_planar_project(invalid,1e9;mx=12,my=8)
end
