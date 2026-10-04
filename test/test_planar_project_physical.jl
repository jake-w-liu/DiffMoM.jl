using DiffMoM,Test,LinearAlgebra

function _physical_project(kind;direction="up",subdivisions=1)
    return planar_project_from_dict(Dict(
        "project"=>Dict("unit"=>"mm"),"box"=>Dict("width"=>3.,"length"=>2.),
        "mesh"=>Dict("nx"=>6,"ny"=>4),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5),Dict("thickness"=>.5)]),
        "metals"=>Dict("wire"=>Dict("type"=>kind,"conductivity"=>"1e4 S/m",
            "thickness"=>"20 um","direction"=>direction,"axial_subdivisions"=>subdivisions)),
        "polygons"=>[Dict("name"=>"bar","level"=>1,"metal"=>"wire",
            "vertices"=>[[0.,.5],[3.,.5],[3.,1.5],[0.,1.5]])],
        "ports"=>[Dict("name"=>"a","polygon"=>"bar","edge"=>4),
            Dict("name"=>"b","polygon"=>"bar","edge"=>2)]))
end

@testset "project complex passive dispersive materials" begin
    p=_physical_project("sheet")
    p.data["stackup"]["layers"][1]["eps_r"]="=2+3/(1+complex(0,2*pi*freq*1e-10))"
    p.data["stackup"]["layers"][1]["eps_r_z"]="=2+2/(1+complex(0,2*pi*freq*2e-10))"
    p.data["stackup"]["layers"][1]["mu_r"]="=1.5+complex(0,-.01*freq/1e9)"
    p.data["stackup"]["layers"][1]["tan_delta"]=.02
    p.data["stackup"]["layers"][1]["tan_delta_z"]=.01
    p.data["stackup"]["layers"][1]["conductivity"]=".001 S/m"
    f=1e9;layer=planar_project_layout(p;freq=f).layout.problem.stack.layers[1]
    transverse=2+3/(1+im*2pi*f*1e-10)
    axial=2+2/(1+im*2pi*f*2e-10)
    @test layer.epsr≈transverse-im*real(transverse)*.02-im*.001/(2pi*f*DiffMoM._EPS0)
    @test layer.epsr_z≈axial-im*real(axial)*.01-im*.001/(2pi*f*DiffMoM._EPS0)
    @test layer.mur≈1.5-.01im
    r=solve_planar_project(p,f;mx=24,my=20)
    @test opnorm(r.s)<=1+1e-10
    @test r.s≈transpose(r.s) rtol=1e-11
    p.data["stackup"]["layers"][1]["eps_r"]="=complex(2,.001)"
    @test_throws ArgumentError planar_project_layout(p)
end

@testset "project physical conductor stack and source contracts" begin
    for direction in ("up","down")
        p=_physical_project("volume";direction)
        model=planar_project_layout(p;freq=1e6);prob=model.layout.source_problem
        @test length(prob.sheets)==0
        @test length(prob.vols)==1
        @test real(prob.stack.layers[prob.vols[1].layer].thickness)≈20e-6
        @test model.volume_sigma==[1e4]
        @test prob.ports[1].wall===:volume_west
        @test prob.ports[2].wall===:volume_east
        r=solve_planar_project(p,1e6;mx=24,my=20)
        @test real(-inv(r.y[1,2]))≈.003/(1e4*.001*20e-6) rtol=1e-6
        @test r.s≈transpose(r.s) rtol=1e-10
        @test opnorm(r.s)<=1+1e-10
        @test any(map->size(map.jx)==(6,4) && norm(map.jx)>0,
            planar_current_maps(r;incident_waves=[1.,0.]))
    end
    p=_physical_project("volume";subdivisions=2)
    refined=solve_planar_project(p,1e6;mx=24,my=20)
    @test length(refined.model.layout.source_problem.vols)==2
    @test refined.model.layout.contraction==[1. 0.;1. 0.;0. 1.;0. 1.]
    @test real(-inv(refined.y[1,2]))≈15. rtol=1e-6
    @test maximum(refined.em.raw.relative_residuals)<1e-8
    p.data["metals"]["wire"]["conductivity"]="=1e4*freq/1e6"
    dynamic=solve_planar_project(p,2e6;mx=24,my=20)
    @test real(-inv(dynamic.y[1,2]))≈7.5 rtol=1e-6
    dynamic_model=planar_project_layout(p;freq=1e6)
    @test solve_planar(dynamic_model.layout,2e6;mx=24,my=20).s≈dynamic.s rtol=1e-12
    p=_physical_project("thick")
    model=planar_project_layout(p;freq=1e6);prob=model.layout.source_problem
    @test length(prob.sheets)==2
    @test isempty(prob.vols)
    @test length(prob.vias)==1
    z=real.(planar_interfaces(prob.stack))
    @test z[prob.sheets[2].interface+1]-z[prob.sheets[1].interface+1]≈20e-6
    @test model.layout.contraction==[1. 0.;1. 0.;0. 1.;0. 1.]
    @test model.layout.via_models==[Inf]
    r=solve_planar_project(p,1e6;mx=24,my=20)
    @test real(-inv(r.y[1,2]))≈15. rtol=1e-6
    fast=solve_planar_project(p,1e6;mx=24,my=20,method=:dense_fft)
    @test fast.s≈r.s rtol=1e-10
    p.data["metals"]["wire"]["plating"]=[Dict("conductivity"=>"2e4 S/m","thickness"=>"2 um")]
    plated=planar_project_layout(p;freq=1e9)
    expected=planar_layered_surface_zs(1e9,[PlanarConductorLayer(2e4,1e-6),PlanarConductorLayer(1e4,10e-6)])
    @test plated.layout.materials[1](1e9)≈expected rtol=1e-13
    @test real(plated.layout.problem.stack.layers[2].thickness)≈22e-6
    @test_throws ArgumentError planar_project_layout(p;max_bytes=1)
    p.data["metals"]["wire"]["thickness"]="=.0001*freq/1e9"
    @test_throws ArgumentError planar_project_layout(p)
    p=_physical_project("volume")
    p.data["ports"][1]["type"]="internal"
    p.data["polygons"][1]["vertices"][1][1]=.5
    p.data["polygons"][1]["vertices"][4][1]=.5
    @test_throws ArgumentError planar_project_layout(p)
end

@testset "project loaded radiation uses actual physical coefficients" begin
    project=load_planar_project(joinpath(@__DIR__,"..","examples","planar_project_line.toml"))
    raw=solve_planar_project(project,1e9;mx=16,my=12)
    source=raw.model.layout.source_problem
    radiation=PlanarStackup(source.stack.layers,TERM_SPACE,TERM_SPACE,source.stack.a,source.stack.b)
    waves=ComplexF64[1,.2im];v=sqrt.(raw.z0).*(waves+raw.s*waves)
    coeff=raw.em.currents*v
    direct=planar_farfield(source,coeff,1e9;theta=[.2,.6],phi=[.1,.7],radiation_stack=radiation)
    mapped=planar_farfield(raw;incident_waves=waves,theta=[.2,.6],phi=[.1,.7],radiation_stack=radiation)
    @test mapped.etheta≈direct.etheta rtol=1e-12
    @test mapped.ephi≈direct.ephi rtol=1e-12
    voltage=planar_farfield(raw;voltages=v,theta=[.2,.6],phi=[.1,.7],radiation_stack=radiation)
    @test voltage.etheta≈mapped.etheta rtol=1e-10
    project.data["components"]=[Dict("type"=>"resistor","name"=>"shunt","value"=>"100 ohm","ports"=>[2,0])]
    project.data["ports"][2]["external"]=true
    project.data["ports"][1]["refplane"]=Dict("length"=>".1 mm","zc"=>"50 ohm","gamma"=>"=complex(0,freq/1e8)")
    loaded=solve_planar_project(project,1e9;mx=16,my=12)
    nodes=loaded.circuit.voltages*waves
    transfer=loaded.em.gap_voltage_transfer
    coeff=loaded.em.raw.currents*(transfer*nodes)
    direct=planar_farfield(source,coeff,1e9;theta=[.2,.6],phi=[.1,.7],radiation_stack=radiation)
    mapped=planar_farfield(loaded;incident_waves=waves,theta=[.2,.6],phi=[.1,.7],radiation_stack=radiation)
    @test mapped.etheta≈direct.etheta rtol=1e-12
    @test mapped.ephi≈direct.ephi rtol=1e-12
    manual=planar_radiated_power(source,coeff,1e9;ntheta=4,nphi=8,refine=false,radiation_stack=radiation)
    actual=planar_radiated_power(loaded;incident_waves=waves,ntheta=4,nphi=8,refine=false,radiation_stack=radiation)
    @test actual.power≈manual.power rtol=1e-12
    @test_throws ArgumentError planar_farfield(loaded;incident_waves=[NaN,0.],theta=[.2],phi=[.1])
    @test_throws ArgumentError planar_farfield(loaded;incident_waves=waves,voltages=nodes,theta=[.2],phi=[.1])
    @test_throws ArgumentError planar_farfield(loaded;max_bytes=1,theta=[.2],phi=[.1])
end
