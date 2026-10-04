using DiffMoM,Test,LinearAlgebra

function _spice_project_fixture()
    planar_project_from_dict(Dict("project"=>Dict("name"=>"spice_loaded","unit"=>"mm"),
        "variables"=>Dict("series_resistance"=>"10 ohm"),
        "box"=>Dict("width"=>3.,"length"=>2.),"mesh"=>Dict("nx"=>6,"ny"=>4),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5,"eps_r"=>2.),Dict("thickness"=>.5)]),
        "metals"=>Dict("film"=>Dict("type"=>"surface_impedance","rs"=>.1)),
        "polygons"=>[Dict("name"=>"wire","level"=>1,"metal"=>"film",
            "vertices"=>[[0.,.5],[3.,.5],[3.,1.5],[0.,1.5]])],
        "ports"=>[Dict("name"=>"input","number"=>1,"polygon"=>"wire","edge"=>4,"external"=>true,
                "refplane"=>Dict("length"=>.1,"zc"=>50.,"gamma"=>"=complex(0,2*pi*freq/299792458)")),
            Dict("name"=>"output","number"=>2,"polygon"=>"wire","edge"=>2,"external"=>true)],
        "sweep"=>Dict("frequencies"=>[".9 GHz","1 GHz","1.1 GHz"])))
end

@testset "Project named SPICE direct primitive load, provenance and physical outputs" begin
    project=_spice_project_fixture()
    path=joinpath(@__DIR__,"fixtures","spice_linear_ngspice47","ladder.lib")
    project.data["components"]=[Dict("name"=>"LadderLoad","type"=>"subckt","path"=>path,
        "subckt"=>"LADDER","nodes"=>["input","output","gnd"],
        "parameters"=>Dict("rval"=>"=series_resistance"))]
    project=planar_project_from_dict(planar_project_dict(project))
    direct=solve_planar_project(project,1e9;mx=20,my=16)
    @test direct.circuit!==nothing
    @test size(direct.circuit.voltages,1)>length(direct.model.node_names)
    @test direct.circuit.gauge_nodes==Int[]
    @test direct.em isa PlanarCalibratedResult
    @test all(e->length(e.owner.model.provenance)==2,
        filter(e->e isa DiffMoM._CircuitSpicePrimitive,DiffMoM._project_circuit(direct.model,1e9,nothing,10^7).elements))
    compiled=planar_spice_model(planar_read_spice(path),"ladder";parameters=Dict("rval"=>10.),pin_nodes=Dict("ref"=>"0"))
    independent=planar_spice_sparams(compiled,1e9;pin_pairs=[("p1","ref"),("p2","ref")])
    reference=_spice_project_fixture()
    reference.data["components"]=[Dict("name"=>"Independent","type"=>"network","ports"=>[1,2])]
    adapted=solve_planar_project(reference,1e9;mx=20,my=16,
        component_response=(p,c,f)->(response=independent.y,format=:y,z0=50.))
    @test direct.s≈adapted.s rtol=1e-11 atol=1e-12
    @test direct.y≈adapted.y rtol=1e-11 atol=1e-12
    waves=ComplexF64[.3+.2im,-.5im]
    original_nodes=view(direct.circuit.voltages*waves,1:length(direct.model.node_names))
    voltage=transpose(direct.model.source_incidence)*original_nodes
    actual=planar_current_maps(direct;incident_waves=waves)
    expected=planar_current_maps(direct.em;voltages=voltage)
    @test actual[1].jx≈expected[1].jx rtol=1e-13
    @test actual[1].jy≈expected[1].jy rtol=1e-13
    excitation=DiffMoM._project_radiation_excitation(direct;incident_waves=waves,max_bytes=10^7)
    @test excitation.coefficients≈DiffMoM._planar_coefficient_columns(direct.em)*voltage rtol=1e-13
    fields=planar_farfield(direct;incident_waves=waves,theta=[.3,.6],phi=[.2])
    equivalent=planar_farfield(excitation.problem,excitation.coefficients,1e9;
        theta=[.3,.6],phi=[.2],accepted_power=excitation.accepted)
    @test fields.etheta≈equivalent.etheta rtol=1e-13
    @test fields.ephi≈equivalent.ephi rtol=1e-13
    @test_throws ArgumentError planar_current_maps(direct;max_bytes=1)
    @test_throws ArgumentError planar_farfield(direct;max_bytes=1)
    data=planar_project_sweep(project;mx=20,my=16)
    @test length(data.frequencies)==3
    @test data.s[2]≈direct.s rtol=1e-12
    mktempdir() do directory
        # Relative source/include edges retain the project directory contract.
        cp(path,joinpath(directory,"ladder.lib"));cp(joinpath(dirname(path),"section.inc"),joinpath(directory,"section.inc"))
        project.data["components"][1]["path"]="ladder.lib"
        output=joinpath(directory,"loaded.toml");write_planar_project(output,project)
        restored=load_planar_project(output)
        @test solve_planar_project(restored,1e9;mx=20,my=16).s≈direct.s rtol=1e-12
        @test length(DiffMoM._project_circuit(planar_project_layout(restored;freq=1e9),1e9,nothing,10^7).elements)==6
    end
    for bad in (Dict("nodes"=>[["input","gnd"],["output","gnd"]]),
            Dict("parameters"=>[1,2]),Dict("nodes"=>["input","output"]),Dict("subckt"=>"missing"))
        broken=_spice_project_fixture();broken.data["components"]=[merge(Dict("type"=>"subckt","name"=>"Bad","path"=>path,
            "subckt"=>"ladder","nodes"=>["input","output","gnd"]),bad)]
        @test_throws Exception solve_planar_project(planar_project_from_dict(planar_project_dict(broken)),1e9;mx=20,my=16)
    end
end

@testset "Project adaptive parameters and storage reject before technology evaluation" begin
    for (field,value) in (("n_eval",1),("max_points",2),("rel_tol",0.),("n_eval",10^9))
        project=_spice_project_fixture()
        project.data["stackup"]["layers"][1]["eps_r"]="=undefined_technology"
        project.data["sweep"]=merge(Dict("frequencies"=>[".9 GHz","1.1 GHz"],"adaptive"=>true),Dict(field=>value))
        error=try planar_project_sweep(project;max_bytes=10^7);nothing catch e;e end
        @test error isa ArgumentError
        @test !occursin("undefined_technology",sprint(showerror,error))
    end
end
