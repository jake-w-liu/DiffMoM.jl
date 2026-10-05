using DiffMoM,Test,LinearAlgebra

function _project_test_fixture(;gap=false,reference=false)
    variable=Dict("w"=>"1 mm","yc"=>"1 mm","er"=>2.)
    left=Dict("name"=>"left","level"=>1,"metal"=>"copper","net"=>"wire",
        "vertices"=>[[0.,"=yc-w/2"],[gap ? 1. : 3.,"=yc-w/2"],
            [gap ? 1. : 3.,"=yc+w/2"],[0.,"=yc+w/2"]])
    polygons=Any[left]
    gap && push!(polygons,Dict("name"=>"right","level"=>1,"metal"=>"copper","net"=>"wire",
        "vertices"=>[[2.,.5],[3.,.5],[3.,1.5],[2.,1.5]]))
    ports=Any[Dict("name"=>"input","number"=>1,"polygon"=>"left","edge"=>4,"type"=>"box_wall"),
        Dict("name"=>"output","number"=>2,"polygon"=>gap ? "right" : "left","edge"=>2,"type"=>"box_wall")]
    if gap
        push!(ports,Dict("name"=>"pin_left","number"=>3,"polygon"=>"left","edge"=>2,"type"=>"internal"))
        push!(ports,Dict("name"=>"pin_right","number"=>4,"polygon"=>"right","edge"=>4,"type"=>"internal"))
    end
    reference && (ports[1]["refplane"]=Dict("length"=>.2,"zc"=>50.,
        "gamma"=>"=complex(0,2*pi*freq/299792458)"))
    return planar_project_from_dict(Dict("project"=>Dict("name"=>"portable_project","unit"=>"mm"),
        "variables"=>variable,"box"=>Dict("width"=>3.,"length"=>2.),"mesh"=>Dict("nx"=>6,"ny"=>4),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5,"eps_r"=>"=er+.01*freq/1e9"),
            Dict("thickness"=>.5,"eps_r"=>1.)]),
        "metals"=>Dict("copper"=>Dict("type"=>"surface_impedance","rs"=>"=.1+.01*freq/1e9")),
        "polygons"=>polygons,"ports"=>ports,
        "components"=>gap ? [Dict("name"=>"load","type"=>"resistor","value"=>"10 ohm","ports"=>[3,4])] : Any[],
        "sweep"=>Dict("frequencies"=>[".9 GHz","1 GHz","1.1 GHz"])))
end

@testset "planar project: via technology and all circuit adapters" begin
    via=_project_test_fixture()
    via.data["via_types"]=Dict("solid"=>Dict("metal"=>"pec","kind"=>"uniform"))
    via.data["metals"]["pec"]=Dict("type"=>"pec")
    via.data["tech_layers"]=Dict("hole"=>Dict("kind"=>"via","from_level"=>1,"to_level"=>"gnd","via_type"=>"solid"))
    via.data["vias"]=[Dict("name"=>"ground_tie","tech_layer"=>"hole", "vertices"=>[[1.,.5],[1.5,.5],[1.5,1.],[1.,1.]])]
    model=planar_project_layout(via;freq=1e9)
    @test length(model.layout.problem.vias)==1
    @test count(model.layout.problem.vias[1].uni)==1
    @test model.layout.via_models==[Inf]
    @test planar_connectivity(via).grounded_components==[1]
    @test opnorm(solve_planar_project(via,1e9;mx=16,my=12).s)<=1+1e-10
    via.data["ports"]=[Dict("name"=>"axial","number"=>1,"type"=>"via","via"=>"ground_tie",
        "layer"=>1,"cells"=>[9,9],"polarity"=>-1,
        "refplane"=>Dict("length"=>0.,"zc"=>50.,"gamma"=>"=complex(0,freq/1e8)"))]
    axial=planar_project_layout(via;freq=1e9)
    @test axial.layout.problem.ports[1].wall===:via
    @test axial.layout.problem.ports[1].polarity==-1
    @test solve_planar_project(via,1e9;mx=16,my=12).em isa PlanarCalibratedResult
    via.data["ports"][1]["cells"]=[1,1]
    @test_throws ArgumentError planar_project_layout(via)
    for specification in (
            Dict("type"=>"capacitor","value"=>"=1e-12*(freq/1e8)"),
            Dict("type"=>"inductor","value"=>"1 nH"),
            Dict("type"=>"rlc","r"=>"10 ohm","l"=>"1 nH","c"=>"1 pF","topology"=>"parallel"),
            Dict("type"=>"transmission_line","zc"=>"50 ohm","gamma"=>"=complex(0,2*pi*freq/299792458)","length"=>"1 mm"),
            Dict("type"=>"transformer","ratio"=>1.5))
        project=_project_test_fixture(gap=true)
        project.data["components"][1]=merge(Dict("name"=>"branch","ports"=>[3,4]),specification)
        result=solve_planar_project(project,1e8;mx=20,my=16)
        @test result.s≈transpose(result.s) rtol=1e-10
        @test opnorm(result.s)<=1+1e-9
    end
    through=[[0.,1.],[1.,0.]]
    project=_project_test_fixture(gap=true)
    project.data["components"][1]=Dict("name"=>"inline","type"=>"inline_s","ports"=>[3,4],
        "z0"=>["50 ohm","50 ohm"],"frequencies"=>["100 MHz","200 MHz"],
        "s_real"=>[through,through])
    inline=solve_planar_project(project,150e6;mx=20,my=16)
    @test abs(inline.s[2,1])>.95
    mktempdir() do dir
        file=joinpath(dir,"through.s2p")
        planar_write_touchstone(file,PlanarNetworkData([100e6,200e6],
            [ComplexF64[0 1;1 0],ComplexF64[0 1;1 0]];z0=50))
        project.data["components"][1]=Dict("name"=>"file","type"=>"sparam_file","ports"=>[3,4],
            "path"=>"through.s2p","z0"=>["75 ohm","75 ohm"])
        path=joinpath(dir,"loaded.toml");write_planar_project(path,project)
        loaded=load_planar_project(path)
        result=solve_planar_project(loaded,150e6;mx=20,my=16)
        @test result.s≈inline.s rtol=1e-10
        @test_throws ArgumentError solve_planar_project(loaded,250e6;mx=20,my=16)
    end
    project.data["components"][1]=Dict("name"=>"vendor","type"=>"subckt","ports"=>[3,4],"path"=>"vendor.sp")
    @test_throws ArgumentError solve_planar_project(project,150e6;mx=20,my=16)
    response=(project,component,freq)->(response=ComplexF64[0 1;1 0],format=:s,z0=50.)
    callback=solve_planar_project(project,150e6;component_response=response,mx=20,my=16)
    @test callback.s≈inline.s rtol=1e-11
    calls=Ref(0)
    @test_throws ArgumentError solve_planar_project(project,150e6;max_bytes=1,
        component_response=(p,c,f)->(calls[]+=1;response(p,c,f)))
    @test calls[]==0
    mktempdir() do dir
        path=joinpath(dir,"sentinel.toml");write(path,"sentinel")
        project.data["components"][1]["unsupported"]=1
        @test_throws ArgumentError write_planar_project(path,project)
        @test read(path,String)=="sentinel"
    end
end

@testset "planar project: safe SI variables, units and roundtrip" begin
    project=_project_test_fixture()
    @test planar_project_value(project,1.;dimension=:length)==.001
    @test planar_project_value(project,"=2*w";dimension=:length)==.002
    @test planar_project_value(project,"50 ohm";dimension=:resistance)==50.
    @test planar_project_value(project,"=sin(pi/2)+sqrt(er)")≈1+sqrt(2)
    @test planar_project_value(project,"=complex(1,freq/1e9)";freq=2e9)==1+2im
    @test PlanarProject(planar_project_dict(project)).data==project.data
    @test_throws ArgumentError planar_project_value(project,"=freq")
    @test_throws ArgumentError planar_project_value(project,"1 GHz";dimension=:length)
    @test_throws ArgumentError planar_project_value(project,"1e1000")
    @test_throws ArgumentError planar_project_value(project,"=Main.x")
    @test_throws ArgumentError planar_project_value(project,"=run(1)")
    @test_throws ArgumentError planar_project_value(project,"=begin;1;end")
    @test_throws ArgumentError planar_project_value(project,"=unknown+1")
    cycle=planar_project_dict(project);cycle["variables"]=Dict("a"=>"=b","b"=>"=a")
    @test_throws ArgumentError planar_project_from_dict(cycle)
    reserved=planar_project_dict(project);reserved["variables"]["freq"]=1.
    @test_throws ArgumentError planar_project_from_dict(reserved)
    mktempdir() do dir
        path=joinpath(dir,"project.toml");write_planar_project(path,project)
        loaded=load_planar_project(path)
        @test planar_project_dict(loaded)==planar_project_dict(project)
        @test loaded.source==abspath(path)
        @test_throws ArgumentError load_planar_project(path;max_bytes=1)
        @test_throws ArgumentError solve_planar_project(path,1e9;max_bytes=1)
        write(path,"sentinel")
        project.data["unsupported"]=1
        @test_throws ArgumentError write_planar_project(path,project)
        @test read(path,String)=="sentinel"
        delete!(project.data,"unsupported")
    end
end

@testset "planar project: dispersive physical layout and response workflow" begin
    project=_project_test_fixture();model=planar_project_layout(project;freq=1e9)
    @test model.port_names==["input","output"]
    @test model.layout.problem.sheets[1].interface==1
    @test model.layout.problem.stack.layers[1].epsr≈2.01
    @test planar_project_layout(project;freq=2e9).layout.problem.stack.layers[1].epsr≈2.02
    @test count(model.layout.problem.sheets[1].mask)==12
    @test planar_project_metal_zs(project,"copper",1e9)≈.11
    @test planar_project_metal_zs(project,"copper",2e9)≈.12
    result=solve_planar_project(project,1e9;mx=16,my=12)
    direct=solve_planar(model.layout.problem,1e9;mx=16,my=12,surface_zs=.11)
    @test result.s≈direct.s rtol=1e-12
    @test result.s≈transpose(result.s) rtol=1e-12
    @test opnorm(result.s)<=1+1e-10
    @test all(isfinite,planar_current_maps(result;incident_waves=[1.,0.])[1].jx)
    @test planar_connectivity(project).component_count==1
    data=planar_project_sweep(project;mx=16,my=12)
    @test data.frequencies==[.9e9,1e9,1.1e9]
    @test data.s[2]≈result.s rtol=1e-12
    @test data.port_names==model.port_names
    project.data["sweep"]=Dict("start"=>".9 GHz","stop"=>"1.1 GHz","points"=>2,
        "adaptive"=>true,"max_points"=>4,"n_eval"=>9)
    abs=planar_project_sweep(project;mx=16,my=12)
    @test all(all(isfinite,S) for S in abs.dense_s)
    @test_throws ArgumentError solve_planar_project(project,1e9;max_bytes=1)
    huge=planar_project_dict(project)
    huge["sweep"]=Dict("start"=>1e9,"stop"=>2e9,"points"=>"=10^12")
    @test_throws ArgumentError planar_project_frequencies(planar_project_from_dict(huge);max_bytes=64)
    freqgeom=planar_project_dict(project);freqgeom["polygons"][1]["vertices"][1][1]="=freq"
    @test_throws ArgumentError planar_project_layout(planar_project_from_dict(freqgeom))
    legacy=planar_project_dict(project);legacy["stackup"]["order"]="ascent"
    legacy["polygons"][1]["level"]=0
    old=planar_project_layout(planar_project_from_dict(legacy);freq=1e9)
    @test old.layout.problem.stack.layers[2].epsr≈2.01
    @test old.layout.problem.sheets[1].interface==1
    shifted=_project_test_fixture();shifted.data["box"]=Dict("auto_margin"=>.5)
    for port in shifted.data["ports"];port["type"]="terminal";end
    moved=planar_project_layout(shifted;grid=(8,4))
    @test moved.layout.problem.grid.a≈.004
    @test moved.layout.contraction!==nothing
end

@testset "planar project: components, calibration and named technology" begin
    project=_project_test_fixture(gap=true)
    result=solve_planar_project(project,1e8;mx=20,my=16)
    @test result.port_names==["input","output"]
    @test result.em isa PlanarContractedResult
    @test size(result.em.currents,2)==4
    @test abs(result.s[2,1])>.8
    @test opnorm(result.s)<=1+1e-10
    model=result.model
    circuit=PlanarCircuit(4,[1,2];z0=50.)
    circuit_add_rlc!(circuit,3,4;r=10.)
    circuit_add_network!(circuit,1:4,result.em.y;format=:y)
    manual=solve_planar_circuit(circuit,1e8)
    @test result.s≈manual.s rtol=1e-12
    maps=planar_current_maps(result;incident_waves=[1.,.3im])
    direct=planar_current_maps(result.em;voltages=result.circuit.voltages*ComplexF64[1,.3im])
    @test maps[1].jx≈direct[1].jx rtol=1e-13
    @test maps[end].jz≈direct[end].jz rtol=1e-13
    reference=_project_test_fixture(reference=true)
    calibrated=solve_planar_project(reference,1e9;mx=16,my=12)
    @test calibrated.em isa PlanarCalibratedResult
    @test calibrated.s≈planar_y_to_s(deembed_ports(calibrated.em.raw.y,calibrated.em.chains),[50.,50.])
    @test all(isfinite,planar_current_maps(calibrated)[1].jx)
    tech=_project_test_fixture()
    tech.data["tech_layers"]=Dict("drawing"=>Dict("kind"=>"metal","level"=>1,"metal"=>"copper","stream_layer"=>7,"datatype"=>2))
    delete!(tech.data["polygons"][1],"level");delete!(tech.data["polygons"][1],"metal")
    tech.data["polygons"][1]["tech_layer"]="drawing"
    @test planar_project_layout(tech).layout.problem.sheets[1].interface==1
    thick=_project_test_fixture()
    thick.data["metals"]["copper"]=Dict("type"=>"thick","conductivity"=>"5.8e7 S/m","thickness"=>"20 um",
        "plating"=>[Dict("conductivity"=>"1.4e7 S/m","thickness"=>"2 um")],
        "roughness"=>Dict("model"=>"hammerstad","rms"=>"1 um","loss_only"=>true))
    expected=planar_layered_surface_zs(1e9,[PlanarConductorLayer(1.4e7,2e-6),PlanarConductorLayer(5.8e7,20e-6)];
        roughness=HammerstadRoughness(1e-6),loss_only=true)
    @test planar_project_metal_zs(thick,"copper",1e9)≈expected rtol=1e-13
    @test planar_material_preset("gaas").eps_r==12.9
    @test_throws ArgumentError planar_material_preset("unknown")
end

function _project_zero_budget_rejection_bytes(project)
    reject()=try
        planar_project_layout(project;max_bytes=0)
        error("zero layout budget unexpectedly accepted")
    catch err
        err isa ArgumentError && occursin("max_bytes",sprint(showerror,err)) || rethrow()
        nothing
    end
    reject();reject()
    return @allocated reject()
end

@testset "planar project: invalid budgets reject before layer-dependent work" begin
    for count in (2,128,1024)
        project=_project_test_fixture()
        layer=first(project.data["stackup"]["layers"])
        project.data["stackup"]["layers"]=[copy(layer) for _ in 1:count]
        saved=planar_project_dict(project)
        @test _project_zero_budget_rejection_bytes(project)<=4096
        @test project.data==saved
        for budget in (-1,big(typemax(Int))+1)
            @test_throws ArgumentError planar_project_layout(project;max_bytes=budget)
        end
    end
    project=_project_test_fixture()
    err=try planar_project_layout(project;freq=-1.,max_bytes=0);nothing catch e;e end
    @test err isa ArgumentError && occursin("frequency",sprint(showerror,err))
end
