using DiffMoM,Test,LinearAlgebra,SHA,JSON
const _spectre_fixture=joinpath(@__DIR__,"fixtures","spice_spectre_ngspice47")

@testset "Explicit Spectre native exports against retained actual ngspice Y and S" begin
    hashes=JSON.parse(read(joinpath(_spectre_fixture,"sha256.json"),String))
    @test all(bytes2hex(sha256(read(joinpath(_spectre_fixture,replace(path,'\\'=>'/')))))==digest for (path,digest) in hashes)
    comparison=JSON.parsefile(joinpath(_spectre_fixture,"comparison.json"))
    @test comparison["engine_sha256"]=="22d5cae2bd32b2e39157a8d27bf457122f68285b72a9ebefdf41551b628233ab"
    for case in comparison["cases"]
        name=split(replace(case["source"],'\\'=>'/'),'/')[end]
        folder=joinpath(_spectre_fixture,splitext(name)[1]);source=joinpath(folder,name)
        library=planar_read_spectre(source);top=only(keys(library.definitions))
        model=planar_spice_model(library,top);np=case["ports"]
        refs=collect(range(50.;step=25.,length=np));pins=case["formal_pins"]
        ports=[(p,"REF") for p in pins[1:np]]
        @test library.dialect===:spectre
        @test library.definitions[top].pins==pins
        @test library.provenance[realpath(source)]==case["source_sha256"]
        @test length(model.couplings)==case["mutual_cards"]
        @test length(model.elements)==length(library.definitions[top].cards)-length(model.couplings)
        @test all(c->c.text in readlines(source)&&c.line>0,library.records)
        frequencies=Float64[];Y=Matrix{ComplexF64}[];S=Matrix{ComplexF64}[]
        for drive in 0:np-1
            rows=[parse.(Float64,split(line)) for line in readlines(joinpath(folder,"drive$drive.dat"))[2:end]]
            if drive==0
                frequencies=first.(rows);Y=[zeros(ComplexF64,np,np) for _ in rows];S=deepcopy(Y)
            end
            @test first.(rows)==frequencies
            for i in eachindex(rows),p in 1:np
                @test length(rows[i])==1+2np
                Y[i][p,drive+1]=-complex(rows[i][2p],rows[i][2p+1])
            end
            fs,waves=_retained_spice_wave(joinpath(folder,"wave$drive.dat"),np,refs)
            @test fs==frequencies
            for i in eachindex(fs);S[i][:,drive+1]=waves[i];end
        end
        for (f,y,s) in zip(frequencies,Y,S)
            result=planar_spice_sparams(model,f;pin_pairs=ports,z0=refs)
            @test norm(result.y-y)/norm(y)<1e-10
            @test maximum(abs,result.s-s)<1e-10
            @test opnorm(result.s)<=1+1e-12
            @test result.s≈transpose(result.s) rtol=1e-11 atol=1e-12
        end
        @test_throws ArgumentError planar_spice_model(library,uppercase(top))
        @test_throws ArgumentError planar_spice_model(library,top;pin_nodes=Dict("ref"=>"0"))
        @test_throws ArgumentError planar_spice_model(library,top;parameters=Dict("invented"=>1.))
    end
end

@testset "Spectre explicit numeric dialect, literal ground and source diagnostics" begin
    for (token,value) in (("1M",1e6),("1m",1e-3),("2KOhms",2e3),("3kOhms",3e3),
            ("2uH",2e-6),("7nF",7e-9),(".5p",.5e-12),("1_",1.),("1%",.01),
            ("2c",.02),("4T",4e12),("2G",2e9),("8a",8e-18),("-3.7e-12",-3.7e-12))
        @test DiffMoM._spectre_number(token)≈value rtol=2eps(Float64)
    end
    @test DiffMoM._spice_number("1M")==1e-3
    @test DiffMoM._spectre_number("1meg")==1e-3 # milli with ignored unit text, never SPICE mega
    for bad in ("1Ohm","1H","1e3k","1Q","1e309","NaN","Inf","{3}","(3)","1e-1000","1e-320a")
        @test_throws ArgumentError DiffMoM._spectre_number(bad)
    end
    @test DiffMoM._spectre_number("0e-1000")==0
    @test DiffMoM._spice_number("0e-1000")==0
    @test_throws ArgumentError DiffMoM._spice_number("1e-1000")
    @test_throws ArgumentError DiffMoM._spice_number("1e-320a")
    mktempdir() do directory
        source=joinpath(directory,"case.scs")
        function readcase(body;kw...)
            write(source,"; literal native dialect\nsimulator lang=spectre\nsubckt Case A a gnd REF\n"*body*"\nends Case\n")
            planar_read_spectre(source;kw...)
        end
        lib=readcase("R_A A gnd resistor r=50\nR_a a gnd resistor r=75\nR_ground REF 0 resistor r=100")
        model=planar_spice_model(lib,"Case")
        @test length(model.elements)==3 && model.pins["A"]!=model.pins["a"]
        result=planar_spice_sparams(model,1e9;pin_pairs=[("A","gnd"),("a","gnd")],z0=[50.,75.])
        @test maximum(abs,result.s)<1e-14
        @test length(result.gauge_nodes)==1
        @test model.pins["gnd"]>0 && "pin.gnd" in keys(model.nodes)
        @test_throws ArgumentError planar_spice_sparams(model,1e9;pin_pairs=[("A","GND")])
        # Both readable library and pre-K compiled-model constructor arities remain valid.
        legacy=PlanarSpiceLibrary(lib.source,lib.root,lib.definitions,lib.parameters,lib.provenance,
            lib.payload,lib.records,lib.includes)
        @test legacy.dialect===:spice
        compatibility=PlanarSpiceModel(model.elements,model.nodes,model.pins,model.provenance,
            model.payload,model.library,model.subckt,model.parameters)
        @test isempty(compatibility.couplings) && isempty(compatibility.inductive_rows)
        @test planar_spice_sparams(compatibility,1e9;pin_pairs=[("A","gnd"),("a","gnd")],z0=[50.,75.]).s==result.s
        # Forward mutual properties may be reordered; native winding order is retained.
        forward=readcase("M_native mutual_inductor ind2=L_a coupling=-.8 ind1=L_A\nL_A A gnd inductor l=1n\nL_a a gnd inductor l=2n")
        compiled=planar_spice_model(forward,"Case")
        @test only(compiled.couplings).coefficient==-.8
        @test [compiled.elements[i].name for i in only(compiled.couplings).inductors]==["xroot/L_A","xroot/L_a"]
        for body in ("include other.scs","parameters x=1","X1 A a child","D1 A gnd diode",
                "R1 (A gnd) resistor r=1","R1 A gnd resistor r={1+2}","R1 A gnd resistor r=1 \$comment",
                "R1 A gnd resistor r=1 ; comment","r1 A gnd resistor r=1\nr1 a gnd resistor r=2",
                "M1 mutual_inductor coupling=.8 ind1=L1 ind1=L2","L1 A gnd inductor l=1e309",
                "ends other","simulator lang=spectre","simulator lang=spice","tran stop=1n")
            error=try readcase(body);nothing catch e;e end
            @test error isa ArgumentError
            @test occursin(realpath(source)*":",sprint(showerror,error))
        end
        valid=readcase("R1 A gnd resistor r=1")
        for kw in ((max_bytes=1,),(max_records=2,),(max_line_bytes=10,))
            @test_throws ArgumentError planar_read_spectre(source;kw...)
        end
        @test_throws ArgumentError planar_read_spectre(source;root=joinpath(directory,"missing"))
        mktempdir() do outside
            @test_throws ArgumentError planar_read_spectre(source;root=outside)
        end
        for kw in ((max_records=0,),(max_line_bytes=0,),(max_bytes=false,))
            @test_throws ArgumentError planar_read_spectre(source;kw...)
        end
        @test_throws ArgumentError planar_spice_model(valid,"Case";max_bytes=1)
    end
end

function _spectre_project_fixture(path)
    planar_project_from_dict(Dict("project"=>Dict("name"=>"spectre_loaded","unit"=>"mm"),
        "box"=>Dict("width"=>3.,"length"=>2.),"mesh"=>Dict("nx"=>6,"ny"=>4),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5,"eps_r"=>2.),Dict("thickness"=>.5)]),
        "metals"=>Dict("film"=>Dict("type"=>"surface_impedance","rs"=>.1)),
        "polygons"=>[Dict("name"=>"wire","level"=>1,"metal"=>"film","vertices"=>[[0.,.5],[3.,.5],[3.,1.5],[0.,1.5]])],
        "ports"=>[Dict("name"=>"input","number"=>1,"polygon"=>"wire","edge"=>4,"external"=>true),
            Dict("name"=>"output","number"=>2,"polygon"=>"wire","edge"=>2,"external"=>true)],
        "components"=>[Dict("name"=>"InstalledExport","type"=>"subckt","dialect"=>"spectre","path"=>path,
            "subckt"=>"_1nH_oct_inductor","nodes"=>["input","output","gnd"])],
        "sweep"=>Dict("frequencies"=>[".9 GHz","1 GHz","1.1 GHz"])))
end

@testset "Spectre explicit project lowering, retained source and physical node projection" begin
    path=joinpath(_spectre_fixture,"1nH_oct_inductor","1nH_oct_inductor.scs")
    project=_spectre_project_fixture(path)
    direct=solve_planar_project(project,1e9;mx=20,my=16)
    circuit=DiffMoM._project_circuit(direct.model,1e9,nothing,10^7)
    @test length(circuit.elements)==11 && circuit.elements[1].owner.model.library.dialect===:spectre
    @test direct.circuit!==nothing && size(direct.circuit.voltages,1)>length(direct.model.node_names)
    independent=planar_spice_sparams(planar_spice_model(planar_read_spectre(path),"_1nH_oct_inductor";
        pin_nodes=Dict("REF"=>"0")),1e9;pin_pairs=[("1","REF"),("4","REF")])
    reference=_spectre_project_fixture(path)
    reference.data["components"]=[Dict("name"=>"Independent","type"=>"network","ports"=>[1,2])]
    equivalent=solve_planar_project(reference,1e9;mx=20,my=16,
        component_response=(p,c,f)->(response=independent.y,format=:y,z0=50.))
    @test direct.s≈equivalent.s rtol=1e-11 atol=1e-12
    waves=ComplexF64[.3+.2im,-.5im]
    v=transpose(direct.model.source_incidence)*view(direct.circuit.voltages*waves,1:length(direct.model.node_names))
    maps=planar_current_maps(direct;incident_waves=waves)
    expected=planar_current_maps(direct.em;voltages=v)
    @test maps[1].jx≈expected[1].jx rtol=1e-13
    @test maps[1].jy≈expected[1].jy rtol=1e-13
    excitation=DiffMoM._project_radiation_excitation(direct;incident_waves=waves,max_bytes=10^7)
    @test excitation.coefficients≈DiffMoM._planar_coefficient_columns(direct.em)*v rtol=1e-13
    far=planar_farfield(direct;incident_waves=waves,theta=[.3,.6],phi=[.2])
    physical=planar_farfield(excitation.problem,excitation.coefficients,1e9;theta=[.3,.6],phi=[.2],accepted_power=excitation.accepted)
    @test far.etheta≈physical.etheta rtol=1e-13
    @test far.ephi≈physical.ephi rtol=1e-13
    @test planar_project_sweep(project;mx=20,my=16).s[2]≈direct.s rtol=1e-12
    @test_throws ArgumentError solve_planar_project(project,1e9;max_bytes=1)
    for dialect in ("guess","Spectre")
        broken=_spectre_project_fixture(path);broken.data["components"][1]["dialect"]=dialect
        @test_throws ArgumentError planar_project_from_dict(planar_project_dict(broken))
    end
    mktempdir() do directory
        cp(path,joinpath(directory,"model.scs"));project.data["components"][1]["path"]="model.scs"
        file=joinpath(directory,"loaded.toml");write_planar_project(file,project)
        @test solve_planar_project(load_planar_project(file),1e9;mx=20,my=16).s≈direct.s rtol=1e-12
    end
end

@testset "Installed coupled Spectre model loads physical EM nodes" begin
    path=joinpath(_spectre_fixture,"ct_inductor_noshield_IMEnet","ct_inductor_noshield_IMEnet.scs")
    project=_spectre_project_fixture(path);component=project.data["components"][1]
    component["subckt"]="ct_inductor_noshield_IMEnet";component["nodes"]=["input","output","gnd","gnd"]
    direct=solve_planar_project(project,1e9;mx=20,my=16)
    model=planar_spice_model(planar_read_spectre(path),component["subckt"];pin_nodes=Dict("3"=>"0","REF"=>"0"))
    independent=planar_spice_sparams(model,1e9;pin_pairs=[("1","REF"),("2","REF")])
    adapted=_spectre_project_fixture(path)
    adapted.data["components"]=[Dict("name"=>"Independent","type"=>"network","ports"=>[1,2])]
    equivalent=solve_planar_project(adapted,1e9;mx=20,my=16,
        component_response=(p,c,f)->(response=independent.y,format=:y,z0=50.))
    @test length(model.couplings)==1
    @test direct.s≈equivalent.s rtol=1e-11 atol=1e-12
    waves=ComplexF64[.2im,.4-.1im]
    voltage=transpose(direct.model.source_incidence)*view(direct.circuit.voltages*waves,1:length(direct.model.node_names))
    maps=planar_current_maps(direct;incident_waves=waves);physical=planar_current_maps(direct.em;voltages=voltage)
    @test maps[1].jx≈physical[1].jx rtol=1e-13
    @test maps[1].jy≈physical[1].jy rtol=1e-13
    excitation=DiffMoM._project_radiation_excitation(direct;incident_waves=waves,max_bytes=10^7)
    @test excitation.coefficients≈DiffMoM._planar_coefficient_columns(direct.em)*voltage rtol=1e-13
end
