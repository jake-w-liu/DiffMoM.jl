using DiffMoM,Test,LinearAlgebra,SHA,TOML
const _spice_lines_fixture=joinpath(@__DIR__,"fixtures","spice_lines_ngspice47")
const _spice_lines_pairs=[("p1","g1"),("p2","g2")]

function _spice_lines_engine(name;shift=false,dc=false,gmin=nothing)
    suffix=(gmin===nothing ? "" : "_gmin$(gmin)")*(dc ? "_dc" : "")
    output=Matrix{ComplexF64}[];frequencies=Float64[]
    for drive in 1:2
        path=joinpath(_spice_lines_fixture,"$(name)_device_$(shift ? "shift" : "zero")$(suffix)_drive$drive.cir.dat")
        lines=readlines(path)
        if drive==1
            output=[zeros(ComplexF64,2,2) for _ in lines[2:end]]
        end
        for (index,line) in enumerate(lines[2:end])
            t=parse.(Float64,split(line));@test length(t)==(dc ? 5 : 9)
            f=dc ? 0. : t[1]
            drive==1 ? push!(frequencies,f) : (@test frequencies[index]==f)
            for p in 1:2
                v=dc ? complex(t[2p]) : complex(t[4p-2],t[4p-1])
                i=dc ? -complex(t[2p+1]) : -complex(t[4p],t[4p+1])
                z=p==1 ? 50. : 75.
                output[index][p,drive]=(v-z*i)/(2sqrt(z))
            end
        end
    end
    frequencies,output
end

function _spice_lines_independent(spec,f;gmin=0.)
    setprecision(BigFloat,512) do
        freq=BigFloat(f);pi_value=BigFloat(pi)
        if spec.kind===:T
            z=Complex{BigFloat}(spec.z0);gl=2pi_value*im*freq*BigFloat(spec.td)
            a=d=cosh(gl);b=z*sinh(gl);c=sinh(gl)/z
        else
            series=BigFloat(spec.r)+2pi_value*im*freq*BigFloat(spec.l)
            shunt=BigFloat(spec.g)+2pi_value*im*freq*BigFloat(spec.c)
            len=BigFloat(spec.length);gl=sqrt(series)*sqrt(shunt)*len
            sinhc=iszero(gl) ? one(gl) : sinh(gl)/gl
            a=d=cosh(gl);b=series*len*sinhc;c=shunt*len*sinhc
            b*=1+BigFloat(gmin);c*=1+BigFloat(gmin)
        end
        z1,z2=BigFloat(50),BigFloat(75);den=a*z2+b+c*z1*z2+d*z1
        ComplexF64[(a*z2+b-c*z1*z2-d*z1)/den 2sqrt(z1*z2)*(a*d-b*c)/den;
            2sqrt(z1*z2)/den (-a*z2+b-c*z1*z2+d*z1)/den]
    end
end

@testset "SPICE T/O provenance, actual engine full S and physical branch currents" begin
    manifest=TOML.parsefile(joinpath(_spice_lines_fixture,"sha256.toml"))
    @test length(manifest["files"])==272
    @test manifest["engine_sha256"]=="22d5cae2bd32b2e39157a8d27bf457122f68285b72a9ebefdf41551b628233ab"
    for (path,hash) in manifest["files"]
        @test bytes2hex(sha256(read(joinpath(_spice_lines_fixture,path))))==hash
    end
    for name in manifest["cases"]
        library=planar_read_spice(joinpath(_spice_lines_fixture,name*".lib"))
        model=planar_spice_model(library,"device")
        @test all(e->e.kind in ('T','O') || !haskey(model.lines,findfirst(==(e),model.elements)),model.elements)
        refs=[50.,75.]
        circuit=PlanarCircuit(4,[(1,2),(3,4)];z0=refs)
        circuit_add_spice!(circuit,[1,2,3,4],model)
        @test length(DiffMoM._circuit_gauge_nodes(circuit))==2
        @test sum(length(e.terminals) for e in circuit.elements)==length(model.elements)+length(model.lines)
        fs,expected=_spice_lines_engine(name;gmin=name=="rg" ? 0. : nothing)
        shifted_fs,shifted=_spice_lines_engine(name;shift=true,gmin=name=="rg" ? 0. : nothing)
        @test fs==shifted_fs
        for (index,f) in enumerate(fs)
            actual=solve_planar_circuit(circuit,f;floating_gauge=:auto)
            @test maximum(abs,actual.s-expected[index])<1e-10
            @test maximum(abs,expected[index]-shifted[index])<1e-10
            @test maximum(abs,actual.s-transpose(actual.s))<1e-11
            @test opnorm(actual.s)<=1+1e-11
            @test length(actual.gauge_nodes)==2
            @test norm(actual.voltages[[1,3],:]-actual.voltages[[2,4],:]-
                sqrt.(refs).*(Matrix{ComplexF64}(I,2,2)+actual.s))<1e-11
            if length(model.elements)==1
                spec=only(values(model.lines))
                @test maximum(abs,actual.s-_spice_lines_independent(spec,f))<1e-11
            end
        end
        if name in ("explicit_td","frequency_length","lc","rc","rg","rlc","nested_params")
            _,reference=_spice_lines_engine(name;dc=true,gmin=name=="rg" ? 0. : nothing)
            @test maximum(abs,only(reference)-solve_planar_circuit(circuit,0.;floating_gauge=:auto).s)<1e-10
        end
        if name=="transient_controls"
            options=Dict(only(values(model.lines)).options)
            @test options==Dict("rel"=>1.,"abs"=>.1,"compactrel"=>.001,"compactabs"=>1e-14,"steplimit"=>1.,"lininterp"=>1.)
        end
    end
    rg=planar_spice_model(planar_read_spice(joinpath(_spice_lines_fixture,"rg.lib")),"device")
    _,default=_spice_lines_engine("rg")
    _,zero=_spice_lines_engine("rg";gmin=0.)
    @test maximum(abs,first(default)-_spice_lines_independent(only(values(rg.lines)),1e6;gmin=1e-12))<1e-11
    @test maximum(abs,first(default)-transpose(first(default)))>1e-9
    @test maximum(abs,first(zero)-transpose(first(zero)))<1e-10
end

@testset "Mixed line and F/H/K physical current branch indices" begin
    for name in TOML.parsefile(joinpath(_spice_lines_fixture,"sha256.toml"))["mixed_cases"]
        model=planar_spice_model(planar_read_spice(joinpath(_spice_lines_fixture,name*".lib")),"device")
        fs,reference=_spice_lines_engine(name)
        _,shifted=_spice_lines_engine(name;shift=true)
        for (f,s,t) in zip(fs,reference,shifted)
            actual=planar_spice_sparams(model,f;pin_pairs=_spice_lines_pairs,z0=[50.,75.])
            @test maximum(abs,actual.s-s)<1e-10
            @test maximum(abs,s-t)<1e-10
        end
    end
end

@testset "Line near-DC, scale-safe coefficients and independent reference modes" begin
    for name in ("rc","rlc","lc")
        model=planar_spice_model(planar_read_spice(joinpath(_spice_lines_fixture,name*".lib")),"device")
        spec=only(values(model.lines))
        for f in (0.,1e-300,1e-100,1e-20,1e-6,1.,1e6,1e9)
            actual=planar_spice_sparams(model,f;pin_pairs=_spice_lines_pairs,z0=[50.,75.])
            @test maximum(abs,actual.s-_spice_lines_independent(spec,f))<1e-11
            @test length(actual.gauge_nodes)==2
        end
        ordinary=planar_spice_sparams(model,1e9;pin_pairs=_spice_lines_pairs,z0=[50.,75.])
        complex_refs=ComplexF64[50+20im,75-10im]
        changed=planar_spice_sparams(model,1e9;pin_pairs=_spice_lines_pairs,z0=f->complex_refs)
        @test changed.s≈planar_renormalize_s(ordinary.s,[50.,75.],complex_refs) rtol=1e-12 atol=1e-13
        @test changed.y≈ordinary.y rtol=1e-12 atol=1e-13
    end
    literal_product=setprecision(BigFloat,512) do
        Float64(BigFloat(1e300)*BigFloat(1e300)*BigFloat(1e-300))
    end
    @test DiffMoM._spice_line_product(1e300,1e300,1e-300)==literal_product+0im
    @test DiffMoM._spice_line_product(1e-300,1e-300,1e300)≈1e-300+0im rtol=1e-15
    @test_throws ArgumentError DiffMoM._spice_line_product(1e300,1e300)
    @test_throws ArgumentError DiffMoM._spice_line_product(1e-300,1e-300)
    # Final omega*L is finite even when 2pi*f alone overflows.
    mktempdir() do directory
        path=joinpath(directory,"scale.lib")
        write(path,".subckt line a ga b gb\nO1 a ga b gb m\n.model m LTRA L=1e-308 C=1e-308 LEN=1\n.ends line\n")
        model=planar_spice_model(planar_read_spice(path),"line")
        actual=planar_spice_sparams(model,1e308;pin_pairs=[("a","ga"),("b","gb")],z0=[50.,75.])
        @test all(isfinite,actual.s)
        @test maximum(abs,actual.s-_spice_lines_independent(only(values(model.lines)),1e308))<1e-11
    end
end

@testset "Bounded line grammar and transactional attachment" begin
    mktempdir() do directory
        path=joinpath(directory,"line.lib")
        body(card)=".subckt line a ga b gb\n$card\n.ends line\n"
        invalid=["T1 a ga b gb Z0=0 TD=1n","T1 a ga b gb Z0=50","T1 a ga b gb Z0=50 TD=1n F=1Meg",
            "T1 a ga b gb Z0=50 TD=-1n","T1 a ga b gb Z0=50 F=0","T1 a ga b gb Z0=50 TD=1n NL=.25",
            "T1 a ga b gb Z0=50 F=1e308 NL=1e-308","T1 a ga b gb Z0=50 TD=1n IC=0,0,0,0",
            "T1 a ga b","O1 a ga b gb absent","O1 a ga b gb m IC=0,0,0,0",
            "O1 a ga b gb m\n.model m LTRA R=1 L=1n G=.1 C=1p LEN=1",
            "O1 a ga b gb m\n.model m LTRA R=1 L=1n C=1p",
            "O1 a ga b gb m\n.model m LTRA R=1 L=1n C=1p LEN=1 LININTERP=2",
            "O1 a ga b gb m\n.model m LTRA R=1 L=1n C=1p LEN=1 COMPACTREL=2",
            "O1 a ga b gb m\n.model m LTRA R=1 L=1n C=1p LEN=1 UNKNOWN",
            "O1 a ga b gb m\n.model m LTRA R=1 L=1n C=1p LEN=1\n.model M LTRA R=2 L=1n C=1p LEN=1"]
        for text in invalid
            write(path,body(text))
            @test_throws ArgumentError planar_spice_model(planar_read_spice(path),"line")
        end
        for text in (".model diode D IS=1e-14\n", ".subckt line a b\nW1 a b\n.ends line\n")
            write(path,text);@test_throws ArgumentError planar_read_spice(path)
        end
        write(path,body("T1 a ga b gb Z0=50 TD=1n"))
        library=planar_read_spice(path);model=planar_spice_model(library,"line")
        @test_throws ArgumentError planar_read_spice(path;max_bytes=1)
        @test_throws ArgumentError planar_spice_model(library,"line";max_bytes=1)
        @test_throws ArgumentError planar_spice_model(library,"line";max_nodes=1)
        circuit=PlanarCircuit(4,[(1,2),(3,4)];z0=f->50.)
        original=(circuit.nnodes,circuit.elements,circuit.z0)
        @test_throws ArgumentError circuit_add_spice!(circuit,[1,2,3,4],model;max_bytes=1)
        @test circuit.nnodes==original[1] && circuit.elements===original[2] && circuit.z0===original[3]
        @test_throws ArgumentError circuit_add_spice!(circuit,[1,2,3,9],model)
        @test circuit.elements===original[2]
        circuit_add_spice!(circuit,[1,2,3,4],model;name="line")
        @test_throws ArgumentError circuit_add_spice!(circuit,[1,2,3,4],model;name="LINE")
        @test length(circuit.elements)==1
        calls=Ref(0);circuit.z0=f->(calls[]+=1;50.)
        @test_throws ArgumentError solve_planar_circuit(circuit,1e9;max_bytes=1,floating_gauge=:auto)
        @test calls[]==0
        @test_throws ArgumentError solve_planar_circuit(circuit,BigFloat("1e-1000");floating_gauge=:auto)
        @test calls[]==0
        compatibility=PlanarSpiceModel(model.elements,model.nodes,model.pins,model.provenance,
            model.payload,model.library,model.subckt,model.parameters,model.couplings,model.inductive_rows)
        @test isempty(compatibility.lines)
        # Wrapper storage has the same nonzero representability contract as
        # literal numbers, and rejects before optional reference callbacks.
        write(path,".subckt line a ga b gb delay=1n\nT1 a ga b gb Z0=50 TD={delay}\n.ends line\n")
        library=planar_read_spice(path)
        for value in (BigFloat("1e-1000"),BigFloat("1e1000"))
            @test_throws ArgumentError planar_spice_model(library,"line";parameters=Dict("delay"=>value))
        end
        finite=planar_spice_model(library,"line";parameters=Dict("delay"=>BigFloat("1.25e-9")))
        @test only(values(finite.lines)).td==1.25e-9
        for f in (BigFloat("1e-1000"),BigFloat("1e1000"))
            @test_throws ArgumentError planar_spice_sparams(finite,f;pin_pairs=[("a","ga"),("b","gb")],
                z0=f->(calls[]+=1;50.))
        end
        @test calls[]==0
        # Source tokens already remove inline comments; model parentheses
        # must not mistakenly consume the retained raw comment text.
        write(path,body("O1 a ga b gb m\n.model m LTRA (R=12.45 C=.468p LEN=16) ; retained comment"))
        commented=planar_spice_model(planar_read_spice(path),"line")
        @test only(values(commented.lines)).r==12.45
        @test occursin("retained comment",only(values(commented.lines)).model_card.text)
        # Several independent small line instances keep separate namespaces.
        write(path,".subckt cell a ga b gb\nT1 a ga b gb Z0=50 TD=1n\n.ends cell\n.subckt line p1 g1 p2 g2\n"*
            join(("X$i a$i ga$i b$i gb$i cell\n" for i in 1:128))*".ends line\n")
        library=planar_read_spice(path);many=planar_spice_model(library,"line")
        @test length(many.lines)==128
        @test length(many.nodes)==516
        @test_throws ArgumentError planar_spice_model(library,"line";max_bytes=many.payload-1)
        @test_throws ArgumentError planar_spice_model(library,"line";max_elements=127)
    end
end

@testset "Project line subcircuit retains physical EM current and radiation projection" begin
    project=planar_project_from_dict(Dict("project"=>Dict("name"=>"line_loaded","unit"=>"mm"),
        "box"=>Dict("width"=>3.,"length"=>2.),"mesh"=>Dict("nx"=>6,"ny"=>4),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5,"eps_r"=>2.),Dict("thickness"=>.5)]),
        "metals"=>Dict("film"=>Dict("type"=>"surface_impedance","rs"=>.1)),
        "polygons"=>[Dict("name"=>"wire","level"=>1,"metal"=>"film","vertices"=>[[0.,.5],[3.,.5],[3.,1.5],[0.,1.5]])],
        "ports"=>[Dict("name"=>"input","number"=>1,"polygon"=>"wire","edge"=>4,"external"=>true),
            Dict("name"=>"output","number"=>2,"polygon"=>"wire","edge"=>2,"external"=>true)]))
    path=joinpath(_spice_lines_fixture,"nested_params.lib")
    project.data["components"]=[Dict("name"=>"Line","type"=>"subckt","path"=>path,"subckt"=>"device",
        "nodes"=>["input","gnd","output","gnd"])]
    direct=solve_planar_project(project,1e9;mx=20,my=16)
    model=planar_spice_model(planar_read_spice(path),"device")
    independent=planar_spice_sparams(model,1e9;pin_pairs=_spice_lines_pairs)
    alternative=planar_project_from_dict(planar_project_dict(project))
    alternative.data["components"]=[Dict("name"=>"Network","type"=>"network","ports"=>[1,2])]
    reference=solve_planar_project(alternative,1e9;mx=20,my=16,
        component_response=(p,c,f)->(response=independent.y,format=:y,z0=50.))
    @test direct.s≈reference.s rtol=1e-11 atol=1e-12
    @test direct.y≈reference.y rtol=1e-11 atol=1e-12
    waves=ComplexF64[.3+.2im,-.5im]
    nodes=view(direct.circuit.voltages*waves,1:length(direct.model.node_names))
    voltage=transpose(direct.model.source_incidence)*nodes
    maps=planar_current_maps(direct;incident_waves=waves)
    equivalent=planar_current_maps(direct.em;voltages=voltage)
    @test maps[1].jx≈equivalent[1].jx rtol=1e-13
    @test maps[1].jy≈equivalent[1].jy rtol=1e-13
    excitation=DiffMoM._project_radiation_excitation(direct;incident_waves=waves,max_bytes=10^7)
    @test excitation.coefficients≈DiffMoM._planar_coefficient_columns(direct.em)*voltage rtol=1e-13
    @test size(direct.circuit.voltages,1)>length(direct.model.node_names)
end
