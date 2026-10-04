using DiffMoM,Test,LinearAlgebra,SHA,TOML

const _spice_fixture=joinpath(@__DIR__,"fixtures","spice_linear_ngspice47")
@testset "SPICE original actual ngspice evidence and safe library grammar" begin
    manifest=TOML.parsefile(joinpath(_spice_fixture,"sha256.toml"))
    @test all(bytes2hex(sha256(read(joinpath(_spice_fixture,path))))==digest for (path,digest) in manifest)
    evidence=TOML.parsefile(joinpath(_spice_fixture,"comparison.toml"))
    @test evidence["engine_sha256"]=="22d5cae2bd32b2e39157a8d27bf457122f68285b72a9ebefdf41551b628233ab"
    for (literal,value) in (("1M",1e-3),("1MEGohm",1e6),("1000Hz",1000.),("1mSec",.001),
            ("1mil",25.4e-6),("1a",1e-18),(".25kV",250.),("1e-3uF",1e-9),("10V",10.))
        @test DiffMoM._spice_number(literal)≈value
    end
    @test DiffMoM._spice_expression("{r2 * 2 + 1k}",Dict("r2"=>25.))==1050
    @test DiffMoM._spice_expression("'sqrt(9)+2meg'",Dict{String,Float64}())==2000003
    for text in ("{run(`echo hi`)}","{open(\"file\")}","{[1,2]}","{unknown}","1e999",repeat("1+",3000)*"1",repeat("(",33)*"1"*repeat(")",33))
        @test_throws Exception DiffMoM._spice_expression(text,Dict{String,Float64}())
    end
    path=joinpath(_spice_fixture,"ladder.lib");lib=planar_read_spice(path)
    @test length(lib.provenance)==2
    @test lib.provenance[realpath(joinpath(_spice_fixture,"section.inc"))]==bytes2hex(sha256(read(joinpath(_spice_fixture,"section.inc"))))
    @test length(lib.includes)==1
    @test occursin(".include",lib.includes[1][3])
    @test all(c->isfile(c.path)&&c.line>0&&!isempty(c.text),lib.records)
    model=planar_spice_model(lib,"LADDER";parameters=Dict("rval"=>10.))
    @test length(model.elements)==6
    @test [e.value for e in model.elements if e.kind=='R']==[10.,20.]
    @test length(unique(e.name for e in model.elements))==6
    for kw in ((max_bytes=1,),(max_records=1,),(max_depth=1,),(max_line_bytes=10,))
        @test_throws ArgumentError planar_read_spice(path;kw...)
    end
    for kw in ((max_elements=1,),(max_nodes=1,),(max_bytes=1,),(max_depth=1,))
        @test_throws ArgumentError planar_spice_model(lib,"ladder";kw...)
    end
    @test_throws ArgumentError planar_spice_model(lib,"ladder";parameters=Dict("rval"=>1,"RVAL"=>2))
    @test_throws ArgumentError planar_spice_model(lib,"ladder";parameters=Dict("rval"=>Inf))
    @test_throws ArgumentError planar_spice_model(lib,"ladder";pin_nodes=Dict("p1"=>"x","P1"=>"x"))
    @test_throws ArgumentError planar_read_spice(path;root=path)
    @test_throws ArgumentError planar_spice_sparams(model,1e9;pin_pairs=[("p1","ref"),("p2","ref")],max_bytes=1)
    for name in ("nonlinear","behavioural","poly","model","global","analysis","acsource","waveform",
            "selfparam","undefined","duplicate","badcontrol","unknownx","xpin","recursive","endname")
        @test_throws Exception planar_spice_model(planar_read_spice(joinpath(_spice_fixture,name*".lib")),"bad")
    end
    for name in ("cyclic","absolute","escape","duplicate_subckt","zero_pin","orphan")
        @test_throws Exception planar_read_spice(joinpath(_spice_fixture,name*".lib"))
    end
end

@testset "SPICE real exported rational model ingestion vs independent equations and engine" begin
    fixture=joinpath(@__DIR__,"fixtures","spice_export_ngspice47")
    manifest=TOML.parsefile(joinpath(fixture,"sha256.toml"))
    @test all(bytes2hex(sha256(read(joinpath(fixture,path))))==digest for (path,digest) in manifest)
    D=[.02 -.005;-.005 .03];E=[1e-12 -2e-13;-2e-13 2e-12]
    R=[1e8 -2e7;-2e7 8e7];Rc=ComplexF64[1e7+2e6im -2e6+1e6im;-2e6+1e6im 8e6+1e6im]
    pole=-1e9+3e9im
    for name in ("real_pole_affine_capacitance","complex_conjugate_pair"),floating in (false,true)
        folder=joinpath(fixture,name*(floating ? "_floating_reference" : "_grounded"))
        model=planar_spice_model(planar_read_spice(joinpath(folder,"model.cir")),"diffmom_nport";
            pin_nodes=floating ? Dict{String,String}() : Dict("dm_ref"=>"0"))
        lines=readlines(joinpath(folder,"drive1.dat"));frequencies=Float64[]
        Y=[zeros(ComplexF64,2,2) for _ in 2:length(lines)]
        for drive in 1:2
            data=readlines(joinpath(folder,"drive$drive.dat"))
            for k in 2:length(data)
                v=parse.(Float64,split(data[k]));drive==1 && push!(frequencies,v[1])
                for p in 1:2;Y[k-1][p,drive]=-complex(v[2p],v[2p+1]);end
            end
        end
        for (k,f) in enumerate(frequencies)
            s=2pi*im*f
            analytic=name=="real_pole_affine_capacitance" ? D+s*E+R/(s+2e9) :
                D+Rc/(s-pole)+conj.(Rc)/(s-conj(pole))
            result=planar_spice_sparams(model,f;pin_pairs=[("p1","dm_ref"),("p2","dm_ref")],z0=[50,75])
            @test result.y≈analytic rtol=1e-10 atol=1e-12
            @test result.y≈Y[k] rtol=1e-10 atol=1e-12
            @test result.s≈planar_y_to_s(Y[k],[50,75]) rtol=1e-10 atol=1e-12
            @test length(result.gauge_nodes)==(floating ? 1 : 0)
        end
    end
end

# Original engine files are literal terminal voltages from independent
# Thevenin sources. These are not regenerated by DiffMoM in the default suite.
@testset "SPICE production ingestion vs actual ngspice full complex S" begin
    frequencies=collect(range(1e3,1e10;length=21))
    for name in ("ladder","controlled","tie","gain"),floating in (false,true)
        parameters=name=="ladder" ? Dict("rval"=>10.) : Dict{String,Float64}()
        model=planar_spice_model(planar_read_spice(joinpath(_spice_fixture,name*".lib")),name;
            parameters,pin_nodes=floating ? Dict{String,String}() : Dict("ref"=>"0"))
        refs=[50.,75.];folder=joinpath(_spice_fixture,name*(floating ? "_floating" : "_grounded"))
        actual=[zeros(ComplexF64,2,2) for _ in frequencies]
        for drive in 1:2
            lines=readlines(joinpath(folder,"drive$drive.dat"))
            @test length(lines)==22
            for (k,f) in enumerate(frequencies)
                v=parse.(Float64,split(lines[k+1]));@test v[1]≈f rtol=1e-13
                for p in 1:2;actual[k][p,drive]=complex(v[2p],v[2p+1])/sqrt(refs[p])-(p==drive);end
            end
        end
        for (k,f) in enumerate(frequencies)
            result=planar_spice_sparams(model,f;pin_pairs=[("p1","ref"),("p2","ref")],z0=refs)
            @test maximum(abs,result.s-actual[k])<1e-10
            @test length(result.gauge_nodes)==(floating ? 1 : 0)
            @test size(result.voltages,1)>=length(model.pins)-(floating ? 0 : 1)
        end
    end
    model=planar_spice_model(planar_read_spice(joinpath(_spice_fixture,"global_ground.lib")),"global_ground")
    refs=[50.,75.,60.];actual=[zeros(ComplexF64,3,3) for _ in frequencies]
    for drive in 1:3
        lines=readlines(joinpath(_spice_fixture,"global_ground_grounded","drive$drive.dat"))
        for (k,f) in enumerate(frequencies)
            v=parse.(Float64,split(lines[k+1]));@test v[1]≈f rtol=1e-13
            for p in 1:3;actual[k][p,drive]=complex(v[2p],v[2p+1])/sqrt(refs[p])-(p==drive);end
        end
    end
    for (k,f) in enumerate(frequencies)
        result=planar_spice_sparams(model,f;pin_pairs=[("p1","0"),("p2","0"),("ref","0")],z0=refs)
        @test maximum(abs,result.s-actual[k])<1e-10
        @test isempty(result.gauge_nodes)
    end
end

@testset "SPICE exact constraints, transactional attachment and control falsifiers" begin
    tie=planar_spice_model(planar_read_spice(joinpath(_spice_fixture,"tie.lib")),"tie")
    @test planar_spice_sparams(tie,0.;pin_pairs=[("p1","ref"),("p2","ref")]).s≈[-.2 .8;.8 -.2] atol=1e-13
    @test_throws ArgumentError planar_spice_sparams(tie,1e9;pin_pairs=[("p1","ref"),("p2","ref")],floating_gauge=:reject)
    c=PlanarCircuit(3,[(1,3),(2,3)]);original=c.elements
    @test_throws ArgumentError circuit_add_spice!(c,[1,2,3],tie;max_bytes=1)
    @test c.nnodes==3 && c.elements===original
    @test_throws ArgumentError circuit_add_spice!(c,[1,2],tie)
    @test_throws ArgumentError circuit_add_spice!(c,[1,2,4],tie)
    @test circuit_add_spice!(c,[1,2,3],tie;name="U1")===c
    @test_throws ArgumentError circuit_add_spice!(c,[1,2,3],tie;name="u1")
    @test c.elements[1].owner.model===tie
    @test c.elements[1].origin.card.line>0
    shorted=planar_spice_model(tie.library,"tie";pin_nodes=Dict("ref"=>"0"))
    c=PlanarCircuit(3,[(1,3),(2,3)]);original=c.elements
    @test_throws ArgumentError circuit_add_spice!(c,[1,2,3],shorted)
    @test c.nnodes==3 && c.elements===original
    aliased=planar_spice_model(tie.library,"tie";pin_nodes=Dict("p1"=>"common","p2"=>"common"))
    @test_throws ArgumentError circuit_add_spice!(c,[1,2,3],aliased)
    calls=Ref(0);small=PlanarCircuit(3,[(1,3),(2,3)];z0=f->(calls[]+=1;50.))
    circuit_add_spice!(small,[1,2,3],tie)
    @test_throws ArgumentError solve_planar_circuit(small,1e9;max_bytes=tie.payload-1,floating_gauge=:auto)
    @test calls[]==0
    mktempdir() do directory
        # Empty expansion still owns lexical contexts; it cannot evade its
        # storage budget by having zero primitive elements.
        path=joinpath(directory,"contexts.lib")
        write(path,".subckt leaf a k=1\n.ends leaf\n.subckt tree a\n"*
            join(("X$k a leaf k=$k" for k in 1:100),"\n")*"\n.ends tree\n")
        library=planar_read_spice(path)
        @test_throws ArgumentError planar_spice_model(library,"tree";max_bytes=library.payload+2000)
        model=planar_spice_model(library,"tree")
        @test isempty(model.elements) && model.payload>library.payload+100*512
    end
    mktempdir() do directory
        path=joinpath(directory,"falsifiers.lib")
        write(path,"""
        .subckt zero p1 p2 ref
        Rin p1 ref 100
        Fzero c ref Vsense 0
        Hzero h ref Vsense 0
        Ezero e ref p1 ref 0
        Gzero p2 ref p1 ref 0
        Rc c ref 200
        Rh h p2 100
        Re e p2 100
        Vsense p1 sense 0
        Rsense sense ref 100
        .ends zero
        .subckt disconnected p1 p2 unused
        Ronly p1 p2 100
        .ends disconnected
        .subckt literal p1 p2 ref
        R1 p1 ref 50
        R2 p2 ref 75
        Rg ref GND 200
        .ends literal
        """)
        lib=planar_read_spice(path)
        zero=planar_spice_model(lib,"zero")
        result=planar_spice_sparams(zero,1e9;pin_pairs=[("p1","ref"),("p2","ref")])
        @test result.s≈diagm([0.,0.]) atol=1e-13 # both actual input resistances are 50 Ω
        disconnected=planar_spice_sparams(planar_spice_model(lib,"disconnected"),1e9;pin_pairs=[("p1","p2")])
        @test disconnected.gauge_nodes==[1,3]
        @test disconnected.s[1,1]≈1/3
        @test disconnected.voltages[3,1]==0
        literal=planar_spice_model(lib,"literal")
        @test any(e->0 in e.nodes,literal.elements)
        result=planar_spice_sparams(literal,1e9;pin_pairs=[("p1","0"),("p2","0"),("ref","0")],z0=[50,75,60])
        D=[1. 0.;0. 1.;-1. -1.]
        Y=D*Diagonal([1/50,1/75])*transpose(D)+diagm([0.,0.,1/200])
        @test result.s≈planar_y_to_s(Y,[50,75,60]) rtol=1e-13 atol=1e-14
    end
end

@testset "Accepted null common-mode Y blocks retain the correct floating gauge" begin
    for refs in ([50.],[50+20im]),orientation in (1,-1),returnnode in (0,3)
        n=returnnode==0 ? 2 : 3;c=PlanarCircuit(n,[(1,2)];z0=refs)
        terminals=orientation==1 ? [(1,returnnode),(2,returnnode)] : [(returnnode,1),(returnnode,2)]
        circuit_add_network!(c,terminals,.01*[1 -1;-1 1];format=:y)
        @test_throws ArgumentError solve_planar_circuit(c,1e9)
        result=solve_planar_circuit(c,1e9;floating_gauge=:auto)
        @test result.s≈planar_y_to_s(fill(.01,1,1),refs) rtol=1e-13
        @test length(result.gauge_nodes)==(returnnode==0 ? 1 : 2)
        @test (result.voltages[1,1]-result.voltages[2,1])/100≈result.currents[1,1]
    end
    c=PlanarCircuit(2,[(1,2)])
    # A small finite path must never become a gauge through a tolerance.
    H=.01*[1 -1;-1 1]+1e-8I;circuit_add_network!(c,[(1,0),(2,0)],H;format=:y)
    result=solve_planar_circuit(c,1e9;floating_gauge=:auto)
    @test isempty(result.gauge_nodes)
    @test result.y[1,1]≈(.01+5e-9) rtol=1e-10
    @test !DiffMoM._circuit_uniform_y_null([1 -1;0 0])
    @test !DiffMoM._circuit_uniform_y_null([1 0;-1 0])
    calls=Ref(0);c=PlanarCircuit(2,[(1,2)])
    circuit_add_network!(c,[(1,0),(2,0)],f->(calls[]+=1;.01*[1 -1;-1 1]);format=:y)
    @test_throws ArgumentError solve_planar_circuit(c,1e9;floating_gauge=:auto,max_bytes=1)
    @test calls[]==0
    @test solve_planar_circuit(c,1e9;floating_gauge=:auto).s[1,1]≈1/3
    @test calls[]==1
    sparse=PlanarCircuit(11,[(1,11)]);circuit_add_rlc!(sparse,1,11;r=100.)
    @test_throws ArgumentError solve_planar_circuit(sparse,1e9)
    result=solve_planar_circuit(sparse,1e9;floating_gauge=:auto)
    @test result.gauge_nodes==collect(1:10)
    @test result.s[1,1]≈1/3
    @test all(iszero,view(result.voltages,2:10,:))
end
