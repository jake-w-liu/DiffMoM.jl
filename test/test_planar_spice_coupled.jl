using DiffMoM,Test,LinearAlgebra,SHA,TOML,JSON
const _coupled_fixture=joinpath(@__DIR__,"fixtures","spice_coupling_ngspice47")

function _retained_spice_wave(path,np,z0)
    lines=readlines(path);frequencies=Float64[];columns=Vector{ComplexF64}[]
    for line in lines[2:end]
        t=parse.(Float64,split(line));@test length(t)==1+4np
        push!(frequencies,t[1]);wave=ComplexF64[]
        for p in 1:np
            voltage=complex(t[4p-2],t[4p-1]);current=-complex(t[4p],t[4p+1])
            push!(wave,(voltage-z0[p]*current)/(2sqrt(z0[p])))
        end
        push!(columns,wave)
    end
    frequencies,columns
end

@testset "Coupled SPICE metadata and retained actual ngspice full S" begin
    hashes=JSON.parse(read(joinpath(_coupled_fixture,"sha256.json"),String))
    @test all(bytes2hex(sha256(read(joinpath(_coupled_fixture,replace(path,'\\'=>'/')))))==digest for (path,digest) in hashes)
    reference=TOML.parsefile(joinpath(_coupled_fixture,"comparison.toml"))
    @test reference["engine_sha256"]=="22d5cae2bd32b2e39157a8d27bf457122f68285b72a9ebefdf41551b628233ab"
    for name in ("forward","reverse_winding","negative_coupling","zero_coupling",
            "unit_coupling","negative_unit_coupling","nested_global_zero","three_inductor_equal_k")
        folder=joinpath(_coupled_fixture,name);top=name=="nested_global_zero" ? "wrap" : "coils"
        library=planar_read_spice(joinpath(folder,"model.lib"));model=planar_spice_model(library,top)
        formal=library.definitions[top].pins;ports=[(p,"ref") for p in formal if p!="ref"]
        np=length(ports);refs=collect(range(50.;step=25.,length=np))
        @test length(model.couplings)==(name=="nested_global_zero" ? 2 : 1)
        @test all(e->e.kind!='K',model.elements)
        @test all(c->isfile(c.card.path)&&c.card.line>0,model.couplings)
        expected=Matrix{ComplexF64}[];frequencies=Float64[]
        for drive in 1:np
            fs,columns=_retained_spice_wave(joinpath(folder,"$(top)_wave$(drive).cir.dat"),np,refs)
            drive==1 && (frequencies=fs;expected=[zeros(ComplexF64,np,np) for f in fs])
            @test fs==frequencies
            for i in eachindex(fs);expected[i][:,drive]=columns[i];end
        end
        for (f,S) in zip(frequencies,expected)
            actual=planar_spice_sparams(model,f;pin_pairs=ports,z0=refs)
            @test maximum(abs,actual.s-S)<1e-10
            @test actual.s≈transpose(actual.s) rtol=1e-11 atol=1e-12
            @test opnorm(actual.s)<=1+1e-12
        end
        circuit=PlanarCircuit(length(formal),collect(1:np);z0=refs)
        circuit_add_spice!(circuit,collect(1:length(formal)),model)
        @test length(circuit.elements)==length(model.elements)
    end
end

@testset "Coupled physical branch voltage and perfect winding constraints" begin
    for name in ("forward","unit_coupling","negative_unit_coupling")
        model=planar_spice_model(planar_read_spice(joinpath(_coupled_fixture,name,"model.lib")),"coils")
        k=only(model.couplings).coefficient;L1,L2=1e-9,2e-9
        for f in (0.,1e6,2e9,20e9)
            r=planar_spice_sparams(model,f;pin_pairs=[("p1","ref"),("p2","ref")],z0=[50.,75.])
            v=r.voltages;id(s)=model.nodes[s]
            a,b,g=v[id("xroot/a"),:],v[id("xroot/b"),:],v[id("pin.ref"),:]
            i1=(v[id("pin.p1"),:]-a)/20;i2=(v[id("pin.p2"),:]-b)/30
            mutual=k*sqrt(L1)*sqrt(L2)
            @test norm(a-g-2pi*1im*f*(L1*i1+mutual*i2))<1e-11
            @test norm(b-g-2pi*1im*f*(mutual*i1+L2*i2))<1e-11
            abs(k)==1 && @test norm((a-g)/sqrt(L1)-k*(b-g)/sqrt(L2))<1e-8
        end
    end
end

@testset "Coupled isolated returns retain two gauges and common-mode invariance" begin
    model=planar_spice_model(planar_read_spice(joinpath(_coupled_fixture,"isolated.lib")),"isolation")
    refs=[50.,75.];ports=[("p1","g1"),("p2","g2")]
    plain=Matrix{ComplexF64}[];shifted=Matrix{ComplexF64}[];fs=Float64[]
    for (prefix,target) in (("isolation",plain),("isolation_shifted",shifted)),drive in 1:2
        frequencies,columns=_retained_spice_wave(joinpath(_coupled_fixture,"$(prefix)_wave$(drive).cir.dat"),2,refs)
        drive==1 && (append!(target,[zeros(ComplexF64,2,2) for f in frequencies]);fs=frequencies)
        for i in eachindex(fs);target[i][:,drive]=columns[i];end
    end
    for (f,S,T) in zip(fs,plain,shifted)
        actual=planar_spice_sparams(model,f;pin_pairs=ports,z0=refs)
        @test length(actual.gauge_nodes)==2
        @test maximum(abs,actual.s-S)<1e-10
        @test maximum(abs,S-T)<1e-10
    end
end

@testset "K exact energy, zero L, scale safety and transactional resources" begin
    mktempdir() do directory
        function compile(body;kw...)
            path=joinpath(directory,"case.lib");write(path,".subckt case p1 p2 ref\n"*body*"\n.ends case\n")
            planar_spice_model(planar_read_spice(path),"case";kw...)
        end
        for name in ("bad_missing.lib","bad_same.lib","bad_wrongkind.lib","bad_too_negative.lib",
                "bad_nonfinite.lib","bad_too_large.lib","bad_duplicate_pair.lib","bad_repeated_name.lib","bad_multi_duplicate.lib","indefinite.lib")
            @test_throws Exception planar_spice_model(planar_read_spice(joinpath(_coupled_fixture,name)),name=="indefinite.lib" ? "indefinite" : "bad")
        end
        for a in (sqrt(.5),nextfloat(sqrt(.5)))
            @test_throws ArgumentError compile("L1 p1 ref 1\nL2 p2 ref 1\nL3 p1 p2 1\nK12 L1 L2 $a\nK23 L2 L3 $a")
        end
        @test length(compile("L1 p1 ref 1\nL2 p2 ref 1\nL3 p1 p2 1\nK12 L1 L2 $(prevfloat(sqrt(.5)))\nK23 L2 L3 $(prevfloat(sqrt(.5)))").couplings)==2
        @test length(compile("L1 p1 ref 1\nL2 p2 ref 1\nL3 p1 p2 1\nK12 L1 L2 1\nK13 L1 L3 .5\nK23 L2 L3 .5").couplings)==3
        @test_throws ArgumentError compile("L1 p1 ref 1e-300\nL2 p2 ref 1e300\nK1 L1 L2 1e-300")
        @test_throws ArgumentError compile("L1 p1 ref -1n\nL2 p2 ref 1n\nK1 L1 L2 .5")
        zero=compile("R1 p1 a 20\nR2 p2 b 30\nL1 a ref 0\nL2 b ref 1n\nK1 L1 L2 1")
        @test isempty(zero.inductive_rows)
        @test all(isfinite,planar_spice_sparams(zero,1e9;pin_pairs=[("p1","ref"),("p2","ref")]).s)
        for (L,f) in ((1e300,1e-300),(1e-300,1e300))
            model=compile("R1 p1 a 20\nR2 p2 b 30\nL1 a ref $L\nL2 b ref $L\nK1 L1 L2 .5")
            actual=planar_spice_sparams(model,f;pin_pairs=[("p1","ref"),("p2","ref")],z0=[50.,75.])
            control=compile("R1 p1 a 20\nR2 p2 b 30\nL1 a ref 1\nL2 b ref 1\nK1 L1 L2 .5")
            @test actual.s≈planar_spice_sparams(control,1.;pin_pairs=[("p1","ref"),("p2","ref")],z0=[50.,75.]).s rtol=1e-12
            @test abs(actual.s[1,2])>1e-3
        end
        source=planar_read_spice(joinpath(_coupled_fixture,"forward","model.lib"))
        model=planar_spice_model(source,"coils")
        for kw in ((max_bytes=1,),(max_elements=1,),(max_nodes=1,),(max_coupled_size=1,),(max_coupling_pairs=1,max_elements=1))
            @test_throws ArgumentError planar_spice_model(source,"coils";kw...)
        end
        c=PlanarCircuit(3,[1,2]);old=copy(c.elements);nn=c.nnodes
        @test_throws ArgumentError circuit_add_spice!(c,[1,2,3],model;max_bytes=1)
        @test c.nnodes==nn && c.elements==old
        circuit_add_spice!(c,[1,2,3],model;name="coupled")
        @test_throws ArgumentError circuit_add_spice!(c,[1,2,3],model;name="COUPLED")
        @test_throws ArgumentError solve_planar_circuit(c,1e9;max_bytes=1)
        # Many independent small groups use sequential exact workspaces.
        body=join(("L$(i)a p1 ref 1n\nL$(i)b p2 ref 2n\nK$i L$(i)a L$(i)b .8" for i in 1:150),"\n")
        many=compile(body;max_bytes=2*1024^2)
        @test length(many.couplings)==150 && length(many.elements)==300
        chain=join(vcat(["L$i p1 ref 1n" for i in 1:257],
            ["K$i L$i L$(i+1) .1" for i in 1:256]),"\n")
        oversized=try compile(chain;max_bytes=16*1024^2);nothing catch e;e end
        @test oversized isa ArgumentError
        @test occursin("max_coupled_size",sprint(showerror,oversized))
        # Pair limits include zero interactions, even though their energy rows decouple.
        @test_throws ArgumentError compile("L1 p1 ref 1n\nL2 p2 ref 1n\nL3 p1 p2 1n\nK1 L1 L2 L3 0";
            max_coupling_pairs=2)
    end
end
