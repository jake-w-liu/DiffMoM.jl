using DiffMoM, Test, LinearAlgebra, SHA, TOML

function _native_component_budget_project(kind=:ideal)
    p=read_sonnet_project(joinpath(@__DIR__,"fixtures","native_ideal_component_units","res","ohm","ohm.son"))
    t=only(filter(r->first(r.tokens)=="TYPE",only(p.components))).tokens
    kind===:ideal && return p
    empty!(t);append!(t,kind===:none ? ["TYPE","NONE"] : ["TYPE","SPROJ","child.son"])
    p
end
function _native_component_float_project(kind,value,unit)
    p=_native_component_budget_project()
    p.units[kind]=unit
    type=only(filter(r->first(r.tokens)=="TYPE",only(p.components)))
    type.tokens[:]=["TYPE","IDEAL",kind,value]
    for record in only(p.components)
        first(record.tokens)=="GNDREF" && (record.tokens[end]="FLOAT")
    end
    pins=filter(r->first(r.tokens)=="SMDP",only(p.components))
    pins[2].tokens[6]="-3"
    p
end

@testset "Native component cumulative geometry/model resources" begin
    fixture=joinpath(@__DIR__,"fixtures","native_component_budget_float")
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["files"]
    @test length(hashes)==19
    for entry in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,entry["path"]))))==entry["sha256"]
    end
    @test occursin("max_bytes=20000",read(joinpath(fixture,"native_ideal_geometry_budget_audit.log"),String))
    @test occursin("accepted",read(joinpath(fixture,"native_ideal_geometry_budget_audit.log"),String))
    @test occursin("direct_stored=0.01",read(joinpath(fixture,"native_floating_ideal_units_audit.log"),String))
    @test occursin("BigFloat_override_underflow = 0.0",read(joinpath(fixture,"native_floating_ideal_units_audit.toml"),String))
    @test occursin("allocated=16964254",read(joinpath(fixture,"native_component_callback_shape_before.log"),String))
    @test TOML.parsefile(joinpath(fixture,"native_component_callback_shape_before.toml"))["circuit_sha256"]==
        bytes2hex(sha256(read(joinpath(fixture,"PlanarCircuit_network_shape_before.jl"))))
    @testset "All model classes reject before geometry and callbacks" begin
        for kind in (:ideal,:none,:callback,:sparam)
            p=kind===:sparam ? read_sonnet_project(joinpath(@__DIR__,"fixtures","native_sparam_model_files","native","sparameter.son")) :
                _native_component_budget_project(kind)
            saved=deepcopy(p)
            calls=Ref(0)
            callback=(args...)->(calls[]+=1;(response=ComplexF64[0 1;1 0],format=:s,z0=50.))
            kw=kind===:callback ? (;component_response=callback) : (;)
            for grid in ((32,40),(128,160)),limit in (1,20000)
                @test_throws ArgumentError sonnet_component_model(p,200e6;grid,max_bytes=limit,kw...)
                @test calls[]==0
                allocation=@allocated try
                    sonnet_component_model(p,200e6;grid,max_bytes=limit,kw...)
                catch e
                    e isa ArgumentError || rethrow()
                end
                @test allocation<100000
            end
            model=sonnet_component_model(p,200e6;kw...)
            @test calls[]==(kind===:callback ? 1 : 0)
            @test model.payload>20000
            @test model.payload==DiffMoM._sonnet_component_retained_payload(model)
            @test model.payload>sizeof(ComplexF64)*32*40
            @test_throws ArgumentError sonnet_component_model(p,200e6;max_bytes=model.payload-1,kw...)
            @test calls[]==(kind===:callback ? 1 : 0)
            @test p.units==saved.units && p.variables==saved.variables
            @test [r.tokens for c in p.components for r in c]==[r.tokens for c in saved.components for r in c]
            # Real node voltages retain the complete physical source set.
            solved=solve_sonnet_project(p,200e6;raw=true,mx=64,my=80,kw...)
            @test all(isfinite,solved.s) && opnorm(solved.s)<=1+1e-10
            @test solved.s≈transpose(solved.s) rtol=1e-10
            waves=ComplexF64[.3+.2im,-.1im]
            maps=sonnet_component_current_maps(solved;incident_waves=waves)
            manual=planar_current_maps(solved.raw;voltages=solved.contraction*(solved.circuit.voltages*waves))
            @test maps[1].jx≈manual[1].jx rtol=1e-10
            @test maps[end].jz≈manual[end].jz rtol=1e-10
        end
        p=_native_component_budget_project(:callback)
        calls=Ref(0)
        @test_throws ArgumentError sonnet_component_model(p,200e6;
            variables=Dict("UNUSED"=>1.),max_bytes=1,component_response=(args...)->(calls[]+=1;error("callback")))
        @test calls[]==0
        @test_throws ArgumentError sonnet_component_model(p,200e6;variables=:invalid)
        # The callback's preexisting malformed table is never copied into
        # the two-pin block. Its own memory belongs to the external caller.
        response=zeros(ComplexF64,1024,1024)
        callback=(args...)->(response=response,format=:s,z0=50.)
        @test_throws ArgumentError sonnet_component_model(p,200e6;
            component_response=callback,max_bytes=10000000)
        allocation=@allocated try
            sonnet_component_model(p,200e6;component_response=callback,max_bytes=10000000)
        catch e
            e isa ArgumentError || rethrow()
        end
        @test allocation<1000000
    end
end

@testset "Native FLOAT uses shared stored IDEAL units and preflight" begin
    waves=ComplexF64[.3+.2im,-.1im]
    @testset "Equivalent physical units, oriented source nodes and currents" begin
        for (kind,field,basevalue,baseunit,value,unit) in (
                ("RES",:r,"16.77","OH",".01677","KOH"),
                ("CAP",:c,"1","PF",".001","NF"),
                ("IND",:l,"1","NH",".001","UH"))
            base=_native_component_float_project(kind,basevalue,baseunit)
            equivalent=_native_component_float_project(kind,value,unit)
            a=sonnet_floating_model(base,200e6);b=sonnet_floating_model(equivalent,200e6)
            wrapper=sonnet_component_model(equivalent,200e6)
            @test getproperty(only(a.circuit.elements),field)≈getproperty(only(b.circuit.elements),field) rtol=2eps(Float64)
            @test getproperty(only(wrapper.circuit.elements),field)==getproperty(only(b.circuit.elements),field)
            @test a.source_terminals==b.source_terminals==[(1,0),(2,0),(3,4)]
            @test isempty(a.problem.vias) && isempty(b.problem.vias)
            left=solve_sonnet_project(base,200e6;raw=true,mx=64,my=80)
            right=solve_sonnet_project(equivalent,200e6;raw=true,mx=64,my=80)
            @test maximum(abs,left.s-right.s)<1e-10
            @test left.circuit.voltages≈right.circuit.voltages rtol=1e-10
            @test left.circuit.gauge_nodes==right.circuit.gauge_nodes==[3]
            @test right.s≈transpose(right.s) rtol=1e-10
            @test opnorm(right.s)<=1+1e-10
            actual=planar_current_maps(right;incident_waves=waves)
            original=planar_current_maps(left;incident_waves=waves)
            @test actual[1].jx≈original[1].jx rtol=1e-10
            expected=planar_current_maps(right.em;voltages=transpose(right.source_incidence)*(right.circuit.voltages*waves))
            @test actual[1].jx≈expected[1].jx rtol=1e-10
            changed=_native_component_float_project(kind,basevalue,unit)
            @test getproperty(only(sonnet_floating_model(changed,200e6).circuit.elements),field)>
                100getproperty(only(a.circuit.elements),field)
        end
    end
    @testset "Stored domains reject before geometry/providers" begin
        p=_native_component_float_project("RES","RVALUE","KOH")
        p.variables["RVALUE"]=".01677"
        values=Dict("RVALUE"=>BigFloat(".01677"));saved=copy(values)
        @test only(sonnet_floating_model(p,BigFloat("200000000");variables=values).circuit.elements).r≈16.77
        @test values==saved
        calls=Ref(0)
        callback=(args...)->(calls[]+=1;(response=reshape(ComplexF64[.1],1,1),format=:y,z0=50.))
        for f in (BigFloat("1e-1000"),BigFloat("1e1000"),-1.,0.,Inf,NaN)
            @test_throws ArgumentError sonnet_floating_model(p,f;component_response=callback)
            @test_throws ArgumentError solve_sonnet_project(p,f;raw=true,component_response=callback)
            @test calls[]==0
        end
        for value in (BigFloat("1e-1000"),BigFloat("1e1000"),Inf,NaN,1im)
            @test_throws ArgumentError sonnet_floating_model(p,200e6;variables=Dict("RVALUE"=>value),component_response=callback)
            @test calls[]==0
        end
        @test_throws ArgumentError sonnet_floating_model(p,200e6;variables=:invalid)
        @test_throws ArgumentError sonnet_floating_model(p,200e6;variables=values,max_bytes=1,component_response=callback)
        @test calls[]==0 && values==saved
        bad=_native_component_float_project("RES","1","INVALID")
        first(filter(r->first(r.tokens)=="SMDP",only(bad.components))).tokens[3]=".1"
        err=try sonnet_floating_model(bad,200e6;component_response=callback);nothing catch e;e end
        @test err isa ArgumentError && occursin("IDEAL RES units",sprint(showerror,err))
        @test calls[]==0
        underflow=_native_component_float_project("RES","1e-1000","OH")
        @test_throws ArgumentError sonnet_floating_model(underflow,200e6)
    end
    @testset "FREQ Hz and explicit frequency callbacks preserve loaded physical modes" begin
        for f in (200e6,400e6)
            dynamic=_native_component_float_project("RES",".01677*FREQ/200000000","KOH")
            equivalent=_native_component_float_project("RES",string(16.77*f/200e6),"OH")
            actual=solve_sonnet_project(dynamic,f;raw=true,mx=64,my=80)
            reference=solve_sonnet_project(equivalent,f;raw=true,mx=64,my=80)
            @test maximum(abs,actual.s-reference.s)<1e-10
            @test actual.circuit.voltages≈reference.circuit.voltages rtol=1e-10
            @test planar_current_maps(actual;incident_waves=waves)[1].jx≈
                planar_current_maps(reference;incident_waves=waves)[1].jx rtol=1e-10
            vendor=_native_component_float_project("RES","1","OH")
            t=only(filter(r->first(r.tokens)=="TYPE",only(vendor.components))).tokens
            empty!(t);append!(t,["TYPE","SPROJ","explicit.son"])
            calls=Ref(0)
            callback=(p,c,f)->(calls[]+=1;(response=reshape(ComplexF64[200e6/(16.77*f)],1,1),format=:y,z0=f->50+10im))
            loaded=solve_sonnet_project(vendor,f;raw=true,mx=64,my=80,component_response=callback)
            @test calls[]==1
            @test maximum(abs,loaded.s-reference.s)<1e-10
            @test loaded.circuit.voltages≈reference.circuit.voltages rtol=1e-10
            @test planar_current_maps(loaded;incident_waves=waves)[1].jx≈
                planar_current_maps(reference;incident_waves=waves)[1].jx rtol=1e-10
        end
    end
end
