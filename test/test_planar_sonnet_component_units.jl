using Test, DiffMoM, SHA, TOML, LinearAlgebra

function _ideal_unit_project(project,kind,value;unit=nothing)
    p=deepcopy(project)
    record=only(filter(r->first(r.tokens)=="TYPE",only(p.components)))
    record.tokens[:]=["TYPE","IDEAL",kind,string(value)]
    unit===nothing || (p.units[kind]=unit)
    p
end

@testset "Native IDEAL component DIM units and stored domains" begin
    fixture=joinpath(@__DIR__,"fixtures","native_ideal_component_units")
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["files"]
    @test length(hashes)==116
    for entry in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,entry["path"]))))==entry["sha256"]
    end
    @test occursin("storedR=16.77 fullS_native_error=0.8505",
        read(joinpath(fixture,"native_ideal_units_before.log"),String))
    @test occursin("storedR=16770.0",read(joinpath(fixture,"native_ideal_units_after.log"),String))
    cases=(
        ("res","ohm",:r,16.77),
        ("res","kohm_same_literal",:r,16770.),
        ("res","kohm_same_physical",:r,16.77),
        ("reactive","cap_base",:c,1e-12),
        ("reactive","cap_same_literal",:c,1e-9),
        ("reactive","cap_same_physical",:c,1e-12),
        ("reactive","ind_base",:l,1e-9),
        ("reactive","ind_same_literal",:l,1e-6),
        ("reactive","ind_same_physical",:l,1e-9))
    projects=Dict{String,SonnetProject}()
    results=Dict{String,SonnetComponentResult}()
    natives=Dict{String,Matrix{ComplexF64}}()
    waves=ComplexF64[.2+.1im,-.3im]
    @testset "Exact native sources, full complex response and loaded physical currents" begin
        for (group,tag,field,value) in cases
            dir=joinpath(fixture,group,tag)
            metadata=TOML.parsefile(joinpath(dir,"metadata.toml"))
            @test metadata["process_success"] && !metadata["deembedded"]
            @test metadata["actual_cell_counts"]==[32,40]
            source=joinpath(dir,tag*".son")
            @test bytes2hex(sha256(read(source)))==metadata["source_sha256"]
            check=metadata["touchstone_selected_log_checks"]["native_device.s2p"]
            @test check["status"]=="PASS" && check["intended_log_section"]=="raw"
            @test check["output_reference_real"]==[[50.,50.]]
            @test check["output_reference_imag"]==[[0.,0.]]
            p=read_sonnet_project(source);projects[tag]=p
            model=sonnet_component_model(p,200e6)
            @test getproperty(only(model.circuit.elements),field)≈value rtol=2eps(Float64)
            @test model.external_labels==[1,2] && model.labels==[1,2,3,4]
            @test model.circuit.z0==ComplexF64[50,50]
            result=solve_sonnet_project(p,200e6;raw=true,mx=64,my=80)
            results[tag]=result
            native=planar_network_response(planar_read_touchstone(joinpath(dir,"native_device.s2p")),200e6)
            natives[tag]=native
            # Predeclared raw physical-return fixture gate, not pin co-calibration.
            @test maximum(abs,result.s-native)<.03
            @test result.s≈transpose(result.s) rtol=1e-10
            @test opnorm(result.s)<=1+1e-10
            actual=sonnet_component_current_maps(result;incident_waves=waves)
            physical_nodes=result.circuit.voltages*waves
            independent=planar_current_maps(result.raw;voltages=result.contraction*physical_nodes)
            @test actual[1].jx≈independent[1].jx rtol=1e-10
            @test actual[1].jy≈independent[1].jy rtol=1e-10
            @test actual[end].jz≈independent[end].jz rtol=1e-10
            @test all(isfinite,physical_nodes)
        end
        for (base,equal,changed) in (("ohm","kohm_same_physical","kohm_same_literal"),
                ("cap_base","cap_same_physical","cap_same_literal"),
                ("ind_base","ind_same_physical","ind_same_literal"))
            @test natives[base]==natives[equal]
            @test maximum(abs,natives[base]-natives[changed])>.8
            @test maximum(abs,results[base].s-results[equal].s)<1e-10
            @test maximum(abs,results[base].s-results[changed].s)>.8
            @test results[base].circuit.voltages≈results[equal].circuit.voltages rtol=1e-10
            left=sonnet_component_current_maps(results[base];incident_waves=waves)
            right=sonnet_component_current_maps(results[equal];incident_waves=waves)
            @test left[1].jx≈right[1].jx rtol=1e-10
            @test left[1].jy≈right[1].jy rtol=1e-10
            @test left[end].jz≈right[end].jz rtol=1e-10
        end
    end
    base=projects["ohm"]
    @testset "Scalar, frequency and callback preflight" begin
        calls=Ref(0)
        callback=(args...)->(calls[]+=1;(response=zeros(ComplexF64,2,2),format=:s,z0=50.))
        for (unit,literal) in (("OH","16.77"),("OHMS","16.77"),("KOH",".01677"),("MOH",".00001677"))
            p=_ideal_unit_project(base,"RES",literal;unit)
            @test only(sonnet_component_model(p,200e6).circuit.elements).r≈16.77 rtol=2eps(Float64)
        end
        for (kind,unit,literal) in (("RES","KOH","1e308"),("CAP","FF","1e-310"),
                ("IND","PH","1e-313"),("RES","OH","1e-1000"),
                ("CAP","PF","-1"),("IND","NH","Inf"))
            p=_ideal_unit_project(base,kind,literal;unit)
            # A preceding vendor callback and malformed geometry falsify late validation.
            vendor=deepcopy(only(p.components))
            type_tokens=only(filter(r->first(r.tokens)=="TYPE",vendor)).tokens
            empty!(type_tokens);append!(type_tokens,["TYPE","SPROJ","unused.son"])
            pushfirst!(p.components,vendor)
            first(filter(r->first(r.tokens)=="SMDP",vendor)).tokens[3]=".1"
            err=try sonnet_component_model(p,200e6;component_response=callback);nothing catch e;e end
            @test err isa ArgumentError
            @test occursin("IDEAL",sprint(showerror,err)) || occursin("unknown Sonnet variable Inf",sprint(showerror,err))
            @test calls[]==0
        end
        for (kind,unit) in (("RES","INVALID"),("CAP","INVALID"),("IND","INVALID"))
            @test_throws ArgumentError sonnet_component_model(_ideal_unit_project(base,kind,"1";unit),
                200e6;component_response=callback)
            @test calls[]==0
        end
        for f in (BigFloat("1e1000"),BigFloat("1e-1000"),0.,-1.,Inf,NaN)
            @test_throws ArgumentError sonnet_component_model(base,f;component_response=callback)
            @test_throws ArgumentError solve_sonnet_project(base,f;raw=true,component_response=callback)
            @test calls[]==0
        end
        variable=_ideal_unit_project(base,"RES","RVALUE")
        variable.variables["RVALUE"]="16.77"
        for value in (BigFloat("1e1000"),BigFloat("1e-1000"),Inf,NaN,1im)
            @test_throws ArgumentError sonnet_component_model(variable,200e6;
                variables=Dict("RVALUE"=>value),component_response=callback)
            @test calls[]==0
        end
        @test_throws ArgumentError sonnet_component_model(variable,200e6;variables=Dict(1=>1.))
        values=Dict("RVALUE"=>BigFloat("16.77"))
        saved=copy(values)
        @test only(sonnet_component_model(variable,BigFloat("200000000");variables=values).circuit.elements).r==16.77
        @test values==saved
        equivalent=solve_sonnet_project(variable,BigFloat("200000000");variables=values,raw=true,mx=64,my=80)
        @test maximum(abs,equivalent.s-results["ohm"].s)<1e-10
        @test equivalent.circuit.voltages≈results["ohm"].circuit.voltages rtol=1e-10
        @test_throws ArgumentError sonnet_component_model(variable,200e6;variables=values,max_bytes=1,
            component_response=callback)
        @test calls[]==0 && values==saved
        dynamic=_ideal_unit_project(base,"RES","16.77*(1+FREQ/200000000)")
        @test only(sonnet_component_model(dynamic,200e6).circuit.elements).r==33.54
        @test only(sonnet_component_model(dynamic,400e6).circuit.elements).r==50.31
        for (kind,field) in (("RES",:r),("CAP",:c),("IND",:l))
            p=_ideal_unit_project(base,kind,"0")
            @test getproperty(only(sonnet_component_model(p,200e6).circuit.elements),field)==0
        end
    end
end
