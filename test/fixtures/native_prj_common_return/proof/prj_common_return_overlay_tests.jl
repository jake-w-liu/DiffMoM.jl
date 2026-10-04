using DiffMoM, LinearAlgebra, Test, TOML, SHA
const D=DiffMoM
const repo=normpath(joinpath(@__DIR__,"..",".."))
const evidence=last(sort(filter(path->isfile(joinpath(path,"comparison.toml")),
    filter(isdir,readdir(joinpath(repo,"data","sonnet_validation");join=true)));
    by=path->occursin("native_prj_common_return_",basename(path)) ? stat(path).mtime : -Inf))
occursin("native_prj_common_return_",basename(evidence)) || error("native evidence missing")
const native=TOML.parsefile(joinpath(evidence,"comparison.toml"))
const report=Dict{String,Any}("native_evidence"=>evidence,
    "source_before"=>Dict(n=>bytes2hex(sha256(read(joinpath(repo,"src","planar",n))))
        for n in ("PlanarSonnetIO.jl","PlanarSonnetProjectFiles.jl","PlanarCircuit.jl")))
# The coordinated main-source integration may proceed while this isolated
# proof runs. Restore only captured original methods in this Julia process
# so the before assertions always describe the preserved source bytes.
baseline=joinpath(@__DIR__,"prj_common_return_overlay")
oldio=read(joinpath(baseline,"PlanarSonnetIO_source_before.jl"),String)
a=findfirst("function sonnet_planar_circuit(",oldio).start
b=findnext("\nfunction _sonnet_termination",oldio,a).start
Base.include_string(D,oldio[a:prevind(oldio,b)],"captured_PRJ_IO_before")
oldprojects=read(joinpath(baseline,"PlanarSonnetProjectFiles_source_before.jl"),String)
a=findfirst("_sonnet_project_path_key",oldprojects).start
Base.include_string(D,oldprojects[a:end],"captured_PRJ_ProjectFiles_before")
report["captured_runtime_baseline"]=Dict(n=>bytes2hex(sha256(read(joinpath(baseline,n*"_source_before.jl"))))
    for n in ("PlanarSonnetIO","PlanarSonnetProjectFiles"))
child_y(f)=ComplexF64[.2 -.2;-.2 .2+im*(2pi*f*10e-12-1/(2pi*f*25e-9))]
child_s(f)=planar_y_to_s(child_y(f),[50.,50.])

@testset "Preserved public before failure: valid common-return PRJ" begin
    for case in native["cases"]
        case["case"]=="implicit_ground" && continue
        p=read_sonnet_project(joinpath(evidence,case["case"]*".son"))
        calls=Ref(0)
        @test_throws ArgumentError sonnet_planar_circuit(p;project_response=(path,f)->(calls[]+=1;child_s(f)))
        @test calls[]==0
    end
    mktempdir() do directory
        fixture=joinpath(repo,"test","fixtures","native_sparam_model_files","native","sparameter.son")
        parent=joinpath(directory,"parent.son")
        write(parent,replace(read(fixture,String),"TYPE SPARAM 1"=>"TYPE SPROJ device.son\nINHSWP N"))
        write(joinpath(directory,"device.son"),replace(read(joinpath(evidence,"common_return_resistor.son"),String),"child.son"=>"primitive.son"))
        cp(joinpath(evidence,"child.son"),joinpath(directory,"primitive.son"))
        error=try
            sonnet_component_files(read_sonnet_project(parent),300e6)
            nothing
        catch e
            e
        end
        @test error isa ArgumentError
        @test occursin("invalid linear PRJ port count",sprint(showerror,error))
    end
end

include("prj_common_return_overlay/install_overlay.jl")
@testset "Isolated common-return overlay: actual native complete S and nodes" begin
    worst=0.
    for case in native["cases"]
        p=read_sonnet_project(joinpath(evidence,case["case"]*".son"))
        n=case["ports"];calls=Ref(0)
        circuit=sonnet_planar_circuit(p;project_response=(path,f)->(calls[]+=1;child_s(f)))
        data=planar_read_touchstone(joinpath(evidence,case["case"],"native_parent.s$(n)p"))
        @test calls[]==0
        @test_throws ArgumentError solve_planar_circuit(circuit,300e6;max_bytes=1)
        @test calls[]==0
        for (f,s) in zip(data.frequencies,data.s)
            result=solve_planar_circuit(circuit,f;floating_gauge=:auto)
            err=maximum(abs,result.s-s);worst=max(worst,err)
            @test err<1e-10
            @test norm(result.s-transpose(result.s))<1e-12
            @test opnorm(result.s)<=1+1e-12
            @test solve_sonnet_project(p,f;project_response=(path,f)->child_s(f),floating_gauge=:auto).s==result.s
            # Literal current return: sparse native node7 compacts to the last
            # node without becoming a third model pin or a ground clamp.
            if case["case"]=="common_return_resistor"
                waves=ComplexF64[.4+.2im,-.3im]
                v=result.voltages*waves
                current=child_y(f)*(v[1:2].-v[3])
                @test sum(current)≈v[3]/13 rtol=1e-12
            elseif occursin("external",case["case"])
                @test result.s*ones(n)≈ones(n) atol=1e-12
            end
        end
    end
    report["full_s_native_max_error"]=worst
end

@testset "Malformed return records reject before model response" begin
    p=read_sonnet_project(joinpath(evidence,"common_return_resistor.son"))
    for tokens in (["PRJ","1","2","7","9","child.son","2","1"],
                   ["PRJ","1","2","-7","child.son","2","1"],
                   ["PRJ","1","2","1","child.son","2","1"],
                   ["PRJ","1","2","7","child.son","0","1"])
        rows=deepcopy(p.circuit);rows[1]=SonnetRecord(rows[1].line,tokens)
        bad=SonnetNetlistProject(p.source,p.units,p.frequency_scale,rows,p.records)
        calls=Ref(0)
        @test_throws ArgumentError sonnet_planar_circuit(bad;project_response=(path,f)->(calls[]+=1;child_s(f)))
        @test calls[]==0
    end
end

@testset "Physical calibrated geometry common-return voltage/current projection" begin
    mktempdir() do directory
        geometry=joinpath(directory,"geometry.son")
        cp(joinpath(repo,"validation","sonnet_stripline","s50.son"),geometry)
        g=read_sonnet_project(geometry);f=2e9
        chains=[planar_line_abcd(50.,.003im),ComplexF64[1 2+.3im;0 1]]
        child=solve_sonnet_project(g,f;grid=(8,8),calibration=chains,mx=24,my=24)
        parent=read_sonnet_project(joinpath(evidence,"common_return_resistor.son"))
        rows=deepcopy(parent.circuit);q=copy(rows[1].tokens);q[5]="geometry.son";rows[1]=SonnetRecord(rows[1].line,q)
        p=SonnetNetlistProject(joinpath(directory,"parent.son"),parent.units,parent.frequency_scale,rows,parent.records)
        circuit=sonnet_planar_circuit(p;project_response=(path,f)->child.s)
        result=solve_planar_circuit(circuit,f;floating_gauge=:auto)
        direct=PlanarCircuit(3,[1,2]);circuit_add_network!(direct,[(1,3),(2,3)],child.s;z0=child.z0)
        circuit_add_rlc!(direct,3,0;r=13.)
        hand=solve_planar_circuit(direct,f;floating_gauge=:auto)
        @test result.s==hand.s
        waves=ComplexF64[.3+.2im,-.4im];v=result.voltages*waves
        incidence=Float64[1 0;0 1;-1 -1]
        yn=incidence*child.y*transpose(incidence);yn[3,3]+=1/13
        nodal=copy(yn);nodal[1,1]+=1/50;nodal[2,2]+=1/50
        independent=nodal\vcat(2waves/sqrt(50.),0im)
        @test v≈independent rtol=1e-11
        terminal=transpose(incidence)*v
        @test terminal≈v[1:2].-v[3]
        maps=planar_current_maps(child;voltages=terminal)
        rawmaps=planar_current_maps(child.raw;voltages=child.voltage_transfer*terminal)
        @test maps[1].jx==rawmaps[1].jx
        @test maps[1].jy==rawmaps[1].jy
        @test child.y*terminal≈2waves/sqrt(50.).-v[1:2]/50 rtol=1e-11
        @test sum(child.y*terminal)≈v[3]/13 rtol=1e-11
        @test child.y*(transpose(incidence)*(v.+(2+.7im)))≈child.y*terminal rtol=1e-11
        raw=child.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            q=raw.problem.basis.port[b]
            q>0 && (rhs[b,q]=-D._planar_port_sign(raw.problem.ports[q])*D._planar_port_weight(raw.problem.basis,b))
        end
        @test norm(raw.z_mom*raw.currents-rhs)/norm(rhs)<1e-10
    end
end

@testset "Automatic bounded recursive SPROJ common return and loaded physical nodes" begin
    fixture=joinpath(repo,"test","fixtures","native_sparam_model_files","native","sparameter.son")
    mktempdir() do directory
        parent=joinpath(directory,"parent.son");device=joinpath(directory,"device.son");primitive=joinpath(directory,"primitive.son")
        write(parent,replace(read(fixture,String),"TYPE SPARAM 1"=>"TYPE SPROJ device.son\nINHSWP N"))
        write(device,replace(read(joinpath(evidence,"common_return_resistor.son"),String),"child.son"=>"primitive.son"))
        cp(joinpath(evidence,"child.son"),primitive)
        p=read_sonnet_project(parent);f=300e6
        files=sonnet_component_files(p,f;grid=(32,40))
        binding=only(files.bindings)
        actualchild=solve_sonnet_project(read_sonnet_project(device),f;project_response=(path,f)->child_s(f),floating_gauge=:auto)
        @test binding.response≈actualchild.s rtol=1e-12
        @test length(files.circuits)==2 && length(files.sources)==3
        @test all(source->source.sha256==bytes2hex(sha256(source.bytes)),values(files.sources))
        loaded=solve_sonnet_project(files;raw=true,mx=32,my=40)
        direct=solve_sonnet_project(p,f;grid=(32,40),raw=true,mx=32,my=40,
            component_response=(p,records,f)->(;response=actualchild.s,format=:s,z0=actualchild.z0))
        @test maximum(abs,loaded.s-direct.s)<1e-10
        waves=ComplexF64[.2+.3im,-.4im]
        maps=sonnet_component_current_maps(loaded;incident_waves=waves)
        manual=sonnet_component_current_maps(direct;incident_waves=waves)
        @test maps[1].jx≈manual[1].jx rtol=1e-10
        @test maps[1].jy≈manual[1].jy rtol=1e-10
        before=copy(binding.response);bytes=copy(files.sources[realpath(primitive)].bytes)
        write(primitive,replace(read(primitive,String),"R=5"=>"R=9"))
        @test binding.response==before && files.sources[realpath(primitive)].bytes==bytes
        @test solve_sonnet_project(files;raw=true,mx=32,my=40).s==loaded.s
        @test only(sonnet_component_files(p,f;grid=(32,40)).bindings).response!=before
        calls=Ref(0);reference=f->(calls[]+=1;50.)
        @test_throws ArgumentError sonnet_component_files(p,f;max_bytes=1000,requested_reference=reference)
        @test calls[]==0
        attached=PlanarCircuit(4,[1,2];z0=reference)
        original_elements=attached.elements;original_refs=attached.z0
        @test_throws ArgumentError circuit_add_sonnet_files!(attached,files,[1,2,3,3])
        @test attached.elements===original_elements && attached.z0===original_refs && calls[]==0
        @test_throws ArgumentError circuit_add_sonnet_files!(attached,files,[1,2,3,4];max_bytes=1)
        @test attached.elements===original_elements && attached.z0===original_refs && calls[]==0
        circuit_add_sonnet_files!(attached,files,[1,2,3,4])
        @test length(attached.elements)==1 && attached.z0===original_refs && calls[]==0
        @test only(attached.elements).terminals==[(3,0),(4,0)]
        corrupt=replace(read(device,String),"PRJ 1 2 7"=>"PRJ 1 2 7 9")
        write(device,corrupt)
        @test_throws ArgumentError sonnet_component_files(p,f;requested_reference=reference)
        @test calls[]==0
    end
end

report["source_after"]=Dict(n=>bytes2hex(sha256(read(joinpath(repo,"src","planar",n)))) for n in keys(report["source_before"]))
open(joinpath(@__DIR__,"prj_common_return_overlay_tests.toml"),"w") do io;TOML.print(io,report);end
