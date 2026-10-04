using Test, DiffMoM, LinearAlgebra, SHA, TOML, JSON, CRC32c

let repo=normpath(joinpath(@__DIR__,"..")),
    fixture=joinpath(@__DIR__,"fixtures","native_prj_common_return")
    D=DiffMoM
    manifest=JSON.parsefile(joinpath(fixture,"sha256.json"))
    @testset "Native common-return proof integrity" begin
        for entry in manifest["files"]
            bytes=read(joinpath(fixture,entry["path"]))
            @test length(bytes)==entry["bytes"]
            @test bytes2hex(sha256(bytes))==entry["sha256"]
            @test string(CRC32c.crc32c(bytes);base=16,pad=8)==entry["crc32c"]
        end
    end
    evidence=joinpath(fixture,"native","native_prj_common_return_77wGmW")
    native=TOML.parsefile(joinpath(evidence,"comparison.toml"))
    report=Dict{String,Any}()
    child_y(f)=ComplexF64[.2 -.2;-.2 .2+im*(2pi*f*10e-12-1/(2pi*f*25e-9))]
    child_s(f)=planar_y_to_s(child_y(f),[50.,50.])
@testset "Native common-return public path: actual native complete S and nodes" begin
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

end
