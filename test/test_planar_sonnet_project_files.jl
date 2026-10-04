using Test, DiffMoM, SHA, LinearAlgebra, JSON

function _sproj_child(path,body;reference=50.)
    write(path,"""
FTYP SONNETPRJ 19
VER "18.53"
DIM
FREQ MHZ
CAP PF
IND NH
LNG MIL
RES OH
END DIM
CKT
$body
END CKT
VARSWP
ENABLED Y
FREQ Y AN SWEEP 200 400 200
END
END VARSWP
""")
end

@testset "Bounded native linear SPROJ snapshots" begin
    evidence=joinpath(@__DIR__,"fixtures","native_sproj_linear")
    for entry in JSON.parsefile(joinpath(evidence,"sha256.json"))["files"]
        @test bytes2hex(sha256(read(joinpath(evidence,entry["path"]))))==entry["sha256"]
    end
    fixture=joinpath(@__DIR__,"fixtures","native_sparam_model_files","native","sparameter.son")
    mktempdir() do directory
        parent=joinpath(directory,"parent.son");child=joinpath(directory,"child.son")
        original=replace(read(fixture,String),"TYPE SPARAM 1"=>"TYPE SPROJ child.son\nINHSWP N")
        write(parent,original)
        _sproj_child(child,"RES 1 2 R=16.77\nDEF2P 1 2 Device R 50")
        p=read_sonnet_project(parent)
        plan=sonnet_component_files(p,300e6)
        b=only(plan.bindings)
        @test b.file_index==0 && b.inherited_sweep===false
        @test b.geometry_labels==[3,4] && b.pin_indices==[1,2]
        @test length(plan.sources)==2 && length(plan.circuits)==1 && isempty(plan.networks)
        @test b.z0==ComplexF64[50,50]
        @test b.response≈ComplexF64[16.77 100;100 16.77]/116.77 rtol=1e-14
        for source in values(plan.sources)
            @test source.sha256==bytes2hex(sha256(source.bytes))
        end
        native=planar_read_touchstone(joinpath(@__DIR__,"fixtures","native_sproj_linear","native_child.s2p"))
        @test maximum(abs,b.response-planar_network_response(native,300e6))<4e-14
        actual=solve_sonnet_project(plan;raw=true,mx=64,my=80)
        ideal=deepcopy(p)
        filter!(r->first(r.tokens)!="INHSWP",ideal.components[1])
        type=only(filter(r->first(r.tokens)=="TYPE",ideal.components[1])).tokens
        empty!(type);append!(type,["TYPE","IDEAL","RES","16.77"])
        expected=solve_sonnet_project(ideal,300e6;raw=true,mx=64,my=80)
        @test maximum(abs,actual.s-expected.s)<1e-10
        waves=ComplexF64[.3+.2im,-.4im]
        maps=sonnet_component_current_maps(actual;incident_waves=waves)
        oracle=sonnet_component_current_maps(expected;incident_waves=waves)
        @test maps[1].jx≈oracle[1].jx rtol=1e-10
        @test maps[1].jy≈oracle[1].jy rtol=1e-10
        before=copy(b.response);sourcebytes=copy(plan.sources[realpath(child)].bytes)
        _sproj_child(child,"RES 1 2 R=30\nDEF2P 1 2 Device R 50")
        @test b.response==before && plan.sources[realpath(child)].bytes==sourcebytes
        @test solve_sonnet_project(plan;raw=true,mx=64,my=80).s≈actual.s rtol=1e-13
        @test only(sonnet_component_files(p,300e6).bindings).response!=before

        @testset "Continuous reactive child evaluation and reference basis" begin
            _sproj_child(child,"RES 1 2 R=5\nCAP 2 0 C=10\nIND 2 0 L=25\nDEF2P 1 2 Reactive R 75")
            for inherit in (false,true),f in (250e6,300e6,350e6)
                inherited=deepcopy(p)
                only(filter(r->first(r.tokens)=="INHSWP",inherited.components[1])).tokens[2]=inherit ? "Y" : "N"
                staged=sonnet_component_files(inherited,f)
                binding=only(staged.bindings)
                y=ComplexF64[.2 -.2;-.2 .2+im*(2pi*f*10e-12-1/(2pi*f*25e-9))]
                @test binding.response≈planar_y_to_s(y,[75.,75.]) rtol=2e-13
                @test binding.z0==ComplexF64[75,75] && binding.inherited_sweep===inherit
                changed=only(sonnet_component_files(inherited,f;requested_reference=[43+12im,92-21im]).bindings)
                @test planar_s_to_y(changed.response,changed.z0)≈y rtol=2e-12
                low=planar_y_to_s(ComplexF64[.2 -.2;-.2 .2+im*(2pi*200e6*10e-12-1/(2pi*200e6*25e-9))],[75.,75.])
                high=planar_y_to_s(ComplexF64[.2 -.2;-.2 .2+im*(2pi*400e6*10e-12-1/(2pi*400e6*25e-9))],[75.,75.])
                interpolated=low+(f-200e6)/200e6*(high-low)
                @test maximum(abs,binding.response-interpolated)>0.05
            end
        end

        @testset "Archived native PRJ continuous-frequency controls" begin
            worst=0.
            for case in ("coarse_child_flag0","coarse_child_flag1","fine_child_flag0")
                flag=endswith(case,"flag1") ? 1 : 0
                _sproj_child(joinpath(directory,"grandchild.son"),
                    "RES 1 2 R=5\nCAP 2 0 C=10\nIND 2 0 L=25\nDEF2P 1 2 Reactive R 50")
                _sproj_child(child,"PRJ 1 2 grandchild.son 2 $flag\nDEF2P 1 2 Parent R 50")
                source=joinpath(@__DIR__,"fixtures","native_sproj_linear",case,"response_nd.sid")
                rows=0
                for line in eachline(source)
                    values=tryparse.(Float64,split(line))
                    length(values)==9 && all(!isnothing,values) || continue
                    f=values[1]
                    actual=reshape(ComplexF64[complex(values[k],values[k+1]) for k in 2:2:8],2,2)
                    expected=only(sonnet_component_files(p,f).bindings).response
                    error=maximum(abs,actual-expected);worst=max(worst,error)
                    @test error<1e-13
                    rows+=1
                end
                @test rows==3
            end
            println("Native full-precision PRJ control maximum complete-S error: ",worst)
        end

        @testset "Recursive projects, earlier DEF, data snapshots and ordered pins" begin
            grandchild=joinpath(directory,"grandchild.son")
            _sproj_child(grandchild,"RES 1 0 R=25\nRES 2 0 R=100\nDEF2P 1 2 Asymmetric R 75")
            _sproj_child(child,"PRJ 1 2 grandchild.son 2 1 DATE 10/04/2026 03:00:00\nDEF2P 1 2 Local R 50\nLocal 11 7\nDEF2P 11 7 Final R 50")
            staged=sonnet_component_files(p,300e6)
            @test length(staged.sources)==3 && length(staged.circuits)==2
            @test only(staged.bindings).response≈planar_y_to_s(ComplexF64[.04 0;0 .01],[50.,50.]) rtol=2e-13
            @test_throws ArgumentError solve_planar_circuit(staged.circuits[realpath(child)],250e6)
            shared=deepcopy(p);second=deepcopy(p.components[1])
            only(filter(r->first(r.tokens)=="ID",second)).tokens[2]="2"
            push!(shared.components,second)
            @test length(sonnet_component_files(shared,300e6).sources)==3
            @test length(sonnet_component_files(shared,300e6).bindings)==2
            datafile=joinpath(directory,"data.s2p")
            data=PlanarNetworkData([200e6,400e6],[planar_y_to_s(ComplexF64[.04 0;0 .01],[75.,75.]) for _ in 1:2];z0=75.)
            planar_write_touchstone(datafile,data)
            _sproj_child(grandchild,"S2P 1 2 data.s2p\nDEF2P 1 2 Data R 75")
            captured=sonnet_component_files(p,300e6)
            @test length(captured.sources)==4 && length(captured.networks)==1
            @test only(captured.bindings).response≈planar_y_to_s(ComplexF64[.04 0;0 .01],[50.,50.]) rtol=2e-13
            write(datafile,"changed")
            @test solve_planar_circuit(captured.circuits[realpath(child)],300e6;floating_gauge=:auto).s≈only(captured.bindings).response rtol=1e-13
            permutation=deepcopy(p)
            pins=filter(r->first(r.tokens)=="SMDP",permutation.components[1])
            pins[1].tokens[6:7]=["44","2"];pins[2].tokens[6:7]=["23","1"]
            _sproj_child(child,"RES 1 0 R=25\nRES 2 0 R=100\nDEF2P 1 2 Final R 50")
            @test only(sonnet_component_files(permutation,300e6).bindings).geometry_labels==[23,44]
        end

        @testset "Dependency, unit and resource rejection before reference callbacks" begin
            calls=Ref(0);reference=f->(calls[]+=1;50.)
            _sproj_child(child,"RES 1 2 R=5\nCAP 2 0 C=10\nIND 2 0 L=25\nDEF2P 1 2 Device R 50")
            for options in ((max_bytes=1,),(max_bytes=100000,),(max_dependencies=1,),
                    (max_project_nodes=1,),(max_project_elements=2,),(max_project_depth=0,))
                @test_throws ArgumentError sonnet_component_files(p,300e6;requested_reference=reference,options...)
                @test calls[]==0
            end
            for body in ("CAP 1 2 C=1e-320\nDEF2P 1 2 Device R 50",
                    "RES 1 2 R=1e400\nDEF2P 1 2 Device R 50",
                    "RES 1 2 R=FREQ\nDEF2P 1 2 Device R 50",
                    "PRJ 1 2 parent.son 2 0\nDEF2P 1 2 Device R 50",
                    "PRJ 1 2 child.son 2 0\nDEF2P 1 2 Device R 50",
                    "PRJ 1 2 grandchild.son 2 0 VARIABLE x=3\nDEF2P 1 2 Device R 50",
                    "RES 1 2 R=5\nDEF1P 1 Device R 50",
                    "RES 1 2 R=5\nDEF2P 1 01 Device R 50")
                _sproj_child(child,body)
                @test_throws ArgumentError sonnet_component_files(p,300e6;requested_reference=reference)
                @test calls[]==0
            end
            _sproj_child(joinpath(directory,"grandchild.son"),"RES 1 2 R=5\nDEF2P 1 2 Device R 50")
            _sproj_child(child,"PRJ 1 2 grandchild.son 2 0\nDEF2P 1 2 Device R 50")
            @test_throws ArgumentError sonnet_component_files(p,300e6;max_project_depth=1,requested_reference=reference)
            @test calls[]==0
            @test_throws ArgumentError sonnet_component_files(p,300e6;max_dependencies=2,requested_reference=reference)
            @test calls[]==0
            outside=joinpath(dirname(directory),"outside_$(basename(directory)).son")
            try
                _sproj_child(outside,"RES 1 2 R=5\nDEF2P 1 2 Device R 50")
                _sproj_child(child,"PRJ 1 2 \"$outside\" 2 0\nDEF2P 1 2 Device R 50")
                @test_throws ArgumentError sonnet_component_files(p,300e6;requested_reference=reference)
                @test calls[]==0
            finally
                isfile(outside) && rm(outside)
            end
            write(child,read(fixture,String))
            @test_throws ArgumentError sonnet_component_files(p,300e6;requested_reference=reference)
            @test calls[]==0
            _sproj_child(child,"RES 1 2 R=5\nDEF2P 1 2 Device R 50")
            _sproj_child(child,"RES 1 2 R=5\nCAP 2 0 C=2.6e-312\nDEF2P 1 2 Device R 50")
            tiny=sonnet_component_files(p,300e6)
            @test tiny.circuits[realpath(child)].elements[2].c==
                Float64(BigFloat("2.6e-312")*BigFloat(1e-12))==nextfloat(0.)
            _sproj_child(child,"RES 1 2 R=5\nDEF2P 1 2 Device R 50")
            badlater=deepcopy(p);later=deepcopy(first(p.components))
            only(filter(r->first(r.tokens)=="ID",later)).tokens[2]="2"
            only(filter(r->first(r.tokens)=="TYPE",later)).tokens[3]="missing.son"
            push!(badlater.components,later)
            @test_throws ArgumentError sonnet_component_files(badlater,300e6;requested_reference=reference)
            @test calls[]==0
            for invalid in (NaN,Inf,0.,BigFloat("1e-1000"),BigFloat("1e1000"))
                @test_throws ArgumentError sonnet_component_files(p,invalid;requested_reference=reference)
                @test calls[]==0
            end
            for bad in ("INHSWP N"=>"INHSWP INVALID","TYPE SPROJ child.son"=>"TYPE SPROJ child.son VARIABLE x=3")
                write(parent,replace(original,bad))
                @test_throws ArgumentError sonnet_component_files(read_sonnet_project(parent),300e6;requested_reference=reference)
                @test calls[]==0
            end
            write(parent,original)
            good=sonnet_component_files(p,300e6)
            c=PlanarCircuit(4,[1,2]);old=copy(c.elements)
            @test_throws ArgumentError circuit_add_sonnet_files!(c,good,[1,2,3,3])
            @test c.elements==old
            circuit_add_sonnet_files!(c,good,[1,2,3,4])
            @test only(c.elements).terminals==[(3,0),(4,0)]
        end
    end
end
