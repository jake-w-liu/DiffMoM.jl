using Test, DiffMoM, SHA, LinearAlgebra, Tar, JSON

_sparam_file_hashes(plan)=Dict(k=>v.sha256 for (k,v) in plan.sources)
_sparam_file_bytes(plan)=Dict(k=>v.bytes for (k,v) in plan.sources)

@testset "Native SPARAM physical adapter" begin
    fixture=joinpath(@__DIR__,"fixtures","native_sparam_model_files")
    manifest=JSON.parsefile(joinpath(fixture,"manifest.json"))
    for entry in manifest["files"]
        @test bytes2hex(sha256(read(joinpath(fixture,entry["path"]))))==entry["sha256"]
    end
    @test occursin("does not allow data file components",read(joinpath(fixture,"native","sparameter_engine_stderr.log"),String))
    mktempdir() do proof
        original=joinpath(fixture,"native")
        corpus=joinpath(fixture,"installed")
coupon=joinpath(proof,"coupon.son")
cp(joinpath(original,"sparameter.son"),coupon;force=true)
modelpath=joinpath(proof,"device_model.s2p")
frequencies=[200e6,300e6,400e6]
conductance=ComplexF64[.015 -.005;-.005 .025]
capacitance=ComplexF64[.4 -.1;-.1 .6]*1e-12
refs=[ComplexF64[50+20im,75-10im],ComplexF64[55+25im,70-5im],ComplexF64[60+30im,65]]
Y=[conductance+2pi*im*f*capacitance for f in frequencies]
data=PlanarNetworkData(frequencies,[planar_y_to_s(y,z) for (y,z) in zip(Y,refs)];reference_series=refs)
# Retain dynamic complex references as literal Sonnet FTERM extension.
open(modelpath,"w") do io
    println(io,"# HZ S RI R 50")
    println(io,"! FTERM 50 20 0 0 75 -10 0 0")
    # This source below deliberately uses fixed complex FTERM references.
    for (f,y) in zip(frequencies,Y)
        s=planar_y_to_s(y,refs[1]);print(io,f)
        for x in (s[1,1],s[2,1],s[1,2],s[2,2]);print(io," ",real(x)," ",imag(x));end
        println(io)
    end
end
project=read_sonnet_project(coupon)

function _sparam_handwired(p,f,source;swapped=false)
    geometry=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,p.metals,
        p.top,p.bottom,p.polygons,p.ports,p.variables,Vector{SonnetRecord}[],p.sweeps,p.records)
    base=sonnet_planar_problem(geometry;freq=f,_materials=true,_details=true)
    terminal=[PlanarPort(1,:terminal_x,7,15:16,50.;metal_side=:negative),
        PlanarPort(1,:terminal_x,9,15:16,50.;metal_side=:positive)]
    physical=planar_terminal_returns(base.problem,terminal)
    raw=solve_planar(physical.problem,f;mx=64,my=80,surface_zs=base.sheet_zs)
    y=transpose(physical.contraction)*raw.y*physical.contraction
    c=PlanarCircuit(4,[1,2];z0=base.z0)
    z=DiffMoM._network_reference_at(source,f)
    circuit_add_network!(c,swapped ? [4,3] : [3,4],planar_network_response(source,f);z0=z)
    circuit_add_network!(c,[1,2,3,4],y;format=:y)
    result=solve_planar_circuit(c,f)
    (;result,raw,physical,c)
end

worst=Ref(0.)
@testset "Observed automatic SPARAM table, snapshots, references and interpolation" begin
    source=planar_read_touchstone(modelpath)
    for f in (200e6,250e6,300e6,350e6,400e6)
        plan=sonnet_component_files(project,f)
        @test length(plan.bindings)==1 && length(_sparam_file_hashes(plan))==2
        binding=only(plan.bindings)
        @test binding.geometry_labels==[3,4] && binding.pin_indices==[1,2]
        @test binding.z0==ComplexF64[50+20im,75-10im]
        @test _sparam_file_hashes(plan)[realpath(modelpath)]==bytes2hex(sha256(_sparam_file_bytes(plan)[realpath(modelpath)]))
        expected=planar_network_response(source,f)
        @test binding.response==expected
        refplan=sonnet_component_files(project,f;requested_reference=[43+12im,92-21im])
        @test planar_s_to_y(binding.response,binding.z0)≈planar_s_to_y(only(refplan.bindings).response,only(refplan.bindings).z0) rtol=1e-12
        automatic=solve_sonnet_project(plan;raw=true,mx=64,my=80)
        manual=_sparam_handwired(project,f,source)
        error=maximum(abs,automatic.s-manual.result.s);worst[]=max(worst[],error)
        @test error<1e-10
        @test opnorm(automatic.s)<=1+1e-10
        @test automatic.s≈transpose(automatic.s) rtol=1e-10
        waves=ComplexF64[.2+.1im,-.4im]
        actual=sonnet_component_current_maps(automatic;incident_waves=waves)
        expectedmaps=planar_current_maps(manual.raw;voltages=manual.physical.contraction*(manual.result.voltages*waves))
        @test actual[1].jx≈expectedmaps[1].jx rtol=1e-10
        @test actual[1].jy≈expectedmaps[1].jy rtol=1e-10
        # AUTO retains the measured global-reference shunt conductance.
        @test norm(planar_s_to_y(binding.response,binding.z0)*ones(ComplexF64,2))>0.02
    end
    plan=sonnet_component_files(project,250e6)
    left,right=filter(r->first(r.tokens)=="SMDP",project.components[1])
    permuted=deepcopy(project)
    pins=filter(r->first(r.tokens)=="SMDP",permuted.components[1])
    pins[1].tokens[6:7]=["44","2"];pins[2].tokens[6:7]=["23","1"]
    permutedplan=sonnet_component_files(permuted,250e6)
    @test only(permutedplan.bindings).geometry_labels==[23,44]
    actual=solve_sonnet_project(permutedplan;raw=true,mx=64,my=80)
    manual=_sparam_handwired(permuted,250e6,source;swapped=true)
    @test maximum(abs,actual.s-manual.result.s)<1e-10
    @test maximum(abs,actual.s-solve_sonnet_project(plan;raw=true,mx=64,my=80).s)>1e-3
    maps=sonnet_component_current_maps(actual;incident_waves=[.4im,.7])
    independent=planar_current_maps(manual.raw;voltages=manual.physical.contraction*(manual.result.voltages*ComplexF64[.4im,.7]))
    @test maps[1].jx≈independent[1].jx rtol=1e-10
    @test norm([maps[1].jx;maps[1].jy]-[independent[1].jx;independent[1].jy])/
        norm([independent[1].jx;independent[1].jy])<1e-10
end

@testset "Preflight, immutable source snapshots and transactional physical attachment" begin
    plan=sonnet_component_files(project,300e6)
    c=PlanarCircuit(4,[1,2]);circuit_add_rlc!(c,1,2;r=100.)
    old=copy(c.elements)
    @test_throws ArgumentError circuit_add_sonnet_files!(c,plan,[1,2,3,4];max_bytes=1)
    @test c.elements==old && c.nnodes==4
    @test_throws ArgumentError circuit_add_sonnet_files!(c,plan,[1,2,3,3])
    @test c.elements==old
    circuit_add_sonnet_files!(c,plan,[1,2,3,4])
    @test length(c.elements)==2 && c.elements[2].terminals==[(3,0),(4,0)]
    calls=Ref(0);provider=f->(calls[]+=1;50+3im)
    for f in (-1.,0.,NaN,Inf,BigFloat("1e1000"),BigFloat("1e-1000"))
        @test_throws ArgumentError sonnet_component_files(project,f;requested_reference=provider)
        @test calls[]==0
    end
    for options in ((max_bytes=1,),(max_bytes=20000,),(max_dependencies=1,))
        @test_throws ArgumentError sonnet_component_files(project,300e6;requested_reference=provider,options...)
        @test calls[]==0
    end
    for f in (prevfloat(200e6),nextfloat(400e6))
        @test_throws ArgumentError sonnet_component_files(project,f;requested_reference=provider)
        @test calls[]==0
    end
    sonnet_component_files(project,300e6;requested_reference=provider)
    @test calls[]>0
    # Archive bundle is portable, and fresh staging detects changed dependencies.
    archive=joinpath(proof,"portable_archive");mkpath(archive)
    cp(coupon,joinpath(archive,"coupon.son");force=true);cp(modelpath,joinpath(archive,"device_model.s2p");force=true)
    saved=read_sonnet_project(joinpath(archive,"coupon.son"));firstplan=sonnet_component_files(saved,300e6)
    @test only(firstplan.bindings).response==only(plan.bindings).response
    altered=read(joinpath(archive,"device_model.s2p"),String)*"! dependency edit\n"
    write(joinpath(archive,"device_model.s2p"),altered)
    secondplan=sonnet_component_files(saved,300e6)
    @test _sparam_file_hashes(firstplan)!=_sparam_file_hashes(secondplan)
    @test _sparam_file_bytes(firstplan)!=_sparam_file_bytes(secondplan)
    @test only(firstplan.bindings).response==only(secondplan.bindings).response
    # The original plan's bytes continue to describe exactly what it evaluated.
    @test _sparam_file_hashes(firstplan)[realpath(joinpath(archive,"device_model.s2p"))]==
        bytes2hex(sha256(_sparam_file_bytes(firstplan)[realpath(joinpath(archive,"device_model.s2p"))]))
    planar_write_touchstone(joinpath(archive,"device_model.s2p"),
        PlanarNetworkData(frequencies,[planar_y_to_s(10y,[50.,50.]) for y in Y]);version="1.0")
    thirdplan=sonnet_component_files(saved,300e6)
    @test norm(only(thirdplan.bindings).response-only(firstplan.bindings).response)>.1
    @test only(firstplan.bindings).response==only(plan.bindings).response
    for (recordkind,tokens,message) in (("TERMW",["TERMW","CUST","40"],"TERMW CUST"),
            ("TERMW",["TERMW","CELL"],"width adapter"),("GNDREF",["GNDREF","FLOAT"],"AUTO ground"),
            ("GNDREF",["GNDREF","POLYGON","AUTO"],"AUTO ground"),
            ("TYPE",["TYPE","SPARAM","999"],"unknown SMDFILES"))
        bad=deepcopy(project);record=first(filter(r->first(r.tokens)==recordkind,bad.components[1]))
        empty!(record.tokens);append!(record.tokens,tokens)
        count=Ref(0);ref=f->(count[]+=1;50.)
        e=try sonnet_component_files(bad,300e6;requested_reference=ref);nothing catch error;error end
        @test e isa ArgumentError && occursin(message,sprint(showerror,e))
        @test count[]==0
    end
    for tag in ("DRP1","DRP2","UNKNOWN")
        bad=deepcopy(project);push!(bad.components[1],SonnetRecord(500,[tag,"LEFT","FIX","50"]))
        @test_throws ArgumentError sonnet_component_files(bad,300e6)
    end
    bad=deepcopy(project);pin=first(filter(r->first(r.tokens)=="SMDP",bad.components[1]));pin.tokens[7]="2"
    @test_throws ArgumentError sonnet_component_files(bad,300e6)
    bad=deepcopy(project);bad.top[4]="1"
    @test_throws ArgumentError sonnet_component_files(bad,300e6)
    bad=deepcopy(project);slot=findfirst(r->r.tokens==["END","SMDFILES"],bad.records)
    insert!(bad.records,slot,SonnetRecord(700,["1","device_model.s2p"]))
    @test_throws ArgumentError sonnet_component_files(bad,300e6)
    bad=deepcopy(project);entry=only(DiffMoM._sonnet_section(bad.records,"SMDFILES",bad.source));entry.tokens[2]="missing.s2p"
    @test_throws ArgumentError sonnet_component_files(bad,300e6)
    @test_throws ArgumentError sonnet_component_files(project,300e6;root=archive)
    # Caller-extracted archive workflow retains the exact relative dependency
    # graph. No native compressed-model-file grammar is inferred.
    tarpath=joinpath(proof,"portable_bundle.tar")
    isfile(tarpath) && rm(tarpath)
    Tar.create(archive,tarpath)
    extraction=joinpath(proof,"extracted_archive");mkpath(extraction)
    Tar.extract(tarpath,extraction)
    extracted=sonnet_component_files(read_sonnet_project(joinpath(extraction,"coupon.son")),300e6;root=extraction)
    @test only(extracted.bindings).response==only(thirdplan.bindings).response
    @test sort!(collect(values(_sparam_file_hashes(extracted))))==sort!(collect(values(_sparam_file_hashes(thirdplan))))
    escaped=deepcopy(project)
    only(DiffMoM._sonnet_section(escaped.records,"SMDFILES",escaped.source)).tokens[2]=
        relpath(joinpath(original,"device_model.s2p"),proof)
    @test_throws ArgumentError sonnet_component_files(escaped,300e6)
    absolute=deepcopy(project)
    only(DiffMoM._sonnet_section(absolute.records,"SMDFILES",absolute.source)).tokens[2]=modelpath
    @test only(sonnet_component_files(absolute,300e6).bindings).response==only(plan.bindings).response
    invalid=joinpath(proof,"invalid_device.s2p");write(invalid,"# HZ S RI R50\n300000000 NaN 0 0 0 0 0 0 0\n")
    broken=deepcopy(project);only(DiffMoM._sonnet_section(broken.records,"SMDFILES",broken.source)).tokens[2]=basename(invalid)
    badcalls=Ref(0)
    @test_throws Exception sonnet_component_files(broken,300e6;requested_reference=f->(badcalls[]+=1;50.))
    @test badcalls[]==0
end

@testset "Dynamic file references and explicit post-interpolation output basis" begin
    dynamic=joinpath(proof,"dynamic_device.s2p")
    references=[PlanarPortImpedance(r=50,x=20,l=3e-9,c=.5e-12,topology=:parallel),
        PlanarPortImpedance(r=75,x=-10,l=2e-9,c=.3e-12,topology=:parallel)]
    evaluated=[[z(f) for z in references] for f in frequencies]
    scattering=[planar_y_to_s(y,z) for (y,z) in zip(Y,evaluated)]
    open(dynamic,"w") do io
        println(io,"# HZ S RI R 50\n! FTERM 50 20 3e-9 5e-13 75 -10 2e-9 3e-13")
        for (f,s) in zip(frequencies,scattering)
            print(io,f)
            for x in (s[1,1],s[2,1],s[1,2],s[2,2]);print(io," ",real(x)," ",imag(x));end
            println(io)
        end
    end
    p=deepcopy(project);entry=only(DiffMoM._sonnet_section(p.records,"SMDFILES",p.source));entry.tokens[2]="dynamic_device.s2p"
    for (f,k,t) in ((200e6,1,0.),(250e6,1,.5),(300e6,2,0.),(350e6,2,.5))
        plan=sonnet_component_files(p,f);binding=only(plan.bindings)
        expected_refs=(1-t)*evaluated[k]+t*evaluated[k+1]
        expected_s=(1-t)*planar_renormalize_s(scattering[k],evaluated[k],expected_refs)+
            t*planar_renormalize_s(scattering[k+1],evaluated[k+1],expected_refs)
        @test binding.z0≈expected_refs rtol=1e-14
        @test binding.response≈expected_s rtol=1e-13
        output=sonnet_component_files(p,f;requested_reference=[50.,50.])
        @test planar_s_to_y(only(output.bindings).response,[50.,50.])≈planar_s_to_y(binding.response,binding.z0) rtol=1e-12
    end
    # S/Y/Z file classes map through the same observed SMDFILES declaration.
    for parameter in (:y,:z)
        path=joinpath(proof,"literal_$(parameter).ts")
        open(path,"w") do io
            println(io,"[Version] 2.0\n# HZ $(uppercase(string(parameter))) RI R 50\n[Number of Ports] 2\n[Number of Frequencies] 3\n[Reference] 50 50\n[Two-Port Data Order] 21_12\n[Network Data]")
            for (f,y) in zip(frequencies,Y)
                matrix=parameter===:y ? y : inv(y);print(io,f)
                for x in (matrix[1,1],matrix[2,1],matrix[1,2],matrix[2,2]);print(io," ",real(x)," ",imag(x));end
                println(io)
            end
            println(io,"[End]")
        end
        p=deepcopy(project);only(DiffMoM._sonnet_section(p.records,"SMDFILES",p.source)).tokens[2]=basename(path)
        response=only(sonnet_component_files(p,300e6).bindings)
        @test planar_s_to_y(response.response,response.z0)≈Y[2] rtol=1e-12
    end
end

@testset "Both installed SPARAM schemas are independently classified" begin
    amp=read_sonnet_project(joinpath(corpus,"amp","amp.son"))
    plan=sonnet_component_files(amp,16e9)
    @test length(plan.bindings)==1 && only(plan.bindings).geometry_labels==[3,4]
    @test length(only(plan.bindings).pin_indices)==2
    @test only(plan.bindings).source==realpath(joinpath(corpus,"amp","amp_dev.s2p"))
    @test all(isfinite,only(plan.bindings).response)
    # Reading/model ordering succeeds independently of EM/native rendering acceptance.
    precise=read_sonnet_project(joinpath(corpus,"7GHz_amp","7GHz_amp.son"))
    count=Ref(0);provider=f->(count[]+=1;50.)
    e=try sonnet_component_files(precise,7e9;requested_reference=provider);nothing catch error;error end
    @test e isa ArgumentError
    @test occursin("TERMW CUST 40",sprint(showerror,e))
    @test count[]==0
end

@testset "Owned configuration, callable circuit references and all-model preflight" begin
    files=sonnet_component_files(project,300e6)
    frompath=sonnet_component_files(coupon,300e6)
    @test only(frompath.bindings).response==only(files.bindings).response
    @test frompath.project !== project && files.project !== project
    @test frompath.configuration_sha256==files.configuration_sha256
    # macOS /var aliases, Unix symlinks and Windows filename casing must
    # identify the same effective configuration and retained dependency.
    alias=if Sys.iswindows()
        joinpath(dirname(coupon),uppercase(basename(coupon)))
    else
        path=joinpath(proof,"coupon_alias.son")
        symlink(coupon,path)
        path
    end
    aliased=sonnet_component_files(alias,300e6)
    @test aliased.project.source==realpath(coupon)==files.project.source
    @test aliased.configuration_sha256==files.configuration_sha256
    @test _sparam_file_hashes(aliased)==_sparam_file_hashes(files)
    before=copy(files.project.box)
    edited=deepcopy(project);staged=sonnet_component_files(edited,300e6)
    edited.box[2]="800"
    @test staged.project.box==before
    edited_config=deepcopy(project)
    config_pins=filter(r->first(r.tokens)=="SMDP",edited_config.components[1])
    config_pins[1].tokens[6:7]=["44","2"];config_pins[2].tokens[6:7]=["23","1"]
    edited_files=sonnet_component_files(edited_config,300e6)
    @test edited_files.configuration_sha256!=files.configuration_sha256
    @test edited_files.sources[realpath(coupon)].sha256==files.sources[realpath(coupon)].sha256
    @test files.sources[realpath(coupon)].bytes==read(coupon)
    @test_throws ArgumentError solve_sonnet_project(files,301e6;raw=true)
    @test_throws ArgumentError solve_sonnet_project(files;raw=true,grid=(16,20))
    @test_throws ArgumentError sonnet_component_files(project,300e6;max_bytes=true)
    @test_throws ArgumentError sonnet_component_files(project,300e6;max_dependencies=true)
    @test_throws ArgumentError sonnet_component_files(project,300e6;grid=(typemax(Int),typemax(Int)))
    @test_throws ArgumentError sonnet_component_files(project,300e6;variables=Dict("X"=>BigFloat("1e-1000")))
    @test_throws ArgumentError sonnet_component_files(project,300e6;variables=Dict("X"=>BigFloat("1e1000")))
    provider_calls=Ref(0);provider=f->(provider_calls[]+=1;50+2im)
    c=PlanarCircuit(4,[1,2];z0=provider)
    circuit_add_rlc!(c,1,2;r=100.)
    original=copy(c.elements)
    @test_throws ArgumentError circuit_add_sonnet_files!(c,files,[1,2,3,3])
    @test c.elements==original && c.z0===provider && provider_calls[]==0
    @test_throws ArgumentError circuit_add_sonnet_files!(c,files,[1,2,3,4];max_bytes=1)
    @test c.elements==original && c.z0===provider && provider_calls[]==0
    circuit_add_sonnet_files!(c,files,[1,2,3,4])
    @test length(c.elements)==2 && c.z0===provider && provider_calls[]==0
    result=solve_planar_circuit(c,300e6)
    @test all(isfinite,result.s) && provider_calls[]>0

    callback_calls=Ref(0);callback=f->(callback_calls[]+=1;50.)
    for mutation in (:misaligned,:missing_width,:missing_first_width)
        bad=deepcopy(project)
        if mutation===:misaligned
            first(filter(r->first(r.tokens)=="SMDP",bad.components[1])).tokens[3]="88"
        else
            widths=findall(r->first(r.tokens)=="TERMW",bad.components[1])
            deleteat!(bad.components[1],mutation===:missing_width ? last(widths) : first(widths))
        end
        @test_throws ArgumentError sonnet_component_files(bad,300e6;requested_reference=callback)
        @test callback_calls[]==0
    end
    # A later component's coverage error must precede every reference callback.
    two=deepcopy(project);second=deepcopy(only(two.components))
    only(filter(r->first(r.tokens)=="ID",second)).tokens[2]="101"
    only(filter(r->first(r.tokens)=="TYPE",second)).tokens[3]="2"
    push!(two.components,second)
    endpoint=joinpath(proof,"late_only.s2p")
    planar_write_touchstone(endpoint,PlanarNetworkData([400e6,500e6],
        [planar_y_to_s(Y[1],[50.,50.]),planar_y_to_s(Y[2],[50.,50.])]);version="1.0")
    position=findfirst(r->r.tokens==["END","SMDFILES"],two.records)
    insert!(two.records,position,SonnetRecord(900,["2",basename(endpoint)]))
    @test_throws ArgumentError sonnet_component_files(two,300e6;requested_reference=callback)
    @test callback_calls[]==0
    only(filter(r->first(r.tokens)=="TYPE",second)).tokens[3]="1"
    shared=sonnet_component_files(two,300e6)
    @test length(shared.bindings)==2 && length(shared.networks)==1 && length(shared.sources)==2
    rejected=deepcopy(two)
    type=only(filter(r->first(r.tokens)=="TYPE",rejected.components[2]))
    empty!(type.tokens);append!(type.tokens,["TYPE","SPROJ","child.son"])
    @test_throws ArgumentError sonnet_component_files(rejected,300e6;requested_reference=callback)
    @test callback_calls[]==0
    caller=deepcopy(project)
    changed=sonnet_component_files(caller,300e6;requested_reference=f->begin
        caller.box[2]="800";50.
    end)
    @test changed.project.box[2]=="400" && caller.box[2]=="800"

    owned=deepcopy(project);owned.box[2]="WBOX";owned.variables["WBOX"]="400"
    varied=sonnet_component_files(owned,300e6;variables=Dict("WBOX"=>400.),grid=(32,40))
    @test varied.variables==Dict("WBOX"=>400.) && varied.grid==(32,40)
    @test_throws ArgumentError solve_sonnet_project(varied;raw=true,variables=Dict("WBOX"=>800.))
    actual=solve_sonnet_project(files;raw=true,mx=64,my=80)
    automatic=solve_sonnet_project(project,300e6;raw=true,mx=64,my=80)
    @test actual.model_files===files && automatic.model_files isa SonnetComponentFiles
    @test actual.s≈automatic.s rtol=1e-13
    @test solve_sonnet_project(varied;raw=true,mx=64,my=80).s≈actual.s rtol=1e-13
    old=SonnetComponentResult(actual.project,actual.raw,actual.port_numbers,actual.y,actual.s,
        actual.contraction,actual.circuit,actual.terminal_nodes,actual.incident_transfer,actual.gap_transfer)
    @test old.model_files===nothing && old.s==actual.s && old.z0==actual.z0
    count=Ref(0)
    response=(p,c,f)->(count[]+=1;(response=only(files.bindings).response,format=:s,z0=only(files.bindings).z0))
    supplied=solve_sonnet_project(project,300e6;raw=true,mx=64,my=80,component_response=response)
    @test supplied.model_files===nothing && count[]==1
    @test supplied.s≈actual.s rtol=1e-13
    archive=joinpath(proof,"portable_archive")
    archived=sonnet_component_files(read_sonnet_project(joinpath(archive,"coupon.son")),300e6)
    oldresponse=solve_sonnet_project(archived;raw=true,mx=64,my=80).s
    write(joinpath(archive,"device_model.s2p"),"invalid replacement\n")
    @test solve_sonnet_project(archived;raw=true,mx=64,my=80).s==oldresponse
    @test_throws ArgumentError sonnet_component_files(read_sonnet_project(joinpath(archive,"coupon.son")),300e6)
end

@testset "Linked static technology remains an exact separate dependency" begin
    technology=joinpath(proof,"mini.stf")
    xml="""<technology_file version="1700"><units/><public><materials>
      <dielectric name="Air"><params/></dielectric>
      <dielectric name="Alumina"><params erel="9.8"/></dielectric>
      </materials><metal_model_defs><metal_model name="Normal" model_type="Normal"/></metal_model_defs>
      <stackup><TOP material="Lossless" model="Normal"/>
      <diel name="Upper" dielectric="Air" thickness="6350"/>
      <diel name="Lower" dielectric="Alumina" thickness="635"/>
      <BOTTOM material="Lossless" model="Normal"/></stackup></public></technology_file>"""
    write(technology,xml)
    linkedpath=joinpath(proof,"linked.son")
    write(linkedpath,replace(read(coupon,String),"TMET \"Lossless\""=>"STF mini.stf\nTMET \"Lossless\""))
    linked=read_sonnet_linked_project(linkedpath)
    @test linked.source==realpath(linkedpath)
    @test linked.technology.source==realpath(technology)
    files=sonnet_component_files(linked,300e6)
    @test files.linked!==linked && length(files.sources)==3
    @test files.sources[linked.technology.source].sha256==linked.technology.sha256
    @test files.sources[linked.source].sha256==linked.sha256
    @test files.sources[linked.technology.source].bytes==Vector{UInt8}(codeunits(xml))
    @test files.project.polygons[1].id==project.polygons[1].id
    @test files.project.components[1][1].tokens==project.components[1][1].tokens
    savedname=files.linked.technology.root.name
    linked.technology.root.attributes["caller_edit"]="mutated"
    @test !haskey(files.linked.technology.root.attributes,"caller_edit")
    @test files.linked.technology.root.name==savedname
    @test_throws ArgumentError sonnet_component_files(linked,300e6;max_bytes=1)
    actual=solve_sonnet_project(files;raw=true,mx=64,my=80)
    independent=solve_sonnet_project(project,300e6;raw=true,mx=64,my=80)
    @test maximum(abs,actual.s-independent.s)<1e-10
    write(technology,xml*"\n<!-- dependency edit -->\n")
    @test solve_sonnet_project(files;raw=true,mx=64,my=80).s==actual.s
    fresh=sonnet_component_files(read_sonnet_linked_project(linkedpath),300e6)
    @test fresh.sources[linked.technology.source].sha256!=files.sources[linked.technology.source].sha256
    @test only(fresh.bindings).response==only(files.bindings).response
end

    end
end
