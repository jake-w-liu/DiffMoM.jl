using Test
using DiffMoM
using LinearAlgebra
using TOML
using SHA, JSON

include("test_planar_sonnet_scalar_units.jl")
include("test_planar_sonnet_scalar_files.jl")
include("test_planar_sonnet_logarithms.jl")
include("test_planar_sonnet_functions.jl")
include("test_planar_sonnet_complex_native_arguments.jl")

@testset "Actual native dielectric resistivity loss selection" begin
    fixture=joinpath(@__DIR__,"fixtures","native_dielectric_resistivity_gui")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
    end
    native_pairs=Dict{Int,Dict{String,Matrix{ComplexF64}}}()
    for ghz in (1,5,10)
        g=CellGrid(.001,.001,20,20);sheet=sheet_level(1,20,20)
        sheet.mask[:,9:12].=true;sheet.connect_west[9:12].=true;sheet.connect_east[9:12].=true
        expected_sigma=100/7
        eps=11.9-im*expected_sigma/(2pi*ghz*1e9*8.8541878128e-12)
        stack=PlanarStackup([PlanarLayer(eps,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
        handprob=build_planar_problem(stack,g,[sheet],[PlanarPort(1,:west,9:12,50.),PlanarPort(1,:east,9:12,50.)])
        hand=solve_planar(handprob,ghz*1e9;mx=160,my=160)
        rhs=zeros(ComplexF64,length(handprob.basis.kind),2)
        for b in eachindex(handprob.basis.kind)
            q=handprob.basis.port[b];q==0 && continue
            rhs[b,q]=(q==1 ? -1 : 1)*handprob.basis.width[b]
        end
        @test norm(hand.z_mom*hand.currents-rhs)/norm(rhs)<1e-9
        native_pairs[ghz]=Dict{String,Matrix{ComplexF64}}()
        for mode in ("rsvy","conductivity")
            tag="$(mode)_$(ghz)GHz";directory=joinpath(fixture,tag)
            p=read_sonnet_project(joinpath(directory,tag*".son"))
            native=only(planar_read_touchstone(joinpath(directory,"native.s2p")).s)
            native_pairs[ghz][mode]=native
            imported=solve_sonnet_project(p,ghz*1e9;raw=true,mx=160,my=160)
            @test maximum(abs,imported.s-native)<.005
            @test maximum(abs,imported.s-hand.s)<2e-10
            @test imported.raw.problem.stack.layers[1].epsr≈eps rtol=2e-9
            @test only(imported.raw.problem.sheets).mask==sheet.mask
            @test imported.raw.currents≈hand.currents rtol=1e-8
            @test norm(imported.raw.z_mom*imported.raw.currents-rhs)/norm(rhs)<1e-9
            voltages=ComplexF64[1+.2im,-.3+.4im]
            actual=only(planar_current_maps(imported.raw;voltages))
            expected=only(planar_current_maps(hand;voltages))
            @test actual.jx≈expected.jx rtol=1e-8
            # Compare the vector field with a total-field relative norm;
            # a component can vanish in this symmetric straight strip.
            @test hypot(norm(actual.jx-expected.jx),norm(actual.jy-expected.jy))<
                1e-8*hypot(norm(expected.jx),norm(expected.jy))
            metadata=TOML.parsefile(joinpath(directory,"metadata.toml"))
            @test metadata["source_sha256"]==bytes2hex(sha256(read(p.source)))
            @test startswith(metadata["engine_version"],"18.53-Lite (64-bit Windows)")
            @test metadata["actual_cell_counts"]==[20,20]
            @test metadata["native_subsections"]==[38]
            @test metadata["touchstone_selected_log_checks"]["native.s2p"]["status"]=="PASS"
        end
        @test maximum(abs,native_pairs[ghz]["rsvy"]-native_pairs[ghz]["conductivity"])<2e-13
    end
    p=read_sonnet_project(joinpath(fixture,"rsvy_1GHz","rsvy_1GHz.son"))
    @test occursin("Rho = 7 Ohm-cm",read(joinpath(fixture,"rsvy_1GHz","engine_stdout.log"),String))
    @test DiffMoM._sonnet_dielectric_sigma(p,p.layers[2],1e9,Dict("Rho"=>14.))≈50/7
    # Dielectric marker units are distinct from conductor NOR/SRVY flags.
    for unit in ("OH","KOH","MOH")
        changed=deepcopy(p);changed.units["RES"]=unit
        @test DiffMoM._sonnet_dielectric_sigma(changed,changed.layers[2],1e9,Dict())≈100/7
    end
    # Lite 18.53: only the first token after the layer name is the loss
    # selector — RSVY reads ohm-cm and SRVY reads ohms-per-square through
    # the 0.1 mm layer; every other token, including CDVY, selects plain
    # conductivity and later tokens are inert.
    for (suffix,sigma) in ((["UNKNOWN"],7.),(["ANISO"],7.),(["CDVY","RSVY"],7.),
            (["RSVY","UNKNOWN"],100/7),(["SRVY"],1/(7*.1*.001)))
        changed=deepcopy(p);resize!(changed.layers[2],8);append!(changed.layers[2],suffix)
        @test DiffMoM._sonnet_dielectric_sigma(changed,changed.layers[2],1e9,Dict())≈sigma
    end
    for value in ("0","-1","1e-320")
        bad=deepcopy(p);bad.variables["Rho"]=value
        @test_throws ArgumentError sonnet_planar_problem(bad;freq=1e9)
    end
    bad=deepcopy(p);bad.variables["Rho"]="1e309"
    @test_throws ArgumentError sonnet_planar_problem(bad;freq=1e9)
    @test_throws ArgumentError sonnet_planar_problem(p;freq=1e9,variables=Dict("Rho"=>Inf))
    bad=deepcopy(p);pop!(bad.layers[2]);bad.variables["Rho"]="-1"
    @test_throws ArgumentError sonnet_planar_problem(bad;freq=1e9)
    before=TOML.parsefile(joinpath(fixture,"replay_before.toml"))
    @test all(r->r["reader_delta_s"]>.16,filter(r->startswith(r["tag"],"rsvy"),before["runs"]))
end

@testset "Native material GUI field-order contract" begin
    directory=joinpath(@__DIR__,"..","validation","sonnet_stripline",
        "native_material_reference")
    p=read_sonnet_project(joinpath(directory,"planar_order_probe.son"))
    # Actual installed18.53 GUI displays SUP Rdc1/Rrf2/Xdc3/Ls4,
    # and SFC Rdc1/Xdc1e-5/Rrf.2 — matching the materialized row order
    # `SFC rdc xdc rrf` decoded via soncmd -LoadTechnology.
    @test p.metals[1]==["Planar","1","SUP","1","2","3","4"]
    @test p.metals[2]==["Bulk","1","SFC","1","1e-5",".2"]
    f=1e9
    rf=(1+im)*2sqrt(f)
    @test sonnet_metal_zs(p,p.metals[1],f)≈
        rf/tanh(rf)+im*(3+2pi*f*4e-12)
    # SFC on a sheet is the surface impedance Zs=transition(Rdc,Rrf*f^.5)+jXdc.
    ref=DiffMoM._sonnet_loss_transition(1,.2*sqrt(f))
    @test sonnet_metal_zs(p,p.metals[2],f)≈complex(real(ref),imag(ref)+1e-5)
    # The SFC-material RING via materializes as a surface tube.
    prob=sonnet_planar_problem(p;grid=(8,8),freq=f,_materials=true)
    @test !isempty(prob.vias)
end

@testset "Native primitive resistance units oracle" begin
    directory=joinpath(@__DIR__,"..","validation","sonnet_stripline","native_circuit_units_reference")
    expected=ComplexF64[1/3 2/3;2/3 1/3]
    for unit in ("OH","OHMS","KOH","MOH")
        project=read_sonnet_project(joinpath(directory,unit*".son"))
        native=planar_read_touchstone(joinpath(directory,unit*".s2p"))
        circuit=sonnet_planar_circuit(project)
        @test circuit.z0==[50.,50.]
        for (f,actual) in zip(native.frequencies,native.s)
            @test maximum(abs.(actual-expected))<=4e-14
            @test maximum(abs.(solve_planar_circuit(circuit,f).s-actual))<=4e-14
        end
        @test project.units["RES"]==unit
        # Unknown spellings must fail explicitly, even if the native engine
        # silently falls back to ohms for some undocumented tokens.
        units=copy(project.units);units["RES"]="UOH"
        unsupported=SonnetNetlistProject(project.source,units,project.frequency_scale,
            project.circuit,project.records)
        @test_throws ArgumentError sonnet_planar_circuit(unsupported)
    end
end

@testset "Native complex reference GUI oracle" begin
    directory=joinpath(@__DIR__,"..","validation","sonnet_stripline","native_reference")
    p=read_sonnet_project(joinpath(directory,"baseline.son"))
    baseline=planar_read_touchstone(joinpath(directory,"baseline.s2p"))
    variants=(
        ("reactance",[(50.,20.,0.,0.),(75.,-10.,0.,0.)]),
        ("inductance",[(50.,0.,1.,0.),(75.,0.,.5,0.)]),
        ("capacitance",[(50.,0.,0.,2.),(75.,0.,0.,4.)]),
        ("combined",[(50.,10.,.25,2.),(75.,-10.,.5,4.)]))
    clone(ports,units=p.units,variables=p.variables)=SonnetProject(p.source,units,
        p.length_scale,p.frequency_scale,p.box,p.layers,p.metals,p.top,p.bottom,
        p.polygons,ports,variables,p.components,p.sweeps,p.records)
    nativeports(values)=[SonnetPortSpec(ps.kind,ps.polygon,ps.edge,ps.number,
        vcat(ps.values[1:1],string.(collect(values[i])),ps.values[6:end]),ps.records)
        for (i,ps) in enumerate(p.ports)]
    for (name,values) in variants
        ports=nativeports(values);project=clone(ports)
        native=planar_read_touchstone(joinpath(directory,"native_graph_"*name*".s2p"))
        circuit=SonnetNetlistProject(joinpath(directory,"corpus_wrapper.son"),p.units,
            p.frequency_scale,[SonnetRecord(1,["S2P","1","2","native_graph_"*name*".s2p"]),
                SonnetRecord(2,["DEF2P","1","2","NativeGUI","50"])],SonnetRecord[])
        @test native.frequencies==baseline.frequencies==[1e9,5e9,10e9]
        for (k,f) in enumerate(baseline.frequencies)
            # Independent literal native RLC topology, fixed port field units.
            refs=[inv(inv(r+im*x+im*2pi*f*l*1e-9)+im*2pi*f*c*1e-12)
                for (r,x,l,c) in values]
            evaluated=[DiffMoM._sonnet_port_reference(project,ps,f) for ps in ports]
            @test evaluated≈refs rtol=1e-14
            exported=native.reference_series===nothing ? native.z0 : native.reference_series[k]
            @test exported≈refs rtol=1e-14
            y=planar_s_to_y(baseline.s[k],baseline.z0)
            @test maximum(abs.(planar_y_to_s(y,evaluated)-native.s[k]))<1e-9
            @test planar_s_to_y(native.s[k],exported)≈y rtol=1e-9 atol=1e-11
            @test solve_sonnet_project(circuit,f).s≈baseline.s[k] rtol=1e-9 atol=1e-10
            changed_units=copy(p.units)
            changed_units["RES"]="KOH";changed_units["IND"]="UH";changed_units["CAP"]="NF"
            @test [DiffMoM._sonnet_port_reference(clone(ports,changed_units),ps,f) for ps in ports]==evaluated
        end
    end
    ports=nativeports(last(variants)[2]);project=clone(ports)
    real_result=solve_sonnet_project(p,5e9;raw=true,grid=(8,8),mx=16,my=16)
    result=solve_sonnet_project(project,5e9;raw=true,grid=(8,8),mx=16,my=16)
    @test result.y≈real_result.y rtol=1e-13
    @test result.z0==[DiffMoM._sonnet_port_reference(project,ps,5e9) for ps in ports]
    @test result.s≈planar_y_to_s(real_result.y,result.z0) rtol=1e-13
    a=ComplexF64[.7+.2im,-.1+.3im];b=result.s*a
    v=(conj.(result.z0).*a+result.z0.*b)./sqrt.(real.(result.z0))
    maps=only(planar_current_maps(result;incident_waves=a))
    independent=only(planar_current_maps(result.raw;voltages=result.voltage_transfer*v))
    @test maps.jx≈independent.jx rtol=1e-13
    @test maps.jy≈independent.jy rtol=1e-13
    vars=copy(p.variables);vars["ZREF"]="50+FREQ/1000000000"
    exprvalues=copy(ports[1].values);exprvalues[2]="ZREF"
    expression=SonnetPortSpec(ports[1].kind,ports[1].polygon,ports[1].edge,
        ports[1].number,exprvalues,ports[1].records)
    exprproject=clone([expression,ports[2]],p.units,vars)
    expected=inv(inv(55+10im+im*2pi*5e9*.25e-9)+im*2pi*5e9*2e-12)
    @test DiffMoM._sonnet_port_reference(exprproject,expression,5e9)≈expected
    invalid=copy(ports[1].values);invalid[2]="0";invalid[3:5].="0"
    bad=SonnetPortSpec(ports[1].kind,ports[1].polygon,ports[1].edge,ports[1].number,invalid,ports[1].records)
    @test_throws ArgumentError DiffMoM._sonnet_port_reference(p,bad,1e9)
    field=planar_farfield(result;incident_waves=a,theta=[.4,.9],phi=[.2])
    direct=planar_farfield(result.raw;voltages=result.voltage_transfer*v,theta=[.4,.9],phi=[.2])
    @test field.intensity≈direct.intensity
    @test field.etheta≈direct.etheta
    @test field.ephi≈direct.ephi
    @test field.accepted_power===direct.accepted_power
    unit=planar_farfield(result;theta=[.4],phi=[.2])
    onewave=planar_farfield(result;incident_waves=ComplexF64[1,0],theta=[.4],phi=[.2])
    @test unit.intensity≈onewave.intensity
    @test isfinite(unit.frequency)
    voltagefield=planar_farfield(result;voltages=v,theta=[.4,.9],phi=[.2])
    @test voltagefield.intensity≈field.intensity
    power=planar_radiated_power(result;incident_waves=a,ntheta=8,nphi=16)
    manual=planar_radiated_power(result.raw;voltages=result.voltage_transfer*v,ntheta=8,nphi=16)
    @test power.power≈manual.power
    @test power.coarse_power≈manual.coarse_power
    @test power.relative_change===manual.relative_change
    @test_throws ArgumentError planar_farfield(result;voltages=v,incident_waves=a)
    @test_throws ArgumentError planar_farfield(result;port=3)
    @test_throws ArgumentError planar_farfield(result;incident_waves=[NaN,0])
    @test_throws ArgumentError planar_radiated_power(result;voltages=[1.])
    @test_throws ArgumentError planar_radiated_power(result;max_bytes=1)
end

@testset "Native balanced current terminal contract" begin
    # Two coupled differential groups, duplicate positive/negative pads,
    # a height-weighted multilayer via path, and a grounded third source.
    labels=[1,1,1,2,2,2,2,3];signs=[1,1,-1,1,1,-1,-1,1]
    weights=[1.,1.,1.,.4,.6,.4,.6,1.]
    C=zeros(8,3)
    for i in eachindex(labels);C[i,labels[i]]=weights[i];end
    maps=DiffMoM._sonnet_terminal_maps(C,signs)
    @test size(maps.floating_common)==(8,2)
    @test maps.contraction[:,3]==C[:,3]
    @test maps.contraction[:,1:2]==C[:,1:2]/2
    # Independent physical nodal MNA: positive and negative pads share
    # their respective potentials; differential ideal sources enforce Vp−Vn.
    physical_nodes=[1,1,2,3,3,4,4,5];G=zeros(8,5)
    for i in eachindex(labels);G[i,physical_nodes[i]]=weights[i];end
    A=[1. 0 0;-1 0 0;0 1 0;0 -1 0;0 0 1]
    R=[(i==j ? 3. : .03/(1+abs(i-j))) for i in 1:8,j in 1:8]
    Yphys=ComplexF64.(R)+im.*[.2/(1+abs(i-j)) for i in 1:8,j in 1:8]
    Yraw=Diagonal(signs)*Yphys*Diagonal(signs)
    Yn=transpose(G)*Yphys*G
    mna=[Yn -A;transpose(A) zeros(3,3)]\[zeros(5,3);Matrix{Float64}(I,3,3)]
    balanced=DiffMoM._sonnet_balanced_y(Yraw,maps)
    @test balanced.y≈mna[6:8,:] rtol=2e-14
    @test Diagonal(signs)*balanced.voltage_transfer≈G*mna[1:5,:] rtol=2e-14
    @test norm(transpose(maps.floating_common)*Yraw*balanced.voltage_transfer)<2e-15
    @test balanced.y≈transpose(balanced.y) rtol=2e-14
    @test minimum(eigvals(Hermitian(real.(balanced.y))))>0
    @test_throws ArgumentError DiffMoM._sonnet_terminal_maps(ones(2,1),[-1,-1])
    @test_throws ArgumentError DiffMoM._sonnet_balanced_y(zeros(2,2),ones(2,1)/2,reshape([1.,-1.],2,1))
    @test_throws DimensionMismatch DiffMoM._sonnet_balanced_y(Yraw,zeros(7,3),zeros(8,2))
    @test_throws ArgumentError DiffMoM._sonnet_balanced_y(fill(NaN,8,8),maps)
end

@testset "Native Sonnet geometry import" begin
    fixture=joinpath(@__DIR__,"..","validation","sonnet_stripline","s50.son")
    p=read_sonnet_project(fixture)
    @test p.length_scale==1e-3
    @test p.frequency_scale==1e9
    @test length(p.layers)==2
    @test length(p.polygons)==1
    @test p.polygons[1].vertices[1,2]≈4.996541e-3
    @test [x.number for x in p.ports]==[1,2]
    prob=sonnet_planar_problem(p;grid=(16,16),freq=15e9)
    native=sonnet_planar_problem(p;freq=15e9)
    @test native.grid.nx==parse(Int,p.box[4])÷2
    @test native.grid.ny==parse(Int,p.box[5])÷2
    @test prob.grid.nx==16
    @test prob.sheets[1].interface==1
    @test count(prob.sheets[1].mask)==32
    @test prob.ports[1].wall==:west
    @test prob.ports[2].wall==:east
    @test prob.ports[1].cells==8:9
    negvalues=copy(p.ports[2].values);negvalues[1]="-1"
    negport=SonnetPortSpec(p.ports[2].kind,p.ports[2].polygon,p.ports[2].edge,
        -1,negvalues,p.ports[2].records)
    paired=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,
        p.box,p.layers,p.metals,p.top,p.bottom,p.polygons,[p.ports[1],negport],
        p.variables,p.components,p.sweeps,p.records)
    native_balanced=solve_sonnet_project(paired,1e9;raw=true,grid=(8,8),mx=16,my=16)
    transferred=only(planar_current_maps(native_balanced))
    independently_transferred=only(planar_current_maps(native_balanced.raw;
        voltages=vec(native_balanced.voltage_transfer)))
    @test transferred.jx==independently_transferred.jx
    @test transferred.jy==independently_transferred.jy
    native_waves=only(planar_current_maps(native_balanced;incident_waves=[1.]))
    native_voltage=sqrt(50.)*(1+native_balanced.s[1,1])
    @test native_waves.jx≈transferred.jx*native_voltage
    launch=[planar_line_abcd(50.,.01im)]
    calibrated_balanced=solve_sonnet_project(paired,1e9;raw=true,calibration=launch,
        grid=(8,8),mx=16,my=16)
    @test calibrated_balanced.y≈deembed_ports(native_balanced.y,launch)
    @test calibrated_balanced.voltage_transfer≈native_balanced.voltage_transfer*
        (launch[1][1,1]+launch[1][1,2]*calibrated_balanced.y[1,1])
    @test_throws ArgumentError planar_current_maps(native_balanced;voltages=[1.],incident_waves=[1.])
    @test_throws ArgumentError planar_current_maps(native_balanced;max_bytes=1)
    portless=SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,
        p.box,p.layers,p.metals,p.top,p.bottom,p.polygons,SonnetPortSpec[],
        p.variables,p.components,p.sweeps,p.records)
    @test_throws ArgumentError sonnet_planar_problem(portless;grid=(16,16))
    base_geometry=sonnet_planar_problem(portless;grid=(16,16),_allow_portless=true,_details=true)
    @test isempty(base_geometry.problem.ports)
    @test size(base_geometry.contraction)==(0,0)
    @test base_geometry.problem.sheets[1].mask==prob.sheets[1].mask
    @test sonnet_variable_value(p,"2*pi*FREQ";freq=15e9)≈30e9*pi
    @test_throws ArgumentError sonnet_variable_value(p,"run(`bad`)")
    @test_throws ArgumentError sonnet_variable_value(p,"missing_parameter")
    @test_throws ArgumentError sonnet_variable_value(p,"Inf")
    @test_throws ArgumentError sonnet_planar_problem(p;freq=0.)
    # `OPTIONS -d` auto-de-embeds wall-port launch sections natively (Lite
    # 18.53: de-embedded output even without DRP records) — no longer a
    # calibration-required rejection.
    deembedded=solve_sonnet_project(p,15e9;grid=(8,8))
    uncalibrated=solve_sonnet_project(p,15e9;grid=(8,8),raw=true)
    @test deembedded.s!=uncalibrated.s
    @test sonnet_metal_zs(p,["Sheet","1","RES","2"],1e9)==2+0im
    @test sonnet_metal_zs(p,["Sense","1","SEN","1000000"],1e9)==im*1e6
    @test sonnet_metal_zs(p,["PEC","1","SUP","0","0","0","0"],1e9)==0im
    space=["Free Space","0","FREESPACE","376.7303136","0","0","0"]
    termination=DiffMoM._sonnet_termination(p,space,1e9,Dict{String,Float64}())
    @test termination===DiffMoM.TERM_SPACE
    @test termination.kind==DiffMoM.TERM_OPEN
    @test_throws ArgumentError sonnet_metal_zs(p,space,1e9)
    @test sonnet_metal_zs(p,["PEC","1","NOR","INF",".5","0"],1e9)==0im
    @test imag(sonnet_metal_zs(p,["Kinetic","1","SUP","0","0","0",".11"],1e9))≈2pi*1e9*.11e-12
    low=sonnet_metal_zs(p,["Cu","1","NOR","58000000","1",".001"],1.)
    @test real(low)≈1/(58000000*1e-6) rtol=1e-7
    skin=planar_surface_zs(1e9,58e6);k=(1+.5^2)/(1+.5)^2
    expected=k*skin/tanh(k*58e6*skin*1e-6)
    @test sonnet_metal_zs(p,["Cu","1","NOR","58000000",".5",".001"],1e9)≈expected rtol=1e-13
    @test_throws ArgumentError sonnet_metal_zs(p,["Thick","1","TMM","58000000",".5",".1","2"],1e9)

    mktempdir() do dir
        source=replace(read(fixture,String),"\r\n"=>"\n")
        path=joinpath(dir,"case.son")
        # GUI independently verifies BOX stores half-cell counts.
        write(path,replace(source,r"(?m)^(BOX 1 [^ ]+ [^ ]+) \d+ \d+"=>s"\1 32 32"))
        @test sonnet_planar_problem(read_sonnet_project(path)).grid.nx==16
        write(path,replace(read(path,String)," 32 32 100 0"=>" 31 32 100 0"))
        @test_throws ArgumentError sonnet_planar_problem(read_sonnet_project(path))
        tmm=replace(source,"BOX 1"=>"MET \"Cu\" 1 TMM 58000000 .5 .02 2\nBOX 1",
            "0 5 -1 N 9"=>"0 5 0 N 9")
        write(path,tmm)
        thick=sonnet_planar_problem(read_sonnet_project(path);grid=(16,16),freq=1e9,
            _materials=true,_details=true)
        @test length(thick.problem.stack.layers)==3
        @test thick.problem.stack.layers[2].thickness≈20e-6
        @test sort([s.interface for s in thick.problem.sheets])==[1,2]
        @test length(thick.problem.ports)==4
        @test size(thick.contraction)==(4,2)
        @test all(sum(thick.contraction;dims=1).==2)
        @test thick.problem.vias[1].layer==2
        @test only(unique(thick.sheet_zs[1][thick.problem.sheets[1].mask]))≈
            planar_two_sheet_zs(1e9,58e6,20e-6)[1,1]
        write(path,replace(tmm,".02 2\nBOX"=>".02 2 CDVY TDWN\nBOX"))
        down=sonnet_planar_problem(read_sonnet_project(path);grid=(16,16),freq=1e9,
            _materials=true,_details=true)
        @test down.problem.stack.layers[1].thickness≈480e-6
        @test down.problem.stack.layers[2].thickness≈20e-6
        write(path,replace(tmm,".02 2\nBOX"=>".02 3\nBOX"))
        @test_throws ArgumentError sonnet_planar_problem(read_sonnet_project(path);_materials=true)
        base=read_sonnet_project(fixture);box=copy(base.box);box[1]="2"
        rows=deepcopy(base.layers);rows[2][1]=".2";push!(rows,copy(rows[2]));rows[3][1]=".3"
        via=SonnetPolygon(:via,0,-1,99,[.002 .003 .003 .002;.005 .005 .006 .006],
            "GND","","",["RING","NOCOVERS"])
        pv=SonnetPortSpec(:via,99,0,3,["3","50","0","0","0","2.5","5.5"],SonnetRecord[])
        multivia=SonnetProject(base.source,base.units,base.length_scale,base.frequency_scale,
            box,rows,base.metals,base.top,base.bottom,vcat(base.polygons,[via]),
            vcat(base.ports,[pv]),base.variables,base.components,base.sweeps,base.records)
        axial=sonnet_planar_problem(multivia;grid=(16,16),_details=true)
        @test axial.labels==[1,2,3]
        @test Set(axial.contraction[axial.contraction[:,3].>0,3])==Set([.4,.6])
        @test Set(v.layer for v in axial.problem.vias)==Set([1,2])
        coverstd=SonnetPortSpec(:std,pv.polygon,pv.edge,pv.number,pv.values,pv.records)
        standardvia=SonnetProject(multivia.source,multivia.units,multivia.length_scale,
            multivia.frequency_scale,multivia.box,multivia.layers,multivia.metals,
            multivia.top,multivia.bottom,multivia.polygons,vcat(base.ports,[coverstd]),
            multivia.variables,multivia.components,multivia.sweeps,multivia.records)
        @test sonnet_planar_problem(standardvia;grid=(16,16),_details=true).contraction==axial.contraction
        # The constant axial polygon resistance is independent of subsection
        # area and distributed by physical layer height across multiple layers.
        resistive=SonnetPolygon(via.kind,via.level,0,via.id,via.vertices,via.target,
            via.technology,via.technology_mode,via.flags)
        rpv=SonnetProject(base.source,base.units,base.length_scale,base.frequency_scale,
            box,rows,[["Via","1","VOL","2","0","RPV"]],base.top,base.bottom,
            vcat(base.polygons,[resistive]),vcat(base.ports,[pv]),base.variables,
            base.components,base.sweeps,base.records)
        rmodel=sonnet_planar_problem(rpv;grid=(16,16),_materials=true,_details=true)
        @test length(rmodel.via_sigma)==2
        @test rmodel.via_sigma[1]==rmodel.via_sigma[2]
        @test sum(real(rmodel.problem.stack.layers[v.layer].thickness)/
            (rmodel.via_sigma[i]*count(v.uni)*rmodel.problem.grid.dx*rmodel.problem.grid.dy)
            for (i,v) in enumerate(rmodel.problem.vias))≈2
        rmodel_fine=sonnet_planar_problem(rpv;grid=(32,32),_materials=true,_details=true)
        @test sum(real(rmodel_fine.problem.stack.layers[v.layer].thickness)/
            (rmodel_fine.via_sigma[i]*count(v.uni)*rmodel_fine.problem.grid.dx*rmodel_fine.problem.grid.dy)
            for (i,v) in enumerate(rmodel_fine.problem.vias))≈2
        array=SonnetProject(rpv.source,rpv.units,rpv.length_scale,rpv.frequency_scale,
            rpv.box,rpv.layers,[["Array","1","ARR","2","100","RPV",".25"]],
            rpv.top,rpv.bottom,rpv.polygons,rpv.ports,rpv.variables,rpv.components,rpv.sweeps,rpv.records)
        amodel=sonnet_planar_problem(array;grid=(16,16),_materials=true,_details=true)
        @test amodel.via_sigma[1]/rmodel.via_sigma[1]≈.25e6
        # The translator's canonical array record carries only loss and fill
        # factor; its effective conductivity is the volume skin law applied
        # to the filled fraction of the polygon cross-section.
        arrayfill=SonnetProject(rpv.source,rpv.units,rpv.length_scale,rpv.frequency_scale,
            rpv.box,rpv.layers,[["Array","1","ARR","58e6","16"]],
            rpv.top,rpv.bottom,rpv.polygons,rpv.ports,rpv.variables,rpv.components,rpv.sweeps,rpv.records)
        fmodel=sonnet_planar_problem(arrayfill;grid=(16,16),freq=1e9,_materials=true,_details=true)
        fmesh=count(fmodel.problem.vias[1].uni)*fmodel.problem.grid.dx*fmodel.problem.grid.dy
        fexpected=DiffMoM._sonnet_volume_sigma_wide(58e6,1e9,1e-6,16.,1e-3,1e-3,fmesh)
        @test all(s->isapprox(s,fexpected;rtol=1e-10),fmodel.via_sigma)
        grounded=copy(base.ports[1].values);grounded[1]="0"
        groundport=SonnetPortSpec(:std,9,3,0,grounded,base.ports[1].records)
        wallground=SonnetProject(base.source,base.units,base.length_scale,base.frequency_scale,
            base.box,base.layers,base.metals,base.top,base.bottom,base.polygons,
            [groundport,base.ports[2]],base.variables,base.components,base.sweeps,base.records)
        groundmodel=sonnet_planar_problem(wallground;grid=(16,16),_details=true)
        @test groundmodel.labels==[2]
        @test all(groundmodel.problem.sheets[1].connect_west[8:9])
        # A native gap edge drives its whole pad span, not just the cell
        # containing the recorded marker. Here both pad halves share a seam.
        full=only(base.polygons);mid=base.length_scale*parse(Float64,base.box[2])/2
        leftv=copy(full.vertices);leftv[1,leftv[1,:].>mid].=mid
        rightv=copy(full.vertices);rightv[1,rightv[1,:].<mid].=mid
        left=SonnetPolygon(:sheet,0,-1,9,leftv,"","","",String[])
        right=SonnetPolygon(:sheet,0,-1,10,rightv,"","","",String[])
        gap=SonnetPortSpec(:gap,9,1,1,["1","50","0","0","0",string(mid/base.length_scale),
            string(parse(Float64,base.box[3])/2)],SonnetRecord[])
        gaps=SonnetProject(base.source,base.units,base.length_scale,base.frequency_scale,
            base.box,base.layers,base.metals,base.top,base.bottom,[left,right],[gap],
            base.variables,base.components,base.sweeps,base.records)
        gmodel=sonnet_planar_problem(gaps;grid=(16,16),_details=true)
        @test only(gmodel.problem.ports).cells==8:9
        @test only(gmodel.problem.ports).edge==8
        write(path,replace(source,"NUM 1"=>"NUM 2"))
        @test_throws ArgumentError read_sonnet_project(path)
        write(path,replace(source,"POLY 9 1"=>"POLY 999 1"))
        @test_throws ArgumentError read_sonnet_project(path)
        write(path,replace(source,"4.996541 6.4907532"=>"NaN 6.4907532"))
        @test_throws ArgumentError read_sonnet_project(path)
        # Lite routes by content, not the FTYP token: SONNETPRJ on a GEO
        # file runs the EM solve and SONPROJ on a CKT file runs the
        # netlist solve (installed-Lite probes). Both tokens accept both
        # project kinds; only garbage/lowercase types reject.
        write(path,replace(source,"FTYP SONPROJ"=>"FTYP SONNETPRJ"))
        @test read_sonnet_project(path) isa SonnetProject
        write(path,replace(source,"FTYP SONPROJ"=>"FTYP garbage"))
        @test_throws ArgumentError read_sonnet_project(path)
        write(path,replace(source,"GEO\n"=>"GEO\nSTF missing.stf\n";count=1))
        @test_throws ArgumentError read_sonnet_project(path)
        # Restricted expressions retain native units and resolve dependencies.
        write(path,replace(source,"LORGN"=>"VALVAR H LNG 0.5 \"height\"\nVALVAR Twice LNG \"2*H\" \"dependent\"\nLORGN"))
        v=read_sonnet_project(path)
        @test sonnet_variable_value(v,"Twice")==1.0
        @test sonnet_variable_value(v,"Twice";variables=Dict("H"=>.25))==.5
        write(path,replace(read(path,String),"VALVAR H LNG 0.5"=>"VALVAR H LNG Twice"))
        @test_throws ArgumentError sonnet_variable_value(read_sonnet_project(path),"H")
        netlist="""
        FTYP SONNETPRJ 19
        DIM
        LNG MM
        FREQ GHZ
        CAP PF
        IND NH
        END DIM
        CKT
        RES 1 2 R=50
        DEF2P 1 2 SERIES R 50
        SERIES 1 2
        SERIES 2 3
        DEF2P 1 3 CASCADE R 50
        END CKT
        """
        write(path,netlist)
        n=read_sonnet_project(path)
        @test n isa SonnetNetlistProject
        @test length(n.circuit)==5
        circuit=sonnet_planar_circuit(n)
        result=solve_planar_circuit(circuit,1e9)
        @test result.s≈[.5 .5;.5 .5] atol=1e-12
        @test solve_sonnet_project(n,1e9).s≈result.s atol=1e-12
        # The SONPROJ token on netlist content is equally valid natively
        # (installed-Lite: halfnet.son retokened SONPROJ ran the identical
        # circuit-theory sweep).
        write(path,replace(netlist,"FTYP SONNETPRJ"=>"FTYP SONPROJ"))
        n2=read_sonnet_project(path)
        @test n2 isa SonnetNetlistProject
        @test solve_sonnet_project(n2,1e9).s≈result.s atol=1e-12
        # Unknown circuits fail instead of disappearing during import.
        write(path,replace(netlist,"RES 1 2 R=50"=>"MAGIC 1 2 FOO=50"))
        @test_throws ArgumentError sonnet_planar_circuit(read_sonnet_project(path))
    end
end

@testset "Native sparse circuit labels preserve wiring" begin
    mktempdir() do directory
        write(joinpath(directory,"probe.s2p"),
            "# Hz S RI R 50\n1e9 .1 0 .8 .01 .8 .01 .1 0\n"*
            "2e9 .1 0 .8 .01 .8 .01 .1 0\n")
        function netlist(labels)
            a,b,c,d,e,f,g=labels
            return """
            FTYP SONNETPRJ 19
            DIM
            LNG MM
            FREQ GHZ
            CAP PF
            IND NH
            END DIM
            CKT
            RES $a $b R=13
            IND $b $c L=0.75
            CAP $c 0 C=2.5
            DEF2P $a $c BRANCH R 50
            BRANCH $d $e
            S2P $e $f probe.s2p
            PRJ $f $g coupon.son 2 R 50
            RES $g 0 R=80
            DEF2P $d $g FINAL R 50
            END CKT
            """
        end
        densepath=joinpath(directory,"dense.son")
        sparsepath=joinpath(directory,"sparse.son")
        write(densepath,netlist((1,2,3,10,20,30,40)))
        write(sparsepath,netlist((91,9001,70001,1111,100000,1000000001,2000000001)))
        denseproject=read_sonnet_project(densepath)
        sparseproject=read_sonnet_project(sparsepath)
        provider=(path,f)->begin
            @test path==joinpath(realpath(directory),"coupon.son")
            ComplexF64[.2 .8;.8 .2]
        end
        dense=sonnet_planar_circuit(denseproject;project_response=provider)
        sparse=sonnet_planar_circuit(sparseproject;project_response=provider)
        @test sparse.nnodes==dense.nnodes==4
        @test sparse.ports==dense.ports==[(1,0),(4,0)]
        @test sparseproject.circuit[1].tokens==["RES","91","9001","R=13"]
        @test sparseproject.circuit[3].tokens==["CAP","70001","0","C=2.5"]
        for frequency in (1e9,2e9)
            expected=solve_planar_circuit(dense,frequency)
            actual=solve_planar_circuit(sparse,frequency)
            @test actual.s≈expected.s rtol=1e-12 atol=1e-13
            @test actual.y≈expected.y rtol=1e-12 atol=1e-13
            @test solve_sonnet_project(sparseproject,frequency;
                project_response=provider).s≈expected.s rtol=1e-12 atol=1e-13
        end
        write(sparsepath,netlist((-91,9001,70001,1111,100000,1000000001,2000000001)))
        @test_throws ArgumentError sonnet_planar_circuit(read_sonnet_project(sparsepath);
            project_response=provider)
    end
end

module NativeSonnetResponseOracle
include(joinpath(@__DIR__,"..","validation","sonnet_stripline","sonnet_reference.jl"))
end

@testset "Native full-matrix response logs" begin
    reference=NativeSonnetResponseOracle.SonnetReference
    mktempdir() do dir
        path=joinpath(dir,"response.log")
        text="Non-de-embedded S-Parameters\nMagnitude/Angle. Matrix Order (line #1: S11 S12 S13).\n1 1 0 2 0 3 0\n4 0 5 0 6 0\n7 0 8 0 9 0\n"
        write(path,text)
        parsed=reference.parse_sonnet_response(path;deembedded=false,nports=3)
        @test reference.nport_matrix(only(parsed.rows))≈ComplexF64[1 2 3;4 5 6;7 8 9]
        write(path,replace(text,"7 0 8 0 9 0\n"=>""))
        @test_throws ErrorException reference.parse_sonnet_response(path;deembedded=false,nports=3)
        write(path,text*"1 1 0 2 0 3 0\n4 0 5 0 6 0\n7 0 8 0 9 0\n")
        @test_throws ErrorException reference.parse_sonnet_response(path;deembedded=false,nports=3)
        write(path,"Non-de-embedded S-Parameters\n1 .1 0 .2 0 .3 0 .4 0\n")
        parsed=reference.parse_sonnet_response(path;deembedded=false)
        @test reference.nport_matrix(only(parsed.rows))==ComplexF64[.1 .3;.2 .4]
        @test_throws DimensionMismatch reference.nport_matrix(ones(8))
    end
end

@testset "External native Touchstone selected-log guard" begin
    reference=NativeSonnetResponseOracle.SonnetReference
    mktempdir() do directory
        source=joinpath(directory,"source.son")
        write(source,"FTYP SONNETPRJ 19\nDIM\nFREQ GHZ\nEND DIM\nCONTROL\nOPTIONS -d\nEND CONTROL\n")
        function write_metadata(selected)
            open(joinpath(directory,"metadata.toml"),"w") do io
                TOML.print(io,Dict("source"=>source,"deembedded"=>selected))
            end
        end
        function write_touch(path,s;f=1.,z0=50,termination="")
            open(path,"w") do io
                println(io,"# GHz S RI R ",z0)
                isempty(termination) || println(io,termination)
                values=Any[f]
                for value in vec(size(s,1)<=2 ? s : permutedims(s))
                    push!(values,real(value),imag(value))
                end
                println(io,join(values,' '))
            end
        end
        for n in (1,2,4)
            matrix=ComplexF64.(reshape(collect(1:n^2),n,n)/100)
            log=joinpath(directory,"response_$(n).log")
            open(log,"w") do io
                println(io,"Non-de-embedded S-Parameters")
                if n<=2
                    values=Any[1.]
                    for value in vec(matrix);push!(values,abs(value),0.);end
                    println(io,join(values,' '))
                else
                    for row in 1:n
                        values=row==1 ? Any[1.] : Any[]
                        for column in 1:n;push!(values,abs(matrix[row,column]),0.);end
                        println(io,join(values,' '))
                    end
                end
            end
            parsed=reference.parse_sonnet_response(log;deembedded=false,nports=n)
            selected=(;parsed...,output_dir=directory,logf=log)
            write_metadata(false)
            matching=joinpath(directory,"matching_$(n).s$(n)p")
            write_touch(matching,matrix)
            @test reference.checked_native_touchstone(selected,matching;
                deembedded=false,expected_z0=50).s[1]≈matrix atol=1e-14
            metadata=TOML.parsefile(joinpath(directory,"metadata.toml"))
            check=metadata["touchstone_selected_log_checks"][basename(matching)]
            @test check["source_options"]==["-d"]
            @test check["intended_log_section"]=="raw"
            @test check["status"]=="PASS"
            @test check["frequency_scale_hz"]==1e9
            @test check["reference_contract"]=="explicit expected_z0"
            @test check["expected_reference_real"]==[fill(50.,n)]
            @test_throws ErrorException reference.checked_native_touchstone(selected,matching;
                deembedded=false)
            @test_throws UndefKeywordError reference.checked_native_touchstone(selected,matching)
            # Change a single off-diagonal entry for n>1, retaining S11.
            # Every full-matrix element must therefore be checked.
            wrong=copy(matrix);wrong[1,n]+=0.2
            mismatched=joinpath(directory,"mismatched_$(n).s$(n)p")
            write_touch(mismatched,wrong)
            @test_throws ErrorException reference.checked_native_touchstone(selected,mismatched;
                deembedded=false,expected_z0=50)
            metadata=TOML.parsefile(joinpath(directory,"metadata.toml"))
            failed=metadata["touchstone_selected_log_checks"][basename(mismatched)]
            @test failed["status"]=="FAIL"
            @test failed["max_printed_bound_ratio"]>1
            @test haskey(failed,"output_sha256")
            frequency=joinpath(directory,"frequency_$(n).s$(n)p")
            write_touch(frequency,matrix;f=2.)
            @test_throws ErrorException reference.checked_native_touchstone(selected,frequency;
                deembedded=false,expected_z0=50)
            @test_throws ErrorException reference.checked_native_touchstone(selected,matching;
                deembedded=true,expected_z0=50)
            # Values inside printed intervals are permitted without changing
            # the independent physics acceptance or residual thresholds.
            rounded=copy(matrix);rounded[1,1]+=1e-4
            bounded=joinpath(directory,"bounded_$(n).s$(n)p")
            write_touch(bounded,rounded)
            @test reference.checked_native_touchstone(selected,bounded;
                deembedded=false,expected_z0=50).s[1]≈rounded atol=1e-14
            @test size(reference.nport_quantization_bound(only(parsed.row_tokens)))==(n,n)
        end
        @test_throws DimensionMismatch reference.nport_quantization_bound(["1",".1"])
        @test_throws DimensionMismatch reference.nport_quantization_bound(String[])
        # Identical S does not identify the wave-reference basis: a matched
        # thru remains the same under a common real-reference change.
        write(source,"FTYP SONNETPRJ 19\nDIM\nLNG MM\nFREQ GHZ\nRES KOH\nEND DIM\nCKT\nRES 1 2 R=0\nDEF2P 1 2 THRU R 50\nEND CKT\n")
        log=joinpath(directory,"reference_thru.log")
        write(log,"Non-de-embedded S-Parameters. 50 Ohm Port Terminations.\n1 0.000000 0.00 1.000000 0.00 1.000000 0.00 0.000000 0.00\n")
        parsed=reference.parse_sonnet_response(log;deembedded=false)
        selected=(;parsed...,output_dir=directory,logf=log)
        write_metadata(false)
        matrix=ComplexF64[0 1;1 0]
        matching=joinpath(directory,"reference_thru.s2p")
        write_touch(matching,matrix)
        @test reference.checked_native_touchstone(selected,matching;deembedded=false).s[1]==matrix
        @test reference.checked_native_touchstone(selected,matching;deembedded=false,expected_z0=[50,50]).s[1]==matrix
        check=TOML.parsefile(joinpath(directory,"metadata.toml"))["touchstone_selected_log_checks"][basename(matching)]
        @test check["reference_contract"]=="native final DEF reference"
        @test_throws ErrorException reference.checked_native_touchstone(selected,matching;deembedded=false,expected_z0=75)
        wrong=joinpath(directory,"wrong_reference_thru.s2p")
        write_touch(wrong,matrix;z0=75)
        @test_throws ErrorException reference.checked_native_touchstone(selected,wrong;deembedded=false)
        check=TOML.parsefile(joinpath(directory,"metadata.toml"))["touchstone_selected_log_checks"][basename(wrong)]
        @test check["status"]=="FAIL"
        @test check["expected_reference_real"]==[[50.,50.]]
        @test check["output_reference_real"]==[[75.,75.]]
        write_touch(wrong,matrix;termination="! TERM 50 0 75 0")
        @test_throws ErrorException reference.checked_native_touchstone(selected,wrong;deembedded=false)
        write_touch(wrong,matrix;termination="! TERM 50 0 50 1")
        @test_throws ErrorException reference.checked_native_touchstone(selected,wrong;deembedded=false)
        # A caller-supplied frequency contract also checks FTERM, including
        # its evaluated reactive references, rather than just the real header.
        write(source,"FTYP SONNETPRJ 19\nDIM\nFREQ GHZ\nEND DIM\n")
        dynamic=joinpath(directory,"dynamic_reference_thru.s2p")
        write_touch(dynamic,matrix;termination="! FTERM 50 1 1e-9 0 75 -2 2e-9 0")
        providers=[f->50+im*(1+2pi*f*1e-9),f->75+im*(-2+2pi*f*2e-9)]
        @test reference.checked_native_touchstone(selected,dynamic;deembedded=false,expected_z0=providers).s[1]==matrix
        @test_throws ErrorException reference.checked_native_touchstone(selected,dynamic;deembedded=false,expected_z0=[providers[1],f->providers[2](f)+im])
        history=TOML.parsefile(joinpath(directory,"metadata.toml"))["touchstone_selected_log_check_history"]
        @test length(history)==8
        @test history[1]["status"]=="PASS"
        @test history[4]["output_reference_real"]==[[75.,75.]]
    end
end

@testset "Native via mesh semantics" begin
    grid=CellGrid(1.,1.,8,8);mask=falses(8,8);mask[2:6,2:6].=true
    v=[.125 .75 .75 .125;.125 .125 .75 .75]
    ring=DiffMoM._planar_via_mesh_mask(mask,v,grid,:ring)
    @test count(ring)==16
    @test !any(ring[3:5,3:5])
    @test count(DiffMoM._planar_via_mesh_mask(mask,v,grid,:full))==25
    @test count(DiffMoM._planar_via_mesh_mask(mask,v,grid,:vertices))==4
    @test count(DiffMoM._planar_via_mesh_mask(mask,v,grid,:center))==1
    @test_throws ArgumentError DiffMoM._planar_via_mesh_mask(mask,v,grid,:bar)
end

@testset "Native CUP calibration group records" begin
    fixture=joinpath(@__DIR__,"fixtures","native_floating_power_waves","floating_raw.son")
    template=replace(read(fixture,String),"\r\n"=>"\n")
    project=read_sonnet_project(fixture)
    @test [port.kind for port in project.ports]==[:cup,:cup]
    for port in project.ports
        keys=[record.tokens[1] for record in port.records]
        @test keys[end-2:end]==["CUPGRP","ID","END"]
        @test port.records[end-2].tokens==["CUPGRP","Auto","LOCAL"]
    end
    groups=sonnet_calibration_groups(project)
    @test length(groups)==1
    @test groups[1].name=="Auto" && groups[1].ports==[1,2]
    @test groups[1].numbers==[1,-1] && groups[1].groundref===:float
    # Coupled removal consumes member indices; this synthetic 2-port
    # network orders members like `project.ports`.
    Y=ComplexF64[-2im -im;-im -3im];launch=planar_line_abcd(50.,.02+.01im)
    # The coupled 2k×2k chain indexes each ABCD block by port, so an
    # uncoupled pair becomes four diagonal k×k blocks.
    chain=[Diagonal(fill(launch[1,1],2)) Diagonal(fill(launch[1,2],2));
           Diagonal(fill(launch[2,1],2)) Diagonal(fill(launch[2,2],2))]
    lifted=deembed_cocal_group(Y,groups[1].ports,Matrix{ComplexF64}(chain))
    @test lifted≈deembed_ports(Y,[launch,launch]) rtol=1e-13
    mktempdir() do dir
        path=joinpath(dir,"case.son")
        write(path,replace(template,"Auto"=>"A"," LOCAL"=>""))
        manual=sonnet_calibration_groups(read_sonnet_project(path))
        @test only(manual).name=="A" && only(manual).groundref===:float
        write(path,replace(template,"GNDREF FLOAT"=>"GNDREF BOX AUTO";count=1))
        split_groups=sonnet_calibration_groups(read_sonnet_project(path))
        @test length(split_groups)==2
        @test sort([g.groundref for g in split_groups])==[:box_auto,:float]
        @test sort([only(g.ports) for g in split_groups])==[1,2]
        write(path,replace(template,"POR1 CUP Auto"=>"POR1 CUP Auto\nSIMPLE Y";count=1))
        simplified=sonnet_calibration_groups(read_sonnet_project(path))
        @test length(simplified)==1 && only(simplified).ports==[2]
        for (name,text) in (
            "no_cupgrp"=>replace(template,r"CUPGRP \"Auto\" LOCAL\nID 0\nEND\n"=>"END\n"),
            "name_mismatch"=>replace(template,"POR1 CUP Auto"=>"POR1 CUP B";count=1),
            "bad_flag"=>replace(template,"CUPGRP \"Auto\" LOCAL"=>"CUPGRP \"Auto\" REMOTE"),
            "unterminated"=>replace(template,r"CUPGRP \"Auto\" LOCAL\nID 0\nEND\n"=>"CUPGRP \"Auto\" LOCAL\n"),
            "extra_por1"=>replace(template,"POR1 CUP Auto"=>"POR1 CUP Auto X";count=1),
            "stray_cupgrp"=>replace(template,"LORGN 0 1 U"=>"LORGN 0 1 U\nCUPGRP \"X\" LOCAL";count=1))
            write(path,text)
            @test_throws ArgumentError read_sonnet_project(path)
        end
        # A CUPGRP block on a non-CUP token is inert natively (verified:
        # a wall STD+CUPGRP replays bit-identically without the group),
        # so it parses — but Lite's port typing is geometric: these
        # interior edges as standard ports are ungrounded and reject at
        # the solve exactly like Lite's "Port has no ground reference".
        write(path,replace(template,"POR1 CUP Auto"=>"POR1 STD"))
        std_group=read_sonnet_project(path)
        @test isempty(sonnet_calibration_groups(std_group))
        @test_throws ArgumentError sonnet_planar_problem(std_group;
            freq=1e9,grid=(16,16),_materials=true,_details=true)
        write(path,replace(template,"GNDREF FLOAT"=>"GNDREF BOX AUTO"))
        boxref=read_sonnet_project(path)
        @test only(sonnet_calibration_groups(boxref)).groundref===:box_auto
        @test_throws ArgumentError solve_sonnet_project(boxref,1e9;mx=16,my=16)
    end
end

@testset "installed-Lite diagonal port rejection replay" begin
    # Native Lite 18.53 rejects a source port on a non-axis-aligned polygon
    # edge ("Illegal circuit configuration. Port is diagonal." — archived
    # in *.native.log). DiffMoM must reject the same geometry instead of
    # silently staircasing the terminal; the adapter rejection is parity.
    fixture=joinpath(@__DIR__,"fixtures","native_diag_port_probes")
    for name in sort([splitext(f)[1] for f in readdir(fixture) if endswith(f,".son")])
        project=read_sonnet_project(joinpath(fixture,name*".son"))
        @test_throws ArgumentError solve_sonnet_project(project,1e9;raw=true,mx=24,my=16)
    end
end

@testset "native material override coexistence policy" begin
    # solve_sonnet_project is the project-faithful path: its sheet and via
    # loss fields come from the project's own metal records, so caller
    # surface_zs/via_sigma keywords must reject rather than silently
    # shadow declared physics (two impedance sources cannot coexist on
    # the same cells). The sanctioned override route is the lower-level
    # API: sonnet_planar_problem lowers the project, then solve_planar
    # accepts the caller's field — native-derived or custom.
    fixture=joinpath(@__DIR__,"fixtures","native_arr_srvy")
    p=read_sonnet_project(joinpath(fixture,"as_1.son"));f=1e10
    @test_throws ArgumentError solve_sonnet_project(p,f;surface_zs=0.,raw=true,mx=16,my=16)
    @test_throws ArgumentError solve_sonnet_project(p,f;via_sigma=1.,raw=true,mx=16,my=16)
    model=sonnet_planar_problem(p;freq=f,grid=(20,20),_materials=true,_details=true)
    native=solve_planar(model.problem,f;surface_zs=model.sheet_zs,
        via_sigma=model.via_sigma,method=:dense_fft,mx=160,my=160)
    full=solve_sonnet_project(p,f;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
    @test native.y==full.raw.y
    # A caller-supplied field reaches the same solver: unit conductivity
    # nearly opens the bridge the project's own record drives.
    custom=solve_planar(model.problem,f;surface_zs=model.sheet_zs,
        via_sigma=1.,method=:dense_fft,mx=160,my=160)
    @test abs(custom.s[1,2])<abs(native.s[1,2])/10
end

@testset "installed-Lite port kind token is descriptive" begin
    # Lite 18.53 requires a token after POR1 but ignores its value: garbage
    # and wrong-class kinds on sheet-edge, box-wall and via-polygon ports
    # all solve silently as the geometry-determined type ("Standard port"
    # annotations in the archived logs). The structural pieces are the
    # group records, not the token: a CUPGRP block after ANY token
    # de-embeds the port, and a single-port group solves natively in Lite.
    fixture=joinpath(@__DIR__,"fixtures","native_port_kind_token")
    base=read_sonnet_project(joinpath(fixture,"pk_STD.son"))
    reference=sonnet_planar_problem(base;freq=1e9,grid=(20,20),_materials=true,_details=true)
    # Every token variant lowers to the identical port contraction.
    for name in ("pk_XYZ","pk_VIA","pk_AGI","pk_gauto","pk_all")
        model=sonnet_planar_problem(read_sonnet_project(joinpath(fixture,name*".son"));
            freq=1e9,grid=(20,20),_materials=true,_details=true)
        @test model.contraction==reference.contraction
        @test model.via_sigma==reference.via_sigma
    end
    # The shared native reference still replays inside the gate.
    ts=planar_read_touchstone(joinpath(fixture,"pk_STD_raw.s2p"))
    for frequency in (1e9,1e10)
        native=ts.s[only(findall(==(frequency),ts.frequencies))]
        actual=solve_sonnet_project(base,frequency;raw=true,grid=(20,20),mx=160,my=160,method=:dense_fft)
        @test maximum(abs,actual.s-native)<=.06
    end
    # A bare POR1 and a group-less CUP are native parse rejections.
    @test_throws ArgumentError read_sonnet_project(joinpath(fixture,"pk_BARE.son"))
    @test_throws ArgumentError read_sonnet_project(joinpath(fixture,"pk_CUP.son"))
    # CUPGRP is structural but inert on non-CUP ports (verified 18.53:
    # a wall STD+CUPGRP replays bit-identically to the group-less port,
    # and an interior non-CUP port rejects as ungrounded regardless of
    # the group). The grouped non-CUP ports parse, keep their structural
    # validation, and solve by placement — xg2_cup's XYZ token on the
    # wall edge lowers to a standard wall port exactly like pk_XYZ.
    xgrouped=read_sonnet_project(joinpath(fixture,"xg2_cup.son"))
    xmodel=sonnet_planar_problem(xgrouped;freq=1e9,grid=(20,20),
        _materials=true,_details=true)
    @test xmodel.contraction==reference.contraction
    @test isempty(xmodel.cup_standards)
    # A wall-attached CUP token degrades to the standard wall port
    # natively (g2_cup's log types both ports "Standard port"), so its
    # `-d` request is wall feed-line de-embedding — served by the DRP1
    # feed-chain adapter.
    grouped=read_sonnet_project(joinpath(fixture,"g2_cup.son"))
    @test sonnet_planar_problem(grouped;freq=1e9,grid=(20,20),
        _materials=true,_details=true).contraction==reference.contraction
    @test solve_sonnet_project(grouped,1e9;grid=(16,16)).s!=
        solve_sonnet_project(grouped,1e9;grid=(16,16),raw=true).s
    # A two-port group with only a trailing CUPGRP rejects natively too
    # (mp_cup: "Illegal co-calibrated port / Calibration group information
    # not defined") and locally as the CUP-without-group parse error.
    @test_throws ArgumentError read_sonnet_project(joinpath(fixture,"mp_cup.son"))
    # A bogus token on a via polygon rides the via path identically to
    # :via/:std (verified: v_xyz.son solves natively as a via port).
    stripline=read_sonnet_project(joinpath(@__DIR__,"..","validation","sonnet_stripline","s50.son"))
    via=SonnetPolygon(:via,0,-1,99,[.002 .003 .003 .002;.005 .005 .006 .006],
        "GND","","",["RING","NOCOVERS"])
    build_via_project(kind)=SonnetProject(stripline.source,stripline.units,
        stripline.length_scale,stripline.frequency_scale,stripline.box,
        stripline.layers,stripline.metals,stripline.top,stripline.bottom,
        vcat(stripline.polygons,[via]),
        vcat(stripline.ports,[SonnetPortSpec(kind,99,0,3,
            ["3","50","0","0","0","2.5","5.5"],SonnetRecord[])]),
        stripline.variables,stripline.components,stripline.sweeps,stripline.records)
    contractions=map(kind->
        sonnet_planar_problem(build_via_project(kind);grid=(16,16),_details=true).contraction,
        (:xyz,:agi,:via,:std))
    @test allequal(contractions)
    # Interior-terminating vias accept every token natively too: v3_via,
    # v3_std and v3_xyz (a 3-layer stack, via TOLEVEL 1) all solve as
    # "Standard port" with identical output — the earlier cover-
    # termination gate was a false rejection. The 50mil native via is
    # enlarged to stay resolved on the coarse test grid.
    v3=read_sonnet_project(joinpath(fixture,"v3_via.son"))
    big=map(v3.polygons) do poly
        poly.id==99 ? SonnetPolygon(poly.kind,poly.level,poly.material,poly.id,
            [7000. 8000. 8000. 7000.; 7000. 7000. 8000. 8000.].*v3.length_scale,
            poly.target,poly.technology,poly.technology_mode,poly.flags) : poly
    end
    inner=map(("v3_via","v3_std","v3_xyz")) do name
        project=read_sonnet_project(joinpath(fixture,name*".son"))
        variant=SonnetProject(project.source,project.units,project.length_scale,
            project.frequency_scale,project.box,project.layers,project.metals,
            project.top,project.bottom,big,project.ports,project.variables,
            project.components,project.sweeps,project.records)
        sonnet_planar_problem(variant;grid=(24,24),_materials=true,_details=true).contraction
    end
    @test allequal(inner)
end

@testset "native DRP1 reference-plane intake and gating" begin
    fixture=joinpath(@__DIR__,"fixtures","native_drp_reference_planes")
    # Grammar validation happens at intake, matching Lite's "Project File
    # Error" intake failures.
    @test_throws ArgumentError read_sonnet_project(joinpath(fixture,"d_fixabc.son"))
    @test_throws ArgumentError read_sonnet_project(joinpath(fixture,"d_bad.son"))
    @test_throws ArgumentError read_sonnet_project(joinpath(fixture,"d_drp2.son"))
    @test_throws ArgumentError read_sonnet_project(joinpath(fixture,"d_linkbare.son"))
    e=try read_sonnet_project(joinpath(fixture,"d_fixabc.son"));nothing
        catch error;error end
    @test occursin("DRP1 FIX length is not numeric",sprint(showerror,e))
    e=try read_sonnet_project(joinpath(fixture,"d_bad.son"));nothing
        catch error;error end
    @test occursin("invalid DRP1 reference plane type",sprint(showerror,e))
    e=try read_sonnet_project(joinpath(fixture,"d_drp2.son"));nothing
        catch error;error end
    @test occursin("unrepresented native geometry record DRP2",sprint(showerror,e))
    e=try read_sonnet_project(joinpath(fixture,"d_linkbare.son"));nothing
        catch error;error end
    @test occursin("POLY <id> 1 continuation",sprint(showerror,e))
    # Verified-inert records keep the problem solvable and unchanged:
    # nonpositive FIX, bare or nonpositive NONE (native output is
    # bit-identical to base).
    base=sonnet_planar_problem(read_sonnet_project(
        joinpath(fixture,"drp_base.son"));freq=2.5e9,_materials=true)
    for name in ("drp_fixm500","d_FIX_0","d_NONE","d_NONE_4")
        variant=sonnet_planar_problem(read_sonnet_project(
            joinpath(fixture,name*".son"));freq=2.5e9,_materials=true)
        @test [s.mask for s in variant.sheets]==[s.mask for s in base.sheets]
        @test length(variant.ports)==length(base.ports)
    end
    # Positive FIX below the five-half-cell floor fails with the native
    # too-short diagnostic; at or above it the record now lowers to a
    # physical feed spec the solve unrolls after the raw model.
    for name in ("drp_fix500","drp_fix600","d_f625")
        e=try sonnet_planar_problem(
                read_sonnet_project(joinpath(fixture,name*".son"));
                freq=2.5e9,_materials=true);nothing
            catch error;error end
        @test e isa ArgumentError
        @test occursin("on left box wall is too short",sprint(showerror,e))
    end
    # `feed_drp` carries each wall's active spec: FIX declares the plane
    # length, NONE keeps a zero-length marker (its calibration length does
    # not move the plane), LINK resolves its wall-to-vertex distance and
    # duplicate records keep the verified first-wins intake semantics.
    mils=2.54e-5
    for (name,mode,len) in (("d_f630",:fix,630mils),("drp_fix1000",:fix,1000mils),
            ("d_lc",:fix,1000mils),("d_dup",:fix,1000mils),
            ("drp_none500",:none,0.0),("drp_none1000",:none,0.0),
            ("d_link",:link,0.0))
        model=sonnet_planar_problem(
            read_sonnet_project(joinpath(fixture,name*".son"));
            freq=2.5e9,_materials=true,_details=true)
        spec=model.feed_drp[1]
        @test spec!==nothing
        @test spec.mode==mode
        @test spec.length≈len
        @test spec.kind==:wall
    end
end

@testset "Native CUP calibration-group solve (E2)" begin
    # Lite 18.53 archived evidence: an interior open-edge simplified
    # co-calibrated port de-embeds a pure lumped launch-junction
    # capacitance (~0.95fF on this fixture, exactly proportional to f).
    # DiffMoM measures the same junction as the port cells' in-situ
    # self-capacitance marginal — host pad present vs stripped of the
    # port cells, driven through a separate cover-contact column so the
    # column's own parasitic cancels — and removes ~1.5x Lite's value.
    # The gap is a basis-resolution residual in the A7 family; this
    # testset pins the model, the bookkeeping and the bounded residual.
    fixture=joinpath(@__DIR__,"fixtures","native_cup_junction")
    manifest=JSON.parsefile(joinpath(fixture,"manifest.json"))
    for entry in manifest["files"]
        @test bytes2hex(sha256(read(joinpath(fixture,entry["path"]))))==entry["sha256"]
    end
    project=read_sonnet_project(joinpath(fixture,"cup.son"))
    template=replace(read(joinpath(fixture,"cup.son"),String),"\r\n"=>"\n")
    # Only this unchanged baseline is reused; edited projects below are
    # solved independently. Keep small responses, with fresh arrays for
    # every reader, rather than retaining numerical factors or mutable aliases.
    responses=Dict{Tuple{Float64,Bool},NamedTuple}()
    function baseline_response(f;raw=false)
        saved=get!(responses,(Float64(f),raw)) do
            result=solve_sonnet_project(project,f;raw)
            (;s=copy(result.s),y=copy(result.y),voltage_transfer=copy(result.voltage_transfer))
        end
        return deepcopy(saved)
    end

    # Staging: the interior open-edge CUP stages a marginal junction
    # standard — one full host-pad solve plus one solve per member with
    # that member's port cells stripped; contacts are cover-return
    # columns outside every port span and clear of every launch cell's
    # neighborhood (a contact touching a launch measures column-lateral
    # coupling, not the junction capacitance).
    model=sonnet_planar_problem(project;freq=1e9,_materials=true,_details=true)
    @test length(model.cup_standards)==1
    std=only(model.cup_standards)
    @test std.indices==[1]
    @test std.standard.kind===:marginal
    @test length(std.standard.problems)==2
    @test size(std.standard.contraction,2)==1
    @test model.labels==[1,2]
    # Every marginal variant must actually strip launch cells — a minus
    # problem that kept the full mask would silently zero the removal.
    @test count(std.standard.problems[2].sheets[1].mask)<
        count(std.standard.problems[1].sheets[1].mask)

    # The removal is the launch cells' self-capacitance: pure shunt
    # susceptance, exactly proportional to frequency, bounded near the
    # measured ~9uS at 1GHz (Lite removes 5.95uS on this fixture).
    removed=f->imag((baseline_response(f;raw=true).y-baseline_response(f).y)[1,1])
    for (f,lo,hi) in ((1e6,5e-9,2e-8),(1e9,5e-6,2e-5),(1e10,5e-5,2e-4))
        @test lo<removed(f)<hi
    end
    @test removed(1e10)/removed(1e9)≈10 rtol=0.05

    # Parity against archived Lite data: calibrated output vs the
    # de-embedded (D) response, raw output vs ND. Residual stays bounded
    # and grows ~linearly with frequency (the 1.5x junction over-removal).
    # Lite's D stream also de-embeds the wall STD port's feed-line
    # discontinuity — a separate A1-family residual (~0.02 on S22 here) —
    # so the D comparison covers only the CUP-driven terms.
    function sid_rows(name)
        [parse.(Float64,split(strip(l))) for l in
            eachline(joinpath(fixture,name)) if occursin(r"^\d.*e",l)]
    end
    for (name,mode) in (("geo_response_d.sid",false),("geo_response_nd.sid",true))
        for row in sid_rows(name)
            f=row[1]
            f in (1e9,1e10) || continue
            native=ComplexF64[row[2]+row[3]im row[6]+row[7]im;
                              row[4]+row[5]im row[8]+row[9]im]
            ours=baseline_response(f;raw=mode)
            terms=name=="geo_response_d.sid" ? ((1,1),(2,1),(1,2)) :
                ((1,1),(2,1),(1,2),(2,2))
            @test maximum(abs(ours.s[i...]-native[i...]) for i in terms)<=
                (f==1e9 ? 2e-4 : 2e-3)
        end
    end

    # raw=true leaves the network untouched: no standard is applied. The
    # launch-box removal moves the wall port's reference plane, so the
    # voltage transfer shifts only at machine level.
    ours_raw=baseline_response(1e9;raw=true)
    ours_cal=baseline_response(1e9)
    @test ours_cal.y!=ours_raw.y
    @test ours_cal.voltage_transfer≈ours_raw.voltage_transfer

    mktempdir() do dir
        path=joinpath(dir,"case.son")
        # A second open-edge CUP on the other pad sharing CUPGRP g1 makes
        # one coupled group: K+1 problems, a K-by-K junction block over
        # labels [1,2], mutual terms included. The level-1 member must
        # declare ABOVE: Lite groups by the resolved cover, and its AUTO
        # would resolve to the bottom cover — a mixed-reference group
        # Lite rejects natively.
        coupled=replace(template,"SIMPLE Y\n"=>"")
        coupled=replace(coupled,"""POR1 STD
POLY 10 1
1
2 50 0 0 0 1 0.5
NUM 2"""=>"""POR1 CUP g1
GNDREF BOX ABOVE
TERMW FEED
POLY 10 1
3
2 50 0 0 0 0 0.5
CUPGRP g1
END
POR1 STD
POLY 10 1
1
3 50 0 0 0 1 0.5
NUM 2""")
        write(path,coupled)
        group_model=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,_materials=true,_details=true)
        gstd=only(group_model.cup_standards)
        @test group_model.labels==[1,2,3]
        @test gstd.indices==[1,2]
        @test gstd.standard.kind===:marginal
        @test length(gstd.standard.problems)==3
        @test size(gstd.standard.contraction,2)==2
        gcal=solve_sonnet_project(read_sonnet_project(path),1e9)
        graw=solve_sonnet_project(read_sonnet_project(path),1e9;raw=true)
        dy=gcal.y-graw.y
        @test imag(dy[1,1])<-4e-6 && imag(dy[2,2])<-4e-6
        @test imag(dy[1,2])>1e-8
        # Label 3 is the wall STD port — `-d` also removes its launch
        # discontinuity now (feed-chain adapter), so its column moves too.
        @test !iszero(dy[3,3])
        @test gcal.y≈transpose(gcal.y)

        # A coupled group whose members carry opposite rails of one
        # external label folds the member-space junction into label
        # space by rail sign: the pair removes J11+J22−2J12 at label 1
        # rather than dropping all but the last member through a
        # duplicate-index block assignment.
        diff=replace(coupled,"""POLY 10 1
3
2 50"""=>"""POLY 10 1
3
-1 50""")
        write(path,diff)
        diff_model=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,_materials=true,_details=true)
        dstd=only(diff_model.cup_standards)
        @test dstd.indices==[1,1]
        @test dstd.signs==[1,-1]
        @test dstd.standard.kind===:marginal
        dcal=solve_sonnet_project(read_sonnet_project(path),1e9)
        draw=solve_sonnet_project(read_sonnet_project(path),1e9;raw=true)
        ddy=dcal.y-draw.y
        # J11+J22−2J12 ≈ 2·9uS with the small mutual term subtracting.
        @test imag(ddy[1,1])<-1e-5
        # Column 2 is the wall STD port's label — launch de-embedded by `-d`.
        @test !iszero(ddy[2,2])
        @test diff_model.labels==[1,3]

        # Two members sharing one pad leave no contact cell outside
        # every launch's neighborhood — adjacent contacts measure
        # column-lateral coupling, not junction capacitance — so the
        # standard degenerates to the feed-column form and the sign
        # fold still applies to its member-space block. The second
        # member launches off the pad's east edge: its AUTO resolves to
        # the top cover like the first member's — the north and south
        # edges resolve below and a mixed-reference group rejects
        # natively.
        samepad=replace(template,"""POR1 STD
POLY 10 1
1
2 50 0 0 0 1 0.5
NUM 2"""=>"""POR1 CUP g1
GNDREF BOX AUTO
TERMW FEED
POLY 9 1
1
-1 50 0 0 0 0 0.5
CUPGRP g1
END
POR1 STD
POLY 10 1
1
2 50 0 0 0 1 0.5
NUM 2""")
        samepad=replace(samepad,"SIMPLE Y\n"=>"")
        write(path,samepad)
        sp_model=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,_materials=true,_details=true)
        spstd=only(sp_model.cup_standards)
        @test spstd.indices==[1,1]
        @test spstd.signs==[1,-1]
        @test spstd.standard.kind===:column
        sp_d=imag((solve_sonnet_project(read_sonnet_project(path),1e9;raw=true).y-
            solve_sonnet_project(read_sonnet_project(path),1e9).y)[1,1])
        @test sp_d>1e-5

        # Lite rejects a named group whose members resolve to different
        # covers ("mixed ground references"): the level-1 member's AUTO
        # resolves to the bottom cover while the level-0 member's
        # resolves to the top. Auto-named members instead sub-group by
        # resolved cover.
        mixed=replace(coupled,"GNDREF BOX ABOVE\nTERMW FEED"=>
            "GNDREF BOX AUTO\nTERMW FEED")
        write(path,mixed)
        @test_throws ArgumentError sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,_materials=true,_details=true)
        auto=replace(mixed,"POR1 CUP g1"=>"POR1 CUP Auto")
        auto=replace(auto,"CUPGRP g1"=>"CUPGRP \"Auto\" LOCAL")
        write(path,auto)
        auto_model=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,_materials=true,_details=true)
        @test length(auto_model.cup_standards)==2
        @test all(s->length(s.indices)==1,auto_model.cup_standards)

        # SIMPLE Y splits the same CUPGRP into independent singleton
        # standards even when the ports share a group name.
        write(path,replace(template,"""POR1 STD
POLY 10 1
1
2 50 0 0 0 1 0.5
NUM 2"""=>"""POR1 CUP g1
SIMPLE Y
GNDREF BOX AUTO
TERMW FEED
POLY 10 1
3
2 50 0 0 0 0 0.5
CUPGRP g1
END
POR1 STD
POLY 10 1
1
3 50 0 0 0 1 0.5
NUM 2"""))
        simple_model=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,_materials=true,_details=true)
        @test length(simple_model.cup_standards)==2
        @test all(s->s.standard.kind===:marginal,simple_model.cup_standards)
        @test sort!([only(s.indices) for s in simple_model.cup_standards])==[1,2]

        # A pad fully covered by port cells leaves no non-junction
        # contact: the standard degenerates to the coarser feed-column
        # measure rather than inventing a contact.
        shrunk=replace(template,"""0.3 0.45
0.45 0.45
0.45 0.55
0.3 0.55
0.3 0.45"""=>"""0.3 0.45
0.35 0.45
0.35 0.55
0.3 0.55
0.3 0.45""")
        write(path,shrunk)
        col_model=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,_materials=true,_details=true)
        @test only(col_model.cup_standards).standard.kind===:column

        # GNDREF direction selects the return cover: both forced covers
        # stage and solve, and BELOW's two-layer column removes a
        # different in-situ junction admittance than AUTO's near cover.
        for token in ("ABOVE","BELOW")
            write(path,replace(template,"GNDREF BOX AUTO"=>"GNDREF BOX $token"))
            pdir=read_sonnet_project(path)
            @test only(sonnet_planar_problem(pdir;freq=1e9,_materials=true,
                _details=true).cup_standards).standard.kind===:marginal
        end
        write(path,replace(template,"GNDREF BOX AUTO"=>"GNDREF BOX BELOW"))
        pdir=read_sonnet_project(path)
        below_d=imag((solve_sonnet_project(pdir,1e9;raw=true).y-
            solve_sonnet_project(pdir,1e9).y)[1,1])
        @test below_d>removed(1e9)

        # Unsupported forms stay loud.
        write(path,replace(template,"GNDREF BOX AUTO"=>"GNDREF FLOAT"))
        @test_throws ArgumentError sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,_materials=true,_details=true)
        write(path,replace(template,"""3
1 50 0 0 0 0 0.5"""=>"""3
0 50 0 0 0 0 0.5"""))
        @test_throws ArgumentError sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,_materials=true,_details=true)
        # A wall-edge CUP degrades to the standard wall port natively —
        # co-calibration needs an interior open edge — so the problem
        # lowers with no calibration standard, and its `-d` request is
        # wall feed-line de-embedding. The wall pad's own port block
        # drops away so the two wall launches do not share the edge's
        # connected cell run.
        wallcup=replace(template,"POLY 9 1\n3\n1 50"=>"POLY 10 1\n1\n1 50")
        wallcup=replace(wallcup,"POR1 STD\nPOLY 10 1\n1\n2 50 0 0 0 1 0.5\n"=>"")
        write(path,wallcup)
        wallmodel=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,_materials=true,_details=true)
        @test isempty(wallmodel.cup_standards)
        @test solve_sonnet_project(read_sonnet_project(path),1e9).s!=
            solve_sonnet_project(read_sonnet_project(path),1e9;raw=true).s
        # Payload accounting retains every standard problem.
        @test_throws ArgumentError sonnet_planar_problem(project;
            freq=1e9,_materials=true,_details=true,max_bytes=2^10)
        # The non-details path cannot carry the group contract.
        @test_throws ArgumentError sonnet_planar_problem(project;
            freq=1e9,_materials=true,_details=false)
    end
end

@testset "Sonnet native dielectric bricks" begin
    # Grammar verified against 18.53 examples (brickcap, chipcap): BRI
    # declares an isotropic brick material, TECHLAY BRICK binds it to a
    # named brick tech layer whose template header carries the assigned
    # metal level, BRI POLY attaches through TLAYNAM, and the host
    # dielectric record's Z-partition field (token 7) gates support.
    template="""
FTYP SONPROJ 19 ! test
VER "18.53"
HEADER
LIC x
END HEADER
DIM
ANG DEG
CAP PF
CON /OH
FREQ GHZ
IND NH
LNG MM
RES OH
END DIM
GEO
TMET "Lossless" 0 SUP 0 0 0 0
BMET "Lossless" 0 SUP 0 0 0 0
BRI "HiK" 1 10 0.001 0
BOX 3 1 1 16 16 100 0
0.1 1 1 0 0 0 0 "Air_top"
0.1 1 1 0 0 0 0 "Air_mid"
0.1 4 1 0 0 0 1 "Host"
0.1 1 1 0 0 0 0 "Air_bottom"
TECHLAY BRICK "HiKbrick" "HiK" 0 0
BRI POLY
2 0 1 N 0 1 1 1 1 0 0 0 N
END
END
LORGN 0 1 U
POR1 STD
POLY 10 1
1
1 50 0 0 0 1 0.5
NUM 3
0 5 -1 N 9 1 1 100 100 0 0 0 Y
0.3 0.45
0.45 0.45
0.45 0.55
0.3 0.55
0.3 0.45
END
1 5 -1 N 10 1 1 100 100 0 0 0 Y
0.55 0.45
1 0.45
1 0.55
0.55 0.55
0.55 0.45
END
BRI POLY
2 4 1 N 11 1 1 1 1 0 0 0 N
TLAYNAM "HiKbrick" INH
0.3 0.3
0.7 0.3
0.7 0.7
0.3 0.7
END
END GEO
OPT
MAX 100
END OPT
VARSWP
ENABLED Y
FREQ Y AN LIST 1
END
END VARSWP
"""

    mktempdir() do dir
        path=joinpath(dir,"brick.son")
        write(path,template)
        project=read_sonnet_project(path)
        problem=sonnet_planar_problem(project;freq=1e9,grid=(32,32))
        @test length(problem.bricks)==1
        brick=problem.bricks[1]
        # Metal level 2 fills the dielectric below its plane: solver
        # layer L-2 = 2 ("Host"; st.layers is bottom-up, verified on
        # brickcap/chipcap layouts — level+1 would wrongly give 3).
        @test brick.layer==2
        # epsr 10 with tand 0.001 -> 10-0.01im on the e^{+iwt} convention.
        @test brick.epsr[1]≈10-0.01im
        @test brick.epsr[1]==brick.epsr[2]==brick.epsr[3]
        @test 0<count(brick.mask)<32*32
        # Every marked cell contributes a z pulse plus the touched
        # transverse rooftop faces, so brick unknowns exceed one per
        # direction and cell.
        nb=count(DiffMoM._is_brick_kind,problem.basis.kind)
        @test nb>3count(brick.mask)
        result=solve_planar(problem,1e9)
        @test all(x->abs(x)<=1+1e-8,result.s)
        @test all(isfinite,result.s)

        # The brick carries polarization current: a contrasting run
        # without it must differ measurably.
        basis0=DiffMoM.build_planar_basis(problem.grid,problem.sheets,
            problem.ports;vias=problem.vias)
        free=PlanarProblem(problem.stack,problem.grid,problem.sheets,
            problem.ports,problem.vias,basis0)
        @test solve_planar(free,1e9).s!=result.s

        # Brick currents appear in the maps as a :brick payload.
        maps=planar_current_maps(result)
        brickmaps=filter(m->m.kind===:brick,maps)
        @test length(brickmaps)==1
        @test brickmaps[1].mask==brick.mask
        @test any(!iszero,brickmaps[1].jz)

        # zparts>1 needs the axial refinement adapter (chipcap grammar).
        write(path,replace(template,"0.1 4 1 0 0 0 1 \"Host\""=>
            "0.1 4 1 0 0 0 3 \"Host\""))
        @test_throws r"Z-partition 3 > 1" sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,grid=(32,32))

        # A zero/absent Z-partition keeps the brick inert natively: the
        # adapter refuses to guess rather than silently dropping metal.
        write(path,replace(template,"0.1 4 1 0 0 0 1 \"Host\""=>
            "0.1 4 1 0 0 0 0 \"Host\""))
        @test_throws ArgumentError sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,grid=(32,32))
        write(path,replace(template,"0.1 4 1 0 0 0 1 \"Host\""=>
            "0.1 4 1 0 0 0 \"Host\""))
        @test_throws ArgumentError sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,grid=(32,32))

        # A brick without its TLAYNAM tech-layer binding is inert
        # natively; the adapter must not silently drop it.
        write(path,replace(template,"TLAYNAM \"HiKbrick\" INH\n"=>""))
        @test_throws r"lacks a brick tech layer" sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,grid=(32,32))
        write(path,replace(template,"TLAYNAM \"HiKbrick\" INH"=>
            "TLAYNAM \"bogus\" INH"))
        @test_throws r"unknown brick tech layer" sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,grid=(32,32))
        write(path,replace(template,"TECHLAY BRICK \"HiKbrick\" \"HiK\""=>
            "TECHLAY BRICK \"HiKbrick\" \"NoSuchMat\""))
        @test_throws r"undeclared" sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,grid=(32,32))

        # Generated names Brick_<material>_<level> resolve back to the
        # declared material (chipcap grammar).
        write(path,replace(template,"TECHLAY BRICK \"HiKbrick\" \"HiK\""=>
            "TECHLAY BRICK \"HiKbrick\" \"Brick_HiK_2\""))
        generated=sonnet_planar_problem(read_sonnet_project(path);freq=1e9,grid=(32,32))
        @test generated.bricks[1].epsr==brick.epsr

        # Level disagreement between polygon and tech-layer template.
        write(path,replace(template,"BRI POLY\n2 0 1 N 0"=>
            "BRI POLY\n0 0 1 N 0"))
        @test_throws r"disagrees" sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,grid=(32,32))

        # Anisotropic BRI arity rejects rather than guessing semantics.
        write(path,replace(template,"BRI \"HiK\" 1 10 0.001 0"=>
            "BRI \"HiK\" 1 10 0.001 0 9 0.002 0"))
        @test_throws r"anisotropic" sonnet_planar_problem(
            read_sonnet_project(path);freq=1e9,grid=(32,32))

        # Bricks have no parity transform: UFFT and dense_fft are
        # explicit about it instead of mis-mapping elements.
        write(path,template)
        problem=sonnet_planar_problem(read_sonnet_project(path);freq=1e9,grid=(32,32))
        @test_throws r"brick parity transform" solve_planar(problem,1e9;
            method=:ufft)

        # Zero-contrast and empty bricks are caught at problem build.
        contrast=brick_level(2,32,32,4+0im)
        contrast.mask[5:8,5:8].=true
        @test_throws r"contrast" build_planar_problem(problem.stack,
            problem.grid,problem.sheets,problem.ports;bricks=[contrast])
        emptybrick=brick_level(2,32,32,9+0im)
        @test_throws r"marks no cells" build_planar_problem(problem.stack,
            problem.grid,problem.sheets,problem.ports;bricks=[emptybrick])
    end
end

@testset "Sonnet native BAR vias" begin
    # Lite's "Simplifying Bar Vias" rule merges adjacent BAR polygons into
    # one wider via meshed with the Vertices fill.  Two pads share the
    # x=0.5 port edge (same geometry as the brick fixture); BAR vias sit
    # under the driven pad on the level-0 to level-1 span.
    function bar_source(fill,bars;material=-1,metal="")
        head="""
FTYP SONPROJ 19 ! bar tests
VER "18.53"
HEADER
LIC x
END HEADER
DIM
ANG DEG
CAP PF
CON /OH
FREQ GHZ
IND NH
LNG MM
RES OH
END DIM
GEO
TMET "Lossless" 0 SUP 0 0 0 0
BMET "Lossless" 0 SUP 0 0 0 0
$(isempty(metal) ? "" : metal*"\n")BOX 3 1 1 16 16 100 0
0.1 1 1 0 0 0 0 "Air_top"
0.1 1 1 0 0 0 0 "Air_mid"
0.1 4 1 0 0 0 1 "Host"
0.1 1 1 0 0 0 0 "Air_bot"
LORGN 0 1 U
POR1 STD
POLY 10 1
1
1 50 0 0 0 1 0.5
NUM $(2+length(bars))
0 5 -1 N 9 1 1 100 100 0 0 0 Y
0.3 0.45
0.45 0.45
0.45 0.55
0.3 0.55
0.3 0.45
END
1 5 -1 N 10 1 1 100 100 0 0 0 Y
0.55 0.45
1 0.45
1 0.55
0.55 0.55
0.55 0.45
END
"""
        body=""
        for (k,(x0,x1,y0,y1)) in enumerate(bars)
            body*="""
VIA POLYGON
0 5 $(material) N $(10+k) 1 1 100 100 0 0 0 Y
TOLEVEL 1 $(fill) NOCOVERS
$(x0) $(y0)
$(x1) $(y0)
$(x1) $(y1)
$(x0) $(y1)
$(x0) $(y0)
END
"""
        end
        return head*body*"""
END GEO
OPT
MAX 100
END OPT
"""
    end
    # Two 2x2-cell strips abutting at x=0.375: one 4x2 merged region with
    # four outline-corner cells, versus eight cells as two vertex-filled
    # polygons.
    bars=[(0.3125,0.375,0.46875,0.53125),(0.375,0.4375,0.46875,0.53125)]
    mktempdir() do dir
        path=joinpath(dir,"bar.son")
        write(path,bar_source("BAR",bars))
        model=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_details=true)
        @test length(model.problem.vias)==1
        @test count(model.problem.vias[1].uni)==4
        @test model.via_sigma==[Inf]
        result=solve_planar(model.problem,1e9)
        @test all(isfinite,result.s)
        @test all(x->abs(x)<=1+1e-8,result.s)
        write(path,bar_source("VERTICES",bars))
        vertex=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_details=true)
        @test count(vertex.problem.vias[1].uni)==8
        @test solve_planar(vertex.problem,1e9).s!=result.s

        # A lone BAR via is its own Vertices fill: identical mesh, sigma
        # and S-parameters to the VERTICES declaration.
        write(path,bar_source("BAR",bars[1:1]))
        barmodel=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_details=true)
        write(path,bar_source("VERTICES",bars[1:1]))
        vertmodel=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_details=true)
        @test barmodel.problem.vias[1].uni==vertmodel.problem.vias[1].uni
        @test barmodel.via_sigma==vertmodel.via_sigma
        @test solve_planar(barmodel.problem,1e9).s==
            solve_planar(vertmodel.problem,1e9).s

        # Disjoint bars never merge: each keeps its own polygon's fill
        # (the second bar rasterizes to a 1x2 strip with two corners).
        write(path,bar_source("BAR",
            [(0.3125,0.375,0.46875,0.53125),(0.55,0.6,0.46875,0.53125)]))
        far=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_details=true)
        @test count(far.problem.vias[1].uni)==6

        # An L-shaped union keeps the merged outline's six corners.
        write(path,bar_source("BAR",
            [(0.3125,0.375,0.46875,0.53125),(0.34375,0.4375,0.46875,0.5625)]))
        ell=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_details=true)
        @test count(ell.problem.vias[1].uni)==6

        # Constant-RPV bars preserve the parallel DC conductance of the
        # group: two 2-ohm bars merge to one column group totaling 1 ohm.
        rpvmetal="""MET "Via" 1 VOL 2 0 RPV"""
        write(path,bar_source("BAR",bars;material=0,metal=rpvmetal))
        rmodel=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_materials=true,_details=true)
        @test sum(real(rmodel.problem.stack.layers[v.layer].thickness)/
            (rmodel.via_sigma[i]*count(v.uni)*rmodel.problem.grid.dx*rmodel.problem.grid.dy)
            for (i,v) in enumerate(rmodel.problem.vias))≈1
        write(path,bar_source("BAR",bars[1:1];material=0,metal=rpvmetal))
        rmodel1=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_materials=true,_details=true)
        @test sum(real(rmodel1.problem.stack.layers[v.layer].thickness)/
            (rmodel1.via_sigma[i]*count(v.uni)*rmodel1.problem.grid.dx*rmodel1.problem.grid.dy)
            for (i,v) in enumerate(rmodel1.problem.vias))≈2
        # The merged sigma is the parallel sum of the member conductances.
        @test rmodel.via_sigma[1]≈2rmodel1.via_sigma[1]

        # A single loss model evaluates on the merged union outline for
        # non-RPV metals (VOL SOLID and ARR both take this path).
        for (metal,check) in (
                ("""MET "Via" 1 VOL 5.8e7 SOLID 0""",
                    (s2,s1)->isapprox(s2,2s1;rtol=1e-10)),
                ("""MET "Array" 1 ARR 58e6 16""", (s2,s1)->s2!=s1))
            write(path,bar_source("BAR",bars;material=0,metal=metal))
            merged=sonnet_planar_problem(read_sonnet_project(path);
                freq=1e9,grid=(32,32),_materials=true,_details=true)
            write(path,bar_source("BAR",bars[1:1];material=0,metal=metal))
            single=sonnet_planar_problem(read_sonnet_project(path);
                freq=1e9,grid=(32,32),_materials=true,_details=true)
            @test check(merged.via_sigma[1],single.via_sigma[1])
            @test all(isfinite,solve_planar(merged.problem,1e9).s)
        end

        # Bars on different spans cannot form one column: a level-0->1
        # member and a level-0->2 member keep their own fills.  On the
        # shared layer their two four-cell fills land on the common
        # ideal-conductor level (8 cells, not the merged 4); the longer
        # member also emits its second-layer column.
        spanned=replace(bar_source("BAR",bars),
            "TOLEVEL 1 BAR"=>"TOLEVEL 2 BAR",count=1)
        write(path,spanned)
        span=sonnet_planar_problem(read_sonnet_project(path);
            freq=1e9,grid=(32,32),_details=true)
        @test sort([count(v.uni) for v in span.problem.vias])==[4,8]
    end
end
