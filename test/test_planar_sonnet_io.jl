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
    for suffix in (["UNKNOWN"],["RSVY","RSVY"],["ANISO"])
        bad=deepcopy(p);resize!(bad.layers[2],8);append!(bad.layers[2],suffix)
        @test_throws ArgumentError sonnet_planar_problem(bad;freq=1e9)
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
    # and SFC Rdc1/Xdc1e-5/Rrf.2. This is a schema oracle, not an
    # accepted Surface-via EM or nonzero kinetic-inductance solve.
    @test p.metals[1]==["Planar","1","SUP","1","2","3","4"]
    @test p.metals[2]==["Bulk","1","SFC","1","1e-5",".2"]
    f=1e9
    rf=(1+im)*2sqrt(f)
    @test sonnet_metal_zs(p,p.metals[1],f)≈
        rf/tanh(rf)+im*(3+2pi*f*4e-12)
    @test_throws ArgumentError sonnet_metal_zs(p,p.metals[2],f)
    @test_throws ArgumentError sonnet_planar_problem(p;grid=(8,8),freq=f,
        _materials=true)
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
    @test_throws ArgumentError solve_sonnet_project(p,15e9;grid=(8,8))
    @test sonnet_metal_zs(p,["Sheet","1","RES","2"],1e9)==2+0im
    @test sonnet_metal_zs(p,["Sense","1","SEN","1000000"],1e9)==im*1e6
    @test sonnet_metal_zs(p,["PEC","1","SUP","0","0","0","0"],1e9)==0im
    space=["Free Space","0","FREESPACE","376.7303136","0","0","0"]
    termination=DiffMoM._sonnet_termination(p,space,1e9,Dict{String,Float64}())
    @test termination.kind==DiffMoM.TERM_SURFACE
    @test termination.zs==376.7303136+0im
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
            "GND","",["RING","NOCOVERS"])
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
            via.technology,via.flags)
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
        unsupported=SonnetProject(rpv.source,rpv.units,rpv.length_scale,rpv.frequency_scale,
            rpv.box,rpv.layers,[["Array","1","ARR","58e6","16"]],
            rpv.top,rpv.bottom,rpv.polygons,rpv.ports,rpv.variables,rpv.components,rpv.sweeps,rpv.records)
        @test_throws ArgumentError sonnet_planar_problem(unsupported;grid=(16,16),_materials=true)
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
        left=SonnetPolygon(:sheet,0,-1,9,leftv,"","",String[])
        right=SonnetPolygon(:sheet,0,-1,10,rightv,"","",String[])
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
        write(path,replace(source,"FTYP SONPROJ"=>"FTYP SONNETPRJ"))
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
