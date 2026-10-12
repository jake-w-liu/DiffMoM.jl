using Test,LinearAlgebra,DiffMoM

function _return_test_line(;split_layer=false)
    grid=CellGrid(6e-3,4e-3,12,8)
    layers=split_layer ? [PlanarLayer(1,1,.25e-3),PlanarLayer(1,1,.25e-3),PlanarLayer(1,1,.5e-3)] :
        [PlanarLayer(1,1,.5e-3),PlanarLayer(1,1,.5e-3)]
    stack=PlanarStackup(layers,TERM_GND,TERM_GND,grid.a,grid.b)
    sheet=sheet_level(split_layer ? 2 : 1,12,8)
    sheet.mask[2:11,4:5].=true
    terminals=[PlanarPort(1,:terminal_x,1,4:5,50;metal_side=:positive),
        PlanarPort(1,:terminal_x,11,4:5,50;metal_side=:negative)]
    return planar_terminal_returns(stack,grid,[sheet],terminals;ground_direction=:below)
end

@testset "physical terminal returns: independent DC resistance and inductance" begin
    model=_return_test_line()
    @test length(model.problem.ports)==4
    @test all(p->p.wall===:via,model.problem.ports)
    @test all(path->path.direction===:below,model.paths)
    @test model.paths[1].length≈.5e-3
    @test_throws ArgumentError planar_terminal_returns(model.problem,
        [model.paths[1].terminal];max_bytes=128)
    @test_throws ArgumentError planar_terminal_returns(model.problem,
        [model.paths[1].terminal];ground_direction=:sideways)
    @test count(model.problem.sheets[1].mask)==20
    inductances=Float64[]
    for f in (1e6,1e7,1e8)
        raw=solve_planar(model.problem,f;mx=48,my=32,surface_zs=.2)
        Y=transpose(model.contraction)*raw.y*model.contraction
        current=Y*ComplexF64[.5,-.5]
        Z=2/(current[1]-current[2])
        # Uniform current over the two contact-cell footprints ramps from
        # zero to I inside each cell: each dissipates R_sheet*dx/(3width).
        # Eight interior cells carry I, giving a geometry-only DC oracle.
        expected=.2*(8+2/3)*model.problem.grid.dx/(2model.problem.grid.dy)
        @test real(Z)≈expected rtol=3e-5
        @test imag(Z)>0
        push!(inductances,imag(Z)/(2pi*f))
        @test opnorm(planar_y_to_s(Y,model.z0))<=1+1e-10
    end
    @test maximum(inductances)/minimum(inductances)<1.00002
    # Layer subdivision preserves physical return distance and voltage
    # normalization; each series section is driven at half total voltage.
    split=_return_test_line(split_layer=true)
    @test length(split.problem.ports)==8
    @test all(v->v==0 || v==.5,split.contraction)
    a=solve_planar(model.problem,1e7;mx=48,my=32,surface_zs=.2)
    b=solve_planar(split.problem,1e7;mx=48,my=32,surface_zs=.2)
    @test transpose(split.contraction)*b.y*split.contraction≈
        transpose(model.contraction)*a.y*model.contraction rtol=1e-7
    contracted=planar_contract_ports(b,split.contraction;z0=[50,75])
    @test contracted.currents≈b.currents*split.contraction
    @test contracted.y≈transpose(split.contraction)*b.y*split.contraction
    @test contracted.s≈planar_y_to_s(contracted.y,[50,75])
    @test contracted.raw===b
    waves=ComplexF64[1,.2im]
    volts=sqrt.([50,75]).*(waves+contracted.s*waves)
    maps=planar_current_maps(contracted;incident_waves=waves)
    direct=planar_current_maps(b;voltages=split.contraction*volts)
    @test maps[1].jx≈direct[1].jx
    @test maps[end].jz≈direct[end].jz
    @test_throws ArgumentError planar_contract_ports(b,zeros(size(split.contraction));z0=50)
    @test_throws ArgumentError planar_contract_ports(b,split.contraction;max_bytes=128)
end

function _synthetic_smd_project(kind="RES",value="10")
    records(tokens)=SonnetRecord(1,String.(tokens))
    left=SonnetPolygon(:sheet,0,-1,1,[0.0 0.0 .0025 .0025;.0015 .0025 .0025 .0015],"","","",String[])
    right=SonnetPolygon(:sheet,0,-1,2,[.0035 .006 .006 .0035;.0015 .0015 .0025 .0025],"","","",String[])
    ports=[SonnetPortSpec(:box,1,0,1,["1","50","0","0","0","0","2"],SonnetRecord[]),
        SonnetPortSpec(:box,2,1,2,["2","50","0","0","0","6","2"],SonnetRecord[])]
    component=[records(["TYPE","IDEAL",kind,value]),records(["GNDREF","BOX","AUTO"]),
        records(["TERMW","FEED"]),records(["SMDP","0","2.5","2","L","3","1"]),
        records(["SMDP","0","3.5","2","R","4","2"])]
    return SonnetProject("synthetic_smd.son",Dict("LNG"=>"MM","CAP"=>"PF","IND"=>"NH"),.001,1e9,
        ["1","6","4","24","16","20","0"],[[".5","1","1","0","0","0"],
        [".5","1","1","0","0","0"]],Vector{String}[],["PEC","0","RES","0"],
        ["PEC","0","RES","0"],[left,right],ports,Dict{String,String}(),[component],SonnetRecord[],SonnetRecord[])
end

@testset "native components: physical pin loading and retained currents" begin
    project=_synthetic_smd_project()
    model=sonnet_component_model(project,1e8)
    @test model.labels==[1,2,3,4]
    @test length(model.problem.ports)==6
    @test model.pins[1][1].terminal.cells==4:5
    @test model.circuit.elements[1].r==10
    @test length(model.via_sigma)==length(model.problem.vias)
    @test all(isinf,model.via_sigma)
    @test all(path->path.direction===:below,model.reference_paths)
    result=solve_sonnet_project(project,1e8;raw=true,mx=36,my=24)
    @test abs(result.s[2,1])>.8
    @test result.s≈transpose(result.s) rtol=1e-10
    @test opnorm(result.s)<=1+1e-9
    maps=sonnet_component_current_maps(result;incident_waves=ComplexF64[1,0])
    direct=planar_current_maps(result.raw;voltages=result.contraction*result.circuit.voltages[:,1])
    @test maps[1].jx≈direct[1].jx
    @test all(isfinite,maps[end].jz)
    # Supplied pin-group lead calibration retains the raw current voltage
    # transfer, including mutual series terms.
    Zg=ComplexF64[0.02im 0.003im;0.003im 0.03im]
    group=[Matrix{ComplexF64}(I,2,2) Zg;zeros(2,2) Matrix{ComplexF64}(I,2,2)]
    calibrated=solve_sonnet_project(project,1e8;raw=true,mx=36,my=24,pin_calibration=group)
    c_maps=sonnet_component_current_maps(calibrated;incident_waves=ComplexF64[1,0])
    raw_voltage=calibrated.contraction*(calibrated.gap_transfer*calibrated.circuit.voltages[:,1])
    direct_c=planar_current_maps(calibrated.raw;voltages=raw_voltage)
    @test c_maps[1].jx≈direct_c[1].jx
    @test calibrated.gap_transfer!=Matrix{ComplexF64}(I,4,4)
    for (kind,value,field,expected) in (("CAP","5.6",:c,5.6e-12),("IND","3.3",:l,3.3e-9))
        component=sonnet_component_model(_synthetic_smd_project(kind,value),1e8)
        @test getfield(component.circuit.elements[1],field)≈expected
    end
    vendor=_synthetic_smd_project()
    vendor.components[1][1]=SonnetRecord(1,["TYPE","SPARAM","1"])
    @test_throws ArgumentError sonnet_component_model(vendor,1e8)
    supplied=sonnet_component_model(vendor,1e8;
        component_response=(p,c,f)->(response=ComplexF64[0 1;1 0],format=:s,z0=50))
    @test length(supplied.circuit.elements)==1
    @test_throws ArgumentError solve_sonnet_project(project,1e8;raw=true,max_bytes=128)
    resistive=_synthetic_smd_project()
    push!(resistive.metals,["axial_resistor","0","VOL","25","0","RPV"])
    push!(resistive.polygons,SonnetPolygon(:via,0,0,3,
        [.0005 .001 .001 .0005;.0015 .0015 .002 .002],"GND","","",["NOCOVERS"]))
    lossy=sonnet_component_model(resistive,1e8)
    @test isfinite(first(lossy.via_sigma))
    @test all(isinf,lossy.via_sigma[2:end])
    loaded=solve_sonnet_project(resistive,1e8;raw=true,mx=36,my=24)
    direct=solve_planar(lossy.problem,1e8;surface_zs=lossy.sheet_zs,via_sigma=lossy.via_sigma,mx=36,my=24)
    @test loaded.raw.y≈direct.y rtol=1e-12
    @test norm(loaded.s-result.s)>.01
    @test_throws ArgumentError solve_sonnet_project(resistive,1e8;raw=true,via_sigma=Inf)
end

@testset "native component terminal widths resolve FEED/CELL/CUST" begin
    cells(mode)=begin
        p=_synthetic_smd_project();p.components[1][3]=SonnetRecord(1,["TERMW",split(mode)...])
        [pin.terminal.cells for pin in sonnet_component_model(p,1e8).pins[1]]
    end
    @test cells("FEED")==[4:5,4:5]
    @test cells("CELL")==[5:5,5:5]
    @test cells("CUST 0.4")==[4:5,4:5]
    @test cells("CUST 0.25")==[4:5,4:5]
    # A TERMW inside a pin's EXSMDP block overrides the component default
    # for that pin alone.
    project=_synthetic_smd_project();component=project.components[1]
    splice!(component,4:4,[component[4],SonnetRecord(1,["EXSMDP"]),
        SonnetRecord(1,["TERMW","CELL"]),SonnetRecord(1,["END","EXSMDP"])])
    @test [pin.terminal.cells for pin in sonnet_component_model(project,1e8).pins[1]]==[5:5,4:5]
    for (tokens,message) in ((["TERMW","CUST","9"],"exceeds its pad edge"),
            (["TERMW","CUST","-1"],"must be positive"),
            (["TERMW","ODD"],"unsupported SMD terminal width"),
            (["TERMW"],"requires a width mode"))
        bad=_synthetic_smd_project();bad.components[1][3]=SonnetRecord(1,tokens)
        e=try sonnet_component_model(bad,1e8);nothing catch error;error end
        @test e isa ArgumentError && occursin(message,sprint(showerror,e))
    end
    # A bare EXSMDP TERMW with no preceding SMDP pin and malformed short
    # SMDP records fail with controlled diagnostics, never BoundsError.
    orphan=_synthetic_smd_project();splice!(orphan.components[1],3:3,
        [SonnetRecord(1,["EXSMDP"]),SonnetRecord(1,["TERMW","CELL"]),
         SonnetRecord(1,["END","EXSMDP"])])
    @test_throws ArgumentError sonnet_component_model(orphan,1e8)
    short=_synthetic_smd_project();short.components[1][4]=SonnetRecord(1,["SMDP","0","2.5"])
    @test_throws ArgumentError sonnet_component_model(short,1e8)
    # Unsupported physical component records reject instead of dropping.
    for tag in ("DRP1","DRP2","REFPLANE","TWTYPE","TWVALUE")
        bad=_synthetic_smd_project();push!(bad.components[1],SonnetRecord(1,[tag,"1"]))
        @test_throws ArgumentError sonnet_component_model(bad,1e8)
    end
    unknown=_synthetic_smd_project();push!(unknown.components[1],SonnetRecord(1,["BOGUSKEY","1"]))
    @test_throws ArgumentError sonnet_component_model(unknown,1e8)
    # Both modes still solve through the full native path.
    a=solve_sonnet_project(_synthetic_smd_project(),1e8;raw=true,mx=36,my=24)
    cell=_synthetic_smd_project();cell.components[1][3]=SonnetRecord(1,["TERMW","CELL"])
    b=solve_sonnet_project(cell,1e8;raw=true,mx=36,my=24)
    @test a.s!=b.s && all(isfinite,b.s)
end

@testset "native grounded component pins short their model ports" begin
    ground=DiffMoM._sonnet_ground_network
    # Y blocks drop the grounded port's row/column (V=0) directly.
    @test ground(ComplexF64[2 -1 0;-1 2 -1;0 -1 2],:y,[false,true,false],50)==
        ComplexF64[2 0;0 2]
    # Z blocks eliminate the shorted port by a Schur complement:
    # V_u=(Z_uu−Z_ug·Z_gg⁻¹·Z_gu)I_u.
    Z=[3.0+0im 1 2;1 4 1;2 1 5]
    @test ground(Z,:z,[false,true,false],50)≈
        [3-1/4 2-1/4;2-1/4 5-1/4]
    # S blocks terminate the port with its power-wave short. A perfect
    # thru grounded at port 2 reflects with Γ=−z0*/z0: −1 for real
    # references and a unit-magnitude phase term for complex ones.
    thru=ComplexF64[0 1;1 0]
    @test ground(thru,:s,[false,true],50)≈fill(-1,1,1)
    z0c=50+10im
    @test ground(thru,:s,[false,true],z0c)≈fill(-conj(z0c)/z0c,1,1)
    # Sequential per-port elimination agrees with the simultaneous
    # multi-port reduction S_uu+S_ugΓ(I−S_ggΓ)⁻¹S_gu.
    S=ComplexF64[.1 .2 .3;.2 .05 .1;.3 .1 .2]
    g=[true,false,true]
    direct=S[.!g,.!g]-S[.!g,g]*
        ((Matrix{ComplexF64}(I,2,2)+S[g,g])\S[g,.!g])
    @test ground(S,:s,g,50)≈direct rtol=1e-12
    # An already-shorted port makes the termination singular.
    @test_throws ArgumentError ground(ComplexF64[0 1;1 -1],:s,[false,true],50)
    # Callable responses wrap the provider; an all-grounded model reduces
    # to the empty network.
    wrapped=ground(f->thru,:s,[false,true],50)
    @test wrapped(1e9)≈fill(-1,1,1)
    @test size(ground(ComplexF64[.1 .2;.2 .1],:s,[true,true],50))==(0,0)
    @test ground(thru,:s,[false,false],50)===thru
    # A grounded SMD pin takes circuit node 0: the ideal resistor becomes
    # a shunt to box ground and the pin's embedding row stays zero.
    project=_synthetic_smd_project()
    project.components[1][5]=SonnetRecord(1,["SMDP","0","3.5","2","R","0","2"])
    model=sonnet_component_model(project,1e8)
    @test model.labels==[1,2,3]
    @test model.circuit.elements[1].terminals==[(3,0)]
    loaded=solve_sonnet_project(project,1e8;raw=true,mx=36,my=24)
    baseline=solve_sonnet_project(_synthetic_smd_project(),1e8;raw=true,mx=36,my=24)
    @test loaded.port_numbers==[1,2]
    @test loaded.s≈transpose(loaded.s) rtol=1e-10
    @test opnorm(loaded.s)<=1+1e-9
    @test loaded.s!=baseline.s
    # A grounded pin on a supplied SPARAM model reduces its network to the
    # ungrounded ports.
    vendor=_synthetic_smd_project()
    vendor.components[1][1]=SonnetRecord(1,["TYPE","SPARAM","1"])
    vendor.components[1][5]=SonnetRecord(1,["SMDP","0","3.5","2","R","0","2"])
    reduced=sonnet_component_model(vendor,1e8;
        component_response=(p,c,f)->(response=ComplexF64[0 1;1 0],format=:s,z0=50))
    @test size(reduced.circuit.elements[1].response)==(1,1)
    # Both pins grounded is inert: no branch is added and the pads remain
    # clamped terminals.
    inert=_synthetic_smd_project()
    inert.components[1][4]=SonnetRecord(1,["SMDP","0","2.5","2","L","0","1"])
    inert.components[1][5]=SonnetRecord(1,["SMDP","0","3.5","2","R","0","2"])
    inert_model=sonnet_component_model(inert,1e8)
    @test inert_model.labels==[1,2]
    @test isempty(inert_model.circuit.elements)
end
