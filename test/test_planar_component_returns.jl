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
    left=SonnetPolygon(:sheet,0,-1,1,[0.0 0.0 .0025 .0025;.0015 .0025 .0025 .0015],"","",String[])
    right=SonnetPolygon(:sheet,0,-1,2,[.0035 .006 .006 .0035;.0015 .0015 .0025 .0025],"","",String[])
    ports=[SonnetPortSpec(:box,1,0,1,["1","50","0","0","0","0","2"],SonnetRecord[]),
        SonnetPortSpec(:box,2,1,2,["2","50","0","0","0","6","2"],SonnetRecord[])]
    component=[records(["TYPE","IDEAL",kind,value]),records(["GNDREF","AUTO"]),
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
        [.0005 .001 .001 .0005;.0015 .0015 .002 .002],"GND","",["NOCOVERS"]))
    lossy=sonnet_component_model(resistive,1e8)
    @test isfinite(first(lossy.via_sigma))
    @test all(isinf,lossy.via_sigma[2:end])
    loaded=solve_sonnet_project(resistive,1e8;raw=true,mx=36,my=24)
    direct=solve_planar(lossy.problem,1e8;surface_zs=lossy.sheet_zs,via_sigma=lossy.via_sigma,mx=36,my=24)
    @test loaded.raw.y≈direct.y rtol=1e-12
    @test norm(loaded.s-result.s)>.01
    @test_throws ArgumentError solve_sonnet_project(resistive,1e8;raw=true,via_sigma=Inf)
end
