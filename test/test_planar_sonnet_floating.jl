using DiffMoM,Test,LinearAlgebra

function _native_floating_pad_project(;cup=false)
    r(t)=SonnetRecord(1,String.(t))
    left=SonnetPolygon(:sheet,0,-1,1,
        [.00025 .0004375 .0004375 .00025 .00025;.0004375 .0004375 .0005625 .0005625 .0004375],"","",String[])
    right=SonnetPolygon(:sheet,0,-1,2,
        [.0005625 .00075 .00075 .0005625 .0005625;.0004375 .0004375 .0005625 .0005625 .0004375],"","",String[])
    pins=[r(["TYPE","NONE"]),r(["GNDREF","FLOAT"]),r(["TERMW","FEED"]),
        r(["SMDP","0",".4375",".5","L","1","1"]),r(["SMDP","0",".5625",".5","R","-1","2"])]
    ports=SonnetPortSpec[]
    if cup
        for (poly,edge,num,x) in ((1,1,1,".4375"),(2,3,-1,".5625"))
            records=[r(["POR1","CUP","Auto"]),r(["GNDREF","FLOAT"]),r(["TERMW","FEED"])]
            push!(ports,SonnetPortSpec(:cup,poly,edge,num,[string(num),"50","0","0","0",x,".5"],records))
        end
    end
    return SonnetProject("floating_contract.son",Dict("LNG"=>"MM","CAP"=>"PF","IND"=>"NH"),.001,1e9,
        ["1","1","1","32","32","20","0"],[[".5","1","1","0","0","0"],[".5","1","1","0","0","0"]],
        Vector{String}[],["PEC","0","RES","0"],["PEC","0","RES","0"],[left,right],ports,
        Dict{String,String}(),cup ? Vector{SonnetRecord}[] : [pins],SonnetRecord[],SonnetRecord[])
end

@testset "native paired FLOAT preserves physical differential source" begin
    smd=_native_floating_pad_project();cup=_native_floating_pad_project(;cup=true)
    a=solve_sonnet_project(smd,1e9;raw=true,mx=32,my=32)
    b=solve_sonnet_project(cup,1e9;raw=true,mx=32,my=32)
    @test a isa SonnetFloatingResult && b isa SonnetFloatingResult
    @test a.port_numbers==b.port_numbers==[1]
    @test a.y≈b.y rtol=1e-12
    @test a.s≈b.s rtol=1e-12
    @test a.source_incidence==[1.;-1.;;]
    @test a.circuit.gauge_nodes==[1]
    @test isempty(a.em.problem.vias) && isempty(b.em.problem.vias)
    @test only(a.bridges).polarity==-1
    @test Set(only(a.bridges).cells)==Set([(i,j) for i in 8:9,j in 8:9])
    @test opnorm(a.s)<=1+1e-10
    @test a.em.y≈transpose(a.em.y) rtol=1e-12
    # Independent physical geometry/current source, with no native parser
    # or component adapter in its construction.
    g=CellGrid(.001,.001,16,16);sh=sheet_level(1,16,16)
    sh.mask[5:7,8:9].=true;sh.mask[10:12,8:9].=true
    stack=PlanarStackup([PlanarLayer(1,1,.0005),PlanarLayer(1,1,.0005)],TERM_GND,TERM_GND,g.a,g.b)
    bare=PlanarProblem(stack,g,[sh],PlanarPort[],build_planar_basis(g,[sh],PlanarPort[]))
    physical=planar_floating_bridge(bare,PlanarPort(1,:terminal_x,7,8:9,50;metal_side=:negative),
        PlanarPort(1,:terminal_x,9,8:9,50;metal_side=:positive))
    oracle=solve_planar(physical.problem,1e9;mx=32,my=32)
    @test a.y≈oracle.y rtol=1e-12
    @test planar_current_maps(a;incident_waves=[1.])[1].jx≈
        planar_current_maps(a.em;voltages=transpose(a.source_incidence)*a.circuit.voltages[:,1])[1].jx rtol=1e-12
    u=solve_sonnet_project(cup,1e9;raw=true,method=:ufft,mx=32,my=32,memory=80,rtol=1e-10)
    @test u.y≈a.y rtol=1e-9
    model=sonnet_floating_model(cup,1e9)
    @test model.source_terminals==[(1,2)]
    @test model.external_labels==[1] && model.labels==[1]
    @test all(iszero,model.sheet_zs[1])
    @test_throws ArgumentError sonnet_floating_model(cup,1e9;max_bytes=1)
    @test_throws ArgumentError solve_sonnet_project(smd,1e9;raw=true,max_bytes=1)
    @test_throws ArgumentError planar_current_maps(a;max_bytes=1)
    bad=_native_floating_pad_project();bad.components[1][5]=SonnetRecord(1,["SMDP","0",".5625",".5","R","2","2"])
    @test_throws ArgumentError sonnet_floating_model(bad,1e9)
    width=_native_floating_pad_project();width.components[1][3]=SonnetRecord(1,["TERMW","CELL"])
    @test_throws ArgumentError sonnet_floating_model(width,1e9)
    @test_throws ArgumentError sonnet_floating_model(cup,1e9;bridge_zs=-1)
    @test_throws ArgumentError sonnet_floating_model(cup,1e9;bridge_zs=big"1e400")
    unknown=_native_floating_pad_project();push!(unknown.components[1],SonnetRecord(1,["UNKNOWN_PHYSICS","1"]))
    @test_throws ArgumentError sonnet_floating_model(unknown,1e9)
    group=Matrix{ComplexF64}(I,2,2)
    calibrated=solve_sonnet_project(cup,1e9;raw=true,pin_calibration=group,mx=32,my=32)
    @test calibrated.y≈a.y rtol=1e-12
    @test planar_current_maps(calibrated)[1].jx≈planar_current_maps(a)[1].jx rtol=1e-12
end

@testset "native FLOAT loading and mixed reference modes" begin
    p=_synthetic_smd_project()
    p.components[1][2]=SonnetRecord(1,["GNDREF","FLOAT"])
    p.components[1][4]=SonnetRecord(1,["SMDP","0","2.5","2","L","3","1"])
    p.components[1][5]=SonnetRecord(1,["SMDP","0","3.5","2","R","-3","2"])
    model=sonnet_component_model(p,1e8)
    @test model.labels==[1,2,3]
    @test model.external_labels==[1,2]
    @test model.source_terminals==[(1,0),(2,0),(3,4)]
    @test isempty(model.problem.vias)
    loaded=solve_sonnet_project(p,1e8;raw=true,mx=36,my=24)
    @test loaded.circuit.gauge_nodes==[3]
    @test loaded.s≈transpose(loaded.s) rtol=1e-10
    @test opnorm(loaded.s)<=1+1e-10
    @test abs(loaded.s[2,1])>.8
    # Native ideal resistor has the independent circuit value 10 ohms;
    # no globally grounded approximation or invented common mode is used.
    direct=PlanarCircuit(4,[(1,0),(2,0)])
    circuit_add_em!(direct,model.source_terminals,loaded.em)
    circuit_add_rlc!(direct,3,4;r=10.)
    oracle=solve_planar_circuit(direct,1e8;floating_gauge=:auto)
    @test oracle.s≈loaded.s rtol=1e-12
    @test planar_current_maps(loaded;incident_waves=[1.,0.])[1].jx≈
        planar_current_maps(loaded.em;voltages=transpose(loaded.source_incidence)*loaded.circuit.voltages[:,1])[1].jx rtol=1e-12
    # Asymmetric native negative wall sources require their common-mode
    # potential to be eliminated jointly with the floating component source.
    p.ports[2]=SonnetPortSpec(:box,2,1,-1,["-1","50","0","0","0","6","2"],SonnetRecord[])
    balanced=solve_sonnet_project(p,1e8;raw=true,mx=36,my=24)
    @test balanced.port_numbers==[1]
    @test size(balanced.em.voltage_transfer)==(3,2)
    @test all(isfinite,balanced.s)
    @test opnorm(balanced.s)<=1+1e-10
end

@testset "native cover components retain balanced external current constraint" begin
    p=_synthetic_smd_project()
    p.ports[2]=SonnetPortSpec(:box,2,1,-1,["-1","50","0","0","0","6","2"],SonnetRecord[])
    model=sonnet_component_model(p,1e8)
    @test size(model.floating_common)==(6,1)
    r=solve_sonnet_project(p,1e8;raw=true,mx=36,my=24)
    @test r.port_numbers==[1]
    v=r.contraction*(r.gap_transfer*r.circuit.voltages[:,1])
    @test norm(transpose(model.floating_common)*r.raw.y*v)<1e-10norm(r.raw.y*v)
    @test planar_current_maps(r.raw;voltages=v)[1].jx≈sonnet_component_current_maps(r)[1].jx rtol=1e-12
    @test opnorm(r.s)<=1+1e-10
end
