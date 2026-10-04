using DiffMoM,Test,LinearAlgebra

@testset "circuit floating gauges preserve terminal power and incidence" begin
    D=planar_port_incidence(4,[(1,2),(3,4)])
    @test D==[1. 0.;-1. 0.;0. 1.;0. -1.]
    v=ComplexF64[1,.3im,2,.2];i=ComplexF64[.02+.01im,.03im]
    @test dot(v,D*i)≈dot(transpose(D)*v,i)
    c=PlanarCircuit(2,[(1,2)]);circuit_add_rlc!(c,1,2;r=100.)
    @test_throws ArgumentError solve_planar_circuit(c,1e9)
    r=solve_planar_circuit(c,1e9;floating_gauge=:auto)
    @test r.gauge_nodes==[1]
    @test r.s[1,1]≈1/3
    @test r.voltages[1,1]≈0.
    @test (r.voltages[1,1]-r.voltages[2,1])/100≈r.currents[1,1]
    shifted=r.voltages[:,1].+17.3im
    @test transpose(planar_port_incidence(2,[(1,2)]))*shifted≈
        transpose(planar_port_incidence(2,[(1,2)]))*r.voltages[:,1]
    transformer=PlanarCircuit(4,[(1,2)])
    circuit_add_transformer!(transformer,[(1,2),(3,4)],2.)
    circuit_add_rlc!(transformer,3,4;r=100.)
    t=solve_planar_circuit(transformer,1e9;floating_gauge=:auto)
    @test t.gauge_nodes==[1,3]
    @test t.s[1,1]≈(400-50)/(400+50)
    vp=t.voltages[1,1]-t.voltages[2,1];vs=t.voltages[3,1]-t.voltages[4,1]
    @test vp≈2vs
    @test .5real(conj(vp)*t.currents[1,1])≈.5abs2(vs)/100
    orphan=PlanarCircuit(2,[1]);circuit_add_rlc!(orphan,1,0;r=100)
    isolated=solve_planar_circuit(orphan,1e9;floating_gauge=:auto)
    @test isolated.gauge_nodes==[2]
    @test isolated.s[1,1]≈1/3
    @test isolated.voltages[2,1]==0
    @test_throws ArgumentError solve_planar_circuit(c,1e9;floating_gauge=:unknown)
    @test_throws ArgumentError planar_port_incidence(4,[(1,2),(3,4)];max_bytes=1)
end

@testset "project explicit S/G nodes and represented common modes" begin
    p=_floating_project()
    p.data["ports"][1]["nodes"]=["S","G"]
    p.data["ports"][1]["external"]=true
    unloaded=solve_planar_project(p,1e9;mx=36,my=24)
    modes=planar_project_source_modes(unloaded.model)
    @test modes.node_names==["S","G"]
    @test modes.incidence==[1.;-1.;;]
    @test modes.represented_modes==1
    @test size(modes.unexcited_voltage_modes)==(2,1)
    @test norm(transpose(modes.incidence)*modes.unexcited_voltage_modes)<1e-14
    p.data["components"]=[Dict("name"=>"physical_load","type"=>"resistor","nodes"=>["S","G"],"value"=>"100 ohm")]
    loaded=solve_planar_project(p,1e9;mx=36,my=24)
    @test loaded.y≈unloaded.y.+.01 rtol=1e-10
    @test loaded.circuit.gauge_nodes==[1]
    @test loaded.model.port_terminals==[(1,2)]
    v=transpose(loaded.model.source_incidence)*loaded.circuit.voltages[:,1]
    @test planar_current_maps(loaded;incident_waves=[1.])[1].jx≈
        planar_current_maps(loaded.em;voltages=v)[1].jx rtol=1e-12
    # Add an actual reference-to-box source rather than inventing its
    # common-mode response from the differential bridge data.
    p=_floating_project();p.data["polygons"][1]["vertices"][1][1]=0.
    p.data["polygons"][1]["vertices"][4][1]=0.
    p.data["ports"][1]["nodes"]=["S","G"];p.data["ports"][1]["external"]=true
    push!(p.data["ports"],Dict("name"=>"ground_mode","number"=>2,"polygon"=>"reference","edge"=>4,
        "nodes"=>["G","gnd"],"external"=>true))
    complete=solve_planar_project(p,1e9;mx=36,my=24)
    report=planar_project_source_modes(complete.model)
    @test report.incidence==[1. 0.;-1. 1.]
    @test report.represented_modes==2
    @test size(report.unexcited_voltage_modes)==(2,0)
    p.data["components"]=[Dict("type"=>"resistor","name"=>"signal_to_box","nodes"=>["S","gnd"],"value"=>"100 ohm")]
    coupled=solve_planar_project(p,1e9;mx=36,my=24)
    @test coupled.y≈complete.y.+.01ones(2,2) rtol=1e-10
    @test isempty(coupled.circuit.gauge_nodes)
    D=report.incidence;direct=PlanarCircuit(2,[(1,2),(2,0)])
    circuit_add_network!(direct,[(1,0),(2,0)],D*complete.y*transpose(D);format=:y)
    circuit_add_rlc!(direct,1,0;r=100.)
    oracle=solve_planar_circuit(direct,1e9)
    @test oracle.s≈coupled.s rtol=1e-10
    attach=PlanarCircuit(2,[(1,2),(2,0)])
    circuit_add_em!(attach,[(1,2),(2,0)],complete.em)
    reactive=PlanarCircuit(2,[(1,2),(2,0)])
    refs=[50+1im,50]
    response=(;s=planar_renormalize_s(complete.em.s,complete.em.z0,refs),z0=refs)
    @test circuit_add_em!(reactive,[(1,2),(2,0)],response)===reactive
    @test solve_planar_circuit(reactive,1e9).s≈complete.s rtol=1e-10
    @test circuit_add_em!(PlanarCircuit(1,[1]),[1],(;s=zeros(1,1),z0=50+2im)) isa PlanarCircuit
    @test solve_planar_circuit(attach,1e9).s≈complete.s rtol=1e-10
    @test_throws ArgumentError planar_project_source_modes(complete.model;max_bytes=1)
    @test_throws ArgumentError planar_current_maps(coupled;max_bytes=1)
    broken=deepcopy(p.data);broken["ports"][1]["nodes"]=["S","S"]
    @test_throws ArgumentError planar_project_layout(planar_project_from_dict(broken))
end
