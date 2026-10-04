using DiffMoM,Test,LinearAlgebra,Random

# Independent physical boundary V+Z I=2sqrt(ReZ) a, with I=Y V.
function _pw_oracle(Y,refs)
    roots=sqrt.(real.(refs))
    voltage=(I+Diagonal(refs)*Y)\Diagonal(2roots)
    current=Y*voltage
    return Diagonal(inv.(2roots))*(voltage-Diagonal(conj.(refs))*current)
end

function _pw_project()
    return PlanarProject(Dict("project"=>Dict("name"=>"complex-reference","unit"=>"mm"),
        "box"=>Dict("width"=>3.,"length"=>2.),"mesh"=>Dict("nx"=>6,"ny"=>4),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5),Dict("thickness"=>.5)]),
        "metals"=>Dict("film"=>Dict("type"=>"surface_impedance","rs"=>.1)),
        "polygons"=>[Dict("name"=>"wire","level"=>1,"metal"=>"film",
            "vertices"=>[[0.,.5],[3.,.5],[3.,1.5],[0.,1.5]])],
        "ports"=>[Dict("name"=>"input","number"=>1,"polygon"=>"wire","edge"=>4,
                "impedance"=>Dict("r"=>"=50+freq/1e9","x"=>"=20*freq/1e9","l"=>"-1 nH")),
            Dict("name"=>"output","number"=>2,"polygon"=>"wire","edge"=>2,
                "impedance"=>"=complex(75,-10*freq/1e9)")],
        "sweep"=>Dict("frequencies"=>[".8 GHz","1 GHz","1.2 GHz"])))
end

@testset "Kurokawa waves: physical energy and reference providers" begin
    rng=MersenneTwister(10326)
    for n in 1:5
        r=randn(rng,n,n);x=randn(rng,n,n)
        Y=.01*(r'*r+I)+im*.01*(x+x')
        refs=20 .+80rand(rng,n).+im*(60rand(rng,n).-30)
        S=planar_y_to_s(Y,refs)
        @test S≈_pw_oracle(Y,refs) rtol=2e-14 atol=2e-14
        @test planar_s_to_y(S,refs)≈Y rtol=2e-14 atol=2e-14
        @test S≈transpose(S) atol=3e-14
        @test opnorm(S)<=1+1e-14
        a=randn(rng,ComplexF64,n)
        V=planar_wave_voltages(S,a;z0=refs);current=Y*V
        waves=planar_power_waves(V,current;z0=refs)
        @test waves.a≈a rtol=3e-14 atol=3e-14
        @test waves.b≈S*a rtol=3e-14 atol=3e-14
        @test waves.accepted_power≈.5(sum(abs2,a)-sum(abs2,S*a)) atol=3e-14
        @test DiffMoM._planar_incident_from_voltage(S,V,refs)≈a rtol=3e-14 atol=3e-14
    end
    z=PlanarPortImpedance(r=50,x=-10,l=-1e-9,c=-1e-12)
    @test z(1e9)≈50+im*(-10-2pi+1/(2pi*1e9*1e-12))
    @test_throws ArgumentError z(0.)
    @test PlanarPortImpedance(r=50,c=1e-12,topology=:parallel)(0.)==50
    @test PlanarPortImpedance(r=50,x=20,l=1e-9,c=1e-12,topology=:parallel)(1e9)≈
        inv(inv(50+im*(20+2pi))+im*2pi*1e9*1e-12)
    calls=Ref(0);provider(f)=(calls[]+=1;50+im*f/1e8)
    circuit=PlanarCircuit(1,[1];z0=provider)
    circuit_add_network!(circuit,[1],zeros(1,1);z0=provider)
    @test calls[]==0
    @test_throws ArgumentError solve_planar_circuit(circuit,1e9;max_bytes=1)
    @test calls[]==0
    solved=solve_planar_circuit(circuit,1e9)
    @test solved.z0==[50+10im]
    @test abs(solved.s[1,1])<1e-14
    @test calls[]>0
    calls[]=0
    @test_throws ArgumentError planar_power_waves([1.],[.01];z0=provider,freq=1e9,max_bytes=1)
    @test calls[]==0
    @test_throws ArgumentError planar_wave_voltages(zeros(1,1),[1.];z0=provider,freq=1e9,max_bytes=1)
    @test calls[]==0
    @test_throws ArgumentError PlanarPortImpedance(r=0)
    @test_throws ArgumentError PlanarPortImpedance(c=big"1e-400")
    @test_throws ArgumentError PlanarPortImpedance(l=Inf)
    @test_throws ArgumentError PlanarPortImpedance(topology=:unknown)
    @test_throws ArgumentError planar_y_to_s(ones(1,1),[1im])
    @test_throws ArgumentError planar_power_waves([big"1e400"],[1.])
    @test_throws ArgumentError planar_wave_voltages(zeros(1,1),[big"1e400"])
    @test_throws ArgumentError planar_s_to_y(reshape([1im],1,1),[50+50im])
    @test DiffMoM._planar_wave_voltage(ComplexF64[-1+eps(Float64);;],[1.],[50.])==
        sqrt.(real.([50.])).*([1.]+ComplexF64[-1+eps(Float64);;]*[1.])
    @test PlanarPortImpedance(r=50,x=10,c=1e-12,topology=:parallel)(0.)==50+10im
end

@testset "complex-reference MNA and physical cascades" begin
    ext=[45+17im,80-13im];localrefs=[30-9im,90+31im]
    Y1=ComplexF64[.04+.03im -.01-.01im;-.01-.01im .03+.02im]
    Y2=ComplexF64[.03+.01im -.005;-.005 .02+.02im]
    circuit=PlanarCircuit(2,[1,2];z0=ext)
    circuit_add_network!(circuit,[1,2],_pw_oracle(Y1,localrefs);z0=localrefs)
    circuit_add_network!(circuit,[1,2],inv(Y2);format=:z)
    result=solve_planar_circuit(circuit,1e9)
    @test result.y≈Y1+Y2 rtol=2e-14
    @test result.s≈_pw_oracle(Y1+Y2,ext) rtol=2e-14
    @test result.s≈transpose(result.s) atol=3e-14
    @test opnorm(result.s)<1
    @test result.voltages≈DiffMoM._planar_wave_voltage_matrix(result.s,ext) rtol=2e-14
    for (value,expected) in ((0.,-conj(ext[1])/ext[1]),(conj(ext[1]),0.))
        c=PlanarCircuit(1,[1];z0=ext[1]);circuit_add_network!(c,[1],reshape([value],1,1);format=:z)
        @test solve_planar_circuit(c,1e9).s[1,1]≈expected atol=1e-14
    end
    left=planar_line_abcd(42+3im,.02+.2im)
    device=planar_line_abcd(61+7im,.03+.4im)
    right=planar_line_abcd(55-4im,.01+.3im)
    ls=planar_y_to_s(DiffMoM._y_of_abcd(left),localrefs)
    ds=planar_y_to_s(DiffMoM._y_of_abcd(device),[75-10im,50+20im])
    rs=planar_y_to_s(DiffMoM._y_of_abcd(right),ext)
    embedded=planar_embed_2port(ds,ls,rs;z0=[75-10im,50+20im],left_z0=localrefs,right_z0=ext)
    expected=planar_y_to_s(DiffMoM._y_of_abcd(left*device*right),[localrefs[1],ext[2]])
    @test embedded≈expected rtol=3e-14 atol=3e-14
end

@testset "complex measured TRL and independently known line impedance" begin
    refs=[50+20im,75-10im];zc=50+8im;gl=.04+.7im
    left=ComplexF64[1 3+.8im;0 1]*ComplexF64[1 0;.001+.002im 1]
    right=ComplexF64[1 0;.002+.003im 1]*ComplexF64[1 5+1.2im;0 1]
    scattering(T)=planar_y_to_s(DiffMoM._y_of_abcd(T),refs)
    thru=scattering(left*right);line=scattering(left*planar_line_abcd(zc,gl)*right)
    zl=left[1,1]/left[2,1];zr=right[2,2]/right[2,1]
    reflect=Diagonal(ComplexF64[(zl-conj(refs[1]))/(zl+refs[1]),
        (zr-conj(refs[2]))/(zr+refs[2])])
    calibration=planar_line_calibrate(thru,line;z0=refs,line_impedance=zc,
        delta_length=.002,reflect_standard=reflect,reflection=1.)
    @test calibration.gamma≈gl/.002 rtol=1e-12
    dut=planar_line_abcd(zc,.07+.3im)
    measured=scattering(left*dut*right)
    recovered=planar_calibration_apply(measured,calibration;output_z0=refs)
    @test recovered≈scattering(dut) rtol=1e-11 atol=1e-11
    @test opnorm(recovered)<=1+1e-12
    unknown=planar_line_calibrate(thru,line;z0=refs,delta_length=.002,
        reflect_standard=reflect,reflection=1.)
    @test_throws ArgumentError planar_calibration_apply(measured,unknown;output_z0=refs)
end

@testset "project physical currents, calibration and dynamic-reference sweeps" begin
    p=_pw_project();f=1e9
    raw=solve_planar_project(p,f;mx=16,my=12)
    @test raw.z0≈[51+im*(20-2pi),75-10im]
    @test raw.s≈_pw_oracle(raw.y,raw.z0) rtol=2e-13
    a=ComplexF64[1,.2im];v=planar_wave_voltages(raw.s,a;z0=raw.z0)
    maps=planar_current_maps(raw;incident_waves=a)
    explicit=planar_current_maps(raw.em;voltages=v)
    @test maps[1].jx≈explicit[1].jx rtol=2e-13
    @test maps[1].jy≈explicit[1].jy rtol=2e-13
    C=[1. 0.;0. 1.]
    contracted=solve_planar_contracted(raw.model.layout.problem,f,C;z0=raw.z0,mx=16,my=12,surface_zs=.1)
    @test contracted.y≈raw.y rtol=2e-12
    @test planar_current_maps(contracted;incident_waves=a)[1].jx≈maps[1].jx rtol=2e-12
    fast=solve_planar_contracted(raw.model.layout.problem,f,C;z0=raw.z0,mx=16,my=12,
        surface_zs=.1,method=:ufft,memory=40,rtol=1e-9)
    @test fast.s≈contracted.s rtol=2e-9 atol=2e-9
    filled=solve_planar_contracted(raw.model.layout.problem,f,C;z0=raw.z0,mx=16,my=12,
        surface_zs=.1,method=:dense_fft)
    @test filled.s≈contracted.s rtol=2e-12
    splits=[planar_subdivide(raw.model.layout.problem,:x,3;z0=z) for z in (50.,40+15im)]
    recombined=[solve_planar_subdivision(plan,f;mx=16,my=12,surface_zs=.1,retain_results=true) for plan in splits]
    @test recombined[1].y≈recombined[2].y rtol=2e-12
    @test recombined[1].z0==recombined[2].z0==raw.z0
    @test recombined[1].s≈recombined[2].s rtol=2e-12
    @test planar_subdivision_currents(recombined[1];incident_waves=a)[1][1].jx≈
        planar_subdivision_currents(recombined[2];incident_waves=a)[1][1].jx rtol=2e-12
    p.data["ports"][1]["refplane"]=Dict("length"=>".2 mm","zc"=>"=complex(42,3)","gamma"=>"=complex(0,freq/1e8)")
    calibrated=solve_planar_project(p,f;mx=16,my=12)
    @test calibrated.em.z0==calibrated.z0
    cv=planar_wave_voltages(calibrated.s,a;z0=calibrated.z0)
    gap=calibrated.em.gap_voltage_transfer*cv
    @test planar_current_maps(calibrated;incident_waves=a)[1].jx≈
        planar_current_maps(calibrated.em.raw;voltages=gap)[1].jx rtol=3e-13
    data=planar_project_sweep(p;mx=16,my=12)
    @test data.reference_series!==nothing
    for (k,freq) in enumerate(data.frequencies)
        point=solve_planar_project(p,freq;mx=16,my=12)
        @test data.reference_series[k]==point.z0
        @test data.s[k]≈point.s rtol=2e-13
    end
    p.data["sweep"]=Dict("start"=>".8 GHz","stop"=>"1.2 GHz","adaptive"=>true,
        "n_eval"=>9,"max_points"=>4,"rel_tol"=>1e3)
    adaptive=planar_project_sweep(p;mx=16,my=12)
    @test adaptive.z0==data.reference_series[1]
    for (freq,S) in zip(adaptive.freqs,adaptive.s)
        point=solve_planar_project(p,freq;mx=16,my=12)
        @test S≈planar_renormalize_s(point.s,point.z0,adaptive.z0) rtol=2e-13
    end
    p.data["ports"][1]["resistance"]=50.
    @test_throws ArgumentError planar_project_layout(p;freq=f)
end

@testset "inline project model keeps frequency-specific S references" begin
    p=_pw_project();p.data["ports"][2]["external"]=false
    p.data["components"]=[Dict("name"=>"load","type"=>"inline_s","ports"=>["output"],
        "frequencies"=>[".8 GHz","1 GHz","1.2 GHz"],"s_real"=>[[[0.]],[[0.]],[[0.]]],
        "z0"=>Dict("r"=>100.,"x"=>"=5*freq/1e9","c"=>"1 pF","topology"=>"parallel"))]
    result=solve_planar_project(p,1e9;mx=16,my=12)
    ref=PlanarPortImpedance(r=100,x=5,c=1e-12,topology=:parallel)(1e9)
    oracle=PlanarCircuit(2,[1];z0=result.z0)
    circuit_add_em!(oracle,[1,2],result.em)
    circuit_add_network!(oracle,[2],zeros(1,1);z0=ref)
    expected=solve_planar_circuit(oracle,1e9)
    @test result.s≈expected.s rtol=2e-13
    @test result.y≈expected.y rtol=2e-13
    @test result.circuit.voltages≈expected.voltages rtol=2e-13
    @test planar_current_maps(result;incident_waves=[1.])[1].jx≈
        planar_current_maps(result.em;voltages=expected.voltages[:,1])[1].jx rtol=2e-13
end

@testset "native FLOAT reference and physical calibrated current transfer" begin
    path=joinpath(@__DIR__,"fixtures","native_floating_power_waves","floating_raw.son")
    @test bytes2hex(DiffMoM.SHA.sha256(read(path)))==
        "267689d24b7e832dc19b0e9cb705097907a8941dab511790e7cac83436e1b170"
    source=read_sonnet_project(path)
    baseline=solve_sonnet_project(source,1e9;raw=true,mx=24,my=24)
    for port in source.ports
        port.values[3]="20";port.values[4]="1";port.values[5]=".2"
    end
    raw=solve_sonnet_project(source,1e9;raw=true,mx=24,my=24)
    reference=inv(inv(50+im*(20+2pi))+im*2pi*1e9*.2e-12)
    @test raw.z0≈[reference] rtol=2e-14
    @test raw.y≈baseline.y rtol=2e-12
    @test raw.s≈_pw_oracle(raw.y,raw.z0) rtol=2e-13
    chain=planar_line_abcd(43+2im,.01+.1im)
    calibrated=solve_sonnet_project(source,1e9;raw=true,mx=24,my=24,calibration=[chain])
    vc=planar_wave_voltages(calibrated.s,[1.];z0=calibrated.z0);ic=calibrated.y*vc
    vr=chain[1,1]*vc+chain[1,2]*ic;ir=chain[2,1]*vc+chain[2,2]*ic
    incident=planar_power_waves(vr,ir;z0=calibrated.z0).a
    @test calibrated.incident_transfer[:,1]≈incident rtol=2e-13
    @test planar_current_maps(calibrated;incident_waves=[1.])[1].jx≈
        planar_current_maps(raw;incident_waves=incident)[1].jx rtol=2e-12
end
