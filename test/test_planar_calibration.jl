using Test,LinearAlgebra,DiffMoM

@testset "planar calibration: general reciprocal SOC launches" begin
    z0 = 50.0
    from_abcd(T) = planar_y_to_s(DiffMoM._y_of_abcd(T),[z0,z0])
    # Different reciprocal launches contain both series and shunt terms;
    # their shunt-only double-delay approximation is not valid.
    X = ComplexF64[1 3+0.8im;0 1]*ComplexF64[1 0;0.001+0.002im 1]
    Y = ComplexF64[1 0;0.002+0.003im 1]*ComplexF64[1 5+1.2im;0 1]
    left,right = from_abcd(X),from_abcd(Y)
    gl = 0.05+0.7im
    matched_line = ComplexF64[0 exp(-gl);exp(-gl) 0]
    thru = planar_embed_2port(ComplexF64[0 1;1 0],left,right)
    line = planar_embed_2port(matched_line,left,right)
    load = 0.98exp(-0.2im)
    reflect_one(S) = S[1,1]+S[1,2]*S[2,1]*load/(1-S[2,2]*load)
    measured_reflect = Diagonal(ComplexF64[reflect_one(left),reflect_one(right[[2,1],[2,1]])])
    dut = ComplexF64[0.2+0.1im 0.6-0.15im;0.6-0.15im -0.1+0.05im]
    measured = planar_embed_2port(dut,left,right)
    for known in (nothing,load)
        calibration = planar_line_calibrate(thru,line;delta_length=0.002,
            reflect_standard=measured_reflect,reflection=known)
        @test calibration.gamma ≈ gl/0.002 rtol=1e-10
        @test calibration.reflection ≈ load rtol=1e-10
        @test calibration.residual < 1e-10
        @test planar_calibration_apply(measured,calibration) ≈ dut rtol=1e-10
    end
    @test planar_t_to_s(planar_s_to_t(dut)) ≈ dut
    @test_throws ArgumentError planar_line_calibrate(thru,line;delta_length=0.002,
        reflect_standard=ones(2,2))
end

@testset "planar calibration: physical planes and sister standards" begin
    grid = CellGrid(4e-3,5e-3,8,10)
    stack = PlanarStackup([PlanarLayer(1,1,0.5e-3),PlanarLayer(1,1,0.5e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheet = sheet_level(1,8,10)
    sheet.mask[:,5:6].=true
    sheet.connect_west[5:6].=true;sheet.connect_east[5:6].=true
    seen = Float64[]
    plane = PlanarReferencePlane(0.2e-3,50.0,f -> (push!(seen,f);2pi*1im*f/299792458))
    ports = [PlanarPort(1,:west,5:6,50;refplane=plane),PlanarPort(1,:east,5:6,50)]
    prob = build_planar_problem(stack,grid,[sheet],ports)
    for f in (1e9,2e9)
        raw = solve_planar(prob,f;mx=24,my=30)
        original_s = copy(raw.s)
        calibrated = planar_reference_planes(raw)
        @test raw.s==original_s
        @test calibrated.raw===raw
        chain = planar_line_abcd(50,2pi*1im*f*0.2e-3/299792458)
        @test calibrated.y ≈ deembed_ports(raw.y,[chain,Matrix{ComplexF64}(I,2,2)])
        wave = ComplexF64[1,0]
        vb = sqrt(50)*(wave+calibrated.s*wave)
        va = calibrated.gap_voltage_transfer*vb
        maps = planar_current_maps(calibrated;incident_waves=wave)
        raw_maps = planar_current_maps(raw;voltages=va)
        @test maps[1].jx ≈ raw_maps[1].jx
        @test maps[1].jy ≈ raw_maps[1].jy
    end
    @test seen==[1e9,2e9]
    standards = planar_line_standards(prob;extra_cells=8,reflect_cells=2)
    @test standards.delta_length≈grid.a
    @test standards.line.grid.dx≈grid.dx
    @test standards.line.grid.nx==16
    @test count(standards.thru.sheets[1].mask)==16
    @test count(standards.reflect.sheets[1].mask)==8
    for standard in (standards.thru,standards.line,standards.reflect)
        @test all(isfinite,solve_planar(standard,2e9;mx=32,my=40).s)
    end
    @test_throws ArgumentError PlanarReferencePlane(NaN,50,1im)
    @test_throws ArgumentError PlanarReferencePlane(1e-3,50,"gamma")
end

@testset "planar calibration: coupled group and mixed modes" begin
    Y = ComplexF64[0.02+0.01im -0.002im 0.003;-0.002im 0.03+0.02im -0.004im;
        0.003 -0.004im 0.01+0.01im]
    # A coupled two-port launch network with mutual shunt admittance and
    # mutual series impedance needs full blocks, rather than diagonal chains.
    Zg = ComplexF64[2 0.3im;0.3im 3]
    Yg = ComplexF64[0.001im -0.0002im;-0.0002im 0.002im]
    I2 = Matrix{ComplexF64}(I,2,2)
    chain = [I2 Zg;zeros(2,2) I2]*[I2 zeros(2,2);Yg I2]
    ports = [1,3]
    A,D = Matrix{ComplexF64}(I,3,3),Matrix{ComplexF64}(I,3,3)
    B,C = zeros(ComplexF64,3,3),zeros(ComplexF64,3,3)
    A[ports,ports],B[ports,ports],C[ports,ports],D[ports,ports] =
        chain[1:2,1:2],chain[1:2,3:4],chain[3:4,1:2],chain[3:4,3:4]
    measured = (C+D*Y)/(A+B*Y)
    @test deembed_cocal_group(measured,ports,chain) ≈ Y rtol=1e-12
    individual = [planar_line_abcd(50,0.1im),planar_line_abcd(75,0.2im),Matrix{ComplexF64}(I,2,2)]
    @test deembed_ports(embed_ports(Y,individual),individual) ≈ Y rtol=1e-12
    S = ComplexF64[0.1 0.02 0.6 0.01;0.02 0.1 0.01 0.6;
        0.6 0.01 0.1 0.02;0.01 0.6 0.02 0.1]
    mixed = planar_mixed_mode(S,[(1,2),(3,4)];z0=50)
    @test mixed.z0 == [100,100,25,25]
    @test transpose(mixed.transform)*mixed.s*mixed.transform ≈ S rtol=1e-12
    @test svdvals(mixed.s) ≈ svdvals(S)
    @test abs(mixed.s[1,3]) < 1e-15 # differential/common conversion vanishes
    @test_throws ArgumentError planar_mixed_mode(S,[(1,2),(2,3)])
    @test_throws ArgumentError planar_mixed_mode(S,[(1,2)];z0=[50,75,50,50])
end

@testset "planar calibration: coupled double-delay identification" begin
    # Noncommuting symmetric series/shunt blocks exercise genuinely
    # coupled launch groups, rather than independent single-port chains.
    Zg=ComplexF64[1+0.4im 0.1im;0.1im 2+0.7im]
    Yg=ComplexF64[0.001im -0.0002im;-0.0002im 0.002im]
    Zl=ComplexF64[0.8+12im 2im;2im 0.6+15im]
    Yl=ComplexF64[0.0001+0.005im -0.001im;-0.001im 0.0002+0.006im]
    E=exp([zeros(2,2) Zg;Yg zeros(2,2)])
    L=exp([zeros(2,2) Zl;Yl zeros(2,2)])
    first=planar_abcd_to_y(E*L*E)
    second=planar_abcd_to_y(E*L*L*E)
    cal=planar_group_double_delay_calibrate(first,second;length=0.002)
    @test cal.launch≈E rtol=1e-10
    @test cal.line≈L rtol=1e-10
    @test cal.residual<1e-10
    @test planar_y_to_abcd(planar_abcd_to_y(L))≈L rtol=1e-12
    # Independently selected coupled DUT, embedded in both recovered groups.
    dut=ComplexF64[0.03+0.01im 0.002 -0.005im 0.001;
        0.002 0.04+0.02im 0.001 -0.006im;
        -0.005im 0.001 0.025+0.01im 0.002;
        0.001 -0.006im 0.002 0.035+0.015im]
    A,B,C,D=[zeros(ComplexF64,4,4) for _ in 1:4]
    for indices in (1:2,3:4)
        A[indices,indices]=E[1:2,1:2];B[indices,indices]=E[1:2,3:4]
        C[indices,indices]=E[3:4,1:2];D[indices,indices]=E[3:4,3:4]
    end
    measured=(C+D*dut)/(A+B*dut)
    recovered=deembed_cocal_group(deembed_cocal_group(measured,[1,2],cal.launch),[3,4],cal.launch)
    @test recovered≈dut rtol=1e-10
    @test_throws ArgumentError planar_group_double_delay_calibrate(first,second;
        length=0.002,max_bytes=128)
end
