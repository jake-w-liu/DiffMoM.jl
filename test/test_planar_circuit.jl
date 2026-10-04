using Test, LinearAlgebra, DiffMoM

@testset "planar circuit: native network blocks" begin
    S0 = ComplexF64[0.1+0.2im 0.4-0.1im; 0.3+0.1im -0.2+0.05im]
    c = PlanarCircuit(2,[1,2];z0=[50,75])
    circuit_add_network!(c,[1,2],S0;z0=[50,75])
    @test solve_planar_circuit(c,1e9).s ≈ S0 rtol=1e-12
    # Exact ideal networks have finite S even when finite Y does not exist.
    for reference in (ComplexF64[0 1;1 0], Matrix{ComplexF64}(I,2,2),
            -Matrix{ComplexF64}(I,2,2))
        ci = PlanarCircuit(2,[1,2])
        circuit_add_network!(ci,[1,2],reference)
        @test solve_planar_circuit(ci,1e9).s ≈ reference atol=1e-12
    end
    Y = ComplexF64[0.02+0.01im -0.005im;-0.005im 0.01+0.03im]
    for (value,format) in ((Y,:y),(inv(Y),:z))
        cn = PlanarCircuit(2,[1,2];z0=[50,75])
        circuit_add_network!(cn,[1,2],value;format=format)
        result = solve_planar_circuit(cn,1e9)
        @test result.s ≈ planar_y_to_s(Y,[50,75]) rtol=1e-12
        @test result.y ≈ Y rtol=1e-12
    end
end

@testset "planar circuit: hybrid cascade and component limits" begin
    f,zc,gl,r = 3e9,50.0,0.03+0.7im,25.0
    c = PlanarCircuit(3,[1,3])
    circuit_add_line!(c,[1,2],zc,frequency -> gl * frequency/f)
    circuit_add_rlc!(c,2,3;r=r)
    oracle_t = planar_line_abcd(zc,gl)*ComplexF64[1 r;0 1]
    oracle_y = DiffMoM._y_of_abcd(oracle_t)
    @test solve_planar_circuit(c,f).s ≈ planar_y_to_s(oracle_y,[50,50]) rtol=1e-12
    @test length(planar_circuit_sparams(c,[f,2f])) == 2
    # A one-port parallel RLC has the analytic sum admittance.
    cp = PlanarCircuit(1,[1])
    circuit_add_rlc!(cp,1,0;r=100,l=1e-9,c=0.1e-12,topology=:parallel)
    y = inv(100)+inv(2pi*f*1im*1e-9)+2pi*f*1im*0.1e-12
    @test solve_planar_circuit(cp,f).y[1,1] ≈ y
    @test solve_planar_circuit(cp,0).s[1,1] ≈ -1 # DC inductor short
    cs = PlanarCircuit(1,[1])
    circuit_add_rlc!(cs,1,0;r=25,l=1e-9,c=0.1e-12)
    z = 25+2pi*f*1im*1e-9+inv(2pi*f*1im*0.1e-12)
    @test solve_planar_circuit(cs,f).s[1,1] ≈ (z-50)/(z+50)
    @test solve_planar_circuit(cs,0).s[1,1] ≈ 1 # DC capacitor open
    @test_throws ArgumentError circuit_add_rlc!(cs,1,0)
    @test_throws ArgumentError circuit_add_rlc!(cs,1,0;r=-1)
    @test_throws ArgumentError solve_planar_circuit(cs,f;max_bytes=1)
end

@testset "planar circuit: transformer and differential terminals" begin
    c = PlanarCircuit(2,[1,2];z0=[200,50])
    circuit_add_transformer!(c,[1,2],2)
    result = solve_planar_circuit(c,1e9)
    @test result.s ≈ ComplexF64[0 1;1 0] atol=1e-12
    # Network port's minus terminal is a circuit node, rather than an
    # implicit ground. Its differential voltage is retained by the stamp.
    cd = PlanarCircuit(2,[(1,2)])
    circuit_add_rlc!(cd,1,2;r=100)
    circuit_add_rlc!(cd,2,0;r=20)
    @test solve_planar_circuit(cd,1e9).s[1,1] ≈ 1/3
    # Frequency-dependent measured/EM blocks are sampled at the requested f.
    seen = Float64[]
    cm = PlanarCircuit(1,[1])
    circuit_add_network!(cm,[1],f -> (push!(seen,f);fill(0.3+0.2im,1,1)))
    @test solve_planar_circuit(cm,2e9).s[1,1] ≈ 0.3+0.2im
    @test seen == [2e9]
    @test_throws ArgumentError PlanarCircuit(2,[(1,1)])
    @test_throws ArgumentError circuit_add_transformer!(cm,[1,1],0)
    floating = PlanarCircuit(2,[1])
    @test_throws ArgumentError solve_planar_circuit(floating,1e9)
end
