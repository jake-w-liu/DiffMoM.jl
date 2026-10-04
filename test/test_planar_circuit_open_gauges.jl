module PlanarCircuitOpenGaugeTests
using DiffMoM, Test, LinearAlgebra

@testset "Exact opens retain independent unused voltage coordinates" begin
    for topology in (:series,:parallel),(capacitance,frequency) in ((0.,1e9),(1e-12,0.))
        circuit=PlanarCircuit(2,[1]);circuit_add_rlc!(circuit,1,2;c=capacitance,topology)
        @test_throws ArgumentError solve_planar_circuit(circuit,frequency)
        result=solve_planar_circuit(circuit,frequency;floating_gauge=:auto)
        @test result.s==ones(ComplexF64,1,1)
        @test result.gauge_nodes==[2]
        @test result.voltages[2,:]==[0]
        @test all(iszero,result.currents)
    end
    for format in (:s,:y),n in (1,2,3),frequency in (0.,1e9)
        circuit=PlanarCircuit(n+1,[1])
        matrix=format===:s ? Matrix{ComplexF64}(I,n,n) : zeros(ComplexF64,n,n)
        circuit_add_network!(circuit,[(1,p+1) for p in 1:n],matrix;format,z0=50+20im)
        result=solve_planar_circuit(circuit,frequency;floating_gauge=:auto)
        @test result.s==ones(ComplexF64,1,1)
        @test result.gauge_nodes==collect(2:n+1)
        @test all(iszero,result.currents)
    end
    # The parallel resonance is an exact open, with its branch still present.
    circuit=PlanarCircuit(2,[1]);circuit_add_rlc!(circuit,1,2;l=1.,c=1.,topology=:parallel)
    result=solve_planar_circuit(circuit,1/(2pi);floating_gauge=:auto)
    @test result.s==ones(ComplexF64,1,1)
    @test result.gauge_nodes==[2]
end

@testset "SPICE inactive currents and zero voltage-control gains preserve gauges" begin
    mktempdir() do directory
        path=joinpath(directory,"opens.cir")
        for (card,frequency,gauges) in (("Copen a b 0",1e9,[2,3,4]),
                ("Copen a b 1p",0.,[2,3,4]),("Iopen a b DC 2",1e9,[2,3,4]),
                ("Gopen a b c d 0",1e9,[2,3,4]),("Gopen a b c c 2",1e9,[2,3,4]),
                ("Eshort a b c d 0",1e9,[3,4]))
            write(path,".subckt inactive a b c d\n"*card*"\n.ends inactive\n")
            model=planar_spice_model(planar_read_spice(path),"inactive")
            circuit=PlanarCircuit(4,[1]);circuit_add_spice!(circuit,[1,2,3,4],model)
            result=solve_planar_circuit(circuit,frequency;floating_gauge=:auto)
            @test result.s≈ones(ComplexF64,1,1) atol=1e-15
            @test result.gauge_nodes==gauges
            @test all(iszero,result.currents)
        end
    end
end
end
