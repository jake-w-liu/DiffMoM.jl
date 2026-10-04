module PlanarCircuitDomainTests
using DiffMoM, Test, LinearAlgebra

@testset "Circuit stored scalars reject overflow/underflow transactionally" begin
    for topology in (:series,:parallel),field in (:r,:l,:c),value in
            (BigFloat("1e1000"),BigFloat("1e-1000"),Inf,NaN,-1.)
        circuit=PlanarCircuit(2,[1,2])
        circuit_add_rlc!(circuit,1,2;r=10.)
        elements=copy(circuit.elements)
        parameters=NamedTuple{(field,)}((value,))
        @test_throws ArgumentError circuit_add_rlc!(circuit,1,2;topology,parameters...)
        @test circuit.elements==elements
        @test circuit.nnodes==2
        @test circuit.ports==[(1,0),(2,0)]
    end
    for value in (BigFloat("1e1000"),BigFloat("-1e1000"),BigFloat("1e-1000"),
            BigFloat("-1e-1000"),Inf,NaN,0.)
        circuit=PlanarCircuit(2,[1,2])
        @test_throws ArgumentError circuit_add_transformer!(circuit,[(1,0),(2,0)],value)
        @test isempty(circuit.elements)
        @test circuit.nnodes==2
    end
    for field in (:r,:l,:c),value in (BigFloat("1e200"),BigFloat("1e-200"),
            BigFloat("0"),1//3,nextfloat(0.))
        circuit=PlanarCircuit(2,[1,2]);parameters=NamedTuple{(field,)}((value,))
        circuit_add_rlc!(circuit,1,2;parameters...)
        stored=getproperty(only(circuit.elements),field)
        @test stored==Float64(value)
        @test isfinite(stored)
        @test iszero(stored)==iszero(value)
    end
    for value in (BigFloat("1e200"),BigFloat("1e-200"),-1//3,nextfloat(0.))
        circuit=PlanarCircuit(2,[1,2]);circuit_add_transformer!(circuit,[(1,0),(2,0)],value)
        @test only(circuit.elements).ratio==Float64(value)
        @test isfinite(only(circuit.elements).ratio)
        @test !iszero(only(circuit.elements).ratio)
    end
    for field in (:r,:l,:c)
        circuit=PlanarCircuit(2,[1,2]);parameters=NamedTuple{(field,)}((BigFloat("0"),))
        circuit_add_rlc!(circuit,1,2;parameters...)
        expected=field===:c ? Matrix{ComplexF64}(I,2,2) : ComplexF64[0 1;1 0]
        @test solve_planar_circuit(circuit,1e9).s≈expected atol=1e-14
    end
    real_circuit=PlanarCircuit(2,[1,2]);big_circuit=PlanarCircuit(2,[1,2])
    circuit_add_transformer!(real_circuit,[(1,0),(2,0)],2.)
    circuit_add_transformer!(big_circuit,[(1,0),(2,0)],BigFloat("2"))
    @test solve_planar_circuit(big_circuit,1e9).s==solve_planar_circuit(real_circuit,1e9).s
end

@testset "Circuit frequencies validate stored domain before callbacks" begin
    for frequency in (BigFloat("1e1000"),BigFloat("1e-1000"),Inf,NaN,-1.)
        references=Ref(0);responses=Ref(0)
        circuit=PlanarCircuit(1,[1];z0=f->(references[]+=1;50.))
        circuit_add_network!(circuit,[1],f->(responses[]+=1;zeros(ComplexF64,1,1));format=:y)
        @test_throws ArgumentError solve_planar_circuit(circuit,frequency)
        @test references[]==responses[]==0
        @test_throws ArgumentError solve_planar_circuit(circuit,frequency;max_bytes=1)
        @test references[]==responses[]==0
    end
    for frequency in (BigFloat("1e9"),BigFloat("1e-200"),BigFloat("0"),1//3,
            1e308,nextfloat(0.))
        types=DataType[]
        circuit=PlanarCircuit(1,[1];z0=f->(push!(types,typeof(f));50.))
        circuit_add_network!(circuit,[1],f->(push!(types,typeof(f));zeros(ComplexF64,1,1));format=:y)
        result=solve_planar_circuit(circuit,frequency)
        @test result.freq==Float64(frequency)
        @test isfinite(result.freq)
        @test iszero(result.freq)==iszero(frequency)
        @test types==[Float64,Float64]
        @test result.s==ones(ComplexF64,1,1)
    end
end

@testset "Network callbacks validate converted complex storage" begin
    calls=Ref(0)
    circuit=PlanarCircuit(1,[1])
    circuit_add_network!(circuit,[1],f->fill(Complex{BigFloat}(BigFloat("1e1000"),0),1,1);
        format=:s,z0=f->(calls[]+=1;50.))
    @test_throws ArgumentError solve_planar_circuit(circuit,1e9)
    @test calls[]==0
    circuit=PlanarCircuit(1,[1])
    @test_throws ArgumentError circuit_add_network!(circuit,[1],
        fill(Complex{BigFloat}(BigFloat("1e1000"),0),1,1);format=:s)
    @test isempty(circuit.elements)
    circuit_add_network!(circuit,[1],f->fill(Complex{BigFloat}(BigFloat("0.5"),0),1,1);format=:s)
    @test only(solve_planar_circuit(circuit,BigFloat("1e9")).s)≈.5 atol=1e-14
end
end
