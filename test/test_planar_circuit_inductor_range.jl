module PlanarCircuitInductorRangeTests
using Test, DiffMoM

function reflection_oracle(f,parameters,topology,z0)
    w=2big(pi)*BigFloat(f);ref=Complex{BigFloat}(z0)
    r=get(parameters,:r,nothing);l=get(parameters,:l,nothing);c=get(parameters,:c,nothing)
    if topology===:series
        c!==nothing && (iszero(c) || iszero(f)) && return ComplexF64(1)
        z=(r===nothing ? 0 : BigFloat(r))+(l===nothing ? 0 : im*w*BigFloat(l))+
            (c===nothing ? 0 : inv(im*w*BigFloat(c)))
        return ComplexF64((z-conj(ref))/(z+ref))
    end
    if (r!==nothing && iszero(r)) || (l!==nothing && (iszero(l) || iszero(f)))
        return ComplexF64(-conj(ref)/ref)
    end
    y=(r===nothing ? 0 : inv(BigFloat(r)))+(l===nothing ? 0 : inv(im*w*BigFloat(l)))+
        (c===nothing ? 0 : im*w*BigFloat(c))
    return ComplexF64((1-conj(ref)*y)/(1+ref*y))
end

@testset "RLC rows retain finite power waves across product and reciprocal ranges" begin
    cases=((1e308,(l=1e-308,)),(floatmax(Float64),(l=1e-300,)),
        (floatmax(Float64),(l=floatmax(Float64),)),
        (1e308,(r=20.,l=1e-308)),(1e308,(r=20.,c=1e-308)),
        (1e308,(r=20.,l=1e-308,c=1e-308)),
        (1e9,(r=nextfloat(0.),)),(1e9,(l=nextfloat(0.),)),
        (1e-308,(l=1e-308,)),(1e-308,(r=20.,l=1e-308)),
        (1e-308,(r=20.,c=1e-308)),(nextfloat(0.),(l=nextfloat(0.),)),
        (nextfloat(0.),(r=20.,l=nextfloat(0.),c=nextfloat(0.))),
        (1e9,(r=floatmax(Float64),)),(0.,(l=1e-9,)),
        (0.,(r=20.,l=1e-9,c=1e-12)),(1e9,(r=0.,l=1e-9)),
        (1e9,(r=20.,l=0.,c=0.)))
    for (f,parameters) in cases,topology in (:series,:parallel),z0 in (50.,50. + 20im)
        circuit=PlanarCircuit(1,[1];z0)
        circuit_add_rlc!(circuit,1,0;parameters...,topology)
        result=solve_planar_circuit(circuit,f)
        @test only(result.s)≈reflection_oracle(f,parameters,topology,z0) atol=16eps(Float64) rtol=16eps(Float64)
        @test all(isfinite,result.s) && all(isfinite,result.currents) && all(isfinite,result.voltages)
    end
    # The imaginary reflection for a subnormal inductor remains observable,
    # even though its inverse impedance exceeds Float64's finite range.
    for topology in (:series,:parallel)
        circuit=PlanarCircuit(1,[1]);circuit_add_rlc!(circuit,1,0;l=nextfloat(0.),topology)
        expected=reflection_oracle(1e9,(l=nextfloat(0.),),topology,50.)
        actual=only(solve_planar_circuit(circuit,1e9).s)
        @test !iszero(imag(actual))
        @test abs(imag(actual)-imag(expected))<=64nextfloat(0.)
    end
end

@testset "RLC range scaling retains ordinary RF behavior and floating modes" begin
    for f in (0.,1e6,1e9,100e9),topology in (:series,:parallel),parameters in
            ((r=20.,),(l=1e-9,),(c=1e-12,),(r=20.,l=1e-9),
             (r=20.,c=1e-12),(l=1e-9,c=1e-12),(r=20.,l=1e-9,c=1e-12))
        circuit=PlanarCircuit(2,[(1,2)];z0=50. + 20im)
        circuit_add_rlc!(circuit,1,2;parameters...,topology)
        result=solve_planar_circuit(circuit,f;floating_gauge=:auto)
        @test only(result.s)≈reflection_oracle(f,parameters,topology,50. + 20im) atol=16eps(Float64) rtol=16eps(Float64)
        @test length(result.gauge_nodes)==1
    end
    circuit=PlanarCircuit(1,[1]);circuit_add_rlc!(circuit,1,0;r=20.,l=1e-308)
    @test_throws ArgumentError solve_planar_circuit(circuit,1e308;max_bytes=1)
    element=only(circuit.elements)
    DiffMoM._circuit_rlc_coefficients(element,1e308)
    @test (@allocated DiffMoM._circuit_rlc_coefficients(element,1e308))==0
    # An unexcited series branch beyond the representable impedance range
    # imposes zero current; both arbitrary endpoint coordinates remain free.
    for topology in (:series,:parallel)
        open=PlanarCircuit(4,[1]);circuit_add_rlc!(open,1,0;r=50.)
        circuit_add_rlc!(open,3,4;l=floatmax(Float64),topology)
        result=solve_planar_circuit(open,floatmax(Float64);floating_gauge=:auto)
        @test abs(only(result.s))<=4eps(Float64)
        @test result.gauge_nodes==[2,3,4]
        @test all(isfinite,result.voltages) && all(isfinite,result.currents)
    end
end

@testset "Circuit results omit nonrepresentable admittance" begin
    for topology in (:series,:parallel)
        circuit=PlanarCircuit(1,[1]);circuit_add_rlc!(circuit,1,0;r=1e-310,topology)
        result=solve_planar_circuit(circuit,1e9)
        @test result.s≈fill(-1.,1,1) atol=8eps(Float64)
        @test result.y===nothing
        @test all(isfinite,result.voltages) && all(isfinite,result.currents)
    end
end
end
