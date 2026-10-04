using Test, DiffMoM, LinearAlgebra

@testset "Pure capacitor MNA retains bounded constitutive rows" begin
    for topology in (:series,:parallel),(f,c) in ((300e6,nextfloat(0.)),(floatmax(Float64),nextfloat(0.)),
            (nextfloat(0.),floatmax(Float64)),(1e308,1e-300),
            (1e-300,floatmax(Float64)),(floatmax(Float64),floatmax(Float64)))
        circuit=PlanarCircuit(1,[1]);circuit_add_rlc!(circuit,1,0;c,topology)
        result=solve_planar_circuit(circuit,f)
        admittance=im*(2big(pi)*BigFloat(f)*BigFloat(c))
        expected=ComplexF64((1-50admittance)/(1+50admittance))
        @test result.s[1]≈expected rtol=8eps(Float64)
        @test all(isfinite,result.s) && all(isfinite,result.currents)
        if f==300e6
            @test result.y!==nothing
            @test abs(result.y[1]-ComplexF64(admittance))<=16nextfloat(0.)
            @test !iszero(imag(result.s[1]))
        end
    end
    for topology in (:series,:parallel),(f,c) in ((0.,nextfloat(0.)),(300e6,0.))
        circuit=PlanarCircuit(1,[1]);circuit_add_rlc!(circuit,1,0;c,topology)
        @test solve_planar_circuit(circuit,f).s≈ones(ComplexF64,1,1) rtol=2eps(Float64)
    end
    circuit=PlanarCircuit(2,[(1,2)]);circuit_add_rlc!(circuit,1,2;c=nextfloat(0.))
    @test solve_planar_circuit(circuit,300e6;floating_gauge=:auto).s[1]≈1 rtol=4eps(Float64)
    @test_throws ArgumentError solve_planar_circuit(circuit,300e6;max_bytes=1)
end
