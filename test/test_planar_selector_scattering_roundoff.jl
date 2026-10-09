module StoredScatteringRoundoffTests
using DiffMoM,LinearAlgebra,Test

@testset "Stored scattering retains ordinary mapping within original wave accuracy" begin
    # Exact observed macOS operands: current main run37871699740, failed
    # job113631041093; arithmetic diagnostic run37876311027. These stored
    # bits are measured regression data, not selected tuning parameters.
    number(bits)=reinterpret(Float64,bits)
    Y=reshape(ComplexF64[complex(number(0x3fcb75332d7453b2),number(0xbff28d45a395759e))],1,1)
    recorded=complex(number(0xbfefc9165ab289ce),number(0x3fa0f8021db4c60b))
    refs=ComplexF64[number(0x4049000000000000)]
    a=ComplexF64[1+.2im] # Original radiation fixture excitation.
    # Adjacent representable scattering coordinates exercise the rounding
    # boundary independently of architecture-specific assembly arithmetic.
    for shift_real in (prevfloat,identity,nextfloat),shift_imag in (prevfloat,identity,nextfloat)
        S=reshape(ComplexF64[complex(shift_real(real(recorded)),shift_imag(imag(recorded)))],1,1)
        saved=copy(Y),copy(S),copy(a),copy(refs)
        declared=planar_wave_voltages(S,a;z0=refs)
        selected=@inferred DiffMoM._planar_wave_voltage_retained(Y,S,a,refs)
        incident=@inferred DiffMoM._planar_incident_admittance(Y,selected,refs)
        @test selected==declared
        @test isapprox(incident,a;rtol=3e-14,atol=0) # Existing independent Kurokawa gate.
        @test all(isfinite,selected) && all(isfinite,incident)
        @test (Y,S,a,refs)==saved
        @test selected!==a && selected!==refs
        @test eltype(selected)===ComplexF64
    end
end
end
