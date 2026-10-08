module RetainedSelectorQuantizationTests
using DiffMoM, LinearAlgebra, Random, Test

@testset "Retained wave selector recovers coupled S quantization" begin
    rng = MersenneTwister(10326) # Original power-wave fixture.
    for n in 1:5
        r, x = randn(rng,n,n), randn(rng,n,n)
        base = .01*(r'*r+I)+im*.01*(x+x')
        refs = ComplexF64.(20 .+80rand(rng,n).+im*(60rand(rng,n).-30))
        a = randn(rng,ComplexF64,n)
        for multiplier in (1.0,inv(sqrt(eps(Float64))),inv(eps(Float64)),inv(eps(Float64)^2))
            Y = multiplier*base
            S = planar_y_to_s(Y,refs)
            saved = copy(Y),copy(S),copy(a),copy(refs)
            reference,derivative = setprecision(BigFloat,2precision(Float64)) do
                yy,zz,aa = Complex{BigFloat}.(Y),Complex{BigFloat}.(refs),Complex{BigFloat}.(a)
                matrix = Matrix{Complex{BigFloat}}(I,n,n)+Diagonal(zz)*yy
                voltage = matrix\(2sqrt.(real.(zz)).*aa)
                slope = -(matrix\(Diagonal(zz)*yy*voltage))
                ComplexF64.(voltage),ComplexF64.(slope)
            end
            voltage = @inferred DiffMoM._planar_wave_voltage_retained(Y,S,a,refs)
            incident = @inferred DiffMoM._planar_incident_admittance(Y,voltage,refs)
            @test isapprox(voltage,reference;rtol=3e-14,atol=0)
            @test isapprox(incident,a;rtol=3e-14,atol=0)
            step = cbrt(eps(Float64)) # Second-order central-difference balance.
            positive,negative = (1+step)*Y,(1-step)*Y
            slope = (DiffMoM._planar_wave_voltage_retained(positive,planar_y_to_s(positive,refs),a,refs)-
                     DiffMoM._planar_wave_voltage_retained(negative,planar_y_to_s(negative,refs),a,refs))/(2step)
            @test isapprox(slope,derivative)
            @test (Y,S,a,refs) == saved
            @test all(isfinite,voltage) && all(isfinite,incident) && all(isfinite,slope)
        end
    end
end
end
