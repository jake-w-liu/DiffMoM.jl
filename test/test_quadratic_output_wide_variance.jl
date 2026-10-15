using DiffMoM,LinearAlgebra,Test

@testset "Wide Gaussian factors retain small indefinite quadratic variance" begin
    narrow=ones(ComplexF64,2,1)
    repeated=fill(.5+0im,2,4)
    # Exact dyadic/unitary changes of latent coordinates preserve the law.
    phased=repeated*Diagonal(ComplexF64[1,im,-1,-im])
    hadamard=[1 1 1 1;1 -1 1 -1;1 1 -1 -1;1 -1 -1 1]/2
    concentrated=repeated*hadamard
    for exponent in (-60,-100),amplitude in (0.0+0im,.3+.2im)
        delta=2.0^exponent;weight=ComplexF64[1 delta;delta -1]
        mean=fill(amplitude,2)
        expected=(mean=2delta*(1+abs2(amplitude)),
            variance=4delta^2*(1+2abs2(amplitude)))
        for factor in (narrow,repeated,phased,concentrated)
            outputs=ErrorOutputs(mean,ones(ComplexF64,2,2),factor,copy(mean),(;))
            moments=quadratic_output_moments(outputs,weight)
            @test moments.mean ≈ expected.mean rtol=32eps(Float64)
            @test moments.variance ≈ expected.variance rtol=32eps(Float64)
        end
    end

    # A caller-owned wider dyadic weight can retain a variance below the
    # Float64 exponent range; both its significand and exponent must survive.
    weight,delta=setprecision(BigFloat,6144) do
        d=BigFloat(2)^(-5000)
        ([BigFloat(1) d;d BigFloat(-1)],d)
    end
    caller_precision=precision(BigFloat)
    for factor in (narrow,repeated)
        outputs=ErrorOutputs(zeros(ComplexF64,2),ones(ComplexF64,2,2),factor,zeros(ComplexF64,2),(;))
        moments=quadratic_output_moments(outputs,weight)
        @test moments.mean == 2delta
        @test moments.variance == 4delta^2
        @test precision(BigFloat)==caller_precision
    end
    outputs=ErrorOutputs(zeros(ComplexF64,2),ones(ComplexF64,2,2),repeated,zeros(ComplexF64,2),(;))
    @test_throws ArgumentError quadratic_output_moments(outputs,1e300Matrix{Float64}(I,2,2))
end
