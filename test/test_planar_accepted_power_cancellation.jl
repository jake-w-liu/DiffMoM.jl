using Test, DiffMoM, LinearAlgebra

@testset "accepted terminal power retains finite cancellation" begin
    oracle(v,i) = setprecision(BigFloat,4352) do
        Float64(real(dot(Complex{BigFloat}.(v),Complex{BigFloat}.(i)))/2)
    end
    for (v,i) in ((ComplexF64[1e16,3,1e16],ComplexF64[1,1,-1]),
                  (ComplexF64[1e16+1e16im,3+2im,1e16+1e16im],ComplexF64[1+im,2-im,-1-im]),
                  (fill(ComplexF64(2.0^-538),8),fill(ComplexF64(2.0^-538),8)),
                  (fill(ComplexF64(3*2.0^-538),4),fill(ComplexF64(2.0^-537),4)))
        waves=planar_power_waves(v,i)
        @test waves.accepted_power == oracle(v,i)
        @test isfinite(waves.accepted_power)
    end
    moderate=planar_power_waves(ComplexF64[2+im,1-2im],ComplexF64[.2-.1im,.3+.2im];z0=[50+10im,75-5im])
    @test moderate.accepted_power ≈ .5(sum(abs2,moderate.a)-sum(abs2,moderate.b)) rtol=1e-12
    @test planar_power_waves(zeros(ComplexF64,2),ones(ComplexF64,2)).accepted_power == 0
    @test planar_power_waves(ComplexF64[1],ComplexF64[im];max_bytes=6sizeof(ComplexF64)).accepted_power == 0
    @test_throws ArgumentError planar_power_waves(ComplexF64[1e16,3,1e16],ComplexF64[1,1,-1];max_bytes=18sizeof(ComplexF64))
    v,i=ComplexF64[2+im],ComplexF64[.2+.1im]
    DiffMoM._planar_accepted_power(v,i;max_bytes=1024,retained_bytes=0)
    @test (@allocated DiffMoM._planar_accepted_power(v,i;max_bytes=1024,retained_bytes=0)) == 0
end
