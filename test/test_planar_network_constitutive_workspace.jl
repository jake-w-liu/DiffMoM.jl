module PlanarNetworkConstitutiveWorkspaceTests
using DiffMoM, Test, LinearAlgebra, Random

@testset "Constitutive network conversion preserves physical source equations" begin
    rng=MersenneTwister(122)
    for n in (1,2,8,32)
        E=Matrix{ComplexF64}(I,n,n)
        F=-((randn(rng,n,n)+im*randn(rng,n,n))+(30+10im)*I)
        refs=20 .+80rand(rng,n)
        original_E,original_F,original_refs=copy(E),copy(F),copy(refs)
        # Solve terminal currents and voltages with independent Thevenin
        # wave sources rather than forming production wave coefficients.
        system=[E F;Matrix{ComplexF64}(I,n,n) Diagonal(refs)]
        vi=system\[zeros(ComplexF64,n,n);Diagonal(2sqrt.(refs))]
        expected=Diagonal(inv.(2sqrt.(refs)))*(vi[1:n,:]-Diagonal(refs)*vi[n+1:2n,:])
        actual=DiffMoM._network_constitutive_s(E,F,refs)
        @test actual≈expected atol=2e-14 rtol=2e-13
        @test actual isa Matrix{ComplexF64}
        @test E==original_E && F==original_F && refs==original_refs
    end
    E=ComplexF64[1 -1;0 0];F=ComplexF64[0 0;1 1]
    @test DiffMoM._network_constitutive_s(E,F,[50.,50.])≈ComplexF64[0 1;1 0] atol=2e-15
    @test_throws SingularException DiffMoM._network_constitutive_s(ones(ComplexF64,1,1),
        fill(50.0+0im,1,1),[50.])
end
end
