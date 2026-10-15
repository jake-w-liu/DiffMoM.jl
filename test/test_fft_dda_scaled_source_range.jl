using DiffMoM,Test,LinearAlgebra,StaticArrays

@testset "FFT interactions retain unrepresentable scaled source moments" begin
    grid=VoxelGrid3D((0.,2.),(0.,1.),(0.,1.),2,1,1)
    for epsr in (nextfloat(1.),Diagonal(ComplexF64[nextfloat(1.),nextfloat(1.),1.])), em in (false,true)
        direct=em ? em_dda_operator_3d(grid,1e150,epsr,1.) : dda_operator_3d(grid,1e150,epsr)
        fft=em ? fft_em_dda_operator_3d(grid,1e150,epsr,1.) : fft_dda_operator_3d(grid,1e150,epsr)
        x=zeros(ComplexF64,size(direct,2));x[em ? 8 : 5]=1e-308
        reference=Matrix(direct)*x
        @test abs(reference[2])>1e-26
        @test direct*x ≈ reference rtol=1e-12
        @test fft*x ≈ reference rtol=1e-12
        aliased=copy(x);mul!(aliased,fft,aliased)
        @test aliased ≈ reference rtol=1e-12
        y=ones(ComplexF64,length(x));mul!(y,fft,x,2.,3.)
        @test y ≈ 2reference .+ 3 rtol=1e-12
        # The exceptional path releases the shared work lock for later calls.
        ordinary=ones(ComplexF64,length(x))
        @test fft*ordinary ≈ direct*ordinary rtol=1e-12
    end
end
