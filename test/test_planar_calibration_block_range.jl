using DiffMoM,Test,LinearAlgebra

@testset "Public transfer conversions retain finite product cancellation" begin
    y=ComplexF64[1e154 -1.5e308;-1 2e154]
    t=ComplexF64[2e154 1;1.5e308 1e154]
    remainder=setprecision(BigFloat,256) do
        Float64(BigFloat(1e154)*BigFloat(2e154)-BigFloat(1.5e308))
    end
    expected_t=ComplexF64[2e154 1;remainder 1e154]
    expected_y=ComplexF64[1e154 -remainder;-1 2e154]
    # Two isolated copies use the public [left conductors; right
    # conductors] order and exercise the multiconductor block contraction.
    expand(matrix,n)=[matrix[i,j]*Matrix{ComplexF64}(I,n,n) for i in 1:2,j in 1:2]
    blocks(matrix,n)=reduce(vcat,(reduce(hcat,expand(matrix,n)[i,:]) for i in 1:2))
    for n in (1,2)
        forward=planar_y_to_abcd(blocks(y,n))
        inverse=planar_abcd_to_y(blocks(t,n))
        @test all(isfinite,forward) && all(isfinite,inverse)
        @test forward≈blocks(expected_t,n) rtol=2e-15
        @test inverse≈blocks(expected_y,n) rtol=2e-15
    end
    ordinary=ComplexF64[3 .2 -.8 .1;.2 2 .1 -.7;-.8 .1 3 .2;.1 -.7 .2 2]
    @test planar_abcd_to_y(planar_y_to_abcd(ordinary))≈ordinary rtol=1e-13
    @test_throws ArgumentError planar_abcd_to_y(ComplexF64[1e308 1;0 1e308])
end
