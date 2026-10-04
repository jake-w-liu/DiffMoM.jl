using DiffMoM, Test

@testset "triangle Fourier moments: preserve accepted small area" begin
    d, s = eps(1.), 2.0^-10
    # These exactly represented coordinates have determinant s²*d².
    # Rounded direct products cancel, but the nonzero triangle is valid.
    vertices = [0. s s*(1+d); 0. s*(1-d) s]
    mesh = PlanarConformalMesh(vertices, reshape([1,2,3], 3, 1))
    expected_area = 2.0^-125
    @test only(mesh.areas) == expected_area
    @test DiffMoM._planar_triangle_affine_fourier(vertices,
        (1.,1.,1.), 0., 0.) ≈ expected_area rtol=2eps(1.)
    @test DiffMoM._planar_triangle_affine_fourier(vertices,
        (1.,2.,3.), 0., 0.) ≈ 2expected_area rtol=2eps(1.)
end
