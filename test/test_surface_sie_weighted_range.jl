using DiffMoM, LinearAlgebra, Test

@testset "Dense Muller regional products retain finite cancellation" begin
    vertices = Vec3[Vec3(1, 1, 1), Vec3(-1, -1, 1),
                    Vec3(-1, 1, -1), Vec3(1, -1, -1)]
    faces = ((1, 2, 3), (1, 4, 2), (1, 3, 4), (2, 4, 3))
    triangles = zeros(Int, 3, 4)
    for (triangle, face) in enumerate(faces)
        indices = collect(face)
        a, b, c = vertices[indices[1]], vertices[indices[2]], vertices[indices[3]]
        if dot(cross(b - a, c - a), (a + b + c) / 3) < 0
            indices[2], indices[3] = indices[3], indices[2]
        end
        triangles[:, triangle] = indices
    end
    mesh = TriMesh(hcat(vertices...), triangles)
    rwg = build_rwg(mesh)
    options = (mur_in=complex(-1e308, 1e292), mur_ext=1e308,
               formulation=:muller, eta0=1.0)
    operator = matrixfree_dielectric_sie_operator_3d(mesh, rwg, 1e-296, 1.0; options...)
    dense = assemble_dielectric_sie_3d(mesh, rwg, 1e-296, 1.0; options...)
    expected = setprecision(BigFloat, 4352) do
        ComplexF64(Complex{BigFloat}(operator.c_ze_ext) *
                   Complex{BigFloat}(operator.Ze_ext[1, 1]) +
                   Complex{BigFloat}(operator.c_ze_int) *
                   Complex{BigFloat}(operator.Ze_int[1, 1]))
    end
    @test isfinite(expected)
    @test !isfinite(operator.c_ze_ext * operator.Ze_ext[1, 1])
    @test dense[1, 1] == expected
    @test all(isfinite, dense)
    # The full assembled path must agree with independently indexed regional
    # operators, including the off-diagonal principal-value/Gram blocks.
    @test dense ≈ Matrix(operator) rtol=16eps(Float64)

    # Half the difference is finite even though the unscaled difference is
    # not. Keep this identity coefficient and its local Gram contribution
    # consistent in dense and matrix-free constructors.
    small_mesh = TriMesh(0.1 * mesh.xyz, copy(mesh.tri))
    small_rwg = build_rwg(small_mesh)
    gram_options = (mur_in=complex(-1e308, -1.0), mur_ext=1e308,
                    formulation=:muller, eta0=1e-308)
    gram_operator = matrixfree_dielectric_sie_operator_3d(
        small_mesh, small_rwg, 1e-154, 1.0; gram_options...)
    gram_dense = assemble_dielectric_sie_3d(
        small_mesh, small_rwg, 1e-154, 1.0; gram_options...)
    gram_coefficient = setprecision(BigFloat, 4352) do
        ComplexF64((-Complex{BigFloat}(gram_operator.c_ze_ext) +
                    Complex{BigFloat}(gram_operator.c_ze_int)) / 2)
    end
    @test !isfinite(gram_operator.c_ze_ext - gram_operator.c_ze_int)
    @test isfinite(gram_coefficient)
    @test gram_operator.c_g_e == gram_coefficient
    @test all(isfinite, gram_dense)
    @test gram_dense ≈ Matrix(gram_operator) rtol=32eps(Float64)
end
