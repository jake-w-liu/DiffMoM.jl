# Test 49: Dielectric 3D surface integral equations

using Test
using LinearAlgebra

if isdefined(Main, :DiffMoM)
    using .DiffMoM
else
    using DiffMoM
end

println("\n── Test 49: Dielectric 3D SIE assembly/solve ──")

function _oriented_tetrahedron_mesh()
    verts = Vec3[
        Vec3(1.0, 1.0, 1.0),
        Vec3(-1.0, -1.0, 1.0),
        Vec3(-1.0, 1.0, -1.0),
        Vec3(1.0, -1.0, -1.0),
    ]
    faces = [(1, 2, 3), (1, 4, 2), (1, 3, 4), (2, 4, 3)]
    tri = zeros(Int, 3, length(faces))
    for (t, f) in enumerate(faces)
        inds = collect(f)
        a, b, c = verts[inds[1]], verts[inds[2]], verts[inds[3]]
        n = cross(b - a, c - a)
        center = (a + b + c) / 3
        if dot(n, center) < 0
            inds[2], inds[3] = inds[3], inds[2]
        end
        tri[:, t] .= inds
    end
    xyz = hcat(verts...)
    return TriMesh(xyz, tri)
end

@testset "Dielectric 3D SIE assembly/solve" begin
    mesh = _oriented_tetrahedron_mesh()
    rwg = build_rwg(mesh; allow_boundary=false, require_closed=true)
    N = rwg.nedges
    k0 = 0.7
    eps_in = 2.2 - 0.03im
    mu_in = 1.3 - 0.02im

    K = assemble_magnetic_field_operator_3d(mesh, rwg, k0; quad_order=1)
    K_mf = matrixfree_magnetic_field_operator_3d(mesh, rwg, k0; quad_order=1)
    @test size(K) == (N, N)
    @test size(K_mf) == (N, N)
    @test all(isfinite, real.(K))
    @test all(isfinite, imag.(K))
    xk = ComplexF64[sin(0.2 * i) + 1im * cos(0.17 * i) for i in 1:N]
    yk = zeros(ComplexF64, N)
    mul!(yk, K_mf, xk)
    @test norm(yk - K * xk) / max(norm(K * xk), eps()) < 1e-13

    A_pm = assemble_pmchwt_3d(mesh, rwg, k0, eps_in;
                              mur_in=mu_in,
                              quad_order=1,
                              singular_quad_order=3)
    A_pm_mf = matrixfree_dielectric_sie_operator_3d(mesh, rwg, k0, eps_in;
                                                    mur_in=mu_in,
                                                    formulation=:pmchwt,
                                                    quad_order=1,
                                                    singular_quad_order=3)
    # The :muller formulation is disabled (it produced currents inconsistent with
    # PMCHWT — see SurfaceIE3D); it must raise a clear error rather than assemble.
    @test_throws ErrorException assemble_muller_3d(mesh, rwg, k0, eps_in;
                                                   mur_in=mu_in,
                                                   quad_order=1,
                                                   singular_quad_order=3)
    @test size(A_pm) == (2N, 2N)
    @test size(A_pm_mf) == (2N, 2N)
    @test all(isfinite, real.(A_pm))
    @test all(isfinite, imag.(A_pm))
    @test norm(Matrix(A_pm_mf) - A_pm) / norm(A_pm) < 1e-13

    x = ComplexF64[sin(0.11 * i) + 1im * cos(0.07 * i) for i in 1:2N]
    y_mf = zeros(ComplexF64, 2N)
    mul!(y_mf, A_pm_mf, x)
    @test norm(y_mf - A_pm * x) / norm(A_pm * x) < 1e-13

    rhs0 = zeros(ComplexF64, 2N)
    res0 = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, rhs0;
                                   mur_in=mu_in,
                                   formulation=:pmchwt,
                                   quad_order=1,
                                   singular_quad_order=3)
    @test norm(res0.J) < 1e-13
    @test norm(res0.M) < 1e-13
    @test norm(res0.A * vcat(res0.J, res0.M) - res0.rhs) < 1e-13

    rhs = ComplexF64[sin(0.13 * i) - 0.2im * cos(0.19 * i) for i in 1:2N]
    res_direct = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, rhs;
                                         mur_in=mu_in,
                                         formulation=:pmchwt,
                                         quad_order=1,
                                         singular_quad_order=3)
    res_gmres = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, rhs;
                                        mur_in=mu_in,
                                        formulation=:pmchwt,
                                        solver=:gmres,
                                        quad_order=1,
                                        singular_quad_order=3,
                                        tol=1e-12,
                                        maxiter=50)
    x_direct = vcat(res_direct.J, res_direct.M)
    x_gmres = vcat(res_gmres.J, res_gmres.M)
    @test res_gmres.A isa MatrixFreeDielectricSIE3D
    @test res_gmres.A_LU === nothing
    @test norm(x_gmres - x_direct) / max(norm(x_direct), eps()) < 1e-9

    pw = make_plane_wave(Vec3(0.0, 0.0, k0), 1.0, Vec3(1.0, 0.0, 0.0))
    res_pw = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, pw;
                                     mur_in=mu_in,
                                     formulation=:pmchwt,
                                     quad_order=1,
                                     singular_quad_order=3)
    @test res_pw.formulation == :pmchwt
    @test norm(res_pw.rhs) > 0
    @test norm(res_pw.A * vcat(res_pw.J, res_pw.M) - res_pw.rhs) /
          max(norm(res_pw.rhs), eps()) < 1e-10
    # :muller is disabled and must error rather than return wrong currents.
    @test_throws ErrorException solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, pw;
                                                        mur_in=mu_in,
                                                        formulation=:muller,
                                                        quad_order=1,
                                                        singular_quad_order=3)

    plate = make_rect_plate(1.0, 1.0, 1, 1)
    plate_rwg = build_rwg(plate)
    @test_throws ErrorException assemble_pmchwt_3d(plate, plate_rwg, k0, eps_in)
    @test_throws ErrorException assemble_dielectric_sie_3d(mesh, rwg, k0, eps_in;
                                                           formulation=:cfie)
end

println("  PASS")
