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

# Subdivided icosahedron projected to a sphere; faces oriented outward.
function _icosphere_mesh(radius::Float64, nsub::Int)
    t = (1 + sqrt(5)) / 2
    verts = Vec3[
        Vec3(-1, t, 0), Vec3(1, t, 0), Vec3(-1, -t, 0), Vec3(1, -t, 0),
        Vec3(0, -1, t), Vec3(0, 1, t), Vec3(0, -1, -t), Vec3(0, 1, -t),
        Vec3(t, 0, -1), Vec3(t, 0, 1), Vec3(-t, 0, -1), Vec3(-t, 0, 1),
    ]
    faces = [
        (1,12,6),(1,6,2),(1,2,8),(1,8,11),(1,11,12),
        (2,6,10),(6,12,5),(12,11,3),(11,8,7),(8,2,9),
        (4,10,5),(4,5,3),(4,3,7),(4,7,9),(4,9,10),
        (5,10,6),(3,5,12),(7,3,11),(9,7,8),(10,9,2),
    ]
    vlist = [v / norm(v) for v in verts]
    vindex = Dict{Vec3,Int}()
    for (i, v) in enumerate(vlist)
        vindex[v] = i
    end
    function midpoint(a::Vec3, b::Vec3)
        m = (a + b) / 2
        m = m / norm(m)
        haskey(vindex, m) && return vindex[m]
        push!(vlist, m)
        vindex[m] = length(vlist)
        return length(vlist)
    end
    for _ in 1:nsub
        newfaces = NTuple{3,Int}[]
        for (i1, i2, i3) in faces
            a, b, c = vlist[i1], vlist[i2], vlist[i3]
            m12 = midpoint(a, b); m23 = midpoint(b, c); m31 = midpoint(c, a)
            push!(newfaces, (i1, m12, m31))
            push!(newfaces, (i2, m23, m12))
            push!(newfaces, (i3, m31, m23))
            push!(newfaces, (m12, m23, m31))
        end
        faces = newfaces
    end
    Nv = length(vlist)
    xyz = zeros(3, Nv)
    for (i, v) in enumerate(vlist)
        xyz[:, i] = radius .* v
    end
    tri = zeros(Int, 3, length(faces))
    for (tt, f) in enumerate(faces)
        i1, i2, i3 = f
        a = Vec3(xyz[:, i1]...); b = Vec3(xyz[:, i2]...); c = Vec3(xyz[:, i3]...)
        n = cross(b - a, c - a)
        center = (a + b + c) / 3
        if dot(n, center) < 0
            i2, i3 = i3, i2
        end
        tri[:, tt] = [i1, i2, i3]
    end
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
    # :muller assembles a different (second-kind) system than PMCHWT (it carries the
    # nhat x Gram identity term); the PMCHWT-vs-Muller currents agreement is checked in
    # its own testset below.
    A_mu = assemble_muller_3d(mesh, rwg, k0, eps_in;
                              mur_in=mu_in,
                              quad_order=1,
                              singular_quad_order=3)
    @test size(A_pm) == (2N, 2N)
    @test size(A_pm_mf) == (2N, 2N)
    @test size(A_mu) == (2N, 2N)
    @test all(isfinite, real.(A_pm))
    @test all(isfinite, imag.(A_pm))
    @test all(isfinite, real.(A_mu))
    @test norm(A_pm - A_mu) / norm(A_pm) > 1e-4   # distinct formulation
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
    # :muller plane-wave solve produces a consistent (weighted) second-kind system.
    res_pw_mu = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, pw;
                                        mur_in=mu_in,
                                        formulation=:muller,
                                        quad_order=1,
                                        singular_quad_order=3)
    @test res_pw_mu.formulation == :muller
    @test norm(res_pw_mu.A * vcat(res_pw_mu.J, res_pw_mu.M) - res_pw_mu.rhs) /
          max(norm(res_pw_mu.rhs), eps()) < 1e-10

    plate = make_rect_plate(1.0, 1.0, 1, 1)
    plate_rwg = build_rwg(plate)
    @test_throws ErrorException assemble_pmchwt_3d(plate, plate_rwg, k0, eps_in)
    @test_throws ErrorException assemble_dielectric_sie_3d(mesh, rwg, k0, eps_in;
                                                           formulation=:cfie)
end

@testset "PMCHWT vs Muller currents agree (dielectric sphere)" begin
    # Decisive Muller oracle: PMCHWT and Muller discretize the same boundary
    # value problem, so the surface currents J, M must match. This passes only
    # when the off-diagonal K blocks are mu/eps-weighted, the RHS is scaled by
    # the exterior row weights, and the second-kind (nhat x Gram) identity term
    # is included on the off-diagonal. Without the identity term the mismatch is
    # ~20-50% (or >100% for the H current); with it the agreement is <1% and
    # tightens under mesh refinement.
    mesh = _icosphere_mesh(1.0, 1)
    rwg = build_rwg(mesh; allow_boundary=false, require_closed=true)
    k0 = 1.0
    pw = make_plane_wave(Vec3(0.0, 0.0, k0), 1.0, Vec3(1.0, 0.0, 0.0))

    for (eps_in, mu_in) in ((2.5 + 0.0im, 1.0 + 0.0im), (2.5 + 0.0im, 1.6 + 0.0im))
        res_pm = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, pw;
                                         mur_in=mu_in, formulation=:pmchwt,
                                         quad_order=3, singular_quad_order=7)
        res_mu = solve_dielectric_sie_3d(mesh, rwg, k0, eps_in, pw;
                                         mur_in=mu_in, formulation=:muller,
                                         quad_order=3, singular_quad_order=7)
        relJ = norm(res_mu.J - res_pm.J) / norm(res_pm.J)
        relM = norm(res_mu.M - res_pm.M) / norm(res_pm.M)
        @test relJ < 0.01
        @test relM < 0.01

        # Muller RHS is scaled by the exterior row weights, so the solved
        # currents must satisfy the (weighted) Muller system to solver tolerance.
        @test norm(res_mu.A * vcat(res_mu.J, res_mu.M) - res_mu.rhs) /
              max(norm(res_mu.rhs), eps()) < 1e-10

        # Dense and matrix-free Muller operators must be identical.
        N = rwg.nedges
        A_mu = assemble_muller_3d(mesh, rwg, k0, eps_in; mur_in=mu_in,
                                  quad_order=3, singular_quad_order=7)
        A_mu_mf = matrixfree_dielectric_sie_operator_3d(mesh, rwg, k0, eps_in;
                      mur_in=mu_in, formulation=:muller,
                      quad_order=3, singular_quad_order=7)
        @test norm(Matrix(A_mu_mf) - A_mu) / norm(A_mu) < 1e-13
        xv = ComplexF64[sin(0.11 * i) + 1im * cos(0.07 * i) for i in 1:2N]
        @test norm(A_mu_mf * xv - A_mu * xv) / norm(A_mu * xv) < 1e-13
    end
end

println("  PASS")
