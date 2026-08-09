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

function _near_pair_query_checksum(near_pairs)
    Nt = size(near_pairs, 1)
    total = 0
    @inbounds for i in 1:10_000
        t1 = mod1(37i, Nt)
        t2 = mod1(101i + 7, Nt)
        total += near_pairs[t1, t2]
    end
    return total
end

function _near_pair_reference(mesh)
    Nt = ntriangles(mesh)
    reference = falses(Nt, Nt)
    for t1 in 1:Nt, t2 in 1:Nt
        t1 == t2 && continue
        reference[t1, t2] = any(
            mesh.tri[v1, t1] == mesh.tri[v2, t2]
            for v1 in 1:3 for v2 in 1:3
        )
    end
    return reference
end

function _test_shared_sie_operator_concurrency(A, dense_A, inputs)
    references = [dense_A * input for input in inputs]
    Threads.nthreads() > 1 || return

    for _ in 1:4
        gate = Base.Event()
        tasks = map(eachindex(inputs)) do i
            Threads.@spawn begin
                wait(gate)
                A * inputs[i]
            end
        end
        yield()
        notify(gate)
        results = fetch.(tasks)
        @test results ≈ references rtol=1e-12
    end
end

@testset "Dielectric 3D SIE assembly/solve" begin
    mesh = _oriented_tetrahedron_mesh()
    rwg = build_rwg(mesh; allow_boundary=false, require_closed=true)
    N = rwg.nedges
    k0 = 0.7
    eps_in = 2.2 - 0.03im
    mu_in = 1.3 - 0.02im

    @test_throws ErrorException dielectric_medium_3d(Inf)
    @test_throws ErrorException dielectric_medium_3d(k0; eta0=Inf)
    large_balanced = dielectric_medium_3d(
        k0, 1e308 + 0im, 1e308 + 0im)
    small_balanced = dielectric_medium_3d(
        k0, 1e-308 + 0im, 1e-308 + 0im)
    @test large_balanced.k == ComplexF64(k0 * 1e308)
    @test small_balanced.k == ComplexF64(k0 * 1e-308)
    @test large_balanced.eta == ComplexF64(DiffMoM._ETA0_DDA)
    @test small_balanced.eta == ComplexF64(DiffMoM._ETA0_DDA)

    high_eps = dielectric_medium_3d(k0, 1e300 + 0im, 1e-300 + 0im)
    high_mu = dielectric_medium_3d(k0, 1e-300 + 0im, 1e300 + 0im)
    low_eta_ref, high_eta_ref = setprecision(BigFloat, 256) do
        eta0_b = BigFloat(DiffMoM._ETA0_DDA)
        return ComplexF64(eta0_b * sqrt(BigFloat(1e-300) / BigFloat(1e300))),
               ComplexF64(eta0_b * sqrt(BigFloat(1e300) / BigFloat(1e-300)))
    end
    @test high_eps.k == ComplexF64(k0)
    @test high_mu.k == ComplexF64(k0)
    @test high_eps.eta == low_eta_ref
    @test high_mu.eta == high_eta_ref

    K = assemble_magnetic_field_operator_3d(mesh, rwg, k0; quad_order=1)
    K_mf = matrixfree_magnetic_field_operator_3d(mesh, rwg, k0; quad_order=1)
    @test size(K) == (N, N)
    @test size(K_mf) == (N, N)
    @test size(K_mf, 3) == 1
    @test_throws BoundsError size(K_mf, 0)
    @test all(isfinite, real.(K))
    @test all(isfinite, imag.(K))
    xk = ComplexF64[sin(0.2 * i) + 1im * cos(0.17 * i) for i in 1:N]
    yk = zeros(ComplexF64, N)
    mul!(yk, K_mf, xk)
    @test (@allocated mul!(yk, K_mf, xk)) < 1024
    @test norm(yk - K * xk) / max(norm(K * xk), eps()) < 1e-13

    overlap_storage = vcat(xk, 0.0 + 0im)
    overlap_x = view(overlap_storage, 1:length(xk))
    overlap_y = view(overlap_storage, 2:(length(xk) + 1))
    overlap_expected = K * copy(overlap_x)
    mul!(overlap_y, K_mf, overlap_x)
    @test overlap_y ≈ overlap_expected rtol=1e-13

    # The near-singular relation is an exact compact Boolean matrix: every
    # distinct triangle pair sharing at least one vertex is present.
    near_ref = _near_pair_reference(mesh)
    @test Matrix(K_mf.near_pairs) == near_ref
    @test length(K_mf.near_pairs.neighbors) == count(near_ref)
    @test_throws BoundsError K_mf.near_pairs[0, 1]
    _near_pair_query_checksum(K_mf.near_pairs)
    @test @allocated(_near_pair_query_checksum(K_mf.near_pairs)) == 0

    # Persistent storage must track the local near-pair count, not Nt².
    mesh_small = make_rect_plate(1.0, 1.0, 10, 10)
    near_small = DiffMoM._triangle_near_pairs_3d(
        mesh_small)
    near_large = DiffMoM._triangle_near_pairs_3d(
        make_rect_plate(1.0, 1.0, 40, 40))
    @test Matrix(near_small) == _near_pair_reference(mesh_small)
    @test Base.summarysize(near_large) < 25 * Base.summarysize(near_small)

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
    @test size(A_pm_mf, 3) == 1
    @test_throws BoundsError size(A_pm_mf, -1)
    @test A_pm_mf.work_lock isa ReentrantLock
    @test size(A_mu) == (2N, 2N)
    @test all(isfinite, real.(A_pm))
    @test all(isfinite, imag.(A_pm))
    @test all(isfinite, real.(A_mu))
    @test norm(A_pm - A_mu) / norm(A_pm) > 1e-4   # distinct formulation
    @test norm(Matrix(A_pm_mf) - A_pm) / norm(A_pm) < 1e-13

    x = ComplexF64[sin(0.11 * i) + 1im * cos(0.07 * i) for i in 1:2N]
    y_mf = zeros(ComplexF64, 2N)
    mul!(y_mf, A_pm_mf, x)
    @test (@allocated mul!(y_mf, A_pm_mf, x)) < 1024
    fill!(y_mf, ComplexF64(NaN, NaN))
    mul!(y_mf, A_pm_mf, x, 1.0 + 0im, 0.0 + 0im)
    @test y_mf ≈ A_pm * x rtol=1e-13
    _test_shared_sie_operator_concurrency(
        A_pm_mf,
        A_pm,
        [x, (0.2 - 0.3im) .* x, reverse(x), conj.(x)],
    )
    A_pm_mf * x
    product_allocation = @allocated A_pm_mf * x
    zeros(ComplexF64, 2N)
    output_allocation = @allocated zeros(ComplexF64, 2N)
    @test product_allocation <= output_allocation + 128
    @test norm(y_mf - A_pm * x) / norm(A_pm * x) < 1e-13

    rhs0 = zeros(ComplexF64, 2N)
    rhs_nonfinite = copy(rhs0)
    rhs_nonfinite[1] = ComplexF64(NaN, 0.0)
    @test_throws ArgumentError solve_dielectric_sie_3d(
        mesh, rwg, k0, eps_in, rhs_nonfinite;
        mur_in=mu_in,
        formulation=:pmchwt,
        quad_order=1,
        singular_quad_order=3)
    @test_throws DimensionMismatch solve_dielectric_sie_3d(
        mesh, rwg, k0, eps_in, rhs0[1:(end - 1)];
        mur_in=mu_in,
        formulation=:pmchwt,
        quad_order=1,
        singular_quad_order=3)
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
                                        tol=1e-10,
                                        maxiter=50)
    x_direct = vcat(res_direct.J, res_direct.M)
    x_gmres = vcat(res_gmres.J, res_gmres.M)
    @test res_gmres.A isa MatrixFreeDielectricSIE3D
    @test res_gmres.A_LU === nothing
    @test norm(x_gmres - x_direct) / max(norm(x_direct), eps()) < 1e-9
    @test_throws ErrorException solve_dielectric_sie_3d(
        mesh, rwg, k0, eps_in, rhs;
        mur_in=mu_in,
        formulation=:pmchwt,
        solver=:gmres,
        quad_order=1,
        singular_quad_order=3,
        tol=1e-14,
        maxiter=1,
        memory=1,
    )
    res_sie_partial = solve_dielectric_sie_3d(
        mesh, rwg, k0, eps_in, rhs;
        mur_in=mu_in,
        formulation=:pmchwt,
        solver=:gmres,
        quad_order=1,
        singular_quad_order=3,
        tol=1e-14,
        maxiter=1,
        memory=1,
        check_gmres_convergence=false,
    )
    @test !res_sie_partial.stats.solved

    pw = make_plane_wave(Vec3(0.0, 0.0, k0), 1.0, Vec3(1.0, 0.0, 0.0))
    exterior = dielectric_medium_3d(k0)
    @test_throws ArgumentError assemble_dielectric_sie_rhs_3d(
        mesh, rwg,
        make_plane_wave(Vec3(0.0, 0.0, 0.0), 1.0, Vec3(1.0, 0.0, 0.0)),
        exterior;
        quad_order=1)
    @test_throws ArgumentError assemble_dielectric_sie_rhs_3d(
        mesh, rwg,
        make_plane_wave(Vec3(0.0, 0.0, k0), NaN, Vec3(1.0, 0.0, 0.0)),
        exterior;
        quad_order=1)
    @test_throws ArgumentError assemble_dielectric_sie_rhs_3d(
        mesh, rwg,
        make_plane_wave(Vec3(0.0, 0.0, k0), 1.0, Vec3(0.0, 0.0, 0.0)),
        exterior;
        quad_order=1)
    @test_throws ArgumentError assemble_dielectric_sie_rhs_3d(
        mesh, rwg,
        make_plane_wave(Vec3(0.0, 0.0, k0), 1.0, Vec3(0.0, 0.0, 1.0)),
        exterior;
        quad_order=1)
    @test_throws ArgumentError assemble_dielectric_sie_rhs_3d(
        mesh, rwg,
        make_plane_wave(Vec3(0.0, 0.0, 2k0), 1.0, Vec3(1.0, 0.0, 0.0)),
        exterior;
        quad_order=1)
    @test_throws ArgumentError assemble_dielectric_sie_rhs_3d(
        mesh, rwg, pw, exterior;
        quad_order=1,
        formulation=:invalid)
    @test_throws ArgumentError assemble_dielectric_sie_rhs_3d(
        mesh, rwg, pw,
        DielectricMedium3D(1.0 + 0im, 1.0 + 0im, k0 + 0.1im, 1.0 + 0im);
        quad_order=1)
    @test_throws ArgumentError assemble_dielectric_sie_rhs_3d(
        mesh, rwg, pw,
        DielectricMedium3D(1.0 + 0im, 1.0 + 0im, k0 + 0im, 0.0 + 0im);
        quad_order=1)

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
