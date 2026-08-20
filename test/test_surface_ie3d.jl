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

function _extreme_cancelling_matvec_input(
        matrix::AbstractMatrix{ComplexF64},
        row::Int,
        columns::NTuple{3,Int};
        fraction::Float64=0.55)
    target = fraction * floatmax(Float64)
    target_terms = (target, target, -target)
    input = zeros(ComplexF64, size(matrix, 2))
    setprecision(BigFloat, 4096) do
        for (column, target_term) in zip(columns, target_terms)
            input[column] = ComplexF64(
                Complex{BigFloat}(BigFloat(target_term), 0) /
                Complex{BigFloat}(matrix[row, column]))
        end
    end
    return input
end

function _bigfloat_matvec_reference(
        matrix::AbstractMatrix{ComplexF64},
        input::AbstractVector{ComplexF64})
    return setprecision(BigFloat, 4096) do
        ComplexF64.(Complex{BigFloat}.(matrix) *
                    Complex{BigFloat}.(input))
    end
end

function _bigfloat_sie_matvec_reference(
        operator::MatrixFreeDielectricSIE3D,
        input::AbstractVector{ComplexF64})
    N = div(length(input), 2)
    J = copy(@view input[1:N])
    M = copy(@view input[(N + 1):(2N)])
    Ze_ext_J = operator.Ze_ext * J
    Ze_int_J = operator.Ze_int * J
    K_ext_M = operator.K_ext * M
    K_int_M = operator.K_int * M
    K_ext_J = operator.K_ext * J
    K_int_J = operator.K_int * J
    Zh_ext_M = operator.Zh_ext * M
    Zh_int_M = operator.Zh_int * M
    Gram_M = operator.c_g_e != 0 ? operator.Gram * M : zeros(ComplexF64, N)
    Gram_J = operator.c_g_h != 0 ? operator.Gram * J : zeros(ComplexF64, N)
    return setprecision(BigFloat, 4096) do
        electric =
            Complex{BigFloat}(operator.c_ze_ext) .* Complex{BigFloat}.(Ze_ext_J) .+
            Complex{BigFloat}(operator.c_ze_int) .* Complex{BigFloat}.(Ze_int_J) .-
            Complex{BigFloat}(operator.c_ze_ext) .* Complex{BigFloat}.(K_ext_M) .-
            Complex{BigFloat}(operator.c_ze_int) .* Complex{BigFloat}.(K_int_M) .+
            Complex{BigFloat}(operator.c_g_e) .* Complex{BigFloat}.(Gram_M)
        magnetic =
            Complex{BigFloat}(operator.c_zh_ext) .* Complex{BigFloat}.(K_ext_J) .+
            Complex{BigFloat}(operator.c_zh_int) .* Complex{BigFloat}.(K_int_J) .+
            Complex{BigFloat}(operator.c_zh_ext) .* Complex{BigFloat}.(Zh_ext_M) .+
            Complex{BigFloat}(operator.c_zh_int) .* Complex{BigFloat}.(Zh_int_M) .+
            Complex{BigFloat}(operator.c_g_h) .* Complex{BigFloat}.(Gram_J)
        return ComplexF64.(vcat(electric, magnetic))
    end
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

    extreme_muller_medium = dielectric_medium_3d(
        k0, 1e-308 + 0im, 1e308 + 0im; eta0=1e-300)
    extreme_muller_coefficients = DiffMoM._surface_sie_coefficients_3d(
        :muller, extreme_muller_medium, extreme_muller_medium)
    @test collect(extreme_muller_coefficients) ≈ fill(0.5 + 0im, 4)

    complex_extreme = ComplexF64(1e308, 1e308)
    complex_extreme_medium = DielectricMedium3D(
        complex_extreme, complex_extreme, 1.0 + 0im, 1.0 + 0im)
    complex_extreme_coefficients = DiffMoM._surface_sie_coefficients_3d(
        :muller, complex_extreme_medium, complex_extreme_medium)
    @test collect(complex_extreme_coefficients) ≈ fill(0.5 + 0im, 4)

    unit_medium = DielectricMedium3D(
        1.0 + 0im, 1.0 + 0im, 1.0 + 0im, 1.0 + 0im)
    cancel_mu_medium = DielectricMedium3D(
        1.0 + 0im, -1.0 + 0im, 1.0 + 0im, 1.0 + 0im)
    cancel_eps_medium = DielectricMedium3D(
        -1.0 + 0im, 1.0 + 0im, 1.0 + 0im, 1.0 + 0im)
    @test_throws ErrorException DiffMoM._surface_sie_coefficients_3d(
        :muller, unit_medium, cancel_mu_medium)
    @test_throws ErrorException DiffMoM._surface_sie_coefficients_3d(
        :muller, unit_medium, cancel_eps_medium)

    K = assemble_magnetic_field_operator_3d(mesh, rwg, k0; quad_order=1)
    dense_block_bytes = sizeof(ComplexF64) * N^2
    @test_throws ArgumentError assemble_magnetic_field_operator_3d(
        mesh, rwg, k0;
        quad_order=1,
        max_output_bytes=dense_block_bytes - 1)
    @test_throws ArgumentError assemble_magnetic_field_operator_3d(
        mesh, rwg, NaN; quad_order=1)
    @test_throws ArgumentError matrixfree_magnetic_field_operator_3d(
        mesh, rwg, "invalid"; quad_order=1)
    @test assemble_magnetic_field_operator_3d(
        mesh, rwg, k0;
        quad_order=1,
        max_output_bytes=dense_block_bytes) == K
    K_mf = matrixfree_magnetic_field_operator_3d(mesh, rwg, k0; quad_order=1)
    growing_K_mf = matrixfree_magnetic_field_operator_3d(
        mesh, rwg, 1.0 + 1000.0im;
        quad_order=1, singular_quad_order=1)
    @test_throws OverflowError growing_K_mf[1, 1]
    growing_dense_error = try
        assemble_magnetic_field_operator_3d(
            mesh, rwg, 1.0 + 1000.0im;
            quad_order=1, singular_quad_order=1)
        nothing
    catch err
        err
    end
    @test growing_dense_error isa CompositeException
    @test occursin(
        "magnetic-field operator entry",
        sprint(showerror, growing_dense_error),
    )
    vertex_incidence = zeros(Int, nvertices(mesh))
    for vertex in mesh.tri
        vertex_incidence[vertex] += 1
    end
    near_pair_records = sum(
        degree * (degree - 1) ÷ 2 for degree in vertex_incidence)
    surface_fixed_bytes = DiffMoM._surface_cache_fixed_payload_bytes_3d(
        ntriangles(mesh), length(K_mf.wq), length(K_mf.wq_hi))
    surface_cache_bound = DiffMoM._surface_cache_work_bytes_3d(
        surface_fixed_bytes, ntriangles(mesh), near_pair_records)
    @test_throws ArgumentError matrixfree_magnetic_field_operator_3d(
        mesh, rwg, k0;
        quad_order=1,
        max_cache_bytes=surface_cache_bound - 1)
    exact_bound_K_mf = matrixfree_magnetic_field_operator_3d(
        mesh, rwg, k0;
        quad_order=1,
        max_cache_bytes=surface_cache_bound)
    @test exact_bound_K_mf.near_pairs.offsets == K_mf.near_pairs.offsets
    @test exact_bound_K_mf.near_pairs.neighbors == K_mf.near_pairs.neighbors
    @test_throws ArgumentError assemble_magnetic_field_operator_3d(
        mesh, rwg, k0; quad_order=1, max_cache_bytes=1)
    @test_throws ArgumentError matrixfree_magnetic_field_operator_3d(
        mesh, rwg, k0; quad_order=1,
        max_near_pairs=near_pair_records - 1)
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

    # At higher electrical size the MFIE row has several order-one terms.
    # These three are individually representable and the complete result is
    # representable, but their ordinary order overflows before cancellation.
    extreme_K_mesh = _icosphere_mesh(1.0, 0)
    extreme_K_rwg = build_rwg(
        extreme_K_mesh; allow_boundary=false, require_closed=true)
    extreme_K_mf = matrixfree_magnetic_field_operator_3d(
        extreme_K_mesh, extreme_K_rwg, 100.0;
        quad_order=1, singular_quad_order=3)
    extreme_K_dense = Matrix(extreme_K_mf)
    extreme_K_input = _extreme_cancelling_matvec_input(
        extreme_K_dense, 1, (10, 5, 18); fraction=0.51)
    extreme_K_terms = extreme_K_dense .* transpose(extreme_K_input)
    extreme_K_reference = _bigfloat_matvec_reference(
        extreme_K_dense, extreme_K_input)
    @test all(isfinite, extreme_K_input)
    @test all(isfinite, extreme_K_terms)
    @test all(isfinite, extreme_K_reference)
    extreme_K_result = extreme_K_mf * extreme_K_input
    @test all(isfinite, extreme_K_result)
    @test extreme_K_result[1] == extreme_K_reference[1]
    @test extreme_K_result ≈ extreme_K_reference rtol=2e-15

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

    # A malformed high-valence vertex creates quadratic candidate records.
    # Reject it after the bounded counting pass, before allocating the pairs.
    shared_vertex_triangles = Matrix{Int}(undef, 3, 20)
    for triangle in axes(shared_vertex_triangles, 2)
        shared_vertex_triangles[:, triangle] .=
            (1, 2triangle, 2triangle + 1)
    end
    shared_vertex_mesh = TriMesh(zeros(3, 41), shared_vertex_triangles)
    @test_throws ArgumentError DiffMoM._triangle_near_pairs_3d(
        shared_vertex_mesh; max_near_pairs=10)

    A_pm = assemble_pmchwt_3d(mesh, rwg, k0, eps_in;
                              mur_in=mu_in,
                              quad_order=1,
                              singular_quad_order=3)
    dense_sie_work_bytes = 10 * dense_block_bytes
    @test_throws ArgumentError assemble_pmchwt_3d(
        mesh, rwg, k0, eps_in;
        mur_in=mu_in,
        quad_order=1,
        singular_quad_order=3,
        max_work_bytes=dense_sie_work_bytes - 1)
    @test assemble_pmchwt_3d(
        mesh, rwg, k0, eps_in;
        mur_in=mu_in,
        quad_order=1,
        singular_quad_order=3,
        max_work_bytes=dense_sie_work_bytes) == A_pm
    A_pm_mf = matrixfree_dielectric_sie_operator_3d(mesh, rwg, k0, eps_in;
                                                    mur_in=mu_in,
                                                    formulation=:pmchwt,
                                                    quad_order=1,
                                                    singular_quad_order=3)
    @test A_pm_mf.Ze_ext.cache.quad_pts === A_pm_mf.Ze_int.cache.quad_pts
    @test A_pm_mf.Ze_ext.cache.quad_pts === A_pm_mf.Zh_ext.cache.quad_pts
    @test A_pm_mf.Ze_ext.cache.quad_pts === A_pm_mf.Zh_int.cache.quad_pts
    @test A_pm_mf.Ze_ext.cache.rwg_vals === A_pm_mf.Zh_int.cache.rwg_vals
    @test A_pm_mf.Ze_ext.cache.adjacent.offsets ===
          A_pm_mf.Zh_int.cache.adjacent.offsets
    @test A_pm_mf.K_ext.pts === A_pm_mf.K_int.pts
    @test A_pm_mf.K_ext.near_pairs.offsets ===
          A_pm_mf.K_int.near_pairs.offsets

    workspace_payload = DiffMoM._checked_array_payload_bytes(
        ComplexF64, 7, N; label="test matrix-free SIE workspace")
    workspace_bound = DiffMoM._estimated_surface_payload_bytes_3d(
        workspace_payload, "test matrix-free SIE workspace")
    edge_incidence = Dict{Tuple{Int,Int},Int}()
    for triangle in axes(mesh.tri, 2)
        v1, v2, v3 = mesh.tri[:, triangle]
        for (first_vertex, second_vertex) in ((v1, v2), (v2, v3), (v3, v1))
            edge = minmax(first_vertex, second_vertex)
            edge_incidence[edge] = get(edge_incidence, edge, 0) + 1
        end
    end
    adjacency_pair_records = sum(
        degree * (degree - 1) ÷ 2 for degree in values(edge_incidence))
    efie_cache = A_pm_mf.Ze_ext.cache
    efie_fixed_bytes = DiffMoM._efie_cache_fixed_payload_bytes(
        N,
        ntriangles(mesh),
        efie_cache.Nq,
        length(efie_cache.wq_hi),
        eltype(efie_cache.div_vals),
        eltype(fieldtype(eltype(efie_cache.rwg_vals), 1)))
    efie_geometry_bound = DiffMoM._efie_cache_retained_bytes(
        efie_fixed_bytes,
        ntriangles(mesh),
        DiffMoM._adjacent_pair_count(efie_cache.adjacent))
    efie_work_bound = DiffMoM._efie_cache_work_bytes(
        efie_fixed_bytes, ntriangles(mesh), adjacency_pair_records)
    sie_surface_fixed_bytes = DiffMoM._surface_cache_fixed_payload_bytes_3d(
        ntriangles(mesh),
        length(A_pm_mf.K_ext.wq),
        length(A_pm_mf.K_ext.wq_hi))
    sie_surface_work_bound = DiffMoM._surface_cache_work_bytes_3d(
        sie_surface_fixed_bytes, ntriangles(mesh), near_pair_records)
    magnetic_geometry_bound = DiffMoM._surface_cache_retained_bytes_3d(
        sie_surface_fixed_bytes,
        ntriangles(mesh),
        DiffMoM._adjacent_pair_count(A_pm_mf.K_ext.near_pairs))
    aggregate_cache_bound = max(
        workspace_bound + efie_work_bound,
        workspace_bound + efie_geometry_bound + sie_surface_work_bound,
        workspace_bound + efie_geometry_bound + magnetic_geometry_bound)
    @test_throws ArgumentError matrixfree_dielectric_sie_operator_3d(
        mesh, rwg, k0, eps_in;
        mur_in=mu_in,
        formulation=:pmchwt,
        quad_order=1,
        singular_quad_order=3,
        max_cache_bytes=aggregate_cache_bound - 1)
    exact_cache_A = matrixfree_dielectric_sie_operator_3d(
        mesh, rwg, k0, eps_in;
        mur_in=mu_in,
        formulation=:pmchwt,
        quad_order=1,
        singular_quad_order=3,
        max_cache_bytes=aggregate_cache_bound)
    @test exact_cache_A[1, 1] == A_pm_mf[1, 1]
    @test_throws ArgumentError matrixfree_dielectric_sie_operator_3d(
        mesh, rwg, k0, eps_in;
        mur_in=mu_in,
        formulation=:pmchwt,
        quad_order=1,
        singular_quad_order=3,
        max_cache_bytes=1)
    @test_throws ArgumentError assemble_pmchwt_3d(
        mesh, rwg, k0, eps_in;
        mur_in=mu_in,
        quad_order=1,
        singular_quad_order=3,
        max_near_pairs=near_pair_records - 1)
    # :muller assembles a different (second-kind) system than PMCHWT (it carries the
    # nhat x Gram identity term); the PMCHWT-vs-Muller currents agreement is checked in
    # its own testset below.
    A_mu = assemble_muller_3d(mesh, rwg, k0, eps_in;
                              mur_in=mu_in,
                              quad_order=1,
                              singular_quad_order=3)
    @test A_pm_mf.Gram isa Matrix{ComplexF64}
    A_mu_mf = matrixfree_dielectric_sie_operator_3d(
        mesh, rwg, k0, eps_in;
        mur_in=mu_in,
        formulation=:muller,
        quad_order=1,
        singular_quad_order=3)
    @test A_mu_mf.Gram isa LocalMassMatrix{ComplexF64}
    @test Matrix(A_mu_mf) ≈ A_mu rtol=1e-13
    gram_storage_bytes = sizeof(Int) *
                         (length(A_mu_mf.Gram.rows) + length(A_mu_mf.Gram.cols)) +
                         sizeof(ComplexF64) * length(A_mu_mf.Gram.vals)
    @test_throws ArgumentError matrixfree_dielectric_sie_operator_3d(
        mesh, rwg, k0, eps_in;
        mur_in=mu_in,
        formulation=:muller,
        quad_order=1,
        singular_quad_order=3,
        max_gram_storage_bytes=gram_storage_bytes - 1)
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

    # Every individual EFIE product is representable, but the ordinary row
    # order can overflow before later cancellation. Exercise both forward and
    # adjoint public matrix-free reductions against a high-precision oracle.
    Ze_mf = A_pm_mf.Ze_ext
    Ze_dense = Matrix(Ze_mf)
    extreme_columns = (2, 6, 1)
    extreme_efie_input = _extreme_cancelling_matvec_input(
        Ze_dense, 1, extreme_columns)
    extreme_efie_terms = Ze_dense .* transpose(extreme_efie_input)
    extreme_efie_reference = _bigfloat_matvec_reference(
        Ze_dense, extreme_efie_input)
    @test all(isfinite, extreme_efie_input)
    @test all(isfinite, extreme_efie_terms)
    @test all(isfinite, extreme_efie_reference)
    @test Ze_mf * extreme_efie_input == extreme_efie_reference

    Ze_adjoint_dense = Matrix(adjoint(Ze_mf))
    extreme_efie_adjoint_input = _extreme_cancelling_matvec_input(
        Ze_adjoint_dense, 1, extreme_columns)
    extreme_efie_adjoint_terms =
        Ze_adjoint_dense .* transpose(extreme_efie_adjoint_input)
    extreme_efie_adjoint_reference = _bigfloat_matvec_reference(
        Ze_adjoint_dense, extreme_efie_adjoint_input)
    @test all(isfinite, extreme_efie_adjoint_input)
    @test all(isfinite, extreme_efie_adjoint_terms)
    @test all(isfinite, extreme_efie_adjoint_reference)
    @test adjoint(Ze_mf) * extreme_efie_adjoint_input ==
          extreme_efie_adjoint_reference

    extreme_sie_input = _extreme_cancelling_matvec_input(
        Matrix(A_pm_mf), 1, (6, 4, 9); fraction=0.51)
    extreme_sie_terms = Matrix(A_pm_mf) .* transpose(extreme_sie_input)
    extreme_sie_reference = _bigfloat_sie_matvec_reference(
        A_pm_mf, extreme_sie_input)
    extreme_sie_dense_reference = _bigfloat_matvec_reference(
        Matrix(A_pm_mf), extreme_sie_input)
    @test all(isfinite, extreme_sie_input)
    @test all(isfinite, extreme_sie_terms)
    @test all(isfinite, extreme_sie_reference)
    extreme_sie_result = A_pm_mf * extreme_sie_input
    @test all(isfinite, extreme_sie_result)
    @test extreme_sie_result[1] == extreme_sie_reference[1]
    @test extreme_sie_result ≈ extreme_sie_reference rtol=2e-15
    @test extreme_sie_result ≈ extreme_sie_dense_reference rtol=2e-15

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

    # The two scaled terms are individually outside Float64 range but cancel
    # exactly. The five-argument mul! contract must combine them before the
    # final ComplexF64 conversion instead of producing Inf - Inf = NaN.
    scaled_product = A_pm_mf * x
    scaled_initial = -scaled_product
    extreme_scale = 1e308 + 0im
    scaled_result = copy(scaled_initial)
    scaled_reference = setprecision(BigFloat, 4096) do
        ComplexF64.(
            Complex{BigFloat}(extreme_scale) .* Complex{BigFloat}.(scaled_product) .+
            Complex{BigFloat}(extreme_scale) .* Complex{BigFloat}.(scaled_initial))
    end
    mul!(scaled_result, A_pm_mf, x, extreme_scale, extreme_scale)
    @test scaled_reference == zeros(ComplexF64, 2N)
    @test all(isfinite, scaled_result)
    @test scaled_result == scaled_reference

    rhs0 = zeros(ComplexF64, 2N)
    @test_throws ArgumentError solve_dielectric_sie_3d(
        mesh, rwg, k0, eps_in, rhs0;
        mur_in=mu_in,
        formulation=:pmchwt,
        quad_order=1,
        singular_quad_order=3,
        max_work_bytes=dense_sie_work_bytes - 1)
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
    rhs_out_of_range = fill(1.0e308 + 0im, 2N)
    @test_throws OverflowError solve_dielectric_sie_3d(
        mesh, rwg, k0, eps_in, rhs_out_of_range;
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

    tiny_direction_component = nextfloat(0.0)
    tiny_direction_wave = PlaneWaveExcitation(
        Vec3(tiny_direction_component, tiny_direction_component, 0.0),
        1.0,
        Vec3(0.0, 0.0, 1.0),
    )
    tiny_direction_medium = DielectricMedium3D(
        1.0 + 0im,
        1.0 + 0im,
        ComplexF64(tiny_direction_component),
        1.0 + 0im,
    )
    tiny_direction_rhs = assemble_dielectric_sie_rhs_3d(
        mesh, rwg, tiny_direction_wave, tiny_direction_medium;
        quad_order=1)
    tiny_direction = Vec3(
        inv(sqrt(2.0)), inv(sqrt(2.0)), 0.0)
    tiny_magnetic_polarization = cross(
        tiny_direction, tiny_direction_wave.pol)
    tiny_magnetic_rhs = assemble_excitation(
        mesh,
        rwg,
        PlaneWaveExcitation(
            tiny_direction_wave.k_vec,
            tiny_direction_wave.E0,
            tiny_magnetic_polarization,
        );
        quad_order=1,
    )
    @test norm(tiny_magnetic_rhs) > 0.0
    @test tiny_direction_rhs[(N + 1):(2N)] ≈
          tiny_magnetic_rhs rtol=2eps(Float64) atol=0.0

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
