using Test, LinearAlgebra

@testset "workflow ACA storage budget" begin
    mesh = make_rect_plate(0.06, 0.04, 2, 2)
    frequency = 1e9
    k = 2pi * frequency / 299792458.0
    wave = make_plane_wave(Vec3(0.0, 0.0, -k), 1.0, Vec3(1.0, 0.0, 0.0))
    @test_throws ArgumentError solve_scattering(mesh, frequency, wave;
        method=:aca_gmres, max_aca_storage_bytes=1,
        check_resolution=false, verbose=false)
    @test_throws ArgumentError solve_scattering(mesh, frequency, wave;
        max_aca_storage_bytes=0, check_resolution=false, verbose=false)
    @test_throws ArgumentError solve_scattering(mesh, frequency, wave;
        method=:aca_gmres, max_triplet_bytes=1,
        check_resolution=false, verbose=false)
    dense = solve_scattering(mesh, frequency, wave;
        method=:dense_direct, check_resolution=false, verbose=false)
    compressed = solve_scattering(mesh, frequency, wave;
        method=:aca_gmres, aca_leaf_size=4, aca_tol=1e-10, aca_max_rank=32,
        max_aca_storage_bytes=1_000_000, gmres_tol=1e-11,
        check_resolution=false, verbose=false)
    @test isapprox(compressed.I_coeffs, dense.I_coeffs; rtol=1e-8, atol=0)
    @test compressed.gmres_residual <= 1e-9
end

@testset "batched ACA canonical entry order" begin
    mesh = make_rect_plate(0.08, 0.06, 2, 2)
    rwg = build_rwg(mesh)
    k = 2pi * 1e9 / 299792458.0
    dense = assemble_Z_efie(mesh, rwg, k; quad_order=3)
    aca = build_aca_operator(mesh, rwg, k;
        leaf_size=rwg.nedges, quad_order=3, aca_tol=1e-12,
        max_rank=rwg.nedges)
    assembled = Matrix(aca)
    @test norm(assembled - dense) <= 1e-12 * norm(dense)
    @test norm(assembled - transpose(assembled)) <= 1e-12 * norm(assembled)

    panel = make_bent_slotted_panel(; panel_length=0.40, panel_width=0.30,
        flange_width=0.15, bend_angle=pi / 2, slot_length=0.12,
        slot_width=0.02, max_edge=0.15)
    panel_rwg = build_rwg(panel)
    panel_cache = DiffMoM._build_efie_cache(
        panel, panel_rwg, 2pi * 3e9 / 299792458.0; quad_order=7)
    block = zeros(ComplexF64, 1, 1)
    DiffMoM._fill_dense_block_batched!(block, panel_cache, [65], [17])
    @test block[1, 1] ≈ DiffMoM._efie_entry(panel_cache, 65, 17) rtol=1e-12
end
