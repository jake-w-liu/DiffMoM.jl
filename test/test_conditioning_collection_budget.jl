using Test
using LinearAlgebra
using SparseArrays
using DiffMoM

@testset "Conditioning defers collection only within both workspace bounds" begin
    policy = DiffMoM._conditioning_can_defer_collection
    # Both the caller's ceiling and the OS estimate include the discarded
    # matrix. One byte less at either boundary requires collection.
    @test policy(680, 10_000, 700, 11_380, 10_680)
    @test !policy(680, 10_000, 700, 11_379, 10_680)
    @test !policy(680, 10_000, 700, 11_380, 10_679)
    @test !policy(680, 10_000, 700, 11_380, 0)
    @test !policy(680, 10_000, 700, 10_699, typemax(UInt64))
    @test policy(typemax(Int) - 100, 50, 50, typemax(Int), typemax(UInt64))

    mesh = make_rect_plate(0.1, 0.08, 2, 2)
    frequency = 3e8
    k = DiffMoM._frequency_to_wavenumber(frequency, 299792458.0, "collection test")
    source = make_plane_wave(Vec3(0, 0, -k), 1.0, Vec3(1, 0, 0))
    state = solve_scattering(mesh, frequency, source;
        return_state=true, verbose=false).state
    pair = build_nested_rwg_pair(mesh)
    fine_operator = assemble_Z_efie(pair.fine_mesh, pair.fine_rwg, k)
    fine_rhs = assemble_excitation(pair.fine_mesh, pair.fine_rwg, source)
    system = prepare_galerkin_error(state, pair, fine_operator, fine_rhs)
    nf, nc = size(pair.P)
    m = size(pair.Q, 2)
    probes = zeros(ComplexF64, 0, m)
    information = zeros(ComplexF64, 0, m)
    observations = ComplexF64[]
    fine_mass = assemble_rwg_gram(pair.fine_mesh, pair.fine_rwg)
    mass = sparse(adjoint(pair.Q) * fine_mass * pair.Q)
    released = Base.summarysize(fine_mass)
    retained = DiffMoM._error_system_storage_bytes(system)
    rows_bytes = DiffMoM._checked_array_payload_bytes(ComplexF64, 8, nc)
    condition_work = DiffMoM._conditioning_work_bytes(m, 0)
    condition_inputs = Base.summarysize((mass, information, observations))
    outer_inputs = Base.summarysize((probes, information, observations))
    nt = ntriangles(pair.fine_mesh)
    profile = DiffMoM._mass_precompute_profile(
        pair.fine_rwg, nt, 3, Float64, nothing, nt)
    merge_bytes = 10sizeof(Int) * profile.entry_count +
                  5sizeof(Float64) * profile.entry_count +
                  sizeof(Int) * (nf + 1) + sizeof(Float64) * nt
    cap = retained + max(rows_bytes + condition_work,
        rows_bytes + profile.work_bytes + merge_bytes,
        outer_inputs + condition_inputs + condition_work)

    @test_throws ArgumentError condition_discretization_error(
        system, probes; tau=0.02, max_work_bytes=cap - 1)
    tight = condition_discretization_error(
        system, probes; tau=0.02, max_work_bytes=cap)
    deferred = condition_discretization_error(
        system, probes; tau=0.02, max_work_bytes=cap + released)
    @test tight.conditioning.mass == deferred.conditioning.mass
    @test tight.conditioning.mean_coefficients == deferred.conditioning.mean_coefficients
    @test tight.conditioning.right_basis == deferred.conditioning.right_basis
    @test tight.conditioning.rank == deferred.conditioning.rank == 0
    @test tight.conditioning.mass == mass
end
