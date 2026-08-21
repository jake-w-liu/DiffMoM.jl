# runtests.jl — Test suite for DiffMoM
#
# Run: julia --project=. test/runtests.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using SparseArrays
using StaticArrays
using Statistics
using Random
using Test
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM

include("test_runtime_contract.jl")

complex_vector_input = ComplexF64[1 + 2im, 3 - 4im]
@assert DiffMoM._complex_vector_input(complex_vector_input) === complex_vector_input
DiffMoM._complex_vector_input(complex_vector_input)  # warm compilation
@assert @allocated(DiffMoM._complex_vector_input(complex_vector_input)) == 0
@assert DiffMoM._complex_vector_input(Float32[1, 2]) == ComplexF64[1, 2]

function _complex_vector_output_allocation(n::Int)
    zeros(ComplexF64, n)  # warm the exact array-size specialization
    return @allocated zeros(ComplexF64, n)
end

function _bit_vector_output_allocation(n::Int)
    BitVector(undef, n)
    return @allocated BitVector(undef, n)
end

function _float_vector_output_allocation(n::Int)
    zeros(Float64, n)
    return @allocated zeros(Float64, n)
end

function _complex_matrix_output_allocation(m::Int, n::Int)
    Matrix{ComplexF64}(undef, m, n)
    return @allocated Matrix{ComplexF64}(undef, m, n)
end

function _radiation_vectors_allocation(mesh, rwg, grid, k)
    return @allocated radiation_vectors(mesh, rwg, grid, k; quad_order=3)
end

function _radiated_power_allocation(E_ff, grid)
    radiated_power(E_ff, grid)
    return @allocated radiated_power(E_ff, grid)
end

function _projected_power_allocation(E_ff, grid, pol, mask)
    projected_power(E_ff, grid, pol; mask=mask)
    return @allocated projected_power(E_ff, grid, pol; mask=mask)
end

function _bistatic_rcs_allocation(E_ff)
    bistatic_rcs(E_ff)
    return @allocated bistatic_rcs(E_ff)
end

function _backscatter_rcs_allocation(E_ff, grid, k_inc_hat)
    backscatter_rcs(E_ff, grid, k_inc_hat)
    return @allocated backscatter_rcs(E_ff, grid, k_inc_hat)
end

function _green_kernel_allocations(r, rp, k)
    greens(r, rp, k)
    greens_smooth(r, rp, k)
    grad_greens(r, rp, k)
    return (
        @allocated(greens(r, rp, k)),
        @allocated(greens_smooth(r, rp, k)),
        @allocated(grad_greens(r, rp, k)),
    )
end

function _spherical_hankel_allocations(l_max::Int, x::Float64)
    DiffMoM.spherical_hankel2_all(l_max, x)
    return @allocated DiffMoM.spherical_hankel2_all(l_max, x)
end

function _sphere_sampling_rejection_allocations(L::Int)
    try
        DiffMoM.make_sphere_sampling(L)
    catch error
        error isa ArgumentError || rethrow()
    end
    return @allocated try
        DiffMoM.make_sphere_sampling(L)
    catch error
        error isa ArgumentError || rethrow()
    end
end

function _mlfma_precision_rejection_allocations(mesh, rwg, k)
    try
        build_mlfma_operator(mesh, rwg, k; precision=1_000_000)
    catch error
        error isa ArgumentError || rethrow()
    end
    return @allocated try
        build_mlfma_operator(mesh, rwg, k; precision=1_000_000)
    catch error
        error isa ArgumentError || rethrow()
    end
end

function _multiple_excitation_rejection_allocations(mesh, rwg, excitations)
    try
        assemble_multiple_excitations(
            mesh, rwg, excitations;
            quad_order=1,
            max_output_bytes=100_000_000,
            max_work_bytes=100_000_000,
            max_terms=100_000_000)
    catch error
        error isa ArgumentError || rethrow()
    end
    return @allocated try
        assemble_multiple_excitations(
            mesh, rwg, excitations;
            quad_order=1,
            max_output_bytes=100_000_000,
            max_work_bytes=100_000_000,
            max_terms=100_000_000)
    catch error
        error isa ArgumentError || rethrow()
    end
end

function _aca_dense_approximation(operator::ACAOperator)
    approximation_tree = zeros(ComplexF64, operator.N, operator.N)
    for block in operator.dense_blocks
        approximation_tree[block.row_range, block.col_range] .= block.data
    end
    for block in operator.lowrank_blocks
        approximation_tree[block.row_range, block.col_range] .=
            block.U * adjoint(block.V)
    end
    inverse_permutation = invperm(operator.tree.perm)
    return approximation_tree[
        inverse_permutation, inverse_permutation]
end

function _aca_operator_with_blocks(
        base::ACAOperator,
        dense_blocks::Vector{DiffMoM.DenseBlock},
        lowrank_blocks::Vector{DiffMoM.LowRankBlock})
    maximum_rank = isempty(lowrank_blocks) ? 1 :
                   max(1, maximum(size(block.U, 2)
                                  for block in lowrank_blocks))
    workspace = DiffMoM.ACAWorkspace(
        Vector{ComplexF64}(undef, base.N),
        zeros(ComplexF64, base.N),
        Vector{ComplexF64}(undef, maximum_rank),
    )
    return ACAOperator(
        base.cache, base.tree, dense_blocks, lowrank_blocks,
        base.N, workspace)
end

function _spherical_hankel_bigfloat_reference(l_max::Int, x::Float64)
    return setprecision(BigFloat, 8192) do
        x_big = BigFloat(x)
        sx, cx = sincos(x_big)
        inv_x = inv(x_big)
        values = Vector{Complex{BigFloat}}(undef, l_max + 1)
        values[1] = Complex{BigFloat}(sx * inv_x, cx * inv_x)
        if l_max >= 1
            values[2] = Complex{BigFloat}(
                (sx * inv_x - cx) * inv_x,
                (cx * inv_x + sx) * inv_x,
            )
        end
        for l in 1:l_max-1
            values[l + 2] = ((2l + 1) * inv_x) * values[l + 1] -
                            values[l]
        end
        return ComplexF64.(values)
    end
end

function _green_kernel_bigfloat_reference(r::Vec3, rp::Vec3, k::Number)
    return setprecision(BigFloat, 8192) do
        dx = BigFloat(r[1]) - BigFloat(rp[1])
        dy = BigFloat(r[2]) - BigFloat(rp[2])
        dz = BigFloat(r[3]) - BigFloat(rp[3])
        distance = sqrt(dx * dx + dy * dy + dz * dz)
        q = Complex{BigFloat}(BigFloat(imag(k)), -BigFloat(real(k)))
        inv_four_pi = inv(4 * BigFloat(π))

        if iszero(distance)
            return (
                0.0 + 0.0im,
                ComplexF64(q * inv_four_pi),
                CVec3(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im),
            )
        end

        phase_argument = q * distance
        phase = exp(phase_argument)
        green = ComplexF64(phase * (inv_four_pi / distance))
        smooth = ComplexF64(
            expm1(phase_argument) * (inv_four_pi / distance))
        radial_factor = phase * inv_four_pi *
                        (phase_argument - one(phase_argument))
        gradient = CVec3(ntuple(component -> begin
            delta = component == 1 ? dx : component == 2 ? dy : dz
            ComplexF64(iszero(delta) ? zero(phase_argument) :
                       ((radial_factor * (delta / distance)) / distance) /
                       distance)
        end, 3))
        return green, smooth, gradient
    end
end

function _static_integral_allocations(P, V1, V2, V3)
    analytical_integral_1overR(P, V1, V2, V3)
    grad_analytical_integral_1overR(P, V1, V2, V3)
    return (
        @allocated(analytical_integral_1overR(P, V1, V2, V3)),
        @allocated(grad_analytical_integral_1overR(P, V1, V2, V3)),
    )
end

mutable struct _CountingIdentityQ <: AbstractMatrix{ComplexF64}
    n::Int
    reads::Base.RefValue{Int}
end

Base.size(Q::_CountingIdentityQ) = (Q.n, Q.n)
function Base.getindex(Q::_CountingIdentityQ, row::Int, column::Int)
    Q.reads[] += 1
    return row == column ? 1.0 + 0im : 0.0 + 0im
end
Base.:*(Q::_CountingIdentityQ, x::Vector{ComplexF64}) = copy(x)
function LinearAlgebra.mul!(
        y::AbstractVector{ComplexF64}, Q::_CountingIdentityQ,
        x::AbstractVector{ComplexF64})
    length(y) == Q.n && length(x) == Q.n || throw(DimensionMismatch())
    copyto!(y, x)
    return y
end

function _bilinear_allocation(left, A, right)
    DiffMoM._dot_left_matrix_right(left, A, right)
    return @allocated DiffMoM._dot_left_matrix_right(left, A, right)
end

function _filter_allocation(W, w_sum, rho)
    apply_filter(W, w_sum, rho)
    return @allocated apply_filter(W, w_sum, rho)
end

function _filter_transpose_allocation(W, w_sum, gradient)
    apply_filter_transpose(W, w_sum, gradient)
    return @allocated apply_filter_transpose(W, w_sum, gradient)
end

function _assert_single_complex_output_allocation(A, x)
    result = A * x
    A * x  # warm the exact operator/vector specialization
    product_allocation = @allocated A * x
    output_allocation = _complex_vector_output_allocation(length(result))
    @assert product_allocation <= output_allocation + 128
    return nothing
end

function _assert_zero_allocation_mul!(A, x)
    result = zeros(ComplexF64, first(size(A)))
    mul!(result, A, x)
    allocation = @allocated mul!(result, A, x)
    @assert allocation <= 128
    return result
end

function _assert_scaled_mul_contract(A, x, y_initial)
    alpha_scale = 1.25 - 0.5im
    beta_scale = -0.75 + 0.25im
    reference = A * x

    result = copy(y_initial)
    mul!(result, A, x, alpha_scale, beta_scale)
    @assert result ≈ alpha_scale .* reference .+ beta_scale .* y_initial rtol=1e-12
    allocation = @allocated mul!(result, A, x, alpha_scale, beta_scale)
    @assert allocation <= 128

    fill!(result, ComplexF64(NaN, NaN))
    mul!(result, A, x, alpha_scale, zero(ComplexF64))
    @assert result ≈ alpha_scale .* reference rtol=1e-12

    poisoned_input = fill(ComplexF64(NaN, NaN), length(x))
    copyto!(result, y_initial)
    mul!(result, A, poisoned_input, zero(ComplexF64), beta_scale)
    @assert result ≈ beta_scale .* y_initial rtol=1e-12
    return nothing
end

function _assert_zero_allocation_overlap_mul_contract(A, x, y_tail)
    alpha_scale = 1.2 - 0.1im
    beta_scale = -0.4 + 0.2im
    storage = vcat(x, ComplexF64(y_tail))
    overlap_x = view(storage, 1:length(x))
    overlap_y = view(storage, 2:(length(x) + 1))
    x_initial = copy(overlap_x)
    y_initial = copy(overlap_y)
    expected = alpha_scale .* (A * x_initial) .+ beta_scale .* y_initial
    mul!(overlap_y, A, overlap_x, alpha_scale, beta_scale)
    @assert overlap_y ≈ expected rtol=1e-12
    allocation = @allocated mul!(
        overlap_y, A, overlap_x, alpha_scale, beta_scale)
    @assert allocation <= 128
    return nothing
end

function _assert_single_workspace_mul!(A, x)
    result = zeros(ComplexF64, first(size(A)))
    mul!(result, A, x)
    allocation = @allocated mul!(result, A, x)
    @assert allocation <=
            _complex_vector_output_allocation(length(result)) + 128
    return result
end

function _matrix_entry_allocation(A, i::Int, j::Int)
    A[i, j]
    return @allocated A[i, j]
end

function _apply_q_allocation(G_mat, grid, pol, x, mask)
    apply_Q(G_mat, grid, pol, x; mask=mask)
    return @allocated apply_Q(G_mat, grid, pol, x; mask=mask)
end

function _sum_imported_source(imported, points)
    result = CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    @inbounds for point in points
        result += imported.source_func(point)
    end
    return result
end

function _assert_shared_workspace_concurrency(operators, inputs; rtol=1e-12)
    length(operators) == length(inputs) ||
        throw(DimensionMismatch("operators and inputs must have matching lengths"))
    references = [operators[i] * inputs[i] for i in eachindex(inputs)]
    Threads.nthreads() > 1 || return nothing

    for _ in 1:4
        gate = Base.Event()
        tasks = map(eachindex(inputs)) do i
            Threads.@spawn begin
                wait(gate)
                operators[i] * inputs[i]
            end
        end
        yield()
        notify(gate)
        results = fetch.(tasks)
        @assert results ≈ references rtol=rtol
    end
    return nothing
end

const DATADIR = joinpath(@__DIR__, "..", "data")
mkpath(DATADIR)

function write_icosphere_obj(path::AbstractString; radius::Float64=0.05, subdivisions::Int=2)
    ϕ = (1 + sqrt(5.0)) / 2
    verts = [
        (-1.0,  ϕ, 0.0), ( 1.0,  ϕ, 0.0), (-1.0, -ϕ, 0.0), ( 1.0, -ϕ, 0.0),
        ( 0.0, -1.0, ϕ), ( 0.0,  1.0, ϕ), ( 0.0, -1.0,-ϕ), ( 0.0,  1.0,-ϕ),
        (  ϕ, 0.0, -1.0), (  ϕ, 0.0,  1.0), ( -ϕ, 0.0, -1.0), ( -ϕ, 0.0,  1.0),
    ]
    verts = [Vec3(v...) / norm(Vec3(v...)) for v in verts]

    faces = [
        (1,12,6), (1,6,2), (1,2,8), (1,8,11), (1,11,12),
        (2,6,10), (6,12,5), (12,11,3), (11,8,7), (8,2,9),
        (4,10,5), (4,5,3), (4,3,7), (4,7,9), (4,9,10),
        (5,10,6), (3,5,12), (7,3,11), (9,7,8), (10,9,2),
    ]

    for _ in 1:subdivisions
        edge_mid = Dict{Tuple{Int,Int},Int}()
        new_faces = NTuple{3,Int}[]

        function midpoint_index(i::Int, j::Int)
            key = i < j ? (i, j) : (j, i)
            if haskey(edge_mid, key)
                return edge_mid[key]
            end
            vmid = (verts[i] + verts[j]) / 2
            vmid /= norm(vmid)
            push!(verts, vmid)
            idx = length(verts)
            edge_mid[key] = idx
            return idx
        end

        for (i, j, k) in faces
            a = midpoint_index(i, j)
            b = midpoint_index(j, k)
            c = midpoint_index(k, i)
            push!(new_faces, (i, a, c))
            push!(new_faces, (j, b, a))
            push!(new_faces, (k, c, b))
            push!(new_faces, (a, b, c))
        end
        faces = new_faces
    end

    open(path, "w") do io
        println(io, "# Icosphere mesh for test gate")
        for v in verts
            println(io, "v $(radius * v[1]) $(radius * v[2]) $(radius * v[3])")
        end
        for (i, j, k) in faces
            println(io, "f $i $j $k")
        end
    end
end

println("="^60)
println("DiffMoM Test Suite")
println("="^60)

# ─────────────────────────────────────────────────
# Gradient verification utility validation
# ─────────────────────────────────────────────────
theta_verify = [1.0, -0.5]
objective_verify = theta -> theta[1]^2 + 3theta[2]^2
gradient_verify = [2.0, -3.0]

@test complex_step_grad(objective_verify, theta_verify, 1) ≈ 2.0
@test complex_step_grad(theta -> 1e-294 * theta[1], [1.0], 1) == 1e-294
@test_throws ArgumentError complex_step_grad(_ -> 1.0, [1.0], 1)
@test fd_grad(objective_verify, theta_verify, 2) ≈ -3.0 rtol=1e-9
@test fd_grad(theta -> theta[1], [1e16], 1; h=3.0) == 1.0
@test fd_grad(
    objective_verify, theta_verify, 1; scheme=:forward) ≈ 2.0 rtol=1e-5
verified_gradient = verify_gradient(
    objective_verify, gradient_verify, theta_verify; indices=(2, 1))
@test getproperty.(verified_gradient, :p) == [2, 1]
@test maximum(getproperty.(verified_gradient, :rel_err_fd)) < 1e-9
tiny_verified_gradient = only(verify_gradient(
    theta -> 1e-300 * theta[1], [2e-300], [1.0];
    eps_cs=1e-30, h_fd=1e-6))
@test tiny_verified_gradient.rel_err_cs ≈ 1.0 rtol=4eps(Float64)
@test tiny_verified_gradient.rel_err_fd ≈ 0.5 rtol=1e-10
@test DiffMoM._relative_gradient_error_to_reference(1.0, 0.0) == Inf
@test DiffMoM._symmetric_relative_gradient_error(0.0, 0.0) == 0.0

@test_throws ArgumentError complex_step_grad(
    objective_verify, theta_verify, 1; eps=0.0)
@test_throws ArgumentError complex_step_grad(
    _ -> Inf, theta_verify, 1)
@test_throws ArgumentError complex_step_grad(
    theta -> 1im + theta[1]^2, theta_verify, 1)
@test_throws ArgumentError complex_step_grad(
    objective_verify, [NaN, 0.0], 1)
@test_throws ArgumentError complex_step_grad(
    objective_verify, theta_verify, 0)
@test_throws ArgumentError fd_grad(
    objective_verify, theta_verify, 1; h=0.0)
@test_throws ArgumentError fd_grad(
    objective_verify, theta_verify, 1; h=Inf)
@test_throws ArgumentError fd_grad(
    objective_verify, [floatmax(Float64), 0.0], 1)
@test_throws ArgumentError fd_grad(
    objective_verify, theta_verify, 1; scheme=:invalid)
@test_throws DimensionMismatch verify_gradient(
    objective_verify, Float64[], theta_verify)
@test_throws ArgumentError verify_gradient(
    objective_verify, [NaN, -3.0], theta_verify)
@test_throws ArgumentError verify_gradient(
    objective_verify, gradient_verify, theta_verify; indices=(3,))
@test_throws ArgumentError verify_gradient(
    objective_verify, gradient_verify, theta_verify; indices=(true,))

# ─────────────────────────────────────────────────
# Test 1: Mesh and RWG Construction
# ─────────────────────────────────────────────────
println("\n── Test 1: Mesh and RWG construction ──")

Lx, Ly = 0.1, 0.1   # 10 cm × 10 cm plate
Nx, Ny = 3, 3
mesh = make_rect_plate(Lx, Ly, Nx, Ny)

println("  Vertices: $(nvertices(mesh)),  Triangles: $(ntriangles(mesh))")
@assert nvertices(mesh) == (Nx+1)*(Ny+1)
@assert ntriangles(mesh) == 2*Nx*Ny

rwg = build_rwg(mesh)
println("  RWG basis functions: $(rwg.nedges)")
@assert rwg.nedges > 0

# RWG geometric factors are tied to the exact mesh object used to build them.
# Mixing a same-topology but geometrically different mesh must fail closed.
mismatched_rwg_mesh = make_rect_plate(2Lx, 2Ly, Nx, Ny)
mismatched_rwg_k = 1.0
mismatched_rwg_excitation = make_plane_wave(
    Vec3(0.0, 0.0, -mismatched_rwg_k), 1.0,
    Vec3(1.0, 0.0, 0.0))
mismatched_rwg_grid = make_sph_grid(2, 4)
@test_throws ArgumentError DiffMoM._validate_mesh_rwg_pair(
    mismatched_rwg_mesh, rwg)
@test_throws ArgumentError rwg_centers(mismatched_rwg_mesh, rwg)
@test_throws ArgumentError build_nearfield_preconditioner(
    Matrix{ComplexF64}(I, rwg.nedges, rwg.nedges),
    mismatched_rwg_mesh, rwg, 1.0;
    factorization=:diag)
@test_throws ArgumentError assemble_excitation(
    mismatched_rwg_mesh, rwg, mismatched_rwg_excitation)
@test_throws ArgumentError assemble_multiple_excitations(
    mismatched_rwg_mesh, rwg, [mismatched_rwg_excitation])
@test_throws ArgumentError assemble_Z_efie(
    mismatched_rwg_mesh, rwg, mismatched_rwg_k;
    mesh_precheck=false)
@test_throws ArgumentError matrixfree_efie_operator(
    mismatched_rwg_mesh, rwg, mismatched_rwg_k;
    mesh_precheck=false)
@test_throws ArgumentError radiation_vectors(
    mismatched_rwg_mesh, rwg, mismatched_rwg_grid, mismatched_rwg_k)
@test_throws ArgumentError precompute_triangle_mass(
    mismatched_rwg_mesh, rwg)
@test_throws ArgumentError compute_nearfield(
    mismatched_rwg_mesh, rwg, zeros(ComplexF64, rwg.nedges),
    Vec3(0.0, 0.0, 1.0), mismatched_rwg_k)

# Verify RWG edge lengths are positive and areas are positive
@assert all(rwg.len .> 0)
@assert all(rwg.area_plus .> 0)
@assert all(rwg.area_minus .> 0)

println("  PASS ✓")

# Mesh generators reject inputs that would otherwise create empty, invalid, or
# non-finite geometry.
@test_throws ArgumentError make_rect_plate(1.0, 1.0, 0, 1)
@test_throws ArgumentError make_rect_plate(Inf, 1.0, 1, 1)
@test_throws ArgumentError make_circular_plate(1.0, 0, 8)
@test_throws ArgumentError make_circular_plate(1.0, 1, 2)
@test_throws ArgumentError make_rect_plate_graded(1.0, 1.0, 1, 0)
@test_throws ArgumentError make_rect_plate_graded(1.0, 1.0, 1, 1; grading_factor=NaN)
@test_throws ArgumentError make_rect_plate_graded(1.0, 1.0, 4, 4; grading_factor=1e6)
@test_throws ArgumentError make_parabolic_reflector(1.0, Inf, 2, 8)
@test_throws ArgumentError estimate_dense_matrix_gib(-1)

# Generator counts and raw payloads are checked before large allocations.
@test DiffMoM._mesh_raw_payload_bytes(4, 2) == 144
@test ntriangles(make_rect_plate(
    1.0, 1.0, 1, 1;
    max_vertices=4, max_triangles=2, max_raw_bytes=144)) == 2
@test_throws ArgumentError make_rect_plate(
    1.0, 1.0, 1, 1; max_raw_bytes=143)
@test_throws ArgumentError make_rect_plate(
    1.0, 1.0, 100_000, 100_000)
@test_throws ArgumentError make_circular_plate(
    1.0, 100_000, 100_000)
@test_throws ArgumentError make_parabolic_reflector(
    1.0, 1.0, 100_000, 100_000)
@test_throws ArgumentError DiffMoM._mesh_raw_payload_bytes(typemax(Int), 1)
for invalid_limit in (false, 0, -1, big(typemax(Int)) + 1)
    @test_throws ArgumentError make_rect_plate(
        1.0, 1.0, 1, 1; max_vertices=invalid_limit)
end

# Radial generators allocate only their final coordinate/connectivity payload
# plus bounded bookkeeping, without a growable connectivity copy.
make_circular_plate(1.0, 40, 200)
make_parabolic_reflector(1.0, 1.0, 40, 200)
GC.gc()
radial_raw_bytes = DiffMoM._mesh_raw_payload_bytes(
    DiffMoM._radial_mesh_counts(40, 200)...)
@test @allocated(make_circular_plate(1.0, 40, 200)) < 1.10 * radial_raw_bytes
@test @allocated(make_parabolic_reflector(1.0, 1.0, 40, 200)) <
      1.10 * radial_raw_bytes
@test_throws ArgumentError make_rect_plate(nextfloat(0.0), 1.0, 1, 1)
odd_subnormal_length = 3 * nextfloat(0.0)
@test_throws ArgumentError make_rect_plate(
    odd_subnormal_length, 1.0, 1, 1)
@test_throws ArgumentError make_rect_plate_graded(
    odd_subnormal_length, 1.0, 1, 1; grading_factor=1.0)
@test_throws ArgumentError make_rect_plate(
    ldexp(1.0, -537), ldexp(1.0, -537), 1, 1)
rect_subnormal_x = 1.0925309630530983e-204
rect_subnormal_y = 8.504075576651869e-120
rect_subnormal = make_rect_plate(
    rect_subnormal_x, rect_subnormal_y, 1, 1)
rect_graded_subnormal = make_rect_plate_graded(
    rect_subnormal_x, rect_subnormal_y, 1, 1; grading_factor=1.0)
@test all(triangle_area(rect_subnormal, t) == nextfloat(0.0)
          for t in 1:ntriangles(rect_subnormal))
@test all(triangle_area(rect_graded_subnormal, t) == nextfloat(0.0)
          for t in 1:ntriangles(rect_graded_subnormal))
@test_throws ArgumentError make_rect_plate_graded(
    1.0e-200, 1.0e-200, 1, 1)
plate_extreme_grid = make_rect_plate(
    floatmax(Float64), 1.0, 3, 1)
@test all(isfinite, plate_extreme_grid.xyz)
@test_throws ArgumentError make_circular_plate(nextfloat(0.0), 1, 3)
circle_subnormal_area = make_circular_plate(
    3.1434555694052576e-162, 1, 3)
@test all(triangle_area(circle_subnormal_area, t) > 0.0
          for t in 1:ntriangles(circle_subnormal_area))
@test_throws ArgumentError make_parabolic_reflector(
    nextfloat(0.0), 1.0, 2, 3)
@test_throws ArgumentError make_parabolic_reflector(
    4.8e-16, 1.0, 2, 3; center=Vec3(2.0, 2.0, 0.0))
@test_throws ArgumentError make_parabolic_reflector(
    2.0, 1.0, 2, 8; center=Vec3(0.0, 0.0, 1.0e308))
@test_throws ArgumentError make_parabolic_reflector(
    1.0e-160, 1.0e100, 2, 8)
@test_throws ArgumentError make_parabolic_reflector(
    1.0e-200, 1.0e-200, 2, 3)
@test_throws ArgumentError make_parabolic_reflector(
    2.0, 1.25e15, 2, 8; center=Vec3(0.0, 0.0, 1.0))
@test_throws ArgumentError make_parabolic_reflector(
    4 * (0.6eps(1.0)), 1.0, 2, 4; center=Vec3(1.0, 0.0, 0.0))
@test_throws ArgumentError make_parabolic_reflector(
    1.335e-15, 1.0, 3, 3; center=Vec3(2.0, 2.0, 0.0))
@test_throws ArgumentError make_parabolic_reflector(
    0.0018349836048392409, 8.530595403323751e-15, 3, 13)
@test DiffMoM._parabolic_reflector_height(
    floatmax(Float64) / 2, floatmax(Float64)) ==
      floatmax(Float64) / 16
parabola_subnormal_radius = 4.362219407585569e-78
parabola_subnormal_focal = 1.4292329290653017e168
@test DiffMoM._parabolic_reflector_height(
    parabola_subnormal_radius, parabola_subnormal_focal) ==
      nextfloat(0.0)
parabola_subnormal_height = make_parabolic_reflector(
    4 * parabola_subnormal_radius,
    parabola_subnormal_focal,
    2,
    3,
)
@test parabola_subnormal_height.xyz[3, 2] == nextfloat(0.0)
@test all(triangle_area(parabola_subnormal_height, t) > 0.0
          for t in 1:ntriangles(parabola_subnormal_height))
@test_throws ArgumentError make_parabolic_reflector(
    floatmax(Float64), floatmax(Float64), 2, 3)
@test_throws ArgumentError make_parabolic_reflector(
    1.6e307, 1.0e307, 2, 3;
    center=Vec3(-1.75e308, 0.0, 0.0))
@test_throws ArgumentError make_parabolic_reflector(
    4.0e-200, 2.5e-301, 2, 3)
reflector_extreme = make_parabolic_reflector(
    2.0e154, 1.0e154, 2, 8)
@test all(isfinite, reflector_extreme.xyz)
@test maximum(reflector_extreme.xyz[3, :]) == 2.5e153
@test_throws OverflowError make_parabolic_reflector(
    1.0e308, nextfloat(0.0), 100, 200;
    max_vertices=30_000, max_triangles=50_000,
    max_raw_bytes=2_000_000)

# The relative area tolerance must follow the actual mesh scale. A valid
# micrometre-scale plate must not be compared against a one-metre floor.
mesh_micro = make_rect_plate(1e-6, 1e-6, 1, 1)
report_micro = mesh_quality_report(mesh_micro)
@test report_micro.n_degenerate_triangles == 0
@test report_micro.area_tol_abs ≈ 2e-24
@test mesh_quality_ok(report_micro; allow_boundary=true)

# Non-finite geometry and empty meshes fail the quality gate explicitly.
xyz_nonfinite = copy(mesh_micro.xyz)
xyz_nonfinite[1, 1] = NaN
report_nonfinite = mesh_quality_report(TriMesh(xyz_nonfinite, copy(mesh_micro.tri)))
@test report_nonfinite.n_invalid_vertices == 1
@test report_nonfinite.n_invalid_triangles == 2
@test !mesh_quality_ok(report_nonfinite; allow_boundary=true)
@test_throws ErrorException assert_mesh_quality(
    TriMesh(xyz_nonfinite, copy(mesh_micro.tri));
    allow_boundary=true,
)

empty_mesh = TriMesh(zeros(3, 0), zeros(Int, 3, 0))
report_empty = mesh_quality_report(empty_mesh)
@test !mesh_quality_ok(report_empty; allow_boundary=true)
@test_throws ErrorException assert_mesh_quality(empty_mesh; allow_boundary=true)

# Face identity is independent of winding. A reversed duplicate must not make
# a coincident two-sided triangle look like a closed manifold surface.
duplicate_face_mesh = TriMesh(
    Float64[0 1 0; 0 0 1; 0 0 0],
    Int[1 3; 2 2; 3 1],
)
duplicate_face_report = mesh_quality_report(duplicate_face_mesh)
@test duplicate_face_report.n_duplicate_triangles == 1
@test duplicate_face_report.duplicate_triangles == [2]
@test duplicate_face_report.n_boundary_edges == 3
@test !mesh_quality_ok(duplicate_face_report; allow_boundary=true)
@test !mesh_quality_ok(
    duplicate_face_report; allow_boundary=false, require_closed=true)
@test_throws ErrorException assert_mesh_quality(
    duplicate_face_mesh; allow_boundary=true)
@test_throws ErrorException assert_mesh_quality(
    duplicate_face_mesh; allow_boundary=false, require_closed=true)
@test_throws ErrorException build_rwg(
    duplicate_face_mesh; allow_boundary=true)
@test_throws ErrorException build_rwg(
    duplicate_face_mesh; allow_boundary=false, require_closed=true)

# Reference-to-physical quadrature mapping must preserve a finite residual
# after large, opposing barycentric contributions cancel.
quad_cancel_first = 1.6348747758132425e226
quad_cancel_third = 1.4481861665399708e200
quad_cancel_mesh = TriMesh(
    Float64[
        quad_cancel_first -quad_cancel_first quad_cancel_third
        0.0 1.0 0.0
        0.0 0.0 1.0
    ],
    reshape(Int[1, 2, 3], 3, 1),
)
quad_cancel_xi = SVector(1 / 3, 1 / 3)
quad_cancel_reference = setprecision(BigFloat, 4352) do
    first_weight = 1 - BigFloat(quad_cancel_xi[1]) -
                   BigFloat(quad_cancel_xi[2])
    Float64(
        first_weight * BigFloat(quad_cancel_first) +
        BigFloat(quad_cancel_xi[1]) * BigFloat(-quad_cancel_first) +
        BigFloat(quad_cancel_xi[2]) * BigFloat(quad_cancel_third))
end
@test only(tri_quad_points(
    quad_cancel_mesh, 1, [quad_cancel_xi]))[1] == quad_cancel_reference

# ─────────────────────────────────────────────────
# Test 1b: OBJ mesh import
# ─────────────────────────────────────────────────
println("\n── Test 1b: OBJ mesh import ──")

obj_path = joinpath(DATADIR, "tmp_quad.obj")
open(obj_path, "w") do io
    println(io, "  v 0 0 0")
    println(io, "v\t1 0 0")
    println(io, "v 1 1 0 1")
    println(io, "v 0 1 0")
    println(io, "vt 0 0")
    println(io, "vn 0 0 1")
    println(io, "f 1/1/1 2/2/1 3//1 -1 # inline comment")
end

mesh_obj = read_obj_mesh(obj_path)
@assert nvertices(mesh_obj) == 4
@assert ntriangles(mesh_obj) == 2
@assert mesh_obj.tri == [1 1; 2 3; 3 4]

obj_invalid_path = joinpath(DATADIR, "tmp_invalid.obj")
open(obj_invalid_path, "w") do io
    println(io, "v NaN 0 0")
    println(io, "v 1 0 0")
    println(io, "v 0 1 0")
    println(io, "f 1 2 3")
end
@test_throws ErrorException read_obj_mesh(obj_invalid_path)

open(obj_invalid_path, "w") do io
    println(io, "v 0 0 0")
    println(io, "v 1 0 0")
    println(io, "v 0 1 0")
    println(io, "f 0 2 3")
end
@test_throws ErrorException read_obj_mesh(obj_invalid_path)

report_obj = assert_mesh_quality(mesh_obj; allow_boundary=true)
@assert report_obj.n_nonmanifold_edges == 0
@assert report_obj.n_orientation_conflicts == 0
@assert report_obj.n_degenerate_triangles == 0

# Orientation-conflict negative test (shared interior edge has same orientation)
tri_bad_orient = hcat([1, 2, 3], [1, 4, 3])
mesh_bad_orient = TriMesh(mesh_obj.xyz, tri_bad_orient)
thrown_orient = try
    assert_mesh_quality(mesh_bad_orient; allow_boundary=true)
    false
catch
    true
end
@assert thrown_orient

# Degenerate-triangle negative test
tri_bad_deg = hcat([1, 1, 2])
mesh_bad_deg = TriMesh(mesh_obj.xyz, tri_bad_deg)
thrown_deg = try
    assert_mesh_quality(mesh_bad_deg; allow_boundary=true)
    false
catch
    true
end
@assert thrown_deg

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1c: Mesh repair utility
# ─────────────────────────────────────────────────
println("\n── Test 1c: Mesh repair utility ──")

repair_orient = repair_mesh_for_simulation(mesh_bad_orient; allow_boundary=true)
@assert repair_orient.after.n_orientation_conflicts == 0
@assert !isempty(repair_orient.flipped_triangles)
@assert mesh_quality_ok(repair_orient.after; allow_boundary=true, require_closed=false)

xyz_bad_mixed = hcat(mesh_obj.xyz, mesh_obj.xyz[:, 1])
tri_bad_mixed = hcat([1, 2, 5], [1, 2, 3], [1, 6, 3])
mesh_bad_mixed = TriMesh(xyz_bad_mixed, tri_bad_mixed)
repair_mixed = repair_mesh_for_simulation(
    mesh_bad_mixed;
    allow_boundary=true,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=false,
)
@assert ntriangles(repair_mixed.mesh) == 1
@assert length(repair_mixed.removed_invalid) == 1
@assert length(repair_mixed.removed_degenerate) == 1

repair_duplicate = repair_mesh_for_simulation(
    duplicate_face_mesh; allow_boundary=true)
@test repair_duplicate.removed_duplicate == [2]
@test ntriangles(repair_duplicate.mesh) == 1
@test repair_duplicate.mesh.tri == reshape(Int[1, 2, 3], 3, 1)
@test repair_duplicate.after.n_duplicate_triangles == 0

# Duplicate removal must precede non-manifold cleanup: the third face below is
# a reversed copy of the first, not a reason to discard the valid square.
square_with_duplicate = TriMesh(
    copy(mesh_obj.xyz),
    hcat(mesh_obj.tri, Int[3, 2, 1]),
)
repair_square_duplicate = repair_mesh_for_simulation(
    square_with_duplicate; allow_boundary=true)
@test repair_square_duplicate.removed_duplicate == [3]
@test repair_square_duplicate.removed_nonmanifold == 0
@test repair_square_duplicate.mesh.tri == mesh_obj.tri

# Dropped invalid triangles and unused vertices are compacted and remapped.
# This lets documented repair discard an unused/non-referenced NaN vertex.
invalid_unused_vertex_mesh = TriMesh(
    Float64[0 1 0 NaN; 0 0 1 0; 0 0 0 0],
    Int[1 1; 2 2; 3 4],
)
repair_invalid_unused = repair_mesh_for_simulation(
    invalid_unused_vertex_mesh;
    allow_boundary=true,
    fix_orientation=false,
)
@test repair_invalid_unused.removed_invalid == [2]
@test repair_invalid_unused.removed_vertices == [4]
@test repair_invalid_unused.vertex_old_to_new == [1, 2, 3, 0]
@test nvertices(repair_invalid_unused.mesh) == 3
@test ntriangles(repair_invalid_unused.mesh) == 1
@test all(isfinite, repair_invalid_unused.mesh.xyz)
@test_throws ErrorException repair_mesh_for_simulation(
    invalid_unused_vertex_mesh;
    allow_boundary=true,
    drop_invalid=false,
)

# Orphan vertices do not define triangle scale. Otherwise a distant but
# unreferenced coordinate can make a valid retained face fail the relative
# area tolerance before compaction runs.
for orphan_coordinate in (1.0e6, 1.0e308)
    orphan_vertex_mesh = TriMesh(
        Float64[0 1 0 orphan_coordinate; 0 0 1 0; 0 0 0 0],
        reshape(Int[1, 2, 3], 3, 1),
    )
    orphan_report = mesh_quality_report(orphan_vertex_mesh)
    @test orphan_report.mesh_scale ≈ sqrt(2.0)
    @test orphan_report.n_degenerate_triangles == 0
    orphan_repair = repair_mesh_for_simulation(
        orphan_vertex_mesh; allow_boundary=true)
    @test orphan_repair.removed_vertices == [4]
    @test orphan_repair.vertex_old_to_new == [1, 2, 3, 0]
    @test triangle_area(orphan_repair.mesh, 1) == 0.5
end

# A triangle is excluded from the scale as a unit when any of its coordinates
# is non-finite; its other finite vertices must not poison retained geometry.
invalid_far_triangle_mesh = TriMesh(
    Float64[0 1 0 1.0e6 0 0; 0 0 1 0 0 0; 0 0 0 0 0 NaN],
    Int[1 4; 2 5; 3 6],
)
invalid_far_report = mesh_quality_report(invalid_far_triangle_mesh)
@test invalid_far_report.mesh_scale ≈ sqrt(2.0)
@test invalid_far_report.invalid_triangles == [2]
@test invalid_far_report.n_degenerate_triangles == 0
invalid_far_repair = repair_mesh_for_simulation(
    invalid_far_triangle_mesh; allow_boundary=true)
@test invalid_far_repair.removed_invalid == [2]
@test invalid_far_repair.removed_vertices == [4, 5, 6]
@test invalid_far_repair.vertex_old_to_new == [1, 2, 3, 0, 0, 0]

degenerate_far_triangle_mesh = TriMesh(
    Float64[0 1 0 1.0e6 2.0e6 3.0e6; 0 0 1 0 0 0; 0 0 0 0 0 0],
    Int[1 4; 2 5; 3 6],
)
degenerate_far_report = mesh_quality_report(degenerate_far_triangle_mesh)
@test degenerate_far_report.mesh_scale ≈ sqrt(2.0)
@test degenerate_far_report.degenerate_triangles == [2]
degenerate_far_repair = repair_mesh_for_simulation(
    degenerate_far_triangle_mesh; allow_boundary=true)
@test degenerate_far_repair.removed_degenerate == [2]
@test degenerate_far_repair.removed_vertices == [4, 5, 6]
@test degenerate_far_repair.vertex_old_to_new == [1, 2, 3, 0, 0, 0]

unused_nonfinite_vertex_mesh = TriMesh(
    Float64[0 1 0 NaN; 0 0 1 0; 0 0 0 0],
    reshape(Int[1, 2, 3], 3, 1),
)
repair_unused_nonfinite = repair_mesh_for_simulation(
    unused_nonfinite_vertex_mesh; allow_boundary=true)
@test isempty(repair_unused_nonfinite.removed_invalid)
@test repair_unused_nonfinite.removed_vertices == [4]
@test repair_unused_nonfinite.vertex_old_to_new == [1, 2, 3, 0]

# Cleaning uses scalar-indexed geometry, one retained connectivity buffer, and
# reuses quality reports on the no-flip path. Keep these bounded allocations
# from regressing to per-triangle column slices or redundant full scans.
mesh_repair_alloc = make_rect_plate(1.0, 1.0, 40, 40)
DiffMoM._clean_mesh_triangles(mesh_repair_alloc)
repair_mesh_for_simulation(mesh_repair_alloc)
GC.gc()
mesh_clean_allocated = @allocated DiffMoM._clean_mesh_triangles(
    mesh_repair_alloc)
GC.gc()
mesh_repair_allocated = @allocated repair_mesh_for_simulation(
    mesh_repair_alloc)
@test mesh_clean_allocated < 1_100_000
@test mesh_repair_allocated < 5_500_000

repair_in_path = joinpath(DATADIR, "tmp_repair_in.obj")
open(repair_in_path, "w") do io
    println(io, "v 0 0 0")
    println(io, "v 1 0 0")
    println(io, "v 1 1 0")
    println(io, "v 0 1 0")
    println(io, "f 1 2 3")
    println(io, "f 1 4 3")
end
repair_out_path = joinpath(DATADIR, "tmp_repair_out.obj")
repair_obj_result = repair_obj_mesh(repair_in_path, repair_out_path; allow_boundary=true)
@assert isfile(repair_obj_result.output_path)
mesh_repair_out = read_obj_mesh(repair_out_path)
report_repair_out = mesh_quality_report(mesh_repair_out)
@assert report_repair_out.n_orientation_conflicts == 0
@assert report_repair_out.n_nonmanifold_edges == 0
@test_throws ArgumentError repair_obj_mesh(
    repair_in_path, repair_out_path;
    reader_kwargs=(max_vertices=2,),
    allow_boundary=true,
)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1d: Mesh coarsening utilities
# ─────────────────────────────────────────────────
println("\n── Test 1d: Mesh coarsening utilities ──")

@assert estimate_dense_matrix_gib(100) > 0

mesh_cluster_in = make_rect_plate(1.0, 1.0, 6, 6)
mesh_cluster_out = cluster_mesh_vertices(mesh_cluster_in, 0.35)
@assert nvertices(mesh_cluster_out) > 0
@assert ntriangles(mesh_cluster_out) > 0
@test_throws ArgumentError cluster_mesh_vertices(mesh_cluster_in, Inf)
@test_throws ArgumentError cluster_mesh_vertices(mesh_cluster_in, NaN)

# Online cluster centroids must not overflow when several individually finite
# extreme coordinates occupy the same voxel.
cluster_extreme_x = 1.0e308
cluster_extreme_step = 1.0e293
mesh_cluster_extreme = TriMesh(
    Float64[
        cluster_extreme_x cluster_extreme_x cluster_extreme_x cluster_extreme_x cluster_extreme_x cluster_extreme_x;
        0 cluster_extreme_step 0 0 cluster_extreme_step 0;
        0 0 cluster_extreme_step 0 0 cluster_extreme_step
    ],
    Int[1 4; 2 5; 3 6],
)
mesh_cluster_extreme_out = cluster_mesh_vertices(
    mesh_cluster_extreme, 1.0e292)
@test all(isfinite, mesh_cluster_extreme_out.xyz)
@test nvertices(mesh_cluster_extreme_out) == 3
@test ntriangles(mesh_cluster_extreme_out) == 1
@test all(==(cluster_extreme_x), mesh_cluster_extreme_out.xyz[1, :])

# Origin subtraction may overflow for finite opposite-sign coordinates even
# when division by a large cell size makes the cell index representable.
mesh_cluster_wide = TriMesh(
    Float64[
        -1.0e308 1.0e308 -1.0e308;
        -1.0e308 -1.0e308 1.0e308;
        0 0 0
    ],
    reshape([1, 2, 3], 3, 1),
)
mesh_cluster_wide_out = cluster_mesh_vertices(mesh_cluster_wide, 1.5e308)
@test mesh_cluster_wide_out.xyz == mesh_cluster_wide.xyz
@test mesh_cluster_wide_out.tri == mesh_cluster_wide.tri
mesh_cluster_huge_index = cluster_mesh_vertices(mesh_cluster_wide, 0.25)
@test mesh_cluster_huge_index.xyz == mesh_cluster_wide.xyz
@test mesh_cluster_huge_index.tri == mesh_cluster_wide.tri
mesh_cluster_min_cell = cluster_mesh_vertices(
    mesh_cluster_in, nextfloat(0.0))
@test mesh_cluster_min_cell.xyz == mesh_cluster_in.xyz
@test mesh_cluster_min_cell.tri == mesh_cluster_in.tri
@test_throws ArgumentError cluster_mesh_vertices(
    mesh_cluster_in, nextfloat(0.0); max_exact_cell_indices=1)

# Merged centers are the correctly rounded mean of the stored coordinates;
# avoid double-rounding a wide exact integer before division by the count.
cluster_mean_base = Float64(2^53)
cluster_mean_third = nextfloat(cluster_mean_base, 5)
cluster_mean_mesh = TriMesh(
    Float64[
        cluster_mean_base cluster_mean_base cluster_mean_third cluster_mean_base cluster_mean_base;
        0 0 0 32 0;
        0 0 0 0 32
    ],
    reshape(Int[1, 4, 5], 3, 1),
)
cluster_mean_result = cluster_mesh_vertices(cluster_mean_mesh, 16.0)
cluster_mean_reference = setprecision(BigFloat, 4352) do
    Float64((BigFloat(cluster_mean_base) + BigFloat(cluster_mean_base) +
             BigFloat(cluster_mean_third)) / 3)
end
@test cluster_mean_result.xyz[1, 1] == cluster_mean_reference

# The common merged-cell path must not allocate one BigFloat mean per output
# coordinate.
mesh_cluster_merge_alloc = make_rect_plate(1.0, 1.0, 100, 100)
cluster_mesh_vertices(mesh_cluster_merge_alloc, 0.015)
GC.gc()
cluster_merge_alloc = @allocated cluster_mesh_vertices(
    mesh_cluster_merge_alloc, 0.015)
@test cluster_merge_alloc < 12_000_000

# A no-merge pass must preserve topology without allocating a temporary
# three-element sort buffer for every triangle.
mesh_cluster_alloc = make_rect_plate(1.0, 1.0, 40, 40)
mesh_cluster_alloc_out = cluster_mesh_vertices(mesh_cluster_alloc, 1.0e-12)
@test mesh_cluster_alloc_out.xyz == mesh_cluster_alloc.xyz
@test mesh_cluster_alloc_out.tri == mesh_cluster_alloc.tri
GC.gc()
cluster_alloc_bytes = @allocated cluster_mesh_vertices(
    mesh_cluster_alloc, 1.0e-12)
@test cluster_alloc_bytes < 1_500_000

mesh_edges_test = make_rect_plate(1.0, 1.0, 1, 1) # two triangles
edges_test = mesh_unique_edges(mesh_edges_test)
@assert length(edges_test) == 5
@test_throws ArgumentError mesh_unique_edges(
    mesh_edges_test; max_edge_records=5)
segments_test = mesh_wireframe_segments(mesh_edges_test)
@assert segments_test.n_edges == 5
@assert length(segments_test.x) == 15
@assert count(isnan, segments_test.x) == 5
wireframe_output_bytes = 3 * length(segments_test.x) * sizeof(Float64)
@test_throws ArgumentError mesh_wireframe_segments(
    mesh_edges_test; max_output_bytes=wireframe_output_bytes - 1)
segments_limited = mesh_wireframe_segments(
    mesh_edges_test; max_output_bytes=wireframe_output_bytes)
@test isequal(segments_limited.x, segments_test.x)
@test isequal(segments_limited.y, segments_test.y)
@test isequal(segments_limited.z, segments_test.z)
@test segments_limited.n_edges == segments_test.n_edges

p_mesh = plot_mesh_wireframe(mesh_edges_test; title="Mesh preview test", linewidth=0.5)
@assert p_mesh !== nothing
@test_throws ArgumentError plot_mesh_wireframe(
    mesh_edges_test; max_output_bytes=wireframe_output_bytes - 1)
p_cmp = plot_mesh_comparison(mesh_edges_test, mesh_edges_test; title_a="A", title_b="B", size=(600, 300))
@assert p_cmp !== nothing
@test_throws ArgumentError plot_mesh_comparison(
    mesh_edges_test, mesh_edges_test;
    max_output_bytes=2wireframe_output_bytes - 1)

axis_limit_translation = 1.0e308
axis_limit_mesh = TriMesh(
    Float64[
        axis_limit_translation axis_limit_translation axis_limit_translation
        0 1 0
        0 0 1
    ],
    reshape(Int[1, 2, 3], 3, 1),
)
axis_limit_ranges = DiffMoM._realistic_axis_limits([axis_limit_mesh])
@test all(limits -> all(isfinite, limits), axis_limit_ranges)
@test all(limits -> limits[1] < limits[2], axis_limit_ranges)
@test axis_limit_ranges[1][1] <= axis_limit_translation <=
      axis_limit_ranges[1][2]
@test_throws ArgumentError DiffMoM._realistic_axis_limits(
    [axis_limit_mesh]; pad_frac=-0.01)
@test_throws ArgumentError DiffMoM._realistic_axis_limits(
    [axis_limit_mesh]; pad_frac=Inf)

xyz_nm = [
    0.0  1.0  0.0  0.0  0.0  2.0  2.0;
    0.0  0.0  1.0 -1.0  0.0  0.0  1.0;
    0.0  0.0  0.0  0.0  1.0  0.0  0.0
]
tri_nm = [
    1  1  1  3;
    2  2  2  6;
    3  4  5  7
]
mesh_nm = TriMesh(xyz_nm, tri_nm)
mesh_nm_clean = drop_nonmanifold_triangles(mesh_nm)
report_nm = mesh_quality_report(mesh_nm_clean)
@assert report_nm.n_nonmanifold_edges == 0
@test_throws ArgumentError drop_nonmanifold_triangles(mesh_nm; max_passes=0)
for malformed_rows in (1, 2)
    malformed_drop_mesh = TriMesh(
        Float64[0 1 0; 0 0 1; 0 0 0],
        reshape(collect(1:malformed_rows), malformed_rows, 1),
    )
    @test_throws DimensionMismatch drop_nonmanifold_triangles(
        malformed_drop_mesh)
end
mesh_nm_alloc = make_rect_plate(1.0, 1.0, 100, 100)
drop_nonmanifold_triangles(mesh_nm_alloc)
GC.gc()
@test @allocated(drop_nonmanifold_triangles(mesh_nm_alloc)) < 5_000_000
mesh_nm_with_orphan = TriMesh(
    hcat(Float64[9.0, 9.0, 9.0], xyz_nm),
    tri_nm .+ 1,
)
mesh_nm_with_orphan_xyz = copy(mesh_nm_with_orphan.xyz)
mesh_nm_with_orphan_tri = copy(mesh_nm_with_orphan.tri)
repair_nm = repair_mesh_for_simulation(
    mesh_nm_with_orphan; allow_boundary=true)
@test repair_nm.removed_nonmanifold == 3
@test repair_nm.removed_vertices == [1, 2, 3, 5, 6]
@test repair_nm.vertex_old_to_new == [0, 0, 0, 1, 0, 0, 2, 3]
@test nvertices(repair_nm.mesh) == 3
@test ntriangles(repair_nm.mesh) == 1
@test mesh_nm_with_orphan.xyz == mesh_nm_with_orphan_xyz
@test mesh_nm_with_orphan.tri == mesh_nm_with_orphan_tri
@test !Base.mightalias(repair_nm.mesh.xyz, mesh_nm_with_orphan.xyz)
@test !Base.mightalias(repair_nm.mesh.tri, mesh_nm_with_orphan.tri)

mesh_coarse_in = make_rect_plate(1.0, 1.0, 12, 12)
rwg_coarse_in = build_rwg(mesh_coarse_in; precheck=true, allow_boundary=true)
@assert rwg_coarse_in.nedges > 60

coarse_exact = coarsen_mesh_to_target_rwg(
    mesh_coarse_in, rwg_coarse_in.nedges; max_iters=1)
@test coarse_exact.mesh === mesh_coarse_in
@test coarse_exact.rwg_count == rwg_coarse_in.nedges
@test coarse_exact.best_gap == 0
@test coarse_exact.iterations == 0

coarse_close_target = rwg_coarse_in.nedges - 1
coarse_close = coarsen_mesh_to_target_rwg(
    mesh_coarse_in, coarse_close_target; max_iters=1)
@test coarse_close.mesh === mesh_coarse_in
@test coarse_close.rwg_count == rwg_coarse_in.nedges
@test coarse_close.best_gap == 1
@test coarse_close.iterations == 0

# A valid boundary mesh can have no interior RWG functions; preserve the
# documented no-op behavior when it is already below the target.
zero_rwg_mesh = TriMesh(
    Float64[0 1 0; 0 0 1; 0 0 0], reshape(Int[1, 2, 3], 3, 1))
zero_rwg_result = coarsen_mesh_to_target_rwg(zero_rwg_mesh, 1)
@test zero_rwg_result.mesh === zero_rwg_mesh
@test zero_rwg_result.rwg_count == 0
@test zero_rwg_result.iterations == 0
for invalid_vertex in (0, 4)
    invalid_coarsen_mesh = TriMesh(
        copy(zero_rwg_mesh.xyz),
        reshape(Int[1, 2, invalid_vertex], 3, 1),
    )
    @test_throws ArgumentError coarsen_mesh_to_target_rwg(
        invalid_coarsen_mesh, 1)
end
for coordinate_rows in (1, 2)
    malformed_coordinate_mesh = TriMesh(
        zeros(Float64, coordinate_rows, 3),
        reshape(Int[1, 2, 3], 3, 1),
    )
    @test_throws DimensionMismatch coarsen_mesh_to_target_rwg(
        malformed_coordinate_mesh, 1)
end
for connectivity_rows in (1, 2)
    malformed_connectivity_mesh = TriMesh(
        copy(zero_rwg_mesh.xyz),
        reshape(collect(1:connectivity_rows), connectivity_rows, 1),
    )
    @test_throws DimensionMismatch coarsen_mesh_to_target_rwg(
        malformed_connectivity_mesh, 1)
end
for invalid_coordinate in (NaN, Inf)
    invalid_orphan_mesh = TriMesh(
        hcat(zero_rwg_mesh.xyz, Float64[invalid_coordinate, 0.0, 0.0]),
        copy(zero_rwg_mesh.tri),
    )
    @test_throws ArgumentError coarsen_mesh_to_target_rwg(
        invalid_orphan_mesh, 1)
end

# Candidate over-collapse and owned quality failures are coarse-side bracket
# events, not fatal errors or resource-limit suppression.
@test DiffMoM._recoverable_coarsening_candidate_error(
    ArgumentError(
        "cluster_mesh_vertices: clustering removed all triangles"))
@test DiffMoM._recoverable_coarsening_candidate_error(
    ErrorException(
        "drop_nonmanifold_triangles: empty mesh after cleanup."))
@test !DiffMoM._recoverable_coarsening_candidate_error(
    OverflowError("Mesh quality precheck failed:"))

target_rwg = 60
coarse_result = coarsen_mesh_to_target_rwg(mesh_coarse_in, target_rwg; max_iters=8)
rwg_coarse_out = build_rwg(coarse_result.mesh; precheck=true, allow_boundary=true)
@assert coarse_result.rwg_count == rwg_coarse_out.nedges
@assert abs(coarse_result.rwg_count - target_rwg) <= target_rwg  # improved complexity scale
@assert rwg_coarse_out.nedges < rwg_coarse_in.nedges
@test_throws ArgumentError coarsen_mesh_to_target_rwg(mesh_coarse_in, target_rwg; max_iters=0)
report_coarse_out = mesh_quality_report(coarse_result.mesh)
@assert mesh_quality_ok(report_coarse_out; allow_boundary=true, require_closed=false)

# Distinct voxel centroids can still become exactly collinear; remove the
# resulting zero-area face just like an index-collapsed face.
cluster_collinear_input = TriMesh(
    Float64[
        0.1 0.1 1.1 1.1 2.1 2.1;
        0.2 0.8 0.2 0.8 0.8 0.2;
        0.0 0.0 0.0 0.0 0.0 0.0
    ],
    Int[1 2; 3 4; 5 6],
)
@test_throws ArgumentError cluster_mesh_vertices(
    cluster_collinear_input, 1.0)
cluster_area_underflow_input = let
    length_scale = 1.0e-161
    offset = 1.0e-162
    TriMesh(
        Float64[
            0 0 length_scale length_scale 2length_scale 2length_scale;
            -offset offset offset -offset -offset nextfloat(offset);
            0 0 0 0 0 0
        ],
        Int[1 2; 3 4; 5 6],
    )
end
@test_throws ArgumentError cluster_mesh_vertices(
    cluster_area_underflow_input, 0.75e-161)
cluster_area_underflow_duplicates = TriMesh(
    cluster_area_underflow_input.xyz,
    repeat(cluster_area_underflow_input.tri, 1, 1_000),
)
try
    cluster_mesh_vertices(
        cluster_area_underflow_duplicates, 0.75e-161)
catch
end
GC.gc()
@test @allocated(try
    cluster_mesh_vertices(
        cluster_area_underflow_duplicates, 0.75e-161)
catch
end) < 500_000
@test_throws ArgumentError cluster_mesh_vertices(
    cluster_area_underflow_input, 0.75e-161;
    max_exact_area_checks=0)

# Coarsening is surface-dimensional and scale invariant; a fixed absolute
# length floor must not collapse a nanoscale plate or overflow a huge one.
for scale in (1.0e-100, 1.0e150, 1.0e155)
    scaled_input = make_rect_plate(scale, scale, 12, 12)
    scaled_result = coarsen_mesh_to_target_rwg(
        scaled_input, target_rwg; max_iters=8)
    if scale < 1.0e155
        @test scaled_result.rwg_count == coarse_result.rwg_count
        @test scaled_result.best_gap == coarse_result.best_gap
    else
        # Some coarser triangles have unrepresentable areas at this scale;
        # they bracket the search while the best valid candidate is retained.
        @test scaled_result.best_gap < rwg_coarse_in.nedges - target_rwg
        @test scaled_result.rwg_count == build_rwg(
            scaled_result.mesh; precheck=true, allow_boundary=true).nedges
    end
    @test mesh_quality_ok(
        mesh_quality_report(scaled_result.mesh); allow_boundary=true)
end

# An overflowing total area is kept as a scaled pair; sizing must not rescan
# every triangle with allocating BigFloat arithmetic.
coarse_area_alloc_mesh = make_rect_plate(1.0e155, 1.0e155, 40, 40)
DiffMoM._mesh_surface_area_for_coarsening(coarse_area_alloc_mesh)
GC.gc()
@test @allocated(
    DiffMoM._mesh_surface_area_for_coarsening(coarse_area_alloc_mesh)) < 5_000_000
coarse_tiny_area_alloc_mesh = make_rect_plate(1.0e-160, 1.0e-160, 40, 40)
DiffMoM._mesh_surface_area_for_coarsening(coarse_tiny_area_alloc_mesh)
GC.gc()
@test @allocated(
    DiffMoM._mesh_surface_area_for_coarsening(
        coarse_tiny_area_alloc_mesh)) < 5_000_000

coarse_tiny_runtime_mesh = make_rect_plate(1.0e-160, 1.0e-160, 12, 12)
coarsen_mesh_to_target_rwg(
    coarse_tiny_runtime_mesh, target_rwg; max_iters=2)
GC.gc()
@test @allocated(coarsen_mesh_to_target_rwg(
    coarse_tiny_runtime_mesh, target_rwg; max_iters=2)) < 20_000_000

# When the physical area tolerance rounds to zero, normalized-space repair
# must not delete representable minsubnormal faces. This multi-face mesh enters
# the candidate path and collapses to one such valid face.
coarse_minsub_vertices = Float64[
    8.513298007032766e-163 1.7984151419124296e-162 2.148749960278492e-162;
    -8.029981995707232e-163 1.273363791932202e-162 -2.1764978143105622e-162;
    -6.030051840716439e-163 5.554832536677963e-163 9.659195691234607e-164
]
coarse_minsub_xyz = Matrix{Float64}(undef, 3, 40)
coarse_minsub_tri = Matrix{Int}(undef, 3, 20)
for group in 0:9
    first = 4group + 1
    coarse_minsub_xyz[:, first] = coarse_minsub_vertices[:, 1]
    coarse_minsub_xyz[:, first + 1] = coarse_minsub_vertices[:, 2]
    coarse_minsub_xyz[:, first + 2] = coarse_minsub_vertices[:, 3]
    coarse_minsub_xyz[:, first + 3] = coarse_minsub_vertices[:, 3]
    coarse_minsub_tri[:, 2group + 1] .=
        (first, first + 1, first + 2)
    coarse_minsub_tri[:, 2group + 2] .=
        (first + 1, first, first + 3)
end
coarse_minsub_mesh = TriMesh(coarse_minsub_xyz, coarse_minsub_tri)
coarse_minsub_result = coarsen_mesh_to_target_rwg(
    coarse_minsub_mesh, 1; max_iters=1, area_tol_rel=0.2)
@test coarse_minsub_result.mesh !== coarse_minsub_mesh
@test coarse_minsub_result.iterations == 1
@test ntriangles(coarse_minsub_result.mesh) == 1
@test triangle_area(coarse_minsub_result.mesh, 1) == nextfloat(0.0)
@test mesh_quality_ok(
    mesh_quality_report(
        coarse_minsub_result.mesh; area_tol_rel=0.2);
    allow_boundary=true)

# A power-of-two quality normalization must preserve the physical side of a
# borderline relative-area decision rather than rounding a degenerate face up.
coarse_borderline_xyz = Float64[
    -13.67789190253212 20.605906214258475 -24.77060790944614;
    -8.71347428782219 6.5380022079351425 -12.739690403750291;
    1.3425004850768831 -6.73902980326434 -8.555526751828038
]
coarse_borderline_mesh = TriMesh(
    coarse_borderline_xyz, reshape(Int[1, 2, 3], 3, 1))
coarse_borderline_tol = 0.09246594018146315
@test mesh_quality_report(
    coarse_borderline_mesh;
    area_tol_rel=coarse_borderline_tol).n_degenerate_triangles == 1
@test_throws ErrorException coarsen_mesh_to_target_rwg(
    coarse_borderline_mesh, 1; area_tol_rel=coarse_borderline_tol)

# If no single power-of-two scale preserves every referenced component, the
# physical cold oracle remains authoritative and a valid no-op still works.
coarse_full_span_mesh = TriMesh(
    Float64[
        floatmax(Float64) prevfloat(floatmax(Float64)) floatmax(Float64);
        0.0 0.0 1.0;
        0.0 0.0 nextfloat(0.0)
    ],
    reshape(Int[1, 2, 3], 3, 1),
)
coarse_full_span_result = coarsen_mesh_to_target_rwg(
    coarse_full_span_mesh, 1; area_tol_rel=0.0)
@test coarse_full_span_result.mesh === coarse_full_span_mesh
@test coarse_full_span_result.rwg_count == 0

# Independently rounded subnormal physical area and tolerance can compare
# differently after scaling; the physical API contract remains authoritative.
coarse_subnormal_tolerance_mesh = TriMesh(
    Float64[
        2.9492954239971775e-162 -1.2160397752307908e-162 -2.4016383330038877e-162;
        4.7200179453485664e-164 -2.6906428858668346e-162 -3.2201182285196426e-162;
        -1.856601432683537e-162 -5.3020301050002416e-163 9.129922820188425e-163
    ],
    reshape(Int[1, 2, 3], 3, 1),
)
@test mesh_quality_report(
    coarse_subnormal_tolerance_mesh;
    area_tol_rel=0.05).n_degenerate_triangles == 1
@test_throws ErrorException coarsen_mesh_to_target_rwg(
    coarse_subnormal_tolerance_mesh, 1; area_tol_rel=0.05)

# Orphan coordinates cannot set the normalization scale of retained geometry.
coarse_orphan_input = TriMesh(
    hcat(mesh_coarse_in.xyz, Float64[1.0e308, 1.0e308, 1.0e308]),
    copy(mesh_coarse_in.tri),
)
coarse_orphan_result = coarsen_mesh_to_target_rwg(
    coarse_orphan_input, target_rwg; max_iters=8)
@test coarse_orphan_result.rwg_count == coarse_result.rwg_count
@test coarse_orphan_result.best_gap == coarse_result.best_gap
@test nvertices(coarse_orphan_result.mesh) < nvertices(coarse_orphan_input)
for orphan_value in (-1.0e308, 1.0e308)
    signed_orphan_input = TriMesh(
        hcat(mesh_coarse_in.xyz,
             Float64[orphan_value, orphan_value, orphan_value]),
        copy(mesh_coarse_in.tri),
    )
    signed_orphan_result = coarsen_mesh_to_target_rwg(
        signed_orphan_input, 200; max_iters=8)
    base_target_200 = coarsen_mesh_to_target_rwg(
        mesh_coarse_in, 200; max_iters=8)
    @test signed_orphan_result.rwg_count == base_target_200.rwg_count
    @test signed_orphan_result.best_gap == base_target_200.best_gap
    @test signed_orphan_result.mesh.xyz == base_target_200.mesh.xyz
    @test signed_orphan_result.mesh.tri == base_target_200.mesh.tri
end

report_res_before = mesh_resolution_report(mesh_edges_test, 3e8; points_per_wavelength=2.0)
@assert report_res_before.wavelength_m ≈ 299792458.0 / 3e8
@assert report_res_before.edge_max_m > report_res_before.target_max_edge_m
@assert !mesh_resolution_ok(report_res_before)
@test_throws ArgumentError mesh_resolution_report(mesh_edges_test, Inf; points_per_wavelength=2.0)
@test_throws ArgumentError mesh_resolution_report(mesh_edges_test, 3e8; points_per_wavelength=Inf)
@test_throws ArgumentError mesh_resolution_report(
    mesh_edges_test, 3e8; points_per_wavelength=2.0, c0=Inf)
@test_throws ArgumentError mesh_resolution_report(
    mesh_edges_test, floatmin(Float64); points_per_wavelength=2.0)

refine_result = refine_mesh_to_target_edge(mesh_edges_test, 0.40; max_iters=3)
@assert refine_result.triangles_after > refine_result.triangles_before
@assert refine_result.edge_max_after_m <= 0.40 + 1e-12
@assert refine_result.converged
@test refine_result.stop_reason == :target_reached
@test_throws ArgumentError refine_mesh_to_target_edge(mesh_edges_test, Inf)
@test_throws ArgumentError refine_mesh_to_target_edge(mesh_edges_test, NaN)
@test_throws ArgumentError refine_mesh_to_target_edge(
    mesh_edges_test, 0.40; max_output_bytes=0)

refine_triangle_limit = refine_mesh_to_target_edge(
    mesh_edges_test, 0.40; max_iters=3, max_triangles=2)
@test !refine_triangle_limit.converged
@test refine_triangle_limit.iterations == 0
@test refine_triangle_limit.stop_reason == :max_triangles
@test refine_triangle_limit.mesh === mesh_edges_test

refine_byte_limit = refine_mesh_to_target_edge(
    mesh_edges_test, 0.40; max_iters=3, max_output_bytes=1)
@test !refine_byte_limit.converged
@test refine_byte_limit.iterations == 0
@test refine_byte_limit.stop_reason == :max_output_bytes
@test refine_byte_limit.mesh === mesh_edges_test

# The payload cap is exact, not a worst-case unique-edge estimate. This plate
# has four input vertices, five unique edges, and eight output triangles.
refine_exact_payload_bytes =
    3 * sizeof(Float64) * (4 + 5) + 3 * sizeof(Int) * 8
refine_exact_byte_limit = refine_mesh_to_target_edge(
    mesh_edges_test, 0.1;
    max_iters=1,
    max_triangles=100,
    max_output_bytes=refine_exact_payload_bytes,
)
@test refine_exact_byte_limit.iterations == 1
@test nvertices(refine_exact_byte_limit.mesh) == 9
@test ntriangles(refine_exact_byte_limit.mesh) == 8

midpoint_first = -0.7380437951395766
midpoint_second = -0.0993679039589958
@test DiffMoM._safe_midpoint_component(midpoint_first, midpoint_second) ==
      DiffMoM._safe_midpoint_component(midpoint_second, midpoint_first)
@test DiffMoM._safe_midpoint_component(midpoint_first, midpoint_second) ==
      -0.41870584954928625
midpoint_minsub = nextfloat(0.0)
@test DiffMoM._safe_midpoint_component(
    -127 * midpoint_minsub, -126 * midpoint_minsub) ==
      -126 * midpoint_minsub
@test DiffMoM._safe_midpoint_component(
    -128 * midpoint_minsub, -126 * midpoint_minsub) ==
      -127 * midpoint_minsub

# Adjacent Float64 coordinates cannot always be split by a representable
# midpoint. Stop before mutation instead of emitting repeated/non-finite
# vertices and NaN edge diagnostics.
refine_resolution_mesh = TriMesh(
    Float64[1.0e308 nextfloat(1.0e308) 1.0e308; 0 0 1; 0 0 0],
    reshape(Int[1, 2, 3], 3, 1),
)
refine_resolution_xyz = copy(refine_resolution_mesh.xyz)
refine_resolution_tri = copy(refine_resolution_mesh.tri)
refine_resolution_result = refine_mesh_to_target_edge(
    refine_resolution_mesh, 0.1; max_iters=1)
@test !refine_resolution_result.converged
@test refine_resolution_result.iterations == 0
@test refine_resolution_result.stop_reason == :coordinate_resolution
@test refine_resolution_result.mesh === refine_resolution_mesh
@test all(isfinite, refine_resolution_result.mesh.xyz)
@test refine_resolution_mesh.xyz == refine_resolution_xyz
@test refine_resolution_mesh.tri == refine_resolution_tri

report_res_after = mesh_resolution_report(refine_result.mesh, 3e8; points_per_wavelength=2.0)
@assert mesh_resolution_ok(report_res_after)

mom_refine = refine_mesh_for_mom(mesh_edges_test, 3e8; points_per_wavelength=2.0, max_iters=3)
@assert mom_refine.report_before.edge_max_m > mom_refine.report_before.target_max_edge_m
@assert mom_refine.report_after.edge_max_m <= mom_refine.report_after.target_max_edge_m
@assert mom_refine.converged

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 1e: Parabolic reflector mesh
# ─────────────────────────────────────────────────
println("\n── Test 1e: Parabolic reflector mesh ──")

D_ref = 0.30
f_ref = 0.105
Nr_ref = 6
Nphi_ref = 20
mesh_ref = make_parabolic_reflector(D_ref, f_ref, Nr_ref, Nphi_ref)
report_ref = mesh_quality_report(mesh_ref)
@assert mesh_quality_ok(report_ref; allow_boundary=true, require_closed=false)
@assert nvertices(mesh_ref) == 1 + Nr_ref * Nphi_ref
@assert ntriangles(mesh_ref) == Nphi_ref + 2 * (Nr_ref - 1) * Nphi_ref

# Paraboloid checks at rim points: z=r²/(4f), and distance-to-focus = z+f.
focus = Vec3(0.0, 0.0, f_ref)
rim_start = 2 + (Nr_ref - 1) * Nphi_ref
for idx in rim_start:4:(rim_start + Nphi_ref - 1)
    p = Vec3(mesh_ref.xyz[:, idx])
    rxy = hypot(p[1], p[2])
    z_expected = rxy^2 / (4 * f_ref)
    @assert abs(p[3] - z_expected) < 1e-12
    d_focus = norm(p - focus)
    @assert abs(d_focus - (p[3] + f_ref)) < 1e-12
end

rwg_ref = build_rwg(mesh_ref; precheck=true, allow_boundary=true)
@assert rwg_ref.nedges > 0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 2: Green's Function
# ─────────────────────────────────────────────────
println("\n── Test 2: Green's function ──")

k0 = 2π / 0.1   # wavelength = 10 cm
r1 = Vec3(0.0, 0.0, 0.0)
r2 = Vec3(0.05, 0.0, 0.0)
R = norm(r2 - r1)

G = greens(r1, r2, k0)
G_expected = exp(-1im * k0 * R) / (4π * R)
@assert abs(G - G_expected) < 1e-14

# Check reciprocity: G(r,r') = G(r',r)
@assert abs(greens(r1, r2, k0) - greens(r2, r1, k0)) < 1e-14
@test_throws ArgumentError greens(r1, r2, NaN)
@test_throws ArgumentError greens_smooth(
    Vec3(NaN, 0.0, 0.0), r2, k0)
@test_throws ArgumentError grad_greens(
    r1, Vec3(Inf, 0.0, 0.0), k0)
@test _green_kernel_allocations(r1, r2, k0) == (0, 0, 0)

# These references evaluate the mathematical kernels at 8192-bit precision
# from the exact Float64 inputs before converting the complete result.
green_tiny_point = Vec3(1e-31, 0.0, 0.0)
green_tiny_reference = _green_kernel_bigfloat_reference(
    r1, green_tiny_point, 1.0)
@test greens(r1, green_tiny_point, 1.0) == green_tiny_reference[1]
green_tiny_smooth = greens_smooth(r1, green_tiny_point, 1.0)
@test isapprox(real(green_tiny_smooth), real(green_tiny_reference[2]);
               rtol=2eps(Float64), atol=0.0)
@test imag(green_tiny_smooth) == imag(green_tiny_reference[2])
@test !iszero(real(green_tiny_smooth))
green_tiny_gradient = grad_greens(r1, green_tiny_point, 1.0)
@test isapprox(real(green_tiny_gradient[1]),
               real(green_tiny_reference[3][1]);
               rtol=2eps(Float64), atol=0.0)
@test isapprox(imag(green_tiny_gradient[1]),
               imag(green_tiny_reference[3][1]);
               rtol=2eps(Float64), atol=0.0)
@test !iszero(imag(green_tiny_gradient[1]))
@test green_tiny_gradient[2:3] == green_tiny_reference[3][2:3]

green_subnormal_point = Vec3(1e-200, 0.0, 0.0)
green_subnormal_reference = _green_kernel_bigfloat_reference(
    r1, green_subnormal_point, 1.0)
@test greens(r1, green_subnormal_point, 1.0) ==
      green_subnormal_reference[1]
@test greens_smooth(r1, green_subnormal_point, 1.0) ==
      green_subnormal_reference[2]
@test_throws OverflowError grad_greens(
    r1, green_subnormal_point, 1.0)

green_large_point = Vec3(1e160, 0.0, 0.0)
green_large_reference = _green_kernel_bigfloat_reference(
    r1, green_large_point, 1e-160)
@test greens(r1, green_large_point, 1e-160) == green_large_reference[1]
@test greens_smooth(r1, green_large_point, 1e-160) ==
      green_large_reference[2]
@test grad_greens(r1, green_large_point, 1e-160) ==
      green_large_reference[3]

green_phase_point = Vec3(2.0, 0.0, 0.0)
green_phase_reference = _green_kernel_bigfloat_reference(
    r1, green_phase_point, 1e308)
@test greens(r1, green_phase_point, 1e308) == green_phase_reference[1]
@test greens_smooth(r1, green_phase_point, 1e308) ==
      green_phase_reference[2]
@test grad_greens(r1, green_phase_point, 1e308) ==
      green_phase_reference[3]

# A finite but large radial phase also needs exact-input reduction before it
# reaches outright Float64 overflow. Rounding kR first changes the unit-circle
# phase by macroscopic amounts for these exactly supplied Float64 operands.
green_finite_phase_point = Vec3(1.0e20, 0.0, 0.0)
green_finite_phase_k = 1.1
green_finite_phase_reference = _green_kernel_bigfloat_reference(
    green_finite_phase_point, r1, green_finite_phase_k)
@test greens(green_finite_phase_point, r1, green_finite_phase_k) ==
      green_finite_phase_reference[1]
@test greens_smooth(green_finite_phase_point, r1, green_finite_phase_k) ==
      green_finite_phase_reference[2]
@test grad_greens(green_finite_phase_point, r1, green_finite_phase_k) ==
      green_finite_phase_reference[3]

green_max_point = Vec3(
    floatmax(Float64), floatmax(Float64), floatmax(Float64))
green_min_point = -green_max_point
green_max_reference = _green_kernel_bigfloat_reference(
    green_max_point, green_min_point, floatmax(Float64))
@test greens(green_max_point, green_min_point, floatmax(Float64)) ==
      green_max_reference[1]
@test greens_smooth(
    green_max_point, green_min_point, floatmax(Float64)) ==
      green_max_reference[2]
@test grad_greens(
    green_max_point, green_min_point, floatmax(Float64)) ==
      green_max_reference[3]

green_growth_k = ComplexF64(0.0, log(1e308) / 1e308)
green_growth_point = Vec3(1e308, 0.0, 0.0)
green_growth_reference = _green_kernel_bigfloat_reference(
    green_growth_point, r1, green_growth_k)
@test greens(green_growth_point, r1, green_growth_k) ==
      green_growth_reference[1]
@test greens_smooth(green_growth_point, r1, green_growth_k) ==
      green_growth_reference[2]
@test grad_greens(green_growth_point, r1, green_growth_k) ==
      green_growth_reference[3]

green_phase_one_reference = _green_kernel_bigfloat_reference(
    r1, Vec3(1e-15, 0.0, 0.0), 1e15)
@test isapprox(
    greens_smooth(r1, Vec3(1e-15, 0.0, 0.0), 1e15),
    green_phase_one_reference[2];
    rtol=2eps(Float64),
    atol=0.0,
)

@test greens(r1, r1, 2.3 + 0.4im) == 0.0 + 0.0im
@test greens_smooth(r1, r1, 2.3 + 0.4im) ==
      _green_kernel_bigfloat_reference(r1, r1, 2.3 + 0.4im)[2]
@test grad_greens(r1, r1, 2.3 + 0.4im) ==
      CVec3(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im)

# Generic SVector inputs promote narrow coordinate arithmetic before forming
# distances; otherwise finite Float32 separations can overflow their dot sum.
green_float32_point = SVector{3,Float32}(3.0f38, 0.0f0, 0.0f0)
green_float32_origin = zero(green_float32_point)
green_float32_k = 1.0f-38
green_float32_reference = setprecision(BigFloat, 256) do
    distance = BigFloat(green_float32_point[1])
    phase_rate = -Complex{BigFloat}(0, 1) * BigFloat(green_float32_k)
    green = exp(phase_rate * distance) /
            (4 * BigFloat(pi) * distance)
    smooth = expm1(phase_rate * distance) /
             (4 * BigFloat(pi) * distance)
    derivative = (phase_rate - inv(distance)) * green
    (
        ComplexF64(green),
        ComplexF64(smooth),
        CVec3(ComplexF64(derivative), 0.0 + 0im, 0.0 + 0im),
    )
end
@test greens(
    green_float32_point, green_float32_origin, green_float32_k) ≈
    green_float32_reference[1] rtol=4eps(Float64)
@test greens_smooth(
    green_float32_point, green_float32_origin, green_float32_k) ≈
    green_float32_reference[2] rtol=4eps(Float64)
@test grad_greens(
    green_float32_point, green_float32_origin, green_float32_k) ≈
    green_float32_reference[3] rtol=4eps(Float64)

# The promoted Float32 backend must retain range-safe final scaling as well as
# range-safe geometry. Here exp(imag(k)R) overflows by itself, while division
# by the large distance leaves every returned kernel representable.
green_float32_growth_k = ComplexF32(0.0f0, 2.5f-36)
green_float32_growth_reference = _green_kernel_bigfloat_reference(
    Vec3(green_float32_point), Vec3(green_float32_origin),
    ComplexF64(green_float32_growth_k))
@test greens(
    green_float32_point, green_float32_origin,
    green_float32_growth_k) == green_float32_growth_reference[1]
@test greens_smooth(
    green_float32_point, green_float32_origin,
    green_float32_growth_k) == green_float32_growth_reference[2]
@test grad_greens(
    green_float32_point, green_float32_origin,
    green_float32_growth_k) == green_float32_growth_reference[3]

# Non-floating generic coordinates cannot be rounded through Float64 before
# subtraction: these two valid Int128 points differ by one even though both
# convert to the same Float64 value.
green_int128_source = SVector{3,Int128}(Int128(2)^100, 0, 0)
green_int128_point = green_int128_source + SVector{3,Int128}(1, 0, 0)
green_unit_reference = _green_kernel_bigfloat_reference(
    Vec3(1.0, 0.0, 0.0), r1, 1.0)
@test greens(green_int128_point, green_int128_source, 1.0) ==
      green_unit_reference[1]
@test greens_smooth(green_int128_point, green_int128_source, 1.0) ==
      green_unit_reference[2]
@test grad_greens(green_int128_point, green_int128_source, 1.0) ==
      green_unit_reference[3]

# These are true Float64 output-range boundaries, not intermediate failures.
@test_throws OverflowError greens(r1, Vec3(1e-310, 0.0, 0.0), 0.0)
@test isfinite(greens(r1, Vec3(1e-309, 0.0, 0.0), 0.0))
@test_throws OverflowError grad_greens(
    r1, Vec3(1e-200, 0.0, 0.0), 0.0)
@test all(isfinite, grad_greens(
    r1, Vec3(3e-155, 0.0, 0.0), 0.0))

@test _green_kernel_allocations(
    r1, Vec3(0.4, -0.2, 0.1), 2.3 + 1e-30im) == (0, 0, 0)
@test _green_kernel_allocations(
    r1, Vec3(1e-15, 0.0, 0.0), 1.0) == (0, 0, 0)
@test sum(_green_kernel_allocations(
    green_max_point, green_min_point, floatmax(Float64))) <= 100_000

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 2b: PEC sphere Mie utilities
# ─────────────────────────────────────────────────
println("\n── Test 2b: PEC Mie-theory utilities ──")

k_mie = 2π / 0.1
a_mie = 0.03
θ_mie = 60 * π / 180
khat_mie = Vec3(0.0, 0.0, 1.0)
pol_mie = Vec3(1.0, 0.0, 0.0)
rhat_mie = Vec3(sin(θ_mie), 0.0, cos(θ_mie))
μ_mie = dot(khat_mie, rhat_mie)

S1_mie, S2_mie = mie_s1s2_pec(k_mie * a_mie, μ_mie)
@assert isfinite(real(S1_mie)) && isfinite(imag(S1_mie))
@assert isfinite(real(S2_mie)) && isfinite(imag(S2_mie))

σ_formula = 4π * abs2(S2_mie) / (k_mie^2)   # φ=0 plane for x-pol
σ_mie = mie_bistatic_rcs_pec(k_mie, a_mie, khat_mie, pol_mie, rhat_mie)
@assert σ_mie >= 0
@assert abs(σ_mie - σ_formula) / max(abs(σ_formula), 1e-30) < 1e-10
S1_clamped, S2_clamped = mie_s1s2_pec(k_mie * a_mie, 1.0 + 5e-13)
S1_endpoint, S2_endpoint = mie_s1s2_pec(k_mie * a_mie, 1.0)
@test S1_clamped == S1_endpoint
@test S2_clamped == S2_endpoint
@test_throws ArgumentError mie_s1s2_pec(Inf, 0.0; nmax=3)
@test_throws ArgumentError mie_s1s2_pec(1.0, NaN; nmax=3)
@test_throws ArgumentError mie_s1s2_pec(1.0, 0.0; nmax=2.5)
@test_throws ArgumentError mie_bistatic_rcs_pec(
    1.0, 1.0, Vec3(0.0, 0.0, 0.0), pol_mie, rhat_mie; nmax=3)
@test_throws OverflowError mie_bistatic_rcs_pec(
    1.0e-300, 1.0e300, khat_mie, pol_mie, rhat_mie; nmax=3)

# Stable exterior Riccati seeds avoid cancellation in j₁ for Rayleigh-size
# spheres. These are independent 100-digit direct-reference values.
mie_small_pec_reference = (
    -7.166666666666666e-49 - 8.5e-25im,
    -3.666666666666666e-49 + 2.0e-25im,
)
mie_small_pec = mie_s1s2_pec(1.0e-8, 0.3; nmax=1)
for component in 1:2
    @test isapprox(
        mie_small_pec[component], mie_small_pec_reference[component];
        rtol=8eps(Float64), atol=0.0)
end
mie_small_pec_overtruncated = mie_s1s2_pec(1.0e-8, 0.3; nmax=4)
mie_s1s2_pec(1.0e-8, 0.3; nmax=4)
mie_small_pec_overtruncated_reference = (
    -7.166666666666667e-49 - 8.500000000000002e-25im,
    -3.666666666666667e-49 + 1.9999999999999998e-25im,
)
for component in 1:2
    @test isapprox(
        mie_small_pec_overtruncated[component],
        mie_small_pec_overtruncated_reference[component];
        rtol=4eps(Float64), atol=0.0)
end
@test (@allocated mie_s1s2_pec(1.0e-8, 0.3; nmax=4)) == 0
mie_series_boundary_reference = 0.04160159881993972
@test isapprox(
    DiffMoM._mie_exterior_initial_pair(0.125)[3],
    mie_series_boundary_reference;
    rtol=4eps(Float64), atol=0.0)

S1_dielectric_mie, S2_dielectric_mie =
    mie_s1s2_dielectric(k_mie * a_mie, μ_mie, 2.5)
@test isfinite(S1_dielectric_mie)
@test isfinite(S2_dielectric_mie)
σ_dielectric_mie = mie_bistatic_rcs_dielectric(
    k_mie, a_mie, khat_mie, pol_mie, rhat_mie, 2.5)
@test isfinite(σ_dielectric_mie) && σ_dielectric_mie >= 0.0
@test_throws ArgumentError mie_s1s2_dielectric(
    1.0, 0.0, Inf; nmax=3)
@test_throws ArgumentError mie_s1s2_dielectric(
    1.0, 0.0, 2.0; mu_r=Inf, nmax=3)
@test_throws OverflowError mie_bistatic_rcs_dielectric(
    1.0e-300, 1.0e300, khat_mie, pol_mie, rhat_mie, 2.0; nmax=3)
mie_small_dielectric_reference = (
    -1.6666666666666676e-49 - 5.000000000000002e-25im,
    -5.000000000000003e-50 - 1.5000000000000022e-25im,
)
mie_small_dielectric = mie_s1s2_dielectric(
    1.0e-8, 0.3, 4.0; nmax=1)
for component in 1:2
    @test isapprox(
        mie_small_dielectric[component],
        mie_small_dielectric_reference[component];
        rtol=8eps(Float64), atol=0.0)
end
mie_small_dielectric_overtruncated = mie_s1s2_dielectric(
    1.0e-8, 0.3, 4.0; nmax=4)
mie_s1s2_dielectric(1.0e-8, 0.3, 4.0; nmax=4)
mie_small_dielectric_overtruncated_reference = (
    -1.6666666666666669e-49 - 5.0000000000000005e-25im,
    -5.000000000000001e-50 - 1.5000000000000001e-25im,
)
for component in 1:2
    @test isapprox(
        mie_small_dielectric_overtruncated[component],
        mie_small_dielectric_overtruncated_reference[component];
        rtol=8eps(Float64), atol=0.0)
end
@test (@allocated mie_s1s2_dielectric(
    1.0e-8, 0.3, 4.0; nmax=4)) == 0
@test mie_s1s2_pec(1.0e-155, 0.3; nmax=3) ==
      (0.0 + 0.0im, 0.0 + 0.0im)
@test mie_s1s2_dielectric(1.0e-155, 0.3, 4.0; nmax=3) ==
      (0.0 + 0.0im, 0.0 + 0.0im)
mie_exact_exterior = mie_s1s2_dielectric(
    1.0e-80, 0.3, 4.0; nmax=3)
@test isapprox(
    mie_exact_exterior[1], -5.0e-241im;
    rtol=8eps(Float64), atol=0.0)
mie_tiny_material = mie_s1s2_dielectric(
    0.25, 0.3, nextfloat(0.0);
    mu_r=nextfloat(0.0), nmax=1)
mie_tiny_material_reference = (
    -4.916654355974263e-5 + 0.009791442314209512im,
    -4.916654355974263e-5 + 0.009791442314209512im,
)
for component in 1:2
    @test isapprox(
        mie_tiny_material[component], mie_tiny_material_reference[component];
        rtol=8eps(Float64), atol=0.0)
end
@test isapprox(
    mie_exact_exterior[2], -1.5e-241im;
    rtol=8eps(Float64), atol=0.0)

# Exterior Neumann growth depends on order as well as the exponent of x.
# These 512-bit direct Riccati references remain finite after the ordinary
# forward recurrence overflows.
mie_high_order_pec = mie_s1s2_pec(0.25, 0.3; nmax=150)
mie_high_order_pec_reference = (
    -1.7995571987718538e-4 - 0.013710967389325831im,
    -8.840154453061714e-5 + 0.0028435522730447292im,
)
for component in 1:2
    @test isapprox(
        mie_high_order_pec[component],
        mie_high_order_pec_reference[component];
        rtol=8eps(Float64), atol=0.0)
end
mie_high_order_dielectric = mie_s1s2_dielectric(
    0.25, 0.3, 4.0; nmax=150)
mie_high_order_dielectric_reference = (
    -4.1697685263328135e-5 - 0.007950907520523177im,
    -1.2514490424262779e-5 - 0.00243484060392419im,
)
for component in 1:2
    @test isapprox(
        mie_high_order_dielectric[component],
        mie_high_order_dielectric_reference[component];
        rtol=8eps(Float64), atol=0.0)
end
mie_large_high_order_pec = mie_s1s2_pec(
    100.0, 0.3; nmax=522)
mie_large_high_order_pec_reference = (
    43.83989971698044 + 24.23536856612706im,
    -43.69747383309498 - 25.03245063864703im,
)
for component in 1:2
    @test isapprox(
        mie_large_high_order_pec[component],
        mie_large_high_order_pec_reference[component];
        rtol=8eps(Float64), atol=0.0)
end
@test !DiffMoM._mie_requires_exact_exterior(10_000.0, 10_089)
mie_large_default = mie_s1s2_pec(10_000.0, 0.3)
@test all(isfinite, mie_large_default)
mie_s1s2_pec(10_000.0, 0.3)
@test @allocated(mie_s1s2_pec(10_000.0, 0.3)) < 4_096
@test_throws ArgumentError mie_s1s2_pec(
    0.25, 0.3; nmax=DiffMoM._MAX_MIE_ORDER)

# A partial-wave amplitude may underflow before division by a tiny physical
# wavenumber restores a representable RCS.  Convert only the final sigma.
mie_tiny_rcs = mie_bistatic_rcs_pec(
    1e-200, 1e90,
    Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 0.0, -1.0); nmax=1)
@test mie_tiny_rcs ≈ 2.827433388230813e-259 rtol=4eps(Float64)
mie_tiny_dielectric_rcs = mie_bistatic_rcs_dielectric(
    1e-200, 1e90,
    Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 0.0, -1.0), 4.0; nmax=1)
@test mie_tiny_dielectric_rcs ≈ π * 1e-260 rtol=8eps(Float64)
# At the quasistatic eps=-2 resonance, the cancellation depth is O(x^2).
# The exact-product path therefore derives its precision from the exponent of
# the exact k*a product and converts only the final, representable RCS.
mie_resonant_tiny_rcs = mie_bistatic_rcs_dielectric(
    nextfloat(0.0), ldexp(1.0, -537),
    Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0),
    Vec3(1.0, 0.0, 0.0), -2.0; nmax=1)
@test mie_resonant_tiny_rcs == reinterpret(Float64, UInt64(0x14))
# Magnetodielectric parameters can cancel both leading denominator terms at
# n=1.  The O(x^3) resonance remains finite and must be resolved before the
# exact-product result is rounded.
mie_double_resonant_rcs = mie_bistatic_rcs_dielectric(
    0.25, nextfloat(0.0),
    Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0),
    Vec3(1.0, 0.0, 0.0), -2.0; mu_r=-5.0, nmax=1)
@test mie_double_resonant_rcs == 144π
mie_tiny_selected_rcs = mie_bistatic_rcs_pec(
    1e-200, 1e130,
    Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0),
    Vec3(sqrt(3) / 2, 0.0, 0.5); nmax=1)
@test isapprox(
    mie_tiny_selected_rcs, 2.5446900494077325e-300;
    rtol=8eps(Float64), atol=0.0)

# A weighted amplitude that is merely subnormal (rather than exactly zero)
# has already lost significant bits before division by a tiny k can recover
# the physical RCS.  The public geometry below selects that component.
mie_weight_unit = nextfloat(0.0)
mie_weighted_rcs = mie_bistatic_rcs_pec(
    4.493409457909064e-300, 1.0e300,
    Vec3(0.0, 0.0, 1.0), Vec3(1.0, mie_weight_unit, 0.0),
    Vec3(1.0, 0.0, 0.0); nmax=1)
@test mie_weighted_rcs ≈ 3.417881064525094e-47 rtol=8eps(Float64)

# An explicit bounded partial-wave request remains evaluable when the physical
# product k*a overflows Float64; only the final RCS is converted to Float64.
mie_overflow_product_rcs = mie_bistatic_rcs_pec(
    2.0, 1e308,
    Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 0.0, -1.0); nmax=1)
@test mie_overflow_product_rcs == 7.0685834705770345
@test_throws ArgumentError mie_bistatic_rcs_pec(
    2.0, 1e308,
    Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 0.0, -1.0))
mie_large_internal_tiny_rcs = mie_bistatic_rcs_dielectric(
    1e-200, 1e90,
    Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 0.0, -1.0), 1e300; nmax=1)
@test isapprox(
    mie_large_internal_tiny_rcs, 2.827433388230814e-259;
    rtol=8eps(Float64), atol=0.0)
# The exact-product dielectric kernel also budgets phase-reduction precision
# from the internal m*x exponent, not just the exterior size parameter.
mie_large_internal_phase_rcs = mie_bistatic_rcs_dielectric(
    ldexp(1.0, 512), floatmax(Float64),
    Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0),
    Vec3(1.0, 0.0, 0.0), floatmax(Float64);
    mu_r=prevfloat(floatmax(Float64)), nmax=1)
@test reinterpret(UInt64, mie_large_internal_phase_rcs) ==
      UInt64(0x0037e7b36280bcf9)
matched_mie_rcs = mie_bistatic_rcs_dielectric(
    1.0, 0.5, khat_mie, pol_mie, rhat_mie, 1.0; nmax=3)
@test matched_mie_rcs == 0.0
for invalid_matched_order in (0, 2.5, DiffMoM._MAX_MIE_ORDER + 1)
    @test_throws ArgumentError mie_bistatic_rcs_dielectric(
        1.0, 0.5, khat_mie, pol_mie, rhat_mie, 1.0;
        nmax=invalid_matched_order)
end
mie_bistatic_rcs_dielectric(
    1.0, 0.5, khat_mie, pol_mie, rhat_mie, 1.0; nmax=3)
@test @allocated(mie_bistatic_rcs_dielectric(
    1.0, 0.5, khat_mie, pol_mie, rhat_mie, 1.0; nmax=3)) < 1_024

# Dielectric truncation follows the exterior size parameter, while the
# internal Riccati-Bessel logarithmic derivative remains stable when n >> |mx|.
dielectric_low_index_auto_reference = (
    4.6088330451342935 - 3.146310706699757im,
    3.561319349966622 + 2.9102874774898675im,
)
dielectric_low_index_converged_reference = (
    4.608833045134293 - 3.1463107067010153im,
    3.561319349966622 + 2.9102874774420013im,
)
dielectric_low_index_auto = mie_s1s2_dielectric(10.0, 0.3, 0.01)
dielectric_low_index_converged =
    mie_s1s2_dielectric(10.0, 0.3, 0.01; nmax=40)
for component in 1:2
    @test isapprox(
        dielectric_low_index_auto[component],
        dielectric_low_index_auto_reference[component];
        rtol=16eps(Float64),
        atol=0.0,
    )
    @test isapprox(
        dielectric_low_index_converged[component],
        dielectric_low_index_converged_reference[component];
        rtol=24eps(Float64),
        atol=0.0,
    )
end

# Normalize both the material factors and the internal function/derivative
# pair so no raw material-times-log-derivative product can overflow.
dielectric_extreme_impedance_reference = (
    -0.1993938916576317 + 0.10751820637457842im,
    -0.4582979515538648 - 0.5883400318961542im,
)
dielectric_extreme_impedance = mie_s1s2_dielectric(
    1.0, 0.3, nextfloat(0.0); mu_r=1.0e308, nmax=1)
for component in 1:2
    @test isapprox(
        dielectric_extreme_impedance[component],
        dielectric_extreme_impedance_reference[component];
        rtol=8eps(Float64),
        atol=0.0,
    )
end

# A conservatively selected, normalized forward recurrence handles very large
# internal real arguments in O(nmax) work instead of exhausting the CF cap.
dielectric_large_internal_reference = (
    -0.44596372945719465 - 0.694106100496583im,
    -0.12062065641203042 + 0.3061794991004623im,
)
dielectric_large_internal = mie_s1s2_dielectric(
    1.0, 0.125, 1.0e16; nmax=7)
for component in 1:2
    @test isapprox(
        dielectric_large_internal[component],
        dielectric_large_internal_reference[component];
        rtol=8eps(Float64),
        atol=0.0,
    )
end

# Direct complex sin/cos overflows for this passive lossy material even though
# the coefficient ratios are finite.
dielectric_lossy_reference = (
    -0.45852058587254385 - 0.6473396256779773im,
    -0.1977908038030188 + 0.1842542218799932im,
)
dielectric_lossy = mie_s1s2_dielectric(
    1.0, 0.3, 1.0 - 1.0e8im; nmax=3)
for component in 1:2
    @test isapprox(
        dielectric_lossy[component],
        dielectric_lossy_reference[component];
        rtol=8eps(Float64),
        atol=0.0,
    )
end

# Balanced material and geometry scales retain a finite internal size
# parameter even though the raw material product underflows.
dielectric_balanced = mie_s1s2_dielectric(
    1.0e300, 0.3, 1.0e-300; mu_r=1.0e-300, nmax=1)
dielectric_balanced_reference =
    -1.582279580752163 - 0.7627820860517617im
@test isapprox(
    dielectric_balanced[1], dielectric_balanced_reference;
    rtol=4eps(Float64), atol=0.0)
@test isapprox(
    dielectric_balanced[2], dielectric_balanced_reference;
    rtol=4eps(Float64), atol=0.0)
@test mie_s1s2_dielectric(10.0, 0.3, 1.0; nmax=20) ==
      (0.0 + 0.0im, 0.0 + 0.0im)
@test_throws ArgumentError mie_s1s2_dielectric(
    1.0, 0.3, 2.0; nmax=DiffMoM._MAX_MIE_ORDER + 1)

mie_s1s2_pec(2.0, 0.2; nmax=20)
mie_s1s2_dielectric(2.0, 0.2, 2.5; nmax=20)
mie_bistatic_rcs_pec(
    2.0, 1.0, khat_mie, pol_mie, rhat_mie; nmax=20)
@test (@allocated mie_s1s2_pec(2.0, 0.2; nmax=20)) == 0
@test (@allocated mie_s1s2_dielectric(
    2.0, 0.2, 2.5; nmax=20)) == 0
@test (@allocated mie_s1s2_dielectric(
    1.0, 0.3, nextfloat(0.0); mu_r=1.0e308, nmax=1)) == 0
@test (@allocated mie_s1s2_dielectric(
    1.0, 0.125, 1.0e16; nmax=7)) == 0
@test (@allocated mie_s1s2_dielectric(
    10.0, 0.3, 0.01; nmax=65)) <= 2_048
@test (@allocated mie_s1s2_dielectric(
    1.0e300, 0.3, 1.0e-300; mu_r=1.0e-300, nmax=1)) <= 10_000
@test (@allocated mie_bistatic_rcs_pec(
    2.0, 1.0, khat_mie, pol_mie, rhat_mie; nmax=20)) == 0

# Independent energy-consistency check:
# integrate differential cross section over sphere and compare to
# coefficient-sum total scattering cross section Csca.
function _mie_csca_coeff_pec(k::Float64, a::Float64; nmax::Int=80)
    x = k * a
    j = zeros(Float64, nmax + 1)  # j[n+1] = j_n
    y = zeros(Float64, nmax + 1)  # y[n+1] = y_n
    j[1] = sin(x) / x
    y[1] = -cos(x) / x
    j[2] = sin(x) / x^2 - cos(x) / x
    y[2] = -cos(x) / x^2 - sin(x) / x
    for n in 1:(nmax - 1)
        j[n + 2] = ((2n + 1) / x) * j[n + 1] - j[n]
        y[n + 2] = ((2n + 1) / x) * y[n + 1] - y[n]
    end

    psi = zeros(Float64, nmax + 1)
    xi  = zeros(ComplexF64, nmax + 1)
    for n in 0:nmax
        psi[n + 1] = x * j[n + 1]
        xi[n + 1]  = x * (j[n + 1] - 1im * y[n + 1])
    end

    csum = 0.0
    for n in 1:nmax
        psi_p = psi[n] - (n / x) * psi[n + 1]
        xi_p  = xi[n] - (n / x) * xi[n + 1]
        an = -psi_p / xi_p
        bn = -psi[n + 1] / xi[n + 1]
        csum += (2n + 1) * (abs2(an) + abs2(bn))
    end
    return (2π / (k^2)) * csum
end

grid_mie = make_sph_grid(181, 180)
σ_mie_grid = [
    mie_bistatic_rcs_pec(k_mie, a_mie, khat_mie, pol_mie, Vec3(grid_mie.rhat[:, q]))
    for q in 1:length(grid_mie.w)
]
Csca_num = sum((σ_mie_grid ./ (4π)) .* grid_mie.w)
Csca_ref = _mie_csca_coeff_pec(k_mie, a_mie; nmax=80)
@assert abs(Csca_num - Csca_ref) / max(abs(Csca_ref), 1e-30) < 2e-3

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 3: EFIE Assembly (PEC plate)
# ─────────────────────────────────────────────────
println("\n── Test 3: EFIE assembly ──")

freq = 3e9            # 3 GHz
c0 = 299792458.0
lambda0 = c0 / freq
k = 2π / lambda0
eta0 = 376.730313668

Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
efie_matrix_bytes = sizeof(ComplexF64) * rwg.nedges^2
@test_throws ArgumentError assemble_Z_efie(
    mesh, rwg, k;
    quad_order=3,
    eta0=eta0,
    max_output_bytes=efie_matrix_bytes - 1)
@test assemble_Z_efie(
    mesh, rwg, k;
    quad_order=3,
    eta0=eta0,
    max_output_bytes=efie_matrix_bytes) == Z_efie
N = rwg.nedges
println("  Z_efie size: $N × $N")
@assert size(Z_efie) == (N, N)
@assert all(isfinite, Z_efie)

for invalid_efie_k in (0.0, Inf, -Inf, NaN)
    @test_throws ArgumentError assemble_Z_efie(
        mesh, rwg, invalid_efie_k; quad_order=1, eta0=eta0,
        mesh_precheck=false)
end
for unrepresentable_efie_k in (1.0e-300, 1.0e300)
    @test_throws OverflowError assemble_Z_efie(
        mesh, rwg, unrepresentable_efie_k; quad_order=1, eta0=eta0,
        mesh_precheck=false)
end
for invalid_efie_eta0 in (0.0, Inf, NaN)
    @test_throws ArgumentError assemble_Z_efie(
        mesh, rwg, k; quad_order=1, eta0=invalid_efie_eta0,
        mesh_precheck=false)
end
@test_throws ArgumentError matrixfree_efie_operator(
    mesh, rwg, 0.0; quad_order=1, eta0=eta0, mesh_precheck=false)
@test_throws ArgumentError matrixfree_efie_operator(
    mesh, rwg, k; quad_order=1, eta0=Inf, mesh_precheck=false)

# Complex multiplication can overflow a rounded intermediate even when both
# exact product components fit in Float64. Preserve that representable
# prefactor instead of rejecting valid finite inputs.
efie_prefactor_scale = sqrt(floatmax(Float64))
efie_prefactor_k = complex(
    -efie_prefactor_scale, -0.5 * efie_prefactor_scale)
efie_prefactor_eta = complex(
    -1.2 * efie_prefactor_scale, -0.4 * efie_prefactor_scale)
efie_prefactor_reference = setprecision(BigFloat, 4352) do
    ComplexF64(
        Complex{BigFloat}(efie_prefactor_k) *
        Complex{BigFloat}(efie_prefactor_eta))
end
_, _, efie_prefactor = DiffMoM._validated_efie_prefactors(
    efie_prefactor_k, efie_prefactor_eta)
@test isfinite(efie_prefactor)
@test efie_prefactor == efie_prefactor_reference
@test_throws OverflowError DiffMoM._validated_efie_prefactors(
    2.0, floatmax(Float64))

# Finite, individually representable physical inputs can still produce an
# entry outside ComplexF64. The low-level/public entry contract must fail
# closed instead of returning an Inf that later contaminates an operator.
efie_overflow_mesh = make_rect_plate(1.0, 1.0, 1, 1)
efie_overflow_rwg = build_rwg(efie_overflow_mesh)
efie_overflow_operator = matrixfree_efie_operator(
    efie_overflow_mesh,
    efie_overflow_rwg,
    1e-3;
    quad_order=1,
    eta0=floatmax(Float64),
    mesh_precheck=false,
)
@test_throws OverflowError efie_entry(efie_overflow_operator, 1, 1)
@test_throws OverflowError DiffMoM._fill_dense_block_batched!(
    zeros(ComplexF64, 1, 1),
    efie_overflow_operator.cache,
    [1],
    [1],
)

Z_efie_complex_k = assemble_Z_efie(
    mesh, rwg, complex(k, 1.0e-20); quad_order=1, eta0=eta0,
    mesh_precheck=false)
@test all(isfinite, Z_efie_complex_k)

# Z should have nonzero entries
@assert norm(Z_efie) > 0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 4: PEC Scattering (plane wave excitation)
# ─────────────────────────────────────────────────
println("\n── Test 4: PEC forward solve ──")

# Normal-incidence plane wave, x-polarized
k_vec = Vec3(0.0, 0.0, -k)    # propagating in -z
E0 = 1.0
pol = Vec3(1.0, 0.0, 0.0)     # x-polarized

v = assemble_v_plane_wave(mesh, rwg, k_vec, E0, pol; quad_order=3)
@assert length(v) == N
@assert norm(v) > 0

# Solve PEC EFIE: Z_efie * I = v
I_pec = Z_efie \ v
println("  |I_pec| = $(norm(I_pec))")
@assert norm(I_pec) > 0

# Residual check
residual = norm(Z_efie * I_pec - v) / norm(v)
println("  Relative residual: $residual")
@assert residual < 1e-10

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 5: Impedance Term and Derivatives
# ─────────────────────────────────────────────────
println("\n── Test 5: Impedance term and derivatives ──")

dense_bilinear_matrix = randn(MersenneTwister(500), ComplexF64, 32, 32)
dense_bilinear_left = randn(MersenneTwister(501), ComplexF64, 32)
dense_bilinear_right = randn(MersenneTwister(502), ComplexF64, 32)
@assert DiffMoM._dot_left_matrix_right(
    dense_bilinear_left, dense_bilinear_matrix, dense_bilinear_right) ≈
    dot(dense_bilinear_left, dense_bilinear_matrix * dense_bilinear_right)
@assert _bilinear_allocation(
    dense_bilinear_left, dense_bilinear_matrix, dense_bilinear_right) <= 128
sparse_bilinear_matrix = sparse(dense_bilinear_matrix)
@assert DiffMoM._dot_left_matrix_right(
    dense_bilinear_left, sparse_bilinear_matrix, dense_bilinear_right) ≈
    dot(dense_bilinear_left, sparse_bilinear_matrix * dense_bilinear_right)
@assert _bilinear_allocation(
    dense_bilinear_left, sparse_bilinear_matrix, dense_bilinear_right) <= 128

Nt = ntriangles(mesh)
# Simple partition: one patch per triangle
partition = PatchPartition(collect(1:Nt), Nt)

@test_throws ArgumentError PatchPartition([1], -1)
@test_throws ArgumentError PatchPartition([0], 1)
@test_throws ArgumentError PatchPartition([2], 1)
@test_throws DimensionMismatch precompute_patch_mass(
    mesh, rwg, PatchPartition([1], 1))
mutated_partition = PatchPartition(fill(1, Nt), 1)
mutated_partition.tri_patch[1] = 0
@test_throws ArgumentError precompute_patch_mass(mesh, rwg, mutated_partition)

Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)
@assert length(Mp) == Nt
mass_patch_nq = length(tri_quad_rule(3)[2])
mass_patch_profile = DiffMoM._mass_precompute_profile(
    rwg, Nt, mass_patch_nq, Float64, partition.tri_patch, partition.P)
@test_throws ArgumentError precompute_patch_mass(
    mesh, rwg, partition;
    quad_order=3,
    max_work_bytes=mass_patch_profile.work_bytes - 1,
    max_terms=mass_patch_profile.term_count)
@test_throws ArgumentError precompute_patch_mass(
    mesh, rwg, partition;
    quad_order=3,
    max_work_bytes=mass_patch_profile.work_bytes,
    max_terms=mass_patch_profile.term_count - 1)
Mp_at_resource_boundary = precompute_patch_mass(
    mesh, rwg, partition;
    quad_order=3,
    max_work_bytes=mass_patch_profile.work_bytes,
    max_terms=mass_patch_profile.term_count)
@test Matrix.(Mp_at_resource_boundary) == Matrix.(Mp)

large_mass_length = 2.0^512
large_mass_mesh = TriMesh(
    Float64[
        0 large_mass_length 0 large_mass_length
        0 0 large_mass_length large_mass_length
        0 0 0 0
    ],
    Int[
        1 2
        2 4
        3 3
    ],
)
large_mass_rwg = build_rwg(large_mass_mesh)
@test large_mass_rwg.nedges == 1
@test all(isfinite, triangle_area.(Ref(large_mass_mesh), 1:2))
@test !isfinite(2 * triangle_area(large_mass_mesh, 1))
large_triangle_mass = precompute_triangle_mass(
    large_mass_mesh, large_mass_rwg)
large_mass_reference = map(1:2) do triangle
    setprecision(BigFloat, DiffMoM._LOCAL_MASS_FALLBACK_PRECISION) do
        xi, weights = tri_quad_rule(3)
        points = tri_quad_points(large_mass_mesh, triangle, xi)
        total = zero(BigFloat)
        for quadrature_index in eachindex(weights)
            basis = eval_rwg(
                large_mass_rwg, 1, points[quadrature_index], triangle)
            total += BigFloat(weights[quadrature_index]) *
                     sum(abs2, BigFloat.(basis))
        end
        Float64(total * 2 * BigFloat(triangle_area(
            large_mass_mesh, triangle)))
    end
end
@test [large_triangle_mass[t][1, 1] for t in 1:2] ≈
      large_mass_reference rtol=2eps(Float64)
large_patch_mass = precompute_patch_mass(
    large_mass_mesh, large_mass_rwg, PatchPartition([1, 1], 1))
large_patch_reference = setprecision(
        BigFloat, DiffMoM._LOCAL_MASS_FALLBACK_PRECISION) do
    Float64(sum(BigFloat, large_mass_reference))
end
large_patch_assembled_reference = setprecision(
        BigFloat, DiffMoM._LOCAL_MASS_FALLBACK_PRECISION) do
    Float64(sum(BigFloat,
        [large_triangle_mass[t][1, 1] for t in 1:2]))
end
@test large_patch_mass[1][1, 1] == large_patch_assembled_reference
@test large_patch_mass[1][1, 1] ≈
      large_patch_reference rtol=2eps(Float64)
dZ_first = assemble_dZ_dtheta(Mp, 1)
@test Matrix(dZ_first) == -Matrix(Mp[1])
@test_throws ArgumentError assemble_dZ_dtheta(Mp, 0)
@test_throws ArgumentError assemble_dZ_dtheta(
    Matrix{Float64}[], 1)
@test_throws ArgumentError assemble_dZ_dtheta(
    [fill(NaN, N, N)], 1)

# Invalid stored indices are rejected before the @inbounds numerical kernels
# can observe them, and custom matrix dimensions follow AbstractArray rules.
@test_throws ArgumentError LocalMassMatrix(-1, Int[], Int[], Float64[])
@test_throws DimensionMismatch LocalMassMatrix(2, [1], [1, 2], [1.0])
@test_throws ArgumentError LocalMassMatrix(2, [0], [1], [1.0])
@test_throws ArgumentError LocalMassMatrix(2, [1], [3], [1.0])
@test size(Mp[1], 3) == 1
@test_throws BoundsError size(Mp[1], 0)
@test_throws BoundsError size(Mp[1], -1)
@test_throws DimensionMismatch assemble_Z_impedance(
    Mp, zeros(Float64, length(Mp) - 1))
@test_throws ArgumentError assemble_Z_impedance(
    LocalMassMatrix{Float64}[], Float64[])
undersized_mass = LocalMassMatrix(1, [1], [1], [1.0])
@test_throws DimensionMismatch assemble_Z_impedance(
    [undersized_mass, Mp[1]], [1.0, 1.0])
@test_throws DimensionMismatch assemble_full_Z(
    Z_efie, Mp, zeros(Float64, length(Mp) - 1))
@test_throws DimensionMismatch assemble_full_Z(
    Z_efie, [undersized_mass], [1.0])
@test_throws DimensionMismatch assemble_full_Z!(
    zeros(ComplexF64, N + 1, N + 1), Z_efie, Mp, zeros(Float64, length(Mp)))
@test_throws ArgumentError assemble_Z_impedance(
    Mp, fill(Inf, length(Mp)))
@test_throws ArgumentError assemble_full_Z(
    Z_efie, Mp, fill(NaN, length(Mp)))
@test_throws ArgumentError LocalMassMatrix(
    1, [1], [1], [NaN])
@test_throws ArgumentError gradient_impedance(
    Matrix{Float64}[], ComplexF64[], ComplexF64[])
@test_throws DimensionMismatch gradient_impedance(
    Mp, zeros(ComplexF64, N + 1), zeros(ComplexF64, N))
@test_throws DimensionMismatch gradient_impedance(
    Mp, zeros(ComplexF64, N), zeros(ComplexF64, N + 1))
gradient_probe_finite = ones(ComplexF64, N)
@test_throws ArgumentError gradient_impedance(
    Mp, fill(ComplexF64(NaN, 0.0), N), gradient_probe_finite)
@test_throws ArgumentError gradient_impedance(
    Mp, gradient_probe_finite, fill(ComplexF64(Inf, 0.0), N))

# Fused bilinear products can overflow before exact cancellation. Exercise the
# dense, sparse, and compact local storage paths without changing ordinary
# output-only allocation behavior.
gradient_extreme_scale = floatmax(Float64)
gradient_extreme_left = ComplexF64[
    gradient_extreme_scale,
    gradient_extreme_scale,
    gradient_extreme_scale,
    gradient_extreme_scale,
    1.0,
]
gradient_extreme_right = copy(gradient_extreme_left)
gradient_extreme_matrix = zeros(ComplexF64, 5, 5)
gradient_extreme_matrix[1, 1:4] .= ComplexF64[
    gradient_extreme_scale,
    gradient_extreme_scale,
    -gradient_extreme_scale,
    -gradient_extreme_scale,
]
gradient_extreme_matrix[5, 5] = 3.0
gradient_extreme_reference = setprecision(BigFloat, 8192) do
    dot(
        Complex{BigFloat}.(gradient_extreme_left),
        Matrix{Complex{BigFloat}}(gradient_extreme_matrix),
        Complex{BigFloat}.(gradient_extreme_right),
    )
end
@test gradient_extreme_reference == Complex{BigFloat}(3)
gradient_extreme_local = LocalMassMatrix(
    5,
    [1, 1, 1, 1, 5],
    [1, 2, 3, 4, 5],
    ComplexF64[
        gradient_extreme_scale,
        gradient_extreme_scale,
        -gradient_extreme_scale,
        -gradient_extreme_scale,
        3.0,
    ],
)
for gradient_matrix in (
    gradient_extreme_matrix,
    sparse(gradient_extreme_matrix),
    gradient_extreme_local,
)
    @test gradient_impedance(
        [gradient_matrix], gradient_extreme_right,
        gradient_extreme_left) == [6.0]
end
@test gradient_impedance(
    [im .* gradient_extreme_matrix],
    gradient_extreme_right,
    gradient_extreme_left;
    reactive=true,
) == [-6.0]
@test_throws OverflowError gradient_impedance(
    [reshape(ComplexF64[1.0], 1, 1)],
    ComplexF64[gradient_extreme_scale],
    ComplexF64[gradient_extreme_scale],
)
gradient_allocation_matrices = [ComplexF64[2.0 0.0; 0.0 3.0]]
gradient_allocation_vector = ComplexF64[1.0, 2.0]
gradient_impedance(
    gradient_allocation_matrices,
    gradient_allocation_vector,
    gradient_allocation_vector,
)
@test @allocated(gradient_impedance(
    gradient_allocation_matrices,
    gradient_allocation_vector,
    gradient_allocation_vector,
)) <= _float_vector_output_allocation(1) + 128

# Five-argument mul! must not read x when alpha is zero, including when x
# contains non-finite values. Verify both beta branches and both orientations.
mass_contract = LocalMassMatrix(
    2, [1, 2], [2, 1], ComplexF64[2 + 0im, 3 + 0im])
mass_nonfinite = fill(ComplexF64(NaN, NaN), 2)
mass_initial = ComplexF64[5 - 2im, -3 + 4im]
for mass_op in (mass_contract, adjoint(mass_contract))
    mass_result = copy(mass_initial)
    mul!(mass_result, mass_op, mass_nonfinite, 0.0 + 0im, 2.0 + 0im)
    @test mass_result == 2 .* mass_initial

    fill!(mass_result, ComplexF64(NaN, NaN))
    mul!(mass_result, mass_op, mass_nonfinite, 0.0 + 0im, 0.0 + 0im)
    @test mass_result == zeros(ComplexF64, 2)

    mass_overlap_storage = ComplexF64[1 + 0im, 4 + 0im, 9 + 0im]
    mass_overlap_x = view(mass_overlap_storage, 1:2)
    mass_overlap_y = view(mass_overlap_storage, 2:3)
    mass_overlap_x_initial = copy(mass_overlap_x)
    mass_overlap_y_initial = copy(mass_overlap_y)
    mass_overlap_expected =
        1.25 .* (mass_op * mass_overlap_x_initial) .+
        0.5 .* mass_overlap_y_initial
    mul!(mass_overlap_y, mass_op, mass_overlap_x,
         1.25 + 0im, 0.5 + 0im)
    @test mass_overlap_y ≈ mass_overlap_expected
end

# Duplicate local triplets and five-argument output scaling must retain a
# finite exact result when ordinary intermediate additions/products overflow.
mass_extreme_term = 0.6 * floatmax(Float64)
mass_extreme = LocalMassMatrix(
    1,
    [1, 1, 1],
    [1, 1, 1],
    ComplexF64[mass_extreme_term, mass_extreme_term, -mass_extreme_term],
)
mass_extreme_reference = setprecision(BigFloat, 4608) do
    ComplexF64(
        BigFloat(mass_extreme_term) + BigFloat(mass_extreme_term) -
        BigFloat(mass_extreme_term))
end
@test !isfinite(mass_extreme_term + mass_extreme_term)
@test isfinite(mass_extreme_reference)
@test mass_extreme[1, 1] == mass_extreme_reference
@test Matrix(mass_extreme)[1, 1] == mass_extreme_reference
@test sparse(mass_extreme)[1, 1] == mass_extreme_reference

mass_extreme_dense_accumulation = zeros(ComplexF64, 1, 1)
DiffMoM._add_scaled_matrix!(
    mass_extreme_dense_accumulation, 1.0 + 0im, mass_extreme)
@test mass_extreme_dense_accumulation[1, 1] == mass_extreme_reference

mass_scale = 1.0e308 + 0im
mass_scale_input = ComplexF64[10]
mass_scale_operator = LocalMassMatrix(1, [1], [1], ComplexF64[1])
for mass_op in (
    mass_extreme,
    adjoint(mass_extreme),
)
    @test (mass_op * ComplexF64[1])[1] == mass_extreme_reference
end
for mass_op in (
    mass_scale_operator,
    adjoint(mass_scale_operator),
)
    mass_scale_product = mass_op * mass_scale_input
    mass_scale_previous = -mass_scale_product
    @test any(!isfinite, mass_scale .* mass_scale_product)
    mass_scale_reference = setprecision(BigFloat, 4608) do
        ComplexF64[
            Complex{BigFloat}(mass_scale) *
                Complex{BigFloat}(mass_scale_product[i]) +
            Complex{BigFloat}(mass_scale) *
                Complex{BigFloat}(mass_scale_previous[i])
            for i in eachindex(mass_scale_product)
        ]
    end
    mass_scale_result = copy(mass_scale_previous)
    mul!(mass_scale_result, mass_op, mass_scale_input,
         mass_scale, mass_scale)
    @test mass_scale_result == mass_scale_reference

    mass_scale_alias = copy(mass_scale_input)
    mul!(mass_scale_alias, mass_op, mass_scale_alias,
         mass_scale, -mass_scale)
    @test mass_scale_alias == zeros(ComplexF64, 1)
end

mass_scale_cancelling_entries = LocalMassMatrix(
    1, [1, 1], [1, 1], ComplexF64[10, -10])
mass_scaled_matrix = mass_scale * mass_scale_cancelling_entries
@test mass_scaled_matrix[1, 1] == 0.0 + 0.0im

# Range loss in a finite matrix-vector product must be detected before
# separately rounded subnormal terms are accumulated.  Exercise both storage
# orientations because the adjoint uses its column-sorted traversal.
mass_underflow_unit = nextfloat(0.0)
mass_underflow_input = fill(ComplexF64(0.6), 2)
mass_underflow_reference = setprecision(BigFloat, 6656) do
    ComplexF64(2 * BigFloat(mass_underflow_unit) * BigFloat(0.6))
end
@test mass_underflow_reference == ComplexF64(mass_underflow_unit)
mass_underflow_forward = LocalMassMatrix(
    2, [1, 1], [1, 2], fill(ComplexF64(mass_underflow_unit), 2))
mass_underflow_adjoint = LocalMassMatrix(
    2, [1, 2], [1, 1], fill(ComplexF64(mass_underflow_unit), 2))
for mass_op in (mass_underflow_forward, adjoint(mass_underflow_adjoint))
    mass_underflow_result = zeros(ComplexF64, 2)
    mul!(mass_underflow_result, mass_op, mass_underflow_input)
    @test mass_underflow_result ==
          ComplexF64[mass_underflow_reference, 0]
end

# A normal component must not hide a rounded-away subnormal component of a
# complex product.  All LocalMassMatrix scaling entry points use the same
# exact stored-factor semantics on this exceptional path.
mass_complex_underflow = LocalMassMatrix(
    1, [1], [1], ComplexF64[complex(
        mass_underflow_unit, -mass_underflow_unit)])
mass_complex_scale = 0.4 + 0.4im
mass_complex_reference = setprecision(BigFloat, 6656) do
    ComplexF64(
        Complex{BigFloat}(mass_complex_scale) *
        Complex{BigFloat}(mass_complex_underflow.vals[1]))
end
@test mass_complex_reference == ComplexF64(mass_underflow_unit)
@test (mass_complex_scale * mass_complex_underflow)[1, 1] ==
      mass_complex_reference
mass_complex_product = zeros(ComplexF64, 1)
mul!(mass_complex_product, mass_complex_underflow, ComplexF64[1],
     mass_complex_scale, 0.0 + 0im)
@test mass_complex_product == ComplexF64[mass_complex_reference]
mass_complex_scaled_output = copy(mass_complex_underflow.vals)
mul!(mass_complex_scaled_output, mass_complex_underflow,
     fill(ComplexF64(NaN, NaN), 1), 0.0 + 0im, mass_complex_scale)
@test mass_complex_scaled_output == ComplexF64[mass_complex_reference]
mass_complex_accumulation = zeros(ComplexF64, 1, 1)
DiffMoM._add_scaled_matrix!(
    mass_complex_accumulation, mass_complex_scale,
    mass_complex_underflow)
@test mass_complex_accumulation[1, 1] == mass_complex_reference

# Canonicalization must retain a finite residue that sequential duplicate
# addition loses. All public views and products share the canonical value.
mass_duplicate_residue = ldexp(1.0, -53)
mass_cancelling_residue = LocalMassMatrix(
    1, [1, 1, 1], [1, 1, 1],
    ComplexF64[1.0, mass_duplicate_residue, -1.0])
@test length(mass_cancelling_residue.vals) == 1
@test mass_cancelling_residue[1, 1] == mass_duplicate_residue
@test Matrix(mass_cancelling_residue)[1, 1] == mass_duplicate_residue
@test sparse(mass_cancelling_residue)[1, 1] == mass_duplicate_residue
mass_residue_product = zeros(ComplexF64, 1)
mul!(mass_residue_product, mass_cancelling_residue, ComplexF64[1])
@test mass_residue_product[1] == mass_duplicate_residue
mass_residue_accumulation = zeros(ComplexF64, 1, 1)
DiffMoM._add_scaled_matrix!(
    mass_residue_accumulation, 1.0, mass_cancelling_residue)
@test mass_residue_accumulation[1, 1] == mass_duplicate_residue

@test_throws OverflowError LocalMassMatrix(
    1, [1, 1], [1, 1],
    ComplexF64[floatmax(Float64), floatmax(Float64)])

mass_allocation_x = ComplexF64[1 + 2im, 3 - 4im]
mass_allocation_y = zeros(ComplexF64, 2)
mul!(mass_allocation_y, mass_contract, mass_allocation_x)
@test (@allocated mul!(mass_allocation_y, mass_contract, mass_allocation_x)) < 128
mul!(mass_allocation_y, mass_contract, mass_allocation_x,
     1.25 - 0.5im, -0.75 + 0.25im)
@test (@allocated mul!(mass_allocation_y, mass_contract, mass_allocation_x,
                      1.25 - 0.5im, -0.75 + 0.25im)) < 128

# Each M_p should be symmetric (real-valued mass matrix)
for p in 1:min(3, Nt)
    @assert norm(Mp[p] - Mp[p]') < 1e-14 * norm(Mp[p])
end

# Test impedance assembly
theta = fill(100.0 + 50.0im, Nt)  # complex impedance
Z_imp = assemble_Z_impedance(Mp, theta)
@assert size(Z_imp) == (N, N)
impedance_output_bytes = sizeof(eltype(Z_imp)) * length(Z_imp)
@test_throws ArgumentError assemble_Z_impedance(
    Mp, theta; max_output_bytes=impedance_output_bytes - 1)
@test assemble_Z_impedance(
    Mp, theta; max_output_bytes=impedance_output_bytes) == Z_imp

# Resistive/reactive decomposition sanity checks
Z_imp_res = assemble_Z_impedance(Mp, fill(100.0, Nt))
Z_imp_reac = assemble_Z_impedance(Mp, 1im .* fill(100.0, Nt))
@assert maximum(abs.(imag.(Z_imp_res))) < 1e-12
@assert maximum(abs.(real.(Z_imp_reac))) < 1e-12

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 6: Far-Field and Q Matrix
# ─────────────────────────────────────────────────
println("\n── Test 6: Far-field and Q matrix ──")

@test_throws ArgumentError make_sph_grid(0, 16)
@test_throws ArgumentError make_sph_grid(8, 0)
@test_throws ArgumentError make_sph_grid(8, 16; max_points=127)
@test_throws ArgumentError make_sph_grid(
    8, 16; max_raw_bytes=128 * 6sizeof(Float64) - 1)
@test length(make_sph_grid(
    8, 16;
    max_points=128,
    max_raw_bytes=128 * 6sizeof(Float64)).w) == 128
grid = make_sph_grid(8, 16)
NΩ = length(grid.w)
println("  Far-field grid: $NΩ directions")

G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
@assert size(G_mat) == (3 * NΩ, N)
@test_throws ArgumentError radiation_vectors(mesh, rwg, grid, Inf; eta0=eta0)
@test_throws ArgumentError radiation_vectors(mesh, rwg, grid, k; eta0=Inf)
radiation_output_limit = sizeof(ComplexF64) * 3NΩ * N
radiation_work_limit = DiffMoM._radiation_vectors_work_bytes(
    ntriangles(mesh), length(tri_quad_rule(3)[2]), NΩ,
    radiation_output_limit)
radiation_term_limit = DiffMoM._radiation_vectors_term_count(
    N, length(tri_quad_rule(3)[2]), NΩ)
@test_throws ArgumentError radiation_vectors(
    mesh, rwg, grid, k;
    quad_order=3,
    eta0=eta0,
    max_output_bytes=radiation_output_limit - 1)
@test_throws ArgumentError radiation_vectors(
    mesh, rwg, grid, k;
    quad_order=3,
    eta0=eta0,
    max_work_bytes=radiation_work_limit - 1)
@test_throws ArgumentError radiation_vectors(
    mesh, rwg, grid, k;
    quad_order=3,
    eta0=eta0,
    max_terms=radiation_term_limit - 1)
@test radiation_vectors(
    mesh, rwg, grid, k;
    quad_order=3,
    eta0=eta0,
    max_output_bytes=radiation_output_limit,
    max_work_bytes=radiation_work_limit,
    max_terms=radiation_term_limit) == G_mat

# A standalone k*eta0 product may overflow even though the completed
# geometry-weighted radiation vector is finite. The exact path retains the
# scale through the final per-direction sum and is linear in eta0.
radiation_tiny_mesh = make_rect_plate(1.0e-154, 1.0e-154, 1, 1)
radiation_tiny_rwg = build_rwg(radiation_tiny_mesh)
radiation_axis_grid = SphGrid(
    reshape(Float64[0.0, 0.0, 1.0], 3, 1), [0.0], [0.0], [1.0])
radiation_exact_terms = DiffMoM._radiation_vectors_term_count(
    radiation_tiny_rwg.nedges, 1, 1)
radiation_exact_work =
    radiation_exact_terms * DiffMoM._RADIATION_EXACT_PRECISION
radiation_scale_reference = radiation_vectors(
    radiation_tiny_mesh, radiation_tiny_rwg, radiation_axis_grid,
    floatmax(Float64);
    quad_order=1,
    eta0=1.0,
    max_exact_work=radiation_exact_work)
@test_throws ArgumentError radiation_vectors(
    radiation_tiny_mesh, radiation_tiny_rwg, radiation_axis_grid,
    floatmax(Float64);
    quad_order=1,
    eta0=2.0,
    max_exact_work=radiation_exact_work - 1)
radiation_scale_exact = radiation_vectors(
    radiation_tiny_mesh, radiation_tiny_rwg, radiation_axis_grid,
    floatmax(Float64);
    quad_order=1,
    eta0=2.0,
    max_exact_work=radiation_exact_work)
@test all(isfinite, radiation_scale_exact)
@test radiation_scale_exact ≈ 2 .* radiation_scale_reference rtol=1e-15 atol=0

# Large finite phase products require exact reduction even when each primitive
# factor is well inside the ordinary exponent band.  A translated plate with
# nearly cancelling directional products used to select an unrelated phase.
radiation_phase_base = make_rect_plate(1.0, 1.0, 1, 1)
radiation_phase_angle = 0.2
radiation_phase_direction = Vec3(
    cos(radiation_phase_angle), sin(radiation_phase_angle), 0.0)
radiation_phase_x = 1.0e12
radiation_phase_y = -Float64(
    (radiation_phase_direction[1] / radiation_phase_direction[2]) *
    radiation_phase_x)
radiation_phase_xyz = copy(radiation_phase_base.xyz)
radiation_phase_xyz[1, :] .+= radiation_phase_x
radiation_phase_xyz[2, :] .+= radiation_phase_y
radiation_phase_mesh = TriMesh(
    radiation_phase_xyz, copy(radiation_phase_base.tri))
radiation_phase_rwg = build_rwg(radiation_phase_mesh)
radiation_phase_grid = SphGrid(
    reshape(collect(radiation_phase_direction), 3, 1),
    [acos(radiation_phase_direction[3])],
    [atan(radiation_phase_direction[2], radiation_phase_direction[1])],
    [1.0])
radiation_phase_xi, radiation_phase_weights = tri_quad_rule(3)
radiation_phase_areas = [
    triangle_area(radiation_phase_mesh, triangle)
    for triangle in 1:ntriangles(radiation_phase_mesh)
]
radiation_phase_terms = DiffMoM._radiation_vectors_term_count(
    radiation_phase_rwg.nedges, length(radiation_phase_weights), 1)
radiation_phase_work =
    radiation_phase_terms * DiffMoM._RADIATION_EXACT_PRECISION
radiation_phase_result = radiation_vectors(
    radiation_phase_mesh, radiation_phase_rwg,
    radiation_phase_grid, 1.0;
    quad_order=3, eta0=1.0,
    max_exact_work=radiation_phase_work)
radiation_phase_reference = zeros(
    ComplexF64, 3, radiation_phase_rwg.nedges)
DiffMoM._radiation_vector_column_exact!(
    radiation_phase_reference, 1, radiation_phase_rwg,
    [radiation_phase_direction], radiation_phase_xi,
    radiation_phase_areas, radiation_phase_weights, 1.0, 1.0)
@test radiation_phase_result == radiation_phase_reference

# The radiation operator is exactly transverse.  Preserve that null when an
# RWG quadrature contribution is parallel to a stored direction whose rounded
# components do not sum to a unit norm, on both numeric paths.
radiation_parallel_mesh = TriMesh(
    Float64[
        -0.5   0.5  -0.5   0.5;
        -0.5  -0.5   0.5   0.5;
         0.0   0.0   0.0   0.0
    ],
    Int[2 3; 4 1; 1 4],
)
radiation_parallel_rwg = build_rwg(radiation_parallel_mesh)
radiation_parallel_triangle = radiation_parallel_rwg.tplus[1]
radiation_parallel_xi, _ = tri_quad_rule(1)
radiation_parallel_value = eval_rwg(
    radiation_parallel_rwg, 1,
    tri_quad_points(
        radiation_parallel_mesh, radiation_parallel_triangle,
        radiation_parallel_xi)[1],
    radiation_parallel_triangle)
radiation_parallel_direction =
    radiation_parallel_value / norm(radiation_parallel_value)
radiation_parallel_grid = SphGrid(
    reshape(collect(radiation_parallel_direction), 3, 1),
    [π / 2],
    [atan(radiation_parallel_direction[2], radiation_parallel_direction[1])],
    [1.0])
for radiation_parallel_impedance in (ldexp(1.0, 120), ldexp(1.0, 200))
    radiation_parallel_result = radiation_vectors(
        radiation_parallel_mesh, radiation_parallel_rwg,
        radiation_parallel_grid, 1.0;
        quad_order=1, eta0=radiation_parallel_impedance)
    @test all(iszero, radiation_parallel_result)
end

# Translating a source multiplies its radiation vector by the phase of the
# normalized observation direction.  Preserve that normalization in the
# exact-geometry path when a tiny direction-norm rounding error is amplified
# by a large translation.
radiation_translation_xyz = Float64[
     0.0   0.0   0.0   0.0;
    -0.5   0.5  -0.5   0.5;
    -0.5  -0.5   0.5   0.5
]
radiation_translation_triangles = Int[1 1; 2 4; 4 3]
radiation_translation_base = TriMesh(
    radiation_translation_xyz, radiation_translation_triangles)
radiation_translation_shift = 1.0e16
radiation_translation_shifted_xyz = copy(radiation_translation_xyz)
radiation_translation_shifted_xyz[1, :] .+= radiation_translation_shift
radiation_translation_shifted = TriMesh(
    radiation_translation_shifted_xyz, copy(radiation_translation_triangles))
radiation_translation_component = inv(sqrt(2.0))
radiation_translation_direction = Vec3(
    radiation_translation_component, radiation_translation_component, 0.0)
radiation_translation_grid = SphGrid(
    reshape(collect(radiation_translation_direction), 3, 1),
    [π / 2], [π / 4], [1.0])
radiation_translation_impedance = ldexp(1.0, 200)
radiation_translation_base_result = radiation_vectors(
    radiation_translation_base, build_rwg(radiation_translation_base),
    radiation_translation_grid, 1.0;
    quad_order=3, eta0=radiation_translation_impedance)
radiation_translation_shifted_result = radiation_vectors(
    radiation_translation_shifted, build_rwg(radiation_translation_shifted),
    radiation_translation_grid, 1.0;
    quad_order=3, eta0=radiation_translation_impedance)
radiation_translation_index =
    argmax(abs.(radiation_translation_base_result[:, 1]))
radiation_translation_ratio =
    radiation_translation_shifted_result[radiation_translation_index, 1] /
    radiation_translation_base_result[radiation_translation_index, 1]
radiation_translation_phase = setprecision(BigFloat, 4608) do
    direction = SVector{3,BigFloat}(
        BigFloat.(radiation_translation_direction))
    ComplexF64(exp(Complex{BigFloat}(
        0, BigFloat(radiation_translation_shift) * direction[1] /
           sqrt(sum(abs2, direction)))))
end
@test radiation_translation_ratio ≈
      radiation_translation_phase rtol=4eps(Float64) atol=0.0

# Phase factors are single-use values. Keep transient storage bounded rather
# than allocating one NΩ × Nq phase matrix for every RWG basis function.
mesh_radiation_alloc = make_rect_plate(1.0, 1.0, 12, 12)
rwg_radiation_alloc = build_rwg(mesh_radiation_alloc)
grid_radiation_alloc = make_sph_grid(8, 16)
radiation_vectors(
    mesh_radiation_alloc, rwg_radiation_alloc, grid_radiation_alloc, 2π;
    quad_order=3)
GC.gc()
radiation_alloc = _radiation_vectors_allocation(
    mesh_radiation_alloc, rwg_radiation_alloc, grid_radiation_alloc, 2π)
radiation_output_bytes =
    sizeof(ComplexF64) * 3 * length(grid_radiation_alloc.w) * rwg_radiation_alloc.nedges
@test radiation_alloc <= radiation_output_bytes + 512_000

bad_rhat_grid = SphGrid(
    hcat(grid.rhat[:, 1:(end - 1)], zeros(3)),
    copy(grid.theta),
    copy(grid.phi),
    copy(grid.w),
)
@test_throws ArgumentError radiation_vectors(mesh, rwg, bad_rhat_grid, k; eta0=eta0)
bad_weight_grid = SphGrid(
    copy(grid.rhat),
    copy(grid.theta),
    copy(grid.phi),
    vcat(grid.w[1:(end - 1)], -1.0),
)
@test_throws ArgumentError radiation_vectors(mesh, rwg, bad_weight_grid, k; eta0=eta0)

dipole_ff_validation = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0),
    :electric,
    1.0e9,
)
k_ff_validation = 2π * 1.0e9 / 299792458.0
E_ff_unit_dir = incident_farfield(
    dipole_ff_validation, Vec3(0.0, 0.0, 1.0), k_ff_validation)
E_ff_scaled_dir = incident_farfield(
    dipole_ff_validation, Vec3(0.0, 0.0, 1.0e300), k_ff_validation)
@test E_ff_scaled_dir ≈ E_ff_unit_dir
@test_throws ArgumentError incident_farfield(
    dipole_ff_validation, Vec3(0.0, 0.0, 0.0), k_ff_validation)
@test_throws ArgumentError incident_farfield(
    dipole_ff_validation, Vec3(NaN, 0.0, 1.0), k_ff_validation)
@test_throws ArgumentError incident_farfield(
    dipole_ff_validation, Vec3(0.0, 0.0, 1.0), 0.0)
@test_throws ArgumentError incident_farfield(
    dipole_ff_validation, Vec3(0.0, 0.0, 1.0), Inf)
@test_throws ArgumentError incident_farfield(
    dipole_ff_validation, Vec3(0.0, 0.0, 1.0), 2 * k_ff_validation)
dipole_ff_bad_frequency = DipoleExcitation(
    Vec3(0.0, 0.0, 0.0),
    CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0),
    :electric,
    Inf,
)
@test_throws ArgumentError incident_farfield(
    dipole_ff_bad_frequency, Vec3(0.0, 0.0, 1.0), k_ff_validation)

# Compute far-field from PEC solution
E_ff = compute_farfield(G_mat, I_pec, NΩ)
@assert size(E_ff) == (3, NΩ)
@test_throws ArgumentError compute_farfield(G_mat, I_pec, 0)
@test_throws DimensionMismatch compute_farfield(
    G_mat[1:(end - 1), :], I_pec, NΩ)
@test_throws DimensionMismatch compute_farfield(
    G_mat, vcat(I_pec, 1.0 + 0im), NΩ)
farfield_nonfinite_G = copy(G_mat)
farfield_nonfinite_G[1, 1] = ComplexF64(NaN, 0.0)
@test_throws ArgumentError compute_farfield(
    farfield_nonfinite_G, I_pec, NΩ)
farfield_nonfinite_I = copy(I_pec)
farfield_nonfinite_I[1] = ComplexF64(NaN, 0.0)
@test_throws ArgumentError compute_farfield(
    G_mat, farfield_nonfinite_I, NΩ)

# Each product rounds to zero separately, but their exact sum rounds to the
# minimum ComplexF64 subnormal and must survive the dense-product restart.
farfield_underflow_G = zeros(ComplexF64, 3, 2)
farfield_underflow_G[1, :] .= nextfloat(0.0)
farfield_underflow_I = fill(ComplexF64(0.4), 2)
farfield_underflow_reference = setprecision(BigFloat, 4352) do
    ComplexF64(sum(
        Complex{BigFloat}(farfield_underflow_G[1, column]) *
        Complex{BigFloat}(farfield_underflow_I[column])
        for column in axes(farfield_underflow_G, 2)))
end
@test farfield_underflow_reference == ComplexF64(nextfloat(0.0))
@test compute_farfield(
    farfield_underflow_G, farfield_underflow_I, 1)[1, 1] ==
      farfield_underflow_reference

# Far-field should be transverse: r̂ · E∞ ≈ 0
max_radial = let mr = 0.0
    for q in 1:NΩ
        rh = Vec3(grid.rhat[:, q])
        Eq = CVec3(E_ff[:, q])
        radial = abs(dot(rh, Eq)) / max(abs(norm(Eq)), 1e-30)
        mr = max(mr, radial)
    end
    mr
end
println("  Max radial E-field component: $max_radial")
@assert max_radial < 0.1  # should be small

# Build Q matrix
pol_mat = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=30 * π / 180)
Q = build_Q(G_mat, grid, pol_mat; mask=mask)
Q_operator = build_Q_operator(G_mat, grid, pol_mat; mask=mask)
q_work_limit = sizeof(ComplexF64) * (NΩ * N + N * N)
@test_throws ArgumentError build_Q(
    G_mat, grid, pol_mat;
    mask=mask,
    max_work_bytes=q_work_limit - 1)
@test build_Q(
    G_mat, grid, pol_mat;
    mask=mask,
    max_work_bytes=q_work_limit) == Q
@test length(Q_operator.work) == N
@test Q_operator.work_lock isa ReentrantLock
oversized_mask = vcat(mask, true)
nonfinite_G = copy(G_mat)
nonfinite_G[1, 1] = ComplexF64(NaN, 0.0)
nonfinite_pol = copy(pol_mat)
nonfinite_pol[1, 1] = ComplexF64(NaN, 0.0)
@test_throws ArgumentError build_Q(
    nonfinite_G, grid, pol_mat; mask=mask)
@test_throws ArgumentError build_Q_operator(
    G_mat, grid, nonfinite_pol; mask=mask)
@test_throws ArgumentError apply_Q(
    G_mat, grid, pol_mat, I_pec; mask=Int.(mask))
@test_throws DimensionMismatch build_Q(
    G_mat, grid, pol_mat; mask=oversized_mask)
@test_throws DimensionMismatch build_Q_operator(
    G_mat, grid, pol_mat; mask=oversized_mask)
@test_throws DimensionMismatch apply_Q(
    G_mat, grid, pol_mat, I_pec; mask=oversized_mask)
@test_throws DimensionMismatch build_Q(
    G_mat[1:(end - 1), :], grid, pol_mat; mask=mask)
@test_throws DimensionMismatch build_Q(
    G_mat, grid, pol_mat[:, 1:(end - 1)]; mask=mask)
@test_throws DimensionMismatch apply_Q(
    G_mat, grid, pol_mat, vcat(I_pec, 1.0 + 0im); mask=mask)

# Q should be Hermitian PSD
@assert norm(Q - Q') < 1e-12 * norm(Q)
eigvals_Q = eigvals(Hermitian(Q))
@assert all(eigvals_Q .>= -1e-12 * maximum(eigvals_Q))
println("  Q is Hermitian PSD ✓")

# Cross-check objective computed two ways:
#   (1) quadratic form I†QI
#   (2) direct angular integration of projected far field
P_qform = real(dot(I_pec, Q * I_pec))
P_direct = projected_power(E_ff, grid, pol_mat; mask=mask)
E_ff_oversized = hcat(E_ff, zeros(ComplexF64, 3))
pol_oversized = hcat(pol_mat, zeros(ComplexF64, 3))
@test_throws DimensionMismatch radiated_power(E_ff_oversized, grid)
@test_throws DimensionMismatch projected_power(
    E_ff_oversized, grid, pol_oversized; mask=vcat(mask, true))
@test_throws DimensionMismatch projected_power(
    E_ff, grid, pol_oversized; mask=mask)
@test_throws DimensionMismatch projected_power(
    E_ff, grid, pol_mat; mask=vcat(mask, true))
@test_throws ArgumentError projected_power(
    E_ff, grid, pol_mat; mask=Int.(mask))
@test_throws ArgumentError radiated_power(E_ff, grid; eta0=0.0)
E_ff_nonfinite = copy(E_ff)
E_ff_nonfinite[1, 1] = ComplexF64(NaN, 0.0)
pol_nonfinite = copy(pol_mat)
pol_nonfinite[1, 1] = ComplexF64(Inf, 0.0)
@test_throws ArgumentError radiated_power(E_ff_nonfinite, grid)
@test_throws ArgumentError projected_power(
    E_ff_nonfinite, grid, pol_mat; mask=mask)
@test_throws ArgumentError projected_power(
    E_ff, grid, pol_nonfinite; mask=mask)
@test_throws OverflowError radiated_power(
    fill(ComplexF64(floatmax(Float64), 0.0), size(E_ff)), grid)

diagnostic_rhat = reshape(Float64[0.0, 0.0, 1.0], 3, 1)
diagnostic_overflow_grid = SphGrid(
    diagnostic_rhat, [0.0], [0.0], [1.0e-200])
diagnostic_underflow_grid = SphGrid(
    diagnostic_rhat, [0.0], [0.0], [1.0e200])
diagnostic_radiated_overflow_field = reshape(
    ComplexF64[1.0e200, 0.0, 0.0], 3, 1)
diagnostic_radiated_underflow_field = reshape(
    ComplexF64[1.0e-200, 0.0, 0.0], 3, 1)
diagnostic_radiated_references = setprecision(BigFloat, 6656) do
    (
        overflow = Float64(
            BigFloat(1.0e-200) * BigFloat(1.0e200)^2 /
            (2 * BigFloat(376.730313668))),
        underflow = Float64(
            BigFloat(1.0e200) * BigFloat(1.0e-200)^2 /
            (2 * BigFloat(376.730313668))),
    )
end
@test radiated_power(
    diagnostic_radiated_overflow_field,
    diagnostic_overflow_grid,
) == diagnostic_radiated_references.overflow
@test radiated_power(
    diagnostic_radiated_underflow_field,
    diagnostic_underflow_grid,
) == diagnostic_radiated_references.underflow

diagnostic_projected_overflow_field = reshape(
    ComplexF64[1.0e100, 0.0, 0.0], 3, 1)
diagnostic_projected_underflow_field = reshape(
    ComplexF64[1.0e-100, 0.0, 0.0], 3, 1)
diagnostic_projected_overflow_pol = reshape(
    ComplexF64[1.0e100, 0.0, 0.0], 3, 1)
diagnostic_projected_underflow_pol = reshape(
    ComplexF64[1.0e-100, 0.0, 0.0], 3, 1)
diagnostic_projected_references = setprecision(BigFloat, 11264) do
    (
        overflow = Float64(
            BigFloat(1.0e-200) *
            (BigFloat(1.0e100) * BigFloat(1.0e100))^2),
        underflow = Float64(
            BigFloat(1.0e200) *
            (BigFloat(1.0e-100) * BigFloat(1.0e-100))^2),
    )
end
@test projected_power(
    diagnostic_projected_overflow_field,
    diagnostic_overflow_grid,
    diagnostic_projected_overflow_pol,
) == diagnostic_projected_references.overflow
@test projected_power(
    diagnostic_projected_underflow_field,
    diagnostic_underflow_grid,
    diagnostic_projected_underflow_pol,
) == diagnostic_projected_references.underflow

diagnostic_allocation_size = 96
diagnostic_allocation_rhat = zeros(3, diagnostic_allocation_size)
diagnostic_allocation_rhat[3, :] .= 1.0
diagnostic_allocation_grid = SphGrid(
    diagnostic_allocation_rhat,
    zeros(diagnostic_allocation_size),
    zeros(diagnostic_allocation_size),
    fill(1.0e-200, diagnostic_allocation_size),
)
diagnostic_allocation_field = zeros(
    ComplexF64, 3, diagnostic_allocation_size)
diagnostic_allocation_field[1, :] .= 1.0e100
diagnostic_allocation_pol = zeros(
    ComplexF64, 3, diagnostic_allocation_size)
diagnostic_allocation_pol[1, :] .= 1.0e100
diagnostic_allocation_mask = trues(diagnostic_allocation_size)
projected_power(
    diagnostic_allocation_field,
    diagnostic_allocation_grid,
    diagnostic_allocation_pol;
    mask=diagnostic_allocation_mask,
)
@test @allocated(projected_power(
    diagnostic_allocation_field,
    diagnostic_allocation_grid,
    diagnostic_allocation_pol;
    mask=diagnostic_allocation_mask,
)) <= 1_000_000
rel_q_err = abs(P_qform - P_direct) / max(abs(P_qform), 1e-30)
println("  Objective consistency (I†QI vs direct projected power): $rel_q_err")
@assert rel_q_err < 1e-12

QI_operator = _assert_zero_allocation_mul!(Q_operator, I_pec)
@assert norm(QI_operator - Q * I_pec) / max(norm(Q * I_pec), 1e-30) < 1e-12
@assert _matrix_entry_allocation(Q_operator, 1, 1) <= 128
Q_sum_operator = DiffMoM.sum_q_matrix(Q_operator, Q_operator)
QI_sum_operator = _assert_zero_allocation_mul!(Q_sum_operator, I_pec)
@assert QI_sum_operator ≈ 2 .* (Q * I_pec) rtol=1e-12
QI_apply = apply_Q(G_mat, grid, pol_mat, I_pec; mask=mask)
@assert norm(QI_apply - Q * I_pec) / max(norm(Q * I_pec), 1e-30) < 1e-12
@assert _apply_q_allocation(G_mat, grid, pol_mat, I_pec, mask) <=
        _complex_vector_output_allocation(N) + 128

q_initial = randn(MersenneTwister(606), ComplexF64, N)
q_scaled = copy(q_initial)
mul!(q_scaled, Q_operator, I_pec, 2.0, -0.5)
@assert q_scaled ≈ 2.0 .* (Q * I_pec) .- 0.5 .* q_initial
for q_operator in (Q_operator, Q_sum_operator)
    _assert_scaled_mul_contract(q_operator, I_pec, q_initial)

    q_overlap_storage = vcat(I_pec, 2.0 - 3.0im)
    q_overlap_x = view(q_overlap_storage, 1:N)
    q_overlap_y = view(q_overlap_storage, 2:(N + 1))
    q_overlap_x_initial = copy(q_overlap_x)
    q_overlap_y_initial = copy(q_overlap_y)
    q_overlap_alpha = 1.2 - 0.1im
    q_overlap_beta = -0.4 + 0.2im
    q_overlap_expected =
        q_overlap_alpha .* (q_operator * q_overlap_x_initial) .+
        q_overlap_beta .* q_overlap_y_initial
    mul!(q_overlap_y, q_operator, q_overlap_x,
         q_overlap_alpha, q_overlap_beta)
    @test q_overlap_y ≈ q_overlap_expected rtol=1e-12
end

q_extreme_pol = reshape(ComplexF64[1, 0, 0], 3, 1)
q_extreme_term = 0.6 * floatmax(Float64)
q_extreme_G = zeros(ComplexF64, 3, 3)
q_extreme_G[1, :] .= 1.0 + 0im
q_extreme_operator = FarFieldQMatrix(
    q_extreme_G, [1.0], q_extreme_pol, nothing, 3)
q_extreme_input = ComplexF64[
    q_extreme_term, q_extreme_term, -q_extreme_term]
q_extreme_reference_value = setprecision(BigFloat, 4608) do
    ComplexF64(
        BigFloat(q_extreme_term) + BigFloat(q_extreme_term) -
        BigFloat(q_extreme_term))
end
@test !isfinite(q_extreme_term + q_extreme_term)
q_extreme_reference = fill(q_extreme_reference_value, 3)
q_extreme_result = q_extreme_operator * q_extreme_input
@test q_extreme_result == q_extreme_reference
q_extreme_alias = copy(q_extreme_input)
mul!(q_extreme_alias, q_extreme_operator, q_extreme_alias)
@test q_extreme_alias == q_extreme_reference

# Exercise the complementary N <= NΩ cold workspace and entry-reduction path
# with positive quadrature weights. The first row has +t,+t,-t contributions;
# the second row remains a small finite value.
q_outer_G = zeros(ComplexF64, 9, 2)
q_outer_G[1, :] .= ComplexF64[q_extreme_term, 1]
q_outer_G[4, :] .= ComplexF64[q_extreme_term, 1]
q_outer_G[7, :] .= ComplexF64[-q_extreme_term, 1]
q_outer_pol = zeros(ComplexF64, 3, 3)
q_outer_pol[1, :] .= 1.0 + 0im
q_outer_operator = FarFieldQMatrix(
    q_outer_G, ones(3), q_outer_pol, nothing, 2)
q_outer_input = ComplexF64[0, 1]
q_outer_reference = ComplexF64[q_extreme_reference_value, 3]
@test q_outer_operator * q_outer_input == q_outer_reference
@test q_outer_operator[1, 2] == q_extreme_reference_value

q_inner_underflow_G = zeros(ComplexF64, 3, 2)
q_inner_underflow_G[1, :] .= ComplexF64[1.0e150, 1.0e-300]
q_inner_underflow_operator = FarFieldQMatrix(
    q_inner_underflow_G, [1.0e150], q_extreme_pol, nothing, 2)
q_inner_underflow_input = ComplexF64[0, 1.0e-100]
q_inner_underflow_reference = ComplexF64[1.0e-100, 0]
@test q_inner_underflow_operator * q_inner_underflow_input ==
      q_inner_underflow_reference
q_inner_underflow_alias = copy(q_inner_underflow_input)
mul!(q_inner_underflow_alias, q_inner_underflow_operator,
     q_inner_underflow_alias)
@test q_inner_underflow_alias == q_inner_underflow_reference

q_outer_underflow_G = zeros(ComplexF64, 3, 2)
q_outer_underflow_G[1, :] .= ComplexF64[1.0e-300, 1]
q_outer_underflow_operator = FarFieldQMatrix(
    q_outer_underflow_G, [1.0e-300], q_extreme_pol, nothing, 2)
q_outer_underflow_input = ComplexF64[0, 1.0e300]
@test q_outer_underflow_operator * q_outer_underflow_input ==
      ComplexF64[1.0e-300, 1]

q_entry_underflow_G = zeros(ComplexF64, 3, 2)
q_entry_underflow_G[1, :] .= ComplexF64[1.0e-300, 1.0e300]
q_entry_underflow_operator = FarFieldQMatrix(
    q_entry_underflow_G, [1.0e-300], q_extreme_pol, nothing, 2)
@test q_entry_underflow_operator[1, 2] == 1.0e-300 + 0im

# Construction-time classification must not become stale when the public
# backing arrays are mutated later.
q_mutation_grid = SphGrid(
    reshape(Float64[0, 0, 1], 3, 1), [0.0], [0.0], [1.0])
q_mutation_G = zeros(ComplexF64, 3, 2)
q_mutation_G[1, :] .= 1.0
q_mutation_pol = reshape(ComplexF64[1, 0, 0], 3, 1)
q_mutation_operator = build_Q_operator(
    q_mutation_G, q_mutation_grid, q_mutation_pol)
q_mutation_G[1, 1] = ldexp(1.0, 1023)
q_mutation_G[1, 2] = ldexp(1.0, -550)
q_mutation_pol[1, 1] = ldexp(1.0, -550)
q_mutation_result = q_mutation_operator *
                    ComplexF64[0, ldexp(1.0, 128)]
@test q_mutation_result[1] == ComplexF64(ldexp(1.0, -499))

# The dense builder and one-shot apply wrapper must use the same checked
# extreme-factor path as FarFieldQMatrix instead of losing a representable
# outer product after an underflowed intermediate multiplication.
q_wrapper_grid = SphGrid(
    reshape(Float64[0, 0, 1], 3, 1),
    [0.0], [0.0], [1.0e-300])
q_wrapper_G = zeros(ComplexF64, 3, 2)
q_wrapper_G[1, :] .= ComplexF64[1.0e-200, 1.0e300]
q_wrapper_reference = setprecision(BigFloat, 12800) do
    projected = BigFloat[1.0e-200, 1.0e300]
    weight = BigFloat(1.0e-300)
    ComplexF64[
        weight * projected[row] * projected[column]
        for row in eachindex(projected), column in eachindex(projected)
    ]
end
q_wrapper_input = ComplexF64[0, 1]
q_wrapper_dense = build_Q(
    q_wrapper_G, q_wrapper_grid, q_extreme_pol)
@test all(isfinite, q_wrapper_dense)
@test q_wrapper_dense[1, 2] == q_wrapper_reference[1, 2]
@test q_wrapper_dense ≈ q_wrapper_reference rtol=5e-16 atol=0.0
@test apply_Q(
    q_wrapper_G, q_wrapper_grid, q_extreme_pol,
    q_wrapper_input) == q_wrapper_reference[:, 2]
@test_throws ArgumentError apply_Q(
    q_wrapper_G, q_wrapper_grid, q_extreme_pol,
    ComplexF64[NaN, 0])

q_scale_operator = FarFieldQMatrix(
    reshape(ComplexF64[1, 0, 0], 3, 1),
    [1.0], q_extreme_pol, nothing, 1)
q_scale_input = ComplexF64[10]
q_scale_product = q_scale_operator * q_scale_input
q_scale_previous = -q_scale_product
q_scale_factor = 1.0e308 + 0im
@test any(!isfinite, q_scale_factor .* q_scale_product)
q_scale_reference = setprecision(BigFloat, 4608) do
    ComplexF64[
        Complex{BigFloat}(q_scale_factor) *
            Complex{BigFloat}(q_scale_product[i]) +
        Complex{BigFloat}(q_scale_factor) *
            Complex{BigFloat}(q_scale_previous[i])
        for i in eachindex(q_scale_product)
    ]
end
q_scale_result = copy(q_scale_previous)
mul!(q_scale_result, q_scale_operator, q_scale_input,
     q_scale_factor, q_scale_factor)
@test q_scale_result == q_scale_reference
q_scale_alias = copy(q_scale_input)
mul!(q_scale_alias, q_scale_operator, q_scale_alias,
     q_scale_factor, -q_scale_factor)
@test q_scale_alias == zeros(ComplexF64, 1)
@test _assert_zero_allocation_mul!(q_scale_operator, q_scale_input) ==
      q_scale_product

q_sum_scale_operator = DiffMoM.sum_q_matrix(
    Matrix{ComplexF64}(I, 1, 1), Matrix{ComplexF64}(I, 1, 1))
q_sum_scale_input = ComplexF64[10]
q_sum_scale_previous = ComplexF64[-20]
q_sum_scale_result = copy(q_sum_scale_previous)
mul!(q_sum_scale_result, q_sum_scale_operator, q_sum_scale_input,
     q_scale_factor, q_scale_factor)
@test q_sum_scale_result == zeros(ComplexF64, 1)

q_sum_scale_alias = copy(q_sum_scale_input)
mul!(q_sum_scale_alias, q_sum_scale_operator, q_sum_scale_alias,
     floatmax(Float64) / 2, -floatmax(Float64))
@test q_sum_scale_alias == zeros(ComplexF64, 1)

# Finite scaled terms can still lose most of a representable residual when
# they nearly cancel. All matrix-free five-argument output combiners use the
# exact fallback for this case.
finite_scale_value = ComplexF64(1.0e30)
finite_scale_previous = ComplexF64(-prevfloat(1.0e30))
finite_scale_reference = setprecision(BigFloat, 4352) do
    ComplexF64(
        Complex{BigFloat}(finite_scale_value)^2 +
        Complex{BigFloat}(finite_scale_value) *
        Complex{BigFloat}(finite_scale_previous))
end
@test finite_scale_value^2 +
      finite_scale_value * finite_scale_previous !=
      finite_scale_reference
for scaled_output in (
        DiffMoM._aca_scaled_output,
        DiffMoM._mlfma_scaled_output,
        DiffMoM._farfield_q_scaled_output,
        DiffMoM._sum_q_scaled_output,
        DiffMoM._composite_scaled_output,
    )
    @test scaled_output(
        finite_scale_value,
        finite_scale_previous,
        finite_scale_value,
        finite_scale_value,
        false,
        1,
    ) == finite_scale_reference
end

finite_sum_q_operator = DiffMoM.sum_q_matrix(
    reshape(ComplexF64[1.0e30], 1, 1),
    reshape(ComplexF64[-prevfloat(1.0e30)], 1, 1),
)
finite_sum_q_reference = setprecision(BigFloat, 4352) do
    ComplexF64(BigFloat(1.0e30) - BigFloat(prevfloat(1.0e30)))
end
@test finite_sum_q_operator[1, 1] == finite_sum_q_reference

# Each summand is a Hermitian PSD outer product. Applying the scale to each
# summand separately overflows their cancelling first components, although the
# exact scaled sum is finite.
q_sum_vector_a = ComplexF64[1.0e154, 1]
q_sum_vector_b = ComplexF64[1.0e154, -1]
q_sum_psd_a = q_sum_vector_a * q_sum_vector_a'
q_sum_psd_b = q_sum_vector_b * q_sum_vector_b'
q_sum_psd_operator = DiffMoM.sum_q_matrix(q_sum_psd_a, q_sum_psd_b)
q_sum_psd_input = ComplexF64[0, 1.0e154]
q_sum_psd_reference = ComplexF64[0, 4.0e154]
q_sum_psd_result = zeros(ComplexF64, 2)
mul!(q_sum_psd_result, q_sum_psd_operator, q_sum_psd_input, 2.0, 0.0)
@test q_sum_psd_result == q_sum_psd_reference
@test q_sum_psd_operator * q_sum_psd_input ==
      ComplexF64[0, 2.0e154]
@test_throws OverflowError q_sum_psd_operator[1, 1]

_assert_shared_workspace_concurrency(
    fill(Q_operator, 4),
    [I_pec, (0.2 - 0.3im) .* I_pec, reverse(I_pec), conj.(I_pec)],
)
_assert_shared_workspace_concurrency(
    fill(q_sum_psd_operator, 4),
    [
        q_sum_psd_input,
        -q_sum_psd_input,
        0.5 .* q_sum_psd_input,
        1im .* q_sum_psd_input,
    ],
)

# RCS helper checks
sigma = bistatic_rcs(E_ff; E0=1.0)
@assert length(sigma) == NΩ
@assert all(sigma .>= 0.0)
@test_throws DimensionMismatch bistatic_rcs(E_ff[1:2, :])
@test_throws ArgumentError bistatic_rcs(E_ff; E0=0.0)
@test_throws ArgumentError bistatic_rcs(E_ff; E0=Inf)
@test_throws ArgumentError bistatic_rcs(E_ff_nonfinite)
@test_throws OverflowError bistatic_rcs(E_ff; E0=nextfloat(0.0))
@test bistatic_rcs(E_ff; E0=floatmax(Float64)) == zeros(NΩ)

diagnostic_rcs_reference = setprecision(BigFloat, 256) do
    Float64(4 * BigFloat(π))
end
@test only(bistatic_rcs(
    diagnostic_radiated_overflow_field; E0=1.0e200)) ==
      diagnostic_rcs_reference
@test only(bistatic_rcs(
    diagnostic_radiated_underflow_field; E0=1.0e-200)) ==
      diagnostic_rcs_reference

bs = backscatter_rcs(E_ff, grid, Vec3(0.0, 0.0, -1.0); E0=1.0)
@assert 1 <= bs.index <= NΩ
@assert bs.sigma >= 0.0
@test_throws ArgumentError backscatter_rcs(
    E_ff, grid, Vec3(0.0, 0.0, 0.0); E0=1.0)
@test_throws ArgumentError backscatter_rcs(
    E_ff_nonfinite, grid, Vec3(0.0, 0.0, -1.0); E0=1.0)
@test backscatter_rcs(
    diagnostic_radiated_overflow_field,
    diagnostic_overflow_grid,
    Vec3(0.0, 0.0, -1.0);
    E0=1.0e200,
).sigma == diagnostic_rcs_reference
@test_throws DimensionMismatch input_power(
    ComplexF64[1.0], ComplexF64[1.0, 2.0])
@test_throws ArgumentError input_power(ComplexF64[], ComplexF64[])
@test_throws ArgumentError input_power(
    ComplexF64[NaN], ComplexF64[1.0])
diagnostic_input_scale = floatmax(Float64)
diagnostic_input_I = ComplexF64[
    diagnostic_input_scale, diagnostic_input_scale, 1.0]
diagnostic_input_v = ComplexF64[
    diagnostic_input_scale, -diagnostic_input_scale, -4.0]
@test input_power(diagnostic_input_I, diagnostic_input_v) == 2.0
@test_throws DomainError energy_ratio(
    ComplexF64[0.0], ComplexF64[1.0],
    zeros(ComplexF64, size(E_ff)), grid)
@test_throws ArgumentError condition_diagnostics(
    zeros(ComplexF64, 0, 0))
@test_throws ArgumentError condition_diagnostics(
    ComplexF64[NaN 0.0; 0.0 1.0])
zero_condition = condition_diagnostics(zeros(ComplexF64, 2, 2))
@test zero_condition.cond == Inf
@test zero_condition.sv_max == 0.0
@test zero_condition.sv_min == 0.0
@assert _radiated_power_allocation(E_ff, grid) <= 128
@assert _projected_power_allocation(E_ff, grid, pol_mat, mask) <= 128
@assert _bistatic_rcs_allocation(E_ff) <=
        _float_vector_output_allocation(NΩ) + 128
@assert _backscatter_rcs_allocation(
    E_ff, grid, Vec3(0.0, 0.0, -1.0)) <= 128
radiated_power(
    diagnostic_allocation_field, diagnostic_allocation_grid)
@test @allocated(radiated_power(
    diagnostic_allocation_field,
    diagnostic_allocation_grid,
)) <= 250_000
input_power(diagnostic_input_I, diagnostic_input_v)
@test @allocated(input_power(
    diagnostic_input_I, diagnostic_input_v)) <= 100_000

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 6b: Scattered Near-Field Evaluation
# ─────────────────────────────────────────────────
println("\n── Test 6b: Scattered near-field evaluation ──")

obs_points = [
    Vec3(0.00, 0.00, 0.15),
    Vec3(0.02, -0.03, 0.18),
]
E_nf = compute_nearfield(mesh, rwg, I_pec, obs_points, k; quad_order=3, eta0=eta0)
@assert size(E_nf) == (3, length(obs_points))
@assert all(isfinite, real.(E_nf))
@assert all(isfinite, imag.(E_nf))
nearfield_work_bytes = DiffMoM._nearfield_work_bytes(
    ntriangles(mesh), rwg.nedges, length(obs_points), 3)
@test compute_nearfield(
    mesh, rwg, I_pec, obs_points, k;
    quad_order=3, eta0=eta0,
    max_work_bytes=nearfield_work_bytes,
    max_interaction_terms=3ntriangles(mesh) * length(obs_points)) == E_nf
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_pec, obs_points, k;
    quad_order=3, eta0=eta0,
    max_work_bytes=nearfield_work_bytes - 1)
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_pec, obs_points, k;
    quad_order=3, eta0=eta0,
    max_interaction_terms=3ntriangles(mesh) * length(obs_points) - 1)
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_pec, obs_points, k;
    quad_order=2, eta0=eta0)
@test size(compute_nearfield(
    mesh, rwg, I_pec, Vec3[], k; quad_order=3, eta0=eta0)) == (3, 0)
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_pec, obs_points, Inf; quad_order=3, eta0=eta0)
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_pec, obs_points, k; quad_order=3, eta0=Inf)
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_pec, obs_points, k; quad_order=3, eta0=-eta0)
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_pec, obs_points, k;
    quad_order=3, eta0=eta0, check_surface=false, surface_tol=Inf)
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_pec, Vec3(NaN, 0.0, 0.15), k;
    quad_order=3, eta0=eta0, check_surface=false)
I_nf_nonfinite = copy(I_pec)
I_nf_nonfinite[1] = Inf + 0im
@test_throws ArgumentError compute_nearfield(
    mesh, rwg, I_nf_nonfinite, obs_points, k;
    quad_order=3, eta0=eta0, check_surface=false)

# eta0 and the current enter both potentials only through their product.  An
# exact power-of-two transfer must preserve that product when an otherwise
# harmless standalone prefactor would overflow.
nearfield_scale_mesh = make_rect_plate(1.0, 1.0, 1, 1)
nearfield_scale_rwg = build_rwg(nearfield_scale_mesh)
nearfield_scale_point = Vec3(0.0, 0.0, 1.0)
nearfield_scale_current = 1.0e-308
nearfield_scale_max = floatmax(Float64)
nearfield_scale_reference = compute_nearfield(
    nearfield_scale_mesh, nearfield_scale_rwg,
    fill(2nearfield_scale_current, nearfield_scale_rwg.nedges),
    nearfield_scale_point, 2.0;
    eta0=nearfield_scale_max / 2, check_surface=false)
nearfield_scale_result = compute_nearfield(
    nearfield_scale_mesh, nearfield_scale_rwg,
    fill(nearfield_scale_current, nearfield_scale_rwg.nedges),
    nearfield_scale_point, 2.0;
    eta0=nearfield_scale_max, check_surface=false)
@test nearfield_scale_result == nearfield_scale_reference

# Uniformly scaling every coordinate by L while changing k to k/L leaves the
# vector-potential contribution unchanged and multiplies the scalar-potential
# prefactor by the compensating 1/L.  A power-of-two L makes that identity
# exact in the stored meshes.  The historical near branch formed the two
# O(1/R²) pieces of ∇G_smooth separately; both overflowed on this valid mesh
# even though the final field is finite.
nearfield_tiny_length = ldexp(1.0, -530)
nearfield_unit_mesh = make_rect_plate(1.0, 1.0, 1, 1)
nearfield_unit_rwg = build_rwg(nearfield_unit_mesh)
nearfield_unit_point = Vec3(0.0, 0.0, 1 / 64)
nearfield_tiny_mesh = make_rect_plate(
    nearfield_tiny_length, nearfield_tiny_length, 1, 1)
nearfield_tiny_rwg = build_rwg(nearfield_tiny_mesh)
nearfield_tiny_reference = compute_nearfield(
    nearfield_unit_mesh, nearfield_unit_rwg, ComplexF64[1],
    nearfield_unit_point, nearfield_tiny_length;
    quad_order=3, eta0=1.0, check_surface=false, surface_tol=0.0)
nearfield_tiny_result = compute_nearfield(
    nearfield_tiny_mesh, nearfield_tiny_rwg, ComplexF64[1],
    nearfield_tiny_length * nearfield_unit_point, 1.0;
    quad_order=3, eta0=1.0, check_surface=false, surface_tol=0.0)
@test nearfield_tiny_result == nearfield_tiny_reference

# The same exact scaling identity must also hold in the standard-quadrature
# branch.  There the unweighted O(1/R²) Green gradient is outside Float64 even
# though applying the triangle weight first makes every contribution finite.
nearfield_far_scale_point = Vec3(0.0, 0.0, 1.0)
nearfield_far_scale_reference = compute_nearfield(
    nearfield_unit_mesh, nearfield_unit_rwg, ComplexF64[1],
    nearfield_far_scale_point, nearfield_tiny_length;
    quad_order=3, eta0=1.0, check_surface=false, surface_tol=0.0)
nearfield_far_scale_result = compute_nearfield(
    nearfield_tiny_mesh, nearfield_tiny_rwg, ComplexF64[1],
    nearfield_tiny_length * nearfield_far_scale_point, 1.0;
    quad_order=3, eta0=1.0, check_surface=false, surface_tol=0.0)
@test all(isapprox.(
    nearfield_far_scale_result, nearfield_far_scale_reference;
    rtol=2eps(Float64), atol=0.0))

# At the opposite end of Float64, a valid triangle can have finite area while
# `2area` overflows.  Its characteristic length and weighted kernels remain
# finite and must preserve the same scaling identity in both integration
# branches.
nearfield_large_length = ldexp(1.0, 512)
nearfield_large_mesh = make_rect_plate(
    nearfield_large_length, nearfield_large_length, 1, 1)
nearfield_large_rwg = build_rwg(nearfield_large_mesh)
for nearfield_large_point in (nearfield_unit_point, nearfield_far_scale_point)
    nearfield_large_reference = compute_nearfield(
        nearfield_unit_mesh, nearfield_unit_rwg, ComplexF64[1],
        nearfield_large_point, 1.0;
        quad_order=3, eta0=1.0, check_surface=false, surface_tol=0.0)
    nearfield_large_result = compute_nearfield(
        nearfield_large_mesh, nearfield_large_rwg, ComplexF64[1],
        nearfield_large_length * nearfield_large_point,
        inv(nearfield_large_length);
        quad_order=3, eta0=1.0, check_surface=false, surface_tol=0.0)
    @test isapprox(
        nearfield_large_result, nearfield_large_reference;
        rtol=1e-12, atol=1e-14)
end

obs_mat = hcat(obs_points...)
E_nf_mat = compute_nearfield(mesh, rwg, I_pec, obs_mat, k; quad_order=3, eta0=eta0)
@assert norm(E_nf - E_nf_mat) < 1e-12 * max(norm(E_nf), 1.0)

E_nf_pt = compute_nearfield(mesh, rwg, I_pec, obs_points[1], k; quad_order=3, eta0=eta0)
@assert norm(E_nf_pt - CVec3(E_nf[:, 1])) < 1e-12 * max(norm(E_nf_pt), 1.0)

I_nf_a = ComplexF64.(randn(N) .+ 1im .* randn(N))
I_nf_b = ComplexF64.(randn(N) .+ 1im .* randn(N))
E_nf_a = compute_nearfield(mesh, rwg, I_nf_a, obs_points, k; quad_order=3, eta0=eta0)
E_nf_b = compute_nearfield(mesh, rwg, I_nf_b, obs_points, k; quad_order=3, eta0=eta0)
E_nf_ab = compute_nearfield(mesh, rwg, I_nf_a .+ I_nf_b, obs_points, k; quad_order=3, eta0=eta0)
rel_nf_lin = norm(E_nf_ab - (E_nf_a + E_nf_b)) / max(norm(E_nf_ab), 1e-30)
println("  Near-field linearity rel. error: $rel_nf_lin")
@assert rel_nf_lin < 1e-12

surface_err = try
    compute_nearfield(mesh, rwg, I_pec, triangle_center(mesh, 1), k; quad_order=3, eta0=eta0)
    false
catch
    true
end
@assert surface_err

# A rounded point-to-surface distance can equal the requested tolerance on
# either side. The exact stored geometry must accept the just-outside point
# and continue to reject the just-inside point.
nearfield_surface_mesh = TriMesh(
    Float64[0 2 0 2; 0 0 2 2; 0 0 0 0],
    Int[1 2; 2 4; 3 3],
)
nearfield_surface_rwg = build_rwg(nearfield_surface_mesh)
nearfield_surface_x = prevfloat(1.0)
nearfield_surface_y = ldexp(1.0, -26)
nearfield_surface_outside =
    Vec3(-nearfield_surface_x, -nearfield_surface_y, 0.0)
nearfield_surface_inside = Vec3(
    -nearfield_surface_x, -prevfloat(nearfield_surface_y), 0.0)
@test DiffMoM._surface_distance(
    nearfield_surface_mesh, nearfield_surface_outside) == 1.0
@test DiffMoM._surface_distance(
    nearfield_surface_mesh, nearfield_surface_inside) == 1.0
@test compute_nearfield(
    nearfield_surface_mesh,
    nearfield_surface_rwg,
    zeros(ComplexF64, nearfield_surface_rwg.nedges),
    nearfield_surface_outside,
    1.0;
    surface_tol=1.0,
) == zero(CVec3)
@test_throws ErrorException compute_nearfield(
    nearfield_surface_mesh,
    nearfield_surface_rwg,
    zeros(ComplexF64, nearfield_surface_rwg.nedges),
    nearfield_surface_inside,
    1.0;
    surface_tol=1.0,
)

grid_nf = make_sph_grid(10, 20)
q_nf = 37
G_nf_far = radiation_vectors(mesh, rwg, grid_nf, k; quad_order=7, eta0=eta0)
E_ff_nf = compute_farfield(G_nf_far, I_pec, length(grid_nf.w))
rhat_nf = Vec3(grid_nf.rhat[:, q_nf])
R_nf = 40.0 * lambda0
obs_far = R_nf * rhat_nf
E_nf_far = compute_nearfield(mesh, rwg, I_pec, obs_far, k; quad_order=7, eta0=eta0)
E_nf_ref = CVec3(E_ff_nf[:, q_nf]) * exp(-1im * k * R_nf) / R_nf
rel_nf_far = norm(E_nf_far - E_nf_ref) / max(norm(E_nf_ref), 1e-30)
println("  Near-field far-zone rel. error: $rel_nf_far")
@assert rel_nf_far < 0.08

# Near-singular accuracy regression (finding #10): evaluate the field at a point
# very close to the surface and compare the singularity-subtracted near-field
# branch against a brute-force, sub-triangle-refined reference that resolves the
# 1/R (vector) and 1/R² (scalar-gradient) singularities directly.  The corrected
# branch must agree well; the historical "J_avg + full ∇G" treatment diverged.
let
    Tc_ref = [Int[] for _ in 1:ntriangles(mesh)]
    for n in 1:rwg.nedges
        push!(Tc_ref[rwg.tplus[n]], n)
        push!(Tc_ref[rwg.tminus[n]], n)
    end
    pv = -1im * k * eta0
    ps = -1im * eta0 / k
    xi_ref, wq_ref = DiffMoM.tri_quad_rule(7)
    function brute_nf(robs; nsub=80)
        E = zeros(ComplexF64, 3)
        for t in 1:ntriangles(mesh)
            v1 = DiffMoM._mesh_vertex(mesh, mesh.tri[1, t])
            v2 = DiffMoM._mesh_vertex(mesh, mesh.tri[2, t])
            v3 = DiffMoM._mesh_vertex(mesh, mesh.tri[3, t])
            divt = sum(ComplexF64(I_pec[n]) * div_rwg(rwg, n, t) for n in Tc_ref[t]; init=0.0 + 0im)
            for ii in 0:nsub-1, jj in 0:nsub-1, which in 0:1
                a0 = ii / nsub; b0 = jj / nsub; ds = 1 / nsub
                if which == 0
                    ca = (a0, b0); cb = (a0 + ds, b0); cc = (a0, b0 + ds)
                else
                    ca = (a0 + ds, b0); cb = (a0 + ds, b0 + ds); cc = (a0, b0 + ds)
                end
                (ca[1] + ca[2] > 1 + 1e-12 || cb[1] + cb[2] > 1 + 1e-12 || cc[1] + cc[2] > 1 + 1e-12) && continue
                mapref(ab) = v1 * (1 - ab[1] - ab[2]) + v2 * ab[1] + v3 * ab[2]
                p1 = mapref(ca); p2 = mapref(cb); p3 = mapref(cc)
                subA = 0.5 * norm(cross(p2 - p1, p3 - p1))
                for q in eachindex(wq_ref)
                    xq = xi_ref[q]
                    rq = p1 * (1 - xq[1] - xq[2]) + p2 * xq[1] + p3 * xq[2]
                    w = wq_ref[q] * (2 * subA)
                    Jq = sum(ComplexF64(I_pec[n]) * eval_rwg(rwg, n, rq, t) for n in Tc_ref[t]; init=CVec3(0, 0, 0))
                    E += pv * Jq * (w * DiffMoM.greens(robs, rq, k))
                    if abs(divt) > 0
                        E += ps * divt * (w * DiffMoM.grad_greens(robs, rq, k))
                    end
                end
            end
        end
        return CVec3(E)
    end
    max_rel_nearsing = 0.0
    for hh in (0.01, 0.005, 0.002)
        robs_ns = Vec3(0.013, -0.007, hh)
        E_ns = compute_nearfield(mesh, rwg, I_pec, robs_ns, k; quad_order=3, eta0=eta0)
        @assert all(isfinite, real(E_ns)) && all(isfinite, imag(E_ns))
        E_ns_ref = brute_nf(robs_ns)
        rel = norm(E_ns - E_ns_ref) / max(norm(E_ns_ref), 1e-30)
        max_rel_nearsing = max(max_rel_nearsing, rel)
    end
    println("  Near-singular branch max rel. error vs refined reference: $max_rel_nearsing")
    @assert max_rel_nearsing < 0.1
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 6c: Total Electric Field Evaluation
# ─────────────────────────────────────────────────
println("\n── Test 6c: Total electric-field evaluation ──")

pw_total = make_plane_wave(k_vec, E0, pol)
I_zero = zeros(ComplexF64, N)

function stack_fields(field_func, points)
    E = zeros(ComplexF64, 3, length(points))
    for i in eachindex(points)
        E[:, i] .= field_func(points[i])
    end
    return E
end

E_tf = compute_total_field(mesh, rwg, I_pec, pw_total, obs_points, k; quad_order=3, eta0=eta0)
@assert size(E_tf) == (3, length(obs_points))
@assert all(isfinite, real.(E_tf))
@assert all(isfinite, imag.(E_tf))

E_tf_mat = compute_total_field(mesh, rwg, I_pec, pw_total, obs_mat, k; quad_order=3, eta0=eta0)
@assert norm(E_tf - E_tf_mat) < 1e-12 * max(norm(E_tf), 1.0)

E_tf_pt = compute_total_field(mesh, rwg, I_pec, pw_total, obs_points[1], k; quad_order=3, eta0=eta0)
@assert E_tf_pt isa CVec3
@assert norm(E_tf_pt - CVec3(E_tf[:, 1])) < 1e-12 * max(norm(E_tf_pt), 1.0)

E_pw_ref = stack_fields(obs_points) do r
    plane_wave_field(r, k_vec, E0, pol)
end
E_tf_zero_pw = compute_total_field(mesh, rwg, I_zero, pw_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_pw - E_pw_ref) < 1e-12 * max(norm(E_pw_ref), 1.0)

dip_total = make_dipole(Vec3(0.03, -0.02, 0.12),
                        CVec3(0.0 + 0im, 0.0 + 0im, 1.5e-9 + 0.4e-9im),
                        Vec3(0.0, 0.0, 1.0),
                        :electric,
                        freq)
E_dip_ref = stack_fields(obs_points) do r
    DiffMoM.dipole_incident_field(r, dip_total)
end
E_tf_zero_dip = compute_total_field(mesh, rwg, I_zero, dip_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_dip - E_dip_ref) < 1e-12 * max(norm(E_dip_ref), 1.0)

loop_total = make_loop(Vec3(-0.04, 0.01, 0.14),
                       Vec3(0.0, 1.0, 0.0),
                       0.012,
                       1.0 + 0.35im,
                       freq)
E_loop_ref = stack_fields(obs_points) do r
    DiffMoM.loop_incident_field(r, loop_total)
end
E_tf_zero_loop = compute_total_field(mesh, rwg, I_zero, loop_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_loop - E_loop_ref) < 1e-12 * max(norm(E_loop_ref), 1.0)

theta_pat = collect(range(0.0, stop=π, length=9))
phi_pat = collect(range(0.0, stop=2π - 2π / 12, length=12))
pat_total = make_analytic_dipole_pattern_feed(dip_total, theta_pat, phi_pat;
                                              phase_center=dip_total.position,
                                              convention=:exp_plus_iwt)
E_pat_ref = stack_fields(obs_points) do r
    pattern_feed_field(r, pat_total)
end
E_tf_zero_pat = compute_total_field(mesh, rwg, I_zero, pat_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_pat - E_pat_ref) < 1e-12 * max(norm(E_pat_ref), 1.0)

imp_total = ImportedExcitation(
    r -> CVec3(0.3 + 0.1im * r[1], -0.2 + 0.05im * r[2], 0.1 - 0.08im * r[3]);
    kind=:electric_field,
    min_quad_order=3,
)
E_imp_ref = stack_fields(obs_points) do r
    imp_total.source_func(r)
end
E_tf_zero_imp = compute_total_field(mesh, rwg, I_zero, imp_total, obs_points, k; quad_order=3, eta0=eta0)
@assert norm(E_tf_zero_imp - E_imp_ref) < 1e-12 * max(norm(E_imp_ref), 1.0)

E_inc_from_total = E_tf - E_nf
rel_tf_inc = norm(E_inc_from_total - E_pw_ref) / max(norm(E_pw_ref), 1e-30)
println("  Total minus scattered rel. error: $rel_tf_inc")
@assert rel_tf_inc < 1e-12

E_tf_far = compute_total_field(mesh, rwg, I_pec, pw_total, obs_far, k; quad_order=7, eta0=eta0)
E_tf_ref = plane_wave_field(obs_far, k_vec, E0, pol) + E_nf_ref
rel_tf_far = norm(E_tf_far - E_tf_ref) / max(norm(E_tf_ref), 1e-30)
println("  Total-field far-zone rel. error: $rel_tf_far")
@assert rel_tf_far < 0.08

v_dip_total = assemble_excitation(mesh, rwg, dip_total; quad_order=3)
I_dip_total = Z_efie \ v_dip_total
w_pw = 0.7 - 0.15im
w_dip = -0.25 + 0.5im
multi_total = make_multi_excitation([pw_total, dip_total], [w_pw, w_dip])
I_combo = w_pw .* I_pec .+ w_dip .* I_dip_total
E_tf_multi = compute_total_field(mesh, rwg, I_combo, multi_total, obs_points, k; quad_order=3, eta0=eta0)
E_tf_multi_ref =
    w_pw .* compute_total_field(mesh, rwg, I_pec, pw_total, obs_points, k; quad_order=3, eta0=eta0) .+
    w_dip .* compute_total_field(mesh, rwg, I_dip_total, dip_total, obs_points, k; quad_order=3, eta0=eta0)
rel_tf_multi = norm(E_tf_multi - E_tf_multi_ref) / max(norm(E_tf_multi_ref), 1e-30)
println("  MultiExcitation total-field rel. error: $rel_tf_multi")
@assert rel_tf_multi < 1e-12

port_total = PortExcitation([1, 2], 1.0 + 0im, 50.0 + 0im)
port_err = try
    compute_total_field(mesh, rwg, I_zero, port_total, obs_points, k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("PortExcitation", sprint(showerror, err))
end
@assert port_err

gap_total = make_delta_gap(1, 1.0 + 0im, 0.01)
gap_err = try
    compute_total_field(mesh, rwg, I_zero, gap_total, obs_points, k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("DeltaGapExcitation", sprint(showerror, err))
end
@assert gap_err

imp_js_total = ImportedExcitation(r -> CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im);
                                  kind=:surface_current_density,
                                  eta_equiv=eta0 + 0im,
                                  min_quad_order=3)
imp_js_err = try
    compute_total_field(mesh, rwg, I_zero, imp_js_total, obs_points, k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("surface_current_density", sprint(showerror, err))
end
@assert imp_js_err

multi_bad = make_multi_excitation([pw_total, gap_total], [1.0 + 0im, 0.25 + 0im])
multi_bad_err = try
    compute_total_field(mesh, rwg, I_zero, multi_bad, obs_points, k; quad_order=3, eta0=eta0)
    false
catch err
    msg = sprint(showerror, err)
    occursin("child 2", msg) && occursin("DeltaGapExcitation", msg)
end
@assert multi_bad_err

pw_k_err = try
    compute_total_field(mesh, rwg, I_zero, pw_total, obs_points, 1.01 * k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("PlaneWaveExcitation", sprint(showerror, err))
end
@assert pw_k_err

dip_k_err = try
    compute_total_field(mesh, rwg, I_zero, dip_total, obs_points, 0.99 * k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("DipoleExcitation", sprint(showerror, err))
end
@assert dip_k_err

pat_k_err = try
    compute_total_field(mesh, rwg, I_zero, pat_total, obs_points, 1.02 * k; quad_order=3, eta0=eta0)
    false
catch err
    occursin("PatternFeedExcitation", sprint(showerror, err))
end
@assert pat_k_err

# Wavenumber consistency is relative at every scale.  Fixed absolute floors
# used to accept order-one relative mismatches and dominant attenuation for
# electrically tiny models.
@test_throws ErrorException DiffMoM._check_incident_wavenumber_match(
    2.0e-100, 1.0e-100, "tiny-wavenumber probe")
@test_throws ErrorException DiffMoM._check_incident_wavenumber_match(
    1.0e-100, ComplexF64(1.0e-100, 1.0e-20),
    "complex-wavenumber probe")
@test DiffMoM._check_incident_wavenumber_match(
    nextfloat(0.0), nextfloat(0.0), "subnormal-wavenumber probe") === nothing

surface_total_err = try
    compute_total_field(mesh, rwg, I_zero, pw_total, triangle_center(mesh, 1), k; quad_order=3, eta0=eta0)
    false
catch
    true
end
@assert surface_total_err

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 7: Adjoint Gradient Verification
# ─────────────────────────────────────────────────
println("\n── Test 7: Adjoint gradient verification (CRITICAL) ──")

# Setup: impedance sheet problem with real impedance parameters
theta_real = fill(200.0, Nt)  # real impedance values
Z_full = assemble_full_Z(Z_efie, Mp, theta_real)
full_Z_output_bytes = sizeof(eltype(Z_full)) * length(Z_full)
@test_throws ArgumentError assemble_full_Z(
    Z_efie, Mp, theta_real;
    max_output_bytes=full_Z_output_bytes - 1)
@test assemble_full_Z(
    Z_efie, Mp, theta_real;
    max_output_bytes=full_Z_output_bytes) == Z_full
I_imp = Z_full \ v

# Objective
J_val = compute_objective(I_imp, Q)
println("  J(θ₀) = $J_val")

# Adjoint gradient
lambda = solve_adjoint(Z_full, Q, I_imp)
g_adj = gradient_impedance(Mp, I_imp, lambda)
println("  |g_adj| = $(norm(g_adj))")

# Gradient verification via central finite differences
#
# Note: complex-step is not used here because J(θ) = I†QI involves
# conjugation (sesquilinear form), which breaks analyticity.
# Central FD with h ≈ 1e-5 provides O(h²) accuracy, sufficient
# for validating the adjoint gradient.

function J_of_theta(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= theta_vec[p] .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

# Sanity check: J_of_theta at baseline should match J_val
J_check = J_of_theta(theta_real)
@assert abs(J_check - J_val) / max(abs(J_val), 1e-30) < 1e-12

println("  Checking adjoint vs central finite difference (h=1e-5)...")
fd_results = Float64[]
adj_results = Float64[]
rel_errors = Float64[]

n_check = min(Nt, 10)  # check first 10 parameters
h_fd = 1e-5
for p in 1:n_check
    g_fd = fd_grad(J_of_theta, theta_real, p; h=h_fd)
    rel_err = abs(g_adj[p] - g_fd) / max(abs(g_adj[p]), 1e-30)
    push!(fd_results, g_fd)
    push!(adj_results, g_adj[p])
    push!(rel_errors, rel_err)
    println("    p=$p: adj=$(g_adj[p])  fd=$g_fd  rel_err=$rel_err")
end

max_rel_err = maximum(rel_errors)
println("  Max relative error (adjoint vs central FD): $max_rel_err")
@assert max_rel_err < 1e-4 "Gradient verification FAILED: max rel error = $max_rel_err"

println("  PASS ✓  (adjoint gradients match central FD)")

# ─────────────────────────────────────────────────
# Test 8: FD Convergence Check
# ─────────────────────────────────────────────────
println("\n── Test 8: FD convergence rate check ──")

# Verify FD error decreases at O(h²) for central differences
# by comparing two step sizes on a single parameter
p_test = 1
h1 = 1e-4
h2 = 1e-5
g_fd1 = fd_grad(J_of_theta, theta_real, p_test; h=h1)
g_fd2 = fd_grad(J_of_theta, theta_real, p_test; h=h2)
err1 = abs(g_adj[p_test] - g_fd1)
err2 = abs(g_adj[p_test] - g_fd2)

if err1 > 1e-15 && err2 > 1e-15
    rate = log10(err1 / err2) / log10(h1 / h2)
    println("  Error at h=$h1: $err1")
    println("  Error at h=$h2: $err2")
    println("  Convergence rate: $rate  (expected ≈ 2 for central FD)")
    # Rate should be near 2 for central differences (O(h²))
    @assert rate > 1.5 "FD convergence rate too low: $rate (expected ~2)"
else
    println("  Errors at machine precision — gradient is exact")
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 9: Reciprocity Check
# ─────────────────────────────────────────────────
println("\n── Test 9: Reciprocity check ──")

# For EFIE on PEC: Z should be symmetric (Z = Z^T) due to reciprocity
# (Galerkin testing with the same basis/test functions)
sym_err = norm(Z_efie - transpose(Z_efie)) / norm(Z_efie)
println("  Symmetry error (EFIE, PEC): $sym_err")
# Note: due to quadrature, small symmetry error is expected
@assert sym_err < 1e-10 "EFIE matrix not symmetric: err = $sym_err"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 10: Optimization Smoke Test
# ─────────────────────────────────────────────────
println("\n── Test 10: Optimization smoke test ──")

theta_init = fill(300.0, Nt)
theta_opt, trace = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init;
    maxiter=10, tol=1e-8, alpha0=0.01, verbose=false
)

# Check that objective decreased
if length(trace) >= 2
    J_first = trace[1].J
    J_last  = trace[end].J
    println("  J(iter=1)  = $J_first")
    println("  J(iter=$(length(trace))) = $J_last")

end

# Fail closed on invalid controls, undefined ratios, and rejected projected
# steps. These small systems make the accepted iterate and solve counts exact.
Z_opt_guard = ComplexF64[1.0;;]
Mp_opt_guard = Matrix{Float64}[[1.0;;]]
v_opt_guard = ComplexF64[1.0]
Q_opt_guard = ComplexF64[1.0;;]
theta_opt_guard = [0.0]
@test optimize_lbfgs(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, theta_opt_guard;
    maxiter=0, verbose=false,
    max_workspace_bytes=sizeof(ComplexF64))[1] == theta_opt_guard
@test_throws ArgumentError optimize_lbfgs(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, theta_opt_guard;
    maxiter=0, verbose=false,
    max_workspace_bytes=sizeof(ComplexF64) - 1)
@test_throws ArgumentError optimize_directivity(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, Q_opt_guard,
    theta_opt_guard;
    maxiter=0, verbose=false,
    max_workspace_bytes=sizeof(ComplexF64) - 1)
@test transform_patch_matrices(
    Mp_opt_guard;
    preconditioner_M=Z_opt_guard,
    max_output_bytes=sizeof(ComplexF64))[1] ==
    Matrix{ComplexF64}[[1.0 + 0im;;]]
@test_throws ArgumentError transform_patch_matrices(
    Mp_opt_guard;
    preconditioner_M=Z_opt_guard,
    max_output_bytes=sizeof(ComplexF64) - 1)
@test_throws ArgumentError optimize_lbfgs(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, theta_opt_guard;
    maxiter=1, tol=Inf, verbose=false)
@test_throws ArgumentError optimize_lbfgs(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, theta_opt_guard;
    maxiter=1, alpha0=Inf, verbose=false)
@test_throws ArgumentError optimize_directivity(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, Q_opt_guard,
    theta_opt_guard;
    maxiter=1, regularization_alpha=Inf, verbose=false)
@test_throws ArgumentError optimize_lbfgs(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, theta_opt_guard;
    maxiter=1, lb=[1.0], ub=[0.0], verbose=false)
@test DiffMoM._recoverable_optimizer_trial_error(SingularException(1))
@test DiffMoM._recoverable_optimizer_trial_error(OverflowError("trial"))
@test !DiffMoM._recoverable_optimizer_trial_error(DomainError(1.0))
@test !DiffMoM._recoverable_optimizer_trial_error(
    ErrorException("sentinel implementation failure"))

theta_fixed_guard, trace_fixed_guard = optimize_lbfgs(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, theta_opt_guard;
    maxiter=3, tol=0.0, lb=[0.0], ub=[0.0], verbose=false)
@test theta_fixed_guard == theta_opt_guard
@test length(trace_fixed_guard) == 1
@test trace_fixed_guard[1].n_fwd == 1
theta_backtracked_guard, trace_backtracked_guard = optimize_lbfgs(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard, theta_opt_guard;
    maxiter=1, tol=0.0, alpha0=0.5, maximize=true, verbose=false)
@test theta_backtracked_guard ≈ [0.5]
@test length(trace_backtracked_guard) == 1
@test all(isfinite, theta_backtracked_guard)

Z_dir_guard = Matrix{ComplexF64}(I, 2, 2)
Mp_dir_guard = Matrix{Float64}[[1.0 0.0; 0.0 0.0]]
v_dir_guard = ComplexF64[1.0, 1.0]
Q_target_guard = ComplexF64[1.0 0.0; 0.0 0.0]
Q_total_guard = Matrix{ComplexF64}(I, 2, 2)
theta_dir_guard, trace_dir_guard = optimize_directivity(
    Z_dir_guard, Mp_dir_guard, v_dir_guard, Q_target_guard, Q_total_guard,
    theta_opt_guard;
    maxiter=3, tol=0.0, lb=[0.0], ub=[0.0], verbose=false)
@test theta_dir_guard == theta_opt_guard
@test length(trace_dir_guard) == 1
theta_dir_backtracked_guard, trace_dir_backtracked_guard = optimize_directivity(
    Z_dir_guard, Mp_dir_guard, v_dir_guard, Q_target_guard, Q_total_guard,
    theta_opt_guard;
    maxiter=1, tol=0.0, alpha0=2.0, verbose=false)
@test theta_dir_backtracked_guard ≈ [0.5]
@test length(trace_dir_backtracked_guard) == 1
@test all(isfinite, theta_dir_backtracked_guard)
@test_throws DomainError optimize_directivity(
    Z_opt_guard, Mp_opt_guard, v_opt_guard, Q_opt_guard,
    zeros(ComplexF64, 1, 1), theta_opt_guard;
    maxiter=1, verbose=false)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 11: Paper-consistency metrics from tracked data
# ─────────────────────────────────────────────────
println("\n── Test 11: Paper-consistency metrics ──")

meanval(x) = sum(x) / length(x)

function crossval_metrics(
    df_ref::DataFrame,
    ref_col::Symbol,
    df_cmp::DataFrame,
    cmp_col::Symbol;
    target_theta_deg::Float64=30.0,
)
    left = select(df_ref, :theta_deg, :phi_deg, ref_col)
    right = select(df_cmp, :theta_deg, :phi_deg, cmp_col)
    merged = innerjoin(left, right, on=[:theta_deg, :phi_deg])

    delta = merged[!, cmp_col] .- merged[!, ref_col]
    abs_delta = abs.(delta)

    theta_unique = unique(merged.theta_deg)
    theta_near = theta_unique[argmin(abs.(theta_unique .- target_theta_deg))]
    idx_target = findall(t -> abs(t - theta_near) < 1e-12, merged.theta_deg)

    return (
        n = nrow(merged),
        rmse = sqrt(sum(abs2, delta) / length(delta)),
        mean_abs = meanval(abs_delta),
        target_theta_near = theta_near,
        target_mean_abs = meanval(abs_delta[idx_target]),
    )
end

paper_metric_inputs = [
    joinpath(DATADIR, "convergence_study.csv"),
    joinpath(DATADIR, "gradient_verification.csv"),
    joinpath(DATADIR, "robustness_sweep.csv"),
    joinpath(DATADIR, "beam_steer_farfield.csv"),
    joinpath(DATADIR, "bempp_pec_farfield.csv"),
    joinpath(DATADIR, "julia_impedance_farfield.csv"),
    joinpath(DATADIR, "bempp_impedance_farfield.csv"),
]
imp_beam_csv = joinpath(DATADIR, "impedance_validation_matrix_summary.csv")
imp_beam_csv_alt = joinpath(DATADIR, "impedance_validation_matrix_summary_paper_default.csv")

missing_paper_metric_inputs = [path for path in paper_metric_inputs if !isfile(path)]
if !(isfile(imp_beam_csv) || isfile(imp_beam_csv_alt))
    push!(missing_paper_metric_inputs, imp_beam_csv)
end

if !isempty(missing_paper_metric_inputs)
    missing_names = join(basename.(missing_paper_metric_inputs), ", ")
    println("  SKIPPED (optional paper-consistency artifacts not found: $missing_names)")
else
conv = CSV.read(joinpath(DATADIR, "convergence_study.csv"), DataFrame)
grad = CSV.read(joinpath(DATADIR, "gradient_verification.csv"), DataFrame)
rob = CSV.read(joinpath(DATADIR, "robustness_sweep.csv"), DataFrame)

max_grad_mesh = maximum(conv.max_grad_err)
min_energy_ratio = minimum(conv.energy_ratio)
max_grad_ref = maximum(grad.rel_error)

idx_nom = findfirst(rob.case .== "f_nom")
idx_p2 = findfirst(rob.case .== "f_+2pct")
@assert idx_nom !== nothing
@assert idx_p2 !== nothing

J_opt_nom = rob.J_opt_pct[idx_nom]
J_pec_nom = rob.J_pec_pct[idx_nom]
peak_theta_p2 = rob.peak_theta_opt_deg[idx_p2]

df_pec_julia = CSV.read(joinpath(DATADIR, "beam_steer_farfield.csv"), DataFrame)
df_pec_bempp = CSV.read(joinpath(DATADIR, "bempp_pec_farfield.csv"), DataFrame)
df_imp_julia = CSV.read(joinpath(DATADIR, "julia_impedance_farfield.csv"), DataFrame)
df_imp_bempp = CSV.read(joinpath(DATADIR, "bempp_impedance_farfield.csv"), DataFrame)

if !isfile(imp_beam_csv)
    imp_beam_csv = imp_beam_csv_alt
end
df_imp_beam = CSV.read(imp_beam_csv, DataFrame)

pec_cv = crossval_metrics(df_pec_julia, :dir_pec_dBi, df_pec_bempp, :dir_bempp_dBi)
imp_cv = crossval_metrics(df_imp_julia, :dir_julia_imp_dBi, df_imp_bempp, :dir_bempp_imp_dBi)

n_beam_cases = nrow(df_imp_beam)
n_pass_main_theta = count(df_imp_beam.pass_main_theta_le_3deg)
n_pass_main_level = count(df_imp_beam.pass_main_level_le_1p5db)
n_pass_sll = count(df_imp_beam.pass_sll_le_3db)

println("  Max grad rel. err (reference): $max_grad_ref")
println("  Max grad rel. err (mesh sweep): $max_grad_mesh")
println("  Min energy ratio: $min_energy_ratio")
println("  Nominal J_opt/J_pec (%): $J_opt_nom / $J_pec_nom")
println("  +2% freq peak theta (deg): $peak_theta_p2")
println("  PEC CV RMSE / near-target |ΔD| (dB): $(pec_cv.rmse) / $(pec_cv.target_mean_abs)")
println("  IMP CV RMSE / near-target |ΔD| (dB): $(imp_cv.rmse) / $(imp_cv.target_mean_abs)")
println("  Beam-centric passes (main θ / main L / SLL): $n_pass_main_theta/$n_beam_cases, $n_pass_main_level/$n_beam_cases, $n_pass_sll/$n_beam_cases")

# These checks track manuscript quantitative claims.
@assert max_grad_ref < 3e-7
@assert max_grad_mesh < 3e-6
@assert min_energy_ratio > 0.98
@assert J_opt_nom > J_pec_nom
@assert peak_theta_p2 < 5.0
@assert pec_cv.target_mean_abs < 0.5
@assert n_pass_main_theta == n_beam_cases
@assert n_pass_main_level == n_beam_cases
@assert n_pass_sll == n_beam_cases

println("  PASS ✓")
end  # if isfile(convergence_study.csv)

# ─────────────────────────────────────────────────
# Test 12: Conditioning / preconditioning consistency
# ─────────────────────────────────────────────────
println("\n── Test 12: Conditioning and preconditioning ──")

theta_c = copy(theta_real)
Z_raw = assemble_full_Z(Z_efie, Mp, theta_c)
I_raw = Z_raw \ v

# Build mass-based regularizer and left preconditioner
R_mass = make_mass_regularizer(Mp)
M_left = make_left_preconditioner(Mp; eps_rel=1e-6)
mass_regularizer_bytes = sizeof(ComplexF64) * N^2
@test_throws ArgumentError make_mass_regularizer(
    Mp; max_output_bytes=mass_regularizer_bytes - 1)
@test make_mass_regularizer(
    Mp; max_output_bytes=mass_regularizer_bytes) == R_mass
@test_throws ArgumentError make_left_preconditioner(
    Mp;
    eps_rel=1e-6,
    max_output_bytes=mass_regularizer_bytes - 1)
make_mass_regularizer(Mp)
make_left_preconditioner(Mp; eps_rel=1e-6)
@test (@allocated make_mass_regularizer(Mp)) <=
      _complex_matrix_output_allocation(N, N) + 128
@test (@allocated make_left_preconditioner(Mp; eps_rel=1e-6)) <=
      _complex_matrix_output_allocation(N, N) + 128
@test ishermitian(R_mass)
@test_throws ArgumentError make_left_preconditioner(Mp; eps_rel=0.0)
@test_throws ArgumentError make_left_preconditioner(Mp; eps_rel=Inf)
@test_throws ArgumentError make_mass_regularizer(
    [ComplexF64[NaN 0.0; 0.0 1.0]])

@test_throws ArgumentError select_preconditioner(
    Matrix{ComplexF64}[]; mode=:off)
@test_throws ArgumentError select_preconditioner(
    Mp; mode=:off, n_threshold=-1)
@test_throws ArgumentError select_preconditioner(
    Mp; mode=:off, eps_rel=NaN)
@test_throws DimensionMismatch select_preconditioner(
    Mp; preconditioner_M=ones(ComplexF64, N + 1, N + 1))
bad_preconditioner_probe = Matrix{ComplexF64}(I, N, N)
bad_preconditioner_probe[1, 1] = NaN + 0im
@test_throws ArgumentError select_preconditioner(
    Mp; preconditioner_M=bad_preconditioner_probe)

# Auto-preconditioning selector behavior
M_auto_off, enabled_auto_off, reason_auto_off = select_preconditioner(
    Mp;
    mode=:auto,
    n_threshold=10_000,
    iterative_solver=false,
)
println("  Auto preconditioning (high threshold): enabled=$enabled_auto_off ($reason_auto_off)")
@assert !enabled_auto_off
@assert M_auto_off === nothing

M_auto_on, enabled_auto_on, reason_auto_on = select_preconditioner(
    Mp;
    mode=:auto,
    n_threshold=1,
    iterative_solver=false,
    eps_rel=1e-6,
)
println("  Auto preconditioning (low threshold): enabled=$enabled_auto_on ($reason_auto_on)")
@assert enabled_auto_on
@assert M_auto_on !== nothing

# Left-preconditioned system should preserve the same solution
Z_pre, v_pre, fac_pre = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=0.0,
    regularization_R=nothing,
    preconditioner_M=M_left,
)
I_pre = Z_pre \ v_pre
rel_I_pre = norm(I_pre - I_raw) / max(norm(I_raw), 1e-30)
println("  Left-preconditioned solution mismatch: $rel_I_pre")
@assert rel_I_pre < 1e-10

# Auto-selected preconditioner (activated) should also preserve the solution
Z_pre_auto, v_pre_auto, _ = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=0.0,
    regularization_R=nothing,
    preconditioner_M=M_auto_on,
)
I_pre_auto = Z_pre_auto \ v_pre_auto
rel_I_pre_auto = norm(I_pre_auto - I_raw) / max(norm(I_raw), 1e-30)
println("  Auto-preconditioned solution mismatch: $rel_I_pre_auto")
@assert rel_I_pre_auto < 1e-10

# Regularization should alter the solve (for nonzero alpha)
alpha_reg = 1e-3
Z_reg, v_reg, _ = prepare_conditioned_system(
    Z_raw,
    v;
    regularization_alpha=alpha_reg,
    regularization_R=R_mass,
    preconditioner_M=nothing,
)
@test_throws DimensionMismatch prepare_conditioned_system(
    ones(ComplexF64, 2, 1), ComplexF64[1.0, 2.0])
@test_throws ArgumentError prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=NaN,
    regularization_R=R_mass,
)
@test_throws ArgumentError prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=-1.0,
    regularization_R=R_mass,
)
@test_throws DimensionMismatch prepare_conditioned_system(
    Z_raw, v;
    regularization_alpha=1.0,
    regularization_R=ones(ComplexF64, N + 1, N + 1),
)
@test_throws ArgumentError prepare_conditioned_system(
    Z_raw, v; preconditioner_M=bad_preconditioner_probe)
I_reg = Z_reg \ v_reg
rel_I_reg = norm(I_reg - I_raw) / max(norm(I_raw), 1e-30)
println("  Regularized solution change (alpha=$alpha_reg): $rel_I_reg")
@assert rel_I_reg > 1e-9

# Adjoint gradient with left preconditioning should match FD on the
# equivalently preconditioned system objective.
Mp_pre, fac_pre = transform_patch_matrices(
    Mp;
    preconditioner_M=M_left,
    preconditioner_factor=fac_pre,
)
@test_throws DimensionMismatch transform_patch_matrices(
    Mp; preconditioner_M=ones(ComplexF64, N + 1, N + 1))
@test_throws ArgumentError transform_patch_matrices(
    Mp; preconditioner_M=bad_preconditioner_probe)

conditioning_tiny_scale = nextfloat(0.0)
conditioning_rhs_scale = floatmin(Float64)
conditioning_preconditioner = Matrix(Diagonal(
    fill(ComplexF64(conditioning_tiny_scale), 2)))
conditioning_rhs_matrix = Matrix(Diagonal(
    fill(ComplexF64(conditioning_rhs_scale), 2)))
conditioning_rhs_vector = fill(ComplexF64(conditioning_rhs_scale), 2)
conditioning_reference_scale = setprecision(BigFloat, 4096) do
    Float64(BigFloat(conditioning_rhs_scale) /
            BigFloat(conditioning_tiny_scale))
end
@test conditioning_reference_scale == 2.0^52
conditioning_reference_matrix = Matrix(Diagonal(
    fill(ComplexF64(conditioning_reference_scale), 2)))
conditioning_reference_vector = fill(
    ComplexF64(conditioning_reference_scale), 2)

conditioning_Mp, conditioning_factor = transform_patch_matrices(
    [conditioning_rhs_matrix];
    preconditioner_M=conditioning_preconditioner,
)
@test conditioning_Mp[1] == conditioning_reference_matrix
conditioning_Mp_factor_only, _ = transform_patch_matrices(
    [conditioning_rhs_matrix];
    preconditioner_factor=conditioning_factor,
)
@test conditioning_Mp_factor_only[1] == conditioning_reference_matrix

conditioning_Z, conditioning_v, _ = prepare_conditioned_system(
    conditioning_rhs_matrix,
    conditioning_rhs_vector;
    preconditioner_M=conditioning_preconditioner,
)
@test conditioning_Z == conditioning_reference_matrix
@test conditioning_v == conditioning_reference_vector
conditioning_Z_factor_only, conditioning_v_factor_only, _ =
    prepare_conditioned_system(
        conditioning_rhs_matrix,
        conditioning_rhs_vector;
        preconditioner_factor=conditioning_factor,
    )
@test conditioning_Z_factor_only == conditioning_reference_matrix
@test conditioning_v_factor_only == conditioning_reference_vector
conditioning_Z_regularized, conditioning_v_regularized, _ =
    prepare_conditioned_system(
        zeros(ComplexF64, 2, 2),
        conditioning_rhs_vector;
        regularization_alpha=1.0,
        regularization_R=conditioning_rhs_matrix,
        preconditioner_M=conditioning_preconditioner,
    )
@test conditioning_Z_regularized == conditioning_reference_matrix
@test conditioning_v_regularized == conditioning_reference_vector

conditioning_zero_Mp = [zeros(ComplexF64, 2, 2)]
conditioning_zero_Q = zeros(ComplexF64, 2, 2)
conditioning_theta, conditioning_trace = optimize_lbfgs(
    conditioning_rhs_matrix,
    conditioning_zero_Mp,
    conditioning_rhs_vector,
    conditioning_zero_Q,
    [0.0];
    maxiter=1,
    verbose=false,
    preconditioner_M=conditioning_preconditioner,
)
@test conditioning_theta == [0.0]
@test conditioning_trace[1].J == 0.0
conditioning_directivity_theta, conditioning_directivity_trace =
    optimize_directivity(
        conditioning_rhs_matrix,
        conditioning_zero_Mp,
        conditioning_rhs_vector,
        conditioning_zero_Q,
        Matrix{ComplexF64}(I, 2, 2),
        [0.0];
        maxiter=1,
        verbose=false,
        preconditioner_M=conditioning_preconditioner,
    )
@test conditioning_directivity_theta == [0.0]
@test conditioning_directivity_trace[1].J == 0.0
lambda_pre = solve_adjoint(Z_pre, Q, I_pre)
g_pre = gradient_impedance(Mp_pre, I_pre, lambda_pre)

function J_of_theta_pre(theta_vec)
    Z_t = assemble_full_Z(Z_efie, Mp, theta_vec)
    Zp, vp, _ = prepare_conditioned_system(
        Z_t,
        v;
        regularization_alpha=0.0,
        regularization_R=nothing,
        preconditioner_M=M_left,
        preconditioner_factor=fac_pre,
    )
    I_t = Zp \ vp
    return real(dot(I_t, Q * I_t))
end

p_pre = 1
g_fd_pre = fd_grad(J_of_theta_pre, theta_c, p_pre; h=1e-5)
rel_g_pre = abs(g_pre[p_pre] - g_fd_pre) / max(abs(g_pre[p_pre]), 1e-30)
println("  Preconditioned gradient rel. error (p=$p_pre): $rel_g_pre")
@assert rel_g_pre < 1e-4

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 13: Sphere MoM-vs-Mie benchmark CI gate
# ─────────────────────────────────────────────────
println("\n── Test 13: Sphere Mie benchmark gate ──")

obj_sphere = joinpath(DATADIR, "tmp_sphere_gate.obj")
write_icosphere_obj(obj_sphere; radius=0.05, subdivisions=2)

mesh_s = read_obj_mesh(obj_sphere)
rwg_s = build_rwg(mesh_s)

freq_s = 2.0e9
c0_s = 299792458.0
lambda_s = c0_s / freq_s
k_s = 2π / lambda_s
eta0_s = 376.730313668
k_vec_s = Vec3(0.0, 0.0, -k_s)
khat_s = k_vec_s / norm(k_vec_s)
pol_s = Vec3(1.0, 0.0, 0.0)

ctr_s = vec(sum(mesh_s.xyz, dims=2) ./ nvertices(mesh_s))
radii_s = [norm(Vec3(mesh_s.xyz[:, i]) - Vec3(ctr_s)) for i in 1:nvertices(mesh_s)]
a_s = sum(radii_s) / length(radii_s)

Z_s = assemble_Z_efie(mesh_s, rwg_s, k_s; quad_order=3, eta0=eta0_s)
v_s = assemble_v_plane_wave(mesh_s, rwg_s, k_vec_s, 1.0, pol_s; quad_order=3)
I_s = solve_forward(Z_s, v_s)
res_s = norm(Z_s * I_s - v_s) / max(norm(v_s), 1e-30)
@assert res_s < 1e-10

grid_s = make_sph_grid(37, 18)
G_s = radiation_vectors(mesh_s, rwg_s, grid_s, k_s; quad_order=3, eta0=eta0_s)
E_s = compute_farfield(G_s, I_s, length(grid_s.w))
σ_mom_s = bistatic_rcs(E_s; E0=1.0)

phi_target_s = grid_s.phi[argmin(grid_s.phi)]
idx_cut_s = [q for q in 1:length(grid_s.w) if abs(grid_s.phi[q] - phi_target_s) < 1e-12]
idx_cut_s = idx_cut_s[sortperm(grid_s.theta[idx_cut_s])]

σ_mie_s = [
    mie_bistatic_rcs_pec(k_s, a_s, khat_s, pol_s, Vec3(grid_s.rhat[:, q]))
    for q in idx_cut_s
]

dB_mom_s = 10 .* log10.(max.(σ_mom_s[idx_cut_s], 1e-30))
dB_mie_s = 10 .* log10.(max.(σ_mie_s, 1e-30))
ΔdB_s = dB_mom_s .- dB_mie_s

mae_s = sum(abs.(ΔdB_s)) / length(ΔdB_s)
rmse_s = sqrt(sum(abs2, ΔdB_s) / length(ΔdB_s))
maxabs_s = maximum(abs.(ΔdB_s))

σ_bs_mom_s = backscatter_rcs(E_s, grid_s, khat_s; E0=1.0).sigma
σ_bs_mie_s = mie_bistatic_rcs_pec(k_s, a_s, khat_s, pol_s, -khat_s)
Δbs_s = 10 * log10(max(σ_bs_mom_s, 1e-30)) - 10 * log10(max(σ_bs_mie_s, 1e-30))

println("  MAE(dB): $mae_s")
println("  RMSE(dB): $rmse_s")
println("  Max |Δ|(dB): $maxabs_s")
println("  Backscatter Δ(dB): $Δbs_s")

# Dedicated CI thresholds for the sphere benchmark (coarse grid tolerances)
@assert mae_s < 0.50 "Sphere Mie gate failed: MAE(dB)=$mae_s"
@assert rmse_s < 0.60 "Sphere Mie gate failed: RMSE(dB)=$rmse_s"
@assert maxabs_s < 1.20 "Sphere Mie gate failed: max |Δ|(dB)=$maxabs_s"
@assert abs(Δbs_s) < 1.20 "Sphere Mie gate failed: |backscatter Δ(dB)|=$(abs(Δbs_s))"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 14: Excitation-model physics sanity checks
# ─────────────────────────────────────────────────
println("\n── Test 14: Excitation physics sanity checks ──")

mesh_exc = make_rect_plate(0.02, 0.02, 3, 3)
rwg_exc = build_rwg(mesh_exc)

freq_exc = 1.0e9
k_exc = 2π * freq_exc / 299792458.0
k_vec_exc = Vec3(0.0, 0.0, -k_exc)
pol_exc = Vec3(1.0, 0.0, 0.0)

@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc,
    make_plane_wave(Vec3(0.0, 0.0, 0.0), 1.0, pol_exc);
    quad_order=3)
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc,
    make_plane_wave(k_vec_exc, NaN, pol_exc);
    quad_order=3)
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc,
    make_plane_wave(k_vec_exc, 1.0, Vec3(0.0, 0.0, 0.0));
    quad_order=3)
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc,
    make_plane_wave(k_vec_exc, 1.0, k_vec_exc);
    quad_order=3)
@test_throws ArgumentError make_plane_wave(
    k_vec_exc, 1.0, Vec3(2.0, 0.0, 0.0))

v_old_exc = assemble_v_plane_wave(mesh_exc, rwg_exc, k_vec_exc, 1.0, pol_exc; quad_order=3)
v_new_exc = assemble_excitation(mesh_exc, rwg_exc, make_plane_wave(k_vec_exc, 1.0, pol_exc); quad_order=3)
rel_rhs_exc = norm(v_new_exc - v_old_exc) / max(norm(v_old_exc), 1e-30)
println("  Plane-wave path-consistency RHS rel. diff: $rel_rhs_exc")
@assert rel_rhs_exc < 1e-13

# Surface integration must combine a finite extreme triangle area with a tiny
# incident amplitude before either factor overflows or underflows.  Scaling
# coordinates by L, the wave vector by 1/L, and E0 by 1/L² leaves this RHS
# exactly invariant.
excitation_surface_scale = ldexp(1.0, 512)
excitation_surface_unit_mesh = make_rect_plate(1.0, 1.0, 1, 1)
excitation_surface_unit_rwg = build_rwg(excitation_surface_unit_mesh)
excitation_surface_large_mesh = make_rect_plate(
    excitation_surface_scale, excitation_surface_scale, 1, 1)
excitation_surface_large_rwg = build_rwg(excitation_surface_large_mesh)
excitation_surface_reference = assemble_excitation(
    excitation_surface_unit_mesh, excitation_surface_unit_rwg,
    make_plane_wave(Vec3(0.0, 0.0, -1.0), 1.0, pol_exc);
    quad_order=3)
excitation_surface_result = assemble_excitation(
    excitation_surface_large_mesh, excitation_surface_large_rwg,
    make_plane_wave(
        Vec3(0.0, 0.0, -inv(excitation_surface_scale)),
        inv(excitation_surface_scale)^2,
        pol_exc,
    );
    quad_order=3)
@test isapprox(
    excitation_surface_result, excitation_surface_reference;
    rtol=2eps(Float64), atol=0.0)
wrong_mesh_cache = DiffMoM.ExcitationQuadCache(
    make_rect_plate(0.08, 0.04, 3, 3))
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc, make_plane_wave(k_vec_exc, 1.0, pol_exc);
    quad_order=3, quad_cache=wrong_mesh_cache)
shared_quad_cache = DiffMoM.ExcitationQuadCache(mesh_exc)
@test shared_quad_cache.cache_lock isa ReentrantLock
if Threads.nthreads() > 1
    cache_orders = (3, 4, 3, 4, 3, 4, 3, 4)
    for _ in 1:20
        empty_cache = DiffMoM.ExcitationQuadCache(mesh_exc)
        cache_gate = Base.Event()
        cache_tasks = map(cache_orders) do order
            Threads.@spawn begin
                wait(cache_gate)
                DiffMoM._quad_cache_for(empty_cache, mesh_exc, order)
            end
        end
        yield()
        notify(cache_gate)
        cache_results = fetch.(cache_tasks)
        @test sort!(collect(keys(empty_cache.by_order))) == [3, 4]
        for (result, order) in zip(cache_results, cache_orders)
            @test result === empty_cache.by_order[order]
        end
    end
end

# Explicit quadrature check for plane-wave RHS assembly
xi_exc, wq_exc = tri_quad_rule(3)
v_manual_exc = zeros(ComplexF64, rwg_exc.nedges)
for n in 1:rwg_exc.nedges
    for t in (rwg_exc.tplus[n], rwg_exc.tminus[n])
        A = triangle_area(mesh_exc, t)
        pts = tri_quad_points(mesh_exc, t, xi_exc)
        for q in eachindex(wq_exc)
            rq = pts[q]
            fn = eval_rwg(rwg_exc, n, rq, t)
            Einc = pol_exc * exp(-1im * dot(k_vec_exc, rq))
            v_manual_exc[n] += -wq_exc[q] * dot(fn, Einc) * (2 * A)
        end
    end
end
rel_rhs_manual_exc = norm(v_new_exc - v_manual_exc) / max(norm(v_manual_exc), 1e-30)
println("  Plane-wave vs explicit quadrature RHS rel. diff: $rel_rhs_manual_exc")
@assert rel_rhs_manual_exc < 1e-12

gap_a = make_delta_gap(1, 1.0 + 0im, 1e-3)
gap_b = make_delta_gap(1, 1.0 + 0im, 2e-3)
@test_throws ArgumentError make_delta_gap(1, 1.0 + 0im, Inf)
@test_throws ArgumentError make_delta_gap(1, Inf + 0im, 1e-3)
v_gap_a = assemble_excitation(mesh_exc, rwg_exc, gap_a)
v_gap_b = assemble_excitation(mesh_exc, rwg_exc, gap_b)
ratio_gap = abs(v_gap_a[1]) / max(abs(v_gap_b[1]), 1e-30)
println("  Delta-gap scaling ratio (g=1mm/g=2mm): $ratio_gap")
@assert abs(ratio_gap - 2.0) < 1e-12
@assert norm(v_gap_a[2:end]) < 1e-12

port_exc = PortExcitation([1, 2], 2.0 + 0im, 50.0 + 0im)
v_port = assemble_excitation(mesh_exc, rwg_exc, port_exc)
@assert abs(v_port[1] - (2.0 / rwg_exc.len[1])) < 1e-14
@assert abs(v_port[2] - (2.0 / rwg_exc.len[2])) < 1e-14
@assert norm(v_port[3:end]) < 1e-14
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc, PortExcitation([1], Inf + 0im, 50.0 + 0im))
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc, PortExcitation([1], 1.0 + 0im, Inf + 0im))

# A partially invalid port definition must fail closed rather than silently
# dropping intended driven edges.
port_oob = PortExcitation([1, rwg_exc.nedges + 10], 1.0 + 0im, 50.0 + 0im)
@test_throws ArgumentError assemble_excitation(mesh_exc, rwg_exc, port_oob)

thrown_multi = try
    bad_multi = make_multi_excitation([gap_a, gap_b], [1 + 0im])
    assemble_excitation(mesh_exc, rwg_exc, bad_multi)
    false
catch
    true
end
@assert thrown_multi
@test_throws ArgumentError make_multi_excitation(
    typeof(gap_a)[], ComplexF64[])
@test_throws ArgumentError make_multi_excitation(
    [gap_a], ComplexF64[Inf + 0im])

cyclic_multi_children = DiffMoM.AbstractExcitation[]
cyclic_multi = MultiExcitation(
    cyclic_multi_children, ComplexF64[1.0 + 0im])
push!(cyclic_multi_children, cyclic_multi)
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc, cyclic_multi)
@test_throws ArgumentError assemble_multiple_excitations(
    mesh_exc, rwg_exc, [cyclic_multi])

too_deep_multi = let nested::DiffMoM.AbstractExcitation = gap_a
    for _ in 1:(DiffMoM._MAX_MULTI_EXCITATION_DEPTH + 1)
        nested = MultiExcitation(
            DiffMoM.AbstractExcitation[nested],
            ComplexF64[1.0 + 0im])
    end
    nested
end
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc, too_deep_multi)

# Reject malformed public excitation structs before allocating the dense batch
# result.  Only `nedges` is intentionally enlarged: child validation must run
# before any RWG array is inspected or an N-by-M output is allocated.
invalid_batch_multi = MultiExcitation(
    DiffMoM.AbstractExcitation[], ComplexF64[])
large_invalid_batch_rwg = typeof(rwg_exc)(
    rwg_exc.mesh, 100_000, rwg_exc.tplus, rwg_exc.tminus, rwg_exc.evert,
    rwg_exc.vplus_opp, rwg_exc.vminus_opp, rwg_exc.len,
    rwg_exc.area_plus, rwg_exc.area_minus, rwg_exc.coeff_plus,
    rwg_exc.coeff_minus, rwg_exc.has_periodic_bloch)
@test _multiple_excitation_rejection_allocations(
    mesh_exc, large_invalid_batch_rwg, [invalid_batch_multi]) <= 100_000

V_exc = assemble_multiple_excitations(mesh_exc, rwg_exc, [gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)]; quad_order=3)
@assert size(V_exc) == (rwg_exc.nedges, 2)
@assert norm(V_exc[:, 1] - v_gap_a) / max(norm(v_gap_a), 1e-30) < 1e-13
@assert norm(V_exc[:, 2] - v_new_exc) / max(norm(v_new_exc), 1e-30) < 1e-13
multiple_excitation_bytes = sizeof(ComplexF64) * length(V_exc)
multiple_excitation_nq = length(tri_quad_rule(3)[2])
multiple_excitation_work_bytes = DiffMoM._multiple_excitation_work_bytes(
    rwg_exc.nedges, 2, ntriangles(mesh_exc), multiple_excitation_nq,
    multiple_excitation_bytes)
multiple_excitation_terms = DiffMoM._multiple_excitation_term_count(
    rwg_exc.nedges, 2, multiple_excitation_nq)
@test_throws ArgumentError assemble_multiple_excitations(
    mesh_exc, rwg_exc,
    [gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)];
    quad_order=3,
    max_output_bytes=multiple_excitation_bytes - 1)
@test_throws ArgumentError assemble_multiple_excitations(
    mesh_exc, rwg_exc,
    [gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)];
    quad_order=3,
    max_work_bytes=multiple_excitation_work_bytes - 1)
@test_throws ArgumentError assemble_multiple_excitations(
    mesh_exc, rwg_exc,
    [gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)];
    quad_order=3,
    max_terms=multiple_excitation_terms - 1)
@test assemble_multiple_excitations(
    mesh_exc, rwg_exc,
    [gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)];
    quad_order=3,
    max_output_bytes=multiple_excitation_bytes,
    max_work_bytes=multiple_excitation_work_bytes,
    max_terms=multiple_excitation_terms) == V_exc

# Imported excitations can raise the effective quadrature order.  Batch limits
# must charge that order, and a mixed batch must charge both retained caches.
batch_imported_order7 = ImportedExcitation(
    _ -> CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im);
    min_quad_order=7)
batch_imported_output_bytes = sizeof(ComplexF64) * rwg_exc.nedges
batch_imported_work_order1 = DiffMoM._multiple_excitation_work_bytes(
    rwg_exc.nedges, 1, ntriangles(mesh_exc), 1,
    batch_imported_output_bytes)
batch_imported_work_order7 = DiffMoM._multiple_excitation_work_bytes(
    rwg_exc.nedges, 1, ntriangles(mesh_exc), 7,
    batch_imported_output_bytes)
batch_imported_terms_order1 = DiffMoM._multiple_excitation_term_count(
    rwg_exc.nedges, 1, 1)
batch_imported_terms_order7 = DiffMoM._multiple_excitation_term_count(
    rwg_exc.nedges, 1, 7)
@test_throws ArgumentError assemble_multiple_excitations(
    mesh_exc, rwg_exc, [batch_imported_order7];
    quad_order=1,
    max_output_bytes=batch_imported_output_bytes,
    max_work_bytes=batch_imported_work_order1,
    max_terms=batch_imported_terms_order7)
@test_throws ArgumentError assemble_multiple_excitations(
    mesh_exc, rwg_exc, [batch_imported_order7];
    quad_order=1,
    max_output_bytes=batch_imported_output_bytes,
    max_work_bytes=batch_imported_work_order7,
    max_terms=batch_imported_terms_order1)
@test size(assemble_multiple_excitations(
    mesh_exc, rwg_exc, [batch_imported_order7];
    quad_order=1,
    max_output_bytes=batch_imported_output_bytes,
    max_work_bytes=batch_imported_work_order7,
    max_terms=batch_imported_terms_order7)) == (rwg_exc.nedges, 1)

batch_mixed_orders = [
    make_plane_wave(k_vec_exc, 1.0, pol_exc), batch_imported_order7]
batch_mixed_output_bytes =
    sizeof(ComplexF64) * rwg_exc.nedges * length(batch_mixed_orders)
batch_mixed_single_cache_bytes = DiffMoM._multiple_excitation_work_bytes(
    rwg_exc.nedges, length(batch_mixed_orders), ntriangles(mesh_exc), 7,
    batch_mixed_output_bytes)
batch_mixed_work_bytes = DiffMoM._multiple_excitation_profile_work_bytes(
    rwg_exc.nedges, ntriangles(mesh_exc), (1, 7), 1,
    batch_mixed_output_bytes)
batch_mixed_terms = DiffMoM._multiple_excitation_term_count(
    rwg_exc.nedges, BigInt(1 + 7))
@test batch_mixed_work_bytes > batch_mixed_single_cache_bytes
@test_throws ArgumentError assemble_multiple_excitations(
    mesh_exc, rwg_exc, batch_mixed_orders;
    quad_order=1,
    max_output_bytes=batch_mixed_output_bytes,
    max_work_bytes=batch_mixed_single_cache_bytes,
    max_terms=batch_mixed_terms)
@test size(assemble_multiple_excitations(
    mesh_exc, rwg_exc, batch_mixed_orders;
    quad_order=1,
    max_output_bytes=batch_mixed_output_bytes,
    max_work_bytes=batch_mixed_work_bytes,
    max_terms=batch_mixed_terms)) ==
      (rwg_exc.nedges, length(batch_mixed_orders))

# A late exceptional child completes the normal MultiExcitation pass and then
# reassembles every child exactly.  `max_terms` must bound both passes.
batch_multi_source_calls = Ref(0)
batch_multi_normal = ImportedExcitation(
    _ -> begin
        batch_multi_source_calls[] += 1
        CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    end;
    min_quad_order=1)
batch_multi_tiny = ImportedExcitation(
    _ -> begin
        batch_multi_source_calls[] += 1
        CVec3(1.0e-300 + 0im, 0.0 + 0im, 0.0 + 0im)
    end;
    min_quad_order=1)
batch_late_exact_multi = make_multi_excitation(
    [batch_multi_normal, batch_multi_tiny],
    ComplexF64[1.0 + 0im, 1.0e300 + 0im])
batch_late_exact_terms = DiffMoM._multiple_excitation_term_count(
    rwg_exc.nedges, BigInt(4))
@test_throws ArgumentError assemble_multiple_excitations(
    mesh_exc, rwg_exc, [batch_late_exact_multi];
    quad_order=1, max_terms=batch_late_exact_terms - 1)
@test batch_multi_source_calls[] == 0
@test size(assemble_multiple_excitations(
    mesh_exc, rwg_exc, [batch_late_exact_multi];
    quad_order=1, max_terms=batch_late_exact_terms)) ==
      (rwg_exc.nedges, 1)
@test batch_multi_source_calls[] == batch_late_exact_terms

weights_exc = ComplexF64[0.3 - 0.1im, -0.2 + 0.7im]
multi_exc = make_multi_excitation([gap_a, make_plane_wave(k_vec_exc, 1.0, pol_exc)], weights_exc)
v_multi_exc = assemble_excitation(mesh_exc, rwg_exc, multi_exc; quad_order=3)
v_multi_ref = V_exc * weights_exc
rel_multi_exc = norm(v_multi_exc - v_multi_ref) / max(norm(v_multi_ref), 1e-30)
println("  Multi-excitation linearity rel. diff: $rel_multi_exc")
@assert rel_multi_exc < 1e-13

# Imported excitation semantics:
# kind=:electric_field should match direct electric-field import exactly.
E_field_test(r) = CVec3(r[1] + 0im, (0.5 * r[2]) + 0im, 0.0 + 0im)
cur_E = make_imported_excitation(E_field_test; kind=:electric_field, min_quad_order=3)
imp_E = ImportedExcitation(E_field_test; kind=:electric_field, min_quad_order=3)
v_cur_E = assemble_excitation(mesh_exc, rwg_exc, cur_E; quad_order=3)
v_imp_E = assemble_excitation(mesh_exc, rwg_exc, imp_E; quad_order=3)
rel_cur_imp = norm(v_cur_E - v_imp_E) / max(norm(v_imp_E), 1e-30)
println("  ImportedExcitation(:electric_field) self-consistency rel. diff: $rel_cur_imp")
@assert rel_cur_imp < 1e-13
imported_points = [Vec3(i / 128, 0.25, -0.5) for i in 1:128]
imported_sum_ref = sum(E_field_test, imported_points)
@assert _sum_imported_source(imp_E, imported_points) == imported_sum_ref
_sum_imported_source(imp_E, imported_points)
@assert @allocated(_sum_imported_source(imp_E, imported_points)) <= 128

# Source function can return tuple/vector-like 3-component data.
E_field_tuple(r) = (r[1] + 0im, (0.5 * r[2]) + 0im, 0.0 + 0im)
cur_E_tuple = make_imported_excitation(E_field_tuple; kind=:electric_field, min_quad_order=3)
v_cur_E_tuple = assemble_excitation(mesh_exc, rwg_exc, cur_E_tuple; quad_order=3)
rel_cur_tuple = norm(v_cur_E_tuple - v_imp_E) / max(norm(v_imp_E), 1e-30)
println("  ImportedExcitation tuple-return rel. diff: $rel_cur_tuple")
@assert rel_cur_tuple < 1e-13

# surface-current mode uses local equivalent-sheet map E ≈ η Js.
Js_test(r) = CVec3((2r[1]) + 0im, 0.0 + 0im, 0.0 + 0im)
eta_test = 120.0 + 30.0im
cur_Js = make_imported_excitation(Js_test; kind=:surface_current_density, eta_equiv=eta_test, min_quad_order=3)
imp_etaJs = ImportedExcitation(r -> eta_test * Js_test(r); kind=:electric_field, min_quad_order=3)
v_cur_Js = assemble_excitation(mesh_exc, rwg_exc, cur_Js; quad_order=3)
v_imp_Js = assemble_excitation(mesh_exc, rwg_exc, imp_etaJs; quad_order=3)
rel_cur_js = norm(v_cur_Js - v_imp_Js) / max(norm(v_imp_Js), 1e-30)
println("  ImportedExcitation(:surface_current_density) map rel. diff: $rel_cur_js")
@assert rel_cur_js < 1e-13
@test_throws ArgumentError make_imported_excitation(
    Js_test; kind=:surface_current_density, eta_equiv=Inf + 0im)

# Imported field can also be tuple/vector-like.
imp_tuple = ImportedExcitation(r -> (r[1] + 0im, 0.0 + 0im, 0.0 + 0im); kind=:electric_field, min_quad_order=3)
v_imp_tuple = assemble_excitation(mesh_exc, rwg_exc, imp_tuple; quad_order=3)
@assert all(isfinite, real.(v_imp_tuple))
@assert all(isfinite, imag.(v_imp_tuple))

thrown_imp_bad_dim = try
    imp_bad_dim = ImportedExcitation(r -> ComplexF64[1 + 0im, 2 + 0im]; kind=:electric_field)
    assemble_excitation(mesh_exc, rwg_exc, imp_bad_dim; quad_order=3)
    false
catch
    true
end
@assert thrown_imp_bad_dim

thrown_imp_nonfinite = try
    imp_nonfinite = ImportedExcitation(r -> CVec3(NaN + 0im, 0 + 0im, 0 + 0im); kind=:electric_field)
    assemble_excitation(mesh_exc, rwg_exc, imp_nonfinite; quad_order=3)
    false
catch
    true
end
@assert thrown_imp_nonfinite

# Constructor guard
thrown_cur = try
    make_imported_excitation(E_field_test; min_quad_order=0)
    false
catch
    true
end
@assert thrown_cur
@test_throws ArgumentError make_imported_excitation(
    E_field_test; min_quad_order=8)
@test_throws ArgumentError make_imported_excitation(
    E_field_test; kind=:unsupported)
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc, imp_E; quad_order=0)
@test_throws ArgumentError assemble_excitation(
    mesh_exc, rwg_exc, imp_E; quad_order=8)
@test DiffMoM._effective_quad_order(2, 1) == 3

thrown_cur_bad_dim = try
    cur_bad_dim = make_imported_excitation(r -> ComplexF64[1 + 0im, 2 + 0im];
                                           kind=:electric_field,
                                           min_quad_order=3)
    assemble_excitation(mesh_exc, rwg_exc, cur_bad_dim; quad_order=3)
    false
catch
    true
end
@assert thrown_cur_bad_dim

thrown_cur_nonfinite = try
    cur_nonfinite = make_imported_excitation(r -> CVec3(NaN + 0im, 0 + 0im, 0 + 0im);
                                             kind=:electric_field,
                                             min_quad_order=3)
    assemble_excitation(mesh_exc, rwg_exc, cur_nonfinite; quad_order=3)
    false
catch
    true
end
@assert thrown_cur_nonfinite

# Hard-break API check: legacy excitation wrappers are intentionally removed.
@assert !isdefined(DiffMoM, :CurrentDistributionExcitation)
@assert !isdefined(DiffMoM, :ImportedFieldExcitation)
@assert !isdefined(DiffMoM, :make_current_distribution)

@test_throws ArgumentError make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0),
    :electric,
    Inf,
)
@test_throws ArgumentError make_loop(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    -0.1,
    1.0 + 0im,
    freq_exc,
)
@test_throws ArgumentError make_monopole(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.1,
    1.0 + 0im,
    Inf,
)
monopole_axis_reference = make_monopole(
    Vec3(0.0, 0.0, 0.0),
    Vec3(1.0, 1.0, 0.0),
    0.1,
    1.0 + 0im,
    1.0,
)
for axis_scale in (nextfloat(0.0), floatmax(Float64))
    scaled_axis_monopole = make_monopole(
        Vec3(0.0, 0.0, 0.0),
        Vec3(axis_scale, axis_scale, 0.0),
        0.1,
        1.0 + 0im,
        1.0,
    )
    @test scaled_axis_monopole.axis == monopole_axis_reference.axis
end
pattern_guard_theta = [0.0, π]
pattern_guard_phi = [0.0, π]
pattern_guard_F = ones(ComplexF64, 2, 2)
pattern_guard_storage_bytes =
    sizeof(Float64) * 4 + 2 * sizeof(ComplexF64) * 4
@test make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    pattern_guard_F, pattern_guard_F, freq_exc;
    max_storage_bytes=pattern_guard_storage_bytes) isa PatternFeedExcitation
@test_throws ArgumentError make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    pattern_guard_F, pattern_guard_F, freq_exc;
    max_storage_bytes=pattern_guard_storage_bytes - 1)
@test_throws ArgumentError make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    pattern_guard_F, pattern_guard_F, Inf)
@test_throws ArgumentError make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    pattern_guard_F, pattern_guard_F, big"1e10000")

# Invalid scalar metadata and malformed dipole models must be rejected before
# scanning/copying large angle grids or allocating analytical coefficient
# matrices.
pattern_guard_large_theta = range(0.0, stop=π, length=200_000)
pattern_guard_invalid_frequency = () -> make_pattern_feed(
    pattern_guard_large_theta, pattern_guard_phi,
    pattern_guard_F, pattern_guard_F, Inf)
try
    pattern_guard_invalid_frequency()
catch
end
@test (@allocated try
    pattern_guard_invalid_frequency()
catch
end) < 4_096
pattern_guard_object = (
    x=pattern_guard_large_theta,
    y=pattern_guard_phi,
    U=pattern_guard_F,
)
pattern_guard_invalid_object_frequency = () -> make_pattern_feed(
    pattern_guard_object, pattern_guard_object, Inf)
try
    pattern_guard_invalid_object_frequency()
catch
end
@test (@allocated try
    pattern_guard_invalid_object_frequency()
catch
end) < 4_096
pattern_guard_invalid_dipole = DipoleExcitation(
    Vec3(0.0, 0.0, 0.0),
    CVec3(NaN + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0), :electric, freq_exc)
pattern_guard_analytic_phi =
    range(0.0, stop=2π - 2π / 500, length=500)
pattern_guard_invalid_analytic = () ->
    make_analytic_dipole_pattern_feed(
        pattern_guard_invalid_dipole,
        range(0.0, stop=π, length=500), pattern_guard_analytic_phi)
try
    pattern_guard_invalid_analytic()
catch
end
@test (@allocated try
    pattern_guard_invalid_analytic()
catch
end) < 4_096
pattern_guard_F_bad = copy(pattern_guard_F)
pattern_guard_F_bad[1, 1] = Inf + 0im
@test_throws ArgumentError make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    pattern_guard_F_bad, pattern_guard_F, freq_exc)

# Direct point-field evaluators must reject invalid models/points rather than
# leaking NaN/Inf fields. Their checked public paths remain allocation-free;
# assembly and total-field loops use already-validated internal kernels.
field_guard_r = Vec3(0.4, 0.2, 0.3)
field_guard_k = Vec3(k_exc, 0.0, 0.0)
field_guard_pol = Vec3(0.0, 1.0, 0.0)
@test_throws ArgumentError plane_wave_field(
    Vec3(NaN, 0.0, 0.0), field_guard_k, 1.0, field_guard_pol)
@test_throws ArgumentError plane_wave_field(
    field_guard_r, Vec3(Inf, 0.0, 0.0), 1.0, field_guard_pol)
@test_throws ArgumentError plane_wave_field(
    field_guard_r, field_guard_k, Inf, field_guard_pol)
@test_throws ArgumentError plane_wave_field(
    field_guard_r, field_guard_k, 1.0, Vec3(0.0, 2.0, 0.0))
@test_throws ArgumentError plane_wave_field(
    field_guard_r, field_guard_k, 1.0, Vec3(1.0, 0.0, 0.0))
field_guard_extreme_phase = setprecision(BigFloat, 512) do
    argument = BigFloat(1.0e308) * BigFloat(1.0e308)
    ComplexF64(exp(Complex{BigFloat}(0, -argument)))
end
@test plane_wave_field(
    Vec3(1.0e308, 0.0, 0.0),
    Vec3(1.0e308, 0.0, 0.0),
    1.0,
    field_guard_pol,
) == CVec3(0.0 + 0im, field_guard_extreme_phase, 0.0 + 0im)

# Retain an extreme complex amplitude through the phase rotation.  Rounding
# the amplitude and phase products separately doubles this min-subnormal
# result even though the exact final field is representable.
field_guard_unit = nextfloat(0.0)
field_guard_subnormal_amplitude = complex(
    field_guard_unit, field_guard_unit)
field_guard_subnormal_point = Vec3(Float64(π) / 4, 0.0, 0.0)
field_guard_subnormal_reference = setprecision(BigFloat, 4352) do
    phase = exp(Complex{BigFloat}(
        0, -BigFloat(field_guard_subnormal_point[1])))
    CVec3(
        0.0 + 0im,
        ComplexF64(
            Complex{BigFloat}(field_guard_subnormal_amplitude) * phase),
        0.0 + 0im)
end
@test plane_wave_field(
    field_guard_subnormal_point,
    Vec3(1.0, 0.0, 0.0),
    field_guard_subnormal_amplitude,
    field_guard_pol,
) == field_guard_subnormal_reference

pattern_guard = make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    pattern_guard_F, pattern_guard_F, freq_exc)
pattern_guard_bad = PatternFeedExcitation(
    copy(pattern_guard.theta),
    copy(pattern_guard.phi),
    copy(pattern_guard.Ftheta),
    copy(pattern_guard.Fphi),
    pattern_guard.frequency,
    pattern_guard.phase_center,
    pattern_guard.convention,
)
pattern_guard_bad.Ftheta[1, 1] = NaN + 0im
@test_throws ArgumentError pattern_feed_field(
    Vec3(NaN, 0.0, 0.0), pattern_guard)
@test_throws ArgumentError pattern_feed_field(
    field_guard_r, pattern_guard_bad)

dipole_guard = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(1e-12 + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0),
    :electric,
    freq_exc,
)
loop_guard = make_loop(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.01,
    1.0 + 0im,
    freq_exc,
)
monopole_guard = make_monopole(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.05,
    1.0 + 0im,
    freq_exc,
)
@test_throws ArgumentError DiffMoM.dipole_incident_field(
    Vec3(NaN, 0.0, 0.0), dipole_guard)
@test_throws ArgumentError DiffMoM.loop_incident_field(
    Vec3(NaN, 0.0, 0.0), loop_guard)
@test_throws ArgumentError monopole_incident_field(
    Vec3(NaN, 0.0, 0.0), monopole_guard)

# Source fields are scale-covariant below a picometre; only the exact source
# point is singular.  An absolute distance cutoff used to erase these finite
# fields.
source_geometry_scale = ldexp(1.0, -44)
source_geometry_frequency = 1.0e6
source_geometry_scaled_frequency =
    ldexp(source_geometry_frequency, 44)

dipole_scale_moment = 1.0e-12
dipole_scale_reference = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(dipole_scale_moment + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0), :electric, source_geometry_frequency)
dipole_scale_tiny = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(ComplexF64(ldexp(dipole_scale_moment, -132)), 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0), :electric,
    source_geometry_scaled_frequency)
@test DiffMoM.dipole_incident_field(
    Vec3(0.0, 0.0, source_geometry_scale), dipole_scale_tiny) ==
    DiffMoM.dipole_incident_field(
        Vec3(0.0, 0.0, 1.0), dipole_scale_reference)

pattern_scale_zero = zeros(ComplexF64, 2, 2)
pattern_scale_reference = make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    pattern_guard_F, pattern_scale_zero, source_geometry_frequency)
pattern_scale_tiny = make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    source_geometry_scale .* pattern_guard_F, pattern_scale_zero,
    source_geometry_scaled_frequency)
@test pattern_feed_field(
    Vec3(source_geometry_scale, 0.0, 0.0), pattern_scale_tiny) ==
    pattern_feed_field(Vec3(1.0, 0.0, 0.0), pattern_scale_reference)

pattern_radial_minimum = make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    fill(ComplexF64(nextfloat(0.0)), 2, 2), pattern_scale_zero, 1.0)
@test pattern_feed_field(
    Vec3(nextfloat(0.0), 0.0, 0.0), pattern_radial_minimum) ==
    CVec3(cos(π / 2) + 0im, 0.0 + 0im, -1.0 + 0im)

pattern_radial_frequency = 1.0e200
pattern_radial_distance = 1.0e200
pattern_radial_huge = make_pattern_feed(
    pattern_guard_theta, pattern_guard_phi,
    pattern_guard_F, pattern_scale_zero, pattern_radial_frequency)
pattern_radial_field = pattern_feed_field(
    Vec3(pattern_radial_distance, 0.0, 0.0), pattern_radial_huge)
pattern_radial_k = DiffMoM._frequency_to_wavenumber(
    pattern_radial_frequency, DiffMoM._C0, "pattern radial test")
pattern_radial_reference_z = setprecision(BigFloat, 2304) do
    phase = exp(Complex{BigFloat}(
        0, -BigFloat(pattern_radial_k) * BigFloat(pattern_radial_distance)))
    ComplexF64(-phase / BigFloat(pattern_radial_distance))
end
@test all(isfinite, pattern_radial_field)
@test pattern_radial_field[3] == pattern_radial_reference_z

monopole_scale_reference = make_monopole(
    Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
    0.1, 1.0 + 0im, source_geometry_frequency)
monopole_scale_tiny = make_monopole(
    Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
    0.1 * source_geometry_scale, ComplexF64(source_geometry_scale),
    source_geometry_scaled_frequency)
@test monopole_incident_field(
    source_geometry_scale * Vec3(0.2, 0.0, 0.3), monopole_scale_tiny) ==
    monopole_incident_field(
        Vec3(0.2, 0.0, 0.3), monopole_scale_reference)

# A finite kR near 1e20 has insufficient Float64 phase resolution even though
# the final integrated field is representable. The exact path retains the
# stored Float64 source geometry through the complete Simpson accumulation.
monopole_radial_k = 1.1
monopole_radial_frequency =
    monopole_radial_k * DiffMoM._C0 / (2π)
monopole_radial = make_monopole(
    Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
    0.1, 1.0 + 0im, monopole_radial_frequency;
    include_image=false)
monopole_radial_point = Vec3(1.0e20, 0.0, 0.0)
monopole_radial_intervals = 64
monopole_radial_reference = DiffMoM._monopole_incident_field_exact(
    monopole_radial_point,
    monopole_radial,
    monopole_radial_k,
    monopole_radial_intervals,
    DiffMoM._MAX_MONOPOLE_EXACT_WORK,
)
@test monopole_incident_field(
    monopole_radial_point, monopole_radial) == monopole_radial_reference
@test isapprox(
    monopole_radial_reference[3],
    ComplexF64(-1.6659000403245365e-23, 2.5213022494156855e-23);
    rtol=2e-14, atol=0.0)
@test_throws ArgumentError monopole_incident_field(
    monopole_radial_point, monopole_radial; max_exact_work=1_000)

# The Simpson workload is bounded before Float64-to-Int conversion. The helper
# accepts the exact configured boundary without allocating and both public
# field paths reject an electrically impossible workload with a domain error.
@test DiffMoM._monopole_simpson_interval_count(
    999.99, 2π, 1.0, 64, "test monopole") ==
    DiffMoM._MAX_MONOPOLE_SIMPSON_INTERVALS
DiffMoM._monopole_simpson_interval_count(
    999.99, 2π, 1.0, 64, "test monopole")
@test (@allocated DiffMoM._monopole_simpson_interval_count(
    999.99, 2π, 1.0, 64, "test monopole")) == 0
@test_throws ArgumentError DiffMoM._monopole_simpson_interval_count(
    1000.01, 2π, 1.0, 64, "test monopole")

resource_frequency = 1.0e100
resource_monopole = make_monopole(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    1.0e220,
    1.0 + 0im,
    resource_frequency,
)
resource_k = 2π * resource_frequency / 299792458.0
for resource_call in (
    () -> monopole_incident_field(
        Vec3(1.0, 0.0, 1.0), resource_monopole),
    () -> incident_farfield(
        resource_monopole, Vec3(1.0, 0.0, 0.0), resource_k),
)
    resource_error = try
        resource_call()
        nothing
    catch err
        err
    end
    @test resource_error isa ArgumentError
    @test occursin(
        "more than $(DiffMoM._MAX_MONOPOLE_SIMPSON_INTERVALS) Simpson intervals",
        sprint(showerror, resource_error),
    )
end

source_scaling_k = 1.0e200
source_scaling_frequency =
    (source_scaling_k / (2π)) * 299792458.0
source_scaling_dipole = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(1.0e-300 + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0),
    :electric,
    source_scaling_frequency,
)
source_scaling_dipole_far = incident_farfield(
    source_scaling_dipole, Vec3(0.0, 0.0, 1.0), source_scaling_k)
source_scaling_dipole_reference = setprecision(BigFloat, 512) do
    scale = BigFloat(source_scaling_k)^2 /
            (4 * BigFloat(pi) * BigFloat(DiffMoM._EPS0)) *
            BigFloat(1.0e-300)
    CVec3(ComplexF64(scale), 0.0 + 0im, 0.0 + 0im)
end
@test source_scaling_dipole_far == source_scaling_dipole_reference
@test (@allocated incident_farfield(
    source_scaling_dipole,
    Vec3(0.0, 0.0, 1.0),
    source_scaling_k,
)) <= 20_000

# Preserve the transverse projection when the dipole moment is almost
# parallel to an ordinary observation direction.  Rounded dot/subtract
# evaluation used to return components with the wrong sign and four times the
# correct magnitude.
dipole_projection_direction = Vec3(1.0, 1.0, 0.0)
dipole_projection_moment =
    CVec3(1.0 + 0im, prevfloat(1.0) + 0im, 0.0 + 0im)
dipole_projection = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    dipole_projection_moment,
    Vec3(1.0, 0.0, 0.0),
    :electric,
    freq_exc,
)
dipole_projection_far = incident_farfield(
    dipole_projection, dipole_projection_direction, k_exc)
dipole_projection_reference = setprecision(BigFloat, 512) do
    direction = SVector{3,BigFloat}(
        BigFloat.(dipole_projection_direction))
    direction /= sqrt(sum(abs2, direction))
    moment = SVector{3,Complex{BigFloat}}(
        Complex{BigFloat}.(dipole_projection_moment))
    projection = cross(cross(direction, moment), direction)
    scale = BigFloat(k_exc)^2 /
            (4 * BigFloat(pi) * BigFloat(DiffMoM._EPS0))
    CVec3(ComplexF64.(scale * projection))
end
@test dipole_projection_far ≈ dipole_projection_reference rtol=4eps(Float64)
@test (@allocated incident_farfield(
    dipole_projection, dipole_projection_direction, k_exc)) == 0

# A tiny off-axis direction component can produce a projection that underflows
# before multiplication by k² even though the final field is representable.
tiny_projection_component = exp2(-400.0)
tiny_projection_k = exp2(100.0)
tiny_projection_frequency =
    tiny_projection_k * DiffMoM._C0 / (2π)
tiny_projection_direction =
    Vec3(1.0, tiny_projection_component, 0.0)
tiny_projection_dipole = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(1.0 + 0im, 0.0 + 0im, 0.0 + 0im),
    Vec3(1.0, 0.0, 0.0),
    :electric,
    tiny_projection_frequency,
)
tiny_projection_far = incident_farfield(
    tiny_projection_dipole, tiny_projection_direction, tiny_projection_k)
tiny_projection_reference = setprecision(BigFloat, 2048) do
    direction = SVector{3,BigFloat}(
        BigFloat.(tiny_projection_direction))
    direction /= sqrt(sum(abs2, direction))
    moment = SVector{3,Complex{BigFloat}}(
        Complex{BigFloat}.(tiny_projection_dipole.moment))
    projection = cross(cross(direction, moment), direction)
    scale = BigFloat(tiny_projection_k)^2 /
            (4 * BigFloat(pi) * BigFloat(DiffMoM._EPS0))
    CVec3(ComplexF64.(scale * projection))
end
@test tiny_projection_far == tiny_projection_reference
@test all(isfinite, tiny_projection_far)
@test (@allocated incident_farfield(
    tiny_projection_dipole,
    tiny_projection_direction,
    tiny_projection_k,
)) <= 25_000

# Normalize a non-axial observation direction before forming a large source
# translation phase.  Rounding each normalized component first changes this
# finite phase by more than half a radian.
directional_phase_direction = Vec3(1.0, 1.0, 0.0)
directional_phase_position = Vec3(1.0e16, 0.0, 0.0)
directional_phase_k = 1.0
directional_phase_frequency =
    directional_phase_k * DiffMoM._C0 / (2π)
directional_phase_dipole = make_dipole(
    directional_phase_position,
    CVec3(0.0 + 0im, 0.0 + 0im, 1.0 + 0im),
    Vec3(0.0, 0.0, 1.0),
    :electric,
    directional_phase_frequency,
)
directional_phase_far = incident_farfield(
    directional_phase_dipole,
    directional_phase_direction,
    directional_phase_k,
)
directional_phase_reference, directional_phase_field_reference =
        setprecision(BigFloat, 2048) do
    direction = SVector{3,BigFloat}(
        BigFloat.(directional_phase_direction))
    direction /= sqrt(sum(abs2, direction))
    moment = SVector{3,Complex{BigFloat}}(
        Complex{BigFloat}.(directional_phase_dipole.moment))
    projection = cross(cross(direction, moment), direction)
    phase_argument = BigFloat(directional_phase_k) * sum(
        direction[component] *
        BigFloat(directional_phase_position[component])
        for component in 1:3)
    scale = BigFloat(directional_phase_k)^2 /
            (4 * BigFloat(pi) * BigFloat(DiffMoM._EPS0))
    phase = exp(Complex{BigFloat}(0, phase_argument))
    ComplexF64(phase), CVec3(ComplexF64.(scale * projection * phase))
end
directional_phase_normalized = DiffMoM._validated_farfield_direction(
    directional_phase_direction)
@test DiffMoM._source_directional_phase(
    directional_phase_k,
    directional_phase_direction,
    directional_phase_normalized,
    directional_phase_position,
    1.0,
    "directional phase regression",
) == directional_phase_reference
@test directional_phase_far ≈
      directional_phase_field_reference rtol=2eps(Float64)
@test (@allocated incident_farfield(
    directional_phase_dipole,
    directional_phase_direction,
    directional_phase_k,
)) <= 20_000

# Form phase arguments without overflowing Float64 and retain small terms when
# their exponent is far below another term in the same dot product.
source_phase_first = Vec3(1.0e200, 1.0, 0.0)
source_phase_second = Vec3(1.0e200, 1.0, 0.0)
source_phase_reference = setprecision(BigFloat, 8192) do
    argument = sum(
        BigFloat(source_phase_first[i]) * BigFloat(source_phase_second[i])
        for i in 1:3)
    ComplexF64(exp(Complex{BigFloat}(0, argument)))
end
@test DiffMoM._source_phase(
    1.0, source_phase_first, source_phase_second, 1.0,
    "source phase regression") == source_phase_reference
@test plane_wave_field(
    source_phase_first,
    source_phase_second,
    1.0,
    Vec3(0.0, 1.0, 0.0),
) == CVec3(0.0 + 0im, conj(source_phase_reference), 0.0 + 0im)
@test (@allocated DiffMoM._source_phase(
    1.0, source_phase_first, source_phase_second, 1.0,
    "source phase regression")) <= 20_000

# A large but finite radial distance needs exact-Float geometry and phase
# reduction.  The ordinary hypot/product path loses an observable fraction of
# the magnetic-dipole field well before either endpoint is near Float64 range.
dipole_radial_frequency = DiffMoM._C0 / (2π)
dipole_radial = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, 1.0 + 0im),
    Vec3(0.0, 0.0, 1.0),
    :magnetic,
    dipole_radial_frequency,
)
dipole_radial_point = Vec3(1.0e12, 1.0e12, 0.0)
dipole_radial_k = DiffMoM._frequency_to_wavenumber(
    dipole_radial.frequency,
    inv(sqrt(DiffMoM._MU0 * DiffMoM._EPS0)),
    "dipole radial reference",
)
dipole_radial_reference = setprecision(BigFloat, 2048) do
    displacement = SVector{3,BigFloat}(
        BigFloat(dipole_radial_point[1]),
        BigFloat(dipole_radial_point[2]),
        BigFloat(dipole_radial_point[3]),
    )
    distance = sqrt(sum(abs2, displacement))
    direction = displacement / distance
    moment = SVector{3,Complex{BigFloat}}(
        Complex{BigFloat}.(dipole_radial.moment))
    wavenumber = BigFloat(dipole_radial_k)
    phase = exp(Complex{BigFloat}(0, -wavenumber * distance))
    value = (BigFloat(DiffMoM._ETA0) / (4 * BigFloat(pi))) *
            (wavenumber^2 / distance -
             Complex{BigFloat}(0, 1) * wavenumber / distance^2) *
            phase * cross(moment, direction)
    CVec3(ComplexF64.(value))
end
@test DiffMoM.dipole_incident_field(
    dipole_radial_point, dipole_radial) == dipole_radial_reference

translated_scaling_dipole = make_dipole(
    Vec3(1.0e200, 0.0, 0.0),
    CVec3(0.0 + 0im, 1.0e-300 + 0im, 0.0 + 0im),
    Vec3(0.0, 1.0, 0.0),
    :electric,
    source_scaling_frequency,
)
translated_scaling_far = incident_farfield(
    translated_scaling_dipole, Vec3(1.0, 0.0, 0.0), source_scaling_k)
translated_scaling_reference = setprecision(BigFloat, 8192) do
    amplitude = BigFloat(source_scaling_k)^2 /
                (4 * BigFloat(pi) * BigFloat(DiffMoM._EPS0)) *
                BigFloat(1.0e-300)
    phase = exp(Complex{BigFloat}(
        0, BigFloat(source_scaling_k) * BigFloat(1.0e200)))
    CVec3(
        0.0 + 0im,
        ComplexF64(amplitude) * ComplexF64(phase),
        0.0 + 0im,
    )
end
@test translated_scaling_far == translated_scaling_reference
@test all(isfinite, translated_scaling_far)
@test (@allocated incident_farfield(
    translated_scaling_dipole,
    Vec3(1.0, 0.0, 0.0),
    source_scaling_k,
)) <= 20_000

# The direct angular-frequency product overflows at this finite frequency, but
# the represented wavenumber and far field remain finite.
high_source_frequency = 1.0e308
high_source_k = DiffMoM._frequency_to_wavenumber(
    high_source_frequency, DiffMoM._C0, "DipoleExcitation")
high_frequency_dipole = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 1.0e-320 + 0im, 0.0 + 0im),
    Vec3(0.0, 1.0, 0.0),
    :electric,
    high_source_frequency,
)
high_frequency_far = incident_farfield(
    high_frequency_dipole, Vec3(1.0, 0.0, 0.0), high_source_k)
high_frequency_reference = setprecision(BigFloat, 512) do
    amplitude = BigFloat(high_source_k)^2 /
                (4 * BigFloat(pi) * BigFloat(DiffMoM._EPS0)) *
                BigFloat(1.0e-320)
    CVec3(0.0 + 0im, ComplexF64(amplitude), 0.0 + 0im)
end
@test high_frequency_far == high_frequency_reference
@test all(isfinite, high_frequency_far)

let frequency = 1.0e9
    DiffMoM._frequency_to_wavenumber(
        frequency, DiffMoM._C0, "allocation regression")
    @test (@allocated DiffMoM._frequency_to_wavenumber(
        frequency, DiffMoM._C0, "allocation regression")) == 0
end

source_scaling_pattern = make_analytic_dipole_pattern_feed(
    source_scaling_dipole, [0.0, π], [0.0, π])
@test source_scaling_pattern.Ftheta[1, 1] ==
      source_scaling_dipole_reference[1]
@test all(isfinite, source_scaling_pattern.Ftheta)
@test all(isfinite, source_scaling_pattern.Fphi)

source_scaling_loop = make_loop(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    1.0e200,
    1.0e-300 + 0im,
    freq_exc,
)
source_scaling_loop_far = incident_farfield(
    source_scaling_loop, Vec3(1.0, 0.0, 0.0), k_exc)
source_scaling_loop_reference = setprecision(BigFloat, 512) do
    moment = BigFloat(1.0e-300) * BigFloat(pi) *
             BigFloat(1.0e200)^2
    scale = BigFloat(DiffMoM._ETA0) * BigFloat(k_exc)^2 /
            (4 * BigFloat(pi)) * moment
    CVec3(0.0 + 0im, ComplexF64(scale), 0.0 + 0im)
end
@test source_scaling_loop_far == source_scaling_loop_reference
@test all(isfinite, DiffMoM.loop_incident_field(
    Vec3(1.0, 0.0, 0.0), source_scaling_loop))

# Keep the analytically finite monopole pattern when the legacy current-first
# product overflows, and when its dimensional Simpson integral underflows.
for (monopole_k, monopole_height, monopole_amplitude) in (
    (1.0e-108, 1.0, 1.0e308),
    (1.0, 1.0e-200, 1.0e100),
)
    monopole_frequency =
        (monopole_k / (2π)) * DiffMoM._C0
    scaled_monopole = make_monopole(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        monopole_height,
        ComplexF64(monopole_amplitude),
        monopole_frequency,
    )
    scaled_monopole_far = incident_farfield(
        scaled_monopole, Vec3(1.0, 0.0, 0.0), monopole_k)
    scaled_monopole_reference = setprecision(BigFloat, 2304) do
        electrical_height =
            BigFloat(monopole_k) * BigFloat(monopole_height)
        pattern = 2 * sin(electrical_height / 2)^2
        CVec3(
            0.0 + 0im,
            0.0 + 0im,
            -ComplexF64(BigFloat(monopole_amplitude) * pattern),
        )
    end
    @test scaled_monopole_far == scaled_monopole_reference

    scaled_half_monopole = make_monopole(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        monopole_height,
        ComplexF64(monopole_amplitude),
        monopole_frequency;
        include_image=false,
    )
    @test incident_farfield(
        scaled_half_monopole, Vec3(1.0, 0.0, 0.0), monopole_k) ==
        scaled_monopole_reference / 2
end

# A near-axis monopole pattern is small but not zero.  Cancel the pattern and
# polarization sin(theta) factors analytically so both ordinary and extreme
# amplitudes retain this representable field.
monopole_far_axis_k = 1.0
monopole_far_axis_height = 0.1
monopole_far_axis_frequency =
    monopole_far_axis_k * DiffMoM._C0 / (2π)
monopole_far_axis_direction = Vec3(1.0e-13, 0.0, 1.0)
for monopole_far_axis_amplitude in (1.0, ldexp(1.0, 200))
    monopole_far_axis_source = make_monopole(
        Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
        monopole_far_axis_height, monopole_far_axis_amplitude,
        monopole_far_axis_frequency)
    monopole_far_axis_result = incident_farfield(
        monopole_far_axis_source, monopole_far_axis_direction,
        monopole_far_axis_k)
    monopole_far_axis_reference = setprecision(BigFloat, 512) do
        direction = SVector{3,BigFloat}(
            BigFloat.(monopole_far_axis_direction))
        direction /= sqrt(sum(abs2, direction))
        axis = SVector{3,BigFloat}(0, 0, 1)
        cosine = dot(direction, axis)
        electrical_height = BigFloat(monopole_far_axis_k) *
                            BigFloat(monopole_far_axis_height)
        first_argument = electrical_height * (1 + cosine) / 2
        second_argument = electrical_height * (1 - cosine) / 2
        first_sinc = sin(first_argument) / first_argument
        second_sinc = iszero(second_argument) ? one(BigFloat) :
                      sin(second_argument) / second_argument
        angular_factor = BigFloat(monopole_far_axis_amplitude) *
                         electrical_height^2 *
                         first_sinc * second_sinc / 2
        CVec3(angular_factor * (cosine * direction - axis))
    end
    @test !all(iszero, monopole_far_axis_result)
    @test monopole_far_axis_result ≈
          monopole_far_axis_reference rtol=8eps(Float64) atol=0.0
end

multi_farfield_theta = [0.0, π]
multi_farfield_phi = [0.0, π]
multi_farfield_maximum = floatmax(Float64)
multi_farfield_children = [
    make_pattern_feed(
        multi_farfield_theta,
        multi_farfield_phi,
        fill(ComplexF64(value), 2, 2),
        zeros(ComplexF64, 2, 2),
        freq_exc,
    ) for value in (
        multi_farfield_maximum,
        multi_farfield_maximum,
        -multi_farfield_maximum,
    )
]
multi_farfield = make_multi_excitation(multi_farfield_children)
@test incident_farfield(
    multi_farfield, Vec3(0.0, 0.0, 1.0), k_exc) ==
    CVec3(ComplexF64(multi_farfield_maximum), 0.0 + 0im, 0.0 + 0im)
multi_incident_point = Vec3(0.0, 0.0, 1.0)
@test DiffMoM._incident_electric_field(
    multi_farfield, multi_incident_point, k_exc) ==
    DiffMoM._incident_electric_field(
        multi_farfield_children[1], multi_incident_point, k_exc)

large_phi_origin = 1.0e16
large_phi_grid = [
    large_phi_origin,
    nextfloat(large_phi_origin),
    nextfloat(nextfloat(large_phi_origin)),
]
large_phi_Ftheta = zeros(ComplexF64, 2, 3)
large_phi_Ftheta[:, 1] .= 1.0 + 0im
large_phi_pattern = make_pattern_feed(
    [0.0, π],
    large_phi_grid,
    large_phi_Ftheta,
    zeros(ComplexF64, 2, 3),
    freq_exc,
)
large_phi_fraction = setprecision(BigFloat, 512) do
    origin = BigFloat(large_phi_origin)
    period = 2 * BigFloat(pi)
    wrapped = mod(-origin, period) + origin
    Float64(
        (wrapped - BigFloat(large_phi_grid[end])) /
        (origin + period - BigFloat(large_phi_grid[end])))
end
@test DiffMoM._bracket_periodic_phi(large_phi_grid, 0.0) ==
      (3, 1, large_phi_fraction)
@test incident_farfield(
    large_phi_pattern, Vec3(0.0, 0.0, 1.0), k_exc) ==
    CVec3(ComplexF64(large_phi_fraction), 0.0 + 0im, 0.0 + 0im)

multi_rhs_value = ComplexF64(1.0e308)
multi_rhs_children = [
    make_delta_gap(1, value, 1.0) for value in
    (multi_rhs_value, multi_rhs_value, -multi_rhs_value)
]
multi_rhs = make_multi_excitation(multi_rhs_children)
multi_rhs_mesh = make_rect_plate(1.0, 1.0, 1, 1)
multi_rhs_rwg = build_rwg(multi_rhs_mesh)
@test multi_rhs_rwg.nedges == 1
@test assemble_excitation(
    multi_rhs_mesh, multi_rhs_rwg, multi_rhs) == [multi_rhs_value]
@test_throws ArgumentError assemble_excitation(
    multi_rhs_mesh, multi_rhs_rwg, multi_rhs; max_exact_bytes=1)

incident_farfield(dipole_guard, Vec3(0.0, 0.0, 1.0), k_exc)
DiffMoM._loop_equivalent_moment(loop_guard)
let dipole = dipole_guard,
    direction = Vec3(0.0, 0.0, 1.0),
    wavenumber = k_exc,
    loop = loop_guard
    @test (@allocated incident_farfield(
        dipole, direction, wavenumber)) == 0
    @test (@allocated DiffMoM._loop_equivalent_moment(loop)) == 0
end

plane_wave_field(field_guard_r, field_guard_k, 1.0, field_guard_pol)
pattern_feed_field(field_guard_r, pattern_guard)
DiffMoM.dipole_incident_field(field_guard_r, dipole_guard)
DiffMoM.loop_incident_field(field_guard_r, loop_guard)
monopole_incident_field(field_guard_r, monopole_guard)
@test (@allocated plane_wave_field(
    field_guard_r, field_guard_k, 1.0, field_guard_pol)) == 0
@test (@allocated pattern_feed_field(field_guard_r, pattern_guard)) == 0
@test (@allocated DiffMoM.dipole_incident_field(
    field_guard_r, dipole_guard)) == 0
@test (@allocated DiffMoM.loop_incident_field(
    field_guard_r, loop_guard)) == 0
@test (@allocated monopole_incident_field(
    field_guard_r, monopole_guard)) == 0

# Near-longitudinal electric dipoles retain a small radiating component.  Form
# its double cross without losing the product roundoff at high electrical
# distance, while the ordinary path remains allocation-free.
dipole_projection_r = Vec3(3.0, 4.0, 0.0)
dipole_projection_direction = dipole_projection_r / 5.0
dipole_projection_transverse = Vec3(
    -dipole_projection_direction[2], dipole_projection_direction[1], 0.0)
dipole_projection_moment = CVec3(complex.(
    dipole_projection_direction +
    1.0e-8 * dipole_projection_transverse))
dipole_projection_c0 = inv(sqrt(DiffMoM._EPS0 * DiffMoM._MU0))
dipole_projection_frequency =
    ldexp(1.0, 28) * dipole_projection_c0 / (2π)
dipole_projection_source = make_dipole(
    Vec3(0.0, 0.0, 0.0), dipole_projection_moment,
    dipole_projection_direction, :electric, dipole_projection_frequency)
dipole_projection_result = DiffMoM.dipole_incident_field(
    dipole_projection_r, dipole_projection_source)
dipole_projection_k = DiffMoM._frequency_to_wavenumber(
    dipole_projection_frequency, dipole_projection_c0,
    "dipole projection regression")
dipole_projection_reference = setprecision(BigFloat, 512) do
    displacement = SVector{3,BigFloat}(BigFloat.(dipole_projection_r))
    distance = sqrt(sum(abs2, displacement))
    direction = displacement / distance
    moment = SVector{3,Complex{BigFloat}}(
        Complex{BigFloat}.(dipole_projection_moment))
    wavenumber = BigFloat(dipole_projection_k)
    phase = exp(Complex{BigFloat}(0, -wavenumber * distance))
    transverse = cross(cross(direction, moment), direction) * wavenumber^2
    near = (3 * direction * dot(direction, moment) - moment) *
           (inv(distance^2) +
            Complex{BigFloat}(0, 1) * wavenumber / distance)
    CVec3((transverse + near) * phase /
          (4 * BigFloat(π) * BigFloat(DiffMoM._EPS0) * distance))
end
@test dipole_projection_result ≈
      dipole_projection_reference rtol=4eps(Float64) atol=0.0
dipole_cross_source = make_dipole(
    Vec3(0.0, 0.0, 0.0), dipole_projection_moment,
    dipole_projection_direction, :magnetic, dipole_projection_frequency)
dipole_cross_result = DiffMoM.dipole_incident_field(
    dipole_projection_r, dipole_cross_source)
dipole_cross_reference = setprecision(BigFloat, 512) do
    displacement = SVector{3,BigFloat}(BigFloat.(dipole_projection_r))
    distance = sqrt(sum(abs2, displacement))
    direction = displacement / distance
    moment = SVector{3,Complex{BigFloat}}(
        Complex{BigFloat}.(dipole_projection_moment))
    wavenumber = BigFloat(dipole_projection_k)
    phase = exp(Complex{BigFloat}(0, -wavenumber * distance))
    CVec3((BigFloat(DiffMoM._ETA0) / (4 * BigFloat(π))) *
          (wavenumber^2 / distance -
           Complex{BigFloat}(0, 1) * wavenumber / distance^2) *
          phase * cross(moment, direction))
end
@test dipole_cross_result ≈
      dipole_cross_reference rtol=4eps(Float64) atol=0.0

# The Hertzian transverse factor and basis contain reciprocal sin(theta)
# factors.  Cancel them algebraically so a small but radiatively amplified
# near-axis field is not discarded by an angular cutoff.
monopole_near_axis_k = ldexp(1.0, 28)
monopole_near_axis_frequency =
    monopole_near_axis_k * DiffMoM._C0 / (2π)
monopole_near_axis_source = make_monopole(
    Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
    1.0e-9, 1.0 + 0im, monopole_near_axis_frequency;
    include_image=true)
monopole_near_axis_point = Vec3(5.0e-13, 0.0, 5.0)
monopole_near_axis_result = monopole_incident_field(
    monopole_near_axis_point, monopole_near_axis_source)
monopole_near_axis_intervals = DiffMoM._monopole_simpson_interval_count(
    monopole_near_axis_source.height, monopole_near_axis_k,
    2.0, 128, "near-axis monopole regression")
monopole_near_axis_reference = DiffMoM._monopole_incident_field_exact(
    monopole_near_axis_point, monopole_near_axis_source,
    monopole_near_axis_k, monopole_near_axis_intervals, 2_000_000)
@test abs(monopole_near_axis_reference[1]) >
      1.0e-5 * abs(monopole_near_axis_reference[3])
@test monopole_near_axis_result ≈
      monopole_near_axis_reference rtol=1.0e-9 atol=0.0

m_mag = CVec3(0.0 + 0im, 0.0 + 0im, 1e-4 + 0im) # A·m²
dip_mag = make_dipole(Vec3(0.0, 0.0, 0.0), m_mag, Vec3(0.0, 0.0, 1.0), :magnetic, freq_exc)
Rfar = 5.0
E_mag_num = DiffMoM.dipole_incident_field(Vec3(Rfar, 0.0, 0.0), dip_mag)
# Far-field E of a magnetic dipole: E = -ikη₀(∇G×m) → radiating term has a REAL
# coefficient (dual to the electric dipole). Observed on +x with m∥ẑ, the
# φ̂-component is +η₀k²m_z e^{-ikR}/(4πR) (no factor i).
E_mag_ref = eta0 * k_exc^2 * m_mag[3] * exp(-1im * k_exc * Rfar) / (4π * Rfar)
rel_mag = abs(E_mag_num[2] - E_mag_ref) / max(abs(E_mag_ref), 1e-30)
println("  Magnetic-dipole far-field rel. error: $rel_mag")
@assert rel_mag < 0.03

# Electric dipole: broadside far-field amplitude and axial quasi-static behavior
p_e = CVec3(0.0 + 0im, 0.0 + 0im, 1e-12 + 0im)
dip_e = make_dipole(Vec3(0.0, 0.0, 0.0), p_e, Vec3(0.0, 0.0, 1.0), :electric, freq_exc)
Rfar_e = 8.0
E_e_num = DiffMoM.dipole_incident_field(Vec3(Rfar_e, 0.0, 0.0), dip_e)
eps0_exc = 8.854187817e-12
E_e_ref = p_e[3] * k_exc^2 * exp(-1im * k_exc * Rfar_e) / (4π * eps0_exc * Rfar_e)
rel_e = abs(E_e_num[3] - E_e_ref) / max(abs(E_e_ref), 1e-30)
println("  Electric-dipole broadside far-field rel. error: $rel_e")
@assert rel_e < 0.03

dip_e_low = make_dipole(Vec3(0.0, 0.0, 0.0), p_e, Vec3(0.0, 0.0, 1.0), :electric, 1.0e6)
R_quasi = 0.5
E_e_axial = DiffMoM.dipole_incident_field(Vec3(0.0, 0.0, R_quasi), dip_e_low)
E_e_static = 2 * p_e[3] / (4π * eps0_exc * R_quasi^3)
rel_e_quasi = abs(real(E_e_axial[3]) - real(E_e_static)) / max(abs(real(E_e_static)), 1e-30)
println("  Electric-dipole axial quasi-static rel. error: $rel_e_quasi")
@assert rel_e_quasi < 0.05

# Loop field must match equivalent magnetic dipole field and RHS exactly
loop_eq = make_loop(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), 0.01, 2.0 + 0im, freq_exc)
m_eq = (2.0 + 0im) * π * (0.01)^2
dip_eq = make_dipole(Vec3(0.0, 0.0, 0.0), CVec3(0.0 + 0im, 0.0 + 0im, m_eq), Vec3(0.0, 0.0, 1.0), :magnetic, freq_exc)
E_loop_eq = DiffMoM.loop_incident_field(Vec3(Rfar, 0.0, 0.0), loop_eq)
E_dip_eq = DiffMoM.dipole_incident_field(Vec3(Rfar, 0.0, 0.0), dip_eq)
rel_loop_field = norm(E_loop_eq - E_dip_eq) / max(norm(E_dip_eq), 1e-30)
println("  Loop vs equivalent magnetic-dipole field rel. diff: $rel_loop_field")
@assert rel_loop_field < 1e-13

v_loop_eq = assemble_excitation(mesh_exc, rwg_exc, loop_eq; quad_order=3)
v_dip_eq = assemble_excitation(mesh_exc, rwg_exc, dip_eq; quad_order=3)
rel_loop_rhs = norm(v_loop_eq - v_dip_eq) / max(norm(v_dip_eq), 1e-30)
println("  Loop vs equivalent magnetic-dipole RHS rel. diff: $rel_loop_rhs")
@assert rel_loop_rhs < 1e-13

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 15: Dipole/loop far-field pattern gate
# ─────────────────────────────────────────────────
println("\n── Test 15: Dipole/loop far-field pattern gate ──")

freq_pat = 1.0e9
k_pat = 2π * freq_pat / 299792458.0
lambda_pat = 2π / k_pat
Rfar_pat = 50 * lambda_pat
theta_deg_pat = collect(0.0:1.0:180.0)
theta_pat = deg2rad.(theta_deg_pat)

dip_pat = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, 1e-12 + 0im),   # electric dipole (C·m)
    Vec3(0.0, 0.0, 1.0),
    :electric,
    freq_pat
)
loop_pat = make_loop(
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    0.01,                                        # 1 cm radius
    1.0 + 0im,
    freq_pat
)

P_dip_num = zeros(Float64, length(theta_pat))
P_loop_num = zeros(Float64, length(theta_pat))
P_ana = sin.(theta_pat) .^ 2
E_dip_theta_cmp = zeros(ComplexF64, length(theta_pat))
E_loop_phi_cmp = zeros(ComplexF64, length(theta_pat))

for i in eachindex(theta_pat)
    th = theta_pat[i]
    rhat = Vec3(sin(th), 0.0, cos(th))          # φ = 0 cut
    r = Rfar_pat * rhat
    E_d = DiffMoM.dipole_incident_field(r, dip_pat)
    E_l = DiffMoM.loop_incident_field(r, loop_pat)
    e_theta = Vec3(cos(th), 0.0, -sin(th))
    e_phi = Vec3(0.0, 1.0, 0.0)
    E_dip_theta_cmp[i] = dot(E_d, e_theta)
    E_loop_phi_cmp[i] = dot(E_l, e_phi)
    P_dip_num[i] = norm(E_d)^2
    P_loop_num[i] = norm(E_l)^2
end

P_dip_num ./= maximum(P_dip_num)
P_loop_num ./= maximum(P_loop_num)
P_ana ./= maximum(P_ana)

err_dip = P_dip_num .- P_ana
err_loop = P_loop_num .- P_ana

rmse_dip = sqrt(sum(abs2, err_dip) / length(err_dip))
rmse_loop = sqrt(sum(abs2, err_loop) / length(err_loop))
maxabs_dip = maximum(abs.(err_dip))
maxabs_loop = maximum(abs.(err_loop))

null_max_dip = max(P_dip_num[1], P_dip_num[end])
null_max_loop = max(P_loop_num[1], P_loop_num[end])

# Polarization-resolved check over a coarse (θ,φ) grid.
phi_pat = deg2rad.(collect(0.0:10.0:350.0))
theta_pol_pat = deg2rad.(collect(1.0:2.0:179.0))
crossfrac_dip = Float64[]
crossfrac_loop = Float64[]
for th in theta_pol_pat, ph in phi_pat
    rhat = Vec3(sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
    r = Rfar_pat * rhat
    e_theta = Vec3(cos(th) * cos(ph), cos(th) * sin(ph), -sin(th))
    e_phi = Vec3(-sin(ph), cos(ph), 0.0)
    E_d = DiffMoM.dipole_incident_field(r, dip_pat)
    E_l = DiffMoM.loop_incident_field(r, loop_pat)
    E_d_theta = dot(E_d, e_theta)
    E_d_phi = dot(E_d, e_phi)
    E_l_theta = dot(E_l, e_theta)
    E_l_phi = dot(E_l, e_phi)
    P_d = abs2(E_d_theta) + abs2(E_d_phi)
    P_l = abs2(E_l_theta) + abs2(E_l_phi)
    push!(crossfrac_dip, abs(E_d_phi) / sqrt(P_d))
    push!(crossfrac_loop, abs(E_l_theta) / sqrt(P_l))
end
max_crossfrac_dip = maximum(crossfrac_dip)
max_crossfrac_loop = maximum(crossfrac_loop)

# Co-pol phase consistency on φ=0 cut. Both Eθ(electric dipole) and Eϕ(loop /
# magnetic dipole) have REAL far-field coefficients (× e^{-ikr}), being duals of
# each other, so their ratio is real: Δϕ ∈ {0°, 180°}. Verify each sample lands
# near 0° or 180° and that the deviation is consistent across the cut.
wrap_to_pi(x) = atan(sin(x), cos(x))
phase_dev_deg = Float64[]   # signed distance to the nearest of {0°, 180°}
amp_floor_phase = 1e-12 * max(maximum(abs.(E_dip_theta_cmp)), maximum(abs.(E_loop_phi_cmp)))
for i in eachindex(theta_pat)
    if abs(E_dip_theta_cmp[i]) > amp_floor_phase && abs(E_loop_phi_cmp[i]) > amp_floor_phase
        Δϕ = angle(E_loop_phi_cmp[i] / E_dip_theta_cmp[i])
        d0 = wrap_to_pi(Δϕ)
        dπ = wrap_to_pi(Δϕ - π)
        dev = abs(d0) <= abs(dπ) ? d0 : dπ
        push!(phase_dev_deg, rad2deg(dev))
    end
end
phase_std_deg = std(phase_dev_deg)
phase_max_err_copol_deg = maximum(abs.(phase_dev_deg))

println("  Dipole pattern RMSE:      $rmse_dip")
println("  Dipole pattern max |err|: $maxabs_dip")
println("  Loop pattern RMSE:        $rmse_loop")
println("  Loop pattern max |err|:   $maxabs_loop")
println("  Dipole null max:          $null_max_dip")
println("  Loop null max:            $null_max_loop")
println("  Dipole max cross-pol frac: $max_crossfrac_dip")
println("  Loop max cross-pol frac:   $max_crossfrac_loop")
println("  Phase dev std (deg):           $phase_std_deg")
println("  Phase max err to {0,180}° (deg): $phase_max_err_copol_deg")

# CI thresholds (pattern-shape gate)
@assert rmse_dip < 1e-4 "Dipole pattern gate failed: RMSE=$rmse_dip"
@assert maxabs_dip < 2e-4 "Dipole pattern gate failed: max |err|=$maxabs_dip"
@assert rmse_loop < 1e-10 "Loop pattern gate failed: RMSE=$rmse_loop"
@assert maxabs_loop < 1e-9 "Loop pattern gate failed: max |err|=$maxabs_loop"
@assert null_max_dip < 1e-3 "Dipole pattern gate failed: null level=$null_max_dip"
@assert null_max_loop < 1e-10 "Loop pattern gate failed: null level=$null_max_loop"
@assert max_crossfrac_dip < 1e-10 "Dipole polarization gate failed: max cross-pol frac=$max_crossfrac_dip"
@assert max_crossfrac_loop < 1e-10 "Loop polarization gate failed: max cross-pol frac=$max_crossfrac_loop"
@assert phase_max_err_copol_deg < 1.0 "Dipole/loop phase gate failed: max err to {0,180}° = $phase_max_err_copol_deg deg"
@assert phase_std_deg < 0.1 "Dipole/loop phase gate failed: phase dev std = $phase_std_deg deg"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 16: Pattern-feed excitation gate
# ─────────────────────────────────────────────────
println("\n── Test 16: Pattern-feed excitation gate ──")

const EPS0_PAT = 8.854187817e-12
const C0_PAT = 299792458.0

freq_pf = 1.0e9
k_pf = 2π * freq_pf / C0_PAT
λ_pf = C0_PAT / freq_pf
Rfar_pf = 80 * λ_pf
pz_pf = 1e-12 + 0im

dip_pf = make_dipole(
    Vec3(0.0, 0.0, 0.0),
    CVec3(0.0 + 0im, 0.0 + 0im, pz_pf),
    Vec3(0.0, 0.0, 1.0),
    :electric,
    freq_pf,
)

theta_pat_deg_pf = collect(0.0:2.0:180.0)
phi_pat_deg_pf = collect(0.0:6.0:354.0)
pat_plus = make_analytic_dipole_pattern_feed(
    dip_pf,
    theta_pat_deg_pf,
    phi_pat_deg_pf;
    angles_in_degrees=true,
)
pat_plus_storage_bytes =
    sizeof(Float64) * (length(theta_pat_deg_pf) + length(phi_pat_deg_pf)) +
    2 * sizeof(ComplexF64) * length(theta_pat_deg_pf) * length(phi_pat_deg_pf)
@test make_analytic_dipole_pattern_feed(
    dip_pf, theta_pat_deg_pf, phi_pat_deg_pf;
    angles_in_degrees=true,
    max_storage_bytes=pat_plus_storage_bytes) isa PatternFeedExcitation
@test_throws ArgumentError make_analytic_dipole_pattern_feed(
    dip_pf, theta_pat_deg_pf, phi_pat_deg_pf;
    angles_in_degrees=true,
    max_storage_bytes=pat_plus_storage_bytes - 1)

function _pattern_feed_constructor_bytes(pat::PatternFeedExcitation)
    return @allocated make_pattern_feed(
        pat.theta, pat.phi, pat.Ftheta, pat.Fphi, pat.frequency;
        phase_center=pat.phase_center,
        convention=pat.convention,
    )
end
_pattern_feed_constructor_bytes(pat_plus)
pat_plus_copy = make_pattern_feed(
    pat_plus.theta, pat_plus.phi, pat_plus.Ftheta, pat_plus.Fphi,
    pat_plus.frequency;
    phase_center=pat_plus.phase_center,
    convention=pat_plus.convention,
)
pattern_constructor_bytes = _pattern_feed_constructor_bytes(pat_plus)
pattern_stored_bytes =
    Base.summarysize(pat_plus_copy.theta) +
    Base.summarysize(pat_plus_copy.phi) +
    Base.summarysize(pat_plus_copy.Ftheta) +
    Base.summarysize(pat_plus_copy.Fphi)
@test pattern_constructor_bytes <= pattern_stored_bytes + 32768

# Adapter-style constructor using two pattern objects with fields x, y, U
struct _PatternLike
    x::Vector{Float64}
    y::Vector{Float64}
    U::Matrix{ComplexF64}
end
pat_like_θ = _PatternLike(copy(pat_plus.theta), copy(pat_plus.phi), copy(pat_plus.Ftheta))
pat_like_ϕ = _PatternLike(copy(pat_plus.theta), copy(pat_plus.phi), copy(pat_plus.Fphi))
pat_from_like = make_pattern_feed(pat_like_θ, pat_like_ϕ, pat_plus.frequency; angles_in_degrees=false)

probe_like_dirs = (Vec3(1.0, 0.5, 2.0), Vec3(-0.8, 0.3, 1.4), Vec3(0.1, -0.9, 1.1))
rel_like = let worst = 0.0
    for d in probe_like_dirs
        r = Rfar_pf * d / norm(d)
        E_ref = pattern_feed_field(r, pat_plus)
        E_new = pattern_feed_field(r, pat_from_like)
        worst = max(worst, norm(E_ref - E_new) / max(norm(E_ref), 1e-30))
    end
    worst
end
println("  Pattern-object adapter mismatch:    $rel_like")
@assert rel_like < 1e-13

# Matrix-shape tolerance: transposed input should auto-correct
# Expected-by-design shape-tolerance behavior: auto-transpose with warnings.
pat_transposed = @test_logs (
    :warn,
    r"Ftheta matrix shape .* appears transposed; auto-transposing to .*",
) (
    :warn,
    r"Fphi matrix shape .* appears transposed; auto-transposing to .*",
) make_pattern_feed(
    pat_plus.theta,
    pat_plus.phi,
    permutedims(pat_plus.Ftheta),
    permutedims(pat_plus.Fphi),
    pat_plus.frequency;
    angles_in_degrees=false,
)
rel_transposed = let worst = 0.0
    for d in probe_like_dirs
        r = Rfar_pf * d / norm(d)
        E_ref = pattern_feed_field(r, pat_plus)
        E_new = pattern_feed_field(r, pat_transposed)
        worst = max(worst, norm(E_ref - E_new) / max(norm(E_ref), 1e-30))
    end
    worst
end
println("  Pattern-transpose auto-fix mismatch: $rel_transposed")
@assert rel_transposed < 1e-12

# Field-level comparison on a fine φ=0 cut against closed-form dipole far-field.
theta_eval_deg_pf = collect(0.0:0.5:180.0)
theta_eval_pf = deg2rad.(theta_eval_deg_pf)
P_num_pf = zeros(Float64, length(theta_eval_pf))
P_ref_pf = zeros(Float64, length(theta_eval_pf))
Etheta_num_pf = zeros(ComplexF64, length(theta_eval_pf))
Etheta_ref_pf = zeros(ComplexF64, length(theta_eval_pf))
cross_ratio_pf = zeros(Float64, length(theta_eval_pf))

for i in eachindex(theta_eval_pf)
    θ = theta_eval_pf[i]
    ϕ = 0.0
    rhat = Vec3(sin(θ), 0.0, cos(θ))
    r = Rfar_pf * rhat
    eθ = Vec3(cos(θ), 0.0, -sin(θ))
    eϕ = Vec3(0.0, 1.0, 0.0)

    E_num = pattern_feed_field(r, pat_plus)
    Eθ_num = dot(E_num, eθ)
    Eϕ_num = dot(E_num, eϕ)

    Eθ_ref = (k_pf^2 * pz_pf / (4π * EPS0_PAT)) * sin(θ) * exp(-1im * k_pf * Rfar_pf) / Rfar_pf
    Eϕ_ref = 0.0 + 0im

    Etheta_num_pf[i] = Eθ_num
    Etheta_ref_pf[i] = Eθ_ref
    P_num_pf[i] = abs2(Eθ_num) + abs2(Eϕ_num)
    P_ref_pf[i] = abs2(Eθ_ref) + abs2(Eϕ_ref)
    cross_ratio_pf[i] = abs(Eϕ_num) / max(sqrt(P_num_pf[i]), 1e-30)
end

P_num_pf ./= maximum(P_num_pf)
P_ref_pf ./= maximum(P_ref_pf)
err_lin_pf = P_num_pf .- P_ref_pf
rmse_pf = sqrt(mean(abs2, err_lin_pf))
maxabs_pf = maximum(abs.(err_lin_pf))
max_cross_pf = maximum(cross_ratio_pf)

phase_err_deg_pf = fill(NaN, length(theta_eval_pf))
phase_floor_pf = 1e-12 * maximum(abs.(Etheta_ref_pf))
for i in eachindex(theta_eval_pf)
    if abs(Etheta_ref_pf[i]) > phase_floor_pf && abs(Etheta_num_pf[i]) > phase_floor_pf
        phase_err_deg_pf[i] = rad2deg(angle(Etheta_num_pf[i] / Etheta_ref_pf[i]))
    end
end

phase_valid_pf = phase_err_deg_pf[.!isnan.(phase_err_deg_pf)]
phase_mean_pf = mean(phase_valid_pf)
phase_std_pf = std(phase_valid_pf)
phase_max_pf = maximum(abs.(phase_valid_pf))
phase_resid_pf = [rad2deg(atan(sin(deg2rad(x - phase_mean_pf)), cos(deg2rad(x - phase_mean_pf)))) for x in phase_valid_pf]
phase_resid_std_pf = std(phase_resid_pf)
phase_resid_max_pf = maximum(abs.(phase_resid_pf))

# Convention conversion check:
# If imported data comes from exp(-iωt), using conjugated coefficients with
# convention=:exp_minus_iwt must reproduce the same physical field.
pat_minus = make_pattern_feed(
    pat_plus.theta,
    pat_plus.phi,
    conj.(pat_plus.Ftheta),
    conj.(pat_plus.Fphi),
    pat_plus.frequency;
    convention=:exp_minus_iwt,
)
probe_dirs_pf = (
    Vec3(1.2, -0.4, 2.1),
    Vec3(-0.8, 1.5, 1.2),
    Vec3(0.5, 0.9, -1.7),
)
conv_mismatch_pf = let mismatch = 0.0
    for d in probe_dirs_pf
        r = Rfar_pf * d / norm(d)
        E_plus = pattern_feed_field(r, pat_plus)
        E_minus = pattern_feed_field(r, pat_minus)
        mismatch = max(mismatch, norm(E_plus - E_minus) / max(norm(E_plus), 1e-30))
    end
    mismatch
end

# RHS consistency against direct imported electric-field path.
mesh_pf = make_rect_plate(0.04, 0.04, 4, 4)
rwg_pf = build_rwg(mesh_pf)
v_pf_pat = assemble_excitation(mesh_pf, rwg_pf, pat_plus; quad_order=3)
imp_pf = ImportedExcitation(r -> pattern_feed_field(r, pat_plus); kind=:electric_field, min_quad_order=3)
v_pf_imp = assemble_excitation(mesh_pf, rwg_pf, imp_pf; quad_order=3)
rhs_rel_pf = norm(v_pf_pat - v_pf_imp) / max(norm(v_pf_pat), 1e-30)

println("  Pattern-feed RMSE (linear):        $rmse_pf")
println("  Pattern-feed max |err| (linear):   $maxabs_pf")
println("  Pattern-feed max cross-pol ratio:  $max_cross_pf")
println("  Pattern-feed phase mean (deg):     $phase_mean_pf")
println("  Pattern-feed phase std (deg):      $phase_std_pf")
println("  Pattern-feed phase max |err| (deg): $phase_max_pf")
println("  Pattern-feed phase residual std (deg): $phase_resid_std_pf")
println("  Pattern-feed phase residual max |err| (deg): $phase_resid_max_pf")
println("  Convention conversion mismatch:    $conv_mismatch_pf")
println("  RHS path mismatch (pattern vs imported): $rhs_rel_pf")

@assert rmse_pf < 2e-4 "Pattern-feed gate failed: RMSE=$rmse_pf"
@assert maxabs_pf < 5e-4 "Pattern-feed gate failed: max |err|=$maxabs_pf"
@assert max_cross_pf < 1e-8 "Pattern-feed gate failed: cross-pol ratio=$max_cross_pf"
@assert phase_resid_std_pf < 0.2 "Pattern-feed gate failed: phase residual std=$phase_resid_std_pf"
@assert phase_resid_max_pf < 0.5 "Pattern-feed gate failed: phase residual max |err|=$phase_resid_max_pf"
@assert conv_mismatch_pf < 1e-12 "Pattern-feed gate failed: convention mismatch=$conv_mismatch_pf"
@assert rhs_rel_pf < 1e-12 "Pattern-feed gate failed: RHS mismatch=$rhs_rel_pf"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 17: GMRES solver and dispatch
# ─────────────────────────────────────────────────
println("\n── Test 17: GMRES solver and dispatch ──")

# Use impedance-loaded system from Test 7
Z_gm = Matrix{ComplexF64}(Z_full)
I_gm_direct = Z_gm \ v

# GMRES forward solve (no preconditioner)
gmres_workspace_bytes = DiffMoM._gmres_workspace_bytes(
    size(Z_gm, 1), 20)
I_gmres_nop, stats_nop = solve_gmres(Z_gm, Vector{ComplexF64}(v);
                                       tol=1e-10, maxiter=500,
                                       max_workspace_bytes=gmres_workspace_bytes)
@test_throws ArgumentError solve_gmres(
    Z_gm, Vector{ComplexF64}(v);
    tol=1e-10, maxiter=500,
    max_workspace_bytes=gmres_workspace_bytes - 1)
rel_gmres_nop = norm(I_gmres_nop - I_gm_direct) / max(norm(I_gm_direct), 1e-30)
println("  GMRES (no precond) rel error: $rel_gmres_nop  iters: $(stats_nop.niter)")
@assert rel_gmres_nop < 1e-6 "GMRES without preconditioner inaccurate: $rel_gmres_nop"

# GMRES adjoint solve (no preconditioner)
rhs_adj_gm = Vector{ComplexF64}(Q * I_gm_direct)
lam_gm_direct = Z_gm' \ rhs_adj_gm
lam_gmres_nop, stats_adj_nop = solve_gmres_adjoint(Z_gm, rhs_adj_gm;
                                                      tol=1e-10, maxiter=500,
                                                      max_workspace_bytes=
                                                          gmres_workspace_bytes)
@test_throws ArgumentError solve_gmres_adjoint(
    Z_gm, rhs_adj_gm;
    tol=1e-10, maxiter=500,
    max_workspace_bytes=gmres_workspace_bytes - 1)
rel_adj_nop = norm(lam_gmres_nop - lam_gm_direct) / max(norm(lam_gm_direct), 1e-30)
println("  GMRES adjoint (no precond) rel error: $rel_adj_nop  iters: $(stats_adj_nop.niter)")
@assert rel_adj_nop < 1e-6

# Krylov's internal absolute breakdown threshold must not classify a globally
# tiny, perfectly conditioned system as an inconsistent least-squares solve.
for tiny_gmres_scale in (1e-200, 1e-300, nextfloat(0.0))
    tiny_gmres_matrix = reshape(
        ComplexF64[tiny_gmres_scale], 1, 1)
    tiny_gmres_rhs = ComplexF64[tiny_gmres_scale]
    tiny_gmres_solution, tiny_gmres_stats = solve_gmres(
        tiny_gmres_matrix, tiny_gmres_rhs;
        tol=1e-8, maxiter=3,
    )
    @test tiny_gmres_stats.solved
    @test !tiny_gmres_stats.inconsistent
    @test isapprox(
        tiny_gmres_solution[1], 1.0 + 0.0im;
        rtol=eps(Float64), atol=0.0,
    )
    @test DiffMoM._assert_true_residual(
        tiny_gmres_matrix, tiny_gmres_solution, tiny_gmres_rhs,
        "tiny GMRES regression";
        tol=1e-8, factor=1.0,
    ) <= eps(Float64)
    @test solve_forward(
        tiny_gmres_matrix, tiny_gmres_rhs;
        solver=:gmres, gmres_tol=1e-8, gmres_maxiter=3,
    ) == tiny_gmres_solution
end

tiny_gmres_matrix = reshape(ComplexF64[1e-200], 1, 1)
tiny_gmres_krylov_bytes = DiffMoM._gmres_workspace_bytes(1, 3)
tiny_gmres_scaling_bytes = 2sizeof(ComplexF64)
@test_throws ArgumentError solve_gmres(
    tiny_gmres_matrix, ComplexF64[1e-200];
    tol=1e-8, maxiter=3,
    max_workspace_bytes=
        tiny_gmres_krylov_bytes + tiny_gmres_scaling_bytes - 1)
@test solve_gmres(
    tiny_gmres_matrix, ComplexF64[1e-200];
    tol=1e-8, maxiter=3,
    max_workspace_bytes=
        tiny_gmres_krylov_bytes + tiny_gmres_scaling_bytes)[1] ==
      ComplexF64[1.0]
large_gmres_solution, large_gmres_stats = solve_gmres(
    tiny_gmres_matrix, ComplexF64[1.0];
    tol=1e-8, maxiter=3,
)
@test large_gmres_stats.solved && !large_gmres_stats.inconsistent
@test large_gmres_solution == ComplexF64[1e200]

imaginary_gmres_matrix = reshape(ComplexF64[1e-200im], 1, 1)
imaginary_gmres_rhs = ComplexF64[1e-200im]
imaginary_adjoint_solution, imaginary_adjoint_stats =
    solve_gmres_adjoint(
        imaginary_gmres_matrix, imaginary_gmres_rhs;
        tol=1e-8, maxiter=3,
    )
@test imaginary_adjoint_stats.solved
@test !imaginary_adjoint_stats.inconsistent
@test imaginary_adjoint_solution == ComplexF64[-1.0]

identity_gmres_preconditioner = DiagonalPreconditionerData(
    ComplexF64[1.0], Inf, 1.0)
for tiny_gmres_side in (:left, :right)
    preconditioned_tiny_solution, preconditioned_tiny_stats = solve_gmres(
        tiny_gmres_matrix, ComplexF64[1e-200];
        preconditioner=identity_gmres_preconditioner,
        precond_side=tiny_gmres_side,
        tol=1e-8,
        maxiter=3,
    )
    @test preconditioned_tiny_solution == ComplexF64[1.0]
    @test preconditioned_tiny_stats.solved
    @test !preconditioned_tiny_stats.inconsistent
end

@test DiffMoM._true_residual_ratio(
    tiny_gmres_matrix,
    ComplexF64[0.0],
    ComplexF64[1e-200],
    "tiny residual probe",
) == 1.0
@test_throws ErrorException DiffMoM._assert_true_residual(
    tiny_gmres_matrix,
    ComplexF64[0.0],
    ComplexF64[1e-200],
    "tiny residual probe";
    tol=1e-8,
    factor=1.0,
)
@test DiffMoM._assert_true_residual(
    tiny_gmres_matrix,
    ComplexF64[0.0],
    ComplexF64[0.0],
    "zero RHS residual probe";
    tol=1e-8,
    factor=1.0,
) == 0.0
@test_throws ErrorException DiffMoM._assert_gmres_converged(
    (solved=true,
     inconsistent=true,
     niter=0,
     status="found approximate least-squares solution",
     residuals=[0.0]),
    "inconsistent stats regression";
    tol=1e-8,
    maxiter=3,
)

# solve_forward dispatch: :direct
I_sf_direct = solve_forward(Z_gm, Vector{ComplexF64}(v))
rel_sf_direct = norm(I_sf_direct - I_gm_direct) / max(norm(I_gm_direct), 1e-30)
@assert rel_sf_direct < 1e-12
@test_throws DimensionMismatch solve_forward(
    ones(ComplexF64, 2, 1), ComplexF64[1.0, 2.0])
@test_throws ArgumentError solve_forward(
    ComplexF64[NaN 0.0; 0.0 1.0], ComplexF64[1.0, 1.0])
@test_throws ArgumentError solve_forward(
    ComplexF64[1.0 0.0; 0.0 1.0], ComplexF64[NaN, 1.0])
@test_throws ArgumentError solve_system(
    ComplexF64[1.0 0.0; 0.0 1.0], ComplexF64[Inf, 1.0])

# Direct LU substitution can overflow on an extreme finite RHS even when the
# mathematical solution is finite. Exercise every public direct dispatch and
# the corresponding subnormal scale-up path.
Z_extreme_rhs = ComplexF64[1.0 -2.0; -2.0 1.0]
rhs_extreme = fill(ComplexF64(0.8 * floatmax(Float64)), 2)
extreme_reference = setprecision(BigFloat, 4096) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(Z_extreme_rhs) \
        Complex{BigFloat}.(rhs_extreme))
end
Q_extreme_rhs = Matrix{ComplexF64}(I, 2, 2)
direct_extreme_solutions = (
    solve_forward(Z_extreme_rhs, rhs_extreme),
    solve_system(Z_extreme_rhs, rhs_extreme),
    solve_adjoint_rhs(Z_extreme_rhs, rhs_extreme),
    solve_adjoint(Z_extreme_rhs, Q_extreme_rhs, rhs_extreme),
)
for solution in direct_extreme_solutions
    @test all(isfinite, solution)
    @test all(
        isapprox(real(solution[index]), real(extreme_reference[index]);
                 rtol=8eps(Float64), atol=0.0) &&
        isapprox(imag(solution[index]), imag(extreme_reference[index]);
                 rtol=8eps(Float64), atol=0.0)
        for index in eachindex(extreme_reference))
end
real_extreme_solution = solve_forward(
    Float64[1.0 -2.0; -2.0 1.0],
    fill(0.8 * floatmax(Float64), 2))
@test all(
    isapprox(real_extreme_solution[index], real(extreme_reference[index]);
             rtol=8eps(Float64), atol=0.0)
    for index in eachindex(extreme_reference))
tiny_direct_rhs = fill(ComplexF64(nextfloat(0.0)), 2)
@test solve_forward(Matrix{ComplexF64}(I, 2, 2), tiny_direct_rhs) ==
      tiny_direct_rhs

# A subnormal matrix can make LAPACK substitution form a non-finite reciprocal
# even when jointly scaling the matrix and RHS reveals a modest exact solution.
tiny_direct_matrix = reshape(ComplexF64[nextfloat(0.0)], 1, 1)
tiny_matrix_rhs = ComplexF64[floatmin(Float64)]
tiny_matrix_reference = setprecision(BigFloat, 4096) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(tiny_direct_matrix) \
        Complex{BigFloat}.(tiny_matrix_rhs))
end
@test tiny_matrix_reference == ComplexF64[2.0^52]
tiny_matrix_solutions = (
    solve_forward(tiny_direct_matrix, tiny_matrix_rhs),
    solve_system(tiny_direct_matrix, tiny_matrix_rhs),
    solve_adjoint_rhs(tiny_direct_matrix, tiny_matrix_rhs),
    solve_adjoint(
        tiny_direct_matrix,
        reshape(ComplexF64[1.0], 1, 1),
        tiny_matrix_rhs,
    ),
)
@test all(solution == tiny_matrix_reference
          for solution in tiny_matrix_solutions)
@test solve_forward(
    reshape(Float64[nextfloat(0.0)], 1, 1),
    Float64[floatmin(Float64)],
) == Float64[2.0^52]
@test solve_forward(
    reshape(Float32[nextfloat(0.0f0)], 1, 1),
    Float32[floatmin(Float32)],
) == Float32[2.0f0^23]
@test solve_forward(
    reshape(ComplexF32[nextfloat(0.0f0)], 1, 1),
    ComplexF32[floatmin(Float32)],
) == ComplexF32[2.0f0^23]
@test_throws OverflowError solve_forward(
    tiny_direct_matrix,
    ComplexF64[floatmax(Float64)],
)

# A finite LU result is not sufficient evidence of a correct solve. LAPACK's
# unscaled elimination loses the second row of this mixed-range matrix and
# returns [0.5, -0.5], whose componentwise backward error is 0.2. The direct
# paths must retry with row/column power-of-two equilibration and verify the
# result against the physical matrix.
direct_mixed_matrix = ComplexF64[
    1e200  1e200
    1e-200 2e-200
]
direct_mixed_rhs = ComplexF64[0.0, -1e-200]
direct_mixed_reference = setprecision(BigFloat, 512) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(direct_mixed_matrix) \
        Complex{BigFloat}.(direct_mixed_rhs))
end
@test direct_mixed_reference == ComplexF64[1.0, -1.0]
direct_unverified_candidate = ComplexF64[0.5, -0.5]
@test DiffMoM._direct_backward_error(
    direct_mixed_matrix,
    direct_unverified_candidate,
    direct_mixed_rhs,
    ComplexF64,
) == 0.2
@test DiffMoM._direct_backward_error(
    direct_mixed_matrix,
    direct_mixed_reference,
    direct_mixed_rhs,
    ComplexF64,
) <= DiffMoM._direct_backward_error_limit(Float64, 2)

direct_mixed_Q = Matrix{ComplexF64}(I, 2, 2)
for solution in (
        solve_forward(direct_mixed_matrix, direct_mixed_rhs),
        solve_system(direct_mixed_matrix, direct_mixed_rhs),
        solve_adjoint(
            Matrix(adjoint(direct_mixed_matrix)),
            direct_mixed_Q,
            direct_mixed_rhs,
        ),
        solve_adjoint_rhs(
            Matrix(adjoint(direct_mixed_matrix)), direct_mixed_rhs),
    )
    @test solution ≈ direct_mixed_reference rtol=4eps(Float64)
end
@test solve_forward(
    real.(direct_mixed_matrix), real.(direct_mixed_rhs)) ≈
      real.(direct_mixed_reference) rtol=4eps(Float64)

direct_mixed_adjoint_rhs = ComplexF64[0.0, -1.0]
direct_mixed_adjoint_reference = setprecision(BigFloat, 512) do
    ComplexF64.(
        adjoint(Matrix{Complex{BigFloat}}(direct_mixed_matrix)) \
        Complex{BigFloat}.(direct_mixed_adjoint_rhs))
end
direct_mixed_adjoint_solution = solve_adjoint_rhs(
    direct_mixed_matrix, direct_mixed_adjoint_rhs)
@test direct_mixed_adjoint_solution ≈
      direct_mixed_adjoint_reference rtol=4eps(Float64)
@test dot(direct_mixed_adjoint_rhs, direct_mixed_reference) ≈
      dot(direct_mixed_adjoint_solution, direct_mixed_rhs) rtol=4eps(Float64)

direct_mixed_rhs_matrix = hcat(
    direct_mixed_rhs, ComplexF64[1e200, 1e-200])
direct_mixed_matrix_reference = setprecision(BigFloat, 512) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(direct_mixed_matrix) \
        Matrix{Complex{BigFloat}}(direct_mixed_rhs_matrix))
end
direct_mixed_factor = lu(direct_mixed_matrix)
@test DiffMoM._solve_factored_linear_system(
    direct_mixed_factor,
    direct_mixed_matrix,
    direct_mixed_rhs_matrix,
    "mixed-range matrix RHS",
) ≈ direct_mixed_matrix_reference rtol=8eps(Float64)
direct_mixed_alias = copy(direct_mixed_rhs_matrix)
@test DiffMoM._solve_factored_linear_system!(
    direct_mixed_alias,
    direct_mixed_factor,
    direct_mixed_matrix,
    direct_mixed_alias,
    "mixed-range aliased matrix RHS",
) ≈ direct_mixed_matrix_reference rtol=8eps(Float64)

direct_matrix_alias = ComplexF64[2.0 0.0; 0.0 3.0]
direct_matrix_alias_factor = lu(copy(direct_matrix_alias))
@test DiffMoM._solve_factored_linear_system!(
    direct_matrix_alias,
    direct_matrix_alias_factor,
    direct_matrix_alias,
    Matrix{ComplexF64}(I, 2, 2),
    "physical matrix alias",
) ≈ ComplexF64[0.5 0.0; 0.0 1 / 3] rtol=4eps(Float64)

_, direct_mixed_conditioning_factor = transform_patch_matrices(
    [Matrix{ComplexF64}(I, 2, 2)];
    preconditioner_M=direct_mixed_matrix,
)
@test direct_mixed_conditioning_factor \ direct_mixed_rhs ≈
      direct_mixed_reference rtol=4eps(Float64)
direct_mixed_conditioning_destination = copy(direct_mixed_rhs)
direct_mixed_conditioning_returned = ldiv!(
    direct_mixed_conditioning_factor,
    direct_mixed_conditioning_destination,
)
@test direct_mixed_conditioning_returned ===
      direct_mixed_conditioning_destination
@test direct_mixed_conditioning_destination ≈
      direct_mixed_reference rtol=4eps(Float64)

# A raw LU factor does not retain the physical input matrix, so factor-only
# verification must fail closed. Factors returned by the conditioning APIs
# retain their matrix and remain reusable through the documented factor-only
# path below.
direct_factor_only_destination = copy(direct_mixed_rhs)
@test_throws ArgumentError DiffMoM._solve_factored_linear_system!(
    direct_factor_only_destination,
    direct_mixed_factor,
    nothing,
    direct_mixed_rhs,
    "mixed-range factor-only solve",
)
@test direct_factor_only_destination == direct_mixed_rhs

# Ordinary factor exponents do not make Matrix(factor) a physical oracle
# either: Wilkinson growth loses input entries while reconstructing L*U.
direct_wilkinson_size = 55
direct_wilkinson_matrix = zeros(
    ComplexF64, direct_wilkinson_size, direct_wilkinson_size)
for row in 1:direct_wilkinson_size
    direct_wilkinson_matrix[row, row] = 1.0
    for column in 1:(row - 1)
        direct_wilkinson_matrix[row, column] = -1.0
    end
    direct_wilkinson_matrix[row, end] =
        1.0 + (mod(7row, 17) - 8) * eps(Float64)
end
direct_wilkinson_matrix[end, end] = 1.0
direct_wilkinson_reference = ComplexF64[
    (-1)^index * (1 + index / 100)
    for index in 1:direct_wilkinson_size
]
direct_wilkinson_rhs = setprecision(BigFloat, 512) do
    ComplexF64[
        sum(
            Complex{BigFloat}(direct_wilkinson_matrix[row, column]) *
            Complex{BigFloat}(direct_wilkinson_reference[column])
            for column in 1:direct_wilkinson_size
        )
        for row in 1:direct_wilkinson_size
    ]
end
direct_wilkinson_factor = lu(direct_wilkinson_matrix)
direct_wilkinson_factor_exponent = maximum(
    exponent(abs(value)) for value in direct_wilkinson_factor.factors
    if !iszero(value))
@test direct_wilkinson_factor_exponent <= 128
@test maximum(abs, Matrix(direct_wilkinson_factor) -
                   direct_wilkinson_matrix) >= 0.5
direct_wilkinson_bad = direct_wilkinson_factor \ direct_wilkinson_rhs
@test DiffMoM._direct_backward_error(
    direct_wilkinson_matrix,
    direct_wilkinson_bad,
    direct_wilkinson_rhs,
    ComplexF64,
) > DiffMoM._direct_backward_error_limit(
    Float64, direct_wilkinson_size)
@test_throws ArgumentError prepare_conditioned_system(
    Matrix{ComplexF64}(I, direct_wilkinson_size, direct_wilkinson_size),
    direct_wilkinson_rhs;
    preconditioner_factor=direct_wilkinson_factor,
)
_, direct_wilkinson_solution, _ = prepare_conditioned_system(
    Matrix{ComplexF64}(I, direct_wilkinson_size, direct_wilkinson_size),
    direct_wilkinson_rhs;
    preconditioner_M=direct_wilkinson_matrix,
    preconditioner_factor=direct_wilkinson_factor,
)
@test DiffMoM._direct_backward_error(
    direct_wilkinson_matrix,
    direct_wilkinson_solution,
    direct_wilkinson_rhs,
    ComplexF64,
) <= DiffMoM._direct_backward_error_limit(Float64, direct_wilkinson_size)
@test direct_wilkinson_solution ≈
      direct_wilkinson_reference rtol=8eps(Float64)

# Equilibration may round away an irrelevant component; the physical residual
# check determines whether doing so was safe.
direct_mixed_component_matrix = copy(direct_mixed_matrix)
direct_mixed_component_matrix[1, 1] += nextfloat(0.0)im
direct_mixed_component_reference = setprecision(BigFloat, 512) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(direct_mixed_component_matrix) \
        Complex{BigFloat}.(direct_mixed_rhs))
end
@test solve_forward(
    direct_mixed_component_matrix, direct_mixed_rhs) ≈
      direct_mixed_component_reference rtol=8eps(Float64)

# A subnormal component can be the only rank-restoring information. The
# balanced Float64 factor intentionally rounds it away, so the final exact
# physical-matrix factorization must recover the representable solution.
direct_rank_restoring_scale = nextfloat(0.0)
direct_rank_restoring_large = ldexp(1.0, 500)
direct_rank_restoring_small = ldexp(1.0, -500)
direct_rank_restoring_matrix = ComplexF64[
    complex(direct_rank_restoring_large, direct_rank_restoring_scale) direct_rank_restoring_large
    direct_rank_restoring_small                                    direct_rank_restoring_small
]
direct_rank_restoring_rhs = ComplexF64[
    complex(0.0, direct_rank_restoring_scale), 0.0]
direct_rank_restoring_reference = setprecision(BigFloat, 4352) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(direct_rank_restoring_matrix) \
        Complex{BigFloat}.(direct_rank_restoring_rhs))
end
@test direct_rank_restoring_reference == ComplexF64[1.0, -1.0]
direct_rank_restoring_factor = DiffMoM._factor_dense_linear_system(
    direct_rank_restoring_matrix,
    ComplexF64,
    "rank-restoring regression",
)
@test direct_rank_restoring_factor isa DiffMoM._BigFloatDenseLUPlan
@test isnothing(DiffMoM._validate_bigfloat_plan_size(
    DiffMoM._MAX_DIRECT_BIGFLOAT_VALUES - 1,
    1,
    "direct fallback resource boundary",
))
@test_throws ArgumentError DiffMoM._validate_bigfloat_plan_size(
    DiffMoM._MAX_DIRECT_BIGFLOAT_VALUES,
    1,
    "direct fallback resource overflow",
)
@test solve_forward(
    direct_rank_restoring_matrix, direct_rank_restoring_rhs) ==
      direct_rank_restoring_reference
direct_rank_restoring_adjoint_rhs = ComplexF64[
    complex(direct_rank_restoring_large, -direct_rank_restoring_scale),
    direct_rank_restoring_large,
]
@test adjoint(direct_rank_restoring_factor) \
      direct_rank_restoring_adjoint_rhs == ComplexF64[1.0, 0.0]

# The initial unscaled LU can also report a false zero pivot: its elimination
# multiplier underflows although the exact determinant is -1.
direct_false_singular_matrix = ComplexF64[
    1e200  1e200
    1e-200 0.0
]
direct_false_singular_rhs = ComplexF64[1e200, 0.0]
@test_throws LinearAlgebra.SingularException lu(direct_false_singular_matrix)
direct_false_singular_reference = setprecision(BigFloat, 512) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(direct_false_singular_matrix) \
        Complex{BigFloat}.(direct_false_singular_rhs))
end
@test direct_false_singular_reference == ComplexF64[0.0, 1.0]
@test solve_forward(
    direct_false_singular_matrix, direct_false_singular_rhs) ≈
      direct_false_singular_reference rtol=4eps(Float64)
@test solve_adjoint_rhs(
    Matrix(adjoint(direct_false_singular_matrix)),
    direct_false_singular_rhs) ≈
      direct_false_singular_reference rtol=4eps(Float64)
direct_false_singular_inverse = setprecision(BigFloat, 512) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(direct_false_singular_matrix) \
        Matrix{Complex{BigFloat}}(I, 2, 2))
end
direct_false_singular_Mp, direct_false_singular_factor =
    transform_patch_matrices(
        [Matrix{ComplexF64}(I, 2, 2)];
        preconditioner_M=direct_false_singular_matrix,
    )
@test direct_false_singular_Mp[1] ≈
      direct_false_singular_inverse rtol=8eps(Float64)
@test direct_false_singular_factor.factorization isa
      DiffMoM._EquilibratedDenseLUPlan
direct_false_singular_Z, direct_false_singular_v, _ =
    prepare_conditioned_system(
        Matrix{ComplexF64}(I, 2, 2),
        direct_false_singular_rhs;
        preconditioner_factor=direct_false_singular_factor,
    )
@test direct_false_singular_Z ≈
      direct_false_singular_inverse rtol=8eps(Float64)
@test direct_false_singular_v ≈
      direct_false_singular_reference rtol=8eps(Float64)

# BLAS can overflow while forming Q*I even when cancellation makes the exact
# adjoint RHS finite. The exceptional product restart must recover the full
# sum without adding workspace allocations to ordinary dense products.
adjoint_product_scale = 0.8 * floatmax(Float64)
Q_extreme_product = zeros(ComplexF64, 4, 4)
Q_extreme_product[1, :] .= ComplexF64[1.0, 1.0, -1.0, -1.0]
I_extreme_product = fill(ComplexF64(adjoint_product_scale), 4)
ordinary_extreme_product = Q_extreme_product * I_extreme_product
@test !isfinite(ordinary_extreme_product[1])
extreme_product_reference = setprecision(BigFloat, 4096) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(Q_extreme_product) *
        Complex{BigFloat}.(I_extreme_product))
end
@test extreme_product_reference == zeros(ComplexF64, 4)
@test solve_adjoint(
    Matrix{ComplexF64}(I, 4, 4),
    Q_extreme_product,
    I_extreme_product,
) == extreme_product_reference
Q_extreme_operands = zeros(ComplexF64, 4, 4)
Q_extreme_operands[1, :] .= ComplexF64[
    floatmax(Float64),
    floatmax(Float64),
    -floatmax(Float64),
    -floatmax(Float64),
]
I_extreme_operands = fill(ComplexF64(floatmax(Float64)), 4)
@test !all(isfinite, Q_extreme_operands * I_extreme_operands)
@test solve_adjoint(
    Matrix{ComplexF64}(I, 4, 4),
    Q_extreme_operands,
    I_extreme_operands,
) == zeros(ComplexF64, 4)
Q_extreme_product_real = real.(Q_extreme_product)
@test DiffMoM._finite_matrix_vector_product(
    Q_extreme_product_real,
    fill(adjoint_product_scale, 4),
    "real product",
) == zeros(Float64, 4)
dense_underflow_matrix = fill(nextfloat(0.0), 1, 2)
dense_underflow_input = fill(0.4, 2)
dense_underflow_reference = setprecision(BigFloat, 4352) do
    Float64[sum(
        BigFloat(dense_underflow_matrix[1, column]) *
        BigFloat(dense_underflow_input[column])
        for column in axes(dense_underflow_matrix, 2))]
end
@test dense_underflow_reference == Float64[nextfloat(0.0)]
@test DiffMoM._finite_matrix_vector_product(
    dense_underflow_matrix, dense_underflow_input,
    "Float64 underflow regression") == dense_underflow_reference
dense_underflow_workspace = zeros(Float64, 1)
DiffMoM._finite_matrix_vector_product!(
    dense_underflow_workspace,
    dense_underflow_matrix, dense_underflow_input,
    "in-place Float64 underflow regression")
@test dense_underflow_workspace == dense_underflow_reference

dense_underflow_matrix32 = fill(nextfloat(0.0f0), 1, 2)
dense_underflow_input32 = fill(0.4f0, 2)
dense_underflow_reference32 = setprecision(BigFloat, 4352) do
    Float32[sum(
        BigFloat(dense_underflow_matrix32[1, column]) *
        BigFloat(dense_underflow_input32[column])
        for column in axes(dense_underflow_matrix32, 2))]
end
@test dense_underflow_reference32 == Float32[nextfloat(0.0f0)]
@test DiffMoM._finite_matrix_vector_product(
    dense_underflow_matrix32, dense_underflow_input32,
    "Float32 underflow regression") == dense_underflow_reference32

dense_mixed_component_matrix = ComplexF64[
    1.0 + nextfloat(0.0)im  -1.0 + nextfloat(0.0)im
]
dense_mixed_component_input = fill(ComplexF64(0.4), 2)
dense_mixed_component_reference = setprecision(BigFloat, 4352) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(dense_mixed_component_matrix) *
        Complex{BigFloat}.(dense_mixed_component_input))
end
@test dense_mixed_component_reference ==
      ComplexF64[Complex(0.0, nextfloat(0.0))]
@test DiffMoM._finite_matrix_vector_product(
    dense_mixed_component_matrix,
    dense_mixed_component_input,
    "mixed-component Float64 underflow regression",
) == dense_mixed_component_reference
dense_mixed_component_workspace = zeros(ComplexF64, 1)
DiffMoM._finite_matrix_vector_product!(
    dense_mixed_component_workspace,
    dense_mixed_component_matrix,
    dense_mixed_component_input,
    "in-place mixed-component Float64 underflow regression",
)
@test dense_mixed_component_workspace == dense_mixed_component_reference

dense_mixed_component_matrix32 = ComplexF32[
    1.0f0 + nextfloat(0.0f0)im  -1.0f0 + nextfloat(0.0f0)im
]
dense_mixed_component_input32 = fill(ComplexF32(0.4f0), 2)
dense_mixed_component_reference32 = setprecision(BigFloat, 4352) do
    ComplexF32.(
        Matrix{Complex{BigFloat}}(dense_mixed_component_matrix32) *
        Complex{BigFloat}.(dense_mixed_component_input32))
end
@test dense_mixed_component_reference32 ==
      ComplexF32[Complex(0.0f0, nextfloat(0.0f0))]
@test DiffMoM._finite_matrix_vector_product(
    dense_mixed_component_matrix32,
    dense_mixed_component_input32,
    "mixed-component Float32 underflow regression",
) == dense_mixed_component_reference32
@test_throws OverflowError solve_adjoint(
    Matrix{ComplexF64}(I, 1, 1),
    reshape(ComplexF64[2.0], 1, 1),
    ComplexF64[floatmax(Float64)],
)
I_product_allocation = fill(1.0 + 0.0im, 4)
DiffMoM._finite_matrix_vector_product(
    Q_extreme_product, I_product_allocation, "allocation probe")
@test @allocated(DiffMoM._finite_matrix_vector_product(
    Q_extreme_product, I_product_allocation, "allocation probe")) <=
      _complex_vector_output_allocation(length(I_product_allocation)) + 128

# solve_forward dispatch: :gmres (unpreconditioned)
I_sf_gmres = solve_forward(Z_gm, Vector{ComplexF64}(v);
                            solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
rel_sf_gmres = norm(I_sf_gmres - I_gm_direct) / max(norm(I_gm_direct), 1e-30)
println("  solve_forward :gmres rel error: $rel_sf_gmres")
@assert rel_sf_gmres < 1e-6

# solve_adjoint dispatch: :direct
lam_sa_direct = solve_adjoint(Z_gm, Q, I_gm_direct)
rel_sa_direct = norm(lam_sa_direct - lam_gm_direct) / max(norm(lam_gm_direct), 1e-30)
@assert rel_sa_direct < 1e-12
@test_throws ArgumentError solve_adjoint(
    ComplexF64[1.0 0.0; 0.0 1.0],
    ComplexF64[NaN 0.0; 0.0 1.0],
    ComplexF64[1.0, 1.0],
)
adjoint_operator_size = 512
adjoint_operator_probe = Diagonal(
    ones(ComplexF64, adjoint_operator_size))
adjoint_operator_Q = Matrix{ComplexF64}(
    I, adjoint_operator_size, adjoint_operator_size)
adjoint_operator_I = ones(ComplexF64, adjoint_operator_size)
unsupported_direct_adjoint = () -> solve_adjoint(
    adjoint_operator_probe, adjoint_operator_Q, adjoint_operator_I)
try
    unsupported_direct_adjoint()
catch
end
@test (@allocated try
    unsupported_direct_adjoint()
catch
end) < 4_096
@test_throws ArgumentError solve_adjoint_rhs(
    ComplexF64[1.0 0.0; 0.0 1.0],
    ComplexF64[NaN, 1.0],
)
objective_probe_I = ComplexF64[1.0, 2.0]
objective_probe_Q = ComplexF64[2.0 0.5; 0.5 3.0]
objective_probe_value = compute_objective(
    objective_probe_I, objective_probe_Q)
@test objective_probe_value ≈
      real(dot(objective_probe_I, objective_probe_Q * objective_probe_I))
compute_objective(objective_probe_I, objective_probe_Q)
@test @allocated(compute_objective(
    objective_probe_I, objective_probe_Q)) == 0
@test_throws ArgumentError compute_objective(
    objective_probe_I, ComplexF64[NaN 0.0; 0.0 1.0])
objective_extreme_scale = floatmax(Float64)
objective_extreme_I = ComplexF64[
    objective_extreme_scale,
    objective_extreme_scale,
    objective_extreme_scale,
    objective_extreme_scale,
    1.0,
]
objective_extreme_Q = zeros(ComplexF64, 5, 5)
objective_extreme_Q[1, 1:4] .= ComplexF64[
    objective_extreme_scale,
    objective_extreme_scale,
    -objective_extreme_scale,
    -objective_extreme_scale,
]
objective_extreme_Q[5, 5] = 3.0
@test !isfinite(real(DiffMoM._dot_left_matrix_right(
    objective_extreme_I, objective_extreme_Q, objective_extreme_I)))
objective_extreme_reference = setprecision(BigFloat, 8192) do
    real(dot(
        Complex{BigFloat}.(objective_extreme_I),
        Matrix{Complex{BigFloat}}(objective_extreme_Q),
        Complex{BigFloat}.(objective_extreme_I),
    ))
end
@test objective_extreme_reference == BigFloat(3)
@test compute_objective(objective_extreme_I, objective_extreme_Q) == 3.0
@test compute_objective(
    Float64[objective_extreme_scale, objective_extreme_scale,
            objective_extreme_scale, objective_extreme_scale, 1.0],
    real.(objective_extreme_Q),
) == 3.0

# A Hermitian quadratic form can contain representable three-factor terms
# even when either staged matrix-vector orientation rounds one term to zero.
objective_underflow_unit = nextfloat(0.0)
objective_underflow_Q = ComplexF64[
    0.0 objective_underflow_unit
    objective_underflow_unit 0.0
]
objective_underflow_local = LocalMassMatrix(
    2,
    [1, 2],
    [2, 1],
    fill(ComplexF64(objective_underflow_unit), 2),
)
for objective_underflow_I in (
    ComplexF64[ldexp(1.0, 664), ldexp(1.0, -664)],
    ComplexF64[ldexp(1.0, -664), ldexp(1.0, 664)],
)
    objective_underflow_reference = setprecision(BigFloat, 6656) do
        Float64(real(dot(
            Complex{BigFloat}.(objective_underflow_I),
            Matrix{Complex{BigFloat}}(objective_underflow_Q),
            Complex{BigFloat}.(objective_underflow_I),
        )))
    end
    @test objective_underflow_reference ==
          2 * objective_underflow_unit
    @test compute_objective(
        objective_underflow_I, objective_underflow_Q,
    ) == objective_underflow_reference
    for objective_underflow_matrix in (
        objective_underflow_Q,
        sparse(objective_underflow_Q),
        objective_underflow_local,
    )
        @test DiffMoM._finite_bilinear_component(
            objective_underflow_I,
            objective_underflow_matrix,
            objective_underflow_I,
            Val(:real),
            "quadratic underflow regression",
        ) == objective_underflow_reference
    end
    objective_underflow_product = zeros(ComplexF64, 2)
    objective_underflow_product_used_fallback =
        DiffMoM._finite_matrix_vector_product_status!(
            objective_underflow_product,
            objective_underflow_Q,
            objective_underflow_I,
            "quadratic underflow product regression",
        )
    @test objective_underflow_product_used_fallback
    @test DiffMoM._quadratic_objective_from_product(
        objective_underflow_I,
        objective_underflow_Q,
        objective_underflow_product,
    ) == objective_underflow_reference
    @test DiffMoM._quadratic_objective_from_product(
        objective_underflow_I,
        objective_underflow_Q,
        objective_underflow_product,
        objective_underflow_product_used_fallback,
    ) == objective_underflow_reference
end

objective_underflow_duplicate_local = LocalMassMatrix(
    2,
    [1, 1, 2, 2],
    [2, 2, 1, 1],
    fill(ComplexF64(objective_underflow_unit), 4),
)
objective_underflow_duplicate_I = ComplexF64[
    ldexp(1.0, 664), ldexp(1.0, -664)
]
@test DiffMoM._finite_bilinear_component(
    objective_underflow_duplicate_I,
    objective_underflow_duplicate_local,
    objective_underflow_duplicate_I,
    Val(:real),
    "duplicate LocalMassMatrix quadratic underflow regression",
) == 4 * objective_underflow_unit

objective_underflow_unit32 = nextfloat(0.0f0)
objective_underflow_Q32 = ComplexF32[
    0.0f0 objective_underflow_unit32
    objective_underflow_unit32 0.0f0
]
objective_underflow_I32 = ComplexF32[
    ldexp(1.0f0, 60), ldexp(1.0f0, -60)
]
objective_underflow_reference32 = setprecision(BigFloat, 6656) do
    Float32(real(dot(
        Complex{BigFloat}.(objective_underflow_I32),
        Matrix{Complex{BigFloat}}(objective_underflow_Q32),
        Complex{BigFloat}.(objective_underflow_I32),
    )))
end
@test objective_underflow_reference32 ==
      2 * objective_underflow_unit32
@test compute_objective(
    objective_underflow_I32, objective_underflow_Q32,
) == objective_underflow_reference32
objective_underflow_product32 = zeros(ComplexF32, 2)
objective_underflow_product_used_fallback32 =
    DiffMoM._finite_matrix_vector_product_status!(
        objective_underflow_product32,
        objective_underflow_Q32,
        objective_underflow_I32,
        "Float32 quadratic underflow product regression",
    )
@test objective_underflow_product_used_fallback32
@test DiffMoM._quadratic_objective_from_product(
    objective_underflow_I32,
    objective_underflow_Q32,
    objective_underflow_product32,
    objective_underflow_product_used_fallback32,
) == objective_underflow_reference32

# The exceptional accumulator must have storage-bounded allocation even when
# every dense entry contributes. This case previously allocated per entry.
objective_dense_fallback_size = 96
objective_dense_fallback_signs = ComplexF64[
    isodd(index) ? 1.0 : -1.0
    for index in 1:objective_dense_fallback_size
]
objective_dense_fallback_Q =
    objective_dense_fallback_signs *
    transpose(objective_dense_fallback_signs)
objective_dense_fallback_I = ones(
    ComplexF64, objective_dense_fallback_size)
objective_dense_fallback_I[1] = nextfloat(0.0)
@test compute_objective(
    objective_dense_fallback_I, objective_dense_fallback_Q) == 1.0
compute_objective(
    objective_dense_fallback_I, objective_dense_fallback_Q)
@test @allocated(compute_objective(
    objective_dense_fallback_I,
    objective_dense_fallback_Q,
)) <= 1_000_000

objective_product_workspace = similar(objective_extreme_I)
DiffMoM._finite_matrix_vector_product!(
    objective_product_workspace,
    objective_extreme_Q,
    objective_extreme_I,
    "optimizer objective regression",
)
@test objective_product_workspace == ComplexF64[0, 0, 0, 0, 3]
objective_product_probe = similar(objective_probe_I)
DiffMoM._finite_matrix_vector_product!(
    objective_product_probe,
    objective_probe_Q,
    objective_probe_I,
    "optimizer allocation probe",
)
@test @allocated(DiffMoM._finite_matrix_vector_product!(
    objective_product_probe,
    objective_probe_Q,
    objective_probe_I,
    "optimizer allocation probe",
)) == 0
@test !DiffMoM._finite_matrix_vector_product_status!(
    objective_product_probe,
    objective_probe_Q,
    objective_probe_I,
    "optimizer status allocation probe",
)
@test @allocated(DiffMoM._finite_matrix_vector_product_status!(
    objective_product_probe,
    objective_probe_Q,
    objective_probe_I,
    "optimizer status allocation probe",
)) == 0

objective_optimizer_Z = Matrix{ComplexF64}(I, 5, 5)
objective_optimizer_Mp = [zeros(ComplexF64, 5, 5)]
objective_optimizer_theta, objective_optimizer_trace = optimize_lbfgs(
    objective_optimizer_Z,
    objective_optimizer_Mp,
    objective_extreme_I,
    objective_extreme_Q,
    [0.0];
    maxiter=1,
    verbose=false,
)
@test objective_optimizer_theta == [0.0]
@test objective_optimizer_trace[1].J == 3.0

objective_directivity_Q = zeros(ComplexF64, 5, 5)
objective_directivity_Q[5, 5] = 1.0
directivity_optimizer_theta, directivity_optimizer_trace =
    optimize_directivity(
        objective_optimizer_Z,
        objective_optimizer_Mp,
        objective_extreme_I,
        objective_extreme_Q,
        objective_directivity_Q,
        [0.0];
        maxiter=1,
        verbose=false,
    )
@test directivity_optimizer_theta == [0.0]
@test directivity_optimizer_trace[1].J == 3.0

directivity_overflow_scale = 0.5 * floatmax(Float64)
directivity_overflow_Q = reshape(
    ComplexF64[directivity_overflow_scale], 1, 1)
directivity_overflow_reference = setprecision(BigFloat, 4096) do
    objective = BigFloat(2.0) *
                BigFloat(directivity_overflow_scale) *
                BigFloat(2.0)
    @test objective > BigFloat(floatmax(Float64))
    Float64(objective / objective)
end
directivity_overflow_theta, directivity_overflow_trace =
    optimize_directivity(
        reshape(ComplexF64[1.0], 1, 1),
        [zeros(ComplexF64, 1, 1)],
        ComplexF64[2.0],
        directivity_overflow_Q,
        directivity_overflow_Q,
        [0.0];
        maxiter=1,
        verbose=false,
    )
@test directivity_overflow_reference == 1.0
@test directivity_overflow_theta == [0.0]
@test directivity_overflow_trace[1].J == 1.0

directivity_scale_Z = Matrix{ComplexF64}(I, 2, 2)
directivity_scale_Mp = [ComplexF64[1.0 0.0; 0.0 0.0]]
directivity_scale_rhs = ones(ComplexF64, 2)
directivity_scale_target = Matrix{ComplexF64}(I, 2, 2)
directivity_scale_total = ComplexF64[1.0 0.0; 0.0 0.5]
_, directivity_scale_reference_trace = optimize_directivity(
    directivity_scale_Z,
    directivity_scale_Mp,
    directivity_scale_rhs,
    directivity_scale_target,
    directivity_scale_total,
    [0.0];
    maxiter=1,
    tol=10.0,
    verbose=false,
)
directivity_common_scale = 0.75 * floatmax(Float64)
directivity_scale_theta, directivity_scale_trace = optimize_directivity(
    directivity_scale_Z,
    directivity_scale_Mp,
    directivity_scale_rhs,
    directivity_common_scale .* directivity_scale_target,
    directivity_common_scale .* directivity_scale_total,
    [0.0];
    maxiter=1,
    tol=10.0,
    verbose=false,
)
@test directivity_scale_theta == [0.0]
@test directivity_scale_trace[1].J ≈
      directivity_scale_reference_trace[1].J rtol=4eps(Float64)
@test directivity_scale_trace[1].gnorm ≈
      directivity_scale_reference_trace[1].gnorm rtol=4eps(Float64)

directivity_product_target = similar(directivity_scale_rhs)
directivity_product_total = similar(directivity_scale_rhs)
DiffMoM._directivity_products_and_objectives!(
    directivity_product_target,
    directivity_product_total,
    directivity_scale_target,
    directivity_scale_total,
    directivity_scale_rhs,
    "directivity allocation probe",
)
@test @allocated(DiffMoM._directivity_products_and_objectives!(
    directivity_product_target,
    directivity_product_total,
    directivity_scale_target,
    directivity_scale_total,
    directivity_scale_rhs,
    "directivity allocation probe",
)) == 0

# A valid ratio and gradient can remain representable even when the two QI
# vectors cannot share any lossless Float64 scale. The cold path must combine
# the normalized adjoint RHS before converting it back to Float64.
directivity_span_max = floatmax(Float64)
directivity_span_min = nextfloat(0.0)
directivity_span_current = ComplexF64[2.0, 0.5]
directivity_span_target = ComplexF64[
    0.5 * directivity_span_max 0.0
    0.0 8.0 * directivity_span_min
]
directivity_span_total = ComplexF64[
    directivity_span_max 0.0
    0.0 directivity_span_min
]
directivity_span_Mp = ComplexF64[
    0.0 0.0
    0.0 directivity_span_max
]
directivity_span_reference = setprecision(BigFloat, 8192) do
    current_big = Complex{BigFloat}.(directivity_span_current)
    target_big = Matrix{Complex{BigFloat}}(directivity_span_target)
    total_big = Matrix{Complex{BigFloat}}(directivity_span_total)
    Mp_big = Matrix{Complex{BigFloat}}(directivity_span_Mp)
    target_product = target_big * current_big
    total_product = total_big * current_big
    mass_product = Mp_big * current_big
    numerator = real(dot(current_big, target_product))
    denominator = real(dot(current_big, total_product))
    ratio = numerator / denominator
    target_gradient = 2 * real(dot(target_product, mass_product))
    total_gradient = 2 * real(dot(total_product, mass_product))
    gradient = (target_gradient - ratio * total_gradient) / denominator
    return Float64(ratio), Float64(gradient)
end
@test directivity_span_reference == (0.5, directivity_span_min)
directivity_span_theta, directivity_span_trace = optimize_directivity(
    Matrix{ComplexF64}(I, 2, 2),
    [directivity_span_Mp],
    directivity_span_current,
    directivity_span_target,
    directivity_span_total,
    [0.0];
    maxiter=1,
    tol=1.0,
    verbose=false,
)
@test directivity_span_theta == [0.0]
@test directivity_span_trace[1].J == directivity_span_reference[1]
@test directivity_span_trace[1].gnorm == directivity_span_reference[2]
directivity_span_reactive_ratio, directivity_span_reactive_gradient =
    DiffMoM._directivity_ratio_gradient_bigfloat(
        Matrix{ComplexF64}(I, 2, 2),
        [im .* directivity_span_Mp],
        directivity_span_target,
        directivity_span_total,
        directivity_span_current,
        "reactive directivity span regression";
        reactive=true,
    )
@test directivity_span_reactive_ratio == directivity_span_reference[1]
@test directivity_span_reactive_gradient == [-directivity_span_reference[2]]
@test_throws OverflowError compute_objective(
    ComplexF64[objective_extreme_scale],
    reshape(ComplexF64[1.0], 1, 1),
)

# solve_adjoint dispatch: :gmres (unpreconditioned)
lam_sa_gmres = solve_adjoint(Z_gm, Q, I_gm_direct;
                              solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
rel_sa_gmres = norm(lam_sa_gmres - lam_gm_direct) / max(norm(lam_gm_direct), 1e-30)
println("  solve_adjoint :gmres rel error: $rel_sa_gmres")
@assert rel_sa_gmres < 1e-6

# Right-preconditioned wrapper dispatch. This is important for optimization
# runs that audit the true physical residual rather than only the left-
# preconditioned residual used by Krylov convergence.
P_diag_gm = build_nearfield_preconditioner(Z_gm, mesh, rwg, 0.0;
    factorization=:diag)
I_sf_right = solve_forward(Z_gm, Vector{ComplexF64}(v);
                            solver=:gmres,
                            preconditioner=P_diag_gm,
                            gmres_precond_side=:right,
                            gmres_tol=1e-10,
                            gmres_maxiter=500,
                            check_true_residual=true)
rel_sf_right = norm(I_sf_right - I_gm_direct) / max(norm(I_gm_direct), 1e-30)
@assert rel_sf_right < 1e-6 "right-preconditioned solve_forward inaccurate: $rel_sf_right"
lam_rhs_right = solve_adjoint_rhs(Z_gm, rhs_adj_gm;
                                  solver=:gmres,
                                  preconditioner=P_diag_gm,
                                  gmres_precond_side=:right,
                                  gmres_tol=1e-10,
                                  gmres_maxiter=500,
                                  check_true_residual=true)
rel_adj_right = norm(lam_rhs_right - lam_gm_direct) / max(norm(lam_gm_direct), 1e-30)
@assert rel_adj_right < 1e-6 "right-preconditioned solve_adjoint_rhs inaccurate: $rel_adj_right"

# Optimization-facing GMRES wrappers must fail closed on unconverged solves.
Z_fail = ComplexF64[4 1 0; 1 3 1; 0 1 2]
rhs_fail = ComplexF64[1, 2, 3]
thrown_forward_unconverged = try
    solve_forward(Z_fail, rhs_fail; solver=:gmres, gmres_tol=1e-14, gmres_maxiter=1)
    false
catch e
    occursin("GMRES did not converge", sprint(showerror, e))
end
@assert thrown_forward_unconverged "Expected unconverged forward GMRES wrapper to fail closed"

thrown_adjoint_unconverged = try
    solve_adjoint_rhs(Z_fail, rhs_fail; solver=:gmres, gmres_tol=1e-14, gmres_maxiter=1)
    false
catch e
    occursin("GMRES did not converge", sprint(showerror, e))
end
@assert thrown_adjoint_unconverged "Expected unconverged adjoint GMRES wrapper to fail closed"

# The low-level public entry points enforce the same default. An explicit
# opt-out returns the partial iterate together with unsolved Krylov stats.
@test_throws ErrorException solve_gmres(
    Z_fail, rhs_fail;
    tol=1e-14, maxiter=1, memory=1,
)
_, stats_forward_partial = solve_gmres(
    Z_fail, rhs_fail;
    tol=1e-14, maxiter=1, memory=1,
    check_gmres_convergence=false,
)
@test !stats_forward_partial.solved

@test_throws ErrorException solve_gmres_adjoint(
    Z_fail, rhs_fail;
    tol=1e-14, maxiter=1, memory=1,
)
_, stats_adjoint_partial = solve_gmres_adjoint(
    Z_fail, rhs_fail;
    tol=1e-14, maxiter=1, memory=1,
    check_gmres_convergence=false,
)
@test !stats_adjoint_partial.solved

# Invalid Krylov controls fail before entering the iterative kernel.
@test_throws ArgumentError solve_gmres(Z_fail, rhs_fail; tol=NaN)
@test_throws ArgumentError solve_gmres(Z_fail, rhs_fail; maxiter=0)
@test_throws ArgumentError solve_gmres(Z_fail, rhs_fail; memory=0)
@test_throws ArgumentError solve_gmres(Z_fail, rhs_fail; precond_side=:invalid)

thrown_forward_true_residual = try
    solve_forward(Z_fail, rhs_fail; solver=:gmres, gmres_tol=1e-14,
                  gmres_maxiter=1, check_gmres_convergence=false,
                  check_true_residual=true, true_residual_factor=1.0)
    false
catch e
    occursin("true residual too large", sprint(showerror, e))
end
@assert thrown_forward_true_residual "Expected forward true-residual guard to fail closed"
@test_throws ArgumentError solve_forward(
    Z_fail, rhs_fail;
    solver=:gmres, gmres_tol=1e-14, gmres_maxiter=1,
    check_gmres_convergence=false, check_true_residual=true,
    true_residual_factor=Inf,
)
@test_throws ArgumentError solve_forward(
    Z_fail, rhs_fail;
    solver=:gmres, gmres_tol=1e-14, gmres_maxiter=1,
    check_gmres_convergence=false, check_true_residual=true,
    true_residual_factor=NaN,
)

thrown_adjoint_true_residual = try
    solve_adjoint_rhs(Z_fail, rhs_fail; solver=:gmres, gmres_tol=1e-14,
                      gmres_maxiter=1, check_gmres_convergence=false,
                      check_true_residual=true, true_residual_factor=1.0)
    false
catch e
    occursin("true residual too large", sprint(showerror, e))
end
@assert thrown_adjoint_true_residual "Expected adjoint true-residual guard to fail closed"
@test_throws ArgumentError solve_adjoint_rhs(
    Z_fail, rhs_fail;
    solver=:gmres, gmres_tol=1e-14, gmres_maxiter=1,
    check_gmres_convergence=false, check_true_residual=true,
    true_residual_factor=Inf,
)

# Matrix-free EFIE operator: A*x should match dense Z*x
A_mf = matrixfree_efie_operator(mesh, rwg, k; quad_order=3)
x_probe = randn(ComplexF64, N)
Ax_dense = Z_efie * x_probe
Ax_mf = A_mf * x_probe
rel_matvec = norm(Ax_mf - Ax_dense) / max(norm(Ax_dense), 1e-30)
println("  matrix-free matvec rel error: $rel_matvec")
@assert rel_matvec < 1e-10

for efie_op in (A_mf, adjoint(A_mf))
    efie_overlap_storage = vcat(x_probe, 0.0 + 0im)
    efie_overlap_x = view(efie_overlap_storage, 1:N)
    efie_overlap_y = view(efie_overlap_storage, 2:(N + 1))
    efie_overlap_expected = efie_op * copy(efie_overlap_x)
    mul!(efie_overlap_y, efie_op, efie_overlap_x)
    @test efie_overlap_y ≈ efie_overlap_expected rtol=1e-12

    efie_nonalias_y = zeros(ComplexF64, N)
    mul!(efie_nonalias_y, efie_op, x_probe)
    @test (@allocated mul!(efie_nonalias_y, efie_op, x_probe)) < 128
end

# GMRES on matrix-free operator
I_mf_gmres, stats_mf = solve_gmres(A_mf, v; tol=1e-8, maxiter=300)
rel_mf = norm(I_mf_gmres - I_pec) / max(norm(I_pec), 1e-30)
println("  matrix-free GMRES rel error: $rel_mf  iters: $(stats_mf.niter)")
@assert rel_mf < 1e-6

# solve_forward / solve_adjoint dispatch on matrix-free operator
I_sf_mf = solve_forward(A_mf, v; solver=:gmres, gmres_tol=1e-8, gmres_maxiter=300)
rel_sf_mf = norm(I_sf_mf - I_pec) / max(norm(I_pec), 1e-30)
@assert rel_sf_mf < 1e-6

lam_sa_mf = solve_adjoint(A_mf, Q, I_pec; solver=:gmres, gmres_tol=1e-8, gmres_maxiter=300)
lam_sa_mf_ref = Z_efie' \ (Q * I_pec)
rel_sa_mf = norm(lam_sa_mf - lam_sa_mf_ref) / max(norm(lam_sa_mf_ref), 1e-30)
println("  matrix-free adjoint GMRES rel error: $rel_sa_mf")
@assert rel_sa_mf < 1e-6

thrown_direct_operator = try
    solve_forward(A_mf, v; solver=:direct)
    false
catch
    true
end
@assert thrown_direct_operator "Expected direct solve failure on matrix-free operator"

# Bad solver symbol should error
thrown_bad_solver = try
    solve_forward(Z_gm, Vector{ComplexF64}(v); solver=:unknown)
    false
catch
    true
end
@assert thrown_bad_solver "Expected error for unknown solver"

thrown_bad_solver_adj = try
    solve_adjoint(Z_gm, Q, I_gm_direct; solver=:unknown)
    false
catch
    true
end
@assert thrown_bad_solver_adj "Expected error for unknown adjoint solver"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 18: GMRES adjoint gradient verification
# ─────────────────────────────────────────────────
println("\n── Test 18: GMRES adjoint gradient verification ──")

# Resistive case: forward and adjoint with GMRES (unpreconditioned)
I_gm_res = solve_forward(Z_gm, Vector{ComplexF64}(v);
                           solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
lam_gm_res = solve_adjoint(Z_gm, Q, I_gm_res;
                             solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
g_adj_gmres = gradient_impedance(Mp, I_gm_res, lam_gm_res)

# Compare GMRES gradient against finite differences
println("  Checking GMRES adjoint gradient vs central FD (h=1e-5)...")
rel_errors_gm = Float64[]
n_check_gm = min(Nt, 10)
for p in 1:n_check_gm
    g_fd = fd_grad(J_of_theta, theta_real, p; h=1e-5)
    rel_err = abs(g_adj_gmres[p] - g_fd) / max(abs(g_adj_gmres[p]), abs(g_fd), 1e-30)
    push!(rel_errors_gm, rel_err)
end
max_rel_err_gm = maximum(rel_errors_gm)
println("  Max rel error (GMRES adjoint vs FD): $max_rel_err_gm")
@assert max_rel_err_gm < 1e-3 "GMRES gradient verification FAILED: max rel error = $max_rel_err_gm"

# Also compare GMRES gradient against direct gradient
rel_gm_vs_direct = norm(g_adj_gmres - g_adj) / max(norm(g_adj), 1e-30)
println("  GMRES vs direct gradient rel diff: $rel_gm_vs_direct")
@assert rel_gm_vs_direct < 1e-4 "GMRES gradient diverges from direct: $rel_gm_vs_direct"

# Reactive case
theta_reac = fill(150.0, Nt)
Z_reac = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_reac; reactive=true))
I_reac_direct = Z_reac \ v

lam_reac_dir = solve_adjoint(Z_reac, Q, I_reac_direct)
g_reac_dir = gradient_impedance(Mp, I_reac_direct, lam_reac_dir; reactive=true)

I_reac_gm = solve_forward(Z_reac, Vector{ComplexF64}(v);
                            solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
lam_reac_gm = solve_adjoint(Z_reac, Q, I_reac_gm;
                              solver=:gmres, gmres_tol=1e-10, gmres_maxiter=500)
g_reac_gm = gradient_impedance(Mp, I_reac_gm, lam_reac_gm; reactive=true)

function J_of_theta_reac(theta_vec)
    Z_t = copy(Z_efie)
    for p in eachindex(theta_vec)
        Z_t .-= (1im * theta_vec[p]) .* Mp[p]
    end
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

rel_errors_reac = Float64[]
n_check_reac = min(Nt, 5)
for p in 1:n_check_reac
    g_fd = fd_grad(J_of_theta_reac, theta_reac, p; h=1e-5)
    rel_err_gm = abs(g_reac_gm[p] - g_fd) / max(abs(g_fd), 1e-30)
    push!(rel_errors_reac, rel_err_gm)
end
max_rel_err_reac = maximum(rel_errors_reac)
println("  Max rel error (GMRES reactive gradient vs FD): $max_rel_err_reac")
@assert max_rel_err_reac < 1e-3 "Reactive GMRES gradient failed: $max_rel_err_reac"

rel_reac_gm_vs_dir = norm(g_reac_gm - g_reac_dir) / max(norm(g_reac_dir), 1e-30)
println("  Reactive gradient GMRES vs direct: $rel_reac_gm_vs_dir")
@assert rel_reac_gm_vs_dir < 1e-4

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 19: Near-field sparse preconditioner
# ─────────────────────────────────────────────────
println("\n── Test 19: Near-field sparse preconditioner ──")

# Test rwg_centers
centers = rwg_centers(mesh, rwg)
@assert length(centers) == N
@assert all(c -> length(c) == 3, centers)

# Averaging same-sign, upper-range triangle centroids must not overflow.
rwg_center_translation = 1.0e308
rwg_center_mesh = TriMesh(
    Float64[
        rwg_center_translation rwg_center_translation rwg_center_translation rwg_center_translation
        0 1 0 1
        0 0 1 1
    ],
    Int[1 2; 2 4; 3 3],
)
rwg_center_data = build_rwg(rwg_center_mesh)
rwg_center_extreme = rwg_centers(rwg_center_mesh, rwg_center_data)
@test all(center -> all(isfinite, center), rwg_center_extreme)
@test all(center -> center[1] == rwg_center_translation, rwg_center_extreme)

# Build near-field preconditioner at 1.0λ cutoff
P_nf = build_nearfield_preconditioner(Z_efie, mesh, rwg, lambda0)
@assert P_nf.cutoff == lambda0
@assert 0.0 < P_nf.nnz_ratio <= 1.0

# Compare default spatial search against brute-force reference
P_nf_bruteforce = build_nearfield_preconditioner(
    Z_efie, mesh, rwg, lambda0; neighbor_search=:bruteforce
)
@assert abs(P_nf.nnz_ratio - P_nf_bruteforce.nnz_ratio) < 1e-12
x_nf_cmp = randn(ComplexF64, N)
y_nf_spatial = NearFieldOperator(P_nf) * x_nf_cmp
y_nf_bruteforce = NearFieldOperator(P_nf_bruteforce) * x_nf_cmp
rel_nf_spatial_vs_brute = norm(y_nf_spatial - y_nf_bruteforce) / max(norm(y_nf_bruteforce), 1e-30)
println("  NF spatial vs brute rel diff: $rel_nf_spatial_vs_brute")
@assert rel_nf_spatial_vs_brute < 1e-12

# Spatial hashing must be translation-invariant even when coordinate/cutoff
# ratios exceed Int, and distance comparisons must not turn Inf <= Inf into a
# false near-field classification.
translated_nf_xyz = copy(mesh.xyz)
translated_nf_xyz[1, :] .+= 1.0e9
translated_nf_mesh = TriMesh(translated_nf_xyz, copy(mesh.tri))
translated_nf_rwg = build_rwg(translated_nf_mesh)
tiny_cutoff = 1.0e-12
P_nf_translated = build_nearfield_preconditioner(
    Z_efie, translated_nf_mesh, translated_nf_rwg, tiny_cutoff)
P_nf_translated_brute = build_nearfield_preconditioner(
    Z_efie, translated_nf_mesh, translated_nf_rwg, tiny_cutoff;
    neighbor_search=:bruteforce)
@test P_nf_translated.nnz_ratio == P_nf_translated_brute.nnz_ratio

extreme_centers = Vec3[Vec3(0.0, 0.0, 0.0), Vec3(1.0e200, 0.0, 0.0)]
extreme_I, extreme_J, _ = DiffMoM._nearfield_triplets_bruteforce(
    extreme_centers, 1.0e190, (m, n) -> 1.0 + 0im)
@test collect(zip(extreme_I, extreme_J)) == [(1, 1), (2, 2)]

# A rounded distance exactly equal to the cutoff is ambiguous: settle the
# inclusion from the exact stored centers so a just-outside pair is excluded
# while its just-inside counterpart remains included.
nearfield_boundary_x = prevfloat(1.0)
nearfield_boundary_y = ldexp(1.0, -26)
nearfield_rounded_outside = Vec3[
    Vec3(0.0, 0.0, 0.0),
    Vec3(nearfield_boundary_x, nearfield_boundary_y, 0.0),
]
nearfield_rounded_inside = Vec3[
    Vec3(0.0, 0.0, 0.0),
    Vec3(nearfield_boundary_x, prevfloat(nearfield_boundary_y), 0.0),
]
@test hypot(nearfield_rounded_outside[2]...) == 1.0
@test hypot(nearfield_rounded_inside[2]...) == 1.0
outside_I, outside_J, _ = DiffMoM._nearfield_triplets_bruteforce(
    nearfield_rounded_outside, 1.0, (m, n) -> ComplexF64(m, n))
inside_I, inside_J, _ = DiffMoM._nearfield_triplets_bruteforce(
    nearfield_rounded_inside, 1.0, (m, n) -> ComplexF64(m, n))
@test collect(zip(outside_I, outside_J)) == [(1, 1), (2, 2)]
@test collect(zip(inside_I, inside_J)) ==
      [(1, 1), (1, 2), (2, 1), (2, 2)]

# Triplet payload limits are enforced before all-pairs preallocation, and the
# exact raw boundary (two Int indices plus one ComplexF64 per entry) is usable.
nearfield_pair_bytes = 2 * sizeof(Int) + sizeof(ComplexF64)
bounded_I, bounded_J, bounded_V = DiffMoM._nearfield_triplets_bruteforce(
    extreme_centers, Inf, (m, n) -> ComplexF64(m, n);
    max_triplet_bytes=4 * nearfield_pair_bytes)
@test length(bounded_I) == length(bounded_J) == length(bounded_V) == 4
@test_throws ArgumentError DiffMoM._nearfield_triplets_bruteforce(
    extreme_centers, Inf, (m, n) -> ComplexF64(m, n);
    max_triplet_bytes=4 * nearfield_pair_bytes - 1)
@test_throws ArgumentError build_nearfield_preconditioner(
    Z_efie, mesh, rwg, lambda0;
    max_triplet_bytes=max(1, N * nearfield_pair_bytes - 1))

# Build near-field preconditioner without dense Z (matrix-free and geometry paths)
P_nf_mf = build_nearfield_preconditioner(A_mf, lambda0)
@assert P_nf_mf.cutoff == lambda0
@assert 0.0 < P_nf_mf.nnz_ratio <= 1.0
@assert abs(P_nf_mf.nnz_ratio - P_nf.nnz_ratio) < 1e-12

# A one-matrix Green workspace disables retention while preserving the exact
# assembled preconditioner; a smaller workspace fails before that matrix is
# allocated when a regular triangle pair is encountered.
nearfield_green_bytes = sizeof(ComplexF64) * A_mf.cache.Nq^2
P_nf_mf_scratch = build_nearfield_preconditioner(
    A_mf, lambda0;
    max_green_cache_bytes=nearfield_green_bytes,
    max_green_cache_entries=1)
@test P_nf_mf_scratch.nnz_ratio == P_nf_mf.nnz_ratio
scratch_probe = NearFieldOperator(P_nf_mf_scratch) * x_nf_cmp
@test isapprox(
    scratch_probe, NearFieldOperator(P_nf_mf) * x_nf_cmp;
    rtol=1e-12, atol=1e-12)
@test_throws ArgumentError build_nearfield_preconditioner(
    A_mf, lambda0;
    max_green_cache_bytes=max(1, nearfield_green_bytes - 1))

P_nf_mf_bruteforce = build_nearfield_preconditioner(
    A_mf, lambda0; neighbor_search=:bruteforce
)
@assert abs(P_nf_mf.nnz_ratio - P_nf_mf_bruteforce.nnz_ratio) < 1e-12

P_nf_geom = build_nearfield_preconditioner(mesh, rwg, k, lambda0; quad_order=3)
@assert P_nf_geom.cutoff == lambda0
@assert abs(P_nf_geom.nnz_ratio - P_nf.nnz_ratio) < 1e-12

# Invalid neighbor-search mode should error
thrown_bad_neighbor_search = try
    build_nearfield_preconditioner(Z_efie, mesh, rwg, lambda0; neighbor_search=:invalid_mode)
    false
catch
    true
end
@assert thrown_bad_neighbor_search "Expected error for invalid neighbor_search mode"

# Invalid factorization mode should error
thrown_bad_factorization = try
    build_nearfield_preconditioner(A_mf, lambda0; factorization=:invalid_mode)
    false
catch
    true
end
@assert thrown_bad_factorization "Expected error for invalid factorization mode"

@test_throws ArgumentError build_nearfield_preconditioner(
    Z_efie, mesh, rwg, NaN)
@test_throws ArgumentError build_nearfield_preconditioner(
    Z_efie, mesh, rwg, -1.0)
@test_throws ArgumentError build_nearfield_preconditioner(
    Z_efie, mesh, rwg, lambda0;
    neighbor_search=:invalid_mode, factorization=:diag)
Z_nf_nonfinite = copy(Z_efie)
Z_nf_nonfinite[1, 1] = NaN + 0im
@test_throws ArgumentError build_nearfield_preconditioner(
    Z_nf_nonfinite, mesh, rwg, lambda0)
@test_throws ArgumentError build_nearfield_preconditioner(
    Z_nf_nonfinite, mesh, rwg, lambda0; factorization=:diag)

# Diagonal/Jacobi preconditioner path
P_diag = build_nearfield_preconditioner(A_mf, lambda0; factorization=:diag)
@assert P_diag isa DiagonalPreconditionerData
@assert P_diag.cutoff == lambda0
@assert 0.0 < P_diag.nnz_ratio <= 1.0
M_diag = NearFieldOperator(P_diag)
M_diag_adj = NearFieldAdjointOperator(P_diag)
@test_throws DimensionMismatch M_diag * ones(ComplexF64, N - 1)
@test_throws DimensionMismatch M_diag * ones(ComplexF64, N + 1)
@test_throws DimensionMismatch M_diag_adj * ones(ComplexF64, N - 1)
@test_throws DimensionMismatch mul!(
    zeros(ComplexF64, N - 1), M_diag, ones(ComplexF64, N - 1))
@test_throws DimensionMismatch mul!(
    zeros(ComplexF64, N), M_diag_adj, ones(ComplexF64, N - 1))
I_diag, stats_diag = solve_gmres(A_mf, v; preconditioner=P_diag, tol=1e-8, maxiter=300)
rel_diag = norm(I_diag - I_pec) / max(norm(I_pec), 1e-30)
println("  Diag precond + matrix-free rel error: $rel_diag  iters: $(stats_diag.niter)")
@assert rel_diag < 1e-6 "Diagonal-preconditioned matrix-free solve inaccurate: $rel_diag"

# GMRES with near-field preconditioner
I_nf, stats_nf = solve_gmres(Z_efie, v; preconditioner=P_nf, tol=1e-8, maxiter=200)
rel_nf = norm(I_nf - I_pec) / max(norm(I_pec), 1e-30)
println("  NF precond (1.0λ) rel error: $rel_nf  iters: $(stats_nf.niter)")
@assert rel_nf < 1e-6 "Near-field preconditioned solve inaccurate: $rel_nf"

# GMRES on matrix-free operator with near-field preconditioner built without dense Z
I_nf_mf, stats_nf_mf = solve_gmres(A_mf, v; preconditioner=P_nf_mf, tol=1e-8, maxiter=300)
rel_nf_mf = norm(I_nf_mf - I_pec) / max(norm(I_pec), 1e-30)
println("  NF precond + matrix-free rel error: $rel_nf_mf  iters: $(stats_nf_mf.niter)")
@assert rel_nf_mf < 1e-6 "Matrix-free near-field preconditioned solve inaccurate: $rel_nf_mf"

# Compare iteration count: near-field should help vs unpreconditioned
I_nop_nf, stats_nop_nf = solve_gmres(Z_efie, v; tol=1e-8, maxiter=200)
println("  Iterations: no_precond=$(stats_nop_nf.niter), NF=$(stats_nf.niter)")

# NearFieldOperator / NearFieldAdjointOperator wrappers
M_nf = NearFieldOperator(P_nf)
@assert size(M_nf) == (N, N)
@assert eltype(M_nf) == ComplexF64
y_nf = M_nf * v
@assert length(y_nf) == N

M_nf_adj = NearFieldAdjointOperator(P_nf)
@assert size(M_nf_adj) == (N, N)
y_nf_adj = M_nf_adj * v

# Adjoint consistency: ⟨P⁻¹x, y⟩ ≈ ⟨x, P⁻ᴴy⟩
x_test = randn(ComplexF64, N)
y_test = randn(ComplexF64, N)
lhs_nf = dot(M_nf * x_test, y_test)
rhs_nf = dot(x_test, M_nf_adj * y_test)
adj_err = abs(lhs_nf - rhs_nf) / max(abs(lhs_nf), 1e-30)
println("  Adjoint consistency: $adj_err")
@assert adj_err < 1e-12 "Near-field adjoint inconsistent: $adj_err"

# GMRES adjoint solve with near-field
rhs_adj_nf = Q * I_pec
lambda_nf, stats_adj_nf = solve_gmres_adjoint(Z_efie, rhs_adj_nf;
                                                preconditioner=P_nf, tol=1e-8, maxiter=200)
lambda_direct_nf = Z_efie' \ rhs_adj_nf
rel_adj_nf = norm(lambda_nf - lambda_direct_nf) / max(norm(lambda_direct_nf), 1e-30)
println("  NF adjoint solve rel error: $rel_adj_nf  iters: $(stats_adj_nf.niter)")
@assert rel_adj_nf < 1e-6 "Near-field adjoint solve inaccurate: $rel_adj_nf"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 20: Right preconditioning
# ─────────────────────────────────────────────────
println("\n── Test 20: Right preconditioning ──")

# Right-preconditioned GMRES with near-field
I_nf_right, stats_nf_right = solve_gmres(Z_efie, v;
                                           preconditioner=P_nf, precond_side=:right,
                                           tol=1e-8, maxiter=200)
rel_nf_right = norm(I_nf_right - I_pec) / max(norm(I_pec), 1e-30)
println("  NF right precond rel error: $rel_nf_right  iters: $(stats_nf_right.niter)")
@assert rel_nf_right < 1e-6 "NF right-preconditioned solve inaccurate: $rel_nf_right"

# Right-preconditioned adjoint with near-field
lambda_nf_right, stats_adj_nf_right = solve_gmres_adjoint(Z_efie, rhs_adj_nf;
                                                             preconditioner=P_nf, precond_side=:right,
                                                             tol=1e-8, maxiter=200)
rel_adj_nf_right = norm(lambda_nf_right - lambda_direct_nf) / max(norm(lambda_direct_nf), 1e-30)
println("  NF right adjoint rel error: $rel_adj_nf_right  iters: $(stats_adj_nf_right.niter)")
@assert rel_adj_nf_right < 1e-6 "NF right adjoint solve inaccurate: $rel_adj_nf_right"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 21: Optimization with GMRES solver
# ─────────────────────────────────────────────────
println("\n── Test 21: Optimization with GMRES solver ──")

theta_init_gm = fill(300.0, Nt)
theta_opt_gm, trace_gm = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_gm;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:gmres, gmres_tol=1e-8, gmres_maxiter=300,
)
@assert length(trace_gm) >= 2 "GMRES optimization should run at least 2 iterations"

# Direct solver comparison
theta_opt_dir, trace_dir = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_gm;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:direct,
)
rel_J0 = abs(trace_gm[1].J - trace_dir[1].J) / max(abs(trace_dir[1].J), 1e-30)
println("  First-iteration J agreement (lbfgs): $rel_J0")
@assert rel_J0 < 1e-4 "GMRES and direct first-iter J disagree: $rel_J0"

# optimize_directivity with GMRES
Q_total_test = build_Q(G_mat, grid, pol_mat)
theta_opt_dir_d, trace_dir_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_gm;
    maxiter=5, tol=1e-8, verbose=false, solver=:direct,
)
theta_opt_gm_d, trace_gm_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_gm;
    maxiter=5, tol=1e-8, verbose=false,
    solver=:gmres, gmres_tol=1e-8, gmres_maxiter=300,
)
@assert length(trace_gm_d) >= 2
rel_J0_d = abs(trace_gm_d[1].J - trace_dir_d[1].J) / max(abs(trace_dir_d[1].J), 1e-30)
println("  First-iteration J_ratio agreement (directivity): $rel_J0_d")
@assert rel_J0_d < 1e-3

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 22: Regression — default solver=:direct unchanged
# ─────────────────────────────────────────────────
println("\n── Test 22: Regression — default solver=:direct ──")

# solve_forward default should match Z \ v exactly
I_reg_sf = solve_forward(Z_gm, Vector{ComplexF64}(v))
I_reg_direct = Z_gm \ v
rel_reg = norm(I_reg_sf - I_reg_direct) / max(norm(I_reg_direct), 1e-30)
@assert rel_reg < 1e-12 "Default solve_forward regression: $rel_reg"

# solve_adjoint default should match Z' \ (Q*I)
lam_reg_sa = solve_adjoint(Z_gm, Q, I_reg_direct)
lam_reg_dir = Z_gm' \ (Q * I_reg_direct)
rel_reg_adj = norm(lam_reg_sa - lam_reg_dir) / max(norm(lam_reg_dir), 1e-30)
@assert rel_reg_adj < 1e-12 "Default solve_adjoint regression: $rel_reg_adj"

# optimize_lbfgs default (solver=:direct) should still work
theta_reg_init = fill(300.0, Nt)
theta_reg_opt, trace_reg = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_reg_init;
    maxiter=3, tol=1e-8, verbose=false,
)
@assert length(trace_reg) >= 2

println("  All defaults unchanged")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 23: NF-preconditioned optimization
# ─────────────────────────────────────────────────
println("\n── Test 23: NF-preconditioned optimization ──")

# Build NF preconditioner from PEC EFIE matrix
P_nf_opt = build_nearfield_preconditioner(Z_efie, mesh, rwg, lambda0)
println("  NF preconditioner: cutoff=$(round(P_nf_opt.cutoff, sigdigits=3)), nnz=$(round(P_nf_opt.nnz_ratio*100, digits=1))%")

theta_init_nf = fill(300.0, Nt)

# optimize_lbfgs with NF-preconditioned GMRES
theta_opt_nf, trace_nf = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_nf;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:gmres, nf_preconditioner=P_nf_opt,
    gmres_tol=1e-8, gmres_maxiter=300,
)
@assert length(trace_nf) >= 2 "NF-preconditioned optimization should run at least 2 iterations"

# First-iteration J should agree with direct solver
theta_opt_dir_nf, trace_dir_nf = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_nf;
    maxiter=8, tol=1e-8, alpha0=0.01, verbose=false,
    solver=:direct,
)
rel_J0_nf = abs(trace_nf[1].J - trace_dir_nf[1].J) / max(abs(trace_dir_nf[1].J), 1e-30)
println("  First-iteration J agreement (lbfgs, NF): $rel_J0_nf")
@assert rel_J0_nf < 1e-4 "NF-preconditioned first-iter J disagrees: $rel_J0_nf"

# Gradient check: NF-preconditioned GMRES gradient vs FD
Z_nf_check = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_init_nf))
I_nf_check = solve_forward(Z_nf_check, Vector{ComplexF64}(v);
                             solver=:gmres, preconditioner=P_nf_opt,
                             gmres_tol=1e-10, gmres_maxiter=300)
lam_nf_check = solve_adjoint(Z_nf_check, Q, I_nf_check;
                               solver=:gmres, preconditioner=P_nf_opt,
                               gmres_tol=1e-10, gmres_maxiter=300)
g_nf_check = gradient_impedance(Mp, I_nf_check, lam_nf_check)

function J_of_theta_nf(theta_vec)
    Z_t = Matrix{ComplexF64}(assemble_full_Z(Z_efie, Mp, theta_vec))
    I_t = Z_t \ v
    return real(dot(I_t, Q * I_t))
end

rel_errors_nf = Float64[]
n_check_nf = min(Nt, 5)
for p in 1:n_check_nf
    g_fd = fd_grad(J_of_theta_nf, theta_init_nf, p; h=1e-5)
    rel_err = abs(g_nf_check[p] - g_fd) / max(abs(g_fd), abs(g_nf_check[p]), 1e-30)
    push!(rel_errors_nf, rel_err)
end
max_rel_err_nf = maximum(rel_errors_nf)
println("  NF-preconditioned gradient max rel error vs FD: $max_rel_err_nf")
@assert max_rel_err_nf < 1e-3 "NF-preconditioned gradient inaccurate: $max_rel_err_nf"

# optimize_directivity with NF-preconditioned GMRES
theta_opt_dir_nf_d, trace_dir_nf_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_nf;
    maxiter=5, tol=1e-8, verbose=false, solver=:direct,
)
theta_opt_nf_d, trace_nf_d = optimize_directivity(
    Z_efie, Mp, v, Q, Q_total_test, theta_init_nf;
    maxiter=5, tol=1e-8, verbose=false,
    solver=:gmres, nf_preconditioner=P_nf_opt,
    gmres_tol=1e-8, gmres_maxiter=300,
)
@assert length(trace_nf_d) >= 2
rel_J0_nf_d = abs(trace_nf_d[1].J - trace_dir_nf_d[1].J) / max(abs(trace_dir_nf_d[1].J), 1e-30)
println("  First-iteration J_ratio agreement (directivity, NF): $rel_J0_nf_d")
@assert rel_J0_nf_d < 1e-3

# nf_preconditioner=nothing should behave same as unpreconditioned
theta_opt_none, trace_none = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta_init_nf;
    maxiter=3, tol=1e-8, verbose=false,
    solver=:gmres, nf_preconditioner=nothing,
    gmres_tol=1e-8, gmres_maxiter=300,
)
rel_J0_none = abs(trace_none[1].J - trace_gm[1].J) / max(abs(trace_gm[1].J), 1e-30)
println("  nf_preconditioner=nothing matches unpreconditioned: $rel_J0_none")
@assert rel_J0_none < 1e-10

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 24: Cluster tree construction
# ─────────────────────────────────────────────────
println("\n── Test 24: Cluster tree construction ──")

centers_ct = rwg_centers(mesh, rwg)
@assert length(centers_ct) == N

tree_ct = build_cluster_tree(centers_ct; leaf_size=8)
cluster_node_bound, cluster_storage_bound =
    DiffMoM._cluster_tree_resource_bounds(length(centers_ct), 8)
@test_throws ArgumentError build_cluster_tree(
    centers_ct; leaf_size=8, max_nodes=cluster_node_bound - 1)
@test_throws ArgumentError build_cluster_tree(
    centers_ct; leaf_size=8,
    max_nodes=cluster_node_bound,
    max_storage_bytes=cluster_storage_bound - 1)
@test build_cluster_tree(
    centers_ct; leaf_size=8,
    max_nodes=cluster_node_bound,
    max_storage_bytes=cluster_storage_bound).perm == tree_ct.perm
cluster_nodes_actual = length(tree_ct.nodes)
@test build_cluster_tree(
    centers_ct; leaf_size=8,
    max_nodes=cluster_nodes_actual,
    max_storage_bytes=cluster_storage_bound).perm == tree_ct.perm
@assert length(tree_ct.perm) == N
@assert length(tree_ct.iperm) == N
@test_throws ArgumentError build_cluster_tree(Vec3[])
@test_throws ArgumentError build_cluster_tree(
    [Vec3(NaN, 0.0, 0.0)])
@test_throws ArgumentError build_cluster_tree(
    [Vec3(-floatmax(Float64), 0.0, 0.0),
     Vec3(floatmax(Float64), 0.0, 0.0)])
@test_throws ArgumentError build_cluster_tree(
    centers_ct; leaf_size=0)

# perm and iperm should be inverses
for i in 1:N
    @assert tree_ct.iperm[tree_ct.perm[i]] == i "perm/iperm inverse check failed at i=$i"
end

# All indices 1:N should appear in perm exactly once
@assert sort(tree_ct.perm) == collect(1:N) "perm is not a valid permutation"

# Leaf nodes should have size <= leaf_size
for i in eachindex(tree_ct.nodes)
    if is_leaf(tree_ct, i)
        @assert length(tree_ct.nodes[i].indices) <= tree_ct.leaf_size "Leaf too large at node $i"
    end
end

# Root should cover all indices
@assert tree_ct.nodes[1].indices == 1:N "Root does not cover all indices"

# Admissibility should be false for overlapping clusters, true for well-separated ones
leaves_ct = leaf_nodes(tree_ct)
if length(leaves_ct) >= 2
    # Self-block is never admissible
    @assert !is_admissible(tree_ct, leaves_ct[1], leaves_ct[1])
end
@test_throws ArgumentError is_admissible(
    tree_ct, 1, 1; eta=NaN)
@test_throws ArgumentError is_admissible(
    tree_ct, 1, 1; eta=0.0)

# Rounded box separation can equal the admissibility boundary on both sides.
# Resolve the exact stored bounding boxes so a just-too-close block stays dense.
cluster_boundary_x = prevfloat(1.0)
cluster_boundary_y = ldexp(1.0, -26)
function boundary_cluster_tree(y)
    nodes = ClusterNode[
        ClusterNode(1:1, Vec3(0.0, 0.0, 0.0),
                    Vec3(0.0, 0.0, 1.0), 0, 0, 0),
        ClusterNode(2:2, Vec3(cluster_boundary_x, y, 0.0),
                    Vec3(cluster_boundary_x, y, 1.0), 0, 0, 0),
    ]
    return ClusterTree(nodes, [1, 2], [1, 2], 1)
end
cluster_rounded_inside =
    boundary_cluster_tree(prevfloat(cluster_boundary_y))
cluster_rounded_outside = boundary_cluster_tree(cluster_boundary_y)
@test cluster_distance(cluster_rounded_inside, 1, 2) == 1.0
@test cluster_distance(cluster_rounded_outside, 1, 2) == 1.0
@test !is_admissible(cluster_rounded_inside, 1, 2; eta=1.0)
@test is_admissible(cluster_rounded_outside, 1, 2; eta=1.0)

println("  Tree: $(length(tree_ct.nodes)) nodes, $(length(leaves_ct)) leaves")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 25: ACA low-rank approximation accuracy
# ─────────────────────────────────────────────────
println("\n── Test 25: ACA low-rank approximation accuracy ──")

# Build EFIE cache for ACA entry evaluation
cache_aca = DiffMoM._build_efie_cache(mesh, rwg, k; quad_order=3, eta0=eta0)
for invalid_aca_tol in (0.0, -1.0, Inf, NaN)
    @test_throws ArgumentError aca_lowrank(
        cache_aca, [1], [1]; tol=invalid_aca_tol, max_rank=1)
end
for invalid_aca_rank in (0, -1)
    @test_throws ArgumentError aca_lowrank(
        cache_aca, [1], [1]; tol=1e-6, max_rank=invalid_aca_rank)
end
@test_throws ArgumentError aca_lowrank(
    cache_aca, [1], [1]; tol=1e-6, max_rank=1,
    max_output_bytes=2 * sizeof(ComplexF64) - 1)

# A relative ACA tolerance must be invariant under a common physical scale.
# This 3-RWG plate produces four admissible 1×1 far blocks; the former
# absolute pivot floor erased every one at the smaller impedance.
aca_scale_mesh = make_rect_plate(1.0, 1.0, 1, 2)
aca_scale_rwg = build_rwg(aca_scale_mesh)
aca_scale_errors = Float64[]
aca_scale_ranks = Vector{Int}[]
for aca_scale_eta0 in (1.0, 1e-100)
    aca_scale_operator = build_aca_operator(
        aca_scale_mesh,
        aca_scale_rwg,
        2π;
        leaf_size=1,
        eta=1.5,
        aca_tol=1e-6,
        max_rank=3,
        quad_order=1,
        eta0=aca_scale_eta0,
    )
    aca_scale_dense = assemble_Z_efie(
        aca_scale_mesh,
        aca_scale_rwg,
        2π;
        quad_order=1,
        eta0=aca_scale_eta0,
    )
    aca_scale_approximation = _aca_dense_approximation(aca_scale_operator)
    push!(aca_scale_errors,
          norm(aca_scale_approximation - aca_scale_dense) /
          norm(aca_scale_dense))
    push!(aca_scale_ranks,
          [size(block.U, 2) for block in aca_scale_operator.lowrank_blocks])
end
@test aca_scale_ranks[1] == aca_scale_ranks[2]
@test !isempty(aca_scale_ranks[2])
@test all(>(0), aca_scale_ranks[2])
@test maximum(aca_scale_errors) <= 1e-12

# The relative stopping rule must also remain scale-invariant when squaring a
# representable factor norm would underflow or overflow. Exercise a block that
# genuinely converges before its rank limit; 1×1 blocks cannot expose this.
aca_convergence_mesh = make_rect_plate(2.0, 1.0, 6, 3)
aca_convergence_rwg = build_rwg(aca_convergence_mesh)
aca_convergence_operator = build_aca_operator(
    aca_convergence_mesh,
    aca_convergence_rwg,
    2π;
    leaf_size=4,
    eta=1.0,
    aca_tol=0.1,
    max_rank=6,
    quad_order=1,
    eta0=1.0,
)
aca_converged_block_index = findfirst(
    block -> size(block.U, 2) <
             min(size(block.U, 1), size(block.V, 1), 6),
    aca_convergence_operator.lowrank_blocks,
)
@test aca_converged_block_index !== nothing
aca_converged_block =
    aca_convergence_operator.lowrank_blocks[aca_converged_block_index]
aca_convergence_rows =
    aca_convergence_operator.tree.perm[aca_converged_block.row_range]
aca_convergence_cols =
    aca_convergence_operator.tree.perm[aca_converged_block.col_range]
aca_convergence_reference =
    aca_converged_block.U * aca_converged_block.V'
for aca_extreme_scale in (1e-200, 1e200)
    aca_scaled_cache = DiffMoM._efie_cache_with_prefactors(
        aca_convergence_operator.cache, 2π, aca_extreme_scale)
    aca_scaled_U, aca_scaled_V = aca_lowrank(
        aca_scaled_cache,
        aca_convergence_rows,
        aca_convergence_cols;
        tol=0.1,
        max_rank=6,
    )
    @test size(aca_scaled_U, 2) == size(aca_converged_block.U, 2)
    @test (aca_scaled_U * aca_scaled_V') / aca_extreme_scale ≈
          aca_convergence_reference rtol=5e-13 atol=0.0
end

# Find two well-separated leaf clusters for testing
tree_aca = build_cluster_tree(centers_ct; leaf_size=8)
leaves_aca = leaf_nodes(tree_aca)
found_admissible = false
global aca_row_node = 0
global aca_col_node = 0
for i_leaf in leaves_aca
    for j_leaf in leaves_aca
        if i_leaf != j_leaf && is_admissible(tree_aca, i_leaf, j_leaf; eta=1.5)
            global aca_row_node = i_leaf
            global aca_col_node = j_leaf
            global found_admissible = true
            break
        end
    end
    found_admissible && break
end

if found_admissible
    rn_aca = tree_aca.nodes[aca_row_node]
    cn_aca = tree_aca.nodes[aca_col_node]
    row_idx = [tree_aca.perm[k] for k in rn_aca.indices]
    col_idx = [tree_aca.perm[k] for k in cn_aca.indices]

    # Compute dense sub-block for reference
    m_blk = length(row_idx)
    n_blk = length(col_idx)
    Z_sub = Matrix{ComplexF64}(undef, m_blk, n_blk)
    for jj in 1:n_blk
        for ii in 1:m_blk
            Z_sub[ii, jj] = DiffMoM._efie_entry(cache_aca, row_idx[ii], col_idx[jj])
        end
    end

    # ACA low-rank approximation
    U_aca, V_aca = aca_lowrank(cache_aca, row_idx, col_idx; tol=1e-6, max_rank=30)
    rank_aca = size(U_aca, 2)
    approx_aca = U_aca * V_aca'
    err_aca = norm(approx_aca - Z_sub) / max(norm(Z_sub), 1e-30)

    println("  Block size: $(m_blk) x $(n_blk), ACA rank: $rank_aca, rel error: $err_aca")
    @assert err_aca < 1e-4 "ACA approximation too inaccurate: $err_aca"
    @assert rank_aca < min(m_blk, n_blk) "ACA should compress: rank=$rank_aca >= min($m_blk,$n_blk)"
else
    # Fallback: the default smoke-test mesh can be too small to produce an
    # admissible far block. Use a coarser-quadrature larger plate so ACA's
    # actual low-rank stopping logic is still tested deterministically.
    let
        mesh_aca_fb = make_rect_plate(1.0, 1.0, 20, 20)
        rwg_aca_fb = build_rwg(mesh_aca_fb)
        centers_aca_fb = rwg_centers(mesh_aca_fb, rwg_aca_fb)
        cache_aca_fb = DiffMoM._build_efie_cache(
            mesh_aca_fb, rwg_aca_fb, 2π; quad_order=1, eta0=eta0)
        tree_aca_fb = build_cluster_tree(centers_aca_fb; leaf_size=64)
        leaves_aca_fb = leaf_nodes(tree_aca_fb)

        row_node_fb = 0
        col_node_fb = 0
        min_block_fb = 0
        for i_leaf in leaves_aca_fb, j_leaf in leaves_aca_fb
            if i_leaf != j_leaf && is_admissible(tree_aca_fb, i_leaf, j_leaf; eta=0.8)
                ni = length(tree_aca_fb.nodes[i_leaf].indices)
                nj = length(tree_aca_fb.nodes[j_leaf].indices)
                if min(ni, nj) > min_block_fb
                    row_node_fb = i_leaf
                    col_node_fb = j_leaf
                    min_block_fb = min(ni, nj)
                end
            end
        end
        @assert row_node_fb != 0 "ACA fallback should find an admissible block"

        rn_fb = tree_aca_fb.nodes[row_node_fb]
        cn_fb = tree_aca_fb.nodes[col_node_fb]
        row_idx_fb = [tree_aca_fb.perm[k] for k in rn_fb.indices]
        col_idx_fb = [tree_aca_fb.perm[k] for k in cn_fb.indices]

        m_blk = length(row_idx_fb)
        n_blk = length(col_idx_fb)
        Z_sub = Matrix{ComplexF64}(undef, m_blk, n_blk)
        for jj in 1:n_blk
            for ii in 1:m_blk
                Z_sub[ii, jj] = DiffMoM._efie_entry(
                    cache_aca_fb, row_idx_fb[ii], col_idx_fb[jj])
            end
        end

        U_aca, V_aca = aca_lowrank(
            cache_aca_fb, row_idx_fb, col_idx_fb; tol=1e-4, max_rank=30)
        rank_aca = size(U_aca, 2)
        approx_aca = U_aca * V_aca'
        err_aca = norm(approx_aca - Z_sub) / max(norm(Z_sub), 1e-30)

        println("  Fallback block size: $(m_blk) x $(n_blk), ACA rank: $rank_aca, rel error: $err_aca")
        @assert err_aca < 5e-3 "ACA fallback approximation too inaccurate: $err_aca"
        @assert rank_aca < min(m_blk, n_blk) "ACA fallback should compress: rank=$rank_aca >= min($m_blk,$n_blk)"
    end
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 26: ACA operator matvec accuracy
# ─────────────────────────────────────────────────
println("\n── Test 26: ACA operator matvec accuracy ──")

@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; aca_tol=Inf, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; max_rank=0, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; leaf_size=0, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; eta=NaN, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; max_block_tasks=0, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; max_storage_bytes=0, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; max_cache_bytes=1, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; max_adjacency_pairs=0, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; max_green_cache_bytes=0, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; max_green_cache_entries=-1, mesh_precheck=false)
aca_task_cap_mesh = make_rect_plate(1.0, 1.0, 2, 2)
aca_task_cap_rwg = build_rwg(aca_task_cap_mesh)
@test_throws ArgumentError build_aca_operator(
    aca_task_cap_mesh, aca_task_cap_rwg, k;
    leaf_size=1, eta=1.5, max_block_tasks=1, mesh_precheck=false)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k; max_storage_bytes=1, mesh_precheck=false)

A_aca_op = build_aca_operator(mesh, rwg, k;
                               leaf_size=8, eta=1.5, aca_tol=1e-8,
                               max_rank=50, quad_order=3, mesh_precheck=false)
@assert size(A_aca_op) == (N, N)
@assert A_aca_op.workspace.work_lock isa ReentrantLock

# One Green-matrix slot disables retention and reuses a bounded scratch matrix
# without changing the assembled operator. A smaller slot fails before the
# first regular-pair matrix allocation.
aca_green_bytes = sizeof(ComplexF64) * cache_aca.Nq^2
A_aca_scratch = build_aca_operator(
    mesh, rwg, k;
    leaf_size=8, eta=1.5, aca_tol=1e-8,
    max_rank=50, quad_order=3, mesh_precheck=false,
    max_green_cache_bytes=aca_green_bytes,
    max_green_cache_entries=1)
@test isapprox(
    A_aca_scratch * ones(ComplexF64, N),
    A_aca_op * ones(ComplexF64, N);
    rtol=1e-12, atol=1e-12)
@test_throws ArgumentError build_aca_operator(
    mesh, rwg, k;
    leaf_size=8, eta=1.5, aca_tol=1e-8,
    max_rank=50, quad_order=3, mesh_precheck=false,
    max_green_cache_bytes=max(1, aca_green_bytes - 1))

# Block BLAS must not lose a finite cancellation before the result is
# unpermuted. The matrices below encode exact 2M - 2M cancellations, where
# M is floatmax(Float64), in the forward/adjoint dense and low-rank paths.
aca_range_base = build_aca_operator(
    aca_scale_mesh,
    aca_scale_rwg,
    2π;
    leaf_size=1,
    eta=1.5,
    aca_tol=1e-6,
    max_rank=3,
    quad_order=1,
    eta0=1.0,
)
aca_range_N = aca_range_base.N
aca_range_rows = 1:aca_range_N
aca_range_input = fill(complex(floatmax(Float64)), aca_range_N)
aca_range_zero = zeros(ComplexF64, aca_range_N)

aca_forward_dense_data = zeros(ComplexF64, aca_range_N, aca_range_N)
aca_forward_dense_data[1, 1] = 2.0
aca_forward_dense_data[1, 2] = -2.0
aca_forward_dense = _aca_operator_with_blocks(
    aca_range_base,
    [DiffMoM.DenseBlock(
        aca_range_rows, aca_range_rows, aca_forward_dense_data)],
    DiffMoM.LowRankBlock[],
)
@test aca_forward_dense * aca_range_input == aca_range_zero

aca_adjoint_dense_data = zeros(ComplexF64, aca_range_N, aca_range_N)
aca_adjoint_dense_data[1, 1] = 2.0
aca_adjoint_dense_data[2, 1] = -2.0
aca_adjoint_dense = _aca_operator_with_blocks(
    aca_range_base,
    [DiffMoM.DenseBlock(
        aca_range_rows, aca_range_rows, aca_adjoint_dense_data)],
    DiffMoM.LowRankBlock[],
)
@test adjoint(aca_adjoint_dense) * aca_range_input == aca_range_zero

aca_forward_U = zeros(ComplexF64, aca_range_N, 1)
aca_forward_V = zeros(ComplexF64, aca_range_N, 1)
aca_forward_U[1, 1] = 1.0
aca_forward_V[1, 1] = 2.0
aca_forward_V[2, 1] = -2.0
aca_forward_lowrank = _aca_operator_with_blocks(
    aca_range_base,
    DiffMoM.DenseBlock[],
    [DiffMoM.LowRankBlock(
        aca_range_rows, aca_range_rows,
        aca_forward_U, aca_forward_V)],
)
@test aca_forward_lowrank * aca_range_input == aca_range_zero

aca_adjoint_U = zeros(ComplexF64, aca_range_N, 1)
aca_adjoint_V = zeros(ComplexF64, aca_range_N, 1)
aca_adjoint_U[1, 1] = 2.0
aca_adjoint_U[2, 1] = -2.0
aca_adjoint_V[1, 1] = 1.0
aca_adjoint_lowrank = _aca_operator_with_blocks(
    aca_range_base,
    DiffMoM.DenseBlock[],
    [DiffMoM.LowRankBlock(
        aca_range_rows, aca_range_rows,
        aca_adjoint_U, aca_adjoint_V)],
)
@test adjoint(aca_adjoint_lowrank) * aca_range_input == aca_range_zero

# Apply alpha before converting an underflowed internal product, and retain
# cancellation against beta*y when both Float64 intermediates overflow.
aca_rescued_data = zeros(ComplexF64, aca_range_N, aca_range_N)
aca_rescued_data[1, 1] = 1e-300
aca_rescued_operator = _aca_operator_with_blocks(
    aca_range_base,
    [DiffMoM.DenseBlock(
        aca_range_rows, aca_range_rows, aca_rescued_data)],
    DiffMoM.LowRankBlock[],
)
aca_rescued_input = zeros(ComplexF64, aca_range_N)
aca_rescued_input[aca_range_base.tree.perm[1]] = 1e-100
aca_rescued_result = zeros(ComplexF64, aca_range_N)
mul!(aca_rescued_result, aca_rescued_operator, aca_rescued_input,
     1e300, 0.0)
aca_rescued_reference = setprecision(BigFloat, 512) do
    ComplexF64(
        BigFloat(1e300) * BigFloat(1e-300) * BigFloat(1e-100))
end
@test aca_rescued_result[aca_range_base.tree.perm[1]] ==
      aca_rescued_reference

aca_beta_data = zeros(ComplexF64, aca_range_N, aca_range_N)
aca_beta_data[1, 1] = 2.0
aca_beta_operator = _aca_operator_with_blocks(
    aca_range_base,
    [DiffMoM.DenseBlock(
        aca_range_rows, aca_range_rows, aca_beta_data)],
    DiffMoM.LowRankBlock[],
)
aca_beta_input = zeros(ComplexF64, aca_range_N)
aca_beta_input[aca_range_base.tree.perm[1]] =
    complex(floatmax(Float64))
aca_beta_result = zeros(ComplexF64, aca_range_N)
aca_beta_result[aca_range_base.tree.perm[1]] =
    -complex(floatmax(Float64))
mul!(aca_beta_result, aca_beta_operator, aca_beta_input, 1.0, 2.0)
@test aca_beta_result == aca_range_zero

# Compare matvec against dense Z
Random.seed!(42)
x_test = randn(ComplexF64, N)
y_dense = Z_efie * x_test
y_aca = A_aca_op * x_test
_assert_single_complex_output_allocation(A_aca_op, x_test)
_assert_scaled_mul_contract(A_aca_op, x_test, reverse(x_test))

rel_matvec_err = norm(y_aca - y_dense) / norm(y_dense)
println("  Dense blocks: $(length(A_aca_op.dense_blocks)), Low-rank blocks: $(length(A_aca_op.lowrank_blocks))")
println("  Matvec relative error: $rel_matvec_err")
@assert rel_matvec_err < 1e-5 "ACA matvec too inaccurate: $rel_matvec_err"

# Test adjoint matvec
A_adj = adjoint(A_aca_op)
y_adj_dense = Z_efie' * x_test
y_adj_aca = A_adj * x_test
_assert_single_complex_output_allocation(A_adj, x_test)
_assert_scaled_mul_contract(A_adj, x_test, reverse(x_test))

# Use an exactly represented identity block here.  If `scaled_aca_previous`
# were formed from a rounded general block product, the mathematical
# `alpha*A*x + beta*y` need not cancel: the large scales can legitimately
# expose the rounding error in that earlier, separate `A*x` call.
scaled_aca_identity = Matrix{ComplexF64}(I, aca_range_N, aca_range_N)
scaled_aca_operator = _aca_operator_with_blocks(
    aca_range_base,
    [DiffMoM.DenseBlock(
        aca_range_rows, aca_range_rows, scaled_aca_identity)],
    DiffMoM.LowRankBlock[],
)
scaled_aca_input = ComplexF64[
    1.0e100 * (sin(0.17 * i) + 1im * cos(0.11 * i))
    for i in 1:aca_range_N
]
scaled_aca_factor = 1.0e308 + 0im
for scaled_operator in (scaled_aca_operator, adjoint(scaled_aca_operator))
    scaled_aca_product = scaled_operator * scaled_aca_input
    scaled_aca_previous = -scaled_aca_product
    @test all(isfinite, scaled_aca_product)
    @test any(!isfinite, scaled_aca_factor .* scaled_aca_product)
    scaled_aca_reference = setprecision(BigFloat, 4608) do
        ComplexF64[
            Complex{BigFloat}(scaled_aca_factor) *
                Complex{BigFloat}(scaled_aca_product[i]) +
            Complex{BigFloat}(scaled_aca_factor) *
                Complex{BigFloat}(scaled_aca_previous[i])
            for i in eachindex(scaled_aca_product)
        ]
    end
    @test all(isfinite, scaled_aca_reference)
    scaled_aca_result = copy(scaled_aca_previous)
    mul!(scaled_aca_result, scaled_operator, scaled_aca_input,
         scaled_aca_factor, scaled_aca_factor)
    @test scaled_aca_result == scaled_aca_reference
end

rel_adj_err = norm(y_adj_aca - y_adj_dense) / norm(y_adj_dense)
println("  Adjoint matvec relative error: $rel_adj_err")
@assert rel_adj_err < 1e-5 "ACA adjoint matvec too inaccurate: $rel_adj_err"
_assert_shared_workspace_concurrency(
    [A_aca_op, A_adj, A_aca_op, A_adj],
    [x_test, (0.2 - 0.3im) .* x_test, reverse(x_test), conj.(x_test)],
)

# Test getindex fallback (used by NF preconditioner)
for _ in 1:10
    ii = rand(1:N)
    jj = rand(1:N)
    @assert A_aca_op[ii, jj] == Z_efie[ii, jj] "getindex mismatch at ($ii,$jj)"
end

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 27: ACA operator solve (forward + adjoint)
# ─────────────────────────────────────────────────
println("\n── Test 27: ACA operator solve (forward + adjoint) ──")

# Build NF preconditioner from geometry (not from dense Z)
P_nf_aca = build_nearfield_preconditioner(mesh, rwg, k, lambda0;
                                            quad_order=3, mesh_precheck=false)

# Forward solve via ACA GMRES
I_aca_gm, stats_aca = solve_gmres(A_aca_op, Vector{ComplexF64}(v);
                                    preconditioner=P_nf_aca,
                                    tol=1e-8, maxiter=300)

# Compare against direct dense solve
I_direct_ref = Z_efie \ v
rel_solve_err = norm(I_aca_gm - I_direct_ref) / norm(I_direct_ref)
println("  Forward solve relative error vs direct: $rel_solve_err")
println("  GMRES iterations: $(stats_aca.niter)")
@assert rel_solve_err < 1e-4 "ACA forward solve too inaccurate: $rel_solve_err"

# Adjoint solve via ACA GMRES
rhs_adj = Q * I_aca_gm
lam_aca, stats_adj = solve_gmres_adjoint(A_aca_op, rhs_adj;
                                           preconditioner=P_nf_aca,
                                           tol=1e-8, maxiter=300)
lam_direct = Z_efie' \ (Q * I_direct_ref)
rel_adj_solve = norm(lam_aca - lam_direct) / max(norm(lam_direct), 1e-30)
println("  Adjoint solve relative error: $rel_adj_solve")
@assert rel_adj_solve < 1e-3 "ACA adjoint solve too inaccurate: $rel_adj_solve"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 28: solve_scattering workflow
# ─────────────────────────────────────────────────
println("\n── Test 28: solve_scattering workflow ──")

# Test with small mesh — auto should pick dense_direct
pw_exc = make_plane_wave(k_vec, E0, pol)
result_auto = solve_scattering(mesh, freq, pw_exc;
                                verbose=false, check_resolution=false)
@assert result_auto.method == :dense_direct "Expected :dense_direct for N=$(result_auto.N), got $(result_auto.method)"
@assert result_auto.N == N
workflow_dense_bytes = sizeof(ComplexF64) * N^2
@test_throws ArgumentError solve_scattering(
    mesh, freq, pw_exc;
    max_dense_matrix_bytes=workflow_dense_bytes - 1,
    verbose=false,
    check_resolution=false)
@test solve_scattering(
    mesh, freq, pw_exc;
    max_dense_matrix_bytes=workflow_dense_bytes,
    verbose=false,
    check_resolution=false).I_coeffs == result_auto.I_coeffs
@test_throws ArgumentError solve_scattering(
    mesh, Inf, pw_exc;
    verbose=false,
    check_resolution=false)
@test_throws ArgumentError solve_scattering(
    mesh, freq, pw_exc;
    c0=Inf,
    verbose=false,
    check_resolution=false)
@test_throws ArgumentError solve_scattering(
    mesh, freq,
    make_plane_wave(2 .* k_vec, E0, pol);
    verbose=false,
    check_resolution=false)
@test_throws ArgumentError solve_scattering(
    mesh, freq, fill(ComplexF64(NaN, 0.0), N);
    verbose=false,
    check_resolution=false)

# A finite frequency/wave-speed ratio must remain usable when the naive
# intermediate 2π*frequency would overflow.
workflow_range_mesh = make_rect_plate(1.0, 1.0, 1, 1)
workflow_range_rwg = build_rwg(workflow_range_mesh)
workflow_range_rhs = ones(ComplexF64, workflow_range_rwg.nedges)
workflow_range_reference = solve_scattering(
    workflow_range_mesh, 1.0, workflow_range_rhs;
    c0=1.0,
    method=:dense_direct,
    verbose=false,
    check_resolution=false)
workflow_range_result = solve_scattering(
    workflow_range_mesh, floatmax(Float64), workflow_range_rhs;
    c0=floatmax(Float64),
    method=:dense_direct,
    verbose=false,
    check_resolution=false)
@test workflow_range_result.I_coeffs == workflow_range_reference.I_coeffs
@test workflow_range_result.method == :dense_direct

for invalid_thresholds in (
    (dense_direct_limit=-1, dense_gmres_limit=100, mlfma_threshold=200),
    (dense_direct_limit=100, dense_gmres_limit=99, mlfma_threshold=200),
    (dense_direct_limit=100, dense_gmres_limit=200, mlfma_threshold=199),
)
    @test_throws ArgumentError solve_scattering(
        mesh, freq, pw_exc;
        invalid_thresholds...,
        verbose=false,
        check_resolution=false)
end

single_triangle_mesh = TriMesh(
    Float64[0 1 0; 0 0 1; 0 0 0],
    reshape(Int[1, 2, 3], 3, 1),
)
@test build_rwg(single_triangle_mesh).nedges == 0
@test_throws ArgumentError solve_scattering(
    single_triangle_mesh, freq, ComplexF64[];
    verbose=false,
    check_resolution=false)

# Verify solution matches manual dense solve
rel_workflow_err = norm(result_auto.I_coeffs - I_pec) / norm(I_pec)
println("  Auto (dense_direct) vs manual: rel_err=$rel_workflow_err")
@assert rel_workflow_err < 1e-10 "Workflow dense_direct solution mismatch: $rel_workflow_err"

# Force ACA GMRES method
result_aca_forced = solve_scattering(mesh, freq, pw_exc;
                                      method=:aca_gmres,
                                      aca_tol=1e-8, aca_leaf_size=8,
                                      nf_cutoff_lambda=1.0,
                                      gmres_tol=1e-8, gmres_maxiter=300,
                                      verbose=false, check_resolution=false)
@assert result_aca_forced.method == :aca_gmres
rel_aca_workflow = norm(result_aca_forced.I_coeffs - I_pec) / norm(I_pec)
println("  Forced ACA vs dense direct: rel_err=$rel_aca_workflow")
@assert rel_aca_workflow < 1e-4 "Workflow ACA solution mismatch: $rel_aca_workflow"

# Force dense GMRES method
result_dgm = solve_scattering(mesh, freq, pw_exc;
                               method=:dense_gmres,
                               nf_cutoff_lambda=1.0,
                               gmres_tol=1e-8, gmres_maxiter=300,
                               verbose=false, check_resolution=false)
@assert result_dgm.method == :dense_gmres
rel_dgm = norm(result_dgm.I_coeffs - I_pec) / norm(I_pec)
println("  Forced dense_gmres vs direct: rel_err=$rel_dgm")
@assert rel_dgm < 1e-6 "Workflow dense GMRES solution mismatch: $rel_dgm"

# High-level workflows must not package an unconverged partial iterate as a
# ScatteringResult. Callers can explicitly opt out when diagnosing Krylov.
@test_throws ErrorException solve_scattering(
    mesh, freq, v;
    method=:dense_gmres,
    preconditioner=:none,
    gmres_tol=1e-14,
    gmres_maxiter=1,
    check_resolution=false,
    verbose=false,
)
result_dgm_partial = solve_scattering(
    mesh, freq, v;
    method=:dense_gmres,
    preconditioner=:none,
    gmres_tol=1e-14,
    gmres_maxiter=1,
    check_gmres_convergence=false,
    check_resolution=false,
    verbose=false,
)
@test result_dgm_partial.gmres_iters == 1
@test_throws ArgumentError solve_scattering(
    mesh, freq, v;
    preconditioner=:invalid,
    check_resolution=false,
    verbose=false,
)
for non_mlfma_method in (:dense_gmres, :aca_gmres)
    @test_throws ArgumentError solve_scattering(
        mesh, freq, v;
        method=non_mlfma_method,
        preconditioner=:ilu,
        check_resolution=false,
        verbose=false,
    )
end

# Test with pre-assembled excitation vector
result_vec = solve_scattering(mesh, freq, v;
                               verbose=false, check_resolution=false)
@assert result_vec.method == :dense_direct
@assert norm(result_vec.I_coeffs - I_pec) / norm(I_pec) < 1e-10

# Reject a wrong-sized lazy RHS before attempting to materialize it. This
# keeps an untrusted vector length from driving an irrelevant allocation.
struct WorkflowPoisonVector <: AbstractVector{ComplexF64}
    length::Int
end
Base.size(vector::WorkflowPoisonVector) = (vector.length,)
Base.getindex(::WorkflowPoisonVector, ::Int) =
    error("wrong-sized workflow vector was materialized")
@test_throws DimensionMismatch solve_scattering(
    mesh,
    freq,
    WorkflowPoisonVector(N + 1);
    verbose=false,
    check_resolution=false,
)

# Test mesh resolution warning
warning_freq = 1e6
warning_k = 2π * warning_freq / c0
warning_pw = make_plane_wave(
    Vec3(0.0, 0.0, -warning_k), E0, pol)
result_warn = solve_scattering(mesh, warning_freq, warning_pw;
                                verbose=false, check_resolution=true)
# At 1 MHz, lambda = 300m, mesh is massively over-resolved → no warning
@assert isempty(result_warn.warnings) || result_warn.mesh_report.meets_target

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 29: Physical Optics (PO) solver
# ─────────────────────────────────────────────────
println("\n── Test 29: Physical Optics (PO) solver ──")

# 29a: Illumination test on flat plate
# Plate at z=0, plane wave from +z direction (k along -z)
po_mesh = make_rect_plate(0.2, 0.2, 5, 5)  # reuse existing test plate
po_freq = 3e9
po_c0 = 299792458.0
po_lam = po_c0 / po_freq
po_k = 2π / po_lam

# Wave traveling in -z: should illuminate the +z-facing surface
pw_down = make_plane_wave(Vec3(0.0, 0.0, -po_k), 1.0, Vec3(1.0, 0.0, 0.0))
po_grid = make_sph_grid(36, 72)
@test_throws ArgumentError solve_po(po_mesh, 0.0, pw_down; grid=po_grid)
@test_throws ArgumentError solve_po(
    po_mesh, po_freq,
    make_plane_wave(Vec3(0.0, 0.0, 0.0), 1.0, Vec3(1.0, 0.0, 0.0));
    grid=po_grid)
@test_throws ArgumentError solve_po(
    po_mesh, po_freq,
    make_plane_wave(Vec3(0.0, 0.0, -2po_k), 1.0, Vec3(1.0, 0.0, 0.0));
    grid=po_grid)
@test_throws ArgumentError solve_po(
    po_mesh, po_freq,
    make_plane_wave(Vec3(0.0, 0.0, -po_k), 1.0, Vec3(0.0, 0.0, 1.0));
    grid=po_grid)
@test_throws ArgumentError solve_po(
    po_mesh, po_freq, pw_down; grid=po_grid, eta0=0.0)
@test_throws ArgumentError solve_po(
    po_mesh, po_freq,
    PlaneWaveExcitation(
        Vec3(0.0, 0.0, -po_k), 1.0, Vec3(2.0, 0.0, 0.0));
    grid=po_grid)
@test_throws ErrorException solve_po(
    TriMesh(zeros(3, 0), zeros(Int, 3, 0)),
    po_freq, pw_down; grid=po_grid)
po_tiny_direction_component = nextfloat(0.0)
po_tiny_direction_grid = SphGrid(
    reshape(Float64[0, 0, 1], 3, 1), [0.0], [0.0], [1.0])
po_tiny_direction_excitation = PlaneWaveExcitation(
    Vec3(po_tiny_direction_component, po_tiny_direction_component, 0.0),
    1.0 + 0im,
    Vec3(0.0, 0.0, 1.0),
)
_, _, _, po_tiny_direction = DiffMoM._validate_po_inputs(
    po_tiny_direction_grid,
    po_tiny_direction_component,
    po_tiny_direction_excitation,
    2π,
    376.730313668,
)
@test po_tiny_direction ≈
      Vec3(inv(sqrt(2.0)), inv(sqrt(2.0)), 0.0) rtol=2eps(Float64)
po_large_amplitude = solve_po(
    po_mesh, po_freq,
    PlaneWaveExcitation(
        Vec3(0.0, 0.0, -po_k), floatmax(Float64),
        Vec3(1.0, 0.0, 0.0));
    grid=make_sph_grid(2, 4))
@test all(current -> all(isfinite, current), po_large_amplitude.J_s)
@test all(isfinite, po_large_amplitude.E_ff)
po_result = solve_po(po_mesh, po_freq, pw_down; grid=po_grid)
po_work_bytes = DiffMoM._po_work_bytes(
    ntriangles(po_mesh), length(po_grid.w))
po_capped = solve_po(
    po_mesh, po_freq, pw_down;
    grid=po_grid, max_work_bytes=po_work_bytes)
@test po_capped.E_ff == po_result.E_ff
@test_throws ArgumentError solve_po(
    po_mesh, po_freq, pw_down;
    grid=po_grid, max_work_bytes=po_work_bytes - 1)
@test_throws ArgumentError solve_po(
    po_mesh, po_freq, pw_down; grid=po_grid, max_work_bytes=0)

# Mixed-scale PO currents are evaluated as a complete product.  A formally
# overflowing standalone 2E0/eta0 factor must not reject an exactly zero
# geometric current.
po_zero_current_grid = SphGrid(
    reshape(Float64[0, 0, 1], 3, 1), [0.0], [0.0], [1.0])
po_zero_current_excitation = make_plane_wave(
    Vec3(1.0, 0.0, 0.0), 1.0, Vec3(0.0, 1.0, 0.0))
po_zero_current = solve_po(
    make_rect_plate(1.0, 1.0, 1, 1), 1.0,
    po_zero_current_excitation;
    grid=po_zero_current_grid, c0=2π, eta0=nextfloat(0.0))
@test all(iszero, po_zero_current.J_s)
@test all(iszero, po_zero_current.E_ff)

# The moment-series transition is continuous and independent of cyclic face
# order, including the high-precision geometry path.
po_phase_vertices = (
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0, 0.2, 0.0),
    Vec3(1.0, 5.0e-6, 0.0),
)
po_phase_area = 0.1
po_phase_delta = Vec3(0.0, 1.0, 0.0)
po_phase_reference = DiffMoM._phase_integral_analytical(
    1.0, po_phase_delta, po_phase_vertices..., po_phase_area)
@test DiffMoM._phase_integral_analytical(
    1.0, po_phase_delta,
    po_phase_vertices[3], po_phase_vertices[1], po_phase_vertices[2],
    po_phase_area) == po_phase_reference
for phase_argument in (0.249, 0.251)
    phase_integral = DiffMoM._phase_integral_analytical(
        1.0, Vec3(0.0, phase_argument, 0.0),
        Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 0.0), 0.5)
    phase_oracle = setprecision(BigFloat, 512) do
        argument = BigFloat(phase_argument)
        ComplexF64(2 * (exp(im * argument) - 1 - im * argument) /
                   (im * argument)^2 * BigFloat(0.5))
    end
    @test phase_integral ≈ phase_oracle rtol=8eps(Float64) atol=0.0
end
# The DD-small branch expands Dq-Dp with the opposite sign from its factored
# exponential.  Compare it with the permutation-symmetric bivariate moment
# series around the origin.
po_dd_small_dp = 1.000001e-5
po_dd_small_dq = 2.0e-5
po_dd_small_integral = DiffMoM._phase_integral_analytical(
    1.0, Vec3(po_dd_small_dq, po_dd_small_dp, 0.0),
    Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
    Vec3(0.0, 0.0, 0.0), 0.5)
po_dd_small_reference = setprecision(BigFloat, 256) do
    dp = BigFloat(po_dd_small_dp)
    dq = BigFloat(po_dd_small_dq)
    ComplexF64(sum(
        (im * dp)^first_order * (im * dq)^second_order /
        factorial(big(first_order + second_order + 2))
        for first_order in 0:20, second_order in 0:20))
end
@test po_dd_small_integral ≈
      po_dd_small_reference rtol=2e-12 atol=0.0

# The exact-direction path must form the transverse projection in the exact
# domain. A tiny component of r̂ can disappear from the Float64 dot product,
# yet become representable after multiplication by a large incident field.
po_projection_side = ldexp(1.0, -100)
po_projection_mesh = TriMesh(
    Float64[
        0.0  po_projection_side  0.0;
        0.0  0.0                 2po_projection_side;
        0.0  0.0                 0.0
    ],
    reshape(Int[1, 2, 3], 3, 1),
)
po_projection_grid = SphGrid(
    reshape(Float64[1.0, po_projection_side, 0.0], 3, 1),
    [π / 2], [0.0], [1.0])
po_projection_polarization = Vec3(inv(sqrt(2.0)), inv(sqrt(2.0)), 0.0)
po_projection_result = solve_po(
    po_projection_mesh, 1.0,
    PlaneWaveExcitation(
        Vec3(0.0, 0.0, -1.0), ldexp(1.0, 200),
        po_projection_polarization);
    grid=po_projection_grid, c0=2π)
po_projection_reference = setprecision(BigFloat, 512) do
    rb = SVector{3,BigFloat}(
        BigFloat(1.0), BigFloat(po_projection_side), BigFloat(0.0))
    vb = SVector{3,BigFloat}(BigFloat.(po_projection_polarization))
    projection = rb * dot(rb, vb) - vb
    area = BigFloat(po_projection_side)^2
    scale = Complex{BigFloat}(0, 1) * BigFloat(ldexp(1.0, 200)) /
            BigFloat(2π)
    integral = DiffMoM._phase_integral_analytical_big_value(
        BigFloat(1.0),
        rb - SVector{3,BigFloat}(0, 0, -1),
        Vec3(po_projection_side, 0.0, 0.0),
        Vec3(0.0, 2po_projection_side, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Float64(po_projection_side^2), 1e-5, 5)
    ComplexF64(scale * projection[1] * integral)
end
@test !iszero(po_projection_reference)
@test po_projection_result.E_ff[1, 1] == po_projection_reference

# A stored unit direction generally has a norm a few ulps from one.  The PO
# transverse projection must still preserve the exact null when the surface
# direction is parallel to that observation direction, on both accumulation
# paths.
po_parallel_component = inv(sqrt(2.0))
po_parallel_direction =
    Vec3(po_parallel_component, po_parallel_component, 0.0)
po_parallel_grid = SphGrid(
    reshape(collect(po_parallel_direction), 3, 1),
    [π / 2], [π / 4], [1.0])
for po_parallel_amplitude in (ldexp(1.0, 120), ldexp(1.0, 200))
    po_parallel_result = solve_po(
        make_rect_plate(1.0, 1.0, 1, 1), po_freq,
        PlaneWaveExcitation(
            Vec3(0.0, 0.0, -po_k), po_parallel_amplitude,
            po_parallel_direction);
        grid=po_parallel_grid)
    @test all(iszero, po_parallel_result.E_ff)
end

# The exact PO phase uses normalized supplied incident and observation
# directions.  A large rigid translation must therefore produce the physical
# normalized translation phase, not one amplified from their rounded norms.
po_translation_xyz = Float64[
     0.0   0.0   0.0   0.0;
    -0.5   0.5  -0.5   0.5;
    -0.5  -0.5   0.5   0.5
]
po_translation_triangles = Int[1 1; 2 4; 4 3]
po_translation_base = TriMesh(
    po_translation_xyz, po_translation_triangles)
po_translation_shift = 1.0e16
po_translation_shifted_xyz = copy(po_translation_xyz)
po_translation_shifted_xyz[1, :] .+= po_translation_shift
po_translation_shifted = TriMesh(
    po_translation_shifted_xyz, copy(po_translation_triangles))
po_translation_direction = Vec3(
    po_parallel_component, po_parallel_component, 0.0)
po_translation_grid = SphGrid(
    reshape(collect(po_translation_direction), 3, 1),
    [π / 2], [π / 4], [1.0])
po_translation_excitation = PlaneWaveExcitation(
    Vec3(-1.0, 0.0, 0.0), ldexp(1.0, 200), Vec3(0.0, 0.0, 1.0))
po_translation_base_result = solve_po(
    po_translation_base, 1.0, po_translation_excitation;
    grid=po_translation_grid, c0=2π)
po_translation_shifted_result = solve_po(
    po_translation_shifted, 1.0, po_translation_excitation;
    grid=po_translation_grid, c0=2π)
po_translation_index = argmax(abs.(po_translation_base_result.E_ff[:, 1]))
po_translation_ratio =
    po_translation_shifted_result.E_ff[po_translation_index, 1] /
    po_translation_base_result.E_ff[po_translation_index, 1]
po_translation_phase = setprecision(BigFloat, 4608) do
    direction = SVector{3,BigFloat}(BigFloat.(po_translation_direction))
    normalized_direction = direction / sqrt(sum(abs2, direction))
    ComplexF64(exp(Complex{BigFloat}(
        0, BigFloat(po_translation_shift) *
           (normalized_direction[1] + 1))))
end
@test po_translation_ratio ≈
      po_translation_phase rtol=4eps(Float64) atol=0.0

n_illum = count(po_result.illuminated)
n_total = ntriangles(po_mesh)
# For a plate at z=0 with normals pointing in +z, wave from +z traveling -z
# should illuminate all faces (k̂·n̂ = -1 < 0)
println("  Illumination (wave -z on +z plate): $n_illum / $n_total")
@assert n_illum == n_total "Expected all $n_total triangles illuminated, got $n_illum"

# Wave traveling in +z: should illuminate NO faces (backside)
pw_up = make_plane_wave(Vec3(0.0, 0.0, po_k), 1.0, Vec3(1.0, 0.0, 0.0))
po_result_back = solve_po(po_mesh, po_freq, pw_up; grid=po_grid)
n_illum_back = count(po_result_back.illuminated)
println("  Illumination (wave +z on +z plate): $n_illum_back / $n_total")
@assert n_illum_back == 0 "Expected 0 triangles illuminated, got $n_illum_back"

# 29b: PO specular RCS on flat plate
# Analytical PO: σ_specular = 4π A² / λ² for a flat plate at broadside
Lx_po, Ly_po = 0.2, 0.2
A_plate = Lx_po * Ly_po
sigma_analytical = 4π * A_plate^2 / po_lam^2
sigma_analytical_dB = 10 * log10(sigma_analytical)

# Find the specular direction (θ≈π, backscatter for -z incidence → r̂ = +z)
# Actually for -z incidence, specular reflection from plate at z=0 is +z direction
sigma_po = bistatic_rcs(po_result.E_ff; E0=1.0)
# Find the direction closest to +z (θ≈0)
best_idx = argmax([po_grid.rhat[3, q] for q in 1:length(po_grid.w)])
sigma_spec = sigma_po[best_idx]
sigma_spec_dB = 10 * log10(max(sigma_spec, 1e-30))

println("  PO specular RCS: $(round(sigma_spec_dB, digits=2)) dBsm")
println("  Analytical 4πA²/λ²: $(round(sigma_analytical_dB, digits=2)) dBsm")
po_err_dB = abs(sigma_spec_dB - sigma_analytical_dB)
println("  Error: $(round(po_err_dB, digits=2)) dB")
@assert po_err_dB < 1.5 "PO specular RCS error $(po_err_dB) dB > 1.5 dB tolerance"

# 29c: PO vs MoM comparison on small plate
# At broadside, PO and MoM should agree within a few dB for specular
po_rwg = build_rwg(po_mesh)
po_Z = assemble_Z_efie(po_mesh, po_rwg, po_k)
po_v = assemble_excitation(po_mesh, po_rwg, pw_down)
po_I = po_Z \ po_v
po_G = radiation_vectors(po_mesh, po_rwg, po_grid, po_k)
po_NΩ = length(po_grid.w)
po_Eff_mom = compute_farfield(po_G, Vector{ComplexF64}(po_I), po_NΩ)
sigma_mom = bistatic_rcs(po_Eff_mom; E0=1.0)
sigma_mom_spec_dB = 10 * log10(max(sigma_mom[best_idx], 1e-30))

# Complex-field SIGN check: PO far-field must match MoM's phase (not its negative).
# RCS = |E|² is sign-blind, so this guards the +jk PO prefactor convention, which
# matters for coherent PO+PTD / PO+MoM combination.
po_Espec = po_result.E_ff[:, best_idx]
mom_Espec = po_Eff_mom[:, best_idx]
@assert norm(po_Espec - mom_Espec) < norm(po_Espec + mom_Espec) "PO far-field sign disagrees with MoM (prefactor sign error)"

mom_po_diff_dB = abs(sigma_mom_spec_dB - sigma_spec_dB)
println("  MoM specular RCS: $(round(sigma_mom_spec_dB, digits=2)) dBsm")
println("  MoM vs PO specular diff: $(round(mom_po_diff_dB, digits=2)) dB")
@assert mom_po_diff_dB < 3.0 "MoM vs PO specular difference $(mom_po_diff_dB) dB > 3.0 dB"

# 29d: PTD edge geometry, general-wedge coefficients, and solve path
@test_throws ArgumentError extract_diffraction_edges(
    po_mesh; min_dihedral_deg=NaN)
@test_throws ArgumentError extract_diffraction_edges(
    po_mesh; min_dihedral_deg=-1.0)
@test_throws ArgumentError extract_diffraction_edges(
    po_mesh; min_dihedral_deg=181.0)
@test isempty(extract_diffraction_edges(
    po_mesh; min_dihedral_deg=0.0, include_boundary=false))

ptd_wedge_xyz = [
    0.0  1.0  0.0  0.0;
    0.0  0.0  1.0  0.0;
    0.0  0.0  0.0  1.0
]
ptd_wedge_tri = [
    1  1;
    2  4;
    3  2
]
ptd_wedge_mesh = TriMesh(ptd_wedge_xyz, ptd_wedge_tri)
ptd_wedge_edges = extract_diffraction_edges(
    ptd_wedge_mesh; include_boundary=false)
@test length(ptd_wedge_edges) == 1
@test ptd_wedge_edges[1].alpha ≈ 3π / 2 atol=1e-14
@test_throws ArgumentError solve_ptd(
    ptd_wedge_mesh, po_freq, pw_down;
    grid=make_sph_grid(2, 4), include_boundary=false)

for (n_wedge, delta_s_wedge, delta_i_wedge) in
    ((1.25, 1.1, 0.4), (1.5, 2.0, 0.7), (1.8, 2.5, 1.2))
    gamma_wedge = n_wedge * π
    u_wedge = 0.5 * (delta_s_wedge - delta_i_wedge)
    v_wedge = 0.5 * (delta_s_wedge + delta_i_wedge)
    a_wedge = sin(π / n_wedge) / n_wedge
    X_wedge = a_wedge /
        (cos(π / n_wedge) - cos(2u_wedge / n_wedge))
    Y_wedge = a_wedge /
        (cos(π / n_wedge) - cos(2v_wedge / n_wedge))
    direct_f_wedge =
        X_wedge - Y_wedge - 0.5tan(u_wedge) -
        0.5tan(gamma_wedge - v_wedge)
    direct_g_wedge =
        X_wedge + Y_wedge - 0.5tan(u_wedge) +
        0.5tan(gamma_wedge - v_wedge)
    stable_f_wedge, stable_g_wedge = DiffMoM._ptd_fringe_fg(
        n_wedge, delta_s_wedge, delta_i_wedge, gamma_wedge)
    @test stable_f_wedge ≈ direct_f_wedge rtol=1e-12 atol=1e-12
    @test stable_g_wedge ≈ direct_g_wedge rtol=1e-12 atol=1e-12
end
@test all(isfinite, DiffMoM._ptd_fringe_fg(2.0, Float64(π), 0.0, 2π))
ptd_exact_cap_state = Dict{Int,Vector{Complex{BigFloat}}}()
@test_throws ArgumentError DiffMoM._ptd_register_exact_direction!(
    ptd_exact_cap_state, 1,
    DiffMoM._MAX_PTD_EXACT_DIRECTION_VALUES ÷ 3)
@test isempty(ptd_exact_cap_state)
ptd_exact_cap_state[1] = Complex{BigFloat}[]
@test DiffMoM._ptd_register_exact_direction!(
    ptd_exact_cap_state, 1,
    DiffMoM._MAX_PTD_EXACT_DIRECTION_VALUES ÷ 3) ==
    DiffMoM._MAX_PTD_EXACT_DIRECTION_VALUES ÷ 3

ptd_mesh = make_rect_plate(0.2, 0.2, 1, 1)
ptd_grid = make_sph_grid(4, 8)
ptd_result = solve_ptd(
    ptd_mesh, po_freq, pw_down; grid=ptd_grid)
ptd_required_bytes = DiffMoM._po_work_bytes(
    ntriangles(ptd_mesh), length(ptd_grid.w)) +
    DiffMoM._ptd_additional_work_bytes(
        length(ptd_result.edges), length(ptd_grid.w))
ptd_capped = solve_ptd(
    ptd_mesh, po_freq, pw_down;
    grid=ptd_grid, max_work_bytes=ptd_required_bytes)
@test ptd_capped.E_ff == ptd_result.E_ff
@test_throws ArgumentError solve_ptd(
    ptd_mesh, po_freq, pw_down;
    grid=ptd_grid, max_work_bytes=ptd_required_bytes - 1)
@test_throws ArgumentError solve_ptd(
    ptd_mesh, po_freq, pw_down;
    grid=ptd_grid, max_work_bytes=0)
@test size(ptd_result.E_ff) == (3, length(ptd_grid.w))
@test length(ptd_result.edges) == 4
@test all(isfinite, ptd_result.E_ff)
@test ptd_result.E_ff ≈
      ptd_result.E_ff_po + ptd_result.E_ff_ptd rtol=1e-14

# A rigid translation applies one incident and one outgoing phase to the PTD
# edge field.  It must not apply the incident phase a second time, and both
# supplied directions must be normalized before exact phase reduction.
ptd_translation_excitation = PlaneWaveExcitation(
    Vec3(-1.0, 0.0, 0.0), 1.0, Vec3(0.0, 0.0, 1.0))
ptd_translation_base_result = solve_ptd(
    po_translation_base, 1.0, ptd_translation_excitation;
    grid=po_translation_grid, c0=2π)
ptd_translation_shifted_result = solve_ptd(
    po_translation_shifted, 1.0, ptd_translation_excitation;
    grid=po_translation_grid, c0=2π)
ptd_translation_index =
    argmax(abs.(ptd_translation_base_result.E_ff_ptd[:, 1]))
@test !iszero(
    ptd_translation_base_result.E_ff_ptd[ptd_translation_index, 1])
ptd_translation_ratio =
    ptd_translation_shifted_result.E_ff_ptd[ptd_translation_index, 1] /
    ptd_translation_base_result.E_ff_ptd[ptd_translation_index, 1]
@test ptd_translation_ratio ≈
      po_translation_phase rtol=16eps(Float64) atol=0.0

# A large incident field and a tiny edge length form a finite PTD product.
# The solver must not reject the standalone pre-length coefficient.
ptd_scale_length = 1.0e-150
ptd_scale_kx = sqrt(0.99)
ptd_scale_mesh = make_rect_plate(
    ptd_scale_length, ptd_scale_length, 1, 1)
ptd_scale_grid = SphGrid(
    reshape(Float64[0, 0, 1], 3, 1), [0.0], [0.0], [1.0])
ptd_scale_wavevector = Vec3(ptd_scale_kx, 0.1, 0.0)
ptd_scale_polarization = Vec3(-0.1, ptd_scale_kx, 0.0)
ptd_scale_low = solve_ptd(
    ptd_scale_mesh, 1.0,
    PlaneWaveExcitation(
        ptd_scale_wavevector, floatmax(Float64) / 32,
        ptd_scale_polarization);
    grid=ptd_scale_grid, c0=2π)
ptd_scale_high = solve_ptd(
    ptd_scale_mesh, 1.0,
    PlaneWaveExcitation(
        ptd_scale_wavevector, floatmax(Float64),
        ptd_scale_polarization);
    grid=ptd_scale_grid, c0=2π)
@test all(isfinite, ptd_scale_high.E_ff_ptd)
@test ptd_scale_high.E_ff_ptd ≈
      32 .* ptd_scale_low.E_ff_ptd rtol=5e-14 atol=0.0

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 30: ILU preconditioner
# ─────────────────────────────────────────────────
println("\n── Test 30: ILU preconditioner ──")

# Build a small plate problem for testing
ilu_mesh = make_rect_plate(0.4, 0.4, 4, 4)
ilu_rwg  = build_rwg(ilu_mesh)
ilu_N    = ilu_rwg.nedges
ilu_k    = 2π / 0.4   # λ = 0.4m
ilu_Z    = assemble_Z_efie(ilu_mesh, ilu_rwg, ilu_k)
ilu_pw   = make_plane_wave(Vec3(0.0, 0.0, -ilu_k), 1.0, Vec3(1.0, 0.0, 0.0))
ilu_rhs  = assemble_excitation(ilu_mesh, ilu_rwg, ilu_pw)

# Reference: direct solve
ilu_ref = ilu_Z \ ilu_rhs

# 30a: ILU preconditioner builds without error
cutoff_ilu = 0.3   # ~0.75λ
P_ilu = build_nearfield_preconditioner(ilu_Z, ilu_mesh, ilu_rwg, cutoff_ilu;
    factorization=:ilu, ilu_tau=1e-3)
@assert P_ilu isa ILUPreconditionerData
@assert P_ilu.nnz_ratio > 0 && P_ilu.nnz_ratio <= 1.0
println("  30a: ILU preconditioner built — nnz=$(round(P_ilu.nnz_ratio * 100, digits=1))%, τ=$(P_ilu.tau)")
for invalid_tau in (-1.0, Inf, NaN)
    @test_throws ArgumentError build_nearfield_preconditioner(
        ilu_Z, ilu_mesh, ilu_rwg, cutoff_ilu;
        factorization=:ilu, ilu_tau=invalid_tau)
end

# 30b: ILU-preconditioned GMRES converges
I_ilu, stats_ilu = solve_gmres(ilu_Z, ilu_rhs; preconditioner=P_ilu, tol=1e-8, maxiter=500)
err_ilu = norm(I_ilu - ilu_ref) / norm(ilu_ref)
println("  30b: ILU GMRES — $(stats_ilu.niter) iters, rel error $(round(err_ilu, sigdigits=3))")
@assert stats_ilu.niter < 200 "ILU GMRES took $(stats_ilu.niter) iters (expected < 200)"
@assert err_ilu < 1e-4 "ILU GMRES relative error $(err_ilu) > 1e-4"

# 30c: Compare ILU vs full LU iteration counts
P_lu = build_nearfield_preconditioner(ilu_Z, ilu_mesh, ilu_rwg, cutoff_ilu;
    factorization=:lu)
I_lu, stats_lu = solve_gmres(ilu_Z, ilu_rhs; preconditioner=P_lu, tol=1e-8, maxiter=500)
println("  30c: LU GMRES — $(stats_lu.niter) iters (ILU: $(stats_ilu.niter) iters)")
# ILU should take more iterations than full LU but still converge
@assert stats_ilu.niter >= stats_lu.niter "ILU should take ≥ LU iterations"

# 30d: ILU works with matrix-free operator (mesh, rwg, k overload)
P_ilu_mf = build_nearfield_preconditioner(ilu_mesh, ilu_rwg, ilu_k, cutoff_ilu;
    factorization=:ilu, ilu_tau=1e-3)
@assert P_ilu_mf isa ILUPreconditionerData
println("  30d: Matrix-free ILU build — nnz=$(round(P_ilu_mf.nnz_ratio * 100, digits=1))%")

# 30e: ILU adjoint preconditioner works
I_adj_ilu, stats_adj = solve_gmres_adjoint(ilu_Z, ilu_rhs; preconditioner=P_ilu, tol=1e-8, maxiter=500)
I_adj_ref = adjoint(ilu_Z) \ ilu_rhs
err_adj = norm(I_adj_ilu - I_adj_ref) / norm(I_adj_ref)
println("  30e: ILU adjoint GMRES — $(stats_adj.niter) iters, rel error $(round(err_adj, sigdigits=3))")
@assert err_adj < 1e-4 "ILU adjoint GMRES relative error $(err_adj) > 1e-4"

# 30f: ILU preconditioner adjoint must be the true Hermitian adjoint
ilu_M = NearFieldOperator(P_ilu)
ilu_M_adj = NearFieldAdjointOperator(P_ilu)
ilu_x = randn(ComplexF64, size(ilu_Z, 1))
ilu_y = randn(ComplexF64, size(ilu_Z, 1))
ilu_lhs = dot(ilu_M * ilu_x, ilu_y)
ilu_rhs_adj = dot(ilu_x, ilu_M_adj * ilu_y)
ilu_adj_err = abs(ilu_lhs - ilu_rhs_adj) / max(abs(ilu_lhs), 1e-30)
println("  30f: ILU preconditioner adjoint identity — $(round(ilu_adj_err, sigdigits=3))")
@assert ilu_adj_err < 1e-12 "ILU preconditioner adjoint inconsistent: $ilu_adj_err"

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 31: MLFMA operator
# ─────────────────────────────────────────────────
println("\n--- Test 31: MLFMA operator ---")

# Setup: 2λ × 2λ flat plate at 3 GHz
mlfma_freq = 3e9
mlfma_lambda = 299792458.0 / mlfma_freq
mlfma_k = 2π / mlfma_lambda
mlfma_Lx = 2 * mlfma_lambda
mlfma_Ly = 2 * mlfma_lambda
mlfma_Nx = 6
mlfma_Ny = 6
mlfma_mesh = make_rect_plate(mlfma_Lx, mlfma_Ly, mlfma_Nx, mlfma_Ny)
mlfma_rwg = build_rwg(mlfma_mesh)
mlfma_N = mlfma_rwg.nedges
println("  Setup: $(mlfma_Nx)×$(mlfma_Ny) plate, N=$mlfma_N, freq=$(mlfma_freq/1e9) GHz")

# 31a: Octree construction
mlfma_centers = rwg_centers(mlfma_mesh, mlfma_rwg)
@test_throws ArgumentError build_octree(Vec3[], mlfma_k)
for invalid_octree_k in (0.0, -mlfma_k, Inf, NaN)
    @test_throws ArgumentError build_octree(mlfma_centers, invalid_octree_k)
end
for invalid_leaf_lambda in (0.0, -0.5, Inf, NaN)
    @test_throws ArgumentError build_octree(
        mlfma_centers, mlfma_k; leaf_lambda=invalid_leaf_lambda)
end
@test_throws ArgumentError build_octree(
    [Vec3(NaN, 0.0, 0.0)], mlfma_k)
@test_throws OverflowError build_octree(
    [Vec3(-floatmax(Float64), 0.0, 0.0),
     Vec3(floatmax(Float64), 0.0, 0.0)], mlfma_k)
@test_throws OverflowError build_octree(
    mlfma_centers, mlfma_k; leaf_lambda=1.0e-100)

balanced_octree = build_octree(
    [Vec3(0.0, 0.0, 0.0)], 1e-308; leaf_lambda=1e-308)
@test balanced_octree.levels[end].edge_length == 2π
@test all(all(isfinite, box.center)
          for level in balanced_octree.levels for box in level.boxes)
@test_throws OverflowError build_octree(
    [Vec3(floatmax(Float64), 0.0, 0.0)],
    2π / 5e306;
    leaf_lambda=1.0,
)
@test_throws ArgumentError build_octree(
    mlfma_centers, mlfma_k; max_boxes=0)
@test_throws ArgumentError build_octree(
    mlfma_centers, mlfma_k; max_storage_bytes=0)

octree_fused_center = DiffMoM._validated_octree_box_center(
    Vec3(-floatmax(Float64), 0.0, 0.0),
    (1, 0, 0),
    floatmax(Float64),
)
@test octree_fused_center[1] == floatmax(Float64) / 2
@test all(isfinite, octree_fused_center)

DiffMoM._validated_octree_leaf_edge(2.0, 0.5)
@test (@allocated DiffMoM._validated_octree_leaf_edge(2.0, 0.5)) == 0
@test (@allocated DiffMoM._validated_octree_leaf_edge(1e-308, 1e-308)) <=
      1_024
DiffMoM._validated_octree_box_center(
    Vec3(0.0, 0.0, 0.0), (0, 0, 0), 1.0)
@test (@allocated DiffMoM._validated_octree_box_center(
    Vec3(0.0, 0.0, 0.0), (0, 0, 0), 1.0)) == 0

for invalid_truncation_input in (
    (0.0, mlfma_k, 3), (1.0, 0.0, 3), (1.0, mlfma_k, 0)
)
    @test_throws ArgumentError DiffMoM.truncation_order(
        invalid_truncation_input[1], invalid_truncation_input[2];
        precision=invalid_truncation_input[3])
end
@test DiffMoM.truncation_order(1.1e308, 1e-308; precision=3) == 6
@test_throws ArgumentError DiffMoM.make_sphere_sampling(-1)
@test_throws ArgumentError DiffMoM.make_sphere_sampling(3; max_points=0)
@test_throws ArgumentError DiffMoM.make_sphere_sampling(1_000_000)
@test_throws OverflowError DiffMoM.make_sphere_sampling(typemax(Int))
@test _sphere_sampling_rejection_allocations(1_000_000) <= 4_096

for invalid_hankel_input in ((-1, 1.0), (3, 0.0), (3, Inf), (3, NaN))
    @test_throws ArgumentError DiffMoM.spherical_hankel2_all(
        invalid_hankel_input...)
end

# The half-integer Bessel implementation is independent of the recurrence
# under test. This coarse-level argument is generated by a default-option
# octree interaction and used to make Miller's old start below x.
mlfma_hankel_coarse_L = 56
mlfma_hankel_coarse_x = 130.59355422486368
mlfma_hankel_coarse = DiffMoM.spherical_hankel2_all(
    mlfma_hankel_coarse_L, mlfma_hankel_coarse_x)
mlfma_hankel_coarse_reference = ComplexF64[
    DiffMoM.sphericalbesselj(l, mlfma_hankel_coarse_x) -
    im * DiffMoM.sphericalbessely(l, mlfma_hankel_coarse_x)
    for l in 0:mlfma_hankel_coarse_L
]
@test norm(mlfma_hankel_coarse - mlfma_hankel_coarse_reference) /
      norm(mlfma_hankel_coarse_reference) <= 5e-13
@test all(isfinite, mlfma_hankel_coarse)
@test _spherical_hankel_allocations(
    mlfma_hankel_coarse_L, mlfma_hankel_coarse_x) <= 2_500

mlfma_hankel_tiny_x = 3e-31
mlfma_hankel_tiny = DiffMoM.spherical_hankel2_all(3, mlfma_hankel_tiny_x)
mlfma_hankel_tiny_reference = _spherical_hankel_bigfloat_reference(
    3, mlfma_hankel_tiny_x)
for l in eachindex(mlfma_hankel_tiny)
    @test isapprox(real(mlfma_hankel_tiny[l]),
                   real(mlfma_hankel_tiny_reference[l]);
                   rtol=4eps(Float64), atol=0.0)
    @test isapprox(imag(mlfma_hankel_tiny[l]),
                   imag(mlfma_hankel_tiny_reference[l]);
                   rtol=4eps(Float64), atol=0.0)
end
@test _spherical_hankel_allocations(3, mlfma_hankel_tiny_x) <= 24_000
@test_throws OverflowError DiffMoM.spherical_hankel2_all(3, 1e-100)

mlfma_coarse_sampling = DiffMoM.make_sphere_sampling(mlfma_hankel_coarse_L)
mlfma_coarse_translation = DiffMoM.compute_translation_factor(
    Vec3(24π, 24π, 24π), 1.0, mlfma_coarse_sampling)[1]
@test isapprox(real(mlfma_coarse_translation),
               -0.0272850707742405; rtol=5e-13, atol=0.0)
@test isapprox(imag(mlfma_coarse_translation),
               0.0009201247803121605; rtol=5e-13, atol=0.0)

mlfma_tiny_sampling = DiffMoM.make_sphere_sampling(3)
mlfma_tiny_translation = DiffMoM.compute_translation_factor(
    Vec3(mlfma_hankel_tiny_x, 0.0, 0.0),
    1.0,
    mlfma_tiny_sampling,
)[1]
@test isapprox(real(mlfma_tiny_translation),
               5.77490728152923e123; rtol=32eps(Float64), atol=0.0)
@test isapprox(imag(mlfma_tiny_translation),
               9.394780419190805e91; rtol=32eps(Float64), atol=0.0)
mlfma_balanced_translation = DiffMoM.compute_translation_factor(
    Vec3(-1.05e308, -1.05e308, -1.05e308),
    (2π * 0.25) / 3.5e307,
    DiffMoM.make_sphere_sampling(7),
)
@test isapprox(
    mlfma_balanced_translation[1],
    -0.6934666072411643 - 0.23265439823371686im;
    rtol=8eps(Float64), atol=0.0)
@test all(isfinite, mlfma_balanced_translation)
@test_throws ArgumentError DiffMoM.compute_translation_factor(
    Vec3(1.0, 0.0, 0.0), 1.0,
    DiffMoM.make_sphere_sampling(64); max_terms=1)
mlfma_translation_sampling = DiffMoM.make_sphere_sampling(64)
DiffMoM.compute_translation_factor(
    Vec3(1.0, 0.0, 0.0), 1.0, mlfma_translation_sampling)
GC.gc()
mlfma_translation_alloc = @allocated DiffMoM.compute_translation_factor(
    Vec3(1.0, 0.0, 0.0), 1.0, mlfma_translation_sampling)
@test mlfma_translation_alloc < 500_000
mlfma_zero_legendre_sampling = SphereSampling(
    101, 1, 1, 1, [π / 2], [0.0], [1.0],
    reshape(Float64[1.0, 0.0, 0.0], 3, 1),
)
mlfma_zero_legendre_translation = DiffMoM.compute_translation_factor(
    Vec3(0.0, 0.07, 0.0), 1.0, mlfma_zero_legendre_sampling)
@test mlfma_zero_legendre_translation == ComplexF64[
    1.0 + 4.710198164328786e304im]

@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; precision=0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; eta0=0.0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, 0.0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; leaf_lambda=0.0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_sampling_points=0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_setup_bytes=0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_nearfield_entries=0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_nearfield_bytes=0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_adjacency_pairs=-1)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_translation_terms=0)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; precision=1_000_000)
@test _mlfma_precision_rejection_allocations(
    mlfma_mesh, mlfma_rwg, mlfma_k) <= 4_096
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_setup_bytes=1)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_nearfield_entries=1)
@test_throws ArgumentError build_mlfma_operator(
    mlfma_mesh, mlfma_rwg, mlfma_k; max_nearfield_bytes=1)
octree = build_octree(mlfma_centers, mlfma_k; leaf_lambda=0.5)
octree_box_bound, octree_storage_bound =
    DiffMoM._octree_resource_bounds(mlfma_N, octree.nLevels)
@test_throws ArgumentError build_octree(
    mlfma_centers, mlfma_k;
    leaf_lambda=0.5,
    max_boxes=octree_box_bound - 1,
    max_storage_bytes=octree_storage_bound)
@test_throws ArgumentError build_octree(
    mlfma_centers, mlfma_k;
    leaf_lambda=0.5,
    max_boxes=octree_box_bound,
    max_storage_bytes=octree_storage_bound - 1)
octree_at_limits = build_octree(
    mlfma_centers, mlfma_k;
    leaf_lambda=0.5,
    max_boxes=octree_box_bound,
    max_storage_bytes=octree_storage_bound)
@test octree_at_limits.perm == octree.perm
@test DiffMoM._estimated_octree_storage_bytes(octree) <=
      octree_storage_bound

# Verify all BFs are assigned via permutation
@assert length(octree.perm) == mlfma_N
@assert length(octree.iperm) == mlfma_N
@assert sort(octree.perm) == 1:mlfma_N  "perm must be a permutation of 1:N"

# Verify perm/iperm are inverse of each other
for i in 1:mlfma_N
    @assert octree.iperm[octree.perm[i]] == i "perm/iperm inconsistency at $i"
end

leaf_level = octree.levels[octree.nLevels]
n_leaf_boxes = length(leaf_level.boxes)
println("  31a: Octree — $(octree.nLevels) levels, $n_leaf_boxes leaf boxes")

# Verify neighbors and interaction_list don't overlap
for box in leaf_level.boxes
    nbr_set = Set(box.neighbors)
    il_set = Set(box.interaction_list)
    @assert isempty(intersect(nbr_set, il_set)) "Neighbor/interaction_list overlap for box $(box.id)"
end
println("  31a: PASS — no neighbor/interaction_list overlaps")

# 31b: Near-field matrix accuracy
Z_dense_mlfma = assemble_Z_efie(mlfma_mesh, mlfma_rwg, mlfma_k; mesh_precheck=false)
Z_near_mlfma = assemble_mlfma_nearfield(octree, mlfma_mesh, mlfma_rwg, mlfma_k)
mlfma_nearfield_entries = Int(DiffMoM._mlfma_nearfield_entry_count(octree))
@test_throws ArgumentError assemble_mlfma_nearfield(
    octree, mlfma_mesh, mlfma_rwg, mlfma_k;
    max_nearfield_entries=mlfma_nearfield_entries - 1)
@test_throws ArgumentError assemble_mlfma_nearfield(
    octree, mlfma_mesh, mlfma_rwg, mlfma_k;
    max_nearfield_bytes=
        mlfma_nearfield_entries * DiffMoM._MLFMA_NEARFIELD_TRIPLET_BYTES - 1)

# Check that near-field entries match dense for neighbor pairs
max_nf_err = 0.0
n_checked = 0
for box in leaf_level.boxes
    for nbr_id in box.neighbors
        nbr_box = leaf_level.boxes[nbr_id]
        for m_perm in box.bf_range
            m = octree.perm[m_perm]
            for n_perm in nbr_box.bf_range
                n = octree.perm[n_perm]
                err = abs(Z_near_mlfma[m, n] - Z_dense_mlfma[m, n])
                ref = abs(Z_dense_mlfma[m, n])
                if ref > 1e-15
                    global max_nf_err = max(max_nf_err, err / ref)
                end
                global n_checked += 1
            end
        end
    end
end
println("  31b: Near-field accuracy — checked $n_checked entries, max rel error = $(round(max_nf_err, sigdigits=3))")
@assert max_nf_err < 1e-10 "Near-field entries do not match dense: max error $max_nf_err"
println("  31b: PASS")

# 31c: MLFMA matvec accuracy
A_mlfma = build_mlfma_operator(mlfma_mesh, mlfma_rwg, mlfma_k;
    leaf_lambda=0.5, quad_order=3, verbose=false)
@assert A_mlfma.workspace.work_lock isa ReentrantLock
@test isempty(A_mlfma.workspace.entry_output)
build_nearfield_preconditioner(
    A_mlfma, mlfma_mesh, mlfma_rwg, 0.0; factorization=:diag)
@test isempty(A_mlfma.workspace.entry_output)

# Random test vector
Random.seed!(42)
x_test = randn(ComplexF64, mlfma_N)
y_dense = Z_dense_mlfma * x_test
y_mlfma = A_mlfma * x_test
_assert_single_complex_output_allocation(A_mlfma, x_test)

mlfma_matvec_err = norm(y_mlfma - y_dense) / norm(y_dense)
println("  31c: MLFMA matvec — rel error = $(round(mlfma_matvec_err, sigdigits=3))")
@assert mlfma_matvec_err < 0.05 "MLFMA matvec error too large: $mlfma_matvec_err (expected < 0.05)"
println("  31c: PASS")

mlfma_far_row = 0
mlfma_far_column = 0
for box in leaf_level.boxes
    isempty(box.bf_range) && continue
    for far_box_id in box.interaction_list
        far_box = leaf_level.boxes[far_box_id]
        isempty(far_box.bf_range) && continue
        global mlfma_far_row = octree.perm[first(box.bf_range)]
        global mlfma_far_column = octree.perm[first(far_box.bf_range)]
        break
    end
    mlfma_far_row > 0 && break
end
@test mlfma_far_row > 0
@test iszero(A_mlfma.Z_near[mlfma_far_row, mlfma_far_column])
mlfma_basis_input = zeros(ComplexF64, mlfma_N)
mlfma_basis_input[mlfma_far_column] = 1
mlfma_basis_product = A_mlfma * mlfma_basis_input
mlfma_far_entry = A_mlfma[mlfma_far_row, mlfma_far_column]
@test mlfma_far_entry == mlfma_basis_product[mlfma_far_row]
@test !iszero(mlfma_far_entry)
@test adjoint(A_mlfma)[mlfma_far_column, mlfma_far_row] ==
      conj(mlfma_far_entry)
@test _matrix_entry_allocation(
          A_mlfma, mlfma_far_row, mlfma_far_column) <= 128

# 31d: MLFMA adjoint inner-product identity
y_test = randn(ComplexF64, mlfma_N)
lhs_adj = dot(y_test, A_mlfma * x_test)
rhs_adj = dot(adjoint(A_mlfma) * y_test, x_test)
_assert_single_complex_output_allocation(adjoint(A_mlfma), y_test)
_assert_scaled_mul_contract(A_mlfma, x_test, y_test)
_assert_scaled_mul_contract(adjoint(A_mlfma), y_test, x_test)
_assert_zero_allocation_overlap_mul_contract(A_mlfma, x_test, 2.0 - 3.0im)
_assert_zero_allocation_overlap_mul_contract(
    adjoint(A_mlfma), y_test, -4.0 + 1.5im)

mlfma_extreme_product_unscaled = A_mlfma * x_test
mlfma_extreme_input =
    (10.0 / maximum(abs, mlfma_extreme_product_unscaled)) .* x_test
mlfma_extreme_product = A_mlfma * mlfma_extreme_input
mlfma_extreme_scale = 1.0e308 + 0im
@test any(!isfinite, mlfma_extreme_scale .* mlfma_extreme_product)
mlfma_extreme_result = -mlfma_extreme_product
mul!(mlfma_extreme_result, A_mlfma, mlfma_extreme_input,
     mlfma_extreme_scale, mlfma_extreme_scale)
@test mlfma_extreme_result == zeros(ComplexF64, mlfma_N)

mlfma_adjoint_extreme_product_unscaled = adjoint(A_mlfma) * y_test
mlfma_adjoint_extreme_input =
    (10.0 / maximum(abs, mlfma_adjoint_extreme_product_unscaled)) .*
    y_test
mlfma_adjoint_extreme_product =
    adjoint(A_mlfma) * mlfma_adjoint_extreme_input
@test any(!isfinite,
          mlfma_extreme_scale .* mlfma_adjoint_extreme_product)
mlfma_adjoint_extreme_result = -mlfma_adjoint_extreme_product
mul!(mlfma_adjoint_extreme_result, adjoint(A_mlfma),
     mlfma_adjoint_extreme_input,
     mlfma_extreme_scale, mlfma_extreme_scale)
@test mlfma_adjoint_extreme_result == zeros(ComplexF64, mlfma_N)

# Internal aggregation can overflow before a finite final MLFMA product is
# formed. Recover it with a lossless power-of-two input rescaling, including
# five-argument scaling, cancellation, and aliased output.
mlfma_internal_extreme_input =
    fill(ComplexF64(floatmax(Float64), 0.0), mlfma_N)
mlfma_internal_extreme_scale = inv(floatmax(Float64))
mlfma_internal_scaled_input =
    mlfma_internal_extreme_scale .* mlfma_internal_extreme_input
for mlfma_internal_operator in (A_mlfma, adjoint(A_mlfma))
    mlfma_internal_reference =
        mlfma_internal_operator * mlfma_internal_scaled_input
    mlfma_internal_raw =
        mlfma_internal_operator * mlfma_internal_extreme_input
    @test all(isfinite, mlfma_internal_raw)
    @test mlfma_internal_extreme_scale .* mlfma_internal_raw ==
          mlfma_internal_reference

    mlfma_internal_result = zeros(ComplexF64, mlfma_N)
    mul!(mlfma_internal_result, mlfma_internal_operator,
         mlfma_internal_extreme_input,
         mlfma_internal_extreme_scale, 0.0)
    @test mlfma_internal_result == mlfma_internal_reference

    mlfma_internal_cancellation = -mlfma_internal_reference
    mul!(mlfma_internal_cancellation, mlfma_internal_operator,
         mlfma_internal_extreme_input,
         mlfma_internal_extreme_scale, 1.0)
    @test mlfma_internal_cancellation == zeros(ComplexF64, mlfma_N)

    mlfma_internal_alias = copy(mlfma_internal_extreme_input)
    mul!(mlfma_internal_alias, mlfma_internal_operator,
         mlfma_internal_alias,
         mlfma_internal_extreme_scale, 0.0)
    @test mlfma_internal_alias == mlfma_internal_reference
end

# Conversely, an output scale can recover products that would underflow inside
# the MLFMA passes. Upscale the input exactly before the pass and compensate in
# the final high-precision output combination.
mlfma_internal_tiny_input =
    fill(ComplexF64(nextfloat(0.0), 0.0), mlfma_N)
mlfma_internal_tiny_scale = floatmax(Float64)
mlfma_internal_rescaled_tiny_input =
    mlfma_internal_tiny_scale .* mlfma_internal_tiny_input
for mlfma_internal_operator in (A_mlfma, adjoint(A_mlfma))
    mlfma_internal_tiny_reference =
        mlfma_internal_operator * mlfma_internal_rescaled_tiny_input
    @test all(isfinite, mlfma_internal_tiny_reference)
    @test all(!iszero, mlfma_internal_tiny_reference)

    mlfma_internal_tiny_result = zeros(ComplexF64, mlfma_N)
    mul!(mlfma_internal_tiny_result, mlfma_internal_operator,
         mlfma_internal_tiny_input,
         mlfma_internal_tiny_scale, 0.0)
    @test mlfma_internal_tiny_result ≈
          mlfma_internal_tiny_reference rtol=1e-13 atol=0.0

    mlfma_internal_tiny_cancellation = -mlfma_internal_tiny_result
    mul!(mlfma_internal_tiny_cancellation, mlfma_internal_operator,
         mlfma_internal_tiny_input,
         mlfma_internal_tiny_scale, 1.0)
    @test maximum(abs, mlfma_internal_tiny_cancellation) <=
          eps(Float64) * maximum(abs, mlfma_internal_tiny_result)

    mlfma_internal_tiny_alias = copy(mlfma_internal_tiny_input)
    mul!(mlfma_internal_tiny_alias, mlfma_internal_operator,
         mlfma_internal_tiny_alias,
         mlfma_internal_tiny_scale, 0.0)
    @test mlfma_internal_tiny_alias == mlfma_internal_tiny_result
end

# A normal component must not mask subnormal components that become observable
# after alpha/beta cancellation. Real and imaginary components are assigned to
# exponent bands independently so one ComplexF64 value can straddle bands.
mlfma_banded_normal_input = zeros(ComplexF64, mlfma_N)
mlfma_banded_normal_input[1] = 1.0
mlfma_banded_tiny_input =
    fill(ComplexF64(nextfloat(0.0), 0.0), mlfma_N)
mlfma_banded_tiny_input[1] = ComplexF64(0.0, nextfloat(0.0))
mlfma_banded_mixed_input =
    mlfma_banded_normal_input + mlfma_banded_tiny_input
mlfma_banded_scale = floatmax(Float64)
for mlfma_banded_operator in (A_mlfma, adjoint(A_mlfma))
    mlfma_banded_normal_product =
        mlfma_banded_operator * mlfma_banded_normal_input
    mlfma_banded_reference =
        mlfma_banded_operator *
        (mlfma_banded_scale .* mlfma_banded_tiny_input)
    mlfma_banded_result = -mlfma_banded_normal_product
    mul!(mlfma_banded_result, mlfma_banded_operator,
         mlfma_banded_mixed_input,
         mlfma_banded_scale, mlfma_banded_scale)
    @test all(isfinite, mlfma_banded_result)
    @test all(!iszero, mlfma_banded_result)
    @test mlfma_banded_result ≈
          mlfma_banded_reference rtol=1e-13 atol=0.0

    mlfma_banded_alias = copy(mlfma_banded_mixed_input)
    mlfma_banded_alias_reference = copy(mlfma_banded_mixed_input)
    mul!(mlfma_banded_alias_reference, mlfma_banded_operator,
         mlfma_banded_mixed_input, 1.0, 0.5)
    mul!(mlfma_banded_alias, mlfma_banded_operator,
         mlfma_banded_alias, 1.0, 0.5)
    @test mlfma_banded_alias == mlfma_banded_alias_reference
end

# Direct mul! accepts generic AbstractVector inputs.  It must apply the same
# ComplexF64 conversion as `A * x` before exponent classification and before
# the internal product; otherwise a below-minsub BigFloat can be amplified by
# the operator even though the public ComplexF64 input semantics rounded it to
# zero.
mlfma_big_input = setprecision(BigFloat, 256) do
    values = fill(BigFloat(0), mlfma_N)
    values[1] = BigFloat(1)
    mlfma_N >= 2 && (values[2] = ldexp(BigFloat(1), -1075))
    mlfma_N >= 3 && (values[3] = ldexp(BigFloat(1), -100))
    values
end
mlfma_converted_input = ComplexF64.(mlfma_big_input)
mlfma_unbanded_big_input = setprecision(BigFloat, 256) do
    values = fill(BigFloat(0), mlfma_N)
    values[1] = BigFloat(1)
    mlfma_N >= 2 && (values[2] = ldexp(BigFloat(1), -1075))
    values
end
for raw_input in (mlfma_big_input, mlfma_unbanded_big_input)
    converted_input = ComplexF64.(raw_input)
    for mlfma_generic_operator in (A_mlfma, adjoint(A_mlfma))
        mlfma_generic_reference = mlfma_generic_operator * converted_input
        mlfma_generic_result = zeros(ComplexF64, mlfma_N)
        mul!(mlfma_generic_result, mlfma_generic_operator,
             raw_input, 1.0, 0.0)
        @test mlfma_generic_result == mlfma_generic_reference
    end
end

mlfma_adj_err = abs(lhs_adj - rhs_adj) / max(abs(lhs_adj), abs(rhs_adj), eps())
println("  31d: MLFMA adjoint identity — rel error = $(round(mlfma_adj_err, sigdigits=3))")
@assert mlfma_adj_err < 1e-10 "MLFMA adjoint identity failed: $mlfma_adj_err"
println("  31d: PASS")
_assert_shared_workspace_concurrency(
    [A_mlfma, adjoint(A_mlfma), A_mlfma, adjoint(A_mlfma)],
    [x_test, y_test, reverse(x_test), conj.(y_test)],
)
if Threads.nthreads() > 1
    for _ in 1:4
        mlfma_index_gate = Base.Event()
        mlfma_index_task = Threads.@spawn begin
            wait(mlfma_index_gate)
            A_mlfma[mlfma_far_row, mlfma_far_column]
        end
        mlfma_product_task = Threads.@spawn begin
            wait(mlfma_index_gate)
            (A_mlfma * mlfma_basis_input)[mlfma_far_row]
        end
        mlfma_adjoint_index_task = Threads.@spawn begin
            wait(mlfma_index_gate)
            adjoint(A_mlfma)[mlfma_far_column, mlfma_far_row]
        end
        yield()
        notify(mlfma_index_gate)
        @test fetch(mlfma_index_task) == mlfma_far_entry
        @test fetch(mlfma_product_task) == mlfma_far_entry
        @test fetch(mlfma_adjoint_index_task) == conj(mlfma_far_entry)
    end
end

# 31e: MLFMA + GMRES convergence
mlfma_exc = PlaneWaveExcitation(Vec3(0.0, 0.0, -mlfma_k), 1.0, Vec3(1.0, 0.0, 0.0))
mlfma_v = assemble_excitation(mlfma_mesh, mlfma_rwg, mlfma_exc)
I_dense_ref = Z_dense_mlfma \ mlfma_v

# Build preconditioner from MLFMA near-field
P_mlfma = build_nearfield_preconditioner(A_mlfma.Z_near; factorization=:lu)
@test_throws DimensionMismatch build_nearfield_preconditioner(
    spzeros(ComplexF64, 2, 3); factorization=:diag)
@test_throws ArgumentError build_nearfield_preconditioner(
    spzeros(ComplexF64, 0, 0); factorization=:diag)
nonfinite_sparse_nf = spdiagm(0 => ones(ComplexF64, 2))
nonfinite_sparse_nf[1, 1] = Inf + 0im
@test_throws ArgumentError build_nearfield_preconditioner(
    nonfinite_sparse_nf; factorization=:diag)
@test_throws ArgumentError build_nearfield_preconditioner(
    spzeros(ComplexF64, 2, 2); factorization=:diag)

# Jacobi regularization is relative to the supplied matrix scale.  The former
# absolute unit floor replaced every entry here by 1e-10 and took two Krylov
# dimensions instead of reducing the diagonal system to an identity in one.
tiny_scaled_diagonal = ComplexF64[1e-100, 1e-107]
tiny_scaled_nf = spdiagm(0 => tiny_scaled_diagonal)
tiny_scaled_preconditioner = build_nearfield_preconditioner(
    tiny_scaled_nf; factorization=:diag)
@test tiny_scaled_preconditioner.dinv ≈ inv.(tiny_scaled_diagonal) rtol=2eps(Float64)
tiny_scaled_solution, tiny_scaled_stats = solve_gmres(
    tiny_scaled_nf, copy(tiny_scaled_diagonal);
    preconditioner=tiny_scaled_preconditioner,
    tol=1e-12, maxiter=10, memory=10)
@test tiny_scaled_stats.niter == 1
@test tiny_scaled_solution ≈ ones(ComplexF64, 2) rtol=4eps(Float64)
I_mlfma, stats_mlfma = solve_gmres(A_mlfma, mlfma_v;
    preconditioner=P_mlfma, tol=1e-4, maxiter=200)

mlfma_sol_err = norm(I_mlfma - I_dense_ref) / norm(I_dense_ref)
println("  31e: MLFMA+GMRES — $(stats_mlfma.niter) iters, sol rel error = $(round(mlfma_sol_err, sigdigits=3))")
@assert stats_mlfma.niter < 200 "MLFMA GMRES did not converge in 200 iterations"
@assert mlfma_sol_err < 0.1 "MLFMA solution error too large: $mlfma_sol_err (expected < 0.1)"
println("  31e: PASS")

# 31f: impedance-loaded MLFMA preconditioner
part_mlfma_loaded = assign_patches_grid(mlfma_mesh; nx=2, ny=2, nz=1)
Mp_mlfma_loaded = precompute_patch_mass(mlfma_mesh, mlfma_rwg, part_mlfma_loaded)
theta_mlfma_loaded = fill(150.0, part_mlfma_loaded.P)
A_mlfma_loaded = ImpedanceLoadedOperator(A_mlfma, Mp_mlfma_loaded, theta_mlfma_loaded, false)
P_mlfma_loaded = build_mlfma_preconditioner(A_mlfma, Mp_mlfma_loaded, theta_mlfma_loaded;
    factorization=:lu)
@assert length(P_mlfma_loaded.work) == mlfma_N
_assert_single_workspace_mul!(NearFieldOperator(P_mlfma_loaded), x_test)
_assert_single_workspace_mul!(NearFieldAdjointOperator(P_mlfma_loaded), x_test)
I_mlfma_loaded, stats_mlfma_loaded = solve_gmres(A_mlfma_loaded, mlfma_v;
    preconditioner=P_mlfma_loaded, tol=1e-4, maxiter=200)
loaded_residual = norm(A_mlfma_loaded * I_mlfma_loaded - mlfma_v) / norm(mlfma_v)
println("  31f: Loaded MLFMA preconditioner — $(stats_mlfma_loaded.niter) iters, residual = $(round(loaded_residual, sigdigits=3))")
@assert stats_mlfma_loaded.solved "Loaded MLFMA GMRES should converge"
@assert loaded_residual < 1e-4 "Loaded MLFMA residual too large: $loaded_residual"
println("  31f: PASS")

# 31g: block-Jacobi preconditioner action and allocation
block_storage_bytes = DiffMoM._block_diag_storage_bytes(A_mlfma)
P_mlfma_block = build_block_diag_preconditioner(A_mlfma)
P_mlfma_block_capped = build_block_diag_preconditioner(
    A_mlfma; max_storage_bytes=block_storage_bytes)
@test length(P_mlfma_block_capped.lu_blocks) ==
      length(P_mlfma_block.lu_blocks)
@test_throws ArgumentError build_block_diag_preconditioner(
    A_mlfma; max_storage_bytes=block_storage_bytes - 1)
@test_throws ArgumentError build_block_diag_preconditioner(
    A_mlfma; max_storage_bytes=0)

# Assemble loaded leaf blocks without materializing one dense patch submatrix
# per patch. The exceptional path retains an exact Float-input cancellation
# that rounds to a normal ComplexF64 entry.
block_loaded = build_block_diag_preconditioner(
    A_mlfma, Mp_mlfma_loaded, theta_mlfma_loaded;
    max_storage_bytes=block_storage_bytes)
@test length(block_loaded.lu_blocks) == length(P_mlfma_block.lu_blocks)
@test_throws ArgumentError build_block_diag_preconditioner(
    A_mlfma, Mp_mlfma_loaded, theta_mlfma_loaded;
    max_storage_bytes=block_storage_bytes - 1)

m_block_exact = Int64(2)^52 + 2
a_block_exact = ldexp(Float64(m_block_exact), -52)
b_block_exact = -ldexp(Float64(m_block_exact - 1), -52)
v_block_exact = ldexp(Float64(m_block_exact), -970)
p_block_exact = ldexp(Float64(m_block_exact + 1), -970)
exact_block = ComplexF64[0 0; 0 1]
exact_mass_a = ComplexF64[v_block_exact 0; 0 0]
exact_mass_b = ComplexF64[p_block_exact 0; 0 0]
DiffMoM._load_block_diag_matrix!(
    exact_block, [exact_mass_a, exact_mass_b],
    [a_block_exact, b_block_exact], false, [1, 2],
    Ref(0), 100_000)
exact_block_reference = setprecision(BigFloat, 4352) do
    ComplexF64(-BigFloat(a_block_exact) * BigFloat(v_block_exact) -
               BigFloat(b_block_exact) * BigFloat(p_block_exact))
end
@test exact_block[1, 1] == exact_block_reference
@test exact_block[1, 1] != 0
@test_throws ArgumentError DiffMoM._load_block_diag_matrix!(
    copy(ComplexF64[0 0; 0 1]),
    [exact_mass_a, exact_mass_b],
    [a_block_exact, b_block_exact], false, [1, 2],
    Ref(0), 1)

block_result = _assert_zero_allocation_mul!(
    NearFieldOperator(P_mlfma_block), x_test)
block_adjoint_result = _assert_zero_allocation_mul!(
    NearFieldAdjointOperator(P_mlfma_block), x_test)
block_reference = copy(x_test)
block_adjoint_reference = copy(x_test)
for (factor, indices) in zip(
        P_mlfma_block.lu_blocks, P_mlfma_block.box_bf_indices)
    block_reference[indices] = factor \ x_test[indices]
    block_adjoint_reference[indices] = adjoint(factor) \ x_test[indices]
end
@assert block_result ≈ block_reference
@assert block_adjoint_result ≈ block_adjoint_reference
@assert length(P_mlfma_block.work) ==
        maximum(length, P_mlfma_block.box_bf_indices)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 32: Mesh I/O formats and geometry coverage
# ─────────────────────────────────────────────────
println("\n── Test 32: Mesh I/O formats and geometry coverage ──")

# 32a: write_obj_mesh round-trip
println("  32a: write_obj_mesh round-trip ...")
obj_rt_path = joinpath(DATADIR, "tmp_roundtrip.obj")
write_obj_mesh(obj_rt_path, mesh; header="Round-trip test")
mesh_rt = read_obj_mesh(obj_rt_path)
@assert nvertices(mesh_rt) == nvertices(mesh) "OBJ round-trip vertex count mismatch"
@assert ntriangles(mesh_rt) == ntriangles(mesh) "OBJ round-trip triangle count mismatch"
for i in 1:nvertices(mesh)
    for d in 1:3
        @assert abs(mesh_rt.xyz[d, i] - mesh.xyz[d, i]) < 1e-12 "OBJ round-trip vertex position mismatch at vertex $i dim $d"
    end
end
report_rt = mesh_quality_report(mesh_rt)
@assert mesh_quality_ok(report_rt; allow_boundary=true)

obj_limit_path = joinpath(DATADIR, "tmp_obj_limits.obj")
write(obj_limit_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
obj_limit_size = filesize(obj_limit_path)
obj_limit_mesh = read_obj_mesh(
    obj_limit_path;
    max_vertices=3,
    max_triangles=1,
    max_raw_bytes=96,
    max_input_bytes=obj_limit_size,
    max_line_bytes=7,
)
@test nvertices(obj_limit_mesh) == 3
@test ntriangles(obj_limit_mesh) == 1
@test_throws ArgumentError read_obj_mesh(obj_limit_path; max_vertices=2)
@test_throws ArgumentError read_obj_mesh(obj_limit_path; max_triangles=false)
@test_throws ArgumentError read_obj_mesh(obj_limit_path; max_raw_bytes=95)
@test_throws ArgumentError read_obj_mesh(
    obj_limit_path; max_input_bytes=obj_limit_size - 1)
@test_throws ArgumentError read_obj_mesh(obj_limit_path; max_line_bytes=6)

# Line limits measure parsed content and therefore behave identically for LF
# and CRLF files. The longest content line here is seven bytes (`v 0 0 0`).
obj_crlf_limit_path = joinpath(DATADIR, "tmp_obj_crlf_limits.obj")
write(obj_crlf_limit_path, "v 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nf 1 2 3\r\n")
obj_crlf_limit_mesh = read_obj_mesh(
    obj_crlf_limit_path; max_line_bytes=7)
@test nvertices(obj_crlf_limit_mesh) == 3
@test ntriangles(obj_crlf_limit_mesh) == 1
@test_throws ArgumentError read_obj_mesh(
    obj_crlf_limit_path; max_line_bytes=6)

# The CRLF pair may straddle the bounded reader's 64 KiB refill boundary.
# That must not make the content limit platform- or alignment-dependent.
obj_split_crlf_path = joinpath(DATADIR, "tmp_obj_split_crlf_limits.obj")
obj_split_comment = "#" * repeat("x", 65_534)
obj_split_payload = obj_split_comment *
    "\r\nv 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nf 1 2 3\r\n"
write(obj_split_crlf_path, obj_split_payload)
obj_split_crlf_mesh = read_obj_mesh(
    obj_split_crlf_path;
    max_input_bytes=ncodeunits(obj_split_payload),
    max_line_bytes=65_535,
)
@test nvertices(obj_split_crlf_mesh) == 3
@test ntriangles(obj_split_crlf_mesh) == 1
@test_throws ArgumentError read_obj_mesh(
    obj_split_crlf_path; max_line_bytes=65_534)

# A header is metadata, not an escape hatch into the line-oriented OBJ grammar.
# Reject both Unix and classic-Mac line breaks before replacing an existing file.
obj_header_path = joinpath(DATADIR, "tmp_obj_header_validation.obj")
obj_header_sentinel = UInt8[0x44, 0x69, 0x66, 0x66, 0x4d, 0x6f, 0x4d]
for unsafe_header in ("safe\nv 9 9 9", "safe\rf 1 2 3")
    write(obj_header_path, obj_header_sentinel)
    @test_throws ArgumentError write_obj_mesh(
        obj_header_path, mesh; header=unsafe_header)
    @test read(obj_header_path) == obj_header_sentinel
end

# Invalid meshes must be rejected before opening the destination.  These are
# precisely the structural/value constraints imposed by read_obj_mesh.
obj_validation_path = joinpath(DATADIR, "tmp_obj_mesh_validation.obj")
obj_validation_sentinel = UInt8[0x44, 0x69, 0x66, 0x66, 0x4d, 0x6f, 0x4d]
obj_empty_mesh = TriMesh(zeros(Float64, 3, 0), Matrix{Int}(undef, 3, 0))
obj_nonfinite_mesh = TriMesh(
    Float64[0 Inf 0; 0 0 1; 0 0 0], reshape([1, 2, 3], 3, 1))
obj_bad_index_mesh = TriMesh(
    Float64[0 1 0; 0 0 1; 0 0 0], reshape([1, 2, 4], 3, 1))
for bad_mesh in (obj_empty_mesh, obj_nonfinite_mesh, obj_bad_index_mesh)
    write(obj_validation_path, obj_validation_sentinel)
    @test_throws ArgumentError write_obj_mesh(obj_validation_path, bad_mesh)
    @test read(obj_validation_path) == obj_validation_sentinel
end
for bad_shape_mesh in (
    TriMesh(zeros(Float64, 2, 3), reshape([1, 2, 3], 3, 1)),
    TriMesh(zeros(Float64, 3, 3), reshape([1, 2], 2, 1)),
)
    write(obj_validation_path, obj_validation_sentinel)
    @test_throws DimensionMismatch write_obj_mesh(
        obj_validation_path, bad_shape_mesh)
    @test read(obj_validation_path) == obj_validation_sentinel
end

obj_alloc_path = joinpath(DATADIR, "tmp_alloc.obj")
obj_grid_n = 20
open(obj_alloc_path, "w") do io
    for j in 0:obj_grid_n, i in 0:obj_grid_n
        println(io, "v ", i / obj_grid_n, " ", j / obj_grid_n, " 0")
    end
    for j in 0:(obj_grid_n - 1), i in 0:(obj_grid_n - 1)
        n1 = i + 1 + j * (obj_grid_n + 1)
        n2 = n1 + 1
        n4 = n1 + obj_grid_n + 1
        n3 = n4 + 1
        println(io, "f ", n1, " ", n2, " ", n3)
        println(io, "f ", n1, " ", n3, " ", n4)
    end
end
mesh_obj_alloc = read_obj_mesh(obj_alloc_path)  # warm compilation
@assert nvertices(mesh_obj_alloc) == (obj_grid_n + 1)^2
@assert ntriangles(mesh_obj_alloc) == 2 * obj_grid_n^2
GC.gc()
obj_read_alloc = @allocated read_obj_mesh(obj_alloc_path)
@assert obj_read_alloc < 24 * filesize(obj_alloc_path)
println("  32a: PASS")

# 32b: triangle_area explicit test
println("  32b: triangle_area explicit test ...")
# Right triangle with base=3 along x, height=4 along y → area = 6.0
xyz_right = Float64[0 3 0; 0 0 4; 0 0 0]
tri_right = reshape([1, 2, 3], 3, 1)
mesh_right = TriMesh(xyz_right, tri_right)
@assert abs(triangle_area(mesh_right, 1) - 6.0) < 1e-12 "Right triangle area should be 6.0"
@test triangle_normal(mesh_right, 1) ≈ Vec3(0.0, 0.0, 1.0)
@test triangle_center(mesh_right, 1) == Vec3(1.0, 4 / 3, 0.0)
triangle_area(mesh_right, 1)
triangle_center(mesh_right, 1)
triangle_normal(mesh_right, 1)
@test (@allocated triangle_area(mesh_right, 1)) == 0
@test (@allocated triangle_center(mesh_right, 1)) == 0
@test (@allocated triangle_normal(mesh_right, 1)) == 0

mesh_cancelled_ordinary_center = TriMesh(
    Float64[0.1 0.2 -0.3; 0 1 0; 0 0 0],
    tri_right,
)
ordinary_cancelled_center_reference = setprecision(BigFloat, 256) do
    Float64((BigFloat(0.1) + BigFloat(0.2) - BigFloat(0.3)) / 3)
end
@test triangle_center(mesh_cancelled_ordinary_center, 1)[1] ==
      ordinary_cancelled_center_reference
triangle_center(mesh_cancelled_ordinary_center, 1)
@test (@allocated triangle_center(mesh_cancelled_ordinary_center, 1)) == 0

# Equilateral triangle with side s=2 → area = sqrt(3)
s_eq = 2.0
xyz_eq = Float64[0 s_eq s_eq/2; 0 0 s_eq*sqrt(3)/2; 0 0 0]
tri_eq = reshape([1, 2, 3], 3, 1)
mesh_eq = TriMesh(xyz_eq, tri_eq)
expected_area = s_eq^2 * sqrt(3) / 4
@assert abs(triangle_area(mesh_eq, 1) - expected_area) < 1e-12 "Equilateral triangle area mismatch"
mesh_degenerate_normal = TriMesh(
    Float64[0 1 2; 0 0 0; 0 0 0], tri_right)
@test_throws DomainError triangle_normal(mesh_degenerate_normal, 1)
mesh_extreme_normal = TriMesh(
    Float64[0 1.0e300 0; 0 0 1.0e300; 0 0 0], tri_right)
@test triangle_normal(mesh_extreme_normal, 1) ≈ Vec3(0.0, 0.0, 1.0)

# Endpoint subtraction, determinant products, and centroid sums can overflow
# even when the exact Float64 result is finite. Compare against an independent
# high-precision evaluation of the stored binary coordinates.
max_float = floatmax(Float64)
mesh_opposite_extremes = TriMesh(
    Float64[-max_float max_float -max_float; 0 0 1.0e-308; 0 0 0],
    tri_right,
)
opposite_area_reference = setprecision(BigFloat, 4352) do
    x1, y1, z1 = BigFloat.(mesh_opposite_extremes.xyz[:, 1])
    x2, y2, z2 = BigFloat.(mesh_opposite_extremes.xyz[:, 2])
    x3, y3, z3 = BigFloat.(mesh_opposite_extremes.xyz[:, 3])
    e1x, e1y, e1z = x2 - x1, y2 - y1, z2 - z1
    e2x, e2y, e2z = x3 - x1, y3 - y1, z3 - z1
    cx = e1y * e2z - e1z * e2y
    cy = e1z * e2x - e1x * e2z
    cz = e1x * e2y - e1y * e2x
    Float64(hypot(hypot(cx, cy), cz) / 2)
end
@test triangle_area(mesh_opposite_extremes, 1) == opposite_area_reference
@test triangle_normal(mesh_opposite_extremes, 1) == Vec3(0.0, 0.0, 1.0)

cancel_scale = nextfloat(sqrt(max_float))
cancel_offset = nextfloat(cancel_scale)
mesh_cancelled_cross = TriMesh(
    Float64[0 cancel_scale cancel_scale; 0 cancel_scale cancel_offset; 0 0 0],
    tri_right,
)
cancelled_area_reference = setprecision(BigFloat, 4352) do
    scale_big = BigFloat(cancel_scale)
    offset_big = BigFloat(cancel_offset)
    Float64(abs(scale_big * offset_big - scale_big * scale_big) / 2)
end
@test triangle_area(mesh_cancelled_cross, 1) == cancelled_area_reference
@test triangle_normal(mesh_cancelled_cross, 1) == Vec3(0.0, 0.0, 1.0)

mesh_thin_translated = TriMesh(
    Float64[
         0.7310586165167978 0.22185094509767112 0.03875638900935915
        -0.9841919634964176 -0.003812620537511413 0.3486999802587598
         0.2402064357666356 0.3521966389914236 0.39246468226059505
    ],
    tri_right,
)
thin_area_reference, thin_normal_reference = setprecision(BigFloat, 512) do
    first = BigFloat.(mesh_thin_translated.xyz[:, 1])
    edge1 = BigFloat.(mesh_thin_translated.xyz[:, 2]) - first
    edge2 = BigFloat.(mesh_thin_translated.xyz[:, 3]) - first
    cross_big = (
        edge1[2] * edge2[3] - edge1[3] * edge2[2],
        edge1[3] * edge2[1] - edge1[1] * edge2[3],
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
    )
    cross_norm_big = hypot(hypot(cross_big[1], cross_big[2]), cross_big[3])
    Float64(cross_norm_big / 2),
    Vec3(ntuple(component -> Float64(
        cross_big[component] / cross_norm_big), 3))
end
@test triangle_area(mesh_thin_translated, 1) == thin_area_reference
@test triangle_normal(mesh_thin_translated, 1) == thin_normal_reference
@test DiffMoM._validate_coarsening_candidate_area_range(
    mesh_thin_translated) === nothing
thin_coarsen = coarsen_mesh_to_target_rwg(
    mesh_thin_translated, 1; area_tol_rel=0.0)
@test thin_coarsen.mesh === mesh_thin_translated

mesh_huge_collinear = TriMesh(
    Float64[0 1.0e200 2.0e200; 0 1.0e200 2.0e200; 0 0 0],
    tri_right,
)
@test triangle_area(mesh_huge_collinear, 1) == 0.0
triangle_area(mesh_huge_collinear, 1)
@test (@allocated triangle_area(mesh_huge_collinear, 1)) == 0
@test_throws DomainError triangle_normal(mesh_huge_collinear, 1)

mesh_area_overflow = TriMesh(
    Float64[0 max_float 0; 0 0 max_float; 0 0 0],
    tri_right,
)
@test_throws OverflowError triangle_area(mesh_area_overflow, 1)

# Endpoint subtraction can flip a top-range area classification in either
# direction even when the scaled cross product is well-conditioned.
area_boundary_a = 2.2224518734910558e154
area_boundary_shift = 1.4855885759426881e138
mesh_hidden_area_overflow = TriMesh(
    Float64[
        -area_boundary_shift area_boundary_a -area_boundary_shift
        -area_boundary_shift -area_boundary_shift 1.617756637437081e154
        0 0 0
    ],
    tri_right,
)
@test_throws OverflowError triangle_area(mesh_hidden_area_overflow, 1)

mesh_hidden_finite_area = TriMesh(
    Float64[
        area_boundary_shift area_boundary_a area_boundary_shift
        area_boundary_shift area_boundary_shift 1.6177566374370813e154
        0 0 0
    ],
    tri_right,
)
@test triangle_area(mesh_hidden_finite_area, 1) == floatmax(Float64)

mesh_hidden_subnormal_area = TriMesh(
    Float64[
        -1.2314112129908113e-178 1.6017659002066176e-162 -1.2314112129908113e-178
        -2.4628224259816226e-178 -2.4628224259816226e-178 3.084505955442772e-162
        0 0 0
    ],
    tri_right,
)
@test triangle_area(mesh_hidden_subnormal_area, 1) == nextfloat(0.0)
@test_throws OverflowError DiffMoM._validate_coarsening_candidate_area_range(
    mesh_hidden_area_overflow)
@test DiffMoM._validate_coarsening_candidate_area_range(
    mesh_hidden_finite_area) === nothing
@test DiffMoM._validate_coarsening_candidate_area_range(
    mesh_hidden_subnormal_area) === nothing
hidden_subnormal_coarsen = coarsen_mesh_to_target_rwg(
    mesh_hidden_subnormal_area, 1; area_tol_rel=0.0)
@test hidden_subnormal_coarsen.mesh === mesh_hidden_subnormal_area
@test hidden_subnormal_coarsen.rwg_count == 0

mesh_extreme_center = TriMesh(
    Float64[1.0e308 1.0e308 1.0e308; 0 1 0; 0 0 1],
    tri_right,
)
extreme_center_reference = setprecision(BigFloat, 256) do
    Vec3(ntuple(component -> Float64(
        sum(BigFloat(mesh_extreme_center.xyz[component, vertex]) for vertex in 1:3) / 3), 3))
end
@test triangle_center(mesh_extreme_center, 1) == extreme_center_reference

min_subnormal = nextfloat(0.0)
mesh_cancelled_center = TriMesh(
    Float64[max_float -max_float 2 * min_subnormal; 0 0 0; 0 0 0],
    tri_right,
)
@test triangle_center(mesh_cancelled_center, 1) ==
      Vec3(min_subnormal, 0.0, 0.0)
mesh_permuted_cancelled_center = TriMesh(
    Float64[max_float 2 * min_subnormal -max_float; 0 0 0; 0 0 0],
    tri_right,
)
@test triangle_center(mesh_permuted_cancelled_center, 1) ==
      Vec3(min_subnormal, 0.0, 0.0)

mesh_subnormal_center = TriMesh(
    Float64[-64 * min_subnormal -2 * min_subnormal 64 * min_subnormal;
            0 1 0;
            0 0 1],
    tri_right,
)
subnormal_center_reference = setprecision(BigFloat, 256) do
    Float64((BigFloat(-64 * min_subnormal) +
             BigFloat(-2 * min_subnormal) +
             BigFloat(64 * min_subnormal)) / 3)
end
@test subnormal_center_reference == -min_subnormal
@test triangle_center(mesh_subnormal_center, 1)[1] == subnormal_center_reference

mesh_nonfinite_geometry = TriMesh(
    Float64[0 1 NaN; 0 0 1; 0 0 0],
    tri_right,
)
@test_throws DomainError triangle_area(mesh_nonfinite_geometry, 1)
@test_throws DomainError triangle_center(mesh_nonfinite_geometry, 1)
@test_throws DomainError triangle_normal(mesh_nonfinite_geometry, 1)
@test_throws DomainError mesh_resolution_report(
    mesh_nonfinite_geometry, 1.0; points_per_wavelength=1.0, c0=1.0)

mesh_unrepresentable_edge = TriMesh(
    Float64[0 1.3e308 0; 0 1.3e308 1; 0 0 0],
    tri_right,
)
@test_throws OverflowError mesh_resolution_report(
    mesh_unrepresentable_edge, 1.0; points_per_wavelength=1.0, c0=1.0)
@test_throws OverflowError refine_mesh_to_target_edge(
    mesh_unrepresentable_edge, 1.0e308; max_iters=1)

# Rounded coordinate differences can put an exact over-range diagonal exactly
# at floatmax. The stored-coordinate oracle still requires rejection.
edge_boundary_x = 1.271161006153646e308
edge_boundary_shift = -2.4948003869183998e292
mesh_hidden_edge_overflow = TriMesh(
    Float64[
        edge_boundary_shift edge_boundary_x edge_boundary_shift
        edge_boundary_shift edge_boundary_x edge_boundary_shift
        0 0 1
    ],
    tri_right,
)
@test hypot(
    edge_boundary_x - edge_boundary_shift,
    edge_boundary_x - edge_boundary_shift,
) == floatmax(Float64)
@test_throws OverflowError mesh_resolution_report(
    mesh_hidden_edge_overflow, 1.0;
    points_per_wavelength=1.0,
    c0=floatmax(Float64),
)
@test_throws OverflowError refine_mesh_to_target_edge(
    mesh_hidden_edge_overflow, floatmax(Float64); max_iters=1)

# The reverse boundary case has a rounded `hypot == Inf`, while exact stored
# endpoint differences still round back to floatmax and must remain usable.
edge_finite_d1 = -1.2674639246533443e308
edge_finite_d2 = 1.2748473660926795e308
edge_finite_s1 = -9.959243144578252e291
edge_finite_s2 = 9.959243144578252e291
mesh_hidden_finite_edge = TriMesh(
    Float64[
        edge_finite_s1 edge_finite_d1 edge_finite_s1
        edge_finite_s2 edge_finite_d2 edge_finite_s2
        0 0 1
    ],
    tri_right,
)
@test hypot(
    edge_finite_d1 - edge_finite_s1,
    edge_finite_d2 - edge_finite_s2,
) == Inf
hidden_finite_edge_report = mesh_resolution_report(
    mesh_hidden_finite_edge, 1.0;
    points_per_wavelength=1.0,
    c0=floatmax(Float64),
)
@test hidden_finite_edge_report.edge_max_m == floatmax(Float64)
@test hidden_finite_edge_report.meets_target

mesh_resolution_extreme_mean = TriMesh(
    Float64[0 5.0e307 1.0e308; 0 0 0; 0 0 0],
    tri_right,
)
resolution_extreme_report = mesh_resolution_report(
    mesh_resolution_extreme_mean, 1.0; points_per_wavelength=1.0, c0=1.0)
resolution_mean_reference = setprecision(BigFloat, 256) do
    Float64((BigFloat(5.0e307) + BigFloat(5.0e307) + BigFloat(1.0e308)) / 3)
end
@test resolution_extreme_report.edge_mean_m == resolution_mean_reference
println("  32b: PASS")

# 32c: STL binary round-trip
println("  32c: STL binary round-trip ...")
stl_bin_path = joinpath(DATADIR, "tmp_roundtrip_bin.stl")
mesh_plate = make_rect_plate(0.1, 0.1, 3, 3)
write_stl_mesh(stl_bin_path, mesh_plate)
mesh_stl_bin = read_stl_mesh(stl_bin_path)
# Binary STL integers and Float32 values are little-endian by specification.
stl_bin_bytes = read(stl_bin_path)
nt_stl = ntriangles(mesh_plate)
@assert stl_bin_bytes[81] == UInt8(nt_stl & 0xff)
@assert stl_bin_bytes[82] == UInt8((nt_stl >> 8) & 0xff)
@assert stl_bin_bytes[83] == UInt8((nt_stl >> 16) & 0xff)
@assert stl_bin_bytes[84] == UInt8((nt_stl >> 24) & 0xff)
@assert DiffMoM._stl_is_binary(stl_bin_bytes)
# STL uses Float32 internally, so vertex count may differ slightly from merging.
# But triangle count must match since each facet is independent.
@assert ntriangles(mesh_stl_bin) == ntriangles(mesh_plate) "STL binary round-trip triangle count mismatch: got $(ntriangles(mesh_stl_bin)), expected $(ntriangles(mesh_plate))"
@assert nvertices(mesh_stl_bin) == nvertices(mesh_plate) "STL binary round-trip vertex count mismatch: got $(nvertices(mesh_stl_bin)), expected $(nvertices(mesh_plate))"
report_stl_bin = mesh_quality_report(mesh_stl_bin)
@assert mesh_quality_ok(report_stl_bin; allow_boundary=true) "STL binary round-trip mesh quality check failed"
@test_throws ArgumentError read_stl_mesh(stl_bin_path; max_vertices=1)
@test_throws ArgumentError read_stl_mesh(stl_bin_path; max_triangles=1)
@test_throws ArgumentError read_stl_mesh(stl_bin_path; max_raw_bytes=24)
@test_throws ArgumentError read_mesh(stl_bin_path; max_vertices=1)
# Check vertex positions (Float32 precision ~ 1e-6)
for t in 1:ntriangles(mesh_plate)
    for vi in 1:3
        idx_orig = mesh_plate.tri[vi, t]
        idx_stl = mesh_stl_bin.tri[vi, t]
        for d in 1:3
            @assert abs(mesh_stl_bin.xyz[d, idx_stl] - Float64(Float32(mesh_plate.xyz[d, idx_orig]))) < 1e-10 "STL binary vertex mismatch at tri $t vert $vi dim $d"
        end
    end
end

# Reject invalid or unrepresentable meshes before opening the destination, so
# predictable validation failures cannot replace an existing file with a
# partial STL.  Large finite Float64 coordinates remain valid for ASCII STL.
stl_validation_path = joinpath(DATADIR, "tmp_stl_validation.stl")
stl_validation_sentinel = UInt8[0x44, 0x69, 0x66, 0x66, 0x4d, 0x6f, 0x4d]
mesh_empty_stl = TriMesh(zeros(Float64, 3, 0), Matrix{Int}(undef, 3, 0))
for ascii_stl in (false, true)
    write(stl_validation_path, stl_validation_sentinel)
    @test_throws ArgumentError write_stl_mesh(
        stl_validation_path, mesh_empty_stl; ascii=ascii_stl)
    @test read(stl_validation_path) == stl_validation_sentinel
end
mesh_extreme_stl = TriMesh(
    Float64[0.0 1.0e100 0.0; 0.0 0.0 1.0e100; 0.0 0.0 0.0],
    reshape([1, 2, 3], 3, 1),
)
write(stl_validation_path, stl_validation_sentinel)
@test_throws ArgumentError write_stl_mesh(stl_validation_path, mesh_extreme_stl)
@test read(stl_validation_path) == stl_validation_sentinel

stl_extreme_ascii_path = joinpath(DATADIR, "tmp_stl_extreme_ascii.stl")
write_stl_mesh(stl_extreme_ascii_path, mesh_extreme_stl; ascii=true)
mesh_extreme_stl_rt = read_stl_mesh(stl_extreme_ascii_path)
@test mesh_extreme_stl_rt.xyz == mesh_extreme_stl.xyz
@test mesh_extreme_stl_rt.tri == mesh_extreme_stl.tri

mesh_bad_stl_index = TriMesh(
    Float64[0 1 0; 0 0 1; 0 0 0], reshape([1, 2, 4], 3, 1))
write(stl_validation_path, stl_validation_sentinel)
@test_throws ArgumentError write_stl_mesh(stl_validation_path, mesh_bad_stl_index)
@test read(stl_validation_path) == stl_validation_sentinel

write(stl_validation_path, stl_validation_sentinel)
@test_throws DomainError write_stl_mesh(
    stl_validation_path, mesh_degenerate_normal)
@test read(stl_validation_path) == stl_validation_sentinel

# Binary STL must reject Float32 quantization that changes topology, and it
# must do so before opening an existing destination. ASCII retains Float64.
mesh_stl_quantization_collapse = TriMesh(
    Float64[1.0 1.0 + 1.0e-8 1.0; 0.0 0.0 1.0; 0.0 0.0 0.0],
    reshape(Int[1, 2, 3], 3, 1),
)
write(stl_validation_path, stl_validation_sentinel)
@test_throws ArgumentError write_stl_mesh(
    stl_validation_path, mesh_stl_quantization_collapse)
@test read(stl_validation_path) == stl_validation_sentinel
stl_quantization_ascii_path = joinpath(DATADIR, "tmp_stl_quantization_ascii.stl")
write_stl_mesh(
    stl_quantization_ascii_path, mesh_stl_quantization_collapse; ascii=true)
mesh_stl_quantization_ascii = read_stl_mesh(stl_quantization_ascii_path)
@test nvertices(mesh_stl_quantization_ascii) == 3
@test mesh_quality_ok(
    mesh_quality_report(mesh_stl_quantization_ascii); allow_boundary=true)

# Distinct Float32 vertices can still become collinear after quantization.
mesh_stl_quantized_collinear = TriMesh(
    Float64[0.0 1.0 2.0; 0.0 1.0 2.0 + 1.0e-8; 0.0 0.0 0.0],
    reshape(Int[1, 2, 3], 3, 1),
)
write(stl_validation_path, stl_validation_sentinel)
@test_throws ArgumentError write_stl_mesh(
    stl_validation_path, mesh_stl_quantized_collinear)
@test read(stl_validation_path) == stl_validation_sentinel

# Quantization can also reverse a nondegenerate facet without merging any
# vertex. Preserve the oriented surface or reject binary output.
mesh_stl_quantized_winding = TriMesh(
    Float64[
        -2.0766889876912712e-47 0.9999999719816558 2.0000001011465844
        5.5201520571433606e-46 0.9999999895400845 2.0000001223452344
        0.0 0.0 0.0
    ],
    reshape(Int[1, 2, 3], 3, 1),
)
@test triangle_normal(mesh_stl_quantized_winding, 1) == Vec3(0.0, 0.0, -1.0)
write(stl_validation_path, stl_validation_sentinel)
@test_throws ArgumentError write_stl_mesh(
    stl_validation_path, mesh_stl_quantized_winding)
@test read(stl_validation_path) == stl_validation_sentinel
stl_winding_ascii_path = joinpath(DATADIR, "tmp_stl_winding_ascii.stl")
write_stl_mesh(stl_winding_ascii_path, mesh_stl_quantized_winding; ascii=true)
mesh_stl_winding_ascii = read_stl_mesh(stl_winding_ascii_path)
@test triangle_normal(mesh_stl_winding_ascii, 1) ==
      triangle_normal(mesh_stl_quantized_winding, 1)

# Near orthogonality, rounding the serialized normal itself can give the
# opposite orientation sign. The predicate must use the endpoint geometry.
mesh_stl_near_orthogonal_winding = TriMesh(
    Float64[
        0.0 6.992565167042597e-40 6.992602567486016e-40
        0.0 6.992942005328153e-40 6.992975788750699e-40
        0.0 6.992513989908744e-40 6.992544152654118e-40
    ],
    reshape(Int[1, 2, 3], 3, 1),
)
near_orthogonal_coords = ntuple(local_vertex ->
    DiffMoM._binary_stl_coordinate(
        mesh_stl_near_orthogonal_winding, local_vertex, 1), 3)
@test DiffMoM._binary_stl_orientation_sign_exact(
    mesh_stl_near_orthogonal_winding,
    1,
    near_orthogonal_coords...,
) == -1
write(stl_validation_path, stl_validation_sentinel)
@test_throws ArgumentError write_stl_mesh(
    stl_validation_path, mesh_stl_near_orthogonal_winding)
@test read(stl_validation_path) == stl_validation_sentinel

# The serialized facet normal must describe the quantized coordinates, not
# the higher-precision source triangle.
mesh_stl_quantized_normal = TriMesh(
    Float64[
        293.24681575467264 -190.57731472668434 -526.9051943781989
        511.2936686990998 -526.0714577166923 -327.5233177931973
        -545.1941430238312 1082.7882577480282 -305.01398683138245
    ],
    reshape(Int[1, 2, 3], 3, 1),
)
stl_quantized_normal_path = joinpath(DATADIR, "tmp_stl_quantized_normal.stl")
write_stl_mesh(stl_quantized_normal_path, mesh_stl_quantized_normal)
stl_quantized_normal_bytes = read(stl_quantized_normal_path)
stored_stl_normal = (
    DiffMoM._stl_float32_le(stl_quantized_normal_bytes, 85),
    DiffMoM._stl_float32_le(stl_quantized_normal_bytes, 89),
    DiffMoM._stl_float32_le(stl_quantized_normal_bytes, 93),
)
quantized_normal_mesh = TriMesh(
    Float64.(Float32.(mesh_stl_quantized_normal.xyz)),
    copy(mesh_stl_quantized_normal.tri),
)
expected_stl_normal = Tuple(Float32.(triangle_normal(quantized_normal_mesh, 1)))
source_stl_normal = Tuple(Float32.(triangle_normal(mesh_stl_quantized_normal, 1)))
@test stored_stl_normal == expected_stl_normal
@test stored_stl_normal != source_stl_normal
println("  32c: PASS")

# 32d: STL ASCII round-trip
println("  32d: STL ASCII round-trip ...")
stl_ascii_path = joinpath(DATADIR, "tmp_ascii.stl")
open(stl_ascii_path, "w") do io
    println(io, "solid test")
    println(io, "  facet normal 0 0 1")
    println(io, "    outer loop")
    println(io, "      vertex 0.0 0.0 0.0")
    println(io, "      vertex 1.0 0.0 0.0")
    println(io, "      vertex 1.0 1.0 0.0")
    println(io, "    endloop")
    println(io, "  endfacet")
    println(io, "  facet normal 0 0 1")
    println(io, "    outer loop")
    println(io, "      vertex 0.0 0.0 0.0")
    println(io, "      vertex 1.0 1.0 0.0")
    println(io, "      vertex 0.0 1.0 0.0")
    println(io, "    endloop")
    println(io, "  endfacet")
    println(io, "endsolid test")
end
mesh_stl_ascii = read_stl_mesh(stl_ascii_path)
@assert nvertices(mesh_stl_ascii) == 4 "STL ASCII: expected 4 unique vertices, got $(nvertices(mesh_stl_ascii))"
@assert ntriangles(mesh_stl_ascii) == 2 "STL ASCII: expected 2 triangles, got $(ntriangles(mesh_stl_ascii))"
@test_throws ArgumentError read_stl_mesh(stl_ascii_path; max_vertices=3)
@test_throws ArgumentError read_stl_mesh(stl_ascii_path; max_triangles=1)
@test_throws ArgumentError read_stl_mesh(stl_ascii_path; max_raw_bytes=143)
@test_throws ArgumentError read_stl_mesh(stl_ascii_path; max_input_bytes=1)
@test_throws ArgumentError read_stl_mesh(stl_ascii_path; max_line_bytes=4)

# Facet boundaries, rather than a global vertex count, define ASCII STL
# triangles. Do not regroup six vertices from one facet into two triangles.
stl_malformed_facets_path = joinpath(DATADIR, "tmp_stl_malformed_facets.stl")
open(stl_malformed_facets_path, "w") do io
    println(io, "solid malformed")
    println(io, "facet normal 0 0 1")
    println(io, "outer loop")
    for coord in ((0, 0, 0), (1, 0, 0), (0, 1, 0),
                  (2, 0, 0), (3, 0, 0), (2, 1, 0))
        println(io, "vertex $(coord[1]) $(coord[2]) $(coord[3])")
    end
    println(io, "endloop")
    println(io, "endfacet")
    println(io, "facet normal 0 0 1")
    println(io, "outer loop")
    println(io, "endloop")
    println(io, "endfacet")
    println(io, "endsolid malformed")
end
@test_throws ErrorException read_stl_mesh(stl_malformed_facets_path)
println("  32d: PASS")

# 32e: STL vertex merging (tetrahedron)
println("  32e: STL vertex merging ...")
# Build a tetrahedron: 4 triangles, 4 unique vertices, but 12 raw vertices in STL
xyz_tet = Float64[0 1 0.5 0.5; 0 0 sqrt(3)/2 sqrt(3)/6; 0 0 0 sqrt(2/3)]
tri_tet = [1 1 1 2; 2 2 3 3; 3 4 4 4]
mesh_tet = TriMesh(xyz_tet, tri_tet)
stl_tet_path = joinpath(DATADIR, "tmp_tetra.stl")
write_stl_mesh(stl_tet_path, mesh_tet)
mesh_tet_rt = read_stl_mesh(stl_tet_path)
@assert nvertices(mesh_tet_rt) == 4 "Tetrahedron STL: expected 4 unique vertices after merge, got $(nvertices(mesh_tet_rt))"
@assert ntriangles(mesh_tet_rt) == 4 "Tetrahedron STL: expected 4 triangles, got $(ntriangles(mesh_tet_rt))"

# A positive merge tolerance is a Euclidean distance, including points across
# spatial-hash cell boundaries; it must not merge farther points in one cell.
raw_within_tol = [
    (0.49, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0),
    (0.51, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0),
]
raw_beyond_tol = [
    (0.49, 0.49, 0.49), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0),
    (-0.49, -0.49, -0.49), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0),
]
@assert nvertices(DiffMoM._merge_stl_vertices(
    raw_within_tol, 2; merge_tol=1.0)) == 3
@assert nvertices(DiffMoM._merge_stl_vertices(
    raw_beyond_tol, 2; merge_tol=1.0)) == 4

# Signed zero denotes the same STL coordinate, and a positive tolerance must
# not depend on the mesh's absolute translation.
raw_signed_zero = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (-0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
]
mesh_signed_zero = DiffMoM._merge_stl_vertices(raw_signed_zero, 2)
@test nvertices(mesh_signed_zero) == 3
@test all(x -> !iszero(x) || !signbit(x), mesh_signed_zero.xyz)

raw_translation_origin = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (5.0e-11, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
]
raw_translation_shifted = map(
    coord -> (coord[1] + 1.0e10, coord[2], coord[3]),
    raw_translation_origin,
)
@test nvertices(DiffMoM._merge_stl_vertices(
    raw_translation_origin, 2; merge_tol=1.0e-10)) == 3
@test nvertices(DiffMoM._merge_stl_vertices(
    raw_translation_shifted, 2; merge_tol=1.0e-10)) == 3

# A late full-range coordinate promotes the spatial index to exact BigInt
# cells. The obsolete small-cell hash table must release its O(N) capacity.
stl_promoted_merger = DiffMoM._new_stl_vertex_merger(1.0)
for i in 1:1000
    DiffMoM._merge_stl_vertex!(
        stl_promoted_merger, (10.0 * i, 0.0, 0.0))
end
stl_small_index_bytes = Base.summarysize(stl_promoted_merger.bucket_heads)
@test DiffMoM._merge_stl_vertex!(
    stl_promoted_merger, (floatmax(Float64), 0.0, 0.0)) == 1001
@test stl_promoted_merger.use_big_cells
@test isempty(stl_promoted_merger.bucket_heads)
@test Base.summarysize(stl_promoted_merger.bucket_heads) < 1024
@test stl_small_index_bytes >
      100 * Base.summarysize(stl_promoted_merger.bucket_heads)
for invalid_tol in (-1.0, Inf, NaN)
    @test_throws ArgumentError read_stl_mesh(stl_tet_path; merge_tol=invalid_tol)
end

# The binary reader streams fixed-size facet records; allocation must remain
# well below the former whole-file/per-coordinate-slice implementation.
stl_alloc_path = joinpath(DATADIR, "tmp_alloc_bin.stl")
stl_alloc_mesh = make_rect_plate(1.0, 1.0, 40, 40)
write_stl_mesh(stl_alloc_path, stl_alloc_mesh)
read_stl_mesh(stl_alloc_path)  # warm compilation
GC.gc()
stl_write_alloc = @allocated write_stl_mesh(stl_alloc_path, stl_alloc_mesh)
stl_read_alloc = @allocated read_stl_mesh(stl_alloc_path)
@assert stl_write_alloc < 16 * filesize(stl_alloc_path)
@assert stl_read_alloc < 8 * filesize(stl_alloc_path)
println("  32e: PASS")

# 32f: MSH v2 import
println("  32f: MSH v2 import ...")
msh_v2_path = joinpath(DATADIR, "tmp_v2.msh")
open(msh_v2_path, "w") do io
    println(io, "\$MeshFormat")
    println(io, "2.2 0 8")
    println(io, "\$EndMeshFormat")
    println(io, "\$Nodes")
    println(io, "4")
    println(io, "1 0.0 0.0 0.0")
    println(io, "2 1.0 0.0 0.0")
    println(io, "3 1.0 1.0 0.0")
    println(io, "4 0.0 1.0 0.0")
    println(io, "\$EndNodes")
    println(io, "\$Elements")
    println(io, "3")
    println(io, "1 1 2 0 1 1 2")         # line element (should be skipped)
    println(io, "2 2 2 0 1 1 2 3")       # triangle 1
    println(io, "3 2 2 0 1 1 3 4")       # triangle 2
    println(io, "\$EndElements")
end
mesh_msh_v2 = read_msh_mesh(msh_v2_path)
@assert nvertices(mesh_msh_v2) == 4 "MSH v2: expected 4 vertices, got $(nvertices(mesh_msh_v2))"
@assert ntriangles(mesh_msh_v2) == 2 "MSH v2: expected 2 triangles, got $(ntriangles(mesh_msh_v2))"
report_msh_v2 = mesh_quality_report(mesh_msh_v2)
@assert report_msh_v2.n_invalid_triangles == 0
@assert report_msh_v2.n_degenerate_triangles == 0
@test_throws ArgumentError read_msh_mesh(msh_v2_path; max_vertices=3)
@test_throws ArgumentError read_msh_mesh(msh_v2_path; max_triangles=1)
@test_throws ArgumentError read_msh_mesh(msh_v2_path; max_raw_bytes=143)
@test_throws ArgumentError read_msh_mesh(msh_v2_path; max_input_bytes=1)
@test_throws ArgumentError read_msh_mesh(msh_v2_path; max_line_bytes=3)

# Untrusted declared counts must fail before count-sized hints or tag arrays.
msh_v2_declared_path = joinpath(DATADIR, "tmp_v2_declared_resource.msh")
write(msh_v2_declared_path,
      "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes\n100000\n")
try
    read_msh_mesh(msh_v2_declared_path; max_vertices=200_000)
catch
end
GC.gc()
@test (@allocated try
    read_msh_mesh(msh_v2_declared_path; max_vertices=200_000)
catch
end) < 100_000
@test_throws ArgumentError read_msh_mesh(
    msh_v2_declared_path; max_vertices=100)

# The MSH reader streams records and tokenizes without per-field heap objects.
msh_alloc_path = joinpath(DATADIR, "tmp_alloc_v2.msh")
msh_grid_n = 20
open(msh_alloc_path, "w") do io
    nv_grid = (msh_grid_n + 1)^2
    nt_grid = 2 * msh_grid_n^2
    println(io, "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes")
    println(io, nv_grid)
    for j in 0:msh_grid_n, i in 0:msh_grid_n
        node_id = i + 1 + j * (msh_grid_n + 1)
        println(io, node_id, " ", i / msh_grid_n, " ", j / msh_grid_n, " 0")
    end
    println(io, "\$EndNodes\n\$Elements")
    println(io, nt_grid)
    element_id = 1
    for j in 0:(msh_grid_n - 1), i in 0:(msh_grid_n - 1)
        n1 = i + 1 + j * (msh_grid_n + 1)
        n2 = n1 + 1
        n4 = n1 + msh_grid_n + 1
        n3 = n4 + 1
        println(io, element_id, " 2 0 ", n1, " ", n2, " ", n3)
        element_id += 1
        println(io, element_id, " 2 0 ", n1, " ", n3, " ", n4)
        element_id += 1
    end
    println(io, "\$EndElements")
end
read_msh_mesh(msh_alloc_path)  # warm compilation
GC.gc()
msh_read_alloc = @allocated read_msh_mesh(msh_alloc_path)
@assert msh_read_alloc < 20 * filesize(msh_alloc_path)
println("  32f: PASS")

# 32g: MSH v4 import
println("  32g: MSH v4 import ...")
msh_v4_path = joinpath(DATADIR, "tmp_v4.msh")
open(msh_v4_path, "w") do io
    println(io, "\$MeshFormat")
    println(io, "4.1 0 8")
    println(io, "\$EndMeshFormat")
    println(io, "\$Nodes")
    println(io, "1 3 10 30")             # 1 entity block, 3 nodes, non-contiguous tags
    println(io, "2 1 0 3")               # entity dim=2, tag=1, parametric=0, 3 nodes
    println(io, "10 20 30")               # node tags may share one whitespace-delimited line
    println(io, "0.0 0.0 0.0")          # node coordinates
    println(io, "1.0 0.0 0.0")
    println(io, "0.5 1.0 0.0")
    println(io, "\$EndNodes")
    println(io, "\$Elements")
    println(io, "1 1 1 1")               # 1 entity block, 1 element, min tag 1, max tag 1
    println(io, "2 1 2 1")               # entity dim=2, tag=1, type=2 (triangle), 1 element
    println(io, "1 10 20 30")            # element tag 1, nodes 10 20 30
    println(io, "\$EndElements")
end
mesh_msh_v4 = read_msh_mesh(msh_v4_path)
@assert nvertices(mesh_msh_v4) == 3 "MSH v4: expected 3 vertices, got $(nvertices(mesh_msh_v4))"
@assert ntriangles(mesh_msh_v4) == 1 "MSH v4: expected 1 triangle, got $(ntriangles(mesh_msh_v4))"
# Verify vertex positions
@assert abs(mesh_msh_v4.xyz[1, 1] - 0.0) < 1e-12
@assert abs(mesh_msh_v4.xyz[1, 2] - 1.0) < 1e-12
@assert abs(mesh_msh_v4.xyz[2, 3] - 1.0) < 1e-12

msh_v4_declared_path = joinpath(DATADIR, "tmp_v4_declared_resource.msh")
write(msh_v4_declared_path,
      "\$MeshFormat\n4.1 0 8\n\$EndMeshFormat\n" *
      "\$Nodes\n1 100000 1 100000\n2 1 0 100000\n")
try
    read_msh_mesh(msh_v4_declared_path; max_vertices=200_000)
catch
end
GC.gc()
@test (@allocated try
    read_msh_mesh(msh_v4_declared_path; max_vertices=200_000)
catch
end) < 100_000
@test_throws ArgumentError read_msh_mesh(
    msh_v4_declared_path; max_vertices=100)

msh_binary_path = joinpath(DATADIR, "tmp_binary_header.msh")
open(msh_binary_path, "w") do io
    println(io, "\$MeshFormat\n4.1 1 8\n\$EndMeshFormat")
end
@test_throws ErrorException read_msh_mesh(msh_binary_path)

msh_unsupported_path = joinpath(DATADIR, "tmp_unsupported_version.msh")
open(msh_unsupported_path, "w") do io
    println(io, "\$MeshFormat\n3.0 0 8\n\$EndMeshFormat")
end
@test_throws ErrorException read_msh_mesh(msh_unsupported_path)

msh_missing_node_path = joinpath(DATADIR, "tmp_missing_node.msh")
open(msh_missing_node_path, "w") do io
    println(io, "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes\n3")
    println(io, "1 0 0 0\n2 1 0 0\n3 0 1 0\n\$EndNodes")
    println(io, "\$Elements\n1\n1 2 0 1 2 99\n\$EndElements")
end
@test_throws ErrorException read_msh_mesh(msh_missing_node_path)
@test_throws ErrorException DiffMoM._parse_msh_v2_node("1 NaN 0 0", msh_v2_path)
println("  32g: PASS")

# 32h: Unified read_mesh / write_mesh dispatcher
println("  32h: Unified read_mesh / write_mesh dispatcher ...")
# OBJ dispatch
mesh_dispatch_obj = read_mesh(obj_rt_path)
@assert nvertices(mesh_dispatch_obj) == nvertices(mesh) "read_mesh .obj dispatch failed"

# STL dispatch
mesh_dispatch_stl = read_mesh(stl_bin_path)
@assert ntriangles(mesh_dispatch_stl) == ntriangles(mesh_plate) "read_mesh .stl dispatch failed"

# MSH dispatch
mesh_dispatch_msh = read_mesh(msh_v2_path)
@assert ntriangles(mesh_dispatch_msh) == 2 "read_mesh .msh dispatch failed"

# write_mesh OBJ
write_out_obj = joinpath(DATADIR, "tmp_write_dispatch.obj")
write_mesh(write_out_obj, mesh)
@assert isfile(write_out_obj)

# write_mesh STL
write_out_stl = joinpath(DATADIR, "tmp_write_dispatch.stl")
write_mesh(write_out_stl, mesh)
@assert isfile(write_out_stl)

# Unsupported extension
thrown_ext = try
    read_mesh(joinpath(DATADIR, "fake.xyz"))
    false
catch
    true
end
@assert thrown_ext "read_mesh should throw on unsupported extension"

thrown_ext_w = try
    write_mesh(joinpath(DATADIR, "fake.xyz"), mesh)
    false
catch
    true
end
@assert thrown_ext_w "write_mesh should throw on unsupported extension"
println("  32h: PASS")

# 32i: convert_cad_to_mesh (skip if gmsh not available)
println("  32i: convert_cad_to_mesh (gmsh check) ...")
gmsh_available = Sys.which("gmsh") !== nothing
mktempdir() do cad_test_dir
    cad_test_path = joinpath(cad_test_dir, "probe.step")
    write(cad_test_path, "")
    cad_test_output = joinpath(cad_test_dir, "probe.msh")
    missing_gmsh = joinpath(cad_test_dir, "missing-gmsh")
    for invalid_mesh_size in (-1.0, NaN, Inf)
        mesh_size_error = try
            convert_cad_to_mesh(
                cad_test_path, cad_test_output;
                mesh_size=invalid_mesh_size, gmsh_exe=missing_gmsh)
            nothing
        catch err
            err
        end
        @test mesh_size_error isa ArgumentError
        @test occursin("mesh_size", sprint(showerror, mesh_size_error))
    end

    if !gmsh_available
        # Verify the availability diagnostic after all local arguments pass.
        thrown_gmsh = try
            convert_cad_to_mesh(cad_test_path, cad_test_output)
            false
        catch e
            occursin("Gmsh", sprint(showerror, e))
        end
        @assert thrown_gmsh "convert_cad_to_mesh should mention gmsh in error"
    end
end
if !gmsh_available
    println("  32i: SKIP (gmsh not installed) — error message verified")
else
    println("  32i: SKIP (gmsh available but no test CAD file) — presence verified")
end

# 32j: Closed-surface mesh workflow
println("  32j: Closed-surface mesh workflow ...")
ico_path = joinpath(DATADIR, "tmp_icosphere.obj")
write_icosphere_obj(ico_path; radius=0.05, subdivisions=2)
mesh_ico = read_obj_mesh(ico_path)
report_ico = mesh_quality_report(mesh_ico)
@assert report_ico.n_boundary_edges == 0 "Icosphere should have no boundary edges, got $(report_ico.n_boundary_edges)"
@assert report_ico.n_nonmanifold_edges == 0 "Icosphere should have no non-manifold edges"
@assert mesh_quality_ok(report_ico; allow_boundary=false, require_closed=true) "Icosphere should pass closed-surface quality check"
println("  32j: PASS")

# 32k: mesh_resolution_ok with :p95 and :median criteria
println("  32k: mesh_resolution_ok criteria ...")
# Use a coarse mesh that fails :max but could pass :p95 or :median
mesh_res_test = make_rect_plate(1.0, 1.0, 2, 2)
report_res_test = mesh_resolution_report(mesh_res_test, 3e8; points_per_wavelength=2.0)
# The mesh has edges of similar length, so all criteria should give same result
res_max = mesh_resolution_ok(report_res_test; criterion=:max)
res_p95 = mesh_resolution_ok(report_res_test; criterion=:p95)
res_med = mesh_resolution_ok(report_res_test; criterion=:median)
# :median is most lenient, :max is strictest
# If :max passes, all must pass. If :max fails, :median may still pass.
if res_max
    @assert res_p95 && res_med ":max passed but :p95 or :median failed — logic error"
end
# :p95 and :median should never be stricter than :max
if !res_p95
    @assert !res_max ":p95 failed but :max passed — impossible"
end
# Verify the criteria use different statistics
@assert report_res_test.edge_max_m >= report_res_test.edge_p95_m >= report_res_test.edge_median_m "Edge statistics ordering violated"
# Unsupported criterion should throw
thrown_crit = try
    mesh_resolution_ok(report_res_test; criterion=:bogus)
    false
catch
    true
end
@assert thrown_crit "mesh_resolution_ok should throw on unknown criterion"
println("  32k: PASS")

# 32l: STL ASCII write and read-back
println("  32l: STL ASCII write round-trip ...")
stl_ascii_rt_path = joinpath(DATADIR, "tmp_ascii_roundtrip.stl")
mesh_small = make_rect_plate(0.05, 0.05, 2, 2)
write_stl_mesh(stl_ascii_rt_path, mesh_small; ascii=true)
mesh_ascii_rt = read_stl_mesh(stl_ascii_rt_path)
@assert ntriangles(mesh_ascii_rt) == ntriangles(mesh_small) "STL ASCII round-trip triangle mismatch"
@assert nvertices(mesh_ascii_rt) == nvertices(mesh_small) "STL ASCII round-trip vertex mismatch"
stl_header_path = joinpath(DATADIR, "tmp_stl_header_validation.stl")
stl_header_sentinel = UInt8[0x44, 0x69, 0x66, 0x66, 0x4d, 0x6f, 0x4d]
for unsafe_header in ("safe\nvertex 9 9 9", "safe\rfacet normal 0 0 1")
    write(stl_header_path, stl_header_sentinel)
    @test_throws ArgumentError write_stl_mesh(
        stl_header_path, mesh_small; ascii=true, header=unsafe_header)
    @test read(stl_header_path) == stl_header_sentinel
end
println("  32l: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 33: Spatial patch assignment
# ─────────────────────────────────────────────────
println("\n── Test 33: Spatial patch assignment ──")

# 33a: Grid-based partitioning
println("  33a: Grid-based partitioning ...")
part_grid = assign_patches_grid(mesh; nx=3, ny=2, nz=1)
@assert length(part_grid.tri_patch) == ntriangles(mesh) "Grid partition should cover all triangles"
@assert part_grid.P >= 1 "Grid partition should have at least 1 patch"
@assert part_grid.P <= 6 "Grid partition should have at most nx*ny patches"
@assert all(1 .<= part_grid.tri_patch .<= part_grid.P) "All patch IDs should be in [1, P]"
translated_xyz = copy(mesh.xyz)
translated_xyz[3, :] .+= 1.0e9
translated_grid = assign_patches_grid(
    TriMesh(translated_xyz, copy(mesh.tri)); nx=3, ny=2, nz=1)
@test translated_grid.tri_patch == part_grid.tri_patch
@test_throws ArgumentError assign_patches_grid(empty_mesh)
@test_throws ArgumentError assign_patches_grid(mesh; nx=0)
@test_throws ArgumentError assign_patches_grid(
    mesh; nx=typemax(Int), ny=2, nz=1)
println("    $Nt triangles → $(part_grid.P) patches")
println("  33a: PASS")

# 33b: Region-based partitioning
println("  33b: Region-based partitioning ...")
# Split plate at y=0 (mesh is centered)
region_top = region_halfspace(; axis=:y, threshold=0.0, above=true)
region_bot = region_halfspace(; axis=:y, threshold=0.0, above=false)
part_region = assign_patches_by_region(mesh, [region_top, region_bot])
@assert part_region.P == 2 "Two half-space regions should give exactly 2 patches"
@assert all(1 .<= part_region.tri_patch .<= 2) "All triangles should be assigned to patch 1 or 2"
n_top = count(==(1), part_region.tri_patch)
n_bot = count(==(2), part_region.tri_patch)
@assert n_top > 0 && n_bot > 0 "Both regions should have triangles"
@test_throws ArgumentError assign_patches_by_region(empty_mesh, [region_top])
@test_throws ArgumentError assign_patches_by_region(mesh, Function[])
@test_throws ArgumentError region_halfspace(axis=:invalid, threshold=0.0)
@test_throws ArgumentError region_halfspace(axis=:x, threshold=NaN)
println("    Region 1 (y>=0): $n_top tris,  Region 2 (y<0): $n_bot tris")
println("  33b: PASS")

# 33c: Sphere and box predicates
println("  33c: Sphere and box predicates ...")
pred_sphere = region_sphere(; center=Vec3(0.0, 0.0, 0.0), radius=0.01)
pred_box = region_box(; lo=Vec3(-0.01, -0.01, -1.0), hi=Vec3(0.01, 0.01, 1.0))
# Test that predicates are callable and return Bool
@assert pred_sphere(Vec3(0.0, 0.0, 0.0)) == true "Origin should be inside sphere"
@assert pred_sphere(Vec3(100.0, 0.0, 0.0)) == false "Far point should be outside sphere"
pred_unit_sphere = region_sphere(
    center=Vec3(0.0, 0.0, 0.0), radius=1.0)
@test !pred_unit_sphere(
    Vec3(prevfloat(1.0), ldexp(1.0, -26), 0.0))
pred_sphere_extreme = region_sphere(
    center=Vec3(floatmax(Float64), 0.0, 0.0),
    radius=floatmax(Float64))
@test !pred_sphere_extreme(
    Vec3(-floatmax(Float64), 0.0, 0.0))
@assert pred_box(Vec3(0.0, 0.0, 0.0)) == true "Origin should be inside box"
@assert pred_box(Vec3(1.0, 0.0, 0.0)) == false "Point outside should fail"
@test_throws ArgumentError region_sphere(
    center=Vec3(NaN, 0.0, 0.0), radius=1.0)
@test_throws ArgumentError region_sphere(
    center=Vec3(0.0, 0.0, 0.0), radius=-1.0)
@test_throws ArgumentError region_box(
    lo=Vec3(1.0, 0.0, 0.0), hi=Vec3(0.0, 1.0, 1.0))
@test_throws ArgumentError region_box(
    lo=Vec3(-Inf, -1.0, -1.0), hi=Vec3(Inf, 1.0, 1.0))
println("  33c: PASS")

# 33d: Uniform k-means partitioning
println("  33d: Uniform k-means partitioning ...")
n_target = 5
part_kmeans = assign_patches_uniform(mesh; n_patches=n_target)
@test_throws ArgumentError assign_patches_uniform(empty_mesh; n_patches=1)
@test_throws ArgumentError assign_patches_uniform(mesh; n_patches=0)
@test_throws ArgumentError assign_patches_grid(
    mesh; nx=10_000_001, ny=1, nz=1)
@assert length(part_kmeans.tri_patch) == ntriangles(mesh)
@assert part_kmeans.P >= 1 && part_kmeans.P <= n_target
@assert all(1 .<= part_kmeans.tri_patch .<= part_kmeans.P)
println("    Requested $n_target patches → got $(part_kmeans.P)")

# Coincident centroids can empty a k-means cluster; IDs must still be compact.
duplicate_patch_mesh = TriMesh(
    Float64[0 1 0; 0 0 1; 0 0 0],
    [1 1; 2 2; 3 3],
)
duplicate_partition = assign_patches_uniform(
    duplicate_patch_mesh; n_patches=2)
@test duplicate_partition.P == 1
@test duplicate_partition.tri_patch == [1, 1]

# Translating finite centroids near the top of the Float64 range must not
# overflow center sums or change the geometric clustering.
patch_translation_base = make_rect_plate(1.0, 4.0, 1, 4)
patch_translation_reference = assign_patches_uniform(
    patch_translation_base; n_patches=2)
patch_translation_mesh = TriMesh(
    copy(patch_translation_base.xyz), copy(patch_translation_base.tri))
patch_translation_mesh.xyz[1, :] .+= 1.0e308
patch_translation_result = assign_patches_uniform(
    patch_translation_mesh; n_patches=2)
@test patch_translation_result.P == patch_translation_reference.P == 2
@test all(
    (patch_translation_result.tri_patch[i] ==
     patch_translation_result.tri_patch[j]) ==
    (patch_translation_reference.tri_patch[i] ==
     patch_translation_reference.tri_patch[j])
    for i in eachindex(patch_translation_result.tri_patch),
        j in eachindex(patch_translation_result.tri_patch))
patch_distance_point = Vec3(floatmax(Float64), 0.0, 0.0)
patch_distance_candidate = Vec3(-floatmax(Float64), 0.0, 0.0)
patch_distance_incumbent = Vec3(
    -floatmax(Float64), nextfloat(0.0), 0.0)
@test DiffMoM._patch_candidate_is_nearer(
    patch_distance_point,
    patch_distance_candidate,
    patch_distance_incumbent,
    DiffMoM._patch_distance(patch_distance_point, patch_distance_candidate),
    DiffMoM._patch_distance(patch_distance_point, patch_distance_incumbent),
)
patch_tie_point = Vec3(0.0, 0.0, 0.0)
patch_tie_candidate = Vec3(1.0, 0.0, 0.0)
patch_tie_incumbent = Vec3(1.0, eps(Float64), 0.0)
@test DiffMoM._patch_candidate_is_nearer(
    patch_tie_point,
    patch_tie_candidate,
    patch_tie_incumbent,
    DiffMoM._patch_distance(patch_tie_point, patch_tie_candidate),
    DiffMoM._patch_distance(patch_tie_point, patch_tie_incumbent),
)
patch_rounded_candidate = Vec3(
    5.722957319730639e-187,
    -8.573417650413093e-188,
    1.475425973239854e-187,
)
patch_rounded_incumbent = Vec3(
    5.72295731973064e-187,
    -8.573417650413092e-188,
    1.4754259732398537e-187,
)
@test DiffMoM._patch_candidate_is_nearer(
    patch_tie_point,
    patch_rounded_candidate,
    patch_rounded_incumbent,
    DiffMoM._patch_distance(patch_tie_point, patch_rounded_candidate),
    DiffMoM._patch_distance(patch_tie_point, patch_rounded_incumbent),
)

# Overflowing centroid spans are assigned with exponent scaling, without a
# BigFloat allocation per ordinary interior centroid.
patch_wide_count = 2_000
patch_wide_xyz = zeros(Float64, 3, 3 * patch_wide_count)
patch_wide_tri = Matrix{Int}(undef, 3, patch_wide_count)
for triangle in 1:patch_wide_count
    centroid_x = triangle == 1 ? -floatmax(Float64) :
                 triangle == 2 ? floatmax(Float64) : 0.0
    first_vertex = 3 * triangle - 2
    patch_wide_xyz[:, first_vertex] .= (centroid_x, 0.0, 0.0)
    patch_wide_xyz[:, first_vertex + 1] .= (centroid_x, 1.0, 0.0)
    patch_wide_xyz[:, first_vertex + 2] .= (centroid_x, 0.0, 1.0)
    patch_wide_tri[:, triangle] .=
        (first_vertex, first_vertex + 1, first_vertex + 2)
end
patch_wide_mesh = TriMesh(patch_wide_xyz, patch_wide_tri)
patch_wide_partition = assign_patches_grid(
    patch_wide_mesh; nx=4, ny=1, nz=1)
@test patch_wide_partition.tri_patch[1] == 1
@test patch_wide_partition.tri_patch[2] == patch_wide_partition.P
assign_patches_grid(patch_wide_mesh; nx=4, ny=1, nz=1)
GC.gc()
@test @allocated(assign_patches_grid(
    patch_wide_mesh; nx=4, ny=1, nz=1)) < 2_000_000
@test DiffMoM._patch_grid_axis_index(
    prevfloat(floatmax(Float64) / 2),
    -floatmax(Float64), floatmax(Float64), 4) == 2
@test DiffMoM._patch_grid_axis_index(
    -nextfloat(0.0), -1.0, 1.0, 4) == 1

# Distinct centers may be exactly equidistant without requiring a
# high-precision comparison for every symmetric triangle.
patch_symmetric_count = 2_000
patch_symmetric_xyz = zeros(Float64, 3, 3 * patch_symmetric_count)
patch_symmetric_tri = Matrix{Int}(undef, 3, patch_symmetric_count)
patch_symmetric_indices = randperm(
    MersenneTwister(42), patch_symmetric_count)[1:2]
for triangle in 1:patch_symmetric_count
    xcenter = triangle == patch_symmetric_indices[1] ? -1.0 :
              triangle == patch_symmetric_indices[2] ? 1.0 : 0.0
    first_vertex = 3 * triangle - 2
    patch_symmetric_xyz[:, first_vertex] .= (xcenter - 0.25, 0.0, 0.0)
    patch_symmetric_xyz[:, first_vertex + 1] .= (xcenter + 0.25, 0.0, 0.0)
    patch_symmetric_xyz[:, first_vertex + 2] .= (xcenter, 1.0, 0.0)
    patch_symmetric_tri[:, triangle] .=
        (first_vertex, first_vertex + 1, first_vertex + 2)
end
patch_symmetric_mesh = TriMesh(
    patch_symmetric_xyz, patch_symmetric_tri)
assign_patches_uniform(patch_symmetric_mesh; n_patches=2)
GC.gc()
@test @allocated(assign_patches_uniform(
    patch_symmetric_mesh; n_patches=2)) < 1_000_000

# The K==Nt endpoint groups exact duplicate centroids in linear work rather
# than scanning every triangle against every center.
patch_endpoint = assign_patches_uniform(
    patch_translation_base;
    n_patches=ntriangles(patch_translation_base))
@test patch_endpoint.P == ntriangles(patch_translation_base)
patch_endpoint_large = make_rect_plate(1.0, 1.0, 1, 800)
assign_patches_uniform(
    patch_endpoint_large; n_patches=ntriangles(patch_endpoint_large))
GC.gc()
patch_endpoint_alloc = @allocated assign_patches_uniform(
    patch_endpoint_large; n_patches=ntriangles(patch_endpoint_large))
@test patch_endpoint_alloc < 2_000_000

# Duplicate centers and exact equidistance must not trigger a BigFloat
# allocation per distance comparison.
duplicate_lloyd_count = 2_000
duplicate_lloyd_mesh = TriMesh(
    Float64[0 1 0; 0 0 1; 0 0 0],
    repeat(reshape(Int[1, 2, 3], 3, 1), 1, duplicate_lloyd_count),
)
assign_patches_uniform(duplicate_lloyd_mesh; n_patches=2)
GC.gc()
@test @allocated(assign_patches_uniform(
    duplicate_lloyd_mesh; n_patches=2)) < 500_000
patch_single_mesh = make_rect_plate(1.0, 1.0, 20, 20)
assign_patches_uniform(patch_single_mesh; n_patches=1)
GC.gc()
@test @allocated(assign_patches_uniform(
    patch_single_mesh; n_patches=1)) < 500_000
@test assign_patches_uniform(
    patch_translation_base;
    n_patches=1,
    max_distance_evaluations=1).tri_patch ==
      fill(1, ntriangles(patch_translation_base))

# A work cap is checked before centroid/permutation allocation and never
# returns a silently unconverged Lloyd state.
for invalid_work_limit in (false, 0, -1, big(typemax(Int)) + 1)
    @test_throws ArgumentError assign_patches_uniform(
        patch_translation_base;
        n_patches=2,
        max_distance_evaluations=invalid_work_limit)
end
@test_throws ArgumentError assign_patches_uniform(
    patch_translation_base;
    n_patches=2,
    max_distance_evaluations=1)

patch_reference_x = [0.0, -3.0, -8.0, 2.0, 2.0, -10.0, -10.0, 7.0]
patch_reference_xyz = Matrix{Float64}(undef, 3, 3 * length(patch_reference_x))
patch_reference_tri = Matrix{Int}(undef, 3, length(patch_reference_x))
for (triangle, xcenter) in enumerate(patch_reference_x)
    first_vertex = 3 * triangle - 2
    patch_reference_xyz[:, first_vertex] .= (xcenter - 0.25, 0.0, 0.0)
    patch_reference_xyz[:, first_vertex + 1] .= (xcenter + 0.25, 0.0, 0.0)
    patch_reference_xyz[:, first_vertex + 2] .= (xcenter, 1.0, 0.0)
    patch_reference_tri[:, triangle] .=
        (first_vertex, first_vertex + 1, first_vertex + 2)
end
patch_reference_mesh = TriMesh(patch_reference_xyz, patch_reference_tri)
@test assign_patches_uniform(
    patch_reference_mesh; n_patches=2).tri_patch ==
      [2, 2, 1, 2, 2, 1, 1, 2]
@test_throws ArgumentError assign_patches_uniform(
    patch_reference_mesh;
    n_patches=2,
    max_distance_evaluations=16)

# Cluster sums and member counts are reused across k-means iterations. Keep a
# regression ratchet against rebuilding one temporary member vector per cluster.
patch_alloc_mesh = make_rect_plate(1.0, 1.0, 20, 20)
assign_patches_uniform(patch_alloc_mesh; n_patches=20)  # warm compilation
GC.gc()
patch_uniform_alloc = @allocated assign_patches_uniform(
    patch_alloc_mesh; n_patches=20)
@assert patch_uniform_alloc < 500_000 "Uniform patch assignment allocated $patch_uniform_alloc bytes"
println("  33d: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 34: CompositeOperator (ImpedanceLoadedOperator)
# ─────────────────────────────────────────────────
println("\n── Test 34: CompositeOperator ──")

# 34a: Forward matvec accuracy
println("  34a: Forward matvec accuracy ...")
theta_test = randn(MersenneTwister(123), Nt) .* 100.0
x_test = randn(MersenneTwister(456), ComplexF64, N)

# Reference: dense assemble_full_Z
Z_ref = assemble_full_Z(Z_efie, Mp, theta_test; reactive=false)
y_ref = Z_ref * x_test

# Composite operator
A_comp = ImpedanceLoadedOperator(Z_efie, Mp, theta_test, false)
@test_throws ArgumentError ImpedanceLoadedOperator(
    Z_efie, Mp, fill(Inf, length(theta_test)), false)
@test_throws DimensionMismatch ImpedanceLoadedOperator(
    Z_efie, Mp, theta_test[1:(end - 1)], false)
@test_throws DimensionMismatch ImpedanceLoadedOperator(
    Z_efie, [undersized_mass], [1.0], false)
A_comp_resized = ImpedanceLoadedOperator(
    Z_efie, Mp, copy(theta_test), false)
pop!(A_comp_resized.theta)
@test_throws DimensionMismatch A_comp_resized * x_test
@test_throws DimensionMismatch adjoint(A_comp_resized) * x_test
A_comp_nonfinite = ImpedanceLoadedOperator(
    Z_efie, Mp, copy(theta_test), false)
A_comp_nonfinite.theta[1] = NaN
@test_throws ArgumentError A_comp_nonfinite * x_test
@test_throws ArgumentError adjoint(A_comp_nonfinite) * x_test
y_comp = A_comp * x_test
@test A_comp[1, 1] ≈ Z_ref[1, 1] rtol=1e-12
@test adjoint(A_comp)[1, 1] ≈ Z_ref'[1, 1] rtol=1e-12
@test _matrix_entry_allocation(A_comp, 1, 1) <= 128
_assert_single_complex_output_allocation(A_comp, x_test)
_assert_scaled_mul_contract(A_comp, x_test, reverse(x_test))

composite_alpha = 1.2 - 0.1im
composite_beta = -0.4 + 0.2im
composite_overlap_storage = vcat(x_test, 5.0 + 4.0im)
composite_overlap_x = view(composite_overlap_storage, 1:N)
composite_overlap_y = view(composite_overlap_storage, 2:(N + 1))
composite_overlap_x_initial = copy(composite_overlap_x)
composite_overlap_y_initial = copy(composite_overlap_y)
composite_overlap_expected =
    composite_alpha .* (A_comp * composite_overlap_x_initial) .+
    composite_beta .* composite_overlap_y_initial
mul!(composite_overlap_y, A_comp, composite_overlap_x,
     composite_alpha, composite_beta)
@test composite_overlap_y ≈ composite_overlap_expected rtol=1e-12

composite_identity = Matrix{ComplexF64}(I, 1, 1)
composite_scale_operator = ImpedanceLoadedOperator(
    composite_identity, [composite_identity], [-1.0], false)
composite_scale_input = ComplexF64[10]
composite_scale_previous = ComplexF64[-20]
composite_scale_result = copy(composite_scale_previous)
mul!(composite_scale_result, composite_scale_operator,
     composite_scale_input, 1.0e308, 1.0e308)
@test composite_scale_result == zeros(ComplexF64, 1)

composite_scale_adjoint_result = copy(composite_scale_previous)
mul!(composite_scale_adjoint_result, adjoint(composite_scale_operator),
     composite_scale_input, 1.0e308, 1.0e308)
@test composite_scale_adjoint_result == zeros(ComplexF64, 1)

composite_scale_alias = copy(composite_scale_input)
mul!(composite_scale_alias, composite_scale_operator,
     composite_scale_alias,
     floatmax(Float64) / 2, -floatmax(Float64))
@test composite_scale_alias == zeros(ComplexF64, 1)

composite_finite_base = reshape(ComplexF64[1.0e30], 1, 1)
composite_finite_mass = reshape(
    ComplexF64[prevfloat(1.0e30)], 1, 1)
composite_finite_operator = ImpedanceLoadedOperator(
    composite_finite_base, [composite_finite_mass], [1.0], false)
composite_finite_reference = setprecision(BigFloat, 8704) do
    ComplexF64(BigFloat(1.0e30) - BigFloat(prevfloat(1.0e30)))
end
@test composite_finite_operator[1, 1] == composite_finite_reference
@test composite_finite_operator * ComplexF64[1.0] ==
      ComplexF64[composite_finite_reference]
@test adjoint(composite_finite_operator) * ComplexF64[1.0] ==
      ComplexF64[composite_finite_reference]

composite_max_matrix = fill(
    ComplexF64(floatmax(Float64), 0.0), 1, 1)
composite_cancel_operator = ImpedanceLoadedOperator(
    composite_max_matrix, [composite_max_matrix], [1.0], false)
composite_cancel_input = ComplexF64[2]
@test composite_cancel_operator * composite_cancel_input ==
      zeros(ComplexF64, 1)
@test adjoint(composite_cancel_operator) * composite_cancel_input ==
      zeros(ComplexF64, 1)
@test composite_cancel_operator[1, 1] == 0

composite_overflow_operator = ImpedanceLoadedOperator(
    composite_max_matrix, [composite_identity], [0.0], false)
@test_throws OverflowError composite_overflow_operator *
                           composite_cancel_input

composite_reactive_scale_operator = ImpedanceLoadedOperator(
    composite_identity, [composite_identity], [1.0], true)
composite_reactive_result = ComplexF64[-10 + 10im]
mul!(composite_reactive_result, composite_reactive_scale_operator,
     composite_scale_input, 1.0e308, 1.0e308)
@test composite_reactive_result == zeros(ComplexF64, 1)
composite_reactive_adjoint_result = ComplexF64[-10 - 10im]
mul!(composite_reactive_adjoint_result,
     adjoint(composite_reactive_scale_operator),
     composite_scale_input, 1.0e308, 1.0e308)
@test composite_reactive_adjoint_result == zeros(ComplexF64, 1)

matvec_err = norm(y_comp - y_ref) / norm(y_ref)
println("    Forward matvec relative error: $matvec_err")
@assert matvec_err < 1e-12 "Forward matvec error too large: $matvec_err"
@assert size(A_comp) == (N, N) "Size mismatch"
println("  34a: PASS")

# 34b: Adjoint matvec accuracy
println("  34b: Adjoint matvec accuracy ...")
y_adj_ref = Z_ref' * x_test
y_adj_comp = adjoint(A_comp) * x_test
_assert_single_complex_output_allocation(adjoint(A_comp), x_test)
_assert_scaled_mul_contract(adjoint(A_comp), x_test, reverse(x_test))

composite_adjoint_overlap_storage = vcat(x_test, -2.0 + 3.0im)
composite_adjoint_overlap_x = view(composite_adjoint_overlap_storage, 1:N)
composite_adjoint_overlap_y = view(composite_adjoint_overlap_storage, 2:(N + 1))
composite_adjoint_overlap_x_initial = copy(composite_adjoint_overlap_x)
composite_adjoint_overlap_y_initial = copy(composite_adjoint_overlap_y)
composite_adjoint_overlap_expected =
    composite_alpha .* (adjoint(A_comp) * composite_adjoint_overlap_x_initial) .+
    composite_beta .* composite_adjoint_overlap_y_initial
mul!(composite_adjoint_overlap_y, adjoint(A_comp), composite_adjoint_overlap_x,
     composite_alpha, composite_beta)
@test composite_adjoint_overlap_y ≈ composite_adjoint_overlap_expected rtol=1e-12

adj_err = norm(y_adj_comp - y_adj_ref) / norm(y_adj_ref)
println("    Adjoint matvec relative error: $adj_err")
@assert adj_err < 1e-12 "Adjoint matvec error too large: $adj_err"
println("  34b: PASS")

# 34c: Reactive mode
println("  34c: Reactive mode ...")
Z_ref_rx = assemble_full_Z(Z_efie, Mp, theta_test; reactive=true)
A_comp_rx = ImpedanceLoadedOperator(Z_efie, Mp, theta_test, true)
y_rx_ref = Z_ref_rx * x_test
y_rx_comp = A_comp_rx * x_test
rx_err = norm(y_rx_comp - y_rx_ref) / norm(y_rx_ref)
@assert rx_err < 1e-12 "Reactive forward matvec error: $rx_err"
println("  34c: PASS")

# 34d: GMRES convergence with composite operator
println("  34d: GMRES solve via composite operator ...")
theta_small = fill(300.0, Nt)
A_gmres = ImpedanceLoadedOperator(Z_efie, Mp, theta_small, false)
I_gmres = solve_forward(A_gmres, v; solver=:gmres, gmres_tol=1e-8, gmres_maxiter=200)

Z_dense = assemble_full_Z(Z_efie, Mp, theta_small)
I_dense = Z_dense \ v
gmres_diff = norm(I_gmres - I_dense) / norm(I_dense)
println("    GMRES vs direct relative diff: $gmres_diff")
@assert gmres_diff < 1e-4 "GMRES solution too different from direct: $gmres_diff"
println("  34d: PASS")

# 34e: solve_adjoint_rhs
println("  34e: solve_adjoint_rhs ...")
rhs_adj = Q * I_dense
rhs_adj_before = copy(rhs_adj)
lam_direct = Z_dense' \ rhs_adj
lam_rhs = solve_adjoint_rhs(Z_dense, rhs_adj; solver=:direct)
@assert norm(lam_rhs - lam_direct) / norm(lam_direct) < 1e-12 "solve_adjoint_rhs direct mismatch"
@test rhs_adj == rhs_adj_before
lam_gmres = solve_adjoint_rhs(A_gmres, rhs_adj; solver=:gmres, gmres_tol=1e-8, gmres_maxiter=200)
@test rhs_adj == rhs_adj_before
adj_rhs_diff = norm(lam_gmres - lam_direct) / norm(lam_direct)
println("    GMRES adjoint_rhs vs direct: $adj_rhs_diff")
@assert adj_rhs_diff < 1e-4 "solve_adjoint_rhs GMRES too different: $adj_rhs_diff"
println("  34e: PASS")

_assert_shared_workspace_concurrency(
    fill(A_comp, 4),
    [x_test, reverse(x_test), conj.(x_test), (0.2 - 0.3im) .* x_test],
)
_assert_shared_workspace_concurrency(
    fill(adjoint(A_comp), 4),
    [x_test, reverse(x_test), conj.(x_test), (0.2 - 0.3im) .* x_test],
)
_assert_shared_workspace_concurrency(
    fill(composite_cancel_operator, 4),
    [
        composite_cancel_input,
        -composite_cancel_input,
        1im .* composite_cancel_input,
        (0.5 - 0.25im) .* composite_cancel_input,
    ],
)

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 35: Multi-angle RCS optimization (smoke test)
# ─────────────────────────────────────────────────
println("\n── Test 35: Multi-angle RCS optimization ──")

# 35a: direction_mask
println("  35a: direction_mask ...")
grid_test = make_sph_grid(18, 36)
mask_z = direction_mask(grid_test, Vec3(0.0, 0.0, 1.0); half_angle=10.0 * π / 180)
mask_mz = direction_mask(grid_test, Vec3(0.0, 0.0, -1.0); half_angle=10.0 * π / 180)
@test_throws ArgumentError cap_mask(grid_test; theta_max=-0.1)
@test_throws ArgumentError cap_mask(grid_test; theta_max=π + 0.1)
@test_throws ArgumentError direction_mask(
    grid_test, Vec3(0.0, 0.0, 0.0); half_angle=0.1)
@test_throws ArgumentError direction_mask(
    grid_test, Vec3(0.0, 0.0, 1.0); half_angle=-0.1)
@test_throws ArgumentError direction_mask(
    grid_test, Vec3(0.0, 0.0, 1.0); half_angle=π + 0.1)
direction_mask(grid_test, Vec3(0.0, 0.0, 1.0); half_angle=0.1)
@test @allocated(direction_mask(
    grid_test, Vec3(0.0, 0.0, 1.0); half_angle=0.1)) <=
    _bit_vector_output_allocation(length(grid_test.w)) + 128
@assert sum(mask_z) > 0 "Should have points near +z"
@assert sum(mask_mz) > 0 "Should have points near -z"
# +z and -z masks should not overlap (for small cone)
@assert sum(mask_z .& mask_mz) == 0 "+z and -z 10° cones should not overlap"
println("    +z mask: $(sum(mask_z)) pts,  -z mask: $(sum(mask_mz)) pts")
println("  35a: PASS")

# 35b: build_multiangle_configs
println("  35b: build_multiangle_configs ...")
angles_2 = [
    (theta_inc=0.0, phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0),     # +z incidence
    (theta_inc=π/4, phi_inc=0.0, pol=Vec3(0.0, 1.0, 0.0), weight=1.0),     # 45° incidence
]
grid_opt = make_sph_grid(12, 24)
configs_test = build_multiangle_configs(mesh, rwg, k, angles_2;
                                         grid=grid_opt, backscatter_cone=15.0)
@assert length(configs_test) == 2 "Should have 2 angle configs"
@assert length(configs_test[1].v) == N "Excitation vector should have length N"
@assert size(configs_test[1].Q) == (N, N) "Q matrix should be N×N"
@assert configs_test[1].weight == 1.0
configs_mfree = build_multiangle_configs(mesh, rwg, k, angles_2;
                                          grid=grid_opt, backscatter_cone=15.0,
                                          matrix_free_Q=true)
@assert configs_mfree[1].Q isa FarFieldQMatrix "matrix_free_Q should use FarFieldQMatrix"
x_q = randn(MersenneTwister(355), ComplexF64, N)
Qx_dense = configs_test[1].Q * x_q
Qx_mfree = configs_mfree[1].Q * x_q
q_rel = norm(Qx_dense - Qx_mfree) / max(norm(Qx_dense), 1e-30)
@assert q_rel < 1e-12 "Matrix-free Q action mismatch: $q_rel"
configs_total = build_multiangle_configs(mesh, rwg, k, angles_2;
                                          grid=grid_opt, backscatter_cone=15.0,
                                          rcs_component=:total)
configs_total_mfree = build_multiangle_configs(mesh, rwg, k, angles_2;
                                                grid=grid_opt, backscatter_cone=15.0,
                                                matrix_free_Q=true,
                                                rcs_component=:total)
@assert configs_total_mfree[1].Q isa SumQMatrix "total matrix_free_Q should use SumQMatrix"
configs_crosspol_mfree = build_multiangle_configs(
    mesh, rwg, k, angles_2;
    grid=grid_opt, backscatter_cone=15.0,
    matrix_free_Q=true, rcs_component=:crosspol)
@test configs_crosspol_mfree[1].Q isa FarFieldQMatrix
Qx_total_dense = configs_total[1].Q * x_q
Qx_total_mfree = configs_total_mfree[1].Q * x_q
q_total_rel = norm(Qx_total_dense - Qx_total_mfree) / max(norm(Qx_total_dense), 1e-30)
@assert q_total_rel < 1e-12 "Matrix-free total-Q action mismatch: $q_total_rel"
@assert _assert_zero_allocation_mul!(configs_total_mfree[1].Q, x_q) ≈
        Qx_total_dense
_assert_single_complex_output_allocation(configs_total_mfree[1].Q, x_q)

q_sum_initial = randn(MersenneTwister(356), ComplexF64, N)
q_sum_scaled = copy(q_sum_initial)
mul!(q_sum_scaled, configs_total_mfree[1].Q, x_q, -1.25, 0.75)
@assert q_sum_scaled ≈ -1.25 .* Qx_total_dense .+ 0.75 .* q_sum_initial
configs_proj = build_multiangle_configs(
    mesh, rwg, k,
    [(theta_inc=π/4, phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0)];
    grid=grid_opt,
    backscatter_cone=15.0,
)
khat_proj = Vec3(sin(π/4), 0.0, cos(π/4))
@assert abs(dot(khat_proj, configs_proj[1].pol)) < 1e-12 "Multi-angle polarization must be transverse"
@assert abs(norm(configs_proj[1].pol) - 1.0) < 1e-12 "Multi-angle polarization must be unit length"
@test DiffMoM._transverse_unit_pol(
    Vec3(0.0, 0.0, 1.0),
    Vec3(nextfloat(0.0), 0.0, 0.0)) == Vec3(1.0, 0.0, 0.0)
multiangle_diagonal_direction =
    Vec3(inv(sqrt(2.0)), inv(sqrt(2.0)), 0.0)
multiangle_large_pol = Vec3(
    floatmax(Float64), floatmax(Float64), floatmax(Float64))
@test DiffMoM._transverse_unit_pol(
    multiangle_diagonal_direction, multiangle_large_pol) ≈
      Vec3(0.0, 0.0, 1.0) atol=2eps(Float64)
configs_default_pol = build_multiangle_configs(
    mesh, rwg, k,
    [(theta_inc=π/4, phi_inc=0.2, weight=1.0)];
    grid=grid_opt,
    backscatter_cone=15.0,
)
khat_default = Vec3(sin(π/4) * cos(0.2), sin(π/4) * sin(0.2), cos(π/4))
@assert abs(dot(khat_default, configs_default_pol[1].pol)) < 1e-12 "Default multi-angle polarization must be transverse"
@test_throws ArgumentError build_multiangle_configs(
    mesh, rwg, k, NamedTuple[];
    grid=grid_opt,
)
@test_throws ArgumentError build_multiangle_configs(
    mesh, rwg, k,
    [(theta_inc=Inf, phi_inc=0.0, weight=1.0)];
    grid=grid_opt,
)
@test_throws ArgumentError build_multiangle_configs(
    mesh, rwg, k,
    [(theta_inc=0.0, phi_inc=0.0, weight=Inf)];
    grid=grid_opt,
)
@test_throws ArgumentError build_multiangle_configs(
    mesh, rwg, k, angles_2;
    grid=grid_opt,
    backscatter_cone=181.0,
)

# Multi-angle plane waves share one mapped-triangle quadrature cache, and a
# single-component objective retains only its selected polarization matrix.
# The warm call excludes compilation from this allocation regression.
multiangle_allocation_mesh = make_rect_plate(0.02, 0.02, 8, 8)
multiangle_allocation_rwg = build_rwg(multiangle_allocation_mesh)
multiangle_allocation_grid = make_sph_grid(4, 8)
multiangle_allocation_angles = fill(
    (theta_inc=0.0, phi_inc=0.0,
     pol=Vec3(1.0, 0.0, 0.0), weight=1.0),
    32)
build_multiangle_configs(
    multiangle_allocation_mesh, multiangle_allocation_rwg, 2π,
    multiangle_allocation_angles;
    grid=multiangle_allocation_grid, matrix_free_Q=true)
multiangle_config_allocation = @allocated build_multiangle_configs(
    multiangle_allocation_mesh, multiangle_allocation_rwg, 2π,
    multiangle_allocation_angles;
    grid=multiangle_allocation_grid, matrix_free_Q=true)
@test multiangle_config_allocation < 750_000

# Stagnation is relative to the objective scale.  The former absolute 1e-30
# denominator floor stopped this problem at iteration 11 despite a 68% drop in
# a finite subnormal-scale objective.
@test !DiffMoM._multiangle_objective_stagnated(2e-300, 1e-300)
@test DiffMoM._multiangle_objective_stagnated(
    nextfloat(1e-300), 1e-300)
multiangle_stagnation_Z = ComplexF64[1;;]
multiangle_stagnation_Mp = Matrix{ComplexF64}[ComplexF64[1;;]]
multiangle_stagnation_config = AngleConfig(
    Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
    ComplexF64[1e-150], ComplexF64[1;;], 1.0)
_, multiangle_stagnation_trace = optimize_multiangle_rcs(
    multiangle_stagnation_Z, multiangle_stagnation_Mp,
    [multiangle_stagnation_config], [0.0];
    maxiter=20, tol=0.0, alpha0=1e299, m_lbfgs=0,
    verbose=false, fallback_to_steepest=false,
    lbfgs_line_search_maxiter=40)
@test length(multiangle_stagnation_trace) == 20
@test last(multiangle_stagnation_trace).J <
      0.5first(multiangle_stagnation_trace).J
println("  35b: PASS")

# 35c: optimize_multiangle_rcs smoke test
println("  35c: optimize_multiangle_rcs (5 iterations) ...")
# Use spatial grid patches instead of one-per-triangle (fewer parameters = faster)
part_opt = assign_patches_grid(mesh; nx=3, ny=3, nz=1)
Mp_opt = precompute_patch_mass(mesh, rwg, part_opt; quad_order=3)
theta_init = fill(200.0, part_opt.P)

@test optimize_multiangle_rcs(
    Z_opt_guard, Mp_opt_guard,
    [AngleConfig(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                 v_opt_guard, Q_opt_guard, 1.0)],
    theta_opt_guard;
    maxiter=0, verbose=false,
    max_workspace_bytes=sizeof(ComplexF64))[1] == theta_opt_guard
@test_throws ArgumentError optimize_multiangle_rcs(
    Z_opt_guard, Mp_opt_guard,
    [AngleConfig(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
                 v_opt_guard, Q_opt_guard, 1.0)],
    theta_opt_guard;
    maxiter=0, verbose=false,
    max_workspace_bytes=sizeof(ComplexF64) - 1)

theta_opt_35, trace_35 = optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=5, tol=1e-12, alpha0=0.01,
    reactive=false, verbose=false,
    lb=fill(0.0, part_opt.P), ub=fill(1000.0, part_opt.P)
)
@assert length(trace_35) == 5 "Should run exactly 5 iterations"
@assert all(t -> isfinite(t.J), trace_35) "All objectives should be finite"
@assert all(t -> isfinite(t.gnorm), trace_35) "All gradients should be finite"
J_trace_35 = [t.J for t in trace_35]
@assert all(J_trace_35[2:end] .<= J_trace_35[1:end-1] .+ 1e-12) "line search should not accept uphill objective steps"
# Objective should generally decrease (check last < first with tolerance)
println("    J: $(round(trace_35[1].J, sigdigits=4)) → $(round(trace_35[end].J, sigdigits=4))")
println("    |g|: $(round(trace_35[1].gnorm, sigdigits=4)) → $(round(trace_35[end].gnorm, sigdigits=4))")

# Dense Q products at both accepted iterates and exploratory line-search points
# must restart exact accumulation when BLAS loses an otherwise finite sum.
multiangle_extreme_scale = 0.8 * floatmax(Float64)
multiangle_extreme_Q = zeros(ComplexF64, 4, 4)
multiangle_extreme_Q[1, :] .= ComplexF64[1.0, 1.0, -1.0, -1.0]
multiangle_extreme_v = fill(ComplexF64(multiangle_extreme_scale), 4)
@test !isfinite((multiangle_extreme_Q * multiangle_extreme_v)[1])
multiangle_extreme_reference = setprecision(BigFloat, 4096) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(multiangle_extreme_Q) *
        Complex{BigFloat}.(multiangle_extreme_v))
end
@test multiangle_extreme_reference == zeros(ComplexF64, 4)
multiangle_extreme_config = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    multiangle_extreme_v,
    multiangle_extreme_Q,
    1.0,
)
multiangle_extreme_theta, multiangle_extreme_trace =
    optimize_multiangle_rcs(
        Matrix{ComplexF64}(I, 4, 4),
        [zeros(ComplexF64, 4, 4)],
        [multiangle_extreme_config],
        [0.0];
        maxiter=1,
        verbose=false,
    )
@test multiangle_extreme_theta == [0.0]
@test length(multiangle_extreme_trace) == 1
@test multiangle_extreme_trace[1].J == 0.0
@test multiangle_extreme_trace[1].gnorm == 0.0

multiangle_line_Q = zeros(ComplexF64, 5, 5)
multiangle_line_coefficient = 0.4 * floatmax(Float64)
multiangle_line_Q[1, 1:4] .= ComplexF64[
    multiangle_line_coefficient,
    multiangle_line_coefficient,
    -multiangle_line_coefficient,
    -multiangle_line_coefficient,
]
multiangle_line_Q[5, 5] = -0.5
multiangle_line_trial_current = fill(ComplexF64(2.0), 5)
@test !isfinite((multiangle_line_Q * multiangle_line_trial_current)[1])
multiangle_line_reference = setprecision(BigFloat, 8192) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(multiangle_line_Q) *
        Complex{BigFloat}.(multiangle_line_trial_current))
end
@test multiangle_line_reference == ComplexF64[0.0, 0.0, 0.0, 0.0, -1.0]
@test real(dot(
    multiangle_line_trial_current,
    multiangle_line_reference,
)) == -2.0
multiangle_line_config = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    ones(ComplexF64, 5),
    multiangle_line_Q,
    1.0,
)
multiangle_line_theta, multiangle_line_trace = optimize_multiangle_rcs(
    Matrix{ComplexF64}(I, 5, 5),
    [Matrix{ComplexF64}(I, 5, 5)],
    [multiangle_line_config],
    [0.0];
    maxiter=1,
    tol=0.0,
    m_lbfgs=0,
    alpha0=0.5,
    verbose=false,
    fallback_to_steepest=false,
)
@test multiangle_line_theta == [0.5]
@test length(multiangle_line_trace) == 1

multiangle_tiny_matrix = reshape(ComplexF64[nextfloat(0.0)], 1, 1)
multiangle_tiny_rhs = ComplexF64[floatmin(Float64)]
multiangle_tiny_reference = setprecision(BigFloat, 4096) do
    ComplexF64.(
        Matrix{Complex{BigFloat}}(multiangle_tiny_matrix) \
        Complex{BigFloat}.(multiangle_tiny_rhs))
end
@test multiangle_tiny_reference == ComplexF64[2.0^52]
multiangle_tiny_config = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    multiangle_tiny_rhs,
    zeros(ComplexF64, 1, 1),
    1.0,
)
multiangle_tiny_theta, multiangle_tiny_trace = optimize_multiangle_rcs(
    multiangle_tiny_matrix,
    [zeros(ComplexF64, 1, 1)],
    [multiangle_tiny_config],
    [0.0];
    maxiter=1,
    verbose=false,
)
@test multiangle_tiny_theta == [0.0]
@test multiangle_tiny_trace[1].J == 0.0

multiangle_false_singular_matrix = ComplexF64[
    1e200  1e200
    1e-200 0.0
]
multiangle_false_singular_rhs = ComplexF64[1e200, 0.0]
multiangle_false_singular_config = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    multiangle_false_singular_rhs,
    Matrix{ComplexF64}(I, 2, 2),
    1.0,
)
multiangle_false_singular_theta, multiangle_false_singular_trace =
    optimize_multiangle_rcs(
        multiangle_false_singular_matrix,
        [zeros(ComplexF64, 2, 2)],
        [multiangle_false_singular_config],
        [0.0];
        maxiter=1,
        verbose=false,
    )
@test multiangle_false_singular_theta == [0.0]
@test multiangle_false_singular_trace == [(
    iter=1,
    J=1.0,
    gnorm=0.0,
    n_fwd=1,
    n_adj=1,
)]

multiangle_adjoint_matrix = ComplexF64[1.0 -2.0; -2.0 1.0]
multiangle_adjoint_current = ComplexF64[1.0, -1.0]
multiangle_adjoint_qscale = 0.4 * floatmax(Float64)
multiangle_adjoint_Q = ComplexF64[
    multiangle_adjoint_qscale -multiangle_adjoint_qscale;
    multiangle_adjoint_qscale -multiangle_adjoint_qscale
]
multiangle_adjoint_rhs =
    multiangle_adjoint_Q * multiangle_adjoint_current
multiangle_adjoint_reference = setprecision(BigFloat, 4096) do
    ComplexF64.(
        adjoint(Matrix{Complex{BigFloat}}(multiangle_adjoint_matrix)) \
        Complex{BigFloat}.(multiangle_adjoint_rhs))
end
@test multiangle_adjoint_reference == fill(
    ComplexF64(-0.8 * floatmax(Float64)), 2)
multiangle_adjoint_config = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    multiangle_adjoint_matrix * multiangle_adjoint_current,
    multiangle_adjoint_Q,
    1.0,
)
multiangle_adjoint_theta, multiangle_adjoint_trace =
    optimize_multiangle_rcs(
        multiangle_adjoint_matrix,
        [zeros(ComplexF64, 2, 2)],
        [multiangle_adjoint_config],
        [0.0];
        maxiter=1,
        verbose=false,
    )
@test multiangle_adjoint_theta == [0.0]
@test multiangle_adjoint_trace[1].J == 0.0

multiangle_trial_scale = floatmin(Float64)
multiangle_trial_alpha = multiangle_trial_scale - nextfloat(0.0)
multiangle_trial_config = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    ComplexF64[multiangle_trial_scale],
    reshape(ComplexF64[-multiangle_trial_scale / 2], 1, 1),
    1.0,
)
multiangle_trial_theta, multiangle_trial_trace = optimize_multiangle_rcs(
    reshape(ComplexF64[multiangle_trial_scale], 1, 1),
    [reshape(ComplexF64[1.0], 1, 1)],
    [multiangle_trial_config],
    [0.0];
    maxiter=1,
    tol=0.0,
    m_lbfgs=0,
    alpha0=multiangle_trial_alpha,
    verbose=false,
    fallback_to_steepest=false,
)
@test multiangle_trial_theta == [multiangle_trial_alpha]
@test length(multiangle_trial_trace) == 1
println("  35c: PASS")

# 35d: normalized smooth worst-angle objective
println("  35d: smoothmax_log normalized objective ...")
J_probe = [2.0e-3, 7.5e-4]
w_probe = [1.0, 1.3]
ref_probe = [3.0e-3, 2.0e-3]
beta_probe = 6.0
phi_probe, scale_probe = DiffMoM._multiangle_objective_scales(
    J_probe, w_probe, :smoothmax_log, ref_probe, beta_probe)
@assert isfinite(phi_probe) "smoothmax objective should be finite"
@assert all(isfinite, scale_probe) "smoothmax objective scales should be finite"

multiangle_scalar_scale = floatmax(Float64)
multiangle_linear_values = Float64[2.0, -2.0]
multiangle_scalar_weights = fill(multiangle_scalar_scale, 2)
multiangle_linear_reference = setprecision(BigFloat, 4096) do
    total = zero(BigFloat)
    for index in eachindex(
        multiangle_linear_values, multiangle_scalar_weights)
        total += BigFloat(multiangle_scalar_weights[index]) *
                 BigFloat(multiangle_linear_values[index])
    end
    Float64(total)
end
multiangle_linear_value, multiangle_linear_scales =
    DiffMoM._multiangle_objective_scales(
        multiangle_linear_values,
        multiangle_scalar_weights,
        :linear,
        ones(2),
        beta_probe,
    )
@test multiangle_linear_reference == 0.0
@test multiangle_linear_value == multiangle_linear_reference
@test multiangle_linear_scales == multiangle_scalar_weights
@test_throws OverflowError DiffMoM._multiangle_objective_scales(
    Float64[2.0, 2.0],
    multiangle_scalar_weights,
    :linear,
    ones(2),
    beta_probe,
)

multiangle_sum_log_values = Float64[10.0, 10.0]
multiangle_sum_log_references = Float64[
    10.0 / exp(2.0),
    10.0 / exp(-2.0),
]
multiangle_sum_log_reference = setprecision(BigFloat, 4096) do
    total = zero(BigFloat)
    for index in eachindex(
        multiangle_sum_log_values,
        multiangle_scalar_weights,
        multiangle_sum_log_references,
    )
        total += BigFloat(multiangle_scalar_weights[index]) * (
            log(BigFloat(multiangle_sum_log_values[index])) -
            log(BigFloat(multiangle_sum_log_references[index]))
        )
    end
    Float64(total)
end
multiangle_sum_log_value, multiangle_sum_log_scales =
    DiffMoM._multiangle_objective_scales(
        multiangle_sum_log_values,
        multiangle_scalar_weights,
        :sum_log,
        multiangle_sum_log_references,
        beta_probe,
    )
@test multiangle_sum_log_value ≈
      multiangle_sum_log_reference rtol=2eps(Float64)
@test multiangle_sum_log_scales ==
      multiangle_scalar_weights ./ multiangle_sum_log_values

# Positive subnormal objectives remain distinct instead of being clipped to
# an absolute 1e-300 floor. Their derivative ratios are evaluated before
# conversion back to Float64.
multiangle_subnormal_sum_value, multiangle_subnormal_sum_scales =
    DiffMoM._multiangle_objective_scales(
        Float64[1e-310], Float64[1e-310], :sum_log,
        ones(1), beta_probe)
multiangle_subnormal_sum_reference = setprecision(BigFloat, 4352) do
    Float64(BigFloat(1e-310) * log(BigFloat(1e-310)))
end
@test multiangle_subnormal_sum_value ==
      multiangle_subnormal_sum_reference
@test multiangle_subnormal_sum_scales == [1.0]
multiangle_subnormal_smooth_value, multiangle_subnormal_smooth_scales =
    DiffMoM._multiangle_objective_scales(
        Float64[1e-307], ones(1), :smoothmax_log,
        ones(1), beta_probe)
multiangle_subnormal_smooth_reference = setprecision(BigFloat, 4352) do
    Float64(log(BigFloat(1e-307)))
end
@test multiangle_subnormal_smooth_value ==
      multiangle_subnormal_smooth_reference
@test multiangle_subnormal_smooth_scales ≈ [1e307] rtol=2eps(Float64)
@test_throws OverflowError DiffMoM._multiangle_objective_scales(
    Float64[1e-310], ones(1), :smoothmax_log,
    ones(1), beta_probe)
_, multiangle_negative_smooth_scales =
    DiffMoM._multiangle_objective_scales(
        Float64[-1.0], ones(1), :smoothmax_log,
        ones(1), beta_probe)
@test multiangle_negative_smooth_scales == [0.0]

# The smooth maximum tends to the weighted mean as beta approaches zero.
# Evaluate the cancellation-prone normalizer without collapsing the finite
# limiting value.
multiangle_tiny_beta_value, multiangle_tiny_beta_scales =
    DiffMoM._multiangle_objective_scales(
        Float64[1.0, exp(1.0)], Float64[0.5, 0.5],
        :smoothmax_log, ones(2), 1.0e-300)
multiangle_tiny_beta_reference = setprecision(BigFloat, 4352) do
    Float64(log(sum(BigFloat(0.5) *
                    exp(BigFloat(1.0e-300) *
                        log(BigFloat(value)))
                    for value in (1.0, exp(1.0)))) /
            BigFloat(1.0e-300))
end
@test multiangle_tiny_beta_value == multiangle_tiny_beta_reference == 0.5
@test all(isfinite, multiangle_tiny_beta_scales)

# Matrix-free objectives must not scan Q through getindex merely to classify
# the scalar reduction; the operator action already certifies the product.
multiangle_counting_reads = Ref(0)
multiangle_counting_Q = _CountingIdentityQ(64, multiangle_counting_reads)
multiangle_counting_product, multiangle_counting_objective =
    DiffMoM._multiangle_q_product_and_objective(
        multiangle_counting_Q, ones(ComplexF64, 64),
        "counting matrix-free Q")
@test multiangle_counting_product == ones(ComplexF64, 64)
@test multiangle_counting_objective == 64.0
@test multiangle_counting_reads[] == 0

sum_q_counting_reads = Ref(0)
sum_q_counting_child = _CountingIdentityQ(64, sum_q_counting_reads)
sum_q_counting_operator = DiffMoM.sum_q_matrix(
    sum_q_counting_child, sum_q_counting_child)
@test sum_q_counting_operator * ones(ComplexF64, 64) ==
      fill(2.0 + 0.0im, 64)
@test sum_q_counting_reads[] == 0

# Accumulating two individually subnormal angle-gradient contributions must
# preserve their representable sum.
multiangle_gradient_unit = nextfloat(0.0)
multiangle_gradient_config = AngleConfig(
    Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
    ComplexF64[1.0], reshape(ComplexF64[1.0], 1, 1),
    multiangle_gradient_unit,
)
_, multiangle_tiny_gradient_trace = optimize_multiangle_rcs(
    reshape(ComplexF64[1.0], 1, 1),
    [reshape(ComplexF64[0.2], 1, 1)],
    [multiangle_gradient_config, multiangle_gradient_config],
    [0.0]; maxiter=1, tol=0.0, verbose=false)
multiangle_tiny_gradient_reference = setprecision(BigFloat, 4352) do
    Float64(2 * BigFloat(multiangle_gradient_unit) * BigFloat(0.4))
end
@test multiangle_tiny_gradient_trace[1].gnorm ==
      multiangle_tiny_gradient_reference == nextfloat(0.0)

multiangle_scalar_config_positive = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    ComplexF64[1.0],
    reshape(ComplexF64[2.0], 1, 1),
    multiangle_scalar_scale,
)
multiangle_scalar_config_negative = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    ComplexF64[1.0],
    reshape(ComplexF64[-2.0], 1, 1),
    multiangle_scalar_scale,
)
multiangle_scalar_theta, multiangle_scalar_trace = optimize_multiangle_rcs(
    reshape(ComplexF64[1.0], 1, 1),
    [zeros(ComplexF64, 1, 1)],
    [multiangle_scalar_config_positive, multiangle_scalar_config_negative],
    [0.0];
    maxiter=1,
    verbose=false,
)
@test multiangle_scalar_theta == [0.0]
@test multiangle_scalar_trace[1].J == 0.0

multiangle_gradient_config_positive = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    ComplexF64[1.0],
    reshape(ComplexF64[1.0], 1, 1),
    multiangle_scalar_scale,
)
multiangle_gradient_config_negative = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    ComplexF64[1.0],
    reshape(ComplexF64[-1.0], 1, 1),
    multiangle_scalar_scale,
)
multiangle_gradient_reference = setprecision(BigFloat, 4096) do
    BigFloat(multiangle_scalar_scale) * BigFloat(2.0) +
    BigFloat(multiangle_scalar_scale) * BigFloat(-2.0)
end
multiangle_gradient_theta, multiangle_gradient_trace =
    optimize_multiangle_rcs(
        reshape(ComplexF64[1.0], 1, 1),
        [reshape(ComplexF64[1.0], 1, 1)],
        [multiangle_gradient_config_positive,
         multiangle_gradient_config_negative],
        [0.0];
        maxiter=1,
        verbose=false,
    )
@test multiangle_gradient_reference == 0.0
@test multiangle_gradient_theta == [0.0]
@test multiangle_gradient_trace[1].J == 0.0
@test multiangle_gradient_trace[1].gnorm == 0.0

DiffMoM._multiangle_objective_scales(
    J_probe, w_probe, :linear, ref_probe, beta_probe)
@test @allocated(DiffMoM._multiangle_objective_scales(
    J_probe, w_probe, :linear, ref_probe, beta_probe)) <=
      _float_vector_output_allocation(length(J_probe)) + 128
@test_throws ArgumentError DiffMoM._multiangle_objective_scales(
    J_probe, w_probe, :smoothmax_log, ref_probe, Inf)
@test_throws ArgumentError DiffMoM._multiangle_objective_scales(
    J_probe, w_probe, :smoothmax_log, ref_probe, NaN)
@test_throws ArgumentError DiffMoM._multiangle_objective_scales(
    Float64[], Float64[], :linear, Float64[], beta_probe)
phi_sharp_probe, scale_sharp_probe = DiffMoM._multiangle_objective_scales(
    J_probe, w_probe, :smoothmax_log, ref_probe, floatmax(Float64))
@test isfinite(phi_sharp_probe)
@test all(isfinite, scale_sharp_probe)
for j in eachindex(J_probe)
    h = 1e-7 * max(abs(J_probe[j]), 1.0e-12)
    Jp = copy(J_probe); Jp[j] += h
    Jm = copy(J_probe); Jm[j] -= h
    phip, _ = DiffMoM._multiangle_objective_scales(
        Jp, w_probe, :smoothmax_log, ref_probe, beta_probe)
    phim, _ = DiffMoM._multiangle_objective_scales(
        Jm, w_probe, :smoothmax_log, ref_probe, beta_probe)
    fd_scale = (phip - phim) / (2h)
    rel_scale = abs(fd_scale - scale_probe[j]) / max(abs(scale_probe[j]), 1e-30)
    @assert rel_scale < 1e-5 "smoothmax objective scale mismatch at $j: $rel_scale"
end

@test_throws ArgumentError optimize_multiangle_rcs(
    Z_efie, Mp_opt, AngleConfig[], theta_init;
    maxiter=1, verbose=false,
)
@test_throws ArgumentError optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=1, tol=Inf, verbose=false,
)
@test_throws ArgumentError optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=1, tol=NaN, verbose=false,
)
@test_throws ArgumentError optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=1, alpha0=Inf, verbose=false,
)
@test_throws ArgumentError optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, fill(NaN, part_opt.P);
    maxiter=0, verbose=false,
)
@test_throws ArgumentError optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=0, lb=NaN, verbose=false,
)
@test_throws ArgumentError optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=0, lb=2.0, ub=1.0, verbose=false,
)
@test_throws DimensionMismatch optimize_multiangle_rcs(
    ones(ComplexF64, N, N + 1), Mp_opt, configs_test, theta_init;
    maxiter=0, verbose=false,
)
@test_throws DimensionMismatch optimize_multiangle_rcs(
    Z_efie, Matrix{ComplexF64}[], configs_test, theta_init;
    maxiter=0, verbose=false,
)
bad_config_35 = AngleConfig(
    configs_test[1].k_vec,
    configs_test[1].pol,
    ComplexF64[configs_test[1].v; 0.0 + 0im],
    configs_test[1].Q,
    configs_test[1].weight,
)
@test_throws DimensionMismatch optimize_multiangle_rcs(
    Z_efie, Mp_opt, [bad_config_35], theta_init;
    maxiter=0, verbose=false,
)
@test_throws DimensionMismatch optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=0, objective=:sum_log,
    reference_objectives=Float64[], verbose=false,
)
@test_throws ArgumentError optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=0, objective=:smoothmax_log,
    reference_objectives=ones(length(configs_test)),
    smooth_beta=Inf, verbose=false,
)

# Projected Armijo must use the actual feasible step. The first synthetic
# variable has a dominant outward gradient but is pinned at its lower bound;
# the second variable remains free and must still take a decreasing step.
projected_cfg_35 = AngleConfig(
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    ComplexF64[1.0 + 0im],
    ComplexF64[1.0 + 0im;;],
    1.0,
)
projected_theta_35, projected_trace_35 = optimize_multiangle_rcs(
    ComplexF64[5.0 + 0im;;],
    [ComplexF64[1000.0 + 0im;;], ComplexF64[1.0 + 0im;;]],
    [projected_cfg_35],
    [0.0, 0.0];
    maxiter=2,
    tol=0.0,
    alpha0=0.01,
    lb=[0.0, -Inf],
    ub=[Inf, Inf],
    verbose=false,
)
@test length(projected_trace_35) == 2
@test projected_theta_35[1] == 0.0
@test projected_theta_35[2] < 0.0
@test projected_trace_35[2].J < projected_trace_35[1].J

refs_35 = Float64[]
Z_ref_35 = assemble_full_Z(Z_efie, Mp_opt, theta_init)
for cfg in configs_test
    I_ref = Z_ref_35 \ cfg.v
    push!(refs_35, real(dot(I_ref, cfg.Q * I_ref)))
end
theta_bal_35, trace_bal_35 = optimize_multiangle_rcs(
    Z_efie, Mp_opt, configs_test, theta_init;
    maxiter=3, tol=1e-12, alpha0=0.01,
    reactive=false, verbose=false,
    lb=fill(0.0, part_opt.P), ub=fill(1000.0, part_opt.P),
    objective=:smoothmax_log,
    reference_objectives=refs_35,
    smooth_beta=6.0,
)
@assert length(trace_bal_35) == 3 "smoothmax objective should run exactly 3 iterations"
@assert all(t -> isfinite(t.J), trace_bal_35) "smoothmax objectives should be finite"
@assert all(t -> isfinite(t.gnorm), trace_bal_35) "smoothmax gradients should be finite"
J_trace_bal_35 = [t.J for t in trace_bal_35]
@assert all(J_trace_bal_35[2:end] .<= J_trace_bal_35[1:end-1] .+ 1e-12) "smoothmax line search should not accept uphill objective steps"
println("    smoothmax Φ: $(round(trace_bal_35[1].J, sigdigits=4)) → $(round(trace_bal_35[end].J, sigdigits=4))")
println("  35d: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 36: MLFMA + optimizer integration
# ─────────────────────────────────────────────────
println("\n── Test 36: MLFMA + optimizer integration ──")

# Build a larger mesh for MLFMA (icosphere, ~1920 unknowns)
println("  36a: Build icosphere + MLFMA operator ...")
ico_opt_path = joinpath(DATADIR, "tmp_icosphere_opt.obj")
write_icosphere_obj(ico_opt_path; radius=0.05, subdivisions=2)
mesh_ico_opt = read_obj_mesh(ico_opt_path)
rwg_ico = build_rwg(mesh_ico_opt)
N_ico = rwg_ico.nedges
println("    Icosphere: $(nvertices(mesh_ico_opt)) verts, $(ntriangles(mesh_ico_opt)) tris, $N_ico RWG")

freq_ico = 2e9
k_ico = 2π * freq_ico / 299792458.0

# Build MLFMA operator
A_mlfma = build_mlfma_operator(mesh_ico_opt, rwg_ico, k_ico;
                                leaf_lambda=1.0, precision=3, verbose=false)
println("    MLFMA operator built: $(size(A_mlfma))")
println("  36a: PASS")

# 36b: Composite operator with MLFMA base
println("  36b: Composite operator + MLFMA ...")
part_ico = assign_patches_grid(mesh_ico_opt; nx=3, ny=3, nz=3)
Mp_ico = precompute_patch_mass(mesh_ico_opt, rwg_ico, part_ico; quad_order=3)
theta_ico = fill(200.0, part_ico.P)
println("    Icosphere: $(part_ico.P) spatial patches")

A_loaded = ImpedanceLoadedOperator(A_mlfma, Mp_ico, theta_ico, false)
@assert size(A_loaded) == (N_ico, N_ico)

# Forward solve with MLFMA composite
pw_ico = PlaneWaveExcitation(Vec3(0.0, 0.0, -k_ico), 1.0, Vec3(1.0, 0.0, 0.0))
v_ico = assemble_excitation(mesh_ico_opt, rwg_ico, pw_ico)

# Build preconditioner from the impedance-loaded MLFMA near-field
P_mlfma = build_mlfma_preconditioner(A_mlfma, Mp_ico, theta_ico;
    factorization=:ilu, ilu_tau=1e-2)
_assert_zero_allocation_mul!(NearFieldOperator(P_mlfma), v_ico)
_assert_zero_allocation_mul!(
    NearFieldAdjointOperator(P_mlfma), v_ico)

I_ico = solve_forward(A_loaded, v_ico; solver=:gmres, preconditioner=P_mlfma,
                       gmres_tol=1e-6, gmres_maxiter=300)
@assert length(I_ico) == N_ico
@assert norm(I_ico) > 0 "Solution should be nonzero"
println("    Forward solve OK, |I| = $(round(norm(I_ico), sigdigits=4))")
println("  36b: PASS")

# 36c: Adjoint solve with MLFMA composite
println("  36c: Adjoint solve with MLFMA composite ...")
rhs_adj_ico = randn(MersenneTwister(789), ComplexF64, N_ico)
lam_ico = solve_adjoint_rhs(A_loaded, rhs_adj_ico;
                             solver=:gmres, preconditioner=P_mlfma,
                             gmres_tol=1e-6, gmres_maxiter=300)
@assert length(lam_ico) == N_ico
@assert norm(lam_ico) > 0
println("    Adjoint solve OK, |λ| = $(round(norm(lam_ico), sigdigits=4))")
println("  36c: PASS")

# 36d: Multi-angle RCS optimization with MLFMA (3 iterations)
println("  36d: Multi-angle RCS + MLFMA (3 iterations) ...")
grid_ico = make_sph_grid(12, 24)
angles_ico = [
    (theta_inc=0.0, phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0),
    (theta_inc=π/3, phi_inc=0.0, pol=Vec3(1.0, 0.0, 0.0), weight=1.0),
]
configs_ico = build_multiangle_configs(mesh_ico_opt, rwg_ico, k_ico, angles_ico;
                                        grid=grid_ico, backscatter_cone=15.0,
                                        matrix_free_Q=true)
@assert configs_ico[1].Q isa FarFieldQMatrix "MLFMA optimization should use matrix-free Q in this test"

builder_calls_ico = Ref(0)
duplicate_builder_calls_ico = Ref(0)
last_builder_theta_ico = Ref{Union{Nothing, Vector{Float64}}}(nothing)
preconditioner_builder_ico = θ -> begin
    if last_builder_theta_ico[] !== nothing && last_builder_theta_ico[] == θ
        duplicate_builder_calls_ico[] += 1
    end
    last_builder_theta_ico[] = copy(θ)
    builder_calls_ico[] += 1
    build_mlfma_preconditioner(A_mlfma, Mp_ico, θ; factorization=:ilu, ilu_tau=1e-2)
end

theta_opt_ico, trace_ico = optimize_multiangle_rcs(
    A_mlfma, Mp_ico, configs_ico, theta_ico;
    maxiter=3, tol=1e-12, alpha0=0.01,
    reactive=false, verbose=false,
    lb=fill(0.0, part_ico.P), ub=fill(1000.0, part_ico.P),
    preconditioner_builder=preconditioner_builder_ico,
    gmres_tol=1e-6, gmres_maxiter=300
)
@assert length(trace_ico) == 3 "Should run exactly 3 iterations"
@assert all(t -> isfinite(t.J), trace_ico) "All objectives should be finite"
@assert all(t -> isfinite(t.gnorm), trace_ico) "All gradients should be finite"
@assert builder_calls_ico[] >= length(trace_ico) "Dynamic MLFMA preconditioner builder was not exercised"
@assert duplicate_builder_calls_ico[] == 0 "Dynamic preconditioner should be cached for unchanged theta"
println("    J: $(round(trace_ico[1].J, sigdigits=4)) → $(round(trace_ico[end].J, sigdigits=4))")
println("    |g|: $(round(trace_ico[1].gnorm, sigdigits=4)) → $(round(trace_ico[end].gnorm, sigdigits=4))")
println("  36d: PASS")

# 36e: Reuse accepted-iterate preconditioner for exploratory line-search trials,
# with rebuilt-trial fallback when a solve fails the residual guard.
println("  36e: MLFMA adaptive line-search trial preconditioner reuse ...")
builder_calls_current_ico = Ref(0)
preconditioner_builder_current_ico = θ -> begin
    builder_calls_current_ico[] += 1
    build_mlfma_preconditioner(A_mlfma, Mp_ico, θ; factorization=:ilu, ilu_tau=1e-2)
end

theta_current_ico, trace_current_ico = optimize_multiangle_rcs(
    A_mlfma, Mp_ico, configs_ico, theta_ico;
    maxiter=2, tol=1e-12, alpha0=0.01,
    reactive=false, verbose=false,
    lb=fill(0.0, part_ico.P), ub=fill(1000.0, part_ico.P),
    preconditioner_builder=preconditioner_builder_current_ico,
    trial_preconditioner_mode=:current_then_rebuild,
    gmres_tol=1e-6, gmres_maxiter=300,
    check_gmres_true_residual=true,
)
@assert length(theta_current_ico) == part_ico.P
@assert length(trace_current_ico) == 2 "Current-preconditioner mode should run exactly 2 iterations"
@assert builder_calls_current_ico[] <= length(trace_current_ico) + 1 "Line-search trials should avoid dynamic preconditioner rebuilds when current-preconditioner solves pass"
J_trace_current_ico = [t.J for t in trace_current_ico]
@assert all(J_trace_current_ico[2:end] .<= J_trace_current_ico[1:end-1] .+ 1e-12) "Current-preconditioner mode should not accept uphill objective steps"
bad_trial_mode = try
    optimize_multiangle_rcs(
        A_mlfma, Mp_ico, configs_ico, theta_ico;
        maxiter=0, verbose=false,
        preconditioner_builder=preconditioner_builder_current_ico,
        trial_preconditioner_mode=:invalid,
    )
    false
catch err
    occursin("trial_preconditioner_mode", sprint(showerror, err))
end
@assert bad_trial_mode "Invalid trial_preconditioner_mode should fail closed"
bad_lbfgs_cap = try
    optimize_multiangle_rcs(
        A_mlfma, Mp_ico, configs_ico, theta_ico;
        maxiter=0, verbose=false,
        lbfgs_line_search_maxiter=0,
    )
    false
catch err
    occursin("lbfgs_line_search_maxiter", sprint(showerror, err))
end
@assert bad_lbfgs_cap "Invalid L-BFGS line-search cap should fail closed"
bad_steepest_cap = try
    optimize_multiangle_rcs(
        A_mlfma, Mp_ico, configs_ico, theta_ico;
        maxiter=0, verbose=false,
        steepest_line_search_maxiter=0,
    )
    false
catch err
    occursin("steepest_line_search_maxiter", sprint(showerror, err))
end
@assert bad_steepest_cap "Invalid steepest line-search cap should fail closed"
println("    builder calls: $(builder_calls_current_ico[]) for $(length(trace_current_ico)) iterations")
println("  36e: PASS")

println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 43: Graded Mesh Generation
# ─────────────────────────────────────────────────
println("\nTest 43: Graded mesh generation")

mesh_g = make_rect_plate_graded(1.0, 1.0, 10, 10; grading_factor=3.0)
mesh_u = make_rect_plate(1.0, 1.0, 10, 10)

# Same topology: same number of triangles and vertices
@assert ntriangles(mesh_g) == ntriangles(mesh_u) "Graded mesh should have same triangle count"
@assert size(mesh_g.xyz, 2) == size(mesh_u.xyz, 2) "Graded mesh should have same vertex count"

# Bounding box preserved
@assert maximum(mesh_g.xyz[1,:]) ≈ 0.5 atol=1e-12 "x_max should be Lx/2"
@assert minimum(mesh_g.xyz[1,:]) ≈ -0.5 atol=1e-12 "x_min should be -Lx/2"
@assert maximum(mesh_g.xyz[2,:]) ≈ 0.5 atol=1e-12 "y_max should be Ly/2"
@assert minimum(mesh_g.xyz[2,:]) ≈ -0.5 atol=1e-12 "y_min should be -Ly/2"

# Grading effect: edge triangles should be smaller than center triangles
edge_areas = Float64[]
center_areas = Float64[]
for t in 1:ntriangles(mesh_g)
    cx = sum(mesh_g.xyz[1, mesh_g.tri[j, t]] for j in 1:3) / 3
    cy = sum(mesh_g.xyz[2, mesh_g.tri[j, t]] for j in 1:3) / 3
    a = triangle_area(mesh_g, t)
    if abs(cx) > 0.35 || abs(cy) > 0.35
        push!(edge_areas, a)
    elseif abs(cx) < 0.15 && abs(cy) < 0.15
        push!(center_areas, a)
    end
end
@assert !isempty(edge_areas) && !isempty(center_areas) "Should have both edge and center triangles"
@assert mean(edge_areas) < mean(center_areas) "Edge triangles should be smaller than center (grading effect)"
ratio = mean(center_areas) / mean(edge_areas)
@assert ratio > 2.0 "Center-to-edge area ratio should be > 2 for grading_factor=3, got $ratio"
println("  Center/edge area ratio: $(round(ratio, digits=1))")

# RWG construction should work
rwg_g = build_rwg(mesh_g)
@assert rwg_g.nedges > 0 "RWG construction should succeed on graded mesh"

# Near-zero grading_factor should give approximately uniform mesh
mesh_g0 = make_rect_plate_graded(1.0, 1.0, 10, 10; grading_factor=0.01)
max_vertex_diff = maximum(abs.(mesh_g0.xyz .- mesh_u.xyz))
@assert max_vertex_diff < 1e-3 "Near-zero grading should approximate uniform, max diff = $max_vertex_diff"
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 44: Analytical integral 1/R
# ─────────────────────────────────────────────────
println("\nTest 44: Analytical integral 1/R")

# Reference triangle: (0,0,0), (1,0,0), (0,1,0)
V1 = Vec3(0.0, 0.0, 0.0)
V2 = Vec3(1.0, 0.0, 0.0)
V3 = Vec3(0.0, 1.0, 0.0)

# Test at centroid — compare against high-order numerical quadrature
P_cent = (V1 + V2 + V3) / 3
S_analytical = analytical_integral_1overR(P_cent, V1, V2, V3)
@assert isfinite(S_analytical) "Analytical integral should be finite"
@assert S_analytical > 0 "Integral of 1/R should be positive"

# Reference value verified independently via brute-force midpoint rule (N=1000,
# converges to 2.4090 with ~0.07% error due to the centroid singularity).
# The analytical formula gives the exact coplanar result.
S_ref = 2.4072299231640093
@assert abs(S_analytical - S_ref) < 1e-10 "Analytical 1/R at centroid should match reference, got $S_analytical vs $S_ref"
println("  S(centroid) = $(round(S_analytical, digits=6))")

# Test at a vertex — should still be finite (the formula handles d_i ≈ 0 edges)
S_vertex = analytical_integral_1overR(V1, V1, V2, V3)
@assert isfinite(S_vertex) "Integral at vertex should be finite"
println("  S(vertex) = $(round(S_vertex, digits=6))")

# Endpoint subtraction over the full Float64 span is handled in the exact
# geometry path, while the final integral remains representable.
static_span_max = floatmax(Float64)
static_span_unit = nextfloat(0.0)
static_span_point = Vec3(-static_span_max, 0.0, 0.0)
static_span_second = Vec3(static_span_max, 0.0, 0.0)
static_span_third = Vec3(-static_span_max, static_span_unit, 0.0)
static_span_value = analytical_integral_1overR(
    static_span_point, static_span_point,
    static_span_second, static_span_third)
@test static_span_value == 7.194e-321

# A global scaling must not erase a representable off-plane height.  The
# one-sided static-field limits have opposite normal signs.
static_scale = ldexp(1.0, 600)
static_height = ldexp(1.0, -600)
static_large_v1 = Vec3(0.0, 0.0, 0.0)
static_large_v2 = Vec3(static_scale, 0.0, 0.0)
static_large_v3 = Vec3(0.0, static_scale, 0.0)
static_large_above = grad_analytical_integral_1overR(
    Vec3(static_scale / 4, static_scale / 4, static_height),
    static_large_v1, static_large_v2, static_large_v3)
static_large_below = grad_analytical_integral_1overR(
    Vec3(static_scale / 4, static_scale / 4, -static_height),
    static_large_v1, static_large_v2, static_large_v3)
@test static_large_above[3] ≈ -2π rtol=2eps(Float64)
@test static_large_below[3] ≈ 2π rtol=2eps(Float64)

# Cancellation in the signed-height dot product can also erase the side even
# when all normalized coordinate components remain nonzero.
static_tilt_unit = nextfloat(0.0)
static_tilt_v1 = Vec3(1.0, -1.0, 0.0)
static_tilt_v2 = Vec3(1.0, 0.0, -4.0)
static_tilt_v3 = Vec3(-2.0, 1.0, 4.0)
static_tilt_above = grad_analytical_integral_1overR(
    Vec3(0.0, 0.0, 3static_tilt_unit),
    static_tilt_v1, static_tilt_v2, static_tilt_v3)
static_tilt_below = grad_analytical_integral_1overR(
    Vec3(0.0, 0.0, -3static_tilt_unit),
    static_tilt_v1, static_tilt_v2, static_tilt_v3)
@test static_tilt_above == Vec3(
    -0x1.eeb65755bf836p+1,
    -0x1.3a20837c2e029p+2,
    -0x1.005f5f9297a89p+0)
@test static_tilt_below == Vec3(
    0x1.38a66cb26d166p+2,
    0x1.ebc229c23daafp+1,
    0x1.2fa238cab52f8p+0)

# Test symmetry: integral from centroid should not depend on triangle orientation
V1b = Vec3(0.0, 0.0, 0.0)
V2b = Vec3(0.0, 1.0, 0.0)
V3b = Vec3(1.0, 0.0, 0.0)
S_flipped = analytical_integral_1overR(P_cent, V1b, V2b, V3b)
@assert abs(S_analytical - S_flipped) < 1e-12 "Integral should be orientation-invariant"

# Test with scaled triangle — integral scales linearly with triangle size
scale = 2.0
S_scaled = analytical_integral_1overR(scale * P_cent, scale * V1, scale * V2, scale * V3)
@assert abs(S_scaled - scale * S_analytical) < 1e-10 "Integral should scale linearly, got $(S_scaled) vs $(scale * S_analytical)"
println("  Scaling test: PASS")

# Test off-plane: P above centroid at height h=0.5
P_offplane = P_cent + Vec3(0.0, 0.0, 0.5)
S_offplane = analytical_integral_1overR(P_offplane, V1, V2, V3)
S_offplane_ref = 0.8515737774153572  # verified via convergence of midpoint rule (N→∞)
@assert abs(S_offplane - S_offplane_ref) < 1e-10 "Off-plane integral at h=0.5 should match reference, got $S_offplane vs $S_offplane_ref"
println("  S(centroid, h=0.5) = $(round(S_offplane, digits=6))")

# Off-plane should decrease monotonically with height
S_h01 = analytical_integral_1overR(P_cent + Vec3(0,0,0.1), V1, V2, V3)
S_h10 = analytical_integral_1overR(P_cent + Vec3(0,0,1.0), V1, V2, V3)
@assert S_analytical > S_h01 > S_offplane > S_h10 > 0 "Integral should decrease with height"

# Symmetry: integral should be the same for +h and -h
S_neg_h = analytical_integral_1overR(P_cent - Vec3(0.0, 0.0, 0.5), V1, V2, V3)
@assert abs(S_offplane - S_neg_h) < 1e-12 "Integral should be symmetric in h sign"

# The exported gradient must agree with central differences of the scalar
# closed form above and reject invalid geometry instead of returning zeros.
grad_offplane = grad_analytical_integral_1overR(
    P_offplane, V1, V2, V3)
h_grad = 1e-6
grad_offplane_fd = Vec3(ntuple(axis -> begin
    displacement = Vec3(ntuple(
        component -> component == axis ? h_grad : 0.0, 3))
    (analytical_integral_1overR(
         P_offplane + displacement, V1, V2, V3) -
     analytical_integral_1overR(
         P_offplane - displacement, V1, V2, V3)) / (2h_grad)
end, 3))
@test grad_offplane ≈ grad_offplane_fd rtol=1e-8 atol=1e-10
@test_throws ArgumentError analytical_integral_1overR(
    Vec3(NaN, 0.0, 0.0), V1, V2, V3)
@test_throws ArgumentError analytical_integral_1overR(
    P_offplane, V1, V1, V1)
@test_throws ArgumentError grad_analytical_integral_1overR(
    P_offplane, V1, V1, V1)
@test _static_integral_allocations(
    P_offplane, V1, V2, V3) == (0, 0)
println("  Off-plane tests: PASS")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Test 45: Adjacent-cell contribution
# ─────────────────────────────────────────────────
println("\nTest 45: Adjacent-cell near-singular integration")

# Build a small plate and compare Z matrices with and without adjacent-cell fix
freq_adj = 3e9
c0_adj = 299792458.0
k_adj = 2π * freq_adj / c0_adj
mesh_adj = make_rect_plate(0.2, 0.2, 4, 4)
rwg_adj = build_rwg(mesh_adj)

Z_adj = assemble_Z_efie(mesh_adj, rwg_adj, k_adj; quad_order=4)

# Basic sanity: Z should be finite and non-zero
@assert all(isfinite, Z_adj) "All Z entries should be finite"
@assert maximum(abs, Z_adj) > 0 "Z should be non-zero"

# Symmetry check (EFIE should be symmetric for real-valued RWG)
sym_err = maximum(abs, Z_adj - transpose(Z_adj))
rel_sym = sym_err / maximum(abs, Z_adj)
@assert rel_sym < 0.02 "Relative symmetry error should be < 2%, got $(round(100*rel_sym, digits=2))%"
println("  Z size: $(size(Z_adj)), max|Z|=$(round(maximum(abs, Z_adj), digits=2))")
println("  Relative symmetry error: $(round(100*rel_sym, digits=4))%")

# Verify adjacent pairs are detected in the cache
cache_adj = DiffMoM._build_efie_cache(mesh_adj, rwg_adj, k_adj; quad_order=4)
n_adj_pairs = DiffMoM._adjacent_pair_count(cache_adj.adjacent)
@assert n_adj_pairs == 40 "4×4 split-quad grid should have 40 edge-adjacent pairs"
@assert length(cache_adj.adjacent.offsets) == ntriangles(mesh_adj) + 1
@assert length(cache_adj.adjacent.neighbors) == 2n_adj_pairs
cache_adj_coef = promote_type(
    eltype(rwg_adj.coeff_plus), eltype(rwg_adj.coeff_minus))
cache_adj_vec = SVector{3,cache_adj_coef}
cache_adj_fixed_bytes = DiffMoM._efie_cache_fixed_payload_bytes(
    rwg_adj.nedges,
    ntriangles(mesh_adj),
    cache_adj.Nq,
    length(cache_adj.wq_hi),
    cache_adj_coef,
    cache_adj_vec)
cache_adj_bound = DiffMoM._efie_cache_work_bytes(
    cache_adj_fixed_bytes, ntriangles(mesh_adj), n_adj_pairs)
@test_throws ArgumentError DiffMoM._build_efie_cache(
    mesh_adj, rwg_adj, k_adj;
    quad_order=4,
    max_cache_bytes=cache_adj_bound - 1)
@test DiffMoM._build_efie_cache(
    mesh_adj, rwg_adj, k_adj;
    quad_order=4,
    max_cache_bytes=cache_adj_bound).tri_ids == cache_adj.tri_ids
@test_throws ArgumentError assemble_Z_efie(
    mesh_adj, rwg_adj, k_adj; max_cache_bytes=1)
@test_throws ArgumentError matrixfree_efie_operator(
    mesh_adj, rwg_adj, k_adj; max_cache_bytes=1)

shared_edge_triangles = Matrix{Int}(undef, 3, 20)
for triangle in axes(shared_edge_triangles, 2)
    shared_edge_triangles[:, triangle] .= (1, 2, triangle + 2)
end
shared_edge_mesh = TriMesh(zeros(3, 22), shared_edge_triangles)
@test_throws ArgumentError DiffMoM._build_triangle_adjacency(
    shared_edge_mesh; max_adjacency_pairs=10)
# For a 4×4 grid of quads (32 triangles), there are many internal edges
# Each internal edge connects 2 triangles → pair stored in both compact rows.
println("  Adjacent pairs: $n_adj_pairs")
println("  High-order quad points: $(length(cache_adj.wq_hi)) (standard: $(cache_adj.Nq))")

# The resident adjacency representation must scale with mesh edges, not all
# triangle pairs. A 4× increase in linear grid resolution creates 16× as many
# triangles; compact storage should grow by roughly 16×, while a dense pair
# matrix would grow by 256×.
adj_mem_small = DiffMoM._build_triangle_adjacency(make_rect_plate(1.0, 1.0, 10, 10))
adj_mem_large = DiffMoM._build_triangle_adjacency(make_rect_plate(1.0, 1.0, 40, 40))
adj_bytes_small = Base.summarysize(adj_mem_small)
adj_bytes_large = Base.summarysize(adj_mem_large)
@assert adj_bytes_large < 20adj_bytes_small
@assert length(adj_mem_large.neighbors) <= 3 * (length(adj_mem_large.offsets) - 1)
println("  Adjacency storage: $adj_bytes_small → $adj_bytes_large bytes (200 → 3200 triangles)")

# Convergence test: compare Z at two quad orders
Z_lo = assemble_Z_efie(mesh_adj, rwg_adj, k_adj; quad_order=3)
Z_hi = assemble_Z_efie(mesh_adj, rwg_adj, k_adj; quad_order=4)
diff_Z = maximum(abs, Z_hi - Z_lo) / maximum(abs, Z_hi)
@assert diff_Z < 0.15 "Z should converge with quad order, relative diff = $(round(100*diff_Z, digits=2))%"
println("  Quad convergence (order 3→4): $(round(100*diff_Z, digits=4))% relative diff")
println("  PASS ✓")

# ─────────────────────────────────────────────────
# Periodic MoM + Topology Optimization Tests (37-42)
# ─────────────────────────────────────────────────
include("test_periodic_topology.jl")

# ─────────────────────────────────────────────────
# 2D TM volume-integral-equation solver
# ─────────────────────────────────────────────────
include("test_mom2d.jl")

# ─────────────────────────────────────────────────
# 3D vector material DDA solver
# ─────────────────────────────────────────────────
include("test_material_models3d.jl")
include("test_mom3d.jl")
include("test_mom3d_adjoint.jl")
include("test_mom3d_fft.jl")
include("test_mom3d_em.jl")
include("test_surface_ie3d.jl")

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
println("\n" * "="^60)
println("ALL 52 TESTS PASSED")
println("="^60)
