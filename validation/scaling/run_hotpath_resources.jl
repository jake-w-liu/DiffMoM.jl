#!/usr/bin/env julia

# Run with one Julia/BLAS thread for comparable CPU samples:
# julia --project=. --startup-file=no validation/scaling/run_hotpath_resources.jl
# An optional argument selects the TOML report path. These measurements cover
# warmed cumulative Julia allocations, not peak RSS or native-library memory.
using DiffMoM
using LinearAlgebra
using Random
using SparseArrays
using TOML

function measure_hotpath(label, operation::F; iterations=5, samples=5) where {F}
    operation()
    operation()
    allocated = minimum(@allocated(operation()) for _ in 1:samples)
    elapsed = minimum(1:samples) do _
        started = time_ns()
        for _ in 1:iterations
            operation()
        end
        (time_ns() - started) / (1e9 * iterations)
    end
    println(rpad(label, 36), " ", elapsed, " s; ", allocated, " bytes")
    return Dict("name" => label, "seconds" => elapsed,
                "allocated_bytes" => allocated, "iterations" => iterations,
                "samples" => samples)
end

function main()
    rng = MersenneTwister(15102026)
    rows = Dict{String,Any}[]
    record(label, operation; kwargs...) =
        push!(rows, measure_hotpath(label, operation; kwargs...))

    mesh = make_rect_plate(0.2, 0.15, 3, 3)
    rwg = build_rwg(mesh)
    k = 2pi
    efie = matrixfree_efie_operator(mesh, rwg, k; quad_order=3)
    x = randn(rng, ComplexF64, rwg.nedges)
    y = similar(x)
    record("EFIE mul!, 21 RWGs", () -> mul!(y, efie, x))

    grid = make_sph_grid(10, 20)
    radiation = randn(rng, ComplexF64, 3length(grid.w), 48)
    pol = pol_linear_x(grid)
    q = build_Q_operator(radiation, grid, pol)
    qx = randn(rng, ComplexF64, 48)
    qy = similar(qx)
    record("Q mul!, 48 RWGs/200 directions", () -> mul!(qy, q, qx))
    fields = compute_farfield(radiation, qx, length(grid.w))
    record("radiated power, 200 directions", () -> radiated_power(fields, grid))
    record("PEC Mie, size parameter 10", () -> mie_s1s2_pec(10.0, 0.3))

    # Increase geometry and unknown count together so both fast operators
    # exercise far interactions, rather than a near-only tiny fixture.
    for side in (8, 16)
        fast_mesh = make_rect_plate(2.0, 1.0, side, side)
        fast_rwg = build_rwg(fast_mesh)
        nfast = fast_rwg.nedges
        fast_x = randn(rng, ComplexF64, nfast)
        fast_y = similar(fast_x)
        aca = build_aca_operator(fast_mesh, fast_rwg, k; leaf_size=16)
        mlfma = build_mlfma_operator(fast_mesh, fast_rwg, k; leaf_lambda=0.25)
        for (name, operator) in (("ACA", aca), ("MLFMA", mlfma))
            record("$name mul!, $nfast RWGs", () -> mul!(fast_y, operator, fast_x))
            record("$name adjoint, $nfast RWGs", () -> mul!(fast_y, adjoint(operator), fast_x))
        end
    end

    near_mesh = make_rect_plate(0.2, 0.15, 4, 4)
    near_rwg = build_rwg(near_mesh)
    near_current = randn(rng, ComplexF64, near_rwg.nedges)
    observations = vec([Vec3(px, py, 0.2) for px in range(-0.15, 0.15; length=5),
                       py in range(-0.1, 0.1; length=5)])
    near_k = 2pi * 3e9 / 299792458.0
    record("near field, 40 RWGs/25 points", () -> compute_nearfield(
        near_mesh, near_rwg, near_current, observations, near_k))

    po_grid = make_sph_grid(6, 12)
    po_frequency = 3e9
    po_wave = PlaneWaveExcitation(Vec3(0, 0, -near_k), 1.0, Vec3(1, 0, 0))
    for side in (4, 8)
        po_mesh = make_rect_plate(0.2, 0.15, side, side)
        triangles = ntriangles(po_mesh)
        record("PO, $triangles triangles/72 looks", () -> solve_po(
            po_mesh, po_frequency, po_wave; grid=po_grid))
        record("PTD, $triangles triangles/72 looks", () -> solve_ptd(
            po_mesh, po_frequency, po_wave; grid=po_grid))
    end

    filter_mesh = make_rect_plate(0.2, 0.15, 20, 20)
    weights, sums = build_filter_weights(filter_mesh, 0.025)
    density = rand(rng, length(sums))
    record("density filter, 800 triangles", () -> apply_filter(weights, sums, density))
    record("filter transpose, 800 triangles", () -> apply_filter_transpose(weights, sums, density))
    projected = collect(range(0.0, 1.0; length=1000))
    for beta in (8.0, 64.0, 128.0, 256.0)
        record("projection, beta=$beta/N=1000", () -> heaviside_project(projected, beta))
        record("derivative, beta=$beta/N=1000", () -> heaviside_derivative(projected, beta))
    end

    for side in (2, 4)
        voxels = VoxelGrid3D((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), side, side, side)
        incident = planewave_dda_3d(voxels, Vec3(0, 0, k), 1.0 + 0.2im, Vec3(1, 0.3, 0))
        solved = solve_dda_3d(voxels, k, 2.5 - 0.05im, incident)
        dda = dda_operator_3d(voxels, k, 2.5 - 0.05im)
        dda_x = reduce(vcat, solved.E_total)
        dda_y = similar(dda_x)
        record("DDA near-solved mul!, $(side^3) voxels", () -> mul!(dda_y, dda, dda_x);
               iterations=1, samples=3)
        random_x = randn(rng, ComplexF64, length(dda_x))
        record("DDA random mul!, $(side^3) voxels", () -> mul!(dda_y, dda, random_x))
        fft_dda = fft_dda_operator_3d(voxels, k, 2.5 - 0.05im)
        record("FFT DDA mul!, $(side^3) voxels", () -> mul!(dda_y, fft_dda, random_x))
        fft_em = fft_em_dda_operator_3d(voxels, k, 2.5 - 0.05im, 1.0)
        em_x = randn(rng, ComplexF64, size(fft_em, 2))
        em_y = similar(em_x)
        record("FFT EM-DDA mul!, $(side^3) voxels", () -> mul!(em_y, fft_em, em_x))
    end

    mesh2d = Mesh2D((-0.15, 0.15), (-0.15, 0.15), 8, 8)
    contrast = fill(1.5, mesh2d.ncells)
    incident2d = planewave_2d(mesh2d, k, 0.0)
    record("2D VIE assembly, 64 cells", () -> assemble_vie_2d(mesh2d, k, contrast))
    record("2D VIE solve, 64 cells", () -> solve_vie_2d(mesh2d, k, contrast, incident2d))

    # The Touchstone converter owns its solve buffers. Its constitutive
    # kernel is measured directly to exclude filesystem and text parsing.
    nports = 64
    e = Matrix{ComplexF64}(I, nports, nports)
    f = -(randn(rng, ComplexF64, nports, nports) + (30 + 10im) * I)
    refs = 20 .+ 80rand(rng, nports)
    record("network conversion, 64 ports", () -> DiffMoM._network_constitutive_s(e, f, refs))

    output = isempty(ARGS) ? joinpath(@__DIR__, "..", "..", "data", "hotpath_resources.toml") : ARGS[1]
    mkpath(dirname(abspath(output)))
    report = Dict("julia_version" => string(VERSION), "cpu" => Sys.CPU_NAME,
                  "kernel" => string(Sys.KERNEL), "julia_threads" => Threads.nthreads(),
                  "blas_threads" => BLAS.get_num_threads(), "measured_at_unix" => time(),
                  "measurements" => rows,
                  "scope" => "Warmed CPU latency and cumulative Julia allocations; excludes peak RSS and native allocation accounting.")
    open(output, "w") do io
        TOML.print(io, report; sorted=true)
    end
    println("Saved ", abspath(output))
end

main()
