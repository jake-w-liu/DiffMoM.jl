#!/usr/bin/env julia

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames
using DiffMoM

const DATADIR = normpath(joinpath(@__DIR__, "..", "..", "data"))
mkpath(DATADIR)

const C0 = 299792458.0
const ETA0 = 376.730313668
const FREF = 3e9
const ROBUSTNESS_MAX_RELATIVE_RESIDUAL = 1e-10
const ROBUSTNESS_MAX_ENERGY_RATIO_ERROR = 0.02
const REGENERATE_DESIGN_COMMAND =
    "julia --project=. validation/paper/run_beam_steering_case.jl"

function relative_residual(A, x, rhs)
    rhs_norm = norm(rhs)
    residual_norm = norm(A * x - rhs)
    return iszero(rhs_norm) ?
           (iszero(residual_norm) ? 0.0 : Inf) :
           residual_norm / rhs_norm
end

function load_beam_design(path::AbstractString, mesh::TriMesh)
    isfile(path) || error(
        "Saved beam design not found at $path. Run `$REGENERATE_DESIGN_COMMAND` " *
        "from the repository root, then rerun this sweep.")

    table = CSV.read(path, DataFrame)
    required_columns = (:patch, :cx, :cy, :theta_opt)
    missing_columns = filter(
        column -> column ∉ propertynames(table), required_columns)
    isempty(missing_columns) || error(
        "Saved beam design $path is missing required columns: " *
        join(string.(missing_columns), ", ") * ". Run " *
        "`$REGENERATE_DESIGN_COMMAND` to regenerate it.")

    expected_rows = ntriangles(mesh)
    nrow(table) == expected_rows || error(
        "Saved beam design $path has $(nrow(table)) rows; the reference mesh " *
        "has $expected_rows triangles. Run `$REGENERATE_DESIGN_COMMAND` to " *
        "regenerate a matching design.")

    valid_patch = all(value ->
        value isa Real && isfinite(value) && isinteger(value), table.patch)
    valid_patch || error(
        "Saved beam design $path has a non-integer or non-finite `patch` value. " *
        "Run `$REGENERATE_DESIGN_COMMAND` to regenerate it.")
    Int.(table.patch) == collect(1:expected_rows) || error(
        "Saved beam design $path is not in triangle order 1:$expected_rows. " *
        "Run `$REGENERATE_DESIGN_COMMAND` to regenerate it.")

    for column in (:cx, :cy, :theta_opt)
        all(value -> value isa Real && isfinite(value), table[!, column]) ||
            error(
                "Saved beam design $path has a non-finite or non-numeric " *
                "`$column` value. Run `$REGENERATE_DESIGN_COMMAND` to " *
                "regenerate it.")
    end

    centers = [triangle_center(mesh, triangle) for triangle in 1:expected_rows]
    coordinate_scale = max(maximum(abs, mesh.xyz), 1.0)
    coordinate_tolerance = 128eps(coordinate_scale)
    coordinate_error = maximum(
        max(
            abs(Float64(table.cx[triangle]) - centers[triangle][1]),
            abs(Float64(table.cy[triangle]) - centers[triangle][2]),
        )
        for triangle in 1:expected_rows
    )
    coordinate_error <= coordinate_tolerance || error(
        "Saved beam design $path does not match the reference mesh " *
        "(maximum triangle-center error $coordinate_error m, allowed " *
        "$coordinate_tolerance m). Run `$REGENERATE_DESIGN_COMMAND` to " *
        "regenerate it.")

    return Vector{Float64}(table.theta_opt)
end

function build_target_mask(
    grid,
    theta_target_deg::Float64,
    cone_deg::Float64,
    phi_target_deg::Float64,
)
    θ0 = deg2rad(theta_target_deg)
    ϕ0 = deg2rad(phi_target_deg)
    r0 = Vec3(sin(θ0) * cos(ϕ0), sin(θ0) * sin(ϕ0), cos(θ0))
    return BitVector([begin
        rh = Vec3(grid.rhat[:, q])
        angle = acos(clamp(dot(rh, r0), -1.0, 1.0))
        angle <= deg2rad(cone_deg)
    end for q in 1:length(grid.w)])
end

function mean_dir_at_theta(
    theta_vals::Vector{Float64},
    dir_vals::Vector{Float64},
    theta_target::Float64,
)
    θuniq = unique(theta_vals)
    θnear = θuniq[argmin(abs.(θuniq .- theta_target))]
    idx = findall(t -> abs(t - θnear) < 1e-12, theta_vals)
    return θnear, sum(dir_vals[idx]) / length(idx)
end

function main()
    println("Robustness sweep (frequency/angle perturbations)")

    # Match the physical aperture used by run_beam_steering_case.jl.
    λref = C0 / FREF
    Lx = λref
    Ly = λref
    Nx = Ny = 12

    mesh = make_rect_plate(Lx, Ly, Nx, Ny)
    rwg = build_rwg(mesh)
    Nt = ntriangles(mesh)
    partition = PatchPartition(collect(1:Nt), Nt)
    Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

    imp_path = joinpath(DATADIR, "beam_steer_impedance.csv")
    theta_opt = load_beam_design(imp_path, mesh)

    grid = make_sph_grid(180, 72)
    mask = build_target_mask(grid, 30.0, 5.0, 0.0)

    cases = DataFrame(
        case = ["f_-2pct", "f_nom", "f_+2pct", "ang_-2deg", "ang_+2deg"],
        freq_GHz = [2.94, 3.00, 3.06, 3.00, 3.00],
        theta_inc_deg = [0.0, 0.0, 0.0, -2.0, 2.0],
    )

    J_opt = Float64[]
    J_pec = Float64[]
    gain_target_dB = Float64[]
    target_theta_deg = Float64[]
    peak_theta_opt_deg = Float64[]
    peak_opt_dBi = Float64[]
    residual_pec = Float64[]
    residual_opt = Float64[]
    energy_ratio_pec = Float64[]
    energy_ratio_opt = Float64[]

    for row in eachrow(cases)
        f = row.freq_GHz * 1e9
        θinc = deg2rad(row.theta_inc_deg)
        λ = C0 / f
        k = 2π / λ

        println("  Case $(row.case): f=$(row.freq_GHz) GHz, theta_inc=$(row.theta_inc_deg) deg")

        k_dir = Vec3(sin(θinc), 0.0, -cos(θinc))
        k_vec = k * k_dir
        pol = normalize(Vec3(cos(θinc), 0.0, sin(θinc))) # transverse to k_dir

        Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=ETA0)
        v = assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, pol; quad_order=3)

        G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=ETA0)
        pol_mat = pol_linear_x(grid)
        Q_target = build_Q(G_mat, grid, pol_mat; mask=mask)
        Q_total = build_Q(G_mat, grid, pol_mat)

        I_pec = Z_efie \ v
        push!(residual_pec, relative_residual(Z_efie, I_pec, v))
        f_pec = real(dot(I_pec, Q_target * I_pec))
        g_pec = real(dot(I_pec, Q_total * I_pec))
        isfinite(g_pec) && g_pec > 0.0 || error(
            "Case $(row.case) produced a non-positive or non-finite PEC " *
            "total-pattern objective: $g_pec. Inspect that case's solve and " *
            "far-field grid before using the sweep.")
        J_pec_case = f_pec / g_pec

        Z_opt = assemble_full_Z(Z_efie, Mp, theta_opt; reactive=true)
        I_opt = Z_opt \ v
        push!(residual_opt, relative_residual(Z_opt, I_opt, v))
        f_opt = real(dot(I_opt, Q_target * I_opt))
        g_opt = real(dot(I_opt, Q_total * I_opt))
        isfinite(g_opt) && g_opt > 0.0 || error(
            "Case $(row.case) produced a non-positive or non-finite saved-design " *
            "total-pattern objective: $g_opt. Inspect that case's loaded solve " *
            "and far-field grid before using the sweep.")
        J_opt_case = f_opt / g_opt

        E_ff_pec = compute_farfield(G_mat, I_pec, length(grid.w))
        E_ff_opt = compute_farfield(G_mat, I_opt, length(grid.w))
        push!(energy_ratio_pec, energy_ratio(I_pec, v, E_ff_pec, grid))
        push!(energy_ratio_opt, energy_ratio(I_opt, v, E_ff_opt, grid))
        p_pec = [real(dot(E_ff_pec[:, q], E_ff_pec[:, q])) for q in 1:length(grid.w)]
        p_opt = [real(dot(E_ff_opt[:, q], E_ff_opt[:, q])) for q in 1:length(grid.w)]
        Ppec = sum(p_pec[q] * grid.w[q] for q in 1:length(grid.w))
        Popt = sum(p_opt[q] * grid.w[q] for q in 1:length(grid.w))
        isfinite(Ppec) && Ppec > 0.0 || error(
            "Case $(row.case) produced non-positive or non-finite PEC sampled " *
            "far-field power: $Ppec. Inspect the PEC field and spherical grid.")
        isfinite(Popt) && Popt > 0.0 || error(
            "Case $(row.case) produced non-positive or non-finite saved-design " *
            "sampled far-field power: $Popt. Inspect the saved-design field " *
            "and spherical grid.")
        Dpec = [4π * p_pec[q] / Ppec for q in 1:length(grid.w)]
        Dopt = [4π * p_opt[q] / Popt for q in 1:length(grid.w)]
        dir_pec = 10 .* log10.(max.(Dpec, 1e-30))
        dir_opt = 10 .* log10.(max.(Dopt, 1e-30))

        dphi = 2π / 72
        phi0_idx = [
            q for q in eachindex(grid.w)
            if min(grid.phi[q], 2π - grid.phi[q]) <= dphi / 2 + 1e-10
        ]
        theta_cut = rad2deg.(grid.theta[phi0_idx])
        dir_pec_cut = dir_pec[phi0_idx]
        dir_opt_cut = dir_opt[phi0_idx]

        θtarget, pec_at_target = mean_dir_at_theta(theta_cut, dir_pec_cut, 30.0)
        _, opt_at_target = mean_dir_at_theta(theta_cut, dir_opt_cut, 30.0)
        gain = opt_at_target - pec_at_target

        # Peak of optimized pattern in phi~0 cut
        θuniq = unique(theta_cut)
        mean_opt_per_theta = [sum(dir_opt_cut[findall(t -> abs(t - θ) < 1e-12, theta_cut)]) /
                              length(findall(t -> abs(t - θ) < 1e-12, theta_cut)) for θ in θuniq]
        idx_peak = argmax(mean_opt_per_theta)

        push!(J_opt, J_opt_case * 100)
        push!(J_pec, J_pec_case * 100)
        push!(gain_target_dB, gain)
        push!(target_theta_deg, θtarget)
        push!(peak_theta_opt_deg, θuniq[idx_peak])
        push!(peak_opt_dBi, mean_opt_per_theta[idx_peak])
    end

    out = DataFrame(
        case = cases.case,
        freq_GHz = cases.freq_GHz,
        theta_inc_deg = cases.theta_inc_deg,
        J_opt_pct = J_opt,
        J_pec_pct = J_pec,
        gain_target_dB = gain_target_dB,
        target_theta_deg = target_theta_deg,
        peak_theta_opt_deg = peak_theta_opt_deg,
        peak_opt_dBi = peak_opt_dBi,
        residual_pec = residual_pec,
        residual_opt = residual_opt,
        energy_ratio_pec = energy_ratio_pec,
        energy_ratio_opt = energy_ratio_opt,
    )

    scalar_columns = (
        J_opt=J_opt,
        J_pec=J_pec,
        gain_target_dB=gain_target_dB,
        target_theta_deg=target_theta_deg,
        peak_theta_opt_deg=peak_theta_opt_deg,
        peak_opt_dBi=peak_opt_dBi,
        residual_pec=residual_pec,
        residual_opt=residual_opt,
        energy_ratio_pec=energy_ratio_pec,
        energy_ratio_opt=energy_ratio_opt,
    )
    nonfinite_columns = [
        string(name) for (name, values) in pairs(scalar_columns)
        if !all(isfinite, values)
    ]
    isempty(nonfinite_columns) || error(
        "robustness sweep produced non-finite columns: " *
        join(nonfinite_columns, ", ") *
        ". Inspect the corresponding case output before using the sweep.")
    checks = [
        ("all PEC relative residuals <= $ROBUSTNESS_MAX_RELATIVE_RESIDUAL",
         all(value <= ROBUSTNESS_MAX_RELATIVE_RESIDUAL
             for value in residual_pec)),
        ("all saved-design relative residuals <= " *
         "$ROBUSTNESS_MAX_RELATIVE_RESIDUAL",
         all(value <= ROBUSTNESS_MAX_RELATIVE_RESIDUAL
             for value in residual_opt)),
        ("all PEC energy-ratio errors <= " *
         "$(100 * ROBUSTNESS_MAX_ENERGY_RATIO_ERROR)%",
         all(abs(value - 1.0) <= ROBUSTNESS_MAX_ENERGY_RATIO_ERROR
             for value in energy_ratio_pec)),
        ("all saved-design energy-ratio errors <= " *
         "$(100 * ROBUSTNESS_MAX_ENERGY_RATIO_ERROR)%",
         all(abs(value - 1.0) <= ROBUSTNESS_MAX_ENERGY_RATIO_ERROR
             for value in energy_ratio_opt)),
        ("all objective fractions are nonnegative",
         all(value >= 0.0 for value in J_pec) &&
         all(value >= 0.0 for value in J_opt)),
        ("saved design respects the 500 ohm box",
         all(abs(value) <= 500.0 + 8eps(500.0) for value in theta_opt)),
    ]
    println("\nVerification")
    for (label, pass) in checks
        println("[$(pass ? "PASS" : "FAIL")] $label")
    end
    failed_checks = first.(filter(check -> !last(check), checks))
    isempty(failed_checks) || error(
        "Robustness-sweep checks failed: " * join(failed_checks, "; ") *
        ". Review the case metrics before using the CSV output.")

    output_path = joinpath(DATADIR, "robustness_sweep.csv")
    CSV.write(output_path, out)
    println("Saved $output_path")
end

main()
